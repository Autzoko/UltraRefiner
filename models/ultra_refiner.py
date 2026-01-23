"""
UltraRefiner: End-to-end differentiable segmentation refinement model.
Combines TransUNet for initial segmentation with SAMRefiner for mask refinement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .transunet import VisionTransformer, CONFIGS
from .sam import sam_model_registry
from .sam_refiner import DifferentiableSAMRefiner


class UltraRefiner(nn.Module):
    """
    End-to-end segmentation refinement model.

    Architecture:
        1. TransUNet: Produces initial coarse segmentation
        2. SAMRefiner: Refines the coarse mask using SAM

    The entire pipeline is differentiable, allowing gradients to flow
    from SAMRefiner back to TransUNet during training.
    """

    def __init__(
        self,
        transunet_config,
        img_size: int = 224,
        sam_img_size: int = 1024,
        num_classes: int = 2,
        sam_model_type: str = 'vit_b',
        sam_checkpoint: Optional[str] = None,
        transunet_checkpoint: Optional[str] = None,
        freeze_sam_image_encoder: bool = True,
        freeze_sam_prompt_encoder: bool = False,
        use_point_prompt: bool = True,
        use_box_prompt: bool = True,
        use_mask_prompt: bool = True,
        mask_prompt_style: str = 'direct',  # 'direct' for E2E (TransUNet outputs are smooth)
    ):
        """
        Args:
            transunet_config: Configuration for TransUNet
            img_size: Input image size for TransUNet
            sam_img_size: Input image size for SAM (typically 1024)
            num_classes: Number of segmentation classes
            sam_model_type: SAM model variant ('vit_b', 'vit_l', 'vit_h')
            sam_checkpoint: Path to SAM/MedSAM checkpoint
            transunet_checkpoint: Path to pre-trained TransUNet checkpoint
            freeze_sam_image_encoder: Whether to freeze SAM's image encoder
            freeze_sam_prompt_encoder: Whether to freeze SAM's prompt encoder
            use_point_prompt: Whether to use point prompts in refinement
            use_box_prompt: Whether to use box prompts in refinement
            use_mask_prompt: Whether to use mask prompts in refinement
            mask_prompt_style: Style for mask prompt ('direct' for E2E, 'gaussian' for SAM finetuning)
        """
        super().__init__()

        self.img_size = img_size
        self.sam_img_size = sam_img_size
        self.num_classes = num_classes

        # Build TransUNet
        transunet_config.n_classes = num_classes
        self.transunet = VisionTransformer(
            config=transunet_config,
            img_size=img_size,
            num_classes=num_classes,
        )

        # Load TransUNet checkpoint if provided
        if transunet_checkpoint is not None:
            self._load_transunet_checkpoint(transunet_checkpoint)

        # Build SAM
        sam_builder = sam_model_registry[sam_model_type]
        self.sam = sam_builder(checkpoint=sam_checkpoint)

        # Freeze SAM components as specified
        if freeze_sam_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        if freeze_sam_prompt_encoder:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False

        # Build SAMRefiner
        # For E2E training, use 'direct' style since TransUNet outputs are already smooth
        # For SAM-only finetuning with augmented data, use 'gaussian' to smooth artificial edges
        self.sam_refiner = DifferentiableSAMRefiner(
            sam_model=self.sam,
            use_point_prompt=use_point_prompt,
            use_box_prompt=use_box_prompt,
            use_mask_prompt=use_mask_prompt,
            freeze_image_encoder=freeze_sam_image_encoder,
            freeze_prompt_encoder=freeze_sam_prompt_encoder,
            mask_prompt_style=mask_prompt_style,
        )

        # SAM normalization parameters
        self.register_buffer(
            'sam_pixel_mean',
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'sam_pixel_std',
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        )

    def _load_transunet_checkpoint(self, checkpoint_path: str):
        """Load pre-trained TransUNet weights."""
        import numpy as np
        if checkpoint_path.endswith('.npz'):
            # Load ViT pre-trained weights
            weights = np.load(checkpoint_path)
            self.transunet.load_from(weights)
            print(f"Loaded TransUNet ViT weights from {checkpoint_path}")
        else:
            # Load full model checkpoint
            # weights_only=False for PyTorch 2.6+ compatibility
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.transunet.load_state_dict(state_dict, strict=False)
            print(f"Loaded TransUNet checkpoint from {checkpoint_path}")

    def prepare_sam_input(self, image: torch.Tensor) -> torch.Tensor:
        """
        Prepare image for SAM input.

        Args:
            image: Image tensor (B, C, H, W), can be 1-channel or 3-channel

        Returns:
            SAM-ready image (B, 3, sam_img_size, sam_img_size)
        """
        B, C, H, W = image.shape

        # Convert to 3 channels if grayscale
        if C == 1:
            image = image.repeat(1, 3, 1, 1)

        # Denormalize if needed (assuming input is normalized to [0, 1])
        image = image * 255.0

        # Resize to SAM size preserving aspect ratio
        scale = self.sam_img_size / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)

        image_resized = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )

        # Pad to square
        pad_h = self.sam_img_size - new_h
        pad_w = self.sam_img_size - new_w
        image_padded = F.pad(image_resized, (0, pad_w, 0, pad_h), mode='constant', value=0)

        # Normalize for SAM
        image_normalized = (image_padded - self.sam_pixel_mean) / self.sam_pixel_std

        return image_normalized

    def forward(
        self,
        image: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full UltraRefiner pipeline.

        Args:
            image: Input image (B, C, H, W), normalized to [0, 1]
            return_all: Whether to return all intermediate outputs

        Returns:
            Dictionary containing:
                - 'coarse_mask': Coarse mask from TransUNet (B, num_classes, H, W)
                - 'refined_mask': Refined mask from SAMRefiner (B, H, W)
                - 'iou_predictions': IoU predictions from SAM (B, 3)
                - Additional outputs if return_all=True
        """
        # Step 1: TransUNet segmentation
        transunet_output = self.transunet(image)  # (B, num_classes, H, W)

        # Get soft mask (probability) for the foreground class
        if self.num_classes == 2:
            coarse_mask_prob = torch.softmax(transunet_output, dim=1)[:, 1]  # (B, H, W)
        else:
            coarse_mask_prob = torch.sigmoid(transunet_output[:, 1:].sum(dim=1))  # (B, H, W)

        # Step 2: Prepare image for SAM
        sam_image = self.prepare_sam_input(image)

        # Step 3: Resize coarse mask to SAM input size
        coarse_mask_resized = F.interpolate(
            coarse_mask_prob.unsqueeze(1),
            size=(self.sam_img_size, self.sam_img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        # Step 4: SAM Refinement
        sam_output = self.sam_refiner(
            sam_image,
            coarse_mask_resized,
            return_intermediate=return_all,
        )

        # Resize refined mask back to original size
        refined_mask = F.interpolate(
            sam_output['masks'].unsqueeze(1),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        result = {
            'coarse_logits': transunet_output,
            'coarse_mask': coarse_mask_prob,
            'refined_mask': refined_mask,
            'refined_mask_logits': sam_output['masks'],
            'iou_predictions': sam_output['iou_predictions'],
        }

        if return_all:
            result['sam_masks_all'] = sam_output['masks_all']
            result['low_res_masks'] = sam_output['low_res_masks']
            if 'prompts' in sam_output:
                result['prompts'] = sam_output['prompts']

        return result

    def forward_transunet_only(self, image: torch.Tensor) -> torch.Tensor:
        """Forward through TransUNet only (for Phase 1 training)."""
        return self.transunet(image)

    def forward_sam_only(
        self,
        image: torch.Tensor,
        coarse_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward through SAM only (for Phase 2 training)."""
        sam_image = self.prepare_sam_input(image)
        coarse_mask_resized = F.interpolate(
            coarse_mask.unsqueeze(1) if coarse_mask.dim() == 3 else coarse_mask,
            size=(self.sam_img_size, self.sam_img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        return self.sam_refiner(sam_image, coarse_mask_resized)

    def get_transunet_params(self):
        """Get TransUNet parameters for separate optimization."""
        return self.transunet.parameters()

    def get_sam_params(self):
        """Get SAM parameters (only trainable ones) for separate optimization."""
        params = []
        for name, param in self.sam.named_parameters():
            if param.requires_grad:
                params.append(param)
        for name, param in self.sam_refiner.named_parameters():
            if param.requires_grad and 'sam' not in name:
                params.append(param)
        return params


def build_ultra_refiner(
    vit_name: str = 'R50-ViT-B_16',
    img_size: int = 224,
    num_classes: int = 2,
    sam_model_type: str = 'vit_b',
    sam_checkpoint: Optional[str] = None,
    transunet_checkpoint: Optional[str] = None,
    n_skip: int = 3,
    **kwargs,
) -> UltraRefiner:
    """
    Build UltraRefiner model with specified configuration.

    Args:
        vit_name: TransUNet backbone name
        img_size: Input image size
        num_classes: Number of segmentation classes
        sam_model_type: SAM model type
        sam_checkpoint: Path to SAM/MedSAM checkpoint
        transunet_checkpoint: Path to TransUNet checkpoint
        n_skip: Number of skip connections for TransUNet

    Returns:
        UltraRefiner model
    """
    # Get TransUNet config
    config = CONFIGS[vit_name]
    config.n_classes = num_classes
    config.n_skip = n_skip

    if vit_name.startswith('R50'):
        config.patches.grid = (
            img_size // 16,
            img_size // 16,
        )

    model = UltraRefiner(
        transunet_config=config,
        img_size=img_size,
        num_classes=num_classes,
        sam_model_type=sam_model_type,
        sam_checkpoint=sam_checkpoint,
        transunet_checkpoint=transunet_checkpoint,
        **kwargs,
    )

    return model
