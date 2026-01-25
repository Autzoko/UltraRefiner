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
from .sam_refiner import DifferentiableSAMRefiner, GatedResidualSAMRefiner


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
        sharpen_coarse_mask: bool = False,  # Sharpen soft masks to be more binary-like
        sharpen_temperature: float = 10.0,  # Temperature for sharpening (higher = sharper)
        use_roi_crop: bool = False,  # Whether to crop to ROI before SAM processing
        roi_expand_ratio: float = 0.2,  # Ratio to expand ROI box
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
            sharpen_coarse_mask: Whether to sharpen soft masks to match Phase 2 training distribution
            sharpen_temperature: Temperature for sharpening (higher = more binary-like)
            use_roi_crop: Whether to crop to ROI before SAM processing. Focuses SAM on lesion area.
            roi_expand_ratio: Ratio to expand ROI box (0.2 = 20% expansion on each side)
        """
        super().__init__()

        self.img_size = img_size
        self.sam_img_size = sam_img_size
        self.num_classes = num_classes
        self.sharpen_coarse_mask = sharpen_coarse_mask
        self.sharpen_temperature = sharpen_temperature

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
            use_roi_crop=use_roi_crop,
            roi_expand_ratio=roi_expand_ratio,
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

            # Debug: Check key matching
            model_keys = set(self.transunet.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            matched = model_keys & ckpt_keys
            missing_in_ckpt = model_keys - ckpt_keys
            unexpected = ckpt_keys - model_keys

            print(f"TransUNet checkpoint loading:")
            print(f"  - Model keys: {len(model_keys)}")
            print(f"  - Checkpoint keys: {len(ckpt_keys)}")
            print(f"  - Matched keys: {len(matched)}")
            print(f"  - Missing in checkpoint: {len(missing_in_ckpt)}")
            print(f"  - Unexpected in checkpoint: {len(unexpected)}")

            if len(matched) == 0:
                print("  WARNING: No keys matched! Checkpoint may be incompatible.")
                print(f"  Sample model keys: {list(model_keys)[:3]}")
                print(f"  Sample ckpt keys: {list(ckpt_keys)[:3]}")

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

        # Optional: Sharpen soft mask to be more binary-like (matches Phase 2 training distribution)
        # This uses a differentiable soft-threshold: sigmoid((x - 0.5) * temperature)
        # Higher temperature = sharper (more binary-like)
        if self.sharpen_coarse_mask:
            coarse_mask_prob = torch.sigmoid((coarse_mask_prob - 0.5) * self.sharpen_temperature)

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


# =============================================================================
# GATED ULTRA REFINER
# =============================================================================
# Uses gated residual refinement where SAM acts as a controlled error corrector
# instead of directly replacing the coarse prediction.
# =============================================================================


class GatedUltraRefiner(nn.Module):
    """
    UltraRefiner with gated residual refinement.

    Instead of directly using SAM's output, this model computes:
        final = coarse + gate * (sam_output - coarse)

    The gate is computed based on uncertainty, limiting corrections to uncertain
    regions and preventing degradation of already accurate coarse predictions.

    This design:
    1. Constrains SAM to act as a controlled error corrector
    2. Prevents degradation of already accurate TransUNet predictions
    3. Limits corrections to uncertain regions (coarse ≈ 0.5)
    4. Maintains full differentiability for end-to-end training
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
        mask_prompt_style: str = 'gaussian',
        sharpen_coarse_mask: bool = False,
        sharpen_temperature: float = 10.0,
        use_roi_crop: bool = False,
        roi_expand_ratio: float = 0.2,
        # Gated refinement parameters
        gate_type: str = 'uncertainty',
        gate_gamma: float = 1.0,
        gate_min: float = 0.0,
        gate_max: float = 0.8,
        learned_gate_channels: int = 32,
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
            use_point_prompt: Whether to use point prompts
            use_box_prompt: Whether to use box prompts
            use_mask_prompt: Whether to use mask prompts
            mask_prompt_style: Style for mask prompt ('direct' or 'gaussian')
            sharpen_coarse_mask: Whether to sharpen soft masks
            sharpen_temperature: Temperature for sharpening
            use_roi_crop: Whether to crop to ROI before SAM processing
            roi_expand_ratio: Ratio to expand ROI box
            gate_type: Type of gate ('uncertainty', 'learned', 'hybrid')
            gate_gamma: Gamma parameter for uncertainty gate
            gate_min: Minimum gate value
            gate_max: Maximum gate value (cap correction strength)
            learned_gate_channels: Hidden channels for learned gate network
        """
        super().__init__()

        self.img_size = img_size
        self.sam_img_size = sam_img_size
        self.num_classes = num_classes
        self.sharpen_coarse_mask = sharpen_coarse_mask
        self.sharpen_temperature = sharpen_temperature

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

        # Build Gated SAM Refiner
        self.sam_refiner = GatedResidualSAMRefiner(
            sam_model=self.sam,
            gate_type=gate_type,
            gate_gamma=gate_gamma,
            gate_min=gate_min,
            gate_max=gate_max,
            learned_hidden_channels=learned_gate_channels,
            use_point_prompt=use_point_prompt,
            use_box_prompt=use_box_prompt,
            use_mask_prompt=use_mask_prompt,
            freeze_image_encoder=freeze_sam_image_encoder,
            freeze_prompt_encoder=freeze_sam_prompt_encoder,
            mask_prompt_style=mask_prompt_style,
            use_roi_crop=use_roi_crop,
            roi_expand_ratio=roi_expand_ratio,
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
            weights = np.load(checkpoint_path)
            self.transunet.load_from(weights)
            print(f"Loaded TransUNet ViT weights from {checkpoint_path}")
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model' in state_dict:
                state_dict = state_dict['model']

            model_keys = set(self.transunet.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            matched = model_keys & ckpt_keys

            print(f"TransUNet checkpoint loading:")
            print(f"  - Matched keys: {len(matched)}/{len(model_keys)}")

            self.transunet.load_state_dict(state_dict, strict=False)
            print(f"Loaded TransUNet checkpoint from {checkpoint_path}")

    def prepare_sam_input(self, image: torch.Tensor) -> torch.Tensor:
        """Prepare image for SAM input."""
        B, C, H, W = image.shape

        if C == 1:
            image = image.repeat(1, 3, 1, 1)

        image = image * 255.0

        scale = self.sam_img_size / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)

        image_resized = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )

        pad_h = self.sam_img_size - new_h
        pad_w = self.sam_img_size - new_w
        image_padded = F.pad(image_resized, (0, pad_w, 0, pad_h), mode='constant', value=0)

        image_normalized = (image_padded - self.sam_pixel_mean) / self.sam_pixel_std

        return image_normalized

    def forward(
        self,
        image: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Gated UltraRefiner pipeline.

        Args:
            image: Input image (B, C, H, W), normalized to [0, 1]
            return_all: Whether to return all intermediate outputs

        Returns:
            Dictionary containing:
                - 'coarse_mask': Coarse mask from TransUNet (B, H, W)
                - 'refined_mask': Gated refined mask (B, H, W)
                - 'gate': Gate values showing where corrections were applied (B, H, W)
                - 'residual': Residual (sam - coarse) showing SAM's proposed corrections
                - 'iou_predictions': IoU predictions from SAM
        """
        # Step 1: TransUNet segmentation
        transunet_output = self.transunet(image)  # (B, num_classes, H, W)

        # Get soft mask (probability) for the foreground class
        if self.num_classes == 2:
            coarse_mask_prob = torch.softmax(transunet_output, dim=1)[:, 1]  # (B, H, W)
        else:
            coarse_mask_prob = torch.sigmoid(transunet_output[:, 1:].sum(dim=1))  # (B, H, W)

        # Optional: Sharpen soft mask
        if self.sharpen_coarse_mask:
            coarse_mask_prob = torch.sigmoid((coarse_mask_prob - 0.5) * self.sharpen_temperature)

        # Step 2: Prepare image for SAM
        sam_image = self.prepare_sam_input(image)

        # Step 3: Resize coarse mask to SAM input size
        coarse_mask_resized = F.interpolate(
            coarse_mask_prob.unsqueeze(1),
            size=(self.sam_img_size, self.sam_img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        # Step 4: Gated SAM Refinement
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

        # Resize gate back to original size
        gate = F.interpolate(
            sam_output['gate'].unsqueeze(1),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        # Resize residual back to original size
        residual = F.interpolate(
            sam_output['residual'].unsqueeze(1),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        result = {
            'coarse_logits': transunet_output,
            'coarse_mask': coarse_mask_prob,
            'refined_mask': refined_mask,
            'gate': gate,
            'residual': residual,
            'iou_predictions': sam_output['iou_predictions'],
            # SAM's raw output before gating
            'sam_mask': F.interpolate(
                sam_output['masks_sam'].unsqueeze(1),
                size=image.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1),
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

    def get_transunet_params(self):
        """Get TransUNet parameters for separate optimization."""
        return self.transunet.parameters()

    def get_sam_params(self):
        """Get SAM and gate parameters (only trainable ones) for separate optimization."""
        params = []
        for name, param in self.sam.named_parameters():
            if param.requires_grad:
                params.append(param)
        for name, param in self.sam_refiner.named_parameters():
            if param.requires_grad and 'sam' not in name:
                params.append(param)
        return params

    def get_gate_params(self):
        """Get gate network parameters (for learned gate only)."""
        params = []
        for name, param in self.sam_refiner.gated_refiner.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params


def build_gated_ultra_refiner(
    vit_name: str = 'R50-ViT-B_16',
    img_size: int = 224,
    num_classes: int = 2,
    sam_model_type: str = 'vit_b',
    sam_checkpoint: Optional[str] = None,
    transunet_checkpoint: Optional[str] = None,
    n_skip: int = 3,
    gate_type: str = 'uncertainty',
    gate_gamma: float = 1.0,
    gate_min: float = 0.0,
    gate_max: float = 0.8,
    **kwargs,
) -> GatedUltraRefiner:
    """
    Build GatedUltraRefiner model with specified configuration.

    This model uses gated residual refinement where SAM acts as a controlled
    error corrector instead of directly replacing the coarse prediction.

    Args:
        vit_name: TransUNet backbone name
        img_size: Input image size
        num_classes: Number of segmentation classes
        sam_model_type: SAM model type
        sam_checkpoint: Path to SAM/MedSAM checkpoint
        transunet_checkpoint: Path to TransUNet checkpoint
        n_skip: Number of skip connections for TransUNet
        gate_type: Type of gate
            - 'uncertainty': Based on coarse mask uncertainty (default, no extra params)
            - 'learned': Learned gate network (adds trainable params)
            - 'hybrid': Combination of both
        gate_gamma: Controls uncertainty curve shape (default 1.0)
            - gamma > 1: More aggressive gating (only very uncertain regions)
            - gamma < 1: Softer gating (more regions get corrections)
        gate_min: Minimum gate value (default 0.0)
        gate_max: Maximum gate value (default 0.8, caps correction strength)
        **kwargs: Additional arguments passed to GatedUltraRefiner

    Returns:
        GatedUltraRefiner model

    Example usage:
        # For unfinetuned SAM with uncertainty gating
        model = build_gated_ultra_refiner(
            sam_checkpoint='./pretrained/medsam_vit_b.pth',
            transunet_checkpoint='./checkpoints/transunet/best.pth',
            gate_type='uncertainty',
            gate_gamma=1.0,
            gate_max=0.8,  # Cap corrections at 80%
            mask_prompt_style='gaussian',
        )
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

    model = GatedUltraRefiner(
        transunet_config=config,
        img_size=img_size,
        num_classes=num_classes,
        sam_model_type=sam_model_type,
        sam_checkpoint=sam_checkpoint,
        transunet_checkpoint=transunet_checkpoint,
        gate_type=gate_type,
        gate_gamma=gate_gamma,
        gate_min=gate_min,
        gate_max=gate_max,
        **kwargs,
    )

    return model
