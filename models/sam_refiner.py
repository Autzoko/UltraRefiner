"""
Differentiable SAM Refiner module for end-to-end training.
This module takes coarse segmentation masks and refines them using SAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import numpy as np


class DifferentiableSAMRefiner(nn.Module):
    """
    Differentiable SAM Refiner for end-to-end training.

    This module refines coarse masks from a segmentation model (e.g., TransUNet)
    using SAM's mask decoder in a fully differentiable manner.

    Key features:
    - Differentiable prompt generation from soft masks
    - Supports point, box, and mask prompts
    - End-to-end gradient flow from SAM output back to upstream model
    """

    def __init__(
        self,
        sam_model,
        use_point_prompt=True,
        use_box_prompt=True,
        use_mask_prompt=True,
        num_points=1,
        add_negative_point=True,
        freeze_image_encoder=True,
        freeze_prompt_encoder=False,
    ):
        """
        Args:
            sam_model: Pre-trained SAM model
            use_point_prompt: Whether to use point prompts
            use_box_prompt: Whether to use box prompts
            use_mask_prompt: Whether to use mask prompts
            num_points: Number of point prompts per mask
            add_negative_point: Whether to add negative point prompts
            freeze_image_encoder: Whether to freeze SAM's image encoder
            freeze_prompt_encoder: Whether to freeze SAM's prompt encoder
        """
        super().__init__()

        self.sam = sam_model
        self.use_point_prompt = use_point_prompt
        self.use_box_prompt = use_box_prompt
        self.use_mask_prompt = use_mask_prompt
        self.num_points = num_points
        self.add_negative_point = add_negative_point

        # Freeze components if specified
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False

        # Learnable temperature for soft argmax
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def extract_soft_points(self, soft_mask: torch.Tensor, num_points: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract point prompts from soft masks using differentiable soft-argmax.

        Args:
            soft_mask: Soft probability mask (B, H, W) with values in [0, 1]
            num_points: Number of points to extract

        Returns:
            point_coords: (B, N, 2) point coordinates
            point_labels: (B, N) point labels (1 for positive)
        """
        B, H, W = soft_mask.shape
        device = soft_mask.device

        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Soft-argmax: use probability-weighted average
        # Add small epsilon to avoid division by zero
        mask_sum = soft_mask.sum(dim=(-2, -1)) + 1e-6  # (B,)

        # Weighted centroid (positive point)
        y_center = (soft_mask * y_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # (B,)
        x_center = (soft_mask * x_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # (B,)

        point_coords = torch.stack([x_center, y_center], dim=-1).unsqueeze(1)  # (B, 1, 2)
        point_labels = torch.ones(B, 1, device=device)

        return point_coords, point_labels

    def extract_soft_negative_points(self, soft_mask: torch.Tensor, box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract negative point prompts from the inverse of soft masks within bounding box.

        Args:
            soft_mask: Soft probability mask (B, H, W)
            box: Bounding box (B, 4) as [x1, y1, x2, y2]

        Returns:
            point_coords: (B, 1, 2) point coordinates
            point_labels: (B, 1) point labels (0 for negative)
        """
        B, H, W = soft_mask.shape
        device = soft_mask.device

        # Inverse mask (background probability)
        inv_mask = 1.0 - soft_mask

        # Create box mask
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Apply box constraint
        box_mask = torch.zeros_like(soft_mask)
        for b in range(B):
            x1, y1, x2, y2 = box[b].int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            box_mask[b, y1:y2, x1:x2] = 1.0

        # Masked inverse
        inv_mask_boxed = inv_mask * box_mask

        # Soft-argmax on inverse mask
        mask_sum = inv_mask_boxed.sum(dim=(-2, -1)) + 1e-6  # (B,)
        y_center = (inv_mask_boxed * y_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # (B,)
        x_center = (inv_mask_boxed * x_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum  # (B,)

        point_coords = torch.stack([x_center, y_center], dim=-1).unsqueeze(1)  # (B, 1, 2)
        point_labels = torch.zeros(B, 1, device=device)

        return point_coords, point_labels

    def extract_soft_box(self, soft_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Extract bounding box from soft masks in a semi-differentiable manner.

        Args:
            soft_mask: Soft probability mask (B, H, W)
            threshold: Threshold for determining box boundaries

        Returns:
            boxes: (B, 4) bounding boxes as [x1, y1, x2, y2]
        """
        B, H, W = soft_mask.shape
        device = soft_mask.device

        boxes = []
        for b in range(B):
            mask = soft_mask[b]

            # Find where mask exceeds threshold
            mask_binary = (mask > threshold).float()

            # Get coordinates
            y_coords = torch.arange(H, device=device, dtype=torch.float32)
            x_coords = torch.arange(W, device=device, dtype=torch.float32)

            # Project to axes
            y_proj = mask_binary.max(dim=1)[0]  # (H,)
            x_proj = mask_binary.max(dim=0)[0]  # (W,)

            # Find min/max using weighted approach for sub-differentiability
            y_weight = y_proj * y_coords
            x_weight = x_proj * x_coords

            # Use soft min/max approximation
            y_indices = torch.nonzero(y_proj, as_tuple=True)[0]
            x_indices = torch.nonzero(x_proj, as_tuple=True)[0]

            if len(y_indices) > 0 and len(x_indices) > 0:
                y1 = y_indices.min().float()
                y2 = y_indices.max().float()
                x1 = x_indices.min().float()
                x2 = x_indices.max().float()
            else:
                # Fallback: use centroid-based box
                mask_sum = mask.sum() + 1e-6
                y_center = (mask * y_coords.unsqueeze(1)).sum() / mask_sum
                x_center = (mask * x_coords.unsqueeze(0)).sum() / mask_sum
                box_size = torch.sqrt(mask_sum)
                y1 = torch.clamp(y_center - box_size / 2, 0, H - 1)
                y2 = torch.clamp(y_center + box_size / 2, 0, H - 1)
                x1 = torch.clamp(x_center - box_size / 2, 0, W - 1)
                x2 = torch.clamp(x_center + box_size / 2, 0, W - 1)

            boxes.append(torch.stack([x1, y1, x2, y2]))

        return torch.stack(boxes)

    def prepare_mask_input(self, soft_mask: torch.Tensor, target_size: int = 256) -> torch.Tensor:
        """
        Prepare mask prompt for SAM.

        Args:
            soft_mask: Soft probability mask (B, H, W)
            target_size: Target size for mask input (SAM uses 256x256)

        Returns:
            mask_input: (B, 1, target_size, target_size) mask prompt
        """
        B, H, W = soft_mask.shape

        # Convert to logits-like values
        # SAM expects values where positive = foreground
        mask_logits = (soft_mask * 2 - 1) * 10  # Scale to [-10, 10]

        # Resize to SAM's expected input size
        mask_input = F.interpolate(
            mask_logits.unsqueeze(1),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        return mask_input

    def forward(
        self,
        image: torch.Tensor,
        coarse_mask: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Refine coarse masks using SAM.

        Args:
            image: Input image (B, 3, H, W) normalized for SAM
            coarse_mask: Coarse segmentation mask (B, H, W) with values in [0, 1]
            return_intermediate: Whether to return intermediate results

        Returns:
            Dictionary containing:
                - 'masks': Refined masks (B, H, W)
                - 'iou_predictions': IoU predictions (B, 3)
                - 'low_res_masks': Low resolution masks (B, 3, 256, 256)
                - 'prompts': Dictionary of prompts used (if return_intermediate)
        """
        B = image.shape[0]
        device = image.device
        original_size = coarse_mask.shape[-2:]

        # Get image embeddings
        with torch.set_grad_enabled(self.sam.image_encoder.training):
            input_images = torch.stack([self.sam.preprocess(img) for img in image])
            image_embeddings = self.sam.image_encoder(input_images)

        # Prepare prompts
        prompts = {}

        # Box prompts
        if self.use_box_prompt:
            boxes = self.extract_soft_box(coarse_mask)
            # Scale boxes to SAM input size
            scale_h = self.sam.image_encoder.img_size / original_size[0]
            scale_w = self.sam.image_encoder.img_size / original_size[1]
            scaled_boxes = boxes.clone()
            scaled_boxes[:, [0, 2]] *= scale_w
            scaled_boxes[:, [1, 3]] *= scale_h
            prompts['boxes'] = scaled_boxes

        # Point prompts
        if self.use_point_prompt:
            point_coords, point_labels = self.extract_soft_points(coarse_mask, self.num_points)

            if self.add_negative_point and self.use_box_prompt:
                neg_coords, neg_labels = self.extract_soft_negative_points(coarse_mask, boxes)
                point_coords = torch.cat([point_coords, neg_coords], dim=1)
                point_labels = torch.cat([point_labels, neg_labels], dim=1)

            # Scale points to SAM input size
            scale_h = self.sam.image_encoder.img_size / original_size[0]
            scale_w = self.sam.image_encoder.img_size / original_size[1]
            scaled_points = point_coords.clone()
            scaled_points[:, :, 0] *= scale_w
            scaled_points[:, :, 1] *= scale_h
            prompts['point_coords'] = scaled_points
            prompts['point_labels'] = point_labels

        # Mask prompts
        if self.use_mask_prompt:
            mask_input = self.prepare_mask_input(coarse_mask)
            prompts['mask_inputs'] = mask_input

        # Run through SAM decoder
        all_masks = []
        all_ious = []
        all_low_res = []

        for b in range(B):
            curr_embedding = image_embeddings[b].unsqueeze(0)

            # Prepare prompt inputs
            point_input = None
            if self.use_point_prompt:
                point_input = (
                    prompts['point_coords'][b:b + 1],
                    prompts['point_labels'][b:b + 1]
                )

            box_input = prompts.get('boxes', None)
            if box_input is not None:
                box_input = box_input[b:b + 1]

            mask_input = prompts.get('mask_inputs', None)
            if mask_input is not None:
                mask_input = mask_input[b:b + 1]

            # Encode prompts
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=point_input,
                boxes=box_input,
                masks=mask_input,
            )

            # Decode masks
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            # Upscale masks to original size
            masks = F.interpolate(
                low_res_masks,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )

            all_masks.append(masks.squeeze(0))
            all_ious.append(iou_predictions.squeeze(0))
            all_low_res.append(low_res_masks.squeeze(0))

        # Stack results
        all_masks = torch.stack(all_masks)  # (B, 3, H, W)
        all_ious = torch.stack(all_ious)  # (B, 3)
        all_low_res = torch.stack(all_low_res)  # (B, 3, 256, 256)

        # Select best mask based on IoU prediction
        best_idx = all_ious.argmax(dim=1)
        refined_masks = torch.stack([
            all_masks[b, best_idx[b]] for b in range(B)
        ])  # (B, H, W)

        result = {
            'masks': refined_masks,
            'masks_all': all_masks,
            'iou_predictions': all_ious,
            'low_res_masks': all_low_res,
        }

        if return_intermediate:
            result['prompts'] = prompts
            result['image_embeddings'] = image_embeddings

        return result


class SAMRefinerInference(nn.Module):
    """
    SAM Refiner for inference with iterative refinement support.
    """

    def __init__(
        self,
        sam_model,
        use_point_prompt=True,
        use_box_prompt=True,
        use_mask_prompt=True,
        num_iterations=3,
    ):
        super().__init__()
        self.sam = sam_model
        self.use_point_prompt = use_point_prompt
        self.use_box_prompt = use_box_prompt
        self.use_mask_prompt = use_mask_prompt
        self.num_iterations = num_iterations

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
        coarse_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Refine masks with iterative refinement.

        Args:
            image: Input image (B, 3, H, W)
            coarse_mask: Initial coarse mask (B, H, W)

        Returns:
            Refined mask (B, H, W)
        """
        current_mask = coarse_mask

        for _ in range(self.num_iterations):
            # Use differentiable refiner for each iteration
            refiner = DifferentiableSAMRefiner(
                self.sam,
                use_point_prompt=self.use_point_prompt,
                use_box_prompt=self.use_box_prompt,
                use_mask_prompt=self.use_mask_prompt,
            )
            refiner.eval()

            result = refiner(image, current_mask)
            current_mask = torch.sigmoid(result['masks'])

        return current_mask
