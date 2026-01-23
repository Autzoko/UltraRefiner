"""
Differentiable SAM Refiner module for end-to-end training.
This module takes coarse segmentation masks and refines them using SAM.

IMPORTANT: All prompt extraction methods must be differentiable to allow
gradient flow from SAM output back to the upstream segmentation model.
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

    Mathematical guarantee of differentiability:
    - Point extraction: Uses soft-argmax (weighted average)
    - Box extraction: Uses soft-min/max with temperature
    - Mask selection: Uses soft selection with IoU-weighted combination
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
        selection_temperature=0.1,
        box_temperature=0.01,
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
            selection_temperature: Temperature for soft mask selection (lower = sharper)
            box_temperature: Temperature for soft box extraction (lower = sharper)
        """
        super().__init__()

        self.sam = sam_model
        self.use_point_prompt = use_point_prompt
        self.use_box_prompt = use_box_prompt
        self.use_mask_prompt = use_mask_prompt
        self.num_points = num_points
        self.add_negative_point = add_negative_point
        self.selection_temperature = selection_temperature
        self.box_temperature = box_temperature

        # Freeze components if specified
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False

        # Learnable temperature for soft argmax (optional)
        self.register_buffer('_dummy', torch.tensor(0.0))  # For device tracking

    def extract_soft_points(self, soft_mask: torch.Tensor, num_points: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract point prompts from soft masks using differentiable soft-argmax.

        Mathematical formulation:
            x_center = Σ(p(i,j) · x_j) / Σ(p(i,j))
            y_center = Σ(p(i,j) · y_i) / Σ(p(i,j))

        This is fully differentiable as it's a weighted average.

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
        mask_sum = soft_mask.sum(dim=(-2, -1), keepdim=True) + 1e-6  # (B, 1, 1)

        # Weighted centroid (positive point)
        # ∂y_center/∂soft_mask = (y_grid - y_center) / mask_sum  [non-zero gradient!]
        y_center = (soft_mask * y_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum.squeeze()  # (B,)
        x_center = (soft_mask * x_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum.squeeze()  # (B,)

        point_coords = torch.stack([x_center, y_center], dim=-1).unsqueeze(1)  # (B, 1, 2)
        point_labels = torch.ones(B, 1, device=device)

        return point_coords, point_labels

    def extract_soft_negative_points(self, soft_mask: torch.Tensor, box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract negative point prompts using differentiable soft box masking.

        Instead of hard box boundaries, we use a soft sigmoid transition.

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

        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Create SOFT box mask using sigmoid (differentiable!)
        # This creates smooth transitions at box boundaries
        sharpness = 1.0  # Controls transition sharpness

        x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]  # (B,) each

        # Soft box: sigmoid transitions at boundaries
        # left_boundary:  σ(sharpness * (x - x1))
        # right_boundary: σ(sharpness * (x2 - x))
        left = torch.sigmoid(sharpness * (x_grid.unsqueeze(0) - x1.view(B, 1, 1)))
        right = torch.sigmoid(sharpness * (x2.view(B, 1, 1) - x_grid.unsqueeze(0)))
        top = torch.sigmoid(sharpness * (y_grid.unsqueeze(0) - y1.view(B, 1, 1)))
        bottom = torch.sigmoid(sharpness * (y2.view(B, 1, 1) - y_grid.unsqueeze(0)))

        soft_box_mask = left * right * top * bottom  # (B, H, W)

        # Masked inverse
        inv_mask_boxed = inv_mask * soft_box_mask

        # Soft-argmax on inverse mask
        mask_sum = inv_mask_boxed.sum(dim=(-2, -1), keepdim=True) + 1e-6
        y_center = (inv_mask_boxed * y_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum.squeeze()
        x_center = (inv_mask_boxed * x_grid.unsqueeze(0)).sum(dim=(-2, -1)) / mask_sum.squeeze()

        point_coords = torch.stack([x_center, y_center], dim=-1).unsqueeze(1)  # (B, 1, 2)
        point_labels = torch.zeros(B, 1, device=device)

        return point_coords, point_labels

    def extract_soft_box(self, soft_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract bounding box from soft masks in a FULLY DIFFERENTIABLE manner.

        Mathematical formulation using soft-min/max:
            For soft-min: weighted average with negative temperature softmax
            x_min ≈ Σ(x_i · softmax(-x_i / τ · w_i))

            where w_i is the mask projection weight at position i.

        This provides non-zero gradients everywhere the mask is non-zero.

        Args:
            soft_mask: Soft probability mask (B, H, W)

        Returns:
            boxes: (B, 4) bounding boxes as [x1, y1, x2, y2]
        """
        B, H, W = soft_mask.shape
        device = soft_mask.device
        tau = self.box_temperature

        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)

        # Project mask to axes (soft projection)
        # y_proj[i] = max_j(mask[i,j]) approximated by logsumexp
        y_proj = torch.logsumexp(soft_mask / tau, dim=2) * tau  # (B, H) - soft max over columns
        x_proj = torch.logsumexp(soft_mask / tau, dim=1) * tau  # (B, W) - soft max over rows

        # Normalize projections to get weights
        y_weights = F.softmax(y_proj / tau, dim=1)  # (B, H)
        x_weights = F.softmax(x_proj / tau, dim=1)  # (B, W)

        # Soft-min: use negative temperature
        # x_min = Σ(x_i · softmax(-x_i · w_i / τ))
        y_min_weights = F.softmax(-y_coords.unsqueeze(0) * y_weights / tau, dim=1)  # (B, H)
        y_max_weights = F.softmax(y_coords.unsqueeze(0) * y_weights / tau, dim=1)   # (B, H)
        x_min_weights = F.softmax(-x_coords.unsqueeze(0) * x_weights / tau, dim=1)  # (B, W)
        x_max_weights = F.softmax(x_coords.unsqueeze(0) * x_weights / tau, dim=1)   # (B, W)

        # Compute soft min/max coordinates
        y1 = (y_coords.unsqueeze(0) * y_min_weights).sum(dim=1)  # (B,)
        y2 = (y_coords.unsqueeze(0) * y_max_weights).sum(dim=1)  # (B,)
        x1 = (x_coords.unsqueeze(0) * x_min_weights).sum(dim=1)  # (B,)
        x2 = (x_coords.unsqueeze(0) * x_max_weights).sum(dim=1)  # (B,)

        # Add small margin to ensure box contains object
        margin = 2.0
        y1 = torch.clamp(y1 - margin, min=0)
        y2 = torch.clamp(y2 + margin, max=H - 1)
        x1 = torch.clamp(x1 - margin, min=0)
        x2 = torch.clamp(x2 + margin, max=W - 1)

        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (B, 4)
        return boxes

    def prepare_mask_input(self, soft_mask: torch.Tensor, target_size: int = 256) -> torch.Tensor:
        """
        Prepare mask prompt for SAM.

        Mathematical formulation:
            logits = (p - 0.5) * 20 = (p * 2 - 1) * 10

        Maps [0, 1] probability to [-10, 10] logits.
        Fully differentiable via linear transformation + bilinear interpolation.

        Args:
            soft_mask: Soft probability mask (B, H, W)
            target_size: Target size for mask input (SAM uses 256x256)

        Returns:
            mask_input: (B, 1, target_size, target_size) mask prompt
        """
        # Convert to logits-like values
        # SAM expects values where positive = foreground
        mask_logits = (soft_mask * 2 - 1) * 10  # Scale to [-10, 10]

        # Resize to SAM's expected input size (bilinear is differentiable)
        mask_input = F.interpolate(
            mask_logits.unsqueeze(1),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        return mask_input

    def soft_mask_selection(self, masks: torch.Tensor, iou_predictions: torch.Tensor) -> torch.Tensor:
        """
        Select refined mask using DIFFERENTIABLE soft selection.

        Instead of argmax (non-differentiable), we use softmax-weighted combination:
            refined = Σ(mask_i · softmax(iou_i / τ))

        This allows gradients to flow through all mask candidates.

        Args:
            masks: All candidate masks (B, 3, H, W)
            iou_predictions: IoU predictions (B, 3)

        Returns:
            refined_mask: Soft-selected mask (B, H, W)
        """
        B, num_masks, H, W = masks.shape
        tau = self.selection_temperature

        # Soft selection weights via softmax
        selection_weights = F.softmax(iou_predictions / tau, dim=1)  # (B, 3)

        # Weighted combination of all masks
        # refined = Σ(w_i · mask_i)
        refined_mask = (masks * selection_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # (B, H, W)

        return refined_mask

    def forward(
        self,
        image: torch.Tensor,
        coarse_mask: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Refine coarse masks using SAM with full differentiability.

        Gradient flow:
            L_refined → refined_mask → soft_selection → SAM_decoder
                     → prompt_encoder → [points, boxes, masks]
                     → extract_soft_* → coarse_mask → TransUNet

        Args:
            image: Input image (B, 3, H, W) normalized for SAM
            coarse_mask: Coarse segmentation mask (B, H, W) with values in [0, 1]
            return_intermediate: Whether to return intermediate results

        Returns:
            Dictionary containing:
                - 'masks': Refined masks (B, H, W) - soft-selected
                - 'iou_predictions': IoU predictions (B, 3)
                - 'low_res_masks': Low resolution masks (B, 3, 256, 256)
                - 'prompts': Dictionary of prompts used (if return_intermediate)
        """
        B = image.shape[0]
        device = image.device
        original_size = coarse_mask.shape[-2:]

        # Get image embeddings (frozen, but keep in graph for proper device handling)
        with torch.set_grad_enabled(self.sam.image_encoder.training):
            input_images = torch.stack([self.sam.preprocess(img) for img in image])
            image_embeddings = self.sam.image_encoder(input_images)

        # Prepare prompts (ALL DIFFERENTIABLE)
        prompts = {}

        # Box prompts (differentiable soft extraction)
        if self.use_box_prompt:
            boxes = self.extract_soft_box(coarse_mask)
            # Scale boxes to SAM input size
            scale_h = self.sam.image_encoder.img_size / original_size[0]
            scale_w = self.sam.image_encoder.img_size / original_size[1]
            scaled_boxes = boxes.clone()
            scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]] * scale_w
            scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]] * scale_h
            prompts['boxes'] = scaled_boxes

        # Point prompts (differentiable soft-argmax)
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
            scaled_points[:, :, 0] = scaled_points[:, :, 0] * scale_w
            scaled_points[:, :, 1] = scaled_points[:, :, 1] * scale_h
            prompts['point_coords'] = scaled_points
            prompts['point_labels'] = point_labels

        # Mask prompts (differentiable linear transform + interpolation)
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

        # DIFFERENTIABLE mask selection using soft weighting
        refined_masks = self.soft_mask_selection(all_masks, all_ious)  # (B, H, W)

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
    Uses hard selection for efficiency (no gradient needed at inference).
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

            # At inference, use hard selection for best mask
            best_idx = result['iou_predictions'].argmax(dim=1)
            current_mask = torch.sigmoid(torch.stack([
                result['masks_all'][b, best_idx[b]] for b in range(image.shape[0])
            ]))

        return current_mask
