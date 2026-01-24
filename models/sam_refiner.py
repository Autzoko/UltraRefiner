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


class DifferentiableROICropper(nn.Module):
    """
    Differentiable ROI (Region of Interest) cropping for focused SAM processing.

    This module extracts the ROI from an image/mask based on the coarse mask's
    bounding box, processes at full SAM resolution (1024x1024), and pastes
    the result back. All operations are fully differentiable.

    Benefits:
    - Focuses SAM computation on the lesion area
    - Higher effective resolution for the lesion
    - Consistent distribution between Phase 2 and Phase 3
    """

    def __init__(
        self,
        target_size: int = 1024,
        expand_ratio: float = 0.2,
        box_temperature: float = 0.01,
        min_roi_size: int = 64,
    ):
        """
        Args:
            target_size: Size to resize ROI to (SAM's expected input size)
            expand_ratio: Ratio to expand the ROI box (0.2 = 20% expansion on each side)
            box_temperature: Temperature for soft bounding box extraction
            min_roi_size: Minimum ROI size in pixels (prevents too small crops)
        """
        super().__init__()
        self.target_size = target_size
        self.expand_ratio = expand_ratio
        self.box_temperature = box_temperature
        self.min_roi_size = min_roi_size

    def extract_roi_box(self, soft_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract ROI bounding box from soft mask using differentiable soft-min/max.

        Args:
            soft_mask: Soft probability mask (B, H, W)

        Returns:
            boxes: (B, 4) as [x1, y1, x2, y2] with expansion applied
        """
        B, H, W = soft_mask.shape
        device = soft_mask.device
        tau = self.box_temperature

        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)

        # Soft projection to axes
        y_proj = torch.logsumexp(soft_mask / tau, dim=2) * tau
        x_proj = torch.logsumexp(soft_mask / tau, dim=1) * tau

        # Normalize to get weights
        y_weights = F.softmax(y_proj / tau, dim=1)
        x_weights = F.softmax(x_proj / tau, dim=1)

        # Soft-min/max
        y_min_weights = F.softmax(-y_coords.unsqueeze(0) * y_weights / tau, dim=1)
        y_max_weights = F.softmax(y_coords.unsqueeze(0) * y_weights / tau, dim=1)
        x_min_weights = F.softmax(-x_coords.unsqueeze(0) * x_weights / tau, dim=1)
        x_max_weights = F.softmax(x_coords.unsqueeze(0) * x_weights / tau, dim=1)

        y1 = (y_coords.unsqueeze(0) * y_min_weights).sum(dim=1)
        y2 = (y_coords.unsqueeze(0) * y_max_weights).sum(dim=1)
        x1 = (x_coords.unsqueeze(0) * x_min_weights).sum(dim=1)
        x2 = (x_coords.unsqueeze(0) * x_max_weights).sum(dim=1)

        # Expand box by ratio
        box_h = y2 - y1
        box_w = x2 - x1
        expand_h = box_h * self.expand_ratio
        expand_w = box_w * self.expand_ratio

        y1 = y1 - expand_h
        y2 = y2 + expand_h
        x1 = x1 - expand_w
        x2 = x2 + expand_w

        # Ensure minimum size
        center_y = (y1 + y2) / 2
        center_x = (x1 + x2) / 2
        half_size = max(self.min_roi_size, 1) / 2

        # Soft minimum size enforcement
        box_h = y2 - y1
        box_w = x2 - x1
        y1 = torch.where(box_h < self.min_roi_size, center_y - half_size, y1)
        y2 = torch.where(box_h < self.min_roi_size, center_y + half_size, y2)
        x1 = torch.where(box_w < self.min_roi_size, center_x - half_size, x1)
        x2 = torch.where(box_w < self.min_roi_size, center_x + half_size, x2)

        # Clamp to image bounds
        y1 = torch.clamp(y1, min=0)
        y2 = torch.clamp(y2, max=H - 1)
        x1 = torch.clamp(x1, min=0)
        x2 = torch.clamp(x2, max=W - 1)

        return torch.stack([x1, y1, x2, y2], dim=1)

    def crop_and_resize(
        self,
        tensor: torch.Tensor,
        boxes: torch.Tensor,
        is_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable crop and resize using grid_sample.

        Args:
            tensor: Input tensor (B, C, H, W) or (B, H, W)
            boxes: ROI boxes (B, 4) as [x1, y1, x2, y2]
            is_mask: If True, use nearest interpolation for masks

        Returns:
            cropped: Cropped and resized tensor (B, C, target_size, target_size)
            crop_info: Information needed for pasting back
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, C, H, W = tensor.shape
        device = tensor.device

        # Normalize box coordinates to [-1, 1] for grid_sample
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Create sampling grid
        # grid_sample expects coordinates in [-1, 1] where -1 is left/top, 1 is right/bottom
        theta = torch.zeros(B, 2, 3, device=device)

        # Scale factors
        scale_x = (x2 - x1) / W
        scale_y = (y2 - y1) / H

        # Translation (center of ROI in normalized coords)
        trans_x = (x1 + x2) / W - 1
        trans_y = (y1 + y2) / H - 1

        theta[:, 0, 0] = scale_x
        theta[:, 1, 1] = scale_y
        theta[:, 0, 2] = trans_x
        theta[:, 1, 2] = trans_y

        # Generate grid
        grid = F.affine_grid(theta, [B, C, self.target_size, self.target_size], align_corners=False)

        # Sample
        mode = 'nearest' if is_mask else 'bilinear'
        cropped = F.grid_sample(tensor, grid, mode=mode, padding_mode='zeros', align_corners=False)

        if squeeze_output:
            cropped = cropped.squeeze(1)

        crop_info = {
            'boxes': boxes,
            'original_size': (H, W),
            'theta': theta,
        }

        return cropped, crop_info

    def paste_back(
        self,
        cropped: torch.Tensor,
        crop_info: Dict,
        background: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable paste cropped region back to full size.

        Uses a soft blending approach for differentiability.

        Args:
            cropped: Cropped tensor (B, C, target_size, target_size) or (B, target_size, target_size)
            crop_info: Information from crop_and_resize
            background: Optional background tensor. If None, uses zeros.

        Returns:
            full: Full-size tensor with ROI pasted (B, C, H, W)
        """
        if cropped.dim() == 3:
            cropped = cropped.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, C, _, _ = cropped.shape
        H, W = crop_info['original_size']
        boxes = crop_info['boxes']
        device = cropped.device

        # Create inverse affine transform
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Inverse transformation: from full image coords to cropped coords
        inv_theta = torch.zeros(B, 2, 3, device=device)

        # Inverse scale
        inv_scale_x = W / (x2 - x1 + 1e-6)
        inv_scale_y = H / (y2 - y1 + 1e-6)

        # Inverse translation
        inv_trans_x = -((x1 + x2) / W - 1) * inv_scale_x
        inv_trans_y = -((y1 + y2) / H - 1) * inv_scale_y

        inv_theta[:, 0, 0] = inv_scale_x
        inv_theta[:, 1, 1] = inv_scale_y
        inv_theta[:, 0, 2] = inv_trans_x
        inv_theta[:, 1, 2] = inv_trans_y

        # Generate grid for full size
        grid = F.affine_grid(inv_theta, [B, C, H, W], align_corners=False)

        # Sample from cropped to get full size
        full = F.grid_sample(cropped, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Create soft ROI mask for blending
        # This ensures gradients flow properly
        roi_mask = self._create_soft_roi_mask(boxes, H, W, device)

        if background is not None:
            if background.dim() == 3:
                background = background.unsqueeze(1)
            full = full * roi_mask + background * (1 - roi_mask)
        else:
            # Keep only ROI region, zero elsewhere
            full = full * roi_mask

        if squeeze_output:
            full = full.squeeze(1)

        return full

    def _create_soft_roi_mask(
        self,
        boxes: torch.Tensor,
        H: int,
        W: int,
        device: torch.device,
        edge_softness: float = 5.0
    ) -> torch.Tensor:
        """
        Create a soft ROI mask with smooth edges for differentiable blending.

        Args:
            boxes: ROI boxes (B, 4) as [x1, y1, x2, y2]
            H, W: Output size
            device: Device
            edge_softness: Softness of edges (higher = sharper)

        Returns:
            mask: Soft ROI mask (B, 1, H, W)
        """
        B = boxes.shape[0]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Expand for batch
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)

        # Soft boundaries using sigmoid
        left = torch.sigmoid(edge_softness * (x_grid - x1.view(B, 1, 1)))
        right = torch.sigmoid(edge_softness * (x2.view(B, 1, 1) - x_grid))
        top = torch.sigmoid(edge_softness * (y_grid - y1.view(B, 1, 1)))
        bottom = torch.sigmoid(edge_softness * (y2.view(B, 1, 1) - y_grid))

        mask = left * right * top * bottom
        return mask.unsqueeze(1)  # (B, 1, H, W)


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
        mask_prompt_style='gaussian',
        use_roi_crop=False,
        roi_expand_ratio=0.2,
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
            mask_prompt_style: Style for mask prompt preparation
                - 'gaussian': Apply Gaussian blur (softer boundaries, recommended)
                - 'direct': Direct conversion (sharp boundaries)
                - 'distance': Distance-weighted confidence (SDF-like)
            use_roi_crop: Whether to crop to ROI before SAM processing.
                         This focuses SAM computation on the lesion area and
                         provides higher effective resolution. Fully differentiable.
            roi_expand_ratio: Ratio to expand ROI box (0.2 = 20% on each side)
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
        self.mask_prompt_style = mask_prompt_style
        self.use_roi_crop = use_roi_crop
        self.roi_expand_ratio = roi_expand_ratio

        # Freeze components if specified
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False

        # ROI cropper for focused processing
        if use_roi_crop:
            self.roi_cropper = DifferentiableROICropper(
                target_size=self.sam.image_encoder.img_size,
                expand_ratio=roi_expand_ratio,
                box_temperature=box_temperature,
            )

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

    def prepare_mask_input(
        self,
        soft_mask: torch.Tensor,
        target_size: int = 256,
        style: str = 'gaussian'
    ) -> torch.Tensor:
        """
        Prepare mask prompt for SAM with different styles.

        Args:
            soft_mask: Soft probability mask (B, H, W) with values in [0, 1]
            target_size: Target size for mask input (SAM uses 256x256)
            style: Mask prompt style
                - 'direct': Direct conversion to logits (sharp boundaries)
                - 'gaussian': Apply Gaussian blur for softer boundaries (recommended)
                - 'distance': Distance-weighted confidence (SDF-like)

        Returns:
            mask_input: (B, 1, target_size, target_size) mask prompt
        """
        B, H, W = soft_mask.shape

        if style == 'gaussian':
            # Apply Gaussian blur for softer, more natural boundaries
            # This helps SAM generalize better to various input qualities
            kernel_size = max(3, int(H / 64) * 2 + 1)  # Adaptive kernel size
            sigma = kernel_size / 3.0

            # Create Gaussian kernel
            x = torch.arange(kernel_size, device=soft_mask.device, dtype=torch.float32)
            x = x - (kernel_size - 1) / 2
            gauss_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
            gauss_2d = gauss_2d / gauss_2d.sum()
            gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

            # Apply Gaussian blur (differentiable)
            padding = kernel_size // 2
            soft_mask_4d = soft_mask.unsqueeze(1)  # (B, 1, H, W)
            blurred_mask = F.conv2d(soft_mask_4d, gauss_2d, padding=padding)
            blurred_mask = blurred_mask.squeeze(1)  # (B, H, W)

            # Convert to logits
            mask_logits = (blurred_mask * 2 - 1) * 10  # [-10, 10]

        elif style == 'distance':
            # Distance-weighted: higher confidence near mask center, lower near boundaries
            # Approximates SDF-like behavior using soft distance transform
            # Uses iterative erosion to estimate distance

            # Compute approximate distance from boundary using soft operations
            mask_4d = soft_mask.unsqueeze(1)  # (B, 1, H, W)

            # Soft erosion to find "core" regions
            kernel_size = 5
            erosion_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=soft_mask.device)
            erosion_kernel = erosion_kernel / (kernel_size ** 2)

            # Multiple erosion passes for distance estimation
            distance_map = torch.zeros_like(soft_mask)
            current_mask = mask_4d
            for i in range(5):
                # Soft erosion: convolve and threshold
                eroded = F.conv2d(current_mask, erosion_kernel, padding=kernel_size // 2)
                eroded = torch.sigmoid((eroded - 0.5) * 10)  # Soft threshold
                distance_map = distance_map + eroded.squeeze(1)
                current_mask = eroded

            # Normalize distance map
            distance_map = distance_map / 5.0

            # Combine with original mask: high confidence in core, lower at boundaries
            confidence_mask = soft_mask * (0.5 + 0.5 * distance_map)

            # Convert to logits
            mask_logits = (confidence_mask * 2 - 1) * 10

        else:  # 'direct'
            # Direct conversion: sharp boundaries preserved
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
        image_already_normalized: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Refine coarse masks using SAM with full differentiability.

        Gradient flow:
            L_refined → refined_mask → soft_selection → SAM_decoder
                     → prompt_encoder → [points, boxes, masks]
                     → extract_soft_* → coarse_mask → TransUNet

        Args:
            image: Input image (B, 3, H, W)
                   If image_already_normalized=True: expects SAM-normalized image
                   If image_already_normalized=False: expects [0, 255] range
            coarse_mask: Coarse segmentation mask (B, H, W) with values in [0, 1]
            return_intermediate: Whether to return intermediate results
            image_already_normalized: If True, skip SAM preprocessing (avoid double normalization)

        Returns:
            Dictionary containing:
                - 'masks': Refined masks (B, H, W) - soft-selected
                - 'iou_predictions': IoU predictions (B, 3)
                - 'low_res_masks': Low resolution masks (B, 3, 256, 256)
                - 'prompts': Dictionary of prompts used (if return_intermediate)
                - 'roi_boxes': ROI boxes used (if use_roi_crop=True and return_intermediate)
        """
        B = image.shape[0]
        device = image.device
        original_size = coarse_mask.shape[-2:]

        # ROI Cropping Mode: Focus SAM computation on lesion area
        if self.use_roi_crop:
            return self._forward_with_roi(
                image, coarse_mask, return_intermediate, image_already_normalized
            )

        # Standard full-image mode
        # Get image embeddings (frozen, but keep in graph for proper device handling)
        with torch.set_grad_enabled(self.sam.image_encoder.training):
            if image_already_normalized:
                # Image is already normalized for SAM - just ensure correct size with padding
                H, W = image.shape[-2:]
                target_size = self.sam.image_encoder.img_size
                if H != target_size or W != target_size:
                    # Pad to target size if needed
                    padh = target_size - H
                    padw = target_size - W
                    input_images = F.pad(image, (0, padw, 0, padh))
                else:
                    input_images = image
            else:
                # Image is raw [0, 255] - apply SAM preprocessing
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

        # Mask prompts (differentiable, with configurable style)
        if self.use_mask_prompt:
            mask_input = self.prepare_mask_input(coarse_mask, style=self.mask_prompt_style)
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

    def _forward_with_roi(
        self,
        image: torch.Tensor,
        coarse_mask: torch.Tensor,
        return_intermediate: bool = False,
        image_already_normalized: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ROI cropping for focused SAM processing.

        This method:
        1. Extracts ROI bounding box from coarse mask
        2. Crops image and mask to ROI at full SAM resolution
        3. Processes through SAM
        4. Pastes result back to full image size

        All operations are fully differentiable.

        Args:
            Same as forward()

        Returns:
            Same as forward(), plus 'roi_boxes' if return_intermediate=True
        """
        B = image.shape[0]
        device = image.device
        original_size = coarse_mask.shape[-2:]
        target_size = self.sam.image_encoder.img_size

        # Step 1: Extract ROI boxes from coarse mask
        roi_boxes = self.roi_cropper.extract_roi_box(coarse_mask)

        # Step 2: Crop and resize image and mask to ROI
        # The cropped regions will be at full SAM resolution (1024x1024)
        image_cropped, crop_info = self.roi_cropper.crop_and_resize(
            image, roi_boxes, is_mask=False
        )
        mask_cropped, _ = self.roi_cropper.crop_and_resize(
            coarse_mask, roi_boxes, is_mask=False  # Use bilinear for soft masks
        )

        # Step 3: Get image embeddings for cropped region
        with torch.set_grad_enabled(self.sam.image_encoder.training):
            if image_already_normalized:
                input_images = image_cropped
            else:
                input_images = torch.stack([self.sam.preprocess(img) for img in image_cropped])

            image_embeddings = self.sam.image_encoder(input_images)

        # Step 4: Extract prompts from CROPPED mask (coordinates are in cropped space)
        prompts = {}
        cropped_size = (target_size, target_size)

        # Box prompts - in cropped space, the box should roughly fill the image
        if self.use_box_prompt:
            boxes = self.extract_soft_box(mask_cropped)
            prompts['boxes'] = boxes

        # Point prompts
        if self.use_point_prompt:
            point_coords, point_labels = self.extract_soft_points(mask_cropped, self.num_points)

            if self.add_negative_point and self.use_box_prompt:
                neg_coords, neg_labels = self.extract_soft_negative_points(mask_cropped, boxes)
                point_coords = torch.cat([point_coords, neg_coords], dim=1)
                point_labels = torch.cat([point_labels, neg_labels], dim=1)

            prompts['point_coords'] = point_coords
            prompts['point_labels'] = point_labels

        # Mask prompts
        if self.use_mask_prompt:
            mask_input = self.prepare_mask_input(mask_cropped, style=self.mask_prompt_style)
            prompts['mask_inputs'] = mask_input

        # Step 5: Run through SAM decoder
        all_masks = []
        all_ious = []
        all_low_res = []

        for b in range(B):
            curr_embedding = image_embeddings[b].unsqueeze(0)

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

            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=point_input,
                boxes=box_input,
                masks=mask_input,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            # Upscale to cropped size (1024x1024)
            masks = F.interpolate(
                low_res_masks,
                size=cropped_size,
                mode='bilinear',
                align_corners=False
            )

            all_masks.append(masks.squeeze(0))
            all_ious.append(iou_predictions.squeeze(0))
            all_low_res.append(low_res_masks.squeeze(0))

        all_masks = torch.stack(all_masks)  # (B, 3, 1024, 1024)
        all_ious = torch.stack(all_ious)  # (B, 3)
        all_low_res = torch.stack(all_low_res)  # (B, 3, 256, 256)

        # Step 6: Soft mask selection in cropped space
        refined_masks_cropped = self.soft_mask_selection(all_masks, all_ious)  # (B, 1024, 1024)

        # Step 7: Paste back to original size
        # Create zero background for areas outside ROI
        background = torch.zeros(B, *original_size, device=device)
        refined_masks = self.roi_cropper.paste_back(
            refined_masks_cropped, crop_info, background=None
        )

        result = {
            'masks': refined_masks,
            'masks_all': all_masks,  # In cropped space
            'iou_predictions': all_ious,
            'low_res_masks': all_low_res,
        }

        if return_intermediate:
            result['prompts'] = prompts
            result['image_embeddings'] = image_embeddings
            result['roi_boxes'] = roi_boxes
            result['crop_info'] = crop_info

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
