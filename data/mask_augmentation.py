"""
Comprehensive Mask Augmentation for Phase 2 SAM Refinement Training.

This module generates pseudo-coarse masks from ground truth to simulate realistic
TransUNet failure modes. The augmented masks train SAM to refine various types of
segmentation errors.

Pipeline:
1. Start from GT mask
2. Sample one primary error type (12 types)
3. Optionally apply 0-2 secondary perturbations
4. Optionally convert to soft pseudo-probability map
5. Generate prompts matching Phase 3 logic

Primary Error Types:
1. Identity/Near-Perfect (10-20%)
2. Over-Segmentation (15-20%)
3. Giant Over-Segmentation (8-12%)
4. Under-Segmentation (15-20%)
5. Missing Chunk (10-15%)
6. Internal Holes (8-12%)
7. Bridge/Adhesion (8-12%)
8. False Positive Islands (15-20%)
9. Fragmentation (8-12%)
10. Shift/Wrong Location (8-12%)
11. Empty/Near-Empty (3-6%)
12. Noise-Only Scatter (2-4%)
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from typing import Tuple, Optional, Dict, List, Union
import random


class MaskAugmentor:
    """
    Comprehensive mask augmentation for simulating TransUNet failure modes.

    Usage:
        augmentor = MaskAugmentor()
        coarse_mask, aug_info = augmentor(gt_mask)
    """

    # Default probabilities for each error type (should sum to ~1.0)
    DEFAULT_ERROR_PROBS = {
        'identity': 0.15,           # 10-20%
        'over_segmentation': 0.17,  # 15-20%
        'giant_over_seg': 0.10,     # 8-12%
        'under_segmentation': 0.17, # 15-20%
        'missing_chunk': 0.12,      # 10-15%
        'internal_holes': 0.10,     # 8-12%
        'bridge_adhesion': 0.10,    # 8-12%
        'false_positive_islands': 0.17,  # 15-20%
        'fragmentation': 0.10,      # 8-12%
        'shift_wrong_location': 0.10,    # 8-12%
        'empty_prediction': 0.04,   # 3-6%
        'noise_only_scatter': 0.03, # 2-4%
    }

    def __init__(
        self,
        error_probs: Optional[Dict[str, float]] = None,
        secondary_prob: float = 0.5,
        soft_mask_prob: float = 0.8,
        soft_mask_temperature: Tuple[float, float] = (2.0, 8.0),
        seed: Optional[int] = None,
    ):
        """
        Args:
            error_probs: Custom probabilities for each error type. If None, uses defaults.
            secondary_prob: Probability of applying secondary perturbations (0-1).
            soft_mask_prob: Probability of converting to soft mask (0-1).
            soft_mask_temperature: Temperature range for soft mask conversion.
            seed: Random seed for reproducibility.
        """
        self.error_probs = error_probs or self.DEFAULT_ERROR_PROBS.copy()
        self.secondary_prob = secondary_prob
        self.soft_mask_prob = soft_mask_prob
        self.soft_mask_temperature = soft_mask_temperature

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Normalize probabilities
        total = sum(self.error_probs.values())
        self.error_probs = {k: v / total for k, v in self.error_probs.items()}

        # Error type functions mapping
        self.error_functions = {
            'identity': self._identity,
            'over_segmentation': self._over_segmentation,
            'giant_over_seg': self._giant_over_segmentation,
            'under_segmentation': self._under_segmentation,
            'missing_chunk': self._missing_chunk,
            'internal_holes': self._internal_holes,
            'bridge_adhesion': self._bridge_adhesion,
            'false_positive_islands': self._false_positive_islands,
            'fragmentation': self._fragmentation,
            'shift_wrong_location': self._shift_wrong_location,
            'empty_prediction': self._empty_prediction,
            'noise_only_scatter': self._noise_only_scatter,
        }

    def __call__(
        self,
        gt_mask: np.ndarray,
        force_error_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate augmented coarse mask from ground truth.

        Args:
            gt_mask: Binary ground truth mask (H, W), values in {0, 1}.
            force_error_type: If specified, use this error type instead of sampling.

        Returns:
            coarse_mask: Augmented coarse mask (H, W), values in [0, 1].
            aug_info: Dictionary with augmentation details.
        """
        # Ensure binary mask
        gt_mask = (gt_mask > 0.5).astype(np.float32)

        # Check for empty GT
        if gt_mask.sum() < 10:
            # GT is empty, return empty mask
            return gt_mask.copy(), {'error_type': 'empty_gt', 'secondary': [], 'soft': False}

        # Sample error type
        if force_error_type is not None:
            error_type = force_error_type
        else:
            error_types = list(self.error_probs.keys())
            probs = list(self.error_probs.values())
            error_type = np.random.choice(error_types, p=probs)

        # Apply primary error
        error_func = self.error_functions[error_type]
        coarse_mask = error_func(gt_mask)

        # Apply secondary perturbations (0-2)
        secondary_applied = []
        if np.random.random() < self.secondary_prob:
            num_secondary = np.random.randint(1, 3)  # 1 or 2
            secondary_types = ['boundary_jitter', 'threshold_fluctuation']
            selected = np.random.choice(secondary_types, size=min(num_secondary, len(secondary_types)), replace=False)

            for sec_type in selected:
                if sec_type == 'boundary_jitter':
                    coarse_mask = self._boundary_jitter(coarse_mask)
                elif sec_type == 'threshold_fluctuation':
                    coarse_mask = self._threshold_fluctuation(coarse_mask)
                secondary_applied.append(sec_type)

        # Convert to soft mask
        is_soft = False
        if np.random.random() < self.soft_mask_prob:
            temperature = np.random.uniform(*self.soft_mask_temperature)
            coarse_mask = self._to_soft_mask(coarse_mask, temperature)
            is_soft = True

        # Compute Dice score
        dice = self._compute_dice(gt_mask, coarse_mask)

        aug_info = {
            'error_type': error_type,
            'secondary': secondary_applied,
            'soft': is_soft,
            'dice': dice,
        }

        return coarse_mask, aug_info

    # =========================================================================
    # PRIMARY ERROR TYPES
    # =========================================================================

    def _identity(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Identity / Near-Perfect Case.
        Simulates coarse predictions that are already correct.
        """
        choice = np.random.random()

        if choice < 0.3:
            # Use GT directly
            return gt_mask.copy()
        elif choice < 0.6:
            # Very small boundary jitter (1-2 pixels)
            jitter = np.random.randint(1, 3)
            if np.random.random() < 0.5:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*jitter+1, 2*jitter+1))
                return cv2.dilate(gt_mask, kernel, iterations=1)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*jitter+1, 2*jitter+1))
                return cv2.erode(gt_mask, kernel, iterations=1)
        else:
            # Minimal dilation/erosion (1-2 pixels)
            radius = np.random.randint(1, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            if np.random.random() < 0.5:
                return cv2.dilate(gt_mask, kernel, iterations=1)
            else:
                result = cv2.erode(gt_mask, kernel, iterations=1)
                # Ensure we don't completely remove the mask
                if result.sum() < gt_mask.sum() * 0.8:
                    return gt_mask.copy()
                return result

    def _over_segmentation(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Over-Segmentation (Moderate Expansion).
        Simulates prediction expanding beyond GT boundaries.
        Area increase: 1.2x-3x GT.
        """
        # Compute target area increase
        target_ratio = np.random.uniform(1.2, 3.0)
        gt_area = gt_mask.sum()
        target_area = gt_area * target_ratio

        # Binary search for dilation radius
        result = gt_mask.copy()
        for radius in range(3, 20):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            dilated = cv2.dilate(gt_mask, kernel, iterations=1)
            if dilated.sum() >= target_area:
                result = dilated
                break
            result = dilated

        # Optionally apply directional expansion
        if np.random.random() < 0.3:
            result = self._directional_expand(result, gt_mask)

        return result

    def _giant_over_segmentation(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Giant Over-Segmentation / Background Takeover.
        Simulates extreme cases where large tissue regions are predicted as tumor.
        Area increase: 3x-20x GT.
        """
        gt_area = gt_mask.sum()
        target_ratio = np.random.uniform(3.0, 20.0)
        target_area = gt_area * target_ratio

        # Get bounding box for scale reference
        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            return gt_mask.copy()

        y1, x1, y2, x2 = bbox
        bbox_size = max(y2 - y1, x2 - x1)

        # Strong dilation
        max_radius = int(bbox_size * 0.5)
        result = gt_mask.copy()

        for radius in range(5, max(max_radius, 30)):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            dilated = cv2.dilate(gt_mask, kernel, iterations=1)
            if dilated.sum() >= target_area:
                result = dilated
                break
            result = dilated

        # Add irregular outward blobs
        if np.random.random() < 0.5:
            result = self._add_irregular_blobs(result, gt_mask, num_blobs=np.random.randint(2, 6))

        return result

    def _under_segmentation(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Under-Segmentation (Shrinkage).
        Simulates prediction missing outer boundary regions.
        Area decrease: 0.4x-0.9x GT.
        """
        target_ratio = np.random.uniform(0.4, 0.9)
        gt_area = gt_mask.sum()
        target_area = gt_area * target_ratio

        result = gt_mask.copy()
        for radius in range(2, 15):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            eroded = cv2.erode(gt_mask, kernel, iterations=1)
            if eroded.sum() <= target_area:
                result = eroded
                break
            if eroded.sum() < gt_area * 0.1:  # Don't erode too much
                break
            result = eroded

        # Ensure we don't completely remove the mask
        if result.sum() < gt_area * 0.1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            result = cv2.erode(gt_mask, kernel, iterations=1)

        return result

    def _missing_chunk(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Missing Chunk / Boundary Cutout.
        Simulates prediction missing a part of the lesion.
        Cutout area: 5-30% of GT.
        """
        cutout_ratio = np.random.uniform(0.05, 0.30)

        # Get centroid and bounding box
        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            return gt_mask.copy()

        y1, x1, y2, x2 = bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        result = gt_mask.copy()
        H, W = gt_mask.shape

        # Create wedge-shaped or irregular cutout
        if np.random.random() < 0.6:
            # Wedge-shaped cutout
            angle_start = np.random.uniform(0, 2 * np.pi)
            angle_span = np.random.uniform(np.pi / 6, np.pi / 2)

            yy, xx = np.ogrid[:H, :W]
            angles = np.arctan2(yy - cy, xx - cx)

            # Normalize angles to [0, 2*pi]
            angles = (angles + 2 * np.pi) % (2 * np.pi)
            angle_end = (angle_start + angle_span) % (2 * np.pi)

            if angle_start < angle_end:
                wedge_mask = (angles >= angle_start) & (angles <= angle_end)
            else:
                wedge_mask = (angles >= angle_start) | (angles <= angle_end)

            result[wedge_mask & (gt_mask > 0.5)] = 0
        else:
            # Irregular blob cutout
            gt_area = gt_mask.sum()
            cutout_size = int(np.sqrt(gt_area * cutout_ratio))

            # Find a point on the boundary
            boundary = cv2.Canny((gt_mask * 255).astype(np.uint8), 100, 200)
            boundary_points = np.where(boundary > 0)
            if len(boundary_points[0]) > 0:
                idx = np.random.randint(len(boundary_points[0]))
                cut_y, cut_x = boundary_points[0][idx], boundary_points[1][idx]

                # Create elliptical cutout
                cutout = np.zeros_like(gt_mask)
                cv2.ellipse(cutout, (cut_x, cut_y),
                           (cutout_size, cutout_size // 2),
                           np.random.uniform(0, 360), 0, 360, 1, -1)
                result[cutout > 0] = 0

        return result

    def _internal_holes(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Internal Holes / Hollow Lesion.
        Simulates internal missed regions.
        1-3 holes, each 2-20% of GT area.
        """
        num_holes = np.random.randint(1, 4)
        result = gt_mask.copy()
        gt_area = gt_mask.sum()

        # Get interior points (eroded mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        interior = cv2.erode(gt_mask, kernel, iterations=2)
        interior_points = np.where(interior > 0.5)

        if len(interior_points[0]) < 10:
            return result

        for _ in range(num_holes):
            hole_ratio = np.random.uniform(0.02, 0.20)
            hole_area = int(gt_area * hole_ratio)
            hole_radius = int(np.sqrt(hole_area / np.pi))

            # Random interior point
            idx = np.random.randint(len(interior_points[0]))
            cy, cx = interior_points[0][idx], interior_points[1][idx]

            # Create elliptical hole
            hole = np.zeros_like(gt_mask)
            aspect = np.random.uniform(0.5, 2.0)
            cv2.ellipse(hole, (cx, cy),
                       (hole_radius, int(hole_radius * aspect)),
                       np.random.uniform(0, 360), 0, 360, 1, -1)

            result[hole > 0] = 0

        return result

    def _bridge_adhesion(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Bridge / Adhesion.
        Simulates erroneous connection between tumor and nearby structures.
        Thin band (2-6 px width, 10-60 px length).
        """
        result = gt_mask.copy()
        H, W = gt_mask.shape

        # Get boundary points
        boundary = cv2.Canny((gt_mask * 255).astype(np.uint8), 100, 200)
        boundary_points = np.where(boundary > 0)

        if len(boundary_points[0]) < 10:
            return result

        # Select random boundary point
        idx = np.random.randint(len(boundary_points[0]))
        start_y, start_x = boundary_points[0][idx], boundary_points[1][idx]

        # Determine outward direction (away from centroid)
        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            return result
        y1, x1, y2, x2 = bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        # Direction away from centroid
        dy = start_y - cy
        dx = start_x - cx
        length = np.sqrt(dy**2 + dx**2) + 1e-6
        dy, dx = dy / length, dx / length

        # Add some randomness to direction
        angle_offset = np.random.uniform(-np.pi/4, np.pi/4)
        cos_a, sin_a = np.cos(angle_offset), np.sin(angle_offset)
        new_dx = dx * cos_a - dy * sin_a
        new_dy = dx * sin_a + dy * cos_a
        dx, dy = new_dx, new_dy

        # Bridge parameters
        bridge_length = np.random.randint(10, 60)
        bridge_width = np.random.randint(2, 7)

        # Draw bridge
        end_x = int(start_x + dx * bridge_length)
        end_y = int(start_y + dy * bridge_length)
        end_x = np.clip(end_x, 0, W - 1)
        end_y = np.clip(end_y, 0, H - 1)

        cv2.line(result, (start_x, start_y), (end_x, end_y), 1, bridge_width)

        # Optionally add false positive blob at end
        if np.random.random() < 0.5:
            blob_radius = np.random.randint(5, 20)
            cv2.circle(result, (end_x, end_y), blob_radius, 1, -1)

        return result

    def _false_positive_islands(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        False Positive Islands / Multi-Component Noise.
        Simulates multiple disconnected predictions.
        Keep GT, add 1-5 blobs (or 5-30 in severe cases).
        Blob area: 0.2%-10% of GT.
        """
        result = gt_mask.copy()
        H, W = gt_mask.shape
        gt_area = gt_mask.sum()

        # Determine severity
        if np.random.random() < 0.3:
            num_blobs = np.random.randint(5, 30)  # Severe case
        else:
            num_blobs = np.random.randint(1, 6)   # Normal case

        # Get bounding box for spatial reference
        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            return result
        y1, x1, y2, x2 = bbox

        # Expand search region
        margin = max(y2 - y1, x2 - x1) // 2
        search_y1 = max(0, y1 - margin)
        search_y2 = min(H, y2 + margin)
        search_x1 = max(0, x1 - margin)
        search_x2 = min(W, x2 + margin)

        for _ in range(num_blobs):
            blob_ratio = np.random.uniform(0.002, 0.10)
            blob_area = int(gt_area * blob_ratio)
            blob_radius = max(2, int(np.sqrt(blob_area / np.pi)))

            # Random position in search region (avoiding GT)
            for _ in range(10):  # Max attempts
                cx = np.random.randint(search_x1, search_x2)
                cy = np.random.randint(search_y1, search_y2)

                # Check if position is outside GT (with margin)
                if gt_mask[max(0, cy-5):min(H, cy+5), max(0, cx-5):min(W, cx+5)].sum() < 5:
                    break

            # Create elliptical blob
            aspect = np.random.uniform(0.5, 2.0)
            cv2.ellipse(result, (cx, cy),
                       (blob_radius, int(blob_radius * aspect)),
                       np.random.uniform(0, 360), 0, 360, 1, -1)

        return result

    def _fragmentation(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Fragmentation / Lesion Split.
        Simulates lesion broken into pieces.
        Insert 1-5 narrow black strips cutting through GT.
        Width: 2-8 px.
        """
        result = gt_mask.copy()
        H, W = gt_mask.shape

        num_strips = np.random.randint(1, 6)

        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            return result
        y1, x1, y2, x2 = bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        for _ in range(num_strips):
            strip_width = np.random.randint(2, 9)

            # Random angle
            angle = np.random.uniform(0, np.pi)

            # Create line passing through or near centroid
            offset = np.random.randint(-20, 20)
            length = max(y2 - y1, x2 - x1) + 20

            dx = np.cos(angle) * length / 2
            dy = np.sin(angle) * length / 2

            start_x = int(cx + offset * np.sin(angle) - dx)
            start_y = int(cy - offset * np.cos(angle) - dy)
            end_x = int(cx + offset * np.sin(angle) + dx)
            end_y = int(cy - offset * np.cos(angle) + dy)

            # Draw black strip
            cv2.line(result, (start_x, start_y), (end_x, end_y), 0, strip_width)

        # Optionally combine with mild erosion
        if np.random.random() < 0.3:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result = cv2.erode(result, kernel, iterations=1)

        return result

    def _shift_wrong_location(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Shift / Wrong Location.
        Simulates prediction displaced from the correct location.
        Translation: 5-30% of bounding box size.
        Optional rotation: +-5-20 degrees.
        """
        H, W = gt_mask.shape

        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            return gt_mask.copy()
        y1, x1, y2, x2 = bbox
        bbox_h, bbox_w = y2 - y1, x2 - x1

        # Translation
        shift_ratio = np.random.uniform(0.05, 0.30)
        shift_x = int(bbox_w * shift_ratio * np.random.choice([-1, 1]))
        shift_y = int(bbox_h * shift_ratio * np.random.choice([-1, 1]))

        # Create translation matrix
        M_translate = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        result = cv2.warpAffine(gt_mask, M_translate, (W, H))

        # Optional rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(5, 20) * np.random.choice([-1, 1])
            cy, cx = (y1 + y2) // 2 + shift_y, (x1 + x2) // 2 + shift_x
            M_rotate = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            result = cv2.warpAffine(result, M_rotate, (W, H))

        return result

    def _empty_prediction(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Empty / Near-Empty Prediction.
        Simulates complete or near complete miss.
        """
        H, W = gt_mask.shape

        if np.random.random() < 0.5:
            # Completely empty
            return np.zeros((H, W), dtype=np.float32)
        else:
            # Retain only 1-5% small fragment
            retain_ratio = np.random.uniform(0.01, 0.05)
            gt_area = gt_mask.sum()
            target_area = gt_area * retain_ratio

            # Aggressive erosion
            result = gt_mask.copy()
            for radius in range(3, 30):
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
                eroded = cv2.erode(gt_mask, kernel, iterations=1)
                if eroded.sum() <= target_area:
                    result = eroded
                    break
                if eroded.sum() < 10:
                    result = eroded
                    break
                result = eroded

            return result

    def _noise_only_scatter(self, gt_mask: np.ndarray) -> np.ndarray:
        """
        Noise-Only Scatter.
        Simulates completely scattered false positives with no true object.
        Generate 10-50 tiny blobs (0.1-1% GT area) with no GT overlap.
        """
        H, W = gt_mask.shape
        gt_area = gt_mask.sum()

        result = np.zeros((H, W), dtype=np.float32)
        num_blobs = np.random.randint(10, 50)

        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            # No GT, scatter across entire image
            for _ in range(num_blobs):
                cx = np.random.randint(10, W - 10)
                cy = np.random.randint(10, H - 10)
                radius = np.random.randint(2, 8)
                cv2.circle(result, (cx, cy), radius, 1, -1)
            return result

        y1, x1, y2, x2 = bbox

        # Expand search region
        margin = max(y2 - y1, x2 - x1)
        search_y1 = max(10, y1 - margin)
        search_y2 = min(H - 10, y2 + margin)
        search_x1 = max(10, x1 - margin)
        search_x2 = min(W - 10, x2 + margin)

        for _ in range(num_blobs):
            blob_ratio = np.random.uniform(0.001, 0.01)
            blob_radius = max(2, int(np.sqrt(gt_area * blob_ratio / np.pi)))

            # Random position avoiding GT
            for _ in range(10):
                cx = np.random.randint(search_x1, search_x2)
                cy = np.random.randint(search_y1, search_y2)

                # Check no overlap with GT
                if gt_mask[max(0, cy-blob_radius):min(H, cy+blob_radius),
                          max(0, cx-blob_radius):min(W, cx+blob_radius)].sum() < 1:
                    break

            cv2.circle(result, (cx, cy), blob_radius, 1, -1)

        return result

    # =========================================================================
    # SECONDARY PERTURBATIONS
    # =========================================================================

    def _boundary_jitter(self, mask: np.ndarray, max_jitter: int = 5) -> np.ndarray:
        """
        Apply small random boundary distortions.
        """
        jitter = np.random.randint(1, max_jitter + 1)

        # Create morphological noise
        if np.random.random() < 0.5:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*jitter+1, 2*jitter+1))
            if np.random.random() < 0.5:
                result = cv2.dilate(mask, kernel, iterations=1)
            else:
                result = cv2.erode(mask, kernel, iterations=1)
        else:
            # Random morphological opening/closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*jitter+1, 2*jitter+1))
            if np.random.random() < 0.5:
                result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            else:
                result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return result

    def _threshold_fluctuation(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply very small dilation/erosion to simulate threshold instability.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if np.random.random() < 0.5:
            return cv2.dilate(mask, kernel, iterations=1)
        else:
            return cv2.erode(mask, kernel, iterations=1)

    def _to_soft_mask(self, mask: np.ndarray, temperature: float = 5.0) -> np.ndarray:
        """
        Convert binary mask to soft pseudo-probability map using distance transform.

        P(x) = sigmoid(d(x) / temperature)

        where d(x) is the signed distance to the boundary:
        - Positive inside the mask
        - Negative outside the mask
        """
        binary_mask = (mask > 0.5).astype(np.uint8)

        if binary_mask.sum() < 1:
            return mask.astype(np.float32)

        # Compute distance transforms
        dist_inside = distance_transform_edt(binary_mask)
        dist_outside = distance_transform_edt(1 - binary_mask)

        # Signed distance: positive inside, negative outside
        signed_dist = dist_inside - dist_outside

        # Apply sigmoid with temperature
        soft_mask = 1.0 / (1.0 + np.exp(-signed_dist / temperature))

        return soft_mask.astype(np.float32)

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================

    def _get_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box (y1, x1, y2, x2) of mask."""
        nonzero = np.where(mask > 0.5)
        if len(nonzero[0]) == 0:
            return None
        y1, y2 = nonzero[0].min(), nonzero[0].max() + 1
        x1, x2 = nonzero[1].min(), nonzero[1].max() + 1
        return y1, x1, y2, x2

    def _directional_expand(self, mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
        """Apply directional expansion to one side."""
        H, W = mask.shape

        bbox = self._get_bbox(gt_mask)
        if bbox is None:
            return mask
        y1, x1, y2, x2 = bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        # Random direction
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        expand_amount = np.random.randint(5, 20)

        result = mask.copy()

        if direction == 'up':
            kernel = np.ones((expand_amount, 1), dtype=np.uint8)
            upper_half = result.copy()
            upper_half[cy:, :] = 0
            expanded = cv2.dilate(upper_half, kernel, iterations=1)
            result = np.maximum(result, expanded)
        elif direction == 'down':
            kernel = np.ones((expand_amount, 1), dtype=np.uint8)
            lower_half = result.copy()
            lower_half[:cy, :] = 0
            expanded = cv2.dilate(lower_half, kernel, iterations=1)
            result = np.maximum(result, expanded)
        elif direction == 'left':
            kernel = np.ones((1, expand_amount), dtype=np.uint8)
            left_half = result.copy()
            left_half[:, cx:] = 0
            expanded = cv2.dilate(left_half, kernel, iterations=1)
            result = np.maximum(result, expanded)
        elif direction == 'right':
            kernel = np.ones((1, expand_amount), dtype=np.uint8)
            right_half = result.copy()
            right_half[:, :cx] = 0
            expanded = cv2.dilate(right_half, kernel, iterations=1)
            result = np.maximum(result, expanded)

        return result

    def _add_irregular_blobs(self, mask: np.ndarray, gt_mask: np.ndarray, num_blobs: int = 3) -> np.ndarray:
        """Add irregular outward blobs connected to the mask."""
        H, W = mask.shape
        result = mask.copy()

        # Get boundary points
        boundary = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
        boundary_points = np.where(boundary > 0)

        if len(boundary_points[0]) < 10:
            return result

        gt_area = gt_mask.sum()

        for _ in range(num_blobs):
            # Random boundary point
            idx = np.random.randint(len(boundary_points[0]))
            by, bx = boundary_points[0][idx], boundary_points[1][idx]

            # Blob parameters
            blob_ratio = np.random.uniform(0.05, 0.3)
            blob_radius = int(np.sqrt(gt_area * blob_ratio / np.pi))

            # Outward offset
            bbox = self._get_bbox(gt_mask)
            if bbox:
                y1, x1, y2, x2 = bbox
                cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
                dy, dx = by - cy, bx - cx
                length = np.sqrt(dy**2 + dx**2) + 1e-6
                dy, dx = dy / length, dx / length

                offset = np.random.randint(5, 15)
                blob_x = int(bx + dx * offset)
                blob_y = int(by + dy * offset)
                blob_x = np.clip(blob_x, blob_radius, W - blob_radius - 1)
                blob_y = np.clip(blob_y, blob_radius, H - blob_radius - 1)
            else:
                blob_x, blob_y = bx, by

            # Draw irregular blob
            aspect = np.random.uniform(0.5, 2.0)
            cv2.ellipse(result, (blob_x, blob_y),
                       (blob_radius, int(blob_radius * aspect)),
                       np.random.uniform(0, 360), 0, 360, 1, -1)

        return result

    def _compute_dice(self, gt: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> float:
        """Compute Dice score."""
        gt_binary = (gt > threshold).astype(np.float32)
        pred_binary = (pred > threshold).astype(np.float32)

        intersection = (gt_binary * pred_binary).sum()
        union = gt_binary.sum() + pred_binary.sum()

        if union < 1e-6:
            return 1.0 if gt_binary.sum() < 1e-6 else 0.0

        return (2 * intersection + 1e-6) / (union + 1e-6)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

AUGMENTOR_PRESETS = {
    'default': MaskAugmentor.DEFAULT_ERROR_PROBS,

    'mild': {
        'identity': 0.30,
        'over_segmentation': 0.15,
        'giant_over_seg': 0.02,
        'under_segmentation': 0.15,
        'missing_chunk': 0.10,
        'internal_holes': 0.08,
        'bridge_adhesion': 0.05,
        'false_positive_islands': 0.08,
        'fragmentation': 0.05,
        'shift_wrong_location': 0.02,
        'empty_prediction': 0.00,
        'noise_only_scatter': 0.00,
    },

    'severe': {
        'identity': 0.05,
        'over_segmentation': 0.10,
        'giant_over_seg': 0.20,
        'under_segmentation': 0.10,
        'missing_chunk': 0.10,
        'internal_holes': 0.10,
        'bridge_adhesion': 0.08,
        'false_positive_islands': 0.10,
        'fragmentation': 0.07,
        'shift_wrong_location': 0.05,
        'empty_prediction': 0.03,
        'noise_only_scatter': 0.02,
    },

    'boundary_focus': {
        'identity': 0.15,
        'over_segmentation': 0.25,
        'giant_over_seg': 0.05,
        'under_segmentation': 0.25,
        'missing_chunk': 0.05,
        'internal_holes': 0.05,
        'bridge_adhesion': 0.05,
        'false_positive_islands': 0.05,
        'fragmentation': 0.05,
        'shift_wrong_location': 0.05,
        'empty_prediction': 0.00,
        'noise_only_scatter': 0.00,
    },

    'structural': {
        'identity': 0.10,
        'over_segmentation': 0.05,
        'giant_over_seg': 0.05,
        'under_segmentation': 0.05,
        'missing_chunk': 0.20,
        'internal_holes': 0.20,
        'bridge_adhesion': 0.10,
        'false_positive_islands': 0.05,
        'fragmentation': 0.15,
        'shift_wrong_location': 0.05,
        'empty_prediction': 0.00,
        'noise_only_scatter': 0.00,
    },
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_augmentor(
    preset: str = 'default',
    **kwargs
) -> MaskAugmentor:
    """
    Create augmentor with preset configurations.

    Presets:
        'default': Balanced distribution across all error types
        'mild': More identity/near-perfect, less severe errors
        'severe': More severe errors (giant over-seg, empty, scatter)
        'boundary_focus': Focus on boundary errors (over/under-seg, jitter)
        'structural': Focus on structural errors (holes, fragmentation, missing)
    """
    error_probs = AUGMENTOR_PRESETS.get(preset, AUGMENTOR_PRESETS['default'])
    return MaskAugmentor(error_probs=error_probs, **kwargs)


def augment_mask(
    gt_mask: np.ndarray,
    augmentor: Optional[MaskAugmentor] = None,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to augment a single mask.

    Args:
        gt_mask: Ground truth mask (H, W).
        augmentor: MaskAugmentor instance. If None, creates default.
        **kwargs: Arguments passed to MaskAugmentor if creating new instance.

    Returns:
        coarse_mask: Augmented mask.
        aug_info: Augmentation information.
    """
    if augmentor is None:
        augmentor = MaskAugmentor(**kwargs)

    return augmentor(gt_mask)
