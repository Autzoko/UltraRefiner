"""
SDF-Based Mask Augmentation for SAM Refiner Training

This module implements mathematically grounded mask augmentation by operating
in the continuous Signed Distance Function (SDF) domain rather than directly
on binary masks.

Key advantages over pixel-level augmentation:
1. Smooth, anatomically plausible deformations
2. Explicit control over boundary displacement magnitude
3. Smoothness and topological constraints
4. No unrealistic pixel-level artifacts

Mathematical Framework:
- GT mask M → SDF φ (zero level set defines contour)
- Perturbation: φ' = φ + δ(x,y)
- Where δ(x,y) = c (global offset) + G(x,y) (Gaussian random field)
- Thresholding: M' = (φ' < 0)

The perturbation field δ(x,y) can be decomposed into:
1. Global offset c: uniform erosion (c > 0) or dilation (c < 0)
2. Low-frequency GRF: spatially varying boundary shifts
3. Localized perturbations: holes, attachments, bridges

Regularization constraints:
- Total variation: limits boundary roughness
- Curvature constraint: preserves smooth contours
- Area constraint: controls size deviation
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.fft import fft2, ifft2
from typing import Tuple, Optional, Dict
import warnings


class SDFAugmentor:
    """
    SDF-based mask augmentation with explicit geometric control.

    All perturbations are performed in the continuous SDF domain,
    ensuring smooth and anatomically plausible deformations.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    # =========================================================================
    # SDF Computation
    # =========================================================================

    def mask_to_sdf(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert binary mask to Signed Distance Function.

        The SDF φ(x,y) is defined as:
        - φ < 0 inside the object
        - φ = 0 on the boundary (zero level set)
        - φ > 0 outside the object

        Args:
            mask: Binary mask (H, W) with values in {0, 1} or [0, 1]

        Returns:
            sdf: Signed distance function (H, W)
        """
        binary = (mask > 0.5).astype(np.float64)

        # Handle empty mask
        if np.sum(binary) == 0:
            return np.ones_like(mask, dtype=np.float64) * 1000.0

        # Handle full mask
        if np.sum(binary) == binary.size:
            return np.ones_like(mask, dtype=np.float64) * -1000.0

        # Compute unsigned distance transforms
        dist_inside = distance_transform_edt(binary)
        dist_outside = distance_transform_edt(1 - binary)

        # Signed distance: negative inside, positive outside
        sdf = dist_outside - dist_inside

        return sdf

    def sdf_to_mask(self, sdf: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Convert SDF back to binary mask by thresholding.

        Args:
            sdf: Signed distance function (H, W)
            threshold: Level set threshold (default 0.0 = boundary)

        Returns:
            mask: Binary mask (H, W)
        """
        return (sdf < threshold).astype(np.float32)

    # =========================================================================
    # Gaussian Random Field Generation
    # =========================================================================

    def generate_gaussian_random_field(
        self,
        shape: Tuple[int, int],
        correlation_length: float,
        amplitude: float
    ) -> np.ndarray:
        """
        Generate a low-frequency Gaussian Random Field (GRF).

        The GRF is generated in Fourier space with a power spectrum
        P(k) ∝ exp(-k²σ²/2) where σ is the correlation length.

        This produces smooth, spatially correlated perturbations
        that mimic realistic boundary uncertainty.

        Args:
            shape: Output shape (H, W)
            correlation_length: Spatial correlation length (larger = smoother)
            amplitude: Standard deviation of the field

        Returns:
            grf: Gaussian random field (H, W)
        """
        H, W = shape

        # Create frequency grids
        fy = np.fft.fftfreq(H)
        fx = np.fft.fftfreq(W)
        FY, FX = np.meshgrid(fy, fx, indexing='ij')

        # Frequency magnitude
        k_squared = FX**2 + FY**2

        # Power spectrum: Gaussian decay in frequency domain
        # Correlation length σ controls the smoothness
        sigma = correlation_length / (2 * np.pi)
        power_spectrum = np.exp(-k_squared / (2 * sigma**2 + 1e-10))

        # Generate complex noise in Fourier space
        noise_real = self.rng.randn(H, W)
        noise_imag = self.rng.randn(H, W)
        noise_fourier = noise_real + 1j * noise_imag

        # Apply power spectrum filter
        filtered_fourier = noise_fourier * np.sqrt(power_spectrum)

        # Transform back to spatial domain
        grf = np.real(ifft2(filtered_fourier))

        # Normalize to desired amplitude
        grf = grf / (np.std(grf) + 1e-10) * amplitude

        return grf

    def generate_multiscale_grf(
        self,
        shape: Tuple[int, int],
        scales: list,
        amplitudes: list
    ) -> np.ndarray:
        """
        Generate multi-scale GRF by combining fields at different scales.

        This allows for both large-scale shape variations and
        small-scale boundary perturbations.

        Args:
            shape: Output shape (H, W)
            scales: List of correlation lengths
            amplitudes: List of amplitudes for each scale

        Returns:
            combined_grf: Multi-scale Gaussian random field
        """
        combined = np.zeros(shape, dtype=np.float64)

        for scale, amp in zip(scales, amplitudes):
            grf = self.generate_gaussian_random_field(shape, scale, amp)
            combined += grf

        return combined

    # =========================================================================
    # SDF Perturbation Operations
    # =========================================================================

    def global_offset(self, sdf: np.ndarray, offset: float) -> np.ndarray:
        """
        Apply global offset to SDF (uniform erosion/dilation).

        Mathematical interpretation:
        - offset > 0: Shrinks the object (erosion)
        - offset < 0: Expands the object (dilation)

        The offset magnitude directly corresponds to pixel displacement
        of the boundary in the normal direction.

        Args:
            sdf: Signed distance function
            offset: Global offset value (positive = erosion)

        Returns:
            perturbed_sdf: SDF with global offset applied
        """
        return sdf + offset

    def apply_grf_perturbation(
        self,
        sdf: np.ndarray,
        correlation_length: float,
        amplitude: float
    ) -> np.ndarray:
        """
        Apply Gaussian Random Field perturbation to SDF.

        This creates spatially varying boundary shifts that mimic
        locally uncertain contours from segmentation models.

        Args:
            sdf: Signed distance function
            correlation_length: Spatial correlation (larger = smoother variations)
            amplitude: Magnitude of perturbation (in pixels)

        Returns:
            perturbed_sdf: SDF with GRF perturbation
        """
        grf = self.generate_gaussian_random_field(
            sdf.shape, correlation_length, amplitude
        )
        return sdf + grf

    def apply_boundary_weighted_perturbation(
        self,
        sdf: np.ndarray,
        perturbation: np.ndarray,
        boundary_weight_sigma: float = 10.0
    ) -> np.ndarray:
        """
        Apply perturbation weighted by distance to boundary.

        Perturbations are strongest near the boundary (|φ| ≈ 0)
        and decay away from it. This ensures modifications affect
        only the boundary region, preserving interior/exterior.

        Weight function: w(φ) = exp(-φ²/2σ²)

        Args:
            sdf: Signed distance function
            perturbation: Perturbation field to apply
            boundary_weight_sigma: Width of boundary region

        Returns:
            perturbed_sdf: SDF with boundary-weighted perturbation
        """
        # Gaussian weight centered on boundary
        weight = np.exp(-sdf**2 / (2 * boundary_weight_sigma**2))

        return sdf + perturbation * weight

    # =========================================================================
    # Regularization and Constraints
    # =========================================================================

    def compute_curvature(self, sdf: np.ndarray) -> np.ndarray:
        """
        Compute mean curvature of the level sets.

        Curvature κ = div(∇φ/|∇φ|)

        High curvature indicates sharp corners or thin structures.
        """
        # Compute gradients
        gy, gx = np.gradient(sdf)
        grad_mag = np.sqrt(gx**2 + gy**2 + 1e-10)

        # Normalized gradient
        nx = gx / grad_mag
        ny = gy / grad_mag

        # Divergence of normalized gradient = curvature
        nxy, nxx = np.gradient(nx)
        nyy, nyx = np.gradient(ny)

        curvature = nxx + nyy

        return curvature

    def regularize_sdf(
        self,
        sdf: np.ndarray,
        original_sdf: np.ndarray,
        tv_weight: float = 0.1,
        area_weight: float = 0.01
    ) -> np.ndarray:
        """
        Regularize perturbed SDF to maintain realistic geometry.

        Constraints:
        1. Total Variation: Limits boundary roughness
        2. Area constraint: Prevents excessive size changes

        This is a simplified version using iterative smoothing.
        For production, consider solving the regularized optimization problem.

        Args:
            sdf: Perturbed SDF to regularize
            original_sdf: Original SDF for area constraint
            tv_weight: Weight for total variation smoothing
            area_weight: Weight for area preservation

        Returns:
            regularized_sdf: Smoothed SDF with constraints
        """
        # Total variation regularization via smoothing
        if tv_weight > 0:
            smooth_sigma = tv_weight * 3
            sdf_smooth = gaussian_filter(sdf, sigma=smooth_sigma)
            sdf = (1 - tv_weight) * sdf + tv_weight * sdf_smooth

        # Area constraint: soft penalty for size deviation
        if area_weight > 0:
            original_area = np.sum(original_sdf < 0)
            current_area = np.sum(sdf < 0)

            if original_area > 0:
                area_ratio = current_area / original_area

                # Adjust global offset to correct area
                if area_ratio > 1.5:  # Too large
                    correction = area_weight * np.log(area_ratio)
                    sdf = sdf + correction
                elif area_ratio < 0.5:  # Too small
                    correction = area_weight * np.log(area_ratio)
                    sdf = sdf + correction

        return sdf

    # =========================================================================
    # High-Level Augmentation Functions
    # =========================================================================

    def uniform_erosion_dilation(
        self,
        mask: np.ndarray,
        displacement: float
    ) -> Tuple[np.ndarray, float]:
        """
        Apply uniform erosion or dilation via SDF offset.

        Args:
            mask: Input binary mask
            displacement: Boundary displacement in pixels
                         positive = erosion, negative = dilation

        Returns:
            augmented_mask: Modified mask
            actual_displacement: Applied displacement
        """
        sdf = self.mask_to_sdf(mask)
        sdf_perturbed = self.global_offset(sdf, displacement)
        return self.sdf_to_mask(sdf_perturbed), displacement

    def spatially_varying_boundary_shift(
        self,
        mask: np.ndarray,
        correlation_length: float = 20.0,
        amplitude: float = 5.0,
        global_offset: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply spatially varying boundary perturbation using GRF.

        This simulates locally uncertain contours that are common
        in segmentation model outputs.

        Args:
            mask: Input binary mask
            correlation_length: Spatial smoothness of variations (pixels)
            amplitude: Standard deviation of boundary shifts (pixels)
            global_offset: Additional uniform offset

        Returns:
            augmented_mask: Modified mask
            info: Dictionary with perturbation statistics
        """
        sdf = self.mask_to_sdf(mask)

        # Generate GRF perturbation
        grf = self.generate_gaussian_random_field(
            mask.shape, correlation_length, amplitude
        )

        # Apply perturbation
        sdf_perturbed = sdf + grf + global_offset

        # Regularize
        sdf_perturbed = self.regularize_sdf(sdf_perturbed, sdf)

        augmented = self.sdf_to_mask(sdf_perturbed)

        info = {
            'correlation_length': correlation_length,
            'amplitude': amplitude,
            'global_offset': global_offset,
            'grf_range': (grf.min(), grf.max()),
        }

        return augmented, info

    def small_lesion_shrinkage(
        self,
        mask: np.ndarray,
        shrink_factor: float = 0.5,
        disappearance_prob: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Model small-lesion failure via controlled SDF bias.

        Small lesions are probabilistically shrunk or eliminated
        by applying a negative bias proportional to object size.

        Mathematical model:
        - Shrinkage: φ' = φ + c where c > 0 is proportional to sqrt(area)
        - Disappearance: φ' = max_value (everywhere positive)

        Args:
            mask: Input binary mask
            shrink_factor: Fraction of equivalent radius to shrink (0-1)
            disappearance_prob: Probability of complete disappearance

        Returns:
            augmented_mask: Modified mask (possibly empty)
            info: Dictionary with transformation details
        """
        sdf = self.mask_to_sdf(mask)

        # Compute equivalent radius
        area = np.sum(mask > 0.5)
        equiv_radius = np.sqrt(area / np.pi)

        # Check for complete disappearance
        if self.rng.random() < disappearance_prob:
            return np.zeros_like(mask), {'disappeared': True}

        # Apply shrinkage offset
        shrink_offset = equiv_radius * shrink_factor
        sdf_perturbed = sdf + shrink_offset

        # Add small GRF for realistic boundary
        grf = self.generate_gaussian_random_field(
            mask.shape,
            correlation_length=max(5, equiv_radius * 0.5),
            amplitude=shrink_offset * 0.2
        )
        sdf_perturbed = sdf_perturbed + grf

        augmented = self.sdf_to_mask(sdf_perturbed)

        info = {
            'original_radius': equiv_radius,
            'shrink_offset': shrink_offset,
            'disappeared': False,
            'new_area': np.sum(augmented > 0.5),
        }

        return augmented, info

    def create_hole(
        self,
        mask: np.ndarray,
        hole_center: Optional[Tuple[int, int]] = None,
        hole_radius: float = 10.0,
        smoothness: float = 2.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Create smooth hole inside the mask using SDF manipulation.

        The hole is created by adding a localized positive bias to the SDF
        using a smooth radial function.

        Args:
            mask: Input binary mask
            hole_center: (y, x) center of hole (None = random inside mask)
            hole_radius: Radius of the hole in pixels
            smoothness: Transition smoothness (larger = softer edges)

        Returns:
            augmented_mask: Mask with hole
            info: Hole location and size
        """
        sdf = self.mask_to_sdf(mask)
        H, W = mask.shape

        # Find hole center if not specified
        if hole_center is None:
            inside_coords = np.argwhere(mask > 0.5)
            if len(inside_coords) < 10:
                return mask.copy(), {'no_hole': True}
            idx = self.rng.randint(0, len(inside_coords))
            hole_center = tuple(inside_coords[idx])

        cy, cx = hole_center

        # Create distance from hole center
        yy, xx = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((yy - cy)**2 + (xx - cx)**2)

        # Smooth hole function: positive inside hole, zero outside
        # Using tanh for smooth transition
        hole_field = hole_radius - dist_from_center
        hole_field = hole_radius * (1 + np.tanh(hole_field / smoothness)) / 2

        # Add hole to SDF (positive values = outside object)
        sdf_perturbed = sdf + hole_field

        augmented = self.sdf_to_mask(sdf_perturbed)

        info = {
            'hole_center': hole_center,
            'hole_radius': hole_radius,
        }

        return augmented, info

    def create_attachment(
        self,
        mask: np.ndarray,
        attach_direction: Optional[Tuple[float, float]] = None,
        attach_length: float = 20.0,
        attach_width: float = 10.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Create smooth attachment/protrusion using SDF manipulation.

        The attachment is created by subtracting a localized elongated
        bias from the SDF along a specified direction.

        Args:
            mask: Input binary mask
            attach_direction: (dy, dx) unit vector for attachment direction
            attach_length: Length of attachment
            attach_width: Width of attachment

        Returns:
            augmented_mask: Mask with attachment
            info: Attachment details
        """
        sdf = self.mask_to_sdf(mask)
        H, W = mask.shape

        # Find boundary point for attachment
        boundary = np.abs(sdf) < 2.0
        boundary_coords = np.argwhere(boundary)

        if len(boundary_coords) < 5:
            return mask.copy(), {'no_attachment': True}

        idx = self.rng.randint(0, len(boundary_coords))
        start_y, start_x = boundary_coords[idx]

        # Direction (outward from centroid if not specified)
        if attach_direction is None:
            inside_coords = np.argwhere(mask > 0.5)
            if len(inside_coords) > 0:
                centroid = np.mean(inside_coords, axis=0)
                direction = np.array([start_y - centroid[0], start_x - centroid[1]])
                direction = direction / (np.linalg.norm(direction) + 1e-10)
            else:
                angle = self.rng.uniform(0, 2 * np.pi)
                direction = np.array([np.sin(angle), np.cos(angle)])
        else:
            direction = np.array(attach_direction)
            direction = direction / (np.linalg.norm(direction) + 1e-10)

        dy, dx = direction

        # Create elongated attachment field
        yy, xx = np.ogrid[:H, :W]

        # Project onto attachment axis
        proj_along = (yy - start_y) * dy + (xx - start_x) * dx
        proj_perp = np.abs((yy - start_y) * dx - (xx - start_x) * dy)

        # Attachment field: negative where we want to add material
        # Only in forward direction, within width
        along_mask = (proj_along > 0) & (proj_along < attach_length)
        width_mask = proj_perp < attach_width / 2

        attachment_field = np.zeros_like(sdf)
        active = along_mask & width_mask

        # Smooth falloff
        attachment_field[active] = -(attach_width / 2 - proj_perp[active])
        attachment_field = gaussian_filter(attachment_field, sigma=2.0)

        # Apply to SDF (negative = inside object)
        sdf_perturbed = sdf + attachment_field

        augmented = self.sdf_to_mask(sdf_perturbed)

        info = {
            'start_point': (start_y, start_x),
            'direction': (dy, dx),
            'length': attach_length,
            'width': attach_width,
        }

        return augmented, info

    # =========================================================================
    # Composite Augmentation with Target Dice Control
    # =========================================================================

    def augment_to_target_dice(
        self,
        mask: np.ndarray,
        target_dice: Tuple[float, float],
        max_attempts: int = 20
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Apply SDF-based augmentation to achieve target Dice range.

        Uses iterative parameter adjustment to hit the target Dice score.

        Args:
            mask: Ground truth mask
            target_dice: (min_dice, max_dice) target range
            max_attempts: Maximum optimization attempts

        Returns:
            augmented_mask: Modified mask
            actual_dice: Achieved Dice score
            info: Augmentation parameters used
        """
        if target_dice == 'perfect':
            return mask.copy(), 1.0, {'augmentation': 'none'}

        target_min, target_max = target_dice
        target_mid = (target_min + target_max) / 2

        # Analyze mask properties
        area = np.sum(mask > 0.5)
        if area < 10:
            return mask.copy(), 1.0, {'error': 'mask_too_small'}

        equiv_radius = np.sqrt(area / np.pi)

        best_result = mask.copy()
        best_dice = 1.0
        best_info = {}

        for attempt in range(max_attempts):
            # Determine augmentation strategy based on target
            if target_mid < 0.75:
                # Severe degradation: large perturbations
                strategy = self.rng.choice([
                    'large_erosion', 'large_dilation', 'grf_heavy',
                    'shrinkage', 'multiple_holes'
                ])
            elif target_mid < 0.88:
                # Moderate degradation
                strategy = self.rng.choice([
                    'medium_erosion', 'medium_dilation', 'grf_medium',
                    'single_hole', 'attachment'
                ])
            else:
                # Light degradation
                strategy = self.rng.choice([
                    'light_erosion', 'light_dilation', 'grf_light'
                ])

            # Apply chosen strategy
            result, info = self._apply_strategy(mask, strategy, equiv_radius, attempt)

            # Compute Dice
            dice = self._compute_dice(result, mask)

            # Check if in target range
            if target_min <= dice <= target_max:
                return result, dice, {'strategy': strategy, **info}

            # Track best
            if abs(dice - target_mid) < abs(best_dice - target_mid):
                best_result = result
                best_dice = dice
                best_info = {'strategy': strategy, **info}

        return best_result, best_dice, best_info

    def _apply_strategy(
        self,
        mask: np.ndarray,
        strategy: str,
        equiv_radius: float,
        attempt: int
    ) -> Tuple[np.ndarray, Dict]:
        """Apply a specific augmentation strategy."""

        # Add randomness based on attempt number
        noise_factor = 1.0 + attempt * 0.1

        if strategy == 'large_erosion':
            displacement = equiv_radius * self.rng.uniform(0.2, 0.4) * noise_factor
            result, _ = self.uniform_erosion_dilation(mask, displacement)
            return result, {'displacement': displacement}

        elif strategy == 'large_dilation':
            displacement = -equiv_radius * self.rng.uniform(0.2, 0.4) * noise_factor
            result, _ = self.uniform_erosion_dilation(mask, displacement)
            return result, {'displacement': displacement}

        elif strategy == 'medium_erosion':
            displacement = equiv_radius * self.rng.uniform(0.1, 0.2) * noise_factor
            result, _ = self.uniform_erosion_dilation(mask, displacement)
            return result, {'displacement': displacement}

        elif strategy == 'medium_dilation':
            displacement = -equiv_radius * self.rng.uniform(0.1, 0.2) * noise_factor
            result, _ = self.uniform_erosion_dilation(mask, displacement)
            return result, {'displacement': displacement}

        elif strategy == 'light_erosion':
            displacement = equiv_radius * self.rng.uniform(0.02, 0.08)
            result, _ = self.uniform_erosion_dilation(mask, displacement)
            return result, {'displacement': displacement}

        elif strategy == 'light_dilation':
            displacement = -equiv_radius * self.rng.uniform(0.02, 0.08)
            result, _ = self.uniform_erosion_dilation(mask, displacement)
            return result, {'displacement': displacement}

        elif strategy == 'grf_heavy':
            corr_len = equiv_radius * self.rng.uniform(0.5, 1.5)
            amplitude = equiv_radius * self.rng.uniform(0.15, 0.3) * noise_factor
            global_off = self.rng.uniform(-3, 3)
            result, info = self.spatially_varying_boundary_shift(
                mask, corr_len, amplitude, global_off
            )
            return result, info

        elif strategy == 'grf_medium':
            corr_len = equiv_radius * self.rng.uniform(0.3, 1.0)
            amplitude = equiv_radius * self.rng.uniform(0.08, 0.15)
            result, info = self.spatially_varying_boundary_shift(
                mask, corr_len, amplitude
            )
            return result, info

        elif strategy == 'grf_light':
            corr_len = equiv_radius * self.rng.uniform(0.2, 0.5)
            amplitude = equiv_radius * self.rng.uniform(0.02, 0.06)
            result, info = self.spatially_varying_boundary_shift(
                mask, corr_len, amplitude
            )
            return result, info

        elif strategy == 'shrinkage':
            shrink = self.rng.uniform(0.3, 0.7) * noise_factor
            result, info = self.small_lesion_shrinkage(mask, shrink)
            return result, info

        elif strategy == 'single_hole':
            hole_r = equiv_radius * self.rng.uniform(0.15, 0.35)
            result, info = self.create_hole(mask, hole_radius=hole_r)
            return result, info

        elif strategy == 'multiple_holes':
            result = mask.copy()
            num_holes = self.rng.randint(2, 4)
            infos = []
            for _ in range(num_holes):
                hole_r = equiv_radius * self.rng.uniform(0.1, 0.2)
                result, info = self.create_hole(result, hole_radius=hole_r)
                infos.append(info)
            return result, {'holes': infos}

        elif strategy == 'attachment':
            length = equiv_radius * self.rng.uniform(0.3, 0.6)
            width = equiv_radius * self.rng.uniform(0.15, 0.3)
            result, info = self.create_attachment(
                mask, attach_length=length, attach_width=width
            )
            return result, info

        else:
            return mask.copy(), {'strategy': 'unknown'}

    def _compute_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Dice coefficient."""
        pred_bin = (pred > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)

        intersection = np.sum(pred_bin * gt_bin)
        union = np.sum(pred_bin) + np.sum(gt_bin)

        if union == 0:
            return 1.0 if np.sum(gt_bin) == 0 else 0.0

        return 2.0 * intersection / union


def compute_dice(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Dice score between prediction and ground truth."""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > threshold).astype(np.float32)

    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)

    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0

    return 2.0 * intersection / union


# =============================================================================
# Integration with Existing Pipeline
# =============================================================================

class SDFSegmentationFailureSimulator:
    """
    SDF-based segmentation failure simulator.

    Drop-in replacement for SegmentationFailureSimulator that uses
    mathematically grounded SDF operations instead of pixel-level manipulation.
    """

    def __init__(self, seed: int = 42):
        self.sdf_aug = SDFAugmentor(seed=seed)
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed: int):
        self.sdf_aug.set_seed(seed)
        self.rng = np.random.RandomState(seed)

    def compose_augmentations(
        self,
        mask: np.ndarray,
        target_dice,
        max_attempts: int = 20
    ) -> Tuple[np.ndarray, float, list]:
        """
        Apply SDF-based augmentations to achieve target Dice score.

        Compatible interface with the original SegmentationFailureSimulator.
        """
        result, dice, info = self.sdf_aug.augment_to_target_dice(
            mask, target_dice, max_attempts
        )

        # Format info as list of strings for compatibility
        aug_list = [f"sdf_{info.get('strategy', 'unknown')}"]

        return result, dice, aug_list
