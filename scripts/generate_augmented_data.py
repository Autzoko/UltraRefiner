"""
Generate Augmented Training Data for SAM Refiner Finetuning

This script generates augmented training data by applying realistic segmentation
failure simulations to ground truth masks. It expands ~3K samples to ~100K samples
with controlled Dice score distribution.

Key Design Principles:
1. Simulate realistic failure modes from actual segmentation models
2. Condition augmentation intensity on lesion SIZE and SHAPE COMPLEXITY
3. Include PERFECT masks so the Refiner learns "when not to modify"
4. Mix perfect, mildly corrupted, and heavily corrupted masks

Target Distribution:
- Dice 0.9-0.99 (Good): 25% of samples - Minor artifacts, mostly preserve
- Dice 0.8-0.9 (Medium): 40% of samples - Moderate errors, needs refinement
- Dice 0.6-0.8 (Poor): 35% of samples - Severe failures, needs strong correction

Note: No Dice=1.0 (perfect/unchanged) masks are generated. This prevents the model
from learning to simply copy the input when it sees high-quality masks.

Augmentation Types (mimicking real segmentation model failures):

Under-segmentation:
1. Boundary erosion - Missing boundary regions
2. Partial breakage - Object split into disconnected parts
3. Small-lesion disappearance - Critical: tiny lesions completely missed
4. Extreme shrinkage - Severe size reduction

Over-segmentation:
5. Boundary dilation - Bleeding into background
6. Attachment to nearby structures - Unintended connection to adjacent regions
7. Artificial bridges - Thin connections linking to nearby structures

Boundary artifacts:
8. Contour jitter - Pixel-level noise along edge band
9. Edge roughening - Jagged/rough boundaries
10. Elastic deformation - Wavy/distorted boundaries

Internal failures:
11. Internal holes - Missing low-contrast central regions
12. Partial dropout - Missing object regions

False positives:
13. False-positive islands - Small spurious blobs near lesion

All augmentation intensities are SIZE and SHAPE-CONDITIONED:
- Tiny lesions (<500 pixels): Higher probability of disappearance/shrinkage
- Irregular shapes (low circularity): Higher probability of breakage
- Large lesions: Higher probability of holes and partial dropout

Augmentation Backends:
1. Pixel-level (default): Direct morphological operations on binary masks
2. SDF-based (--use_sdf): Mathematically grounded operations in signed distance
   function domain, producing smoother and more anatomically plausible deformations

Soft Mask Mode (--soft_masks):
    When enabled, coarse masks are saved as soft probability maps (NPY float files)
    with Gaussian-blurred boundaries, matching the output distribution of TransUNet.
    This is CRITICAL for Phase 2 -> Phase 3 compatibility in end-to-end training.

Usage:
    # Single dataset
    python scripts/generate_augmented_data.py \
        --data_root ./dataset/processed \
        --output_dir ./dataset/augmented \
        --datasets BUSI \
        --target_samples 100000

    # Multiple datasets combined (recommended)
    python scripts/generate_augmented_data.py \
        --data_root ./dataset/processed \
        --output_dir ./dataset/augmented \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --target_samples 100000

    # SDF-based augmentation (smoother deformations)
    python scripts/generate_augmented_data.py \
        --data_root ./dataset/processed \
        --output_dir ./dataset/augmented \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --target_samples 100000 \
        --use_sdf

    # Soft masks for Phase 3 compatibility (RECOMMENDED for E2E training)
    python scripts/generate_augmented_data.py \
        --data_root ./dataset/processed \
        --output_dir ./dataset/augmented_soft \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --target_samples 100000 \
        --soft_masks
"""

import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import morphology
import json
from collections import defaultdict
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_dice(pred, gt, threshold=0.5):
    """Compute Dice score between prediction and ground truth."""
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > threshold).astype(np.float32)

    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)

    if union == 0:
        return 1.0 if np.sum(gt_binary) == 0 else 0.0

    return 2.0 * intersection / union


def create_soft_mask(binary_mask, blur_sigma=None, boundary_noise_std=0.1):
    """
    Convert a binary mask to a soft probability mask that matches TransUNet output distribution.

    TransUNet produces soft probability maps with:
    1. Smooth boundaries (due to bilinear upsampling and softmax)
    2. High confidence in core regions
    3. Gradual transitions at edges

    This function simulates this distribution by:
    1. Applying Gaussian blur to create smooth boundaries
    2. Adding slight noise to simulate prediction uncertainty
    3. Ensuring values stay in [0, 1] range

    Args:
        binary_mask: Binary mask (H, W) with values in {0, 1} or [0, 1]
        blur_sigma: Gaussian blur sigma. If None, auto-computed based on mask size.
                   Larger sigma = smoother boundaries.
        boundary_noise_std: Standard deviation of noise added at boundaries.

    Returns:
        soft_mask: Soft probability mask (H, W) with values in [0, 1]
    """
    H, W = binary_mask.shape

    # Ensure binary
    binary = (binary_mask > 0.5).astype(np.float32)

    # Auto-compute blur sigma based on mask size (similar to TransUNet's upsampling effect)
    if blur_sigma is None:
        # TransUNet uses 16x upsampling from 14x14 to 224x224
        # The effective blur is proportional to image size / patch_grid_size
        blur_sigma = max(2.0, min(H, W) / 56.0)  # Typical range: 2-10 for 224-512 images

    # Step 1: Apply Gaussian blur to create smooth boundaries
    soft_mask = gaussian_filter(binary, sigma=blur_sigma)

    # Step 2: Find boundary region and add slight uncertainty noise
    # Boundary is where the soft mask is between 0.1 and 0.9
    boundary_mask = (soft_mask > 0.1) & (soft_mask < 0.9)

    if boundary_noise_std > 0 and np.any(boundary_mask):
        noise = np.random.randn(H, W) * boundary_noise_std
        # Only add noise in boundary region, scaled by distance from 0.5
        # This simulates TransUNet's uncertainty at boundaries
        uncertainty_weight = 1.0 - 2.0 * np.abs(soft_mask - 0.5)
        soft_mask = soft_mask + noise * uncertainty_weight * boundary_mask

    # Step 3: Ensure valid probability range
    soft_mask = np.clip(soft_mask, 0.0, 1.0)

    return soft_mask


class SegmentationFailureSimulator:
    """
    Simulates realistic segmentation model failures with controllable severity.

    Each augmentation method is designed to achieve a target Dice score range
    by adjusting the intensity of the degradation.

    Key features:
    - Size-conditioned: Augmentation probability/intensity varies with lesion size
    - Shape-conditioned: Irregular shapes get different failure modes
    - Perfect mask support: Can return unmodified masks for preservation learning
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    # =========================================================================
    # Lesion Property Analysis (for size/shape-conditioned augmentation)
    # =========================================================================

    def analyze_lesion(self, mask):
        """
        Analyze lesion properties for size/shape-conditioned augmentation.

        Returns:
            dict with:
            - area: Total pixel area
            - diameter: Equivalent circular diameter
            - circularity: How circular (1.0 = perfect circle)
            - is_tiny: Whether lesion is very small (<500 pixels)
            - is_small: Whether lesion is small (<2000 pixels)
            - is_irregular: Whether shape is highly irregular (circularity < 0.5)
            - complexity: Overall shape complexity score (0-1)
        """
        binary = (mask > 0.5).astype(np.uint8)
        area = np.sum(binary)

        if area < 10:
            return {
                'area': area,
                'diameter': 0,
                'circularity': 0,
                'is_tiny': True,
                'is_small': True,
                'is_irregular': False,
                'complexity': 0
            }

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {
                'area': area,
                'diameter': 2 * np.sqrt(area / np.pi),
                'circularity': 0.5,
                'is_tiny': area < 500,
                'is_small': area < 2000,
                'is_irregular': False,
                'complexity': 0.5
            }

        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)

        # Circularity: 4π × area / perimeter²
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        circularity = min(circularity, 1.0)

        # Equivalent diameter
        diameter = 2 * np.sqrt(area / np.pi)

        # Complexity based on contour irregularity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)  # How convex

        complexity = 1.0 - (circularity * 0.5 + solidity * 0.5)

        return {
            'area': area,
            'diameter': diameter,
            'circularity': circularity,
            'is_tiny': area < 500,
            'is_small': area < 2000,
            'is_irregular': circularity < 0.5,
            'complexity': complexity,
            'solidity': solidity
        }

    # =========================================================================
    # Core Augmentation Methods
    # =========================================================================

    def boundary_erosion(self, mask, intensity=0.5):
        """
        Simulate under-segmentation by eroding boundaries.

        This mimics segmentation models that:
        - Miss thin structures
        - Under-predict boundary regions
        - Shrink objects due to conservative predictions

        Args:
            mask: Binary mask (H, W) with values in [0, 1]
            intensity: 0.0-1.0, controls erosion strength
                      0.1 -> 1-2 pixel erosion (Dice ~0.95)
                      0.5 -> 5-8 pixel erosion (Dice ~0.80)
                      1.0 -> 10-15 pixel erosion (Dice ~0.60)
        """
        if np.sum(mask > 0.5) < 50:  # Skip if mask too small
            return mask.copy()

        # Convert to binary
        binary = (mask > 0.5).astype(np.uint8)

        # Calculate erosion amount based on intensity and mask size
        mask_area = np.sum(binary)
        mask_diameter = 2 * np.sqrt(mask_area / np.pi)

        # Erosion amount: 2-15% of object diameter
        erosion_fraction = 0.02 + 0.13 * intensity
        erosion_pixels = max(1, int(mask_diameter * erosion_fraction))

        # Apply morphological erosion
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_pixels * 2 + 1, erosion_pixels * 2 + 1)
        )
        eroded = cv2.erode(binary, kernel, iterations=1)

        # Add slight noise to boundary for realism
        noise = self.rng.randn(*mask.shape) * 0.05
        result = eroded.astype(np.float32) + noise

        return np.clip(result, 0, 1)

    def boundary_dilation(self, mask, intensity=0.5):
        """
        Simulate over-segmentation by dilating boundaries.

        This mimics segmentation models that:
        - Over-predict object boundaries
        - Include surrounding tissue/background
        - Have bleeding artifacts

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls dilation strength
        """
        if np.sum(mask > 0.5) < 50:
            return mask.copy()

        binary = (mask > 0.5).astype(np.uint8)

        mask_area = np.sum(binary)
        mask_diameter = 2 * np.sqrt(mask_area / np.pi)

        # Dilation amount: 2-15% of object diameter
        dilation_fraction = 0.02 + 0.13 * intensity
        dilation_pixels = max(1, int(mask_diameter * dilation_fraction))

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1)
        )
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Add soft boundary transition
        blurred = gaussian_filter(dilated.astype(np.float32), sigma=1.0)

        return np.clip(blurred, 0, 1)

    def elastic_deformation(self, mask, intensity=0.5):
        """
        Simulate imprecise/wavy boundaries through elastic deformation.

        This mimics segmentation models that:
        - Produce spatially inconsistent predictions
        - Have checkerboard artifacts
        - Show boundary wobbling

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls deformation strength
        """
        H, W = mask.shape

        # Deformation parameters based on intensity
        alpha = 5 + 25 * intensity  # Displacement magnitude
        sigma = 3 + 7 * intensity    # Smoothness

        # Generate random displacement fields
        dx = self.rng.randn(H, W) * alpha
        dy = self.rng.randn(H, W) * alpha

        # Smooth the displacement fields
        dx = gaussian_filter(dx, sigma=sigma)
        dy = gaussian_filter(dy, sigma=sigma)

        # Create meshgrid
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # Apply displacement
        new_x = np.clip(x + dx, 0, W - 1).astype(np.float32)
        new_y = np.clip(y + dy, 0, H - 1).astype(np.float32)

        # Remap using bilinear interpolation
        deformed = cv2.remap(
            mask.astype(np.float32),
            new_x, new_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return np.clip(deformed, 0, 1)

    def add_holes(self, mask, intensity=0.5):
        """
        Simulate false negatives by adding holes inside the mask.

        This mimics segmentation models that:
        - Miss internal structures
        - Have dropout artifacts
        - Produce incomplete predictions

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls hole size and count
        """
        binary = (mask > 0.5).astype(np.float32)
        fg_coords = np.argwhere(binary > 0.5)

        if len(fg_coords) < 100:
            return mask.copy()

        result = binary.copy()
        H, W = mask.shape

        # Number of holes: 1-6 based on intensity
        num_holes = int(1 + 5 * intensity)

        # Hole size: 3-15% of object area
        mask_area = np.sum(binary)
        hole_area_fraction = 0.03 + 0.12 * intensity
        total_hole_area = mask_area * hole_area_fraction
        hole_area_per = total_hole_area / num_holes
        hole_radius = int(np.sqrt(hole_area_per / np.pi))
        hole_radius = max(3, min(hole_radius, 30))

        for _ in range(num_holes):
            # Random center inside foreground
            idx = self.rng.randint(0, len(fg_coords))
            cy, cx = fg_coords[idx]

            # Vary hole size
            r = hole_radius + self.rng.randint(-2, 3)
            r = max(2, r)

            # Create elliptical hole with random aspect ratio
            ry = r + self.rng.randint(-2, 3)
            rx = r + self.rng.randint(-2, 3)

            # Apply hole
            yy, xx = np.ogrid[:H, :W]
            hole_mask = ((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2 <= 1
            result[hole_mask] *= self.rng.uniform(0, 0.2)

        return np.clip(result, 0, 1)

    def add_false_positives(self, mask, intensity=0.5):
        """
        Simulate false positive detections in the background.

        This mimics segmentation models that:
        - Detect spurious objects
        - Have noise-induced false alarms
        - Confuse similar-looking structures

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls blob size and count
        """
        binary = (mask > 0.5).astype(np.float32)
        bg_coords = np.argwhere(binary < 0.3)

        if len(bg_coords) < 100:
            return mask.copy()

        result = binary.copy()
        H, W = mask.shape

        # Number of blobs: 1-5 based on intensity
        num_blobs = int(1 + 4 * intensity)

        # Blob size: 1-8% of object area
        mask_area = max(np.sum(binary), 100)
        blob_area_fraction = 0.01 + 0.07 * intensity
        total_blob_area = mask_area * blob_area_fraction
        blob_area_per = total_blob_area / num_blobs
        blob_radius = int(np.sqrt(blob_area_per / np.pi))
        blob_radius = max(2, min(blob_radius, 20))

        for _ in range(num_blobs):
            # Random center in background
            idx = self.rng.randint(0, len(bg_coords))
            cy, cx = bg_coords[idx]

            # Vary blob size
            r = blob_radius + self.rng.randint(-2, 3)
            r = max(2, r)

            # Create blob
            ry = r + self.rng.randint(-2, 3)
            rx = r + self.rng.randint(-2, 3)

            yy, xx = np.ogrid[:H, :W]
            blob_mask = ((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2 <= 1

            # Soft blob with Gaussian falloff
            blob_value = self.rng.uniform(0.6, 1.0)
            result[blob_mask] = np.maximum(result[blob_mask], blob_value)

        return np.clip(result, 0, 1)

    def partial_dropout(self, mask, intensity=0.5):
        """
        Simulate missing object regions through partial dropout.

        This mimics segmentation models that:
        - Miss part of the object
        - Have attention dropout
        - Fail on occluded regions

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls dropout fraction
        """
        binary = (mask > 0.5).astype(np.float32)

        if np.sum(binary) < 100:
            return mask.copy()

        H, W = mask.shape

        # Dropout fraction: 5-35% based on intensity
        dropout_fraction = 0.05 + 0.30 * intensity

        # Choose dropout strategy
        strategy = self.rng.choice(['corner', 'side', 'random_region'])

        if strategy == 'corner':
            # Drop a corner quadrant
            fg_coords = np.argwhere(binary > 0.5)
            cy, cx = np.mean(fg_coords, axis=0).astype(int)

            quadrant = self.rng.randint(0, 4)
            dropout_mask = np.zeros_like(binary, dtype=bool)

            if quadrant == 0:  # Top-left
                dropout_mask[:cy, :cx] = True
            elif quadrant == 1:  # Top-right
                dropout_mask[:cy, cx:] = True
            elif quadrant == 2:  # Bottom-left
                dropout_mask[cy:, :cx] = True
            else:  # Bottom-right
                dropout_mask[cy:, cx:] = True

            # Apply partial dropout
            dropout_mask = dropout_mask & (binary > 0.5)
            result = binary.copy()
            result[dropout_mask] *= self.rng.uniform(0, 0.3)

        elif strategy == 'side':
            # Drop from one side
            fg_coords = np.argwhere(binary > 0.5)
            min_y, min_x = fg_coords.min(axis=0)
            max_y, max_x = fg_coords.max(axis=0)

            side = self.rng.randint(0, 4)
            result = binary.copy()

            if side == 0:  # Top
                cut_y = int(min_y + (max_y - min_y) * dropout_fraction)
                result[:cut_y, :] *= self.rng.uniform(0, 0.3)
            elif side == 1:  # Bottom
                cut_y = int(max_y - (max_y - min_y) * dropout_fraction)
                result[cut_y:, :] *= self.rng.uniform(0, 0.3)
            elif side == 2:  # Left
                cut_x = int(min_x + (max_x - min_x) * dropout_fraction)
                result[:, :cut_x] *= self.rng.uniform(0, 0.3)
            else:  # Right
                cut_x = int(max_x - (max_x - min_x) * dropout_fraction)
                result[:, cut_x:] *= self.rng.uniform(0, 0.3)

        else:  # random_region
            # Drop a random elliptical region
            fg_coords = np.argwhere(binary > 0.5)
            idx = self.rng.randint(0, len(fg_coords))
            cy, cx = fg_coords[idx]

            mask_area = np.sum(binary)
            dropout_area = mask_area * dropout_fraction
            r = int(np.sqrt(dropout_area / np.pi))

            yy, xx = np.ogrid[:H, :W]
            ry = r + self.rng.randint(-5, 6)
            rx = r + self.rng.randint(-5, 6)
            dropout_region = ((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2 <= 1

            result = binary.copy()
            result[dropout_region] *= self.rng.uniform(0, 0.3)

        return np.clip(result, 0, 1)

    def add_prediction_noise(self, mask, intensity=0.5):
        """
        Simulate prediction uncertainty and soft probability artifacts.

        This mimics segmentation models that:
        - Produce uncertain predictions
        - Have noise in probability maps
        - Show speckle artifacts

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls noise level
        """
        # Noise level: 0.05-0.25 based on intensity
        noise_std = 0.05 + 0.20 * intensity

        noise = self.rng.randn(*mask.shape) * noise_std
        result = mask + noise

        # Add spatially correlated noise (low-frequency)
        if intensity > 0.3:
            low_freq_noise = self.rng.randn(*mask.shape) * noise_std * 2
            low_freq_noise = gaussian_filter(low_freq_noise, sigma=5)
            result += low_freq_noise

        return np.clip(result, 0, 1)

    def edge_roughening(self, mask, intensity=0.5):
        """
        Simulate jagged/rough boundary predictions.

        This mimics segmentation models that:
        - Have aliasing artifacts
        - Produce staircase boundaries
        - Show pixelated edges

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls roughness
        """
        binary = (mask > 0.5).astype(np.uint8)

        # Get boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary, kernel)
        eroded = cv2.erode(binary, kernel)
        boundary = dilated - eroded

        # Add noise to boundary region
        noise_level = 0.2 + 0.5 * intensity
        noise = self.rng.randn(*mask.shape) * noise_level

        result = binary.astype(np.float32)
        boundary_mask = boundary > 0
        result[boundary_mask] += noise[boundary_mask]

        # Random boundary jitter
        if intensity > 0.3:
            jitter_mask = boundary_mask & (self.rng.rand(*mask.shape) < intensity)
            result[jitter_mask] = 1 - result[jitter_mask]

        return np.clip(result, 0, 1)

    def shape_distortion(self, mask, intensity=0.5):
        """
        Simulate morphological shape distortion.

        This mimics segmentation models that:
        - Distort object shape
        - Have aspect ratio errors
        - Produce skewed predictions

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls distortion magnitude
        """
        H, W = mask.shape

        # Affine transformation parameters
        scale_x = 1.0 + self.rng.uniform(-0.15, 0.15) * intensity
        scale_y = 1.0 + self.rng.uniform(-0.15, 0.15) * intensity
        shear = self.rng.uniform(-0.1, 0.1) * intensity
        rotation = self.rng.uniform(-10, 10) * intensity  # degrees

        # Get center
        fg_coords = np.argwhere(mask > 0.5)
        if len(fg_coords) < 10:
            return mask.copy()

        cy, cx = np.mean(fg_coords, axis=0)

        # Build transformation matrix
        angle_rad = np.radians(rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Combined rotation, scale, shear
        M = np.array([
            [cos_a * scale_x, -sin_a + shear, cx - cx * cos_a * scale_x + cy * sin_a],
            [sin_a * scale_y, cos_a + shear, cy - cx * sin_a * scale_y - cy * cos_a]
        ], dtype=np.float32)

        # Apply transformation
        result = cv2.warpAffine(
            mask.astype(np.float32), M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return np.clip(result, 0, 1)

    # =========================================================================
    # NEW: Critical Failure Modes for Small/Irregular Lesions
    # =========================================================================

    def small_lesion_disappearance(self, mask, intensity=0.5):
        """
        Simulate complete disappearance or extreme shrinkage of small lesions.

        This is the MOST CRITICAL failure mode - small lesions are often
        completely missed by segmentation models due to:
        - Feature map resolution loss in deep networks
        - Pooling operations removing small structures
        - Class imbalance causing bias toward background

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls disappearance probability
                      0.3 -> partial shrinkage
                      0.7 -> severe shrinkage
                      1.0 -> complete disappearance
        """
        props = self.analyze_lesion(mask)
        binary = (mask > 0.5).astype(np.float32)

        if props['area'] < 10:
            return mask.copy()

        # For tiny lesions, high intensity = complete disappearance
        if props['is_tiny'] and intensity > 0.7:
            # Complete disappearance
            return np.zeros_like(mask)

        # Extreme shrinkage - erode until very small
        if intensity > 0.5:
            shrink_factor = 0.3 + 0.5 * (1 - intensity)  # 0.3-0.8 of original
            target_area = props['area'] * shrink_factor

            # Iteratively erode
            result = binary.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            while np.sum(result > 0.5) > target_area and np.sum(result > 0.5) > 5:
                result = cv2.erode(result.astype(np.uint8), kernel).astype(np.float32)

            return result

        # Moderate shrinkage
        shrink_pixels = int(props['diameter'] * 0.1 * intensity)
        if shrink_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (shrink_pixels * 2 + 1, shrink_pixels * 2 + 1)
            )
            result = cv2.erode(binary.astype(np.uint8), kernel).astype(np.float32)
            return result

        return binary

    def partial_breakage(self, mask, intensity=0.5):
        """
        Simulate object breaking into disconnected parts.

        Common in irregular lesions where thin connections are lost due to:
        - Erosion during processing
        - Feature map quantization
        - Attention mechanism gaps

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls breakage severity
        """
        binary = (mask > 0.5).astype(np.uint8)
        props = self.analyze_lesion(mask)

        if props['area'] < 100:
            return mask.copy()

        H, W = mask.shape

        # Find skeleton to identify thin connections
        skeleton = morphology.skeletonize(binary > 0)

        # Find points along skeleton to break
        skel_points = np.argwhere(skeleton)
        if len(skel_points) < 5:
            return mask.copy()

        result = binary.astype(np.float32)

        # Number of breaks based on intensity
        num_breaks = int(1 + 3 * intensity)

        for _ in range(num_breaks):
            # Select random point on skeleton (avoid endpoints)
            idx = self.rng.randint(len(skel_points) // 4, 3 * len(skel_points) // 4)
            cy, cx = skel_points[idx]

            # Create break gap
            gap_size = int(3 + 7 * intensity)

            yy, xx = np.ogrid[:H, :W]
            gap_mask = ((yy - cy)**2 + (xx - cx)**2) <= gap_size**2

            # Remove the gap
            result[gap_mask] = 0

        return np.clip(result, 0, 1)

    def attachment_to_nearby(self, mask, intensity=0.5):
        """
        Simulate unintended attachment to nearby structures.

        This mimics over-segmentation where the model incorrectly
        includes adjacent anatomical structures due to:
        - Similar texture/intensity
        - Proximity in feature space
        - Upsampling artifacts

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls attachment size
        """
        binary = (mask > 0.5).astype(np.uint8)
        props = self.analyze_lesion(mask)

        if props['area'] < 50:
            return mask.copy()

        H, W = mask.shape
        result = binary.astype(np.float32)

        # Find boundary points
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary, kernel)
        boundary = dilated - binary
        boundary_points = np.argwhere(boundary > 0)

        if len(boundary_points) < 10:
            return result

        # Create 1-3 attachment regions
        num_attachments = int(1 + 2 * intensity)

        for _ in range(num_attachments):
            # Select random boundary point
            idx = self.rng.randint(0, len(boundary_points))
            cy, cx = boundary_points[idx]

            # Direction away from centroid (outward)
            fg_coords = np.argwhere(binary > 0)
            centroid_y, centroid_x = np.mean(fg_coords, axis=0)

            dy = cy - centroid_y
            dx = cx - centroid_x
            length = np.sqrt(dy**2 + dx**2) + 1e-6
            dy, dx = dy / length, dx / length

            # Create elongated attachment region
            attach_length = int(props['diameter'] * (0.1 + 0.3 * intensity))
            attach_width = int(props['diameter'] * (0.05 + 0.15 * intensity))

            # Generate attachment blob
            for step in range(attach_length):
                py = int(cy + dy * step)
                px = int(cx + dx * step)

                if 0 <= py < H and 0 <= px < W:
                    # Elliptical blob at each step
                    yy, xx = np.ogrid[:H, :W]
                    r = attach_width * (1 - step / attach_length * 0.5)  # Taper
                    blob = ((yy - py)**2 + (xx - px)**2) <= r**2
                    result[blob] = 1.0

        return np.clip(result, 0, 1)

    def artificial_bridges(self, mask, intensity=0.5):
        """
        Simulate thin artificial connections to nearby regions.

        Creates thin bridges/connections that shouldn't exist, mimicking:
        - Interpolation artifacts
        - Feature leakage in attention
        - Upsampling errors

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls bridge thickness and length
        """
        binary = (mask > 0.5).astype(np.uint8)
        props = self.analyze_lesion(mask)

        if props['area'] < 100:
            return mask.copy()

        H, W = mask.shape
        result = binary.astype(np.float32)

        # Find boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary, kernel)
        boundary = dilated - binary
        boundary_points = np.argwhere(boundary > 0)

        if len(boundary_points) < 10:
            return result

        # Create 1-2 bridges
        num_bridges = 1 + int(intensity > 0.5)

        for _ in range(num_bridges):
            # Select random boundary point as bridge start
            idx = self.rng.randint(0, len(boundary_points))
            start_y, start_x = boundary_points[idx]

            # Direction away from centroid
            fg_coords = np.argwhere(binary > 0)
            centroid_y, centroid_x = np.mean(fg_coords, axis=0)

            dy = start_y - centroid_y
            dx = start_x - centroid_x

            # Add random angle variation
            angle = np.arctan2(dy, dx) + self.rng.uniform(-0.5, 0.5)
            dy = np.sin(angle)
            dx = np.cos(angle)

            # Bridge parameters
            bridge_length = int(props['diameter'] * (0.2 + 0.4 * intensity))
            bridge_width = max(1, int(2 + 3 * intensity))

            # Draw bridge line
            for step in range(bridge_length):
                py = int(start_y + dy * step)
                px = int(start_x + dx * step)

                if 0 <= py < H and 0 <= px < W:
                    # Thin line
                    yy, xx = np.ogrid[:H, :W]
                    line_mask = ((yy - py)**2 + (xx - px)**2) <= bridge_width**2
                    result[line_mask] = 1.0

            # Add small blob at end
            end_y = int(start_y + dy * bridge_length)
            end_x = int(start_x + dx * bridge_length)
            if 0 <= end_y < H and 0 <= end_x < W:
                blob_r = int(3 + 5 * intensity)
                yy, xx = np.ogrid[:H, :W]
                blob = ((yy - end_y)**2 + (xx - end_x)**2) <= blob_r**2
                result[blob] = 1.0

        return np.clip(result, 0, 1)

    def contour_jitter(self, mask, intensity=0.5):
        """
        Apply pixel-level noise specifically along the edge band.

        Simulates prediction uncertainty at boundaries:
        - Soft probability transitions
        - Aliasing from low resolution
        - Quantization noise

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls jitter magnitude
        """
        binary = (mask > 0.5).astype(np.uint8)

        if np.sum(binary) < 50:
            return mask.copy()

        # Create edge band (dilate - erode)
        band_width = int(2 + 6 * intensity)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width))

        dilated = cv2.dilate(binary, kernel)
        eroded = cv2.erode(binary, kernel)
        edge_band = (dilated - eroded) > 0

        result = binary.astype(np.float32)

        # Apply random jitter in edge band
        jitter_prob = 0.2 + 0.4 * intensity
        jitter_mask = edge_band & (self.rng.rand(*mask.shape) < jitter_prob)

        # Flip values in jittered region
        result[jitter_mask] = 1.0 - result[jitter_mask]

        # Add soft noise in edge band
        noise = self.rng.randn(*mask.shape) * (0.1 + 0.2 * intensity)
        result[edge_band] += noise[edge_band]

        return np.clip(result, 0, 1)

    def internal_hole_lowcontrast(self, mask, intensity=0.5):
        """
        Simulate missing internal regions due to low contrast.

        Central regions are often missed because:
        - Lower contrast than boundaries
        - Homogeneous texture confused with background
        - Attention focused on edges

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls hole size
        """
        binary = (mask > 0.5).astype(np.float32)
        props = self.analyze_lesion(mask)

        if props['area'] < 200:
            return mask.copy()

        H, W = mask.shape

        # Find centroid
        fg_coords = np.argwhere(binary > 0.5)
        cy, cx = np.mean(fg_coords, axis=0).astype(int)

        # Create central hole
        hole_fraction = 0.1 + 0.3 * intensity  # 10-40% of diameter
        hole_radius = int(props['diameter'] * hole_fraction / 2)
        hole_radius = max(3, hole_radius)

        # Irregular hole shape
        yy, xx = np.ogrid[:H, :W]
        ry = hole_radius + self.rng.randint(-3, 4)
        rx = hole_radius + self.rng.randint(-3, 4)

        hole_mask = ((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2 <= 1
        hole_mask = hole_mask & (binary > 0.5)

        result = binary.copy()
        result[hole_mask] *= self.rng.uniform(0, 0.2)

        return np.clip(result, 0, 1)

    def false_positive_islands(self, mask, intensity=0.5):
        """
        Add small spurious blobs NEAR the lesion (not random background).

        These simulate:
        - Speckle noise responses
        - Similar texture in vicinity
        - Satellite lesions incorrectly detected

        Args:
            mask: Binary mask (H, W)
            intensity: 0.0-1.0, controls number and size of islands
        """
        binary = (mask > 0.5).astype(np.float32)
        props = self.analyze_lesion(mask)

        if props['area'] < 50:
            return mask.copy()

        H, W = mask.shape
        result = binary.copy()

        # Find near-boundary background (within 2x diameter)
        search_radius = int(props['diameter'] * 1.5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (search_radius, search_radius))
        nearby_region = cv2.dilate(binary.astype(np.uint8), kernel)

        # Near background = nearby but not foreground
        near_bg = (nearby_region > 0) & (binary < 0.5)
        near_bg_coords = np.argwhere(near_bg)

        if len(near_bg_coords) < 20:
            return result

        # Add 1-4 small islands
        num_islands = int(1 + 3 * intensity)

        for _ in range(num_islands):
            idx = self.rng.randint(0, len(near_bg_coords))
            cy, cx = near_bg_coords[idx]

            # Small island (3-10% of lesion diameter)
            island_r = int(props['diameter'] * (0.03 + 0.07 * intensity))
            island_r = max(2, min(island_r, 15))

            yy, xx = np.ogrid[:H, :W]
            ry = island_r + self.rng.randint(-1, 2)
            rx = island_r + self.rng.randint(-1, 2)
            island = ((yy - cy) / max(ry, 1))**2 + ((xx - cx) / max(rx, 1))**2 <= 1

            # Soft island
            island_value = self.rng.uniform(0.6, 1.0)
            result[island] = np.maximum(result[island], island_value)

        return np.clip(result, 0, 1)

    # =========================================================================
    # Composite Augmentation (combining multiple failure types)
    # =========================================================================

    def compose_augmentations(self, mask, target_dice, max_attempts=15):
        """
        Apply multiple augmentations to achieve a target Dice score.

        Uses SIZE and SHAPE-CONDITIONED selection:
        - Tiny lesions: Higher probability of disappearance/shrinkage
        - Irregular shapes: Higher probability of breakage
        - Large lesions: Higher probability of holes, dropout

        Args:
            mask: Ground truth mask (H, W)
            target_dice: Target Dice score range (min, max)
            max_attempts: Maximum attempts to hit target range

        Returns:
            augmented_mask: Augmented mask
            actual_dice: Achieved Dice score
            augmentation_info: Description of applied augmentations
        """
        target_min, target_max = target_dice

        # Clamp max to 0.99 to avoid generating Dice=1.0 (unchanged) masks
        target_max = min(target_max, 0.99)

        # Analyze lesion properties for size/shape conditioning
        props = self.analyze_lesion(mask)

        # Build augmentation pool conditioned on lesion properties
        augmentation_pool = self._build_conditioned_augmentation_pool(
            props, target_min, target_max
        )

        if len(augmentation_pool) == 0:
            return mask.copy(), 1.0, ['no_valid_augmentations']

        # Determine number of augmentations based on target Dice
        if target_max <= 0.8:
            num_augmentations = self.rng.randint(2, 5)
        elif target_max <= 0.9:
            num_augmentations = self.rng.randint(1, 4)
        else:
            num_augmentations = self.rng.randint(1, 2)

        # Iteratively apply augmentations to hit target
        best_result = mask.copy()
        best_dice = 1.0
        best_info = []
        target_mid = (target_min + target_max) / 2

        for attempt in range(max_attempts):
            result = mask.copy()
            applied = []

            # Shuffle and select augmentations
            pool_indices = list(range(len(augmentation_pool)))
            self.rng.shuffle(pool_indices)
            selected = pool_indices[:min(num_augmentations, len(pool_indices))]

            for idx in selected:
                aug_name, base_intensity, weight = augmentation_pool[idx]

                # Add randomness to intensity
                intensity = base_intensity * self.rng.uniform(0.7, 1.3)
                intensity = np.clip(intensity, 0.05, 0.95)

                # Apply augmentation
                result = self._apply_single_augmentation(result, aug_name, intensity)
                applied.append(f"{aug_name}({intensity:.2f})")

            # Compute Dice
            dice = compute_dice(result, mask)

            # Check if in target range
            if target_min <= dice <= target_max:
                return result, dice, applied

            # Track best result (closest to target midpoint)
            if abs(dice - target_mid) < abs(best_dice - target_mid):
                best_result = result
                best_dice = dice
                best_info = applied

        return best_result, best_dice, best_info

    def _build_conditioned_augmentation_pool(self, props, target_min, target_max):
        """
        Build augmentation pool conditioned on lesion size and shape.

        Returns list of (aug_name, base_intensity, weight) tuples.
        Weight determines selection probability.
        """
        pool = []

        is_tiny = props.get('is_tiny', False)
        is_small = props.get('is_small', False)
        is_irregular = props.get('is_irregular', False)

        # Severe degradation (Dice 0.6-0.8)
        if target_max <= 0.8:
            # Under-segmentation (more likely for small/irregular)
            pool.append(('erosion', 0.6, 1.0))
            pool.append(('partial_dropout', 0.6, 1.0))

            if is_tiny:
                # Critical: small lesion disappearance
                pool.append(('small_lesion_disappearance', 0.8, 2.0))
            if is_small:
                pool.append(('small_lesion_disappearance', 0.5, 1.5))
            if is_irregular:
                pool.append(('partial_breakage', 0.6, 1.5))

            # Over-segmentation
            pool.append(('dilation', 0.6, 1.0))
            pool.append(('attachment_to_nearby', 0.6, 0.8))
            pool.append(('artificial_bridges', 0.6, 0.7))

            # Boundary artifacts
            pool.append(('elastic', 0.6, 0.8))
            pool.append(('contour_jitter', 0.7, 0.6))

            # Internal failures (more likely for large lesions)
            if not is_small:
                pool.append(('internal_hole_lowcontrast', 0.7, 1.2))
                pool.append(('holes', 0.6, 1.0))

            # False positives
            pool.append(('false_positive_islands', 0.6, 0.8))

        # Moderate degradation (Dice 0.8-0.9)
        elif target_max <= 0.9:
            # Under-segmentation
            pool.append(('erosion', 0.35, 1.0))
            pool.append(('partial_dropout', 0.3, 0.8))

            if is_tiny:
                pool.append(('small_lesion_disappearance', 0.4, 1.5))
            if is_irregular:
                pool.append(('partial_breakage', 0.35, 1.2))

            # Over-segmentation
            pool.append(('dilation', 0.35, 1.0))
            pool.append(('attachment_to_nearby', 0.3, 0.6))
            pool.append(('artificial_bridges', 0.3, 0.5))

            # Boundary artifacts
            pool.append(('elastic', 0.35, 1.0))
            pool.append(('contour_jitter', 0.4, 0.8))
            pool.append(('edge_roughening', 0.4, 0.8))
            pool.append(('noise', 0.5, 0.7))

            # Internal failures
            if not is_small:
                pool.append(('internal_hole_lowcontrast', 0.4, 0.8))
                pool.append(('holes', 0.3, 0.7))

            # False positives
            pool.append(('false_positive_islands', 0.35, 0.6))

        # Light degradation (Dice 0.9+)
        else:
            # Very light under/over-segmentation
            pool.append(('erosion', 0.15, 1.0))
            pool.append(('dilation', 0.15, 1.0))

            # Minor boundary artifacts
            pool.append(('elastic', 0.15, 1.0))
            pool.append(('contour_jitter', 0.2, 1.0))
            pool.append(('edge_roughening', 0.2, 0.8))
            pool.append(('noise', 0.3, 1.0))

            # Very light false positives (tiny islands)
            pool.append(('false_positive_islands', 0.15, 0.5))

        return pool

    def _apply_single_augmentation(self, mask, aug_name, intensity):
        """Apply a single augmentation by name."""
        if aug_name == 'erosion':
            return self.boundary_erosion(mask, intensity)
        elif aug_name == 'dilation':
            return self.boundary_dilation(mask, intensity)
        elif aug_name == 'elastic':
            return self.elastic_deformation(mask, intensity)
        elif aug_name == 'holes':
            return self.add_holes(mask, intensity)
        elif aug_name == 'false_positives':
            return self.add_false_positives(mask, intensity)
        elif aug_name == 'partial_dropout':
            return self.partial_dropout(mask, intensity)
        elif aug_name == 'noise':
            return self.add_prediction_noise(mask, intensity)
        elif aug_name == 'edge_roughening':
            return self.edge_roughening(mask, intensity)
        elif aug_name == 'shape_distortion':
            return self.shape_distortion(mask, intensity)
        elif aug_name == 'small_lesion_disappearance':
            return self.small_lesion_disappearance(mask, intensity)
        elif aug_name == 'partial_breakage':
            return self.partial_breakage(mask, intensity)
        elif aug_name == 'attachment_to_nearby':
            return self.attachment_to_nearby(mask, intensity)
        elif aug_name == 'artificial_bridges':
            return self.artificial_bridges(mask, intensity)
        elif aug_name == 'contour_jitter':
            return self.contour_jitter(mask, intensity)
        elif aug_name == 'internal_hole_lowcontrast':
            return self.internal_hole_lowcontrast(mask, intensity)
        elif aug_name == 'false_positive_islands':
            return self.false_positive_islands(mask, intensity)
        else:
            return mask


def process_single_augmentation(task, use_sdf=False, soft_masks=False, blur_sigma=None):
    """
    Worker function to process a single augmentation task.

    Args:
        task: tuple of (file_idx, within_level_idx, dataset_name, name, image_path, mask_path,
                       dice_range, level, aug_seed, out_dirs)
        use_sdf: Whether to use SDF-based augmentation
        soft_masks: Whether to save coarse masks as soft probability maps (NPY)
                   for Phase 3 E2E training compatibility
        blur_sigma: Gaussian blur sigma for soft masks (None = auto)

    Returns:
        dict with result info or None if skipped
    """
    (file_idx, within_level_idx, dataset_name, name, image_path, mask_path,
     dice_range, level, aug_seed, out_dirs) = task

    out_image_dir, out_mask_dir, out_coarse_dir = out_dirs

    # Create stable output filename: {dataset}_{name}_{level}_{idx}
    # This naming is independent of global ordering and only depends on:
    # - Original sample identity (dataset + name)
    # - Augmentation level (perfect/high/medium/low)
    # - Index within that level for this sample
    out_name = f"{dataset_name}_{name}_{level}_{within_level_idx:04d}"

    # Check if already exists (for resume)
    # For soft masks, check for .npy file; for binary masks, check for .png
    coarse_ext = '.npy' if soft_masks else '.png'
    out_coarse_path = os.path.join(out_coarse_dir, f"{out_name}{coarse_ext}")
    if os.path.exists(out_coarse_path):
        return None  # Skip, already processed

    # Initialize simulator (each worker needs its own instance)
    if use_sdf:
        from sdf_augmentation import SDFSegmentationFailureSimulator
        simulator = SDFSegmentationFailureSimulator(seed=aug_seed)
    else:
        simulator = SegmentationFailureSimulator(seed=aug_seed)

    # Load data
    if image_path.endswith('.npy'):
        image = np.load(image_path)
    else:
        image = np.array(Image.open(image_path))

    if mask_path.endswith('.npy'):
        gt_mask = np.load(mask_path)
    else:
        gt_mask = np.array(Image.open(mask_path).convert('L'))

    # Normalize mask to [0, 1]
    if gt_mask.max() > 1:
        gt_mask = gt_mask.astype(np.float32) / 255.0
    else:
        gt_mask = gt_mask.astype(np.float32)

    # Generate augmented mask
    coarse_mask, actual_dice, aug_info = simulator.compose_augmentations(
        gt_mask, dice_range, max_attempts=15
    )

    # Save image (copy original)
    if image_path.endswith('.npy'):
        np.save(os.path.join(out_image_dir, f"{out_name}.npy"), image)
    else:
        Image.fromarray(image).save(os.path.join(out_image_dir, f"{out_name}.png"))

    # Save ground truth mask
    gt_save = (gt_mask * 255).astype(np.uint8)
    Image.fromarray(gt_save).save(os.path.join(out_mask_dir, f"{out_name}.png"))

    # Save coarse mask
    if soft_masks:
        # Convert to soft probability map matching TransUNet output distribution
        # This is CRITICAL for Phase 2 -> Phase 3 compatibility
        soft_coarse = create_soft_mask(coarse_mask, blur_sigma=blur_sigma)
        np.save(os.path.join(out_coarse_dir, f"{out_name}.npy"), soft_coarse.astype(np.float32))
    else:
        # Save as binary PNG (legacy mode)
        coarse_save = (coarse_mask * 255).astype(np.uint8)
        Image.fromarray(coarse_save).save(os.path.join(out_coarse_dir, f"{out_name}.png"))

    return {
        'name': out_name,
        'original': name,
        'source_dataset': dataset_name,
        'dice': float(actual_dice),
        'augmentations': aug_info,
        'level': level,
        'soft_masks': soft_masks,
    }


def generate_augmented_dataset(
    data_root: str,
    output_dir: str,
    datasets: list,
    target_samples: int = 100000,
    seed: int = 42,
    use_sdf: bool = False,
    num_workers: int = None,
    resume: bool = True,
    soft_masks: bool = False,
    blur_sigma: float = None,
):
    """
    Generate augmented dataset with controlled Dice distribution.

    Args:
        data_root: Root directory containing preprocessed datasets
        output_dir: Output directory for augmented data
        datasets: List of dataset names to combine
        target_samples: Target number of total samples
        seed: Random seed
        use_sdf: If True, use SDF-based augmentation (smoother deformations)
        num_workers: Number of parallel workers (default: CPU count)
        resume: If True, skip already generated samples (default: True)
        soft_masks: If True, save coarse masks as soft probability maps (NPY)
                   matching TransUNet output distribution. RECOMMENDED for E2E training.
        blur_sigma: Gaussian blur sigma for soft masks (None = auto-compute based on image size)
    """
    # Set number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free

    print(f"Using {num_workers} parallel workers")
    print(f"Augmentation backend: {'SDF-based' if use_sdf else 'Pixel-level'}")
    print(f"Mask output format: {'Soft (NPY float, TransUNet-like)' if soft_masks else 'Binary (PNG uint8)'}")

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Collect all samples from all datasets
    all_samples = []  # List of (dataset_name, image_file, image_path, mask_path)

    print(f"\nLoading samples from {len(datasets)} dataset(s): {', '.join(datasets)}")

    for dataset_name in datasets:
        image_dir = os.path.join(data_root, dataset_name, 'train', 'images')
        mask_dir = os.path.join(data_root, dataset_name, 'train', 'masks')

        if not os.path.exists(image_dir):
            print(f"Warning: Image directory not found: {image_dir}, skipping {dataset_name}")
            continue

        # Get all samples from this dataset
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.npy'))])

        for image_file in image_files:
            name = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_dir, image_file)

            # Find corresponding mask
            mask_file = image_file.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask_file)

            if not os.path.exists(mask_path):
                for ext in ['.png', '.npy', '.jpg']:
                    alt_path = os.path.join(mask_dir, name + ext)
                    if os.path.exists(alt_path):
                        mask_path = alt_path
                        break

            if os.path.exists(mask_path):
                all_samples.append((dataset_name, image_file, image_path, mask_path))
            else:
                print(f"Warning: Mask not found for {dataset_name}/{image_file}")

        print(f"  {dataset_name}: {len([s for s in all_samples if s[0] == dataset_name])} samples")

    num_original = len(all_samples)

    if num_original == 0:
        print("Error: No samples found!")
        return

    print(f"\nTotal samples from all datasets: {num_original}")
    print(f"Target: {target_samples} augmented samples")

    # Calculate augmentations per sample
    augs_per_sample = target_samples // num_original
    remainder = target_samples % num_original

    print(f"Augmentations per sample: {augs_per_sample} (+ {remainder} extra)")

    # Distribution targets per sample (NO Dice=1.0 perfect masks)
    # Good (0.9-0.99): 25% - Minor artifacts, should mostly preserve
    # Medium (0.8-0.9): 40% - Moderate errors, needs refinement
    # Low (0.6-0.8): 35% - Severe failures, needs strong correction
    high_count = int(augs_per_sample * 0.25)        # Dice 0.9-0.99
    medium_count = int(augs_per_sample * 0.40)      # Dice 0.8-0.9
    low_count = augs_per_sample - high_count - medium_count  # Dice 0.6-0.8

    print(f"\nPer-sample distribution (no Dice=1.0 perfect masks):")
    print(f"  High (0.9-0.99):  {high_count}")
    print(f"  Medium (0.8-0.9): {medium_count}")
    print(f"  Low (0.6-0.8):    {low_count}")

    # Create output directories (combined output for all datasets)
    combined_name = '_'.join(sorted(datasets)) if len(datasets) > 1 else datasets[0]
    out_image_dir = os.path.join(output_dir, combined_name, 'images')
    out_mask_dir = os.path.join(output_dir, combined_name, 'masks')
    out_coarse_dir = os.path.join(output_dir, combined_name, 'coarse_masks')

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(out_coarse_dir, exist_ok=True)

    # Build list of all tasks
    print("\nBuilding task list...")
    all_tasks = []
    out_dirs = (out_image_dir, out_mask_dir, out_coarse_dir)
    sample_idx = 0

    for file_idx, (dataset_name, image_file, image_path, mask_path) in enumerate(all_samples):
        name = os.path.splitext(image_file)[0]

        # Determine augmentation counts for this sample
        extra = 1 if file_idx < remainder else 0
        total_augs = augs_per_sample + extra

        # Distribution (NO Dice=1.0 perfect masks)
        high_n = int(total_augs * 0.25)
        medium_n = int(total_augs * 0.40)
        low_n = total_augs - high_n - medium_n

        # Generate augmentations for each Dice range (max 0.99, no 1.0)
        augmentation_configs = [
            ((0.90, 0.99), high_n, 'high'),
            ((0.80, 0.90), medium_n, 'medium'),
            ((0.60, 0.80), low_n, 'low'),
        ]

        for dice_range, count, level in augmentation_configs:
            for within_level_idx in range(count):
                # Use file_idx and within_level_idx for stable seed (independent of global order)
                aug_seed = seed + file_idx * 10000 + hash(level) % 1000 + within_level_idx
                # Use per-sample naming: {dataset}_{name}_{level}_{idx}
                # This is stable and doesn't depend on global sample ordering
                task = (file_idx, within_level_idx, dataset_name, name, image_path, mask_path,
                       dice_range, level, aug_seed, out_dirs)
                all_tasks.append(task)
                sample_idx += 1

    total_tasks = len(all_tasks)
    print(f"Total tasks to process: {total_tasks}")

    # Check for existing files (resume capability)
    coarse_ext = '.npy' if soft_masks else '.png'
    if resume:
        existing_count = 0
        tasks_to_process = []
        for task in all_tasks:
            file_idx, within_level_idx, dataset_name, name, _, _, _, level, _, _ = task
            # Stable naming: uses original sample name + level + index within that level
            out_name = f"{dataset_name}_{name}_{level}_{within_level_idx:04d}"
            out_coarse_path = os.path.join(out_coarse_dir, f"{out_name}{coarse_ext}")
            if os.path.exists(out_coarse_path):
                existing_count += 1
            else:
                tasks_to_process.append(task)

        if existing_count > 0:
            print(f"Resume mode: Found {existing_count} existing samples, skipping them")
            print(f"Remaining tasks: {len(tasks_to_process)}")
        all_tasks = tasks_to_process

    if len(all_tasks) == 0:
        print("All samples already generated. Nothing to do.")
        return

    # Process tasks in parallel
    print(f"\nProcessing {len(all_tasks)} augmentation tasks with {num_workers} workers...")
    start_time = time.time()

    # Use multiprocessing Pool
    worker_fn = partial(process_single_augmentation, use_sdf=use_sdf,
                        soft_masks=soft_masks, blur_sigma=blur_sigma)

    results = []
    with Pool(num_workers) as pool:
        # Use imap_unordered for better progress tracking
        for result in tqdm(pool.imap_unordered(worker_fn, all_tasks, chunksize=10),
                          total=len(all_tasks), desc='Generating augmentations'):
            if result is not None:
                results.append(result)

    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.1f} seconds")
    print(f"Speed: {len(results) / elapsed_time:.1f} samples/second")

    # Compute statistics from results
    dice_scores = [r['dice'] for r in results]
    dice_distribution = {'high': 0, 'medium': 0, 'low': 0}
    augmentation_stats = defaultdict(int)

    for r in results:
        actual_dice = r['dice']
        if actual_dice >= 0.9:
            dice_distribution['high'] += 1
        elif actual_dice >= 0.8:
            dice_distribution['medium'] += 1
        else:
            dice_distribution['low'] += 1

        for aug in r['augmentations']:
            aug_name = aug.split('(')[0]
            augmentation_stats[aug_name] += 1

    # Metadata
    metadata = {
        'datasets': datasets,
        'original_samples': num_original,
        'target_samples': target_samples,
        'seed': seed,
        'soft_masks': soft_masks,  # Important: tells dataset loader which format to expect
        'blur_sigma': blur_sigma,
        'samples': results
    }

    # Print statistics
    print(f"\n{'='*60}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*60}")

    print(f"\nTotal samples generated: {len(results)}")

    print(f"\nDice Score Distribution (no Dice=1.0 perfect masks):")
    total = sum(dice_distribution.values())
    if total > 0:
        print(f"  High (0.9-0.99):  {dice_distribution['high']:6d} ({100*dice_distribution['high']/total:.1f}%)")
        print(f"  Medium (0.8-0.9): {dice_distribution['medium']:6d} ({100*dice_distribution['medium']/total:.1f}%)")
        print(f"  Low (0.6-0.8):    {dice_distribution['low']:6d} ({100*dice_distribution['low']/total:.1f}%)")
    else:
        print("  No samples generated")

    if len(dice_scores) > 0:
        print(f"\nDice Score Statistics:")
        print(f"  Mean: {np.mean(dice_scores):.4f}")
        print(f"  Std:  {np.std(dice_scores):.4f}")
        print(f"  Min:  {np.min(dice_scores):.4f}")
        print(f"  Max:  {np.max(dice_scores):.4f}")

    print(f"\nAugmentation Type Usage:")
    for aug_name, count in sorted(augmentation_stats.items(), key=lambda x: -x[1]):
        print(f"  {aug_name}: {count}")

    # Save metadata
    metadata['statistics'] = {
        'total_samples': len(results),
        'dice_distribution': dice_distribution,
        'dice_mean': float(np.mean(dice_scores)) if dice_scores else 0.0,
        'dice_std': float(np.std(dice_scores)) if dice_scores else 0.0,
        'augmentation_stats': dict(augmentation_stats)
    }

    metadata_path = os.path.join(output_dir, combined_name, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Output directory: {os.path.join(output_dir, combined_name)}")


def main():
    parser = argparse.ArgumentParser(description='Generate augmented training data')

    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing preprocessed datasets')
    parser.add_argument('--output_dir', type=str, default='./dataset/augmented',
                        help='Output directory for augmented data')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Dataset names to combine (e.g., BUSI BUSBRA BUS)')
    parser.add_argument('--target_samples', type=int, default=100000,
                        help='Target number of augmented samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_sdf', action='store_true',
                        help='Use SDF-based augmentation (smoother, more anatomically plausible)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--no_resume', action='store_true',
                        help='Disable resume mode (regenerate all samples)')
    parser.add_argument('--soft_masks', action='store_true',
                        help='Save coarse masks as soft probability maps (NPY float) matching '
                             'TransUNet output distribution. RECOMMENDED for Phase 3 E2E training.')
    parser.add_argument('--blur_sigma', type=float, default=None,
                        help='Gaussian blur sigma for soft masks (default: auto-compute based on image size)')

    args = parser.parse_args()

    print(f"Generating augmented dataset from: {', '.join(args.datasets)}")
    print(f"Target samples: {args.target_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentation backend: {'SDF-based' if args.use_sdf else 'Pixel-level'}")
    print(f"Resume mode: {'Disabled' if args.no_resume else 'Enabled'}")
    print(f"Mask format: {'Soft (NPY, TransUNet-like)' if args.soft_masks else 'Binary (PNG)'}")
    if args.soft_masks:
        print(f"  Blur sigma: {'auto' if args.blur_sigma is None else args.blur_sigma}")

    generate_augmented_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        datasets=args.datasets,
        target_samples=args.target_samples,
        seed=args.seed,
        use_sdf=args.use_sdf,
        num_workers=args.num_workers,
        resume=not args.no_resume,
        soft_masks=args.soft_masks,
        blur_sigma=args.blur_sigma,
    )


if __name__ == '__main__':
    main()
