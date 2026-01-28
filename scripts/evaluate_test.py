#!/usr/bin/env python3
"""
Evaluate UltraRefiner on test sets of each dataset.

Shows scores for:
1. Without refinement (TransUNet coarse output)
2. With refinement (SAM refined output)

Usage:
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM

    # Evaluate at SAM's native resolution (1024x1024)
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --refined_eval_size 1024

    # With visualization (10 samples per dataset)
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets BUSI \
        --visualize --num_visualize 10

    # Visualize best and worst cases (most improved / most degraded)
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets BUSI \
        --visualize --visualize_best_worst --num_visualize 20

    # Visualize ALL samples
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets BUSI \
        --visualize --num_visualize -1

    # With rejection rules (discard bad refinements)
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets BUSI \
        --use_rejection_rules \
        --reject_iou_threshold 0.5 \
        --reject_area_ratio_min 0.3 \
        --reject_area_ratio_max 3.0

    # With boundary-band fusion (refine only near boundaries)
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets BUSI \
        --use_boundary_fusion \
        --boundary_band_width 15

    # Combined stabilization (rejection + boundary fusion)
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets BUSI \
        --use_rejection_rules \
        --use_boundary_fusion \
        --boundary_band_width 20

    # Cross-dataset generalization: evaluate on unseen UDIAT dataset
    # (model trained on BUSI, tested on UDIAT)
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --datasets UDIAT

    # Evaluate on all datasets including unseen ones
    python scripts/evaluate_test.py \
        --checkpoint ./checkpoints/ultra_refiner/fold_0/best.pth \
        --data_root ./dataset/processed \
        --include_unseen
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, label as ndimage_label
from scipy.ndimage import distance_transform_edt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_ultra_refiner, build_gated_ultra_refiner, CONFIGS
from data import get_test_dataloader, SUPPORTED_DATASETS, UNSEEN_DATASETS, ALL_DATASETS


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate UltraRefiner on test sets')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to UltraRefiner checkpoint')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='TransUNet ViT model variant')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        help='SAM model variant')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size for TransUNet')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of segmentation classes')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names to evaluate (default: all training datasets)')
    parser.add_argument('--include_unseen', action='store_true',
                        help='Include unseen test-only datasets (e.g., UDIAT) in evaluation')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Evaluation arguments
    parser.add_argument('--refined_eval_size', type=int, default=224,
                        help='Resolution for evaluating SAM refined output. '
                             '224 = downsample SAM to match label (default), '
                             '1024 = upsample label to match SAM (preserves boundary details)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarizing predictions (default: 0.5)')

    # Model configuration (must match training)
    parser.add_argument('--mask_prompt_style', type=str, default='direct',
                        choices=['gaussian', 'direct', 'distance'],
                        help='Mask prompt style (must match training)')
    parser.add_argument('--use_roi_crop', action='store_true',
                        help='Use ROI cropping (must match training)')
    parser.add_argument('--roi_expand_ratio', type=float, default=0.3,
                        help='ROI expansion ratio (must match training)')
    parser.add_argument('--use_gated_refinement', action='store_true',
                        help='Use gated refinement (must match training)')
    parser.add_argument('--gate_type', type=str, default='uncertainty',
                        help='Gate type for gated refinement')
    parser.add_argument('--gate_gamma', type=float, default=1.0,
                        help='Gate gamma')
    parser.add_argument('--gate_min', type=float, default=0.0,
                        help='Gate min')
    parser.add_argument('--gate_max', type=float, default=0.8,
                        help='Gate max')

    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization images')
    parser.add_argument('--num_visualize', type=int, default=10,
                        help='Number of samples to visualize per dataset (-1 or 0 for all)')
    parser.add_argument('--visualize_best_worst', action='store_true',
                        help='Visualize best and worst cases based on improvement')

    # Inference-time stabilization arguments
    parser.add_argument('--use_rejection_rules', action='store_true',
                        help='Apply rejection rules to discard bad refinements')
    parser.add_argument('--reject_iou_threshold', type=float, default=0.5,
                        help='Reject refined if IoU with coarse < threshold')
    parser.add_argument('--reject_area_ratio_min', type=float, default=0.3,
                        help='Reject if refined_area/coarse_area < min')
    parser.add_argument('--reject_area_ratio_max', type=float, default=3.0,
                        help='Reject if refined_area/coarse_area > max')
    parser.add_argument('--reject_max_components', type=int, default=5,
                        help='Reject if refined has more than N connected components')

    parser.add_argument('--use_boundary_fusion', action='store_true',
                        help='Apply boundary-band fusion (refine only near boundaries)')
    parser.add_argument('--boundary_band_width', type=int, default=15,
                        help='Width of boundary band in pixels (default: 15)')

    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction masks')
    parser.add_argument('--output_dir', type=str, default='./results/test_evaluation',
                        help='Output directory for predictions')

    return parser.parse_args()


def compute_hd95(pred_binary_np, target_binary_np):
    """Compute the 95th percentile Hausdorff Distance (HD95).

    Args:
        pred_binary_np: Predicted binary mask (H, W) as numpy array
        target_binary_np: Ground truth binary mask (H, W) as numpy array

    Returns:
        HD95 value (float), or None if either mask is empty (skipped)
    """
    # Handle empty masks
    if pred_binary_np.sum() == 0 and target_binary_np.sum() == 0:
        return 0.0
    if pred_binary_np.sum() == 0 or target_binary_np.sum() == 0:
        return None  # Skip this sample for HD95 (empty prediction or GT)

    # Compute surface (boundary) points
    # Boundary = mask XOR eroded_mask
    pred_boundary = pred_binary_np ^ binary_erosion(pred_binary_np, iterations=1)
    target_boundary = target_binary_np ^ binary_erosion(target_binary_np, iterations=1)

    # Handle case where erosion removes all pixels (very small masks)
    if pred_boundary.sum() == 0:
        pred_boundary = pred_binary_np
    if target_boundary.sum() == 0:
        target_boundary = target_binary_np

    # Compute distance transforms
    # Distance from each pixel to the nearest boundary pixel of the other mask
    dt_pred = distance_transform_edt(~pred_boundary)
    dt_target = distance_transform_edt(~target_boundary)

    # Directed Hausdorff: distances from target boundary to pred
    dist_target_to_pred = dt_pred[target_boundary]
    # Directed Hausdorff: distances from pred boundary to target
    dist_pred_to_target = dt_target[pred_boundary]

    # Combine both directions
    all_distances = np.concatenate([dist_target_to_pred, dist_pred_to_target])

    # 95th percentile
    hd95 = np.percentile(all_distances, 95)

    return float(hd95)


def compute_metrics(pred, target, threshold=0.5):
    """Compute segmentation metrics.

    Args:
        pred: Predicted probability mask (B, H, W) in [0, 1]
        target: Ground truth binary mask (B, H, W)
        threshold: Threshold for binarization

    Returns:
        Dictionary of metrics
    """
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # Flatten for computation
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    # True positives, false positives, false negatives, true negatives
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()

    # Metrics
    eps = 1e-7

    # Dice / F1
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    # IoU / Jaccard
    iou = (tp + eps) / (tp + fp + fn + eps)

    # Precision
    precision = (tp + eps) / (tp + fp + eps)

    # Recall / Sensitivity
    recall = (tp + eps) / (tp + fn + eps)

    # Accuracy
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    # Specificity
    specificity = (tn + eps) / (tn + fp + eps)

    # HD95 (computed per-sample on numpy)
    pred_np = pred_binary.squeeze().cpu().numpy().astype(bool)
    target_np = target_binary.squeeze().cpu().numpy().astype(bool)
    hd95 = compute_hd95(pred_np, target_np)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'specificity': specificity.item(),
        'hd95': hd95,
    }


# ============================================================================
# Inference-time Stabilization Methods
# ============================================================================

def apply_rejection_rules(coarse_mask, refined_mask,
                          iou_threshold=0.5,
                          area_ratio_min=0.3,
                          area_ratio_max=3.0,
                          max_components=5):
    """Apply rejection rules to decide whether to keep refined or fall back to coarse.

    Rejects refined predictions when they deviate excessively from the coarse mask,
    based on IoU consistency, area ratio constraints, and connected component count.

    Args:
        coarse_mask: Coarse prediction (H, W), binary or probability
        refined_mask: Refined prediction (H, W), binary or probability
        iou_threshold: Reject if IoU between coarse and refined < threshold
        area_ratio_min: Reject if refined_area / coarse_area < min
        area_ratio_max: Reject if refined_area / coarse_area > max
        max_components: Reject if refined has more connected components than this

    Returns:
        output_mask: Either refined_mask (if accepted) or coarse_mask (if rejected)
        rejected: Boolean indicating if the refinement was rejected
        reject_reason: String describing why rejected (or None if accepted)
    """
    # Convert to numpy for processing
    if torch.is_tensor(coarse_mask):
        coarse_np = coarse_mask.cpu().numpy()
    else:
        coarse_np = coarse_mask

    if torch.is_tensor(refined_mask):
        refined_np = refined_mask.cpu().numpy()
    else:
        refined_np = refined_mask

    # Binarize
    coarse_binary = (coarse_np > 0.5).astype(np.float32)
    refined_binary = (refined_np > 0.5).astype(np.float32)

    # Rule 1: IoU consistency check
    intersection = (coarse_binary * refined_binary).sum()
    union = coarse_binary.sum() + refined_binary.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)

    if iou < iou_threshold:
        return coarse_mask, True, f'IoU too low: {iou:.3f} < {iou_threshold}'

    # Rule 2: Area ratio check
    coarse_area = coarse_binary.sum()
    refined_area = refined_binary.sum()

    if coarse_area > 0:
        area_ratio = refined_area / coarse_area
        if area_ratio < area_ratio_min:
            return coarse_mask, True, f'Area ratio too small: {area_ratio:.3f} < {area_ratio_min}'
        if area_ratio > area_ratio_max:
            return coarse_mask, True, f'Area ratio too large: {area_ratio:.3f} > {area_ratio_max}'

    # Rule 3: Connected component count check
    labeled_array, num_components = ndimage_label(refined_binary)
    if num_components > max_components:
        return coarse_mask, True, f'Too many components: {num_components} > {max_components}'

    # All rules passed - accept refinement
    return refined_mask, False, None


def apply_boundary_fusion(coarse_mask, refined_mask, band_width=15):
    """Apply boundary-band fusion to restrict refinement to boundary regions.

    Restricts refinement to a narrow morphological band around the coarse mask
    boundary, while preserving the interior and exterior regions from the coarse mask.

    Args:
        coarse_mask: Coarse prediction (H, W), probability in [0, 1]
        refined_mask: Refined prediction (H, W), probability in [0, 1]
        band_width: Width of boundary band in pixels (e.g., 10-20)

    Returns:
        fused_mask: Mask with refinement only in boundary band
    """
    # Convert to numpy for morphological operations
    if torch.is_tensor(coarse_mask):
        coarse_np = coarse_mask.cpu().numpy()
        is_tensor = True
        device = coarse_mask.device
    else:
        coarse_np = coarse_mask
        is_tensor = False
        device = None

    if torch.is_tensor(refined_mask):
        refined_np = refined_mask.cpu().numpy()
    else:
        refined_np = refined_mask

    # Binarize coarse for morphological operations
    coarse_binary = (coarse_np > 0.5).astype(np.uint8)

    # Create structuring element
    struct = np.ones((3, 3), dtype=np.uint8)

    # Compute boundary band using dilation and erosion
    # Outer boundary: dilated - original
    dilated = binary_dilation(coarse_binary, structure=struct, iterations=band_width // 2)
    # Inner boundary: original - eroded
    eroded = binary_erosion(coarse_binary, structure=struct, iterations=band_width // 2)

    # Boundary band = dilated AND NOT eroded = region within band_width of boundary
    boundary_band = dilated.astype(np.float32) - eroded.astype(np.float32)
    boundary_band = np.clip(boundary_band, 0, 1)

    # Interior region (definitely inside, far from boundary)
    interior = eroded.astype(np.float32)

    # Exterior region (definitely outside, far from boundary)
    exterior = 1.0 - dilated.astype(np.float32)

    # Fuse: interior from coarse, exterior from coarse, boundary from refined
    # fused = interior * coarse + exterior * (1 - coarse) + boundary * refined
    # Simplified: fused = coarse * (1 - boundary_band) + refined * boundary_band
    fused_np = coarse_np * (1 - boundary_band) + refined_np * boundary_band

    # Convert back to tensor if input was tensor
    if is_tensor:
        fused_mask = torch.from_numpy(fused_np).float().to(device)
    else:
        fused_mask = fused_np

    return fused_mask


def apply_stabilization(coarse_mask, refined_mask,
                        use_rejection=False, use_boundary_fusion=False,
                        rejection_params=None, fusion_params=None):
    """Apply inference-time stabilization to refined predictions.

    Args:
        coarse_mask: Coarse prediction (B, H, W) or (H, W)
        refined_mask: Refined prediction (B, H, W) or (H, W)
        use_rejection: Whether to apply rejection rules
        use_boundary_fusion: Whether to apply boundary-band fusion
        rejection_params: Dict with rejection rule parameters
        fusion_params: Dict with boundary fusion parameters

    Returns:
        stabilized_mask: Stabilized refined prediction
        stats: Dict with stabilization statistics
    """
    if rejection_params is None:
        rejection_params = {}
    if fusion_params is None:
        fusion_params = {}

    # Handle batched input
    if coarse_mask.dim() == 3:
        batch_size = coarse_mask.shape[0]
        stabilized = []
        total_rejected = 0
        reject_reasons = []

        for i in range(batch_size):
            coarse_i = coarse_mask[i]
            refined_i = refined_mask[i]

            # Apply rejection rules first
            if use_rejection:
                refined_i, rejected, reason = apply_rejection_rules(
                    coarse_i, refined_i, **rejection_params
                )
                if rejected:
                    total_rejected += 1
                    reject_reasons.append(reason)

            # Apply boundary fusion (only if not rejected or always)
            if use_boundary_fusion:
                refined_i = apply_boundary_fusion(
                    coarse_i, refined_i, **fusion_params
                )

            stabilized.append(refined_i)

        # Stack back to batch
        if torch.is_tensor(stabilized[0]):
            stabilized_mask = torch.stack(stabilized, dim=0)
        else:
            stabilized_mask = np.stack(stabilized, axis=0)

        stats = {
            'rejected_count': total_rejected,
            'rejected_ratio': total_rejected / batch_size,
            'reject_reasons': reject_reasons
        }
    else:
        # Single sample
        stabilized_mask = refined_mask
        stats = {'rejected_count': 0, 'rejected_ratio': 0.0, 'reject_reasons': []}

        if use_rejection:
            stabilized_mask, rejected, reason = apply_rejection_rules(
                coarse_mask, stabilized_mask, **rejection_params
            )
            if rejected:
                stats['rejected_count'] = 1
                stats['rejected_ratio'] = 1.0
                stats['reject_reasons'] = [reason]

        if use_boundary_fusion:
            stabilized_mask = apply_boundary_fusion(
                coarse_mask, stabilized_mask, **fusion_params
            )

    return stabilized_mask, stats


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_sample(image, label, coarse_pred, refined_pred,
                     coarse_metrics, refined_metrics, sample_idx,
                     output_dir, dataset_name):
    """Visualize a single sample with semi-transparent mask overlays and metrics sidebar.

    Layout: 1 row x 4 panels
      [Image] [GT + Coarse overlay] [GT + Refined overlay] [Metrics sidebar]

    Overlays use semi-transparent colored masks:
      - GT: green (alpha=0.3)
      - Coarse/Refined: red (alpha=0.3)

    Args:
        image: Input image (H, W) or (H, W, 3)
        label: Ground truth mask (H, W)
        coarse_pred: Coarse prediction (H, W) in [0, 1]
        refined_pred: Refined prediction (H, W) in [0, 1]
        coarse_metrics: Dict with 'dice', 'iou', 'hd95'
        refined_metrics: Dict with 'dice', 'iou', 'hd95'
        sample_idx: Sample index for filename
        output_dir: Output directory
        dataset_name: Dataset name for subdirectory
    """
    # Convert to numpy if tensor
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()
    if torch.is_tensor(coarse_pred):
        coarse_pred = coarse_pred.cpu().numpy()
    if torch.is_tensor(refined_pred):
        refined_pred = refined_pred.cpu().numpy()

    # Normalize image for display
    if image.max() > 1:
        image = image / 255.0

    # Binarize
    coarse_binary = (coarse_pred > 0.5).astype(bool)
    refined_binary = (refined_pred > 0.5).astype(bool)
    label_binary = (label > 0.5).astype(bool)

    # Convert grayscale to RGB for overlay
    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image

    def overlay_masks(base_img, gt_mask, pred_mask):
        """Create overlay with GT in green and prediction in red, semi-transparent."""
        overlay = base_img.copy()
        alpha = 0.35
        # GT region: green
        overlay[gt_mask] = overlay[gt_mask] * (1 - alpha) + np.array([0, 0.8, 0]) * alpha
        # Prediction region: red
        overlay[pred_mask] = overlay[pred_mask] * (1 - alpha) + np.array([0.9, 0.15, 0.15]) * alpha
        # Overlap region (both GT and pred): blend to yellow-ish
        overlap = gt_mask & pred_mask
        overlay[overlap] = base_img[overlap] * (1 - alpha) + np.array([0.8, 0.7, 0]) * alpha
        return np.clip(overlay, 0, 1)

    # Create figure: 3 image panels + 1 metrics sidebar
    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.7], wspace=0.08)

    # Panel 1: Input Image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image_rgb)
    ax0.set_title('Input Image', fontsize=11, fontweight='bold', pad=8)
    ax0.axis('off')

    # Panel 2: GT (green) + Coarse (red) overlay
    ax1 = fig.add_subplot(gs[0, 1])
    overlay_coarse = overlay_masks(image_rgb, label_binary, coarse_binary)
    ax1.imshow(overlay_coarse)
    ax1.set_title('GT / Coarse', fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')

    # Panel 3: GT (green) + Refined (red) overlay
    ax2 = fig.add_subplot(gs[0, 2])
    overlay_refined = overlay_masks(image_rgb, label_binary, refined_binary)
    ax2.imshow(overlay_refined)
    ax2.set_title('GT / Refined', fontsize=11, fontweight='bold', pad=8)
    ax2.axis('off')

    # Panel 4: Metrics sidebar
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')

    # Format HD95
    def fmt_hd(val):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 'N/A'
        return f'{val:.2f}'

    c_dice = coarse_metrics['dice']
    r_dice = refined_metrics['dice']
    c_iou = coarse_metrics['iou']
    r_iou = refined_metrics['iou']
    c_hd95 = coarse_metrics['hd95']
    r_hd95 = refined_metrics['hd95']

    dice_imp = r_dice - c_dice
    iou_imp = r_iou - c_iou

    # HD95 delta
    if c_hd95 is not None and r_hd95 is not None and np.isfinite(c_hd95) and np.isfinite(r_hd95):
        hd95_delta = f'{c_hd95 - r_hd95:+.2f}'
    else:
        hd95_delta = 'N/A'

    lines = [
        f'Sample #{sample_idx}',
        '',
        '---- Coarse ----',
        f'  Dice:  {c_dice:.4f}',
        f'  IoU:   {c_iou:.4f}',
        f'  HD95:  {fmt_hd(c_hd95)}',
        '',
        '---- Refined ----',
        f'  Dice:  {r_dice:.4f}',
        f'  IoU:   {r_iou:.4f}',
        f'  HD95:  {fmt_hd(r_hd95)}',
        '',
        '---- Delta ----',
        f'  Dice:  {dice_imp:+.4f}',
        f'  IoU:   {iou_imp:+.4f}',
        f'  HD95:  {hd95_delta}',
        '',
        '---- Legend ----',
        '  Green:   GT',
        '  Red:     Prediction',
        '  Yellow:  Overlap',
    ]

    text = '\n'.join(lines)
    ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    # Save
    vis_dir = os.path.join(output_dir, dataset_name, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    improvement = r_dice - c_dice
    filename = f'sample_{sample_idx:04d}_imp{improvement:+.4f}.png'
    plt.savefig(os.path.join(vis_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_summary(results_list, output_dir, dataset_name):
    """Create a summary visualization showing improvement distribution.

    Args:
        results_list: List of (coarse_dice, refined_dice, improvement) tuples
        output_dir: Output directory
        dataset_name: Dataset name
    """
    coarse_dices = [r[0] for r in results_list]
    refined_dices = [r[1] for r in results_list]
    improvements = [r[2] for r in results_list]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Dice distribution
    axes[0].hist(coarse_dices, bins=20, alpha=0.7, label='Coarse', color='red')
    axes[0].hist(refined_dices, bins=20, alpha=0.7, label='Refined', color='blue')
    axes[0].set_xlabel('Dice Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{dataset_name}: Dice Distribution')
    axes[0].legend()
    axes[0].axvline(np.mean(coarse_dices), color='red', linestyle='--', label=f'Coarse mean: {np.mean(coarse_dices):.4f}')
    axes[0].axvline(np.mean(refined_dices), color='blue', linestyle='--', label=f'Refined mean: {np.mean(refined_dices):.4f}')

    # Improvement distribution
    axes[1].hist(improvements, bins=20, alpha=0.7, color='green')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=2)
    axes[1].axvline(np.mean(improvements), color='red', linestyle='--')
    axes[1].set_xlabel('Dice Improvement')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Improvement Distribution (mean: {np.mean(improvements):+.4f})')

    # Scatter plot: coarse vs refined
    axes[2].scatter(coarse_dices, refined_dices, alpha=0.5, s=20)
    axes[2].plot([0, 1], [0, 1], 'k--', label='No change')
    axes[2].set_xlabel('Coarse Dice')
    axes[2].set_ylabel('Refined Dice')
    axes[2].set_title('Coarse vs Refined Dice')
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([0, 1])
    axes[2].legend()

    # Count improvements vs degradations
    n_improved = sum(1 for i in improvements if i > 0)
    n_degraded = sum(1 for i in improvements if i < 0)
    n_same = sum(1 for i in improvements if i == 0)
    axes[2].text(0.05, 0.95, f'Improved: {n_improved}\nDegraded: {n_degraded}\nSame: {n_same}',
                 transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save to dataset's own directory
    vis_dir = os.path.join(output_dir, dataset_name, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_dataset(model, dataloader, device, refined_eval_size=224, use_gated=False,
                     visualize=False, num_visualize=10, visualize_best_worst=False,
                     output_dir=None, dataset_name=None,
                     use_rejection=False, rejection_params=None,
                     use_boundary_fusion=False, fusion_params=None,
                     threshold=0.5):
    """Evaluate model on a dataset.

    Returns:
        Dictionary with coarse and refined metrics
    """
    model.eval()

    coarse_metrics = defaultdict(list)
    refined_metrics = defaultdict(list)

    # For visualization
    vis_data = []  # Store (image, label, coarse, refined, coarse_dice, refined_dice, idx)
    sample_idx = 0

    # Stabilization statistics
    total_rejected = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            outputs = model(image)

            # Get predictions
            coarse_pred = outputs['coarse_mask']  # (B, H, W) at 224x224

            if use_gated:
                refined_pred = outputs['refined_mask']  # Already probabilities
            else:
                refined_pred = torch.sigmoid(outputs['refined_mask'])  # (B, H, W)

            # Apply inference-time stabilization if enabled
            if use_rejection or use_boundary_fusion:
                refined_pred, stab_stats = apply_stabilization(
                    coarse_pred, refined_pred,
                    use_rejection=use_rejection,
                    use_boundary_fusion=use_boundary_fusion,
                    rejection_params=rejection_params,
                    fusion_params=fusion_params
                )
                total_rejected += stab_stats['rejected_count']
                total_samples += coarse_pred.shape[0]

            # Coarse: evaluate at 224x224 (TransUNet native resolution)
            # Refined: evaluate at refined_eval_size
            if refined_eval_size >= 1024 and refined_pred.shape[-2:] != label.shape[-2:]:
                # Upsample label to match SAM output (preserves boundary details)
                label_for_refined = F.interpolate(
                    label.unsqueeze(1),
                    size=refined_pred.shape[-2:],
                    mode='nearest'
                ).squeeze(1)
                refined_pred_eval = refined_pred
            elif refined_pred.shape[-2:] != label.shape[-2:]:
                # Downsample SAM output to match label (default behavior)
                refined_pred_eval = F.interpolate(
                    refined_pred.unsqueeze(1),
                    size=label.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                label_for_refined = label
            else:
                label_for_refined = label
                refined_pred_eval = refined_pred

            # Compute per-sample metrics for visualization
            for i in range(image.shape[0]):
                sample_coarse_metrics = compute_metrics(
                    coarse_pred[i:i+1], label[i:i+1], threshold=threshold
                )
                sample_refined_metrics = compute_metrics(
                    refined_pred_eval[i:i+1], label_for_refined[i:i+1], threshold=threshold
                )

                coarse_dice = sample_coarse_metrics['dice']
                refined_dice = sample_refined_metrics['dice']

                for k, v in sample_coarse_metrics.items():
                    coarse_metrics[k].append(v)
                for k, v in sample_refined_metrics.items():
                    refined_metrics[k].append(v)

                # Store data for visualization
                if visualize:
                    # Downsample refined to match coarse for visualization
                    refined_for_vis = F.interpolate(
                        refined_pred[i:i+1].unsqueeze(1),
                        size=label.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                    vis_data.append({
                        'image': image[i, 0].cpu(),  # Assume grayscale
                        'label': label[i].cpu(),
                        'coarse': coarse_pred[i].cpu(),
                        'refined': refined_for_vis.cpu(),
                        'coarse_dice': coarse_dice,
                        'refined_dice': refined_dice,
                        'coarse_iou': sample_coarse_metrics['iou'],
                        'refined_iou': sample_refined_metrics['iou'],
                        'coarse_hd95': sample_coarse_metrics['hd95'],
                        'refined_hd95': sample_refined_metrics['hd95'],
                        'improvement': refined_dice - coarse_dice,
                        'idx': sample_idx
                    })

                sample_idx += 1

    # Average metrics (filter None values for HD95)
    def safe_mean(values):
        valid = [v for v in values if v is not None]
        return np.mean(valid) if valid else float('nan')

    avg_coarse = {k: safe_mean(v) if k == 'hd95' else np.mean(v) for k, v in coarse_metrics.items()}
    avg_refined = {k: safe_mean(v) if k == 'hd95' else np.mean(v) for k, v in refined_metrics.items()}

    # Generate visualizations
    if visualize and output_dir and dataset_name:
        print(f"  Generating visualizations...")

        # num_visualize <= 0 means visualize all
        visualize_all = num_visualize <= 0
        n_vis = len(vis_data) if visualize_all else num_visualize

        if visualize_best_worst and not visualize_all:
            # Sort by improvement and visualize best/worst
            vis_data_sorted = sorted(vis_data, key=lambda x: x['improvement'])

            # Worst cases (refinement hurt most)
            worst_cases = vis_data_sorted[:n_vis // 2]
            # Best cases (refinement helped most)
            best_cases = vis_data_sorted[-(n_vis // 2):]

            samples_to_vis = worst_cases + best_cases
        elif visualize_all:
            # Visualize all samples
            samples_to_vis = vis_data
        else:
            # Visualize evenly spaced samples
            step = max(1, len(vis_data) // n_vis)
            samples_to_vis = vis_data[::step][:n_vis]

        for sample in tqdm(samples_to_vis, desc='Visualizing', leave=False):
            visualize_sample(
                image=sample['image'],
                label=sample['label'],
                coarse_pred=sample['coarse'],
                refined_pred=sample['refined'],
                coarse_metrics={
                    'dice': sample['coarse_dice'],
                    'iou': sample['coarse_iou'],
                    'hd95': sample['coarse_hd95'],
                },
                refined_metrics={
                    'dice': sample['refined_dice'],
                    'iou': sample['refined_iou'],
                    'hd95': sample['refined_hd95'],
                },
                sample_idx=sample['idx'],
                output_dir=output_dir,
                dataset_name=dataset_name
            )

        # Generate summary visualization
        results_list = [(d['coarse_dice'], d['refined_dice'], d['improvement'])
                        for d in vis_data]
        visualize_summary(results_list, output_dir, dataset_name)

    result = {
        'coarse': avg_coarse,
        'refined': avg_refined,
        'n_samples': len(dataloader.dataset)
    }

    # Add stabilization stats if enabled
    if use_rejection or use_boundary_fusion:
        result['stabilization'] = {
            'rejected_count': total_rejected,
            'rejected_ratio': total_rejected / max(1, total_samples),
            'use_rejection': use_rejection,
            'use_boundary_fusion': use_boundary_fusion,
        }

    return result


def print_results_table(results, datasets):
    """Print results in a formatted table."""

    print("\n" + "=" * 120)
    print("                              TEST SET EVALUATION RESULTS")
    print("=" * 120)

    # Header
    print(f"\n{'Dataset':<15} {'Samples':>8} {'':^3} "
          f"{'Coarse (TransUNet)':^35} {'':^3} "
          f"{'Refined (SAM)':^35}")
    print(f"{'':<15} {'':>8} {'':^3} "
          f"{'Dice':>10} {'IoU':>10} {'HD95':>10} {'':^3} "
          f"{'Dice':>10} {'IoU':>10} {'HD95':>10} {'':^5}")
    print("-" * 120)

    # Per-dataset results
    total_coarse_dice = []
    total_refined_dice = []
    total_coarse_hd95 = []
    total_refined_hd95 = []
    total_samples = 0

    def fmt_hd95(val):
        """Format HD95 value, handling inf/nan."""
        if val is None or np.isnan(val) or np.isinf(val):
            return f"{'N/A':>10}"
        return f"{val:>10.2f}"

    for ds in datasets:
        if ds not in results:
            continue
        r = results[ds]
        n = r['n_samples']
        c = r['coarse']
        f = r['refined']

        # Calculate improvement
        dice_improvement = f['dice'] - c['dice']
        c_hd = c['hd95'] if c['hd95'] is not None and np.isfinite(c['hd95']) else None
        f_hd = f['hd95'] if f['hd95'] is not None and np.isfinite(f['hd95']) else None
        hd95_impr_str = f"HD95 {c_hd - f_hd:+.2f}" if c_hd is not None and f_hd is not None else "HD95 N/A"

        print(f"{ds:<15} {n:>8} {'':^3} "
              f"{c['dice']:>10.4f} {c['iou']:>10.4f} {fmt_hd95(c['hd95'])} {'':^3} "
              f"{f['dice']:>10.4f} {f['iou']:>10.4f} {fmt_hd95(f['hd95'])} "
              f"(Dice {dice_improvement:+.4f}, {hd95_impr_str})")

        total_coarse_dice.append(c['dice'])
        total_refined_dice.append(f['dice'])
        # Filter invalid values (nan/inf) for averaging
        if c['hd95'] is not None and np.isfinite(c['hd95']):
            total_coarse_hd95.append(c['hd95'])
        if f['hd95'] is not None and np.isfinite(f['hd95']):
            total_refined_hd95.append(f['hd95'])
        total_samples += n

    print("-" * 120)

    # Average across datasets
    avg_coarse_dice = np.mean(total_coarse_dice)
    avg_refined_dice = np.mean(total_refined_dice)
    avg_improvement = avg_refined_dice - avg_coarse_dice
    avg_coarse_hd95 = np.mean(total_coarse_hd95) if total_coarse_hd95 else float('nan')
    avg_refined_hd95 = np.mean(total_refined_hd95) if total_refined_hd95 else float('nan')
    avg_hd95_str = (f"HD95 {avg_coarse_hd95 - avg_refined_hd95:+.2f}"
                    if np.isfinite(avg_coarse_hd95) and np.isfinite(avg_refined_hd95) else "HD95 N/A")

    print(f"{'AVERAGE':<15} {total_samples:>8} {'':^3} "
          f"{avg_coarse_dice:>10.4f} {'':>10} {fmt_hd95(avg_coarse_hd95)} {'':^3} "
          f"{avg_refined_dice:>10.4f} {'':>10} {fmt_hd95(avg_refined_hd95)} "
          f"(Dice {avg_improvement:+.4f}, {avg_hd95_str})")

    print("=" * 120)

    # Detailed metrics for each dataset
    print("\n" + "=" * 120)
    print("                              DETAILED METRICS")
    print("=" * 120)

    for ds in datasets:
        if ds not in results:
            continue
        r = results[ds]
        c = r['coarse']
        f = r['refined']

        print(f"\n{ds} ({r['n_samples']} samples):")
        print(f"  Coarse (TransUNet):")
        print(f"    Dice: {c['dice']:.4f}, IoU: {c['iou']:.4f}, "
              f"Precision: {c['precision']:.4f}, Recall: {c['recall']:.4f}, "
              f"Accuracy: {c['accuracy']:.4f}, HD95: {fmt_hd95(c['hd95'])}")
        print(f"  Refined (SAM):")
        print(f"    Dice: {f['dice']:.4f}, IoU: {f['iou']:.4f}, "
              f"Precision: {f['precision']:.4f}, Recall: {f['recall']:.4f}, "
              f"Accuracy: {f['accuracy']:.4f}, HD95: {fmt_hd95(f['hd95'])}")
        c_hd95 = c['hd95'] if c['hd95'] is not None and np.isfinite(c['hd95']) else None
        f_hd95 = f['hd95'] if f['hd95'] is not None and np.isfinite(f['hd95']) else None
        hd95_str = f"HD95 {c_hd95 - f_hd95:+.2f} (lower is better)" if c_hd95 is not None and f_hd95 is not None else "HD95 N/A"
        print(f"  Improvement: Dice {f['dice'] - c['dice']:+.4f}, "
              f"IoU {f['iou'] - c['iou']:+.4f}, "
              f"{hd95_str}")

    print("\n" + "=" * 120)

    return {
        'avg_coarse_dice': avg_coarse_dice,
        'avg_refined_dice': avg_refined_dice,
        'avg_improvement': avg_improvement,
        'avg_coarse_hd95': avg_coarse_hd95,
        'avg_refined_hd95': avg_refined_hd95,
    }


def main():
    args = get_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Default datasets
    if args.datasets is None:
        if args.include_unseen:
            args.datasets = ALL_DATASETS
        else:
            args.datasets = SUPPORTED_DATASETS

    print(f"\nEvaluating on datasets: {args.datasets}")
    print(f"Refined evaluation size: {args.refined_eval_size}")
    print(f"Binarization threshold: {args.threshold}")

    # Print stabilization settings
    if args.use_rejection_rules:
        print(f"\nRejection rules ENABLED:")
        print(f"  IoU threshold: {args.reject_iou_threshold}")
        print(f"  Area ratio range: [{args.reject_area_ratio_min}, {args.reject_area_ratio_max}]")
        print(f"  Max components: {args.reject_max_components}")

    if args.use_boundary_fusion:
        print(f"\nBoundary-band fusion ENABLED:")
        print(f"  Band width: {args.boundary_band_width} pixels")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get config from checkpoint if available
    config = checkpoint.get('config', {})
    vit_name = config.get('vit_name', args.vit_name)
    sam_model_type = config.get('sam_model_type', args.sam_model_type)
    img_size = config.get('img_size', args.img_size)
    num_classes = config.get('num_classes', args.num_classes)

    print(f"Model config: vit_name={vit_name}, sam_model_type={sam_model_type}, img_size={img_size}")

    # Build model
    print("\nBuilding model...")
    if args.use_gated_refinement:
        model = build_gated_ultra_refiner(
            vit_name=vit_name,
            img_size=img_size,
            num_classes=num_classes,
            sam_model_type=sam_model_type,
            freeze_sam_image_encoder=True,
            freeze_sam_prompt_encoder=True,  # Freeze for inference
            mask_prompt_style=args.mask_prompt_style,
            use_roi_crop=args.use_roi_crop,
            roi_expand_ratio=args.roi_expand_ratio,
            gate_type=args.gate_type,
            gate_gamma=args.gate_gamma,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
        )
    else:
        model = build_ultra_refiner(
            vit_name=vit_name,
            img_size=img_size,
            num_classes=num_classes,
            sam_model_type=sam_model_type,
            freeze_sam_image_encoder=True,
            freeze_sam_prompt_encoder=True,  # Freeze for inference
            mask_prompt_style=args.mask_prompt_style,
            use_roi_crop=args.use_roi_crop,
            roi_expand_ratio=args.roi_expand_ratio,
        )

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")

    # Evaluate on each dataset
    results = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}...")
        print(f"{'='*60}")

        # Check if test set exists
        test_dir = os.path.join(args.data_root, dataset_name, 'test')
        if not os.path.exists(test_dir):
            print(f"  Warning: Test set not found at {test_dir}, skipping...")
            continue

        # Create dataloader
        try:
            dataloader = get_test_dataloader(
                data_root=args.data_root,
                dataset_name=dataset_name,
                batch_size=args.batch_size,
                img_size=img_size,
                num_workers=args.num_workers,
                for_sam=False
            )
            print(f"  Loaded {len(dataloader.dataset)} test samples")
        except Exception as e:
            print(f"  Error loading dataset: {e}, skipping...")
            continue

        # Prepare stabilization parameters
        rejection_params = {
            'iou_threshold': args.reject_iou_threshold,
            'area_ratio_min': args.reject_area_ratio_min,
            'area_ratio_max': args.reject_area_ratio_max,
            'max_components': args.reject_max_components,
        }
        fusion_params = {
            'band_width': args.boundary_band_width,
        }

        # Evaluate
        dataset_results = evaluate_dataset(
            model, dataloader, device,
            refined_eval_size=args.refined_eval_size,
            use_gated=args.use_gated_refinement,
            visualize=args.visualize,
            num_visualize=args.num_visualize,
            visualize_best_worst=args.visualize_best_worst,
            output_dir=args.output_dir,
            dataset_name=dataset_name,
            use_rejection=args.use_rejection_rules,
            rejection_params=rejection_params,
            use_boundary_fusion=args.use_boundary_fusion,
            fusion_params=fusion_params,
            threshold=args.threshold
        )
        results[dataset_name] = dataset_results

        # Print quick summary
        c = dataset_results['coarse']
        f = dataset_results['refined']
        summary_str = f"  Coarse Dice: {c['dice']:.4f}, Refined Dice: {f['dice']:.4f} (improvement: {f['dice'] - c['dice']:+.4f})"

        # Print stabilization stats if enabled
        if 'stabilization' in dataset_results:
            stab = dataset_results['stabilization']
            summary_str += f"\n  Stabilization: rejected {stab['rejected_count']}/{dataset_results['n_samples']} ({stab['rejected_ratio']*100:.1f}%)"

        print(summary_str)

        # Save per-dataset results to dataset's own directory
        if args.output_dir:
            import json

            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                return obj

            dataset_dir = os.path.join(args.output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)

            dataset_results_path = os.path.join(dataset_dir, 'test_result.json')
            with open(dataset_results_path, 'w') as f:
                json.dump(convert_to_serializable({
                    'dataset': dataset_name,
                    'args': vars(args),
                    'results': dataset_results,
                }), f, indent=2)
            print(f"  Results saved to: {dataset_results_path}")

    # Print final results table
    summary = print_results_table(results, args.datasets)

    # Save combined summary
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        import json

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        # Save combined summary at root level
        summary_path = os.path.join(args.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(convert_to_serializable({
                'args': vars(args),
                'results': results,
                'summary': summary,
            }), f, indent=2)
        print(f"\nCombined summary saved to: {summary_path}")

    print("\nEvaluation complete!")
    return results


if __name__ == '__main__':
    main()
