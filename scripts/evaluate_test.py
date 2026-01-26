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
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_ultra_refiner, build_gated_ultra_refiner, CONFIGS
from data import get_test_dataloader, SUPPORTED_DATASETS


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
                        help='Dataset names to evaluate (default: all)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Evaluation arguments
    parser.add_argument('--refined_eval_size', type=int, default=224,
                        help='Resolution for evaluating SAM refined output. '
                             '224 = downsample SAM to match label (default), '
                             '1024 = upsample label to match SAM (preserves boundary details)')

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

    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save prediction masks')
    parser.add_argument('--output_dir', type=str, default='./results/test_evaluation',
                        help='Output directory for predictions')

    return parser.parse_args()


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

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item(),
        'specificity': specificity.item(),
    }


def evaluate_dataset(model, dataloader, device, refined_eval_size=224, use_gated=False):
    """Evaluate model on a dataset.

    Returns:
        Dictionary with coarse and refined metrics
    """
    model.eval()

    coarse_metrics = defaultdict(list)
    refined_metrics = defaultdict(list)

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

            # Coarse: evaluate at 224x224 (TransUNet native resolution)
            batch_coarse_metrics = compute_metrics(coarse_pred, label)
            for k, v in batch_coarse_metrics.items():
                coarse_metrics[k].append(v)

            # Refined: evaluate at refined_eval_size
            if refined_eval_size >= 1024 and refined_pred.shape[-2:] != label.shape[-2:]:
                # Upsample label to match SAM output (preserves boundary details)
                label_for_refined = F.interpolate(
                    label.unsqueeze(1),
                    size=refined_pred.shape[-2:],
                    mode='nearest'
                ).squeeze(1)
            elif refined_pred.shape[-2:] != label.shape[-2:]:
                # Downsample SAM output to match label (default behavior)
                refined_pred = F.interpolate(
                    refined_pred.unsqueeze(1),
                    size=label.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                label_for_refined = label
            else:
                label_for_refined = label

            batch_refined_metrics = compute_metrics(refined_pred, label_for_refined)
            for k, v in batch_refined_metrics.items():
                refined_metrics[k].append(v)

    # Average metrics
    avg_coarse = {k: np.mean(v) for k, v in coarse_metrics.items()}
    avg_refined = {k: np.mean(v) for k, v in refined_metrics.items()}

    return {
        'coarse': avg_coarse,
        'refined': avg_refined,
        'n_samples': len(dataloader.dataset)
    }


def print_results_table(results, datasets):
    """Print results in a formatted table."""

    print("\n" + "=" * 100)
    print("                           TEST SET EVALUATION RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Dataset':<15} {'Samples':>8} {'':^3} {'Coarse (TransUNet)':^30} {'':^3} {'Refined (SAM)':^30}")
    print(f"{'':<15} {'':>8} {'':^3} {'Dice':>10} {'IoU':>10} {'Prec':>10} {'':^3} {'Dice':>10} {'IoU':>10} {'Prec':>10}")
    print("-" * 100)

    # Per-dataset results
    total_coarse_dice = []
    total_refined_dice = []
    total_samples = 0

    for ds in datasets:
        if ds not in results:
            continue
        r = results[ds]
        n = r['n_samples']
        c = r['coarse']
        f = r['refined']

        # Calculate improvement
        dice_improvement = f['dice'] - c['dice']

        print(f"{ds:<15} {n:>8} {'':^3} "
              f"{c['dice']:>10.4f} {c['iou']:>10.4f} {c['precision']:>10.4f} {'':^3} "
              f"{f['dice']:>10.4f} {f['iou']:>10.4f} {f['precision']:>10.4f} "
              f"({dice_improvement:+.4f})")

        total_coarse_dice.append(c['dice'])
        total_refined_dice.append(f['dice'])
        total_samples += n

    print("-" * 100)

    # Average across datasets
    avg_coarse_dice = np.mean(total_coarse_dice)
    avg_refined_dice = np.mean(total_refined_dice)
    avg_improvement = avg_refined_dice - avg_coarse_dice

    print(f"{'AVERAGE':<15} {total_samples:>8} {'':^3} "
          f"{avg_coarse_dice:>10.4f} {'':>10} {'':>10} {'':^3} "
          f"{avg_refined_dice:>10.4f} {'':>10} {'':>10} "
          f"({avg_improvement:+.4f})")

    print("=" * 100)

    # Detailed metrics for each dataset
    print("\n" + "=" * 100)
    print("                           DETAILED METRICS")
    print("=" * 100)

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
              f"Accuracy: {c['accuracy']:.4f}")
        print(f"  Refined (SAM):")
        print(f"    Dice: {f['dice']:.4f}, IoU: {f['iou']:.4f}, "
              f"Precision: {f['precision']:.4f}, Recall: {f['recall']:.4f}, "
              f"Accuracy: {f['accuracy']:.4f}")
        print(f"  Improvement: Dice {f['dice'] - c['dice']:+.4f}, IoU {f['iou'] - c['iou']:+.4f}")

    print("\n" + "=" * 100)

    return {
        'avg_coarse_dice': avg_coarse_dice,
        'avg_refined_dice': avg_refined_dice,
        'avg_improvement': avg_improvement,
    }


def main():
    args = get_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Default datasets
    if args.datasets is None:
        args.datasets = SUPPORTED_DATASETS

    print(f"\nEvaluating on datasets: {args.datasets}")
    print(f"Refined evaluation size: {args.refined_eval_size}")

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

        # Evaluate
        dataset_results = evaluate_dataset(
            model, dataloader, device,
            refined_eval_size=args.refined_eval_size,
            use_gated=args.use_gated_refinement
        )
        results[dataset_name] = dataset_results

        # Print quick summary
        c = dataset_results['coarse']
        f = dataset_results['refined']
        print(f"  Coarse Dice: {c['dice']:.4f}, Refined Dice: {f['dice']:.4f} "
              f"(improvement: {f['dice'] - c['dice']:+.4f})")

    # Print final results table
    summary = print_results_table(results, args.datasets)

    # Save results
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

        results_path = os.path.join(args.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(convert_to_serializable({
                'args': vars(args),
                'results': results,
                'summary': summary,
            }), f, indent=2)
        print(f"\nResults saved to: {results_path}")

    print("\nEvaluation complete!")
    return results


if __name__ == '__main__':
    main()
