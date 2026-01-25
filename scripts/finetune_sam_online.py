"""
Finetune SAM with Online Mask Augmentation (Phase 2)

This script finetunes SAM's mask decoder using on-the-fly mask augmentation,
eliminating the need for pre-generated augmented data.

Benefits of Online Augmentation:
- No disk storage required for augmented masks
- New augmentation every epoch (unlimited diversity)
- 12 primary error types simulating TransUNet failures
- Soft mask conversion matching TransUNet output distribution

The augmentation system includes:
- 12 primary error types (identity, over/under-seg, holes, bridges, etc.)
- Secondary perturbations (boundary jitter, threshold fluctuation)
- Soft mask conversion via signed distance transform
- Configurable presets (default, mild, severe, boundary_focus, structural)

Usage:
    python scripts/finetune_sam_online.py \
        --data_root ./dataset/processed \
        --datasets BUSI BUSBRA BUS \
        --sam_checkpoint ./pretrained/medsam_vit_b.pth \
        --output_dir ./checkpoints/sam_finetuned \
        --augmentor_preset default \
        --mask_prompt_style direct \
        --transunet_img_size 224 \
        --use_roi_crop \
        --roi_expand_ratio 0.2 \
        --epochs 50
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_online_augmented_dataloaders, OnlineAugmentedDataset, MultiDatasetOnlineAugmented
from models.sam_refiner import DifferentiableSAMRefiner
from utils import dice_score, iou_score


def get_args():
    parser = argparse.ArgumentParser(description='Finetune SAM with online augmentation')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Dataset names (e.g., BUSI BUSBRA BUS)')

    # SAM arguments
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')

    # Augmentation arguments
    parser.add_argument('--augmentor_preset', type=str, default='default',
                        choices=['default', 'mild', 'severe', 'boundary_focus', 'structural'],
                        help='Augmentation preset: default (balanced), mild (more identity), '
                             'severe (more extreme failures), boundary_focus (50%% over/under-seg), '
                             'structural (40%% holes+missing+fragmentation)')
    parser.add_argument('--soft_mask_prob', type=float, default=0.8,
                        help='Probability of converting to soft mask (0.8 = 80%%)')
    parser.add_argument('--soft_mask_temp_min', type=float, default=2.0,
                        help='Minimum temperature for soft mask conversion')
    parser.add_argument('--soft_mask_temp_max', type=float, default=8.0,
                        help='Maximum temperature for soft mask conversion')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Speed optimizations
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (AMP) for faster training')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)')
    parser.add_argument('--compile_model', action='store_true',
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Number of batches to prefetch per worker')

    # Prompt arguments
    parser.add_argument('--use_point_prompt', action='store_true', default=True,
                        help='Use point prompts')
    parser.add_argument('--use_box_prompt', action='store_true', default=True,
                        help='Use box prompts')
    parser.add_argument('--use_mask_prompt', action='store_true', default=True,
                        help='Use mask prompts')
    parser.add_argument('--mask_prompt_style', type=str, default='direct',
                        choices=['gaussian', 'direct', 'distance'],
                        help='Mask prompt style: direct (RECOMMENDED for soft masks), '
                             'gaussian (adds blur), distance (SDF-like)')

    # Quality-aware loss arguments
    parser.add_argument('--change_penalty_weight', type=float, default=0.5,
                        help='Weight for quality-aware change penalty (0=disabled, 0.5=default)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sam_finetuned',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/sam_finetuned',
                        help='TensorBoard log directory')

    # Phase 3 compatibility
    parser.add_argument('--transunet_img_size', type=int, default=224,
                        help='Intermediate resolution to simulate TransUNet output path')

    # ROI cropping
    parser.add_argument('--use_roi_crop', action='store_true',
                        help='Enable ROI cropping for higher effective resolution')
    parser.add_argument('--roi_expand_ratio', type=float, default=0.2,
                        help='Ratio to expand ROI bounding box')

    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint save interval (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--split_ratio', type=float, default=0.9,
                        help='Train/val split ratio')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_dice_batch(pred, gt, threshold=0.5):
    """Compute Dice score for batch."""
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > threshold).float()

    intersection = (pred_binary * gt_binary).sum(dim=(-2, -1))
    union = pred_binary.sum(dim=(-2, -1)) + gt_binary.sum(dim=(-2, -1))

    dice = (2 * intersection + 1) / (union + 1)
    return dice


def quality_aware_loss(pred_masks, gt_masks, coarse_masks, iou_preds=None,
                       change_penalty_weight=0.5):
    """
    Quality-aware loss function with change penalty.

    Key insight: The Refiner should learn "when not to modify".
    - High-quality inputs (high Dice coarse vs GT) should be PRESERVED
    - Low-quality inputs should be STRONGLY CORRECTED
    """
    pred_sigmoid = torch.sigmoid(pred_masks)

    # BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)

    # Dice loss
    intersection = (pred_sigmoid * gt_masks).sum(dim=(-2, -1))
    union = pred_sigmoid.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1))
    dice_loss = 1 - (2 * intersection + 1) / (union + 1)
    dice_loss = dice_loss.mean()

    # Focal loss for hard examples
    pt = torch.where(gt_masks == 1, pred_sigmoid, 1 - pred_sigmoid)
    focal_weight = (1 - pt) ** 2
    focal_loss = F.binary_cross_entropy_with_logits(
        pred_masks, gt_masks, reduction='none'
    )
    focal_loss = (focal_weight * focal_loss).mean()

    # Quality-aware change penalty
    with torch.no_grad():
        input_quality = compute_dice_batch(coarse_masks, gt_masks)
        penalty_weight = torch.clamp((input_quality - 0.6) / 0.4, 0, 1)

    change_magnitude = (pred_sigmoid - coarse_masks).pow(2).mean(dim=(-2, -1))
    change_penalty = (change_magnitude * penalty_weight).mean()

    # IoU prediction loss
    if iou_preds is not None:
        with torch.no_grad():
            actual_iou = intersection / (union - intersection + 1)
        iou_loss = F.mse_loss(iou_preds, actual_iou)
    else:
        iou_loss = torch.tensor(0.0)

    # Combined loss
    total_loss = (
        bce_loss +
        dice_loss +
        0.5 * focal_loss +
        change_penalty_weight * change_penalty +
        0.1 * iou_loss
    )

    return total_loss, {
        'bce': bce_loss.item(),
        'dice': dice_loss.item(),
        'focal': focal_loss.item(),
        'change_penalty': change_penalty.item(),
        'iou_loss': iou_loss.item() if isinstance(iou_loss, torch.Tensor) else 0,
        'avg_input_quality': input_quality.mean().item(),
    }


def train_epoch(model, dataloader, optimizer, device, epoch, change_penalty_weight=0.5,
                scaler=None, grad_accum_steps=1):
    """Train for one epoch with quality-aware loss and optional AMP."""
    model.train()

    total_loss = 0
    loss_components = {'bce': 0, 'dice': 0, 'focal': 0, 'change_penalty': 0, 'iou_loss': 0, 'avg_input_quality': 0}
    error_type_counts = {}
    num_batches = 0

    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        # Non-blocking transfer for speed
        image = batch['image'].to(device, non_blocking=True)
        gt_mask = batch['label'].to(device, non_blocking=True)
        coarse_mask = batch['coarse_mask'].to(device, non_blocking=True)

        # Track error types for logging
        if 'error_type' in batch:
            for et in batch['error_type']:
                error_type_counts[et] = error_type_counts.get(et, 0) + 1

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=use_amp):
            result = model(image, coarse_mask, image_already_normalized=True)
            refined_masks = result['masks']

            # Compute quality-aware loss
            loss, components = quality_aware_loss(
                refined_masks, gt_mask, coarse_mask,
                change_penalty_weight=change_penalty_weight
            )
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

        # Backward pass with AMP
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (with gradient accumulation)
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Track losses (unscaled)
        total_loss += loss.item() * grad_accum_steps
        for k, v in components.items():
            if k in loss_components:
                loss_components[k] += v
        num_batches += 1

        pbar.set_postfix({
            'loss': loss.item() * grad_accum_steps,
            'dice': components['dice'],
            'chg_pen': components['change_penalty'],
            'inp_qual': components['avg_input_quality'],
        })

    # Handle remaining gradients if batch count not divisible by grad_accum_steps
    if num_batches % grad_accum_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    # Average losses
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    # Log error type distribution
    loss_components['error_types'] = error_type_counts

    return avg_loss, loss_components


def validate(model, dataloader, device, use_amp=False):
    """Validate the model."""
    model.eval()

    coarse_metrics = {'dice': [], 'iou': []}
    refined_metrics = {'dice': [], 'iou': []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image = batch['image'].to(device, non_blocking=True)
            gt_mask = batch['label'].to(device, non_blocking=True)
            coarse_mask = batch['coarse_mask'].to(device, non_blocking=True)

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                result = model(image, coarse_mask, image_already_normalized=True)
                refined_masks = torch.sigmoid(result['masks'])

            # Compute metrics for each sample
            for i in range(image.shape[0]):
                gt_np = gt_mask[i].cpu().numpy()
                coarse_np = coarse_mask[i].cpu().numpy()
                refined_np = refined_masks[i].cpu().numpy()

                # Coarse metrics
                coarse_metrics['dice'].append(dice_score(coarse_np, gt_np))
                coarse_metrics['iou'].append(iou_score(coarse_np, gt_np))

                # Refined metrics
                refined_metrics['dice'].append(dice_score(refined_np, gt_np))
                refined_metrics['iou'].append(iou_score(refined_np, gt_np))

    # Average metrics
    result = {
        'coarse': {k: np.mean(v) for k, v in coarse_metrics.items()},
        'refined': {k: np.mean(v) for k, v in refined_metrics.items()},
    }

    # Compute deltas
    result['delta'] = {
        k: result['refined'][k] - result['coarse'][k]
        for k in result['refined']
    }

    return result


def print_validation_comparison(val_result, is_best=False):
    """Print validation metrics comparison table."""
    print("\n" + "=" * 70)
    if is_best:
        print("                    VALIDATION RESULTS (NEW BEST)")
    else:
        print("                    VALIDATION RESULTS")
    print("=" * 70)
    print(f"{'Metric':<15} {'Coarse':<15} {'Refined':<15} {'Delta':<15}")
    print("-" * 70)

    for metric in ['dice', 'iou']:
        coarse = val_result['coarse'][metric]
        refined = val_result['refined'][metric]
        delta = val_result['delta'][metric]

        if delta > 0.005:
            indicator = "+"
        elif delta < -0.005:
            indicator = "-"
        else:
            indicator = "="

        print(f"{metric.upper():<15} {coarse:>12.4f}    {refined:>12.4f}    {delta:>+10.4f} {indicator}")

    print("=" * 70)


def print_error_type_distribution(error_types):
    """Print error type distribution for the epoch."""
    if not error_types:
        return

    total = sum(error_types.values())
    print("\nError Type Distribution:")
    print("-" * 40)
    for et, count in sorted(error_types.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {et:<25} {count:>5} ({pct:>5.1f}%)")
    print("-" * 40)


def main():
    args = get_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create dataset name for output
    dataset_str = '_'.join(args.datasets) if len(args.datasets) > 1 else args.datasets[0]

    # Create output directories
    output_dir = os.path.join(args.output_dir, dataset_str)
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(args.log_dir, dataset_str)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # Print augmentation settings
    print("\n" + "=" * 60)
    print("ONLINE AUGMENTATION SETTINGS")
    print("=" * 60)
    print(f"Augmentor preset: {args.augmentor_preset}")
    print(f"Soft mask probability: {args.soft_mask_prob}")
    print(f"Soft mask temperature: [{args.soft_mask_temp_min}, {args.soft_mask_temp_max}]")
    print(f"TransUNet resolution path: {args.transunet_img_size}x{args.transunet_img_size} -> 1024x1024")
    if args.use_roi_crop:
        print(f"ROI cropping enabled: expand_ratio={args.roi_expand_ratio}")
    print("=" * 60 + "\n")

    # Speed optimization settings
    print("SPEED OPTIMIZATIONS")
    print("=" * 60)
    print(f"Mixed precision (AMP): {args.use_amp}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"Compile model: {args.compile_model}")
    print(f"Prefetch factor: {args.prefetch_factor}")
    print("=" * 60 + "\n")

    # Load SAM model
    print(f'Loading SAM model from {args.sam_checkpoint}')
    from segment_anything import sam_model_registry
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam = sam.to(device)

    # Create SAM refiner
    model = DifferentiableSAMRefiner(
        sam_model=sam,
        use_point_prompt=args.use_point_prompt,
        use_box_prompt=args.use_box_prompt,
        use_mask_prompt=args.use_mask_prompt,
        freeze_image_encoder=True,
        freeze_prompt_encoder=False,
        mask_prompt_style=args.mask_prompt_style,
        use_roi_crop=args.use_roi_crop,
        roi_expand_ratio=args.roi_expand_ratio,
    )
    model = model.to(device)

    # Optional: Compile model for faster training (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled successfully")

    # Get dataloaders with online augmentation (with speed optimizations)
    print(f'Loading datasets from {args.data_root}')
    print(f'Datasets: {args.datasets}')

    train_loader, val_loader = get_online_augmented_dataloaders(
        data_root=args.data_root,
        dataset_names=args.datasets,
        batch_size=args.batch_size,
        img_size=1024,
        transunet_img_size=args.transunet_img_size,
        augmentor_preset=args.augmentor_preset,
        num_workers=args.num_workers,
        split_ratio=args.split_ratio,
        seed=args.seed,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )

    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.1f}%)')

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Setup AMP scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    if args.use_amp:
        print("Mixed precision training enabled (AMP)")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            print(f'Resumed from epoch {start_epoch - 1}, best_dice: {best_dice:.4f}')
        else:
            print(f'Warning: checkpoint not found at {args.resume}, starting from scratch')

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*60}')

        # Train (with AMP and gradient accumulation)
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, device, epoch,
            change_penalty_weight=args.change_penalty_weight,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
        )

        print(f'\nTraining Loss: {train_loss:.4f}')
        print(f'  BCE: {loss_components["bce"]:.4f}')
        print(f'  Dice: {loss_components["dice"]:.4f}')
        print(f'  Focal: {loss_components["focal"]:.4f}')
        print(f'  Change Penalty: {loss_components["change_penalty"]:.4f}')
        print(f'  Avg Input Quality: {loss_components["avg_input_quality"]:.4f}')

        # Print error type distribution
        if 'error_types' in loss_components:
            print_error_type_distribution(loss_components['error_types'])

        # Log to TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        for k, v in loss_components.items():
            if k != 'error_types':
                writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Validation
        if epoch % args.val_interval == 0:
            val_result = validate(model, val_loader, device, use_amp=args.use_amp)

            is_best = val_result['refined']['dice'] > best_dice
            if is_best:
                best_dice = val_result['refined']['dice']

            print_validation_comparison(val_result, is_best)

            # Log validation metrics
            for prefix in ['coarse', 'refined', 'delta']:
                for metric, value in val_result[prefix].items():
                    writer.add_scalar(f'val/{prefix}_{metric}', value, epoch)

            # Save best checkpoint
            if is_best:
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'val_result': val_result,
                    'args': vars(args),
                }
                torch.save(checkpoint, os.path.join(output_dir, 'best.pth'))

                # Save SAM-compatible checkpoint for Phase 3
                sam_checkpoint = {
                    'model': model.sam.state_dict(),
                }
                torch.save(sam_checkpoint, os.path.join(output_dir, 'best_sam.pth'))
                print(f'Saved best checkpoint with Dice: {best_dice:.4f}')

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch}.pth'))

            sam_checkpoint = {'model': model.sam.state_dict()}
            torch.save(sam_checkpoint, os.path.join(output_dir, f'epoch_{epoch}_sam.pth'))

        scheduler.step()

    # Final summary
    print(f'\n{"="*60}')
    print('TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'Best Dice: {best_dice:.4f}')
    print(f'Checkpoints saved to: {output_dir}')
    print(f'Logs saved to: {log_dir}')

    writer.close()


if __name__ == '__main__':
    main()
