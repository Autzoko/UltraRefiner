"""
Finetune SAM with Augmented Training Data

This script finetunes SAM's mask decoder using pre-generated augmented data
where coarse masks simulate various segmentation failure patterns with
controlled Dice score distribution.

The augmented data provides:
- 30% samples with Dice 0.6-0.8 (severe failures)
- 50% samples with Dice 0.8-0.9 (moderate errors)
- 20% samples with Dice 0.9+ (minor artifacts)

Usage:
    python scripts/finetune_sam_augmented.py \
        --data_root ./dataset/augmented \
        --dataset BUSI \
        --sam_checkpoint ./checkpoints/sam/sam_vit_b_01ec64.pth \
        --output_dir ./checkpoints/sam_finetuned \
        --epochs 50 \
        --batch_size 4
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

from data import get_augmented_dataloaders, CurriculumAugmentedDataset
from models.sam_refiner import DifferentiableSAMRefiner
from utils import dice_score, iou_score


def get_args():
    parser = argparse.ArgumentParser(description='Finetune SAM with augmented data')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/augmented',
                        help='Root directory containing augmented datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')

    # SAM arguments
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')

    # Prompt arguments
    parser.add_argument('--use_point_prompt', action='store_true', default=True,
                        help='Use point prompts')
    parser.add_argument('--use_box_prompt', action='store_true', default=True,
                        help='Use box prompts')
    parser.add_argument('--use_mask_prompt', action='store_true', default=True,
                        help='Use mask prompts')
    parser.add_argument('--mask_prompt_style', type=str, default='gaussian',
                        choices=['gaussian', 'direct', 'distance'],
                        help='Mask prompt style: gaussian (soft boundaries), direct (sharp), distance (SDF-like)')

    # Quality-aware loss arguments
    parser.add_argument('--change_penalty_weight', type=float, default=0.5,
                        help='Weight for quality-aware change penalty (0=disabled, 0.5=default)')

    # Training strategy
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning (easy to hard)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to use (for debugging)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sam_finetuned',
                        help='Output directory for checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs/sam_finetuned',
                        help='TensorBoard log directory')

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

    The change penalty encourages preservation when the input is already good:
    - change_penalty = ||refined - coarse|| * input_quality
    - Where input_quality = Dice(coarse, GT)

    For high-quality inputs (Dice ~0.95+): strong penalty for changes
    For low-quality inputs (Dice ~0.6): minimal penalty for changes

    Args:
        pred_masks: Predicted/refined masks (B, H, W) - logits
        gt_masks: Ground truth masks (B, H, W)
        coarse_masks: Input coarse masks (B, H, W)
        iou_preds: IoU predictions (B,) or None
        change_penalty_weight: Weight for change penalty term

    Returns:
        Total loss and component breakdown
    """
    pred_sigmoid = torch.sigmoid(pred_masks)

    # =========================================================================
    # 1. Standard segmentation losses (to match GT)
    # =========================================================================

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

    # =========================================================================
    # 2. Quality-aware change penalty (to preserve good inputs)
    # =========================================================================

    # Compute input quality: Dice between coarse mask and GT
    with torch.no_grad():
        input_quality = compute_dice_batch(coarse_masks, gt_masks)  # (B,)
        # Normalize to [0, 1] with emphasis on high-quality inputs
        # quality=0.6 -> penalty_weight=0.0, quality=1.0 -> penalty_weight=1.0
        penalty_weight = torch.clamp((input_quality - 0.6) / 0.4, 0, 1)

    # Change magnitude: L2 difference between refined and coarse
    change_magnitude = (pred_sigmoid - coarse_masks).pow(2).mean(dim=(-2, -1))  # (B,)

    # Quality-weighted change penalty
    # High quality (penalty_weight ~1.0) -> penalize changes heavily
    # Low quality (penalty_weight ~0.0) -> don't penalize changes
    change_penalty = (change_magnitude * penalty_weight).mean()

    # =========================================================================
    # 3. IoU prediction loss
    # =========================================================================
    if iou_preds is not None:
        with torch.no_grad():
            actual_iou = intersection / (union - intersection + 1)
        iou_loss = F.mse_loss(iou_preds, actual_iou)
    else:
        iou_loss = torch.tensor(0.0)

    # =========================================================================
    # 4. Combined loss
    # =========================================================================
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


def combined_loss(pred_masks, gt_masks, iou_preds=None):
    """
    Standard combined loss function (backward compatible).
    Use quality_aware_loss for full functionality.
    """
    # BCE + Dice loss
    bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)

    # Dice loss
    pred_sigmoid = torch.sigmoid(pred_masks)
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

    # IoU prediction loss
    if iou_preds is not None:
        with torch.no_grad():
            actual_iou = intersection / (union - intersection + 1)
        iou_loss = F.mse_loss(iou_preds, actual_iou)
    else:
        iou_loss = 0

    total_loss = bce_loss + dice_loss + 0.5 * focal_loss + 0.1 * iou_loss

    return total_loss, {
        'bce': bce_loss.item(),
        'dice': dice_loss.item(),
        'focal': focal_loss.item(),
        'iou_loss': iou_loss.item() if isinstance(iou_loss, torch.Tensor) else 0,
    }


def train_epoch(model, dataloader, optimizer, device, epoch, change_penalty_weight=0.5):
    """
    Train for one epoch with quality-aware loss.

    The quality-aware loss includes a change penalty that:
    - Penalizes modifications to high-quality inputs (preservation learning)
    - Allows strong modifications to low-quality inputs (refinement learning)
    """
    model.train()

    total_loss = 0
    loss_components = {'bce': 0, 'dice': 0, 'focal': 0, 'change_penalty': 0, 'iou_loss': 0, 'avg_input_quality': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        image = batch['image'].to(device)
        gt_mask = batch['label'].to(device)
        coarse_mask = batch['coarse_mask'].to(device)

        # Forward pass
        # Note: image is already normalized by the dataset (for_sam=True)
        optimizer.zero_grad()
        result = model(image, coarse_mask, image_already_normalized=True)

        refined_masks = result['masks']  # (B, H, W) - logits

        # Compute quality-aware loss with change penalty
        loss, components = quality_aware_loss(
            refined_masks, gt_mask, coarse_mask,
            change_penalty_weight=change_penalty_weight
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        for k, v in components.items():
            if k in loss_components:
                loss_components[k] += v
        num_batches += 1

        pbar.set_postfix({
            'loss': loss.item(),
            'dice': components['dice'],
            'chg_pen': components['change_penalty'],
            'inp_qual': components['avg_input_quality'],
        })

    # Average losses
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches

    return avg_loss, loss_components


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()

    coarse_metrics = {'dice': [], 'iou': []}
    refined_metrics = {'dice': [], 'iou': []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image = batch['image'].to(device)
            gt_mask = batch['label'].to(device)
            coarse_mask = batch['coarse_mask'].to(device)

            # Forward pass (image is already normalized by dataset)
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
            indicator = "↑"
        elif delta < -0.005:
            indicator = "↓"
        else:
            indicator = "→"

        print(f"{metric.upper():<15} {coarse:>12.4f}    {refined:>12.4f}    {delta:>+10.4f} {indicator}")

    print("=" * 70)


def main():
    args = get_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directories
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    log_dir = os.path.join(args.log_dir, args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

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
        freeze_image_encoder=True,  # Always freeze for efficiency
        freeze_prompt_encoder=False,
        mask_prompt_style=args.mask_prompt_style,
    )
    print(f'Mask prompt style: {args.mask_prompt_style}')
    model = model.to(device)

    # Get dataloaders
    print(f'Loading augmented data from {args.data_root}/{args.dataset}')

    if args.curriculum:
        # Use curriculum learning dataset
        from torch.utils.data import DataLoader
        train_dataset = CurriculumAugmentedDataset(
            data_root=args.data_root,
            dataset_name=args.dataset,
            img_size=1024,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        # Validation loader (no curriculum)
        from data import AugmentedSAMDataset
        val_dataset = AugmentedSAMDataset(
            data_root=args.data_root,
            dataset_name=args.dataset,
            img_size=1024,
            split_ratio=0.9,
            is_train=False,
            seed=args.seed,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader, val_loader = get_augmented_dataloaders(
            data_root=args.data_root,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            img_size=1024,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            split_ratio=0.9,
            seed=args.seed,
        )

    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')

    # Setup optimizer (only train non-frozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)

    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.1f}%)')

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*60}')

        # Update curriculum difficulty if enabled
        if args.curriculum:
            train_dataset.update_difficulty(epoch - 1, args.epochs)
            print(f'Curriculum difficulty: {train_dataset.difficulty:.2f}')
            print(f'Current training samples: {len(train_dataset)}')

        # Train with quality-aware loss
        train_loss, loss_components = train_epoch(
            model, train_loader, optimizer, device, epoch,
            change_penalty_weight=args.change_penalty_weight
        )

        print(f'\nTraining Loss: {train_loss:.4f}')
        print(f'  BCE: {loss_components["bce"]:.4f}')
        print(f'  Dice: {loss_components["dice"]:.4f}')
        print(f'  Focal: {loss_components["focal"]:.4f}')
        print(f'  Change Penalty: {loss_components["change_penalty"]:.4f}')
        print(f'  Avg Input Quality: {loss_components["avg_input_quality"]:.4f}')

        # Log to TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        for k, v in loss_components.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)

        # Validation
        if epoch % args.val_interval == 0:
            val_result = validate(model, val_loader, device)

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
