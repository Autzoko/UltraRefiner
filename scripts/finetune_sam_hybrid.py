#!/usr/bin/env python3
"""
Phase 2: Finetune SAM with Hybrid Data (Real Predictions + Augmented GT).

This script uses a combination of:
1. Real TransUNet predictions (captures actual failure modes)
2. On-the-fly augmented GT masks (provides diversity)

Usage:
    # First, generate TransUNet predictions:
    python scripts/generate_transunet_predictions.py \
        --data_root ./dataset/processed \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --checkpoint_dir ./checkpoints/transunet \
        --output_dir ./dataset/transunet_preds

    # Then, train with hybrid data:
    python scripts/finetune_sam_hybrid.py \
        --gt_data_root ./dataset/processed \
        --pred_data_root ./dataset/transunet_preds \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --real_ratio 0.5 \
        --sam_checkpoint ./pretrained/sam_vit_b_01ec64.pth
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sam_refiner import SAMRefiner
from data import get_hybrid_dataloaders


def get_args():
    parser = argparse.ArgumentParser(description='Phase 2: SAM Finetuning with Hybrid Data')

    # Data paths
    parser.add_argument('--gt_data_root', type=str, required=True,
                        help='Root directory containing GT datasets')
    parser.add_argument('--pred_data_root', type=str, required=True,
                        help='Root directory containing TransUNet predictions')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Dataset names (e.g., BUSI BUSBRA)')
    parser.add_argument('--real_ratio', type=float, default=0.5,
                        help='Ratio of real predictions (0.5 = 50%% real, 50%% augmented)')

    # SAM configuration
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM pretrained checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model type')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Data loading workers')

    # Augmentation for GT masks
    parser.add_argument('--augmentor_preset', type=str, default='default',
                        choices=['default', 'mild', 'severe', 'boundary_focus', 'structural'],
                        help='Augmentation preset for GT masks')
    parser.add_argument('--soft_mask_prob', type=float, default=0.8,
                        help='Probability of soft mask for augmented samples')
    parser.add_argument('--change_penalty_weight', type=float, default=0.5,
                        help='Weight for change penalty loss')
    parser.add_argument('--fast_soft_mask', action='store_true',
                        help='Use fast Gaussian blur for soft mask (~10x faster)')

    # Phase 3 compatibility
    parser.add_argument('--transunet_img_size', type=int, default=224,
                        help='TransUNet resolution for resolution path matching')
    parser.add_argument('--mask_prompt_style', type=str, default='direct',
                        choices=['gaussian', 'direct', 'distance'],
                        help='How to encode coarse mask for SAM')
    parser.add_argument('--use_roi_crop', action='store_true',
                        help='Use ROI cropping during training')
    parser.add_argument('--roi_expand_ratio', type=float, default=0.3,
                        help='ROI expansion ratio')

    # Speed optimizations
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training (AMP)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--compile_model', action='store_true',
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='Prefetch factor for dataloader')

    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sam_finetuned_hybrid',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    return parser.parse_args()


def quality_aware_loss(pred_masks, gt_masks, coarse_masks, change_penalty_weight=0.5):
    """
    Quality-aware loss function.

    Components:
    1. BCE loss between prediction and GT
    2. Dice loss for better segmentation
    3. Change penalty to prevent unnecessary changes to good regions
    """
    pred_masks = pred_masks.squeeze(1)
    gt_masks = gt_masks.float()
    coarse_masks = coarse_masks.float()

    # BCE loss
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks)

    # Dice loss
    pred_sigmoid = torch.sigmoid(pred_masks)
    intersection = (pred_sigmoid * gt_masks).sum(dim=(1, 2))
    union = pred_sigmoid.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2))
    dice_loss = 1 - (2 * intersection + 1) / (union + 1)
    dice_loss = dice_loss.mean()

    # Change penalty: penalize changes in regions where coarse was already correct
    coarse_binary = (coarse_masks > 0.5).float()
    correct_regions = (coarse_binary == gt_masks).float()
    changes = torch.abs(pred_sigmoid - coarse_binary)
    change_penalty = (changes * correct_regions).mean()

    # Total loss
    total_loss = bce_loss + dice_loss + change_penalty_weight * change_penalty

    return total_loss, {
        'bce': bce_loss.item(),
        'dice': dice_loss.item(),
        'change_penalty': change_penalty.item(),
        'total': total_loss.item(),
    }


def compute_dice(pred, gt):
    """Compute Dice score."""
    pred_binary = (pred > 0.5).float()
    gt_binary = (gt > 0.5).float()
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    return (2 * intersection + 1e-6) / (union + 1e-6)


def train_epoch(model, train_loader, optimizer, scaler, device, use_amp, grad_accum_steps):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    total_improvement = 0
    real_count = 0
    aug_count = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        image = batch['image'].to(device, non_blocking=True)
        gt_mask = batch['label'].to(device, non_blocking=True)
        coarse_mask = batch['coarse_mask'].to(device, non_blocking=True)
        sources = batch['source']

        # Count real vs augmented
        for s in sources:
            if s == 'real':
                real_count += 1
            else:
                aug_count += 1

        with autocast(enabled=use_amp):
            result = model(image, coarse_mask, image_already_normalized=True)
            refined_masks = result['masks']

            loss, loss_components = quality_aware_loss(
                refined_masks, gt_mask, coarse_mask, change_penalty_weight=0.5
            )
            loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            pred_sigmoid = torch.sigmoid(refined_masks.squeeze(1))
            for i in range(gt_mask.shape[0]):
                refined_dice = compute_dice(pred_sigmoid[i], gt_mask[i])
                coarse_dice = compute_dice(coarse_mask[i], gt_mask[i])
                total_dice += refined_dice.item()
                total_improvement += (refined_dice - coarse_dice).item()

        total_loss += loss_components['total']

    n_batches = len(train_loader)
    n_samples = len(train_loader.dataset)

    return {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_samples,
        'improvement': total_improvement / n_samples,
        'real_ratio': real_count / (real_count + aug_count) if (real_count + aug_count) > 0 else 0,
    }


@torch.no_grad()
def validate(model, val_loader, device, use_amp):
    """Validate the model."""
    model.eval()
    total_dice = 0
    total_improvement = 0
    total_coarse_dice = 0
    real_count = 0
    aug_count = 0

    for batch in val_loader:
        image = batch['image'].to(device, non_blocking=True)
        gt_mask = batch['label'].to(device, non_blocking=True)
        coarse_mask = batch['coarse_mask'].to(device, non_blocking=True)
        sources = batch['source']

        for s in sources:
            if s == 'real':
                real_count += 1
            else:
                aug_count += 1

        with autocast(enabled=use_amp):
            result = model(image, coarse_mask, image_already_normalized=True)
            refined_masks = result['masks']

        pred_sigmoid = torch.sigmoid(refined_masks.squeeze(1))

        for i in range(gt_mask.shape[0]):
            refined_dice = compute_dice(pred_sigmoid[i], gt_mask[i])
            coarse_dice = compute_dice(coarse_mask[i], gt_mask[i])
            total_dice += refined_dice.item()
            total_coarse_dice += coarse_dice.item()
            total_improvement += (refined_dice - coarse_dice).item()

    n_samples = len(val_loader.dataset)

    return {
        'dice': total_dice / n_samples,
        'coarse_dice': total_coarse_dice / n_samples,
        'improvement': total_improvement / n_samples,
        'real_ratio': real_count / (real_count + aug_count) if (real_count + aug_count) > 0 else 0,
    }


def main():
    args = get_args()

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    dataset_str = '_'.join(args.datasets)
    output_dir = os.path.join(args.output_dir, dataset_str)
    os.makedirs(output_dir, exist_ok=True)

    # Create dataloaders
    print("\nCreating hybrid dataloaders...")
    train_loader, val_loader = get_hybrid_dataloaders(
        gt_data_root=args.gt_data_root,
        pred_data_root=args.pred_data_root,
        dataset_names=args.datasets,
        batch_size=args.batch_size,
        img_size=1024,
        transunet_img_size=args.transunet_img_size,
        real_ratio=args.real_ratio,
        augmentor_preset=args.augmentor_preset,
        use_fast_soft_mask=args.fast_soft_mask,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Real prediction ratio: {args.real_ratio:.0%}")

    # Create model
    print("\nCreating SAM Refiner...")
    model = SAMRefiner(
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=device,
        trainable_components=['mask_decoder'],
        mask_prompt_style=args.mask_prompt_style,
        use_roi_crop=args.use_roi_crop,
        roi_expand_ratio=args.roi_expand_ratio,
    )
    model = model.to(device)

    # Optional: compile model
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_dice = ckpt.get('best_dice', 0)
        if args.use_amp and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Hybrid Training")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device,
            args.use_amp, args.grad_accum_steps
        )

        # Validate
        val_metrics = validate(model, val_loader, device, args.use_amp)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_metrics['loss']:.4f}, dice={train_metrics['dice']:.4f}, "
              f"impr={train_metrics['improvement']:+.4f}, real={train_metrics['real_ratio']:.0%}")
        print(f"  Val:   dice={val_metrics['dice']:.4f}, coarse={val_metrics['coarse_dice']:.4f}, "
              f"impr={val_metrics['improvement']:+.4f}, real={val_metrics['real_ratio']:.0%}")

        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            print(f"  -> New best Dice: {best_dice:.4f}")

            # Save full checkpoint
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_dice': best_dice,
                'args': vars(args),
            }
            if args.use_amp:
                ckpt['scaler'] = scaler.state_dict()
            torch.save(ckpt, os.path.join(output_dir, 'best.pth'))

            # Save SAM weights only (for Phase 3)
            sam_state = {k: v for k, v in model.state_dict().items() if k.startswith('sam.')}
            torch.save(sam_state, os.path.join(output_dir, 'best_sam.pth'))

        # Save periodic checkpoint
        if (epoch + 1) % 50 == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_dice': best_dice,
            }
            if args.use_amp:
                ckpt['scaler'] = scaler.state_dict()
            torch.save(ckpt, os.path.join(output_dir, f'epoch_{epoch+1}.pth'))

    print("\n" + "=" * 60)
    print(f"Training complete! Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
