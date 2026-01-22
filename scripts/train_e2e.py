"""
Phase 3: End-to-end training of TransUNet + SAMRefiner.

This script trains the complete UltraRefiner pipeline with gradients flowing
from the SAM refinement back to TransUNet. Both models are jointly optimized.

Uses K-fold cross-validation within the training set for validation.

Usage:
    python scripts/train_e2e.py \
        --data_root ./dataset/processed \
        --transunet_checkpoint ./checkpoints/transunet/best.pth \
        --sam_checkpoint ./checkpoints/sam_finetuned/best.pth \
        --fold 0
"""
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_ultra_refiner, CONFIGS
from data import (
    get_combined_kfold_dataloaders,
    RandomGenerator,
    SUPPORTED_DATASETS
)
from utils import DiceLoss, BCEDiceLoss, MetricTracker


def get_args():
    parser = argparse.ArgumentParser(description='End-to-end training of UltraRefiner')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing preprocessed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names to use for training (default: all)')

    # K-fold cross-validation arguments
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for K-fold cross-validation (0 to n_splits-1)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for K-fold cross-validation')

    # Model arguments
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        help='TransUNet ViT model variant')
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        help='SAM model variant')
    parser.add_argument('--transunet_checkpoint', type=str, default=None,
                        help='Path to pre-trained TransUNet checkpoint')
    parser.add_argument('--sam_checkpoint', type=str, default=None,
                        help='Path to finetuned SAM checkpoint')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size for TransUNet')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of segmentation classes')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--transunet_lr', type=float, default=1e-4,
                        help='Learning rate for TransUNet')
    parser.add_argument('--sam_lr', type=float, default=1e-5,
                        help='Learning rate for SAM components')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Loss weights
    parser.add_argument('--coarse_loss_weight', type=float, default=0.3,
                        help='Weight for coarse mask loss')
    parser.add_argument('--refined_loss_weight', type=float, default=0.7,
                        help='Weight for refined mask loss')

    # SAM freezing options
    parser.add_argument('--freeze_sam_image_encoder', action='store_true', default=True,
                        help='Freeze SAM image encoder')
    parser.add_argument('--freeze_sam_prompt_encoder', action='store_true', default=False,
                        help='Freeze SAM prompt encoder')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/ultra_refiner',
                        help='Output directory for checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class EndToEndLoss(nn.Module):
    """
    Combined loss for end-to-end training.
    """
    def __init__(self, coarse_weight=0.3, refined_weight=0.7, n_classes=2):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.refined_weight = refined_weight
        self.n_classes = n_classes

        # Loss for coarse mask (TransUNet output)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes=n_classes)

        # Loss for refined mask (SAM output)
        self.bce_dice = BCEDiceLoss()

    def forward(self, outputs, target):
        """
        Args:
            outputs: Dictionary from UltraRefiner forward pass
            target: Ground truth mask (B, H, W)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        coarse_logits = outputs['coarse_logits']
        refined_mask = outputs['refined_mask_logits']

        # Resize target for refined mask if needed
        if refined_mask.shape[-2:] != target.shape[-2:]:
            target_refined = F.interpolate(
                target.unsqueeze(1),
                size=refined_mask.shape[-2:],
                mode='nearest'
            ).squeeze(1)
        else:
            target_refined = target

        # Coarse loss
        loss_ce = self.ce_loss(coarse_logits, target.long())
        loss_dice = self.dice_loss(coarse_logits, target, softmax=True)
        coarse_loss = 0.5 * loss_ce + 0.5 * loss_dice

        # Refined loss
        refined_loss = self.bce_dice(refined_mask.unsqueeze(1), target_refined.unsqueeze(1))

        # Combined loss
        total_loss = self.coarse_weight * coarse_loss + self.refined_weight * refined_loss

        loss_dict = {
            'coarse_ce': loss_ce.item(),
            'coarse_dice': loss_dice.item(),
            'coarse_total': coarse_loss.item(),
            'refined': refined_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, args):
    """Train for one epoch."""
    model.train()
    metric_tracker_coarse = MetricTracker()
    metric_tracker_refined = MetricTracker()
    iter_num = epoch * len(dataloader)

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        label = batch['label'].to(device)

        # Forward pass
        outputs = model(image)

        # Compute loss
        loss, loss_dict = criterion(outputs, label)
        loss = loss / args.gradient_accumulation

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            coarse_pred = outputs['coarse_mask']
            refined_pred = torch.sigmoid(outputs['refined_mask'])

            # Resize refined prediction to original size if needed
            if refined_pred.shape[-2:] != label.shape[-2:]:
                refined_pred = F.interpolate(
                    refined_pred.unsqueeze(1),
                    size=label.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            metric_tracker_coarse.update(coarse_pred, label)
            metric_tracker_refined.update(refined_pred, label, loss_dict['total'])

        # Logging
        for key, value in loss_dict.items():
            writer.add_scalar(f'train/{key}', value, iter_num)

        iter_num += 1

        pbar.set_postfix({
            'loss': f'{loss_dict["total"]:.4f}',
            'dice_c': f'{metric_tracker_coarse.get_average()["dice"]:.4f}',
            'dice_r': f'{metric_tracker_refined.get_average()["dice"]:.4f}'
        })

    return {
        'coarse': metric_tracker_coarse.get_average(),
        'refined': metric_tracker_refined.get_average()
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    metric_tracker_coarse = MetricTracker()
    metric_tracker_refined = MetricTracker()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            outputs = model(image)
            loss, loss_dict = criterion(outputs, label)

            coarse_pred = outputs['coarse_mask']
            refined_pred = torch.sigmoid(outputs['refined_mask'])

            # Resize if needed
            if refined_pred.shape[-2:] != label.shape[-2:]:
                refined_pred = F.interpolate(
                    refined_pred.unsqueeze(1),
                    size=label.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            metric_tracker_coarse.update(coarse_pred, label)
            metric_tracker_refined.update(refined_pred, label, loss_dict['total'])

    return {
        'coarse': metric_tracker_coarse.get_average(),
        'refined': metric_tracker_refined.get_average()
    }


def main():
    args = get_args()

    # Set default datasets
    if args.datasets is None:
        args.datasets = SUPPORTED_DATASETS

    # Auto-generate experiment name
    if args.exp_name is None:
        args.exp_name = f'ultra_refiner_e2e_fold{args.fold}'

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    snapshot_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(snapshot_path, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'Arguments: {args}')

    # Create dataloaders with K-fold cross-validation
    train_loader, val_loader = get_combined_kfold_dataloaders(
        data_root=args.data_root,
        dataset_names=args.datasets,
        fold_idx=args.fold,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        for_sam=False,
        seed=args.seed
    )

    logging.info(f'Training on datasets: {args.datasets}')
    logging.info(f'Fold {args.fold}/{args.n_splits}')
    logging.info(f'Train samples: {len(train_loader.dataset)}')
    logging.info(f'Val samples: {len(val_loader.dataset)}')

    # Build UltraRefiner model
    model = build_ultra_refiner(
        vit_name=args.vit_name,
        img_size=args.img_size,
        num_classes=args.num_classes,
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
        transunet_checkpoint=args.transunet_checkpoint,
        n_skip=args.n_skip,
        freeze_sam_image_encoder=args.freeze_sam_image_encoder,
        freeze_sam_prompt_encoder=args.freeze_sam_prompt_encoder,
    ).to(device)

    logging.info('Built UltraRefiner model')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Total parameters: {total_params:,}')
    logging.info(f'Trainable parameters: {trainable_params:,}')

    # Loss function
    criterion = EndToEndLoss(
        coarse_weight=args.coarse_loss_weight,
        refined_weight=args.refined_loss_weight,
        n_classes=args.num_classes
    )

    # Optimizer with different learning rates
    transunet_params = list(model.get_transunet_params())
    sam_params = list(model.get_sam_params())

    optimizer = optim.AdamW([
        {'params': transunet_params, 'lr': args.transunet_lr},
        {'params': sam_params, 'lr': args.sam_lr}
    ], weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=1e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logging.info(f'Resumed from epoch {start_epoch}')

    # Tensorboard writer
    writer = SummaryWriter(os.path.join(snapshot_path, 'logs'))

    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, writer, args
        )
        logging.info(f'Epoch {epoch} Train Coarse: {train_metrics["coarse"]}')
        logging.info(f'Epoch {epoch} Train Refined: {train_metrics["refined"]}')

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logging.info(f'Epoch {epoch} Val Coarse: {val_metrics["coarse"]}')
        logging.info(f'Epoch {epoch} Val Refined: {val_metrics["refined"]}')

        # Update scheduler
        scheduler.step()

        # Log to tensorboard
        for key, value in val_metrics['coarse'].items():
            writer.add_scalar(f'val/coarse_{key}', value, epoch)
        for key, value in val_metrics['refined'].items():
            writer.add_scalar(f'val/refined_{key}', value, epoch)

        writer.add_scalar('train/lr_transunet', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('train/lr_sam', scheduler.get_last_lr()[1], epoch)

        # Save checkpoint (based on refined dice)
        refined_dice = val_metrics['refined']['dice']
        is_best = refined_dice > best_dice
        best_dice = max(refined_dice, best_dice)

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_dice': best_dice,
            'config': {
                'vit_name': args.vit_name,
                'sam_model_type': args.sam_model_type,
                'img_size': args.img_size,
                'num_classes': args.num_classes,
                'fold': args.fold,
                'n_splits': args.n_splits,
                'datasets': args.datasets,
            }
        }

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))
            logging.info(f'New best model saved with Dice: {best_dice:.4f}')

        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, os.path.join(snapshot_path, f'epoch_{epoch}.pth'))

    writer.close()
    logging.info(f'Training finished. Best Dice: {best_dice:.4f}')


if __name__ == '__main__':
    main()
