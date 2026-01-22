"""
Phase 1: Train TransUNet independently on breast ultrasound datasets.

Supports two modes:
1. Train on a single dataset with K-fold cross-validation
2. Train on combined datasets with K-fold cross-validation

Usage:
    # Single dataset training
    python scripts/train_transunet.py --data_root ./dataset/processed --dataset BUSI --fold 0

    # Combined dataset training
    python scripts/train_transunet.py --data_root ./dataset/processed --datasets BUSI BUSBRA --fold 0
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
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transunet import VisionTransformer, CONFIGS
from data import (
    get_kfold_dataloaders,
    get_combined_kfold_dataloaders,
    get_test_dataloader,
    get_combined_test_dataloader,
    RandomGenerator,
    SUPPORTED_DATASETS
)
from utils import DiceLoss, MetricTracker


def get_args():
    parser = argparse.ArgumentParser(description='Train TransUNet on breast ultrasound data')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing preprocessed datasets')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name to train on (for per-dataset training)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names to use for combined training')

    # K-fold cross-validation arguments
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for K-fold cross-validation (0 to n_splits-1)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for K-fold cross-validation')

    # Model arguments
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16',
                        choices=['R50-ViT-B_16', 'R50-ViT-L_16', 'ViT-B_16', 'ViT-L_16'],
                        help='ViT model variant')
    parser.add_argument('--vit_pretrained', type=str, default=None,
                        help='Path to pretrained ViT weights (.npz file)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of segmentation classes')
    parser.add_argument('--n_skip', type=int, default=3,
                        help='Number of skip connections')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size per GPU')
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Maximum number of epochs')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/transunet',
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

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, ce_loss, dice_loss, device, epoch, writer, base_lr, max_iterations):
    """Train for one epoch."""
    model.train()
    metric_tracker = MetricTracker()
    iter_num = epoch * len(dataloader)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        label = batch['label'].to(device)

        # Forward pass
        outputs = model(image)

        # Compute loss
        loss_ce = ce_loss(outputs, label.long())
        loss_dice = dice_loss(outputs, label, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate (poly scheduler)
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        # Compute metrics
        with torch.no_grad():
            pred = torch.softmax(outputs, dim=1)[:, 1]
            metric_tracker.update(pred, label, loss.item())

        # Logging
        writer.add_scalar('train/loss', loss.item(), iter_num)
        writer.add_scalar('train/loss_ce', loss_ce.item(), iter_num)
        writer.add_scalar('train/loss_dice', loss_dice.item(), iter_num)
        writer.add_scalar('train/lr', lr_, iter_num)

        iter_num += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metric_tracker.get_average()["dice"]:.4f}'
        })

    return metric_tracker.get_average()


def validate(model, dataloader, ce_loss, dice_loss, device):
    """Validate the model."""
    model.eval()
    metric_tracker = MetricTracker()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            outputs = model(image)

            loss_ce = ce_loss(outputs, label.long())
            loss_dice = dice_loss(outputs, label, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            pred = torch.softmax(outputs, dim=1)[:, 1]
            metric_tracker.update(pred, label, loss.item())
            total_loss += loss.item()

    avg_metrics = metric_tracker.get_average()
    avg_metrics['loss'] = total_loss / len(dataloader)
    return avg_metrics


def main():
    args = get_args()

    # Validate arguments
    if args.dataset is None and args.datasets is None:
        args.datasets = SUPPORTED_DATASETS
        print(f"No dataset specified, using all datasets: {args.datasets}")
    elif args.dataset is not None and args.datasets is not None:
        print("Warning: Both --dataset and --datasets provided, using --dataset (single dataset mode)")
        args.datasets = None

    # Auto-generate experiment name
    if args.exp_name is None:
        if args.dataset:
            args.exp_name = f'{args.dataset}_fold{args.fold}'
        else:
            args.exp_name = f'combined_fold{args.fold}'

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
    if args.dataset:
        # Single dataset training
        train_loader, val_loader = get_kfold_dataloaders(
            data_root=args.data_root,
            dataset_name=args.dataset,
            fold_idx=args.fold,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers,
            for_sam=False,
            seed=args.seed
        )
        logging.info(f'Training on single dataset: {args.dataset}')
    else:
        # Combined dataset training
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
        logging.info(f'Training on combined datasets: {args.datasets}')

    logging.info(f'Fold {args.fold}/{args.n_splits}')
    logging.info(f'Train samples: {len(train_loader.dataset)}')
    logging.info(f'Val samples: {len(val_loader.dataset)}')

    # Build model
    config = CONFIGS[args.vit_name]
    config.n_classes = args.num_classes
    config.n_skip = args.n_skip

    if args.vit_name.startswith('R50'):
        config.patches.grid = (args.img_size // 16, args.img_size // 16)

    model = VisionTransformer(
        config=config,
        img_size=args.img_size,
        num_classes=args.num_classes
    ).to(device)

    # Load pretrained ViT weights
    if args.vit_pretrained:
        weights = np.load(args.vit_pretrained)
        model.load_from(weights)
        logging.info(f'Loaded pretrained ViT weights from {args.vit_pretrained}')

    # Loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=args.num_classes)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logging.info(f'Resumed from epoch {start_epoch}')

    # Tensorboard writer
    writer = SummaryWriter(os.path.join(snapshot_path, 'logs'))

    # Training loop
    max_iterations = args.max_epochs * len(train_loader)
    logging.info(f'Max iterations: {max_iterations}')

    for epoch in range(start_epoch, args.max_epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, ce_loss, dice_loss,
            device, epoch, writer, args.base_lr, max_iterations
        )
        logging.info(f'Epoch {epoch} Train: {train_metrics}')

        # Validate
        val_metrics = validate(model, val_loader, ce_loss, dice_loss, device)
        logging.info(f'Epoch {epoch} Val: {val_metrics}')

        # Log to tensorboard
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)

        # Save checkpoint
        is_best = val_metrics['dice'] > best_dice
        best_dice = max(val_metrics['dice'], best_dice)

        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dice': best_dice,
            'config': {
                'vit_name': args.vit_name,
                'img_size': args.img_size,
                'num_classes': args.num_classes,
                'n_skip': args.n_skip,
                'fold': args.fold,
                'n_splits': args.n_splits,
                'dataset': args.dataset,
                'datasets': args.datasets,
            },
        }

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))
            logging.info(f'New best model saved with Dice: {best_dice:.4f}')

        if (epoch + 1) % 50 == 0:
            torch.save(checkpoint, os.path.join(snapshot_path, f'epoch_{epoch}.pth'))

    writer.close()
    logging.info(f'Training finished. Best Dice: {best_dice:.4f}')


if __name__ == '__main__':
    main()
