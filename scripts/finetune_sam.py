"""
Phase 2: Finetune SAM (from MedSAM checkpoint) on breast ultrasound datasets.

This script finetunes SAM's prompt encoder and mask decoder while keeping
the image encoder frozen. The training uses ground truth masks to generate
prompts, simulating the coarse masks that will come from TransUNet.

Uses K-fold cross-validation within the training set for validation.

Usage:
    python scripts/finetune_sam.py \
        --data_root ./dataset/processed \
        --medsam_checkpoint ./pretrained/medsam_vit_b.pth \
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

from models.sam import sam_model_registry, build_sam_for_training, build_sam_with_lora
from models.lora import get_lora_params, get_lora_state_dict, print_lora_info
from models.sam_refiner import DifferentiableSAMRefiner
from data import (
    get_combined_kfold_dataloaders,
    SAMRandomGenerator,
    SUPPORTED_DATASETS
)
from utils import BCEDiceLoss, SAMLoss, MetricTracker, TrainingLogger


def get_args():
    parser = argparse.ArgumentParser(description='Finetune SAM on breast ultrasound data')

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
    parser.add_argument('--sam_model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='SAM model variant')
    parser.add_argument('--medsam_checkpoint', type=str, required=True,
                        help='Path to MedSAM checkpoint')
    parser.add_argument('--freeze_image_encoder', action='store_true', default=True,
                        help='Freeze SAM image encoder')
    parser.add_argument('--freeze_prompt_encoder', action='store_true', default=False,
                        help='Freeze SAM prompt encoder')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--base_lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Prompt simulation arguments
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise level for simulating coarse masks')
    parser.add_argument('--use_point_prompt', action='store_true', default=True,
                        help='Use point prompts')
    parser.add_argument('--use_box_prompt', action='store_true', default=True,
                        help='Use box prompts')
    parser.add_argument('--use_mask_prompt', action='store_true', default=True,
                        help='Use mask prompts')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sam_finetuned',
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


def simulate_coarse_mask(gt_mask, noise_level=0.1):
    """
    Simulate coarse mask from ground truth by adding noise and morphological operations.
    This simulates the output quality of TransUNet.

    Args:
        gt_mask: Ground truth mask (B, H, W)
        noise_level: Amount of noise to add

    Returns:
        Coarse mask (B, H, W)
    """
    coarse_mask = gt_mask.clone()

    # Add Gaussian noise
    noise = torch.randn_like(coarse_mask) * noise_level
    coarse_mask = coarse_mask + noise
    coarse_mask = torch.clamp(coarse_mask, 0, 1)

    # Random erosion/dilation simulation
    if random.random() > 0.5:
        # Simulated erosion (shrink mask slightly)
        kernel_size = random.choice([3, 5])
        coarse_mask = F.max_pool2d(
            1 - coarse_mask.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze(1)
        coarse_mask = 1 - coarse_mask
    else:
        # Simulated dilation (expand mask slightly)
        kernel_size = random.choice([3, 5])
        coarse_mask = F.max_pool2d(
            coarse_mask.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze(1)

    return coarse_mask


def train_one_epoch(model, sam_refiner, dataloader, optimizer, criterion, device, epoch, writer, args):
    """Train for one epoch."""
    model.train()
    sam_refiner.train()

    metric_tracker = MetricTracker()
    iter_num = epoch * len(dataloader)

    # SAM normalization
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)

    pbar = tqdm(dataloader, desc=f'  Train', leave=False)
    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)  # (B, 3, 1024, 1024)
        label = batch['label'].to(device)  # (B, 1024, 1024)

        # Normalize image for SAM
        image_normalized = (image - pixel_mean) / pixel_std

        # Simulate coarse mask from ground truth
        coarse_mask = simulate_coarse_mask(label, noise_level=args.noise_level)

        # Forward through SAM refiner
        output = sam_refiner(image_normalized, coarse_mask)

        # Get predictions
        pred_masks = output['masks_all']  # (B, 3, H, W)
        pred_ious = output['iou_predictions']  # (B, 3)

        # Compute loss
        loss, loss_dict = criterion(pred_masks, pred_ious, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Select best mask for metrics
        best_idx = pred_ious.argmax(dim=1)
        pred_best = torch.stack([
            pred_masks[b, best_idx[b]] for b in range(pred_masks.size(0))
        ])
        pred_best = torch.sigmoid(pred_best)

        metric_tracker.update(pred_best, label, loss.item())

        # Logging
        writer.add_scalar('train/loss', loss.item(), iter_num)
        for key, value in loss_dict.items():
            writer.add_scalar(f'train/{key}', value, iter_num)

        iter_num += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metric_tracker.get_average()["dice"]:.4f}'
        })

    return metric_tracker.get_average()


def validate(model, sam_refiner, dataloader, criterion, device, args):
    """Validate the model."""
    model.eval()
    sam_refiner.eval()

    metric_tracker = MetricTracker(
        metrics=['dice', 'iou', 'jaccard', 'precision', 'recall', 'accuracy']
    )

    # SAM normalization
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='  Valid', leave=False):
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            # Normalize image for SAM
            image_normalized = (image - pixel_mean) / pixel_std

            # Simulate coarse mask (without random augmentation for validation)
            coarse_mask = simulate_coarse_mask(label, noise_level=args.noise_level * 0.5)

            # Forward
            output = sam_refiner(image_normalized, coarse_mask)

            pred_masks = output['masks_all']
            pred_ious = output['iou_predictions']

            loss, _ = criterion(pred_masks, pred_ious, label)

            # Best mask
            best_idx = pred_ious.argmax(dim=1)
            pred_best = torch.stack([
                pred_masks[b, best_idx[b]] for b in range(pred_masks.size(0))
            ])
            pred_best = torch.sigmoid(pred_best)

            metric_tracker.update(pred_best, label, loss.item())

    return metric_tracker.get_average()


def main():
    args = get_args()

    # Set default datasets
    if args.datasets is None:
        args.datasets = SUPPORTED_DATASETS

    # Auto-generate experiment name
    if args.exp_name is None:
        # Structure: checkpoints/sam_finetuned/fold_{i}/
        args.exp_name = f'fold_{args.fold}'

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
        num_workers=args.num_workers,
        for_sam=True,
        seed=args.seed
    )

    logging.info(f'Training on datasets: {args.datasets}')
    logging.info(f'Fold {args.fold}/{args.n_splits}')
    logging.info(f'Train samples: {len(train_loader.dataset)}')
    logging.info(f'Val samples: {len(val_loader.dataset)}')

    # Initialize beautiful logger
    logger = TrainingLogger(
        experiment_name=args.exp_name,
        total_epochs=args.max_epochs
    )
    logger.print_header(
        dataset='Combined (All)',
        fold=args.fold,
        n_splits=args.n_splits,
        train_samples=len(train_loader.dataset),
        val_samples=len(val_loader.dataset)
    )

    # Build SAM model
    sam = build_sam_for_training(
        model_type=args.sam_model_type,
        checkpoint=args.medsam_checkpoint,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder
    ).to(device)

    logging.info(f'Loaded MedSAM checkpoint from {args.medsam_checkpoint}')

    # Build SAM Refiner
    sam_refiner = DifferentiableSAMRefiner(
        sam_model=sam,
        use_point_prompt=args.use_point_prompt,
        use_box_prompt=args.use_box_prompt,
        use_mask_prompt=args.use_mask_prompt,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder
    ).to(device)

    # Loss function
    criterion = SAMLoss(mask_weight=1.0, iou_weight=1.0)

    # Optimizer - only for trainable parameters
    # Collect trainable parameters (avoid duplicates using id())
    trainable_params = []
    seen_params = set()

    for p in sam.parameters():
        if p.requires_grad and id(p) not in seen_params:
            trainable_params.append(p)
            seen_params.add(id(p))

    for p in sam_refiner.parameters():
        if p.requires_grad and id(p) not in seen_params:
            trainable_params.append(p)
            seen_params.add(id(p))

    logging.info(f'Trainable parameters: {len(trainable_params)}')
    logging.info(f'Total trainable params: {sum(p.numel() for p in trainable_params):,}')

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

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
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        sam.load_state_dict(checkpoint['sam_model'])
        sam_refiner.load_state_dict(checkpoint['sam_refiner'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logging.info(f'Resumed from epoch {start_epoch}')

    # Tensorboard writer
    writer = SummaryWriter(os.path.join(snapshot_path, 'logs'))

    # Training loop
    best_epoch = 0
    for epoch in range(start_epoch, args.max_epochs):
        # Train
        train_metrics = train_one_epoch(
            sam, sam_refiner, train_loader, optimizer, criterion,
            device, epoch, writer, args
        )
        logging.info(f'Epoch {epoch} Train: {train_metrics}')

        # Validate
        val_metrics = validate(sam, sam_refiner, val_loader, criterion, device, args)
        logging.info(f'Epoch {epoch} Val: {val_metrics}')

        # Update scheduler
        scheduler.step()

        # Log to tensorboard
        for key, value in val_metrics.items():
            writer.add_scalar(f'val/{key}', value, epoch)
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('train/lr', current_lr, epoch)

        # Save checkpoint
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            best_epoch = epoch

        # Print beautiful epoch summary
        logger.print_epoch_summary(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            lr=current_lr,
            is_best=is_best
        )

        checkpoint = {
            'epoch': epoch,
            'sam_model': sam.state_dict(),
            'sam_refiner': sam_refiner.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_dice': best_dice,
            'config': {
                'sam_model_type': args.sam_model_type,
                'fold': args.fold,
                'n_splits': args.n_splits,
                'datasets': args.datasets,
            },
        }

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))
            logging.info(f'New best model saved with Dice: {best_dice:.4f}')

        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, os.path.join(snapshot_path, f'epoch_{epoch}.pth'))

    writer.close()
    logger.print_training_complete(best_dice, best_epoch)
    logging.info(f'Training finished. Best Dice: {best_dice:.4f}')


if __name__ == '__main__':
    main()
