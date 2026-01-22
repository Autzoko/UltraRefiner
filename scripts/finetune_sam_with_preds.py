"""
Phase 2 (Alternative): Finetune SAM using actual TransUNet predictions.

This script finetunes SAM using pre-computed TransUNet predictions as coarse masks,
instead of simulating coarse masks from ground truth. This provides more realistic
training conditions for SAM.

Workflow:
1. First run: python scripts/inference_transunet.py --dataset BUSI --visualize
2. Then run this script with the predictions

Usage:
    python scripts/finetune_sam_with_preds.py \
        --data_root ./dataset/processed \
        --pred_root ./predictions/transunet \
        --medsam_checkpoint ./pretrained/medsam_vit_b.pth \
        --datasets BUSI BUSBRA \
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sam import sam_model_registry, build_sam_for_training
from models.sam_refiner import DifferentiableSAMRefiner
from utils import BCEDiceLoss, SAMLoss, MetricTracker, TrainingLogger
from data import SUPPORTED_DATASETS


def get_args():
    parser = argparse.ArgumentParser(description='Finetune SAM with TransUNet predictions')

    # Data arguments
    parser.add_argument('--data_root', type=str, default='./dataset/processed',
                        help='Root directory containing preprocessed datasets')
    parser.add_argument('--pred_root', type=str, default='./predictions/transunet',
                        help='Root directory containing TransUNet predictions')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Dataset names to use (default: all)')

    # K-fold arguments
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index (must match TransUNet predictions)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds')

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

    # Prompt arguments
    parser.add_argument('--use_point_prompt', action='store_true', default=True,
                        help='Use point prompts')
    parser.add_argument('--use_box_prompt', action='store_true', default=True,
                        help='Use box prompts')
    parser.add_argument('--use_mask_prompt', action='store_true', default=True,
                        help='Use mask prompts')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sam_finetuned_with_preds',
                        help='Output directory for checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')

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


class SAMDatasetWithPredictions(Dataset):
    """
    Dataset that loads images, ground truth, and TransUNet predictions.
    """
    def __init__(
        self,
        data_root: str,
        pred_root: str,
        dataset_name: str,
        fold: int,
        split: str = 'train',
        img_size: int = 1024,
    ):
        """
        Args:
            data_root: Root directory of processed data
            pred_root: Root directory of TransUNet predictions
            dataset_name: Name of the dataset
            fold: Fold index
            split: 'train' or 'val' (uses val predictions for training SAM)
            img_size: Target image size for SAM (1024)
        """
        self.data_root = data_root
        self.pred_root = pred_root
        self.dataset_name = dataset_name
        self.fold = fold
        self.img_size = img_size

        # Load split file
        split_file = os.path.join(data_root, dataset_name, f'{split}_fold{fold}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]

        # Predictions directory
        self.pred_dir = os.path.join(pred_root, dataset_name, f'fold{fold}', 'predictions')
        if not os.path.exists(self.pred_dir):
            raise FileNotFoundError(f"Predictions not found: {self.pred_dir}")

        # Verify predictions exist
        self._verify_predictions()

        print(f"Loaded {len(self.samples)} samples for {dataset_name} fold {fold} ({split})")

    def _verify_predictions(self):
        """Verify that predictions exist for all samples."""
        num_preds = len([f for f in os.listdir(self.pred_dir) if f.endswith('_pred.npy')])
        if num_preds < len(self.samples):
            print(f"Warning: Found {num_preds} predictions but {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]

        # Load image
        img_path = os.path.join(self.data_root, self.dataset_name, 'images', f'{sample_name}.png')
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)

        # Load ground truth
        mask_path = os.path.join(self.data_root, self.dataset_name, 'masks', f'{sample_name}.png')
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)  # (H, W)

        # Load TransUNet prediction (coarse mask)
        pred_path = os.path.join(self.pred_dir, f'{idx:04d}_pred.npy')
        if os.path.exists(pred_path):
            coarse_mask = np.load(pred_path)
            # Resize to SAM input size
            coarse_mask = torch.from_numpy(coarse_mask).unsqueeze(0).unsqueeze(0)
            coarse_mask = F.interpolate(coarse_mask, size=(self.img_size, self.img_size),
                                        mode='bilinear', align_corners=False)
            coarse_mask = coarse_mask.squeeze()
        else:
            # Fallback: use ground truth with noise if prediction not found
            print(f"Warning: Prediction not found for {idx}, using noisy GT")
            coarse_mask = mask + torch.randn_like(mask) * 0.1
            coarse_mask = torch.clamp(coarse_mask, 0, 1)

        return {
            'image': image,
            'label': mask,
            'coarse_mask': coarse_mask,
            'case_name': sample_name
        }


def create_dataloaders(args):
    """Create training and validation dataloaders."""
    datasets = args.datasets if args.datasets else SUPPORTED_DATASETS

    train_datasets = []
    val_datasets = []

    for dataset_name in datasets:
        try:
            # For SAM training, we use TransUNet's validation predictions
            # because TransUNet was trained on the training set
            train_ds = SAMDatasetWithPredictions(
                data_root=args.data_root,
                pred_root=args.pred_root,
                dataset_name=dataset_name,
                fold=args.fold,
                split='val',  # Use validation set predictions
                img_size=1024
            )
            train_datasets.append(train_ds)

            # For SAM validation, we can use a subset or different fold
            # Here we just use a portion of the training data
            val_ds = SAMDatasetWithPredictions(
                data_root=args.data_root,
                pred_root=args.pred_root,
                dataset_name=dataset_name,
                fold=args.fold,
                split='val',
                img_size=1024
            )
            val_datasets.append(val_ds)

        except FileNotFoundError as e:
            print(f"Skipping {dataset_name}: {e}")
            continue

    if not train_datasets:
        raise RuntimeError("No valid datasets found!")

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    # Split val_dataset for actual validation (use 20%)
    val_size = len(val_dataset) // 5
    train_size = len(val_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(model, sam_refiner, dataloader, optimizer, criterion, device, epoch, writer):
    """Train for one epoch using actual TransUNet predictions."""
    model.train()
    sam_refiner.train()

    metric_tracker = MetricTracker()
    iter_num = epoch * len(dataloader)

    # SAM normalization
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(device)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(device)

    pbar = tqdm(dataloader, desc='  Train', leave=False)
    for batch_idx, batch in enumerate(pbar):
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        coarse_mask = batch['coarse_mask'].to(device)  # Actual TransUNet prediction

        # Normalize image for SAM
        image_normalized = (image - pixel_mean) / pixel_std

        # Forward through SAM refiner with actual coarse mask
        output = sam_refiner(image_normalized, coarse_mask)

        # Get predictions
        pred_masks = output['masks_all']
        pred_ious = output['iou_predictions']

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


def validate(model, sam_refiner, dataloader, criterion, device):
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
            coarse_mask = batch['coarse_mask'].to(device)

            # Normalize image for SAM
            image_normalized = (image - pixel_mean) / pixel_std

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
        args.exp_name = f'sam_with_preds_fold{args.fold}'

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

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)

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
        dataset='Combined (with TransUNet preds)',
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

    # Optimizer
    trainable_params = [p for p in sam.parameters() if p.requires_grad]
    trainable_params += [p for p in sam_refiner.parameters()
                         if p.requires_grad and p not in trainable_params]

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
    best_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
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
    for epoch in range(start_epoch, args.max_epochs):
        # Train
        train_metrics = train_one_epoch(
            sam, sam_refiner, train_loader, optimizer, criterion,
            device, epoch, writer
        )
        logging.info(f'Epoch {epoch} Train: {train_metrics}')

        # Validate
        val_metrics = validate(sam, sam_refiner, val_loader, criterion, device)
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
