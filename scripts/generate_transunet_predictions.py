#!/usr/bin/env python3
"""
Generate TransUNet predictions on training data for Phase 2 training.

IMPORTANT: Uses out-of-fold prediction strategy:
- For each fold (0-4), load that fold's trained model
- Predict ONLY on that fold's validation set (data the model hasn't seen)
- Combine all 5 folds' validation predictions to cover entire training set

This ensures predictions represent real TransUNet failure modes, not
artificially good predictions on training data.

Usage:
    python scripts/generate_transunet_predictions.py \
        --data_root ./dataset/processed \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --checkpoint_dir ./checkpoints/transunet \
        --output_dir ./dataset/transunet_preds \
        --img_size 224 \
        --n_folds 5
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VisionTransformer, CONFIGS


def get_args():
    parser = argparse.ArgumentParser(description='Generate TransUNet predictions (out-of-fold)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Dataset names (e.g., BUSI BUSBRA)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing TransUNet checkpoints (one per dataset/fold)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--img_size', type=int, default=224,
                        help='TransUNet input size')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds used during training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training seed for correct fold splits)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


def load_transunet(checkpoint_path, img_size, device):
    """Load TransUNet model from checkpoint."""
    config = CONFIGS['R50-ViT-B_16']
    config.n_classes = 2
    config.n_skip = 3
    if hasattr(config.patches, 'grid'):
        config.patches.grid = (img_size // 16, img_size // 16)

    model = VisionTransformer(config=config, img_size=(img_size, img_size), num_classes=2)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()

    return model


def get_all_samples(data_root, dataset_name):
    """Get all training samples for a dataset (sorted for reproducibility)."""
    train_dir = os.path.join(data_root, dataset_name, 'train')
    image_dir = os.path.join(train_dir, 'images')
    mask_dir = os.path.join(train_dir, 'masks')

    samples = []
    for f in sorted(os.listdir(image_dir)):
        if f.endswith(('.png', '.jpg', '.jpeg', '.npy')):
            name = os.path.splitext(f)[0]
            image_path = os.path.join(image_dir, f)
            mask_path = os.path.join(mask_dir, f"{name}.png")
            if os.path.exists(mask_path):
                samples.append({
                    'name': name,
                    'image_path': image_path,
                    'mask_path': mask_path,
                })
    return samples


def create_fold_indices(n_samples, n_folds, seed):
    """
    Create fold indices matching KFoldCrossValidator logic.

    Returns list of (train_indices, val_indices) for each fold.
    """
    random.seed(seed)
    indices = list(range(n_samples))
    random.shuffle(indices)

    fold_size = n_samples // n_folds
    folds = []

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples

        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]

        folds.append((train_indices, val_indices))

    return folds


def load_and_preprocess(image_path, img_size):
    """Load and preprocess image for TransUNet."""
    if image_path.endswith('.npy'):
        image = np.load(image_path)
    else:
        image = np.array(Image.open(image_path).convert('RGB'))

    # Convert to tensor
    image = torch.from_numpy(image).float()

    # Handle different formats
    if image.ndim == 2:
        image = image.unsqueeze(0)  # (H, W) -> (1, H, W)
    elif image.shape[-1] == 3:
        image = image.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        # Convert RGB to grayscale for TransUNet
        image = image.mean(dim=0, keepdim=True)  # (1, H, W)
    elif image.shape[-1] == 1:
        image = image.squeeze(-1).unsqueeze(0)

    # Normalize to [0, 1]
    if image.max() > 1:
        image = image / 255.0

    # Resize
    image = F.interpolate(image.unsqueeze(0), size=(img_size, img_size), mode='bilinear', align_corners=False)

    return image  # (1, 1, H, W)


def find_checkpoint(checkpoint_dir, dataset_name, fold):
    """Find checkpoint path, trying both original and lowercase names."""
    name_options = [dataset_name, dataset_name.lower()]

    for name in name_options:
        ckpt_path = os.path.join(checkpoint_dir, name, f'fold_{fold}', 'best.pth')
        if os.path.exists(ckpt_path):
            return ckpt_path

    return None


def main():
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nOut-of-Fold Prediction Strategy:")
    print(f"  - For each fold, predict on that fold's VALIDATION set only")
    print(f"  - This ensures predictions are on unseen data (real failure modes)")
    print(f"  - {args.n_folds} folds × validation sets = entire training set")

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Get all samples
        samples = get_all_samples(args.data_root, dataset_name)
        n_samples = len(samples)
        print(f"Total samples: {n_samples}")

        # Create fold indices (must match training splits)
        fold_indices = create_fold_indices(n_samples, args.n_folds, args.seed)

        # Create output directories
        output_dir = os.path.join(args.output_dir, dataset_name, 'train')
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'coarse_masks'), exist_ok=True)

        # Track predictions and stats
        all_dice_scores = []
        samples_processed = set()

        # Process each fold
        for fold in range(args.n_folds):
            # Find checkpoint for this fold
            ckpt_path = find_checkpoint(args.checkpoint_dir, dataset_name, fold)
            if ckpt_path is None:
                print(f"  Fold {fold}: Checkpoint not found, skipping...")
                continue

            print(f"\n  Fold {fold}:")
            print(f"    Checkpoint: {ckpt_path}")

            # Load model for this fold
            model = load_transunet(ckpt_path, args.img_size, device)

            # Get validation indices for this fold
            _, val_indices = fold_indices[fold]
            print(f"    Validation samples: {len(val_indices)}")

            fold_dice_scores = []

            with torch.no_grad():
                for idx in tqdm(val_indices, desc=f"    Predicting fold {fold}"):
                    sample = samples[idx]

                    # Skip if already processed (shouldn't happen with proper folds)
                    if sample['name'] in samples_processed:
                        print(f"    Warning: {sample['name']} already processed!")
                        continue
                    samples_processed.add(sample['name'])

                    # Load and preprocess image
                    image = load_and_preprocess(sample['image_path'], args.img_size)
                    image = image.to(device)

                    # Get prediction
                    output = model(image)
                    pred = torch.softmax(output, dim=1)[:, 1]  # (1, H, W)
                    pred_np = pred[0].cpu().numpy()  # (H, W), values in [0, 1]

                    # Load GT for Dice computation
                    gt_mask = np.array(Image.open(sample['mask_path']).convert('L'))
                    gt_mask = (gt_mask > 127).astype(np.float32)
                    gt_resized = F.interpolate(
                        torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0),
                        size=(args.img_size, args.img_size),
                        mode='nearest'
                    )[0, 0].numpy()

                    # Compute Dice
                    pred_binary = (pred_np > 0.5).astype(np.float32)
                    intersection = (pred_binary * gt_resized).sum()
                    union = pred_binary.sum() + gt_resized.sum()
                    dice = (2 * intersection + 1e-6) / (union + 1e-6)
                    fold_dice_scores.append(dice)
                    all_dice_scores.append(dice)

                    # Save coarse mask (soft probability as .npy)
                    coarse_path = os.path.join(output_dir, 'coarse_masks', f"{sample['name']}.npy")
                    np.save(coarse_path, pred_np.astype(np.float32))

                    # Create symlinks to original image and mask
                    src_image = sample['image_path']
                    dst_image = os.path.join(output_dir, 'images', os.path.basename(src_image))
                    if not os.path.exists(dst_image):
                        os.symlink(os.path.abspath(src_image), dst_image)

                    src_mask = sample['mask_path']
                    dst_mask = os.path.join(output_dir, 'masks', f"{sample['name']}.png")
                    if not os.path.exists(dst_mask):
                        os.symlink(os.path.abspath(src_mask), dst_mask)

            # Print fold statistics
            fold_dice = np.array(fold_dice_scores)
            print(f"    Fold {fold} Dice: mean={fold_dice.mean():.4f}, std={fold_dice.std():.4f}")

            # Clean up model to save memory
            del model
            torch.cuda.empty_cache()

        # Print overall statistics
        if all_dice_scores:
            dice_scores = np.array(all_dice_scores)
            print(f"\nOverall Dice Score Statistics for {dataset_name}:")
            print(f"  Samples processed: {len(samples_processed)} / {n_samples}")
            print(f"  Mean: {dice_scores.mean():.4f}")
            print(f"  Std:  {dice_scores.std():.4f}")
            print(f"  Min:  {dice_scores.min():.4f}")
            print(f"  Max:  {dice_scores.max():.4f}")
            print(f"  < 0.9: {(dice_scores < 0.9).sum()} ({100*(dice_scores < 0.9).mean():.1f}%)")
            print(f"  < 0.8: {(dice_scores < 0.8).sum()} ({100*(dice_scores < 0.8).mean():.1f}%)")
            print(f"  < 0.7: {(dice_scores < 0.7).sum()} ({100*(dice_scores < 0.7).mean():.1f}%)")
            print(f"\nSaved to: {output_dir}")
        else:
            print(f"\nNo predictions generated for {dataset_name}")

    print(f"\n{'='*60}")
    print("Done! Out-of-fold TransUNet predictions saved.")
    print("Each sample was predicted by a model that did NOT see it during training.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
