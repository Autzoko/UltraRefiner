#!/usr/bin/env python3
"""
Generate TransUNet predictions on training data for Phase 2 training.

This script runs TransUNet inference on all training images and saves
the soft probability predictions. These can then be used as "real"
coarse masks for Phase 2 SAM finetuning.

Usage:
    python scripts/generate_transunet_predictions.py \
        --data_root ./dataset/processed \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --checkpoint_dir ./checkpoints/transunet \
        --output_dir ./dataset/transunet_preds \
        --img_size 224
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import VisionTransformer, CONFIGS


def get_args():
    parser = argparse.ArgumentParser(description='Generate TransUNet predictions')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Dataset names (e.g., BUSI BUSBRA)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing TransUNet checkpoints (one per dataset)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for predictions')
    parser.add_argument('--img_size', type=int, default=224,
                        help='TransUNet input size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--use_all_folds', action='store_true',
                        help='Generate predictions using all fold checkpoints (ensemble)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold to use if not using all folds')
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


def get_samples(data_root, dataset_name):
    """Get all training samples for a dataset."""
    train_dir = os.path.join(data_root, dataset_name, 'train')
    image_dir = os.path.join(train_dir, 'images')
    mask_dir = os.path.join(train_dir, 'masks')

    samples = []
    for f in os.listdir(image_dir):
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


def main():
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Find checkpoint
        if args.use_all_folds:
            # Use ensemble of all folds
            checkpoint_paths = []
            for fold in range(5):
                ckpt_path = os.path.join(args.checkpoint_dir, dataset_name, f'fold_{fold}', 'best.pth')
                if os.path.exists(ckpt_path):
                    checkpoint_paths.append(ckpt_path)
            if not checkpoint_paths:
                print(f"No checkpoints found for {dataset_name}, skipping...")
                continue
            print(f"Using ensemble of {len(checkpoint_paths)} folds")
        else:
            ckpt_path = os.path.join(args.checkpoint_dir, dataset_name, f'fold_{args.fold}', 'best.pth')
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}, skipping...")
                continue
            checkpoint_paths = [ckpt_path]

        # Load models
        models = []
        for ckpt_path in checkpoint_paths:
            model = load_transunet(ckpt_path, args.img_size, device)
            models.append(model)
            print(f"Loaded: {ckpt_path}")

        # Get samples
        samples = get_samples(args.data_root, dataset_name)
        print(f"Found {len(samples)} samples")

        # Create output directories
        output_dir = os.path.join(args.output_dir, dataset_name, 'train')
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'coarse_masks'), exist_ok=True)

        # Generate predictions
        dice_scores = []
        with torch.no_grad():
            for sample in tqdm(samples, desc=f"Generating predictions"):
                # Load image
                image = load_and_preprocess(sample['image_path'], args.img_size)
                image = image.to(device)

                # Ensemble prediction
                pred_sum = None
                for model in models:
                    output = model(image)
                    pred = torch.softmax(output, dim=1)[:, 1]  # (1, H, W)
                    if pred_sum is None:
                        pred_sum = pred
                    else:
                        pred_sum = pred_sum + pred
                pred = pred_sum / len(models)  # Average

                # Convert to numpy
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
                dice_scores.append(dice)

                # Save coarse mask (soft probability as .npy)
                coarse_path = os.path.join(output_dir, 'coarse_masks', f"{sample['name']}.npy")
                np.save(coarse_path, pred_np.astype(np.float32))

                # Copy original image (as symlink or copy)
                src_image = sample['image_path']
                dst_image = os.path.join(output_dir, 'images', os.path.basename(src_image))
                if not os.path.exists(dst_image):
                    if src_image.endswith('.npy'):
                        os.symlink(os.path.abspath(src_image), dst_image)
                    else:
                        os.symlink(os.path.abspath(src_image), dst_image)

                # Copy GT mask (as symlink)
                src_mask = sample['mask_path']
                dst_mask = os.path.join(output_dir, 'masks', f"{sample['name']}.png")
                if not os.path.exists(dst_mask):
                    os.symlink(os.path.abspath(src_mask), dst_mask)

        # Print statistics
        dice_scores = np.array(dice_scores)
        print(f"\nDice Score Statistics:")
        print(f"  Mean: {dice_scores.mean():.4f}")
        print(f"  Std:  {dice_scores.std():.4f}")
        print(f"  Min:  {dice_scores.min():.4f}")
        print(f"  Max:  {dice_scores.max():.4f}")
        print(f"  < 0.8: {(dice_scores < 0.8).sum()} ({100*(dice_scores < 0.8).mean():.1f}%)")
        print(f"  < 0.7: {(dice_scores < 0.7).sum()} ({100*(dice_scores < 0.7).mean():.1f}%)")
        print(f"\nSaved to: {output_dir}")

    print(f"\n{'='*60}")
    print("Done! TransUNet predictions saved.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
