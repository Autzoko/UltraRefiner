#!/usr/bin/env python3
"""
Generate augmented masks offline for Phase 2 SAM training.

This script pre-generates augmented masks using the 12 error types,
saving them to disk for fast loading during training.

Usage:
    python scripts/generate_augmented_masks.py \
        --data_root ./dataset/processed \
        --datasets BUSI BUSBRA BUS BUS_UC BUS_UCLM \
        --output_dir ./dataset/augmented_masks \
        --num_augmentations 5 \
        --augmentor_preset default
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mask_augmentation import create_augmentor


def get_args():
    parser = argparse.ArgumentParser(description='Generate augmented masks offline')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing processed datasets')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                        help='Dataset names (e.g., BUSI BUSBRA)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for augmented masks')
    parser.add_argument('--num_augmentations', type=int, default=5,
                        help='Number of augmented versions per sample')
    parser.add_argument('--augmentor_preset', type=str, default='default',
                        choices=['default', 'mild', 'severe', 'boundary_focus', 'structural'],
                        help='Augmentation preset')
    parser.add_argument('--soft_mask_prob', type=float, default=0.8,
                        help='Probability of soft mask conversion')
    parser.add_argument('--use_fast_soft_mask', action='store_true',
                        help='Use fast Gaussian blur for soft mask')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def get_all_samples(data_root, dataset_name):
    """Get all training samples for a dataset."""
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


def augment_single_sample(args):
    """Augment a single sample (for parallel processing)."""
    sample, aug_idx, augmentor_config, output_dir = args

    # Create augmentor in worker process
    augmentor = create_augmentor(
        preset=augmentor_config['preset'],
        soft_mask_prob=augmentor_config['soft_mask_prob'],
        use_fast_soft_mask=augmentor_config['use_fast_soft_mask'],
    )

    # Load GT mask
    gt_mask = np.array(Image.open(sample['mask_path']).convert('L'))
    if gt_mask.max() > 1:
        gt_mask = gt_mask.astype(np.float32) / 255.0
    else:
        gt_mask = gt_mask.astype(np.float32)

    # Skip empty masks
    if gt_mask.sum() < 10:
        return None

    # Apply augmentation
    coarse_mask, aug_info = augmentor(gt_mask)

    # Create output filename
    output_name = f"{sample['name']}_aug{aug_idx}"
    output_path = os.path.join(output_dir, 'coarse_masks', f"{output_name}.npy")

    # Save augmented mask
    np.save(output_path, coarse_mask.astype(np.float32))

    # Return info for metadata
    return {
        'name': output_name,
        'original_name': sample['name'],
        'aug_idx': aug_idx,
        'error_type': aug_info.get('error_type', 'unknown'),
        'dice': float(aug_info.get('dice', 1.0)),
        'soft': aug_info.get('soft', False),
    }


def main():
    args = get_args()
    np.random.seed(args.seed)

    print(f"Generating offline augmented masks")
    print(f"  Preset: {args.augmentor_preset}")
    print(f"  Augmentations per sample: {args.num_augmentations}")
    print(f"  Soft mask probability: {args.soft_mask_prob}")
    print(f"  Workers: {args.num_workers}")

    augmentor_config = {
        'preset': args.augmentor_preset,
        'soft_mask_prob': args.soft_mask_prob,
        'use_fast_soft_mask': args.use_fast_soft_mask,
    }

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        # Get all samples
        samples = get_all_samples(args.data_root, dataset_name)
        print(f"Found {len(samples)} samples")

        # Create output directories
        output_dir = os.path.join(args.output_dir, dataset_name, 'train')
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'coarse_masks'), exist_ok=True)

        # Create symlinks to original images and masks
        print("Creating symlinks to original data...")
        for sample in tqdm(samples, desc="Symlinks"):
            # For each augmentation, we'll reference the same original image/mask
            for aug_idx in range(args.num_augmentations):
                output_name = f"{sample['name']}_aug{aug_idx}"

                # Symlink image
                src_image = os.path.abspath(sample['image_path'])
                dst_image = os.path.join(output_dir, 'images', f"{output_name}.png")
                if not os.path.exists(dst_image):
                    try:
                        os.symlink(src_image, dst_image)
                    except FileExistsError:
                        pass

                # Symlink GT mask
                src_mask = os.path.abspath(sample['mask_path'])
                dst_mask = os.path.join(output_dir, 'masks', f"{output_name}.png")
                if not os.path.exists(dst_mask):
                    try:
                        os.symlink(src_mask, dst_mask)
                    except FileExistsError:
                        pass

        # Prepare tasks for parallel processing
        tasks = []
        for sample in samples:
            for aug_idx in range(args.num_augmentations):
                tasks.append((sample, aug_idx, augmentor_config, output_dir))

        print(f"Generating {len(tasks)} augmented masks...")

        # Process in parallel
        metadata = []
        error_type_counts = {}
        dice_scores = []

        if args.num_workers > 1:
            # Use multiprocessing
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(augment_single_sample, task): task for task in tasks}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting"):
                    result = future.result()
                    if result is not None:
                        metadata.append(result)
                        error_type = result['error_type']
                        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                        dice_scores.append(result['dice'])
        else:
            # Single process
            for task in tqdm(tasks, desc="Augmenting"):
                result = augment_single_sample(task)
                if result is not None:
                    metadata.append(result)
                    error_type = result['error_type']
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                    dice_scores.append(result['dice'])

        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'num_samples': len(metadata),
                'num_augmentations': args.num_augmentations,
                'augmentor_preset': args.augmentor_preset,
                'samples': metadata,
            }, f, indent=2)

        # Print statistics
        dice_scores = np.array(dice_scores)
        print(f"\nGenerated {len(metadata)} augmented masks")
        print(f"\nError Type Distribution:")
        for error_type, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count} ({100*count/len(metadata):.1f}%)")

        print(f"\nDice Score Statistics:")
        print(f"  Mean: {dice_scores.mean():.4f}")
        print(f"  Std:  {dice_scores.std():.4f}")
        print(f"  Min:  {dice_scores.min():.4f}")
        print(f"  Max:  {dice_scores.max():.4f}")
        print(f"  < 0.9: {(dice_scores < 0.9).sum()} ({100*(dice_scores < 0.9).mean():.1f}%)")
        print(f"  < 0.8: {(dice_scores < 0.8).sum()} ({100*(dice_scores < 0.8).mean():.1f}%)")

        print(f"\nSaved to: {output_dir}")

    print(f"\n{'='*60}")
    print("Done! Offline augmented masks generated.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
