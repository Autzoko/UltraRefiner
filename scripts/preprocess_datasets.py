"""
Dataset Preprocessing Script for UltraRefiner

This script preprocesses all breast ultrasound datasets and creates fixed train/test splits.
The same splits are used for all training phases:
- Phase 1: TransUNet training (per dataset)
- Phase 2: SAM finetuning (all datasets combined)
- Phase 3: End-to-end training

IMPORTANT: Samples with blank masks (no lesions / normal cases) are automatically excluded.

Datasets:
- BUSI: Breast Ultrasound Images (benign + malignant, excluding normal)
- BUSBRA: Breast Ultrasound Dataset from Brazil
- BUS: Breast Ultrasound Dataset B
- BUS_UC: Breast Ultrasound from UC
- BUS_UCLM: Breast Ultrasound from UCLM

Usage:
    python scripts/preprocess_datasets.py --raw_dir ./dataset/raw --output_dir ./dataset/processed
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from PIL import Image
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def is_mask_blank(mask_path: str) -> bool:
    """
    Check if a mask is blank (contains no foreground pixels / no lesion).

    Args:
        mask_path: Path to the mask image

    Returns:
        True if mask is blank (no lesion), False if mask contains lesion
    """
    mask = Image.open(mask_path)

    # Convert to grayscale if RGB
    if mask.mode == 'RGB' or mask.mode == 'RGBA':
        mask = mask.convert('L')
    elif mask.mode == '1':
        mask = mask.convert('L')

    mask_array = np.array(mask)

    # Check if any pixel is non-zero (foreground)
    return np.sum(mask_array > 0) == 0


def is_merged_mask_blank(mask_paths: List[str]) -> bool:
    """
    Check if merged masks are blank (for BUSI with multiple masks).

    Args:
        mask_paths: List of mask paths to merge and check

    Returns:
        True if merged mask is blank, False if contains lesion
    """
    merged_mask = None
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert('L'))
        if merged_mask is None:
            merged_mask = mask
        else:
            merged_mask = np.maximum(merged_mask, mask)

    return np.sum(merged_mask > 0) == 0


def create_directory_structure(output_dir: str, dataset_name: str):
    """Create the output directory structure for a dataset."""
    for split in ['train', 'test']:
        for subdir in ['images', 'masks']:
            path = os.path.join(output_dir, dataset_name, split, subdir)
            os.makedirs(path, exist_ok=True)


def process_mask(mask_path: str, output_path: str):
    """
    Process mask to ensure it's binary (0 and 255).
    Handles different input formats (RGB, grayscale, 1-bit).
    """
    mask = Image.open(mask_path)

    # Convert to grayscale if RGB
    if mask.mode == 'RGB' or mask.mode == 'RGBA':
        mask = mask.convert('L')
    elif mask.mode == '1':
        mask = mask.convert('L')

    # Convert to numpy for processing
    mask_array = np.array(mask)

    # Binarize: anything > 0 becomes 255
    mask_array = (mask_array > 0).astype(np.uint8) * 255

    # Save as PNG
    mask_pil = Image.fromarray(mask_array, mode='L')
    mask_pil.save(output_path)


def process_image(image_path: str, output_path: str):
    """
    Process image to ensure consistent format (grayscale PNG).
    """
    image = Image.open(image_path)

    # Convert to grayscale if RGB
    if image.mode == 'RGB' or image.mode == 'RGBA':
        image = image.convert('L')
    elif image.mode == '1':
        image = image.convert('L')

    # Save as PNG
    image.save(output_path)


def train_test_split(items: List, test_ratio: float = 0.2, seed: int = 42) -> Tuple[List, List]:
    """Split items into train and test sets."""
    set_seed(seed)
    items = list(items)
    random.shuffle(items)
    split_idx = int(len(items) * (1 - test_ratio))
    return items[:split_idx], items[split_idx:]


def process_busi(raw_dir: str, output_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Process BUSI dataset.

    Structure:
        raw/BUSI/benign/benign (1).png, benign (1)_mask.png
        raw/BUSI/malignant/malignant (1).png, malignant (1)_mask.png
        raw/BUSI/normal/ (excluded - no lesions, all masks are blank)

    Note: Normal cases (entire folder) and any samples with blank masks are excluded.
    """
    print("\n" + "="*60)
    print("Processing BUSI dataset...")
    print("="*60)

    dataset_name = "BUSI"
    create_directory_structure(output_dir, dataset_name)

    busi_dir = os.path.join(raw_dir, "BUSI")

    # Count normal cases (entire folder excluded - all are no-lesion samples)
    excluded_normal = 0
    normal_dir = os.path.join(busi_dir, "normal")
    if os.path.exists(normal_dir):
        # Count image files (not masks) in normal folder
        normal_files = [f for f in os.listdir(normal_dir) if f.endswith('.png') and '_mask' not in f]
        excluded_normal = len(normal_files)
        print(f"Excluding {excluded_normal} normal cases (no lesions) from 'normal' folder")

    # Collect all image-mask pairs (excluding normal - no lesions)
    pairs = []
    excluded_blank = 0

    for category in ['benign', 'malignant']:
        category_dir = os.path.join(busi_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: {category_dir} not found")
            continue

        # Get all image files (not masks)
        for f in os.listdir(category_dir):
            if f.endswith('.png') and '_mask' not in f:
                image_path = os.path.join(category_dir, f)
                # Find corresponding mask
                base_name = f.replace('.png', '')
                mask_name = f"{base_name}_mask.png"
                mask_path = os.path.join(category_dir, mask_name)

                if os.path.exists(mask_path):
                    # Handle multiple masks - merge them
                    mask_paths = [mask_path]
                    # Check for additional masks like _mask_1.png
                    for i in range(1, 10):
                        extra_mask = os.path.join(category_dir, f"{base_name}_mask_{i}.png")
                        if os.path.exists(extra_mask):
                            mask_paths.append(extra_mask)

                    # Check if merged mask is blank (no lesion)
                    if is_merged_mask_blank(mask_paths):
                        excluded_blank += 1
                        continue

                    pairs.append({
                        'image': image_path,
                        'masks': mask_paths,
                        'category': category,
                        'name': f"{category}_{base_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                    })

    print(f"Found {len(pairs)} valid image-mask pairs (excluded {excluded_blank} blank masks from benign/malignant)")

    # Split
    train_pairs, test_pairs = train_test_split(pairs, test_ratio, seed)
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Total excluded = normal cases + blank masks in benign/malignant
    total_excluded = excluded_normal + excluded_blank

    # Process and save
    split_info = {
        'train': [],
        'test': [],
        'excluded_blank': excluded_blank,
        'excluded_normal': excluded_normal,
        'total_excluded': total_excluded
    }

    for split_name, split_pairs in [('train', train_pairs), ('test', test_pairs)]:
        for pair in tqdm(split_pairs, desc=f"Processing {split_name}"):
            output_name = f"{pair['name']}.png"

            # Process image
            img_output = os.path.join(output_dir, dataset_name, split_name, 'images', output_name)
            process_image(pair['image'], img_output)

            # Process and merge masks if multiple
            if len(pair['masks']) == 1:
                mask_output = os.path.join(output_dir, dataset_name, split_name, 'masks', output_name)
                process_mask(pair['masks'][0], mask_output)
            else:
                # Merge multiple masks
                merged_mask = None
                for mask_path in pair['masks']:
                    mask = np.array(Image.open(mask_path).convert('L'))
                    if merged_mask is None:
                        merged_mask = mask
                    else:
                        merged_mask = np.maximum(merged_mask, mask)

                merged_mask = (merged_mask > 0).astype(np.uint8) * 255
                mask_output = os.path.join(output_dir, dataset_name, split_name, 'masks', output_name)
                Image.fromarray(merged_mask, mode='L').save(mask_output)

            split_info[split_name].append({
                'name': pair['name'],
                'category': pair['category'],
                'original_image': pair['image']
            })

    # Save split info
    with open(os.path.join(output_dir, dataset_name, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"BUSI processing complete. Train: {len(split_info['train'])}, Test: {len(split_info['test'])}")
    print(f"  Excluded {excluded_normal} normal cases (entire 'normal' folder)")
    print(f"  Excluded {excluded_blank} samples with blank masks from benign/malignant")
    print(f"  Total excluded: {total_excluded}")
    return split_info


def process_busbra(raw_dir: str, output_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Process BUSBRA dataset.

    Structure:
        raw/BUSBRA/Images/bus_0001-l.png
        raw/BUSBRA/Masks/mask_0001-l.png

    Note: Samples with blank masks (no lesions) are excluded.
    """
    print("\n" + "="*60)
    print("Processing BUSBRA dataset...")
    print("="*60)

    dataset_name = "BUSBRA"
    create_directory_structure(output_dir, dataset_name)

    busbra_dir = os.path.join(raw_dir, "BUSBRA")
    images_dir = os.path.join(busbra_dir, "Images")
    masks_dir = os.path.join(busbra_dir, "Masks")

    # Collect all image-mask pairs
    pairs = []
    excluded_blank = 0

    for f in os.listdir(images_dir):
        if f.startswith('bus_') and f.endswith('.png'):
            image_path = os.path.join(images_dir, f)
            # Convert bus_0001-l.png to mask_0001-l.png
            mask_name = f.replace('bus_', 'mask_')
            mask_path = os.path.join(masks_dir, mask_name)

            if os.path.exists(mask_path):
                # Check if mask is blank (no lesion)
                if is_mask_blank(mask_path):
                    excluded_blank += 1
                    continue

                pairs.append({
                    'image': image_path,
                    'mask': mask_path,
                    'name': f.replace('.png', '').replace('bus_', 'busbra_')
                })

    print(f"Found {len(pairs)} valid image-mask pairs (excluded {excluded_blank} blank masks)")

    # Split
    train_pairs, test_pairs = train_test_split(pairs, test_ratio, seed)
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Process and save
    split_info = {'train': [], 'test': [], 'excluded_blank': excluded_blank}

    for split_name, split_pairs in [('train', train_pairs), ('test', test_pairs)]:
        for pair in tqdm(split_pairs, desc=f"Processing {split_name}"):
            output_name = f"{pair['name']}.png"

            img_output = os.path.join(output_dir, dataset_name, split_name, 'images', output_name)
            mask_output = os.path.join(output_dir, dataset_name, split_name, 'masks', output_name)

            process_image(pair['image'], img_output)
            process_mask(pair['mask'], mask_output)

            split_info[split_name].append({
                'name': pair['name'],
                'original_image': pair['image']
            })

    # Save split info
    with open(os.path.join(output_dir, dataset_name, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"BUSBRA processing complete. Train: {len(split_info['train'])}, Test: {len(split_info['test'])}")
    print(f"  Excluded {excluded_blank} samples with blank masks (no lesions)")
    return split_info


def process_bus(raw_dir: str, output_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Process BUS dataset.

    Structure:
        raw/BUS/original/000001.png
        raw/BUS/GT/000001.png

    Note: Samples with blank masks (no lesions) are excluded.
    """
    print("\n" + "="*60)
    print("Processing BUS dataset...")
    print("="*60)

    dataset_name = "BUS"
    create_directory_structure(output_dir, dataset_name)

    bus_dir = os.path.join(raw_dir, "BUS")
    images_dir = os.path.join(bus_dir, "original")
    masks_dir = os.path.join(bus_dir, "GT")

    # Collect all image-mask pairs
    pairs = []
    excluded_blank = 0

    for f in os.listdir(images_dir):
        if f.endswith('.png'):
            image_path = os.path.join(images_dir, f)
            mask_path = os.path.join(masks_dir, f)

            if os.path.exists(mask_path):
                # Check if mask is blank (no lesion)
                if is_mask_blank(mask_path):
                    excluded_blank += 1
                    continue

                pairs.append({
                    'image': image_path,
                    'mask': mask_path,
                    'name': f"bus_{f.replace('.png', '')}"
                })

    print(f"Found {len(pairs)} valid image-mask pairs (excluded {excluded_blank} blank masks)")

    # Split
    train_pairs, test_pairs = train_test_split(pairs, test_ratio, seed)
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Process and save
    split_info = {'train': [], 'test': [], 'excluded_blank': excluded_blank}

    for split_name, split_pairs in [('train', train_pairs), ('test', test_pairs)]:
        for pair in tqdm(split_pairs, desc=f"Processing {split_name}"):
            output_name = f"{pair['name']}.png"

            img_output = os.path.join(output_dir, dataset_name, split_name, 'images', output_name)
            mask_output = os.path.join(output_dir, dataset_name, split_name, 'masks', output_name)

            process_image(pair['image'], img_output)
            process_mask(pair['mask'], mask_output)

            split_info[split_name].append({
                'name': pair['name'],
                'original_image': pair['image']
            })

    # Save split info
    with open(os.path.join(output_dir, dataset_name, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"BUS processing complete. Train: {len(split_info['train'])}, Test: {len(split_info['test'])}")
    print(f"  Excluded {excluded_blank} samples with blank masks (no lesions)")
    return split_info


def process_bus_uc(raw_dir: str, output_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Process BUS_UC dataset.

    Structure:
        raw/BUS_UC/All/images/001.png
        raw/BUS_UC/All/masks/001.png

    Note: Samples with blank masks (no lesions) are excluded.
    """
    print("\n" + "="*60)
    print("Processing BUS_UC dataset...")
    print("="*60)

    dataset_name = "BUS_UC"
    create_directory_structure(output_dir, dataset_name)

    bus_uc_dir = os.path.join(raw_dir, "BUS_UC", "All")
    images_dir = os.path.join(bus_uc_dir, "images")
    masks_dir = os.path.join(bus_uc_dir, "masks")

    # Collect all image-mask pairs
    pairs = []
    excluded_blank = 0

    for f in os.listdir(images_dir):
        if f.endswith('.png'):
            image_path = os.path.join(images_dir, f)
            mask_path = os.path.join(masks_dir, f)

            if os.path.exists(mask_path):
                # Check if mask is blank (no lesion)
                if is_mask_blank(mask_path):
                    excluded_blank += 1
                    continue

                pairs.append({
                    'image': image_path,
                    'mask': mask_path,
                    'name': f"bus_uc_{f.replace('.png', '')}"
                })

    print(f"Found {len(pairs)} valid image-mask pairs (excluded {excluded_blank} blank masks)")

    # Split
    train_pairs, test_pairs = train_test_split(pairs, test_ratio, seed)
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Process and save
    split_info = {'train': [], 'test': [], 'excluded_blank': excluded_blank}

    for split_name, split_pairs in [('train', train_pairs), ('test', test_pairs)]:
        for pair in tqdm(split_pairs, desc=f"Processing {split_name}"):
            output_name = f"{pair['name']}.png"

            img_output = os.path.join(output_dir, dataset_name, split_name, 'images', output_name)
            mask_output = os.path.join(output_dir, dataset_name, split_name, 'masks', output_name)

            process_image(pair['image'], img_output)
            process_mask(pair['mask'], mask_output)

            split_info[split_name].append({
                'name': pair['name'],
                'original_image': pair['image']
            })

    # Save split info
    with open(os.path.join(output_dir, dataset_name, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"BUS_UC processing complete. Train: {len(split_info['train'])}, Test: {len(split_info['test'])}")
    print(f"  Excluded {excluded_blank} samples with blank masks (no lesions)")
    return split_info


def process_bus_uclm(raw_dir: str, output_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Process BUS_UCLM dataset.

    Structure:
        raw/BUS_UCLM/images/ALWI_000.png
        raw/BUS_UCLM/masks/ALWI_000.png

    Note: Samples with blank masks (no lesions) are excluded.
    """
    print("\n" + "="*60)
    print("Processing BUS_UCLM dataset...")
    print("="*60)

    dataset_name = "BUS_UCLM"
    create_directory_structure(output_dir, dataset_name)

    bus_uclm_dir = os.path.join(raw_dir, "BUS_UCLM")
    images_dir = os.path.join(bus_uclm_dir, "images")
    masks_dir = os.path.join(bus_uclm_dir, "masks")

    # Collect all image-mask pairs
    pairs = []
    excluded_blank = 0

    for f in os.listdir(images_dir):
        if f.endswith('.png'):
            image_path = os.path.join(images_dir, f)
            mask_path = os.path.join(masks_dir, f)

            if os.path.exists(mask_path):
                # Check if mask is blank (no lesion)
                if is_mask_blank(mask_path):
                    excluded_blank += 1
                    continue

                pairs.append({
                    'image': image_path,
                    'mask': mask_path,
                    'name': f"bus_uclm_{f.replace('.png', '')}"
                })

    print(f"Found {len(pairs)} valid image-mask pairs (excluded {excluded_blank} blank masks)")

    # Split
    train_pairs, test_pairs = train_test_split(pairs, test_ratio, seed)
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    # Process and save
    split_info = {'train': [], 'test': [], 'excluded_blank': excluded_blank}

    for split_name, split_pairs in [('train', train_pairs), ('test', test_pairs)]:
        for pair in tqdm(split_pairs, desc=f"Processing {split_name}"):
            output_name = f"{pair['name']}.png"

            img_output = os.path.join(output_dir, dataset_name, split_name, 'images', output_name)
            mask_output = os.path.join(output_dir, dataset_name, split_name, 'masks', output_name)

            process_image(pair['image'], img_output)
            process_mask(pair['mask'], mask_output)

            split_info[split_name].append({
                'name': pair['name'],
                'original_image': pair['image']
            })

    # Save split info
    with open(os.path.join(output_dir, dataset_name, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"BUS_UCLM processing complete. Train: {len(split_info['train'])}, Test: {len(split_info['test'])}")
    print(f"  Excluded {excluded_blank} samples with blank masks (no lesions)")
    return split_info


def create_combined_split_info(output_dir: str, all_splits: Dict):
    """Create combined split info for all datasets."""
    combined = {
        'train': [],
        'test': [],
        'datasets': {},
        'statistics': {},
        'excluded': {}
    }

    total_train = 0
    total_test = 0
    total_excluded = 0

    for dataset_name, split_info in all_splits.items():
        # Get excluded counts (BUSI has both excluded_normal and excluded_blank)
        excluded_blank = split_info.get('excluded_blank', 0)
        excluded_normal = split_info.get('excluded_normal', 0)
        dataset_total_excluded = split_info.get('total_excluded', excluded_blank)

        combined['datasets'][dataset_name] = {
            'train': len(split_info['train']),
            'test': len(split_info['test']),
            'excluded_blank': excluded_blank,
            'excluded_normal': excluded_normal,
            'total_excluded': dataset_total_excluded
        }
        combined['excluded'][dataset_name] = dataset_total_excluded
        total_train += len(split_info['train'])
        total_test += len(split_info['test'])
        total_excluded += dataset_total_excluded

        # Add to combined lists with dataset info
        for item in split_info['train']:
            combined['train'].append({
                'dataset': dataset_name,
                'name': item['name']
            })
        for item in split_info['test']:
            combined['test'].append({
                'dataset': dataset_name,
                'name': item['name']
            })

    combined['statistics'] = {
        'total_train': total_train,
        'total_test': total_test,
        'total': total_train + total_test,
        'total_excluded': total_excluded,
        'train_ratio': total_train / (total_train + total_test) if (total_train + total_test) > 0 else 0,
        'test_ratio': total_test / (total_train + total_test) if (total_train + total_test) > 0 else 0
    }

    # Save combined info
    with open(os.path.join(output_dir, 'combined_split_info.json'), 'w') as f:
        json.dump(combined, f, indent=2)

    return combined


def main():
    parser = argparse.ArgumentParser(description='Preprocess breast ultrasound datasets')
    parser.add_argument('--raw_dir', type=str, default='./dataset/raw',
                        help='Directory containing raw datasets')
    parser.add_argument('--output_dir', type=str, default='./dataset/processed',
                        help='Output directory for processed datasets')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Ratio of test set (default: 0.2 = 20%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    args = parser.parse_args()

    # Convert to absolute paths
    raw_dir = os.path.abspath(args.raw_dir)
    output_dir = os.path.abspath(args.output_dir)

    print("="*60)
    print("UltraRefiner Dataset Preprocessing")
    print("="*60)
    print(f"Raw data directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Random seed: {args.seed}")
    print("\nNOTE: Samples with blank masks (no lesions) will be EXCLUDED")
    print("="*60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each dataset
    all_splits = {}

    # BUSI
    if os.path.exists(os.path.join(raw_dir, "BUSI")):
        all_splits['BUSI'] = process_busi(raw_dir, output_dir, args.test_ratio, args.seed)
    else:
        print("BUSI dataset not found, skipping...")

    # BUSBRA
    if os.path.exists(os.path.join(raw_dir, "BUSBRA")):
        all_splits['BUSBRA'] = process_busbra(raw_dir, output_dir, args.test_ratio, args.seed)
    else:
        print("BUSBRA dataset not found, skipping...")

    # BUS
    if os.path.exists(os.path.join(raw_dir, "BUS")):
        all_splits['BUS'] = process_bus(raw_dir, output_dir, args.test_ratio, args.seed)
    else:
        print("BUS dataset not found, skipping...")

    # BUS_UC
    if os.path.exists(os.path.join(raw_dir, "BUS_UC")):
        all_splits['BUS_UC'] = process_bus_uc(raw_dir, output_dir, args.test_ratio, args.seed)
    else:
        print("BUS_UC dataset not found, skipping...")

    # BUS_UCLM
    if os.path.exists(os.path.join(raw_dir, "BUS_UCLM")):
        all_splits['BUS_UCLM'] = process_bus_uclm(raw_dir, output_dir, args.test_ratio, args.seed)
    else:
        print("BUS_UCLM dataset not found, skipping...")

    # Create combined split info
    combined = create_combined_split_info(output_dir, all_splits)

    # Print summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<15} {'Train':>10} {'Test':>10} {'Total':>10} {'Excluded':>10}")
    print("-" * 60)
    for dataset_name, counts in combined['datasets'].items():
        total = counts['train'] + counts['test']
        excluded = counts.get('total_excluded', counts.get('excluded_blank', 0))
        print(f"{dataset_name:<15} {counts['train']:>10} {counts['test']:>10} {total:>10} {excluded:>10}")
    print("-" * 60)
    stats = combined['statistics']
    print(f"{'TOTAL':<15} {stats['total_train']:>10} {stats['total_test']:>10} {stats['total']:>10} {stats['total_excluded']:>10}")
    print(f"\nTrain/Test ratio: {stats['train_ratio']:.1%} / {stats['test_ratio']:.1%}")
    print(f"Total samples excluded (no lesions): {stats['total_excluded']}")
    print(f"\nSplit information saved to: {os.path.join(output_dir, 'combined_split_info.json')}")
    print("\nIMPORTANT: This fixed split should be used for ALL training phases:")
    print("  - Phase 1: TransUNet training (per dataset)")
    print("  - Phase 2: SAM finetuning (all datasets combined)")
    print("  - Phase 3: End-to-end training")
    print("\nTest set is completely held out and should ONLY be used for final evaluation!")
    print("\nNOTE: All samples with blank masks (no lesions) and BUSI 'normal' folder have been excluded.")


if __name__ == '__main__':
    main()
