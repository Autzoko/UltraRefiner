"""
Offline Augmented Dataset for Phase 2 SAM Refinement Training.

Loads pre-generated augmented masks from disk for fast training.
Use with generate_augmented_masks.py to create the data.

Key optimizations (matching AugmentedSAMDataset):
- Uses metadata.json for fast sample indexing (no directory scanning)
- No symlinks (files are copied)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Union


class OfflineAugmentedDataset(Dataset):
    """
    Dataset that loads pre-generated augmented masks from disk.
    Uses metadata.json for fast indexing (no directory scanning).

    Directory structure:
        data_root/
        └── {dataset}/
            └── train/
                ├── images/         (copied files, not symlinks)
                ├── masks/          (GT masks, copied files)
                ├── coarse_masks/   (augmented masks as .npy)
                └── metadata.json   (sample index for fast lookup)
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        img_size: int = 1024,
        transunet_img_size: int = 224,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.transunet_img_size = transunet_img_size
        self.is_train = is_train

        # Setup paths
        self.base_dir = os.path.join(data_root, dataset_name, 'train')
        self.image_dir = os.path.join(self.base_dir, 'images')
        self.mask_dir = os.path.join(self.base_dir, 'masks')
        self.coarse_mask_dir = os.path.join(self.base_dir, 'coarse_masks')

        # Load metadata for fast indexing (no directory scanning)
        metadata_path = os.path.join(self.base_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                self.samples = meta.get('samples', [])
                # Create lookup dict for metadata
                self.sample_meta = {s['name']: s for s in self.samples}
        else:
            # Fallback: scan directory (slower)
            print(f"Warning: metadata.json not found, scanning directory (slower)...")
            self.samples = self._scan_samples()
            self.sample_meta = {}

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * split_ratio)

        if is_train:
            selected = indices[:split_idx]
        else:
            selected = indices[split_idx:]

        self.samples = [self.samples[i] for i in selected]

        # SAM normalization parameters
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        print(f"Loaded {len(self.samples)} {'train' if is_train else 'val'} samples from {dataset_name}")

    def _scan_samples(self) -> List[dict]:
        """Fallback: scan directory for samples (slower)."""
        samples = []
        for f in sorted(os.listdir(self.coarse_mask_dir)):
            if f.endswith('.npy'):
                name = os.path.splitext(f)[0]
                samples.append({'name': name})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample['name'] if isinstance(sample, dict) else sample

        # Get metadata
        meta = self.sample_meta.get(name, {})
        dice = meta.get('dice', 1.0)
        error_type = meta.get('error_type', 'unknown')

        # Load image
        image = None
        for ext in ['.png', '.jpg', '.jpeg', '.npy']:
            image_path = os.path.join(self.image_dir, f"{name}{ext}")
            if os.path.exists(image_path):
                if ext == '.npy':
                    image = np.load(image_path)
                else:
                    image = np.array(Image.open(image_path).convert('RGB'))
                break

        if image is None:
            raise FileNotFoundError(f"Image not found for {name}")

        # Load GT mask
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        gt_mask = np.array(Image.open(mask_path).convert('L'))

        if gt_mask.max() > 1:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        else:
            gt_mask = gt_mask.astype(np.float32)

        # Load pre-generated coarse mask
        coarse_path = os.path.join(self.coarse_mask_dir, f"{name}.npy")
        coarse_mask = np.load(coarse_path)

        # Resize coarse_mask to match gt_mask if needed
        if coarse_mask.shape != gt_mask.shape:
            import cv2
            coarse_mask = cv2.resize(
                coarse_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Convert to tensors
        image = torch.from_numpy(image).float()
        gt_mask = torch.from_numpy(gt_mask).float()
        coarse_mask = torch.from_numpy(coarse_mask).float()

        # Handle grayscale images
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        elif image.shape[-1] == 1:
            image = image.squeeze(-1).unsqueeze(0).repeat(3, 1, 1)

        original_size = (image.shape[-2], image.shape[-1])

        # Resolution path matching
        if self.transunet_img_size > 0 and self.transunet_img_size < self.img_size:
            image_small = TF.resize(image, [self.transunet_img_size, self.transunet_img_size])
            coarse_mask_small = TF.resize(
                coarse_mask.unsqueeze(0), [self.transunet_img_size, self.transunet_img_size],
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)

            image = TF.resize(image_small, [self.img_size, self.img_size])
            coarse_mask = TF.resize(
                coarse_mask_small.unsqueeze(0), [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)
        else:
            image = TF.resize(image, [self.img_size, self.img_size])
            coarse_mask = TF.resize(
                coarse_mask.unsqueeze(0), [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)

        gt_mask = TF.resize(
            gt_mask.unsqueeze(0), [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST
        ).squeeze(0)

        # SAM preprocessing
        image = (image - self.pixel_mean) / self.pixel_std

        return {
            'image': image,
            'label': gt_mask,
            'coarse_mask': coarse_mask,
            'name': name,
            'original_size': original_size,
            'dice': dice,
            'error_type': error_type,
            'source': 'offline_augmented',
        }


class MultiDatasetOfflineAugmented(Dataset):
    """Combined dataset from multiple sources with offline augmentation."""

    def __init__(
        self,
        data_root: str,
        dataset_names: List[str],
        img_size: int = 1024,
        transunet_img_size: int = 224,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.datasets = []
        self.cumulative_lengths = [0]

        for ds_name in dataset_names:
            ds = OfflineAugmentedDataset(
                data_root=data_root,
                dataset_name=ds_name,
                img_size=img_size,
                transunet_img_size=transunet_img_size,
                split_ratio=split_ratio,
                is_train=is_train,
                seed=seed,
            )
            self.datasets.append(ds)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

        self.total_length = self.cumulative_lengths[-1]
        print(f"Combined offline dataset: {self.total_length} samples from {len(dataset_names)} datasets")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                local_idx = idx - start
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")


def get_offline_augmented_dataloaders(
    data_root: str,
    dataset_names: Union[str, List[str]],
    batch_size: int = 4,
    img_size: int = 1024,
    transunet_img_size: int = 224,
    num_workers: int = 4,
    split_ratio: float = 0.9,
    seed: int = 42,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders with offline augmented data."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    if len(dataset_names) == 1:
        train_dataset = OfflineAugmentedDataset(
            data_root=data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = OfflineAugmentedDataset(
            data_root=data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            split_ratio=split_ratio,
            is_train=False,
            seed=seed,
        )
    else:
        train_dataset = MultiDatasetOfflineAugmented(
            data_root=data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = MultiDatasetOfflineAugmented(
            data_root=data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            split_ratio=split_ratio,
            is_train=False,
            seed=seed,
        )

    use_persistent = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return train_loader, val_loader
