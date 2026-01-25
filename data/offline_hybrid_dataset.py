"""
Offline Hybrid Dataset for Phase 2 SAM Refinement Training.

Combines pre-generated data from two sources:
1. Real TransUNet predictions (70% by default)
2. Offline augmented GT masks (30% by default)

No online augmentation - everything is pre-generated for fast loading.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Union


class OfflineHybridDataset(Dataset):
    """
    Combines real TransUNet predictions with offline augmented masks.
    Both data sources are pre-generated for fast loading.

    Directory structure:
        pred_data_root/
        └── {dataset}/
            └── train/
                ├── images/
                ├── masks/
                ├── coarse_masks/   (TransUNet predictions .npy)
                └── metadata.json

        aug_data_root/
        └── {dataset}/
            └── train/
                ├── images/
                ├── masks/
                ├── coarse_masks/   (augmented masks .npy)
                └── metadata.json
    """

    def __init__(
        self,
        pred_data_root: str,
        aug_data_root: str,
        dataset_name: str,
        img_size: int = 1024,
        transunet_img_size: int = 224,
        real_ratio: float = 0.7,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.pred_data_root = pred_data_root
        self.aug_data_root = aug_data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.transunet_img_size = transunet_img_size
        self.real_ratio = real_ratio
        self.is_train = is_train

        # Setup paths for real predictions
        self.pred_dir = os.path.join(pred_data_root, dataset_name, 'train')
        self.pred_image_dir = os.path.join(self.pred_dir, 'images')
        self.pred_mask_dir = os.path.join(self.pred_dir, 'masks')
        self.pred_coarse_dir = os.path.join(self.pred_dir, 'coarse_masks')

        # Setup paths for augmented data
        self.aug_dir = os.path.join(aug_data_root, dataset_name, 'train')
        self.aug_image_dir = os.path.join(self.aug_dir, 'images')
        self.aug_mask_dir = os.path.join(self.aug_dir, 'masks')
        self.aug_coarse_dir = os.path.join(self.aug_dir, 'coarse_masks')

        # Load samples from both sources
        self.pred_samples = self._load_samples(self.pred_dir, 'real')
        self.aug_samples = self._load_samples(self.aug_dir, 'augmented')

        # Train/val split for both
        np.random.seed(seed)

        pred_indices = np.random.permutation(len(self.pred_samples))
        pred_split = int(len(pred_indices) * split_ratio)
        if is_train:
            self.pred_samples = [self.pred_samples[i] for i in pred_indices[:pred_split]]
        else:
            self.pred_samples = [self.pred_samples[i] for i in pred_indices[pred_split:]]

        np.random.seed(seed + 1)  # Different seed for augmented to get different split
        aug_indices = np.random.permutation(len(self.aug_samples))
        aug_split = int(len(aug_indices) * split_ratio)
        if is_train:
            self.aug_samples = [self.aug_samples[i] for i in aug_indices[:aug_split]]
        else:
            self.aug_samples = [self.aug_samples[i] for i in aug_indices[aug_split:]]

        # Calculate how many samples from each source based on ratio
        total_real = len(self.pred_samples)
        total_aug = len(self.aug_samples)

        # Target: real_ratio of samples from predictions, rest from augmented
        # Combine and let random sampling handle the ratio
        self.all_samples = []

        # Add real prediction samples
        for s in self.pred_samples:
            s['source_type'] = 'real'
            s['image_dir'] = self.pred_image_dir
            s['mask_dir'] = self.pred_mask_dir
            s['coarse_dir'] = self.pred_coarse_dir
            self.all_samples.append(s)

        # Add augmented samples
        for s in self.aug_samples:
            s['source_type'] = 'augmented'
            s['image_dir'] = self.aug_image_dir
            s['mask_dir'] = self.aug_mask_dir
            s['coarse_dir'] = self.aug_coarse_dir
            self.all_samples.append(s)

        # Shuffle combined samples
        np.random.seed(seed + 2)
        np.random.shuffle(self.all_samples)

        # Subsample to match desired ratio
        n_real = int(len(self.all_samples) * real_ratio)
        n_aug = len(self.all_samples) - n_real

        real_samples = [s for s in self.all_samples if s['source_type'] == 'real']
        aug_samples = [s for s in self.all_samples if s['source_type'] == 'augmented']

        # Take appropriate number from each
        selected_real = real_samples[:min(n_real, len(real_samples))]
        selected_aug = aug_samples[:min(n_aug, len(aug_samples))]

        self.samples = selected_real + selected_aug
        np.random.shuffle(self.samples)

        # SAM normalization parameters
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        n_real_final = len([s for s in self.samples if s['source_type'] == 'real'])
        n_aug_final = len([s for s in self.samples if s['source_type'] == 'augmented'])
        print(f"Loaded {len(self.samples)} {'train' if is_train else 'val'} samples from {dataset_name}")
        print(f"  Real predictions: {n_real_final} ({100*n_real_final/len(self.samples):.0f}%)")
        print(f"  Augmented: {n_aug_final} ({100*n_aug_final/len(self.samples):.0f}%)")

    def _load_samples(self, base_dir: str, source: str) -> List[dict]:
        """Load samples from metadata.json or scan directory."""
        metadata_path = os.path.join(base_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                samples = meta.get('samples', [])
                return samples
        else:
            # Fallback: scan coarse_masks directory
            coarse_dir = os.path.join(base_dir, 'coarse_masks')
            if not os.path.exists(coarse_dir):
                return []
            samples = []
            for f in sorted(os.listdir(coarse_dir)):
                if f.endswith('.npy'):
                    name = os.path.splitext(f)[0]
                    samples.append({'name': name})
            return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        name = sample['name']
        source_type = sample['source_type']
        image_dir = sample['image_dir']
        mask_dir = sample['mask_dir']
        coarse_dir = sample['coarse_dir']

        # Get metadata
        dice = sample.get('dice', 1.0)
        error_type = sample.get('error_type', 'unknown')

        # Load image
        image = None
        for ext in ['.png', '.jpg', '.jpeg', '.npy']:
            image_path = os.path.join(image_dir, f"{name}{ext}")
            if os.path.exists(image_path):
                if ext == '.npy':
                    image = np.load(image_path)
                else:
                    image = np.array(Image.open(image_path).convert('RGB'))
                break

        if image is None:
            raise FileNotFoundError(f"Image not found for {name} in {image_dir}")

        # Load GT mask
        mask_path = os.path.join(mask_dir, f"{name}.png")
        gt_mask = np.array(Image.open(mask_path).convert('L'))

        if gt_mask.max() > 1:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        else:
            gt_mask = gt_mask.astype(np.float32)

        # Load coarse mask
        coarse_path = os.path.join(coarse_dir, f"{name}.npy")
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
            'source': source_type,
        }


class MultiDatasetOfflineHybrid(Dataset):
    """Combined offline hybrid dataset from multiple sources."""

    def __init__(
        self,
        pred_data_root: str,
        aug_data_root: str,
        dataset_names: List[str],
        img_size: int = 1024,
        transunet_img_size: int = 224,
        real_ratio: float = 0.7,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.datasets = []
        self.cumulative_lengths = [0]

        for ds_name in dataset_names:
            ds = OfflineHybridDataset(
                pred_data_root=pred_data_root,
                aug_data_root=aug_data_root,
                dataset_name=ds_name,
                img_size=img_size,
                transunet_img_size=transunet_img_size,
                real_ratio=real_ratio,
                split_ratio=split_ratio,
                is_train=is_train,
                seed=seed,
            )
            self.datasets.append(ds)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

        self.total_length = self.cumulative_lengths[-1]
        print(f"Combined offline hybrid dataset: {self.total_length} samples from {len(dataset_names)} datasets")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                local_idx = idx - start
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")


def get_offline_hybrid_dataloaders(
    pred_data_root: str,
    aug_data_root: str,
    dataset_names: Union[str, List[str]],
    batch_size: int = 4,
    img_size: int = 1024,
    transunet_img_size: int = 224,
    real_ratio: float = 0.7,
    num_workers: int = 4,
    split_ratio: float = 0.9,
    seed: int = 42,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaders with offline hybrid data."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    if len(dataset_names) == 1:
        train_dataset = OfflineHybridDataset(
            pred_data_root=pred_data_root,
            aug_data_root=aug_data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = OfflineHybridDataset(
            pred_data_root=pred_data_root,
            aug_data_root=aug_data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
            split_ratio=split_ratio,
            is_train=False,
            seed=seed,
        )
    else:
        train_dataset = MultiDatasetOfflineHybrid(
            pred_data_root=pred_data_root,
            aug_data_root=aug_data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = MultiDatasetOfflineHybrid(
            pred_data_root=pred_data_root,
            aug_data_root=aug_data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
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
