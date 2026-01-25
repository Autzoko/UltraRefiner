"""
Hybrid Dataset for Phase 2 SAM Refinement Training.

Combines:
1. Real TransUNet predictions (actual failure modes)
2. Augmented GT masks (controlled diversity)

This gives the best of both worlds:
- Real predictions capture actual model behavior
- Augmented masks provide more diversity and edge cases

Usage:
    from data import get_hybrid_dataloaders

    train_loader, val_loader = get_hybrid_dataloaders(
        gt_data_root='./dataset/processed',
        pred_data_root='./dataset/transunet_preds',
        dataset_names=['BUSI', 'BUSBRA'],
        real_ratio=0.5,  # 50% real predictions, 50% augmented
    )
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Union

from .mask_augmentation import MaskAugmentor, create_augmentor


class HybridDataset(Dataset):
    """
    Dataset that combines real TransUNet predictions with augmented GT masks.

    For each sample:
    - With probability `real_ratio`: use real TransUNet prediction
    - With probability `1 - real_ratio`: use augmented GT mask

    Directory structure for predictions:
        pred_data_root/
        └── {dataset}/
            └── train/
                ├── images/
                ├── masks/
                └── coarse_masks/  # TransUNet predictions (.npy)
    """

    def __init__(
        self,
        gt_data_root: str,
        pred_data_root: str,
        dataset_name: str,
        img_size: int = 1024,
        transunet_img_size: int = 224,
        real_ratio: float = 0.5,
        augmentor: Optional[MaskAugmentor] = None,
        augmentor_preset: str = 'default',
        soft_mask_prob: float = 0.8,
        use_fast_soft_mask: bool = True,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            gt_data_root: Root directory containing GT datasets.
            pred_data_root: Root directory containing TransUNet predictions.
            dataset_name: Dataset name (e.g., BUSI, BUSBRA).
            img_size: Output image size (SAM input size, typically 1024).
            transunet_img_size: TransUNet resolution for resolution path matching.
            real_ratio: Ratio of real predictions vs augmented (0.5 = 50% each).
            augmentor: Custom MaskAugmentor instance.
            augmentor_preset: Preset for augmentor if not provided.
            soft_mask_prob: Probability of soft mask for augmented samples.
            use_fast_soft_mask: Use fast Gaussian blur for augmentation.
            split_ratio: Train/val split ratio.
            is_train: Whether this is training set.
            seed: Random seed.
        """
        self.gt_data_root = gt_data_root
        self.pred_data_root = pred_data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.transunet_img_size = transunet_img_size
        self.real_ratio = real_ratio
        self.is_train = is_train

        # Setup paths
        self.gt_dir = os.path.join(gt_data_root, dataset_name, 'train')
        self.pred_dir = os.path.join(pred_data_root, dataset_name, 'train')

        self.gt_image_dir = os.path.join(self.gt_dir, 'images')
        self.gt_mask_dir = os.path.join(self.gt_dir, 'masks')
        self.coarse_mask_dir = os.path.join(self.pred_dir, 'coarse_masks')

        # Check if predictions exist
        self.has_predictions = os.path.exists(self.coarse_mask_dir)
        if not self.has_predictions:
            print(f"Warning: No predictions found at {self.coarse_mask_dir}")
            print(f"  Will use 100% augmented data for {dataset_name}")
            self.real_ratio = 0.0

        # Get all samples
        self.samples = self._load_samples()

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * split_ratio)

        if is_train:
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [self.samples[i] for i in indices[split_idx:]]

        # Create augmentor for GT augmentation
        if augmentor is not None:
            self.augmentor = augmentor
        else:
            self.augmentor = create_augmentor(
                preset=augmentor_preset,
                soft_mask_prob=soft_mask_prob,
                use_fast_soft_mask=use_fast_soft_mask,
            )

        # SAM normalization parameters
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        print(f"Loaded {len(self.samples)} {'train' if is_train else 'val'} samples from {dataset_name}")
        print(f"  Real prediction ratio: {self.real_ratio:.0%}")
        print(f"  Augmented ratio: {1 - self.real_ratio:.0%}")

    def _load_samples(self) -> List[str]:
        """Load sample names from image directory."""
        samples = []
        for f in os.listdir(self.gt_image_dir):
            if f.endswith(('.png', '.jpg', '.jpeg', '.npy')):
                name = os.path.splitext(f)[0]
                # Check GT mask exists
                mask_path = os.path.join(self.gt_mask_dir, f"{name}.png")
                if os.path.exists(mask_path):
                    samples.append(name)
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]

        # Load image
        image_path = os.path.join(self.gt_image_dir, f"{name}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.gt_image_dir, f"{name}.npy")

        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))

        # Load GT mask
        mask_path = os.path.join(self.gt_mask_dir, f"{name}.png")
        gt_mask = np.array(Image.open(mask_path).convert('L'))

        # Normalize to [0, 1]
        if gt_mask.max() > 1:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        else:
            gt_mask = gt_mask.astype(np.float32)

        # Decide: real prediction or augmented?
        use_real = (np.random.random() < self.real_ratio) and self.has_predictions

        if use_real:
            # Load real TransUNet prediction
            coarse_path = os.path.join(self.coarse_mask_dir, f"{name}.npy")
            if os.path.exists(coarse_path):
                coarse_mask = np.load(coarse_path)
                # Compute Dice
                pred_binary = (coarse_mask > 0.5).astype(np.float32)
                gt_binary = (gt_mask > 0.5).astype(np.float32)
                intersection = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum()
                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                source = 'real'
                error_type = 'transunet_prediction'
            else:
                # Fallback to augmentation if prediction missing
                use_real = False

        if not use_real:
            # Apply on-the-fly augmentation
            if gt_mask.sum() < 10:
                coarse_mask = gt_mask.copy()
                dice = 1.0
                error_type = 'empty_gt'
            else:
                coarse_mask, aug_info = self.augmentor(gt_mask)
                dice = aug_info.get('dice', 1.0)
                error_type = aug_info.get('error_type', 'unknown')
            source = 'augmented'

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
            # Step 1: Resize to TransUNet resolution
            image_small = TF.resize(image, [self.transunet_img_size, self.transunet_img_size])
            coarse_mask_small = TF.resize(
                coarse_mask.unsqueeze(0), [self.transunet_img_size, self.transunet_img_size],
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)

            # Step 2: Resize to SAM input size
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

        # GT mask: resize directly to SAM size
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
            'source': source,  # 'real' or 'augmented'
        }


class MultiDatasetHybrid(Dataset):
    """Combined hybrid dataset from multiple sources."""

    def __init__(
        self,
        gt_data_root: str,
        pred_data_root: str,
        dataset_names: List[str],
        img_size: int = 1024,
        transunet_img_size: int = 224,
        real_ratio: float = 0.5,
        augmentor: Optional[MaskAugmentor] = None,
        augmentor_preset: str = 'default',
        use_fast_soft_mask: bool = True,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        self.datasets = []
        self.cumulative_lengths = [0]

        # Share augmentor across datasets
        if augmentor is None:
            augmentor = create_augmentor(
                preset=augmentor_preset,
                use_fast_soft_mask=use_fast_soft_mask,
            )

        for ds_name in dataset_names:
            ds = HybridDataset(
                gt_data_root=gt_data_root,
                pred_data_root=pred_data_root,
                dataset_name=ds_name,
                img_size=img_size,
                transunet_img_size=transunet_img_size,
                real_ratio=real_ratio,
                augmentor=augmentor,
                split_ratio=split_ratio,
                is_train=is_train,
                seed=seed,
            )
            self.datasets.append(ds)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

        self.total_length = self.cumulative_lengths[-1]
        print(f"Combined hybrid dataset: {self.total_length} samples from {len(dataset_names)} datasets")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                local_idx = idx - start
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")


def get_hybrid_dataloaders(
    gt_data_root: str,
    pred_data_root: str,
    dataset_names: Union[str, List[str]],
    batch_size: int = 4,
    img_size: int = 1024,
    transunet_img_size: int = 224,
    real_ratio: float = 0.5,
    augmentor_preset: str = 'default',
    use_fast_soft_mask: bool = True,
    num_workers: int = 4,
    split_ratio: float = 0.9,
    seed: int = 42,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders with hybrid data.

    Args:
        gt_data_root: Root directory containing GT datasets.
        pred_data_root: Root directory containing TransUNet predictions.
        dataset_names: Single dataset name or list of names to combine.
        batch_size: Batch size.
        img_size: SAM input size (typically 1024).
        transunet_img_size: TransUNet resolution for resolution path matching.
        real_ratio: Ratio of real predictions (0.5 = 50% real, 50% augmented).
        augmentor_preset: Augmentation preset for GT augmentation.
        use_fast_soft_mask: Use fast Gaussian blur for soft mask conversion.
        num_workers: Data loading workers.
        split_ratio: Train/val split.
        seed: Random seed.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.

    Returns:
        train_loader, val_loader
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Create shared augmentor
    augmentor = create_augmentor(
        preset=augmentor_preset,
        use_fast_soft_mask=use_fast_soft_mask,
    )

    if len(dataset_names) == 1:
        train_dataset = HybridDataset(
            gt_data_root=gt_data_root,
            pred_data_root=pred_data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
            augmentor=augmentor,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = HybridDataset(
            gt_data_root=gt_data_root,
            pred_data_root=pred_data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
            augmentor=augmentor,
            split_ratio=split_ratio,
            is_train=False,
            seed=seed,
        )
    else:
        train_dataset = MultiDatasetHybrid(
            gt_data_root=gt_data_root,
            pred_data_root=pred_data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
            augmentor=augmentor,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = MultiDatasetHybrid(
            gt_data_root=gt_data_root,
            pred_data_root=pred_data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            real_ratio=real_ratio,
            augmentor=augmentor,
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
