"""
Online Augmented Dataset for Phase 2 SAM Refinement Training.

This dataset applies on-the-fly mask augmentation to simulate TransUNet failure
modes, rather than loading pre-generated augmented masks from disk.

Benefits:
- Unlimited augmentation diversity (new augmentation each epoch)
- No disk storage required for augmented masks
- Easy to adjust augmentation distribution during training
- Matches Phase 3 prompt generation logic exactly
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Dict, Union

from .mask_augmentation import MaskAugmentor, create_augmentor


class OnlineAugmentedDataset(Dataset):
    """
    Dataset with on-the-fly mask augmentation for Phase 2 SAM training.

    For each sample, the GT mask is augmented in real-time to create a
    pseudo-coarse mask that simulates TransUNet failure modes.

    Directory structure:
        data_root/
        └── {dataset}/
            └── train/
                ├── images/
                └── masks/
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        img_size: int = 1024,
        transunet_img_size: int = 224,
        augmentor: Optional[MaskAugmentor] = None,
        augmentor_preset: str = 'default',
        soft_mask_prob: float = 0.8,
        soft_mask_temperature: Tuple[float, float] = (2.0, 8.0),
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
        return_augmentation_info: bool = True,
        use_fast_soft_mask: bool = False,
    ):
        """
        Args:
            data_root: Root directory containing datasets.
            dataset_name: Dataset name (e.g., BUSI, BUSBRA).
            img_size: Output image size (SAM input size, typically 1024).
            transunet_img_size: TransUNet resolution for resolution path matching.
            augmentor: Custom MaskAugmentor instance. If None, creates from preset.
            augmentor_preset: Preset name if creating augmentor ('default', 'mild', 'severe', etc.)
            soft_mask_prob: Probability of converting to soft mask.
            soft_mask_temperature: Temperature range for soft mask conversion.
            split_ratio: Train/val split ratio.
            is_train: Whether this is training set.
            seed: Random seed.
            return_augmentation_info: Whether to return augmentation details.
            use_fast_soft_mask: Use fast Gaussian blur instead of distance transform (~10x faster).
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.transunet_img_size = transunet_img_size
        self.is_train = is_train
        self.return_augmentation_info = return_augmentation_info

        # Setup paths
        self.base_dir = os.path.join(data_root, dataset_name, 'train')
        self.image_dir = os.path.join(self.base_dir, 'images')
        self.mask_dir = os.path.join(self.base_dir, 'masks')

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

        # Create or use provided augmentor
        if augmentor is not None:
            self.augmentor = augmentor
        else:
            self.augmentor = create_augmentor(
                preset=augmentor_preset,
                soft_mask_prob=soft_mask_prob,
                soft_mask_temperature=soft_mask_temperature,
                use_fast_soft_mask=use_fast_soft_mask,
            )

        # SAM normalization parameters
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        print(f"Loaded {len(self.samples)} {'train' if is_train else 'val'} samples from {dataset_name}")
        print(f"  Augmentor preset: {augmentor_preset}")
        print(f"  Resolution path: {transunet_img_size}x{transunet_img_size} -> {img_size}x{img_size}")

    def _load_samples(self) -> List[str]:
        """Load sample names from image directory."""
        samples = []
        for f in os.listdir(self.image_dir):
            if f.endswith(('.png', '.jpg', '.jpeg', '.npy')):
                name = os.path.splitext(f)[0]
                # Check mask exists
                mask_path = os.path.join(self.mask_dir, f"{name}.png")
                if os.path.exists(mask_path):
                    samples.append(name)
        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]

        # Load image
        image_path = os.path.join(self.image_dir, f"{name}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f"{name}.npy")

        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))

        # Load GT mask
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        gt_mask = np.array(Image.open(mask_path).convert('L'))

        # Normalize to [0, 1]
        if gt_mask.max() > 1:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        else:
            gt_mask = gt_mask.astype(np.float32)

        # Skip empty masks
        if gt_mask.sum() < 10:
            # Return GT as coarse for empty masks
            coarse_mask = gt_mask.copy()
            aug_info = {'error_type': 'empty_gt', 'dice': 1.0}
        else:
            # Apply on-the-fly augmentation
            coarse_mask, aug_info = self.augmentor(gt_mask)

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

        # Resolution path matching Phase 3:
        # TransUNet outputs at transunet_img_size, then upscaled to img_size
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
            # Direct resize
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

        result = {
            'image': image,
            'label': gt_mask,
            'coarse_mask': coarse_mask,
            'name': name,
            'original_size': original_size,
        }

        if self.return_augmentation_info:
            result['dice'] = aug_info.get('dice', 1.0)
            result['error_type'] = aug_info.get('error_type', 'unknown')
            # Convert list to string for batching compatibility
            secondary = aug_info.get('secondary', [])
            result['augmentations'] = ','.join(secondary) if secondary else ''

        return result


class MultiDatasetOnlineAugmented(Dataset):
    """
    Combined dataset from multiple sources with online augmentation.
    """

    def __init__(
        self,
        data_root: str,
        dataset_names: List[str],
        img_size: int = 1024,
        transunet_img_size: int = 224,
        augmentor: Optional[MaskAugmentor] = None,
        augmentor_preset: str = 'default',
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_root: Root directory.
            dataset_names: List of dataset names to combine.
            Other args same as OnlineAugmentedDataset.
        """
        self.datasets = []
        self.cumulative_lengths = [0]

        # Share augmentor across datasets
        if augmentor is None:
            augmentor = create_augmentor(preset=augmentor_preset)

        for ds_name in dataset_names:
            ds = OnlineAugmentedDataset(
                data_root=data_root,
                dataset_name=ds_name,
                img_size=img_size,
                transunet_img_size=transunet_img_size,
                augmentor=augmentor,
                split_ratio=split_ratio,
                is_train=is_train,
                seed=seed,
            )
            self.datasets.append(ds)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

        self.total_length = self.cumulative_lengths[-1]
        print(f"Combined dataset: {self.total_length} samples from {len(dataset_names)} datasets")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                local_idx = idx - start
                return self.datasets[i][local_idx]
        raise IndexError(f"Index {idx} out of range")


def get_online_augmented_dataloaders(
    data_root: str,
    dataset_names: Union[str, List[str]],
    batch_size: int = 4,
    img_size: int = 1024,
    transunet_img_size: int = 224,
    augmentor_preset: str = 'default',
    num_workers: int = 4,
    split_ratio: float = 0.9,
    seed: int = 42,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    use_fast_soft_mask: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders with online augmentation.

    Args:
        data_root: Root directory.
        dataset_names: Single dataset name or list of names to combine.
        batch_size: Batch size.
        img_size: SAM input size (typically 1024).
        transunet_img_size: TransUNet resolution for resolution path matching.
        augmentor_preset: Augmentation preset ('default', 'mild', 'severe', etc.)
        num_workers: Data loading workers.
        split_ratio: Train/val split.
        seed: Random seed.
        persistent_workers: Keep workers alive between epochs (faster).
        prefetch_factor: Number of batches to prefetch per worker.
        use_fast_soft_mask: Use fast Gaussian blur instead of distance transform (~10x faster).

    Returns:
        train_loader, val_loader
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # Create shared augmentor
    augmentor = create_augmentor(preset=augmentor_preset, use_fast_soft_mask=use_fast_soft_mask)

    if len(dataset_names) == 1:
        train_dataset = OnlineAugmentedDataset(
            data_root=data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            augmentor=augmentor,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = OnlineAugmentedDataset(
            data_root=data_root,
            dataset_name=dataset_names[0],
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            augmentor=augmentor,
            split_ratio=split_ratio,
            is_train=False,
            seed=seed,
        )
    else:
        train_dataset = MultiDatasetOnlineAugmented(
            data_root=data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            augmentor=augmentor,
            split_ratio=split_ratio,
            is_train=True,
            seed=seed,
        )
        val_dataset = MultiDatasetOnlineAugmented(
            data_root=data_root,
            dataset_names=dataset_names,
            img_size=img_size,
            transunet_img_size=transunet_img_size,
            augmentor=augmentor,
            split_ratio=split_ratio,
            is_train=False,
            seed=seed,
        )

    # Use persistent workers and prefetch for faster loading
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
