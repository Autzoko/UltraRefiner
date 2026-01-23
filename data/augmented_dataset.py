"""
Dataset for augmented training data with pre-generated coarse masks.

This dataset loads:
- Original images
- Ground truth masks
- Pre-generated coarse masks (simulated segmentation failures)

The coarse masks have controlled Dice score distribution:
- 0.6-0.8 (Poor): 30% of samples
- 0.8-0.9 (Medium): 50% of samples
- 0.9+ (Good): 20% of samples
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Dict


class AugmentedSAMDataset(Dataset):
    """
    Dataset for SAM finetuning with pre-generated augmented coarse masks.

    Directory structure:
        augmented_data/
        └── {dataset}/
            ├── images/          # Original images
            ├── masks/           # Ground truth masks
            ├── coarse_masks/    # Pre-generated coarse masks
            └── metadata.json    # Sample information
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        img_size: int = 1024,
        for_sam: bool = True,
        dice_range: Optional[Tuple[float, float]] = None,
        max_samples: Optional[int] = None,
        split_ratio: float = 0.9,
        is_train: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_root: Root directory containing augmented datasets
            dataset_name: Dataset name (e.g., BUSI)
            img_size: Output image size
            for_sam: Whether to prepare data for SAM (1024x1024, normalized)
            dice_range: Optional (min, max) Dice score filter
            max_samples: Maximum number of samples to use
            split_ratio: Train/val split ratio
            is_train: Whether this is training set
            seed: Random seed for reproducible split
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.for_sam = for_sam
        self.dice_range = dice_range

        # Setup paths
        self.base_dir = os.path.join(data_root, dataset_name)
        self.image_dir = os.path.join(self.base_dir, 'images')
        self.mask_dir = os.path.join(self.base_dir, 'masks')
        self.coarse_dir = os.path.join(self.base_dir, 'coarse_masks')

        # Load metadata
        metadata_path = os.path.join(self.base_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Get sample list
        self.samples = self.metadata['samples']

        # Filter by Dice range if specified
        if dice_range is not None:
            min_dice, max_dice = dice_range
            self.samples = [
                s for s in self.samples
                if min_dice <= s['dice'] <= max_dice
            ]

        # Limit samples if specified
        if max_samples is not None and len(self.samples) > max_samples:
            np.random.seed(seed)
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * split_ratio)

        if is_train:
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        self.samples = [self.samples[i] for i in selected_indices]

        # SAM normalization parameters
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

        print(f"Loaded {len(self.samples)} samples from {dataset_name}")
        if dice_range:
            print(f"  Dice range filter: {dice_range}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        name = sample_info['name']

        # Load image
        image_path = os.path.join(self.image_dir, f"{name}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f"{name}.npy")

        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))

        # Load ground truth mask
        mask_path = os.path.join(self.mask_dir, f"{name}.png")
        gt_mask = np.array(Image.open(mask_path).convert('L'))

        # Load coarse mask
        coarse_path = os.path.join(self.coarse_dir, f"{name}.png")
        coarse_mask = np.array(Image.open(coarse_path).convert('L'))

        # Normalize masks to [0, 1]
        if gt_mask.max() > 1:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        if coarse_mask.max() > 1:
            coarse_mask = coarse_mask.astype(np.float32) / 255.0

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

        # Resize to target size
        original_size = (image.shape[-2], image.shape[-1])

        image = TF.resize(image, [self.img_size, self.img_size])
        gt_mask = TF.resize(
            gt_mask.unsqueeze(0), [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST
        ).squeeze(0)
        coarse_mask = TF.resize(
            coarse_mask.unsqueeze(0), [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.BILINEAR
        ).squeeze(0)

        # SAM preprocessing
        if self.for_sam:
            # Normalize with SAM parameters
            image = (image - self.pixel_mean) / self.pixel_std
        else:
            # Standard normalization
            image = image / 255.0

        return {
            'image': image,
            'label': gt_mask,
            'coarse_mask': coarse_mask,
            'name': name,
            'original_size': original_size,
            'dice': sample_info['dice'],
            'augmentations': sample_info.get('augmentations', []),
        }


def get_augmented_dataloaders(
    data_root: str,
    dataset_name: str,
    batch_size: int = 4,
    img_size: int = 1024,
    num_workers: int = 4,
    dice_range: Optional[Tuple[float, float]] = None,
    max_samples: Optional[int] = None,
    split_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders for augmented data.

    Args:
        data_root: Root directory containing augmented datasets
        dataset_name: Dataset name
        batch_size: Batch size
        img_size: Output image size
        num_workers: Number of data loading workers
        dice_range: Optional Dice score filter
        max_samples: Maximum samples to use
        split_ratio: Train/val split ratio
        seed: Random seed

    Returns:
        train_loader, val_loader
    """
    train_dataset = AugmentedSAMDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        img_size=img_size,
        dice_range=dice_range,
        max_samples=max_samples,
        split_ratio=split_ratio,
        is_train=True,
        seed=seed,
    )

    val_dataset = AugmentedSAMDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        img_size=img_size,
        dice_range=dice_range,
        max_samples=max_samples,
        split_ratio=split_ratio,
        is_train=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class CurriculumAugmentedDataset(AugmentedSAMDataset):
    """
    Curriculum learning variant that progressively increases difficulty.

    Training starts with high Dice samples (easy) and gradually includes
    lower Dice samples (harder) as training progresses.
    """

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        img_size: int = 1024,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        # Load all samples without filtering
        super().__init__(
            data_root=data_root,
            dataset_name=dataset_name,
            img_size=img_size,
            dice_range=None,
            max_samples=max_samples,
            split_ratio=1.0,  # Use all for curriculum
            is_train=True,
            seed=seed,
        )

        # Sort samples by Dice score (high to low for curriculum)
        self.samples = sorted(self.samples, key=lambda x: -x['dice'])

        # Current difficulty level (0.0 = easiest, 1.0 = all samples)
        self.difficulty = 0.0

        # Cache sorted indices
        self._full_samples = self.samples.copy()

    def set_difficulty(self, difficulty: float):
        """
        Set curriculum difficulty level.

        Args:
            difficulty: 0.0-1.0, fraction of samples to include
                       0.0 = only highest Dice samples
                       1.0 = all samples
        """
        self.difficulty = np.clip(difficulty, 0.0, 1.0)

        # Include top (1 - difficulty) to 100% of samples
        num_samples = int(len(self._full_samples) * self.difficulty)
        num_samples = max(100, num_samples)  # Minimum 100 samples

        self.samples = self._full_samples[:num_samples]

    def update_difficulty(self, epoch: int, total_epochs: int):
        """
        Automatically update difficulty based on training progress.

        Uses linear warmup: start with easy samples, include all by mid-training.
        """
        # Linear curriculum: reach full difficulty at 50% of training
        warmup_epochs = total_epochs // 2
        if epoch < warmup_epochs:
            difficulty = 0.2 + 0.8 * (epoch / warmup_epochs)
        else:
            difficulty = 1.0

        self.set_difficulty(difficulty)
