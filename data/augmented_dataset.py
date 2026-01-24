"""
Dataset for augmented training data with pre-generated coarse masks.

This dataset loads:
- Original images
- Ground truth masks
- Pre-generated coarse masks (simulated segmentation failures)

The coarse masks have controlled Dice score distribution (no Dice=1.0):
- Dice 0.9-0.99 (Good): 25% of samples - minor artifacts
- Dice 0.8-0.9 (Medium): 40% of samples - moderate errors
- Dice 0.6-0.8 (Poor): 35% of samples - severe failures

Coarse Mask Formats:
- Binary (PNG): Legacy format, hard 0/255 masks
- Soft (NPY): Soft probability maps matching TransUNet output distribution
  (RECOMMENDED for Phase 3 E2E training compatibility)
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
        transunet_img_size: int = 224,
    ):
        """
        Args:
            data_root: Root directory containing augmented datasets
            dataset_name: Dataset name (e.g., BUSI)
            img_size: Output image size (SAM input size, typically 1024)
            for_sam: Whether to prepare data for SAM (1024x1024, normalized)
            dice_range: Optional (min, max) Dice score filter
            max_samples: Maximum number of samples to use
            split_ratio: Train/val split ratio
            is_train: Whether this is training set
            seed: Random seed for reproducible split
            transunet_img_size: Intermediate resolution to simulate TransUNet output path.
                               In Phase 3, TransUNet outputs at this resolution before
                               upscaling to SAM input size. Setting this ensures Phase 2
                               has the same resolution path as Phase 3 for consistent
                               prompt distribution. Set to 0 to disable (legacy behavior).
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.for_sam = for_sam
        self.dice_range = dice_range
        self.transunet_img_size = transunet_img_size

        # Setup paths
        self.base_dir = os.path.join(data_root, dataset_name)
        self.image_dir = os.path.join(self.base_dir, 'images')
        self.mask_dir = os.path.join(self.base_dir, 'masks')
        self.coarse_dir = os.path.join(self.base_dir, 'coarse_masks')

        # Load metadata
        metadata_path = os.path.join(self.base_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Check if soft masks are used (NPY format matching TransUNet output distribution)
        # This is CRITICAL for Phase 2 -> Phase 3 compatibility
        self.soft_masks = self.metadata.get('soft_masks', False)

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
        print(f"  Coarse mask format: {'Soft (NPY, TransUNet-like)' if self.soft_masks else 'Binary (PNG)'}")
        if self.transunet_img_size > 0 and self.transunet_img_size < self.img_size:
            print(f"  Resolution path: {self.transunet_img_size}x{self.transunet_img_size} -> {self.img_size}x{self.img_size} (matches Phase 3)")
        else:
            print(f"  Resolution path: original -> {self.img_size}x{self.img_size} (legacy)")
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

        # Load coarse mask (soft NPY or binary PNG depending on metadata)
        if self.soft_masks:
            # Soft probability map (NPY float) - matches TransUNet output distribution
            coarse_path = os.path.join(self.coarse_dir, f"{name}.npy")
            if os.path.exists(coarse_path):
                coarse_mask = np.load(coarse_path).astype(np.float32)
            else:
                # Fallback to PNG if NPY not found (backward compatibility)
                coarse_path = os.path.join(self.coarse_dir, f"{name}.png")
                coarse_mask = np.array(Image.open(coarse_path).convert('L'))
                if coarse_mask.max() > 1:
                    coarse_mask = coarse_mask.astype(np.float32) / 255.0
        else:
            # Binary mask (PNG uint8) - legacy format
            coarse_path = os.path.join(self.coarse_dir, f"{name}.png")
            coarse_mask = np.array(Image.open(coarse_path).convert('L'))
            if coarse_mask.max() > 1:
                coarse_mask = coarse_mask.astype(np.float32) / 255.0
            else:
                coarse_mask = coarse_mask.astype(np.float32)

        # Normalize GT mask to [0, 1]
        if gt_mask.max() > 1:
            gt_mask = gt_mask.astype(np.float32) / 255.0
        else:
            gt_mask = gt_mask.astype(np.float32)

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

        # CRITICAL: To match Phase 3 distribution, resize through TransUNet resolution first
        # In Phase 3: TransUNet outputs at transunet_img_size (224), then upscaled to img_size (1024)
        # This creates additional smoothing that we must replicate in Phase 2
        if self.transunet_img_size > 0 and self.transunet_img_size < self.img_size:
            # Step 1: Resize to TransUNet resolution (simulates TransUNet output)
            image_small = TF.resize(image, [self.transunet_img_size, self.transunet_img_size])
            coarse_mask_small = TF.resize(
                coarse_mask.unsqueeze(0), [self.transunet_img_size, self.transunet_img_size],
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)

            # Step 2: Resize to SAM input size (simulates Phase 3 upscaling)
            image = TF.resize(image_small, [self.img_size, self.img_size])
            coarse_mask = TF.resize(
                coarse_mask_small.unsqueeze(0), [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)
        else:
            # Legacy behavior: direct resize to SAM input size
            image = TF.resize(image, [self.img_size, self.img_size])
            coarse_mask = TF.resize(
                coarse_mask.unsqueeze(0), [self.img_size, self.img_size],
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0)

        # GT mask: resize directly to SAM size (not through TransUNet resolution)
        gt_mask = TF.resize(
            gt_mask.unsqueeze(0), [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST
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
    transunet_img_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders for augmented data.

    Args:
        data_root: Root directory containing augmented datasets
        dataset_name: Dataset name
        batch_size: Batch size
        img_size: Output image size (SAM input size, typically 1024)
        num_workers: Number of data loading workers
        dice_range: Optional Dice score filter
        max_samples: Maximum samples to use
        split_ratio: Train/val split ratio
        seed: Random seed
        transunet_img_size: Intermediate resolution to simulate TransUNet output path.
                           This ensures Phase 2 has the same resolution path as Phase 3.
                           Set to 0 to disable (legacy behavior).

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
        transunet_img_size=transunet_img_size,
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
        transunet_img_size=transunet_img_size,
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
        transunet_img_size: int = 224,
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
            transunet_img_size=transunet_img_size,
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
