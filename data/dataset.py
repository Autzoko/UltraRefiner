"""
Unified dataset module for breast ultrasound segmentation.
Supports: BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM, UDIAT datasets.

Uses preprocessed data from dataset/processed/ with fixed train/test splits.
Supports K-fold cross-validation within the training set.
"""
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from PIL import Image
from scipy.ndimage import zoom, rotate
from scipy import ndimage
import cv2
from typing import List, Tuple, Optional, Dict


# Supported datasets
# Training datasets (used for train/test splits)
SUPPORTED_DATASETS = ['BUSI', 'BUSBRA', 'BUS', 'BUS_UC', 'BUS_UCLM']

# Unseen test-only datasets (for cross-dataset generalization evaluation)
UNSEEN_DATASETS = ['UDIAT']

# All datasets
ALL_DATASETS = SUPPORTED_DATASETS + UNSEEN_DATASETS


class RandomGenerator:
    """
    Random data augmentation generator for TransUNet training.
    """
    def __init__(self, output_size, random_rotate=True, random_flip=True):
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_flip = random_flip

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Random rotation
        if self.random_rotate and random.random() > 0.5:
            angle = random.uniform(-20, 20)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            label = ndimage.rotate(label, angle, order=0, reshape=False)

        # Random flip
        if self.random_flip and random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        # Resize to exact output size using cv2 (more reliable than scipy.zoom)
        h, w = image.shape[:2]
        target_h, target_w = self.output_size[0], self.output_size[1]
        if h != target_h or w != target_w:
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Ensure exact output size (safeguard against any floating point issues)
        image = image[:target_h, :target_w]
        label = label[:target_h, :target_w]

        # Convert to tensor
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label}
        return sample


class SAMRandomGenerator:
    """
    Random data augmentation generator for SAM training.
    Outputs images at SAM resolution (1024x1024) with proper normalization.
    """
    def __init__(self, output_size=1024, random_rotate=True, random_flip=True):
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_flip = random_flip

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Random rotation
        if self.random_rotate and random.random() > 0.5:
            angle = random.uniform(-20, 20)
            image = ndimage.rotate(image, angle, order=0, reshape=False, mode='constant', cval=0)
            label = ndimage.rotate(label, angle, order=0, reshape=False, mode='constant', cval=0)

        # Random flip
        if self.random_flip and random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        # Resize with aspect ratio preservation
        h, w = image.shape[:2] if len(image.shape) > 2 else image.shape
        scale = self.output_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        if len(image.shape) == 2:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        label = cv2.resize(label.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad to square
        pad_h = self.output_size - new_h
        pad_w = self.output_size - new_w

        if len(image.shape) == 2:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

        label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        # Convert to tensor
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # HWC to CHW
        label = torch.from_numpy(label.astype(np.float32))

        sample = {
            'image': image,
            'label': label,
            'original_size': (h, w),
            'input_size': (new_h, new_w)
        }
        return sample


class BreastUltrasoundDataset(Dataset):
    """
    Unified dataset for breast ultrasound segmentation.
    Uses preprocessed data from dataset/processed/{dataset_name}/{split}/
    """
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        split: str = 'train',
        transform=None,
        img_size: int = 224,
        return_original: bool = False,
        indices: Optional[List[int]] = None
    ):
        """
        Args:
            data_root: Root directory containing processed datasets (dataset/processed)
            dataset_name: Name of the dataset (BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM)
            split: 'train' or 'test'
            transform: Data augmentation transform
            img_size: Target image size
            return_original: Whether to return original image for SAM processing
            indices: Optional list of indices for K-fold cross-validation
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.return_original = return_original

        self.image_list, self.mask_list = self._load_data_list()

        # Apply indices for K-fold cross-validation
        if indices is not None:
            self.image_list = [self.image_list[i] for i in indices]
            self.mask_list = [self.mask_list[i] for i in indices]

        print(f"Loaded {len(self.image_list)} samples from {dataset_name} ({split})")

    def _load_data_list(self) -> Tuple[List[str], List[str]]:
        """Load image and mask paths from preprocessed directory structure."""
        # Path to preprocessed data: data_root/{dataset_name}/{split}/images|masks
        img_dir = os.path.join(self.data_root, self.dataset_name, self.split, 'images')
        mask_dir = os.path.join(self.data_root, self.dataset_name, self.split, 'masks')

        if not os.path.exists(img_dir):
            raise ValueError(f"Image directory not found: {img_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"Mask directory not found: {mask_dir}")

        image_list = []
        mask_list = []

        for img_name in sorted(os.listdir(img_dir)):
            if img_name.endswith('.png'):
                img_path = os.path.join(img_dir, img_name)
                mask_path = os.path.join(mask_dir, img_name)

                if os.path.exists(mask_path):
                    image_list.append(img_path)
                    mask_list.append(mask_path)
                else:
                    print(f"Warning: Mask not found for {img_name}")

        return image_list, mask_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = self.mask_list[idx]

        # Load image (grayscale)
        image = np.array(Image.open(img_path).convert('L'))
        image = image.astype(np.float32) / 255.0

        # Load mask (binary)
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)

        sample = {'image': image, 'label': mask}

        if self.return_original:
            sample['original_image'] = np.array(Image.open(img_path).convert('RGB'))
            sample['image_path'] = img_path

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = os.path.basename(img_path).rsplit('.', 1)[0]
        sample['dataset'] = self.dataset_name

        return sample


class KFoldCrossValidator:
    """
    K-Fold cross-validation utility for training within the training set.
    """
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        n_splits: int = 5,
        seed: int = 42
    ):
        """
        Args:
            data_root: Root directory containing processed datasets
            dataset_name: Name of the dataset
            n_splits: Number of folds
            seed: Random seed for reproducibility
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.seed = seed

        # Load training data indices
        img_dir = os.path.join(data_root, dataset_name, 'train', 'images')
        self.n_samples = len([f for f in os.listdir(img_dir) if f.endswith('.png')])

        # Create fold indices
        self.fold_indices = self._create_folds()

    def _create_folds(self) -> List[Tuple[List[int], List[int]]]:
        """Create fold indices for K-fold cross-validation."""
        random.seed(self.seed)
        indices = list(range(self.n_samples))
        random.shuffle(indices)

        fold_size = self.n_samples // self.n_splits
        folds = []

        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else self.n_samples

            val_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]

            folds.append((train_indices, val_indices))

        return folds

    def get_fold(
        self,
        fold_idx: int,
        transform_train=None,
        transform_val=None,
        img_size: int = 224,
        return_original: bool = False
    ) -> Tuple[Dataset, Dataset]:
        """
        Get train and validation datasets for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_splits-1)
            transform_train: Transform for training data
            transform_val: Transform for validation data
            img_size: Target image size
            return_original: Whether to return original images

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if fold_idx >= self.n_splits:
            raise ValueError(f"Fold index {fold_idx} >= n_splits {self.n_splits}")

        train_indices, val_indices = self.fold_indices[fold_idx]

        train_dataset = BreastUltrasoundDataset(
            data_root=self.data_root,
            dataset_name=self.dataset_name,
            split='train',
            transform=transform_train,
            img_size=img_size,
            return_original=return_original,
            indices=train_indices
        )

        val_dataset = BreastUltrasoundDataset(
            data_root=self.data_root,
            dataset_name=self.dataset_name,
            split='train',  # Validation comes from training set
            transform=transform_val,
            img_size=img_size,
            return_original=return_original,
            indices=val_indices
        )

        return train_dataset, val_dataset


class CombinedKFoldCrossValidator:
    """
    K-Fold cross-validation for combined datasets.
    """
    def __init__(
        self,
        data_root: str,
        dataset_names: Optional[List[str]] = None,
        n_splits: int = 5,
        seed: int = 42
    ):
        """
        Args:
            data_root: Root directory containing processed datasets
            dataset_names: List of dataset names
            n_splits: Number of folds
            seed: Random seed for reproducibility
        """
        self.data_root = data_root
        self.dataset_names = dataset_names or SUPPORTED_DATASETS
        self.n_splits = n_splits
        self.seed = seed

        # Create per-dataset validators
        self.validators = {}
        for ds_name in self.dataset_names:
            ds_path = os.path.join(data_root, ds_name, 'train')
            if os.path.exists(ds_path):
                self.validators[ds_name] = KFoldCrossValidator(
                    data_root=data_root,
                    dataset_name=ds_name,
                    n_splits=n_splits,
                    seed=seed
                )

    def get_fold(
        self,
        fold_idx: int,
        transform_train=None,
        transform_val=None,
        img_size: int = 224,
        return_original: bool = False
    ) -> Tuple[Dataset, Dataset]:
        """
        Get combined train and validation datasets for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_splits-1)
            transform_train: Transform for training data
            transform_val: Transform for validation data
            img_size: Target image size
            return_original: Whether to return original images

        Returns:
            Tuple of (combined_train_dataset, combined_val_dataset)
        """
        train_datasets = []
        val_datasets = []

        for ds_name, validator in self.validators.items():
            train_ds, val_ds = validator.get_fold(
                fold_idx=fold_idx,
                transform_train=transform_train,
                transform_val=transform_val,
                img_size=img_size,
                return_original=return_original
            )
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)

        combined_train = ConcatDataset(train_datasets)
        combined_val = ConcatDataset(val_datasets)

        print(f"Fold {fold_idx}: Train={len(combined_train)}, Val={len(combined_val)}")

        return combined_train, combined_val


def get_dataloader(
    data_root: str,
    dataset_name: str,
    split: str = 'train',
    batch_size: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
    transform=None,
    shuffle: Optional[bool] = None,
    for_sam: bool = False
) -> DataLoader:
    """
    Get dataloader for a single dataset.

    Args:
        data_root: Root directory containing processed datasets
        dataset_name: Name of the dataset
        split: 'train' or 'test'
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of data loading workers
        transform: Custom transform (if None, default is used)
        shuffle: Whether to shuffle (default: True for train, False otherwise)
        for_sam: If True, use SAM-specific transform
    """
    if shuffle is None:
        shuffle = (split == 'train')

    if transform is None:
        if for_sam:
            transform = SAMRandomGenerator(
                output_size=1024,
                random_rotate=(split == 'train'),
                random_flip=(split == 'train')
            )
        else:
            transform = RandomGenerator(
                output_size=[img_size, img_size],
                random_rotate=(split == 'train'),
                random_flip=(split == 'train')
            )

    dataset = BreastUltrasoundDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split=split,
        transform=transform,
        img_size=img_size,
        return_original=for_sam
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


def get_combined_dataloader(
    data_root: str,
    dataset_names: Optional[List[str]] = None,
    split: str = 'train',
    batch_size: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
    transform=None,
    shuffle: Optional[bool] = None,
    for_sam: bool = False
) -> DataLoader:
    """
    Get combined dataloader for multiple datasets.

    Args:
        data_root: Root directory containing processed datasets
        dataset_names: List of dataset names (default: all supported datasets)
        split: 'train' or 'test'
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of data loading workers
        transform: Custom transform
        shuffle: Whether to shuffle
        for_sam: If True, use SAM-specific transform
    """
    if dataset_names is None:
        dataset_names = SUPPORTED_DATASETS

    if shuffle is None:
        shuffle = (split == 'train')

    if transform is None:
        if for_sam:
            transform = SAMRandomGenerator(
                output_size=1024,
                random_rotate=(split == 'train'),
                random_flip=(split == 'train')
            )
        else:
            transform = RandomGenerator(
                output_size=[img_size, img_size],
                random_rotate=(split == 'train'),
                random_flip=(split == 'train')
            )

    datasets = []
    for ds_name in dataset_names:
        ds_path = os.path.join(data_root, ds_name)
        if os.path.exists(ds_path):
            ds = BreastUltrasoundDataset(
                data_root=data_root,
                dataset_name=ds_name,
                split=split,
                transform=transform,
                img_size=img_size,
                return_original=for_sam
            )
            if len(ds) > 0:
                datasets.append(ds)
        else:
            print(f"Warning: Dataset {ds_name} not found at {ds_path}")

    if len(datasets) == 0:
        raise ValueError(f"No valid datasets found in {data_root}")

    combined_dataset = ConcatDataset(datasets)
    print(f"Combined dataset size: {len(combined_dataset)} samples")

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


def get_kfold_dataloaders(
    data_root: str,
    dataset_name: str,
    fold_idx: int,
    n_splits: int = 5,
    batch_size: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
    for_sam: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders for a specific K-fold.

    Args:
        data_root: Root directory containing processed datasets
        dataset_name: Name of the dataset
        fold_idx: Fold index (0 to n_splits-1)
        n_splits: Number of folds
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of workers
        for_sam: If True, use SAM-specific transform
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    if for_sam:
        transform_train = SAMRandomGenerator(output_size=1024, random_rotate=True, random_flip=True)
        transform_val = SAMRandomGenerator(output_size=1024, random_rotate=False, random_flip=False)
    else:
        transform_train = RandomGenerator(output_size=[img_size, img_size], random_rotate=True, random_flip=True)
        transform_val = RandomGenerator(output_size=[img_size, img_size], random_rotate=False, random_flip=False)

    # Create validator and get fold datasets
    validator = KFoldCrossValidator(
        data_root=data_root,
        dataset_name=dataset_name,
        n_splits=n_splits,
        seed=seed
    )

    train_dataset, val_dataset = validator.get_fold(
        fold_idx=fold_idx,
        transform_train=transform_train,
        transform_val=transform_val,
        img_size=img_size,
        return_original=for_sam
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


def get_combined_kfold_dataloaders(
    data_root: str,
    dataset_names: Optional[List[str]] = None,
    fold_idx: int = 0,
    n_splits: int = 5,
    batch_size: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
    for_sam: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get combined train and validation dataloaders for a specific K-fold across all datasets.

    Args:
        data_root: Root directory containing processed datasets
        dataset_names: List of dataset names
        fold_idx: Fold index (0 to n_splits-1)
        n_splits: Number of folds
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of workers
        for_sam: If True, use SAM-specific transform
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if dataset_names is None:
        dataset_names = SUPPORTED_DATASETS

    # Create transforms
    if for_sam:
        transform_train = SAMRandomGenerator(output_size=1024, random_rotate=True, random_flip=True)
        transform_val = SAMRandomGenerator(output_size=1024, random_rotate=False, random_flip=False)
    else:
        transform_train = RandomGenerator(output_size=[img_size, img_size], random_rotate=True, random_flip=True)
        transform_val = RandomGenerator(output_size=[img_size, img_size], random_rotate=False, random_flip=False)

    # Create combined validator
    validator = CombinedKFoldCrossValidator(
        data_root=data_root,
        dataset_names=dataset_names,
        n_splits=n_splits,
        seed=seed
    )

    train_dataset, val_dataset = validator.get_fold(
        fold_idx=fold_idx,
        transform_train=transform_train,
        transform_val=transform_val,
        img_size=img_size,
        return_original=for_sam
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


def get_test_dataloader(
    data_root: str,
    dataset_name: str,
    batch_size: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
    for_sam: bool = False
) -> DataLoader:
    """
    Get test dataloader for final evaluation.

    Args:
        data_root: Root directory containing processed datasets
        dataset_name: Name of the dataset
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of workers
        for_sam: If True, use SAM-specific transform

    Returns:
        Test dataloader
    """
    if for_sam:
        transform = SAMRandomGenerator(output_size=1024, random_rotate=False, random_flip=False)
    else:
        transform = RandomGenerator(output_size=[img_size, img_size], random_rotate=False, random_flip=False)

    dataset = BreastUltrasoundDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split='test',
        transform=transform,
        img_size=img_size,
        return_original=for_sam
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader


def get_combined_test_dataloader(
    data_root: str,
    dataset_names: Optional[List[str]] = None,
    batch_size: int = 8,
    img_size: int = 224,
    num_workers: int = 4,
    for_sam: bool = False
) -> DataLoader:
    """
    Get combined test dataloader for final evaluation across all datasets.

    Args:
        data_root: Root directory containing processed datasets
        dataset_names: List of dataset names
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of workers
        for_sam: If True, use SAM-specific transform

    Returns:
        Combined test dataloader
    """
    if dataset_names is None:
        dataset_names = SUPPORTED_DATASETS

    if for_sam:
        transform = SAMRandomGenerator(output_size=1024, random_rotate=False, random_flip=False)
    else:
        transform = RandomGenerator(output_size=[img_size, img_size], random_rotate=False, random_flip=False)

    datasets = []
    for ds_name in dataset_names:
        ds_path = os.path.join(data_root, ds_name)
        if os.path.exists(ds_path):
            ds = BreastUltrasoundDataset(
                data_root=data_root,
                dataset_name=ds_name,
                split='test',
                transform=transform,
                img_size=img_size,
                return_original=for_sam
            )
            if len(ds) > 0:
                datasets.append(ds)

    if len(datasets) == 0:
        raise ValueError(f"No valid test datasets found in {data_root}")

    combined_dataset = ConcatDataset(datasets)
    print(f"Combined test dataset size: {len(combined_dataset)} samples")

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader


def load_split_info(data_root: str, dataset_name: Optional[str] = None) -> Dict:
    """
    Load split information from JSON file.

    Args:
        data_root: Root directory containing processed datasets
        dataset_name: Name of the dataset (if None, load combined info)

    Returns:
        Dictionary with split information
    """
    if dataset_name:
        split_file = os.path.join(data_root, dataset_name, 'split_info.json')
    else:
        split_file = os.path.join(data_root, 'combined_split_info.json')

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split info file not found: {split_file}")

    with open(split_file, 'r') as f:
        return json.load(f)
