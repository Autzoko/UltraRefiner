"""
Unified dataset module for breast ultrasound segmentation.
Supports: BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM datasets.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from scipy.ndimage import zoom, rotate
from scipy import ndimage
import cv2


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

        # Resize to output size
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

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
    Supports various dataset formats: BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM.
    """
    def __init__(
        self,
        data_root,
        dataset_name,
        split='train',
        transform=None,
        img_size=224,
        return_original=False
    ):
        """
        Args:
            data_root: Root directory containing all datasets
            dataset_name: Name of the dataset (BUSI, BUSBRA, BUS, BUS_UC, BUS_UCLM)
            split: 'train', 'val', or 'test'
            transform: Data augmentation transform
            img_size: Target image size
            return_original: Whether to return original image for SAM processing
        """
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.img_size = img_size
        self.return_original = return_original

        self.image_list, self.mask_list = self._load_data_list()
        print(f"Loaded {len(self.image_list)} samples from {dataset_name} ({split})")

    def _load_data_list(self):
        """Load image and mask paths based on dataset format."""
        dataset_path = os.path.join(self.data_root, self.dataset_name)

        # Try different common dataset structures
        image_list = []
        mask_list = []

        # Structure 1: dataset/split/images and dataset/split/masks
        img_dir = os.path.join(dataset_path, self.split, 'images')
        mask_dir = os.path.join(dataset_path, self.split, 'masks')

        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(img_dir, img_name)
                    # Try different mask naming conventions
                    mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                    mask_name_candidates = [
                        mask_name,
                        mask_name.replace('.png', '_mask.png'),
                        img_name.split('.')[0] + '_mask.png',
                        img_name.split('.')[0] + '_segmentation.png',
                    ]
                    for mn in mask_name_candidates:
                        mask_path = os.path.join(mask_dir, mn)
                        if os.path.exists(mask_path):
                            image_list.append(img_path)
                            mask_list.append(mask_path)
                            break
            return image_list, mask_list

        # Structure 2: dataset/images and dataset/masks with split file
        img_dir = os.path.join(dataset_path, 'images')
        mask_dir = os.path.join(dataset_path, 'masks')
        split_file = os.path.join(dataset_path, f'{self.split}.txt')

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                names = [line.strip() for line in f.readlines()]
            for name in names:
                # Find image
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    img_path = os.path.join(img_dir, name + ext)
                    if os.path.exists(img_path):
                        # Find mask
                        for mask_ext in ['.png', '_mask.png', '_segmentation.png']:
                            mask_path = os.path.join(mask_dir, name + mask_ext)
                            if os.path.exists(mask_path):
                                image_list.append(img_path)
                                mask_list.append(mask_path)
                                break
                        break
            return image_list, mask_list

        # Structure 3: BUSI format - dataset/benign, dataset/malignant, dataset/normal
        if self.dataset_name == 'BUSI':
            for category in ['benign', 'malignant']:
                cat_dir = os.path.join(dataset_path, category)
                if os.path.exists(cat_dir):
                    for f in sorted(os.listdir(cat_dir)):
                        if f.endswith(('.png', '.jpg')) and '_mask' not in f:
                            img_path = os.path.join(cat_dir, f)
                            mask_name = f.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
                            mask_path = os.path.join(cat_dir, mask_name)
                            if os.path.exists(mask_path):
                                image_list.append(img_path)
                                mask_list.append(mask_path)

            # Split data
            n = len(image_list)
            if self.split == 'train':
                image_list = image_list[:int(0.7 * n)]
                mask_list = mask_list[:int(0.7 * n)]
            elif self.split == 'val':
                image_list = image_list[int(0.7 * n):int(0.85 * n)]
                mask_list = mask_list[int(0.7 * n):int(0.85 * n)]
            else:  # test
                image_list = image_list[int(0.85 * n):]
                mask_list = mask_list[int(0.85 * n):]
            return image_list, mask_list

        # Structure 4: Direct image/mask pairs in single directory
        if os.path.exists(dataset_path):
            files = sorted(os.listdir(dataset_path))
            img_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')) and '_mask' not in f and '_seg' not in f]

            for img_name in img_files:
                img_path = os.path.join(dataset_path, img_name)
                base_name = img_name.rsplit('.', 1)[0]
                mask_candidates = [
                    base_name + '_mask.png',
                    base_name + '_segmentation.png',
                    base_name + '_seg.png',
                    base_name + '.png' if not img_name.endswith('.png') else None,
                ]
                for mn in mask_candidates:
                    if mn:
                        mask_path = os.path.join(dataset_path, mn)
                        if os.path.exists(mask_path):
                            image_list.append(img_path)
                            mask_list.append(mask_path)
                            break

            # Split data
            n = len(image_list)
            if self.split == 'train':
                image_list = image_list[:int(0.7 * n)]
                mask_list = mask_list[:int(0.7 * n)]
            elif self.split == 'val':
                image_list = image_list[int(0.7 * n):int(0.85 * n)]
                mask_list = mask_list[int(0.7 * n):int(0.85 * n)]
            else:  # test
                image_list = image_list[int(0.85 * n):]
                mask_list = mask_list[int(0.85 * n):]

        return image_list, mask_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = self.mask_list[idx]

        # Load image
        image = np.array(Image.open(img_path).convert('L'))  # Grayscale
        image = image.astype(np.float32) / 255.0

        # Load mask
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)  # Binary mask

        sample = {'image': image, 'label': mask}

        if self.return_original:
            sample['original_image'] = np.array(Image.open(img_path).convert('RGB'))
            sample['image_path'] = img_path

        if self.transform:
            sample = self.transform(sample)

        sample['name'] = os.path.basename(img_path).rsplit('.', 1)[0]
        sample['dataset'] = self.dataset_name

        return sample


def get_dataloader(
    data_root,
    dataset_name,
    split='train',
    batch_size=8,
    img_size=224,
    num_workers=4,
    transform=None,
    shuffle=None,
    for_sam=False
):
    """
    Get dataloader for a single dataset.

    Args:
        data_root: Root directory containing datasets
        dataset_name: Name of the dataset
        split: 'train', 'val', or 'test'
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
    data_root,
    dataset_names=None,
    split='train',
    batch_size=8,
    img_size=224,
    num_workers=4,
    transform=None,
    shuffle=None,
    for_sam=False
):
    """
    Get combined dataloader for multiple datasets.

    Args:
        data_root: Root directory containing datasets
        dataset_names: List of dataset names (default: all supported datasets)
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of data loading workers
        transform: Custom transform
        shuffle: Whether to shuffle
        for_sam: If True, use SAM-specific transform
    """
    if dataset_names is None:
        dataset_names = ['BUSI', 'BUSBRA', 'BUS', 'BUS_UC', 'BUS_UCLM']

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
