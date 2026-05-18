"""
Data Utilities
--------------
Handles CIFAR-10 loading with proper augmentation for training
and clean transforms for validation/benchmarking.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# CIFAR-10 class names (for dashboard labels)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_transforms(augment: bool = True):
    """
    Returns transforms for training (with augmentation) or eval (clean).

    Why augmentation matters:
    - RandomHorizontalFlip: a cat flipped is still a cat
    - RandomCrop: teaches position invariance
    - ColorJitter: handles lighting variation
    - Normalize: zero-centres pixels per channel (ImageNet-style for CIFAR)
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std  = (0.2023, 0.1994, 0.2010)

    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ])


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str = "./data"
):
    """
    Downloads CIFAR-10 (if not cached) and returns train + val DataLoaders.

    Args:
        batch_size:   Images per batch. Larger = faster GPU utilisation.
        num_workers:  Parallel data loading workers. 2 is safe for Colab.
        data_dir:     Where to cache the dataset.

    Returns:
        (train_loader, val_loader, class_names)
    """
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(augment=True)
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(augment=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # Speeds up CPU→GPU data transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, CLASSES


def get_dataset_stats():
    """Returns a summary dict of dataset statistics."""
    return {
        "train_samples": 50_000,
        "val_samples":   10_000,
        "classes":       10,
        "image_size":    "32×32 RGB",
        "class_names":   CLASSES,
    }
