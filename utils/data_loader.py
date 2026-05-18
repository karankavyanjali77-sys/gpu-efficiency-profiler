"""
Data loading utility — CIFAR-10
Handles download, normalization, and DataLoader creation.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel means and stds (precomputed on full training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_transforms(augment: bool = True):
    """
    augment=True  → training transforms (random flip + crop)
    augment=False → validation/test transforms (just normalize)
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) for CIFAR-10.
    Downloads dataset automatically on first run.
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=get_transforms(augment=True)
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=get_transforms(augment=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # Faster CPU->GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
