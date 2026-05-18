"""
CNN Model — GPU Efficiency Profiler Project
Architecture: 3 Conv blocks + 2 FC layers
Designed to be GPU-bottlenecked (intentionally deep) so CPU vs GPU difference is dramatic
"""

import torch
import torch.nn as nn


class CIFAR10_CNN(nn.Module):
    """
    A moderately deep CNN for CIFAR-10.
    Deliberately has enough parameters (~1.2M) to show meaningful GPU speedup.
    """

    def __init__(self, dropout_rate: float = 0.3):
        super(CIFAR10_CNN, self).__init__()

        # --- Block 1: Input 3x32x32 -> 64x16x16 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        # --- Block 2: 64x16x16 -> 128x8x8 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )

        # --- Block 3: 128x8x8 -> 256x4x4 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CIFAR10_CNN()
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {count_parameters(model):,}")
