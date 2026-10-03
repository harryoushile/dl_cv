"""Lesson 14: a two-stage ConvNeXt for CIFAR-10."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_cifar import fit


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pw2(F.gelu(self.pw1(self.norm(x))))
        x = x.permute(0, 3, 1, 2)
        return shortcut + x


class ConvNeXtTinyCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=2, stride=2)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(64) for _ in range(2)])
        self.down = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(128) for _ in range(2)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down(x))
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description="Lesson 14 ConvNeXt-Tiny CIFAR")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    model = ConvNeXtTinyCIFAR()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
