"""Lesson 7: one Inception module and a small CIFAR GoogLeNet."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_cifar import fit


class Inception(nn.Module):
    def __init__(self, in_ch, ch1, ch3_red, ch3, ch5_red, ch5, pool_proj):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, ch1, 1), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3_red, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3_red, ch3, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5_red, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5_red, ch5, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, 1),
            nn.ReLU(inplace=True),
        )
        self.out_ch = ch1 + ch3 + ch5 + pool_proj

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class MiniGoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.inc1 = Inception(64, 32, 32, 48, 8, 16, 16)
        self.inc2 = Inception(self.inc1.out_ch, 48, 48, 64, 12, 24, 16)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.inc2.out_ch, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool(x)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description="Lesson 7 Mini-GoogLeNet")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    model = MiniGoogLeNet()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
