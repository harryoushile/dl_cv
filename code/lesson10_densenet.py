"""Lesson 10: a small DenseNet with growth rate and a transition layer."""

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


class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x), inplace=True))
        return torch.cat([x, out], dim=1)


class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class MiniDenseNet(nn.Module):
    def __init__(self, growth=12, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        ch = 32
        self.block1 = nn.Sequential(
            *[DenseLayer(ch + i * growth, growth) for i in range(4)]
        )
        ch = ch + 4 * growth
        self.trans = Transition(ch, ch // 2)
        ch = ch // 2
        self.block2 = nn.Sequential(
            *[DenseLayer(ch + i * growth, growth) for i in range(4)]
        )
        ch = ch + 4 * growth
        self.head = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans(x)
        x = self.block2(x)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description="Lesson 10 Mini-DenseNet")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    model = MiniDenseNet()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
