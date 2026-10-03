"""Lesson 11: MobileNetV2 inverted residual blocks for CIFAR-10."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_cifar import fit


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand=6):
        super().__init__()
        hidden = in_ch * expand
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out


class MiniMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            InvertedResidual(16, 24, stride=2),
            InvertedResidual(24, 24, stride=1),
            InvertedResidual(24, 32, stride=2),
            InvertedResidual(32, 32, stride=1),
            InvertedResidual(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(description="Lesson 11 Mini-MobileNetV2")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    model = MiniMobileNetV2()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
