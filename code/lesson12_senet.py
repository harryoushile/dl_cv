"""Lesson 12: squeeze-and-excitation on the CIFAR ResNet-18."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import by module filename. The directory is named code, which is also a
# standard-library module, so "import code.lesson08_resnet" is unreliable
# after torch has been imported.
sys.path.insert(0, str(ROOT / "code"))

from lesson08_resnet import BasicBlock
from common.trace import run_shapes
from common.train_cifar import fit


class SE(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEBasicBlock(nn.Module):
    """Channel reweighting after the residual block.

    The SE-ResNet paper applies SE on the residual branch before the add.
    This teaching block rescales the block output so the change is local.
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = BasicBlock(in_ch, out_ch, stride)
        self.se = SE(out_ch)

    def forward(self, x):
        return self.se(self.block(x))


class SEResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, n=2, stride=1)
        self.layer2 = self._make_layer(64, 128, n=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n=2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_layer(in_ch, out_ch, n, stride):
        blocks = [SEBasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            blocks.append(SEBasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description="Lesson 12 SE-ResNet-18")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    model = SEResNet18CIFAR()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
