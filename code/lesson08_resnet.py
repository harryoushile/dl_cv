"""Lesson 8-9: CIFAR ResNet-18, a Bottleneck block, and a no-shortcut ablation."""

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


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_shortcut=True):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_shortcut:
            out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10, use_shortcut=True):
        super().__init__()
        self.use_shortcut = use_shortcut
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

    def _make_layer(self, in_ch, out_ch, n, stride):
        blocks = [BasicBlock(in_ch, out_ch, stride, self.use_shortcut)]
        for _ in range(1, n):
            blocks.append(BasicBlock(out_ch, out_ch, 1, self.use_shortcut))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description="Lesson 8-9 ResNet-18 CIFAR")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--no-shortcut",
        action="store_true",
        help="Drop residual adds. Used by the lesson 9 ablation.",
    )
    args = parser.parse_args()

    model = ResNet18CIFAR(use_shortcut=not args.no_shortcut)
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
        block = Bottleneck(256, 64, stride=2)
        with torch.no_grad():
            y = block(torch.randn(2, 256, 32, 32))
        print(f"bottleneck {tuple(y.shape)}")
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
