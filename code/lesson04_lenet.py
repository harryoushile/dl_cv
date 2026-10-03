"""Lesson 4: LeNet-5 resized for 28x28 MNIST."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_mnist import fit


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def main():
    parser = argparse.ArgumentParser(description="Lesson 4 LeNet-5")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    model = LeNet5()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 1, 28, 28))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
