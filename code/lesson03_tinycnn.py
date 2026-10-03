"""Lesson 3: two conv-pool stages on MNIST."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_mnist import fit


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def main():
    parser = argparse.ArgumentParser(description="Lesson 3 TinyCNN")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    model = TinyCNN()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 1, 28, 28))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
