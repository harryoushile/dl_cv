"""Lesson 2: flatten an image and classify with one linear layer."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_mnist import fit


class LinearMNIST(nn.Module):
    """MNIST 1x28x28 flattened to 784, then 10 logits."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(1 * 28 * 28, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def main():
    parser = argparse.ArgumentParser(description="Lesson 2 linear classifier")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    model = LinearMNIST()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 1, 28, 28))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
