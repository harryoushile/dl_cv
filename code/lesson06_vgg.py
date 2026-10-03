"""Lesson 6: VGG-style 3x3 stack for CIFAR-10.

BatchNorm is not part of the 2014 VGG paper. It is included so this
network can be trained on CIFAR with the shared SGD recipe.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_cifar import fit


class VGG11CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        layers = []
        in_ch = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [
                    nn.Conv2d(in_ch, v, 3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def main():
    parser = argparse.ArgumentParser(description="Lesson 6 VGG11-CIFAR")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    model = VGG11CIFAR()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
