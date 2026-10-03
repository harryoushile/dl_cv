"""Lesson 13: a tiny Vision Transformer for 32x32 images."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import run_shapes
from common.train_cifar import fit


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=128):
        super().__init__()
        self.num_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    def __init__(self, num_classes=10, dim=128, depth=6):
        super().__init__()
        self.patch = PatchEmbed(dim=dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.patch.num_patches + 1, dim))
        self.blocks = nn.Sequential(*[EncoderBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


def main():
    parser = argparse.ArgumentParser(description="Lesson 13 ViT-Tiny")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    model = ViTTiny()
    if args.mode == "shapes":
        run_shapes(model, torch.randn(2, 3, 32, 32))
    else:
        fit(model, epochs=args.epochs)


if __name__ == "__main__":
    main()
