"""Lesson 15: U-Net trained on synthetic circle masks."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.trace import count_params, run_shapes


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        y = self.up2(x3)
        y = self.conv2(torch.cat([y, x2], dim=1))
        y = self.up1(y)
        y = self.conv1(torch.cat([y, x1], dim=1))
        return self.out(y)


def make_circle_batch(n=8, size=128):
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    images = torch.zeros(n, 1, size, size)
    labels = torch.zeros(n, size, size, dtype=torch.long)
    for i in range(n):
        cy = torch.randint(32, 96, (1,)).item()
        cx = torch.randint(32, 96, (1,)).item()
        radius = torch.randint(12, 28, (1,)).item()
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        labels[i] = mask.long()
        images[i, 0] = mask.float()
    return images, labels


def fit_synthetic(model, steps=50, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device={device}  params={count_params(model):,}")
    model.train()
    for step in range(1, steps + 1):
        x, y = make_circle_batch()
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if step == 1 or step % 10 == 0:
            pred = model(x).argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f"step {step:03d}  loss {loss.item():.3f}  pixel_acc {acc:.3f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Lesson 15 U-Net")
    parser.add_argument("--mode", choices=["shapes", "train"], default="shapes")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    model = UNet()
    x, _ = make_circle_batch(n=2)
    if args.mode == "shapes":
        y = run_shapes(model, x)
        if tuple(y.shape) != (2, 2, 128, 128):
            raise SystemExit(f"expected (2, 2, 128, 128), got {tuple(y.shape)}")
    else:
        fit_synthetic(model, steps=args.steps)


if __name__ == "__main__":
    main()
