"""MNIST loader and a short training loop for lessons 2-4."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.trace import count_params

ROOT = Path(__file__).resolve().parents[1]


def get_loaders(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_root = ROOT / "data"
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    # num_workers=0 avoids Windows spawn issues when the lesson file is __main__.
    return (
        DataLoader(train_set, batch_size, shuffle=True, num_workers=0),
        DataLoader(test_set, batch_size, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def fit(model, epochs=5, lr=1e-3, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    train_loader, test_loader = get_loaders(batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device={device}  params={count_params(model):,}")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
        acc = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch:02d}  "
            f"loss {running / len(train_loader.dataset):.3f}  acc {acc:.3f}"
        )
    return model
