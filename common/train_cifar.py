"""CIFAR-10 loader and training loop shared by lessons 5-14."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.trace import count_params

ROOT = Path(__file__).resolve().parents[1]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_loaders(batch_size=128):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    data_root = ROOT / "data"
    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
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


def fit(model, epochs=20, lr=0.1, batch_size=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    train_loader, test_loader = get_loaders(batch_size)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
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
        sched.step()
        acc = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch:02d}  "
            f"loss {running / len(train_loader.dataset):.3f}  acc {acc:.3f}"
        )
    return model
