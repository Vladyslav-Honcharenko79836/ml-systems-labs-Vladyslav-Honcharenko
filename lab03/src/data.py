"""Fashion-MNIST for Lab 3: ToTensor + Normalize, 55k/5k train/val split of the official train set, official 10k test."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128
MEAN, STD = 0.2860, 0.3530
N_TRAIN, N_VAL = 55_000, 5_000
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
LAB_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LAB_DIR / "results"
DATA_DIR = LAB_DIR.parent / "data"  # git-ignored, shared between labs

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
])


def load_datasets():
    """Returns train (55k), val (5k), test (10k) datasets."""
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=TRANSFORM)
    train, val = random_split(full_train, [N_TRAIN, N_VAL],
                              generator=torch.Generator().manual_seed(SEED))
    return train, val, test


def load_loaders(batch_size=BATCH_SIZE):
    train, val, test = load_datasets()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(SEED)),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


if __name__ == "__main__":
    train, val, test = load_datasets()
    train_dl, val_dl, test_dl = load_loaders()
    xb, yb = next(iter(train_dl))

    # Statistics of the normalized training images: should be close to mean 0, std 1.
    full = torch.stack([train.dataset[i][0] for i in train.indices[:10_000]])

    def class_counts(targets):
        return torch.bincount(targets, minlength=len(CLASSES))

    train_targets = train.dataset.targets[train.indices]
    val_targets = train.dataset.targets[val.indices]
    tr_c, va_c, te_c = class_counts(train_targets), class_counts(val_targets), class_counts(test.targets)

    lines = [
        f"train {len(train)}  val {len(val)}  test {len(test)}",
        f"train/val index overlap: {len(set(train.indices) & set(val.indices))}",
        f"batches per epoch: train {len(train_dl)}  val {len(val_dl)}  test {len(test_dl)}",
        f"one batch: x {tuple(xb.shape)} {xb.dtype}  y {tuple(yb.shape)} {yb.dtype}",
        f"normalized pixels (10k train images): mean {full.mean():.4f}  std {full.std():.4f}  "
        f"min {full.min():.4f}  max {full.max():.4f}",
        "class        name  train   val  test",
    ]
    for c, name in enumerate(CLASSES):
        lines.append(f"{c:>5}  {name:>11}  {tr_c[c]:>5}  {va_c[c]:>4}  {te_c[c]:>4}")
    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dataset.txt").write_text(text + "\n")
