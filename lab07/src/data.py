import torch
from torch.utils.data import random_split
from torchvision import datasets

from common import ROOT


def load_tensors(cfg, device):
    d = cfg["data"]
    root = ROOT / cfg["paths"]["data_dir"]
    full = datasets.FashionMNIST(root, train=True, download=True)
    test = datasets.FashionMNIST(root, train=False, download=True)
    train_split, val_split = random_split(range(len(full)), [d["n_train"], d["n_val"]],
                                          generator=torch.Generator().manual_seed(cfg["split_seed"]))

    def conv(images, targets):
        x = images.float().div(255).sub(d["mean"]).div(d["std"]).unsqueeze(1)
        return x.to(device), targets.to(device)

    tr, va = list(train_split.indices), list(val_split.indices)
    return {"train": conv(full.data[tr], full.targets[tr]),
            "val": conv(full.data[va], full.targets[va]),
            "test": conv(test.data, test.targets)}


class Batches:
    def __init__(self, x, y, batch_size, seed, extra=None):
        self.x, self.y, self.extra = x, y, extra
        self.batch_size = batch_size
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.y), generator=self.gen).to(self.x.device)
        for i in range(0, len(perm), self.batch_size):
            idx = perm[i:i + self.batch_size]
            yield self.x[idx], self.y[idx], (self.extra[idx] if self.extra is not None else None)
