import torch
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision import datasets, transforms

from common import resolve


def _datasets(cfg):
    d = cfg["data"]
    root = resolve(cfg["paths"]["data_dir"])
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((d["mean"],), (d["std"],))])
    full = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    train, val = random_split(full, [d["n_train"], d["n_val"]],
                              generator=torch.Generator().manual_seed(cfg["seed"]))
    return train, val, test


def _to_tensors(images, targets, cfg, device):
    d = cfg["data"]
    x = images.float().div(255).sub(d["mean"]).div(d["std"]).unsqueeze(1)
    return x.to(device), targets.to(device)


def build_loader(train, cfg, device):
    d = cfg["data"]
    sampler_gen = torch.Generator().manual_seed(cfg["seed"])
    workers = d["num_workers"]
    loader = DataLoader(
        train, batch_size=d["batch_size"], sampler=RandomSampler(train, generator=sampler_gen),
        generator=torch.Generator().manual_seed(cfg["seed"] + 1),
        num_workers=workers, pin_memory=d["pin_memory"] and device.type == "cuda",
        persistent_workers=d["persistent_workers"] and workers > 0,
    )
    return loader, sampler_gen


def build_data(cfg, device):
    train, val, test = _datasets(cfg)
    loader, sampler_gen = build_loader(train, cfg, device)
    base = train.dataset
    cpu = torch.device("cpu")
    val_set = _to_tensors(base.data[val.indices], base.targets[val.indices], cfg, cpu)
    test_set = _to_tensors(test.data, test.targets, cfg, cpu)
    return loader, sampler_gen, val_set, test_set


def train_dataset(cfg):
    return _datasets(cfg)[0]


def train_tensors(cfg, device):
    train = train_dataset(cfg)
    base = train.dataset
    return _to_tensors(base.data[train.indices], base.targets[train.indices], cfg, device)


class TensorLoader:
    def __init__(self, x, y, batch_size, seed):
        self.dataset = y
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.y), generator=self.gen).to(self.x.device)
        for i in range(0, len(perm), self.batch_size):
            idx = perm[i:i + self.batch_size]
            yield self.x[idx], self.y[idx]
