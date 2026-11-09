import argparse
import gc
import math
import random
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import tee_results
from engine import Value, vsum


def make_moons(n_samples=200, noise=0.1, seed=0):
    try:
        from sklearn.datasets import make_moons as sk_make_moons

        return sk_make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    except ImportError:
        rng = np.random.default_rng(seed)
        n_out = n_samples // 2
        n_in = n_samples - n_out
        t_out = np.linspace(0, np.pi, n_out)
        t_in = np.linspace(0, np.pi, n_in)
        X = np.vstack([
            np.column_stack([np.cos(t_out), np.sin(t_out)]),
            np.column_stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]),
        ])
        y = np.hstack([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
        X += rng.normal(scale=noise, size=X.shape)
        perm = rng.permutation(n_samples)
        return X[perm], y[perm]


def init_weights(seed=0, n_in=2, n_hidden=8, n_out=1):
    rng = random.Random(seed)
    b1 = 1 / math.sqrt(n_in)
    b2 = 1 / math.sqrt(n_hidden)
    W1 = [[rng.uniform(-b1, b1) for _ in range(n_in)] for _ in range(n_hidden)]
    B1 = [rng.uniform(-b1, b1) for _ in range(n_hidden)]
    W2 = [[rng.uniform(-b2, b2) for _ in range(n_hidden)] for _ in range(n_out)]
    B2 = [rng.uniform(-b2, b2) for _ in range(n_out)]
    return W1, B1, W2, B2


class EngineMLP:
    def __init__(self, W1, B1, W2, B2):
        self.W1 = [[Value(w) for w in row] for row in W1]
        self.B1 = [Value(b) for b in B1]
        self.W2 = [[Value(w) for w in row] for row in W2]
        self.B2 = [Value(b) for b in B2]

    def parameters(self):
        ps = [w for row in self.W1 for w in row] + self.B1
        ps += [w for row in self.W2 for w in row] + self.B2
        return ps

    def forward(self, x):
        h = [vsum(w * xi for w, xi in zip(row, x)) + b
             for row, b in zip(self.W1, self.B1)]
        h = [hi.relu() for hi in h]
        z = vsum(w * hi for w, hi in zip(self.W2[0], h)) + self.B2[0]
        return z.sigmoid()


def bce(p, y, eps=1e-12):
    if y == 1:
        return -(p + eps).log()
    return -((1 - p) + eps).log()


def train_engine(X, y, weights, epochs, lr):
    model = EngineMLP(*weights)
    params = model.parameters()
    Xl, yl = X.tolist(), y.tolist()
    n = len(Xl)
    for _ in range(epochs):
        preds = [model.forward(x) for x in Xl]
        loss = vsum(bce(p, t) for p, t in zip(preds, yl)) * (1.0 / n)
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
    preds = [model.forward(x).data for x in Xl]
    acc = sum((p > 0.5) == bool(t) for p, t in zip(preds, yl)) / n
    return loss.data, acc


class TorchMLP(nn.Module):
    def __init__(self, W1, B1, W2, B2, dtype):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor(W1))
            self.fc1.bias.copy_(torch.tensor(B1))
            self.fc2.weight.copy_(torch.tensor(W2))
            self.fc2.bias.copy_(torch.tensor(B2))
        self.to(dtype)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def train_torch(X, y, weights, epochs, lr, dtype=torch.float64, device="cpu"):
    model = TorchMLP(*weights, dtype=dtype).to(device)
    Xt = torch.tensor(X, dtype=dtype, device=device)
    yt = torch.tensor(y, dtype=dtype, device=device).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        p = model(Xt)
        loss = F.binary_cross_entropy(p, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((model(Xt) > 0.5) == (yt > 0.5)).double().mean().item()
    if device == "cuda":
        torch.cuda.synchronize()
    return loss.item(), acc


def bench(fn, runs):
    times, result = [], None
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    torch.set_num_threads(1)
    X, y = make_moons(args.n, noise=0.1, seed=0)
    w = init_weights(seed=0)
    print(f"make_moons n={args.n}, model 2->8->1 (ReLU, sigmoid), BCE, "
          f"full-batch SGD lr={args.lr}, epochs={args.epochs}, runs={args.runs}\n")

    configs = [
        ("our engine (Python scalars)", lambda: train_engine(X, y, w, args.epochs, args.lr)),
        ("PyTorch CPU float64", lambda: train_torch(X, y, w, args.epochs, args.lr, torch.float64)),
        ("PyTorch CPU float32", lambda: train_torch(X, y, w, args.epochs, args.lr, torch.float32)),
    ]
    if torch.cuda.is_available():
        try:
            train_torch(X, y, w, 2, args.lr, torch.float32, "cuda")
            configs.append(("PyTorch CUDA float32",
                            lambda: train_torch(X, y, w, args.epochs, args.lr, torch.float32, "cuda")))
        except Exception as e:
            print(f"(CUDA skipped: {type(e).__name__}: {e})")

    rows = []
    for name, fn in configs:
        med, times, (loss, acc) = bench(fn, args.runs)
        rows.append((name, med, times, loss, acc))

    base = rows[0][1]
    print(f"{'implementation':<30}{'median, s':>11}{'speedup':>10}{'final loss':>13}{'train acc':>11}")
    for name, med, times, loss, acc in rows:
        print(f"{name:<30}{med:>11.4f}{base / med:>9.1f}x{loss:>13.6f}{acc:>11.3f}")
    print("\nall runs (s):")
    for name, med, times, *_ in rows:
        print(f"  {name:<30} " + ", ".join(f"{t:.4f}" for t in times))
    print(f"\n|loss_engine - loss_torch_f64| = {abs(rows[0][3] - rows[1][3]):.2e}")


if __name__ == "__main__":
    with tee_results("train_moons.txt"):
        main()
