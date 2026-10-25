"""Step 6: mini-batch SGD (batch 32, 50 epochs, reshuffled each epoch) for lr x width grid.

Writes results/training_log.csv (per-epoch), results/final_accuracy.csv and results/loss_curves.png.
`--init he` is a control experiment (He initialisation instead of N(0, 0.01)); its files get a `_he` suffix.
"""
import argparse
import csv
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import RESULTS_DIR, SEED, load_split
from mlp import backward, cross_entropy, forward, init_params, predict, sgd_step

LRS = (0.001, 0.01, 0.1)
WIDTHS = (16, 64)
EPOCHS = 50
BATCH = 32


def train(width, lr, X_tr, Y_tr, y_tr, X_te, Y_te, y_te, scheme="normal"):
    rng = np.random.default_rng(SEED)  # same init and same shuffles for every configuration
    params = init_params(width, rng, scheme=scheme)
    log = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(X_tr.shape[0])
        batch_losses = []
        for s in range(0, X_tr.shape[0], BATCH):
            ids = order[s:s + BATCH]
            P, cache = forward(params, X_tr[ids])
            batch_losses.append(cross_entropy(P, Y_tr[ids]))
            sgd_step(params, backward(params, cache, Y_tr[ids]), lr)
        row = {
            "width": width, "lr": lr, "epoch": epoch,
            "train_loss": cross_entropy(forward(params, X_tr)[0], Y_tr),  # full pass after the epoch
            "mean_batch_loss": float(np.mean(batch_losses)),
            "test_loss": cross_entropy(forward(params, X_te)[0], Y_te),
            "train_acc": float((predict(params, X_tr) == y_tr).mean()),
            "test_acc": float((predict(params, X_te) == y_te).mean()),
        }
        log.append(row)
        print(f"W={width:<3} lr={lr:<6} epoch {epoch:>2}/{EPOCHS}  train_loss {row['train_loss']:.4f}"
              f"  test_loss {row['test_loss']:.4f}  train_acc {row['train_acc']:.4f}  test_acc {row['test_acc']:.4f}")
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", choices=("normal", "he"), default="normal")
    scheme = parser.parse_args().init
    suffix = "" if scheme == "normal" else f"_{scheme}"
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_split()
    logs, finals = [], []
    for width in WIDTHS:
        for lr in LRS:
            t0 = time.perf_counter()
            log = train(width, lr, X_tr, Y_tr, y_tr, X_te, Y_te, y_te, scheme)
            secs = time.perf_counter() - t0
            logs += log
            last = log[-1]
            finals.append({"width": width, "lr": lr, "final_train_loss": last["train_loss"],
                           "final_test_loss": last["test_loss"], "final_train_acc": last["train_acc"],
                           "final_test_acc": last["test_acc"],
                           "best_test_acc": max(r["test_acc"] for r in log),
                           "train_seconds": secs})

    for name, rows in (("training_log", logs), ("final_accuracy", finals)):
        with open(RESULTS_DIR / f"{name}{suffix}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {0.001: "#1f77b4", 0.01: "#ff7f0e", 0.1: "#2ca02c"}
    for ax, width in zip(axes, WIDTHS):
        for lr in LRS:
            rows = [r for r in logs if r["width"] == width and r["lr"] == lr]
            ep = [r["epoch"] for r in rows]
            acc = rows[-1]["test_acc"]
            ax.plot(ep, [r["train_loss"] for r in rows], color=colors[lr], label=f"lr={lr} train (test acc {acc:.3f})")
            ax.plot(ep, [r["test_loss"] for r in rows], color=colors[lr], ls="--", label=f"lr={lr} test")
        ax.axhline(np.log(10), color="grey", lw=0.8, ls=":", label="ln(10)")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cross-entropy loss")
    init_label = "N(0, 0.01)" if scheme == "normal" else "He, std = sqrt(2 / fan_in) (control)"
    fig.suptitle(f"Two-hidden-layer MLP on digits: SGD, batch 32, init {init_label}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"loss_curves{suffix}.png", dpi=150)

    print("\nwidth  lr      final_train_loss  final_test_loss  final_train_acc  final_test_acc  seconds")
    for r in finals:
        print(f"{r['width']:>5}  {r['lr']:<6}  {r['final_train_loss']:>16.4f}  {r['final_test_loss']:>15.4f}"
              f"  {r['final_train_acc']:>15.4f}  {r['final_test_acc']:>14.4f}  {r['train_seconds']:>7.2f}")
