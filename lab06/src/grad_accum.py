import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from common import (INK, MUTED, RESULTS_DIR, SERIES, SURFACE, env_line, load_config, pick_device,
                    seed_everything, style_axes, tee_results, with_overrides, write_csv)
from data import train_tensors
from models import build_model
from pipeline import run

EPOCHS = 5
RUNS = [("b256", 256, 1), ("b64x4", 64, 4), ("b64", 64, 1)]


def one_step_check(cfg):
    device = pick_device(cfg["device"])
    seed_everything(cfg["seed"], cfg["deterministic"])
    model = build_model(cfg).to(device)
    x, y = train_tensors(cfg, device)
    x, y = x[:256], y[:256]
    loss_fn = nn.CrossEntropyLoss()
    model.zero_grad()
    loss_fn(model(x), y).backward()
    big = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for i in range(0, 256, 64):
        (loss_fn(model(x[i:i + 64]), y[i:i + 64]) / 4).backward()
    acc = [p.grad.clone() for p in model.parameters()]
    abs_diff = max((a - b).abs().max().item() for a, b in zip(big, acc))
    scale = max(b.abs().max().item() for b in big)
    return abs_diff, scale


def max_param_diff(m1, m2):
    return max((a - b).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))


if __name__ == "__main__":
    base = load_config()
    results = {}
    with tee_results("grad_accum.txt"):
        print(f"{env_line(pick_device(base['device']))} | Adam lr {base['optim']['lr']}, no scheduler, {EPOCHS} epochs, "
              f"same seed, same initial weights and same shuffled sample order in every run")
        diff, scale = one_step_check(base)
        print(f"one-step check on the first 256 training images: max |grad(batch 256) - sum of 4 x grad(batch 64)/4| "
              f"= {diff:.2e} (largest gradient entry {scale:.2e})\n")
        for name, b, k in RUNS:
            cfg = with_overrides(base, data__batch_size=b, train__accum_steps=k, train__epochs=EPOCHS,
                                 run_name=f"accum-{name}")
            results[name] = res = run(cfg)
            res["optimizer_steps"] = len(res["step_losses"])

        print(f"\n{'run':<6} {'batch':>5} {'accum':>5} {'eff.':>5} {'opt steps':>9} {'samples/s':>9} {'peak MB':>8} "
              f"{'train loss':>10} {'val acc':>7} {'test acc':>8}")
        rows = []
        for name, b, k in RUNS:
            r = results[name]
            h = r["history"]
            thr = statistics.median(x["samples_per_s"] for x in h[1:])
            rows.append({"run": name, "batch_size": b, "accum_steps": k, "effective_batch": b * k,
                         "optimizer_steps": r["optimizer_steps"], "samples_per_s": round(thr),
                         "peak_mem_mb": round(r["peak_mem_mb"], 1), "final_train_loss": round(h[-1]["train_loss"], 4),
                         "final_val_acc": round(h[-1]["val_acc"], 4), "test_acc": round(r["test_acc"], 4)})
            print(f"{name:<6} {b:>5} {k:>5} {b * k:>5} {r['optimizer_steps']:>9} {thr:>9,.0f} "
                  f"{r['peak_mem_mb']:>8.1f} {h[-1]['train_loss']:>10.4f} {h[-1]['val_acc']:>7.4f} "
                  f"{r['test_acc']:>8.4f}")
        write_csv(rows, "grad_accum.csv")

        print("\nper-epoch comparison, b256 vs b64x4")
        print(f"{'epoch':>5} {'train loss 256':>14} {'train loss 64x4':>15} {'val acc 256':>11} {'val acc 64x4':>12}")
        for a, c in zip(results["b256"]["history"], results["b64x4"]["history"]):
            print(f"{a['epoch']:>5} {a['train_loss']:>14.5f} {c['train_loss']:>15.5f} {a['val_acc']:>11.4f} "
                  f"{c['val_acc']:>12.4f}")
        s1 = [x["loss"] for x in results["b256"]["step_losses"]]
        s2 = [x["loss"] for x in results["b64x4"]["step_losses"]]
        d = [abs(a - c) for a, c in zip(s1, s2)]
        print(f"per optimizer step ({len(d)} steps): first-epoch max |loss diff| {max(d[:215]):.2e}, "
              f"whole run max {max(d):.2e}, mean {statistics.mean(d):.2e}")
        print(f"max |weight diff| after {EPOCHS} epochs: b256 vs b64x4 "
              f"{max_param_diff(results['b256']['model'], results['b64x4']['model']):.2e}, "
              f"b256 vs b64 (no accumulation) "
              f"{max_param_diff(results['b256']['model'], results['b64']['model']):.2e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(range(1, len(s1) + 1), s1, color=SERIES[0], lw=2, label="batch 256")
    ax.plot(range(1, len(s2) + 1), s2, color=SERIES[1], lw=1.2, ls="--", label="batch 64 x accumulation 4")
    style_axes(ax, "Training loss per optimizer step", "optimizer step", "loss")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.semilogy(range(1, len(d) + 1), [max(v, 1e-9) for v in d], color=SERIES[2], lw=1.2)
    style_axes(ax, "|loss(256) - loss(64x4)| per step", "optimizer step")
    ax = axes[2]
    for i, (name, label) in enumerate([("b256", "batch 256"), ("b64x4", "batch 64 x accum 4"),
                                       ("b64", "batch 64, no accum")]):
        h = results[name]["history"]
        ax.plot([x["epoch"] for x in h], [x["val_acc"] * 100 for x in h], color=SERIES[i], lw=2,
                marker="osD"[i], ms=7 if i else 9, ls="--" if i == 1 else "-", label=label)
    ax.set_xticks(range(1, EPOCHS + 1))
    style_axes(ax, "Validation accuracy, %", "epoch")
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("Gradient accumulation: batch 64 x 4 vs true batch 256", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "grad_accum.png", dpi=150, facecolor=SURFACE)
