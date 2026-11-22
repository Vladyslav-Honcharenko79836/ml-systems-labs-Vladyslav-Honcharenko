import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from common import (INK, RESULTS_DIR, SERIES, SURFACE, env_line, load_config, pick_device, seed_everything,
                    style_axes, tee_results, with_overrides, write_csv)
from data import TensorLoader, train_tensors
from models import build_model
from pipeline import build_optimizer, run, train_epoch

BATCHES = [32, 64, 128, 256, 512]
EPOCHS = 5


def gpu_only(cfg):
    device = pick_device(cfg["device"])
    seed_everything(cfg["seed"], cfg["deterministic"])
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model.parameters())
    x, y = train_tensors(cfg, device)
    loader = TensorLoader(x, y, cfg["data"]["batch_size"], cfg["seed"])
    train_epoch(model, loader, opt, nn.CrossEntropyLoss(), device)
    t0 = time.perf_counter()
    train_epoch(model, loader, opt, nn.CrossEntropyLoss(), device)
    return len(y) / (time.perf_counter() - t0)


if __name__ == "__main__":
    base = load_config()
    rows = []
    with tee_results("batch_study.txt"):
        print(f"{env_line(pick_device(base['device']))} | Adam lr {base['optim']['lr']} (same for every batch), "
              f"no scheduler, {EPOCHS} epochs, num_workers {base['data']['num_workers']}")
        for b in BATCHES:
            cfg = with_overrides(base, data__batch_size=b, train__epochs=EPOCHS, run_name=f"batch-{b}")
            res = run(cfg)
            h = res["history"]
            thr = statistics.median(r["samples_per_s"] for r in h[1:])
            rows.append({"batch_size": b, "steps_per_epoch": -(-cfg["data"]["n_train"] // b),
                         "samples_per_s": round(thr), "gpu_only_samples_per_s": round(gpu_only(cfg)),
                         "peak_mem_mb": round(res["peak_mem_mb"], 1),
                         "final_train_loss": round(h[-1]["train_loss"], 4), "final_val_acc": round(h[-1]["val_acc"], 4),
                         "test_acc": round(res["test_acc"], 4), "mlflow_run_id": res["run_id"]})
        print(f"\n{'batch':>5} {'steps/ep':>8} {'samples/s':>9} {'GPU-only/s':>10} {'peak MB':>8} "
              f"{'train loss':>10} {'val acc':>7} {'test acc':>8}")
        for r in rows:
            print(f"{r['batch_size']:>5} {r['steps_per_epoch']:>8} {r['samples_per_s']:>9,} "
                  f"{r['gpu_only_samples_per_s']:>10,} {r['peak_mem_mb']:>8.1f} {r['final_train_loss']:>10.4f} "
                  f"{r['final_val_acc']:>7.4f} {r['test_acc']:>8.4f}")
        print("samples/s = median of epochs 2-5 with the real DataLoader; GPU-only/s = same model and batch, "
              "data already on the GPU; peak MB = torch.cuda.max_memory_allocated during training")
    write_csv(rows, "batch_study.csv")

    bs = [r["batch_size"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("samples_per_s", "Throughput, samples/s", "{:,.0f}"),
              ("peak_mem_mb", "Peak GPU memory in training, MB", "{:.0f}"),
              ("test_acc", "Test accuracy after 5 epochs, %", "{:.2f}")]
    for ax, (key, title, fmt) in zip(axes, panels):
        ys = [r[key] * (100 if key == "test_acc" else 1) for r in rows]
        ax.plot(bs, ys, color=SERIES[0], marker="o", lw=2, ms=8, label="real DataLoader")
        for x, y in zip(bs, ys):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color=INK)
        if key == "samples_per_s":
            g = [r["gpu_only_samples_per_s"] for r in rows]
            ax.plot(bs, g, color=SERIES[1], marker="s", lw=2, ms=8, label="data already on GPU")
            ax.legend(frameon=False, fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([str(b) for b in bs])
        style_axes(ax, title, "batch size")
    fig.suptitle("Batch-size study, Fashion-MNIST CNN, Adam lr 1e-3, 5 epochs", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "batch_study.png", dpi=150, facecolor=SURFACE)
