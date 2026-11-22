import statistics
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from common import (INK, MARKERS, MUTED, RESULTS_DIR, SERIES, SURFACE, env_line, load_config, pick_device,
                    seed_everything, style_axes, tee_results, with_overrides, write_csv)
from data import TensorLoader, build_data, train_tensors
from models import build_model
from pipeline import Tracker, build_optimizer, sync, train_epoch

WORKERS = [0, 1, 2, 4, 8]
TRAIN_EPOCHS = 3
DATA_PASSES = 2


def setup(cfg):
    device = pick_device(cfg["device"])
    seed_everything(cfg["seed"], cfg["deterministic"])
    model = build_model(cfg).to(device)
    return device, model, build_optimizer(cfg, model.parameters()), nn.CrossEntropyLoss()


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def data_pass(loader, device):
    for x, y in loader:
        x.to(device, non_blocking=True)
        y.to(device, non_blocking=True)
    sync(device)


def compute_only(cfg):
    device, model, opt, loss_fn = setup(cfg)
    x, y = train_tensors(cfg, device)
    loader = TensorLoader(x, y, cfg["data"]["batch_size"], cfg["seed"])
    train_epoch(model, loader, opt, loss_fn, device)
    return statistics.median(timed(lambda: train_epoch(model, loader, opt, loss_fn, device)) for _ in range(2))


def measure(cfg):
    device, model, opt, loss_fn = setup(cfg)
    t0 = time.perf_counter()
    loader, _, _, _ = build_data(cfg, device)
    it = iter(loader)
    next(it)
    first_batch_s = time.perf_counter() - t0
    del it
    epochs = [timed(lambda: train_epoch(model, loader, opt, loss_fn, device)) for _ in range(TRAIN_EPOCHS)]
    passes = [timed(lambda: data_pass(loader, device)) for _ in range(DATA_PASSES)]
    return first_batch_s, epochs, passes


if __name__ == "__main__":
    base = load_config()
    n = base["data"]["n_train"]
    rows = []
    with tee_results("workers_study.txt"):
        print(f"{env_line(pick_device('cuda'))} | batch {base['data']['batch_size']} | pin_memory on GPU | "
              f"persistent workers | train epoch = median of epochs 2-{TRAIN_EPOCHS}, data-only = median of "
              f"{DATA_PASSES} passes (loader + host-to-device copy, no model)")
        gpu_compute = compute_only(with_overrides(base, device="cuda"))
        cpu_compute = compute_only(with_overrides(base, device="cpu"))
        print(f"compute only (whole train set pre-loaded as tensors, no DataLoader): "
              f"GPU {gpu_compute:.2f} s/epoch ({n / gpu_compute:,.0f} samples/s), "
              f"CPU {cpu_compute:.2f} s/epoch ({n / cpu_compute:,.0f} samples/s)\n", flush=True)
        print(f"{'device':<6} {'workers':>7} {'1st batch s':>11} {'epoch 1 s':>9} {'epoch s':>8} "
              f"{'samples/s':>9} {'data-only s':>11} {'data samples/s':>14}")
        for dev in ("cuda", "cpu"):
            for w in WORKERS:
                cfg = with_overrides(base, device=dev, data__num_workers=w, run_name=f"workers-{dev}-w{w}")
                first, epochs, passes = measure(cfg)
                ep, dp = statistics.median(epochs[1:]), statistics.median(passes)
                row = {"device": dev, "num_workers": w, "first_batch_s": round(first, 3),
                       "epoch1_s": round(epochs[0], 3), "epoch_s": round(ep, 3),
                       "samples_per_s": round(n / ep), "data_only_s": round(dp, 3),
                       "data_samples_per_s": round(n / dp),
                       "compute_only_s": round(gpu_compute if dev == "cuda" else cpu_compute, 3)}
                rows.append(row)
                print(f"{dev:<6} {w:>7} {first:>11.2f} {epochs[0]:>9.2f} {ep:>8.2f} {n / ep:>9,.0f} "
                      f"{dp:>11.2f} {n / dp:>14,.0f}", flush=True)
                tracker = Tracker(cfg, run_name=cfg["run_name"])
                tracker.params({"device": dev, "data.num_workers": w, "data.batch_size": cfg["data"]["batch_size"],
                                "kind": "workers_study"})
                tracker.metrics({k: v for k, v in row.items() if k not in ("device", "num_workers")})
                tracker.end()
    write_csv(rows, "workers_study.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, dev, compute, title in ((axes[0], "cuda", gpu_compute, "GPU training (GTX 1060)"),
                                    (axes[1], "cpu", cpu_compute, "CPU training (8 threads)")):
        r = [x for x in rows if x["device"] == dev]
        ws = [x["num_workers"] for x in r]
        ax.plot(ws, [x["epoch_s"] for x in r], color=SERIES[0], marker=MARKERS[0], lw=2, ms=8,
                label="train epoch (loader + model)")
        ax.plot(ws, [x["data_only_s"] for x in r], color=SERIES[1], marker=MARKERS[1], lw=2, ms=8,
                label="data only (loader + copy)")
        ax.axhline(compute, color=SERIES[2], lw=2, ls="--", label="compute only (tensors in memory)")
        for x in r:
            ax.annotate(f"{x['epoch_s']:.1f}", (x["num_workers"], x["epoch_s"]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=INK)
        ax.set_xticks(WORKERS)
        ax.set_ylim(0, None)
        style_axes(ax, title, "num_workers", "seconds per epoch (55k images)")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.suptitle("DataLoader workers vs epoch time, Fashion-MNIST CNN, batch 128", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "workers_study.png", dpi=150, facecolor=SURFACE)
