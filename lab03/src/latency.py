"""Lab 3 step 3: inference latency at batch 1 (median of 100 runs) and batch 128 (median of 20 runs).

Protocol: eval mode, torch.inference_mode, fixed thread count, real test images, 10 warm-up runs before
timing, time.perf_counter around each forward call. The whole protocol is repeated in 3 passes (model order
rotated each pass) to show that the medians are repeatable. Untrained weights are fine here: latency does
not depend on the values of the weights.
"""
import csv
import json
import os
import platform
import statistics
import time

import torch

from data import RESULTS_DIR, load_datasets
from models import MODELS

RUNS = {1: 100, 128: 20}
WARMUP = 10
PASSES = 3


def time_forward(model, x, runs, warmup=WARMUP):
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    return times


def pct(xs, q):
    return statistics.quantiles(xs, n=100)[q - 1]


if __name__ == "__main__":
    torch.manual_seed(0)
    _, _, test = load_datasets()
    x128 = torch.stack([test[i][0] for i in range(128)])
    inputs = {1: x128[:1].contiguous(), 128: x128}
    models = {name: cls().eval() for name, cls in MODELS.items()}
    names = list(models)

    raw = {name: {str(b): [] for b in RUNS} for name in names}  # per pass: list of per-run seconds
    for p in range(PASSES):
        order = names[p:] + names[:p]
        for name in order:
            for b, runs in RUNS.items():
                raw[name][str(b)].append(time_forward(models[name], inputs[b], runs))

    env = (f"CPU: {platform.processor()} | logical cores {os.cpu_count()} | torch {torch.__version__} | "
           f"threads {torch.get_num_threads()} | warm-up {WARMUP} | passes {PASSES}")
    lines = [env, "",
             "model      batch  runs  median ms  p10 ms  p90 ms  per-image ms  img/s     pass medians ms"]
    rows = []
    for name in names:
        for b, runs in RUNS.items():
            passes = raw[name][str(b)]
            last = passes[-1]
            med = statistics.median(last) * 1e3
            pass_meds = [statistics.median(t) * 1e3 for t in passes]
            row = {
                "model": name, "batch": b, "runs": runs,
                "median_ms": round(med, 4),
                "p10_ms": round(pct(last, 10) * 1e3, 4),
                "p90_ms": round(pct(last, 90) * 1e3, 4),
                "per_image_ms": round(med / b, 5),
                "images_per_s": round(b / (med / 1e3)),
                "pass_medians_ms": " ".join(f"{m:.4f}" for m in pass_meds),
            }
            rows.append(row)
            lines.append(f"{name:<9}  {b:>5}  {runs:>4}  {med:>9.4f}  {row['p10_ms']:>6.4f}  {row['p90_ms']:>6.4f}"
                         f"  {row['per_image_ms']:>12.5f}  {row['images_per_s']:>6}    {row['pass_medians_ms']}")
    lines.append("")
    lines.append("Reported median/p10/p90 come from the last pass; 'pass medians' shows all passes.")

    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latency.txt").write_text(text + "\n", encoding="utf-8")
    with open(RESULTS_DIR / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS_DIR / "latency_raw.json").write_text(json.dumps({"env": env, "seconds": raw}, indent=1))
