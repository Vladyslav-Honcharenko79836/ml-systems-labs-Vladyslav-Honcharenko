"""Step 7: time the forward and backward passes separately (median of 100 runs).

For each width, two batch sizes are timed: the training mini-batch (32) and the full
training set (1257). The FLOP count of the matrix multiplications is computed alongside, and
the matmuls of each pass are also timed on their own to get their share of the pass.
"""
import csv
import json
import os
import platform
import time

import numpy as np

from data import RESULTS_DIR, SEED, load_split
from mlp import N_IN, N_OUT, backward, forward, init_params

RUNS = 100
WARMUP = 10


def matmul_flops(B, width):
    """2*m*k*n per matmul. Forward: 3 matmuls. Backward: 3 weight grads (A^T dZ) + 2 input grads (dZ W^T)."""
    shapes = [(B, N_IN, width), (B, width, width), (B, width, N_OUT)]  # (m, k, n) of X@W1, A1@W2, A2@W3
    fwd = sum(2 * m * k * n for m, k, n in shapes)
    bwd_weight = fwd                                            # A^T @ dZ has the same m*k*n as A @ W
    bwd_input = sum(2 * m * k * n for m, k, n in shapes[1:])   # dZ @ W^T for layers 3 and 2 (no dX needed)
    return fwd, bwd_weight + bwd_input


def forward_matmuls(params, X, A1, A2):
    X @ params["W1"]
    A1 @ params["W2"]
    A2 @ params["W3"]


def backward_matmuls(params, X, A1, A2, dZ3, dZ2, dZ1):
    A2.T @ dZ3
    dZ3 @ params["W3"].T
    A1.T @ dZ2
    dZ2 @ params["W2"].T
    X.T @ dZ1


def time_runs(fn):
    for _ in range(WARMUP):
        fn()
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return np.array(t) * 1e6  # microseconds


if __name__ == "__main__":
    X_tr, _, Y_tr, _, _, _ = load_split()
    rows, raw = [], {}
    for width in (16, 64):
        params = init_params(width, np.random.default_rng(SEED), scheme="he")  # non-degenerate activations
        for B in (32, X_tr.shape[0]):
            X, Y = X_tr[:B], Y_tr[:B]
            _, cache = forward(params, X)
            t_f = time_runs(lambda: forward(params, X))
            t_b = time_runs(lambda: backward(params, cache, Y))
            _, Z1, A1, Z2, A2, P = cache
            dZ3 = (P - Y) / B
            dZ2 = (dZ3 @ params["W3"].T) * (Z2 > 0)
            dZ1 = (dZ2 @ params["W2"].T) * (Z1 > 0)
            t_fm = time_runs(lambda: forward_matmuls(params, X, A1, A2))
            t_bm = time_runs(lambda: backward_matmuls(params, X, A1, A2, dZ3, dZ2, dZ1))
            f_flops, b_flops = matmul_flops(B, width)
            raw[f"W{width}_B{B}"] = {"forward_us": t_f.tolist(), "backward_us": t_b.tolist()}
            rows.append({
                "width": width, "batch": B,
                "forward_median_us": np.median(t_f), "backward_median_us": np.median(t_b),
                "backward_over_forward": np.median(t_b) / np.median(t_f),
                "forward_matmul_median_us": np.median(t_fm), "backward_matmul_median_us": np.median(t_bm),
                "forward_matmul_share": np.median(t_fm) / np.median(t_f),
                "backward_matmul_share": np.median(t_bm) / np.median(t_b),
                "forward_p95_us": np.percentile(t_f, 95), "backward_p95_us": np.percentile(t_b, 95),
                "forward_matmul_flops": f_flops, "backward_matmul_flops": b_flops,
                "flop_ratio": b_flops / f_flops,
            })

    with open(RESULTS_DIR / "timing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS_DIR / "timing_raw.json").write_text(json.dumps(raw))

    lines = [f"{platform.processor()} | {os.cpu_count()} logical CPUs | numpy {np.__version__} | "
             f"median of {RUNS} runs after {WARMUP} warm-up",
             f"{'W':>3} {'batch':>5} {'fwd med us':>11} {'bwd med us':>11} {'bwd/fwd':>8} "
             f"{'fwd MFLOP':>10} {'bwd MFLOP':>10} {'FLOP ratio':>10} {'fwd mm share':>12} {'bwd mm share':>12}"]
    for r in rows:
        lines.append(f"{r['width']:>3} {r['batch']:>5} {r['forward_median_us']:>11.1f} {r['backward_median_us']:>11.1f}"
                     f" {r['backward_over_forward']:>8.2f} {r['forward_matmul_flops'] / 1e6:>10.3f}"
                     f" {r['backward_matmul_flops'] / 1e6:>10.3f} {r['flop_ratio']:>10.2f}"
                     f" {r['forward_matmul_share']:>12.0%} {r['backward_matmul_share']:>12.0%}")
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "timing.txt").write_text(text + "\n")
