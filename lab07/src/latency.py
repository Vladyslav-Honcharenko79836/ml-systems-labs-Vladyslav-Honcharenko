import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnxruntime as ort
import torch

from bench import measure, model_fn
from common import (INK, MUTED, RESULTS_DIR, SERIES, SURFACE, env_line, load_config, pick_device, style_axes,
                    tee_results, write_csv)
from data import load_tensors
from models import effective_macs, total_macs
from zoo import load_model, names


def ort_fns(models, x, threads):
    fns = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n, m in models.items():
            path = Path(tmp) / f"{n}.onnx"
            torch.onnx.export(m, x, str(path), input_names=["x"], output_names=["y"], opset_version=17)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            arr = x.numpy()
            ref = m(x).detach().numpy()
            assert abs(sess.run(None, {"x": arr})[0] - ref).max() < 1e-3, n
            fns[n] = lambda s=sess, a=arr: s.run(None, {"x": a})
    return fns


if __name__ == "__main__":
    cfg = load_config()
    lat = cfg["latency"]
    device = pick_device()
    data = load_tensors(cfg, torch.device("cpu"))
    x1, xb = data["test"][0][:1].clone(), data["test"][0][:lat["big_batch"]].clone()
    models = {n: load_model(cfg, n) for n in names(cfg)}
    cols = {}
    with tee_results("latency.txt"):
        print(env_line())
        print(f"{lat['rounds']} rounds x {lat['runs']} runs per model after {lat['warmup']} warm-up runs in every round; "
              f"models are measured in shuffled order in each round; dense kernels everywhere, pruned weights are "
              f"stored as zeros; no model files are converted to a sparse layout")
        bench = dict(warmup=lat["warmup"], rounds=lat["rounds"], runs=lat["runs"])
        for th in lat["threads"]:
            cols[f"cpu{th}_b1"] = measure({n: model_fn(m, x1) for n, m in models.items()}, threads=th, **bench)
        cols["ort1_b1"] = measure(ort_fns(models, x1, 1), **bench)
        cols[f"cpu8_b{lat['big_batch']}"] = measure({n: model_fn(m, xb) for n, m in models.items()}, warmup=3,
                                                    rounds=lat["rounds"], runs=10, threads=8)
        gm = {n: load_model(cfg, n).to(device) for n in models}
        cols["gpu_b1"] = measure({n: model_fn(m, x1.to(device)) for n, m in gm.items()}, cuda=True, **bench)
        cols[f"gpu_b{lat['big_batch']}"] = measure({n: model_fn(m, xb.to(device)) for n, m in gm.items()}, cuda=True,
                                                   warmup=10, rounds=lat["rounds"], runs=50)
        rows = []
        for n, m in models.items():
            row = {"model": n, "macs_m": total_macs(m) / 1e6, "eff_macs_m": effective_macs(m) / 1e6}
            for c, res in cols.items():
                for k, v in res[n].items():
                    row[f"{c}_{k}"] = round(v, 5)
            rows.append(row)
        print("\nmedian latency in ms, batch 1 (bN = batch of N images); in brackets: speed-up over the baseline")
        print(f"{'model':<16} {'MACs M':>7} {'nz MACs':>8} " + " ".join(f"{c:>16}" for c in cols))
        for r in rows:
            print(f"{r['model']:<16} {r['macs_m']:>7.2f} {r['eff_macs_m']:>8.2f} " + " ".join(
                f"{r[f'{c}_median']:>8.3f} ({rows[0][f'{c}_median'] / r[f'{c}_median']:4.2f}x)" for c in cols))
        print("\nspread: lowest / highest of the per-round medians, and p95 over all runs")
        for r in rows:
            print(f"{r['model']:<16} " + "  ".join(
                f"{c} {r[f'{c}_round_min']:.3f}/{r[f'{c}_round_max']:.3f} p95 {r[f'{c}_p95']:.3f}"
                for c in ("cpu1_b1", "cpu8_b1", "gpu_b1")))
    write_csv(rows, "latency.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"baseline": MUTED, "global": SERIES[0], "layer": SERIES[1], "structured": SERIES[2], "student": SERIES[6]}
    for ax, c, title in ((axes[0], "cpu1_b1", "PyTorch, 1 CPU thread, batch 1"),
                         (axes[1], "ort1_b1", "ONNX Runtime, 1 CPU thread, batch 1")):
        ns = [r["model"] for r in rows]
        ys = [r[f"{c}_median"] for r in rows]
        err = [[y - r[f"{c}_round_min"] for y, r in zip(ys, rows)], [r[f"{c}_round_max"] - y for y, r in zip(ys, rows)]]
        pos = list(range(len(ns)))[::-1]
        ax.barh(pos, ys, height=0.68, color=[colors[n.split("_")[0]] for n in ns], xerr=[
            [max(0, e) for e in err[0]], [max(0, e) for e in err[1]]], ecolor=INK, capsize=2, error_kw={"lw": 0.8})
        ax.set_yticks(pos)
        ax.set_yticklabels(ns, fontsize=9)
        ax.set_xlim(0, max(r[f"{c}_round_max"] for r in rows) * 1.2)
        for p, y, r in zip(pos, ys, rows):
            ax.annotate(f"{y:.2f}", (max(y, r[f"{c}_round_max"]), p), textcoords="offset points", xytext=(5, -3),
                        fontsize=8.5, color=INK)
        style_axes(ax, title, "median latency, ms (bars: range of per-round medians)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Zeroed weights do not change dense-kernel latency; smaller tensors do", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / "latency.png", dpi=150, facecolor=SURFACE)
