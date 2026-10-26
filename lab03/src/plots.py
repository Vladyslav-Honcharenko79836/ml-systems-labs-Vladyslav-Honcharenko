"""Lab 3 step 5: accuracy vs parameters and vs latency, plus access-pattern metrics for the cost discussion.

Reads results/accuracy.csv, flops.csv and latency.csv (steps 3-4); writes tradeoffs.png and tradeoffs.txt/csv.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import RESULTS_DIR  # noqa: E402

# Categorical slots 1-3 of the dataviz reference palette (validated: CVD dE >= 9.2 all pairs);
# each model also gets its own marker shape and a direct label, so identity is never color-only.
STYLE = {"MLP": ("#2a78d6", "o"), "CNN": ("#eb6834", "s"), "Attention": ("#1baf7a", "^")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"


def read_csv(name):
    with open(RESULTS_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def build_table():
    acc = {r["model"]: r for r in read_csv("accuracy.csv")}
    flops = {r["model"]: int(r["hand_flops"]) for r in read_csv("flops.csv")}
    lat = {(r["model"], int(r["batch"])): float(r["median_ms"]) for r in read_csv("latency.csv")}
    rows = []
    for m in acc:
        params, f = int(acc[m]["params"]), flops[m]
        l1, l128 = lat[(m, 1)], lat[(m, 128)]
        rows.append({
            "model": m,
            "params": params,
            "test_acc": float(acc[m]["test_acc"]),
            "flops_per_image": f,
            "macs_per_weight": round(f / 2 / params, 1),  # how many times each weight is reused per image
            "weight_kb_fp32": round(params * 4 / 1024, 1),
            "latency_b1_ms": l1,
            "latency_b128_ms": l128,
            "per_image_b128_ms": round(l128 / 128, 5),
            "batching_speedup": round(l1 * 128 / l128, 1),
            "gflops_b1": round(f / (l1 * 1e-3) / 1e9, 2),
            "gflops_b128": round(f * 128 / (l128 * 1e-3) / 1e9, 1),
        })
    return rows


def style_axes(ax, xlabel):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both")


def scatter_panel(ax, rows, key, xlabel, title, fmt):
    for r in rows:
        color, marker = STYLE[r["model"]]
        x, y = r[key], r["test_acc"] * 100
        ax.scatter(x, y, s=110, color=color, marker=marker, edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(f"{r['model']}\n{fmt(x)}, {y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(10, -4), fontsize=9, color=INK, va="center")
    style_axes(ax, xlabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    lo, hi = min(r[key] for r in rows), max(r[key] for r in rows)
    ax.set_xlim(lo / 1.6, hi * 3.2)  # room for the right-hand labels


if __name__ == "__main__":
    rows = build_table()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    scatter_panel(axes[0], rows, "params", "parameters (log)", "Accuracy vs parameters",
                  lambda v: f"{v / 1000:.0f}k params")
    scatter_panel(axes[1], rows, "latency_b1_ms", "median latency, batch 1, ms (log)",
                  "Accuracy vs latency, batch 1", lambda v: f"{v:.2f} ms")
    scatter_panel(axes[2], rows, "latency_b128_ms", "median latency, batch 128, ms (log)",
                  "Accuracy vs latency, batch 128", lambda v: f"{v:.2f} ms")
    axes[0].set_ylabel("test accuracy, %", color=INK)
    axes[0].set_ylim(85.0, 90.0)
    handles = [plt.Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], linestyle="", markersize=9,
                          markeredgecolor="white", markeredgewidth=1.5, label=m) for m in STYLE]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Fashion-MNIST, 5 epochs, CPU (8 threads): higher and further left is better",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS_DIR / "tradeoffs.png", dpi=150, facecolor="white")

    cols = list(rows[0])
    with open(RESULTS_DIR / "tradeoffs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = ["  ".join(f"{c:>17}" for c in cols)]
    lines += ["  ".join(f"{str(r[c]):>17}" for c in cols) for r in rows]
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "tradeoffs.txt").write_text(text + "\n")
