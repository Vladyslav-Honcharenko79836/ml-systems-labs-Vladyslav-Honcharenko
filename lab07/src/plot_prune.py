import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import INK, MUTED, RESULTS_DIR, SERIES, SURFACE, read_csv, style_axes

STYLE = {"global": (SERIES[0], "o", "global magnitude"), "layer": (SERIES[1], "s", "per-layer magnitude"),
         "structured": (SERIES[2], "^", "structured (whole filters)")}


def load():
    rows = read_csv("prune_summary.csv")
    for r in rows:
        for k in r:
            if k != "kind":
                r[k] = float(r[k])
    return rows


def draw(ax, rows, kinds, xkey, ykey, skey, xscale=1.0, before=False, with_base=False):
    for kind in kinds:
        color, marker, label = STYLE[kind]
        g = [r for r in rows if r["kind"] == kind]
        pts = [(r[xkey] * xscale, r[ykey], r[skey]) for r in g]
        if with_base and kind == "structured":
            b = next(r for r in rows if r["kind"] == "global" and r["target"] == 0)
            pts = [(0.0 if xkey != "eff_macs_m" else b["eff_macs_m"], b["before_mean"], b["before_std"])] + pts
        ax.errorbar([a for a, _, _ in pts], [b for _, b, _ in pts], yerr=[c for _, _, c in pts], color=color,
                    marker=marker, lw=2, ms=8, capsize=3, label=label + (", before fine-tuning" if before else ", fine-tuned"),
                    ls="--" if before else "-", mfc=SURFACE if before else color)


if __name__ == "__main__":
    rows = load()
    base = next(r for r in rows if r["kind"] == "global" and r["target"] == 0)["before_mean"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    draw(ax, rows, ["global", "layer"], "target", "before_mean", "before_std", 100, before=True)
    draw(ax, rows, ["global", "layer"], "target", "after_mean", "after_std", 100)
    style_axes(ax, "Unstructured, before and after 3-epoch fine-tuning", "weight sparsity, %", "test accuracy, %")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    ax = axes[1]
    draw(ax, rows, ["global", "layer"], "target", "after_mean", "after_std", 100)
    ax.axhline(base, color=MUTED, lw=1, ls=":")
    ax.annotate("baseline, no fine-tuning", (2, base), textcoords="offset points", xytext=(0, -13), fontsize=8.5,
                color=MUTED)
    style_axes(ax, "Unstructured after fine-tuning (zoom)", "weight sparsity, %", "test accuracy, %")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    ax = axes[2]
    draw(ax, rows, ["structured"], "target", "before_mean", "before_std", 100, before=True, with_base=True)
    g = [r for r in rows if r["kind"] == "structured"]
    draw(ax, rows, ["structured"], "target", "after_mean", "after_std", 100)
    for r in g:
        ax.annotate(f"{r['after_mean']:.2f}", (100 * r["target"], r["after_mean"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
    ax.axhline(base, color=MUTED, lw=1, ls=":")
    style_axes(ax, "Structured: filters removed from every conv layer", "filters removed, %", "test accuracy, %")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Accuracy vs sparsity, Fashion-MNIST test set, mean ± std over 3 independently trained baselines",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "sparsity_curves.png", dpi=150, facecolor=SURFACE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    draw(ax, rows, ["global", "layer", "structured"], "param_reduction_pct", "after_mean", "after_std")
    ax.axhline(base, color=MUTED, lw=1, ls=":")
    style_axes(ax, "By parameters removed", "parameters removed, %", "test accuracy after fine-tuning, %")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax = axes[1]
    draw(ax, rows, ["global", "layer", "structured"], "eff_macs_m", "after_mean", "after_std")
    ax.axhline(base, color=MUTED, lw=1, ls=":")
    ax.set_xscale("log")
    ticks = [2, 5, 10, 20, 40, 60]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.invert_xaxis()
    style_axes(ax, "By MACs with a non-zero weight", "non-zero MACs, millions (log scale, fewer →)",
               "test accuracy after fine-tuning, %")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("The three pruning methods on common axes, mean ± std over 3 seeds", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "pruning_common_axes.png", dpi=150, facecolor=SURFACE)
