import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import INK, MUTED, RESULTS_DIR, SERIES, SURFACE, load_config, mean_std, read_csv, style_axes

VARIANTS = [("plain", "CE only", SERIES[0], "o"), ("kd", "KD, KL(S‖T)\n(manual)", SERIES[1], "s"),
            ("kd_forward", "KD, KL(T‖S)\n(Hinton)", SERIES[2], "^")]

if __name__ == "__main__":
    cfg = load_config()
    rows = read_csv("distill_runs.csv")
    data = read_csv("distill_data.csv")
    tune0, tune1 = read_csv("tune_distill.csv"), read_csv("tune_distill_clip.csv")
    teacher = next(100 * float(r["test_acc"]) for r in read_csv("baseline.csv")
                   if int(r["seed"]) == cfg["distill"]["teacher_seed"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    for (v, label, c, m) in VARIANTS:
        for t, ls, fill in ((tune0, "--", SURFACE), (tune1, "-", c)):
            lrs = sorted({float(r["lr"]) for r in t})
            ys = [mean_std([100 * float(r["val_acc"]) for r in t if r["variant"] == v and float(r["lr"]) == lr])[0]
                  for lr in lrs]
            ax.plot(lrs, ys, color=c, marker=m, ls=ls, mfc=fill, lw=2, ms=8,
                    label=label.replace("\n", " ") + (" + clipping" if ls == "-" else ""))
    ax.set_xscale("log")
    ax.set_xticks([0.0025, 0.005, 0.01, 0.02, 0.04])
    ax.set_xticklabels(["0.0025", "0.005", "0.01", "0.02", "0.04"])
    ax.minorticks_off()
    ax.set_ylim(86, 94)
    style_axes(ax, "Learning-rate search (validation, 2 seeds;\npoints below 86 % are collapsed runs)", "learning rate",
               "validation accuracy, %")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.95, fontsize=8, loc="lower center")

    ax = axes[1]
    ticks = []
    for i, (v, label, c, m) in enumerate(VARIANTS):
        ys = [100 * float(r["test_acc"]) for r in rows if r["variant"] == v]
        jit = np.linspace(-0.12, 0.12, len(ys))
        ax.scatter(i + jit, ys, color=c, marker=m, s=55, zorder=3)
        mean, sd = mean_std(ys)
        ax.hlines(mean, i - 0.25, i + 0.25, color=INK, lw=2, zorder=4)
        ticks.append(f"{label}\n{mean:.2f} ± {sd:.2f} %")
    ax.axhline(teacher, color=MUTED, ls=":", lw=1.2)
    ax.annotate(f"teacher {teacher:.2f}", (0.55, teacher), textcoords="offset points", xytext=(0, 4), fontsize=9,
                color=MUTED)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ticks)
    ax.set_xlim(-0.5, 2.6)
    style_axes(ax, "Student, 5 seeds, full training set", None, "test accuracy, %")

    ax = axes[2]
    for (v, label, c, m) in VARIANTS[:2]:
        imgs = sorted({int(r["images"]) for r in data})
        stats = [mean_std([100 * float(r["test_acc"]) for r in data if r["variant"] == v and int(r["images"]) == k])
                 for k in imgs]
        ax.errorbar(imgs, [s[0] for s in stats], yerr=[s[1] for s in stats], color=c, marker=m, lw=2, ms=8, capsize=3,
                    label=label.replace("\n", " "))
    ax.set_xscale("log")
    ax.set_xticks(imgs)
    ax.set_xticklabels([f"{k / 1000:g}k" for k in imgs])
    ax.minorticks_off()
    style_axes(ax, "Data study, 3 seeds, equal optimizer steps", "training images (log scale)", "test accuracy, %")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "distillation.png", dpi=150, facecolor=SURFACE)
