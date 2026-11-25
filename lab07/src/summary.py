import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (INK, MUTED, RESULTS_DIR, SERIES, SURFACE, load_config, mean_std, read_csv, style_axes,
                    tee_results, write_csv)

COLS = [("baseline", "baseline (teacher)"), ("global_0.9", "global 90 %"), ("layer_0.9", "per-layer 90 %"),
        ("structured_0.25", "filters −25 %"), ("structured_0.5", "filters −50 %"), ("student_plain", "student, CE"),
        ("student_kd", "student, KD")]


def acc_stats():
    base = read_csv("baseline.csv")
    out = {"baseline": mean_std([100 * float(r["test_acc"]) for r in base])}
    for r in read_csv("prune_summary.csv"):
        out[f"{r['kind']}_{float(r['target'])}"] = (float(r["after_mean"]), float(r["after_std"]))
    for r in read_csv("distill_summary.csv"):
        out[f"student_{r['variant']}"] = (100 * float(r["test_acc_mean"]), 100 * float(r["test_acc_std"]))
    return out


if __name__ == "__main__":
    cfg = load_config()
    n = cfg["data"]["n_train"]
    tdp = cfg["energy"]["cpu_tdp_w"]
    acc = acc_stats()
    lat = {r["model"]: r for r in read_csv("latency.csv")}
    sto = {r["model"]: r for r in read_csv("storage.csv")}
    en = {(r["model"], int(r["batch"])): r for r in read_csv("energy.csv")}
    base = read_csv("baseline.csv")
    base_s, _ = mean_std([float(r["train_s"]) for r in base])
    base_j, _ = mean_std([float(r["train_joules"]) for r in base])
    pr = read_csv("prune_runs.csv")
    dis = {r["variant"]: r for r in read_csv("distill_summary.csv")}
    data_rows = read_csv("distill_data.csv")
    sparse_fmts = ["CSR .pt", "bitmask .npz", "COO .pt"]
    E, FT, DE = cfg["baseline"]["epochs"], cfg["prune"]["finetune_epochs"], cfg["distill"]["epochs"]
    big = cfg["energy"]["big_batch"]

    def ft(model, key):
        kind, t = model.split("_")
        return mean_std([float(r[key]) for r in pr if r["kind"] == kind and float(r["target"]) == float(t)])[0]

    def train_cost(m):
        if m == "baseline":
            return E * n, 0, base_s, base_j
        if m == "student_plain":
            return DE * n, 0, float(dis["plain"]["train_s_mean"]), float(dis["plain"]["train_joules_mean"])
        if m == "student_kd":
            return (DE * n, (E + 1) * n, float(dis["kd"]["train_s_mean"]) + base_s,
                    float(dis["kd"]["train_joules_mean"]) + base_j)
        return FT * n, E * n, ft(m, "ft_seconds") + base_s, ft(m, "ft_joules") + base_j

    def frac_acc(variant, images):
        g = [100 * float(r["test_acc"]) for r in data_rows if r["variant"] == variant and int(r["images"]) == images]
        return mean_std(g)

    table = []

    def add(pillar, metric, fn):
        table.append({"pillar": pillar, "metric": metric, **{c: fn(c) for c, _ in COLS}})

    few = round(min(cfg["distill"]["data_fractions"]) * n)
    add("data", "labelled training images", lambda m: f"{n:,}")
    add("data", "image passes, own training (k)", lambda m: f"{train_cost(m)[0] / 1000:,.0f}")
    add("data", "image passes, incl. teacher / pre-training (k)", lambda m: f"{sum(train_cost(m)[:2]) / 1000:,.0f}")
    add("data", f"test acc with only {few:,} images, %",
        lambda m: "{:.2f} ± {:.2f}".format(*frac_acc(m.split("_")[1], few)) if m.startswith("student") else "—")
    add("model", "test accuracy, % (mean ± std over seeds)", lambda m: "{:.2f} ± {:.2f}".format(*acc[m]))
    add("model", "parameters", lambda m: f"{int(sto[m]['params']):,}")
    add("model", "non-zero parameters", lambda m: f"{int(sto[m]['nonzero']):,}")
    add("model", "file, dense .pt, MB", lambda m: f"{int(sto[m]['dense .pt']) / 1e6:.3f}")
    add("model", "file, best sparse format, MB", lambda m: f"{min(int(sto[m][f]) for f in sparse_fmts) / 1e6:.3f}")
    add("compute", "MACs executed by dense kernels, M", lambda m: f"{float(lat[m]['macs_m']):.2f}")
    add("compute", "MACs with a non-zero weight, M", lambda m: f"{float(lat[m]['eff_macs_m']):.2f}")
    for c, label in (("cpu1_b1", "CPU 1 thread, batch 1, ms"), ("cpu8_b1", "CPU 8 threads, batch 1, ms"),
                     ("ort1_b1", "ONNX Runtime 1 thread, batch 1, ms"), ("gpu_b1", "GPU, batch 1, ms")):
        add("compute", label, lambda m, c=c: f"{float(lat[m][f'{c}_median']):.3f}")
    add("compute", "GPU training time, incl. teacher / pre-training, s", lambda m: f"{train_cost(m)[2]:,.0f}")
    add("energy", "GPU training energy, incl. teacher / pre-training, kJ", lambda m: f"{train_cost(m)[3] / 1000:.1f}")
    add("energy", "GPU inference, batch 1, mJ/image (gross)", lambda m: f"{float(en[(m, 1)]['gross_mj_mean']):.1f}")
    add("energy", f"GPU inference, batch {big}, mJ/image (gross)",
        lambda m: f"{float(en[(m, big)]['gross_mj_mean']):.2f}")
    add("energy", f"CPU inference, upper bound {tdp} W × 8-thread latency, mJ/image",
        lambda m: f"{tdp * float(lat[m]['cpu8_b1_median']):.0f}")

    with tee_results("summary.txt"):
        print("accuracy: mean over seeds; sizes, latency and energy: seed-42 models; training cost: mean over seeds\n")
        for row in table:
            print(f"{row['pillar']:<8} {row['metric']:<58} " + " ".join(f"{row[c]:>18}" for c, _ in COLS))
        print("\ncolumns: " + " | ".join(label for _, label in COLS))
    write_csv(table, "summary.csv")

    md = ["| pillar | metric | " + " | ".join(label for _, label in COLS) + " |",
          "|---|---|" + "---|" * len(COLS)]
    prev = None
    for row in table:
        md.append(f"| {'**' + row['pillar'] + '**' if row['pillar'] != prev else ''} | {row['metric']} | "
                  + " | ".join(row[c] for c, _ in COLS) + " |")
        prev = row["pillar"]
    (RESULTS_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    pal = {"baseline": MUTED, "global": SERIES[0], "layer": SERIES[1], "structured": SERIES[2], "student": SERIES[6]}
    mk = {"baseline": "*", "global": "o", "layer": "s", "structured": "^", "student": "D"}
    names = {"baseline": "baseline", "global": "global pruning", "layer": "per-layer pruning",
             "structured": "structured pruning", "student": "student"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, xfn, xlabel in (
            (axes[0], lambda m: float(lat[m]["cpu1_b1_median"]), "CPU latency, 1 thread, batch 1, ms"),
            (axes[1], lambda m: min(int(sto[m][f]) for f in sparse_fmts + ["dense .pt"]) / 1e6,
             "smallest file over the formats, MB (log scale)")):
        seen = set()
        for m in lat:
            fam = m.split("_")[0]
            ax.errorbar(xfn(m), acc[m][0], yerr=acc[m][1], color=pal[fam], marker=mk[fam], ms=9, lw=0, elinewidth=1.2,
                        capsize=2, mec=SURFACE, label=names[fam] if fam not in seen else None)
            seen.add(fam)
            if m in ("baseline", "global_0.9", "global_0.95", "layer_0.95", "structured_0.5", "student_plain",
                     "student_kd"):
                label = {"student_plain": "student CE", "student_kd": "student KD", "baseline": "baseline"}.get(m, m.replace("_", " "))
                dy = 16 if m == "student_plain" else -14 if m in ("student_kd", "structured_0.5", "global_0.95", "layer_0.95") else 7
                ax.annotate(label, (xfn(m), acc[m][0]), textcoords="offset points", xytext=(7, dy), fontsize=8.5,
                            color=INK)
        if "log" in xlabel:
            ax.set_xscale("log")
            ax.set_xticks([0.2, 0.5, 1, 2, 4])
            ax.set_xticklabels(["0.2", "0.5", "1", "2", "4"])
            ax.minorticks_off()
            ax.set_xlim(0.25, 6.5)
        else:
            ax.set_xlim(0, 4.6)
        style_axes(ax, None, xlabel, "test accuracy, %")
        ax.legend(frameon=False, fontsize=9, loc="lower left" if "ms" in xlabel else "lower right")
    fig.suptitle("Accuracy against real latency and real file size", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS_DIR / "pareto.png", dpi=150, facecolor=SURFACE)
