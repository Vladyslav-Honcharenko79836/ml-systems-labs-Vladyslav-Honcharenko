import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (INK, MARKERS, RESULTS_DIR, SERIES, SURFACE, env_line, load_config, pick_device,
                    style_axes, tee_results, with_overrides, write_csv)
from pipeline import run

EPOCHS = 10
OPTIMIZERS = {"adam": {"optim__name": "adam", "optim__lr": 1e-3},
              "sgd": {"optim__name": "sgd", "optim__lr": 0.05, "optim__momentum": 0.9}}
SCHEDULES = ["step", "cosine", "plateau"]
LABEL = {"adam": "Adam", "sgd": "SGD+momentum", "step": "StepLR", "cosine": "CosineAnnealingLR",
         "plateau": "ReduceLROnPlateau"}

if __name__ == "__main__":
    base = load_config()
    rows, curves = [], {}
    with tee_results("schedule_study.txt"):
        s = base["sched"]
        print(f"{env_line(pick_device(base['device']))} | batch {base['data']['batch_size']} | {EPOCHS} epochs each\n"
              f"Adam lr 1e-3; SGD lr 0.05 momentum 0.9; StepLR step {s['step_size']} gamma {s['gamma']}; "
              f"Cosine T_max {EPOCHS}; Plateau on val loss, factor {s['plateau_factor']}, "
              f"patience {s['plateau_patience']}")
        for opt_name, opt_over in OPTIMIZERS.items():
            for sched in SCHEDULES:
                name = f"sched-{opt_name}-{sched}"
                cfg = with_overrides(base, train__epochs=EPOCHS, sched__name=sched, run_name=name, **opt_over)
                res = run(cfg)
                h = res["history"]
                curves[(opt_name, sched)] = h
                best = max(h, key=lambda r: r["val_acc"])
                rows.append({"optimizer": opt_name, "schedule": sched, "lr_start": h[0]["lr"],
                             "lr_last_epoch": h[-1]["lr"], "final_train_loss": round(h[-1]["train_loss"], 4),
                             "final_val_loss": round(h[-1]["val_loss"], 4), "final_val_acc": round(h[-1]["val_acc"], 4),
                             "best_val_acc": round(best["val_acc"], 4), "best_epoch": best["epoch"],
                             "test_acc": round(res["test_acc"], 4), "mlflow_run_id": res["run_id"]})
        print(f"\n{'optimizer':<9} {'schedule':<8} {'lr ep1':>8} {'lr ep10':>9} {'train loss':>10} {'val loss':>8} "
              f"{'val acc':>7} {'best val':>8} {'@ep':>3} {'test acc':>8}")
        for r in rows:
            print(f"{r['optimizer']:<9} {r['schedule']:<8} {r['lr_start']:>8.0e} {r['lr_last_epoch']:>9.1e} "
                  f"{r['final_train_loss']:>10.4f} {r['final_val_loss']:>8.4f} {r['final_val_acc']:>7.4f} "
                  f"{r['best_val_acc']:>8.4f} {r['best_epoch']:>3} {r['test_acc']:>8.4f}")
    write_csv(rows, "schedule_study.csv")
    write_csv([{"optimizer": o, "schedule": s, **r} for (o, s), h in curves.items() for r in h],
              "schedule_curves.csv")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, (key, title) in enumerate([("val_loss", "Validation loss"), ("val_acc", "Validation accuracy, %"),
                                        ("lr", "Learning rate (log)")]):
        for row, opt_name in enumerate(OPTIMIZERS):
            ax = axes[row, col]
            for i, sched in enumerate(SCHEDULES):
                h = curves[(opt_name, sched)]
                ys = [r[key] * (100 if key == "val_acc" else 1) for r in h]
                ax.plot([r["epoch"] for r in h], ys, color=SERIES[i], marker=MARKERS[i], lw=2, ms=6,
                        label=LABEL[sched])
            if key == "lr":
                ax.set_yscale("log")
            style_axes(ax, f"{LABEL[opt_name]}: {title}", "epoch" if row == 1 else None)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for row in range(2):
        lo = min(min(r["val_acc"] for r in curves[(o, s)]) for o in OPTIMIZERS for s in SCHEDULES) * 100
        axes[row, 1].set_ylim(lo - 0.5, None)
    fig.suptitle("LR schedules x optimizers, 10-epoch budget, Fashion-MNIST CNN", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS_DIR / "schedule_study.png", dpi=150, facecolor=SURFACE)
