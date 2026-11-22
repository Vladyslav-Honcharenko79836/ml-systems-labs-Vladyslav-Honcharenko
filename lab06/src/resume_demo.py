import csv
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (INK, ROOT, RESULTS_DIR, SERIES, SURFACE, load_config, resolve, style_axes, tee_results,
                    write_csv)

TRAIN = [sys.executable, str(ROOT / "lab06" / "src" / "train.py")]
EPOCHS = 10
KILL_AFTER = 3
KEYS = ["train_loss", "train_acc", "val_loss", "val_acc", "lr"]


def read_history(run_dir):
    with open(run_dir / "history.csv", newline="") as f:
        return [{k: (int(v) if k == "epoch" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def train(*args):
    p = subprocess.run(TRAIN + list(args), cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stdout + p.stderr)
    return p


def print_log(proc_out):
    for line in proc_out.splitlines():
        if line.startswith("["):
            print("    " + line)


if __name__ == "__main__":
    runs_dir = resolve(load_config()["paths"]["runs_dir"])
    ref_dir, int_dir = runs_dir / "resume-reference", runs_dir / "resume-interrupted"
    for d in (ref_dir, int_dir):
        for f in ("last.pt", "best.pt", "last_epoch.txt", "history.csv"):
            (d / f).unlink(missing_ok=True)

    with tee_results("resume_demo.txt"):
        print(f"1) reference: uninterrupted run, {EPOCHS} epochs")
        p = train("--set", "run_name=resume-reference", f"train.epochs={EPOCHS}")
        print_log(p.stdout)

        print(f"\n2) same config in a child process, killed (TerminateProcess) once the epoch-{KILL_AFTER} "
              f"checkpoint is on disk")
        proc = subprocess.Popen(TRAIN + ["--set", "run_name=resume-interrupted", f"train.epochs={EPOCHS}"],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        marker = int_dir / "last_epoch.txt"
        while proc.poll() is None:
            if marker.exists() and marker.read_text().strip() == str(KILL_AFTER):
                t_kill = time.perf_counter()
                proc.kill()
                break
            time.sleep(0.02)
        out, _ = proc.communicate()
        print_log(out)
        print(f"    process killed, exit code {proc.returncode}; checkpoint on disk is from epoch "
              f"{marker.read_text().strip()}; epoch {KILL_AFTER + 1} was in progress and is lost")
        before = read_history(int_dir)

        print(f"\n3) resume: train.py --resume {(int_dir / 'last.pt').relative_to(ROOT).as_posix()}")
        p = train("--resume", str(int_dir / "last.pt"))
        print_log(p.stdout)

        ref, res = read_history(ref_dir), read_history(int_dir)
        print(f"\n{'epoch':>5} {'source':<12} " + " ".join(f"{k:>11}" for k in KEYS) + "   max |diff| vs reference")
        rows = []
        for a, b in zip(ref, res):
            src = "interrupted" if b["epoch"] <= KILL_AFTER else "resumed"
            diff = max(abs(a[k] - b[k]) for k in KEYS)
            rows.append({"epoch": b["epoch"], "source": src, **{k: b[k] for k in KEYS},
                         **{f"ref_{k}": a[k] for k in KEYS}, "max_abs_diff": diff})
            print(f"{b['epoch']:>5} {src:<12} " + " ".join(f"{b[k]:>11.6f}" for k in KEYS) + f"   {diff:.1e}")
        print(f"epochs in resumed history: {len(res)} (expected {EPOCHS}); "
              f"largest difference from the uninterrupted run: {max(r['max_abs_diff'] for r in rows):.1e}")
        print("the resumed process logs into the same MLflow run (run id stored in the checkpoint)")
    write_csv(rows, "resume_demo.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title in ((axes[0], "train_loss", "Training loss"), (axes[1], "val_acc", "Validation accuracy, %")):
        scale = 100 if key == "val_acc" else 1
        ax.plot([r["epoch"] for r in ref], [r[key] * scale for r in ref], color=SERIES[0], lw=6, alpha=0.35,
                label="uninterrupted reference")
        first = [r for r in res if r["epoch"] <= KILL_AFTER]
        second = [r for r in res if r["epoch"] > KILL_AFTER]
        ax.plot([r["epoch"] for r in first], [r[key] * scale for r in first], color=SERIES[1], marker="o",
                lw=2, ms=8, label=f"run killed after epoch {KILL_AFTER}")
        ax.plot([r["epoch"] for r in second], [r[key] * scale for r in second], color=SERIES[2], marker="s",
                lw=2, ms=8, label="resumed from checkpoint")
        ax.axvline(KILL_AFTER + 0.5, color=INK, lw=1, ls=":")
        ax.annotate("kill + resume", (KILL_AFTER + 0.5, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", fontsize=9, color=INK)
        style_axes(ax, title, "epoch")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Interrupt at epoch 3 and resume: curves continue on the reference", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "resume_demo.png", dpi=150, facecolor=SURFACE)
