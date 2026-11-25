import contextlib
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

LAB_DIR = Path(__file__).resolve().parents[1]
ROOT = LAB_DIR.parent
RESULTS_DIR = LAB_DIR / "results"
DEFAULT_THREADS = torch.get_num_threads()


def load_config():
    with open(LAB_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ckpt_path(cfg, name):
    d = ROOT / cfg["paths"]["ckpt_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.pt"


def seed_everything(seed, deterministic=True):
    torch.set_num_threads(DEFAULT_THREADS)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_line():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
    return (f"python {sys.version.split()[0]} | torch {torch.__version__} | GPU {gpu} | "
            f"CPU threads {torch.get_num_threads()}")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextlib.contextmanager
def tee_results(filename):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    old = sys.stdout
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        sys.stdout = _Tee(old, f)
        try:
            yield path
        finally:
            sys.stdout = old
    print(f"[saved to {path.relative_to(ROOT).as_posix()}]")


def write_csv(rows, filename):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(filename):
    with open(RESULTS_DIR / filename, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)
