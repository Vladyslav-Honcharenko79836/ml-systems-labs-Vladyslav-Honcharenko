import contextlib
import csv
import os
import random
import time

import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd.profiler import record_function
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from common import ROOT, flatten, pick_device, resolve, seed_everything
from data import build_data
from models import build_model


def build_optimizer(cfg, params):
    o = cfg["optim"]
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "sgd":
        return torch.optim.SGD(params, lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])
    raise ValueError(o["name"])


def build_scheduler(cfg, opt):
    s = cfg["sched"]
    if s["name"] == "none":
        return None
    if s["name"] == "step":
        return StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    if s["name"] == "cosine":
        return CosineAnnealingLR(opt, T_max=s["t_max"] or cfg["train"]["epochs"])
    if s["name"] == "plateau":
        return ReduceLROnPlateau(opt, mode="min", factor=s["plateau_factor"], patience=s["plateau_patience"])
    raise ValueError(s["name"])


def no_regions(name):
    return contextlib.nullcontext()


def profiler_regions(device):
    @contextlib.contextmanager
    def region(name):
        with record_function(name):
            yield
            sync(device)
    return region


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def evaluate(model, data, loss_fn, batch=1000):
    model.eval()
    x, y = data
    loss_sum, correct = 0.0, 0
    device = next(model.parameters()).device
    for i in range(0, len(y), batch):
        xb, yb = x[i:i + batch].to(device), y[i:i + batch].to(device)
        out = model(xb)
        loss_sum += loss_fn(out, yb).item() * len(out)
        correct += (out.argmax(1) == yb).sum().item()
    return loss_sum / len(y), correct / len(y)


def train_epoch(model, loader, opt, loss_fn, device, accum=1, region=no_regions, on_step=None):
    model.train()
    n_total = len(loader.dataset)
    group = loader.batch_size * accum
    n_batches = len(loader)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    group_loss = torch.zeros((), device=device)
    step_losses = []
    opt.zero_grad(set_to_none=True)
    it = iter(loader)
    for i in range(n_batches):
        with region("data_loading"):
            x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        with region("forward"):
            out = model(x)
            loss = loss_fn(out, y)
        weight = len(y) / min(group, n_total - (i // accum) * group)
        with region("backward"):
            (loss * weight).backward()
        group_loss += loss.detach() * weight
        loss_sum += loss.detach() * len(y)
        correct += (out.argmax(1) == y).sum()
        if (i + 1) % accum == 0 or i + 1 == n_batches:
            with region("optimizer_step"):
                opt.step()
                opt.zero_grad(set_to_none=True)
            step_losses.append(group_loss.clone())
            group_loss.zero_()
        if on_step is not None:
            on_step()
    sync(device)
    return loss_sum.item() / n_total, correct.item() / n_total, torch.stack(step_losses).tolist()


class Tracker:
    def __init__(self, cfg, run_id=None, run_name=None):
        self.enabled = cfg["mlflow"]["enabled"]
        self.run_id = run_id
        if not self.enabled:
            return
        import mlflow
        self.mlflow = mlflow
        store = resolve(cfg["paths"]["mlflow_dir"])
        store.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + (store / "mlflow.db").as_posix())
        if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
            mlflow.create_experiment(cfg["experiment"], artifact_location=(store / "artifacts").as_uri())
        mlflow.set_experiment(cfg["experiment"])
        if run_id:
            run = mlflow.start_run(run_id=run_id)
        else:
            run = mlflow.start_run(run_name=run_name or cfg["run_name"])
        self.run_id = run.info.run_id

    def params(self, d):
        if self.enabled:
            self.mlflow.log_params({k: ("null" if v is None else v) for k, v in d.items()})

    def tags(self, d):
        if self.enabled:
            self.mlflow.set_tags(d)

    def metrics(self, d, step=None):
        if self.enabled:
            self.mlflow.log_metrics(d, step=step)

    def artifact(self, path, subdir=None):
        if self.enabled:
            self.mlflow.log_artifact(str(path), subdir)

    def end(self, status="FINISHED"):
        if self.enabled:
            self.mlflow.end_run(status=status)


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def atomic_save(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_text(text, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(cfg, verbose=True):
    device = pick_device(cfg["device"])
    seed_everything(cfg["seed"], cfg["deterministic"])
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model.parameters())
    sched = build_scheduler(cfg, opt)
    train_dl, sampler_gen, val_set, test_set = build_data(cfg, device)
    loss_fn = nn.CrossEntropyLoss()
    epochs, accum = cfg["train"]["epochs"], cfg["train"]["accum_steps"]
    name = cfg["run_name"]

    run_dir = resolve(cfg["paths"]["runs_dir"]) / name
    history, step_losses, best_val, start_epoch, run_id = [], [], -1.0, 1, None
    if cfg["train"]["resume"]:
        ckpt_path = resolve(cfg["train"]["resume"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if sched is not None:
            sched.load_state_dict(ckpt["scheduler"])
        sampler_gen.set_state(ckpt["sampler_gen"])
        set_rng_state(ckpt["rng"])
        history, step_losses, best_val = ckpt["history"], ckpt["step_losses"], ckpt["best_val_acc"]
        start_epoch, run_id = ckpt["epoch"] + 1, ckpt["mlflow_run_id"]
        run_dir = ckpt_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    tracker = Tracker(cfg, run_id)
    if start_epoch == 1:
        tracker.params(flatten(cfg))
        tracker.tags({"device": str(device), "torch": torch.__version__,
                      "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "-"})
    else:
        tracker.tags({"resumed_from_epoch": start_epoch - 1})
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    peak_mb = 0.0
    if verbose:
        what = "start" if start_epoch == 1 else f"resume after epoch {start_epoch - 1}"
        print(f"[{name}] {what}, run dir {run_dir.relative_to(ROOT).as_posix()}", flush=True)

    status = "KILLED"
    try:
        for epoch in range(start_epoch, epochs + 1):
            lr = opt.param_groups[0]["lr"]
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            train_loss, train_acc, steps = train_epoch(model, train_dl, opt, loss_fn, device, accum)
            t_train = time.perf_counter() - t0
            if device.type == "cuda":
                peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 2**20)
            val_loss, val_acc = evaluate(model, val_set, loss_fn)
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_loss)
            elif sched is not None:
                sched.step()
            row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc, "epoch_time_s": t_train,
                   "samples_per_s": len(train_dl.dataset) / t_train}
            history.append(row)
            step_losses += [{"epoch": epoch, "step": len(step_losses) + 1 + k, "loss": v}
                            for k, v in enumerate(steps)]
            tracker.metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict() if sched is not None else None,
                     "sampler_gen": sampler_gen.get_state(), "rng": rng_state(),
                     "history": history, "step_losses": step_losses,
                     "best_val_acc": max(best_val, val_acc), "config": cfg, "mlflow_run_id": tracker.run_id}
            if val_acc > best_val:
                best_val = val_acc
                atomic_save(state, run_dir / "best.pt")
            atomic_save(state, run_dir / "last.pt")
            atomic_text(str(epoch), run_dir / "last_epoch.txt")
            write_rows(history, run_dir / "history.csv")
            if verbose:
                print(f"[{name}] epoch {epoch:>2}  lr {lr:.2e}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
                      f"val loss {val_loss:.4f} acc {val_acc:.4f}  {t_train:.1f}s  "
                      f"{row['samples_per_s']:.0f} samples/s", flush=True)

        test_loss, test_acc = evaluate(model, test_set, loss_fn)
        final = {"test_loss": test_loss, "test_acc": test_acc, "best_val_acc": best_val, "peak_mem_mb": peak_mb}
        tracker.metrics(final)
        write_rows(step_losses, run_dir / "steps.csv")
        for f in ("config.yaml", "history.csv", "steps.csv"):
            tracker.artifact(run_dir / f)
        if cfg["mlflow"]["log_checkpoint"]:
            tracker.artifact(run_dir / "best.pt", "checkpoints")
            tracker.artifact(run_dir / "last.pt", "checkpoints")
        status = "FINISHED"
        if verbose:
            print(f"[{name}] test loss {test_loss:.4f} acc {test_acc:.4f}  best val acc {best_val:.4f}  "
                  f"peak GPU mem (training) {peak_mb:.1f} MB", flush=True)
        return {"history": history, "step_losses": step_losses, "run_id": tracker.run_id,
                "run_dir": run_dir, "model": model, **final}
    finally:
        tracker.end(status)


def timer_regions(device):
    totals = {}

    @contextlib.contextmanager
    def region(name):
        t0 = time.perf_counter()
        yield
        sync(device)
        totals[name] = totals.get(name, 0.0) + time.perf_counter() - t0
    return region, totals
