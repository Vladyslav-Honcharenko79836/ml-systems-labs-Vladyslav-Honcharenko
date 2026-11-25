import threading
import time

import pynvml
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


class PowerMeter:
    def __init__(self, interval=0.02):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval

    def _read(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self._read())
            time.sleep(self.interval)

    def __enter__(self):
        self.samples = []
        self._stop = threading.Event()
        self._counter0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        self.counter_joules = (pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - self._counter0) / 1000.0
        self._stop.set()
        self._thread.join()
        self.mean_watts = sum(self.samples) / max(1, len(self.samples))
        self.joules = self.mean_watts * self.seconds
        return False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def logits_of(model, x, batch=1000):
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


@torch.no_grad()
def evaluate(model, data):
    x, y = data
    out = logits_of(model, x)
    return F.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item()


def ce_loss(out, y, extra):
    return F.cross_entropy(out, y)


def train_model(model, batches, epochs, lr, momentum, weight_decay, device, loss_fn=ce_loss,
                val=None, after_step=None, log=None):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs * len(batches))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        n = 0
        for x, y, extra in batches:
            out = model(x)
            loss = loss_fn(out, y, extra)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if after_step is not None:
                after_step()
            loss_sum += loss.detach() * len(y)
            correct += (out.detach().argmax(1) == y).sum()
            n += len(y)
        sync(device)
        row = {"epoch": epoch, "train_loss": loss_sum.item() / n, "train_acc": correct.item() / n,
               "epoch_s": time.perf_counter() - t0}
        if val is not None:
            row["val_loss"], row["val_acc"] = evaluate(model, val)
        history.append(row)
        if log:
            log(f"  epoch {epoch:>2}  train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f}"
                + (f"  val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f}" if val is not None else "")
                + f"  {row['epoch_s']:.1f} s")
    return history
