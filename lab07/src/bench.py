import random
import time

import numpy as np
import torch

from common import DEFAULT_THREADS


def _time_runs(fn, runs, cuda=False):
    out = np.empty(runs)
    for i in range(runs):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        out[i] = time.perf_counter() - t0
    return out * 1000.0


def measure(fns, warmup, rounds, runs, threads=None, cuda=False, seed=0):
    rng = random.Random(seed)
    samples = {k: [] for k in fns}
    if threads is not None:
        torch.set_num_threads(threads)
    with torch.inference_mode():
        for _ in range(rounds):
            order = list(fns)
            rng.shuffle(order)
            for k in order:
                _time_runs(fns[k], warmup, cuda)
                samples[k].append(_time_runs(fns[k], runs, cuda))
    torch.set_num_threads(DEFAULT_THREADS)
    out = {}
    for k, parts in samples.items():
        ms = np.concatenate(parts)
        meds = [float(np.median(p)) for p in parts]
        out[k] = {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                  "std": float(ms.std(ddof=1)), "round_min": min(meds), "round_max": max(meds)}
    return out


def model_fn(model, x):
    model.eval()
    return lambda: model(x)
