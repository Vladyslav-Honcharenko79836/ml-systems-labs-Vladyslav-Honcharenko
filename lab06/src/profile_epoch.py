import gc
import time
from collections import defaultdict

import torch
from torch import nn
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, schedule

from common import (RESULTS_DIR, env_line, load_config, pick_device, seed_everything, tee_results,
                    with_overrides, write_csv)
from data import build_data
from models import build_model
from pipeline import Tracker, build_optimizer, profiler_regions, timer_regions, train_epoch

LABELS = ["data_loading", "forward", "backward", "optimizer_step"]
SETTINGS = [("cpu", 0), ("cuda", 0), ("cuda", 4)]
CHUNK = 10


def profile_one(cfg):
    gc.collect()
    device = pick_device(cfg["device"])
    seed_everything(cfg["seed"], cfg["deterministic"])
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model.parameters())
    loader, _, _, _ = build_data(cfg, device)
    loss_fn = nn.CrossEntropyLoss()

    train_epoch(model, loader, opt, loss_fn, device)
    region, timer_s = timer_regions(device)
    t0 = time.perf_counter()
    train_epoch(model, loader, opt, loss_fn, device, region=region)
    timer_wall = time.perf_counter() - t0

    region_us, region_calls = defaultdict(float), defaultdict(int)
    step = {"us": 0.0, "n": 0}
    ops = defaultdict(lambda: [0.0, 0.0, 0])

    def collect(prof):
        for e in prof.events():
            if e.device_type != DeviceType.CPU:
                continue
            if e.name in LABELS:
                region_us[e.name] += e.cpu_time_total
                region_calls[e.name] += 1
            elif e.name.startswith("ProfilerStep"):
                step["us"] += e.cpu_time_total
                step["n"] += 1
        for e in prof.key_averages():
            if e.key not in LABELS and not e.key.startswith("ProfilerStep"):
                o = ops[e.key]
                o[0] += e.self_cpu_time_total
                o[1] += getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                o[2] += e.count

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=activities, schedule=schedule(wait=0, warmup=0, active=CHUNK, repeat=0),
                 on_trace_ready=collect) as prof:
        train_epoch(model, loader, opt, loss_fn, device, region=profiler_regions(device), on_step=prof.step)

    top = sorted(ops.items(), key=lambda kv: kv[1][1] if device.type == "cuda" else kv[1][0], reverse=True)
    lines = [f"{'op':<60} {'self CPU s':>10} {'self GPU s':>10} {'calls':>9}"]
    lines += [f"{k[:60]:<60} {v[0] / 1e6:>10.3f} {v[1] / 1e6:>10.3f} {v[2]:>9}" for k, v in top[:25]]
    prof_s = {k: region_us[k] / 1e6 for k in LABELS}
    return {"timer_wall": timer_wall, "timer_s": timer_s, "prof_s": prof_s, "calls": dict(region_calls),
            "steps": step["n"], "step_s": step["us"] / 1e6, "table": "\n".join(lines),
            "threads": torch.get_num_threads()}


if __name__ == "__main__":
    base = load_config()
    rows = []
    with tee_results("profile.txt"):
        print("torch.profiler (CPU + CUDA activities) over one full epoch of 430 steps, after one warm-up epoch.\n"
              "Each step is split into record_function regions data_loading / forward / backward / optimizer_step;\n"
              "every region ends with torch.cuda.synchronize(), so GPU time is charged to the region that launched it.\n"
              f"Events are collected in chunks of {CHUNK} steps (bounded memory). Profiler share = region time / sum of\n"
              "ProfilerStep durations. Check column: the same regions timed with perf_counter in an unprofiled epoch.")
        for dev, workers in SETTINGS:
            name = f"profile-{dev}-w{workers}"
            cfg = with_overrides(base, device=dev, data__num_workers=workers, run_name=name)
            device = pick_device(dev)
            seed_everything(cfg["seed"], cfg["deterministic"])
            print(f"\n== {name}: {env_line(device)}, batch {cfg['data']['batch_size']}, num_workers {workers}",
                  flush=True)
            r = profile_one(cfg)
            total, twall = r["step_s"], r["timer_wall"]
            print(f"profiled steps {r['steps']}, sum of step times {total:.2f} s | "
                  f"unprofiled epoch {twall:.2f} s | intra-op threads at the end {r['threads']}")
            print(f"{'region':<16} {'calls':>6} {'profiler s':>10} {'share':>7}   {'timer s':>8} {'share':>7}")
            row = {"setting": name, "device": dev, "num_workers": workers, "profiled_step_sum_s": round(total, 3),
                   "unprofiled_epoch_s": round(twall, 3)}
            for k in LABELS:
                ps, ts = r["prof_s"][k], r["timer_s"][k]
                print(f"{k:<16} {r['calls'][k]:>6} {ps:>10.3f} {ps / total:>7.1%}   {ts:>8.3f} {ts / twall:>7.1%}")
                row[f"{k}_share"] = round(ps / total, 4)
                row[f"{k}_timer_share"] = round(ts / twall, 4)
            po, to = total - sum(r["prof_s"].values()), twall - sum(r["timer_s"].values())
            print(f"{'other':<16} {'':>6} {po:>10.3f} {po / total:>7.1%}   {to:>8.3f} {to / twall:>7.1%}")
            row["other_share"] = round(po / total, 4)
            row["other_timer_share"] = round(to / twall, 4)
            rows.append(row)
            table_path = RESULTS_DIR / f"profile_ops_{dev}_w{workers}.txt"
            table_path.write_text(f"{name}: top-25 operators of the profiled epoch\n{r['table']}\n", encoding="utf-8")
            print(f"top-25 operators -> results/{table_path.name}", flush=True)

            tracker = Tracker(cfg, run_name=name)
            tracker.params({"device": dev, "data.num_workers": workers,
                            "data.batch_size": cfg["data"]["batch_size"], "kind": "profile"})
            tracker.metrics({f"share_{k}": row[f"{k}_share"] for k in LABELS} | {"epoch_s": twall})
            tracker.artifact(table_path)
            tracker.end()
    write_csv(rows, "profile.csv")
