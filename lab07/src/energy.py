import random
import time

import torch

from common import env_line, load_config, mean_std, pick_device, tee_results, write_csv
from data import load_tensors
from engine import PowerMeter
from zoo import load_model

MODELS = ["baseline", "global_0.9", "layer_0.9", "structured_0.1", "structured_0.25", "structured_0.5",
          "student_plain", "student_kd"]
REPEATS = 3


def burn(model, x, seconds):
    n = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        while time.perf_counter() - t0 < seconds:
            model(x)
            torch.cuda.synchronize()
            n += 1
    return n


if __name__ == "__main__":
    cfg = load_config()
    e = cfg["energy"]
    device = pick_device()
    test_x = load_tensors(cfg, device)["test"][0]
    models = {n: load_model(cfg, n).to(device) for n in MODELS}
    res = {(n, b): [] for n in MODELS for b in (1, e["big_batch"])}
    rng = random.Random(0)
    with tee_results("energy.txt"):
        print(env_line())
        time.sleep(30)
        with PowerMeter() as idle:
            time.sleep(e["seconds"])
        print(f"GPU idle (display + desktop apps, after 30 s rest): {idle.mean_watts:.1f} W (counter "
              f"{idle.counter_joules / idle.seconds:.1f} W)")
        print(f"each measurement: 3 s warm-up, then {e['seconds']} s of synchronized inference under the NVML energy "
              f"counter; {REPEATS} repeats in shuffled order; gross = all GPU energy, net = gross - idle power x time\n")
        for rep in range(REPEATS):
            order = list(res)
            rng.shuffle(order)
            for n, b in order:
                burn(models[n], test_x[:b], 3)
                with PowerMeter() as pm:
                    k = burn(models[n], test_x[:b], e["seconds"])
                images = k * b
                res[(n, b)].append({"watts": pm.counter_joules / pm.seconds, "gross_mj": 1000 * pm.counter_joules / images,
                                    "net_mj": 1000 * (pm.counter_joules - idle.mean_watts * pm.seconds) / images,
                                    "images_s": images / pm.seconds})
        rows = []
        print(f"{'model':<16} {'batch':>5} {'W':>7} {'images/s':>10} {'gross mJ/image':>20} {'net mJ/image':>20}")
        for (n, b), rs in res.items():
            f = {k: mean_std([r[k] for r in rs]) for k in rs[0]}
            rows.append({"model": n, "batch": b, **{f"{k}_{s}": v[i] for k, v in f.items()
                                                    for i, s in enumerate(("mean", "std"))}})
            print(f"{n:<16} {b:>5} {f['watts'][0]:>7.1f} {f['images_s'][0]:>10,.0f} "
                  f"{f['gross_mj'][0]:>11.3f} +- {f['gross_mj'][1]:<6.3f} {f['net_mj'][0]:>11.3f} +- {f['net_mj'][1]:<6.3f}")
    write_csv(rows, "energy.csv")
