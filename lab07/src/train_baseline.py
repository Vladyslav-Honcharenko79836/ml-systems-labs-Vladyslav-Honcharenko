import time

import torch
from thop import profile
from torchinfo import summary

from bench import measure, model_fn
from common import (ckpt_path, env_line, load_config, mean_std, pick_device, seed_everything, tee_results,
                    write_csv)
from data import Batches, load_tensors
from engine import PowerMeter, evaluate, train_model
from models import layer_macs, make_teacher, n_params, total_macs, weight_layers

if __name__ == "__main__":
    cfg = load_config()
    b, lat = cfg["baseline"], cfg["latency"]
    device = pick_device()
    data = load_tensors(cfg, device)
    rows, hist_rows = [], []
    with tee_results("baseline.txt"):
        print(env_line())
        model = make_teacher(cfg)
        print(summary(model, (1, 1, 28, 28), verbose=0, device="cpu",
                      col_names=("output_size", "num_params", "mult_adds")))
        macs = layer_macs(model)
        thop_macs, _ = profile(make_teacher(cfg).eval(), (torch.randn(1, 1, 28, 28),), verbose=False)
        print(f"\n{'layer':<6} {'weights':>9} {'% of weights':>13} {'MACs':>12} {'% of MACs':>10}")
        wl = weight_layers(model)
        nw = sum(m.weight.numel() for _, m in wl)
        for name, m in wl:
            print(f"{name:<6} {m.weight.numel():>9,} {100 * m.weight.numel() / nw:>12.1f}% {macs[name]:>12,} "
                  f"{100 * macs[name] / total_macs(model):>9.1f}%")
        print(f"{'total':<6} {nw:>9,} {'':>13} {total_macs(model):>12,}   (thop {int(thop_macs):,}; "
              f"all parameters with biases {n_params(model):,})")

        for seed in cfg["seeds"]:
            seed_everything(seed, cfg["deterministic"])
            model = make_teacher(cfg).to(device)
            batches = Batches(*data["train"], cfg["data"]["batch_size"], seed)
            print(f"\nbaseline seed {seed}: SGD lr {b['lr']} momentum {b['momentum']} wd {b['weight_decay']}, "
                  f"cosine, {b['epochs']} epochs, batch {cfg['data']['batch_size']}, dropout {b['dropout']}")
            t0 = time.perf_counter()
            with PowerMeter() as pm:
                hist = train_model(model, batches, b["epochs"], b["lr"], b["momentum"], b["weight_decay"], device,
                                   val=data["val"], log=print)
            wall = time.perf_counter() - t0
            test_loss, test_acc = evaluate(model, data["test"])
            path = ckpt_path(cfg, f"baseline_s{seed}")
            torch.save(model.state_dict(), path)
            print(f"  test loss {test_loss:.4f}  test acc {test_acc:.4f}  wall {wall:.1f} s  "
                  f"GPU {pm.mean_watts:.1f} W, {pm.joules:.0f} J (NVML counter {pm.counter_joules:.0f} J)")
            hist_rows += [{"seed": seed, **r} for r in hist]
            rows.append({"seed": seed, "params": n_params(model), "macs": total_macs(model),
                         "train_acc": hist[-1]["train_acc"], "val_acc": hist[-1]["val_acc"],
                         "test_loss": test_loss, "test_acc": test_acc, "file_bytes": path.stat().st_size,
                         "train_s": wall, "train_watts": pm.mean_watts, "train_joules": pm.joules,
                         "train_joules_counter": pm.counter_joules})
        m, s = mean_std([r["test_acc"] for r in rows])
        print(f"\ntest acc over seeds {cfg['seeds']}: {m:.4f} +- {s:.4f}")
        print(f"parameters {rows[0]['params']:,}  file {rows[0]['file_bytes']:,} bytes "
              f"({rows[0]['file_bytes'] / 1e6:.3f} MB, .pt state_dict float32)")

        model = make_teacher(cfg)
        model.load_state_dict(torch.load(ckpt_path(cfg, f"baseline_s{cfg['seeds'][0]}"), map_location="cpu"))
        x = data["test"][0][:1].cpu()
        print(f"\nlatency, batch 1, {lat['rounds']} rounds x {lat['runs']} runs after {lat['warmup']} warm-up runs")
        for th in lat["threads"]:
            r = measure({"b": model_fn(model, x)}, lat["warmup"], lat["rounds"], lat["runs"], threads=th)["b"]
            print(f"  CPU {th} thread(s): median {r['median']:.3f} ms  p95 {r['p95']:.3f} ms  std {r['std']:.3f}")
            rows[0][f"cpu{th}_ms"] = r["median"]
        if device.type == "cuda":
            gm = model.to(device)
            r = measure({"b": model_fn(gm, x.to(device))}, lat["warmup"], lat["rounds"], lat["runs"], cuda=True)["b"]
            print(f"  GPU (synchronized): median {r['median']:.3f} ms  p95 {r['p95']:.3f} ms")
            rows[0]["gpu_ms"] = r["median"]
    write_csv(hist_rows, "baseline_history.csv")
    write_csv([{k: r.get(k, "") for k in rows[0]} for r in rows], "baseline.csv")
