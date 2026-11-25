import copy
import time

import torch

from common import ckpt_path, env_line, load_config, mean_std, pick_device, seed_everything, tee_results, write_csv
from data import Batches, load_tensors
from engine import PowerMeter, evaluate, logits_of, train_model
from models import effective_macs, make_teacher, n_params, nonzero_params, total_macs, weight_layers
from prune import MASKS, apply_masks, filters_to_keep, remove_filters, sparsity_report, zero_filters


def load_baseline(cfg, seed, device):
    m = make_teacher(cfg).to(device)
    m.load_state_dict(torch.load(ckpt_path(cfg, f"baseline_s{seed}"), map_location=device))
    return m


def finetune(model, cfg, data, seed, device, masks=None):
    b, p = cfg["baseline"], cfg["prune"]
    seed_everything(seed, cfg["deterministic"])
    batches = Batches(*data["train"], cfg["data"]["batch_size"], seed + 1000)
    after = (lambda: apply_masks(model, masks)) if masks is not None else None
    return train_model(model, batches, p["finetune_epochs"], p["finetune_lr"], b["momentum"], b["weight_decay"],
                       device, after_step=after)


def record(kind, target, seed, model, data, before, seconds, joules, base_params):
    sp, per = sparsity_report(model)
    _, val = evaluate(model, data["val"])
    _, test = evaluate(model, data["test"])
    return {"kind": kind, "target": target, "seed": seed, "weight_sparsity": round(sp, 6),
            "params": n_params(model), "nonzero_params": nonzero_params(model),
            "param_reduction": round(1 - nonzero_params(model) / base_params, 6),
            "macs": total_macs(model), "effective_macs": round(effective_macs(model)),
            "test_before_ft": before, "val_acc": val, "test_acc": test, "ft_seconds": round(seconds, 2),
            "ft_joules": round(joules, 1), **{f"sp_{k}": round(v, 4) for k, v in per.items()}}


if __name__ == "__main__":
    cfg = load_config()
    p = cfg["prune"]
    device = pick_device()
    data = load_tensors(cfg, device)
    rows = []
    with tee_results("prune_study.txt"):
        print(env_line())
        print(f"fine-tune: {p['finetune_epochs']} epochs, SGD lr {p['finetune_lr']} cosine, masks re-applied after "
              f"every optimizer step; biases are never pruned")
        for seed in cfg["seeds"]:
            base = load_baseline(cfg, seed, device)
            base_params = n_params(base)
            _, acc0 = evaluate(base, data["test"])
            print(f"\nseed {seed}: baseline test {acc0:.4f}")
            for kind in ("global", "layer"):
                for s in p["sparsities"]:
                    if s == 0 and kind == "layer":
                        rows.append({**rows[-len(p["sparsities"])], "kind": "layer"})
                        continue
                    model = copy.deepcopy(base)
                    masks = MASKS[kind](model, s)
                    apply_masks(model, masks)
                    _, before = evaluate(model, data["test"])
                    t0 = time.perf_counter()
                    with PowerMeter() as pm:
                        finetune(model, cfg, data, seed, device, masks)
                    dt = time.perf_counter() - t0
                    assert all(bool((m.weight[masks[n] == 0] == 0).all()) for n, m in weight_layers(model))
                    r = record(kind, s, seed, model, data, before, dt, pm.joules, base_params)
                    rows.append(r)
                    torch.save(model.state_dict(), ckpt_path(cfg, f"{kind}_{s}_s{seed}"))
                    per = " ".join(f"{k[3:]} {r[k]:.2f}" for k in r if k.startswith("sp_"))
                    print(f"  {kind:<10} target {s:<5} real {r['weight_sparsity']:.4f}  nnz {r['nonzero_params']:>9,}  "
                          f"eff MACs {r['effective_macs'] / 1e6:6.2f}M  test {before:.4f} -> {r['test_acc']:.4f}  "
                          f"[{per}]")
            for f in p["filter_fractions"]:
                keep = filters_to_keep(base, f)
                masked = copy.deepcopy(base)
                zero_filters(masked, keep)
                model = remove_filters(base, keep, cfg["baseline"]["dropout"])
                x = data["test"][0][:1000]
                diff = (logits_of(masked, x) - logits_of(model, x)).abs().max().item()
                _, before = evaluate(model, data["test"])
                t0 = time.perf_counter()
                with PowerMeter() as pm:
                    finetune(model, cfg, data, seed, device)
                dt = time.perf_counter() - t0
                r = record("structured", f, seed, model, data, before, dt, pm.joules, base_params)
                rows.append(r)
                torch.save({"widths": model.widths, "state": model.state_dict()},
                           ckpt_path(cfg, f"structured_{f}_s{seed}"))
                print(f"  structured target {f:<5} widths {model.widths}  params {r['params']:>9,}  "
                      f"MACs {r['macs'] / 1e6:6.2f}M  test {before:.4f} -> {r['test_acc']:.4f}  "
                      f"max|masked-removed logits| {diff:.1e}")

        print(f"\nmean +- std over seeds {cfg['seeds']}, test accuracy in %")
        print(f"{'kind':<11} {'target':>6} {'nnz params':>10} {'params -%':>9} {'eff MACs M':>10} "
              f"{'before ft':>14} {'after ft':>14} {'change vs base':>16}")
        base_acc = {}
        for r in rows:
            if r["kind"] == "global" and r["target"] == 0:
                base_acc[r["seed"]] = r["test_before_ft"]
        summary = []
        for kind in ("global", "layer", "structured"):
            for t in (p["filter_fractions"] if kind == "structured" else p["sparsities"]):
                g = [r for r in rows if r["kind"] == kind and r["target"] == t]
                bm, bs = mean_std([100 * r["test_before_ft"] for r in g])
                am, as_ = mean_std([100 * r["test_acc"] for r in g])
                vm, vs = mean_std([100 * r["val_acc"] for r in g])
                dm, ds = mean_std([100 * (r["test_acc"] - base_acc[r["seed"]]) for r in g])
                red, _ = mean_std([100 * r["param_reduction"] for r in g])
                em, _ = mean_std([r["effective_macs"] / 1e6 for r in g])
                nnz, _ = mean_std([r["nonzero_params"] for r in g])
                print(f"{kind:<11} {t:>6} {nnz:>10,.0f} {red:>9.1f} {em:>10.2f} {bm:>7.2f} +- {bs:<4.2f} "
                      f"{am:>7.2f} +- {as_:<4.2f} {dm:>+8.2f} +- {ds:.2f}")
                summary.append({"kind": kind, "target": t, "nonzero_params": round(nnz), "param_reduction_pct": red,
                                "eff_macs_m": em, "before_mean": bm, "before_std": bs, "after_mean": am,
                                "after_std": as_, "val_mean": vm, "val_std": vs, "delta_mean": dm, "delta_std": ds})
    write_csv(rows, "prune_runs.csv")
    write_csv(summary, "prune_summary.csv")
