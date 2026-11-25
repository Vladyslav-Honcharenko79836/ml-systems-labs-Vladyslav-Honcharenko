import time

import torch
import torch.nn.functional as F

from common import ckpt_path, env_line, load_config, mean_std, pick_device, seed_everything, tee_results, write_csv
from data import Batches, load_tensors
from engine import PowerMeter, ce_loss, evaluate, logits_of, train_model
from models import make_student, n_params, total_macs
from prune_study import load_baseline


def kd_manual(T, alpha):
    def loss(out, y, teacher):
        log_s = F.log_softmax(out / T, dim=1)
        log_t = F.log_softmax(teacher / T, dim=1)
        kl = (log_s.exp() * (log_s - log_t)).sum(1).mean()
        return alpha * F.cross_entropy(out, y) + (1 - alpha) * T * T * kl
    return loss


def kd_forward(T, alpha):
    def loss(out, y, teacher):
        kl = F.kl_div(F.log_softmax(out / T, dim=1), F.softmax(teacher / T, dim=1), reduction="batchmean")
        return alpha * F.cross_entropy(out, y) + (1 - alpha) * T * T * kl
    return loss


def losses(cfg):
    T, a = cfg["distill"]["temperature"], cfg["distill"]["alpha"]
    return {"plain": ce_loss, "kd": kd_manual(T, a), "kd_forward": kd_forward(T, a)}


def train_student(cfg, train, seed, loss_fn, teacher_logits, device, epochs, lr, clip):
    b = cfg["baseline"]
    seed_everything(seed, cfg["deterministic"])
    student = make_student(cfg).to(device)
    batches = Batches(*train, cfg["data"]["batch_size"], seed, extra=teacher_logits)
    t0 = time.perf_counter()
    with PowerMeter() as pm:
        hist = train_model(student, batches, epochs, lr, b["momentum"], b["weight_decay"], device,
                           loss_fn=loss_fn, clip=clip)
    return student, hist, time.perf_counter() - t0, pm


def mcnemar(a, b):
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / max(1, n01 + n10)
    return n01, n10, stat


if __name__ == "__main__":
    cfg = load_config()
    d = cfg["distill"]
    device = pick_device()
    data = load_tensors(cfg, device)
    with tee_results("distill.txt"):
        print(env_line())
        teacher = load_baseline(cfg, d["teacher_seed"], device)
        t0 = time.perf_counter()
        with PowerMeter() as pm:
            t_logits = logits_of(teacher, data["train"][0])
            torch.cuda.synchronize()
        _, t_test = evaluate(teacher, data["test"])
        probs = F.softmax(t_logits, 1).max(1).values.mean().item()
        soft = F.softmax(t_logits / d["temperature"], 1).max(1).values.mean().item()
        agree = (t_logits.argmax(1) == data["train"][1]).float().mean().item()
        student = make_student(cfg)
        print(f"teacher: baseline seed {d['teacher_seed']}, {n_params(teacher):,} params, test {t_test:.4f}, "
              f"train acc {agree:.4f}, mean max prob T=1 {probs:.3f}, T={d['temperature']:.0f} {soft:.3f}; "
              f"soft targets for 55 000 images: {time.perf_counter() - t0:.1f} s, {pm.counter_joules:.0f} J")
        print(f"student: {n_params(student):,} params, {total_macs(student):,} MACs; teacher/student: "
              f"{n_params(teacher) / n_params(student):.1f}x params, {total_macs(teacher) / total_macs(student):.1f}x MACs")
        print(f"T={d['temperature']} alpha={d['alpha']} epochs={d['epochs']} lr={d['lr']} (chosen on validation, "
              f"tune_distill.py --clip {d['grad_clip']}), gradient norm clipped at {d['grad_clip']} for all variants; "
              f"same initial weights and batch order for every variant of a seed")
        print("plain      = CE only")
        print("kd         = alpha*CE + (1-alpha)*T^2*KL(softmax(student/T) || softmax(teacher/T))   [manual's formula]")
        print("kd_forward = alpha*CE + (1-alpha)*T^2*KL(softmax(teacher/T) || softmax(student/T))   [Hinton / F.kl_div]\n")

        rows, correct = [], {}
        for seed in d["seeds"]:
            for name, fn in losses(cfg).items():
                model, hist, secs, pm = train_student(cfg, data["train"], seed, fn, t_logits, device, d["epochs"],
                                                      d["lr"][name], d["grad_clip"])
                tr_loss, tr_acc = evaluate(model, data["train"])
                val_loss, val_acc = evaluate(model, data["val"])
                test_loss, test_acc = evaluate(model, data["test"])
                path = ckpt_path(cfg, f"student_{name}_s{seed}")
                torch.save(model.state_dict(), path)
                pred = logits_of(model, data["test"][0]).argmax(1)
                correct[(name, seed)] = pred == data["test"][1]
                tpred = logits_of(teacher, data["test"][0]).argmax(1)
                rows.append({"variant": name, "seed": seed, "train_acc": tr_acc, "train_ce": tr_loss,
                             "val_acc": val_acc, "val_ce": val_loss, "test_acc": test_acc, "test_ce": test_loss,
                             "agree_teacher": (pred == tpred).float().mean().item(),
                             "file_bytes": path.stat().st_size, "train_s": round(secs, 1),
                             "train_joules": round(pm.joules, 1), "train_joules_counter": pm.counter_joules})
                r = rows[-1]
                print(f"seed {seed} {name:<10} train acc {tr_acc:.4f}  val {val_acc:.4f}  test {test_acc:.4f}  "
                      f"test CE {test_loss:.4f}  agreement with teacher {r['agree_teacher']:.4f}  "
                      f"{secs:.1f} s  {r['train_joules']:.0f} J")
        print(f"\nmean +- std over {len(d['seeds'])} seeds (accuracies in %)")
        print(f"{'variant':<11} {'train':>14} {'val':>14} {'test':>14} {'test CE':>16} {'agree teacher':>15}")
        summary = []
        for name in losses(cfg):
            g = [r for r in rows if r["variant"] == name]
            f = {k: mean_std([r[k] for r in g]) for k in
                 ("train_acc", "val_acc", "test_acc", "test_ce", "agree_teacher", "train_s", "train_joules")}
            print(f"{name:<11} " + " ".join(f"{100 * f[k][0]:>7.2f} +- {100 * f[k][1]:<4.2f}"
                                             for k in ("train_acc", "val_acc", "test_acc"))
                  + f" {f['test_ce'][0]:>8.4f} +- {f['test_ce'][1]:.4f} {100 * f['agree_teacher'][0]:>9.2f}")
            summary.append({"variant": name, **{f"{k}_{s}": v[i] for k, v in f.items()
                                                for i, s in enumerate(("mean", "std"))}})
        print("\npaired differences in test accuracy, pp (same seed = same init and batch order)")
        for a, bb in (("kd", "plain"), ("kd_forward", "plain"), ("kd", "kd_forward")):
            diffs = [100 * (next(r["test_acc"] for r in rows if r["variant"] == a and r["seed"] == s)
                            - next(r["test_acc"] for r in rows if r["variant"] == bb and r["seed"] == s))
                     for s in d["seeds"]]
            m, s = mean_std(diffs)
            t = m / (s / len(diffs) ** 0.5) if s else float("nan")
            print(f"  {a} - {bb}: {m:+.2f} +- {s:.2f}  (t = {t:.2f}, n = {len(diffs)}; "
                  f"per seed {' '.join(f'{x:+.2f}' for x in diffs)})")
        s0 = d["seeds"][0]
        n01, n10, stat = mcnemar(correct[("kd", s0)], correct[("plain", s0)])
        print(f"  McNemar, seed {s0}: kd right & plain wrong {n01}, plain right & kd wrong {n10}, chi2 = {stat:.2f} "
              f"(3.84 = 5% level)")

        print(f"\ndata study: fraction of the training set, same number of optimizer steps as {d['epochs']} full epochs")
        frac_rows = []
        n = len(data["train"][1])
        full_steps = d["epochs"] * len(Batches(*data["train"], cfg["data"]["batch_size"], 0))
        for frac in d["data_fractions"]:
            for seed in d["data_seeds"]:
                idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:round(frac * n)].to(device)
                sub = (data["train"][0][idx], data["train"][1][idx])
                steps = len(Batches(*sub, cfg["data"]["batch_size"], 0))
                epochs = max(1, round(full_steps / steps))
                for name in ("plain", "kd"):
                    model, _, secs, _ = train_student(cfg, sub, seed, losses(cfg)[name], t_logits[idx], device, epochs,
                                                          d["lr"][name], d["grad_clip"])
                    _, val_acc = evaluate(model, data["val"])
                    _, test_acc = evaluate(model, data["test"])
                    frac_rows.append({"fraction": frac, "images": len(idx), "epochs": epochs, "seed": seed,
                                      "variant": name, "val_acc": val_acc, "test_acc": test_acc})
                    print(f"  fraction {frac:<5} images {len(idx):>6,} epochs {epochs:>4} seed {seed} {name:<6} "
                          f"val {val_acc:.4f} test {test_acc:.4f}  {secs:.1f} s")
        for seed in d["data_seeds"]:
            for name in ("plain", "kd"):
                r = next(r for r in rows if r["variant"] == name and r["seed"] == seed)
                frac_rows.append({"fraction": 1.0, "images": n, "epochs": d["epochs"], "seed": seed, "variant": name,
                                  "val_acc": r["val_acc"], "test_acc": r["test_acc"]})
        print(f"\n{'images':>7} {'plain test %':>16} {'kd test %':>16} {'kd - plain, pp':>16}")
        for frac in d["data_fractions"] + [1.0]:
            g = {nm: [100 * r["test_acc"] for r in frac_rows if r["fraction"] == frac and r["variant"] == nm]
                 for nm in ("plain", "kd")}
            pm_, km = mean_std(g["plain"]), mean_std(g["kd"])
            dm = mean_std([k - p for k, p in zip(g["kd"], g["plain"])])
            print(f"{round(frac * n):>7,} {pm_[0]:>8.2f} +- {pm_[1]:<4.2f} {km[0]:>8.2f} +- {km[1]:<4.2f} "
                  f"{dm[0]:>+8.2f} +- {dm[1]:<4.2f}")
    write_csv(rows, "distill_runs.csv")
    write_csv(summary, "distill_summary.csv")
    write_csv(frac_rows, "distill_data.csv")
