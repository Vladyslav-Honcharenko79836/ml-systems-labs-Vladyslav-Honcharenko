import argparse

from common import env_line, load_config, mean_std, pick_device, tee_results, write_csv
from data import load_tensors
from distill import losses, train_student
from engine import evaluate, logits_of
from prune_study import load_baseline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=float, default=None)
    clip = ap.parse_args().clip
    tag = "tune_distill_clip" if clip else "tune_distill"
    cfg = load_config()
    d = cfg["distill"]
    device = pick_device()
    data = load_tensors(cfg, device)
    rows = []
    with tee_results(f"{tag}.txt"):
        print(env_line())
        print(f"learning-rate search on the validation set only, seeds {d['tune_seeds']} (not used in the final runs); "
              f"{d['epochs']} epochs; the test set is not touched; gradient clipping: {clip}")
        teacher = load_baseline(cfg, d["teacher_seed"], device)
        t_logits = logits_of(teacher, data["train"][0])
        for name, fn in losses(cfg).items():
            for lr in d["tune_lrs"]:
                for seed in d["tune_seeds"]:
                    model, hist, secs, _ = train_student(cfg, data["train"], seed, fn, t_logits, device,
                                                         d["epochs"], lr, clip)
                    val_loss, val_acc = evaluate(model, data["val"])
                    rows.append({"variant": name, "lr": lr, "seed": seed, "train_acc": hist[-1]["train_acc"],
                                 "val_loss": val_loss, "val_acc": val_acc})
                    print(f"  {name:<10} lr {lr:<7} seed {seed}  train acc {hist[-1]['train_acc']:.4f}  "
                          f"val acc {val_acc:.4f}  val CE {val_loss:.4f}  {secs:.1f} s")
        print("\nmean validation accuracy, %")
        print(f"{'lr':>8} " + " ".join(f"{n:>11}" for n in losses(cfg)))
        table = {(n, lr): mean_std([100 * r["val_acc"] for r in rows if r["variant"] == n and r["lr"] == lr])[0]
                 for n in losses(cfg) for lr in d["tune_lrs"]}
        for lr in d["tune_lrs"]:
            print(f"{lr:>8} " + " ".join(f"{table[(n, lr)]:>11.2f}" for n in losses(cfg)))
        for n in losses(cfg):
            best = max(d["tune_lrs"], key=lambda lr: table[(n, lr)])
            print(f"best lr for {n}: {best} ({table[(n, best)]:.2f} %)")
    write_csv(rows, f"{tag}.csv")
