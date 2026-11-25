import torch

from common import env_line, load_config, pick_device, seed_everything, tee_results, write_csv
from data import Batches, load_tensors
from distill import losses
from engine import evaluate, logits_of
from models import make_student
from prune_study import load_baseline

CASES = [("plain", 0.02, 7), ("kd", 0.02, 45), ("kd_forward", 0.02, 7), ("kd_forward", 0.02, 8)]
STEPS = 1290
MARKS = [1, 10, 25, 50, 100, 200, 300, 430, 645, 860, 1290]

if __name__ == "__main__":
    cfg = load_config()
    device = pick_device()
    data = load_tensors(cfg, device)
    rows = []
    with tee_results("collapse_diag.txt"):
        print(env_line())
        print(f"first {STEPS} steps (3 epochs), constant lr, batch 128, no warm-up; per-batch values, "
              "dead = share of fc1 ReLU units that are zero on all 5 000 validation images")
        teacher = load_baseline(cfg, cfg["distill"]["teacher_seed"], device)
        t_logits = logits_of(teacher, data["train"][0])
        for name, lr, seed in CASES:
            seed_everything(seed, cfg["deterministic"])
            model = make_student(cfg).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg["baseline"]["momentum"],
                                  weight_decay=cfg["baseline"]["weight_decay"])
            batches = Batches(*data["train"], cfg["data"]["batch_size"], seed, extra=t_logits)
            loss_fn = losses(cfg)[name]
            print(f"\n{name} lr {lr} seed {seed}")
            step, peak, peak_step = 0, 0.0, 0
            while step < STEPS:
                for x, y, extra in batches:
                    step += 1
                    model.train()
                    loss = loss_fn(model(x), y, extra)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
                    opt.step()
                    if gn > peak:
                        peak, peak_step = gn, step
                    if step in MARKS:
                        _, val = evaluate(model, data["val"])
                        with torch.no_grad():
                            h = data["val"][0]
                            for n, m in model.net.named_children():
                                h = m(h)
                                if n == "relu4":
                                    break
                            dead = (h == 0).all(0).float().mean().item()
                        rows.append({"variant": name, "lr": lr, "seed": seed, "step": step, "loss": loss.item(),
                                     "grad_norm": gn, "val_acc": val, "dead_fc1": dead})
                        print(f"  step {step:>5}  loss {loss.item():>9.4f}  grad norm {gn:>8.2f}  val acc {val:.4f}  "
                              f"dead {dead:.2f}")
                    if step == STEPS:
                        break
            print(f"  largest gradient norm {peak:.2f} at step {peak_step}")
    write_csv(rows, "collapse_diag.csv")
