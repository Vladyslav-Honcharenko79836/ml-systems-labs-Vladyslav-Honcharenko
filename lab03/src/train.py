"""Lab 3 step 4: train MLP, CNN and attention for 5 epochs (Adam, lr 1e-3, batch 128), record test accuracy.

Every model starts from the same seed and sees the same shuffled batch order. Validation accuracy is logged
after each epoch; the test set is evaluated once, after the last epoch.
"""
import csv
import time

import torch
from torch import nn

from data import DATA_DIR, RESULTS_DIR, SEED, load_loaders
from models import MODELS

EPOCHS = 5
LR = 1e-3
CKPT_DIR = DATA_DIR / "lab03_checkpoints"  # git-ignored


def evaluate(model, loader, loss_fn):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            out = model(x)
            loss_sum += loss_fn(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, correct / n


def train_one(name, cls, log_rows):
    torch.manual_seed(SEED)
    train_dl, val_dl, test_dl = load_loaders()
    model = cls()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())

    total_time = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
        epoch_time = time.perf_counter() - t0
        total_time += epoch_time
        val_loss, val_acc = evaluate(model, val_dl, loss_fn)
        row = {"model": name, "epoch": epoch, "train_loss": round(loss_sum / n, 4),
               "train_acc": round(correct / n, 4), "val_loss": round(val_loss, 4),
               "val_acc": round(val_acc, 4), "epoch_time_s": round(epoch_time, 2)}
        log_rows.append(row)
        print(f"{name:<9} epoch {epoch}  train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f}  "
              f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f}  {epoch_time:.1f}s", flush=True)

    test_loss, test_acc = evaluate(model, test_dl, loss_fn)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / f"{name}.pt")
    return {"model": name, "params": n_params, "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 4), "val_acc": log_rows[-1]["val_acc"],
            "train_time_s": round(total_time, 1)}


if __name__ == "__main__":
    log_rows, results = [], []
    for name, cls in MODELS.items():
        results.append(train_one(name, cls, log_rows))

    lines = [f"Adam lr {LR}, batch 128, {EPOCHS} epochs, seed {SEED}, CPU threads {torch.get_num_threads()}",
             "model      params   val acc  test acc  test loss  train time s"]
    for r in results:
        lines.append(f"{r['model']:<9}  {r['params']:>7,}  {r['val_acc']:>7.4f}  {r['test_acc']:>8.4f}"
                     f"  {r['test_loss']:>9.4f}  {r['train_time_s']:>12.1f}")
    text = "\n".join(lines)
    print("\n" + text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "accuracy.txt").write_text(text + "\n")
    for path, rows in [("training_log.csv", log_rows), ("accuracy.csv", results)]:
        with open(RESULTS_DIR / path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
