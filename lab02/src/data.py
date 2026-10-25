"""Digits dataset for Lab 2: scaled to [0, 1], one-hot labels, 70/30 stratified split."""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
N_CLASSES = 10
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def one_hot(y, n_classes=N_CLASSES):
    Y = np.zeros((y.shape[0], n_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def load_split():
    """Returns X_train, X_test, Y_train, Y_test (one-hot), y_train, y_test (integer labels)."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0  # pixel intensities are 0..16
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    return X_tr, X_te, one_hot(y_tr), one_hot(y_te), y_tr, y_te


if __name__ == "__main__":
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_split()
    lines = [
        f"X_train {X_tr.shape}  X_test {X_te.shape}",
        f"Y_train {Y_tr.shape}  Y_test {Y_te.shape}",
        f"X range [{X_tr.min():.3f}, {X_tr.max():.3f}]",
        f"one-hot rows sum to 1: {bool(np.all(Y_tr.sum(1) == 1) and np.all(Y_te.sum(1) == 1))}",
        f"one-hot argmax == label: {bool(np.all(Y_tr.argmax(1) == y_tr))}",
        "class  train  test  train_share  test_share",
    ]
    for c in range(N_CLASSES):
        ntr, nte = int((y_tr == c).sum()), int((y_te == c).sum())
        lines.append(f"{c:>5}  {ntr:>5}  {nte:>4}  {ntr / len(y_tr):>11.4f}  {nte / len(y_te):>10.4f}")
    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dataset.txt").write_text(text + "\n")
