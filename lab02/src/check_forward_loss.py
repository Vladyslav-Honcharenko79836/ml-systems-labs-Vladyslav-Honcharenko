"""Steps 3-4: forward-pass shape / row-sum check and untrained cross-entropy vs ln(10)."""
import math

import numpy as np

from data import RESULTS_DIR, SEED, load_split
from mlp import cross_entropy, forward, init_params

X_tr, X_te, Y_tr, Y_te, _, _ = load_split()
lines = []
for width in (16, 64):
    params = init_params(width, np.random.default_rng(SEED))
    for name, X in (("batch of 32", X_tr[:32]), ("full train", X_tr), ("full test", X_te)):
        P, cache = forward(params, X)
        max_dev = float(np.abs(P.sum(axis=1) - 1.0).max())
        ok = P.shape == (X.shape[0], 10) and max_dev < 1e-6
        lines.append(
            f"W={width:<3} {name:<12} input {str(X.shape):<10} -> Z1 {str(cache[1].shape):<10}"
            f" Z2 {str(cache[3].shape):<10} P {str(P.shape):<10} max|rowsum-1| {max_dev:.2e}  {'PASS' if ok else 'FAIL'}"
        )
lines.append("")
lines.append(f"ln(10) = {math.log(10):.4f}")
for width in (16, 64):
    params = init_params(width, np.random.default_rng(SEED))
    for name, X, Y in (("train", X_tr, Y_tr), ("test", X_te, Y_te)):
        P, _ = forward(params, X)
        loss = cross_entropy(P, Y)
        lines.append(
            f"W={width:<3} untrained {name:<5} loss {loss:.4f}  |loss-ln10| {abs(loss - math.log(10)):.2e}"
            f"  P min {P.min():.4f} max {P.max():.4f}"
        )
text = "\n".join(lines)
print(text)
(RESULTS_DIR / "forward_loss_check.txt").write_text(text + "\n")
