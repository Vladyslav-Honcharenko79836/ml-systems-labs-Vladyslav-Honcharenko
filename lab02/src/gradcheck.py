"""Step 5: compare analytic gradients with central differences on 10 random weights.

numerical = (L(w + h) - L(w - h)) / (2h),  h = 1e-5
relative error = |analytic - numerical| / max(|analytic|, |numerical|)
Pass: rel. error < 1e-6 for at least 9 of the 10 weights.

The check is run at two points for each width:
  * "trained" (graded): the network after TRAIN_EPOCHS of SGD (lr 0.1, batch 32), i.e. the
    best configuration of step 6. Gradients there are O(1e-3).
  * "init" (diagnostic): the N(0, 0.01) initialisation. The network sits on a plateau, its
    gradients are O(1e-5), and the float64 rounding error of the central difference
    (~1e-11 absolute, independent of the gradient) is then 1e-6 or more of the gradient.
Besides the 10-weight sample, every weight of the network is swept once to show the
pass rate as a function of gradient size.
Weights whose analytic gradient is exactly zero (always-zero pixels, inactive ReLUs) are not
sampled: they pass trivially (0 == 0) and say nothing about the derivation.
"""
import csv

import numpy as np

from data import RESULTS_DIR, SEED, load_split
from mlp import backward, cross_entropy, forward, init_params, sgd_step

H = 1e-5
N_CHECK = 10
BATCH = 32
TRAIN_EPOCHS = 50
TRAIN_LR = 0.1
WEIGHTS = ("W1", "W2", "W3")


def train(params, X, Y, rng):
    for _ in range(TRAIN_EPOCHS):
        order = rng.permutation(X.shape[0])
        for s in range(0, X.shape[0], BATCH):
            ids = order[s:s + BATCH]
            _, cache = forward(params, X[ids])
            sgd_step(params, backward(params, cache, Y[ids]), TRAIN_LR)


def check_one(params, grads, X, Y, name, idx):
    w = params[name]
    old = w[idx]
    w[idx] = old + H
    lp = cross_entropy(forward(params, X)[0], Y)
    w[idx] = old - H
    lm = cross_entropy(forward(params, X)[0], Y)
    w[idx] = old
    num = (lp - lm) / (2 * H)
    ana = grads[name][idx]
    rel = abs(ana - num) / max(abs(ana), abs(num))
    return {"param": name, "index": tuple(int(i) for i in idx), "analytic": ana,
            "numerical": num, "abs_error": abs(ana - num), "rel_error": rel, "pass": rel < 1e-6}


if __name__ == "__main__":
    X_tr, _, Y_tr, _, _, _ = load_split()
    X, Y = X_tr[:BATCH], Y_tr[:BATCH]
    lines, sample_rows, sweep_lines = [], [], []
    for point in ("trained", "init"):
        for width in (16, 64):
            rng = np.random.default_rng(SEED)
            params = init_params(width, np.random.default_rng(SEED))
            if point == "trained":
                train(params, X_tr, Y_tr, rng)
            P, cache = forward(params, X)
            loss = cross_entropy(P, Y)
            grads = backward(params, cache, Y)
            candidates = [(k, idx) for k in WEIGHTS for idx in zip(*np.nonzero(grads[k]))]
            n_zero = sum(params[k].size for k in WEIGHTS) - len(candidates)

            rows = [check_one(params, grads, X, Y, *candidates[c])
                    for c in rng.choice(len(candidates), size=N_CHECK, replace=False)]
            n_pass = sum(r["pass"] for r in rows)
            sample_rows += [{"point": point, "width": width, **r} for r in rows]
            tag = f"after {TRAIN_EPOCHS} epochs SGD lr={TRAIN_LR}" if point == "trained" else "at N(0, 0.01) init"
            lines.append(f"[{point}] W = {width}, {tag}; batch loss {loss:.4f}; batch {BATCH}, h = {H}; "
                         f"{n_zero} zero-gradient weights excluded")
            lines.append(f"  {'param':<5} {'index':<10} {'analytic':>14} {'numerical':>14} {'abs_err':>9} {'rel_err':>9}")
            for r in rows:
                lines.append(f"  {r['param']:<5} {str(r['index']):<10} {r['analytic']:>14.6e} {r['numerical']:>14.6e}"
                             f" {r['abs_error']:>9.1e} {r['rel_error']:>9.1e}  {'PASS' if r['pass'] else 'FAIL'}")
            verdict = "GRADIENT CHECK PASSED" if n_pass >= 9 else "below the 9/10 threshold"
            lines.append(f"  {n_pass}/{N_CHECK} below 1e-6 -> {verdict}\n")

            sweep = [check_one(params, grads, X, Y, *c) for c in candidates]
            mag = np.array([abs(r["analytic"]) for r in sweep])
            rel = np.array([r["rel_error"] for r in sweep])
            ab = np.array([r["abs_error"] for r in sweep])
            row = f"{point:<8} {width:>3} {loss:>7.4f} {len(sweep):>6} {np.median(mag):>9.1e} {np.median(ab):>9.1e}" \
                  f" {np.median(rel):>9.1e} {np.mean(rel < 1e-6):>9.1%}"
            for lo, hi in ((0, 1e-5), (1e-5, 1e-4), (1e-4, np.inf)):
                m = (mag >= lo) & (mag < hi)
                row += f" {np.mean(rel[m] < 1e-6) if m.any() else float('nan'):>11.1%} ({m.sum():>4})"
            sweep_lines.append(row)

    lines.append("Sweep over every weight with a non-zero gradient (pass = rel. error < 1e-6)")
    lines.append(f"{'point':<8} {'W':>3} {'loss':>7} {'n':>6} {'med|g|':>9} {'med abs':>9} {'med rel':>9} {'pass all':>9}"
                 f" {'|g|<1e-5':>18} {'1e-5..1e-4':>18} {'|g|>=1e-4':>18}")
    lines += sweep_lines
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "gradcheck.txt").write_text(text + "\n")
    with open(RESULTS_DIR / "gradcheck.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
