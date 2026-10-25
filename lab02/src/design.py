"""Step 2: print the parameter shapes and counts of the designed network for W = 16 and W = 64."""
import numpy as np

from data import RESULTS_DIR, SEED
from mlp import INIT_STD, PARAM_NAMES, init_params

lines = []
for width in (16, 64):
    params = init_params(width, np.random.default_rng(SEED))
    lines.append(f"W = {width}")
    total = 0
    for k in PARAM_NAMES:
        p = params[k]
        total += p.size
        lines.append(f"  {k}: shape {str(p.shape):<10} count {p.size:>5}  mean {p.mean():+.5f}  std {p.std():.5f}")
    lines.append(f"  total parameters: {total}  ({total * 8} bytes as float64)")
lines.append(f"init: weights ~ N(0, {INIT_STD}) (std), biases = 0")
text = "\n".join(lines)
print(text)
(RESULTS_DIR / "design.txt").write_text(text + "\n")
