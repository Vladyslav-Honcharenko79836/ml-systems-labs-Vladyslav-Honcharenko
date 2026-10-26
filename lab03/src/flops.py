"""Lab 3 step 3: hand-calculated FLOPs (Ch. 4 formulas) vs thop.profile, for one image (batch 1).

Convention: 1 MAC (multiply-accumulate) = 2 FLOPs. The hand count covers matrix multiplies only
(Linear, Conv2d, QKV/out projections, attention scores, attention x V); bias adds, ReLU, max-pool,
softmax, LayerNorm and residual adds are left out, as in the Ch. 4 formulas.
thop reports MACs, so its numbers are doubled before comparing.
"""
import csv

import torch
from thop import profile
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from data import RESULTS_DIR
from models import D_FF, D_MODEL, MODELS, N_LAYERS, N_PATCHES, PATCH_DIM


def linear_flops(n_in, n_out, rows=1):
    return 2 * n_in * n_out * rows


def conv_flops(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def encoder_layer_flops(seq=N_PATCHES, d=D_MODEL, d_ff=D_FF):
    """Matmul FLOPs of one Transformer encoder layer, split by operation."""
    return {
        "QKV projections  2*seq*d*3d": 2 * seq * d * 3 * d,
        "scores QK^T      2*seq^2*d": 2 * seq * seq * d,
        "softmax(.) x V   2*seq^2*d": 2 * seq * seq * d,
        "out projection   2*seq*d*d": 2 * seq * d * d,
        "FFN up           2*seq*d*d_ff": 2 * seq * d * d_ff,
        "FFN down         2*seq*d_ff*d": 2 * seq * d_ff * d,
    }


def hand_layers():
    """Per-layer hand FLOPs for each model (batch 1)."""
    enc = sum(encoder_layer_flops().values())
    return {
        "MLP": {
            "fc1 Linear 784->256": linear_flops(784, 256),
            "fc2 Linear 256->128": linear_flops(256, 128),
            "fc3 Linear 128->10": linear_flops(128, 10),
        },
        "CNN": {
            "conv1 3x3 1->16 @28x28": conv_flops(3, 1, 16, 28, 28),
            "conv2 3x3 16->32 @14x14": conv_flops(3, 16, 32, 14, 14),
            "fc Linear 1568->10": linear_flops(32 * 7 * 7, 10),
        },
        "Attention": {
            "patch proj Linear 49->64 x16": linear_flops(PATCH_DIM, D_MODEL, rows=N_PATCHES),
            **{f"encoder layer {i + 1}": enc for i in range(N_LAYERS)},
            "head Linear 64->10": linear_flops(D_MODEL, 10),
        },
    }


def count_mha(m: nn.MultiheadAttention, x, y):
    """thop rule for nn.MultiheadAttention (batch_first, self-attention): MACs of QKV, QK^T, AV, out_proj."""
    q = x[0]
    b, seq, d = q.shape
    macs = b * (seq * d * 3 * d + 2 * seq * seq * d + seq * d * d)
    m.total_ops += torch.DoubleTensor([macs])


def thop_macs(model, x, custom_ops=None):
    macs, _, layers = profile(model, (x,), custom_ops=custom_ops, verbose=False, ret_layer_info=True)
    return macs, layers


def flop_counter(model, x):
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


def flatten_layers(layers, prefix=""):
    """thop's nested {name: (macs, params, children)} -> [(path, macs)] for leaf modules."""
    out = []
    for name, (macs, _, children) in layers.items():
        path = f"{prefix}{name}"
        if children:
            out += flatten_layers(children, path + ".")
        else:
            out.append((path, int(macs)))
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 1, 28, 28)
    hand = hand_layers()
    rows, lines = [], []

    lines.append("== Hand-calculated FLOPs, one layer of each type (batch 1) ==")
    lines.append(f"Linear  fc1 784->256:  2*784*256 = {linear_flops(784, 256):,}")
    lines.append(f"Conv2d  conv2 3x3 16->32 @14x14:  2*3*3*16*32*14*14 = {conv_flops(3, 16, 32, 14, 14):,}")
    lines.append(f"Attention  one encoder layer, seq={N_PATCHES}, d_model={D_MODEL}, d_ff={D_FF}:")
    for op, f in encoder_layer_flops().items():
        lines.append(f"    {op:<32} {f:>9,}")
    lines.append(f"    {'total':<32} {sum(encoder_layer_flops().values()):>9,}")
    lines.append("")

    for name, cls in MODELS.items():
        model = cls().eval()
        hand_total = sum(hand[name].values())

        torch.backends.mha.set_fastpath_enabled(True)
        macs_default, layers_default = thop_macs(model, x)
        fc_default = flop_counter(model, x)

        torch.backends.mha.set_fastpath_enabled(False)
        macs_nofast, _ = thop_macs(model, x)
        macs_custom, layers_custom = thop_macs(model, x, custom_ops={nn.MultiheadAttention: count_mha})
        fc_nofast = flop_counter(model, x)
        torch.backends.mha.set_fastpath_enabled(True)

        row = {
            "model": name,
            "hand_flops": hand_total,
            "thop_flops": 2 * int(macs_default),
            "thop_flops_no_fastpath": 2 * int(macs_nofast),
            "thop_flops_no_fastpath_mha_rule": 2 * int(macs_custom),
            "flopcounter_flops": fc_default,
            "flopcounter_flops_no_fastpath": fc_nofast,
        }
        rows.append(row)

        lines.append(f"== {name} ==")
        lines.append("hand, per layer (FLOPs):")
        for layer, f in hand[name].items():
            lines.append(f"    {layer:<32} {f:>11,}")
        lines.append(f"    {'total':<32} {hand_total:>11,}")
        lines.append("thop.profile, per leaf module (MACs, x2 = FLOPs), default settings:")
        for path, m in flatten_layers(layers_default):
            lines.append(f"    {path:<44} {m:>9,}  -> {2 * m:>11,}")
        if name == "Attention":
            lines.append("thop.profile, fast path off + MultiheadAttention rule (MACs, x2 = FLOPs):")
            for path, m in flatten_layers(layers_custom):
                lines.append(f"    {path:<44} {m:>9,}  -> {2 * m:>11,}")
        for k, v in row.items():
            if k != "model":
                lines.append(f"{k:<34} {v:>11,}   ({v / hand_total:.3f} x hand)")
        lines.append("")

    lines.append("model      hand FLOPs   thop x2 (default)   ratio")
    for r in rows:
        lines.append(f"{r['model']:<9}  {r['hand_flops']:>10,}   {r['thop_flops']:>17,}   {r['thop_flops'] / r['hand_flops']:.3f}")

    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "flops.txt").write_text(text + "\n", encoding="utf-8")
    with open(RESULTS_DIR / "flops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
