import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from bench import measure
from common import (MUTED, RESULTS_DIR, SERIES, SURFACE, env_line, load_config, style_axes, tee_results, write_csv)
from data import load_tensors
from zoo import load_model


def csr(w):
    c = w.to_sparse_csr()
    return torch.sparse_csr_tensor(c.crow_indices().int(), c.col_indices().int(), c.values(), c.shape)


if __name__ == "__main__":
    cfg = load_config()
    lat = cfg["latency"]
    x = load_tensors(cfg, torch.device("cpu"))["test"][0][:1].clone()
    sp = cfg["prune"]["sparsities"]
    fns, meta = {}, {}
    with tee_results("sparse_kernels.txt"):
        print(env_line())
        print(f"1 CPU thread, per-layer-pruned seed-42 weights, real activations of one test image; median of "
              f"{lat['rounds']} x {lat['runs']} runs, shuffled order\n"
              "fc1   dense = F.linear (MKL gemm/gemv); csr = torch CSR matrix x vector\n"
              "conv4 dense = F.conv2d (oneDNN); unfold+dense = im2col then matmul; unfold+csr = im2col then CSR x dense; "
              "csr only = CSR x dense on ready-made im2col columns\n"
              "model = whole network with dense kernels; model+csr = CSR for fc1 and for conv2-conv4 (im2col)")
        with torch.no_grad():
            for s in sp:
                m = load_model(cfg, "baseline" if s == 0 else f"layer_{s}")
                net = m.net
                a3 = net.pool1(net.relu2(net.conv2(net.relu1(net.conv1(x)))))
                a4 = net.relu3(net.conv3(a3))
                f1 = net.flatten(net.pool2(net.relu4(net.conv4(a4))))
                w_fc, b_fc = net.fc1.weight.detach(), net.fc1.bias.detach()
                w_fc_csr = csr(w_fc)
                conv_csr = {c: csr(getattr(net, c).weight.detach().flatten(1)) for c in ("conv2", "conv3", "conv4")}
                w4, b4 = net.conv4.weight.detach(), net.conv4.bias.detach()
                w4d, w4s = w4.flatten(1), conv_csr["conv4"]
                cols = F.unfold(a4, 3, padding=1)[0]

                def conv_sparse(inp, name, net=net, conv_csr=conv_csr):
                    c = getattr(net, name)
                    u = F.unfold(inp, 3, padding=1)[0]
                    o = (conv_csr[name] @ u).add_(c.bias[:, None])
                    return o.view(1, c.out_channels, inp.shape[2], inp.shape[3])

                def model_sparse(net=net, w=w_fc_csr, b=b_fc, conv_sparse=conv_sparse):
                    h = net.relu1(net.conv1(x))
                    h = net.pool1(net.relu2(conv_sparse(h, "conv2")))
                    h = net.relu3(conv_sparse(h, "conv3"))
                    h = net.pool2(net.relu4(conv_sparse(h, "conv4")))
                    h = net.relu5((w @ h.flatten()[:, None]).T + b)
                    return net.fc2(h)

                assert (model_sparse() - m(x)).abs().max() < 1e-3
                cases = {
                    "fc1 dense": lambda f=f1, w=w_fc, b=b_fc: F.linear(f, w, b),
                    "fc1 csr": lambda f=f1, w=w_fc_csr, b=b_fc: (w @ f.T).T + b,
                    "conv4 dense": lambda a=a4, w=w4, b=b4: F.conv2d(a, w, b, padding=1),
                    "conv4 unfold+dense": lambda a=a4, w=w4d: w @ F.unfold(a, 3, padding=1)[0],
                    "conv4 unfold+csr": lambda a=a4, w=w4s: w @ F.unfold(a, 3, padding=1)[0],
                    "conv4 csr only": lambda u=cols, w=w4s: w @ u,
                    "model": lambda m=m: m(x),
                    "model+csr": model_sparse,
                }
                for k, fn in cases.items():
                    fns[(k, s)] = fn
                meta[s] = 1
        res = measure(fns, lat["warmup"], lat["rounds"], lat["runs"], threads=1)
        kinds = list(dict.fromkeys(k for k, _ in fns))
        rows = []
        print(f"\nmedian ms\n{'sparsity':>8} " + " ".join(f"{k:>19}" for k in kinds))
        for s in sp:
            row = {"sparsity": s, **{k: round(res[(k, s)]["median"], 5) for k in kinds}}
            rows.append(row)
            print(f"{s:>8} " + " ".join(f"{row[k]:>19.4f}" for k in kinds))
    write_csv(rows, "sparse_kernels.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [("fc1 layer (6272 → 128)", [("fc1 dense", MUTED, "x", "dense F.linear"), ("fc1 csr", SERIES[0], "o", "CSR")]),
              ("conv4 layer (128 → 128, 14×14)", [("conv4 dense", MUTED, "x", "dense conv2d (oneDNN)"),
                                                   ("conv4 unfold+dense", SERIES[1], "s", "im2col + dense matmul"),
                                                   ("conv4 unfold+csr", SERIES[0], "o", "im2col + CSR"),
                                                   ("conv4 csr only", SERIES[2], "^", "CSR only, im2col excluded")]),
              ("whole model", [("model", MUTED, "x", "dense kernels"), ("model+csr", SERIES[0], "o", "CSR fc1, conv2–conv4")])]
    for ax, (title, series), ticks in zip(axes, panels, ([0.05, 0.1, 0.2, 0.5, 1], [0.2, 0.5, 1, 2, 5], [3, 4, 6, 8, 12])):
        for k, c, mk, label in series:
            ax.plot([100 * r["sparsity"] for r in rows], [r[k] for r in rows], color=c, marker=mk, lw=2, ms=7,
                    ls="--" if c == MUTED else "-", label=label)
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        style_axes(ax, title, "weight sparsity, %", "median latency, ms (log scale)")
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("CSR kernels against dense kernels on the pruned weights, 1 CPU thread, batch 1", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "sparse_kernels.png", dpi=150, facecolor=SURFACE)
