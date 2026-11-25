import gzip
import io
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import INK, MUTED, RESULTS_DIR, SERIES, SURFACE, env_line, load_config, style_axes, tee_results, write_csv
from models import nonzero_params, n_params
from zoo import load_model, names


def split(state):
    return {k: v for k, v in state.items() if k.endswith("weight")}, {k: v for k, v in state.items()
                                                                       if not k.endswith("weight")}


def save_dense(state, path):
    torch.save(state, path)


def load_dense(path, shapes):
    return torch.load(path)


def save_gzip(state, path):
    buf = io.BytesIO()
    torch.save(state, buf)
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(buf.getvalue())


def load_gzip(path, shapes):
    with gzip.open(path, "rb") as f:
        return torch.load(io.BytesIO(f.read()))


def save_coo(state, path):
    w, rest = split(state)
    torch.save({**{k: v.flatten(1).to_sparse_coo().coalesce() for k, v in w.items()}, **rest}, path)


def load_coo(path, shapes):
    return {k: (v.to_dense().view(shapes[k]) if v.is_sparse else v) for k, v in torch.load(path).items()}


def save_csr(state, path):
    w, rest = split(state)
    out = dict(rest)
    for k, v in w.items():
        c = v.flatten(1).to_sparse_csr()
        out[k] = torch.sparse_csr_tensor(c.crow_indices().int(), c.col_indices().int(), c.values(), c.shape)
    torch.save(out, path)


def load_csr(path, shapes):
    return {k: (v.to_dense().view(shapes[k]) if v.layout == torch.sparse_csr else v)
            for k, v in torch.load(path).items()}


def save_bitmask(state, path):
    w, rest = split(state)
    out = {f"d:{k}": v.numpy() for k, v in rest.items()}
    for k, v in w.items():
        a = v.numpy().ravel()
        out[f"m:{k}"] = np.packbits(a != 0)
        out[f"v:{k}"] = a[a != 0]
    with open(path, "wb") as f:
        np.savez(f, **out)


def load_bitmask(path, shapes):
    z = np.load(path)
    out = {}
    for key in z.files:
        kind, k = key.split(":", 1)
        if kind == "d":
            out[k] = torch.from_numpy(z[key])
        elif kind == "m":
            n = int(np.prod(shapes[k]))
            mask = np.unpackbits(z[key])[:n].astype(bool)
            a = np.zeros(n, dtype=np.float32)
            a[mask] = z[f"v:{k}"]
            out[k] = torch.from_numpy(a).view(shapes[k])
    return out


FORMATS = {"dense .pt": (save_dense, load_dense), "dense .pt.gz": (save_gzip, load_gzip),
           "COO .pt": (save_coo, load_coo), "CSR .pt": (save_csr, load_csr),
           "bitmask .npz": (save_bitmask, load_bitmask)}


if __name__ == "__main__":
    cfg = load_config()
    rows = []
    with tee_results("storage.txt"):
        print(env_line())
        print("on-disk bytes of the fine-tuned seed-42 models; every file is loaded back and compared with the "
              "original weights (bit-exact)")
        print("COO: 2 x int64 + float32 = 20 B per non-zero; CSR: int32 column + float32 = 8 B per non-zero + "
              "4 B per row; bitmask: 4 B per non-zero + 1 bit per weight")
        print(f"\n{'model':<16} {'params':>9} {'non-zero':>9} " + " ".join(f"{f:>13}" for f in FORMATS) + "   [kB]")
        with tempfile.TemporaryDirectory() as tmp:
            for name in names(cfg):
                model = load_model(cfg, name)
                state = {k: v.clone() for k, v in model.state_dict().items()}
                shapes = {k: v.shape for k, v in state.items()}
                row = {"model": name, "params": n_params(model), "nonzero": nonzero_params(model)}
                for fmt, (save, load) in FORMATS.items():
                    path = Path(tmp) / "m.bin"
                    save(state, path)
                    back = load(path, shapes)
                    assert set(back) == set(state)
                    assert all(torch.equal(back[k], state[k]) for k in state), (name, fmt)
                    row[fmt] = path.stat().st_size
                rows.append(row)
                print(f"{name:<16} {row['params']:>9,} {row['nonzero']:>9,} "
                      + " ".join(f"{row[f] / 1000:>13,.1f}" for f in FORMATS))
        base = rows[0]["dense .pt"]
        print(f"\nsize relative to the dense baseline ({base:,} bytes)")
        for r in rows:
            print(f"{r['model']:<16} " + "  ".join(f"{f} {base / r[f]:5.2f}x" for f in FORMATS))
    write_csv(rows, "storage.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    g = [r for r in rows if r["model"] == "baseline" or r["model"].startswith("global_")]
    xs = [0.0] + [100 * float(r["model"].split("_")[1]) for r in g[1:]]
    for (fmt, c, mk) in zip(FORMATS, [MUTED, SERIES[3], SERIES[1], SERIES[0], SERIES[2]], ["x", "D", "s", "o", "^"]):
        ys = [r[fmt] / 1e6 for r in g]
        ax.plot(xs, ys, color=c, marker=mk, lw=2, ms=7, label=fmt, ls="--" if fmt == "dense .pt" else "-")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    style_axes(ax, "File size of globally pruned models by storage format", "weight sparsity, %",
               "file size, MB (log scale)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "storage.png", dpi=150, facecolor=SURFACE)
