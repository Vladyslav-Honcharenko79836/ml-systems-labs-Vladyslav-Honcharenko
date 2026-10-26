"""The three Lab 3 architectures, built exactly as the lab manual specifies them."""
import torch
from torch import nn

from data import RESULTS_DIR

N_CLASSES = 10
PATCH = 7                      # 28x28 image -> 4x4 grid of 7x7 patches
N_PATCHES = (28 // PATCH) ** 2  # 16
PATCH_DIM = PATCH * PATCH       # 49
D_MODEL, N_HEAD, D_FF, N_LAYERS = 64, 4, 128, 2


class MLP(nn.Module):
    """Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Conv(1,16,3,p1) -> ReLU -> MaxPool2 -> Conv(16,32,3,p1) -> ReLU -> MaxPool2 -> Flatten -> Linear(32*7*7, 10)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x28x28 -> 16x14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32x14x14 -> 32x7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def patchify(x):
    """(B, 1, 28, 28) -> (B, 16, 49): row-major 4x4 grid of non-overlapping 7x7 patches."""
    b = x.shape[0]
    p = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)  # (B, 1, 4, 4, 7, 7)
    return p.reshape(b, N_PATCHES, PATCH_DIM)


class AttentionNet(nn.Module):
    """16 patches of 7x7 -> Linear 49->64 (+ learned position embedding) -> 2 Transformer encoder layers
    (d_model 64, 4 heads, FF 128) -> mean pool over patches -> Linear 64->10.

    The manual does not mention position information. Without it, self-attention plus mean pooling is
    invariant to the order of the patches, so the model could not tell top from bottom. We add the
    standard ViT learned position embedding (16 x 64 = 1,024 parameters, under 1.5% of the model).
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(PATCH_DIM, D_MODEL)
        self.pos = nn.Parameter(torch.zeros(1, N_PATCHES, D_MODEL))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEAD, dim_feedforward=D_FF, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.head = nn.Linear(D_MODEL, N_CLASSES)

    def forward(self, x):
        z = self.proj(patchify(x)) + self.pos  # (B, 16, 64)
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": AttentionNet}


def hand_param_count():
    """Parameter counts from the layer shapes (weights + biases), to cross-check torchinfo."""
    lin = lambda i, o: i * o + o
    conv = lambda k, ci, co: k * k * ci * co + co
    ln = lambda d: 2 * d
    enc_layer = (3 * D_MODEL * D_MODEL + 3 * D_MODEL   # QKV in_proj
                 + lin(D_MODEL, D_MODEL)               # attention out_proj
                 + lin(D_MODEL, D_FF) + lin(D_FF, D_MODEL)
                 + 2 * ln(D_MODEL))
    return {
        "MLP": lin(784, 256) + lin(256, 128) + lin(128, 10),
        "CNN": conv(3, 1, 16) + conv(3, 16, 32) + lin(32 * 7 * 7, 10),
        "Attention": lin(PATCH_DIM, D_MODEL) + N_PATCHES * D_MODEL + N_LAYERS * enc_layer + lin(D_MODEL, 10),
    }


if __name__ == "__main__":
    from torchinfo import summary

    torch.manual_seed(0)
    hand = hand_param_count()
    x = torch.randn(2, 1, 28, 28)

    # patchify sanity check: patch k = (row k // 4, col k % 4) of the 4x4 grid
    img = torch.arange(784.0).reshape(1, 1, 28, 28)
    patch_ok = torch.equal(patchify(img)[0, 6].reshape(7, 7), img[0, 0, 7:14, 14:21])

    lines, table = [], ["model      torchinfo   by hand   in 100k-150k target"]
    for name, cls in MODELS.items():
        model = cls()
        s = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                    col_names=("input_size", "output_size", "num_params"), depth=4)
        n = s.total_params
        assert n == sum(p.numel() for p in model.parameters())
        out = model(x)
        table.append(f"{name:<9}  {n:>9,}  {hand[name]:>8,}  {str(100_000 <= n <= 150_000):>6}"
                     f"   (output {tuple(out.shape)})")
        lines += [f"===== {name} =====", str(s), ""]

    table.append(f"patchify places patch 6 at rows 7:14, cols 14:21: {patch_ok}")
    text = "\n".join(table)
    print(text)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "params.txt").write_text(text + "\n\n" + "\n".join(lines), encoding="utf-8")
