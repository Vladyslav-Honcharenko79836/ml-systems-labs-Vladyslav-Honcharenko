import statistics
import time
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.manual_seed(0)


class TinyMLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


class FusableBlock(nn.Module):

    def __init__(self, d=256):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.nn.functional.gelu(h)
        h = h * torch.sigmoid(h) + 0.1 * h
        y = self.fc2(h)
        y = x + torch.tanh(y) * 0.5
        y = (y - y.mean(-1, keepdim=True)) / (y.std(-1, keepdim=True) + 1e-5)
        return y


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=10, iters=50, repeats=5):
    for _ in range(warmup):
        fn()
    sync(device)
    res = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync(device)
        res.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(res)


def part_a():
    print("=" * 78)
    print("PART A: computation graph of TinyMLP (2 -> 8 -> 1)")
    print("=" * 78)
    m = TinyMLP().eval()
    x = torch.randn(4, 2)

    gm = torch.fx.symbolic_trace(m)
    print("\n-- torch.fx graph (Python-level IR, one node per op) --")
    print(gm.graph)
    print("-- torch.fx generated code --")
    print(gm.code)

    traced = torch.jit.trace(m, x)
    print("-- torch.jit.trace graph (TorchScript IR, with shapes/dtypes) --")
    print(traced.graph)

    print("-- autograd graph of the eager forward (grad_fn chain) --")
    out = m(x).sum()
    fn, depth = out.grad_fn, 0
    while fn is not None:
        print("  " * depth + type(fn).__name__)
        nxt = [f for f, _ in fn.next_functions if f is not None and "AccumulateGrad" not in type(f).__name__]
        fn = nxt[0] if nxt else None
        depth += 1
    print()


def try_compile(model, x, device):
    err = None
    for backend in ("inductor", "aot_eager"):
        try:
            torch._dynamo.reset()
            cm = torch.compile(model, backend=backend)
            t0 = time.perf_counter()
            with torch.no_grad():
                cm(x)
            sync(device)
            return cm, backend, time.perf_counter() - t0, err
        except Exception as e:
            if err is None:
                msg = str(e).strip().splitlines()[0][:110]
                err = f"{type(e).__name__}: {msg}"
    return None, None, None, err


def bench_model(name, make_model, batch_sizes, device, d_in):
    rows = []
    for bs in batch_sizes:
        torch._dynamo.reset()
        model = make_model().to(device).eval()
        x = torch.randn(bs, d_in, device=device)

        with torch.no_grad():
            t_eager = timeit(lambda: model(x), device)

            traced = torch.jit.freeze(torch.jit.trace(model, x))
            t_trace = timeit(lambda: traced(x), device)
            max_err_tr = (traced(x) - model(x)).abs().max().item()

        cm, backend, t_first, err = try_compile(model, x, device)
        t_comp = None
        if cm is not None:
            with torch.no_grad():
                t_comp = timeit(lambda: cm(x), device)
        rows.append((bs, t_eager, t_trace, max_err_tr, t_comp, backend, t_first, err))

    print(f"\n{name} on {device.upper()}  (us per forward call, median, after warm-up)")
    print(f"{'batch':>7}{'eager':>10}{'jit.trace':>11}{'compile':>10}{'backend':>11}"
          f"{'trace/eag':>11}{'comp/eag':>10}{'compile 1st':>13}{'trace err':>11}")
    for bs, te, tt, me, tc, be, tf, err in rows:
        comp = f"{tc:>10.1f}" if tc else f"{'n/a':>10}"
        ratio_c = f"{te / tc:>9.2f}x" if tc else f"{'-':>10}"
        first = f"{tf:>12.2f}s" if tf else f"{'-':>13}"
        print(f"{bs:>7}{te:>10.1f}{tt:>11.1f}{comp}{(be or '-'):>11}"
              f"{te / tt:>10.2f}x{ratio_c}{first}{me:>11.1e}")
    for e in {r[-1] for r in rows if r[-1]}:
        print(f"  Inductor (real compilation) unsupported here -> fell back to "
              f"backend='aot_eager' + jit.trace. Reason: {e}")
    return rows


def part_b():
    print("=" * 78)
    print("PART B: eager vs torch.compile vs torch.jit.trace (inference)")
    print("=" * 78)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        bench_model("TinyMLP 2->8->1", TinyMLP, [1, 256, 8192], dev, 2)
        bench_model("FusableBlock d=256", FusableBlock, [1, 64, 1024], dev, 256)
    print()


def describe(name, t):
    print(f"  {name:<22} shape={str(tuple(t.shape)):<14} dtype={str(t.dtype):<14} "
          f"stride={str(t.stride()):<12} contiguous={str(t.is_contiguous()):<5} "
          f"shares_storage={t.untyped_storage().data_ptr() == base_ptr}")


def part_c():
    global base_ptr
    print("=" * 78)
    print("PART C: tensor shape, dtype, stride, contiguity")
    print("=" * 78)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    base_ptr = x.untyped_storage().data_ptr()
    describe("x", x)
    describe("x.t()", x.t())
    describe("x[:, ::2]", x[:, ::2])
    describe("x[1:, 1:]", x[1:, 1:])
    describe("x.view(4, 3)", x.view(4, 3))
    describe("x.t().contiguous()", x.t().contiguous())
    describe("x.half()", x.half())
    try:
        x.t().view(12)
    except RuntimeError as e:
        print(f"  x.t().view(12) -> RuntimeError: {str(e).splitlines()[0][:90]}...")
    print(f"  x.t().reshape(12) works (it silently copies): "
          f"shares_storage={x.t().reshape(12).untyped_storage().data_ptr() == base_ptr}")
    print()


def part_d():
    print("=" * 78)
    print("PART D: effect of a non-contiguous view on operation speed")
    print("=" * 78)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    N = 4096
    for dev in devices:
        a = torch.randn(N, N, device=dev)
        b = torch.randn(N, N, device=dev)
        bt = b.t()
        bt_c = bt.contiguous()
        s2 = torch.randn(N, 2 * N, device=dev)
        s2v = s2[:, ::2]
        s2c = s2v.contiguous()

        tests = [
            ("a + b      (elementwise)", lambda: a + bt_c, lambda: a + bt),
            ("b.sum(dim=1) (reduction)", lambda: bt_c.sum(dim=1), lambda: bt.sum(dim=1)),
            ("b * 2      (strided slice)", lambda: s2c * 2, lambda: s2v * 2),
            ("b.exp()    (unary)", lambda: bt_c.exp(), lambda: bt.exp()),
            ("a @ b      (matmul)", lambda: a @ bt_c, lambda: a @ bt),
        ]
        it = 5 if dev == "cpu" else 20
        print(f"\n{dev.upper()}, {N}x{N} float32 (ms, median)")
        print(f"{'operation':<30}{'contiguous':>12}{'non-contig':>12}{'slowdown':>10}")
        for name, fc, fn in tests:
            tc = timeit(fc, dev, warmup=2, iters=it, repeats=5) / 1000
            tn = timeit(fn, dev, warmup=2, iters=it, repeats=5) / 1000
            print(f"{name:<30}{tc:>12.2f}{tn:>12.2f}{tn / tc:>9.2f}x")
        t_copy = timeit(lambda: bt.contiguous(), dev, warmup=2, iters=it, repeats=5) / 1000
        print(f"{'cost of bt.contiguous() copy':<30}{t_copy:>12.2f}")
    print()


if __name__ == "__main__":
    torch.set_num_threads(4)
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, "
          f"cuda={torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    part_a()
    part_c()
    part_b()
    part_d()
