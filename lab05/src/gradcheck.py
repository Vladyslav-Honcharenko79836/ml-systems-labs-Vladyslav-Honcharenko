import random

import torch

from engine import Value, vsum, print_graph

TOL = 1e-6
torch.set_default_dtype(torch.float64)


def f_engine(a, b, c):
    return vsum(((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c))


def f_engine_relu_inside(a, b, c):
    return vsum(((ai * bi + ci).relu() ** 2) for ai, bi, ci in zip(a, b, c))


def f_float(a, b, c):
    return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c))


def f_float_relu_inside(a, b, c):
    return sum(max(ai * bi + ci, 0.0) ** 2 for ai, bi, ci in zip(a, b, c))


def numerical_grad(f, inputs, h=1e-6):
    grads = []
    for k in range(len(inputs)):
        g = []
        for i in range(len(inputs[k])):
            plus = [list(v) for v in inputs]
            minus = [list(v) for v in inputs]
            plus[k][i] += h
            minus[k][i] -= h
            g.append((f(*plus) - f(*minus)) / (2 * h))
        grads.append(g)
    return grads


def check(name, engine_fn, torch_fn, float_fn, raw):
    vals = [[Value(x) for x in v] for v in raw]
    out = engine_fn(*vals)
    out.backward()
    g_ours = [[x.grad for x in v] for v in vals]

    ts = [torch.tensor(v, requires_grad=True) for v in raw]
    t_out = torch_fn(*ts)
    t_out.backward()
    g_torch = [t.grad.tolist() for t in ts]

    g_num = numerical_grad(float_fn, raw)

    def maxdiff(g1, g2):
        return max(abs(x - y) for v1, v2 in zip(g1, g2) for x, y in zip(v1, v2))

    d_fwd = abs(out.data - t_out.item())
    d_torch = maxdiff(g_ours, g_torch)
    d_num = maxdiff(g_ours, g_num)
    ok = d_fwd < TOL and d_torch < TOL and d_num < TOL
    print(f"{name}")
    print(f"  forward   ours={out.data:.10f}  torch={t_out.item():.10f}  |diff|={d_fwd:.2e}")
    print(f"  max |grad_ours - grad_torch|     = {d_torch:.2e}")
    print(f"  max |grad_ours - grad_numerical| = {d_num:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'} (tol {TOL:g})\n")
    return ok


def accumulation_test():
    x = Value(3.0)
    y = x * x + x
    y.backward()
    ok1 = abs(x.grad - 7.0) < 1e-12

    a = Value(2.0)
    b = a * 3
    c = a ** 2
    d = b * c
    d.backward()
    ok2 = abs(a.grad - 36.0) < 1e-12
    print("Gradient accumulation")
    print(f"  y = x*x + x, x=3: dy/dx = {x.grad} (expected 7)  -> {'PASS' if ok1 else 'FAIL'}")
    print(f"  diamond d = (3a)*(a^2), a=2: dd/da = {a.grad} (expected 36) -> {'PASS' if ok2 else 'FAIL'}\n")
    return ok1 and ok2


def main():
    random.seed(0)
    N = 10
    raw = [[random.uniform(-2, 2) for _ in range(N)] for _ in range(3)]

    a, b, c = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c")
    L = ((a * b + c) ** 2).relu()
    L.backward()
    print("Graph of ((a*b + c)**2).relu() for a=2, b=-3, c=10 after backward():")
    print(print_graph(L))
    print()

    results = [
        check(
            "Task 4: ((a*b + c)**2).relu().sum(), N=10",
            f_engine,
            lambda a, b, c: ((a * b + c) ** 2).relu().sum(),
            f_float,
            raw,
        ),
        check(
            "Extra: ((a*b + c).relu()**2).sum()  (ReLU active on some elements)",
            f_engine_relu_inside,
            lambda a, b, c: ((a * b + c).relu() ** 2).sum(),
            f_float_relu_inside,
            raw,
        ),
        accumulation_test(),
    ]
    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
