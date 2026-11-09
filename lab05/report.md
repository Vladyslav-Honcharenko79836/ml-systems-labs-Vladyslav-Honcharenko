# Lab 5: Frameworks and automatic differentiation

Environment used for the measurements: Python 3.14.2, PyTorch 2.14.1+cu126, NumPy 2.5.4, Windows 10
(not the pinned course set from `requirements.txt`),
GPU NVIDIA GeForce GTX 1060 3GB. scikit-learn is not installed, so
`make_moons` is re-implemented with the same algorithm as sklearn (two noisy
half-circles). It uses sklearn automatically if it is installed.

| File | Tasks | What it does |
|---|---|---|
| `src/engine.py` | 1, 2, 3 | `Value` node, operations, topological backward pass |
| `src/gradcheck.py` | 4 | engine vs PyTorch autograd vs numerical gradients |
| `src/train_moons.py` | 5 | 2→8→1 MLP on make_moons, engine vs PyTorch timing (median of 5) |
| `src/torch_inspect.py` | 6 | PyTorch graphs, eager vs compile/trace, strides and contiguity |
| `src/common.py` | — | paths; prints a script's output to the console and saves it to `results/<script>.txt` |
| `results/*.txt` | — | output of the three scripts, written by the scripts themselves |

Run from the repository root:

```bash
python lab05/src/gradcheck.py      # -> lab05/results/gradcheck.txt      (~1 s)
python lab05/src/train_moons.py    # -> lab05/results/train_moons.txt    (~1.5 min)
python lab05/src/torch_inspect.py  # -> lab05/results/torch_inspect.txt  (~1.5 min)
```

All numbers in this report are taken from those files.

---

## 1. `Value` node [25]

`Value` (in `engine.py`) stores:

* `data`: the forward value (float)
* `grad`: ∂output/∂node, filled in by `backward()`, initially 0
* `_prev`: the parent nodes (edges of the graph)
* `_op`: the operation label
* `_backward`: the **local derivative rule**, a closure that applies the chain
  rule and pushes `out.grad` into the parents' `grad`. For leaf nodes it does nothing.

`__slots__` keeps each node small, because there are about 10⁴ nodes per training step in task 5.

## 2. Operations build the graph [20]

`+`, `*`, `**k` (constant k) and `relu()` each return a new `Value` whose
`_prev` points to the operands and whose `_backward` holds the local rule:

| op | local rule |
|---|---|
| `a + b` | `a.grad += out.grad`, `b.grad += out.grad` |
| `a * b` | `a.grad += b.data*out.grad`, `b.grad += a.data*out.grad` |
| `a ** k` | `a.grad += k*a.data**(k-1)*out.grad` |
| `relu(a)` | `a.grad += (out.data > 0)*out.grad` |

`-`, `/`, and the reflected operators (`2*x`, `1-x`, ...) are built from
these primitives. `exp`, `log` and `sigmoid` were added for task 5. So the
graph is recorded **during the forward pass**, just like PyTorch eager mode
(define-by-run). Graph dump for `((a*b+c)**2).relu()` with a=2, b=-3, c=10
after `backward()`:

```
[relu] data=16 grad=1
  [**2] data=16 grad=1
    [+] data=4 grad=8
      [*] data=-6 grad=8
        [a] data=2 grad=-24
        [b] data=-3 grad=16
      [c] data=10 grad=8
```

## 3. Backward pass [25]

`Value.backward()`:

1. **Topological sort** with an iterative DFS, so deep graphs do not hit
   Python's recursion limit. A node is added only after all of its parents.
2. **Seed** `self.grad = 1.0`.
3. **Reverse order**: nodes are visited from output to inputs, so every
   consumer of a node has already added its share before the node's own
   rule runs.
4. **Accumulation**: every local rule uses `+=`, never `=`. A node used
   several times gets the sum of the contributions of all its paths.

Accumulation tests (in `gradcheck.py`): `y = x*x + x` at x=3 gives `dy/dx = 7` ✓.
The diamond `d = (3a)·(a²)` at a=2 gives `dd/da = 36` ✓. With `=` instead of
`+=` these would be 3 and 12/24.

## 4. Verification against PyTorch and numerical gradients [10]

`L = ((a*b + c)**2).relu().sum()` with a, b, c vectors of 10 random values in
[-2, 2]. PyTorch runs in float64, and the numerical gradient uses central
differences with h = 1e-6.

| check | max abs difference |
|---|---|
| forward value, engine vs torch | 0.0 |
| gradients, engine vs torch autograd | **0.0** |
| gradients, engine vs numerical | **1.8e-9** |

Both are below the required 1e-6 → **PASS**. In this formula ReLU comes after
squaring, so its input is always ≥ 0 and it never clips. For that reason a
second case `((a*b+c).relu()**2).sum()` was also checked, where ReLU *is*
active for some elements: 0.0 vs torch, 1.5e-9 vs numerical → PASS.
Full output: `results/gradcheck.txt`.

## 5. make_moons training, engine vs PyTorch [15]

Setup: make_moons with n=200 and noise=0.1. MLP 2→8→1 with ReLU in the
hidden layer and a sigmoid output, BCE loss, full-batch SGD with lr=0.5 for
300 epochs. **The same data and the same initial weights** are used in both
implementations. PyTorch is limited to 1 thread for a fair comparison with
single-threaded Python. Each configuration is timed 5 times (full training),
and the table reports the median.

| implementation | median time, s | speedup vs engine | final loss | train acc |
|---|---|---|---|---|
| our engine (Python scalars, float64) | 17.19 | 1× | 0.260817 | 0.890 |
| PyTorch CPU float64 | 0.137 | **125×** | 0.260817 | 0.890 |
| PyTorch CPU float32 | 0.138 | 125× | 0.260816 | 0.890 |
| PyTorch CUDA float32 | 0.333 | 52× | 0.260816 | 0.890 |

Engine runs: 17.18, 17.28, 17.19, 17.16, 17.32 s. The spread between runs is under 1%.

The final loss of the engine and of PyTorch float64 differ by only 5e-12,
so the engine computes exactly the same training run. The ~125× slowdown is
pure **framework overhead**. Every scalar op in the engine is one Python
object plus one closure, about 10⁴ nodes per epoch. PyTorch does the same
work as a few vectorized ATen kernels on (200×2) and (200×8) tensors. The GPU
is *slower* than the CPU here: the tensors are tiny, so the run is bound by
kernel-launch and synchronisation overhead, not by FLOPs. Full output:
`results/train_moons.txt`.

Note: the `_backward` closures create reference cycles (node → closure →
node). Python's cyclic GC has to collect them, so `gc.collect()` is called
before each timed run. An earlier version of the script without it (and with
n=300 instead of 200) showed engine times growing from 26 s to 65 s over
the 5 runs.

## 6. PyTorch graphs, eager vs compiled, tensor layout [20]

Full output: `results/torch_inspect.txt`.

### 6a. Graph of a small model

`TinyMLP` (2→8→1) is captured three ways:

* **torch.fx** (`symbolic_trace`): a Python-level IR with nodes
  `placeholder → fc1 → relu → fc2 → sigmoid → output`.
* **torch.jit.trace**: TorchScript IR in which each value carries a concrete
  shape, dtype and stride from the example input, e.g.
  `Float(4, 8, strides=[8, 1], device=cpu) = aten::relu(...)`.
* **autograd graph** (`grad_fn` chain, the PyTorch equivalent of our
  `_prev`/`_backward`): `SumBackward0 → SigmoidBackward0 → AddmmBackward0 →
  ReluBackward0 → AddmmBackward0 → TBackward0`.

### 6b. Eager vs torch.compile vs torch.jit.trace

**torch.compile with the default Inductor backend does not work on this
machine.** On the CPU it needs the MSVC C++ compiler (`InvalidCxxCompiler:
cl is not found`). On the GPU it needs Triton, which does not support the
GTX 1060 (`GPUTooOldForTriton`, Pascal sm_61). As the lab allows, the script
falls back to:

* **torch.jit.trace + torch.jit.freeze** (tracing). On CUDA, TorchScript's
  fuser can merge elementwise chains.
* **torch.compile(backend="aot_eager")**. Dynamo still captures the graph and
  AOTAutograd traces it, but the ops run as normal eager kernels with no code
  generation. This isolates the cost of graph capture and guards.

Timings are µs per forward call, median of 5×50 calls after 10 warm-up calls, inference mode:

| model / device | batch | eager | jit.trace | compile (aot_eager) | trace speedup |
|---|---|---|---|---|---|
| TinyMLP, CPU | 1 | 43.9 | 41.7 | 148.8 | 1.05× |
| TinyMLP, CPU | 8192 | 294.7 | 281.0 | 416.6 | 1.05× |
| FusableBlock, CPU | 1 | 300.7 | 245.9 | 465.7 | 1.22× |
| FusableBlock, CPU | 64 | 1380.9 | 1378.3 | 1638.7 | 1.00× |
| FusableBlock, CPU | 1024 | 13177 | 13024 | 13703 | 1.01× |
| TinyMLP, CUDA | 1 | 204.6 | 188.4 | 318.6 | 1.09× |
| TinyMLP, CUDA | 8192 | 195.8 | 224.0 | 308.0 | 0.87× |
| FusableBlock, CUDA | 1 | 488.4 | **239.5** | 755.6 | **2.04×** |
| FusableBlock, CUDA | 64 | 577.6 | **271.7** | 746.6 | **2.13×** |
| FusableBlock, CUDA | 1024 | 1025.7 | **618.9** | 940.1 | **1.66×** |

`FusableBlock` = Linear(256→1024) → GELU → `h*sigmoid(h)+0.1h` → Linear →
`x + 0.5·tanh(y)` → manual layer-norm, i.e. two matmuls plus a long chain of
elementwise and reduction ops. The traced outputs match eager to within ≤7e-7.

Observations:

* On CUDA, tracing gives a **1.7–2.1× speedup for FusableBlock**. The ~10
  elementwise ops are fused into a few kernels, so there are fewer launches
  (each costs about 10–20 µs on this GPU) and fewer round trips to memory. The
  gain is largest at small batch, where launch overhead dominates, and
  smaller at batch 1024, where the matmuls take more of the time.
* On the CPU, tracing gives almost nothing (0–5%, up to 22% only at batch 1).
  The CPU kernels are already called with little overhead and the CPU
  TorchScript fuser is off by default, so only the Python dispatch for the
  small batch is saved.
* For TinyMLP on CUDA the run is pure launch overhead (~190–200 µs at any
  batch size). Tracing changes it by ±10%, which is within run-to-run noise.
* `compile(aot_eager)` is **slower** than eager for small inputs: it adds the
  cost of the Dynamo guards and the wrapper but no fused kernels. It only
  comes close to eager at large batch (0.96× on CPU, 1.09× on CUDA). The first call (graph capture) takes
  about 0.1–0.15 s, the same as hundreds of eager calls.

### 6c. Tensor shape, dtype, stride, contiguity

For `x = arange(12).reshape(3,4)`, float32:

| tensor | shape | stride | contiguous | shares storage |
|---|---|---|---|---|
| `x` | (3,4) | (4,1) | True | — |
| `x.t()` | (4,3) | (1,4) | **False** | yes (view) |
| `x[:, ::2]` | (3,2) | (4,2) | **False** | yes (view) |
| `x[1:, 1:]` | (2,3) | (4,1) | **False** | yes (view) |
| `x.view(4,3)` | (4,3) | (3,1) | True | yes (view) |
| `x.t().contiguous()` | (4,3) | (3,1) | True | **no (copy)** |
| `x.half()` | (3,4) | (4,1) | True | no (new dtype, copy) |

`x.t().view(12)` raises a RuntimeError ("view size is not compatible with
input tensor's size and stride"). `x.t().reshape(12)` works, but silently
makes a copy.

### 6d. Cost of a non-contiguous view (4096×4096 float32, ms)

| operation | CPU contig. | CPU non-contig. | slowdown | CUDA contig. | CUDA non-contig. | slowdown |
|---|---|---|---|---|---|---|
| `a + b.t()` | 7.65 | 70.12 | **9.2×** | 1.34 | 2.30 | **1.7×** |
| `b.t().sum(dim=1)` | 1.75 | 7.57 | **4.3×** | 0.48 | 0.54 | 1.13× |
| `b[:, ::2] * 2` | 6.08 | 8.17 | 1.34× | 0.91 | 1.33 | 1.46× |
| `b.t().exp()` | 6.91 | 6.09 | 0.88× | 0.91 | 0.90 | 0.99× |
| `a @ b.t()` | 945.9 | 998.1 | 1.06× | 42.1 | 40.4 | 0.96× |
| `b.t().contiguous()` (copy) | 38.16 | | | 1.75 | | |

* The worst case is **mixing layouts**: `a` is row-major and `b.t()` is
  column-major. At least one operand is then read with stride 4096, so every
  element comes from a different cache line (CPU) or the memory accesses are
  not coalesced (GPU). 9.2× slower on the CPU.
* `exp()` on a transposed tensor costs nothing extra. With a single operand,
  TensorIterator simply reorders the loop to follow memory order.
* A reduction along the "wrong" axis is 4.3× slower on the CPU, because it
  can no longer stream along contiguous memory with SIMD.
* A strided slice `[:, ::2]` uses only half of every cache line → about 1.3–1.5×.
* `matmul` barely cares (within ±6%). BLAS/cuBLAS take the transpose as a flag
  (`op(B)=Bᵀ`), so `x.t()` costs nothing as a matmul input.
* An explicit `.contiguous()` costs as much as about 4 elementwise ops on the
  CPU. It pays off only when the tensor is reused several times.

---

## Short note: when compilation helps

Compilation (torch.compile/Inductor, TorchScript, XLA) helps when the
**overhead of eager execution is a large part of the runtime**, and the
compiled graph is **reused many times** with stable shapes:

* **Chains of elementwise and reduction ops** (activations, normalization,
  residuals, masking). Fusing them into one kernel removes kernel launches
  and memory traffic for intermediate tensors. Measured here: up to **2.1×**
  on GPU for FusableBlock with tracing alone, at small and medium batch.
* **Small batches / small tensors on GPU**, where the run is bound by
  launches rather than FLOPs (CUDA graphs via `mode="reduce-overhead"` target exactly this).
* **Long-running training or serving**, where the one-time cost of
  compilation (≥0.1 s here just for graph capture, seconds to minutes with
  Inductor/autotuning) is spread over thousands of steps.

It does **not** help (or makes things worse) when:

* the time goes into **large matmuls/convolutions** that already call
  cuBLAS/cuDNN/MKL. FusableBlock at batch 1024 on the CPU gave 1.01×;
* the model is **tiny and runs only a few times**, so compile time dominates;
* **shapes or control flow keep changing**, causing recompilations and graph breaks;
* the **toolchain is missing**, as on this machine: Inductor needs a C++
  compiler on the CPU and Triton on a GPU of compute capability ≥ 7.0. A
  "compile" without code generation (`aot_eager`) only adds guard overhead,
  measured here as 0.3–0.8× of eager speed on small inputs.

For non-contiguous tensors, the same reasoning applies: a compiler (or a
single explicit `.contiguous()`) helps if the strided tensor is used by
several elementwise ops that mix layouts. Matmul and single-operand ops do
not need it.
