# Lab 6 — Training pipelines and project proposal

## 1. Goal

Turn the best Lab 3 model (the CNN, 89.2 % test accuracy) into a reproducible training pipeline
driven by one YAML config, with seeding, checkpoints and MLflow tracking. Then use the pipeline to
profile an epoch, find the input-pipeline bottleneck, study batch size, LR schedules and
optimizers, emulate a large batch with gradient accumulation, resume training after a kill, and
write the individual project proposal.

| Learning outcome | Evidence in this report |
|---|---|
| Restructure training into a reproducible YAML-driven pipeline | §3.1; `config.yaml`, `src/pipeline.py`, `src/train.py`; identical metrics in 3 independent runs |
| Use MLflow and PyTorch profiling to record experiments and identify bottlenecks | §3.2 profile breakdown (`results/profile.txt`); §3.3 worker study; 30 MLflow runs (`results/mlflow_runs.txt`) |
| Analyze worker count, batch size, throughput, memory, optimizer, and scheduler trade-offs | §3.3–§3.5 study tables and plots |
| Resume interrupted training and prepare a feasible individual project proposal | §3.6 gradient accumulation; §3.7 kill + resume, bit-identical curves; §3.8 `proposal.md` |

## 2. Method

**Hardware and software.** AMD Ryzen 7 2700X (8 cores / 16 threads), 16 GB RAM, NVIDIA GeForce
GTX 1060 3 GB, Windows 10 Pro 19045. Python 3.14.2, PyTorch 2.14.1+cu126, torchvision 0.29.1,
MLflow 3.18.0, PyYAML 6.0.4, matplotlib 3.11.2 (not the pinned course set from
`requirements.txt`, which has no CUDA build and no Python 3.14 wheels; same setup as Lab 5).

**Data and model.** As in Lab 3: Fashion-MNIST, `ToTensor()` + `Normalize(0.2860, 0.3530)`,
55 000 / 5 000 train/validation split of the official training set (seed 42), official 10 000 test
images used once after training. CNN: Conv(1→16, 3×3) → ReLU → MaxPool → Conv(16→32, 3×3) → ReLU →
MaxPool → Linear(1568→10), 20 490 parameters. Training uses the real `DataLoader` with the
per-image torchvision transform; the validation and test sets are converted to tensors once
(identical values, `x/255` then normalize) so evaluation does not interfere with the input-pipeline
measurements.

**Files.**

| File | Step | What it does |
|---|---|---|
| `config.yaml` | 1 | the single config: seed, determinism, device, paths, data, model, optimizer, scheduler, training, MLflow |
| `src/common.py` | — | config loading and `--set key=value` overrides, seeding, result writers, plot style |
| `src/data.py` | 1 | datasets, `DataLoader` with its own seeded sampler generator, tensor loader for compute-only runs |
| `src/models.py` | 1 | the Lab 3 CNN |
| `src/pipeline.py` | 1, 6, 7 | optimizer/scheduler factories, training epoch with gradient accumulation, checkpoints, resume, MLflow tracker |
| `src/train.py` | 1, 7 | CLI: `--config`, `--set`, `--resume` |
| `src/profile_epoch.py` | 2 | `torch.profiler` breakdown of one epoch |
| `src/workers_study.py` | 3 | `num_workers` ∈ {0, 1, 2, 4, 8} on GPU and CPU |
| `src/batch_study.py` | 4 | batch ∈ {32 … 512}, 5 epochs each |
| `src/schedule_study.py` | 5 | {StepLR, Cosine, Plateau} × {Adam, SGD+momentum}, 10 epochs each |
| `src/grad_accum.py` | 6 | batch 64 × 4 accumulation vs batch 256 |
| `src/resume_demo.py` | 7 | kill at epoch 3, resume, compare with an uninterrupted run |
| `src/export_mlflow.py` | — | exports the MLflow runs table to `results/mlflow_runs.{csv,txt}` |
| `proposal.md` | 8 | individual project proposal |

Run from the repository root (each script writes its own `results/*.txt`, `*.csv`, `*.png`):

```bash
python lab06/src/train.py --save-output train_base.txt          # step 1, ~1 min
python lab06/src/train.py --set optim.name=sgd optim.lr=0.05 sched.name=cosine run_name=sgd-cos
python lab06/src/train.py --resume data/lab06_runs/<run>/last.pt  # continue a stopped run
python lab06/src/profile_epoch.py      # step 2, ~10 min (the profiler is slow with num_workers=0)
python lab06/src/workers_study.py      # step 3, ~8 min
python lab06/src/batch_study.py        # step 4, ~4 min
python lab06/src/schedule_study.py     # step 5, ~8 min
python lab06/src/grad_accum.py         # step 6, ~3 min
python lab06/src/resume_demo.py        # step 7, ~2 min
python lab06/src/export_mlflow.py
mlflow ui --backend-store-uri sqlite:///lab06/mlruns/mlflow.db   # browse the runs
```

Checkpoints go to `data/lab06_runs/<run_name>/` and the MLflow store to `lab06/mlruns/`; both are
git-ignored, the exported runs table is committed.

## 3. Results

### 3.1 YAML-driven pipeline [20]

Everything that changes a run lives in `config.yaml`; the scripts hard-code nothing else.
Any key can be overridden from the command line (`--set data.batch_size=256 sched.name=cosine`),
and the studies build their variants with the same override function, so every experiment is
"base config + a few named changes".

* **Seed and determinism.** `seed_everything(42)` seeds Python, NumPy, PyTorch CPU and CUDA, sets
  `cudnn.deterministic = True` and `cudnn.benchmark = False`. The split uses its own generator
  (seed 42) and the shuffling sampler gets its own `torch.Generator` (seed 42), separate from the
  generator that seeds the workers. The base run, the `batch-128` run of §3.4 (first 5 epochs) and
  the `resume-reference` run of §3.7 were started separately and produced **identical** train/val
  metrics for every epoch.
* **Checkpoints.** After every epoch `last.pt` is written, and `best.pt` when validation accuracy
  improves. A checkpoint holds the model, optimizer state (Adam moments and step counts), scheduler
  state, sampler-generator state, Torch/CUDA/NumPy/Python RNG states, the full history, the best
  validation accuracy, the config and the MLflow run id. Files are written to `*.tmp` and then
  renamed with `os.replace`, so a crash during saving cannot leave a half-written checkpoint.
* **MLflow.** Local store `sqlite:///lab06/mlruns/mlflow.db`, experiment `lab06-fashion-cnn`.
  Each run logs all 32 config values as **params** (flattened, e.g. `optim.lr`), per-epoch
  **metrics** (`train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`, `epoch_time_s`,
  `samples_per_s`), final metrics (`test_loss`, `test_acc`, `best_val_acc`, `peak_mem_mb`), tags
  (device, GPU, torch version) and **artifacts** (`config.yaml`, `history.csv`, per-step
  `steps.csv`, `checkpoints/best.pt`, `checkpoints/last.pt`). All 30 runs of this lab are listed in
  `results/mlflow_runs.txt`.

Base run (`results/train_base.txt`): Adam lr 1e-3, batch 128, no scheduler, 10 epochs,
4 workers, GPU.

| epoch | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| train loss | 0.528 | 0.352 | 0.285 | 0.250 | 0.227 | 0.209 |
| val acc | 84.04 % | 88.20 % | 89.60 % | 89.96 % | **90.90 %** | 90.80 % |

**Test accuracy 90.35 %** (Lab 3, 5 epochs on CPU: 89.21 %), 18 600 samples/s (2.9–3.0 s per
epoch, epoch 1: 22 s because of worker start-up, see §3.3), peak GPU memory during training 47 MB.

### 3.2 Profile of one epoch [15]

`torch.profiler` (CPU + CUDA activities) records one full epoch (430 steps) after a warm-up epoch.
Each step is split into four `record_function` regions; each region ends with
`torch.cuda.synchronize()`, otherwise the asynchronous GPU work of forward/backward would be
charged to whatever region happens to wait for it next. With `num_workers=0` the profiler records
about 6 600 operator events per step (every per-image `ToTensor`/`Normalize` op), which ran out of
memory (> 10 GB) on the first attempt, so events are now collected in chunks of 10 steps with
`schedule(active=10)` and summed. Share = region time / total of the `ProfilerStep` spans. The
"timer" column times the same regions with `perf_counter` in an unprofiled epoch, as a check.
Full output: `results/profile.txt`; top operators: `results/profile_ops_*.txt`.

| region | GPU, 0 workers: profiler | timer | GPU, 4 workers: profiler | timer | CPU, 0 workers: profiler | timer |
|---|---|---|---|---|---|---|
| data loading | **64.1 %** | **67.3 %** | 8.3 % | 27.3 % | **70.7 %** | **63.5 %** |
| forward | 9.3 % | 8.0 % | 17.0 % | 13.2 % | 11.9 % | 15.1 % |
| backward | 24.0 % | 21.8 % | **51.2 %** | **41.0 %** | 14.6 % | 19.0 % |
| optimizer step | 1.4 % | 1.7 % | 12.4 % | 10.4 % | 2.2 % | 1.9 % |
| other (metrics, loop) | 1.3 % | 1.2 % | 11.1 % | 8.2 % | 0.6 % | 0.5 % |
| epoch, s | 27.3 | 16.7 | 3.0 | 2.9 | 25.7 | 18.5 |

* **The default setup (`num_workers=0`, as in Lab 3) is input-bound.** Two thirds of the epoch
  is spent producing batches, on GPU and on CPU alike. The top CPU operator of the CPU run is
  `enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__` (11.4 s self time), followed by
  per-image ops called exactly 55 000 times (`aten::sub_`, `aten::permute`, `aten::any`, ...):
  PIL image → `ToTensor` → `Normalize`, one image at a time, in Python, on one core.
* **With 4 workers the GPU run becomes compute-bound**: data loading drops to 8–27 % (the batch
  is usually already waiting in the queue) and backward is the largest region. On GPU backward costs
  about 2.7–3.1× the forward pass, as expected (gradients for both the inputs and the weights of
  every layer); the top GPU kernel is `aten::convolution_backward`.
* **Optimizer step** is small (1–2 %) when the epoch is slow and 10–12 % when it is fast; it is a
  fixed ~0.3 s per epoch (430 Adam updates of 20 k parameters, launch-bound).
* **Profiler overhead** is large with 0 workers (+63 % on GPU, +39 % on CPU: every per-image op is
  recorded) and near zero with 4 workers. The two columns therefore differ in absolute time, but
  the shares agree within a few points. One exception: with 4 workers the profiled loop is slower,
  so the workers have more time to prefetch and the profiler shows less waiting for data (8 % vs
  27 % in the unprofiled epoch).
* GPU backward takes 3.6 s per epoch with 0 workers but 1.2 s with 4 workers for the same
  kernels. The likely cause is that the GPU sits idle ~25 ms between steps while the main process
  decodes images and drops to lower clocks. This was not measured directly.

### 3.3 `num_workers` study [15]

`results/workers_study.{txt,csv,png}`. Batch 128, persistent workers, `pin_memory` on GPU. For each
setting: time to the first batch, train epoch (median of epochs 2–3), and a data-only pass
(loader + host-to-device copy, no model). Compute-only reference: the same training with the whole
train set already in memory as tensors (no DataLoader).

| device | workers | 1st batch s | epoch s | samples/s | data-only s | compute-only s |
|---|---|---|---|---|---|---|
| GPU | 0 | 0.09 | 12.46 | 4 414 | 11.43 | 1.28 |
| GPU | 1 | 5.89 | 10.98 | 5 007 | 10.97 | 1.28 |
| GPU | 2 | 10.55 | 5.57 | 9 879 | 5.51 | 1.28 |
| GPU | 4 | 21.01 | 2.94 | 18 713 | 2.86 | 1.28 |
| GPU | 8 | 42.04 | **1.73** | **31 864** | 1.57 | 1.28 |
| CPU | 0 | 0.10 | 18.63 | 2 952 | 10.73 | 6.82 |
| CPU | 1 | 5.36 | 11.91 | 4 616 | 10.89 | 6.82 |
| CPU | 2 | 10.54 | **7.33** | 7 501 | 5.52 | 6.82 |
| CPU | 4 | 21.08 | 7.36 | 7 474 | 2.85 | 6.82 |
| CPU | 8 | 42.00 | 7.32 | 7 515 | 1.56 | 6.82 |

![workers study](results/workers_study.png)

**The bottleneck is per-image decoding and transformation in the DataLoader.** One process
produces only ~5 000 images/s. On GPU the train-epoch curve lies on the data-only curve at every
worker count: the GPU finishes a step faster than the loader can produce the next batch. Data
throughput scales almost linearly with workers (5.0 k → 10.0 k → 19.2 k → 35.1 k images/s for
1/2/4/8), and the epoch time falls from 12.5 s to 1.7 s (7.2×). Even 8 workers do not quite reach
the compute-only limit of 1.28 s, so the GPU run is still slightly input-bound.
`num_workers=1` gives almost no gain over 0: it is still one decoding process, only moved out of
the main process.

On **CPU** the picture changes after 2 workers: the epoch stops at 7.3 s ≈ compute-only 6.8 s.
The model itself (8 intra-op threads) is now the bottleneck, and adding workers only adds idle
processes.

**Start-up cost on Windows.** Workers are started with `spawn`. Each one starts a new Python
interpreter, imports torch and torchvision and receives a pickled copy of the dataset, which costs
~5.3 s per worker, one after another: 21 s for 4 workers and 42 s for 8. With persistent workers
this is paid once per run, but for a 10-epoch run at 3 s/epoch, 21 s of start-up is ~40 % of the
wall time. 8 workers save 1.2 s per epoch and cost 21 s more at start-up, so they only pay off
after ~17 epochs. The config default stays at **4 workers**.

### 3.4 Batch-size study [15]

`results/batch_study.{txt,csv,png}`. Adam lr 1e-3 (the same for every batch), no scheduler,
5 epochs, 4 workers. Throughput = median of epochs 2–5 with the real DataLoader; "GPU-only" = same
model and batch with the data already on the GPU; peak memory = `torch.cuda.max_memory_allocated`
during training.

| batch | steps/epoch | samples/s | GPU-only samples/s | peak memory MB | final train loss | test acc |
|---|---|---|---|---|---|---|
| 32 | 1719 | 9 891 | 12 459 | 25.5 | 0.219 | **90.79 %** |
| 64 | 860 | 16 579 | 25 035 | 32.4 | 0.242 | 89.58 % |
| 128 | 430 | 18 758 | 43 103 | 46.8 | 0.265 | 89.03 % |
| 256 | 215 | 18 954 | 47 442 | 82.0 | 0.290 | 88.60 % |
| 512 | 108 | 18 933 | 50 469 | 146.4 | 0.329 | 87.61 % |

![batch study](results/batch_study.png)

* **Throughput.** It rises from 9.9 k to 18.8 k samples/s between batch 32 and 128, then stays flat
  at ~19 k. The flat part is exactly the 4-worker data-only limit from §3.3 (19.2 k images/s):
  above batch 128 the GPU could go faster (GPU-only 43 k → 50 k), but the loader cannot feed it.
  At small batches the cost per step dominates (kernel launches, Python, optimizer step): with
  data already on the GPU, batch 32 reaches only 12.5 k samples/s, ¼ of batch 512.
* **Memory** grows linearly with batch above 128: (146.4 − 82.0) / 256 ≈ **0.25 MB per image**
  (activations kept for backward: 16×28×28 and 32×14×14 feature maps before and after ReLU/pool,
  in float32). There is a fixed ~18 MB (cuDNN workspace, weights, gradients and Adam state for 20 k
  parameters). All sizes fit easily into the 3 GB card.
* **Accuracy falls with batch size** (90.8 % → 87.6 %), and so does training loss, so this is
  under-training, not a generalisation gap. The learning rate and the 5-epoch budget are fixed, so
  batch 512 gets 16× fewer updates (540 vs 8 595). Keeping large batches competitive would need a
  higher learning rate (linear scaling) and warm-up.
* **Trade-off on this machine:** batch 128 is the best point: full loader throughput, 47 MB of
  memory, and only 0.4 pp less accuracy than batch 64 after 5 epochs. Batch 32 is most accurate
  per epoch but runs at half the speed.

### 3.5 LR schedules and optimizers, 10-epoch budget [15]

`results/schedule_study.{txt,csv,png}`, `schedule_curves.csv`. Batch 128, 10 epochs per run, same
seed and data order. Adam lr 1e-3; SGD lr 0.05, momentum 0.9. StepLR: step 4, γ 0.1.
CosineAnnealingLR: T_max 10. ReduceLROnPlateau: on validation loss, factor 0.5, patience 1.

| optimizer | schedule | LR at epoch 10 | final train loss | best val acc (epoch) | test acc |
|---|---|---|---|---|---|
| Adam | StepLR | 1.0e-5 | 0.233 | 90.56 % (10) | 89.83 % |
| Adam | Cosine | 2.4e-5 | 0.216 | 90.80 % (10) | 90.28 % |
| Adam | Plateau | 1.0e-3 (never reduced) | 0.209 | 90.90 % (8) | 90.35 % |
| SGD+m | StepLR | 5.0e-4 | 0.174 | 91.58 % (9) | 91.12 % |
| SGD+m | **Cosine** | 1.2e-3 | **0.152** | **91.68 %** (10) | **91.56 %** |
| SGD+m | Plateau | 2.5e-2 (halved after epoch 6) | 0.169 | 91.56 % (10) | 91.20 % |

![schedule study](results/schedule_study.png)

* **SGD+momentum beats Adam by about 1 pp with every schedule** (91.1–91.6 % vs 89.8–90.4 % test)
  under the same 10-epoch budget, and also reaches a lower training loss. Wall-clock cost is the
  same: every run is ~18.7 k samples/s and the optimizer step is only 1–10 % of a step (§3.2).
  SGD keeps one momentum buffer per weight, Adam two moments.
* **Cosine** is best or close to best for both optimizers: the LR stays high early, when progress
  is fastest, and decays smoothly towards the end.
* **StepLR with step 4** cuts the LR by 10× after epoch 4 and again after epoch 8. For Adam this is
  too early: after the first drop validation accuracy improves only slowly (89.6 → 90.6 % over
  6 epochs), and it is the worst run (89.83 %). For SGD the drop after epoch 4 gives a visible jump (val 90.3 → 90.8 → 91.4 % in epochs 4–6, training loss 0.255 → 0.199).
* **ReduceLROnPlateau** reacts to the validation loss. With patience 1 it needs two epochs in a row
  without improvement. For SGD that happened at epochs 5–6, the LR was halved, and validation
  accuracy jumped from 90.0 to 91.3 %. For Adam the validation loss never rose twice in a row, so
  the LR was never reduced and the run is identical to the base run (90.35 %).
* Caveat: one seed per configuration. The validation set has 5 000 images (standard error ≈ 0.4 pp
  at 91 %), so differences between schedules of 0.1–0.4 pp are within noise. The ~1 pp gap between
  SGD and Adam is the same under all three schedules, so it is the robust result.

### 3.6 Gradient accumulation [10]

`results/grad_accum.{txt,csv,png}`. Batch 64 with 4 accumulation steps (effective batch 256) vs a
real batch of 256, both with Adam lr 1e-3, 5 epochs, the same initial weights and the same shuffled
sample order (the sampler generator does not depend on batch size, so 4 consecutive batches of 64
are exactly one batch of 256). Each micro-batch loss is multiplied by
`micro_batch_size / group_size` before `backward()`, so the summed gradient equals the mean over
the 256 samples. This also holds for the last, shorter group (216 = 3 × 64 + 24 samples). The
optimizer steps after every 4th micro-batch and at the end of the epoch. A third run (batch 64,
no accumulation) shows what accumulation is *not*.

One-step check on the first 256 images: max |∇(batch 256) − Σ ∇(4 × batch 64)/4| = **6.3e-8**,
with gradients up to 6.4e-2, i.e. float32 rounding.

| run | batch × accum | optimizer steps | samples/s | peak MB | final train loss | final val acc | test acc |
|---|---|---|---|---|---|---|---|
| b256 | 256 × 1 | 1075 | 18 242 | 82.8 | 0.2899 | 89.20 % | 88.60 % |
| b64x4 | 64 × 4 | 1075 | 16 970 | **31.4** | 0.2898 | 89.18 % | 88.63 % |
| b64 | 64 × 1 | 4300 | 16 366 | 31.8 | 0.2417 | 90.38 % | 89.58 % |

![gradient accumulation](results/grad_accum.png)

**The training behaviour is the same.** Per-epoch training loss agrees to 4 decimals (0.62689 vs
0.62689 in epoch 1, 0.28988 vs 0.28983 in epoch 5) and validation accuracy to within 0.04 pp. The
per-step loss curves lie on top of each other. Their difference starts at float32 rounding (~1e-9)
and grows to ~1e-3 over 1 075 steps, because small rounding differences are amplified by the
training dynamics, but it never moves the curves apart. After 5 epochs the largest weight
difference is 1.4e-2 for b256 vs b64x4, against 0.48 for b256 vs plain b64. Accumulation keeps
**the memory of batch 64** (31 MB instead of 83 MB) at a 7 % throughput cost (4× more,
smaller forward/backward calls). Plain batch 64 behaves differently (4× more updates, higher
accuracy after 5 epochs), so accumulation reproduces the large batch, not the small one.

### 3.7 Interrupt at epoch 3 and resume [10]

`results/resume_demo.{txt,csv,png}`. `resume_demo.py` first runs the base config for 10 epochs
without interruption (reference). It then starts the same config in a child process, waits until
the epoch-3 checkpoint is on disk and **kills the process** (`TerminateProcess`, exit code 1) while
epoch 4 is running. Finally it runs `train.py --resume data/lab06_runs/resume-interrupted/last.pt`.
On resume the config comes from the checkpoint, and model, optimizer, scheduler, sampler generator
and all RNG states are restored. Training continues at epoch 4, and the process re-opens the
**same MLflow run** (run id stored in the checkpoint; tag `resumed_from_epoch=3`), so the MLflow
curves are one continuous run that ends as FINISHED.

| epoch | 1–3 (before the kill) | 4–10 (after resume) |
|---|---|---|
| max \|difference\| from the uninterrupted run (train/val loss, acc, lr) | 0.0 | **0.0** |

![resume](results/resume_demo.png)

The resumed curves match the reference **bit for bit** (test accuracy 90.35 % in both). Exact
resumption needed three things:

1. the full optimizer state (Adam's moment estimates), not just the weights;
2. a dedicated generator for the shuffle sampler, saved in the checkpoint. With the DataLoader's
   own generator, a new set of persistent workers draws one extra seed from it at resume time, and
   the epoch-4 shuffle would then differ from the uninterrupted run;
3. deterministic cuDNN kernels.

The work of the unfinished epoch 4 is lost and repeated. The resumed run also pays the 21 s
worker start-up again (epoch 4: 23.5 s).

### 3.8 Individual project proposal [15]

`proposal.md`: **on-device keyword spotting** on Google Speech Commands v0.02 (12 classes).
Success metric: ≥ 90 % top-1 test accuracy with ≤ 1 pp loss from optimization. Constraints:
**≤ 50 ms** per clip (feature extraction + inference, ONNX Runtime, 1 CPU thread, batch 1) and
**≤ 5 MB** model file. Planned optimization: **INT8 quantization** (post-training static, then
quantization-aware training if the drop exceeds 1 pp), trained with this lab's pipeline.

## 4. Conclusions

* The YAML pipeline makes runs reproducible: separately started runs give identical metrics, and a
  run killed mid-epoch resumes onto exactly the same trajectory. Every run is tracked in MLflow
  with config, curves and checkpoints.
* For this small CNN the training loop is not the bottleneck; the **input pipeline is**. With the
  Lab 3 default (`num_workers=0`) two thirds of every epoch goes to per-image decoding. Workers give
  a 7× speed-up on GPU (12.5 s → 1.7 s per epoch) but cost ~5 s each to start on Windows. On CPU the
  model becomes the bottleneck after 2 workers.
* Larger batches stop helping as soon as the loader saturates (~19 k images/s with 4 workers).
  Memory grows by 0.25 MB per image. Without LR scaling they lose accuracy because of fewer
  updates.
* Under a fixed 10-epoch budget **SGD+momentum with cosine decay** is the best recipe (91.56 %
  test, +1.2 pp over the Lab 3-style Adam run). Gradient accumulation reproduces batch-256 training
  at batch-64 memory.
