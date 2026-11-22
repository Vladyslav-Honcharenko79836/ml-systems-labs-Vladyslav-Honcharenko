# Individual project proposal

**Student:** Vladyslav Honcharenko  **Student ID:** ____________
**Course:** Introduction to Machine Learning Systems  **Date:** 22.11.2026

## Title

On-device keyword spotting: a sub-5 MB, sub-50 ms speech-command classifier with INT8 quantization

## Problem

Voice interfaces on cheap devices (smart switches, toys, wearables) must recognise a few spoken
commands ("yes", "no", "up", "down", "stop", "go", ...) locally: sending audio to the cloud adds
latency, needs a network connection and leaks private speech. The task is to classify a 1-second
audio clip into one of 12 classes (10 command words + "unknown word" + "silence") on a CPU-only
device, within a fixed latency and memory budget.

## Dataset

**Google Speech Commands v0.02** (Warden, 2018; CC BY 4.0), loaded with
`torchaudio.datasets.SPEECHCOMMANDS`: 105 829 one-second 16 kHz clips of 35 words from 2 618
speakers. Official `validation_list.txt` / `testing_list.txt` split, which is by speaker, so no
speaker appears in two splits. The 25 non-target words become "unknown"; "silence" clips are cut
from the bundled background-noise recordings. Input features: 40-band log-Mel spectrogram
(25 ms window, 10 ms hop → 40 × 98). Training augmentation: random time shift ±100 ms and
background noise mixing.

## Success metrics

| Metric | Target |
|---|---|
| Top-1 accuracy on the official test split (12 classes) | **≥ 90 %** for the final optimized model |
| Accuracy drop caused by the optimization | ≤ 1.0 percentage point vs the FP32 model |
| Macro-F1 (checks that "unknown" and "silence" are not traded off) | reported, ≥ 0.88 |

## Deployment constraints

| Constraint | Budget | How it is measured |
|---|---|---|
| Latency | **≤ 50 ms** per 1-s clip, feature extraction + inference | ONNX Runtime, CPU, batch 1, 1 thread, median and p95 of 1 000 runs after warm-up (Lab 3 protocol) |
| Model size | **≤ 5 MB** | size of the exported `.onnx` file on disk |
| Hardware | CPU only, no GPU | AMD Ryzen 7 2700X limited to 1 thread as a stand-in for an edge CPU |
| Privacy | audio never leaves the device | inference runs fully offline |

## Model and planned optimization

**Baseline model (FP32):** a depthwise-separable CNN for keyword spotting (DS-CNN, ~5–6 conv
blocks, 64–172 channels, global average pooling, linear head), trained with the Lab 6 pipeline
(YAML config, seed, checkpoints, MLflow). Expected: ~0.5 M parameters ≈ 2 MB in FP32, already
under 5 MB, so the optimization targets latency and margin rather than only size.

**Planned optimization: INT8 quantization.**

1. Post-training static quantization (ONNX Runtime `quantize_static`, per-channel weights,
   calibration on 500 training clips).
2. If the accuracy drop is above 1 pp: quantization-aware training in PyTorch
   (`torch.ao.quantization`, fake-quant for 3–5 epochs), then export.

Expected effect: about 4× smaller weights (≈ 0.5 MB) and 1.5–3× lower CPU latency.
Fallback if INT8 kernels do not speed up the model on this CPU: width reduction of the DS-CNN
with knowledge distillation from the FP32 model.

## Evaluation plan and deliverables

* FP32 vs INT8 table: accuracy, macro-F1, size, median/p95 latency, peak memory; confusion matrix.
* MLflow runs for every training and quantization experiment; Lab 6 profiler breakdown of the
  final pipeline (feature extraction vs inference).
* Final artifact: INT8 `.onnx` model + a small offline demo script that classifies a `.wav` file.

**Risks:** "unknown" class imbalance (handled with a weighted sampler); log-Mel computation may
take a large share of the 50 ms budget (measured separately and optimized first if needed).
