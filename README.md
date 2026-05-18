# ⚡ GPU Efficiency Profiler

**A production-grade benchmarking system that trains a CNN on CIFAR-10 and measures — with real hardware metrics — exactly why CUDA GPUs exist.**

Built by [Kavyanjali Karan](https://linkedin.com/in/kavyanjali-karan) | [GitHub](https://github.com/karankavyanjali77-sys)

---

## What this project does

Most ML projects just *use* a GPU. This one **measures, quantifies, and visualises** the difference.

| Metric | CPU (Intel Colab) | GPU (T4 CUDA) | Speedup |
|---|---|---|---|
| Avg Epoch Time | ~85s | ~12s | **~7x faster** |
| Throughput | ~590 img/s | ~4,100 img/s | **~7x more** |
| Best Val Accuracy | ~74% | ~76% | +2% |
| Peak VRAM | — | ~680 MB | — |

> *Numbers above are from a real Google Colab T4 run. Your results will vary by hardware.*

---

## Architecture

```
gpu_profiler/
├── models/
│   └── cnn.py           # CIFAR10_CNN — 3 Conv blocks, ~1.2M params
├── utils/
│   ├── data_loader.py   # CIFAR-10 download, augmentation, DataLoader
│   └── profiler.py      # Epoch timing, CUDA memory, throughput tracking
├── train.py             # Full training loop (CPU or GPU, CLI flags)
├── benchmark.py         # Runs CPU + GPU back-to-back, saves comparison JSON
├── app.py               # Streamlit dashboard — live charts of all metrics
└── outputs/             # Auto-generated: report_cpu.json, report_cuda.json
```

**Separated train / benchmark / UI architecture** — each module is independently runnable. No monolithic notebook.

---

## Quickstart — Google Colab (recommended)

```python
# Step 1: Clone
!git clone https://github.com/karankavyanjali77-sys/gpu-efficiency-profiler.git
%cd gpu-efficiency-profiler

# Step 2: Install
!pip install -r requirements.txt

# Step 3: Run full benchmark (CPU + GPU back-to-back)
!python benchmark.py --epochs 5 --batch_size 128

# Step 4: Launch dashboard
!pip install streamlit pyngrok -q
from pyngrok import ngrok
!streamlit run app.py &
public_url = ngrok.connect(8501)
print(public_url)
```

> Make sure Colab runtime is set to **GPU (T4)**: Runtime → Change runtime type → T4 GPU

---

## Local setup

```bash
git clone https://github.com/karankavyanjali77-sys/gpu-efficiency-profiler.git
cd gpu-efficiency-profiler
pip install -r requirements.txt

# Train on GPU only
python train.py --device cuda --epochs 10 --save_model

# Full benchmark (CPU then GPU)
python benchmark.py --epochs 5

# Dashboard
streamlit run app.py
```

---

## Key engineering decisions

### Why `pin_memory=True` in DataLoader?
Pins CPU tensors in page-locked memory, enabling faster async CPU→GPU transfer via DMA. Measurably reduces data-loading bottleneck.

### Why `zero_grad(set_to_none=True)`?
Sets gradients to `None` instead of zero — skips memory write for unused parameters. Slightly faster per step at scale.

### Why `torch.cuda.synchronize()` before timing?
CUDA kernels are asynchronous. Without sync, `time.perf_counter()` measures CPU scheduling time, not actual GPU compute time — giving misleading (too fast) numbers.

### Why `CosineAnnealingLR`?
Avoids sharp LR drops. Reaches similar final accuracy as step-decay with fewer epochs — important when benchmarking with only 5–10 epochs.

### Why label smoothing in CrossEntropyLoss?
Prevents overconfident softmax outputs. Improves generalisation on CIFAR-10 by ~0.5–1% val accuracy.

---

## What the Streamlit dashboard shows

- **Training curves** — val/train accuracy + loss per epoch (CPU vs GPU overlaid)
- **Epoch time comparison** — bar chart + per-epoch line chart
- **Throughput** — images/second processed (CPU vs GPU)
- **GPU VRAM usage** — memory allocated per epoch
- **Full per-epoch metrics table** — downloadable

---

## Skills demonstrated

`PyTorch` · `CUDA` · `CNN architecture` · `GPU memory profiling` · `Throughput benchmarking` · `FastAPI-ready inference architecture` · `Streamlit deployment` · `Data engineering (ETL pipeline design)` · `Systems thinking`

---

## Results interpretation

A **7x speedup** on 5 epochs with batch_size=128 means:
- Training 100 epochs on CPU: ~142 minutes
- Training 100 epochs on GPU: ~20 minutes
- At scale (ResNet-50, ImageNet): this difference is hours vs days

This is why NVIDIA hardware is not a luxury — it's the difference between iterating once a day vs iterating 7 times a day.

---

## Dev Container

A `.devcontainer/devcontainer.json` is included for reproducible development in VS Code or GitHub Codespaces — matching the architecture of my other production systems.

---

*Part of my production ML portfolio — [github.com/karankavyanjali77-sys](https://github.com/karankavyanjali77-sys)*
