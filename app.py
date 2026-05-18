"""
app.py — Streamlit Dashboard
==============================
Visualises benchmark results from outputs/report_cpu.json and report_cuda.json.
Run after benchmark.py completes:
    streamlit run app.py

On Google Colab:
    !streamlit run app.py &
    (use localtunnel or ngrok to expose the port)
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page config ────────────────────────────
st.set_page_config(
    page_title="GPU Efficiency Profiler",
    page_icon="⚡",
    layout="wide",
)

# ─── Load data ──────────────────────────────
OUTPUT_DIR = "./outputs"

@st.cache_data
def load_report(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

cpu_report  = load_report(f"{OUTPUT_DIR}/report_cpu.json")
gpu_report  = load_report(f"{OUTPUT_DIR}/report_cuda.json")
speedup     = load_report(f"{OUTPUT_DIR}/speedup_comparison.json")

# ─── Header ─────────────────────────────────
st.title("⚡ GPU Efficiency Profiler")
st.caption("CNN on CIFAR-10 — CPU vs GPU Training Benchmark | Built by Kavyanjali Karan")
st.divider()

# ─── Check data availability ────────────────
if not cpu_report:
    st.warning("No training data found. Run `python benchmark.py` first.")
    st.code("python benchmark.py --epochs 5 --batch_size 128", language="bash")
    st.stop()

# ─── KPI Cards ──────────────────────────────
has_gpu = gpu_report is not None and speedup is not None

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("CPU Avg Epoch", f"{cpu_report['avg_epoch_time_sec']:.2f}s")

with col2:
    if has_gpu:
        st.metric("GPU Avg Epoch", f"{gpu_report['avg_epoch_time_sec']:.2f}s",
                  delta=f"-{cpu_report['avg_epoch_time_sec'] - gpu_report['avg_epoch_time_sec']:.2f}s faster")
    else:
        st.metric("GPU Avg Epoch", "—", delta="Run with GPU to compare")

with col3:
    if has_gpu:
        st.metric("⚡ GPU Speedup", f"{speedup['time_speedup_x']}x",
                  delta=f"{speedup['throughput_gain_x']}x throughput")
    else:
        st.metric("⚡ GPU Speedup", "—")

with col4:
    if has_gpu and speedup.get("peak_gpu_memory_mb"):
        st.metric("Peak GPU VRAM", f"{speedup['peak_gpu_memory_mb']:.0f} MB")
    else:
        best_acc = cpu_report["best_val_acc"] * 100
        st.metric("CPU Best Acc", f"{best_acc:.2f}%")

st.divider()

# ─── Tabs ───────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Training Curves", "⚡ CPU vs GPU Benchmark", "🔍 Per-Epoch Detail", "📋 System Info"
])

# ──────────────────────────────────────────
# TAB 1: Training Curves
# ──────────────────────────────────────────
with tab1:
    st.subheader("Validation Accuracy Over Epochs")

    fig = go.Figure()

    # CPU trace
    cpu_epochs  = [m["epoch"] for m in cpu_report["epoch_metrics"]]
    cpu_val_acc = [m["val_acc"] * 100 for m in cpu_report["epoch_metrics"]]
    cpu_trn_acc = [m["train_acc"] * 100 for m in cpu_report["epoch_metrics"]]

    fig.add_trace(go.Scatter(
        x=cpu_epochs, y=cpu_val_acc,
        name="CPU Val Acc", mode="lines+markers",
        line=dict(color="#636EFA", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=cpu_epochs, y=cpu_trn_acc,
        name="CPU Train Acc", mode="lines",
        line=dict(color="#636EFA", width=1.5, dash="dot"),
    ))

    if has_gpu:
        gpu_epochs  = [m["epoch"] for m in gpu_report["epoch_metrics"]]
        gpu_val_acc = [m["val_acc"] * 100 for m in gpu_report["epoch_metrics"]]
        gpu_trn_acc = [m["train_acc"] * 100 for m in gpu_report["epoch_metrics"]]

        fig.add_trace(go.Scatter(
            x=gpu_epochs, y=gpu_val_acc,
            name="GPU Val Acc", mode="lines+markers",
            line=dict(color="#EF553B", width=2),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=gpu_epochs, y=gpu_trn_acc,
            name="GPU Train Acc", mode="lines",
            line=dict(color="#EF553B", width=1.5, dash="dot"),
        ))

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Accuracy (%)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Loss curves
    st.subheader("Training Loss Over Epochs")
    fig2 = go.Figure()
    cpu_loss = [m["train_loss"] for m in cpu_report["epoch_metrics"]]
    fig2.add_trace(go.Scatter(x=cpu_epochs, y=cpu_loss, name="CPU Loss",
                              line=dict(color="#636EFA", width=2)))
    if has_gpu:
        gpu_loss = [m["train_loss"] for m in gpu_report["epoch_metrics"]]
        fig2.add_trace(go.Scatter(x=gpu_epochs, y=gpu_loss, name="GPU Loss",
                                  line=dict(color="#EF553B", width=2)))
    fig2.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=360, hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────
# TAB 2: CPU vs GPU Benchmark
# ──────────────────────────────────────────
with tab2:
    if not has_gpu:
        st.info("GPU results not available. Run `python benchmark.py` on a CUDA-enabled machine (e.g. Google Colab T4).")
    else:
        st.subheader("Epoch Time: CPU vs GPU")

        col_a, col_b = st.columns(2)

        with col_a:
            # Bar chart: avg epoch time
            fig_bar = go.Figure(go.Bar(
                x=["CPU", "GPU"],
                y=[cpu_report["avg_epoch_time_sec"], gpu_report["avg_epoch_time_sec"]],
                marker_color=["#636EFA", "#EF553B"],
                text=[f"{cpu_report['avg_epoch_time_sec']:.2f}s", f"{gpu_report['avg_epoch_time_sec']:.2f}s"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title=f"Avg Epoch Time ({speedup['time_speedup_x']}x speedup)",
                yaxis_title="Seconds", height=360,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            # Throughput comparison
            fig_tp = go.Figure(go.Bar(
                x=["CPU", "GPU"],
                y=[cpu_report["avg_samples_per_sec"], gpu_report["avg_samples_per_sec"]],
                marker_color=["#636EFA", "#EF553B"],
                text=[f"{cpu_report['avg_samples_per_sec']:.0f}", f"{gpu_report['avg_samples_per_sec']:.0f}"],
                textposition="outside",
            ))
            fig_tp.update_layout(
                title=f"Throughput — images/sec ({speedup['throughput_gain_x']}x gain)",
                yaxis_title="Images/sec", height=360,
            )
            st.plotly_chart(fig_tp, use_container_width=True)

        # Per-epoch time comparison line chart
        st.subheader("Per-Epoch Time Comparison")
        fig_ep = go.Figure()
        fig_ep.add_trace(go.Scatter(
            x=cpu_epochs,
            y=[m["epoch_time_sec"] for m in cpu_report["epoch_metrics"]],
            name="CPU", mode="lines+markers", line=dict(color="#636EFA"),
        ))
        fig_ep.add_trace(go.Scatter(
            x=gpu_epochs,
            y=[m["epoch_time_sec"] for m in gpu_report["epoch_metrics"]],
            name="GPU", mode="lines+markers", line=dict(color="#EF553B"),
        ))
        fig_ep.update_layout(
            xaxis_title="Epoch", yaxis_title="Time (seconds)", height=360, hovermode="x unified"
        )
        st.plotly_chart(fig_ep, use_container_width=True)

        # GPU Memory over epochs
        if gpu_report.get("peak_gpu_memory_mb"):
            st.subheader("GPU Memory Usage Per Epoch")
            gpu_mem_series = [m.get("gpu_memory_mb") for m in gpu_report["epoch_metrics"] if m.get("gpu_memory_mb")]
            if gpu_mem_series:
                fig_mem = go.Figure(go.Scatter(
                    x=list(range(1, len(gpu_mem_series)+1)),
                    y=gpu_mem_series,
                    fill="tozeroy",
                    line=dict(color="#EF553B"),
                    name="GPU VRAM MB",
                ))
                fig_mem.update_layout(
                    xaxis_title="Epoch", yaxis_title="VRAM (MB)", height=300,
                )
                st.plotly_chart(fig_mem, use_container_width=True)


# ──────────────────────────────────────────
# TAB 3: Per-Epoch Detail Table
# ──────────────────────────────────────────
with tab3:
    st.subheader("CPU — Per-Epoch Metrics")
    cpu_df = pd.DataFrame(cpu_report["epoch_metrics"])
    cpu_df["val_acc"] = (cpu_df["val_acc"] * 100).round(2)
    cpu_df["train_acc"] = (cpu_df["train_acc"] * 100).round(2)
    st.dataframe(cpu_df, use_container_width=True)

    if has_gpu:
        st.subheader("GPU — Per-Epoch Metrics")
        gpu_df = pd.DataFrame(gpu_report["epoch_metrics"])
        gpu_df["val_acc"] = (gpu_df["val_acc"] * 100).round(2)
        gpu_df["train_acc"] = (gpu_df["train_acc"] * 100).round(2)
        st.dataframe(gpu_df, use_container_width=True)


# ──────────────────────────────────────────
# TAB 4: System Info
# ──────────────────────────────────────────
with tab4:
    st.subheader("Benchmark Configuration")

    info = {
        "Model": "CIFAR10_CNN (3 Conv blocks + 2 FC layers)",
        "Parameters": "~1.2M",
        "Dataset": "CIFAR-10 (50,000 train / 10,000 val)",
        "Optimizer": "AdamW (lr=1e-3, weight_decay=1e-4)",
        "LR Scheduler": "CosineAnnealingLR",
        "Loss": "CrossEntropyLoss (label_smoothing=0.1)",
        "Batch Size": str(cpu_report.get("epoch_metrics", [{}])[0].get("device", "—")),
        "CPU Epochs": str(cpu_report["total_epochs"]),
        "GPU Epochs": str(gpu_report["total_epochs"]) if has_gpu else "—",
    }

    for k, v in info.items():
        col_k, col_v = st.columns([1, 2])
        col_k.markdown(f"**{k}**")
        col_v.markdown(v)

    if has_gpu:
        st.divider()
        st.subheader("Speedup Summary")
        for k, v in speedup.items():
            col_k, col_v = st.columns([1, 2])
            col_k.markdown(f"**{k}**")
            col_v.markdown(str(v))
