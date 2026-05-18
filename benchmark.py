"""
benchmark.py — GPU Efficiency Profiler
========================================
Runs CPU training + GPU training back-to-back and produces a
combined speedup comparison JSON.

Usage (Google Colab — recommended):
    python benchmark.py --epochs 5 --batch_size 128

Usage (CPU only — no GPU available):
    python benchmark.py --epochs 3 --cpu_only
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import run_training
from utils.profiler import compute_speedup, BenchmarkReport


def run_benchmark(args):
    print("\n" + "█"*55)
    print("  GPU EFFICIENCY PROFILER — FULL BENCHMARK")
    print("█"*55)

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # ── Phase 1: CPU run ──────────────────────
    print("\n▶  PHASE 1: CPU Training")
    cpu_args = argparse.Namespace(
        device="cpu",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        num_workers=0,    # 0 workers on CPU avoids multiprocessing overhead on Colab
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        save_model=False,
    )
    cpu_report = run_training(cpu_args)
    results["cpu"] = cpu_report

    # ── Phase 2: GPU run ──────────────────────
    if not args.cpu_only:
        if not torch.cuda.is_available():
            print("\n⚠  No CUDA GPU found. Skipping GPU phase.")
            print("   On Google Colab: Runtime → Change runtime type → GPU (T4)")
        else:
            print("\n▶  PHASE 2: GPU (CUDA) Training")
            gpu_args = argparse.Namespace(
                device="cuda",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=1e-3,
                num_workers=2,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                save_model=True,
            )
            gpu_report = run_training(gpu_args)
            results["gpu"] = gpu_report

            # ── Speedup comparison ────────────
            speedup = compute_speedup(cpu_report, gpu_report)
            speedup_path = f"{args.output_dir}/speedup_comparison.json"
            with open(speedup_path, "w") as f:
                json.dump(speedup, f, indent=2)

            print("\n" + "█"*55)
            print("  BENCHMARK RESULTS")
            print("█"*55)
            print(f"  CPU avg epoch    : {speedup['cpu_avg_epoch_sec']:.2f}s")
            print(f"  GPU avg epoch    : {speedup['gpu_avg_epoch_sec']:.2f}s")
            print(f"  ⚡ Speedup        : {speedup['time_speedup_x']}x faster on GPU")
            print(f"  📈 Throughput     : {speedup['throughput_gain_x']}x more images/sec")
            print(f"  🎯 GPU Best Acc   : {speedup['gpu_best_acc']*100:.2f}%")
            print(f"  💾 Peak GPU VRAM  : {speedup['peak_gpu_memory_mb']} MB")
            print(f"\n  Comparison saved : {speedup_path}")
            print("█"*55 + "\n")

    else:
        print("\n  CPU-only mode: skipping GPU phase.")
        print(f"  CPU Best Acc: {cpu_report.best_val_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Efficiency Profiler — Full Benchmark")
    parser.add_argument("--epochs",     type=int, default=5,          help="Epochs per run (5 is enough to show speedup)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--cpu_only",   action="store_true",           help="Skip GPU phase")
    args = parser.parse_args()

    run_benchmark(args)
