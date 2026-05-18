"""
GPU/CPU Profiler Utility
Measures: epoch time, batch throughput, GPU memory usage, CPU utilization.
This is the core differentiator of this project — most students never profile.
"""

import time
import json
import psutil
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class EpochMetrics:
    epoch: int
    device: str
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time_sec: float
    samples_per_sec: float
    gpu_memory_mb: Optional[float] = None    # None when running on CPU
    cpu_percent: Optional[float] = None


@dataclass
class BenchmarkReport:
    device: str
    total_epochs: int
    total_time_sec: float
    avg_epoch_time_sec: float
    avg_samples_per_sec: float
    best_val_acc: float
    peak_gpu_memory_mb: Optional[float]
    speedup_vs_cpu: Optional[float] = None   # Filled in after both runs
    epoch_metrics: list = field(default_factory=list)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        metrics = [EpochMetrics(**m) for m in data.pop("epoch_metrics")]
        obj = cls(**data)
        obj.epoch_metrics = metrics
        return obj


class Profiler:
    """
    Wraps a training epoch and records hardware metrics.
    Usage:
        profiler = Profiler(device)
        with profiler.measure_epoch(epoch_num):
            ... training loop ...
        metrics = profiler.last_metrics
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.device_str = str(device)
        self.last_metrics: Optional[EpochMetrics] = None
        self._start_time = None

    def start(self):
        # Sync GPU before timing (critical for accurate CUDA measurement)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(
        self,
        epoch: int,
        n_samples: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> EpochMetrics:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._start_time
        throughput = n_samples / elapsed

        gpu_mem = None
        if self.device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            torch.cuda.reset_peak_memory_stats(self.device)

        cpu_pct = psutil.cpu_percent(interval=None)

        self.last_metrics = EpochMetrics(
            epoch=epoch,
            device=self.device_str,
            train_loss=round(train_loss, 4),
            train_acc=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 4),
            epoch_time_sec=round(elapsed, 3),
            samples_per_sec=round(throughput, 1),
            gpu_memory_mb=round(gpu_mem, 1) if gpu_mem else None,
            cpu_percent=round(cpu_pct, 1),
        )
        return self.last_metrics


def compute_speedup(cpu_report: BenchmarkReport, gpu_report: BenchmarkReport) -> dict:
    """Computes GPU speedup statistics vs CPU baseline."""
    speedup = cpu_report.avg_epoch_time_sec / gpu_report.avg_epoch_time_sec
    throughput_gain = gpu_report.avg_samples_per_sec / cpu_report.avg_samples_per_sec
    return {
        "time_speedup_x": round(speedup, 2),
        "throughput_gain_x": round(throughput_gain, 2),
        "cpu_avg_epoch_sec": cpu_report.avg_epoch_time_sec,
        "gpu_avg_epoch_sec": gpu_report.avg_epoch_time_sec,
        "cpu_best_acc": cpu_report.best_val_acc,
        "gpu_best_acc": gpu_report.best_val_acc,
        "peak_gpu_memory_mb": gpu_report.peak_gpu_memory_mb,
    }
