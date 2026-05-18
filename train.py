"""
train.py — GPU Efficiency Profiler
===================================
Trains a CNN on CIFAR-10 on either CPU or GPU (auto-detected).
Records per-epoch timing, throughput, memory, and accuracy.
Saves a BenchmarkReport JSON for the Streamlit dashboard.

Usage:
    python train.py --device auto --epochs 10 --batch_size 128
    python train.py --device cpu  --epochs 5
    python train.py --device cuda --epochs 10 --save_model
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from models.cnn import CIFAR10_CNN, count_parameters
from utils.data_loader import get_dataloaders
from utils.profiler import Profiler, BenchmarkReport


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # Faster than zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def run_training(args):
    # ── Device selection ──────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*55}")
    print(f"  GPU Efficiency Profiler — Training Run")
    print(f"{'='*55}")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU Name    : {torch.cuda.get_device_name(device)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  VRAM Total  : {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"{'='*55}\n")

    # ── Data ──────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    n_train = len(train_loader.dataset)
    print(f"  Train samples: {n_train:,} | Val samples: {len(val_loader.dataset):,}")
    print(f"  Batches/epoch: {len(train_loader)}\n")

    # ── Model ─────────────────────────────────
    model = CIFAR10_CNN(dropout_rate=0.3).to(device)
    print(f"  Model parameters: {count_parameters(model):,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Profiler ──────────────────────────────
    profiler = Profiler(device)
    all_metrics = []
    best_val_acc = 0.0
    peak_gpu_mem = 0.0

    # ── Epoch loop ────────────────────────────
    for epoch in range(1, args.epochs + 1):
        profiler.start()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        metrics = profiler.stop(
            epoch=epoch,
            n_samples=n_train,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        all_metrics.append(metrics)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), f"{args.output_dir}/best_model_{device.type}.pth")

        if metrics.gpu_memory_mb:
            peak_gpu_mem = max(peak_gpu_mem, metrics.gpu_memory_mb)

        # Live progress
        gpu_str = f" | GPU mem: {metrics.gpu_memory_mb:.0f}MB" if metrics.gpu_memory_mb else ""
        print(
            f"  Epoch {epoch:02d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.1f}% | "
            f"Val Acc: {val_acc*100:.1f}% | "
            f"Time: {metrics.epoch_time_sec:.1f}s | "
            f"Throughput: {metrics.samples_per_sec:.0f} img/s"
            f"{gpu_str}"
        )

    # ── Save report ───────────────────────────
    total_time = sum(m.epoch_time_sec for m in all_metrics)
    avg_time = total_time / len(all_metrics)
    avg_throughput = sum(m.samples_per_sec for m in all_metrics) / len(all_metrics)

    report = BenchmarkReport(
        device=str(device),
        total_epochs=args.epochs,
        total_time_sec=round(total_time, 2),
        avg_epoch_time_sec=round(avg_time, 3),
        avg_samples_per_sec=round(avg_throughput, 1),
        best_val_acc=round(best_val_acc, 4),
        peak_gpu_memory_mb=round(peak_gpu_mem, 1) if peak_gpu_mem > 0 else None,
        epoch_metrics=[m for m in all_metrics],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = f"{args.output_dir}/report_{device.type}.json"
    report.to_json(report_path)

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
    print(f"  Avg Epoch Time    : {avg_time:.2f}s")
    print(f"  Avg Throughput    : {avg_throughput:.0f} images/sec")
    if peak_gpu_mem:
        print(f"  Peak GPU Memory   : {peak_gpu_mem:.0f} MB")
    print(f"  Report saved to   : {report_path}")
    print(f"{'='*55}\n")

    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Efficiency Profiler — Training")
    parser.add_argument("--device",     type=str,  default="auto",    help="cpu | cuda | auto")
    parser.add_argument("--epochs",     type=int,  default=10,        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,  default=128,       help="Batch size")
    parser.add_argument("--lr",         type=float,default=1e-3,      help="Initial learning rate")
    parser.add_argument("--num_workers",type=int,  default=2,         help="DataLoader workers")
    parser.add_argument("--data_dir",   type=str,  default="./data",  help="CIFAR-10 data directory")
    parser.add_argument("--output_dir", type=str,  default="./outputs",help="Where to save reports + models")
    parser.add_argument("--save_model", action="store_true",           help="Save best model checkpoint")
    args = parser.parse_args()

    run_training(args)
