"""Train a ResNet-18 transfer-learning classifier on GTSRB.

80/20 train/val split, early stopping on validation accuracy, and the best
checkpoint's accuracy / per-class F1 / confusion matrix logged to outputs/.

Usage:
    python -m src.train --epochs 30 --patience 5
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, DEVICE, NUM_CLASSES, OUTPUT_DIR, SEED
from src.dataset import get_dataloaders
from src.labels import CLASS_NAMES
from src.model import build_model, unfreeze_backbone
from src.utils import accuracy, load_checkpoint, save_checkpoint, set_seed


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        outputs = model(images.to(DEVICE))
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_classification_report(y_true, y_pred, out_path):
    target_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=target_names, zero_division=0
    )
    out_path.write_text(report)


def save_per_class_f1(y_true, y_pred, out_path):
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "f1"])
        for class_id, f1 in enumerate(per_class_f1):
            writer.writerow([class_id, CLASS_NAMES[class_id], f"{f1:.4f}"])
    return per_class_f1


def train(epochs, freeze_epochs, lr, batch_size, patience, checkpoint_dir, output_dir):
    set_seed(SEED)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    model = build_model(NUM_CLASSES, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    checkpoint_path = checkpoint_dir / "best_model.pth"
    history = []
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1:
            print("Unfreezing backbone for fine-tuning...")
            unfreeze_backbone(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_path, epoch=epoch, val_acc=val_acc)
            print(f"  Saved new best checkpoint (val_acc={val_acc:.4f}) -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no val_acc improvement in {patience} epochs.")
                break

    with (output_dir / "training_history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(history)

    # Final metrics come from the best checkpoint, not whatever the loop ended on
    load_checkpoint(model, checkpoint_path, device=DEVICE)
    y_true, y_pred = collect_predictions(model, val_loader)

    final_acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    save_classification_report(y_true, y_pred, output_dir / "classification_report.txt")
    save_per_class_f1(y_true, y_pred, output_dir / "per_class_f1.csv")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    print(f"Best val_acc={best_val_acc:.4f} | final_acc={final_acc:.4f} | macro_f1={macro_f1:.4f}")
    print(f"Metrics written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GTSRB classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--freeze-epochs", type=int, default=3, help="Epochs to train with a frozen backbone before fine-tuning"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (epochs without val_acc improvement)"
    )
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
    )
