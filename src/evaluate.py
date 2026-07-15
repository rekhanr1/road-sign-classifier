"""Evaluate a trained GTSRB checkpoint on the held-out test split.

Loads weights (default: src/best_model.pth) and logs accuracy, per-class F1,
and a confusion matrix to outputs/.

Usage:
    python -m src.evaluate
    python -m src.evaluate --weights checkpoints/best_model.pth
"""
import argparse
from pathlib import Path

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB

from src.config import BATCH_SIZE, DATA_DIR, DEVICE, NUM_WORKERS, OUTPUT_DIR, ROOT_DIR
from src.dataset import build_transforms
from src.model import build_model
from src.train import (
    collect_predictions,
    save_classification_report,
    save_confusion_matrix,
    save_per_class_f1,
)
from src.utils import load_checkpoint

DEFAULT_WEIGHTS = ROOT_DIR / "src" / "best_model.pth"


def evaluate(weights_path, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE):
    output_dir.mkdir(parents=True, exist_ok=True)

    test_set = GTSRB(
        root=str(data_dir), split="test", transform=build_transforms(train=False), download=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    model = build_model().to(DEVICE)
    checkpoint = load_checkpoint(model, weights_path, device=DEVICE)

    y_true, y_pred = collect_predictions(model, test_loader)

    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    save_classification_report(y_true, y_pred, output_dir / "classification_report.txt")
    save_per_class_f1(y_true, y_pred, output_dir / "per_class_f1.csv")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    summary_lines = [
        f"Checkpoint: {weights_path}",
    ]
    if "epoch" in checkpoint:
        summary_lines.append(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "val_acc" in checkpoint:
        summary_lines.append(f"Checkpoint val_acc: {checkpoint['val_acc']:.4f}")
    summary_lines.append(f"Test accuracy: {acc:.4f}")
    summary_lines.append(f"Test macro F1: {macro_f1:.4f}")
    (output_dir / "metrics_summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print(f"Metrics written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GTSRB checkpoint on the test split")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    evaluate(
        weights_path=Path(args.weights),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )
