# road-sign-classifier

## What we're building
A road sign image classifier trained on GTSRB (German Traffic Sign Recognition
Benchmark, 43 classes), using transfer learning on ResNet-18. Training
augmentation (rotation, brightness jitter, Gaussian blur) is chosen to
approximate real dashcam viewing conditions rather than studio-quality signage
photos.

## Tech stack
- Python, PyTorch + torchvision (`ResNet18_Weights.DEFAULT` as the pretrained backbone)
- `torchvision.datasets.GTSRB` for data loading, with `download=True` so a fresh
  environment (e.g. Colab) fetches the dataset automatically
- scikit-learn for evaluation metrics (per-class F1, confusion matrix)
- matplotlib for the confusion matrix plot
- Jupyter / Google Colab notebook for GPU training
- GitHub: https://github.com/rekhanr1/road-sign-classifier (public, `master` branch)

## Structure
- `src/config.py` — paths, hyperparameters (80/20 val split, batch size, etc.), CUDA/CPU device detection
- `src/labels.py` — GTSRB class id → sign name mapping
- `src/download_data.py` — standalone script to download the GTSRB train/test splits
- `src/dataset.py` — dataloaders, dashcam-style train-time augmentation, 80/20 train/val split
- `src/model.py` — ResNet-18 builder with freeze/unfreeze of the pretrained backbone
- `src/utils.py` — seeding, checkpoint save/load, accuracy helper
- `src/train.py` — training loop: frozen-backbone warmup → fine-tune, early stopping on val accuracy, saves best checkpoint to `checkpoints/best_model.pth`, logs `training_history.csv`, `classification_report.txt`, `per_class_f1.csv`, `confusion_matrix.png` to `outputs/`
- `notebooks/train_on_colab.ipynb` — clones the repo, installs `requirements.txt`, downloads GTSRB, runs `src.train` on a Colab GPU, downloads the resulting weights
- `requirements.txt`, `.gitignore`

## Phase plan
1. Project scaffold (dirs, `requirements.txt`, `.gitignore`) — **done**
2. Data pipeline (GTSRB auto-download, transforms, dashcam-style augmentation, 80/20 split) — **done**
3. Model + training loop (ResNet-18 transfer learning, freeze-then-fine-tune, early stopping, checkpointing) — **done**
4. Evaluation logging (accuracy, per-class F1, confusion matrix to `outputs/`) — **done**
5. Colab runnability (GPU detection, auto-download, `train_on_colab.ipynb`) — **done**
6. Version control (git init, GitHub repo, push) — **done**
7. Inference script (`src/infer.py` for single-image/folder prediction) — **not started** (a draft was written once but the user rejected that tool call; the file doesn't exist in the repo)
8. An actual training run — **not started**: no data has been downloaded locally yet, `checkpoints/` and `data/` are empty, and no real accuracy/F1 numbers exist
9. Test-set evaluation script (currently `get_dataloaders` returns a `test_loader` that nothing consumes) — **not started**
10. Optional/future: model export (ONNX/TorchScript), a simple demo UI — **not requested yet**

## Done vs. remaining
**Done:** repo scaffold, data pipeline with dashcam-realistic augmentation,
training script with early stopping and full metrics logging, Colab training
notebook, git/GitHub setup with the first two commits pushed.

**Remaining:** write `src/infer.py`; run actual training (locally or via the
Colab notebook) to produce `checkpoints/best_model.pth` and real metrics;
add a script to evaluate on the held-out GTSRB test split.
