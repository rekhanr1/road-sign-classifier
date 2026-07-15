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
- `src/evaluate.py` — standalone evaluation of a trained checkpoint (default `src/best_model.pth`) on the held-out GTSRB **test** split; logs `classification_report.txt`, `per_class_f1.csv`, `confusion_matrix.png`, `metrics_summary.txt` to `outputs/`
- `src/perturbations.py` — `GlareOverlay` / `MotionBlur` / `LowLight` PIL transforms (severity 0-1) simulating dashcam degradations, used by the failure-analysis notebook
- `notebooks/train_on_colab.ipynb` — clones the repo, installs `requirements.txt`, downloads GTSRB, runs `src.train` on a Colab GPU, downloads the resulting weights
- `notebooks/evaluate_failures.ipynb` — loads `src/best_model.pth` and analyzes failure cases: ranks confused class pairs on the clean test set with example images, then sweeps glare/motion-blur/low-light severity and plots accuracy degradation; writes `confusion_pairs.csv`, `top_confused_pairs.png`, `top_confusion_examples.png`, `degradation_results.csv`, `degradation_plot.png`, `perturbation_examples.png` to `outputs/`
- `requirements.txt`, `.gitignore`

## Phase plan
1. Project scaffold (dirs, `requirements.txt`, `.gitignore`) — **done**
2. Data pipeline (GTSRB auto-download, transforms, dashcam-style augmentation, 80/20 split) — **done**
3. Model + training loop (ResNet-18 transfer learning, freeze-then-fine-tune, early stopping, checkpointing) — **done**
4. Evaluation logging (accuracy, per-class F1, confusion matrix to `outputs/`) — **done**
5. Colab runnability (GPU detection, auto-download, `train_on_colab.ipynb`) — **done**
6. Version control (git init, GitHub repo, push) — **done**
7. Inference script (`src/infer.py` for single-image/folder prediction) — **not started** (a draft was written once but the user rejected that tool call; the file doesn't exist in the repo)
8. An actual training run — **done**: trained on Colab, weights downloaded to `src/best_model.pth`. `checkpoints/` and `data/` remain empty locally (nothing has been trained or run locally yet — torch isn't installed in any local env, base conda or `cwq`)
9. Test-set evaluation script (`src/evaluate.py`) — **done**: loads `src/best_model.pth`, evaluates on the held-out GTSRB test split, logs metrics to `outputs/`. Not yet run locally/verified end-to-end (no local torch install)
10. Failure analysis notebook (`notebooks/evaluate_failures.ipynb`) — **done**: confused-class-pair ranking + simulated glare/motion-blur/low-light degradation curves. Not yet run/verified end-to-end
11. Optional/future: model export (ONNX/TorchScript), a simple demo UI — **not requested yet**

## Done vs. remaining
**Done:** repo scaffold, data pipeline with dashcam-realistic augmentation,
training script with early stopping and full metrics logging, Colab training
notebook, git/GitHub setup, an actual Colab training run (`src/best_model.pth`),
a standalone test-set evaluation script, and a failure-analysis notebook
(confusion pairs + condition-robustness sweep via `src/perturbations.py`).

**Remaining:** write `src/infer.py`; actually run `src/evaluate.py` and
`notebooks/evaluate_failures.ipynb` (needs torch installed locally, or run on
Colab) to get real numbers/plots instead of just reviewed-but-unexecuted code;
`outputs/` is still empty.
