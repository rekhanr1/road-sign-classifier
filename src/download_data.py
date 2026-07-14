"""Download the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

Usage:
    python -m src.download_data
"""
import argparse

from torchvision.datasets import GTSRB

from src.config import DATA_DIR


def download(data_dir=DATA_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        print(f"Downloading GTSRB '{split}' split into {data_dir} ...")
        ds = GTSRB(root=str(data_dir), split=split, download=True)
        print(f"  {split}: {len(ds)} images")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the GTSRB dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory to download/extract the dataset into",
    )
    args = parser.parse_args()

    from pathlib import Path

    download(Path(args.data_dir))
