from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import GTSRB

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    NUM_WORKERS,
    SEED,
    VAL_SPLIT,
)

import torch


def build_transforms(train: bool):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        # Simulate real dashcam conditions: viewing angle, exposure, motion blur
        ops += [
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    val_split=VAL_SPLIT,
    num_workers=NUM_WORKERS,
):
    train_full = GTSRB(
        root=str(data_dir), split="train", transform=build_transforms(train=True), download=True
    )
    test_set = GTSRB(
        root=str(data_dir), split="test", transform=build_transforms(train=False), download=True
    )

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

    # validation images should use the eval transform, not the train-time augmentations
    val_set.dataset = GTSRB(root=str(data_dir), split="train", transform=build_transforms(train=False))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
