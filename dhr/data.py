"""Dataset and dataloader helpers for DHR."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


def build_validation_loader(
    model: torch.nn.Module,
    val_dir: str | Path,
    imgcount: int,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Build a deterministic ImageFolder loader over a random validation subset."""
    val_path = Path(val_dir)
    if not val_path.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_path}")

    data_config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_config, is_training=False)
    dataset = ImageFolder(str(val_path), transform=transform)

    if len(dataset) == 0:
        raise ValueError(f"No images found in validation directory: {val_path}")

    sample_count = min(imgcount, len(dataset))
    generator = random.Random(seed)
    indices = generator.sample(range(len(dataset)), sample_count)

    subset = Subset(dataset, indices)
    pin_memory = torch.cuda.is_available()
    workers = 0 if sample_count < batch_size * 2 else (4 if torch.cuda.is_available() else 0)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
