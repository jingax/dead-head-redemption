"""Collect per-head attention feature vectors from validation images."""

from __future__ import annotations

from pathlib import Path

import torch
from timm.models.vision_transformer import VisionTransformer
from tqdm import tqdm

from dhr.attention import accumulate_head_vectors, capture_attention_maps
from dhr.data import build_validation_loader
from dhr.model import get_attention_module

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32


def collect_head_vectors(
    model: VisionTransformer,
    target_layers: list[int],
    val_dir: str | Path,
    imgcount: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Run validation images and accumulate flattened attention vectors per head."""
    loader = build_validation_loader(
        model=model,
        val_dir=val_dir,
        imgcount=imgcount,
        batch_size=batch_size,
        seed=seed,
    )
    attention_modules = {layer_idx: get_attention_module(model, layer_idx) for layer_idx in target_layers}
    accumulators: dict[int, torch.Tensor] = {layer_idx: torch.empty(0) for layer_idx in target_layers}

    model.eval()
    with torch.no_grad(), capture_attention_maps(attention_modules) as storage:
        progress = tqdm(loader, desc="Collecting attention", unit="batch")
        for images, _ in progress:
            images = images.to(device, non_blocking=True)
            model(images)
            accumulate_head_vectors(storage, accumulators)

    return accumulators
