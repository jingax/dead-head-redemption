"""Public reorder API for Dynamic Head Reordering."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from timm.layers import set_fused_attn
from timm.models.vision_transformer import VisionTransformer
from tqdm import tqdm

from dhr.attention import accumulate_head_vectors, capture_attention_maps
from dhr.data import build_validation_loader
from dhr.model import get_attention_module, validate_reorder_inputs
from dhr.ordering import greedy_chain_ordering
from dhr.permute import permute_attention_heads
from dhr.similarity import cosine_similarity_matrix

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 32


def _collect_head_vectors(
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


def reorder(
    model: nn.Module,
    target_layers: list[int],
    new_heads: int,
    val_dir: str | Path,
    imgcount: int = 1000,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: torch.device | str | None = None,
) -> VisionTransformer:
    """Reorder attention heads in selected ViT layers by attention-map similarity.

    Similar heads are placed adjacent to one another to simplify later head-merging
    steps such as 12→6 or 12→4. This function only reorders heads; it does not merge
    them.

    Args:
        model: timm Vision Transformer model.
        target_layers: Transformer block indices to reorder.
        new_heads: Target head count after a future merge step. Used to validate that
            the current head count is divisible by this value.
        val_dir: ImageNet-style validation directory for ImageFolder loading.
        imgcount: Number of randomly sampled validation images to use.
        batch_size: Inference batch size.
        seed: Random seed for deterministic image sampling.
        device: Torch device. Defaults to CUDA when available.

    Returns:
        The same model instance with physically reordered attention heads.
    """
    model, normalized_layers, _ = validate_reorder_inputs(model, target_layers, new_heads)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    set_fused_attn(False)
    model = model.to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head_vectors = _collect_head_vectors(
        model=model,
        target_layers=normalized_layers,
        val_dir=val_dir,
        imgcount=imgcount,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    for layer_idx in normalized_layers:
        similarity = cosine_similarity_matrix(head_vectors[layer_idx])
        ordering = greedy_chain_ordering(similarity)
        permute_attention_heads(get_attention_module(model, layer_idx), ordering)

    return model
