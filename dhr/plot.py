"""Plot per-layer attention-head similarity heatmaps."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from timm.layers import set_fused_attn
from timm.models.vision_transformer import VisionTransformer

from dhr.collect import DEFAULT_BATCH_SIZE, DEFAULT_SEED, collect_head_vectors
from dhr.model import validate_target_layers
from dhr.similarity import cosine_similarity_matrix


def _subplot_grid(num_layers: int) -> tuple[int, int]:
    cols = min(4, num_layers)
    rows = math.ceil(num_layers / cols)
    return rows, cols


def _render_similarity_figure(
    similarity_matrices: dict[int, np.ndarray],
    num_heads: int,
    output_path: Path,
    title: str,
) -> None:
    layer_indices = sorted(similarity_matrices)
    num_layers = len(layer_indices)
    rows, cols = _subplot_grid(num_layers)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    last_im = None
    for subplot_idx, layer_idx in enumerate(layer_indices):
        ax = axes[subplot_idx]
        similarity = similarity_matrices[layer_idx]
        last_im = ax.imshow(similarity, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_heads))
        ax.set_xlabel("Head")
        ax.set_ylabel("Head")

    for ax in axes[num_layers:]:
        ax.axis("off")

    cbar = fig.colorbar(last_im, ax=axes[:num_layers].tolist(), shrink=0.95, pad=0.02)
    cbar.set_label("Cosine similarity")
    fig.suptitle(title, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_similarity(
    model: nn.Module,
    val_dir: str | Path,
    output: str | Path = "head_similarity.png",
    imgcount: int = 1000,
    *,
    target_layers: list[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: torch.device | str | None = None,
) -> Path:
    """Plot per-layer head similarity heatmaps from validation images.

    Samples validation images, accumulates post-softmax attention maps per head,
    computes cosine similarity, and saves one figure with a subplot per layer.

    Args:
        model: timm Vision Transformer model.
        val_dir: ImageNet-style validation directory for ImageFolder loading.
        output: Path where the similarity figure is saved.
        imgcount: Number of randomly sampled validation images to use.
        target_layers: Transformer block indices to include. Defaults to all layers.
        batch_size: Inference batch size.
        seed: Random seed for deterministic image sampling.
        device: Torch device. Defaults to CUDA when available.

    Returns:
        Path to the saved plot.
    """
    model, normalized_layers, num_heads = validate_target_layers(model, target_layers)
    output_path = Path(output)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    set_fused_attn(False)
    model = model.to(device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head_vectors = collect_head_vectors(
        model=model,
        target_layers=normalized_layers,
        val_dir=val_dir,
        imgcount=imgcount,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )

    similarity_matrices = {
        layer_idx: cosine_similarity_matrix(head_vectors[layer_idx]).cpu().numpy()
        for layer_idx in normalized_layers
    }

    model_name = getattr(model, "default_cfg", {}).get("architecture", model.__class__.__name__)
    title = f"Attention head similarity ({model_name}, n={imgcount})"
    _render_similarity_figure(similarity_matrices, num_heads, output_path, title)
    return output_path
