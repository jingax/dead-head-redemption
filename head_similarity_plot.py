#!/usr/bin/env python3
"""Plot per-layer attention-head similarity heatmaps for a timm ViT-Base/16 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.layers import set_fused_attn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import functional as TF


MODEL_NAME = "vit_base_patch16_224"
NUM_LAYERS = 12
NUM_HEADS = 12


def load_model(device: torch.device) -> torch.nn.Module:
    set_fused_attn(False)
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval()
    return model.to(device)


def load_image(image_path: Path, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    config = resolve_data_config(model.pretrained_cfg, model=model)
    image = TF.resize(image, (config["input_size"][-2], config["input_size"][-1]))
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=config["mean"], std=config["std"])
    return tensor.unsqueeze(0).to(device)


def build_attention_extractor(model: torch.nn.Module) -> torch.nn.Module:
    return_nodes = {f"blocks.{layer_idx}.attn.softmax": f"layer_{layer_idx}" for layer_idx in range(NUM_LAYERS)}
    return create_feature_extractor(model, return_nodes=return_nodes)


@torch.no_grad()
def extract_attention_maps(model: torch.nn.Module, image_tensor: torch.Tensor) -> list[torch.Tensor]:
    extractor = build_attention_extractor(model)
    outputs = extractor(image_tensor)
    return [outputs[f"layer_{layer_idx}"][0] for layer_idx in range(NUM_LAYERS)]


def compute_head_similarity(attn: torch.Tensor) -> np.ndarray:
    """Cosine similarity between flattened attention maps for each head pair."""
    heads = attn.reshape(attn.shape[0], -1)
    heads = F.normalize(heads, dim=1, eps=1e-8)
    return (heads @ heads.T).cpu().numpy()


def plot_head_similarity(
    similarity_matrices: list[np.ndarray],
    output_path: Path,
    image_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()

    vmin, vmax = 0.0, 1.0
    last_im = None
    for layer_idx, sim in enumerate(similarity_matrices):
        ax = axes[layer_idx]
        last_im = ax.imshow(sim, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_xticks(range(NUM_HEADS))
        ax.set_yticks(range(NUM_HEADS))
        ax.set_xlabel("Head")
        ax.set_ylabel("Head")

    for ax in axes[NUM_LAYERS:]:
        ax.axis("off")

    cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.95, pad=0.02)
    cbar.set_label("Cosine similarity")
    fig.suptitle(f"Attention head similarity ({MODEL_NAME})\n{image_path.name}", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-layer attention-head similarity heatmaps for ViT-Base/16."
    )
    parser.add_argument("image_path", type=Path, help="Path to an input image.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("head_similarity.png"),
        help="Path to save the output plot (default: head_similarity.png).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    device = torch.device(args.device)
    model = load_model(device)
    image_tensor = load_image(args.image_path, model, device)
    attention_maps = extract_attention_maps(model, image_tensor)
    similarity_matrices = [compute_head_similarity(attn) for attn in attention_maps]
    plot_head_similarity(similarity_matrices, args.output, args.image_path)
    print(f"Saved head similarity plot to {args.output.resolve()}")


if __name__ == "__main__":
    main()
