"""Utilities for validating and inspecting timm Vision Transformer models."""

from __future__ import annotations

import torch.nn as nn
from timm.layers import Attention
from timm.models.vision_transformer import VisionTransformer


def is_vit_model(model: nn.Module) -> bool:
    """Return True if the model is a timm Vision Transformer."""
    return isinstance(model, VisionTransformer)


def get_vit_block(model: VisionTransformer, layer_idx: int) -> nn.Module:
    """Return the transformer block at ``layer_idx``."""
    if layer_idx < 0 or layer_idx >= len(model.blocks):
        raise IndexError(
            f"Layer index {layer_idx} is out of range for model with {len(model.blocks)} blocks."
        )
    return model.blocks[layer_idx]


def get_attention_module(model: VisionTransformer, layer_idx: int) -> Attention:
    """Return the attention module for a transformer block."""
    block = get_vit_block(model, layer_idx)
    attn = getattr(block, "attn", None)
    if not isinstance(attn, Attention):
        raise TypeError(
            f"Layer {layer_idx} uses unsupported attention type {type(attn).__name__}. "
            "Only timm.layers.Attention is supported."
        )
    return attn


def validate_target_layers(
    model: nn.Module,
    target_layers: list[int] | None = None,
) -> tuple[VisionTransformer, list[int], int]:
    """Validate layer indices and return normalized layers plus head count."""
    if not is_vit_model(model):
        raise TypeError("dhr only supports timm VisionTransformer models.")

    depth = len(model.blocks)
    if target_layers is None:
        normalized_layers = list(range(depth))
    else:
        if not target_layers:
            raise ValueError("target_layers must contain at least one layer index.")
        normalized_layers = sorted({int(layer) for layer in target_layers})

    for layer_idx in normalized_layers:
        if layer_idx < 0 or layer_idx >= depth:
            raise IndexError(
                f"Layer index {layer_idx} is out of range for model with {depth} blocks."
            )

    num_heads = get_attention_module(model, normalized_layers[0]).num_heads
    for layer_idx in normalized_layers[1:]:
        layer_heads = get_attention_module(model, layer_idx).num_heads
        if layer_heads != num_heads:
            raise ValueError(
                f"All target layers must share the same head count; layer {layer_idx} has "
                f"{layer_heads} heads, expected {num_heads}."
            )

    return model, normalized_layers, num_heads


def validate_reorder_inputs(
    model: nn.Module,
    target_layers: list[int],
    new_heads: int,
) -> tuple[VisionTransformer, list[int], int]:
    """Validate reorder inputs and return normalized layer indices."""
    model, normalized_layers, num_heads = validate_target_layers(model, target_layers)

    if new_heads <= 0:
        raise ValueError("new_heads must be a positive integer.")
    if num_heads % new_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by new_heads ({new_heads}) "
            "for future merge compatibility."
        )

    return model, normalized_layers, num_heads
