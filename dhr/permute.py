"""In-place attention-head permutation for timm ViT attention modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from timm.layers import Attention


def _permute_segment(
    tensor: torch.Tensor,
    perm: Sequence[int],
    num_heads: int,
    head_dim: int,
    segment_offset: int = 0,
) -> torch.Tensor:
    """Permute contiguous head blocks inside the first dimension of a tensor."""
    reordered = tensor.clone()
    for new_head, old_head in enumerate(perm):
        old_slice = slice(segment_offset + old_head * head_dim, segment_offset + (old_head + 1) * head_dim)
        new_slice = slice(segment_offset + new_head * head_dim, segment_offset + (new_head + 1) * head_dim)
        reordered[new_slice] = tensor[old_slice]
    return reordered


def _permute_qkv(attn: Attention, perm: Sequence[int]) -> None:
    """Permute Q/K/V projection rows so physical head order matches ``perm``."""
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    attn_dim = num_heads * head_dim

    weight = attn.qkv.weight.data
    new_weight = weight.clone()
    for qkv_index in range(3):
        segment_offset = qkv_index * attn_dim
        new_weight[segment_offset : segment_offset + attn_dim] = _permute_segment(
            weight,
            perm,
            num_heads,
            head_dim,
            segment_offset=segment_offset,
        )[segment_offset : segment_offset + attn_dim]
    attn.qkv.weight.data.copy_(new_weight)

    if attn.qkv.bias is not None:
        bias = attn.qkv.bias.data
        new_bias = bias.clone()
        for qkv_index in range(3):
            segment_offset = qkv_index * attn_dim
            new_bias[segment_offset : segment_offset + attn_dim] = _permute_segment(
                bias,
                perm,
                num_heads,
                head_dim,
                segment_offset=segment_offset,
            )[segment_offset : segment_offset + attn_dim]
        attn.qkv.bias.data.copy_(new_bias)


def _permute_proj(attn: Attention, perm: Sequence[int]) -> None:
    """Permute output-projection columns to match the new head order."""
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    weight = attn.proj.weight.data
    new_weight = weight.clone()

    for new_head, old_head in enumerate(perm):
        old_slice = slice(old_head * head_dim, (old_head + 1) * head_dim)
        new_slice = slice(new_head * head_dim, (new_head + 1) * head_dim)
        new_weight[:, new_slice] = weight[:, old_slice]

    attn.proj.weight.data.copy_(new_weight)


def _permute_scale_norm(attn: Attention, perm: Sequence[int]) -> None:
    """Permute scale-normalization parameters when present."""
    norm = attn.norm
    if isinstance(norm, nn.Identity):
        return

    if not hasattr(norm, "weight") or norm.weight is None:
        return

    num_heads = attn.num_heads
    head_dim = attn.head_dim
    norm.weight.data.copy_(_permute_segment(norm.weight.data, perm, num_heads, head_dim))

    if hasattr(norm, "bias") and norm.bias is not None:
        norm.bias.data.copy_(_permute_segment(norm.bias.data, perm, num_heads, head_dim))


def permute_attention_heads(attn: Attention, ordering: Sequence[int]) -> None:
    """Physically reorder attention heads in-place.

    Args:
        attn: timm attention module to modify.
        ordering: ``ordering[new_position] = old_head_index``.
    """
    num_heads = attn.num_heads
    if len(ordering) != num_heads:
        raise ValueError(f"Expected ordering of length {num_heads}, got {len(ordering)}.")
    if sorted(ordering) != list(range(num_heads)):
        raise ValueError("ordering must be a permutation of head indices.")

    _permute_qkv(attn, ordering)
    _permute_proj(attn, ordering)
    _permute_scale_norm(attn, ordering)
