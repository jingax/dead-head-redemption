"""Attention capture utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
import torch.nn.functional as F
from timm.layers import Attention, maybe_add_mask, resolve_self_attn_mask
from timm.layers import set_fused_attn


AttentionStorage = dict[int, list[torch.Tensor]]


def _capture_forward(
    attn: Attention,
    storage: list[torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Build a forward function that records post-softmax attention probabilities."""

    def forward(
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        qkv = attn.qkv(x).reshape(batch_size, num_tokens, 3, attn.num_heads, attn.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = attn.q_norm(query), attn.k_norm(key)

        query = query * attn.scale
        scores = query @ key.transpose(-2, -1)
        attn_bias = resolve_self_attn_mask(num_tokens, scores, attn_mask, is_causal)
        scores = maybe_add_mask(scores, attn_bias)
        probabilities = scores.softmax(dim=-1)
        storage.append(probabilities.detach())

        probabilities = attn.attn_drop(probabilities)
        context = probabilities @ value
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, attn.attn_dim)
        context = attn.norm(context)
        context = attn.proj(context)
        return attn.proj_drop(context)

    return forward


@contextmanager
def capture_attention_maps(
    attention_modules: dict[int, Attention],
) -> Iterator[AttentionStorage]:
    """Temporarily capture post-softmax attention maps for selected layers."""
    set_fused_attn(False)
    storage: AttentionStorage = {layer_idx: [] for layer_idx in attention_modules}
    originals: dict[int, tuple[Callable[..., torch.Tensor], bool]] = {}

    try:
        for layer_idx, attn in attention_modules.items():
            originals[layer_idx] = (attn.forward, attn.fused_attn)
            object.__setattr__(attn, "fused_attn", False)
            attn.forward = _capture_forward(attn, storage[layer_idx])  # type: ignore[method-assign]

        yield storage
    finally:
        for layer_idx, attn in attention_modules.items():
            original_forward, fused_flag = originals[layer_idx]
            attn.forward = original_forward  # type: ignore[method-assign]
            object.__setattr__(attn, "fused_attn", fused_flag)
            storage[layer_idx].clear()


def accumulate_head_vectors(
    storage: AttentionStorage,
    accumulators: dict[int, torch.Tensor],
) -> None:
    """Append flattened per-head attention vectors from the latest capture batch."""
    for layer_idx, tensors in storage.items():
        if not tensors:
            continue

        attention = torch.cat(tensors, dim=0)
        batch_size, num_heads, _, _ = attention.shape
        flattened = attention.reshape(batch_size, num_heads, -1)
        flattened = flattened.permute(1, 0, 2).reshape(num_heads, -1).cpu()

        if accumulators[layer_idx].numel() == 0:
            accumulators[layer_idx] = flattened
        else:
            accumulators[layer_idx] = torch.cat([accumulators[layer_idx], flattened], dim=1)

        tensors.clear()
