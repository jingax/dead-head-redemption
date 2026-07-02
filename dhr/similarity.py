"""Head similarity computation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_similarity_matrix(head_vectors: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity for head feature vectors.

    Args:
        head_vectors: Tensor of shape ``(num_heads, feature_dim)``.

    Returns:
        Similarity matrix of shape ``(num_heads, num_heads)``.
    """
    if head_vectors.ndim != 2:
        raise ValueError(f"Expected head_vectors with shape (H, D), got {tuple(head_vectors.shape)}")

    normalized = F.normalize(head_vectors.float(), dim=1, eps=1e-8)
    return normalized @ normalized.T
