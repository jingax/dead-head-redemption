"""Greedy head-ordering algorithms."""

from __future__ import annotations

import torch


def greedy_chain_ordering(similarity: torch.Tensor) -> list[int]:
    """Find the best greedy head chain over all possible starting heads.

    For each starting head, repeatedly append the unused head with highest cosine
    similarity to the current head. The ordering with the largest sum of adjacent
    pairwise similarities is returned.
    """
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix.")

    num_heads = similarity.shape[0]
    if num_heads == 0:
        return []

    best_ordering: list[int] = []
    best_score = float("-inf")

    for start_head in range(num_heads):
        ordering = [start_head]
        unused = set(range(num_heads)) - {start_head}

        while unused:
            current_head = ordering[-1]
            next_head = max(unused, key=lambda candidate: similarity[current_head, candidate].item())
            ordering.append(next_head)
            unused.remove(next_head)

        score = sum(
            similarity[ordering[index], ordering[index + 1]].item() for index in range(num_heads - 1)
        )
        if score > best_score:
            best_score = score
            best_ordering = ordering

    return best_ordering
