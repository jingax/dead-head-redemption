"""Tests for the dhr package."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import timm
import torch
from PIL import Image
from timm.layers import set_fused_attn
from torchvision.datasets import FakeData
from torchvision.transforms import functional as TF

from dhr.ordering import greedy_chain_ordering
from dhr.permute import permute_attention_heads
from dhr.plot import plot_similarity
from dhr.reorder import reorder
from dhr.similarity import cosine_similarity_matrix


def _make_fake_imagenet(root: Path, count: int = 20) -> Path:
    for class_idx in range(2):
        class_dir = root / f"class_{class_idx}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_idx in range(count // 2):
            image = Image.new("RGB", (64, 64), color=(class_idx * 40, image_idx * 10, 100))
            image.save(class_dir / f"{image_idx}.jpg")
    return root


def test_greedy_chain_ordering_prefers_similar_neighbors() -> None:
    similarity = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.95],
            [0.1, 0.1, 0.95, 1.0],
        ]
    )
    ordering = greedy_chain_ordering(similarity)
    assert ordering in ([0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0])


def test_permutation_preserves_outputs() -> None:
    set_fused_attn(False)
    torch.manual_seed(0)
    model = timm.create_model("vit_small_patch16_224", pretrained=False)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        baseline = model(images)

    ordering = list(range(model.blocks[0].attn.num_heads - 1, -1, -1))
    permute_attention_heads(model.blocks[0].attn, ordering)

    with torch.no_grad():
        reordered = model(images)

    torch.testing.assert_close(baseline, reordered, rtol=1e-5, atol=1e-5)


def test_reorder_preserves_model_outputs(tmp_path: Path) -> None:
    set_fused_attn(False)
    torch.manual_seed(1)

    val_dir = _make_fake_imagenet(tmp_path / "val_outputs", count=12)
    model = timm.create_model("vit_small_patch16_224", pretrained=False)
    image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        baseline = model(image).clone()

    reordered = reorder(
        model,
        target_layers=[0],
        new_heads=3,
        val_dir=val_dir,
        imgcount=8,
        batch_size=4,
        seed=1,
        device="cpu",
    )

    with torch.no_grad():
        after = reordered(image)

    torch.testing.assert_close(baseline, after, rtol=1e-5, atol=1e-5)


def test_reorder_runs_on_fake_validation_dir(tmp_path: Path) -> None:
    set_fused_attn(False)
    random.seed(0)
    torch.manual_seed(0)

    val_dir = _make_fake_imagenet(tmp_path / "val", count=16)
    model = timm.create_model("vit_small_patch16_224", pretrained=False)

    reordered = reorder(
        model,
        target_layers=[0, 1],
        new_heads=3,
        val_dir=val_dir,
        imgcount=8,
        batch_size=4,
        seed=0,
        device="cpu",
    )

    assert reordered is model
    image = TF.to_tensor(Image.new("RGB", (224, 224), color=(120, 80, 40))).unsqueeze(0)
    with torch.no_grad():
        output = reordered(image)
    assert output.shape[0] == 1


def test_plot_similarity_writes_figure(tmp_path: Path) -> None:
    set_fused_attn(False)
    torch.manual_seed(2)

    val_dir = _make_fake_imagenet(tmp_path / "val_plot", count=12)
    model = timm.create_model("vit_small_patch16_224", pretrained=False)
    output = tmp_path / "similarity.png"

    saved_path = plot_similarity(
        model,
        val_dir=val_dir,
        output=output,
        imgcount=8,
        batch_size=4,
        seed=2,
        device="cpu",
    )

    assert saved_path == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_cosine_similarity_matrix_is_symmetric() -> None:
    vectors = torch.randn(6, 128)
    similarity = cosine_similarity_matrix(vectors)
    assert similarity.shape == (6, 6)
    torch.testing.assert_close(similarity, similarity.T)
