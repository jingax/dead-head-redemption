# dead-head-redemption

Tools for analyzing and merging redundant attention heads in Vision Transformers.

## Dynamic Head Reordering (`dhr`)

The `dhr` package reorders attention heads in timm Vision Transformers so that similar heads become physically adjacent before a later head-merging step.

### Install

```bash
pip install -e .
```

### Usage

```python
import timm
import dhr

model = timm.create_model("vit_base_patch16_224", pretrained=True)
model = dhr.reorder(
    model,
    target_layers=[3, 6, 9],
    new_heads=6,
    val_dir="/path/to/imagenet/val",
    imgcount=1000,
)
```

`dhr.reorder` will:

1. Sample validation images with `torchvision.datasets.ImageFolder`
2. Capture post-softmax attention probabilities from the selected layers
3. Build per-head feature vectors across the dataset
4. Compute cosine similarity and run a greedy chain-ordering search
5. Physically permute attention parameters in-place

The reordered model produces identical outputs to the original model (only the internal head order changes).

### Head similarity heatmaps

`head_similarity_plot.py` loads a pretrained `vit_base_patch16_224` model from [timm](https://github.com/huggingface/pytorch-image-models), runs it on an image, and saves a single figure with 12 subplots (one per transformer layer). Each subplot is a 12×12 heatmap of pairwise cosine similarity between attention heads in that layer.

```bash
python head_similarity_plot.py path/to/image.jpg -o head_similarity.png
```

### Development

```bash
pip install -e ".[dev]"
pytest
```
