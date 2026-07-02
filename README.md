# dead-head-redemption

Tools for analyzing and merging redundant attention heads in Vision Transformers.

## Head similarity heatmaps

`head_similarity_plot.py` loads a pretrained `vit_base_patch16_224` model from [timm](https://github.com/huggingface/pytorch-image-models), runs it on an image, and saves a single figure with 12 subplots (one per transformer layer). Each subplot is a 12×12 heatmap of pairwise cosine similarity between attention heads in that layer.

### Setup

```bash
pip install -r requirements.txt
```

### Usage

```bash
python head_similarity_plot.py path/to/image.jpg -o head_similarity.png
```

Optional flags:

- `--device cuda` or `--device cpu` (defaults to CUDA when available)
- `--output` / `-o` path for the saved plot
