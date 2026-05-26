# L2P — Learning to Prompt for Continual Learning

PyTorch implementation of **L2P** from *"Learning to Prompt for Continual Learning"* (Wang et al., CVPR 2022).

L2P uses a frozen pretrained Vision Transformer (ViT) as a feature extractor and learns a small set of **prompt tokens** selected dynamically per input. This enables strong continual learning performance without a replay buffer.

## Overview

![L2P architecture](https://img.shields.io/badge/paper-CVPR%202022-blue)

- **Frozen ViT backbone** — no backbone gradients during continual training.
- **Learnable prompt pool** — `M = 20` groups of `Lp = 5` learnable prompt tokens.
- **Query-key retrieval** — cosine similarity between image features and prompt keys selects top-`k` prompts per input.
- **Per-task classifiers** — one lightweight linear head per task.
- **No replay buffer** — all knowledge retained in the prompt pool.

## Requirements

```
torch >= 2.0
torchvision
timm
numpy
tqdm
```

Install:

```bash
pip install torch torchvision timm numpy tqdm
```

## Quick Start

### Train on Split CIFAR-100

```bash
python main.py
```

This runs 10 sequential tasks (10 classes each). After each task it evaluates on all seen tasks and reports average accuracy + forgetting.

### Override defaults via CLI

```bash
python main.py --epochs 20 --batch-size 128 --model vit_base_patch16_224
```

### Smoke test

```bash
python test.py
```

Runs one epoch on random data to verify the pipeline is intact.

## Project Structure

```
pytorch/
├── config.py      # All hyperparameters in one place
├── model.py       # PromptPool + L2P_ViT model definition
├── dataset.py     # Split CIFAR-100 dataloader
├── train.py       # Training and evaluation loops
├── main.py        # Entry point — continual learning loop
├── utils.py       # Metrics, seeding, parameter helpers
└── test.py        # Smoke test on random data
```

## Configuration

Key defaults in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vit_model_name` | `vit_small_patch16_224` | ViT backbone (timm) |
| `pretrained` | `True` | ImageNet-21k pretrained weights |
| `freeze_backbone` | `True` | Backbone frozen during continual training |
| `pool_size` | `20` | Number of prompt groups in the pool |
| `prompt_length` | `5` | Tokens per prompt group |
| `top_k` | `5` | Prompts selected per input |
| `num_tasks` | `10` | Total continual learning tasks |
| `num_classes_per_task` | `10` | Classes per task (Split CIFAR-100) |
| `num_epochs_per_task` | `10` | Epochs per incremental task |
| `lr` | `1e-3` | Learning rate (AdamW) |
| `prompt_reg_weight` | `0.1` | Diversity regularization on selected keys |
| `batch_size` | `64` | Batch size |
| `amp` | auto | Automatic mixed precision on CUDA |

## Metrics

- **Average Accuracy** — mean test accuracy across all tasks seen so far.
- **Average Forgetting** — mean drop from each old task's historical peak accuracy.

Reported per-task and as final aggregates after the last task.

## Regularization

The diversity loss penalizes cosine similarity among co-selected prompt keys:

$$\mathcal{L}_{\text{reg}} = \frac{1}{B} \sum_b \big\| (K_b K_b^T) \odot (1 - I) \big\|_F^2$$

where $K_b$ holds the normalized prompt keys selected for sample $b$. This encourages the model to pick *diverse* prompts rather than collapsing to a few over-used ones.

## Reference

```bibtex
@inproceedings{wang2022learning,
  title     = {Learning to Prompt for Continual Learning},
  author    = {Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and
               Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and
               Dy, Jennifer and Pfister, Tomas},
  booktitle = {CVPR},
  year      = {2022}
}
```

## License

MIT
