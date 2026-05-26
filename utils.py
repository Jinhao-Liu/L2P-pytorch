"""Utility functions: seeding, runtime config, metrics, and parameter helpers."""

import random

import numpy as np
import torch

from config import cfg


def set_seed(seed: int = cfg.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime():
    if cfg.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(cfg.matmul_precision)


def calculate_avg_acc(acc_matrix, current_task_id):
    """Average accuracy over all tasks seen so far."""
    return acc_matrix[current_task_id, :current_task_id + 1].mean()


def calculate_avg_forgetting(acc_matrix, current_task_id):
    """Average forgetting: mean drop from each old task's peak accuracy."""
    if current_task_id == 0:
        return 0.0
    forgetting_list = []
    for prev_task_id in range(current_task_id):
        max_acc = acc_matrix[:current_task_id, prev_task_id].max()
        current_acc = acc_matrix[current_task_id, prev_task_id]
        forgetting_list.append(max_acc - current_acc)
    return float(np.mean(forgetting_list))


def check_single_gpu_env():
    """Print CUDA environment info. Returns True if GPU is available."""
    print("=" * 40)
    print("PyTorch single-GPU environment")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA was not detected. Training can run on CPU, but it will be very slow.")
        return False

    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current_device)
    print(f"GPU count: {device_count}")
    print(f"Using GPU: {props.name}")
    print(f"GPU memory: {props.total_memory / 1024 ** 3:.2f} GB")
    print(f"AMP: {cfg.amp} | channels_last: {cfg.channels_last} | TF32: enabled")
    print("=" * 40)
    return True


def unwrap_model(model):
    """Return the raw model, unwrapping ``torch.compile`` if active."""
    return getattr(model, "_orig_mod", model)


def trainable_parameters_for_task(model, task_id: int):
    """Collect parameters to optimize for the given task.

    Returns prompt pool + prompt_pos_embed + the task-specific classifier head.
    Old-task heads are excluded.
    """
    model = unwrap_model(model)
    params = []
    params.extend(model.prompt_pool.parameters())
    params.append(model.prompt_pos_embed)
    params.extend(model.class_heads[task_id].parameters())
    return params
