"""Global configuration for L2P training."""

import torch


class Config:
    """L2P training configuration.

    Class-level attributes for lightweight override: ``cfg.batch_size = 128``.
    """

    seed = 42
    project_name = "L2P_PyTorch_SingleGPU"

    # --- Data ---
    data_root = "./data"
    num_tasks = 10
    num_classes_per_task = 10
    image_size = 224
    batch_size = 64
    num_workers = 2

    # --- Model ---
    vit_model_name = "vit_small_patch16_224"
    pretrained = True
    freeze_backbone = True
    pool_size = 20
    prompt_length = 5
    top_k = 5
    key_dim = None
    batchwise_selection = False

    # --- Training ---
    num_epochs_per_task = 10
    lr = 1e-3
    weight_decay = 1e-4
    prompt_reg_weight = 0.1
    loss_type = "cross_entropy"

    # --- Runtime ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = torch.cuda.is_available()
    prefetch_factor = 2
    persistent_workers = num_workers > 0

    amp = torch.cuda.is_available()
    amp_dtype = torch.float16
    channels_last = torch.cuda.is_available()
    cudnn_benchmark = True
    matmul_precision = "high"
    compile_model = False


cfg = Config()
