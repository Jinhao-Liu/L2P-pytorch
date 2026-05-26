"""Minimal smoke test — verifies the training loop runs end-to-end on random data."""

import torch
from torch import nn

from config import cfg
from model import L2P_ViT
from train import train_one_epoch
from utils import configure_torch_runtime, set_seed, trainable_parameters_for_task


class RandomImageDataset(torch.utils.data.Dataset):
    """Synthetic dataset of random images for quick pipeline validation."""

    def __init__(self, size: int = 8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.randn(3, cfg.image_size, cfg.image_size)
        label = torch.randint(0, cfg.num_classes_per_task, ()).item()
        return image, label


def main():
    set_seed()
    configure_torch_runtime()

    original_batch_size = cfg.batch_size
    cfg.batch_size = 2

    loader = torch.utils.data.DataLoader(RandomImageDataset(), batch_size=cfg.batch_size)
    model = L2P_ViT(pretrained=False).to(cfg.device)
    model.set_current_task(0)
    optimizer = torch.optim.AdamW(trainable_parameters_for_task(model, 0), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device == "cuda")

    metrics = train_one_epoch(model, loader, optimizer, criterion, scaler)
    cfg.batch_size = original_batch_size
    print("Smoke test passed:", metrics)


if __name__ == "__main__":
    main()
