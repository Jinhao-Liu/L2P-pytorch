"""Training entry point: runs L2P on Split CIFAR-100 sequentially."""

import argparse

import numpy as np
import torch
from torch import nn

from config import cfg
from dataset import get_split_cifar100
from model import L2P_ViT
from train import evaluate, train_one_epoch
from utils import (
    calculate_avg_acc,
    calculate_avg_forgetting,
    check_single_gpu_env,
    configure_torch_runtime,
    set_seed,
    trainable_parameters_for_task,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch L2P on Split CIFAR-100")
    parser.add_argument("--epochs", type=int, default=cfg.num_epochs_per_task)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--workers", type=int, default=cfg.num_workers)
    parser.add_argument("--model", type=str, default=cfg.vit_model_name)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def apply_args(args):
    cfg.num_epochs_per_task = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.workers
    cfg.persistent_workers = args.workers > 0
    cfg.vit_model_name = args.model
    cfg.pretrained = not args.no_pretrained
    cfg.compile_model = args.compile


def main():
    args = parse_args()
    apply_args(args)
    set_seed()
    configure_torch_runtime()

    print("=" * 60)
    print(f"{cfg.project_name} start")
    print(f"Device: {cfg.device} | model: {cfg.vit_model_name} | batch size: {cfg.batch_size}")
    env_ok = check_single_gpu_env()
    if not env_ok:
        confirm = input("Continue on CPU? (y/n): ")
        if confirm.lower() != "y":
            print("Exit.")
            return

    print("Loading Split CIFAR-100...")
    train_loaders, test_loaders, task_class_mapping = get_split_cifar100()
    print(f"Dataset ready. Tasks: {len(task_class_mapping)}")

    print("Building L2P model...")
    model = L2P_ViT(vit_model_name=cfg.vit_model_name, pretrained=cfg.pretrained).to(cfg.device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device == "cuda")

    print(
        f"Backbone frozen: {cfg.freeze_backbone} | prompt pool: {cfg.pool_size} | "
        f"prompt length: {cfg.prompt_length} | top-k: {cfg.top_k}"
    )
    print("=" * 60)

    # acc_matrix[i, j] = accuracy on task j after training up to task i.
    acc_matrix = np.zeros((cfg.num_tasks, cfg.num_tasks), dtype=np.float32)

    for task_id in range(cfg.num_tasks):
        print("\n" + "=" * 60)
        print(f"Start task {task_id}")
        unwrap_model(model).set_current_task(task_id)

        # Each task rebuilds the optimizer: only prompt params + current head.
        optimizer = torch.optim.AdamW(
            trainable_parameters_for_task(model, task_id),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        for epoch in range(cfg.num_epochs_per_task):
            print(f"\nTask {task_id} | Epoch {epoch + 1}/{cfg.num_epochs_per_task}")
            train_one_epoch(model, train_loaders[task_id], optimizer, criterion, scaler)

        print("\nEvaluating seen tasks...")
        for eval_task_id in range(task_id + 1):
            test_acc = evaluate(model, test_loaders[eval_task_id], eval_task_id)
            acc_matrix[task_id, eval_task_id] = test_acc

        avg_acc = calculate_avg_acc(acc_matrix, task_id)
        avg_forgetting = calculate_avg_forgetting(acc_matrix, task_id)
        print(f"Task {task_id} done | avg acc: {avg_acc:.4f} | avg forgetting: {avg_forgetting:.4f}")

    print("\n" + "=" * 60)
    print("Final results")
    final_avg_acc = calculate_avg_acc(acc_matrix, cfg.num_tasks - 1)
    final_avg_forgetting = calculate_avg_forgetting(acc_matrix, cfg.num_tasks - 1)
    print(f"Final avg acc: {final_avg_acc:.4f}")
    print(f"Final avg forgetting: {final_avg_forgetting:.4f}")
    print("Accuracy matrix:")
    print(np.round(acc_matrix, 4))


if __name__ == "__main__":
    main()
