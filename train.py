"""Training and evaluation loops for L2P."""

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from config import cfg
from model import L2P_ViT
from utils import unwrap_model


def _amp_context():
    return torch.autocast(
        device_type="cuda", dtype=cfg.amp_dtype,
        enabled=cfg.amp and cfg.device == "cuda",
    )


def train_one_epoch(
    model: L2P_ViT,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None = None,
):
    """Train the current task for one epoch.

    Only prompt-pool parameters and the current task's classifier head are updated.
    """
    device = torch.device(cfg.device)
    model.train()
    base_model = unwrap_model(model)
    base_model.prompt_pool.train()
    base_model.class_heads[base_model.current_task_id].train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_reg_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Task {base_model.current_task_id} train", ncols=110)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with _amp_context():
            logits, selected_idx = model(images, return_selected_idx=True)
            ce_loss = criterion(logits, labels)

            # Regularization: penalize cosine similarity among co-selected prompt keys.
            # Equivalent to ||K @ K^T - I||_F^2 where K holds the normalized selected keys.
            selected_keys = base_model.prompt_pool.prompt_key[selected_idx]
            selected_keys_norm = F.normalize(selected_keys, dim=-1)
            similarity_matrix = torch.bmm(
                selected_keys_norm, selected_keys_norm.transpose(1, 2)
            )
            top_k = selected_idx.shape[1]
            mask = torch.eye(top_k, device=device, dtype=similarity_matrix.dtype).unsqueeze(0)
            similarity_matrix = similarity_matrix * (1.0 - mask)
            reg_loss = (
                torch.linalg.matrix_norm(similarity_matrix, ord="fro", dim=(1, 2))
                .pow(2)
                .mean()
            )

            loss = ce_loss + cfg.prompt_reg_weight * reg_loss

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.shape[0]
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.detach().item() * batch_size
        total_ce_loss += ce_loss.detach().item() * batch_size
        total_reg_loss += reg_loss.detach().item() * batch_size
        total_correct += correct
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{loss.detach().item():.4f}",
            "ce": f"{ce_loss.detach().item():.4f}",
            "reg": f"{reg_loss.detach().item():.4f}",
            "acc": f"{correct / batch_size:.4f}",
        })

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print(f"Epoch done | loss: {avg_loss:.4f} | train acc: {avg_acc:.4f}")
    return {
        "avg_loss": avg_loss,
        "avg_acc": avg_acc,
        "avg_ce_loss": total_ce_loss / total_samples,
        "avg_reg_loss": total_reg_loss / total_samples,
    }


@torch.inference_mode()
def evaluate(model: L2P_ViT, test_loader, task_id: int):
    """Evaluate accuracy on a specific task's test set.

    ``task_id`` selects the corresponding classifier head for old-task replay evaluation.
    """
    device = torch.device(cfg.device)
    model.eval()
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with _amp_context():
            logits = model(images, task_id=task_id)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.shape[0]

    test_acc = total_correct / total_samples
    print(f"Task {task_id} test acc: {test_acc:.4f}")
    return test_acc
