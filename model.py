"""L2P model: frozen ViT + learnable prompt pool + per-task classifiers."""

from __future__ import annotations

from typing import Optional

import timm
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from config import cfg


class PromptPool(nn.Module):
    """Learnable prompt pool with key-query retrieval.

    - ``prompt``: [pool_size, prompt_length, embed_dim] — prompt token groups.
    - ``prompt_key``: [pool_size, key_dim] — retrieval keys for cosine-similarity lookup.
    """

    def __init__(
        self,
        pool_size: int = cfg.pool_size,
        prompt_length: int = cfg.prompt_length,
        embed_dim: int = 768,
        top_k: int = cfg.top_k,
        key_dim: Optional[int] = cfg.key_dim,
        batchwise_selection: bool = cfg.batchwise_selection,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.key_dim = key_dim if key_dim is not None else embed_dim
        self.batchwise_selection = batchwise_selection

        self.prompt = nn.Parameter(torch.empty(pool_size, prompt_length, embed_dim))
        self.prompt_key = nn.Parameter(torch.empty(pool_size, self.key_dim))
        nn.init.xavier_uniform_(self.prompt)
        nn.init.xavier_uniform_(self.prompt_key)

    def forward(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """Retrieve top-k prompts via cosine similarity.

        Args:
            query: [B, key_dim] — per-image retrieval vector.

        Returns:
            selected_prompts: [B, top_k * prompt_length, embed_dim].
            selected_idx: [B, top_k] — indices into the prompt pool.
        """
        batch_size = query.shape[0]
        query_norm = F.normalize(query, dim=-1)
        key_norm = F.normalize(self.prompt_key, dim=-1)
        similarity = query_norm @ key_norm.T

        if self.batchwise_selection:
            selected_idx = similarity.mean(dim=0).topk(self.top_k, dim=-1).indices
            selected_idx = selected_idx.unsqueeze(0).expand(batch_size, -1)
        else:
            selected_idx = similarity.topk(self.top_k, dim=-1).indices

        selected_prompts = self.prompt[selected_idx].reshape(batch_size, -1, self.embed_dim)
        return selected_prompts, selected_idx


class L2P_ViT(nn.Module):
    """L2P model with a frozen ViT backbone, shared prompt pool, and per-task heads.

    During task ``t``, only prompt-related parameters and the t-th classifier head
    receive gradients.
    """

    def __init__(
        self,
        vit_model_name: str = cfg.vit_model_name,
        pretrained: bool = cfg.pretrained,
        freeze_backbone: bool = cfg.freeze_backbone,
        num_classes_per_task: int = cfg.num_classes_per_task,
        total_num_tasks: int = cfg.num_tasks,
    ):
        super().__init__()
        self.vit = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        self.freeze_backbone = freeze_backbone
        self.num_classes_per_task = num_classes_per_task
        self.total_num_tasks = total_num_tasks
        self.current_task_id = 0

        if freeze_backbone:
            self.vit.requires_grad_(False)
            self.vit.eval()

        self.prompt_pool = PromptPool(embed_dim=self.embed_dim)
        # Extra learnable position embedding for inserted prompts (since they are
        # inserted after ViT's _pos_embed and would otherwise lack position signal).
        self.prompt_pos_embed = nn.Parameter(
            torch.empty(1, cfg.top_k * cfg.prompt_length, self.embed_dim)
        )
        nn.init.xavier_uniform_(self.prompt_pos_embed)

        self.class_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, num_classes_per_task)
            for _ in range(total_num_tasks)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.vit.eval()
        return self

    def set_current_task(self, task_id: int):
        self.current_task_id = task_id

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Extract CLS features augmented with selected prompts."""
        vit = self.vit

        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        if hasattr(vit, "patch_drop"):
            x = vit.patch_drop(x)
        if hasattr(vit, "norm_pre"):
            x = vit.norm_pre(x)

        # Query = mean of patch tokens (CLS excluded); detached to keep backbone frozen.
        query = x[:, 1:, :].mean(dim=1).detach()
        selected_prompts, selected_idx = self.prompt_pool(query)
        selected_prompts = selected_prompts + self.prompt_pos_embed

        # Token order: [CLS, selected prompts, patch tokens]
        x = torch.cat((x[:, :1, :], selected_prompts, x[:, 1:, :]), dim=1)

        x = vit.blocks(x)
        x = vit.norm(x)

        cls_out = x[:, 0, :]
        return cls_out, selected_idx

    def forward(
        self,
        x: Tensor,
        task_id: Optional[int] = None,
        return_selected_idx: bool = False,
    ):
        if task_id is None:
            task_id = self.current_task_id
        cls_out, selected_idx = self.forward_features(x)
        logits = self.class_heads[task_id](cls_out)
        if return_selected_idx:
            return logits, selected_idx
        return logits
