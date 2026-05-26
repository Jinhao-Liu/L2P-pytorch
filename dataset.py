"""Split CIFAR-100 dataset preparation for continual learning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from config import cfg


class TargetMappedSubset(Dataset):
    """Subset that remaps original CIFAR-100 labels to per-task local labels (0-9)."""

    def __init__(self, dataset: Dataset, indices: list[int], target_mapping: dict[int, int]):
        self.dataset = Subset(dataset, indices)
        self.target_mapping = target_mapping

    def __getitem__(self, idx: int):
        image, raw_label = self.dataset[idx]
        return image, self.target_mapping[int(raw_label)]

    def __len__(self) -> int:
        return len(self.dataset)


def get_transforms():
    """Training (augmented) and test (deterministic) transforms.

    Uses ImageNet statistics for compatibility with pretrained ViT backbones.
    """
    train_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(cfg.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_transform, test_transform


def _targets(dataset: datasets.CIFAR100) -> np.ndarray:
    return np.asarray(dataset.targets, dtype=np.int64)


def _make_loader(dataset: Dataset, train: bool) -> DataLoader:
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": train,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": False,
    }
    if cfg.num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": cfg.persistent_workers,
            "prefetch_factor": cfg.prefetch_factor,
        })
    return DataLoader(dataset, **loader_kwargs)


def get_split_cifar100():
    """Load CIFAR-100 and split into sequential continual learning tasks.

    Returns:
        train_loaders: list of per-task training DataLoaders.
        test_loaders: list of per-task test DataLoaders.
        task_class_mapping: original CIFAR-100 class indices for each task.
    """
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True, transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True, transform=test_transform,
    )

    task_class_mapping = [
        list(range(i * cfg.num_classes_per_task, (i + 1) * cfg.num_classes_per_task))
        for i in range(cfg.num_tasks)
    ]

    train_targets = _targets(train_dataset)
    test_targets = _targets(test_dataset)
    train_loaders, test_loaders = [], []

    for target_classes in task_class_mapping:
        label_map = {raw_label: idx for idx, raw_label in enumerate(target_classes)}

        train_indices = np.flatnonzero(np.isin(train_targets, target_classes)).tolist()
        test_indices = np.flatnonzero(np.isin(test_targets, target_classes)).tolist()

        train_subset = TargetMappedSubset(train_dataset, train_indices, label_map)
        test_subset = TargetMappedSubset(test_dataset, test_indices, label_map)

        train_loaders.append(_make_loader(train_subset, train=True))
        test_loaders.append(_make_loader(test_subset, train=False))

    return train_loaders, test_loaders, task_class_mapping
