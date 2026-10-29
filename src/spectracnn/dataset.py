"""PyTorch dataset and deterministic stratified splits."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class IQDataset(Dataset):
    """Tensor wrapper around ``(n, 2, length)`` frames."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def stratified_split(
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices per class without requiring scikit-learn."""

    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1)")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_size)))
        n_val = max(1, int(round(len(indices) * val_size)))
        test_indices.extend(indices[:n_test].tolist())
        val_indices.extend(indices[n_test : n_test + n_val].tolist())
        train_indices.extend(indices[n_test + n_val :].tolist())

    return (
        np.asarray(sorted(train_indices), dtype=np.int64),
        np.asarray(sorted(val_indices), dtype=np.int64),
        np.asarray(sorted(test_indices), dtype=np.int64),
    )
