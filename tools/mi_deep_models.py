"""Compact PyTorch models for binary motor-imagery classification."""

from __future__ import annotations

import copy
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class SafeLog(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=1e-6))


class EEGNetBinary(nn.Module):
    def __init__(self, channels: int, samples: int, fs: int, dropout: float = 0.35):
        super().__init__()
        del samples
        temporal_kernel = max(16, int(round(fs * 0.5)))
        separable_kernel = max(8, int(round(fs * 0.1)))
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                16,
                16,
                (1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=16,
                bias=False,
            ),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class ShallowConvNetBinary(nn.Module):
    def __init__(self, channels: int, samples: int, fs: int, dropout: float = 0.4):
        super().__init__()
        del samples
        temporal_kernel = max(9, int(round(fs * 0.16)))
        pool_kernel = max(8, int(round(fs * 0.20)))
        pool_stride = max(2, int(round(fs * 0.04)))
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.Conv2d(24, 40, (channels, 1), bias=False),
            nn.BatchNorm2d(40),
            Square(),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
            SafeLog(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.classifier = nn.Linear(40 * 8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def standardize_trials(eeg: np.ndarray) -> np.ndarray:
    x = np.asarray(eeg, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def make_trial_windows(
    eeg: np.ndarray,
    labels: np.ndarray,
    fs: int,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = standardize_trials(eeg)
    labels = np.asarray(labels, dtype=np.int64)
    window = min(x.shape[-1], max(32, int(round(window_seconds * fs))))
    step = max(1, int(round(step_seconds * fs)))
    starts = list(range(0, x.shape[-1] - window + 1, step)) or [0]
    if starts[-1] != x.shape[-1] - window:
        starts.append(x.shape[-1] - window)

    windows = []
    window_labels = []
    trial_ids = []
    for trial_idx in range(len(x)):
        for start in starts:
            windows.append(x[trial_idx, :, start : start + window])
            window_labels.append(labels[trial_idx])
            trial_ids.append(trial_idx)
    return (
        np.stack(windows).astype(np.float32),
        np.asarray(window_labels, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
    )


def build_model(kind: str, channels: int, samples: int, fs: int) -> nn.Module:
    if kind == "eegnet":
        return EEGNetBinary(channels, samples, fs)
    if kind == "shallowconvnet":
        return ShallowConvNetBinary(channels, samples, fs)
    raise ValueError(f"Unknown deep MI model: {kind}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_trial_predictions(probabilities: np.ndarray, trial_ids: np.ndarray, trial_count: int) -> np.ndarray:
    predictions = []
    for trial_idx in range(trial_count):
        mask = trial_ids == trial_idx
        predictions.append(int(np.argmax(np.mean(probabilities[mask], axis=0))))
    return np.asarray(predictions, dtype=np.int64)


def fit_predict_deep(
    kind: str,
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    seed: int = 42,
    max_epochs: int = 100,
    patience: int = 15,
) -> Tuple[np.ndarray, Dict[str, float]]:
    set_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_y = np.asarray(train_y, dtype=np.int64)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=seed)
    inner_train_idx, inner_val_idx = next(inner.split(train_eeg, train_y))
    train_windows, train_labels, _ = make_trial_windows(train_eeg[inner_train_idx], train_y[inner_train_idx], fs)
    val_windows, val_labels, _ = make_trial_windows(train_eeg[inner_val_idx], train_y[inner_val_idx], fs)
    test_windows, _, test_trial_ids = make_trial_windows(
        test_eeg,
        np.zeros(len(test_eeg), dtype=np.int64),
        fs,
    )

    train_tensor = torch.from_numpy(train_windows[:, None, :, :])
    train_label_tensor = torch.from_numpy(train_labels)
    val_tensor = torch.from_numpy(val_windows[:, None, :, :]).to(device)
    val_label_tensor = torch.from_numpy(val_labels).to(device)
    test_tensor = torch.from_numpy(test_windows[:, None, :, :]).to(device)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor, train_label_tensor),
        batch_size=min(32, len(train_tensor)),
        shuffle=True,
        generator=generator,
    )
    model = build_model(kind, train_windows.shape[1], train_windows.shape[2], fs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scale = 0.9 + 0.2 * torch.rand((len(batch_x), 1, 1, 1), device=device)
            noise = 0.015 * torch.randn_like(batch_x)
            logits = model(batch_x * scale + noise)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_tensor), val_label_tensor).item())
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(test_tensor), dim=1).cpu().numpy()
    predictions = aggregate_trial_predictions(probabilities, test_trial_ids, len(test_eeg))
    metadata = {
        "best_epoch": float(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "parameter_count": float(sum(parameter.numel() for parameter in model.parameters())),
        "device_is_cuda": float(device.type == "cuda"),
    }
    return predictions, metadata
