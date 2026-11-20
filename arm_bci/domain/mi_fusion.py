# -*- coding: utf-8 -*-
"""EEGNet and ShallowFBCSPNet probability ensemble for motor imagery."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EEGWindowPreprocessor:
    """Apply the project's persisted EEG preprocessing to neural inputs."""

    def __init__(self, signal_config):
        self.signal_config = signal_config

    def fit(self, eeg, y=None):
        return self

    def transform(self, eeg):
        from ..core import ensure_3d_epochs, preprocess_eeg

        return preprocess_eeg(ensure_3d_epochs(eeg, "eeg"), self.signal_config)

    def fit_transform(self, eeg, y=None):
        return self.fit(eeg, y).transform(eeg)


class _Square(nn.Module):
    def forward(self, x):
        return x.square()


class _Log(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class _EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        f1, depth, f2 = 8, 2, 16
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16), padding="same", groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, (1, 1), bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.35), nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.classifier = nn.Linear(f2 * 8, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class _ShallowFBCSPNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        filters = 40
        self.features = nn.Sequential(
            nn.Conv2d(1, filters, (1, 25), padding="same", bias=False),
            nn.Conv2d(filters, filters, (n_channels, 1), bias=False),
            nn.BatchNorm2d(filters), nn.ELU(), _Square(), nn.AvgPool2d((1, 30), stride=(1, 10)),
            _Log(), nn.Dropout(0.5), nn.AdaptiveAvgPool2d((1, 8)),
        )
        self.classifier = nn.Linear(filters * 8, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class BestMIFusionClassifier:
    """Equal-weight probabilities from three EEGNet and three shallow nets."""

    member_seeds = (33, 34, 35)
    member_weight = 1.0 / 6.0

    def __init__(self, epochs: int = 40, batch_size: int = 32, device: str = "auto"):
        self.epochs = max(1, int(epochs))
        self.batch_size = max(2, int(batch_size))
        self.device = str(device)
        self.classes_ = np.asarray([], dtype=np.int64)
        self.models_ = []
        self.member_families_ = []
        self.n_channels_ = None
        self.n_times_ = None

    def fit(self, eeg, y):
        x = np.asarray(eeg, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.ndim != 3 or len(x) != len(labels) or len(x) == 0:
            raise ValueError("MI 融合训练输入必须是 (trial, channel, sample) 且标签数量匹配")
        if not np.all(np.isfinite(x)):
            raise ValueError("EEG 融合训练输入包含非有限值")
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2:
            raise ValueError("MI 融合分类至少需要两个类别")

        target_device = self.device
        if target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(target_device)
        self.n_channels_, self.n_times_ = x.shape[1:]
        models = []
        families = []
        tx = torch.as_tensor(x[:, None, :, :], dtype=torch.float32)
        encoded = np.searchsorted(self.classes_, labels)
        ty = torch.as_tensor(encoded, dtype=torch.long)
        class_counts = np.bincount(encoded, minlength=len(self.classes_)).astype(np.float32)
        weights = torch.as_tensor(len(labels) / np.maximum(class_counts, 1.0), dtype=torch.float32)

        for family_index, model_type in enumerate((_EEGNet, _ShallowFBCSPNet)):
            for seed in self.member_seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = model_type(x.shape[1], len(self.classes_)).to(self.device_)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.device_))
                generator = torch.Generator().manual_seed(seed)
                dataset = torch.utils.data.TensorDataset(tx, ty)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True,
                    generator=generator, drop_last=False,
                )
                model.train()
                for _ in range(self.epochs):
                    for batch_x, batch_y in loader:
                        batch_x = batch_x.to(self.device_)
                        batch_y = batch_y.to(self.device_)
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_fn(model(batch_x), batch_y)
                        loss.backward()
                        optimizer.step()
                model.eval()
                models.append(model.cpu())
                families.append("EEGNet" if family_index == 0 else "ShallowFBCSPNet")

        self.models_ = models
        self.member_families_ = families
        return self

    def predict_proba(self, eeg):
        if not self.models_:
            raise RuntimeError("EEG 融合分类器尚未训练")
        x = np.asarray(eeg, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (self.n_channels_, self.n_times_):
            raise ValueError("EEG 融合分类输入必须是 (trial, channel, sample)")
        if x.shape[1] <= 0 or not np.all(np.isfinite(x)):
            raise ValueError("EEG 融合分类输入无效")
        tensor = torch.as_tensor(x[:, None, :, :], dtype=torch.float32)
        member_probabilities = []
        with torch.inference_mode():
            for model in self.models_:
                model.eval()
                member_probabilities.append(model(tensor).softmax(dim=1).cpu().numpy())
        return np.mean(member_probabilities, axis=0, dtype=np.float64)

    def predict(self, eeg):
        return self.classes_[np.argmax(self.predict_proba(eeg), axis=1)]
