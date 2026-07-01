#!/usr/bin/env python
"""Fair low-shot comparison of source-pretrained MI deep encoders."""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import mne
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arm_bci_system import bandpass  # noqa: E402
from mi_deep_models import EEGNetBinary, standardize_trials  # noqa: E402
from mi_paths import EXPERIMENTS_DIR, PROCESSED_DATA_DIR  # noqa: E402
from mi_low_shot_innovation import (  # noqa: E402
    load_preprocessed_subjects,
    stratified_low_shot_split,
)
from mi_transfer_regularized_csp import subject_path  # noqa: E402


DEFAULT_SHOTS = (2, 4, 6, 8, 12, 16)
FBC_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 30.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--source-train-subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--source-validation-subject", type=int, default=10)
    parser.add_argument("--validation-subjects", type=int, nargs="*", default=list(range(11, 21)))
    parser.add_argument("--confirmation-subjects", type=int, nargs="*", default=list(range(21, 31)))
    parser.add_argument("--models", nargs="*", default=["eegnet", "fbcnet", "eegconformer"])
    parser.add_argument("--shots", type=int, nargs="*", default=list(DEFAULT_SHOTS))
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENTS_DIR / "physionet_mi_low_shot_deep_compare.json",
    )
    return parser.parse_args()


class LogVariance(nn.Module):
    def __init__(self, segments: int = 4):
        super().__init__()
        self.segments = segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.tensor_split(x, self.segments, dim=-1)
        values = [torch.log(torch.clamp(torch.var(chunk, dim=-1), min=1e-6)) for chunk in chunks]
        return torch.cat(values, dim=-1)


class FBCNetBinary(nn.Module):
    """Compact FBCNet-style bandwise spatial filtering and temporal log-variance."""

    def __init__(self, bands: int, channels: int, spatial_filters: int = 8, segments: int = 4):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(
                bands,
                bands * spatial_filters,
                (channels, 1),
                groups=bands,
                bias=False,
            ),
            nn.BatchNorm2d(bands * spatial_filters),
            nn.SiLU(),
        )
        self.log_variance = LogVariance(segments)
        self.embedding_size = bands * spatial_filters * segments
        self.classifier = nn.Linear(self.embedding_size, 2)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(x).squeeze(2)
        return self.log_variance(spatial).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


class EEGConformerBinary(nn.Module):
    """Lightweight EEG-Conformer-style convolutional patch embedding plus attention."""

    def __init__(self, channels: int, embedding: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False),
            nn.Conv2d(16, embedding, (channels, 1), bias=False),
            nn.BatchNorm2d(embedding),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 4)),
            nn.Dropout(0.35),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding,
            nhead=heads,
            dim_feedforward=embedding * 2,
            dropout=0.35,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(embedding)
        self.classifier = nn.Linear(embedding, 2)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch(x).squeeze(2).transpose(1, 2)
        return self.norm(self.encoder(patches).mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


class EEGNetWithEmbedding(EEGNetBinary):
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_broadband(eeg: np.ndarray) -> np.ndarray:
    return standardize_trials(eeg)[:, None, :, :]


def prepare_filter_bank(eeg: np.ndarray, fs: int) -> np.ndarray:
    bands = []
    for low, high in FBC_BANDS:
        filtered = bandpass(np.asarray(eeg, dtype=np.float64), fs, low, high)
        bands.append(standardize_trials(filtered))
    return np.stack(bands, axis=1).astype(np.float32)


def build_model(kind: str, channels: int, samples: int, fs: int) -> nn.Module:
    if kind == "eegnet":
        return EEGNetWithEmbedding(channels, samples, fs)
    if kind == "fbcnet":
        return FBCNetBinary(len(FBC_BANDS), channels)
    if kind == "eegconformer":
        return EEGConformerBinary(channels)
    raise ValueError(f"Unknown model: {kind}")


def prepare_input(kind: str, eeg: np.ndarray, fs: int) -> np.ndarray:
    if kind == "fbcnet":
        return prepare_filter_bank(eeg, fs)
    return prepare_broadband(eeg)


def fit_epochs(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    validation: Tuple[np.ndarray, np.ndarray] | None = None,
    patience: int = 0,
) -> Tuple[nn.Module, int, float]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.int64))),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = epochs
    stale = 0
    validation_tensors = None
    if validation is not None:
        validation_tensors = (
            torch.from_numpy(validation[0]).to(device),
            torch.from_numpy(validation[1].astype(np.int64)).to(device),
        )
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            scale = 0.9 + 0.2 * torch.rand((len(batch_x), 1, 1, 1), device=device)
            logits = model(batch_x * scale + 0.01 * torch.randn_like(batch_x))
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if validation_tensors is None:
            continue
        model.eval()
        with torch.no_grad():
            current = float(criterion(model(validation_tensors[0]), validation_tensors[1]).item())
        if current < best_loss - 1e-4:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if validation_tensors is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def pretrain_source_encoder(
    kind: str,
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    validation_eeg: np.ndarray,
    validation_y: np.ndarray,
    fs: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> Tuple[nn.Module, Dict[str, float]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = prepare_input(kind, train_eeg, fs)
    validation_x = prepare_input(kind, validation_eeg, fs)
    model = build_model(kind, train_eeg.shape[1], train_eeg.shape[2], fs)
    started = time.perf_counter()
    _, best_epoch, best_loss = fit_epochs(
        model,
        train_x,
        train_y,
        device,
        max_epochs,
        batch_size,
        seed,
        validation=(validation_x, validation_y),
        patience=patience,
    )
    # Refit from scratch on every source subject for the selected number of epochs.
    set_seed(seed)
    combined_eeg = np.concatenate((train_eeg, validation_eeg))
    combined_y = np.concatenate((train_y, validation_y))
    combined_x = prepare_input(kind, combined_eeg, fs)
    model = build_model(kind, combined_eeg.shape[1], combined_eeg.shape[2], fs)
    model, _, _ = fit_epochs(
        model,
        combined_x,
        combined_y,
        device,
        best_epoch,
        batch_size,
        seed,
    )
    elapsed = time.perf_counter() - started
    model.eval()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return model, {
        "selected_epochs": float(best_epoch),
        "source_validation_loss": float(best_loss),
        "source_pretraining_seconds": float(elapsed),
        "total_parameters": float(total_parameters),
        "trainable_parameters": float(trainable_parameters),
        "device_is_cuda": float(device.type == "cuda"),
    }


def extract_embeddings(
    model: nn.Module,
    kind: str,
    eeg: np.ndarray,
    fs: int,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    device = next(model.parameters()).device
    prepared = prepare_input(kind, eeg, fs)
    loader = DataLoader(torch.from_numpy(prepared), batch_size=batch_size, shuffle=False)
    features = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features.append(model.embed(batch.to(device)).cpu().numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(features), elapsed


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    output: Dict[str, object] = {"evaluations": len(rows)}
    for key in (
        "accuracy",
        "balanced_accuracy",
        "kappa",
        "calibration_and_predict_seconds",
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[f"mean_{key}"] = float(np.mean(values))
        output[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def evaluate_targets(
    model: nn.Module,
    kind: str,
    target_ids: Sequence[int],
    subject_eeg: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
    fs: int,
    shots_values: Sequence[int],
    repeats: int,
    batch_size: int,
    seed: int,
) -> Dict[str, object]:
    all_rows: List[Dict[str, float]] = []
    by_shots: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    by_subject: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    embedding_seconds = 0.0
    embedding_trials = 0
    for target_id in target_ids:
        labels = subject_labels[target_id]
        features, elapsed = extract_embeddings(
            model, kind, subject_eeg[target_id], fs, batch_size
        )
        embedding_seconds += elapsed
        embedding_trials += len(labels)
        for shots in shots_values:
            for repeat in range(repeats):
                split_seed = seed + target_id * 100_000 + shots * 1_000 + repeat
                train_idx, test_idx = stratified_low_shot_split(labels, shots, split_seed)
                classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                started = time.perf_counter()
                classifier.fit(features[train_idx], labels[train_idx])
                prediction = classifier.predict(features[test_idx]).astype(np.int64)
                row = {
                    "accuracy": float(accuracy_score(labels[test_idx], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(labels[test_idx], prediction)
                    ),
                    "kappa": float(cohen_kappa_score(labels[test_idx], prediction)),
                    "calibration_and_predict_seconds": float(time.perf_counter() - started),
                }
                all_rows.append(row)
                by_shots[str(shots)].append(row)
                by_subject[f"S{target_id:03d}"].append(row)
    return {
        "overall": summarize(all_rows),
        "by_shots": {key: summarize(value) for key, value in by_shots.items()},
        "by_subject": {key: summarize(value) for key, value in by_subject.items()},
        "embedding_milliseconds_per_trial": float(
            1000.0 * embedding_seconds / max(embedding_trials, 1)
        ),
    }


def concatenate_subjects(
    subject_ids: Sequence[int],
    subject_eeg: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([subject_eeg[subject] for subject in subject_ids]),
        np.concatenate([subject_labels[subject] for subject in subject_ids]),
    )


def validate_disjoint(groups: Dict[str, Sequence[int]]) -> None:
    seen: Dict[int, str] = {}
    for name, subjects in groups.items():
        for subject in subjects:
            if subject in seen:
                raise ValueError(f"S{subject:03d} appears in both {seen[subject]} and {name}")
            seen[subject] = name


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    groups = {
        "source_train": args.source_train_subjects,
        "source_validation": [args.source_validation_subject],
        "validation": args.validation_subjects,
        "confirmation": args.confirmation_subjects,
    }
    validate_disjoint(groups)
    all_subjects = sorted({subject for values in groups.values() for subject in values})
    missing = [subject for subject in all_subjects if not subject_path(args.data_dir, subject).exists()]
    if missing:
        raise FileNotFoundError(f"Missing subject NPZ files: {missing}")
    subject_eeg, _, subject_labels, channels, fs = load_preprocessed_subjects(
        args.data_dir, all_subjects
    )
    train_eeg, train_y = concatenate_subjects(
        args.source_train_subjects, subject_eeg, subject_labels
    )
    validation_eeg = subject_eeg[args.source_validation_subject]
    validation_y = subject_labels[args.source_validation_subject]

    results = {}
    for kind in args.models:
        print(f"Source pretraining {kind}", flush=True)
        model, metadata = pretrain_source_encoder(
            kind,
            train_eeg,
            train_y,
            validation_eeg,
            validation_y,
            fs,
            args.max_epochs,
            args.patience,
            args.batch_size,
            args.seed,
        )
        print(f"Validation targets {kind}", flush=True)
        validation_result = evaluate_targets(
            model,
            kind,
            args.validation_subjects,
            subject_eeg,
            subject_labels,
            fs,
            args.shots,
            args.repeats,
            args.batch_size,
            args.seed,
        )
        print(f"Final confirmation targets {kind}", flush=True)
        confirmation_result = evaluate_targets(
            model,
            kind,
            args.confirmation_subjects,
            subject_eeg,
            subject_labels,
            fs,
            args.shots,
            args.repeats,
            args.batch_size,
            args.seed,
        )
        results[kind] = {
            "metadata": metadata,
            "validation": validation_result,
            "final_confirmation": confirmation_result,
        }
        print(
            f"{kind}: final bacc="
            f"{confirmation_result['overall']['mean_balanced_accuracy']:.4f}",
            flush=True,
        )

    report = {
        "protocol": {
            "dataset": "PhysioNet EEGMMIDB runs 4/8/12, left/right fist motor imagery",
            "source_train_subjects": [f"S{value:03d}" for value in args.source_train_subjects],
            "source_early_stopping_subject": f"S{args.source_validation_subject:03d}",
            "source_refit_uses_all_source_subjects": True,
            "validation_subjects": [f"S{value:03d}" for value in args.validation_subjects],
            "final_untouched_confirmation_subjects": [
                f"S{value:03d}" for value in args.confirmation_subjects
            ],
            "subject_groups_disjoint": True,
            "target_adaptation": "frozen encoder plus shrinkage LDA linear probe",
            "shots_per_class": args.shots,
            "repeats": args.repeats,
            "channels": channels,
            "sampling_rate_hz": fs,
            "broadband_hz": [8.0, 30.0],
            "fbcnet_subbands_hz": [list(band) for band in FBC_BANDS],
            "seed": args.seed,
        },
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
