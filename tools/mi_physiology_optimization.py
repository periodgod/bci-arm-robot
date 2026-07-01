#!/usr/bin/env python
"""Physiology-informed small-sample MI model optimization benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.covariance import oas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arm_bci_system import bandpass, common_average_reference  # noqa: E402
from mi_paths import EXPERIMENTS_DIR, PROCESSED_DATA_DIR, physionet_subject_path  # noqa: E402
from mi_physionet_benchmark import (  # noqa: E402
    MOTOR_CHANNELS,
    load_npz,
    select_channel_indices,
)


BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 30.0))
LAPLACIAN_DEFINITIONS = {
    "C3_lap": ("C3", ("FC3", "C1", "C5", "CP3")),
    "C4_lap": ("C4", ("FC4", "C2", "C6", "CP4")),
    "C1_lap": ("C1", ("FC1", "C3", "Cz", "CP1")),
    "C2_lap": ("C2", ("FC2", "C4", "Cz", "CP2")),
}
TIME_BAND_CANDIDATES = (
    (0.0, 2.0, 8.0, 30.0),
    (0.5, 2.0, 8.0, 30.0),
    (1.0, 2.0, 8.0, 30.0),
    (0.0, 3.0, 8.0, 30.0),
    (0.0, 2.0, 8.0, 14.0),
    (0.5, 2.0, 8.0, 14.0),
    (1.0, 2.0, 8.0, 14.0),
    (0.0, 3.0, 8.0, 14.0),
    (0.0, 2.0, 14.0, 30.0),
    (0.5, 2.0, 14.0, 30.0),
    (1.0, 2.0, 14.0, 30.0),
    (0.0, 3.0, 14.0, 30.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=physionet_subject_path(PROCESSED_DATA_DIR, 1),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENTS_DIR / "physionet_eegmmidb_s001_mi_optimization.json",
    )
    return parser.parse_args()


def crop_time(eeg: np.ndarray, fs: int, start_seconds: float, duration_seconds: float) -> np.ndarray:
    start = max(0, int(round(start_seconds * fs)))
    stop = min(eeg.shape[-1], start + int(round(duration_seconds * fs)))
    if stop - start < max(32, int(round(fs * 0.75))):
        raise ValueError("CSP time window is too short")
    return eeg[..., start:stop]


def fit_predict_csp_lda(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_idx: np.ndarray,
    low: float,
    high: float,
    components: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    train = bandpass(common_average_reference(train_eeg), fs, low, high)[:, channel_idx, :]
    test = bandpass(common_average_reference(test_eeg), fs, low, high)[:, channel_idx, :]
    model = make_pipeline(
        CSP(
            n_components=min(components, len(channel_idx)),
            reg="ledoit_wolf",
            log=True,
            norm_trace=False,
        ),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    model.fit(train, train_y)
    return model.predict(test).astype(np.int64), model.predict_proba(test).astype(np.float64)


def nested_time_frequency_csp(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_idx: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_candidate = TIME_BAND_CANDIDATES[0]
    best_score = -1.0
    for candidate in TIME_BAND_CANDIDATES:
        start, duration, low, high = candidate
        scores = []
        cropped = crop_time(train_eeg, fs, start, duration)
        for inner_train, inner_val in inner.split(cropped, train_y):
            pred, _ = fit_predict_csp_lda(
                cropped[inner_train],
                train_y[inner_train],
                cropped[inner_val],
                fs,
                channel_idx,
                low,
                high,
            )
            scores.append(balanced_accuracy_score(train_y[inner_val], pred))
        score = float(np.mean(scores))
        if score > best_score + 1e-12:
            best_score = score
            best_candidate = candidate
    start, duration, low, high = best_candidate
    prediction, _ = fit_predict_csp_lda(
        crop_time(train_eeg, fs, start, duration),
        train_y,
        crop_time(test_eeg, fs, start, duration),
        fs,
        channel_idx,
        low,
        high,
    )
    return prediction, best_candidate, best_score


def csp_band_ensemble(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_idx: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, List[float]]:
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    band_weights = []
    test_probabilities = []
    for low, high in BANDS:
        scores = []
        for inner_train, inner_val in inner.split(train_eeg, train_y):
            pred, _ = fit_predict_csp_lda(
                train_eeg[inner_train],
                train_y[inner_train],
                train_eeg[inner_val],
                fs,
                channel_idx,
                low,
                high,
                components=4,
            )
            scores.append(balanced_accuracy_score(train_y[inner_val], pred))
        # Values at chance receive almost no influence; stronger bands dominate.
        weight = max(0.01, float(np.mean(scores)) - 0.48) ** 2
        band_weights.append(weight)
        _, proba = fit_predict_csp_lda(
            train_eeg,
            train_y,
            test_eeg,
            fs,
            channel_idx,
            low,
            high,
            components=4,
        )
        test_probabilities.append(proba)
    weights = np.asarray(band_weights, dtype=np.float64)
    weights /= np.sum(weights)
    fused = np.sum(np.stack(test_probabilities, axis=0) * weights[:, None, None], axis=0)
    return np.argmax(fused, axis=1).astype(np.int64), weights.tolist()


def channel_lookup(channel_names: Sequence[str]) -> Dict[str, int]:
    return {str(name).upper(): idx for idx, name in enumerate(channel_names)}


def laplacian_signals(eeg: np.ndarray, channel_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    lookup = channel_lookup(channel_names)
    signals = []
    names = []
    for output_name, (center_name, neighbor_names) in LAPLACIAN_DEFINITIONS.items():
        required = (center_name,) + tuple(neighbor_names)
        if not all(name.upper() in lookup for name in required):
            continue
        center = eeg[:, lookup[center_name.upper()], :]
        neighbors = np.stack([eeg[:, lookup[name.upper()], :] for name in neighbor_names], axis=1)
        signals.append(center - np.mean(neighbors, axis=1))
        names.append(output_name)
    if len(signals) < 2:
        raise ValueError("C3/C4 Laplacian neighborhoods are incomplete")
    return np.stack(signals, axis=1).astype(np.float32), names


def physiological_erd_features(eeg: np.ndarray, fs: int, channel_names: Sequence[str]) -> np.ndarray:
    car = common_average_reference(eeg)
    laplacian, lap_names = laplacian_signals(car, channel_names)
    lap_lookup = {name: idx for idx, name in enumerate(lap_names)}
    segment_bounds = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (0.0, 3.0))
    parts = []
    for low, high in BANDS:
        filtered = bandpass(laplacian, fs, low, high)
        for start_seconds, stop_seconds in segment_bounds:
            start = int(round(start_seconds * fs))
            stop = min(filtered.shape[-1], int(round(stop_seconds * fs)))
            segment = filtered[..., start:stop]
            log_var = np.log(np.var(segment, axis=-1) + 1e-6)
            parts.append(log_var)
            if "C3_lap" in lap_lookup and "C4_lap" in lap_lookup:
                left = log_var[:, lap_lookup["C3_lap"]]
                right = log_var[:, lap_lookup["C4_lap"]]
                parts.append((left - right)[:, None])
                left_power = np.exp(left)
                right_power = np.exp(right)
                lateralization = (left_power - right_power) / (left_power + right_power + 1e-6)
                parts.append(lateralization[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_predict_physiological_erd(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    train_x = physiological_erd_features(train_eeg, fs, channel_names)
    test_x = physiological_erd_features(test_eeg, fs, channel_names)
    models = (
        make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.15, max_iter=2000, class_weight="balanced", solver="liblinear"),
        ),
    )
    predictions = []
    for model in models:
        model.fit(train_x, train_y)
        predictions.append(model.predict(test_x).astype(np.int64))
    return predictions[0], predictions[1]


def spd_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    return (vectors * np.log(values)) @ vectors.T


def spd_exp(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    return (vectors * np.exp(values)) @ vectors.T


def spd_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-8)
    return (vectors * (values ** -0.5)) @ vectors.T


def covariance_trials(eeg: np.ndarray) -> np.ndarray:
    covariances = []
    for trial in eeg:
        centered = trial - np.mean(trial, axis=1, keepdims=True)
        cov, _ = oas(centered.T, assume_centered=True)
        cov = cov / max(float(np.trace(cov)), 1e-12)
        covariances.append(cov.astype(np.float64))
    return np.stack(covariances)


def log_euclidean_mean(covariances: np.ndarray) -> np.ndarray:
    return spd_exp(np.mean(np.stack([spd_log(cov) for cov in covariances]), axis=0))


def affine_distance(covariance: np.ndarray, reference: np.ndarray) -> float:
    inv_sqrt = spd_inv_sqrt(reference)
    whitened = inv_sqrt @ covariance @ inv_sqrt
    return float(np.linalg.norm(spd_log(whitened), ord="fro"))


def riemannian_weighted_filter_bank(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    train_parts = []
    test_parts = []
    weights = []
    direct_train_distances = []
    direct_test_distances = []
    car_train = common_average_reference(train_eeg)
    car_test = common_average_reference(test_eeg)
    for low, high in BANDS:
        train_filtered = bandpass(car_train, fs, low, high)[:, channel_idx, :]
        test_filtered = bandpass(car_test, fs, low, high)[:, channel_idx, :]
        train_cov = covariance_trials(train_filtered)
        test_cov = covariance_trials(test_filtered)
        class_means = [log_euclidean_mean(train_cov[train_y == cls]) for cls in (0, 1)]
        train_distance = np.asarray(
            [[affine_distance(cov, mean) for mean in class_means] for cov in train_cov], dtype=np.float32
        )
        test_distance = np.asarray(
            [[affine_distance(cov, mean) for mean in class_means] for cov in test_cov], dtype=np.float32
        )
        score_axis = train_distance[:, 1] - train_distance[:, 0]
        separation = abs(float(np.mean(score_axis[train_y == 0]) - np.mean(score_axis[train_y == 1])))
        spread = float(np.std(score_axis[train_y == 0]) + np.std(score_axis[train_y == 1]) + 1e-6)
        weight = max(0.02, separation / spread)
        weights.append(weight)
        train_parts.append(train_distance * math.sqrt(weight))
        test_parts.append(test_distance * math.sqrt(weight))
        direct_train_distances.append(train_distance)
        direct_test_distances.append(test_distance)

    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array /= np.sum(weight_array)
    train_x = np.concatenate(train_parts, axis=1)
    test_x = np.concatenate(test_parts, axis=1)
    classifier = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    classifier.fit(train_x, train_y)
    lda_prediction = classifier.predict(test_x).astype(np.int64)

    test_stack = np.stack(direct_test_distances, axis=0)
    fused_distance = np.sum(test_stack * weight_array[:, None, None], axis=0)
    mdm_prediction = np.argmin(fused_distance, axis=1).astype(np.int64)
    return lda_prediction, mdm_prediction, weight_array.tolist()


def summarize(y_true: np.ndarray, y_pred: np.ndarray, elapsed: float) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "elapsed_seconds": float(elapsed),
    }


def benchmark(
    eeg: np.ndarray,
    labels: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    folds: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    y = np.asarray([0 if str(label) == "left" else 1 for label in labels], dtype=np.int64)
    motor_idx, motor_names = select_channel_indices(channel_names, MOTOR_CHANNELS)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = {
        "baseline_compact_csp_lda": np.full(len(y), -1, dtype=np.int64),
        "nested_time_frequency_csp_lda": np.full(len(y), -1, dtype=np.int64),
        "weighted_csp_band_ensemble": np.full(len(y), -1, dtype=np.int64),
        "physio_laplacian_erd_lda": np.full(len(y), -1, dtype=np.int64),
        "physio_laplacian_erd_logreg": np.full(len(y), -1, dtype=np.int64),
        "riemann_weighted_distance_lda": np.full(len(y), -1, dtype=np.int64),
        "riemann_weighted_mdm": np.full(len(y), -1, dtype=np.int64),
    }
    elapsed = {name: 0.0 for name in predictions}
    selected_candidates = []
    csp_band_weights = []
    riemann_band_weights = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(eeg, y), start=1):
        print(f"Fold {fold_idx}/{folds}", flush=True)
        train_eeg, test_eeg = eeg[train_idx], eeg[test_idx]
        train_y = y[train_idx]

        started = time.perf_counter()
        pred, _ = fit_predict_csp_lda(train_eeg, train_y, test_eeg, fs, motor_idx, 8.0, 30.0)
        predictions["baseline_compact_csp_lda"][test_idx] = pred
        elapsed["baseline_compact_csp_lda"] += time.perf_counter() - started

        started = time.perf_counter()
        pred, candidate, inner_score = nested_time_frequency_csp(
            train_eeg, train_y, test_eeg, fs, motor_idx, seed + fold_idx
        )
        predictions["nested_time_frequency_csp_lda"][test_idx] = pred
        elapsed["nested_time_frequency_csp_lda"] += time.perf_counter() - started
        selected_candidates.append({"candidate": list(candidate), "inner_balanced_accuracy": inner_score})

        started = time.perf_counter()
        pred, weights = csp_band_ensemble(train_eeg, train_y, test_eeg, fs, motor_idx, seed + fold_idx)
        predictions["weighted_csp_band_ensemble"][test_idx] = pred
        elapsed["weighted_csp_band_ensemble"] += time.perf_counter() - started
        csp_band_weights.append(weights)

        started = time.perf_counter()
        pred_lda, pred_logreg = fit_predict_physiological_erd(train_eeg, train_y, test_eeg, fs, channel_names)
        predictions["physio_laplacian_erd_lda"][test_idx] = pred_lda
        predictions["physio_laplacian_erd_logreg"][test_idx] = pred_logreg
        shared = time.perf_counter() - started
        elapsed["physio_laplacian_erd_lda"] += shared
        elapsed["physio_laplacian_erd_logreg"] += shared

        started = time.perf_counter()
        pred_lda, pred_mdm, weights = riemannian_weighted_filter_bank(
            train_eeg, train_y, test_eeg, fs, motor_idx
        )
        predictions["riemann_weighted_distance_lda"][test_idx] = pred_lda
        predictions["riemann_weighted_mdm"][test_idx] = pred_mdm
        shared = time.perf_counter() - started
        elapsed["riemann_weighted_distance_lda"] += shared
        elapsed["riemann_weighted_mdm"] += shared
        riemann_band_weights.append(weights)

    results = {name: summarize(y, pred, elapsed[name]) for name, pred in predictions.items()}
    details = {
        "motor_channels": motor_names,
        "bands_hz": [list(band) for band in BANDS],
        "nested_selected_candidates": selected_candidates,
        "nested_candidate_frequency": dict(
            Counter(str(item["candidate"]) for item in selected_candidates)
        ),
        "mean_csp_ensemble_band_weights": np.mean(csp_band_weights, axis=0).tolist(),
        "mean_riemann_band_weights": np.mean(riemann_band_weights, axis=0).tolist(),
    }
    return results, details


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")
    eeg, labels, fs, channel_names, _ = load_npz(args.data)
    print(f"Data: {eeg.shape}, fs={fs}, labels={Counter(map(str, labels))}")
    results, details = benchmark(eeg, labels, fs, channel_names, args.folds, args.seed)
    print("\nPhysiology-informed MI benchmark")
    print("-" * 88)
    print(f"{'model':40s} {'accuracy':>10s} {'balanced':>10s} {'kappa':>10s} {'seconds':>10s}")
    print("-" * 88)
    for name, metrics in sorted(results.items(), key=lambda item: float(item[1]["accuracy"]), reverse=True):
        print(
            f"{name:40s} {metrics['accuracy']:10.4f} {metrics['balanced_accuracy']:10.4f} "
            f"{metrics['kappa']:10.4f} {metrics['elapsed_seconds']:10.2f}"
        )
    report = {
        "data": str(args.data.resolve()),
        "validation": f"{args.folds}-fold stratified trial-level CV; nested selection uses training folds only",
        "seed": args.seed,
        "results": results,
        "details": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
