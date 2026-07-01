#!/usr/bin/env python
"""Development-subject selection and held-out-subject MI validation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mne
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arm_bci_system import DEFAULT_EEG_CHANNELS  # noqa: E402
from mi_paths import EXPERIMENTS_DIR, PROCESSED_DATA_DIR, physionet_subject_path  # noqa: E402
from mi_physionet_benchmark import MOTOR_CHANNELS, load_npz, select_channel_indices  # noqa: E402
from mi_physiology_optimization import (  # noqa: E402
    crop_time,
    fit_predict_csp_lda,
    fit_predict_physiological_erd,
    riemannian_weighted_filter_bank,
)


CSP_CONFIGS = {
    "csp_motor21_8_30_c6": ("motor21", 0.0, 3.0, 8.0, 30.0, 6),
    "csp_motor21_8_30_c4": ("motor21", 0.0, 3.0, 8.0, 30.0, 4),
    "csp_motor21_8_30_c2": ("motor21", 0.0, 3.0, 8.0, 30.0, 2),
    "csp_motor21_8_14_c4": ("motor21", 0.0, 3.0, 8.0, 14.0, 4),
    "csp_motor21_14_30_c4": ("motor21", 0.0, 3.0, 14.0, 30.0, 4),
    "csp_motor21_10_26_c4": ("motor21", 0.0, 3.0, 10.0, 26.0, 4),
    "csp_motor21_mid2s_8_30_c4": ("motor21", 0.5, 2.0, 8.0, 30.0, 4),
    "csp_deployment11_8_30_c4": ("deployment", 0.0, 3.0, 8.0, 30.0, 4),
    "csp_c3_cz_c4_8_30_c2": ("central3", 0.0, 3.0, 8.0, 30.0, 2),
}
NEW_CANDIDATES = tuple(name for name in CSP_CONFIGS if name != "csp_motor21_8_30_c6") + (
    "csp_component_ensemble",
    "csp_time_frequency_ensemble",
    "physio_laplacian_erd_lda",
    "riemann_weighted_distance_lda",
)
ALL_CANDIDATES = ("csp_motor21_8_30_c6",) + NEW_CANDIDATES
NESTED_CSP_CANDIDATES = (
    "csp_motor21_8_30_c2",
    "csp_motor21_8_30_c4",
    "csp_motor21_8_30_c6",
    "csp_motor21_8_14_c4",
    "csp_motor21_14_30_c4",
    "csp_motor21_10_26_c4",
    "csp_motor21_mid2s_8_30_c4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--development-subject", type=int, default=1)
    parser.add_argument("--confirmation-subjects", type=int, nargs="*", default=list(range(2, 11)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENTS_DIR / "physionet_mi_multisubject_validation.json",
    )
    return parser.parse_args()


def data_path(data_dir: Path, subject: int) -> Path:
    return physionet_subject_path(data_dir, subject)


def binary_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray([0 if str(label) == "left" else 1 for label in labels], dtype=np.int64)


def indices_for_group(channel_names: Sequence[str], group: str) -> np.ndarray:
    if group == "motor21":
        indices, _ = select_channel_indices(channel_names, MOTOR_CHANNELS)
        return indices
    if group == "deployment":
        requested = [name for name in MOTOR_CHANNELS if name in DEFAULT_EEG_CHANNELS]
        indices, _ = select_channel_indices(channel_names, requested)
        return indices
    if group == "central3":
        indices, _ = select_channel_indices(channel_names, ("C3", "Cz", "C4"))
        return indices
    raise ValueError(f"Unknown channel group: {group}")


def predict_csp_config(
    name: str,
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    group, start, duration, low, high, components = CSP_CONFIGS[name]
    indices = indices_for_group(channel_names, group)
    return fit_predict_csp_lda(
        crop_time(train_eeg, fs, start, duration),
        train_y,
        crop_time(test_eeg, fs, start, duration),
        fs,
        indices,
        low,
        high,
        components,
    )


def predict_candidate(
    name: str,
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
) -> np.ndarray:
    if name in CSP_CONFIGS:
        prediction, _ = predict_csp_config(name, train_eeg, train_y, test_eeg, fs, channel_names)
        return prediction
    if name == "csp_component_ensemble":
        probabilities = [
            predict_csp_config(config, train_eeg, train_y, test_eeg, fs, channel_names)[1]
            for config in (
                "csp_motor21_8_30_c2",
                "csp_motor21_8_30_c4",
                "csp_motor21_8_30_c6",
            )
        ]
        return np.argmax(np.mean(probabilities, axis=0), axis=1).astype(np.int64)
    if name == "csp_time_frequency_ensemble":
        probabilities = [
            predict_csp_config(config, train_eeg, train_y, test_eeg, fs, channel_names)[1]
            for config in (
                "csp_motor21_8_30_c4",
                "csp_motor21_mid2s_8_30_c4",
                "csp_motor21_8_14_c4",
                "csp_motor21_14_30_c4",
            )
        ]
        return np.argmax(np.mean(probabilities, axis=0), axis=1).astype(np.int64)
    if name == "physio_laplacian_erd_lda":
        prediction, _ = fit_predict_physiological_erd(train_eeg, train_y, test_eeg, fs, channel_names)
        return prediction
    if name == "riemann_weighted_distance_lda":
        motor_idx = indices_for_group(channel_names, "motor21")
        prediction, _, _ = riemannian_weighted_filter_bank(train_eeg, train_y, test_eeg, fs, motor_idx)
        return prediction
    raise ValueError(f"Unknown model: {name}")


def predict_nested_subject_adaptive_csp(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, str, float]:
    """Select CSP physiology parameters using only the current outer training fold."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {}
    for name in NESTED_CSP_CANDIDATES:
        fold_scores = []
        for inner_train, inner_validation in inner.split(train_eeg, train_y):
            prediction, _ = predict_csp_config(
                name,
                train_eeg[inner_train],
                train_y[inner_train],
                train_eeg[inner_validation],
                fs,
                channel_names,
            )
            fold_scores.append(balanced_accuracy_score(train_y[inner_validation], prediction))
        scores[name] = float(np.mean(fold_scores))

    # Deterministic order resolves ties in favor of lower-dimensional CSP models.
    complexity_order = {
        name: rank
        for rank, name in enumerate(
            (
                "csp_motor21_8_30_c2",
                "csp_motor21_8_30_c4",
                "csp_motor21_8_14_c4",
                "csp_motor21_14_30_c4",
                "csp_motor21_10_26_c4",
                "csp_motor21_mid2s_8_30_c4",
                "csp_motor21_8_30_c6",
            )
        )
    }
    selected = max(scores, key=lambda name: (scores[name], -complexity_order[name]))
    prediction, _ = predict_csp_config(
        selected, train_eeg, train_y, test_eeg, fs, channel_names
    )
    return prediction, selected, scores[selected]


def evaluate_subject(
    path: Path,
    candidates: Sequence[str],
    folds: int,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    eeg, labels, fs, channel_names, _ = load_npz(path)
    y = binary_labels(labels)
    splitter = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(eeg, y))
    output = {}
    for name in candidates:
        prediction = np.full(len(y), -1, dtype=np.int64)
        started = time.perf_counter()
        for train_idx, test_idx in splitter:
            prediction[test_idx] = predict_candidate(
                name,
                eeg[train_idx],
                y[train_idx],
                eeg[test_idx],
                fs,
                channel_names,
            )
        elapsed = time.perf_counter() - started
        output[name] = {
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "kappa": float(cohen_kappa_score(y, prediction)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            "five_fold_seconds": float(elapsed),
            "milliseconds_per_trial_cv": float(1000.0 * elapsed / len(y)),
            "trial_count": int(len(y)),
        }
    return output


def evaluate_nested_subject_adaptive(
    path: Path,
    folds: int,
    seed: int,
) -> Dict[str, object]:
    eeg, labels, fs, channel_names, _ = load_npz(path)
    y = binary_labels(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prediction = np.full(len(y), -1, dtype=np.int64)
    selections = []
    inner_scores = []
    started = time.perf_counter()
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(eeg, y)):
        fold_prediction, selected, inner_score = predict_nested_subject_adaptive_csp(
            eeg[train_idx],
            y[train_idx],
            eeg[test_idx],
            fs,
            channel_names,
            seed + fold_index,
        )
        prediction[test_idx] = fold_prediction
        selections.append(selected)
        inner_scores.append(inner_score)
    elapsed = time.perf_counter() - started
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "kappa": float(cohen_kappa_score(y, prediction)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "five_fold_seconds": float(elapsed),
        "milliseconds_per_trial_cv": float(1000.0 * elapsed / len(y)),
        "trial_count": int(len(y)),
        "outer_fold_selections": selections,
        "selection_frequency": dict(Counter(selections)),
        "mean_inner_balanced_accuracy": float(np.mean(inner_scores)),
    }


def aggregate_confirmation(per_subject: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, Dict[str, object]]:
    model_names = sorted({name for subject in per_subject.values() for name in subject})
    aggregate = {}
    for name in model_names:
        rows = [subject[name] for subject in per_subject.values() if name in subject]
        accuracy = np.asarray([row["accuracy"] for row in rows], dtype=float)
        balanced = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
        kappa = np.asarray([row["kappa"] for row in rows], dtype=float)
        milliseconds = np.asarray([row["milliseconds_per_trial_cv"] for row in rows], dtype=float)
        aggregate[name] = {
            "subject_count": len(rows),
            "mean_accuracy": float(np.mean(accuracy)),
            "std_accuracy": float(np.std(accuracy, ddof=1)) if len(rows) > 1 else 0.0,
            "mean_balanced_accuracy": float(np.mean(balanced)),
            "mean_kappa": float(np.mean(kappa)),
            "mean_milliseconds_per_trial_cv": float(np.mean(milliseconds)),
            "subject_accuracies": accuracy.tolist(),
        }
    return aggregate


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")
    development_path = data_path(args.data_dir, args.development_subject)
    development = evaluate_subject(development_path, ALL_CANDIDATES, args.folds, args.seed)
    best_overall = max(development, key=lambda name: development[name]["balanced_accuracy"])
    best_new = max(NEW_CANDIDATES, key=lambda name: development[name]["balanced_accuracy"])
    confirmation_models = tuple(dict.fromkeys(("csp_motor21_8_30_c6", best_overall, best_new)))

    available_subjects = [
        subject for subject in args.confirmation_subjects if data_path(args.data_dir, subject).exists()
    ]
    confirmation = {}
    for subject in available_subjects:
        print(f"Confirmation subject S{subject:03d}: {confirmation_models}", flush=True)
        subject_results = evaluate_subject(
            data_path(args.data_dir, subject), confirmation_models, args.folds, args.seed
        )
        print(f"  nested subject-adaptive CSP", flush=True)
        subject_results["nested_subject_adaptive_csp"] = evaluate_nested_subject_adaptive(
            data_path(args.data_dir, subject), args.folds, args.seed
        )
        confirmation[f"S{subject:03d}"] = subject_results
    aggregate = aggregate_confirmation(confirmation)

    print("\nDevelopment subject S001")
    print("-" * 86)
    for name, metrics in sorted(development.items(), key=lambda item: item[1]["accuracy"], reverse=True):
        print(
            f"{name:40s} acc={metrics['accuracy']:.4f} "
            f"bacc={metrics['balanced_accuracy']:.4f} time={metrics['five_fold_seconds']:.2f}s"
        )
    print(f"\nSelected overall: {best_overall}")
    print(f"Selected new model: {best_new}")
    print("\nHeld-out subject confirmation")
    print("-" * 86)
    for name, metrics in sorted(aggregate.items(), key=lambda item: item[1]["mean_accuracy"], reverse=True):
        print(
            f"{name:40s} mean={metrics['mean_accuracy']:.4f} "
            f"std={metrics['std_accuracy']:.4f} kappa={metrics['mean_kappa']:.4f}"
        )

    report = {
        "protocol": {
            "development_subject": f"S{args.development_subject:03d}",
            "confirmation_subjects": [f"S{subject:03d}" for subject in available_subjects],
            "selection_uses_confirmation_subjects": False,
            "within_subject_validation": f"{args.folds}-fold stratified trial-level CV",
            "seed": args.seed,
        },
        "development_results": development,
        "selected_overall": best_overall,
        "selected_new_model": best_new,
        "confirmation_per_subject": confirmation,
        "confirmation_aggregate": aggregate,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
