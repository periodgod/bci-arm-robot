#!/usr/bin/env python
"""Leakage-safe cross-subject covariance-prior CSP for small-sample MI."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import mne
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arm_bci_system import bandpass, common_average_reference  # noqa: E402
from mi_paths import EXPERIMENTS_DIR, PROCESSED_DATA_DIR, physionet_subject_path  # noqa: E402
from mi_physionet_benchmark import MOTOR_CHANNELS, load_npz, select_channel_indices  # noqa: E402


ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENTS_DIR / "physionet_mi_transfer_regularized_csp.json",
    )
    return parser.parse_args()


def subject_path(data_dir: Path, subject: int) -> Path:
    return physionet_subject_path(data_dir, subject)


def binary_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray([0 if str(label) == "left" else 1 for label in labels], dtype=np.int64)


def normalized_trial_covariances(eeg: np.ndarray) -> np.ndarray:
    centered = eeg - np.mean(eeg, axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->ncd", centered, centered, optimize=True)
    trace = np.trace(covariance, axis1=1, axis2=2)
    return covariance / np.maximum(trace[:, None, None], np.finfo(float).eps)


def class_mean_covariances(covariances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack([np.mean(covariances[labels == label], axis=0) for label in (0, 1)])


def euclidean_alignment_matrix(reference_covariances: np.ndarray) -> np.ndarray:
    reference = np.mean(reference_covariances, axis=0)
    ridge = 1e-7 * np.trace(reference) / reference.shape[0]
    eigenvalues, eigenvectors = eigh(
        reference + ridge * np.eye(reference.shape[0]), check_finite=False
    )
    inverse_root = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12)))
    return eigenvectors @ inverse_root @ eigenvectors.T


def align_eeg(eeg: np.ndarray, reference_covariances: np.ndarray) -> np.ndarray:
    alignment = euclidean_alignment_matrix(reference_covariances)
    return np.einsum("cd,ndt->nct", alignment, eeg, optimize=True)


def equal_subject_source_prior(
    subject_covariances: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
    target_subject: int,
) -> np.ndarray:
    subject_means = [
        class_mean_covariances(subject_covariances[subject], subject_labels[subject])
        for subject in sorted(subject_covariances)
        if subject != target_subject
    ]
    if not subject_means:
        raise ValueError("At least one non-target subject is required for the covariance prior")
    return np.mean(subject_means, axis=0)


def fit_transfer_csp_filters(
    target_covariances: np.ndarray,
    target_labels: np.ndarray,
    source_prior: np.ndarray,
    alpha: float,
    components: int,
) -> np.ndarray:
    target_means = class_mean_covariances(target_covariances, target_labels)
    class_covariances = (1.0 - alpha) * target_means + alpha * source_prior
    composite = class_covariances[0] + class_covariances[1]
    ridge = 1e-7 * np.trace(composite) / composite.shape[0]
    composite = composite + ridge * np.eye(composite.shape[0])
    eigenvalues, eigenvectors = eigh(class_covariances[1], composite, check_finite=False)
    order = np.argsort(eigenvalues)
    side = min(components // 2, len(order) // 2)
    selected = np.concatenate((order[:side], order[-side:]))
    return eigenvectors[:, selected].T


def csp_log_variance(eeg: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = np.einsum("kc,nct->nkt", filters, eeg, optimize=True)
    variance = np.var(projected, axis=-1)
    normalized = variance / np.maximum(np.sum(variance, axis=1, keepdims=True), 1e-12)
    return np.log(np.maximum(normalized, 1e-12))


def fit_predict(
    train_eeg: np.ndarray,
    train_covariances: np.ndarray,
    train_labels: np.ndarray,
    test_eeg: np.ndarray,
    source_prior: np.ndarray,
    alpha: float,
    components: int,
) -> np.ndarray:
    filters = fit_transfer_csp_filters(
        train_covariances, train_labels, source_prior, alpha, components
    )
    train_features = csp_log_variance(train_eeg, filters)
    test_features = csp_log_variance(test_eeg, filters)
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(train_features, train_labels)
    return classifier.predict(test_features).astype(np.int64)


def select_alpha(
    train_eeg: np.ndarray,
    train_covariances: np.ndarray,
    train_labels: np.ndarray,
    source_prior: np.ndarray,
    components: int,
    folds: int,
    seed: int,
) -> Tuple[float, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {}
    for alpha in ALPHAS:
        fold_scores = []
        for fit_idx, validation_idx in splitter.split(train_eeg, train_labels):
            prediction = fit_predict(
                train_eeg[fit_idx],
                train_covariances[fit_idx],
                train_labels[fit_idx],
                train_eeg[validation_idx],
                source_prior,
                alpha,
                components,
            )
            fold_scores.append(
                balanced_accuracy_score(train_labels[validation_idx], prediction)
            )
        scores[alpha] = float(np.mean(fold_scores))
    selected = max(ALPHAS, key=lambda alpha: (scores[alpha], -alpha))
    return selected, scores[selected]


def summarize(labels: np.ndarray, prediction: np.ndarray, elapsed: float) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
        "five_fold_seconds": float(elapsed),
        "milliseconds_per_trial_cv": float(1000.0 * elapsed / len(labels)),
    }


def evaluate_target(
    target: int,
    subject_eeg: Dict[int, np.ndarray],
    subject_covariances: Dict[int, np.ndarray],
    aligned_subject_covariances: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
    folds: int,
    inner_folds: int,
    components: int,
    seed: int,
) -> Dict[str, object]:
    eeg = subject_eeg[target]
    covariances = subject_covariances[target]
    labels = subject_labels[target]
    source_prior = equal_subject_source_prior(subject_covariances, subject_labels, target)
    aligned_source_prior = equal_subject_source_prior(
        aligned_subject_covariances, subject_labels, target
    )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = {f"transfer_alpha_{alpha:.2f}": np.full(len(labels), -1) for alpha in ALPHAS}
    predictions.update(
        {
            f"aligned_transfer_alpha_{alpha:.2f}": np.full(len(labels), -1)
            for alpha in ALPHAS
        }
    )
    predictions["nested_transfer_csp"] = np.full(len(labels), -1)
    elapsed = {name: 0.0 for name in predictions}
    selections = []
    inner_scores = []

    for outer_fold, (train_idx, test_idx) in enumerate(splitter.split(eeg, labels)):
        aligned_all = align_eeg(eeg, covariances[train_idx])
        aligned_covariances = normalized_trial_covariances(aligned_all)
        for alpha in ALPHAS:
            name = f"transfer_alpha_{alpha:.2f}"
            started = time.perf_counter()
            predictions[name][test_idx] = fit_predict(
                eeg[train_idx],
                covariances[train_idx],
                labels[train_idx],
                eeg[test_idx],
                source_prior,
                alpha,
                components,
            )
            elapsed[name] += time.perf_counter() - started

            aligned_name = f"aligned_transfer_alpha_{alpha:.2f}"
            started = time.perf_counter()
            predictions[aligned_name][test_idx] = fit_predict(
                aligned_all[train_idx],
                aligned_covariances[train_idx],
                labels[train_idx],
                aligned_all[test_idx],
                aligned_source_prior,
                alpha,
                components,
            )
            elapsed[aligned_name] += time.perf_counter() - started

        started = time.perf_counter()
        selected, inner_score = select_alpha(
            eeg[train_idx],
            covariances[train_idx],
            labels[train_idx],
            source_prior,
            components,
            inner_folds,
            seed + outer_fold,
        )
        predictions["nested_transfer_csp"][test_idx] = fit_predict(
            eeg[train_idx],
            covariances[train_idx],
            labels[train_idx],
            eeg[test_idx],
            source_prior,
            selected,
            components,
        )
        elapsed["nested_transfer_csp"] += time.perf_counter() - started
        selections.append(selected)
        inner_scores.append(inner_score)

    results = {
        name: summarize(labels, prediction.astype(np.int64), elapsed[name])
        for name, prediction in predictions.items()
    }
    results["nested_transfer_csp"].update(
        {
            "outer_fold_selected_alphas": selections,
            "alpha_selection_frequency": {
                str(alpha): count for alpha, count in Counter(selections).items()
            },
            "mean_inner_balanced_accuracy": float(np.mean(inner_scores)),
        }
    )
    return results


def aggregate(
    per_subject: Dict[str, Dict[str, object]],
    subject_keys: Sequence[str] | None = None,
) -> Dict[str, object]:
    selected_rows = (
        per_subject
        if subject_keys is None
        else {key: per_subject[key] for key in subject_keys}
    )
    model_names = sorted(next(iter(selected_rows.values())).keys())
    output = {}
    for name in model_names:
        accuracy = np.asarray([row[name]["accuracy"] for row in selected_rows.values()])
        balanced = np.asarray(
            [row[name]["balanced_accuracy"] for row in selected_rows.values()]
        )
        kappa = np.asarray([row[name]["kappa"] for row in selected_rows.values()])
        milliseconds = np.asarray(
            [row[name]["milliseconds_per_trial_cv"] for row in selected_rows.values()]
        )
        output[name] = {
            "mean_accuracy": float(np.mean(accuracy)),
            "std_accuracy": float(np.std(accuracy, ddof=1)),
            "mean_balanced_accuracy": float(np.mean(balanced)),
            "mean_kappa": float(np.mean(kappa)),
            "mean_milliseconds_per_trial_cv": float(np.mean(milliseconds)),
            "subject_accuracies": accuracy.tolist(),
        }
    return output


def benchmark_prefiltered_inference(
    target_subject: int,
    subject_eeg: Dict[int, np.ndarray],
    subject_covariances: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
    alpha: float,
    components: int,
    repetitions: int = 1000,
) -> Dict[str, float]:
    eeg = subject_eeg[target_subject]
    covariances = subject_covariances[target_subject]
    labels = subject_labels[target_subject]
    source_prior = equal_subject_source_prior(
        subject_covariances, subject_labels, target_subject
    )
    started = time.perf_counter()
    filters = fit_transfer_csp_filters(
        covariances, labels, source_prior, alpha, components
    )
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(csp_log_variance(eeg, filters), labels)
    training_seconds = time.perf_counter() - started

    trial = eeg[:1]
    for _ in range(20):
        classifier.predict(csp_log_variance(trial, filters))
    latency_ms = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        classifier.predict(csp_log_variance(trial, filters))
        latency_ms.append((time.perf_counter_ns() - started) / 1e6)
    latency = np.asarray(latency_ms)
    return {
        "training_seconds_prefiltered": float(training_seconds),
        "median_inference_ms_prefiltered": float(np.median(latency)),
        "p95_inference_ms_prefiltered": float(np.percentile(latency, 95)),
        "repetitions": repetitions,
        "excludes_streaming_bandpass": True,
    }


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")
    available = [subject for subject in args.subjects if subject_path(args.data_dir, subject).exists()]
    if len(available) < 2:
        raise ValueError("At least two subject files are required")

    subject_eeg = {}
    subject_covariances = {}
    aligned_subject_covariances = {}
    subject_labels = {}
    reference_channels: Sequence[str] | None = None
    sampling_rate = None
    for subject in available:
        eeg, raw_labels, fs, channel_names, _ = load_npz(subject_path(args.data_dir, subject))
        channel_idx, selected_names = select_channel_indices(channel_names, MOTOR_CHANNELS)
        if reference_channels is None:
            reference_channels = selected_names
            sampling_rate = fs
        elif selected_names != list(reference_channels) or fs != sampling_rate:
            raise ValueError(f"S{subject:03d} channel order or sampling rate differs")
        prepared = bandpass(common_average_reference(eeg), fs, 8.0, 30.0)[:, channel_idx, :]
        subject_eeg[subject] = np.asarray(prepared, dtype=np.float64)
        subject_covariances[subject] = normalized_trial_covariances(subject_eeg[subject])
        source_aligned_eeg = align_eeg(subject_eeg[subject], subject_covariances[subject])
        aligned_subject_covariances[subject] = normalized_trial_covariances(source_aligned_eeg)
        subject_labels[subject] = binary_labels(raw_labels)

    per_subject = {}
    for subject in available:
        print(f"Target S{subject:03d}; source subjects={len(available) - 1}", flush=True)
        per_subject[f"S{subject:03d}"] = evaluate_target(
            subject,
            subject_eeg,
            subject_covariances,
            aligned_subject_covariances,
            subject_labels,
            args.folds,
            args.inner_folds,
            args.components,
            args.seed,
        )

    summary = aggregate(per_subject)
    development_key = f"S{available[0]:03d}"
    confirmation_keys = [f"S{subject:03d}" for subject in available[1:]]
    fixed_alpha_models = [f"transfer_alpha_{alpha:.2f}" for alpha in ALPHAS]
    aligned_fixed_alpha_models = [
        f"aligned_transfer_alpha_{alpha:.2f}" for alpha in ALPHAS
    ]
    selected_fixed_alpha_model = max(
        fixed_alpha_models,
        key=lambda name: per_subject[development_key][name]["balanced_accuracy"],
    )
    selected_aligned_fixed_alpha_model = max(
        aligned_fixed_alpha_models,
        key=lambda name: per_subject[development_key][name]["balanced_accuracy"],
    )
    confirmation_summary = aggregate(per_subject, confirmation_keys)
    selected_alpha = float(selected_fixed_alpha_model.rsplit("_", 1)[-1])
    deployment_benchmark = benchmark_prefiltered_inference(
        available[0],
        subject_eeg,
        subject_covariances,
        subject_labels,
        selected_alpha,
        args.components,
    )
    print("\nCross-subject covariance-prior CSP")
    print("-" * 90)
    for name, metrics in sorted(summary.items(), key=lambda item: item[1]["mean_accuracy"], reverse=True):
        print(
            f"{name:28s} mean={metrics['mean_accuracy']:.4f} "
            f"std={metrics['std_accuracy']:.4f} kappa={metrics['mean_kappa']:.4f} "
            f"ms/trial-cv={metrics['mean_milliseconds_per_trial_cv']:.2f}"
        )
    print(
        f"\nDevelopment {development_key} selected: {selected_fixed_alpha_model}; "
        f"confirmation mean={confirmation_summary[selected_fixed_alpha_model]['mean_accuracy']:.4f}"
    )
    print(
        f"Alignment ablation selected: {selected_aligned_fixed_alpha_model}; "
        f"confirmation mean="
        f"{confirmation_summary[selected_aligned_fixed_alpha_model]['mean_accuracy']:.4f}"
    )

    report = {
        "protocol": {
            "dataset": "PhysioNet EEGMMIDB left/right fist motor imagery, runs 4/8/12",
            "subjects": [f"S{subject:03d}" for subject in available],
            "target_validation": f"{args.folds}-fold stratified trial-level CV",
            "alpha_selection": f"{args.inner_folds}-fold CV inside each outer target-training fold",
            "source_prior": "equal-weight class covariance mean from all non-target subjects",
            "target_test_trials_used_for_training_or_selection": False,
            "development_subject": development_key,
            "confirmation_subjects": confirmation_keys,
            "fixed_alpha_selection_uses_confirmation_subjects": False,
            "frequency_band_hz": [8.0, 30.0],
            "channels": list(reference_channels or []),
            "csp_components": args.components,
            "seed": args.seed,
        },
        "per_subject": per_subject,
        "aggregate": summary,
        "selected_fixed_alpha_model": selected_fixed_alpha_model,
        "selected_aligned_fixed_alpha_model": selected_aligned_fixed_alpha_model,
        "confirmation_aggregate": confirmation_summary,
        "deployment_benchmark": deployment_benchmark,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
