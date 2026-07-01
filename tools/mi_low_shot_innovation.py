#!/usr/bin/env python
"""Low-shot MI benchmark with similarity-weighted hierarchical CSP priors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mne
import numpy as np
from scipy.linalg import eigvalsh
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arm_bci_system import bandpass, common_average_reference  # noqa: E402
from mi_paths import EXPERIMENTS_DIR, PROCESSED_DATA_DIR  # noqa: E402
from mi_physionet_benchmark import MOTOR_CHANNELS, load_npz, select_channel_indices  # noqa: E402
from mi_transfer_regularized_csp import (  # noqa: E402
    class_mean_covariances,
    csp_log_variance,
    fit_transfer_csp_filters,
    normalized_trial_covariances,
    subject_path,
)


DEFAULT_SHOTS = (2, 4, 6, 8, 12, 16)
DEFAULT_KAPPAS = (1.0, 2.0, 4.0, 8.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--source-subjects", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--development-subject", type=int, default=1)
    parser.add_argument("--confirmation-subjects", type=int, nargs="*", default=list(range(11, 21)))
    parser.add_argument("--shots", type=int, nargs="*", default=list(DEFAULT_SHOTS))
    parser.add_argument("--kappas", type=float, nargs="*", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENTS_DIR / "physionet_mi_low_shot_innovation.json",
    )
    return parser.parse_args()


def binary_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray([0 if str(label) == "left" else 1 for label in labels], dtype=np.int64)


def mean_covariance(covariances: np.ndarray) -> np.ndarray:
    return np.mean(covariances, axis=0)


def affine_invariant_distance(first: np.ndarray, second: np.ndarray) -> float:
    dimension = first.shape[0]
    ridge_first = 1e-7 * np.trace(first) / dimension
    ridge_second = 1e-7 * np.trace(second) / dimension
    values = eigvalsh(
        second + ridge_second * np.eye(dimension),
        first + ridge_first * np.eye(dimension),
        check_finite=False,
    )
    return float(np.linalg.norm(np.log(np.maximum(values, 1e-12))))


def source_similarity_weights(
    target_covariances: np.ndarray,
    source_covariances: Dict[int, np.ndarray],
    source_ids: Sequence[int],
) -> np.ndarray:
    target_mean = mean_covariance(target_covariances)
    squared_distances = np.asarray(
        [
            affine_invariant_distance(target_mean, mean_covariance(source_covariances[sid])) ** 2
            for sid in source_ids
        ],
        dtype=np.float64,
    )
    positive = squared_distances[squared_distances > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    logits = -squared_distances / max(scale, 1e-12)
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def weighted_source_class_covariance(
    source_covariances: Dict[int, np.ndarray],
    source_labels: Dict[int, np.ndarray],
    source_ids: Sequence[int],
    weights: np.ndarray,
) -> np.ndarray:
    subject_means = np.stack(
        [class_mean_covariances(source_covariances[sid], source_labels[sid]) for sid in source_ids]
    )
    return np.sum(subject_means * weights[:, None, None, None], axis=0)


def equal_source_class_covariance(
    source_covariances: Dict[int, np.ndarray],
    source_labels: Dict[int, np.ndarray],
    source_ids: Sequence[int],
) -> np.ndarray:
    weights = np.full(len(source_ids), 1.0 / len(source_ids), dtype=np.float64)
    return weighted_source_class_covariance(
        source_covariances, source_labels, source_ids, weights
    )


def target_lda_prediction(
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


def standardize_from_training(
    train_features: np.ndarray, test_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.mean(train_features, axis=0)
    scale = np.std(train_features, axis=0, ddof=1)
    scale = np.maximum(scale, 1e-6)
    return (train_features - center) / scale, (test_features - center) / scale


def class_prototypes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.stack([np.mean(features[labels == label], axis=0) for label in (0, 1)])


def pooled_within_class_covariance(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    prototypes = class_prototypes(features, labels)
    residuals = np.concatenate(
        [features[labels == label] - prototypes[label] for label in (0, 1)], axis=0
    )
    denominator = max(1, len(features) - 2)
    return residuals.T @ residuals / denominator


def source_feature_statistics(
    filters: np.ndarray,
    source_eeg: Dict[int, np.ndarray],
    source_labels: Dict[int, np.ndarray],
    source_ids: Sequence[int],
    weights: np.ndarray,
    swap_labels: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    prototypes = []
    covariances = []
    for sid in source_ids:
        features = csp_log_variance(source_eeg[sid], filters)
        center = np.mean(features, axis=0)
        scale = np.maximum(np.std(features, axis=0, ddof=1), 1e-6)
        standardized = (features - center) / scale
        labels = 1 - source_labels[sid] if swap_labels else source_labels[sid]
        prototypes.append(class_prototypes(standardized, labels))
        covariances.append(pooled_within_class_covariance(standardized, labels))
    prototype_prior = np.sum(
        np.stack(prototypes) * weights[:, None, None], axis=0
    )
    covariance_prior = np.sum(np.stack(covariances) * weights[:, None, None], axis=0)
    return prototype_prior, covariance_prior


def hierarchical_prediction(
    train_eeg: np.ndarray,
    train_covariances: np.ndarray,
    train_labels: np.ndarray,
    test_eeg: np.ndarray,
    source_eeg: Dict[int, np.ndarray],
    source_labels: Dict[int, np.ndarray],
    source_ids: Sequence[int],
    source_weights: np.ndarray,
    source_prior: np.ndarray,
    kappa: float,
    components: int,
    swap_source_labels: bool = False,
    reliability_gate: bool = False,
) -> np.ndarray:
    shots_per_class = min(np.sum(train_labels == 0), np.sum(train_labels == 1))
    alpha = float(kappa / (shots_per_class + kappa))
    filters = fit_transfer_csp_filters(
        train_covariances, train_labels, source_prior, alpha, components
    )
    target_train = csp_log_variance(train_eeg, filters)
    target_test = csp_log_variance(test_eeg, filters)
    target_train, target_test = standardize_from_training(target_train, target_test)

    source_prototypes, source_covariance = source_feature_statistics(
        filters,
        source_eeg,
        source_labels,
        source_ids,
        source_weights,
        swap_source_labels,
    )
    target_prototypes = class_prototypes(target_train, train_labels)
    target_covariance = pooled_within_class_covariance(target_train, train_labels)
    prototype_kappa = float(kappa)
    if reliability_gate:
        target_direction = target_prototypes[1] - target_prototypes[0]
        source_direction = source_prototypes[1] - source_prototypes[0]
        denominator = np.linalg.norm(target_direction) * np.linalg.norm(source_direction)
        agreement = (
            float(np.dot(target_direction, source_direction) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        reliability = max(0.0, min(1.0, agreement)) ** 2
        prototype_kappa *= reliability
    class_counts = np.asarray([np.sum(train_labels == label) for label in (0, 1)])
    prototypes = (
        class_counts[:, None] * target_prototypes + prototype_kappa * source_prototypes
    ) / (class_counts[:, None] + prototype_kappa)
    source_equivalent_trials = 2.0 * kappa
    covariance = (
        len(target_train) * target_covariance + source_equivalent_trials * source_covariance
    ) / (len(target_train) + source_equivalent_trials)
    ridge = 1e-3 * np.trace(covariance) / covariance.shape[0]
    precision = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
    scores = np.stack(
        [
            target_test @ precision @ prototypes[label]
            - 0.5 * prototypes[label] @ precision @ prototypes[label]
            for label in (0, 1)
        ],
        axis=1,
    )
    return np.argmax(scores, axis=1).astype(np.int64)


def stratified_low_shot_split(
    labels: np.ndarray, shots_per_class: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    for label in (0, 1):
        candidates = np.flatnonzero(labels == label)
        if shots_per_class >= len(candidates):
            raise ValueError(
                f"shots_per_class={shots_per_class} leaves no test samples for class {label}"
            )
        train_parts.append(rng.choice(candidates, size=shots_per_class, replace=False))
    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx, assume_unique=True)
    return train_idx, test_idx


def summarize_repeats(rows: List[Dict[str, float]]) -> Dict[str, float]:
    output = {"repeats": len(rows)}
    for key in ("accuracy", "balanced_accuracy", "kappa", "elapsed_seconds"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[f"mean_{key}"] = float(np.mean(values))
        output[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def evaluate_subject(
    target_id: int,
    source_ids: Sequence[int],
    subject_eeg: Dict[int, np.ndarray],
    subject_covariances: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
    shots_values: Sequence[int],
    repeats: int,
    kappa: float,
    components: int,
    seed: int,
    development_candidates: bool,
    candidate_kappas: Sequence[float],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    eeg = subject_eeg[target_id]
    covariances = subject_covariances[target_id]
    labels = subject_labels[target_id]
    equal_prior = equal_source_class_covariance(
        subject_covariances, subject_labels, source_ids
    )
    result: Dict[str, Dict[str, Dict[str, float]]] = {}

    for shots in shots_values:
        method_rows: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for repeat in range(repeats):
            split_seed = seed + target_id * 100_000 + shots * 1_000 + repeat
            train_idx, test_idx = stratified_low_shot_split(labels, shots, split_seed)
            train_eeg = eeg[train_idx]
            test_eeg = eeg[test_idx]
            train_covariances = covariances[train_idx]
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]
            weights = source_similarity_weights(
                train_covariances, subject_covariances, source_ids
            )
            weighted_prior = weighted_source_class_covariance(
                subject_covariances, subject_labels, source_ids, weights
            )

            def record(name: str, prediction: np.ndarray, elapsed: float) -> None:
                method_rows[name].append(
                    {
                        "accuracy": float(accuracy_score(test_labels, prediction)),
                        "balanced_accuracy": float(
                            balanced_accuracy_score(test_labels, prediction)
                        ),
                        "kappa": float(cohen_kappa_score(test_labels, prediction)),
                        "elapsed_seconds": float(elapsed),
                    }
                )

            started = time.perf_counter()
            prediction = target_lda_prediction(
                train_eeg,
                train_covariances,
                train_labels,
                test_eeg,
                equal_prior,
                0.0,
                components,
            )
            record("target_only_csp_lda", prediction, time.perf_counter() - started)

            started = time.perf_counter()
            prediction = target_lda_prediction(
                train_eeg,
                train_covariances,
                train_labels,
                test_eeg,
                equal_prior,
                0.25,
                components,
            )
            record("equal_fixed_prior_csp_lda", prediction, time.perf_counter() - started)

            kappas = candidate_kappas if development_candidates else (kappa,)
            for current_kappa in kappas:
                suffix = f"k{current_kappa:g}" if development_candidates else ""
                similarity_name = f"similarity_adaptive_csp_lda{suffix}"
                alpha = float(current_kappa / (shots + current_kappa))
                started = time.perf_counter()
                prediction = target_lda_prediction(
                    train_eeg,
                    train_covariances,
                    train_labels,
                    test_eeg,
                    weighted_prior,
                    alpha,
                    components,
                )
                record(similarity_name, prediction, time.perf_counter() - started)

                hierarchical_name = f"similarity_hierarchical_csp{suffix}"
                started = time.perf_counter()
                prediction = hierarchical_prediction(
                    train_eeg,
                    train_covariances,
                    train_labels,
                    test_eeg,
                    subject_eeg,
                    subject_labels,
                    source_ids,
                    weights,
                    weighted_prior,
                    current_kappa,
                    components,
                )
                record(hierarchical_name, prediction, time.perf_counter() - started)

                gated_name = f"reliability_gated_hierarchical_csp{suffix}"
                started = time.perf_counter()
                prediction = hierarchical_prediction(
                    train_eeg,
                    train_covariances,
                    train_labels,
                    test_eeg,
                    subject_eeg,
                    subject_labels,
                    source_ids,
                    weights,
                    weighted_prior,
                    current_kappa,
                    components,
                    reliability_gate=True,
                )
                record(gated_name, prediction, time.perf_counter() - started)

                if not development_candidates:
                    started = time.perf_counter()
                    prediction = hierarchical_prediction(
                        train_eeg,
                        train_covariances,
                        train_labels,
                        test_eeg,
                        subject_eeg,
                        subject_labels,
                        source_ids,
                        weights,
                        weighted_prior,
                        current_kappa,
                        components,
                        swap_source_labels=True,
                    )
                    record(
                        "source_label_swap_negative_control",
                        prediction,
                        time.perf_counter() - started,
                    )

        result[str(shots)] = {
            name: summarize_repeats(rows) for name, rows in sorted(method_rows.items())
        }
    return result


def select_kappa(
    development: Dict[str, Dict[str, Dict[str, float]]],
    candidate_kappas: Sequence[float],
) -> Tuple[float, Dict[str, float]]:
    low_shot_budgets = [key for key in development if int(key) <= 8]
    scores = {}
    for kappa in candidate_kappas:
        name = f"reliability_gated_hierarchical_cspk{kappa:g}"
        scores[str(kappa)] = float(
            np.mean(
                [development[budget][name]["mean_balanced_accuracy"] for budget in low_shot_budgets]
            )
        )
    selected = max(candidate_kappas, key=lambda value: (scores[str(value)], -value))
    return float(selected), scores


def aggregate_confirmation(
    confirmation: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    shots_values: Sequence[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output = {}
    for shots in shots_values:
        key = str(shots)
        model_names = sorted(next(iter(confirmation.values()))[key])
        output[key] = {}
        for name in model_names:
            accuracy = np.asarray(
                [subject[key][name]["mean_accuracy"] for subject in confirmation.values()]
            )
            balanced = np.asarray(
                [
                    subject[key][name]["mean_balanced_accuracy"]
                    for subject in confirmation.values()
                ]
            )
            kappa = np.asarray(
                [subject[key][name]["mean_kappa"] for subject in confirmation.values()]
            )
            elapsed = np.asarray(
                [subject[key][name]["mean_elapsed_seconds"] for subject in confirmation.values()]
            )
            output[key][name] = {
                "subject_count": len(accuracy),
                "mean_accuracy": float(np.mean(accuracy)),
                "std_accuracy_across_subjects": (
                    float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else 0.0
                ),
                "mean_balanced_accuracy": float(np.mean(balanced)),
                "mean_kappa": float(np.mean(kappa)),
                "mean_fit_predict_seconds": float(np.mean(elapsed)),
                "subject_mean_accuracies": accuracy.tolist(),
                "subject_mean_balanced_accuracies": balanced.tolist(),
            }
    return output


def paired_model_tests(
    aggregate: Dict[str, Dict[str, Dict[str, float]]],
    shots_values: Sequence[int],
    model: str,
    baselines: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    output = {}
    shot_keys = [str(shots) for shots in shots_values]
    model_by_shot = {
        key: np.asarray(aggregate[key][model]["subject_mean_balanced_accuracies"])
        for key in shot_keys
    }
    for baseline in baselines:
        baseline_by_shot = {
            key: np.asarray(aggregate[key][baseline]["subject_mean_balanced_accuracies"])
            for key in shot_keys
        }
        per_budget = {}
        for key in shot_keys:
            difference = model_by_shot[key] - baseline_by_shot[key]
            test = wilcoxon(model_by_shot[key], baseline_by_shot[key])
            per_budget[key] = {
                "mean_difference_percentage_points": float(100.0 * np.mean(difference)),
                "wilcoxon_statistic": float(test.statistic),
                "two_sided_p_value": float(test.pvalue),
            }
        model_subject_mean = np.mean(np.stack(list(model_by_shot.values())), axis=0)
        baseline_subject_mean = np.mean(np.stack(list(baseline_by_shot.values())), axis=0)
        overall_test = wilcoxon(model_subject_mean, baseline_subject_mean)
        output[baseline] = {
            "per_budget": per_budget,
            "overall_across_budgets": {
                "mean_difference_percentage_points": float(
                    100.0 * np.mean(model_subject_mean - baseline_subject_mean)
                ),
                "wilcoxon_statistic": float(overall_test.statistic),
                "two_sided_p_value": float(overall_test.pvalue),
                "subject_count": int(len(model_subject_mean)),
            },
        }
    return output


def load_preprocessed_subjects(
    data_dir: Path, subjects: Iterable[int]
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], List[str], int]:
    subject_eeg = {}
    subject_covariances = {}
    subject_labels = {}
    reference_names: List[str] = []
    reference_fs = 0
    for subject in sorted(set(subjects)):
        eeg, raw_labels, fs, channel_names, _ = load_npz(subject_path(data_dir, subject))
        indices, selected_names = select_channel_indices(channel_names, MOTOR_CHANNELS)
        if not reference_names:
            reference_names = selected_names
            reference_fs = fs
        elif selected_names != reference_names or fs != reference_fs:
            raise ValueError(f"S{subject:03d} channel order or sampling rate differs")
        prepared = bandpass(common_average_reference(eeg), fs, 8.0, 30.0)[:, indices, :]
        subject_eeg[subject] = np.asarray(prepared, dtype=np.float64)
        subject_covariances[subject] = normalized_trial_covariances(subject_eeg[subject])
        subject_labels[subject] = binary_labels(raw_labels)
    return subject_eeg, subject_covariances, subject_labels, reference_names, reference_fs


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")
    all_subjects = set(args.source_subjects) | {args.development_subject} | set(
        args.confirmation_subjects
    )
    missing = [
        subject for subject in sorted(all_subjects) if not subject_path(args.data_dir, subject).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing subject NPZ files: {missing}")
    if args.development_subject not in args.source_subjects:
        raise ValueError("Development subject must belong to the source bank")
    overlap = set(args.source_subjects) & set(args.confirmation_subjects)
    if overlap:
        raise ValueError(f"Confirmation subjects must be unseen in the source bank: {sorted(overlap)}")

    subject_eeg, subject_covariances, subject_labels, channels, fs = load_preprocessed_subjects(
        args.data_dir, all_subjects
    )
    development_sources = [
        subject for subject in args.source_subjects if subject != args.development_subject
    ]
    print(
        f"Development S{args.development_subject:03d}; sources={development_sources}; "
        f"shots={args.shots}; repeats={args.repeats}",
        flush=True,
    )
    development = evaluate_subject(
        args.development_subject,
        development_sources,
        subject_eeg,
        subject_covariances,
        subject_labels,
        args.shots,
        args.repeats,
        args.kappas[0],
        args.components,
        args.seed,
        development_candidates=True,
        candidate_kappas=args.kappas,
    )
    selected_kappa, kappa_scores = select_kappa(development, args.kappas)
    print(f"Selected prior-equivalent shots kappa={selected_kappa:g}: {kappa_scores}", flush=True)

    confirmation = {}
    for subject in args.confirmation_subjects:
        print(f"Confirmation target S{subject:03d}", flush=True)
        confirmation[f"S{subject:03d}"] = evaluate_subject(
            subject,
            args.source_subjects,
            subject_eeg,
            subject_covariances,
            subject_labels,
            args.shots,
            args.repeats,
            selected_kappa,
            args.components,
            args.seed,
            development_candidates=False,
            candidate_kappas=(selected_kappa,),
        )
    aggregate = aggregate_confirmation(confirmation, args.shots)
    statistical_tests = paired_model_tests(
        aggregate,
        args.shots,
        "similarity_hierarchical_csp",
        (
            "target_only_csp_lda",
            "equal_fixed_prior_csp_lda",
            "similarity_adaptive_csp_lda",
            "source_label_swap_negative_control",
            "reliability_gated_hierarchical_csp",
        ),
    )

    print("\nStrict unseen-subject low-shot confirmation")
    print("-" * 104)
    for shots in args.shots:
        rows = aggregate[str(shots)]
        best = max(rows, key=lambda name: rows[name]["mean_balanced_accuracy"])
        values = " | ".join(
            f"{name}={metrics['mean_balanced_accuracy']:.3f}"
            for name, metrics in sorted(rows.items())
        )
        print(f"shots/class={shots:2d} best={best}: {values}")

    report = {
        "protocol": {
            "dataset": "PhysioNet EEGMMIDB runs 4/8/12, left/right fist motor imagery",
            "source_bank_subjects": [f"S{subject:03d}" for subject in args.source_subjects],
            "development_subject": f"S{args.development_subject:03d}",
            "development_source_subjects": [f"S{subject:03d}" for subject in development_sources],
            "confirmation_subjects": [
                f"S{subject:03d}" for subject in args.confirmation_subjects
            ],
            "confirmation_subjects_used_in_source_bank_or_model_selection": False,
            "shots_per_class": args.shots,
            "repeated_stratified_subsampling": args.repeats,
            "test_set": "all remaining target-subject trials after calibration sampling",
            "frequency_band_hz": [8.0, 30.0],
            "channels": channels,
            "sampling_rate_hz": fs,
            "csp_components": args.components,
            "seed": args.seed,
        },
        "innovation": {
            "name": "similarity-weighted adaptive hierarchical CSP",
            "source_weight": "affine-invariant covariance similarity to target calibration data",
            "adaptive_spatial_prior": "alpha=kappa/(shots_per_class+kappa)",
            "feature_prior": "target class prototypes and covariance shrink toward weighted source statistics",
            "feature_reliability_gate": (
                "squared positive cosine agreement between target and source class-difference vectors"
            ),
            "selected_prior_equivalent_shots_kappa": selected_kappa,
            "development_kappa_scores_low_shot_2_to_8": kappa_scores,
        },
        "development_results": development,
        "confirmation_per_subject": confirmation,
        "confirmation_aggregate": aggregate,
        "paired_statistical_tests": statistical_tests,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
