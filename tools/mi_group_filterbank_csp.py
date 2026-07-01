#!/usr/bin/env python
"""Reflection-group filter-bank CSP with hierarchical Fisher feature selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mne
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arm_bci_system import bandpass, common_average_reference  # noqa: E402
from mi_group_graph_csp import fit_group_graph_filters, reflection_matrix  # noqa: E402
from mi_paths import EXPERIMENTS_DIR, PROCESSED_DATA_DIR  # noqa: E402
from mi_low_shot_innovation import (  # noqa: E402
    class_prototypes,
    pooled_within_class_covariance,
    source_similarity_weights,
    standardize_from_training,
    stratified_low_shot_split,
    weighted_source_class_covariance,
)
from mi_physionet_benchmark import MOTOR_CHANNELS, load_npz, select_channel_indices  # noqa: E402
from mi_transfer_regularized_csp import (  # noqa: E402
    csp_log_variance,
    normalized_trial_covariances,
    subject_path,
)


BROAD_BAND = (8.0, 30.0)
SUB_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 30.0))
DEFAULT_SHOTS = (2, 4, 6, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--source-subjects", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--selection-subjects", type=int, nargs="*", default=list(range(11, 21)))
    parser.add_argument("--confirmation-subjects", type=int, nargs="*", default=list(range(21, 31)))
    parser.add_argument("--shots", type=int, nargs="*", default=list(DEFAULT_SHOTS))
    parser.add_argument("--selection-shots", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--selection-repeats", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--etas", type=float, nargs="*", default=[0.5, 0.75, 1.0])
    parser.add_argument("--feature-counts", type=int, nargs="*", default=[4, 6, 8, 10])
    parser.add_argument("--kappa", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENTS_DIR / "physionet_mi_group_filterbank_csp.json",
    )
    return parser.parse_args()


def binary_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray([0 if str(label) == "left" else 1 for label in labels], dtype=np.int64)


def load_filterbank_subjects(
    data_dir: Path,
    subjects: Iterable[int],
) -> Tuple[
    Dict[Tuple[float, float], Dict[int, np.ndarray]],
    Dict[Tuple[float, float], Dict[int, np.ndarray]],
    Dict[int, np.ndarray],
    List[str],
    int,
]:
    bands = (BROAD_BAND,) + SUB_BANDS
    eeg_by_band = {band: {} for band in bands}
    covariance_by_band = {band: {} for band in bands}
    subject_labels = {}
    reference_channels: List[str] = []
    reference_fs = 0
    for subject in sorted(set(subjects)):
        eeg, raw_labels, fs, channel_names, _ = load_npz(subject_path(data_dir, subject))
        indices, selected_names = select_channel_indices(channel_names, MOTOR_CHANNELS)
        if not reference_channels:
            reference_channels = selected_names
            reference_fs = fs
        elif selected_names != reference_channels or fs != reference_fs:
            raise ValueError(f"S{subject:03d} channel order or sampling rate differs")
        motor_eeg = common_average_reference(eeg)[:, indices, :]
        for band in bands:
            prepared = bandpass(motor_eeg, fs, band[0], band[1]).astype(np.float32)
            eeg_by_band[band][subject] = prepared
            covariance_by_band[band][subject] = normalized_trial_covariances(prepared)
        subject_labels[subject] = binary_labels(raw_labels)
        print(f"Prepared filter bank S{subject:03d}", flush=True)
    return eeg_by_band, covariance_by_band, subject_labels, reference_channels, reference_fs


def fit_band_filters(
    train_covariances: Dict[Tuple[float, float], np.ndarray],
    train_labels: np.ndarray,
    source_priors: Dict[Tuple[float, float], np.ndarray],
    alpha: float,
    reflection: np.ndarray,
    bands: Sequence[Tuple[float, float]],
    components_per_band: int,
    eta: float,
) -> Dict[Tuple[float, float], np.ndarray]:
    zeros = np.zeros_like(reflection)
    return {
        band: fit_group_graph_filters(
            train_covariances[band],
            train_labels,
            source_priors[band],
            alpha,
            components_per_band,
            reflection,
            zeros,
            eta,
            0.0,
        )
        for band in bands
    }


def concatenate_features(
    eeg_by_band: Dict[Tuple[float, float], np.ndarray],
    filters: Dict[Tuple[float, float], np.ndarray],
    bands: Sequence[Tuple[float, float]],
) -> np.ndarray:
    return np.concatenate(
        [csp_log_variance(eeg_by_band[band], filters[band]) for band in bands], axis=1
    )


def fisher_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    first = features[labels == 0]
    second = features[labels == 1]
    difference = np.mean(second, axis=0) - np.mean(first, axis=0)
    within = np.var(first, axis=0, ddof=1) + np.var(second, axis=0, ddof=1)
    return difference * difference / np.maximum(within, 1e-6)


def source_statistics_and_fisher(
    filters: Dict[Tuple[float, float], np.ndarray],
    source_eeg: Dict[Tuple[float, float], Dict[int, np.ndarray]],
    source_labels: Dict[int, np.ndarray],
    source_ids: Sequence[int],
    source_weights: np.ndarray,
    bands: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prototypes = []
    covariances = []
    scores = []
    for source in source_ids:
        features = concatenate_features(
            {band: source_eeg[band][source] for band in bands}, filters, bands
        )
        center = np.mean(features, axis=0)
        scale = np.maximum(np.std(features, axis=0, ddof=1), 1e-6)
        standardized = (features - center) / scale
        labels = source_labels[source]
        prototypes.append(class_prototypes(standardized, labels))
        covariances.append(pooled_within_class_covariance(standardized, labels))
        scores.append(fisher_scores(standardized, labels))
    return (
        np.sum(np.stack(prototypes) * source_weights[:, None, None], axis=0),
        np.sum(np.stack(covariances) * source_weights[:, None, None], axis=0),
        np.sum(np.stack(scores) * source_weights[:, None], axis=0),
    )


def predict_filterbank(
    train_eeg: Dict[Tuple[float, float], np.ndarray],
    train_covariances: Dict[Tuple[float, float], np.ndarray],
    train_labels: np.ndarray,
    test_eeg: Dict[Tuple[float, float], np.ndarray],
    source_eeg: Dict[Tuple[float, float], Dict[int, np.ndarray]],
    source_covariances: Dict[Tuple[float, float], Dict[int, np.ndarray]],
    source_labels: Dict[int, np.ndarray],
    source_ids: Sequence[int],
    source_weights: np.ndarray,
    kappa: float,
    reflection: np.ndarray,
    bands: Sequence[Tuple[float, float]],
    components_per_band: int,
    eta: float,
    feature_count: int | None,
    blend_target_fisher: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    shots = min(np.sum(train_labels == 0), np.sum(train_labels == 1))
    alpha = float(kappa / (shots + kappa))
    source_priors = {
        band: weighted_source_class_covariance(
            source_covariances[band], source_labels, source_ids, source_weights
        )
        for band in bands
    }
    filters = fit_band_filters(
        train_covariances,
        train_labels,
        source_priors,
        alpha,
        reflection,
        bands,
        components_per_band,
        eta,
    )
    target_train = concatenate_features(train_eeg, filters, bands)
    target_test = concatenate_features(test_eeg, filters, bands)
    target_train, target_test = standardize_from_training(target_train, target_test)
    source_prototypes, source_covariance, source_fisher = source_statistics_and_fisher(
        filters,
        source_eeg,
        source_labels,
        source_ids,
        source_weights,
        bands,
    )

    selected = np.arange(target_train.shape[1])
    if feature_count is not None and feature_count < target_train.shape[1]:
        if blend_target_fisher:
            target_weight = float(shots / (shots + kappa))
            scores = (
                target_weight * fisher_scores(target_train, train_labels)
                + (1.0 - target_weight) * source_fisher
            )
        else:
            scores = source_fisher
        selected = np.sort(np.argsort(scores)[-feature_count:])
    target_train = target_train[:, selected]
    target_test = target_test[:, selected]
    source_prototypes = source_prototypes[:, selected]
    source_covariance = source_covariance[np.ix_(selected, selected)]

    target_prototypes = class_prototypes(target_train, train_labels)
    target_covariance = pooled_within_class_covariance(target_train, train_labels)
    counts = np.asarray([np.sum(train_labels == label) for label in (0, 1)])
    prototypes = (
        counts[:, None] * target_prototypes + kappa * source_prototypes
    ) / (counts[:, None] + kappa)
    source_equivalent_trials = 2.0 * kappa
    covariance = (
        len(target_train) * target_covariance + source_equivalent_trials * source_covariance
    ) / (len(target_train) + source_equivalent_trials)
    ridge = 1e-3 * np.trace(covariance) / covariance.shape[0]
    precision = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
    discriminants = np.stack(
        [
            target_test @ precision @ prototypes[label]
            - 0.5 * prototypes[label] @ precision @ prototypes[label]
            for label in (0, 1)
        ],
        axis=1,
    )
    return np.argmax(discriminants, axis=1).astype(np.int64), selected


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    output: Dict[str, object] = {"evaluations": len(rows)}
    for key in ("accuracy", "balanced_accuracy", "kappa", "elapsed_seconds"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[f"mean_{key}"] = float(np.mean(values))
        output[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def evaluate(
    target_ids: Sequence[int],
    source_ids: Sequence[int],
    eeg_by_band: Dict[Tuple[float, float], Dict[int, np.ndarray]],
    covariance_by_band: Dict[Tuple[float, float], Dict[int, np.ndarray]],
    subject_labels: Dict[int, np.ndarray],
    shots_values: Sequence[int],
    repeats: int,
    kappa: float,
    seed: int,
    reflection: np.ndarray,
    bands: Sequence[Tuple[float, float]],
    components_per_band: int,
    eta: float,
    feature_count: int | None,
    blend_target_fisher: bool,
) -> Dict[str, object]:
    all_rows: List[Dict[str, float]] = []
    by_shots: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    by_subject: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    feature_counter: Counter[int] = Counter()
    for target in target_ids:
        labels = subject_labels[target]
        broad_covariances = covariance_by_band[BROAD_BAND][target]
        for shots in shots_values:
            for repeat in range(repeats):
                split_seed = seed + target * 100_000 + shots * 1_000 + repeat
                train_idx, test_idx = stratified_low_shot_split(labels, shots, split_seed)
                weights = source_similarity_weights(
                    broad_covariances[train_idx], covariance_by_band[BROAD_BAND], source_ids
                )
                started = time.perf_counter()
                prediction, selected = predict_filterbank(
                    {band: eeg_by_band[band][target][train_idx] for band in bands},
                    {band: covariance_by_band[band][target][train_idx] for band in bands},
                    labels[train_idx],
                    {band: eeg_by_band[band][target][test_idx] for band in bands},
                    eeg_by_band,
                    covariance_by_band,
                    subject_labels,
                    source_ids,
                    weights,
                    kappa,
                    reflection,
                    bands,
                    components_per_band,
                    eta,
                    feature_count,
                    blend_target_fisher,
                )
                elapsed = time.perf_counter() - started
                row = {
                    "accuracy": float(accuracy_score(labels[test_idx], prediction)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(labels[test_idx], prediction)
                    ),
                    "kappa": float(cohen_kappa_score(labels[test_idx], prediction)),
                    "elapsed_seconds": float(elapsed),
                }
                all_rows.append(row)
                by_shots[str(shots)].append(row)
                by_subject[f"S{target:03d}"].append(row)
                feature_counter.update(map(int, selected))
    feature_labels = [
        f"{low:g}-{high:g}Hz/csp{component + 1}"
        for low, high in bands
        for component in range(components_per_band)
    ]
    return {
        "overall": summarize(all_rows),
        "by_shots": {key: summarize(value) for key, value in by_shots.items()},
        "by_subject": {key: summarize(value) for key, value in by_subject.items()},
        "feature_selection_frequency": {
            feature_labels[index]: count for index, count in sorted(feature_counter.items())
        },
    }


def select_parameters(
    args: argparse.Namespace,
    eeg_by_band: Dict[Tuple[float, float], Dict[int, np.ndarray]],
    covariance_by_band: Dict[Tuple[float, float], Dict[int, np.ndarray]],
    subject_labels: Dict[int, np.ndarray],
    reflection: np.ndarray,
) -> Tuple[float, int, Dict[str, float]]:
    scores = {}
    for eta in args.etas:
        for feature_count in args.feature_counts:
            result = evaluate(
                args.selection_subjects,
                args.source_subjects,
                eeg_by_band,
                covariance_by_band,
                subject_labels,
                args.selection_shots,
                args.selection_repeats,
                args.kappa,
                args.seed,
                reflection,
                SUB_BANDS,
                2,
                eta,
                feature_count,
                True,
            )
            key = f"eta={eta:g},features={feature_count}"
            scores[key] = float(result["overall"]["mean_balanced_accuracy"])
            print(f"Selection {key}: bacc={scores[key]:.4f}", flush=True)
    selected_eta, selected_count = max(
        ((eta, count) for eta in args.etas for count in args.feature_counts),
        key=lambda pair: (
            scores[f"eta={pair[0]:g},features={pair[1]}"],
            -pair[1],
            -pair[0],
        ),
    )
    return float(selected_eta), int(selected_count), scores


def paired_tests(results: Dict[str, Dict[str, object]], proposed: str) -> Dict[str, object]:
    subjects = sorted(results[proposed]["by_subject"])
    proposed_values = np.asarray(
        [results[proposed]["by_subject"][sid]["mean_balanced_accuracy"] for sid in subjects]
    )
    output = {}
    for name, result in results.items():
        if name == proposed:
            continue
        baseline = np.asarray(
            [result["by_subject"][sid]["mean_balanced_accuracy"] for sid in subjects]
        )
        if len(subjects) >= 2:
            test = wilcoxon(proposed_values, baseline, zero_method="zsplit")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        else:
            statistic = float("nan")
            p_value = float("nan")
        output[name] = {
            "mean_difference_percentage_points": float(
                100.0 * np.mean(proposed_values - baseline)
            ),
            "wilcoxon_statistic": statistic,
            "two_sided_p_value": p_value,
            "paired_unit": "subject mean across all shots and repeats",
            "subject_count": len(subjects),
        }
    return output


def validate_disjoint(groups: Dict[str, Sequence[int]]) -> None:
    seen: Dict[int, str] = {}
    for group, subjects in groups.items():
        for subject in subjects:
            if subject in seen:
                raise ValueError(f"S{subject:03d} appears in both {seen[subject]} and {group}")
            seen[subject] = group


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")
    groups = {
        "source": args.source_subjects,
        "selection": args.selection_subjects,
        "confirmation": args.confirmation_subjects,
    }
    validate_disjoint(groups)
    all_subjects = sorted({subject for subjects in groups.values() for subject in subjects})
    missing = [subject for subject in all_subjects if not subject_path(args.data_dir, subject).exists()]
    if missing:
        raise FileNotFoundError(f"Missing subject NPZ files: {missing}")
    eeg_by_band, covariance_by_band, labels, channels, fs = load_filterbank_subjects(
        args.data_dir, all_subjects
    )
    reflection = reflection_matrix(channels)
    selected_eta, selected_count, selection_scores = select_parameters(
        args, eeg_by_band, covariance_by_band, labels, reflection
    )
    print(
        f"Selected eta={selected_eta:g}, feature_count={selected_count}", flush=True
    )
    configurations = {
        "reflection_group_broadband_csp": (BROAD_BAND,),
        "reflection_group_fbcsp_all": SUB_BANDS,
        "reflection_group_fbcsp_source_fisher": SUB_BANDS,
        "reflection_group_fbcsp_hierarchical_fisher": SUB_BANDS,
    }
    results = {}
    for name, bands in configurations.items():
        is_broad = name == "reflection_group_broadband_csp"
        use_selection = "fisher" in name
        blend_target = name.endswith("hierarchical_fisher")
        print(f"Final confirmation {name}", flush=True)
        results[name] = evaluate(
            args.confirmation_subjects,
            args.source_subjects,
            eeg_by_band,
            covariance_by_band,
            labels,
            args.shots,
            args.repeats,
            args.kappa,
            args.seed,
            reflection,
            bands,
            6 if is_broad else 2,
            selected_eta,
            selected_count if use_selection else None,
            blend_target,
        )
    proposed = "reflection_group_fbcsp_hierarchical_fisher"
    report = {
        "protocol": {
            "dataset": "PhysioNet EEGMMIDB runs 4/8/12, left/right fist motor imagery",
            "source_subjects": [f"S{value:03d}" for value in args.source_subjects],
            "hyperparameter_selection_subjects": [
                f"S{value:03d}" for value in args.selection_subjects
            ],
            "final_untouched_confirmation_subjects": [
                f"S{value:03d}" for value in args.confirmation_subjects
            ],
            "subject_groups_disjoint": True,
            "shots_per_class": args.shots,
            "repeats": args.repeats,
            "channels": channels,
            "sampling_rate_hz": fs,
            "sub_bands_hz": [list(band) for band in SUB_BANDS],
            "seed": args.seed,
        },
        "method": {
            "name": "reflection-group filter-bank CSP with hierarchical Fisher selection",
            "fisher_shrinkage": "J=(shots/(shots+kappa))*J_target+(kappa/(shots+kappa))*J_source",
            "selected_eta": selected_eta,
            "selected_feature_count": selected_count,
            "kappa": args.kappa,
        },
        "hyperparameter_selection_scores": selection_scores,
        "final_confirmation": results,
        "final_confirmation_paired_tests": paired_tests(results, proposed),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {args.report.resolve()}")
    for name, result in results.items():
        metrics = result["overall"]
        print(
            f"{name}: bacc={metrics['mean_balanced_accuracy']:.4f}, "
            f"fit+predict={1000.0 * metrics['mean_elapsed_seconds']:.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
