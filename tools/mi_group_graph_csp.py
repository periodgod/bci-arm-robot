#!/usr/bin/env python
"""Low-shot MI with bilateral group symmetry and motor-electrode graph regularization."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mne
import numpy as np
from scipy.linalg import eigh
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from mi_low_shot_innovation import (  # noqa: E402
    class_prototypes,
    load_preprocessed_subjects,
    pooled_within_class_covariance,
    source_feature_statistics,
    source_similarity_weights,
    standardize_from_training,
    stratified_low_shot_split,
    weighted_source_class_covariance,
)
from mi_paths import EXPERIMENTS_DIR, PROCESSED_DATA_DIR  # noqa: E402
from mi_transfer_regularized_csp import (  # noqa: E402
    class_mean_covariances,
    csp_log_variance,
    subject_path,
)


DEFAULT_SHOTS = (2, 4, 6, 8, 12, 16)
DEFAULT_ETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_GAMMAS = (0.0,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--source-subjects", type=int, nargs="*", default=list(range(1, 11)))
    parser.add_argument("--selection-subjects", type=int, nargs="*", default=list(range(11, 21)))
    parser.add_argument("--validation-subjects", type=int, nargs="*", default=[])
    parser.add_argument("--confirmation-subjects", type=int, nargs="*", default=list(range(21, 31)))
    parser.add_argument("--shots", type=int, nargs="*", default=list(DEFAULT_SHOTS))
    parser.add_argument("--selection-shots", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--etas", type=float, nargs="*", default=list(DEFAULT_ETAS))
    parser.add_argument("--gammas", type=float, nargs="*", default=list(DEFAULT_GAMMAS))
    parser.add_argument("--selection-repeats", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--kappa", type=float, default=8.0)
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=EXPERIMENTS_DIR / "physionet_mi_reflection_group_csp.json",
    )
    return parser.parse_args()


def channel_coordinate(name: str) -> Tuple[int, int]:
    """Map the 21-channel motor grid to sparse (anterior, left-right) coordinates."""
    if name.startswith("FC"):
        row = 0
        suffix = name[2:]
    elif name.startswith("CP"):
        row = 2
        suffix = name[2:]
    elif name.startswith("C"):
        row = 1
        suffix = name[1:]
    else:
        raise ValueError(f"Unsupported motor channel: {name}")
    columns = {"5": -3, "3": -2, "1": -1, "z": 0, "2": 1, "4": 2, "6": 3}
    if suffix not in columns:
        raise ValueError(f"Unsupported motor channel: {name}")
    return row, columns[suffix]


def reflected_channel(name: str) -> str:
    """Return the sagittal-plane reflection of a 10-20 motor electrode."""
    if name.endswith("z"):
        return name
    mirror = {"1": "2", "2": "1", "3": "4", "4": "3", "5": "6", "6": "5"}
    return f"{name[:-1]}{mirror[name[-1]]}"


def reflection_matrix(channel_names: Sequence[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(channel_names)}
    matrix = np.zeros((len(channel_names), len(channel_names)), dtype=np.float64)
    for row, name in enumerate(channel_names):
        reflected = reflected_channel(name)
        if reflected not in index:
            raise ValueError(f"Missing reflected channel {reflected} for {name}")
        matrix[row, index[reflected]] = 1.0
    return matrix


def motor_graph_laplacian(channel_names: Sequence[str]) -> np.ndarray:
    """Build a normalized 4-neighbour graph Laplacian on the motor electrode grid."""
    coordinates = [channel_coordinate(name) for name in channel_names]
    adjacency = np.zeros((len(channel_names), len(channel_names)), dtype=np.float64)
    for first in range(len(channel_names)):
        for second in range(first + 1, len(channel_names)):
            row_distance = abs(coordinates[first][0] - coordinates[second][0])
            column_distance = abs(coordinates[first][1] - coordinates[second][1])
            if (row_distance == 1 and column_distance == 0) or (
                row_distance == 0 and column_distance == 1
            ):
                adjacency[first, second] = 1.0
                adjacency[second, first] = 1.0
    degree = np.sum(adjacency, axis=1)
    inverse_root = np.diag(1.0 / np.sqrt(np.maximum(degree, 1.0)))
    return np.eye(len(channel_names)) - inverse_root @ adjacency @ inverse_root


def bilateral_group_projection(
    class_covariances: np.ndarray,
    reflection: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Shrink two class covariances toward the C2-equivariant, label-swapping orbit."""
    class_zero_orbit = 0.5 * (
        class_covariances[0] + reflection @ class_covariances[1] @ reflection.T
    )
    class_one_orbit = reflection @ class_zero_orbit @ reflection.T
    orbit = np.stack((class_zero_orbit, class_one_orbit))
    return (1.0 - eta) * class_covariances + eta * orbit


def fit_group_graph_filters(
    target_covariances: np.ndarray,
    target_labels: np.ndarray,
    source_prior: np.ndarray,
    alpha: float,
    components: int,
    reflection: np.ndarray,
    laplacian: np.ndarray,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Solve graph-regularized CSP after source and C2-group covariance shrinkage."""
    target_means = class_mean_covariances(target_covariances, target_labels)
    class_covariances = (1.0 - alpha) * target_means + alpha * source_prior
    class_covariances = bilateral_group_projection(class_covariances, reflection, eta)
    composite = class_covariances[0] + class_covariances[1]
    channel_count = composite.shape[0]
    covariance_scale = np.trace(composite) / channel_count
    denominator = composite + gamma * covariance_scale * laplacian
    ridge = 1e-7 * np.trace(composite) / channel_count
    denominator = denominator + ridge * np.eye(channel_count)
    eigenvalues, eigenvectors = eigh(class_covariances[1], denominator, check_finite=False)
    order = np.argsort(eigenvalues)
    side = min(components // 2, len(order) // 2)
    selected = np.concatenate((order[:side], order[-side:]))
    return eigenvectors[:, selected].T


def group_graph_hierarchical_prediction(
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
    reflection: np.ndarray,
    laplacian: np.ndarray,
    eta: float,
    gamma: float,
    augment_target: bool = False,
) -> np.ndarray:
    shots_per_class = min(np.sum(train_labels == 0), np.sum(train_labels == 1))
    original_train_count = len(train_eeg)
    original_class_counts = np.asarray(
        [np.sum(train_labels == label) for label in (0, 1)]
    )
    if augment_target:
        reflected_eeg = np.einsum("cd,ndt->nct", reflection, train_eeg, optimize=True)
        reflected_covariances = np.einsum(
            "ca,nab,db->ncd",
            reflection,
            train_covariances,
            reflection,
            optimize=True,
        )
        train_eeg = np.concatenate((train_eeg, reflected_eeg))
        train_covariances = np.concatenate((train_covariances, reflected_covariances))
        train_labels = np.concatenate((train_labels, 1 - train_labels))
    alpha = float(kappa / (shots_per_class + kappa))
    filters = fit_group_graph_filters(
        train_covariances,
        train_labels,
        source_prior,
        alpha,
        components,
        reflection,
        laplacian,
        eta,
        gamma,
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
        False,
    )
    target_prototypes = class_prototypes(target_train, train_labels)
    target_covariance = pooled_within_class_covariance(target_train, train_labels)
    prototypes = (
        original_class_counts[:, None] * target_prototypes + kappa * source_prototypes
    ) / (original_class_counts[:, None] + kappa)
    source_equivalent_trials = 2.0 * kappa
    covariance = (
        original_train_count * target_covariance + source_equivalent_trials * source_covariance
    ) / (original_train_count + source_equivalent_trials)
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


def metric_row(labels: np.ndarray, prediction: np.ndarray, elapsed: float) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "kappa": float(cohen_kappa_score(labels, prediction)),
        "elapsed_seconds": float(elapsed),
    }


def summarize_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    output: Dict[str, object] = {"evaluations": len(rows)}
    for key in ("accuracy", "balanced_accuracy", "kappa", "elapsed_seconds"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[f"mean_{key}"] = float(np.mean(values))
        output[f"std_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if key == "balanced_accuracy":
            output["values_balanced_accuracy"] = values.tolist()
    return output


def evaluate_configuration(
    target_ids: Sequence[int],
    source_ids: Sequence[int],
    subject_eeg: Dict[int, np.ndarray],
    subject_covariances: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
    shots_values: Sequence[int],
    repeats: int,
    kappa: float,
    components: int,
    seed: int,
    reflection: np.ndarray,
    laplacian: np.ndarray,
    eta: float,
    gamma: float,
    augment_target: bool = False,
) -> Dict[str, object]:
    rows: List[Dict[str, float]] = []
    by_shots: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    by_subject: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for target_id in target_ids:
        eeg = subject_eeg[target_id]
        covariances = subject_covariances[target_id]
        labels = subject_labels[target_id]
        for shots in shots_values:
            for repeat in range(repeats):
                split_seed = seed + target_id * 100_000 + shots * 1_000 + repeat
                train_idx, test_idx = stratified_low_shot_split(labels, shots, split_seed)
                weights = source_similarity_weights(
                    covariances[train_idx], subject_covariances, source_ids
                )
                source_prior = weighted_source_class_covariance(
                    subject_covariances, subject_labels, source_ids, weights
                )
                started = time.perf_counter()
                prediction = group_graph_hierarchical_prediction(
                    eeg[train_idx],
                    covariances[train_idx],
                    labels[train_idx],
                    eeg[test_idx],
                    subject_eeg,
                    subject_labels,
                    source_ids,
                    weights,
                    source_prior,
                    kappa,
                    components,
                    reflection,
                    laplacian,
                    eta,
                    gamma,
                    augment_target,
                )
                row = metric_row(labels[test_idx], prediction, time.perf_counter() - started)
                rows.append(row)
                by_shots[str(shots)].append(row)
                by_subject[f"S{target_id:03d}"].append(row)
    return {
        "overall": summarize_rows(rows),
        "by_shots": {key: summarize_rows(value) for key, value in by_shots.items()},
        "by_subject": {key: summarize_rows(value) for key, value in by_subject.items()},
    }


def select_hyperparameters(
    target_ids: Sequence[int],
    source_ids: Sequence[int],
    subject_eeg: Dict[int, np.ndarray],
    subject_covariances: Dict[int, np.ndarray],
    subject_labels: Dict[int, np.ndarray],
    shots_values: Sequence[int],
    repeats: int,
    kappa: float,
    components: int,
    seed: int,
    reflection: np.ndarray,
    laplacian: np.ndarray,
    etas: Sequence[float],
    gammas: Sequence[float],
) -> Tuple[float, float, Dict[str, float]]:
    scores = {}
    for eta in etas:
        for gamma in gammas:
            result = evaluate_configuration(
                target_ids,
                source_ids,
                subject_eeg,
                subject_covariances,
                subject_labels,
                shots_values,
                repeats,
                kappa,
                components,
                seed,
                reflection,
                laplacian,
                eta,
                gamma,
                augment_target=False,
            )
            key = f"eta={eta:g},gamma={gamma:g}"
            scores[key] = float(result["overall"]["mean_balanced_accuracy"])
            print(f"Selection {key}: bacc={scores[key]:.4f}", flush=True)
    selected_eta, selected_gamma = max(
        ((eta, gamma) for eta in etas for gamma in gammas),
        key=lambda pair: (scores[f"eta={pair[0]:g},gamma={pair[1]:g}"], -sum(pair)),
    )
    return float(selected_eta), float(selected_gamma), scores


def paired_tests(
    results: Dict[str, Dict[str, object]], proposed_name: str
) -> Dict[str, Dict[str, float]]:
    subject_ids = sorted(results[proposed_name]["by_subject"])
    proposed = np.asarray(
        [
            results[proposed_name]["by_subject"][subject]["mean_balanced_accuracy"]
            for subject in subject_ids
        ],
        dtype=np.float64,
    )
    output = {}
    for name, result in results.items():
        if name == proposed_name:
            continue
        baseline = np.asarray(
            [result["by_subject"][subject]["mean_balanced_accuracy"] for subject in subject_ids],
            dtype=np.float64,
        )
        difference = proposed - baseline
        if len(subject_ids) >= 2:
            test = wilcoxon(proposed, baseline, zero_method="zsplit")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        else:
            statistic = float("nan")
            p_value = float("nan")
        output[name] = {
            "mean_difference_percentage_points": float(100.0 * np.mean(difference)),
            "wilcoxon_statistic": statistic,
            "two_sided_p_value": p_value,
            "paired_unit": "subject mean across all shots and repeats",
            "subject_count": len(subject_ids),
        }
    return output


def validate_subject_partition(groups: Dict[str, Iterable[int]]) -> None:
    seen: Dict[int, str] = {}
    for name, subjects in groups.items():
        for subject in subjects:
            if subject in seen:
                raise ValueError(f"S{subject:03d} appears in both {seen[subject]} and {name}")
            seen[subject] = name


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")
    groups = {
        "source": args.source_subjects,
        "selection": args.selection_subjects,
        "validation": args.validation_subjects,
        "confirmation": args.confirmation_subjects,
    }
    validate_subject_partition(groups)
    all_subjects = sorted({subject for subjects in groups.values() for subject in subjects})
    missing = [subject for subject in all_subjects if not subject_path(args.data_dir, subject).exists()]
    if missing:
        raise FileNotFoundError(f"Missing subject NPZ files: {missing}")
    subject_eeg, subject_covariances, subject_labels, channels, fs = load_preprocessed_subjects(
        args.data_dir, all_subjects
    )
    reflection = reflection_matrix(channels)
    laplacian = motor_graph_laplacian(channels)

    selected_eta, selected_gamma, selection_scores = select_hyperparameters(
        args.selection_subjects,
        args.source_subjects,
        subject_eeg,
        subject_covariances,
        subject_labels,
        args.selection_shots,
        args.selection_repeats,
        args.kappa,
        args.components,
        args.seed,
        reflection,
        laplacian,
        args.etas,
        args.gammas,
    )
    print(
        f"Selected eta={selected_eta:g}, gamma={selected_gamma:g}; "
        f"validating on {[f'S{value:03d}' for value in args.validation_subjects]}",
        flush=True,
    )
    configurations = {
        "hierarchical_csp": (0.0, 0.0, False),
        "reflection_group_csp": (selected_eta, 0.0, False),
    }
    validation = {}
    confirmation = {}
    for name, (eta, gamma, augment_target) in configurations.items():
        if args.validation_subjects:
            print(f"Validation {name}", flush=True)
            validation[name] = evaluate_configuration(
                args.validation_subjects,
                args.source_subjects,
                subject_eeg,
                subject_covariances,
                subject_labels,
                args.shots,
                args.repeats,
                args.kappa,
                args.components,
                args.seed,
                reflection,
                laplacian,
                eta,
                gamma,
                augment_target,
            )
        print(f"Final confirmation {name}", flush=True)
        confirmation[name] = evaluate_configuration(
            args.confirmation_subjects,
            args.source_subjects,
            subject_eeg,
            subject_covariances,
            subject_labels,
            args.shots,
            args.repeats,
            args.kappa,
            args.components,
            args.seed,
            reflection,
            laplacian,
            eta,
            gamma,
            augment_target,
        )

    proposed_name = "reflection_group_csp"
    report = {
        "protocol": {
            "dataset": "PhysioNet EEGMMIDB runs 4/8/12, left/right fist motor imagery",
            "source_subjects": [f"S{value:03d}" for value in args.source_subjects],
            "hyperparameter_selection_subjects": [
                f"S{value:03d}" for value in args.selection_subjects
            ],
            "validation_subjects": [f"S{value:03d}" for value in args.validation_subjects],
            "final_untouched_confirmation_subjects": [
                f"S{value:03d}" for value in args.confirmation_subjects
            ],
            "subject_groups_disjoint": True,
            "shots_per_class": args.shots,
            "repeats": args.repeats,
            "channels": channels,
            "sampling_rate_hz": fs,
            "frequency_band_hz": [8.0, 30.0],
            "components": args.components,
            "seed": args.seed,
        },
        "method": {
            "name": "reflection-group-equivariant hierarchical CSP",
            "group": "C2 sagittal reflection with class-label swap",
            "rejected_ablation": (
                "graph regularization and mirrored pseudo-trial augmentation were removed "
                "after development experiments failed to show stable improvement"
            ),
            "selected_eta": selected_eta,
            "selected_gamma": selected_gamma,
            "kappa": args.kappa,
        },
        "hyperparameter_selection_scores": selection_scores,
        "validation": validation,
        "validation_paired_tests": (
            paired_tests(validation, proposed_name) if validation else {}
        ),
        "final_confirmation": confirmation,
        "final_confirmation_paired_tests": paired_tests(confirmation, proposed_name),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report: {args.report.resolve()}")
    for name, result in confirmation.items():
        overall = result["overall"]
        print(
            f"{name}: bacc={overall['mean_balanced_accuracy']:.4f}, "
            f"fit+predict={1000.0 * overall['mean_elapsed_seconds']:.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
