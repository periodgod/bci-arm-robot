#!/usr/bin/env python
"""Build the reflection-group CSP deployment bundle used by the arm GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm_bci_system import (  # noqa: E402
    DEFAULT_EEG_CHANNELS,
    bandpass,
    common_average_reference,
    csp_log_variance_features,
)
from mi_paths import PROCESSED_DATA_DIR, physionet_subject_path  # noqa: E402
from mi_physionet_benchmark import load_npz, select_channel_indices  # noqa: E402
from mi_transfer_regularized_csp import normalized_trial_covariances  # noqa: E402


DEPLOYMENT_MOTOR_CHANNELS = tuple(
    name
    for name in DEFAULT_EEG_CHANNELS
    if name in {"FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6"}
)
FULL_32_CHANNELS = tuple(DEFAULT_EEG_CHANNELS)
CHANNEL_SETS = {
    "deployment11": DEPLOYMENT_MOTOR_CHANNELS,
    "full32": FULL_32_CHANNELS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 31)))
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument(
        "--channel-set",
        choices=sorted(CHANNEL_SETS),
        default="deployment11",
        help="Channel subset used to build the deployment bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def reflected_channel(name: str) -> str:
    if name.endswith("z"):
        return name
    pairs = {"1": "2", "2": "1", "3": "4", "4": "3", "5": "6", "6": "5"}
    return f"{name[:-1]}{pairs[name[-1]]}"


def reflection_matrix(channel_names: Sequence[str]) -> np.ndarray:
    lookup = {name: idx for idx, name in enumerate(channel_names)}
    matrix = np.zeros((len(channel_names), len(channel_names)), dtype=np.float64)
    for row, name in enumerate(channel_names):
        matrix[row, lookup[reflected_channel(name)]] = 1.0
    return matrix


def load_subjects(
    data_dir: Path,
    subjects: Sequence[int],
    requested_channel_names: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    eeg_by_subject: Dict[str, np.ndarray] = {}
    labels_by_subject: Dict[str, np.ndarray] = {}
    sampling_rate = 0
    for subject in subjects:
        path = physionet_subject_path(data_dir, subject)
        if not path.exists():
            raise FileNotFoundError(path)
        eeg, raw_labels, fs, channel_names, _run_ids = load_npz(path)
        indices, selected_names = select_channel_indices(channel_names, requested_channel_names)
        if tuple(selected_names) != tuple(requested_channel_names):
            raise ValueError(f"Unexpected channel order in {path.name}: {selected_names}")
        selected = eeg[:, indices, :]
        selected = bandpass(common_average_reference(selected), fs, 8.0, 30.0)
        labels = np.asarray([0 if str(value) == "left" else 1 for value in raw_labels], dtype=np.int64)
        key = f"S{subject:03d}"
        eeg_by_subject[key] = selected.astype(np.float32)
        labels_by_subject[key] = labels
        if sampling_rate and sampling_rate != fs:
            raise ValueError(f"Sampling rate mismatch in {path.name}")
        sampling_rate = fs
    return eeg_by_subject, labels_by_subject, sampling_rate


def subject_balanced_class_covariances(
    eeg_by_subject: Dict[str, np.ndarray],
    labels_by_subject: Dict[str, np.ndarray],
    channel_names: Sequence[str],
) -> np.ndarray:
    subject_means: List[np.ndarray] = []
    for subject in sorted(eeg_by_subject):
        covariances = normalized_trial_covariances(eeg_by_subject[subject])
        labels = labels_by_subject[subject]
        subject_means.append(
            np.stack([np.mean(covariances[labels == class_id], axis=0) for class_id in (0, 1)])
        )
    means = np.mean(np.stack(subject_means), axis=0)
    reflection = reflection_matrix(channel_names)
    class_zero = 0.5 * (means[0] + reflection @ means[1] @ reflection.T)
    return np.stack((class_zero, reflection @ class_zero @ reflection.T))


def fit_spatial_filters(class_covariances: np.ndarray, components: int) -> np.ndarray:
    composite = class_covariances[0] + class_covariances[1]
    ridge = 1e-7 * np.trace(composite) / composite.shape[0]
    values, vectors = eigh(
        class_covariances[1], composite + ridge * np.eye(composite.shape[0]), check_finite=False
    )
    order = np.argsort(values)
    side = min(max(1, components // 2), len(order) // 2)
    selected = np.concatenate((order[:side], order[-side:]))
    return vectors[:, selected].T


def main() -> None:
    args = parse_args()
    requested_channel_names = CHANNEL_SETS[args.channel_set]
    eeg_by_subject, labels_by_subject, fs = load_subjects(args.data_dir, args.subjects, requested_channel_names)
    filters = fit_spatial_filters(
        subject_balanced_class_covariances(eeg_by_subject, labels_by_subject, requested_channel_names),
        args.components,
    )
    train_eeg = np.concatenate([eeg_by_subject[key] for key in sorted(eeg_by_subject)], axis=0)
    train_labels = np.concatenate([labels_by_subject[key] for key in sorted(labels_by_subject)])
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(csp_log_variance_features(train_eeg, filters), train_labels)

    first_subject = eeg_by_subject[sorted(eeg_by_subject)[0]]
    bundle = {
        "model_type": "binary_mi_reflection_group_csp_lda",
        "model_name": f"C2 reflection-group CSP + shrinkage LDA ({len(requested_channel_names)}ch)",
        "sampling_rate_hz": int(fs),
        "window_samples": int(first_subject.shape[-1]),
        "input_channel_names": list(requested_channel_names),
        "selected_channel_indices": list(range(len(requested_channel_names))),
        "spatial_filters": filters.astype(np.float64),
        "classifier": classifier,
        "labels": ["left_hand_mi", "right_hand_mi"],
        "group": f"C2 sagittal reflection with class-label swap ({args.channel_set})",
        "channel_set": args.channel_set,
        "group_eta": 1.0,
        "components": int(filters.shape[0]),
        "source_subjects": [f"S{subject:03d}" for subject in args.subjects],
        "training_trials": int(len(train_labels)),
        "preprocessing": {
            "reference": "CAR over deployment motor channels",
            "bandpass_hz": [8.0, 30.0],
        },
    }
    output = args.output or (
        PROJECT_DIR / "models" / ("group_csp_32ch_left_right_mi.joblib" if args.channel_set == "full32" else "group_csp_left_right_mi.joblib")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    metadata_path = output.with_suffix(".json")
    metadata = {key: value for key, value in bundle.items() if key not in {"spatial_filters", "classifier"}}
    metadata["spatial_filter_shape"] = list(filters.shape)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved group CSP model: {output}")
    print(
        f"subjects={len(args.subjects)} trials={len(train_labels)} channels={len(requested_channel_names)} components={filters.shape[0]}"
    )


if __name__ == "__main__":
    main()
