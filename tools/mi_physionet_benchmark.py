#!/usr/bin/env python
"""Download, convert, and benchmark left/right motor-imagery EEG data."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import joblib
import mne
import numpy as np
from mne.datasets import eegbci
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from arm_bci_system import (  # noqa: E402
    DEFAULT_EEG_CHANNELS,
    DecoderConfig,
    FilterBankEEGFeatures,
    HybridBCIModel,
    LABEL_TO_INDEX,
    SignalConfig,
    bandpass,
    common_average_reference,
    labels_to_indices,
    make_classifier,
    make_windows_from_trials,
)
from mi_paths import (  # noqa: E402
    EXPERIMENTS_DIR,
    MI_EXPERIMENT_MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

try:
    from mi_deep_models import fit_predict_deep as _fit_predict_deep
except (ImportError, OSError):
    _fit_predict_deep = None


DEFAULT_RUNS = (4, 8, 12)
DEFAULT_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 30.0))
MOTOR_CHANNELS = (
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
)
ASYMMETRY_PAIRS = (
    ("FC5", "FC6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("CP5", "CP6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--model-dir", type=Path, default=MI_EXPERIMENT_MODEL_DIR)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--runs", type=int, nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deep-epochs", type=int, default=80)
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def dataset_stem(subject: int) -> str:
    return f"physionet_eegmmidb_s{subject:03d}_left_right_mi"


def download_edf(data_dir: Path, subject: int, runs: Sequence[int]) -> List[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = eegbci.load_data(
        subject,
        list(runs),
        path=str(data_dir),
        update_path=False,
        verbose=True,
    )
    return [Path(path) for path in paths]


def expected_edf_paths(data_dir: Path, subject: int, runs: Sequence[int]) -> List[Path]:
    base = data_dir / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / f"S{subject:03d}"
    return [base / f"S{subject:03d}R{run:02d}.edf" for run in runs]


def convert_edf_to_npz(
    edf_paths: Sequence[Path],
    runs: Sequence[int],
    out_path: Path,
    subject: int,
) -> Path:
    all_eeg: List[np.ndarray] = []
    all_labels: List[str] = []
    all_run_ids: List[int] = []
    channel_names: List[str] = []
    fs = 0.0

    for run, edf_path in zip(runs, edf_paths):
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        eegbci.standardize(raw)
        raw.pick("eeg")
        fs = float(raw.info["sfreq"])
        if not channel_names:
            channel_names = list(raw.ch_names)
        elif channel_names != list(raw.ch_names):
            raise ValueError(f"Channel mismatch in {edf_path}")

        event_id = {"T1": 1, "T2": 2}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        tmin = 0.5
        tmax = 3.5 - 1.0 / fs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            proj=False,
            reject_by_annotation=True,
            verbose=False,
        )
        # MNE stores physical EEG in volts; the project and GUI use microvolts.
        eeg_uv = epochs.get_data(copy=True).astype(np.float32) * 1e6
        labels = np.where(epochs.events[:, 2] == event_id["T1"], "left", "right")
        all_eeg.append(eeg_uv)
        all_labels.extend(labels.tolist())
        all_run_ids.extend([int(run)] * len(labels))

    eeg = np.concatenate(all_eeg, axis=0)
    y = np.asarray(all_labels, dtype=object)
    run_ids = np.asarray(all_run_ids, dtype=np.int16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        eeg=eeg,
        y=y,
        fs=np.asarray([int(round(fs))], dtype=np.int32),
        channel_names=np.asarray(channel_names, dtype=object),
        run_ids=run_ids,
        subject=np.asarray([subject], dtype=np.int16),
        source_runs=np.asarray(list(runs), dtype=np.int16),
        epoch_tmin=np.asarray([0.5], dtype=np.float32),
        epoch_tmax=np.asarray([3.5], dtype=np.float32),
        unit=np.asarray(["uV"], dtype=object),
        source=np.asarray(["PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0"], dtype=object),
        source_url=np.asarray(["https://physionet.org/content/eegmmidb/1.0.0/"], dtype=object),
        task=np.asarray(["left/right fist motor imagery; T1=left, T2=right"], dtype=object),
    )
    return out_path


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, int, List[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        eeg = np.asarray(data["eeg"], dtype=np.float32)
        labels = np.asarray(data["y"], dtype=object)
        fs = int(np.asarray(data["fs"]).reshape(-1)[0])
        channel_names = [str(item) for item in data["channel_names"].tolist()]
        run_ids = np.asarray(data["run_ids"], dtype=np.int16)
    return eeg, labels, fs, channel_names, run_ids


def select_channel_indices(channel_names: Sequence[str], requested: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
    lookup = {name.upper(): idx for idx, name in enumerate(channel_names)}
    names = [name for name in requested if name.upper() in lookup]
    if len(names) < 3:
        raise ValueError("Not enough requested motor channels are present")
    return np.asarray([lookup[name.upper()] for name in names], dtype=np.int64), names


def make_signal_config(fs: int, channel_names: Sequence[str], classifier_kind: str = "logreg") -> SignalConfig:
    del classifier_kind
    return SignalConfig(
        fs=fs,
        eeg_channels=list(channel_names),
        window_seconds=1.0,
        step_seconds=0.2,
        calibration_trial_seconds=3.0,
        max_windows_per_trial=8,
        notch_hz=50.0,
        use_csp=True,
        csp_components=4,
    )


def fit_existing_model(
    train_eeg: np.ndarray,
    train_labels: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    classifier_kind: str,
) -> HybridBCIModel:
    cfg = make_signal_config(fs, channel_names, classifier_kind)
    model = HybridBCIModel(cfg, DecoderConfig(eeg_weight=0.85, aux_weight_cap=0.0))
    model.classifier_kind = classifier_kind
    y_project = labels_to_indices(train_labels)
    eeg_windows, y_windows, _ = make_windows_from_trials(train_eeg, y_project, cfg)
    model._fit_core(eeg_windows, y_windows, None)
    return model


def fit_predict_existing_pair(
    train_eeg: np.ndarray,
    train_labels: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = make_signal_config(fs, channel_names)
    left_idx = LABEL_TO_INDEX["left"]
    right_idx = LABEL_TO_INDEX["right"]
    y_project = labels_to_indices(train_labels)
    train_windows, train_y, _ = make_windows_from_trials(train_eeg, y_project, cfg)
    features = FilterBankEEGFeatures(cfg)
    train_x = features.fit_transform(train_windows, train_y)
    classifiers = {
        "logreg": make_classifier("logreg"),
        "lda": make_classifier("lda"),
    }
    for classifier in classifiers.values():
        classifier.fit(train_x, train_y)

    test_window_parts: List[np.ndarray] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for trial in test_eeg:
        windows, _, _ = make_windows_from_trials(
            trial[None, :, :],
            np.asarray([left_idx], dtype=np.int64),
            cfg,
        )
        test_window_parts.append(windows)
        spans.append((cursor, cursor + len(windows)))
        cursor += len(windows)
    test_x = features.transform(np.concatenate(test_window_parts, axis=0))

    output: Dict[str, np.ndarray] = {}
    for name, classifier in classifiers.items():
        window_proba = classifier.predict_proba(test_x)
        classes = np.asarray(classifier.classes_, dtype=np.int64)
        predictions = []
        for start, end in spans:
            predicted_class = int(classes[int(np.argmax(np.mean(window_proba[start:end], axis=0)))])
            predictions.append(0 if predicted_class == left_idx else 1)
        output[name] = np.asarray(predictions, dtype=np.int64)
    return output["logreg"], output["lda"]


def prepare_csp_data(eeg: np.ndarray, fs: int, motor_idx: np.ndarray) -> np.ndarray:
    x = common_average_reference(np.asarray(eeg, dtype=np.float32))
    x = bandpass(x, fs, 8.0, 30.0)
    return x[:, motor_idx, :]


def fit_predict_compact_csp_lda(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    motor_idx: np.ndarray,
) -> np.ndarray:
    train_x = prepare_csp_data(train_eeg, fs, motor_idx)
    test_x = prepare_csp_data(test_eeg, fs, motor_idx)
    model = make_pipeline(
        CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=False),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    model.fit(train_x, train_y)
    return model.predict(test_x).astype(np.int64)


def hemispheric_filter_bank_features(
    eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    motor_idx: np.ndarray,
    motor_names: Sequence[str],
) -> np.ndarray:
    x = common_average_reference(np.asarray(eeg, dtype=np.float32))
    motor_lookup = {name.upper(): pos for pos, name in enumerate(motor_names)}
    parts: List[np.ndarray] = []
    for low, high in DEFAULT_BANDS:
        filtered = bandpass(x, fs, low, high)[:, motor_idx, :]
        log_var = np.log(np.var(filtered, axis=-1) + 1e-6)
        parts.append(log_var)
        asymmetry = []
        for left_name, right_name in ASYMMETRY_PAIRS:
            left_pos = motor_lookup.get(left_name.upper())
            right_pos = motor_lookup.get(right_name.upper())
            if left_pos is not None and right_pos is not None:
                asymmetry.append(log_var[:, left_pos] - log_var[:, right_pos])
        if asymmetry:
            parts.append(np.stack(asymmetry, axis=1))
    del channel_names
    return np.concatenate(parts, axis=1).astype(np.float32)


def fit_predict_hemispheric_fb_lda(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    motor_idx: np.ndarray,
    motor_names: Sequence[str],
) -> np.ndarray:
    train_x = hemispheric_filter_bank_features(train_eeg, fs, channel_names, motor_idx, motor_names)
    test_x = hemispheric_filter_bank_features(test_eeg, fs, channel_names, motor_idx, motor_names)
    model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    model.fit(train_x, train_y)
    return model.predict(test_x).astype(np.int64)


def log_covariance_features(eeg: np.ndarray, fs: int, motor_idx: np.ndarray) -> np.ndarray:
    x = prepare_csp_data(eeg, fs, motor_idx)
    features: List[np.ndarray] = []
    for trial in x:
        centered = trial - np.mean(trial, axis=1, keepdims=True)
        cov = centered @ centered.T / max(1, centered.shape[1] - 1)
        cov += np.eye(cov.shape[0], dtype=np.float32) * max(float(np.trace(cov)) * 1e-6, 1e-6)
        cov /= max(float(np.trace(cov)), 1e-12)
        values, vectors = np.linalg.eigh(cov)
        values = np.maximum(values, 1e-8)
        log_cov = (vectors * np.log(values)) @ vectors.T
        upper = np.triu_indices(log_cov.shape[0])
        vector = log_cov[upper]
        off_diag = upper[0] != upper[1]
        vector[off_diag] *= np.sqrt(2.0)
        features.append(vector.astype(np.float32))
    return np.stack(features)


def fit_predict_logcov_logreg(
    train_eeg: np.ndarray,
    train_y: np.ndarray,
    test_eeg: np.ndarray,
    fs: int,
    motor_idx: np.ndarray,
) -> np.ndarray:
    train_x = log_covariance_features(train_eeg, fs, motor_idx)
    test_x = log_covariance_features(test_eeg, fs, motor_idx)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, max_iter=2000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train_x, train_y)
    return model.predict(test_x).astype(np.int64)


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, elapsed: float) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "label_order": ["left", "right"],
        "elapsed_seconds": float(elapsed),
    }


def cross_validate_models(
    eeg: np.ndarray,
    labels: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    folds: int,
    seed: int,
    include_deep: bool,
    deep_epochs: int,
) -> Dict[str, Dict[str, object]]:
    y = np.asarray([0 if str(label) == "left" else 1 for label in labels], dtype=np.int64)
    motor_idx, motor_names = select_channel_indices(channel_names, MOTOR_CHANNELS)
    project_idx, project_names = select_channel_indices(channel_names, DEFAULT_EEG_CHANNELS)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(eeg, y))
    predictions: Dict[str, np.ndarray] = {
        "existing_fbcsp_logreg": np.full(len(y), -1, dtype=np.int64),
        "existing_fbcsp_lda": np.full(len(y), -1, dtype=np.int64),
        "compact_csp_lda_8_30hz": np.full(len(y), -1, dtype=np.int64),
        "deployment_csp_lda_8_30hz": np.full(len(y), -1, dtype=np.int64),
        "hemispheric_filterbank_lda": np.full(len(y), -1, dtype=np.int64),
        "logcov_motor_logreg": np.full(len(y), -1, dtype=np.int64),
    }
    if include_deep:
        predictions["eegnet_motor"] = np.full(len(y), -1, dtype=np.int64)
        predictions["shallowconvnet_motor"] = np.full(len(y), -1, dtype=np.int64)
    elapsed: Dict[str, float] = {name: 0.0 for name in predictions}

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"Fold {fold_idx}/{folds}: train={len(train_idx)} test={len(test_idx)}", flush=True)
        train_eeg, test_eeg = eeg[train_idx], eeg[test_idx]
        train_labels = labels[train_idx]
        train_y = y[train_idx]
        train_project = train_eeg[:, project_idx, :]
        test_project = test_eeg[:, project_idx, :]
        deployment_idx, _deployment_names = select_channel_indices(project_names, MOTOR_CHANNELS)

        print("  existing FBCSP features + LogReg/LDA", flush=True)
        started = time.perf_counter()
        pred_logreg, pred_lda = fit_predict_existing_pair(
            train_project,
            train_labels,
            test_project,
            fs,
            project_names,
        )
        shared_elapsed = time.perf_counter() - started
        predictions["existing_fbcsp_logreg"][test_idx] = pred_logreg
        predictions["existing_fbcsp_lda"][test_idx] = pred_lda
        elapsed["existing_fbcsp_logreg"] += shared_elapsed
        elapsed["existing_fbcsp_lda"] += shared_elapsed

        print("  compact CSP-LDA", flush=True)
        started = time.perf_counter()
        predictions["compact_csp_lda_8_30hz"][test_idx] = fit_predict_compact_csp_lda(
            train_eeg, train_y, test_eeg, fs, motor_idx
        )
        elapsed["compact_csp_lda_8_30hz"] += time.perf_counter() - started

        print("  deployment-channel CSP-LDA", flush=True)
        started = time.perf_counter()
        predictions["deployment_csp_lda_8_30hz"][test_idx] = fit_predict_compact_csp_lda(
            train_project, train_y, test_project, fs, deployment_idx
        )
        elapsed["deployment_csp_lda_8_30hz"] += time.perf_counter() - started

        print("  hemispheric filter-bank LDA", flush=True)
        started = time.perf_counter()
        predictions["hemispheric_filterbank_lda"][test_idx] = fit_predict_hemispheric_fb_lda(
            train_eeg, train_y, test_eeg, fs, channel_names, motor_idx, motor_names
        )
        elapsed["hemispheric_filterbank_lda"] += time.perf_counter() - started

        print("  log-covariance LogisticRegression", flush=True)
        started = time.perf_counter()
        predictions["logcov_motor_logreg"][test_idx] = fit_predict_logcov_logreg(
            train_eeg, train_y, test_eeg, fs, motor_idx
        )
        elapsed["logcov_motor_logreg"] += time.perf_counter() - started

        if include_deep:
            if _fit_predict_deep is None:
                raise RuntimeError("Deep models were requested, but PyTorch could not be loaded")

            deep_train = bandpass(common_average_reference(train_project), fs, 4.0, 40.0)[:, deployment_idx, :]
            deep_test = bandpass(common_average_reference(test_project), fs, 4.0, 40.0)[:, deployment_idx, :]
            for model_offset, (result_name, model_kind) in enumerate(
                (("eegnet_motor", "eegnet"), ("shallowconvnet_motor", "shallowconvnet"))
            ):
                print(f"  deep model: {model_kind}", flush=True)
                started = time.perf_counter()
                prediction, _metadata = _fit_predict_deep(
                    model_kind,
                    deep_train,
                    train_y,
                    deep_test,
                    fs,
                    seed=seed + fold_idx * 10 + model_offset,
                    max_epochs=deep_epochs,
                )
                predictions[result_name][test_idx] = prediction
                elapsed[result_name] += time.perf_counter() - started

    return {name: summarize_predictions(y, pred, elapsed[name]) for name, pred in predictions.items()}


def train_and_save_existing_model(
    eeg: np.ndarray,
    labels: np.ndarray,
    fs: int,
    channel_names: Sequence[str],
    model_path: Path,
    metrics: Dict[str, object],
) -> None:
    model = fit_existing_model(eeg, labels, fs, channel_names, "logreg")
    model.metrics_ = dict(metrics)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))


def train_and_save_deployment_csp(
    eeg: np.ndarray,
    labels: np.ndarray,
    fs: int,
    input_channel_names: Sequence[str],
    model_path: Path,
) -> None:
    selected_idx, selected_names = select_channel_indices(input_channel_names, MOTOR_CHANNELS)
    y = np.asarray([0 if str(label) == "left" else 1 for label in labels], dtype=np.int64)
    x = prepare_csp_data(eeg, fs, selected_idx)
    model = make_pipeline(
        CSP(n_components=min(6, len(selected_idx)), reg="ledoit_wolf", log=True, norm_trace=False),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    model.fit(x, y)
    bundle = {
        "model_type": "binary_mi_compact_csp_lda",
        "model": model,
        "sampling_rate_hz": fs,
        "input_channel_names": list(input_channel_names),
        "selected_channel_names": list(selected_names),
        "selected_channel_indices": selected_idx.tolist(),
        "bandpass_hz": [8.0, 30.0],
        "labels": ["left_hand_mi", "right_hand_mi"],
        "fusion_role": "EMG selects axis; binary MI selects direction within the axis",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def print_results(results: Dict[str, Dict[str, object]]) -> None:
    print("\nModel benchmark (trial-level cross-validation)")
    print("-" * 79)
    print(f"{'model':34s} {'accuracy':>10s} {'balanced':>10s} {'kappa':>10s} {'seconds':>10s}")
    print("-" * 79)
    for name, metrics in sorted(results.items(), key=lambda item: float(item[1]["accuracy"]), reverse=True):
        print(
            f"{name:34s} {metrics['accuracy']:10.4f} {metrics['balanced_accuracy']:10.4f} "
            f"{metrics['kappa']:10.4f} {metrics['elapsed_seconds']:10.2f}"
        )


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mne.set_log_level("ERROR")
    data_dir = args.data_dir.resolve()
    raw_data_dir = args.raw_data_dir.resolve()
    report_dir = args.report_dir.resolve()
    model_dir = args.model_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / f"{dataset_stem(args.subject)}.npz"
    report_path = report_dir / f"{dataset_stem(args.subject)}_benchmark.json"
    model_path = model_dir / f"{dataset_stem(args.subject)}_existing_fbcsp.joblib"
    deployment_model_path = model_dir / f"{dataset_stem(args.subject)}_deployment_csp_lda.joblib"

    if args.no_download:
        edf_paths = expected_edf_paths(raw_data_dir, args.subject, args.runs)
        missing = [path for path in edf_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing EDF files: {missing}")
    else:
        edf_paths = download_edf(raw_data_dir, args.subject, args.runs)

    if args.force_convert or not npz_path.exists():
        convert_edf_to_npz(edf_paths, args.runs, npz_path, args.subject)

    eeg, labels, fs, channel_names, run_ids = load_npz(npz_path)
    project_idx, project_names = select_channel_indices(channel_names, DEFAULT_EEG_CHANNELS)
    counts = {label: int(np.sum(labels == label)) for label in ("left", "right")}
    print(
        f"Dataset: trials={len(labels)} channels={eeg.shape[1]} samples={eeg.shape[2]} "
        f"fs={fs}Hz counts={counts}"
    )
    results = cross_validate_models(
        eeg,
        labels,
        fs,
        channel_names,
        args.folds,
        args.seed,
        include_deep=not args.skip_deep,
        deep_epochs=args.deep_epochs,
    )
    print_results(results)
    train_and_save_existing_model(
        eeg[:, project_idx, :],
        labels,
        fs,
        project_names,
        model_path,
        results["existing_fbcsp_logreg"],
    )
    train_and_save_deployment_csp(
        eeg[:, project_idx, :],
        labels,
        fs,
        project_names,
        deployment_model_path,
    )

    best_name, best_metrics = max(results.items(), key=lambda item: float(item[1]["accuracy"]))
    report = {
        "dataset": {
            "name": "PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0",
            "source_url": "https://physionet.org/content/eegmmidb/1.0.0/",
            "subject": args.subject,
            "runs": list(args.runs),
            "task": "left/right fist motor imagery",
            "npz_path": str(npz_path),
            "edf_paths": [str(path) for path in edf_paths],
            "trial_count": int(len(labels)),
            "class_counts": counts,
            "channel_count": int(eeg.shape[1]),
            "sample_count": int(eeg.shape[2]),
            "sampling_rate_hz": fs,
            "run_ids": sorted(np.unique(run_ids).astype(int).tolist()),
            "unit": "uV",
        },
        "validation": {
            "method": f"{args.folds}-fold stratified trial-level cross-validation",
            "seed": args.seed,
            "chance_accuracy": 0.5,
            "no_window_leakage": True,
            "deep_epochs_max": args.deep_epochs,
            "deep_models_enabled": not args.skip_deep,
        },
        "models": results,
        "best_model": {"name": best_name, **best_metrics},
        "saved_existing_model": str(model_path),
        "saved_deployment_model": str(deployment_model_path),
        "existing_signal_config": asdict(make_signal_config(fs, project_names)),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved NPZ: {npz_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved existing model: {model_path}")
    print(f"Saved deployment model: {deployment_model_path}")
    print(f"Best cross-validated model: {best_name} accuracy={best_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
