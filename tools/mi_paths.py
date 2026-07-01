# -*- coding: utf-8 -*-
"""Shared filesystem layout for MI data and experiment artifacts."""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
RECORDINGS_DIR = DATA_DIR / "recordings"
MI_EXPERIMENT_MODEL_DIR = PROJECT_DIR / "models" / "mi_experiments"


def physionet_subject_filename(subject: int) -> str:
    return f"physionet_eegmmidb_s{subject:03d}_left_right_mi.npz"


def physionet_subject_path(data_dir: Path, subject: int) -> Path:
    """Return a subject NPZ path, accepting either old data/ or new data/processed."""
    filename = physionet_subject_filename(subject)
    candidates = [
        data_dir / filename,
        data_dir / "processed" / filename,
        PROCESSED_DATA_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def raw_data_dir_for(base_dir: Path) -> Path:
    """Return raw EDF download root for either data/ or data/raw style input."""
    if base_dir.name == "raw" or (base_dir / "MNE-eegbci-data").exists():
        return base_dir
    return base_dir / "raw"


def processed_data_dir_for(base_dir: Path) -> Path:
    """Return processed NPZ root for either data/ or data/processed style input."""
    if base_dir.name == "processed":
        return base_dir
    return base_dir / "processed"
