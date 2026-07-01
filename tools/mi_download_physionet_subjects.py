#!/usr/bin/env python
"""Download and convert PhysioNet EEGMMIDB MI subjects without training models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import mne


PROJECT_DIR = Path(__file__).resolve().parents[1]

from mi_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from mi_physionet_benchmark import (  # noqa: E402
    DEFAULT_RUNS,
    convert_edf_to_npz,
    dataset_stem,
    download_edf,
    expected_edf_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(11, 21)))
    parser.add_argument("--runs", type=int, nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def prepare_subject(
    data_dir: Path,
    raw_data_dir: Path,
    subject: int,
    runs: Sequence[int],
    force_convert: bool,
) -> Path:
    output = data_dir / f"{dataset_stem(subject)}.npz"
    if output.exists() and not force_convert:
        print(f"S{subject:03d}: ready {output.name}", flush=True)
        return output
    edf_paths = expected_edf_paths(raw_data_dir, subject, runs)
    if not all(path.exists() for path in edf_paths):
        print(f"S{subject:03d}: downloading runs {list(runs)}", flush=True)
        edf_paths = download_edf(raw_data_dir, subject, runs)
    print(f"S{subject:03d}: converting {len(edf_paths)} EDF files", flush=True)
    return convert_edf_to_npz(edf_paths, runs, output, subject)


def main() -> None:
    args = parse_args()
    mne.set_log_level("WARNING")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.raw_data_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        prepare_subject(
            args.data_dir,
            args.raw_data_dir,
            subject,
            args.runs,
            args.force_convert,
        )
        for subject in args.subjects
    ]
    print(f"Prepared {len(outputs)} subjects in {args.data_dir.resolve()}")
    print(f"Raw EDF cache: {args.raw_data_dir.resolve()}")


if __name__ == "__main__":
    main()
