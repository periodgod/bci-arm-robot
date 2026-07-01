#!/usr/bin/env python
"""Registry and sequential runner for all reproducible MI experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import List, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
from mi_paths import EXPERIMENTS_DIR  # noqa: E402


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    purpose: str
    dataset_scope: str
    models: Sequence[str]
    script: str
    arguments: Sequence[str]
    report: str
    protocol: str


EXPERIMENTS = (
    ExperimentSpec(
        name="physionet_data_prepare",
        purpose="Ensure the standardized S001-S020 MI subject files exist",
        dataset_scope="PhysioNet EEGMMIDB S001-S020",
        models=(),
        script="tools/mi_download_physionet_subjects.py",
        arguments=("--subjects", *tuple(str(subject) for subject in range(1, 21))),
        report="data/processed/physionet_eegmmidb_s020_left_right_mi.npz",
        protocol="Download missing runs 4/8/12 and convert to standardized trial NPZ",
    ),
    ExperimentSpec(
        name="inventory",
        purpose="Count datasets and derived assets without double-counting raw files",
        dataset_scope="All local data assets",
        models=(),
        script="tools/mi_dataset_inventory.py",
        arguments=(),
        report="data/experiments/mi_dataset_inventory.json",
        protocol="Metadata inventory",
    ),
    ExperimentSpec(
        name="s001_baselines",
        purpose="Compare classical, project, and compact deep MI baselines",
        dataset_scope="PhysioNet EEGMMIDB S001",
        models=(
            "compact CSP-LDA",
            "existing FBCSP-LDA",
            "existing FBCSP-LogReg",
            "deployment CSP-LDA",
            "EEGNet",
            "ShallowConvNet",
        ),
        script="tools/mi_physionet_benchmark.py",
        arguments=("--subject", "1", "--no-download"),
        report="data/experiments/physionet_eegmmidb_s001_left_right_mi_benchmark.json",
        protocol="5-fold stratified trial-level CV",
    ),
    ExperimentSpec(
        name="physiology_candidates",
        purpose="Test physiology-informed frequency, Laplacian, and Riemannian candidates",
        dataset_scope="PhysioNet EEGMMIDB S001",
        models=(
            "nested time-frequency CSP-LDA",
            "weighted CSP band ensemble",
            "physiological Laplacian ERD-LDA",
            "Riemannian weighted distance LDA",
        ),
        script="tools/mi_physiology_optimization.py",
        arguments=(),
        report="data/experiments/physionet_eegmmidb_s001_mi_optimization.json",
        protocol="5-fold trial-level CV; nested selection inside training folds",
    ),
    ExperimentSpec(
        name="multisubject_confirmation",
        purpose="Reject S001-only improvements that do not generalize",
        dataset_scope="PhysioNet EEGMMIDB S001-S010",
        models=("fixed CSP-LDA", "nested subject-adaptive CSP"),
        script="tools/mi_multisubject_validation.py",
        arguments=("--folds", "5", "--seed", "42"),
        report="data/experiments/physionet_mi_multisubject_validation.json",
        protocol="S001 development; S002-S010 fixed confirmation",
    ),
    ExperimentSpec(
        name="transfer_covariance_prior",
        purpose="Test cross-subject covariance-prior CSP and alignment ablations",
        dataset_scope="PhysioNet EEGMMIDB S001-S010",
        models=(
            "transfer-regularized CSP-LDA",
            "nested transfer CSP",
            "Euclidean-aligned transfer CSP",
        ),
        script="tools/mi_transfer_regularized_csp.py",
        arguments=("--folds", "5", "--inner-folds", "3", "--seed", "42"),
        report="data/experiments/physionet_mi_transfer_regularized_csp.json",
        protocol="Target trial-level CV with non-target source covariance priors",
    ),
    ExperimentSpec(
        name="low_shot_innovation",
        purpose="Evaluate MI models with only 2-16 calibration trials per class",
        dataset_scope="PhysioNet source bank S001-S010; unseen confirmation S011-S020",
        models=(
            "target-only CSP-LDA",
            "equal fixed-prior CSP-LDA",
            "similarity-weighted adaptive-prior CSP-LDA",
            "hierarchical source-prototype CSP",
            "reliability-gated hierarchical CSP",
            "source-label-swap negative control",
        ),
        script="tools/mi_low_shot_innovation.py",
        arguments=(
            "--shots",
            "2",
            "4",
            "6",
            "8",
            "12",
            "16",
            "--repeats",
            "12",
            "--kappas",
            "1",
            "2",
            "4",
            "8",
            "--seed",
            "42",
        ),
        report="data/experiments/physionet_mi_low_shot_innovation.json",
        protocol="S001 model selection; S011-S020 absent from source bank and confirmation-only",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs="*",
        default=[],
        help="Experiment names, 'new' for low-shot innovation, or 'all'",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=EXPERIMENTS_DIR / "mi_experiment_registry.json",
    )
    return parser.parse_args()


def artifact_status(spec: ExperimentSpec) -> dict:
    report = PROJECT_DIR / spec.report
    status = {
        "exists": report.exists(),
        "size_bytes": report.stat().st_size if report.exists() else 0,
        "modified_local": (
            datetime.fromtimestamp(report.stat().st_mtime).astimezone().isoformat()
            if report.exists()
            else None
        ),
    }
    return status


def write_registry(path: Path, runs: List[dict] | None = None) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(Path(__file__).relative_to(PROJECT_DIR)),
        "experiments": [
            {**asdict(spec), "artifact": artifact_status(spec)} for spec in EXPERIMENTS
        ],
        "runs_this_invocation": runs or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_experiments(requested: Sequence[str]) -> List[ExperimentSpec]:
    if not requested:
        return []
    if "all" in requested:
        return list(EXPERIMENTS)
    if "new" in requested:
        requested = tuple(name for name in requested if name != "new") + (
            "inventory",
            "low_shot_innovation",
        )
    lookup = {spec.name: spec for spec in EXPERIMENTS}
    unknown = sorted(set(requested) - set(lookup))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}; choices={sorted(lookup)}")
    selected = []
    for spec in EXPERIMENTS:
        if spec.name in requested and spec not in selected:
            selected.append(spec)
    return selected


def run_experiment(spec: ExperimentSpec) -> dict:
    command = [sys.executable, str(PROJECT_DIR / spec.script), *spec.arguments]
    print(f"\n[{spec.name}] {' '.join(command)}", flush=True)
    started = datetime.now(timezone.utc)
    completed = subprocess.run(command, cwd=PROJECT_DIR, check=False)
    ended = datetime.now(timezone.utc)
    result = {
        "name": spec.name,
        "return_code": completed.returncode,
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "report": spec.report,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Experiment {spec.name} failed with code {completed.returncode}")
    return result


def main() -> None:
    args = parse_args()
    selected = select_experiments(args.run)
    runs = []
    for spec in selected:
        runs.append(run_experiment(spec))
    write_registry(args.registry, runs)

    print("\nMI experiment registry")
    print("-" * 112)
    for spec in EXPERIMENTS:
        artifact = artifact_status(spec)
        print(
            f"{spec.name:30s} dataset={spec.dataset_scope:48.48s} "
            f"report={'ready' if artifact['exists'] else 'missing'}"
        )
    print(f"\nRegistry: {args.registry.resolve()}")


if __name__ == "__main__":
    main()
