#!/usr/bin/env python
"""Inventory local MI datasets without double-counting raw and derived files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
from mi_paths import (  # noqa: E402
    DATA_DIR,
    EXPERIMENTS_DIR,
    MI_EXPERIMENT_MODEL_DIR,
    processed_data_dir_for,
    raw_data_dir_for,
)

PHYSIONET_PATTERN = re.compile(r"physionet_eegmmidb_s(?P<subject>\d{3})_left_right_mi\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--json-report", type=Path, default=EXPERIMENTS_DIR / "mi_dataset_inventory.json"
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=PROJECT_DIR / "docs" / "MI_DATASET_INVENTORY.md",
    )
    return parser.parse_args()


def scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    values = np.asarray(data[key]).reshape(-1)
    return values[0].item() if values.size and hasattr(values[0], "item") else values[0]


def inspect_physionet_npz(path: Path) -> Dict[str, Any]:
    match = PHYSIONET_PATTERN.match(path.name)
    if match is None:
        raise ValueError(path.name)
    with np.load(path, allow_pickle=True) as data:
        eeg = np.asarray(data["eeg"])
        labels = [str(item) for item in np.asarray(data["y"], dtype=object).tolist()]
        channels = [str(item) for item in np.asarray(data["channel_names"], dtype=object).tolist()]
        runs = sorted({int(item) for item in np.asarray(data["run_ids"]).tolist()})
        fs = int(scalar(data, "fs"))
        return {
            "file": path.name,
            "subject": int(match.group("subject")),
            "trials": int(eeg.shape[0]),
            "channels": int(eeg.shape[1]),
            "samples_per_trial": int(eeg.shape[2]),
            "duration_seconds": float(eeg.shape[2] / fs),
            "sampling_rate_hz": fs,
            "label_counts": dict(sorted(Counter(labels).items())),
            "channel_names": channels,
            "runs": runs,
            "unit": str(scalar(data, "unit", "unknown")),
            "task": str(scalar(data, "task", "unknown")),
            "size_bytes": path.stat().st_size,
        }


def inspect_other_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {
            key: {"shape": list(np.asarray(data[key]).shape), "dtype": str(np.asarray(data[key]).dtype)}
            for key in data.files
        }
    return {"file": path.name, "arrays": arrays, "size_bytes": path.stat().st_size}


def build_inventory(data_dir: Path) -> Dict[str, Any]:
    processed_dir = processed_data_dir_for(data_dir)
    raw_dir = raw_data_dir_for(data_dir)
    report_dir = data_dir / "experiments" if (data_dir / "experiments").exists() else EXPERIMENTS_DIR
    model_dir = MI_EXPERIMENT_MODEL_DIR if MI_EXPERIMENT_MODEL_DIR.exists() else data_dir / "mi_models"

    physionet = []
    other_npz = []
    for path in sorted(processed_dir.glob("*.npz")):
        if PHYSIONET_PATTERN.match(path.name):
            physionet.append(inspect_physionet_npz(path))
        else:
            other_npz.append(inspect_other_npz(path))

    edf_files = sorted((raw_dir / "MNE-eegbci-data").rglob("*.edf"))
    edf_subjects = Counter(path.parent.name for path in edf_files)
    reports = sorted(path.name for path in report_dir.glob("*.json"))
    models = sorted(path.name for path in model_dir.glob("*"))
    incomplete = []
    for path in raw_dir.rglob("*.partial"):
        try:
            incomplete.append(str(path.relative_to(data_dir)))
        except ValueError:
            incomplete.append(str(path))
    incomplete = sorted(incomplete)

    trial_total = sum(row["trials"] for row in physionet)
    label_total = Counter()
    for row in physionet:
        label_total.update(row["label_counts"])
    family = {
        "name": "PhysioNet EEGMMIDB left/right fist motor imagery",
        "source_url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "subjects": len(physionet),
        "subject_ids": [row["subject"] for row in physionet],
        "derived_npz_files": len(physionet),
        "raw_edf_files": len(edf_files),
        "raw_runs_per_subject": dict(sorted(edf_subjects.items())),
        "trials": trial_total,
        "label_counts": dict(sorted(label_total.items())),
        "sampling_rates_hz": sorted({row["sampling_rate_hz"] for row in physionet}),
        "channel_counts": sorted({row["channels"] for row in physionet}),
        "trial_durations_seconds": sorted({row["duration_seconds"] for row in physionet}),
        "raw_size_bytes": sum(path.stat().st_size for path in edf_files),
        "derived_size_bytes": sum(row["size_bytes"] for row in physionet),
    }
    return {
        "counting_rule": "Raw EDF and derived NPZ are two representations of one dataset family.",
        "usable_mi_dataset_families": 1 if physionet else 0,
        "mi_families": [family] if physionet else [],
        "physionet_subject_files": physionet,
        "other_npz_not_counted_as_left_right_mi": other_npz,
        "experiment_reports": reports,
        "saved_models": models,
        "incomplete_files": incomplete,
    }


def write_markdown(inventory: Dict[str, Any], path: Path) -> None:
    family = inventory["mi_families"][0] if inventory["mi_families"] else None
    lines: List[str] = ["# MI 数据集资产清单", ""]
    lines.append("统计规则：原始 EDF 与派生 NPZ 是同一批信号的不同表示，不重复计数。")
    lines.append("")
    if family is not None:
        lines.extend(
            [
                "## 可用左右手 MI 数据",
                "",
                "| 数据集族 | 受试者 | 原始 run | trial | 左手 | 右手 | 通道 | 采样率 | trial 时长 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                (
                    f"| PhysioNet EEGMMIDB | {family['subjects']} | {family['raw_edf_files']} | "
                    f"{family['trials']} | {family['label_counts'].get('left', 0)} | "
                    f"{family['label_counts'].get('right', 0)} | "
                    f"{','.join(map(str, family['channel_counts']))} | "
                    f"{','.join(map(str, family['sampling_rates_hz']))} Hz | "
                    f"{','.join(map(str, family['trial_durations_seconds']))} s |"
                ),
                "",
                "## 受试者明细",
                "",
                "| 受试者 | trial | 左手 | 右手 | runs | NPZ 大小 |",
                "|---|---:|---:|---:|---|---:|",
            ]
        )
        for row in inventory["physionet_subject_files"]:
            lines.append(
                f"| S{row['subject']:03d} | {row['trials']} | "
                f"{row['label_counts'].get('left', 0)} | {row['label_counts'].get('right', 0)} | "
                f"{','.join(map(str, row['runs']))} | {row['size_bytes'] / 1024**2:.2f} MiB |"
            )

    lines.extend(["", "## 不计入左右手 MI 的文件", ""])
    if inventory["other_npz_not_counted_as_left_right_mi"]:
        for row in inventory["other_npz_not_counted_as_left_right_mi"]:
            lines.append(f"- `{row['file']}`：格式或任务不是当前标准左右手 MI trial。")
    else:
        lines.append("- 无。")
    lines.extend(["", "## 实验资产", ""])
    lines.append(f"- 结果报告：{len(inventory['experiment_reports'])} 个。")
    lines.append(f"- 已保存模型：{len(inventory['saved_models'])} 个。")
    lines.append(f"- 不完整下载：{len(inventory['incomplete_files'])} 个。")
    for item in inventory["incomplete_files"]:
        lines.append(f"- 不完整文件：`{item}`，不可用于实验。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    inventory = build_inventory(args.data_dir)
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(inventory, args.markdown_report)
    family = inventory["mi_families"][0] if inventory["mi_families"] else None
    if family is None:
        print("No standardized left/right MI dataset found")
    else:
        print(
            f"MI families=1 subjects={family['subjects']} raw_edf={family['raw_edf_files']} "
            f"trials={family['trials']} labels={family['label_counts']}"
        )
    print(f"Saved: {args.json_report.resolve()}")
    print(f"Saved: {args.markdown_report.resolve()}")


if __name__ == "__main__":
    main()
