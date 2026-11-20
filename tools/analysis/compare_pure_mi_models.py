# -*- coding: utf-8 -*-
"""Compare all GUI classifiers on one pure-MI recording and save the winner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from arm_bci.core import DecoderConfig, SignalConfig
from arm_bci.eeg8_system import load_eeg8_recordings, train_eeg8_model


CLASSIFIERS = (
    "vote",
    "lda",
    "logreg",
    "svm",
    "extra_trees",
    "random_forest",
    "gaussian_nb",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--output-dir", default="models/pure_mi_comparison")
    parser.add_argument("--online-model", default="models/arm_bci_gui.joblib")
    args = parser.parse_args()

    recording = os.path.abspath(args.recording)
    output_dir = os.path.abspath(args.output_dir)
    online_model = os.path.abspath(args.online_model)
    os.makedirs(output_dir, exist_ok=True)

    eeg, y, aux, fs, channel_names, contract = load_eeg8_recordings(
        [recording], return_metadata=True, log=print
    )
    results = []
    best = None
    best_model = None

    for use_csp in (False, True):
        for classifier in CLASSIFIERS:
            feature = "fbcsp" if use_csp else "bandpower"
            print(f"\n=== {feature} + {classifier} ===", flush=True)
            signal = SignalConfig(fs=fs, window_seconds=2.0, step_seconds=2.0)
            signal.eeg_channels = channel_names
            signal.aux_channel_names = list(contract.get("aux_channel_names", []))
            signal.masseter_gate_capable = bool(contract.get("masseter_gate_capable", False))
            signal.use_csp = use_csp
            model, metrics = train_eeg8_model(
                eeg,
                y,
                aux,
                signal,
                DecoderConfig(),
                classifier,
                data_contract=contract,
                paradigm_key="pure_mi_binary",
                run_cv=True,
            )
            row = {
                "feature": feature,
                "classifier": classifier,
                "accuracy": float(metrics.get("eeg_accuracy", 0.0)),
                "kappa": float(metrics.get("eeg_kappa", 0.0)),
                "class_accuracy": list(metrics.get("class_accuracy", [])),
                "fold_accuracy": list(metrics.get("fold_accuracy", [])),
                "confusion_matrix": metrics.get("confusion_matrix", []),
                "window_count": int(metrics.get("window_count", 0)),
            }
            model.training_strategy_.update(
                {
                    "source_recording": recording,
                    "feature": feature,
                    "selected_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            candidate = os.path.join(output_dir, f"pure_mi_{feature}_{classifier}.joblib")
            model.save(candidate)
            row["model_path"] = candidate
            results.append(row)
            score = (
                row["accuracy"],
                row["kappa"],
                min(row["class_accuracy"] or [0.0]),
            )
            if best is None or score > best:
                best = score
                best_model = (model, row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    assert best_model is not None
    winner_model, winner = best_model
    backup = ""
    if os.path.exists(online_model):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{online_model}.{stamp}.bak"
        shutil.copy2(online_model, backup)
    winner_model.save(online_model)

    report = {
        "recording": recording,
        "recording_trial_count": int(len(y)),
        "online_model": online_model,
        "online_model_backup": backup,
        "winner": winner,
        "results": sorted(
            results,
            key=lambda item: (
                item["accuracy"],
                item["kappa"],
                min(item["class_accuracy"] or [0.0]),
            ),
            reverse=True,
        ),
    }
    report_path = os.path.join(output_dir, "comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"\nREPORT={report_path}")
    print(f"ONLINE_MODEL={online_model}")
    print("WINNER=" + json.dumps(winner, ensure_ascii=False))


if __name__ == "__main__":
    main()
