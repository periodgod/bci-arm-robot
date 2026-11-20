# -*- coding: utf-8 -*-
"""Small GUI-facing helpers for the EEG8 console.

The former command module exposed a second runtime with synthetic data, NPZ
replay, and direct RDA control. EEG8 collection, training, and online control
now live in the GUI workflow, so only helpers shared by its workers remain.
"""

from __future__ import annotations

from typing import Any, Dict


def apply_runtime_control_config(model: Any, decoder_config: Any | None = None) -> None:
    """Attach the selected decoder settings to a loaded EEG8 model."""
    cfg = decoder_config if decoder_config is not None else getattr(model, "decoder_config", None)
    if cfg is None:
        raise ValueError("模型缺少在线解码配置")
    model.decoder_config = cfg


def metrics_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable, user-facing subset of a training or test report."""
    keys = (
        "validation",
        "requested_folds",
        "n_classes",
        "window_count",
        "eeg_accuracy",
        "eeg_kappa",
        "overall_accuracy",
        "kappa",
        "runtime_compatible",
        "warning",
        "note",
    )
    return {key: metrics[key] for key in keys if key in metrics}
