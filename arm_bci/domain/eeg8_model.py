# -*- coding: utf-8 -*-
"""EEG8 classifier used by the active collection and online-control workflow."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import joblib
import numpy as np


EEG8_MODEL_FORMAT_VERSION = 1
"""Persisted-format version for the only supported 2 s EEG8 runtime model."""


class EEG8ModelLoadError(RuntimeError):
    """A safe, operator-facing error while opening a persisted model."""


class LegacyEEG8ModelError(EEG8ModelLoadError):
    """Raised when an archived Hybrid/unknown model reaches the EEG8 runtime."""


LEGACY_EEG8_RETRAIN_MESSAGE = (
    "旧模型不可用于当前 EEG8 运行时，请用当前 2s 窗口重新训练。"
)


def _legacy_model_error(path: str) -> LegacyEEG8ModelError:
    """Return one stable error without exposing pickle/joblib internals."""
    name = os.path.basename(str(path or "模型文件")) or "模型文件"
    return LegacyEEG8ModelError(
        f"无法使用模型“{name}”：检测到旧 HybridBCI 或不兼容的历史模型格式。"
        f"{LEGACY_EEG8_RETRAIN_MESSAGE}"
    )


class EEG8Model:
    """Persisted EEG decoder with AUX artifact baselines."""

    model_format_version = EEG8_MODEL_FORMAT_VERSION

    def __init__(self, signal_config, decoder_config):
        # Persist format metadata with the joblib instance.
        self.model_format_version = EEG8_MODEL_FORMAT_VERSION
        self.signal_config = signal_config
        self.decoder_config = decoder_config
        self.eeg_features = None
        self.eeg_classifier = None
        self.eeg_probability_order_flipped_ = False
        self.classifier_kind = "logreg"
        self.metrics_: Dict = {}
        self.training_strategy_: Dict = {}
        self.paradigm_key = "eeg8"
        self.eeg_abs_center_ = 0.0
        self.eeg_abs_mad_ = 1.0
        self.aux_abs_center_ = 0.0
        self.aux_abs_mad_ = 1.0

    def _fit_core(self, eeg_windows: np.ndarray, y_windows: np.ndarray, aux_windows=None) -> None:
        from ..core import FilterBankEEGFeatures, make_classifier

        self._fit_artifact_baseline(eeg_windows, aux_windows)
        if str(self.classifier_kind).strip().lower() == "best_mi_fusion":
            from .mi_fusion import EEGWindowPreprocessor

            self.eeg_features = EEGWindowPreprocessor(self.signal_config)
        else:
            self.eeg_features = FilterBankEEGFeatures(self.signal_config)
        features = self.eeg_features.fit_transform(eeg_windows, y_windows)
        self.eeg_classifier = make_classifier(self.classifier_kind)
        self.eeg_classifier.fit(features, y_windows)
        self.eeg_probability_order_flipped_ = False
        classes = np.asarray(getattr(self.eeg_classifier, "classes_", []))
        if classes.size == 2 and hasattr(self.eeg_classifier, "predict_proba"):
            raw_proba = np.asarray(
                self.eeg_classifier.predict_proba(features), dtype=np.float64
            )
            hard_pred = np.asarray(self.eeg_classifier.predict(features))
            proba_pred = classes[np.argmax(raw_proba, axis=1)]
            # SVC's small-sample Platt calibration can be globally reversed
            # relative to its decision function. Persist one explicit flag so
            # CV, offline evaluation and online inference use one orientation.
            if hard_pred.size and float(np.mean(proba_pred == hard_pred)) < 0.5:
                self.eeg_probability_order_flipped_ = True

    def _fit_artifact_baseline(self, eeg_windows: np.ndarray, aux_windows=None) -> None:
        from ..core import robust_center_mad

        eeg_abs = np.percentile(np.abs(eeg_windows), 95, axis=(1, 2))
        self.eeg_abs_center_, self.eeg_abs_mad_ = robust_center_mad(eeg_abs)
        if aux_windows is not None and np.asarray(aux_windows).size:
            aux_abs = np.percentile(np.abs(aux_windows), 95, axis=(1, 2))
            self.aux_abs_center_, self.aux_abs_mad_ = robust_center_mad(aux_abs)

    def detect_artifact(self, eeg_window: np.ndarray, aux_window=None) -> Tuple[bool, str]:
        from ..core import ensure_2d_window

        eeg_window = ensure_2d_window(eeg_window, "eeg_window")
        if self._is_outlier(eeg_window, self.eeg_abs_center_, self.eeg_abs_mad_):
            return True, "eeg_artifact"
        if aux_window is not None and self._is_outlier(aux_window, self.aux_abs_center_, self.aux_abs_mad_):
            return True, "aux_artifact"
        return False, "clean"

    def _is_outlier(self, window: np.ndarray, center: float, mad: float) -> bool:
        threshold = float(getattr(self.decoder_config, "artifact_z_threshold", 9.0))
        if not np.isfinite(center) or not np.isfinite(mad) or center <= 0.0 or mad <= 1e-5:
            return False
        score = (np.percentile(np.abs(window), 95) - center) / mad
        return bool(score > threshold)

    def predict_eeg_proba(self, eeg_window_or_batch: np.ndarray) -> np.ndarray:
        from ..core import align_proba, crop_or_pad_window, ensure_3d_epochs

        if self.eeg_features is None or self.eeg_classifier is None:
            raise RuntimeError("模型尚未训练")
        eeg = ensure_3d_epochs(eeg_window_or_batch, "eeg")
        eeg = crop_or_pad_window(eeg, self.signal_config.window_samples)
        features = self.eeg_features.transform(eeg)
        raw_proba = np.asarray(
            self.eeg_classifier.predict_proba(features), dtype=np.float64
        )
        if (
            bool(getattr(self, "eeg_probability_order_flipped_", False))
            and raw_proba.ndim == 2
            and raw_proba.shape[1] == 2
        ):
            raw_proba = raw_proba[:, ::-1]
        return align_proba(raw_proba, self.eeg_classifier.classes_)

    def predict_proba(self, eeg_window: np.ndarray, aux_window=None) -> np.ndarray:
        """Compatibility call used by online code; AUX never changes EEG8 scores."""
        return self.predict_eeg_proba(eeg_window)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.model_format_version = EEG8_MODEL_FORMAT_VERSION
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "EEG8Model":
        from ..core import stabilize_gaussian_nb_estimators

        try:
            model = joblib.load(path)
        except Exception:
            # Old HybridBCI archives reference removed feature classes.
            raise _legacy_model_error(path) from None
        if not isinstance(model, EEG8Model):
            raise _legacy_model_error(path)
        # Require the format version in persisted instance state.
        if getattr(model, "__dict__", {}).get("model_format_version") != EEG8_MODEL_FORMAT_VERSION:
            raise _legacy_model_error(path)
        stabilize_gaussian_nb_estimators(model.eeg_classifier)
        return model
