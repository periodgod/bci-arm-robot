# -*- coding: utf-8 -*-
"""Pure business concepts: configuration, signal semantics and decisions."""

from .eeg8_model import (
    EEG8_MODEL_FORMAT_VERSION,
    LEGACY_EEG8_RETRAIN_MESSAGE,
    EEG8Model,
    EEG8ModelLoadError,
    LegacyEEG8ModelError,
)

__all__ = [
    "EEG8_MODEL_FORMAT_VERSION",
    "LEGACY_EEG8_RETRAIN_MESSAGE",
    "EEG8Model",
    "EEG8ModelLoadError",
    "LegacyEEG8ModelError",
]
