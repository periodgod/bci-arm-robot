import os
import tempfile
import unittest
from unittest.mock import patch

from arm_bci.core import DecoderConfig, SignalConfig
from arm_bci.domain import (
    EEG8_MODEL_FORMAT_VERSION,
    LEGACY_EEG8_RETRAIN_MESSAGE,
    EEG8Model,
    LegacyEEG8ModelError,
)


class EEG8ModelCompatibilityTests(unittest.TestCase):
    def test_new_model_persists_explicit_format_version(self):
        model = EEG8Model(SignalConfig(window_seconds=2.0, step_seconds=2.0), DecoderConfig())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "eeg8.joblib")
            model.save(path)
            loaded = EEG8Model.load(path)
        self.assertEqual(loaded.model_format_version, EEG8_MODEL_FORMAT_VERSION)

    def test_legacy_unpickle_attribute_error_becomes_safe_retrain_message(self):
        with patch(
            "arm_bci.domain.eeg8_model.joblib.load",
            side_effect=AttributeError("module arm_bci.core has no attribute HeadFaceAuxFeatures"),
        ):
            with self.assertRaises(LegacyEEG8ModelError) as caught:
                EEG8Model.load("old_hybrid.joblib")
        message = str(caught.exception)
        self.assertIn(LEGACY_EEG8_RETRAIN_MESSAGE, message)
        self.assertNotIn("HeadFaceAuxFeatures", message)
        self.assertNotIn("Traceback", message)

    def test_model_without_current_format_version_is_rejected(self):
        old_model = EEG8Model(SignalConfig(), DecoderConfig())
        del old_model.model_format_version
        with patch("arm_bci.domain.eeg8_model.joblib.load", return_value=old_model):
            with self.assertRaisesRegex(LegacyEEG8ModelError, "旧模型不可用于当前 EEG8 运行时"):
                EEG8Model.load("old_eeg8.joblib")


if __name__ == "__main__":
    unittest.main()
