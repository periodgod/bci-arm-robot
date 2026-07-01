import unittest
import os
import tempfile

import numpy as np

from arm_bci_system import (
    BinaryMIDeploymentModel,
    DEFAULT_EEG_CHANNELS,
    brainvision_lsl_aux_indices,
    extract_timestamped_epochs,
    fuse_masseter_and_binary_mi,
    inspect_recording_npz,
    load_calibration_npz_many,
    normalize_binary_mi_hand,
    timestamped_recording_path,
)


class FixedClassifier:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, features):
        return np.tile(np.asarray([[0.75, 0.25]], dtype=np.float64), (len(features), 1))


class BinaryMIRuntimeTests(unittest.TestCase):
    def test_hand_aliases(self):
        self.assertEqual(normalize_binary_mi_hand("left"), "left_hand_mi")
        self.assertEqual(normalize_binary_mi_hand("right_hand"), "right_hand_mi")
        self.assertEqual(normalize_binary_mi_hand("左手想象"), "left_hand_mi")

    def test_masseter_mi_fusion_map(self):
        expected = {
            ("left_masseter", "left_hand_mi"): "left",
            ("left_masseter", "right_hand_mi"): "right",
            ("right_masseter", "left_hand_mi"): "forward",
            ("right_masseter", "right_hand_mi"): "backward",
            ("both_masseter", "left_hand_mi"): "up",
            ("both_masseter", "right_hand_mi"): "down",
        }
        for inputs, intent in expected.items():
            with self.subTest(inputs=inputs):
                self.assertEqual(fuse_masseter_and_binary_mi(*inputs), intent)

    def test_invalid_fusion_input(self):
        with self.assertRaises(ValueError):
            fuse_masseter_and_binary_mi("blink", "left_hand_mi")

    def test_brainvision_layout_keeps_32_eeg_channels(self):
        self.assertEqual(brainvision_lsl_aux_indices(36), [32, 33, 34, 35])
        self.assertEqual(brainvision_lsl_aux_indices(43), [32, 33, 34, 35])
        with self.assertRaises(RuntimeError):
            brainvision_lsl_aux_indices(35)

    def test_old_hybrid_model_has_clear_error(self):
        with self.assertRaisesRegex(TypeError, "不是二分类群 CSP 部署包"):
            BinaryMIDeploymentModel(object())

    def test_group_csp_bundle_predicts(self):
        bundle = {
            "model_type": "binary_mi_reflection_group_csp_lda",
            "sampling_rate_hz": 160,
            "window_samples": 480,
            "input_channel_names": ["C3", "Cz", "C4"],
            "selected_channel_indices": [0, 1, 2],
            "spatial_filters": np.asarray([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
            "classifier": FixedClassifier(),
            "labels": ["left_hand_mi", "right_hand_mi"],
        }
        model = BinaryMIDeploymentModel(bundle)
        eeg = np.random.default_rng(7).normal(size=(3, 480)).astype(np.float32)
        result = model.predict(eeg, ["C3", "Cz", "C4"], 160)
        self.assertEqual(result.hand, "left_hand_mi")
        self.assertAlmostEqual(result.confidence, 0.75)

    def test_full_32_channel_bundle_predicts(self):
        channel_names = list(DEFAULT_EEG_CHANNELS)
        bundle = {
            "model_type": "binary_mi_reflection_group_csp_lda",
            "sampling_rate_hz": 250,
            "window_samples": 500,
            "input_channel_names": channel_names,
            "selected_channel_indices": list(range(len(channel_names))),
            "spatial_filters": np.asarray(
                [
                    np.eye(len(channel_names), dtype=np.float64)[0],
                    np.eye(len(channel_names), dtype=np.float64)[-1],
                ]
            ),
            "classifier": FixedClassifier(),
            "labels": ["left_hand_mi", "right_hand_mi"],
        }
        model = BinaryMIDeploymentModel(bundle)
        eeg = np.random.default_rng(11).normal(size=(len(channel_names), 500)).astype(np.float32)
        result = model.predict(eeg, channel_names, 250)
        self.assertEqual(result.hand, "left_hand_mi")
        self.assertAlmostEqual(result.confidence, 0.75)

    def test_timestamped_epoch_interpolation(self):
        fs = 10.0
        timestamps = np.arange(0.0, 3.0, 1.0 / fs)
        samples = np.stack((timestamps, 2.0 * timestamps), axis=1)
        epochs = extract_timestamped_epochs(samples, timestamps, [0.5, 1.5], fs, 1.0)
        self.assertEqual(epochs.shape, (2, 2, 10))
        np.testing.assert_allclose(epochs[0, 0], np.arange(0.5, 1.5, 0.1), atol=1e-6)

    def test_timestamped_recording_path_uses_day_folder(self):
        with tempfile.TemporaryDirectory() as root:
            path = timestamped_recording_path(root, timestamp=0.0)
            self.assertEqual(os.path.basename(os.path.dirname(path)), "1970-01-01")
            self.assertTrue(os.path.basename(path).endswith("_session.npz"))

    def test_multiple_real_recordings_can_be_merged(self):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for index in range(2):
                path = os.path.join(root, f"session_{index}.npz")
                np.savez_compressed(
                    path,
                    eeg=np.full((2, 3, 20), index, dtype=np.float32),
                    aux=np.full((2, 4, 20), index, dtype=np.float32),
                    y=np.asarray(["left", "right"], dtype=object),
                    fs=np.asarray(250),
                    channel_names=np.asarray(["C3", "Cz", "C4"], dtype=object),
                    mode="brainvision_lsl",
                    completed=np.asarray(True),
                )
                paths.append(path)
            meta = inspect_recording_npz(paths[0])
            self.assertEqual(meta["eeg_channels"], 3)
            self.assertEqual(meta["aux_channels"], 4)
            eeg, y, aux, fs, names = load_calibration_npz_many(paths)
            self.assertEqual(eeg.shape, (4, 3, 20))
            self.assertEqual(aux.shape, (4, 4, 20))
            self.assertEqual(len(y), 4)
            self.assertEqual(fs, 250)
            self.assertEqual(names, ["C3", "Cz", "C4"])


if __name__ == "__main__":
    unittest.main()
