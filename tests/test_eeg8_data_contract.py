import json
import os
import tempfile
import unittest

import numpy as np

from arm_bci.core import DecoderConfig, SignalConfig, make_windows_from_trials, validate_eeg8_runtime_window_contract
from arm_bci.eeg8_system import (
    EEG8_INFERRED_CHANNEL_NAMES_TRAINING_ERROR,
    load_eeg8_recordings,
    train_eeg8_model,
    validate_eeg8_training_data_contract,
)


class EEG8DataContractTests(unittest.TestCase):
    FS = 250
    SAMPLES = 3000

    def _records_json(self) -> str:
        return json.dumps(
            [
                {
                    "class_id": 1,
                    "baseline_start_ms": 0,
                    "baseline_duration_s": 5.0,
                    "cue_onset_ms": 5000,
                    "task_start_ms": 5000,
                    "task_end_ms": 10000,
                    "trial_end_ms": 10000,
                    "bad_trial": False,
                    "is_practice": False,
                }
            ],
            ensure_ascii=False,
        )

    def _write_recording(
        self,
        directory: str,
        name: str,
        channel_count: int,
        *,
        legacy_empty_aux_indices: bool = False,
        eeg_names=None,
        raw_eeg_names=None,
        fs: int = FS,
        payload_overrides=None,
        omitted_keys=(),
    ) -> str:
        raw = np.zeros((self.SAMPLES, channel_count), dtype=np.float32)
        raw_names = list(raw_eeg_names or [f"E{index + 1}" for index in range(32)]) + [
            f"Aux{index + 1}" for index in range(max(0, channel_count - 32))
        ]
        aux_indices = [] if legacy_empty_aux_indices else list(range(32, min(36, channel_count)))
        aux = (
            raw[:, aux_indices].T.copy()
            if aux_indices
            else np.zeros((0, 0), dtype=np.float32)
        )
        path = os.path.join(directory, name)
        payload = {
            "raw_stream": raw,
            "raw_timestamps": np.arange(self.SAMPLES, dtype=np.float64) / fs,
            "raw_channel_names": np.asarray(raw_names, dtype=object),
            "channel_names": np.asarray(eeg_names or raw_names[:32], dtype=object),
            "aux": aux,
            "aux_channel_names": np.asarray(
                [raw_names[index] for index in aux_indices], dtype=object
            ),
            "aux_indices": np.asarray(aux_indices, dtype=np.int64),
            "recording_contract_version": np.asarray(
                1 if legacy_empty_aux_indices else 2, dtype=np.int64
            ),
            "fs": np.asarray(fs, dtype=np.int64),
            "trial_records_json": np.asarray(self._records_json()),
        }
        if not legacy_empty_aux_indices:
            payload.update(
                {
                    "eeg_indices": np.asarray(list(range(32)), dtype=np.int64),
                    "eeg_channel_count": np.asarray(32, dtype=np.int64),
                    "aux_channel_count": np.asarray(len(aux_indices), dtype=np.int64),
                    "masseter_gate_capable": np.asarray(channel_count == 36, dtype=bool),
                    "raw_channel_count": np.asarray(channel_count, dtype=np.int64),
                }
            )
        payload.update(payload_overrides or {})
        for key in omitted_keys:
            payload.pop(key, None)
        np.savez_compressed(path, **payload)
        return path

    def test_32_to_36_channel_recordings_preserve_available_aux_and_gate_status(self):
        with tempfile.TemporaryDirectory() as directory:
            for channel_count in range(32, 37):
                with self.subTest(channel_count=channel_count):
                    path = self._write_recording(directory, f"recording_{channel_count}.npz", channel_count)
                    eeg, labels, aux, fs, eeg_names, metadata = load_eeg8_recordings(
                        [path], return_metadata=True
                    )
                    expected_aux = channel_count - 32
                    self.assertEqual(eeg.shape[1], 32)
                    self.assertEqual(labels.tolist(), [0])
                    self.assertEqual(fs, self.FS)
                    self.assertEqual(len(eeg_names), 32)
                    self.assertEqual(metadata["aux_channel_count"], expected_aux)
                    self.assertEqual(metadata["aux_indices"], list(range(32, channel_count)))
                    if expected_aux:
                        self.assertIsNotNone(aux)
                        self.assertEqual(aux.shape[1], expected_aux)
                    else:
                        self.assertIsNone(aux)
                    self.assertEqual(metadata["masseter_gate_capable"], channel_count == 36)
                    if channel_count < 36:
                        self.assertIn("不具备咬肌门控条件", metadata["masseter_gate_status"])

    def test_historical_v1_empty_aux_indices_imports_trailing_raw_aux_without_claiming_full_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_recording(
                directory, "legacy_35_channels.npz", 35, legacy_empty_aux_indices=True
            )
            _eeg, _labels, aux, _fs, _names, metadata = load_eeg8_recordings([path], return_metadata=True)
            self.assertIsNotNone(aux)
            self.assertEqual(aux.shape[1], 3)
            self.assertEqual(metadata["aux_indices"], [32, 33, 34])
            self.assertFalse(metadata["masseter_gate_capable"])
            self.assertIn("AUX4", metadata["masseter_gate_status"])

    def test_v1_without_real_eeg_names_is_read_only_and_rejected_by_training_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_recording(
                directory,
                "legacy_without_eeg_names.npz",
                36,
                legacy_empty_aux_indices=True,
                omitted_keys=("channel_names", "raw_channel_names"),
            )
            _eeg, _labels, _aux, _fs, names, metadata = load_eeg8_recordings(
                [path], return_metadata=True
            )
            self.assertEqual(names, [f"CH{index + 1}" for index in range(32)])
            self.assertTrue(metadata["eeg_channel_names_inferred"])
            with self.assertRaisesRegex(RuntimeError, "缺少真实 32 路 EEG 通道名"):
                validate_eeg8_training_data_contract(metadata)

    def test_v1_with_real_eeg_names_remains_eligible_for_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_recording(
                directory,
                "legacy_with_eeg_names.npz",
                36,
                legacy_empty_aux_indices=True,
            )
            *_result, metadata = load_eeg8_recordings([path], return_metadata=True)
            self.assertFalse(metadata["eeg_channel_names_inferred"])
            validate_eeg8_training_data_contract(metadata)

    def test_cross_file_channel_order_and_sampling_rate_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._write_recording(directory, "first.npz", 36)
            names = [f"E{index + 1}" for index in range(32)]
            names[0], names[1] = names[1], names[0]
            wrong_order = self._write_recording(
                directory, "wrong_order.npz", 36, eeg_names=names, raw_eeg_names=names
            )
            with self.assertRaisesRegex(ValueError, "EEG 通道名称或顺序不一致"):
                load_eeg8_recordings([first, wrong_order])

            wrong_fs = self._write_recording(directory, "wrong_fs.npz", 36, fs=500)
            with self.assertRaisesRegex(ValueError, "采样率不一致"):
                load_eeg8_recordings([first, wrong_fs])

    def test_v2_requires_exact_eeg_metadata_instead_of_defaulting_to_first_32_channels(self):
        cases = (
            ("wrong_eeg_indices", {"eeg_indices": np.asarray(list(range(31)) + [32])}, (), "eeg_indices"),
            ("wrong_eeg_count", {"eeg_channel_count": np.asarray(31)}, (), "eeg_channel_count"),
            ("wrong_raw_count", {"raw_channel_count": np.asarray(35)}, (), "raw_channel_count"),
            ("missing_eeg_indices", {}, ("eeg_indices",), "缺少必填元数据 eeg_indices"),
            ("missing_eeg_count", {}, ("eeg_channel_count",), "缺少必填元数据 eeg_channel_count"),
            ("missing_raw_count", {}, ("raw_channel_count",), "缺少必填元数据 raw_channel_count"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, overrides, omitted_keys, error in cases:
                with self.subTest(name=name):
                    path = self._write_recording(
                        directory,
                        f"{name}.npz",
                        36,
                        payload_overrides=overrides,
                        omitted_keys=omitted_keys,
                    )
                    with self.assertRaisesRegex(ValueError, error):
                        load_eeg8_recordings([path])

    def test_v2_rejects_channel_names_and_declared_aux_gate_metadata_that_disagree_with_raw_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            bad_names = self._write_recording(
                directory,
                "bad_names.npz",
                36,
                payload_overrides={"channel_names": np.asarray(["E2", "E1"] + [f"E{index}" for index in range(3, 33)], dtype=object)},
            )
            with self.assertRaisesRegex(ValueError, "channel_names 与 raw_channel_names"):
                load_eeg8_recordings([bad_names])

            bad_aux_count = self._write_recording(
                directory, "bad_aux_count.npz", 36, payload_overrides={"aux_channel_count": np.asarray(3)}
            )
            with self.assertRaisesRegex(ValueError, "aux_channel_count 与 aux_indices"):
                load_eeg8_recordings([bad_aux_count])

            bad_gate = self._write_recording(
                directory, "bad_gate.npz", 36, payload_overrides={"masseter_gate_capable": np.asarray(False)}
            )
            with self.assertRaisesRegex(ValueError, "masseter_gate_capable 与实际 AUX 契约"):
                load_eeg8_recordings([bad_gate])

    def test_training_and_online_window_contract_is_configurable_but_must_match(self):
        cfg = SignalConfig(fs=self.FS)
        self.assertEqual((cfg.window_seconds, cfg.step_seconds, cfg.window_samples), (2.0, 2.0, 500))
        trial = np.zeros((1, 32, self.FS * 5), dtype=np.float32)
        windows, labels, _aux = make_windows_from_trials(trial, np.asarray([0]), cfg)
        self.assertEqual(windows.shape[-1], cfg.window_samples)
        self.assertTrue(np.all(labels == 0))
        validate_eeg8_runtime_window_contract(cfg, 2.0, 2.0)
        validate_eeg8_runtime_window_contract(
            SignalConfig(fs=self.FS, window_seconds=3.0, step_seconds=3.0),
            3.0,
            3.0,
        )
        with self.assertRaisesRegex(ValueError, "模型窗口不兼容"):
            validate_eeg8_runtime_window_contract(SignalConfig(fs=self.FS, window_seconds=5.0, step_seconds=5.0), 2.0, 2.0)

    def test_training_rejects_partial_label_sets_instead_of_creating_a_non_runtime_model(self):
        cfg = SignalConfig(fs=self.FS)
        eeg = np.zeros((7, 32, cfg.window_samples), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "必须同时包含 C1-C8"):
            train_eeg8_model(
                eeg,
                np.arange(7, dtype=np.int64),
                None,
                cfg,
                DecoderConfig(),
                "logreg",
                run_cv=False,
                data_contract={"eeg_channel_names_inferred": False},
            )

    def test_training_api_requires_data_contract(self):
        cfg = SignalConfig(fs=self.FS)
        eeg = np.zeros((8, 32, cfg.window_samples), dtype=np.float32)
        with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'data_contract'"):
            train_eeg8_model(
                eeg,
                np.arange(8, dtype=np.int64),
                None,
                cfg,
                DecoderConfig(),
                "logreg",
                run_cv=False,
            )

    def test_training_api_rejects_inferred_channel_names_before_fitting(self):
        cfg = SignalConfig(fs=self.FS)
        eeg = np.zeros((8, 32, cfg.window_samples), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "缺少真实 32 路 EEG 通道名") as caught:
            train_eeg8_model(
                eeg,
                np.arange(8, dtype=np.int64),
                None,
                cfg,
                DecoderConfig(),
                "logreg",
                run_cv=False,
                data_contract={"eeg_channel_names_inferred": True},
            )
        self.assertEqual(str(caught.exception), EEG8_INFERRED_CHANNEL_NAMES_TRAINING_ERROR)


if __name__ == "__main__":
    unittest.main()
