import unittest
from dataclasses import asdict

from arm_bci.core import (
    DecoderConfig,
    EEG8_RUNTIME_STRATEGY_EEG_ONLY,
    EEG8_RUNTIME_STRATEGY_MASSETER,
    MasseterTemplateBank,
    SignalConfig,
    build_runtime_session_config,
    eeg8_command_is_continuous,
    freeze_decoder_config,
    migrate_legacy_eeg8_realtime_config,
    resolve_eeg8_confirmation_action,
    resolve_eeg8_fusion_policy,
    runtime_signal_contract_matches,
    runtime_session_robot_output,
    validate_eeg8_ncc_confirmation_timing,
)


class _Model:
    def __init__(self, decoder=None):
        self.signal_config = SignalConfig(fs=250, window_seconds=2.0, step_seconds=2.0)
        self.decoder_config = decoder or DecoderConfig(
            command_confidence=0.71,
            masseter_bite_ratio=2.8,
            continuous_move_speed=0.45,
        )
        self.version = "test-model-v1"


def _runtime_config(decoder=None):
    return {
        "runtime_strategy": EEG8_RUNTIME_STRATEGY_MASSETER,
        "model_path": "E:/models/eeg8.joblib",
        "decode_window_seconds": 2.0,
        "decode_step_seconds": 2.0,
        "use_config_thresholds": True,
        "decoder": decoder or DecoderConfig(
            command_confidence=0.83,
            masseter_bite_ratio=4.2,
            confirmation_masseter_count=4,
            continuous_move_speed=0.72,
            continuous_blink_count=5,
            masseter_state_detectors={"10": "ncc"},
        ),
        "aux_blink_ratio": 4.5,
        "aux_bite_ratio": 4.2,
        "aux_side_ratio": 1.6,
        "aux_blink_count": 3,
        "aux_left_count": 2,
        "aux_right_count": 1,
        "target": "frozen_target",
        "robot_enabled": True,
        "robot_host": "10.1.2.3",
        "robot_port": 5007,
        "robot_speed": 36.0,
    }


class RuntimeSessionConfigTests(unittest.TestCase):
    def test_config_source_is_deep_frozen_after_start(self):
        raw = _runtime_config()
        session = build_runtime_session_config(raw, _Model(), use_config_thresholds=True)

        raw["target"] = "changed_target"
        raw["robot_host"] = "192.168.99.99"
        raw["aux_blink_count"] = 9
        raw["decoder"].command_confidence = 0.10
        raw["decoder"].masseter_state_detectors["10"] = "threshold"

        self.assertEqual(session.decoder_config_source, "config")
        self.assertEqual(session.target, "frozen_target")
        self.assertEqual(session.robot_host, "10.1.2.3")
        self.assertEqual(session.aux_blink_count, 3)
        self.assertAlmostEqual(session.decoder_config.command_confidence, 0.83)
        self.assertEqual(session.decoder_config.confirmation_masseter_count, 4)
        self.assertEqual(session.decoder_config.continuous_blink_count, 5)
        self.assertEqual((session.decoder_config.stable_windows, session.decoder_config.history_windows), (1, 1))
        self.assertEqual(dict(session.decoder_config.masseter_state_detectors), {"10": "ncc"})
        with self.assertRaises(Exception):
            session.decoder_config.command_confidence = 0.10

    def test_model_source_uses_model_decoder_but_never_rewrites_signal_contract(self):
        model = _Model()
        raw = _runtime_config()
        raw["decode_window_seconds"] = 2.0
        raw["decode_step_seconds"] = 2.0
        session = build_runtime_session_config(raw, model, use_config_thresholds=False)

        self.assertEqual(session.decoder_config_source, "model")
        self.assertAlmostEqual(session.decoder_config.command_confidence, 0.71)
        self.assertEqual(session.decoder_config.stable_windows, 1)
        self.assertEqual(session.decoder_config.history_windows, 1)
        self.assertEqual((session.signal_fs, session.signal_window_seconds, session.signal_step_seconds), (250.0, 2.0, 2.0))
        self.assertAlmostEqual(model.signal_config.window_seconds, 2.0)

    def test_model_window_mismatch_is_rejected_instead_of_resampled(self):
        raw = _runtime_config()
        raw["decode_window_seconds"] = 5.0
        raw["decode_step_seconds"] = 5.0
        with self.assertRaisesRegex(
            ValueError, "模型窗口不兼容|当前 EEG8 在线策略只支持 2s EEG 窗口"
        ):
            build_runtime_session_config(raw, _Model(), use_config_thresholds=True)

    def test_fusion_policy_explicitly_declares_aux_confirmation_and_motion(self):
        masseter = resolve_eeg8_fusion_policy(EEG8_RUNTIME_STRATEGY_MASSETER)
        eeg_only = resolve_eeg8_fusion_policy(EEG8_RUNTIME_STRATEGY_EEG_ONLY)

        self.assertTrue(masseter.requires_aux)
        self.assertTrue(masseter.requires_confirmation)
        self.assertTrue(masseter.allows_continuous_motion)
        self.assertFalse(eeg_only.requires_aux)
        self.assertFalse(eeg_only.requires_confirmation)
        self.assertFalse(eeg_only.allows_continuous_motion)
        self.assertTrue(eeg8_command_is_continuous("left", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertTrue(eeg8_command_is_continuous("right", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertTrue(eeg8_command_is_continuous("forward", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertTrue(eeg8_command_is_continuous("backward", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertTrue(eeg8_command_is_continuous("up", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertTrue(eeg8_command_is_continuous("down", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertFalse(eeg8_command_is_continuous("grab", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertFalse(eeg8_command_is_continuous("release", EEG8_RUNTIME_STRATEGY_MASSETER))
        self.assertFalse(eeg8_command_is_continuous("left", EEG8_RUNTIME_STRATEGY_EEG_ONLY))

    def test_unknown_strategy_is_rejected_by_policy_and_confirmation(self):
        with self.assertRaisesRegex(ValueError, "未知 EEG8 在线策略"):
            resolve_eeg8_fusion_policy("unknown")
        action, reason = resolve_eeg8_confirmation_action(
            "unknown", True, 1.0, 1.1, 1.0, "10", "gate"
        )
        self.assertEqual((action, reason), ("cancel", "unknown_runtime_strategy"))
        self.assertFalse(eeg8_command_is_continuous("left", "unknown"))

    def test_ncc_template_must_fit_window1_joint_acquisition(self):
        ncc = DecoderConfig(
            masseter_detector="ncc",
            confirmation_window_seconds=1.0,
        )
        too_long = MasseterTemplateBank(fs=250.0, template_seconds=2.0, states={})
        with self.assertRaisesRegex(ValueError, "窗口1不兼容"):
            validate_eeg8_ncc_confirmation_timing(
                EEG8_RUNTIME_STRATEGY_MASSETER, ncc, too_long
            )

        usable = MasseterTemplateBank(fs=250.0, template_seconds=1.0, states={})
        validate_eeg8_ncc_confirmation_timing(
            EEG8_RUNTIME_STRATEGY_MASSETER, ncc, usable
        )

    def test_ncc_timing_does_not_reject_threshold_or_eeg_only(self):
        bank = MasseterTemplateBank(fs=250.0, template_seconds=1.0, states={})
        threshold = DecoderConfig(
            masseter_detector="threshold",
            confirmation_window_seconds=1.0,
        )
        validate_eeg8_ncc_confirmation_timing(
            EEG8_RUNTIME_STRATEGY_MASSETER, threshold, bank
        )

        ncc = DecoderConfig(
            masseter_detector="ncc",
            confirmation_window_seconds=1.0,
        )
        validate_eeg8_ncc_confirmation_timing(
            EEG8_RUNTIME_STRATEGY_EEG_ONLY, ncc, bank
        )

    def test_precise_ncc_confirmation_path_is_also_checked(self):
        precise_ncc = DecoderConfig(
            masseter_detector="precise",
            masseter_state_detectors={"10": "ncc"},
            confirmation_window_seconds=1.0,
        )
        bank = MasseterTemplateBank(fs=250.0, template_seconds=2.0, states={})
        with self.assertRaisesRegex(ValueError, "窗口1不兼容"):
            validate_eeg8_ncc_confirmation_timing(
                EEG8_RUNTIME_STRATEGY_MASSETER, precise_ncc, bank
            )

    def test_robot_endpoint_and_target_are_session_snapshots(self):
        raw = _runtime_config()
        session = build_runtime_session_config(raw, _Model(), use_config_thresholds=True)
        raw.update({
            "robot_enabled": False,
            "robot_host": "127.0.0.1",
            "robot_port": 9,
            "robot_speed": 1.0,
            "target": "changed_target",
        })

        self.assertEqual(
            runtime_session_robot_output(session),
            (True, "10.1.2.3", 5007, 36.0, "frozen_target"),
        )

    def test_decoder_mapping_round_trip_is_strict_and_preserves_detector_routes(self):
        decoder = DecoderConfig(
            command_confidence=0.79,
            masseter_detector="precise",
            masseter_state_detectors={"00": "threshold", "10": "ncc"},
        )
        frozen = freeze_decoder_config(asdict(decoder))
        self.assertEqual(frozen.command_confidence, 0.79)
        self.assertEqual(frozen.masseter_detector, "precise")
        self.assertEqual(dict(frozen.masseter_state_detectors), {"00": "threshold", "10": "ncc"})
        with self.assertRaisesRegex(ValueError, "未知字段"):
            freeze_decoder_config({**asdict(decoder), "unexpected": 1})
        with self.assertRaisesRegex(ValueError, "仅支持 threshold、ncc 或 precise"):
            freeze_decoder_config({**asdict(decoder), "masseter_detector": "bad"})
        with self.assertRaisesRegex(ValueError, "仅支持状态"):
            freeze_decoder_config({**asdict(decoder), "masseter_state_detectors": {"12": "ncc"}})
        with self.assertRaisesRegex(ValueError, "必须是有限数值"):
            freeze_decoder_config({**asdict(decoder), "command_confidence": "0.7"})

    def test_legacy_strategy_migrates_ui_contract_but_old_model_window_is_rejected(self):
        raw = _runtime_config()
        raw.update({
            "runtime_strategy": "eeg8_5s_masseter_gate",
            "decode_window_seconds": 3.0,
            "decode_step_seconds": 3.0,
        })
        migrated, notes = migrate_legacy_eeg8_realtime_config(raw)
        self.assertEqual(migrated["runtime_strategy"], EEG8_RUNTIME_STRATEGY_MASSETER)
        self.assertEqual((migrated["decode_window_seconds"], migrated["decode_step_seconds"]), (2.0, 2.0))
        self.assertTrue(notes)
        legacy_model = _Model()
        legacy_model.signal_config.window_seconds = 5.0
        legacy_model.signal_config.step_seconds = 5.0
        with self.assertRaisesRegex(ValueError, "模型窗口不兼容"):
            build_runtime_session_config(migrated, legacy_model, use_config_thresholds=True)

    def test_invalid_robot_fields_are_rejected_before_session_start(self):
        for key, value, message in (
            ("robot_enabled", "false", "robot_enabled 必须是布尔值"),
            ("robot_host", "", "robot_host 不能为空"),
            ("robot_port", 70000, "robot_port 必须是 1-65535"),
            ("robot_speed", 0.0, "robot_speed 不在允许范围内"),
            ("target", "", "target 不能为空"),
        ):
            with self.subTest(key=key):
                raw = _runtime_config()
                raw[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    build_runtime_session_config(raw, _Model(), use_config_thresholds=True)

    def test_signal_contract_drift_is_detected_after_session_start(self):
        model = _Model()
        session = build_runtime_session_config(_runtime_config(), model, use_config_thresholds=True)
        self.assertTrue(runtime_signal_contract_matches(session, model.signal_config))
        model.signal_config.filter_bands = ((8.0, 12.0), (14.0, 30.0))
        self.assertFalse(runtime_signal_contract_matches(session, model.signal_config))


if __name__ == "__main__":
    unittest.main()
