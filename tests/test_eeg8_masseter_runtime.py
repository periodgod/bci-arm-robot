import unittest

import numpy as np

from arm_bci.core import (
    ACTION_COMMANDS,
    CONTINUOUS_MOVE_VECTORS,
    MasseterGateResult,
    MasseterTemplateBank,
    SignalConfig,
    apply_masseter_candidate_weight,
    brainvision_lsl_aux_indices,
    count_masseter_events,
    DecoderConfig,
    eeg8_command_is_continuous,
    eeg8_command_eligibility,
    make_masseter_template,
    masseter_binary_state,
    masseter_candidate_indices,
    select_binary_candidate,
    integrate_velocity_position,
    match_masseter_ncc,
    make_classifier,
    resolve_eeg8_confirmation_action,
)
from arm_bci.eeg8_system import CLASS_TO_ARM, prepare_eeg8_action_labels
from arm_bci.domain import EEG8Model
from arm_bci.gui import (
    build_eeg8_realtime_startup_payload,
    resolve_gui_masseter_detector_gate,
    select_gui_eeg8_confirmation_candidate,
)


class EEG8MasseterRuntimeTests(unittest.TestCase):
    def _aux_window(self, side: str, fs: int = 250) -> np.ndarray:
        rng = np.random.default_rng(42)
        aux = rng.normal(0.0, 0.05, size=(4, fs * 5)).astype(np.float32)
        start = fs * 2
        stop = start + int(fs * 0.25)
        if side == "left":
            aux[2, start:stop] += 5.0
        elif side == "right":
            aux[3, start:stop] += 5.0
        elif side == "both":
            aux[2, start:stop] += 5.0
            aux[3, start:stop] += 5.0
        return aux

    def test_ncc_template_matches_current_side(self):
        left = self._aux_window("left")
        right = self._aux_window("right")
        bank = MasseterTemplateBank(
            fs=250,
            template_seconds=1.0,
            states={
                "10": make_masseter_template([left], fs=250, template_seconds=1.0),
                "01": make_masseter_template([right], fs=250, template_seconds=1.0),
            },
        )
        gate = match_masseter_ncc(left, fs=250, template_bank=bank)
        self.assertEqual(gate.side, "10")
        self.assertEqual(gate.candidates, (4, 5))
        self.assertIn("ncc_masseter_binary_10", gate.reason)

        right_gate = match_masseter_ncc(right, fs=250, template_bank=bank)
        self.assertEqual(right_gate.side, "01")
        self.assertEqual(right_gate.candidates, (0, 1))

    def test_ncc_matches_when_action_moves_within_search_window(self):
        fs = 250
        rng = np.random.default_rng(7)

        def window(start_seconds: float) -> np.ndarray:
            aux = rng.normal(0.0, 0.03, size=(4, fs * 2)).astype(np.float32)
            start = int(round(start_seconds * fs))
            stop = start + int(round(0.22 * fs))
            pulse = np.hanning(max(2, stop - start)).astype(np.float32) * 5.0
            aux[2, start:stop] += pulse
            return aux

        training = window(0.25)
        online = window(1.35)
        bank = MasseterTemplateBank(
            fs=fs,
            template_seconds=0.4,
            states={
                "10": make_masseter_template([training], fs=fs, template_seconds=0.4),
                "01": make_masseter_template([self._aux_window("right", fs=fs)], fs=fs, template_seconds=0.4),
            },
        )
        gate = match_masseter_ncc(online, fs=fs, template_bank=bank)
        self.assertEqual(gate.side, "10")
        self.assertIn("offset_", gate.reason)

    def test_ncc_one_second_template_slides_over_complete_two_second_eeg_phase(self):
        fs = 250

        def phase(seconds: float, pulse_start: float, side: str) -> np.ndarray:
            aux = np.zeros((4, int(round(seconds * fs))), dtype=np.float32)
            start = int(round(pulse_start * fs))
            stop = min(aux.shape[1], start + int(round(0.22 * fs)))
            pulse = np.hanning(max(2, stop - start)).astype(np.float32) * 5.0
            if side == "left":
                aux[2, start:stop] = pulse
            else:
                aux[3, start:stop] = pulse
            return aux

        bank = MasseterTemplateBank(
            fs=fs,
            template_seconds=1.0,
            states={
                "10": make_masseter_template([phase(2.0, 0.55, "left")], fs=fs, template_seconds=1.0),
                "01": make_masseter_template([phase(2.0, 0.55, "right")], fs=fs, template_seconds=1.0),
            },
        )
        # Match the 1 s template late in a 2 s online phase.
        gate = match_masseter_ncc(phase(2.0, 1.25, "left"), fs=fs, template_bank=bank)

        self.assertEqual(gate.side, "10")
        self.assertIn("ncc_masseter_binary_10", gate.reason)
        self.assertIn("offset_", gate.reason)
        self.assertNotIn("offset_0.000s", gate.reason)

    def test_ncc_rejects_short_detection_slice_instead_of_stretching_it(self):
        fs = 250
        left = self._aux_window("left", fs=fs)
        right = self._aux_window("right", fs=fs)
        bank = MasseterTemplateBank(
            fs=fs,
            template_seconds=1.0,
            states={
                "10": make_masseter_template([left], fs=fs, template_seconds=1.0),
                "01": make_masseter_template([right], fs=fs, template_seconds=1.0),
            },
        )
        short_detection_slice = left[:, -int(round(0.18 * fs)) :]

        gate = match_masseter_ncc(short_detection_slice, fs=fs, template_bank=bank)

        self.assertEqual(gate.side, "none")
        self.assertEqual(gate.candidates, tuple())
        self.assertTrue(gate.reason.startswith("ncc_input_shorter_than_template_"))

    def test_ncc_matches_complete_one_second_confirmation_phase(self):
        fs = 250

        def confirmation_phase(side: str) -> np.ndarray:
            aux = np.zeros((4, fs), dtype=np.float32)
            start = int(round(0.35 * fs))
            stop = start + int(round(0.22 * fs))
            pulse = np.hanning(max(2, stop - start)).astype(np.float32) * 5.0
            aux[2 if side == "left" else 3, start:stop] = pulse
            return aux

        left = confirmation_phase("left")
        right = confirmation_phase("right")
        bank = MasseterTemplateBank(
            fs=fs,
            template_seconds=1.0,
            states={
                "10": make_masseter_template([left], fs=fs, template_seconds=1.0),
                "01": make_masseter_template([right], fs=fs, template_seconds=1.0),
            },
        )
        # A confirmation phase is exactly one template duration long.
        confirmation_aux = left

        gate = match_masseter_ncc(confirmation_aux, fs=fs, template_bank=bank)

        self.assertEqual(gate.side, "10")
        self.assertIn("ncc_masseter_binary_10", gate.reason)
        self.assertIn("offset_0.000s", gate.reason)

    def test_ncc_rejects_invalid_or_incompatible_template_metadata(self):
        fs = 250
        left = self._aux_window("left", fs=fs)
        right = self._aux_window("right", fs=fs)
        valid_states = {
            "10": make_masseter_template([left], fs=fs, template_seconds=1.0),
            "01": make_masseter_template([right], fs=fs, template_seconds=1.0),
        }
        cases = (
            (
                "invalid_bank_fs",
                MasseterTemplateBank(fs=0.0, template_seconds=1.0, states=valid_states),
                "ncc_invalid_template_bank_fs",
            ),
            (
                "invalid_duration",
                MasseterTemplateBank(fs=fs, template_seconds=float("nan"), states=valid_states),
                "ncc_invalid_template_duration",
            ),
            (
                "incompatible_samples",
                MasseterTemplateBank(
                    fs=fs,
                    template_seconds=1.0,
                    states={"10": np.zeros((2, 100)), "01": valid_states["01"]},
                ),
                "ncc_incompatible_template_10_samples_",
            ),
        )

        for name, bank, expected_reason in cases:
            with self.subTest(name=name):
                gate = match_masseter_ncc(left, fs=fs, template_bank=bank)
                self.assertEqual(gate.side, "none")
                self.assertEqual(gate.candidates, tuple())
                self.assertTrue(gate.reason.startswith(expected_reason))

    def test_ncc_explicitly_resamples_a_valid_calibration_template_to_online_fs(self):
        calibration_fs = 250
        online_fs = 500

        def phase(fs: int, side: str) -> np.ndarray:
            aux = np.zeros((4, fs * 2), dtype=np.float32)
            start = int(round(1.15 * fs))
            stop = start + int(round(0.22 * fs))
            pulse = np.hanning(max(2, stop - start)).astype(np.float32) * 5.0
            aux[2 if side == "left" else 3, start:stop] = pulse
            return aux

        bank = MasseterTemplateBank(
            fs=calibration_fs,
            template_seconds=1.0,
            states={
                "10": make_masseter_template([phase(calibration_fs, "left")], fs=calibration_fs),
                "01": make_masseter_template([phase(calibration_fs, "right")], fs=calibration_fs),
            },
        )

        gate = match_masseter_ncc(phase(online_fs, "left"), fs=online_fs, template_bank=bank)

        self.assertEqual(gate.side, "10")
        self.assertIn("template_fs_250_to_input_fs_500", gate.reason)

    def test_ncc_detector_mode_never_falls_back_to_threshold_gate(self):
        channel_gate = MasseterGateResult(
            "10", 1, 0, 0, 3.0, 0.0, (4, 5), "incremental_masseter_10"
        )
        unavailable_reasons = (
            "ncc_no_templates",
            "ncc_no_valid_templates",
            "no_aux",
            "missing_masseter_channels",
            "ncc_invalid_template_bank_fs",
            "ncc_incompatible_template_10_samples_100_expected_250",
            "ncc_input_shorter_than_template_45_of_250_samples",
        )

        for reason in unavailable_reasons:
            with self.subTest(reason=reason):
                ncc_gate = MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), reason)
                gate = resolve_gui_masseter_detector_gate("ncc", channel_gate, ncc_gate)
                self.assertEqual(gate.side, "none")
                self.assertEqual(gate.candidates, tuple())
                self.assertEqual(gate.reason, reason)

    def test_precise_ncc_required_state_never_falls_back_to_threshold_gate(self):
        channel_gate = MasseterGateResult(
            "10", 1, 0, 0, 3.0, 0.0, (4, 5), "incremental_masseter_10"
        )
        ncc_gate = MasseterGateResult(
            "none", 0, 0, 0, 0.0, 0.0, tuple(), "ncc_no_templates"
        )

        gate = resolve_gui_masseter_detector_gate(
            "precise", channel_gate, ncc_gate, {"10": "ncc"}
        )

        self.assertEqual(gate.side, "none")
        self.assertEqual(gate.candidates, tuple())
        self.assertEqual(gate.reason, "ncc_no_templates")

    def test_threshold_detector_mode_is_the_explicit_threshold_path(self):
        channel_gate = MasseterGateResult(
            "10", 1, 0, 0, 3.0, 0.0, (4, 5), "incremental_masseter_10"
        )
        ncc_gate = MasseterGateResult(
            "none", 0, 0, 0, 0.0, 0.0, tuple(), "ncc_no_templates"
        )

        gate = resolve_gui_masseter_detector_gate("threshold", channel_gate, ncc_gate)

        self.assertEqual(gate.side, "10")
        self.assertEqual(gate.candidates, (4, 5))

    def test_left_masseter_maps_to_execution_pair(self):
        gate = count_masseter_events(
            self._aux_window("left"),
            fs=250,
            baseline_rms=[1.0, 1.0, 1.0, 1.0],
            bite_ratio=2.0,
            side_ratio=1.2,
        )
        self.assertEqual(gate.side, "10")
        self.assertGreaterEqual(gate.left_count, 1)
        self.assertEqual(gate.candidates, (4, 5))
        self.assertEqual(gate.left_peak, gate.left_ratio)
        self.assertEqual(gate.right_peak, gate.right_ratio)

    def test_right_masseter_maps_to_imagery_pair(self):
        gate = count_masseter_events(
            self._aux_window("right"),
            fs=250,
            baseline_rms=[1.0, 1.0, 1.0, 1.0],
            bite_ratio=2.0,
            side_ratio=1.2,
        )
        self.assertEqual(gate.side, "01")
        self.assertGreaterEqual(gate.right_count, 1)
        self.assertEqual(gate.candidates, (0, 1))

    def test_masseter_binary_mapping_table(self):
        self.assertEqual(masseter_binary_state(0, 0), "00")
        self.assertEqual(masseter_binary_state(0, 1), "01")
        self.assertEqual(masseter_binary_state(1, 0), "10")
        self.assertEqual(masseter_binary_state(0, 0, both_count=1), "11")
        self.assertEqual(masseter_candidate_indices("00"), (2, 3))
        self.assertEqual(masseter_candidate_indices("01"), (0, 1))
        self.assertEqual(masseter_candidate_indices("10"), (4, 5))
        self.assertEqual(masseter_candidate_indices("11"), (6, 7))

    def test_masseter_mapping_reaches_expected_robot_commands(self):
        expected = {
            "00": ((3, "forward", "move_forward"), (4, "backward", "move_backward")),
            "01": ((1, "left", "move_left"), (2, "right", "move_right")),
            "10": ((5, "up", "move_up"), (6, "down", "move_down")),
            "11": ((7, "grab", "close_gripper"), (8, "release", "open_gripper")),
        }
        for state, rows in expected.items():
            class_ids = tuple(index + 1 for index in masseter_candidate_indices(state))
            self.assertEqual(class_ids, tuple(row[0] for row in rows))
            for class_id, intent, command in rows:
                self.assertEqual(CLASS_TO_ARM[class_id], intent)
                self.assertEqual(ACTION_COMMANDS[intent], command)

    def test_near_tied_binary_candidates_are_randomized_inside_mapping(self):
        probabilities = np.asarray([0.125, 0.123, 0.0, 0.0], dtype=np.float64)
        selected, second, randomized, gap = select_binary_candidate(
            probabilities, (0, 1), random_margin=0.05, chooser=lambda _items: 1
        )
        self.assertEqual(selected, 1)
        self.assertEqual(second, 0)
        self.assertTrue(randomized)
        self.assertAlmostEqual(gap, 0.002)

        selected, second, randomized, gap = select_binary_candidate(
            np.asarray([0.20, 0.10]), (0, 1), random_margin=0.05
        )
        self.assertEqual((selected, second), (0, 1))
        self.assertFalse(randomized)
        self.assertAlmostEqual(gap, 0.10)

    def test_gui_selection_randomizes_near_tie_before_confirmation(self):
        selected, second, reason = select_gui_eeg8_confirmation_candidate(
            np.asarray([0.125, 0.123, 0.0, 0.0], dtype=np.float64),
            (0, 1),
            "incremental_masseter_01",
            minimum_gap=0.05,
        )
        self.assertIn(selected, (0, 1))
        self.assertIn(second, (0, 1))
        self.assertNotEqual(selected, second)
        self.assertIn("mode_random", reason)

        pending_confirmation = None
        if selected is not None:
            pending_confirmation = {"class_id": selected + 1}
        self.assertEqual(pending_confirmation, {"class_id": selected + 1})

    def test_eeg8_safety_gate_uses_exactly_one_joint_window(self):
        # A legacy multi-window value must not delay the current paradigm.
        cfg = DecoderConfig(command_confidence=0.62, min_margin=0.08, stable_windows=4)

        accepted, reason = eeg8_command_eligibility(
            [("left", 0.80, 0.20, 1.0)], "left", cfg
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "eeg8_single_window_eligible")

        # Only the latest physical window participates in the decision.
        accepted, reason = eeg8_command_eligibility(
            [("left", 0.80, 0.20, 1.0), ("right", 0.70, 0.12, 2.0)], "right", cfg
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "eeg8_single_window_eligible")

        # Confidence and margin are reported values, not rejection thresholds.
        accepted, reason = eeg8_command_eligibility(
            [("left", 0.80, 0.20, 1.0), ("left", 0.61, 0.20, 2.0)], "left", cfg
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "eeg8_single_window_eligible")

        accepted, reason = eeg8_command_eligibility(
            [("left", 0.80, 0.20, 1.0), ("left", 0.70, 0.07, 2.0)], "left", cfg
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "eeg8_single_window_eligible")

        accepted, reason = eeg8_command_eligibility(
            [("left", 0.80, 0.20, 1.0), ("right", 0.70, 0.12, 2.0)], "left", cfg
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "eeg8_intent_mismatch")

    def test_eeg_only_confirmation_is_direct_and_has_no_second_execution(self):
        action, reason = resolve_eeg8_confirmation_action(
            "eeg8_two_phase_eeg_only",
            True,
            100.0,
            100.0,
            1.0,
            "none",
            "no_aux",
        )
        self.assertEqual((action, reason), ("execute", "eeg_only_direct_execute"))

        # A consumed confirmation cannot emit twice.
        action, reason = resolve_eeg8_confirmation_action(
            "eeg8_two_phase_eeg_only",
            False,
            None,
            100.1,
            1.0,
            "none",
            "no_aux",
        )
        self.assertEqual((action, reason), ("none", "no_pending_confirmation"))

    def test_masseter_confirmation_no_aux_cancels_at_fixed_deadline(self):
        # Confirmation timing uses monotonic values, not the LSL epoch.
        action, reason = resolve_eeg8_confirmation_action(
            "eeg8_two_phase_masseter",
            True,
            42.0,
            42.5,
            1.0,
            "none",
            "no_aux",
        )
        self.assertEqual(action, "wait")
        self.assertIn("awaiting_confirmation_count_0_of_2_no_aux", reason)

        action, reason = resolve_eeg8_confirmation_action(
            "eeg8_two_phase_masseter",
            True,
            42.0,
            43.0,
            1.0,
            "none",
            "missing_masseter_channels",
        )
        self.assertEqual(action, "cancel")
        self.assertEqual(
            reason,
            "confirmation_timeout_no_valid_masseter_missing_masseter_channels",
        )

        # Reject a positive gate after the deadline.
        action, reason = resolve_eeg8_confirmation_action(
            "eeg8_two_phase_masseter",
            True,
            42.0,
            43.01,
            1.0,
            "10",
            "incremental_masseter_10",
        )
        self.assertEqual(action, "cancel")
        self.assertEqual(
            reason,
            "confirmation_timeout_no_valid_masseter_incremental_masseter_10",
        )

    def test_runtime_continuous_motion_requires_masseter_stop_channel(self):
        self.assertTrue(eeg8_command_is_continuous("left", "eeg8_two_phase_masseter"))
        self.assertFalse(eeg8_command_is_continuous("left", "eeg8_two_phase_eeg_only"))
        self.assertFalse(eeg8_command_is_continuous("grab", "eeg8_two_phase_masseter"))

    def test_startup_payload_preserves_eeg_only_strategy(self):
        payload = build_eeg8_realtime_startup_payload(
            model=object(),
            source=object(),
            template_bank=None,
            runtime_strategy="eeg8_two_phase_eeg_only",
        )
        self.assertEqual(payload["runtime_strategy"], "eeg8_two_phase_eeg_only")

    def test_continuous_motion_keeps_fractional_positions(self):
        position = integrate_velocity_position(
            (4.0, 0.0, 3.0), (1.0, 0.0, 0.0), speed=0.5, dt=0.02
        )
        self.assertAlmostEqual(position[0], 4.01)
        self.assertAlmostEqual(position[1], 0.0)
        self.assertAlmostEqual(position[2], 3.0)

    def test_only_horizontal_intents_support_continuous_motion(self):
        self.assertEqual(
            set(CONTINUOUS_MOVE_VECTORS),
            {"forward", "backward", "left", "right"},
        )
        self.assertNotIn("up", CONTINUOUS_MOVE_VECTORS)
        self.assertNotIn("down", CONTINUOUS_MOVE_VECTORS)
        for intent, vector in CONTINUOUS_MOVE_VECTORS.items():
            self.assertIn(intent, ACTION_COMMANDS)
            self.assertEqual(len(vector), 3)
            self.assertGreater(sum(abs(value) for value in vector), 0.0)

    def test_masseter_candidate_weight_keeps_all_eight_classes(self):
        proba = np.asarray([0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20], dtype=np.float64)
        weighted = apply_masseter_candidate_weight(proba, masseter_candidate_indices("10"), outside_weight=0.3)
        self.assertEqual(weighted.shape, (8,))
        self.assertAlmostEqual(weighted[4], 0.50)
        self.assertAlmostEqual(weighted[5], 0.40)
        self.assertAlmostEqual(weighted[0], 0.27)
        self.assertAlmostEqual(weighted[1], 0.24)

    def test_missing_or_both_masseter_uses_binary_table(self):
        missing = count_masseter_events(None, fs=250)
        self.assertEqual(missing.side, "none")
        self.assertEqual(missing.candidates, tuple())
        self.assertEqual(masseter_candidate_indices("none"), tuple())

        no_templates = match_masseter_ncc(
            self._aux_window("left"), fs=250, template_bank=None
        )
        self.assertEqual(no_templates.side, "none")
        self.assertEqual(no_templates.candidates, tuple())
        self.assertEqual(no_templates.reason, "ncc_no_templates")

        both = count_masseter_events(
            self._aux_window("both"),
            fs=250,
            baseline_rms=[1.0, 1.0, 1.0, 1.0],
            bite_ratio=2.0,
            side_ratio=1.2,
        )
        self.assertEqual(both.side, "11")
        self.assertEqual(both.candidates, (6, 7))

    def test_eeg8_predict_proba_is_eeg_only_for_real_runtime(self):
        from arm_bci.core import DecoderConfig

        model = EEG8Model(SignalConfig(), DecoderConfig())
        eeg_proba = np.asarray([[0.05, 0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.20]], dtype=np.float64)
        model.predict_eeg_proba = lambda eeg: eeg_proba
        out = model.predict_proba(np.zeros((32, 1250), dtype=np.float32), np.zeros((4, 1250), dtype=np.float32))
        np.testing.assert_allclose(out, eeg_proba)

    def test_brainvision_layout_keeps_32_eeg_channels_and_four_aux(self):
        self.assertEqual(brainvision_lsl_aux_indices(32), [])
        self.assertEqual(brainvision_lsl_aux_indices(33), [32])
        self.assertEqual(brainvision_lsl_aux_indices(34), [32, 33])
        self.assertEqual(brainvision_lsl_aux_indices(35), [32, 33, 34])
        self.assertEqual(brainvision_lsl_aux_indices(36), [32, 33, 34, 35])
        self.assertEqual(brainvision_lsl_aux_indices(43), [32, 33, 34, 35])

    def test_c1_c8_label_contract_rejects_static_rest_and_derived_paths(self):
        y = np.asarray(list(range(8)), dtype=np.int64)

        keep, mapped, info = prepare_eeg8_action_labels(y)
        self.assertEqual(info["name"], "EEG8 C1-C8 动作分类")
        self.assertTrue(info["runtime_compatible"])
        self.assertEqual(mapped.tolist(), list(range(8)))
        self.assertTrue(np.all(keep))
        self.assertEqual(info["class_labels"][0], "C1 左移")
        self.assertEqual(info["class_labels"][7], "C8 释放")

        with self.assertRaisesRegex(ValueError, "必须是 C1-C8"):
            prepare_eeg8_action_labels(np.asarray([0, 8], dtype=np.int64))

    def test_configurable_classifier_backends_are_constructible(self):
        for kind in ["vote", "lda", "fbcsp_lda", "logreg", "svm", "extra_trees", "random_forest", "gaussian_nb"]:
            self.assertIsNotNone(make_classifier(kind))


if __name__ == "__main__":
    unittest.main()
