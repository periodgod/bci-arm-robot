import unittest

import numpy as np

from arm_bci.core import (
    PhaseEmgEventEngine,
    aux_stream_watchdog_expired,
    resolve_continuous_stop_window,
)


class PhaseEmgEventEngineTests(unittest.TestCase):
    FS = 100
    BASELINE = [0.05, 0.05, 0.05, 0.05]

    def _aux(self, samples: int = 200) -> np.ndarray:
        return np.zeros((4, samples), dtype=np.float64)

    @staticmethod
    def _pulse(aux: np.ndarray, channel: int, start: int, width: int = 10, amplitude: float = 8.0) -> None:
        aux[channel, start : start + width] += np.hanning(width) * amplitude

    def _engine(self, **kwargs) -> PhaseEmgEventEngine:
        return PhaseEmgEventEngine(
            fs=self.FS,
            baseline_rms=self.BASELINE,
            blink_ratio=4.0,
            bite_ratio=3.0,
            side_ratio=1.35,
            rms_window_seconds=0.18,
            rms_hop_seconds=0.09,
            blink_min_gap_seconds=0.18,
            masseter_min_gap_seconds=0.25,
            **kwargs,
        )

    def test_repeated_tick_with_same_phase_buffer_does_not_repeat_event(self):
        aux = self._aux(120)
        self._pulse(aux, 2, 30)
        engine = self._engine()
        engine.reset("eeg", 0.0)

        first = engine.update(aux, 1.20)
        repeated = engine.update(aux, 1.20)

        self.assertEqual(first.left_jaw, 1)
        self.assertEqual(repeated.left_jaw, 1)
        self.assertEqual(repeated.right_jaw, 0)

    def test_masseter_uses_250ms_debounce_after_a_release_edge(self):
        aux = self._aux(120)
        # Pulses are separated by less than the 250 ms debounce interval.
        self._pulse(aux, 2, 20)
        self._pulse(aux, 2, 43)
        engine = self._engine()
        engine.reset("confirm", 0.0)

        snapshot = engine.update(aux, 1.20)

        self.assertEqual(snapshot.left_jaw, 1)
        self.assertEqual(snapshot.right_jaw, 0)

    def test_blink_requires_both_eye_channels_in_the_same_frame(self):
        one_eye = self._aux(100)
        self._pulse(one_eye, 0, 30)
        engine = self._engine()
        engine.reset("monitor", 0.0)
        self.assertEqual(engine.update(one_eye, 1.0).blink, 0)

        both_eyes = self._aux(100)
        self._pulse(both_eyes, 0, 30)
        self._pulse(both_eyes, 1, 30)
        engine.reset("monitor", 1.0)
        self.assertEqual(engine.update(both_eyes, 2.0).blink, 1)

    def test_left_right_and_dual_jaw_share_one_edge_contract(self):
        left = self._aux(100)
        self._pulse(left, 2, 30)
        engine = self._engine()
        engine.reset("confirm", 0.0)
        self.assertEqual(engine.update(left, 1.0).as_counts(), {"blink": 0, "left_jaw": 1, "right_jaw": 0})

        right = self._aux(100)
        self._pulse(right, 3, 30)
        engine.reset("confirm", 1.0)
        self.assertEqual(engine.update(right, 2.0).as_counts(), {"blink": 0, "left_jaw": 0, "right_jaw": 1})

        dual = self._aux(100)
        self._pulse(dual, 2, 30)
        self._pulse(dual, 3, 30)
        engine.reset("confirm", 2.0)
        snapshot = engine.update(dual, 3.0)
        self.assertEqual(snapshot.left_jaw, 1)
        self.assertEqual(snapshot.right_jaw, 1)
        self.assertEqual(snapshot.both_jaw, 1)
        self.assertEqual(snapshot.jaw_executions, 1)

    def test_two_separate_bites_are_two_window2_executions(self):
        aux = self._aux(120)
        self._pulse(aux, 2, 20)
        self._pulse(aux, 2, 60)
        engine = self._engine()
        engine.reset("confirm", 0.0)

        snapshot = engine.update(aux, 1.2)

        self.assertEqual(snapshot.jaw_executions, 2)

    def test_phase_switch_discards_completed_counts_and_confirmation_history(self):
        eeg_aux = self._aux(100)
        self._pulse(eeg_aux, 2, 30)
        engine = self._engine()
        engine.reset("eeg", 0.0)
        self.assertEqual(engine.update(eeg_aux, 1.0).left_jaw, 1)

        # Confirmation starts with a new phase-local slice.
        engine.reset("confirm", 1.0)
        confirmation_aux = self._aux(100)
        snapshot = engine.update(confirmation_aux, 2.0)
        self.assertEqual(snapshot.phase, "confirm")
        self.assertEqual(snapshot.as_counts(), {"blink": 0, "left_jaw": 0, "right_jaw": 0})

    def test_window3_uses_independent_fixed_segments(self):
        first_aux = self._aux(200)
        for start in (120, 170):
            self._pulse(first_aux, 0, start)
            self._pulse(first_aux, 1, start)
        engine = self._engine()
        engine.reset("continuous_move", 0.0)
        first = engine.update(first_aux, 2.0)
        self.assertEqual(first.blink, 2)
        self.assertEqual(
            resolve_continuous_stop_window(first.blink, 3, 2.0, 2.0)[0],
            "next_window",
        )

        # The next segment starts at zero; its blink is not added to the prior two.
        engine.reset("continuous_move", 2.0)
        second_aux = self._aux(100)
        self._pulse(second_aux, 0, 20)
        self._pulse(second_aux, 1, 20)
        second = engine.update(second_aux, 3.0)
        self.assertEqual(second.blink, 1)
        self.assertEqual(
            resolve_continuous_stop_window(second.blink, 3, 1.0, 2.0)[0],
            "wait",
        )

    def test_window3_waits_for_segment_boundary_after_blink_target(self):
        action, reason = resolve_continuous_stop_window(3, 3, 1.4, 2.0)
        self.assertEqual(action, "wait")
        self.assertEqual(reason, "window3_blink_target_pending_boundary_3_of_3")
        boundary_action, boundary_reason = resolve_continuous_stop_window(
            3, 3, 2.0, 2.0
        )
        self.assertEqual(boundary_action, "stop")
        self.assertEqual(
            boundary_reason,
            "window3_segment_complete_with_blink_target_3_of_3",
        )

    def test_aux_watchdog_fails_closed_for_missing_or_stale_samples(self):
        self.assertTrue(aux_stream_watchdog_expired(0.0, 10.0, 0.5))
        self.assertFalse(aux_stream_watchdog_expired(9.7, 10.0, 0.5))
        self.assertTrue(aux_stream_watchdog_expired(9.4, 10.0, 0.5))


if __name__ == "__main__":
    unittest.main()
