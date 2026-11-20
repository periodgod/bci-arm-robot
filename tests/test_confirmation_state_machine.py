"""Tests for the EEG8 masseter confirmation state machine."""

import unittest

from arm_bci.core import (
    EEG8_RUNTIME_STRATEGY_MASSETER,
    resolve_continuous_stop_window,
    resolve_eeg8_confirmation_action,
)


class ConfirmationStateMachineTests(unittest.TestCase):
    START = 100.0
    WINDOW = 1.0

    def _resolve(
        self,
        pending: bool,
        now: float,
        side: str,
        reason: str,
        execution_count=None,
    ):
        return resolve_eeg8_confirmation_action(
            EEG8_RUNTIME_STRATEGY_MASSETER,
            pending,
            self.START,
            now,
            self.WINDOW,
            side,
            reason,
            masseter_execution_count=execution_count,
            required_masseter_executions=2,
        )

    def test_valid_phase_local_gate_executes_before_deadline(self):
        action, reason = self._resolve(
            True, 100.35, "01", "incremental_masseter_01", execution_count=2
        )

        self.assertEqual((action, reason), ("execute", "confirmed_by_masseter_count_2_of_2"))

    def test_no_gate_cancels_at_fixed_deadline(self):
        action, reason = self._resolve(
            True, 101.0, "none", "no_masseter_event", execution_count=0
        )

        self.assertEqual(action, "cancel")
        self.assertEqual(
            reason,
            "confirmation_timeout_no_valid_masseter_no_masseter_event",
        )

    def test_gate_reported_after_deadline_cannot_execute(self):
        action, reason = self._resolve(
            True, 101.001, "11", "incremental_masseter_11", execution_count=2
        )

        self.assertEqual(action, "cancel")
        self.assertEqual(
            reason,
            "confirmation_timeout_no_valid_masseter_incremental_masseter_11",
        )

    def test_deadline_bounded_source_slice_survives_delayed_timer_tick(self):
        action, reason = resolve_eeg8_confirmation_action(
            EEG8_RUNTIME_STRATEGY_MASSETER,
            True,
            self.START,
            101.04,
            self.WINDOW,
            "11",
            "incremental_masseter_11",
            gate_observed_before_deadline=True,
            masseter_execution_count=2,
            required_masseter_executions=2,
        )

        self.assertEqual(
            (action, reason),
            ("execute", "confirmed_by_masseter_count_2_of_2_at_deadline"),
        )

    def test_consumed_pending_confirmation_cannot_execute_twice(self):
        pending = True
        action, _ = self._resolve(
            pending, 100.20, "10", "incremental_masseter_10", execution_count=2
        )
        self.assertEqual(action, "execute")

        # Later ticks see a consumed confirmation.
        pending = False
        action, reason = self._resolve(
            pending, 100.21, "10", "incremental_masseter_10", execution_count=2
        )
        self.assertEqual((action, reason), ("none", "no_pending_confirmation"))

    def test_window2_uses_configured_execution_count_without_remapping(self):
        action, _ = resolve_eeg8_confirmation_action(
            EEG8_RUNTIME_STRATEGY_MASSETER,
            True,
            self.START,
            100.4,
            self.WINDOW,
            "count",
            "window2_masseter_count_1",
            masseter_execution_count=1,
            required_masseter_executions=2,
        )
        self.assertEqual(action, "wait")

        action, reason = resolve_eeg8_confirmation_action(
            EEG8_RUNTIME_STRATEGY_MASSETER,
            True,
            self.START,
            100.5,
            self.WINDOW,
            "count",
            "window2_masseter_count_2",
            masseter_execution_count=2,
            required_masseter_executions=2,
        )
        self.assertEqual((action, reason), ("execute", "confirmed_by_masseter_count_2_of_2"))

    def test_window3_segments_are_fixed_and_reset_after_deadline(self):
        self.assertEqual(resolve_continuous_stop_window(2, 3, 1.9, 2.0)[0], "wait")
        self.assertEqual(resolve_continuous_stop_window(2, 3, 2.0, 2.0)[0], "next_window")
        self.assertEqual(resolve_continuous_stop_window(3, 3, 0.8, 2.0)[0], "stop")

    def test_window2_one_jaw_execution_waits_and_then_cancels(self):
        action, reason = resolve_eeg8_confirmation_action(
            EEG8_RUNTIME_STRATEGY_MASSETER,
            True,
            self.START,
            100.40,
            self.WINDOW,
            "10",
            "window2_masseter_count_1",
            masseter_execution_count=1,
            required_masseter_executions=2,
        )
        self.assertEqual(action, "wait")
        self.assertIn("count_1_of_2", reason)

        action, reason = resolve_eeg8_confirmation_action(
            EEG8_RUNTIME_STRATEGY_MASSETER,
            True,
            self.START,
            101.0,
            self.WINDOW,
            "10",
            "window2_masseter_count_1",
            masseter_execution_count=1,
            required_masseter_executions=2,
        )
        self.assertEqual(action, "cancel")
        self.assertIn("window2_masseter_count_1", reason)

    def test_window2_two_jaw_executions_confirm_frozen_window1_result(self):
        action, reason = resolve_eeg8_confirmation_action(
            EEG8_RUNTIME_STRATEGY_MASSETER,
            True,
            self.START,
            100.35,
            self.WINDOW,
            "none",
            "window2_masseter_count_2",
            masseter_execution_count=2,
            required_masseter_executions=2,
        )
        self.assertEqual((action, reason), ("execute", "confirmed_by_masseter_count_2_of_2"))


if __name__ == "__main__":
    unittest.main()
