# -*- coding: utf-8 -*-
"""Qt-free regression tests for the real-arm TCP command output path."""

import unittest

import arm_bci.gui as gui
from arm_bci.core import (
    ACTION_COMMANDS,
    CommandPayloadFanout,
    RobotCommand,
    robot_command_payload,
)
from arm_bci.eeg8_system import CLASS_TO_ARM


class _PayloadSink:
    def __init__(self):
        self.payloads = []

    def execute_payload(self, payload):
        self.payloads.append(payload)


class CommandPayloadFanoutTests(unittest.TestCase):
    EXPECTED_EEG8_ACTIONS = {
        1: "left",
        2: "right",
        3: "forward",
        4: "backward",
        5: "up",
        6: "down",
        7: "grab",
        8: "release",
    }

    def test_gui_uses_the_single_eeg8_action_mapping(self):
        self.assertEqual(CLASS_TO_ARM, self.EXPECTED_EEG8_ACTIONS)
        # GUI and runtime share one action table.
        self.assertIs(gui.CLASS_TO_ARM, CLASS_TO_ARM)

    def test_each_eeg8_command_reaches_real_arm_tcp_from_one_payload(self):
        tcp_sink = _PayloadSink()
        fanout = CommandPayloadFanout(tcp_sink.execute_payload)

        for class_id, intent in self.EXPECTED_EEG8_ACTIONS.items():
            with self.subTest(class_id=class_id):
                command = RobotCommand(
                    name=ACTION_COMMANDS[intent],
                    intent=intent,
                    target="target_1",
                    confidence=0.88,
                    sequence_id=class_id,
                    timestamp=1234.0 + class_id,
                )
                payload = robot_command_payload(command)
                fanout.dispatch(payload)

                self.assertIs(tcp_sink.payloads[-1], payload)
                self.assertEqual(payload["cmd"], ACTION_COMMANDS[intent])
                self.assertEqual(payload["intent"], intent)
                self.assertNotIn("name", payload)


if __name__ == "__main__":
    unittest.main()
