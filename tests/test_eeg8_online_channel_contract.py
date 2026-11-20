# -*- coding: utf-8 -*-
"""Safety tests for the LSL-to-model physical EEG channel contract."""

import unittest

from arm_bci.core import (
    DEFAULT_EEG_CHANNELS,
    normalize_eeg_channel_name,
    validate_eeg8_online_channel_contract,
)


class EEG8OnlineChannelContractTests(unittest.TestCase):
    def setUp(self):
        self.trained = list(DEFAULT_EEG_CHANNELS)

    def test_exact_physical_order_is_accepted(self):
        reported = self.trained + ["AUX1", "AUX2", "AUX3", "AUX4"]

        keys = validate_eeg8_online_channel_contract(reported, self.trained)

        self.assertEqual(keys, tuple(normalize_eeg_channel_name(name) for name in self.trained))

    def test_swapped_electrode_order_is_rejected(self):
        reported = list(self.trained)
        reported[4], reported[5] = reported[5], reported[4]

        with self.assertRaisesRegex(ValueError, "物理通道名/顺序.*不一致"):
            validate_eeg8_online_channel_contract(reported, self.trained)

    def test_missing_or_placeholder_lsl_labels_are_rejected(self):
        cases = (
            ("missing", ["" if index == 0 else name for index, name in enumerate(self.trained)]),
            ("placeholder", [f"CH{index + 1}" for index in range(32)]),
        )
        for name, reported in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "标签缺失或使用 CH/EEG 占位符"):
                    validate_eeg8_online_channel_contract(reported, self.trained)

    def test_case_and_separator_variants_are_accepted(self):
        reported = [f"  {name.upper()[0]}-{name.upper()[1:]}  " for name in self.trained]

        keys = validate_eeg8_online_channel_contract(reported, self.trained)

        self.assertEqual(keys, tuple(normalize_eeg_channel_name(name) for name in self.trained))

    def test_brainvision_bank_channel_ids_are_accepted(self):
        trained = [f"EEG 1_{index + 1}" for index in range(32)]
        reported = trained + [f"AUX 2_{index + 1}" for index in range(4)]

        keys = validate_eeg8_online_channel_contract(reported, trained)

        self.assertEqual(keys, tuple(normalize_eeg_channel_name(name) for name in trained))


if __name__ == "__main__":
    unittest.main()
