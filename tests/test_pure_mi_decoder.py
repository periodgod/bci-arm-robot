# -*- coding: utf-8 -*-
"""纯 MI 可变长解码规则测试。"""

import unittest

from arm_bci.pure_mi_decoder import (
    PureMISequenceDecoder,
    decode_pure_mi_code,
    pure_mi_codebook,
)


class PureMICodeTests(unittest.TestCase):
    def test_one_and_two_bit_direction_codes(self):
        expected = {
            "1": "left",
            "0": "right",
            "10": "forward",
            "01": "backward",
            "11": "up",
            "00": "down",
        }
        for code, intent in expected.items():
            with self.subTest(code=code):
                result = decode_pure_mi_code(code)
                self.assertEqual(result.intent, intent)
                self.assertTrue(result.continuous)
                self.assertFalse(result.requires_confirmation)

    def test_any_two_bit_prefix_can_form_grab_or_release(self):
        for prefix in ("00", "01", "10", "11"):
            grab = decode_pure_mi_code(prefix + "1")
            release = decode_pure_mi_code(prefix + "0")
            self.assertEqual(grab.intent, "grab")
            self.assertEqual(release.intent, "release")
            self.assertTrue(grab.requires_confirmation)
            self.assertTrue(release.requires_confirmation)
            self.assertFalse(grab.continuous)
            self.assertFalse(release.continuous)

    def test_codebook_documents_all_rules(self):
        self.assertEqual(len(pure_mi_codebook()), 8)


class PureMISequenceTests(unittest.TestCase):
    def test_left_masseter_truncates_before_current_window(self):
        decoder = PureMISequenceDecoder()
        decoder.feed_completed_window("left")
        decoder.feed_completed_window("right")
        decoder.begin_window("left")

        result = decoder.handle_masseter(left=True)

        self.assertEqual(result.code, "10")
        self.assertEqual(result.intent, "forward")
        self.assertEqual(decoder.completed_code, "")
        self.assertIsNone(decoder.active_bit)

    def test_right_masseter_rolls_back_exactly_one_completed_window(self):
        decoder = PureMISequenceDecoder()
        decoder.feed_completed_window(1)
        decoder.feed_completed_window(0)
        decoder.begin_window(0)

        result = decoder.handle_masseter(right=True)

        self.assertEqual(result.action, "rollback")
        self.assertEqual(result.code, "1")
        self.assertEqual(decoder.completed_code, "1")
        self.assertIsNone(decoder.active_bit)
        self.assertEqual(decoder.feed_completed_window(0), "10")

    def test_right_masseter_at_first_window_keeps_empty_sequence(self):
        decoder = PureMISequenceDecoder()
        decoder.begin_window(1)

        result = decoder.handle_masseter(right=True)

        self.assertEqual(result.action, "rollback")
        self.assertEqual(result.code, "")
        self.assertEqual(decoder.completed_code, "")
        self.assertIsNone(decoder.active_bit)

    def test_both_masseters_stop_immediately_instead_of_blinks(self):
        decoder = PureMISequenceDecoder()
        decoder.feed_completed_window(1)

        result = decoder.handle_masseter(left=True, right=True)

        self.assertEqual(result.action, "stop")
        self.assertEqual(result.intent, "stop_motion")
        self.assertEqual(decoder.completed_code, "")

    def test_left_masseter_without_completed_window_waits(self):
        decoder = PureMISequenceDecoder()
        decoder.begin_window(1)

        result = decoder.handle_masseter(left=True)

        self.assertEqual(result.action, "wait")
        self.assertIsNone(result.intent)

    def test_no_masseter_after_three_windows_clears_all_current_counts(self):
        decoder = PureMISequenceDecoder()
        decoder.feed_completed_window(0)
        decoder.feed_completed_window(0)
        decoder.feed_completed_window(0)

        code = decoder.feed_completed_window(1)

        self.assertEqual(code, "")
        self.assertEqual(decoder.completed_code, "")


if __name__ == "__main__":
    unittest.main()
