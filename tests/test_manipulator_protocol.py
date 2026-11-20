import json
import socket
import struct
import threading
import time
import unittest

from arm_bci.core import (
    ManipulatorProtocolRobotClient,
    RobotCommand,
    manipulator_message_for_command,
    pack_manipulator_message,
    resolve_robot_intent,
)


def command(name: str, *, duration: float = 0.05, continuous: bool = False) -> RobotCommand:
    return RobotCommand(
        name=name,
        intent=name.removeprefix("move_"),
        target="target_1",
        confidence=1.0,
        sequence_id=1,
        timestamp=123.5,
        duration_seconds=duration,
        continuous=continuous,
    )


class ManipulatorProtocolTests(unittest.TestCase):
    def test_robot_intent_normalization_accepts_command_or_intent(self):
        self.assertEqual(resolve_robot_intent("move_forward"), "forward")
        self.assertEqual(resolve_robot_intent("", "backward"), "backward")
        self.assertEqual(resolve_robot_intent("close_gripper"), "grab")
        self.assertEqual(resolve_robot_intent("home", ""), "home")

    def test_frame_is_big_endian_length_prefixed_utf8_json(self):
        message = {"type": "cmd", "key": "c", "label": "抓取"}
        frame = pack_manipulator_message(message)
        body_size = struct.unpack(">I", frame[:4])[0]
        self.assertEqual(body_size, len(frame) - 4)
        self.assertEqual(json.loads(frame[4:].decode("utf-8")), message)

    def test_all_bci_actions_match_reference_receiver_commands(self):
        expected = {
            "move_forward": ("move_start", 0, 1),
            "move_backward": ("move_start", 0, -1),
            "move_left": ("move_start", 1, 1),
            "move_right": ("move_start", 1, -1),
            "move_up": ("move_start", 2, 1),
            "move_down": ("move_start", 2, -1),
        }
        for name, (key, axis, sign) in expected.items():
            message = manipulator_message_for_command(command(name), speed=25)
            self.assertEqual((message["key"], message["axis"], message["sign"]), (key, axis, sign))
            self.assertEqual(message["speed"], 25.0)
        self.assertEqual(manipulator_message_for_command(command("close_gripper"))["key"], "c")
        self.assertEqual(manipulator_message_for_command(command("open_gripper"))["key"], "v")
        self.assertEqual(manipulator_message_for_command(command("home"))["key"], "r")
        self.assertEqual(manipulator_message_for_command(command("stop_motion"))["key"], "move_stop")

    def test_manual_rpy_actions_match_reference_receiver_axes(self):
        expected = {
            "rotate_roll_positive": (3, 1),
            "rotate_roll_negative": (3, -1),
            "rotate_pitch_positive": (4, 1),
            "rotate_pitch_negative": (4, -1),
            "rotate_yaw_positive": (5, 1),
            "rotate_yaw_negative": (5, -1),
        }
        for name, (axis, sign) in expected.items():
            with self.subTest(name=name):
                message = manipulator_message_for_command(command(name), speed=10)
                self.assertEqual(message["key"], "move_start")
                self.assertEqual((message["axis"], message["sign"]), (axis, sign))
                self.assertEqual(message["speed"], 10.0)

    def test_client_sends_start_and_timed_stop_over_one_tcp_connection(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(2.0)
        received = []

        def receive_two_messages():
            conn, _address = listener.accept()
            with conn:
                conn.settimeout(2.0)
                for _ in range(2):
                    header = self._recv_exact(conn, 4)
                    size = struct.unpack(">I", header)[0]
                    body = self._recv_exact(conn, size)
                    received.append(json.loads(body.decode("utf-8")))

        receiver = threading.Thread(target=receive_two_messages, daemon=True)
        receiver.start()
        client = ManipulatorProtocolRobotClient(
            "127.0.0.1", listener.getsockname()[1], timeout=0.3, speed=40, heartbeat_interval=60
        )
        try:
            client.execute(command("move_up", duration=0.05))
            receiver.join(timeout=2.0)
        finally:
            client.close()
            listener.close()
        self.assertFalse(receiver.is_alive())
        self.assertEqual([item["key"] for item in received], ["move_start", "move_stop"])
        self.assertEqual((received[0]["axis"], received[0]["sign"], received[0]["speed"]), (2, 1, 40.0))

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("socket closed before complete frame")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


if __name__ == "__main__":
    unittest.main()
