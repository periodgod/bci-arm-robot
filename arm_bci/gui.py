# -*- coding: utf-8 -*-
"""PyQt GUI for the arm BCI system."""

from __future__ import annotations

import csv
import copy
from collections import deque
from datetime import datetime
import json
import math
import os
import sys
import threading
import traceback
import uuid
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from arm_bci.commands import apply_runtime_control_config, metrics_summary
    from arm_bci.core import *
    from arm_bci.domain import EEG8Model
    from arm_bci.pure_mi_decoder import PureMISequenceDecoder
    from arm_bci.eeg8_system import (CLASS_NAMES, CLASS_TO_ARM, CLASS_ARM_LABELS,
                                     EEG8_TASKS, eeg8_task_map, eeg8_format_task,
                                     load_eeg8_recordings, train_eeg8_model, eval_eeg8_model,
                                     eeg8_binary_confusion, generate_balanced_block,
                                     eeg8_action_label_info, validate_eeg8_training_data_contract)
    from arm_bci.console_design import (
        CONFIG_PAGE_DESC, CONFIG_PAGE_TITLE,
        CONSOLE_BRAND, CONSOLE_HINT_TEXT, CONSOLE_NAV_CODES,
        CONSOLE_PAGE_NAMES, CONSOLE_SUBTITLE,
        CONSOLE_WINDOW_TITLE, CONSOLE_WORKFLOW_TEXT,
    )
else:
    from .commands import apply_runtime_control_config, metrics_summary
    from .core import *
    from .domain import EEG8Model
    from .pure_mi_decoder import PureMISequenceDecoder
    from .eeg8_system import (CLASS_NAMES, CLASS_TO_ARM, CLASS_ARM_LABELS,
                               EEG8_TASKS, eeg8_task_map, eeg8_format_task,
                               load_eeg8_recordings, train_eeg8_model, eval_eeg8_model,
                               eeg8_binary_confusion, generate_balanced_block,
                               eeg8_action_label_info, validate_eeg8_training_data_contract)
    from .console_design import (
        CONFIG_PAGE_DESC, CONFIG_PAGE_TITLE,
        CONSOLE_BRAND, CONSOLE_HINT_TEXT, CONSOLE_NAV_CODES,
        CONSOLE_PAGE_NAMES, CONSOLE_SUBTITLE,
        CONSOLE_WINDOW_TITLE, CONSOLE_WORKFLOW_TEXT,
    )

def select_gui_eeg8_confirmation_candidate(
    probabilities: np.ndarray,
    candidates: Sequence[int],
    gate_reason: str,
    minimum_gap: float,
) -> Tuple[Optional[int], int, str]:
    """Select from the mapped pair, randomizing when the probability gap is small."""
    proba = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    valid = tuple(int(i) for i in candidates if 0 <= int(i) < min(8, proba.size))
    if not valid:
        return None, 0, f"{gate_reason}_no_mapping_candidates"
    order = np.argsort(proba)[::-1]
    fallback_idx = int(order[0]) if len(order) else 0
    labels = "/".join(f"C{idx + 1}" for idx in valid)
    selected_idx, second_idx, randomized, gap = select_binary_candidate(
        proba,
        valid,
        random_margin=max(0.0, float(minimum_gap)),
    )
    mode = "random" if randomized else "highest"
    return selected_idx, second_idx, (
        f"{gate_reason}_selected_C{selected_idx + 1}_from_{labels}_"
        f"mode_{mode}_gap_{gap:.4f}"
    )


def _masseter_gate_with_reason(gate: MasseterGateResult, reason: str) -> MasseterGateResult:
    """Return a gate copy so detector resolution is free of UI side effects."""
    return MasseterGateResult(
        gate.side,
        gate.left_count,
        gate.right_count,
        gate.both_count,
        gate.left_ratio,
        gate.right_ratio,
        gate.candidates,
        str(reason),
    )


def resolve_gui_masseter_detector_gate(
    detector_mode: object,
    channel_gate: MasseterGateResult,
    ncc_gate: Optional[MasseterGateResult],
    state_detectors: Optional[Dict[str, str]] = None,
) -> MasseterGateResult:
    """Select the configured threshold or NCC gate."""
    mode = str(detector_mode or "threshold").strip().lower()
    if mode == "threshold":
        return channel_gate

    if ncc_gate is None:
        ncc_gate = MasseterGateResult(
            "none", 0, 0, 0, 0.0, 0.0, tuple(), "ncc_not_evaluated"
        )
    if mode == "ncc":
        # NCC failures stay on the NCC path.
        return ncc_gate

    try:
        state_modes = {
            str(key): str(value or "threshold").strip().lower()
            for key, value in dict(state_detectors or {}).items()
        }
    except (TypeError, ValueError):
        state_modes = {}

    # Use NCC when the detected state is configured for NCC.
    channel_side = str(channel_gate.side or "none")
    if (
        channel_side in MASSETER_BINARY_CLASS_MAP
        and channel_side != "00"
        and state_modes.get(channel_side, "threshold") == "ncc"
    ):
        return ncc_gate

    # Resolve per-state detector routing.
    selected = []
    if (
        channel_gate.side != "00"
        and state_modes.get(channel_gate.side, "threshold") == "threshold"
    ):
        selected.append(channel_gate)
    if (
        ncc_gate.side != "00"
        and state_modes.get(ncc_gate.side, "threshold") == "ncc"
    ):
        selected.append(ncc_gate)
    if not selected:
        if state_modes.get(ncc_gate.side, "threshold") == "ncc":
            selected.append(ncc_gate)
        elif state_modes.get(channel_gate.side, "threshold") == "threshold":
            selected.append(channel_gate)
    unique = {item.side: item for item in selected if item.side in MASSETER_BINARY_CLASS_MAP}
    if len(unique) == 1:
        gate = next(iter(unique.values()))
        return _masseter_gate_with_reason(
            gate, f"precise_{state_modes.get(gate.side, 'threshold')}_{gate.reason}"
        )
    if len(unique) > 1:
        return _masseter_gate_with_reason(
            channel_gate,
            f"precise_conflict_channel_{channel_gate.side}_ncc_{ncc_gate.side}",
        )
    return _masseter_gate_with_reason(
        channel_gate, f"precise_no_match_fallback_channel_{channel_gate.side}"
    )


def build_eeg8_realtime_startup_payload(
    model: object,
    source: object,
    template_bank: object,
    runtime_strategy: object,
    runtime_session: Optional[RuntimeSessionConfig] = None,
) -> Dict[str, object]:
    """Build the worker-to-page handoff without dropping the frozen session."""
    return {
        "model": model,
        "source": source,
        "template_bank": template_bank,
        "runtime_strategy": str(runtime_strategy or DEFAULT_EEG8_RUNTIME_STRATEGY),
        "runtime_session": runtime_session,
    }


def launch_gui() -> int:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except Exception as exc:
        print(f"ERROR: PyQt6 is required for the GUI: {exc}")
        print("Install it in your environment, for example:")
        print("  conda install -n bciwork -c conda-forge pyqt=6")
        return 1
    try:
        import pybullet as bullet
        import pybullet_data
    except Exception:
        bullet = None
        pybullet_data = None
    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
    except Exception as exc:
        print(f"ERROR: pyqtgraph and PyOpenGL are required for the optimized 3D view: {exc}")
        print("Install them in your environment, for example:")
        print("  python -m pip install pyqtgraph PyOpenGL")
        return 1
    pg.setConfigOptions(antialias=False)

    class CommandBus(QtCore.QObject):
        command_emitted = QtCore.pyqtSignal(dict)

    class GuiRobotArm(RobotArmClient):
        def __init__(self, bus: CommandBus, log_callback=None):
            self.bus = bus
            self.log_callback = log_callback
            self.commands: List[RobotCommand] = []

        def execute(self, command: RobotCommand) -> None:
            self.commands.append(command)
            payload = robot_command_payload(command)
            self.bus.command_emitted.emit(payload)
            if self.log_callback:
                self.log_callback(
                    f"机械臂指令 seq={command.sequence_id} cmd={command.name} "
                    f"intent={command.intent} conf={command.confidence:.3f} duration={command.duration_seconds:.2f}s "
                    f"continuous={command.continuous} speed={command.move_speed:.2f} velocity={command.velocity}"
                )

    class NetworkRobotForwarder(QtCore.QObject):
        """Forward GUI commands to the Ubuntu receiver."""

        def __init__(self, config, status_callback=None, session_provider=None, parent=None):
            super().__init__(parent)
            self.config = config
            self.status_callback = status_callback
            self.session_provider = session_provider
            self.client: Optional[ManipulatorProtocolRobotClient] = None
            self.endpoint = None

        def _active_session(self) -> Optional[RuntimeSessionConfig]:
            if self.session_provider is None:
                return None
            try:
                session = self.session_provider()
            except Exception:
                return None
            return session if isinstance(session, RuntimeSessionConfig) else None

        def execute_payload(self, payload: Dict) -> None:
            # Keep the endpoint fixed for the active session.
            session = self._active_session()
            if session is not None:
                enabled, host, port, speed, target = runtime_session_robot_output(session)
                endpoint = (host, port, speed)
            else:
                cfg = self.config.realtime_config()
                enabled = bool(cfg.get("robot_enabled", False))
                endpoint = (cfg["robot_host"], int(cfg["robot_port"]), float(cfg["robot_speed"]))
                target = str(payload.get("target", cfg.get("target", "target_1")))
            if not enabled:
                return
            if endpoint != self.endpoint:
                if self.client is not None:
                    self.client.close()
                self.client = ManipulatorProtocolRobotClient(
                    endpoint[0], endpoint[1], speed=endpoint[2], timeout=1.0
                )
                self.endpoint = endpoint
                self.client.start()
            try:
                command = RobotCommand(
                    name=str(payload.get("cmd", "")),
                    intent=str(payload.get("intent", "")),
                    target=target,
                    confidence=float(payload.get("confidence", 1.0)),
                    sequence_id=int(payload.get("sequence_id", 0)),
                    timestamp=float(payload.get("timestamp", time.time())),
                    duration_seconds=float(payload.get("duration_seconds", 0.55)),
                    continuous=bool(payload.get("continuous", False)),
                    move_speed=float(payload.get("move_speed", 0.0)),
                    velocity=tuple(payload.get("velocity", (0.0, 0.0, 0.0))),
                    protocol_speed=float(payload.get("protocol_speed", 0.0)),
                )
                self.client.execute(command)
                if self.status_callback:
                    state = "已连接并排队" if self.client.is_connected else "等待连接，指令已排队"
                    self.status_callback(f"机械臂{state}：{command.name} -> {endpoint[0]}:{endpoint[1]}")
            except (TypeError, ValueError) as exc:
                if self.status_callback:
                    self.status_callback(f"机械臂指令无效：{exc}")

        def close(self) -> None:
            if self.client is not None:
                self.client.close()
                self.client = None

    class Card(QtWidgets.QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("card")
            self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

    class RawTraceWidget(QtWidgets.QWidget):
        def __init__(self, channel_names: Sequence[str], parent=None, row_height: int = 150):
            super().__init__(parent)
            self.channel_names = list(channel_names)
            self.row_height = max(42, int(row_height))
            self.page_seconds = 30.0
            self.uv_per_row = 75.0
            self.uv_per_channel = np.full(len(self.channel_names), self.uv_per_row, dtype=np.float64)
            self.selected = np.ones(len(self.channel_names), dtype=bool)
            self.t = np.zeros((0,), dtype=np.float64)
            self.data = np.zeros((0, len(self.channel_names)), dtype=np.float32)
            self.highlight_ranges: List[Tuple[float, float, QtGui.QColor]] = []
            self._update_minimum_height()
            self.setMinimumWidth(1100)

        def _update_minimum_height(self) -> None:
            self.setMinimumHeight(max(360, len(self.channel_names) * self.row_height + 72))

        def set_channel_names(self, channel_names: Sequence[str]) -> None:
            self.channel_names = list(channel_names)
            n_channels = len(self.channel_names)
            if self.uv_per_channel.size != n_channels:
                self.uv_per_channel = np.full(n_channels, self.uv_per_row, dtype=np.float64)
            if self.selected.size != n_channels:
                self.selected = np.ones(n_channels, dtype=bool)
            self._update_minimum_height()
            self.update()

        def set_config(self, page_seconds: float, uv_per_row: float) -> None:
            self.page_seconds = max(1.0, float(page_seconds))
            arr = np.asarray(uv_per_row, dtype=np.float64)
            if arr.ndim == 0:
                self.uv_per_row = max(1e-6, float(arr))
                self.uv_per_channel = np.full(len(self.channel_names), self.uv_per_row, dtype=np.float64)
            else:
                arr = np.maximum(arr.reshape(-1), 1e-6)
                if arr.size < len(self.channel_names):
                    arr = np.pad(arr, (0, len(self.channel_names) - arr.size), constant_values=float(arr[-1]))
                self.uv_per_channel = arr[: len(self.channel_names)]
                self.uv_per_row = float(np.median(self.uv_per_channel))
            self.update()

        def set_trace_data(self, t: np.ndarray, data: np.ndarray, selected: np.ndarray) -> None:
            self.t = np.asarray(t, dtype=np.float64)
            self.data = np.asarray(data, dtype=np.float32)
            self.selected = np.asarray(selected, dtype=bool)
            self.update()

        def set_highlight_ranges(self, ranges: Sequence[Tuple[float, float]], color: str = "#41b883") -> None:
            base = QtGui.QColor(color)
            base.setAlpha(62)
            self.highlight_ranges = [
                (float(start), float(end), QtGui.QColor(base))
                for start, end in ranges
                if float(end) > float(start)
            ]
            self.update()

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
            rect = self.rect()
            painter.fillRect(rect, QtGui.QColor("#f7f9fb"))

            label_w = 168
            top = 18
            bottom = 42
            right = 14
            plot_left = label_w
            plot_right = max(plot_left + 1, rect.width() - right)
            plot_top = top
            plot_bottom = max(plot_top + 1, rect.height() - bottom)
            plot_w = max(1.0, float(plot_right - plot_left))
            plot_h = max(1.0, float(plot_bottom - plot_top))
            n_channels = max(1, len(self.channel_names))
            row_h = plot_h / n_channels

            painter.fillRect(QtCore.QRect(0, 0, label_w, rect.height()), QtGui.QColor("#eef2f6"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#cfd8e3"), 1))
            painter.drawLine(label_w, 0, label_w, rect.height())

            minor_step = 0.5 if self.page_seconds <= 30 else 1.0
            major_step = 5.0 if self.page_seconds >= 10 else 1.0
            x = 0.0
            while x <= self.page_seconds + 1e-9:
                px = plot_left + int(round((x / self.page_seconds) * plot_w))
                is_major = abs((x / major_step) - round(x / major_step)) < 1e-6
                color = QtGui.QColor("#8ed3a2") if is_major else QtGui.QColor("#cfead7")
                pen = QtGui.QPen(color, 1)
                if not is_major:
                    pen.setStyle(QtCore.Qt.PenStyle.DotLine)
                painter.setPen(pen)
                painter.drawLine(px, plot_top, px, plot_bottom)
                if is_major:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#23313d"), 1))
                    painter.drawText(px - 12, plot_bottom + 18, 28, 18, QtCore.Qt.AlignmentFlag.AlignCenter, f"{int(x)}")
                x += minor_step

            painter.setPen(QtGui.QPen(QtGui.QColor("#d6dee8"), 1))
            for idx in range(n_channels + 1):
                y = int(round(plot_top + idx * row_h))
                painter.drawLine(0, y, plot_right, y)

            for start, end, color in self.highlight_ranges:
                clipped_start = max(0.0, min(self.page_seconds, start))
                clipped_end = max(0.0, min(self.page_seconds, end))
                if clipped_end <= clipped_start:
                    continue
                x1 = plot_left + int(round((clipped_start / self.page_seconds) * plot_w))
                x2 = plot_left + int(round((clipped_end / self.page_seconds) * plot_w))
                painter.fillRect(QtCore.QRect(x1, plot_top, max(1, x2 - x1), int(plot_h)), color)

            font = painter.font()
            font.setPointSize(11)
            painter.setFont(font)
            for idx, name in enumerate(self.channel_names):
                cy = plot_top + (idx + 0.5) * row_h
                painter.setPen(QtGui.QPen(QtGui.QColor("#111111"), 1))
                painter.drawText(8, int(cy - 18), label_w - 16, 18, QtCore.Qt.AlignmentFlag.AlignCenter, str(name))
                painter.setPen(QtGui.QPen(QtGui.QColor("#ff0000"), 1))
                channel_scale = self.uv_per_channel[idx] if idx < len(self.uv_per_channel) else self.uv_per_row
                scale_text = f"{channel_scale:g} uV" if idx < len(self.selected) and self.selected[idx] else "未选"
                painter.drawText(8, int(cy + 1), label_w - 16, 18, QtCore.Qt.AlignmentFlag.AlignCenter, scale_text)

            painter.setPen(QtGui.QPen(QtGui.QColor("#000000"), 1))
            painter.drawText(plot_left + 8, 2, 420, 18, QtCore.Qt.AlignmentFlag.AlignLeft, f"{self.page_seconds:g} 秒/页")

            if self.t.size < 2 or self.data.shape[0] < 2:
                painter.setPen(QtGui.QPen(QtGui.QColor("#666666"), 1))
                painter.drawText(plot_left + 20, plot_top + 30, 280, 24, QtCore.Qt.AlignmentFlag.AlignLeft, "等待原始信号...")
                painter.end()
                return

            t = np.asarray(self.t, dtype=np.float64)
            data = np.asarray(self.data, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] < n_channels:
                painter.end()
                return
            valid = (t >= 0.0) & (t <= self.page_seconds)
            if not np.any(valid):
                painter.end()
                return
            t = t[valid]
            data = data[valid, :]
            painter.setPen(QtGui.QPen(QtGui.QColor("#000000"), 1))
            for idx in range(n_channels):
                if idx >= len(self.selected) or not self.selected[idx]:
                    continue
                row_data = data[:, idx]
                centered = row_data - np.median(row_data)
                channel_scale = self.uv_per_channel[idx] if idx < len(self.uv_per_channel) else self.uv_per_row
                clipped = np.clip(centered / max(1e-6, float(channel_scale)), -0.48, 0.48)
                cy = plot_top + (idx + 0.5) * row_h
                xs_full = plot_left + (t / self.page_seconds) * plot_w
                ys_full = cy - clipped * row_h
                px_int = np.clip(np.rint(xs_full).astype(np.int32), plot_left, plot_right)
                unique_x, first_idx = np.unique(px_int, return_index=True)
                xs = unique_x.astype(np.float64)
                ys = ys_full[first_idx].astype(np.float64)
                if xs.size < 2:
                    continue
                path = QtGui.QPainterPath()
                path.moveTo(float(xs[0]), float(ys[0]))
                for px, py in zip(xs[1:], ys[1:]):
                    path.lineTo(float(px), float(py))
                painter.drawPath(path)

            painter.end()

    class KinematicOpenGLArmWidget(QtWidgets.QWidget):
        state_changed = QtCore.pyqtSignal(str)

        TASK_SCENES = {
            "free": {
                "name": "自由操作",
                "goal": "原始演示场景：自由抓取和移动箱体",
                "homes": [[5.0, -1.0, 0.54], [6.0, 1.0, 0.54], [3.0, -2.0, 0.54], [2.0, 2.0, 0.54]],
                "zones": [[4.0, 0.0, 0.04], [3.0, 1.0, 0.04]],
                "zone_colors": [(0.18, 0.88, 0.36, 0.42), (0.20, 0.56, 0.98, 0.36)],
                "assignments": [],
            },
            "transport": {
                "name": "搬运任务",
                "goal": "将 4 个货箱分别搬到绿色和蓝色托盘",
                "homes": [[6.0, -3.0, 0.54], [6.0, 3.0, 0.54], [3.0, -3.0, 0.54], [3.0, 3.0, 0.54]],
                "zones": [[2.0, -1.0, 0.04], [2.0, 1.0, 0.04]],
                "zone_colors": [(0.18, 0.88, 0.36, 0.42), (0.20, 0.56, 0.98, 0.36)],
                "assignments": [0, 1, 0, 1],
            },
            "stack": {
                "name": "堆叠任务",
                "goal": "将 4 个方块依次堆叠到中央平台",
                "homes": [[6.0, -3.0, 0.54], [6.0, -1.0, 0.54], [6.0, 1.0, 0.54], [6.0, 3.0, 0.54]],
                "zones": [[3.0, 0.0, 0.04]],
                "zone_colors": [(0.18, 0.88, 0.36, 0.42)],
                "assignments": [0, 0, 0, 0],
            },
            "sorting": {
                "name": "分类任务",
                "goal": "按箱体颜色放入 4 个对应颜色的目标区",
                "homes": [[6.0, -3.0, 0.54], [6.0, -1.0, 0.54], [6.0, 1.0, 0.54], [6.0, 3.0, 0.54]],
                "zones": [[2.0, -3.0, 0.04], [2.0, -1.0, 0.04], [2.0, 1.0, 0.04], [2.0, 3.0, 0.04]],
                "zone_colors": [(0.10, 0.67, 0.92, 0.38), (0.93, 0.38, 0.24, 0.38), (0.93, 0.72, 0.18, 0.38), (0.58, 0.42, 0.90, 0.38)],
                "assignments": [0, 1, 2, 3],
            },
        }

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(720, 480)
            self.home_target = [4.0, 0.0, 3.0]
            self.ee_target = self.home_target[:]
            self.motion_start = self.home_target[:]
            self.motion_target: Optional[List[float]] = None
            self.motion_queue: List[Tuple[List[float], str]] = []
            self.motion_started_at = 0.0
            self.motion_duration = 0.0
            self.grid_step = 1.0
            self.grid_cell_duration = 0.5
            self.motion_speed = 0.22
            self.continuous_direction = None
            self.continuous_last_tick = time.perf_counter()
            self.motion_until = 0.0
            self.finger_target = 0.35
            self.pending_grab = False
            self.grabbed_object_index: Optional[int] = None
            self.preferred_drop_zone_index: Optional[int] = None
            self.mode_text = "Ready"
            self.active_task = "free"
            self.grid_x_min = -0.5
            self.grid_x_max = 8.5
            self.grid_y_min = -5.5
            self.grid_y_max = 5.5
            self.floor_z = 0.0
            self.cube_size = 1.0
            self.pad_height = 0.08
            initial_homes = self.TASK_SCENES[self.active_task]["homes"]
            self.object_specs = [
                {"home": initial_homes[0], "size": (1.0, 1.0, 1.0), "color": (0.10, 0.67, 0.92, 0.96)},
                {"home": initial_homes[1], "size": (1.0, 1.0, 1.0), "color": (0.93, 0.38, 0.24, 0.96)},
                {"home": initial_homes[2], "size": (1.0, 1.0, 1.0), "color": (0.93, 0.72, 0.18, 0.96)},
                {"home": initial_homes[3], "size": (1.0, 1.0, 1.0), "color": (0.58, 0.42, 0.90, 0.96)},
            ]
            self.object_positions = [list(spec["home"]) for spec in self.object_specs]
            self.target_zones = [list(zone) for zone in self.TASK_SCENES[self.active_task]["zones"]]
            self.target_sizes = [(1.0, 1.0, self.pad_height)] * 4

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            header = QtWidgets.QHBoxLayout()
            self.task_buttons: Dict[str, QtWidgets.QPushButton] = {}
            self.task_button_group = QtWidgets.QButtonGroup(self)
            self.task_button_group.setExclusive(True)
            for task_key in ["free", "transport", "stack", "sorting"]:
                task_button = QtWidgets.QPushButton(self.TASK_SCENES[task_key]["name"])
                task_button.setCheckable(True)
                task_button.setMinimumWidth(110)
                task_button.setFixedHeight(42)
                task_button.setProperty("taskTab", True)
                task_button.clicked.connect(lambda _checked=False, key=task_key: self.set_task(key))
                self.task_button_group.addButton(task_button)
                self.task_buttons[task_key] = task_button
                header.addWidget(task_button, 1)
            self.task_buttons[self.active_task].setChecked(True)
            self.setStyleSheet(
                "QPushButton[taskTab='true'] {"
                " background: #f8fafc; color: #17324d; border: 1px solid #c7d9e8;"
                " border-radius: 4px; font-weight: 600; padding: 8px 18px; }"
                "QPushButton[taskTab='true']:hover { background: #edf6fb; border-color: #70b9d6; }"
                "QPushButton[taskTab='true']:checked {"
                " background: #087fa8; color: white; border-color: #087fa8; }"
            )
            self.state_badge = QtWidgets.QLabel("Ready")
            self.state_badge.setObjectName("vrBadge")
            self.state_badge.setVisible(False)
            layout.addLayout(header)
            self.status_label = QtWidgets.QLabel(self)
            self.status_label.setObjectName("vrStatus")
            self.status_label.setMinimumHeight(28)
            self.task_goal_label = QtWidgets.QLabel()
            self.task_goal_label.setObjectName("vrStatus")
            self.task_goal_label.setWordWrap(True)
            layout.addWidget(self.task_goal_label)

            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor(QtGui.QColor("#0a1118"))
            self.view.setCameraPosition(distance=15.0, elevation=34, azimuth=42)
            self.view.opts["center"] = QtGui.QVector3D(4.0, 0.0, 1.8)
            self.view.setFixedHeight(560)
            layout.addWidget(self.view, 1)
            layout.addWidget(self.status_label)

            self.arm_line = gl.GLLinePlotItem(color=(0.52, 0.80, 0.95, 1.0), width=8, antialias=True)
            self.gripper_crossbar = gl.GLLinePlotItem(color=(1.0, 0.86, 0.22, 1.0), width=6, antialias=True)
            self.left_gripper = gl.GLLinePlotItem(color=(1.0, 0.80, 0.18, 1.0), width=6, antialias=True)
            self.right_gripper = gl.GLLinePlotItem(color=(1.0, 0.80, 0.18, 1.0), width=6, antialias=True)
            self.object_boxes = [self._make_box_item(spec["size"], spec["color"], "opaque") for spec in self.object_specs]
            target_colors = [
                (0.10, 0.67, 0.92, 0.38),
                (0.93, 0.38, 0.24, 0.38),
                (0.93, 0.72, 0.18, 0.38),
                (0.58, 0.42, 0.90, 0.38),
            ]
            self.target_boxes = [
                self._make_box_item(size, color, "translucent")
                for size, color in zip(self.target_sizes, target_colors)
            ]
            self.grid = gl.GLGridItem(size=QtGui.QVector3D(9.0, 11.0, 1.0), color=(56, 96, 118, 105))
            self.grid.translate(4.0, 0.0, self.floor_z)
            axis_origin = np.asarray([self.grid_x_min, self.grid_y_min, 0.08], dtype=np.float32)
            self.axis_x = gl.GLLinePlotItem(pos=np.asarray([axis_origin, axis_origin + [9.0, 0.0, 0.0]], dtype=np.float32), color=(1.0, 0.24, 0.24, 1.0), width=4, antialias=True)
            self.axis_y = gl.GLLinePlotItem(pos=np.asarray([axis_origin, axis_origin + [0.0, 11.0, 0.0]], dtype=np.float32), color=(0.30, 0.95, 0.42, 1.0), width=4, antialias=True)
            self.axis_z = gl.GLLinePlotItem(pos=np.asarray([axis_origin, axis_origin + [0.0, 0.0, 4.5]], dtype=np.float32), color=(0.32, 0.58, 1.0, 1.0), width=4, antialias=True)
            self.floor_grid_lines: List[object] = []
            grid_color = (0.25, 0.52, 0.66, 0.46)
            z_floor = self.floor_z + 0.03
            for x_value in range(0, 10):
                x_line = self.grid_x_min + float(x_value)
                self.floor_grid_lines.append(
                    gl.GLLinePlotItem(
                        pos=np.asarray([[x_line, self.grid_y_min, z_floor], [x_line, self.grid_y_max, z_floor]], dtype=np.float32),
                        color=grid_color,
                        width=1,
                        antialias=True,
                    )
                )
            for y_value in range(0, 12):
                y_line = self.grid_y_min + float(y_value)
                self.floor_grid_lines.append(
                    gl.GLLinePlotItem(
                        pos=np.asarray([[self.grid_x_min, y_line, z_floor], [self.grid_x_max, y_line, z_floor]], dtype=np.float32),
                        color=grid_color,
                        width=1,
                        antialias=True,
                    )
                )
            for item in [
                self.grid,
                *self.floor_grid_lines,
                *self.target_boxes,
                *self.object_boxes,
                self.arm_line,
                self.gripper_crossbar,
                self.left_gripper,
                self.right_gripper,
            ]:
                self.view.addItem(item)
            self._update_target_boxes()
            self._update_gl_items()
            self._refresh_status()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(16)

        def _make_box_item(self, size: Sequence[float], color: Tuple[float, float, float, float], gl_options: str):
            sx, sy, sz = [float(value) for value in size]
            vertices = np.array(
                [[0.0, 0.0, 0.0], [sx, 0.0, 0.0], [sx, sy, 0.0], [0.0, sy, 0.0],
                 [0.0, 0.0, sz], [sx, 0.0, sz], [sx, sy, sz], [0.0, sy, sz]],
                dtype=np.float32,
            )
            faces = np.array(
                [[0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5], [0, 4, 5], [0, 5, 1],
                 [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]],
                dtype=np.int32,
            )
            return gl.GLMeshItem(
                meshdata=gl.MeshData(vertexes=vertices, faces=faces),
                color=color,
                smooth=False,
                shader="shaded",
                glOptions=gl_options,
                drawEdges=True,
                edgeColor=(0.90, 0.97, 1.0, 0.72),
            )

        def _set_box_transform(self, item, position: Sequence[float], size: Sequence[float]) -> None:
            item.resetTransform()
            item.translate(
                float(position[0]) - float(size[0]) / 2.0,
                float(position[1]) - float(size[1]) / 2.0,
                float(position[2]) - float(size[2]) / 2.0,
            )

        def set_task(self, task_key: str) -> None:
            if task_key not in self.TASK_SCENES:
                return
            self.active_task = task_key
            scene = self.TASK_SCENES[task_key]
            homes = [list(home) for home in scene["homes"]]
            zones = [list(zone) for zone in scene["zones"]]
            if task_key == "sorting":
                homes, zones = self._random_sorting_layout()
            self.target_zones = zones
            for spec, home in zip(self.object_specs, homes):
                spec["home"] = list(home)
            if task_key in self.task_buttons:
                self.task_buttons[task_key].setChecked(True)
            self._update_target_boxes()
            self.reset_scene()

        def _random_sorting_layout(self) -> Tuple[List[List[float]], List[List[float]]]:
            rng = np.random.default_rng()
            object_cells = [(x, y) for x in range(4, 8) for y in range(-4, 5) if (x, y) != (4, 0)]
            target_cells = [(x, y) for x in range(1, 4) for y in range(-4, 5)]

            def sample_spread(cells: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
                for _ in range(50):
                    indices = rng.choice(len(cells), size=4, replace=False)
                    picked = [cells[int(index)] for index in indices]
                    if len({x for x, _y in picked}) >= 2 and len({y for _x, y in picked}) >= 3:
                        return picked
                return [cells[index] for index in range(0, len(cells), max(1, len(cells) // 4))][:4]

            object_positions = sample_spread(object_cells)
            target_positions = sample_spread(target_cells)
            homes = [[float(x), float(y), 0.54] for x, y in object_positions]
            zones = [[float(x), float(y), self.pad_height / 2.0] for x, y in target_positions]
            return homes, zones

        def _update_target_boxes(self) -> None:
            scene = self.TASK_SCENES[self.active_task]
            colors = list(scene.get("zone_colors", []))
            for index, box in enumerate(self.target_boxes):
                visible = index < len(self.target_zones)
                box.setVisible(visible)
                if not visible:
                    continue
                if index < len(colors):
                    box.setColor(colors[index])
                self._set_box_transform(box, self.target_zones[index], self.target_sizes[index])

        def _task_progress(self) -> Tuple[int, int, bool]:
            if self.active_task == "free":
                return 0, 0, False
            scene = self.TASK_SCENES[self.active_task]
            assignments = list(scene.get("assignments", []))
            completed = 0
            if self.active_task == "stack":
                zone_xy = np.asarray(self.target_zones[0][:2], dtype=float)
                completed = sum(
                    float(np.linalg.norm(np.asarray(position[:2], dtype=float) - zone_xy)) <= 0.35
                    for position in self.object_positions
                )
            else:
                for object_index, zone_index in enumerate(assignments):
                    object_xy = np.asarray(self.object_positions[object_index][:2], dtype=float)
                    zone_xy = np.asarray(self.target_zones[zone_index][:2], dtype=float)
                    if float(np.linalg.norm(object_xy - zone_xy)) <= 0.35:
                        completed += 1
            total = len(assignments)
            return int(completed), int(total), bool(total and completed == total)

        def execute_payload(self, payload: dict) -> None:
            duration = payload.get("duration_seconds", payload.get("duration"))
            target_position = payload.get("target_position")
            drop_zone_index = payload.get("drop_zone_index")
            self.execute_command(
                payload.get("cmd", payload.get("name", "")), payload.get("intent", ""), duration,
                target_position, drop_zone_index, bool(payload.get("continuous", False)),
                payload.get("move_speed"), payload.get("velocity"),
            )

        def execute_command(
            self,
            command: str,
            intent: str = "",
            duration_seconds: Optional[float] = None,
            target_position: Optional[Sequence[float]] = None,
            drop_zone_index: Optional[int] = None,
            continuous: bool = False,
            move_speed: Optional[float] = None,
            velocity: Optional[Sequence[float]] = None,
        ) -> None:
            if command == "stop_motion":
                self.stop_motion()
                self.mode_text = "Stopped"
                self._refresh_status()
                self.state_changed.emit(self.mode_text)
                return
            if continuous and command.startswith("move_"):
                self.stop_motion()
                intent_key = command.removeprefix("move_")
                default_vector = CONTINUOUS_MOVE_VECTORS.get(intent_key)
                if default_vector is None:
                    return
                vector = tuple(float(value) for value in (velocity or default_vector))
                if len(vector) != 3 or not any(abs(value) > 1e-9 for value in vector):
                    vector = default_vector
                self.continuous_direction = (vector, float(move_speed or 0.5))
                self.continuous_last_tick = time.perf_counter()
                self.mode_text = f"Move {intent_key}"
                self._refresh_status()
                self.state_changed.emit(self.mode_text)
                return
            if self.continuous_direction is not None:
                self.stop_motion()
            intent = resolve_robot_intent(command, intent)
            if intent == "up":
                self._start_command_motion(0.0, 0.0, 1.0, "Move up", duration_seconds, target_position)
            elif intent == "down":
                self._start_command_motion(0.0, 0.0, -1.0, "Move down", duration_seconds, target_position)
            elif intent == "left":
                self._start_command_motion(0.0, 1.0, 0.0, "Move left", duration_seconds, target_position)
            elif intent == "right":
                self._start_command_motion(0.0, -1.0, 0.0, "Move right", duration_seconds, target_position)
            elif intent == "forward":
                self._start_command_motion(1.0, 0.0, 0.0, "Move forward", duration_seconds, target_position)
            elif intent == "backward":
                self._start_command_motion(-1.0, 0.0, 0.0, "Move backward", duration_seconds, target_position)
            elif intent == "grab":
                self.stop_motion()
                self.finger_target = 0.0
                self.pending_grab = True
                self.mode_text = "Grab"
                self._maybe_grab_object()
            elif intent == "release":
                self.stop_motion()
                if self._opening_gripper_blocked():
                    self.mode_text = "Blocked: release collision"
                    self._refresh_status()
                    self.state_changed.emit(self.mode_text)
                    return
                self.finger_target = 0.35
                self.pending_grab = False
                self.preferred_drop_zone_index = int(drop_zone_index) if drop_zone_index is not None else None
                self._release_object()
                self.mode_text = "Release"
            elif intent == "home":
                self.reset_scene()
                return
            self._update_gl_items()
            self._refresh_status()
            self.state_changed.emit(self.mode_text)

        def _clamp_target(self, target: Sequence[float]) -> List[float]:
            return [
                min(8.0, max(0.0, float(target[0]))),
                min(5.0, max(-5.0, float(target[1]))),
                min(4.0, max(1.0, float(target[2]))),
            ]

        def _grid_cell(self, position: Sequence[float]) -> List[float]:
            clamped = self._clamp_target(position)
            return [float(int(round(clamped[0]))), float(int(round(clamped[1]))), float(int(round(clamped[2])))]

        def _grid_path_to(self, target_position: Sequence[float]) -> List[List[float]]:
            current = self._grid_cell(self.ee_target)
            target = self._grid_cell(target_position)
            path: List[List[float]] = []
            cursor = current[:]
            for axis in range(3):
                while int(round(cursor[axis])) != int(round(target[axis])):
                    direction = 1.0 if target[axis] > cursor[axis] else -1.0
                    cursor[axis] = float(int(round(cursor[axis] + direction)))
                    path.append(cursor[:])
            return path

        def _start_motion_segment(self, target: Sequence[float], mode_text: str) -> None:
            duration = self.grid_cell_duration
            self.motion_queue = []
            self.motion_start = [float(value) for value in self._grid_cell(self.ee_target)]
            self.ee_target = self.motion_start[:]
            self.motion_target = self._clamp_target(target)
            self.motion_started_at = time.perf_counter()
            self.motion_duration = duration
            self.motion_until = self.motion_started_at + duration
            self.mode_text = mode_text

        def _start_command_motion(self, dx: float, dy: float, dz: float, mode_text: str, duration_seconds: Optional[float], target_position: Optional[Sequence[float]]) -> None:
            if target_position is not None:
                self.start_target_motion(target_position, mode_text, self.grid_cell_duration)
                return
            current_cell = [
                round(float(self.ee_target[0])),
                round(float(self.ee_target[1])),
                round(float(self.ee_target[2])),
            ]
            target = [
                current_cell[0] + float(dx) * self.grid_step,
                current_cell[1] + float(dy) * self.grid_step,
                current_cell[2] + float(dz) * self.grid_step,
            ]
            if self._movement_target_blocked(target):
                self.stop_motion()
                self.mode_text = "Blocked: collision"
                self._refresh_status()
                self.state_changed.emit(self.mode_text)
                return
            self.start_target_motion(target, mode_text, self.grid_cell_duration)

        def start_target_motion(self, target_position: Sequence[float], mode_text: str, duration_seconds: Optional[float] = None) -> None:
            path = self._grid_path_to(target_position)
            if any(self._movement_target_blocked(cell) for cell in path):
                self.stop_motion()
                self.mode_text = "Blocked: collision"
                self._refresh_status()
                self.state_changed.emit(self.mode_text)
                return
            if not path:
                self.ee_target = self._grid_cell(target_position)
                self.motion_queue = []
                self.motion_target = None
                self.motion_until = 0.0
                self.motion_duration = 0.0
                self.mode_text = mode_text
                return
            self._start_motion_segment(path[0], mode_text)

        def stop_motion(self) -> None:
            self.continuous_direction = None
            self.motion_queue = []
            if self.motion_target is not None:
                self.ee_target = [float(value) for value in self.motion_target]
            self.motion_target = None
            self.motion_until = 0.0
            self.motion_duration = 0.0

        def reset_scene(self) -> None:
            self.ee_target = self.home_target[:]
            self.motion_start = self.home_target[:]
            self.motion_target = None
            self.motion_queue = []
            self.motion_until = 0.0
            self.motion_duration = 0.0
            self.finger_target = 0.35
            self.pending_grab = False
            self.grabbed_object_index = None
            self.preferred_drop_zone_index = None
            self.object_positions = [list(spec["home"]) for spec in self.object_specs]
            self.mode_text = "Ready"
            self._update_gl_items()
            self._refresh_status()
            self.state_changed.emit(self.mode_text)

        def tick(self) -> None:
            now = time.perf_counter()
            if self.continuous_direction is not None:
                vector, speed = self.continuous_direction
                dt = min(0.05, max(0.0, now - self.continuous_last_tick))
                self.continuous_last_tick = now
                target = integrate_velocity_position(self.ee_target, vector, speed, dt)
                clamped = self._clamp_target(target)
                if not self._movement_target_blocked(clamped):
                    moved = any(abs(float(a) - float(b)) > 1e-8 for a, b in zip(clamped, self.ee_target))
                    if moved:
                        self.ee_target = clamped
                    else:
                        self.continuous_direction = None
                        self.mode_text = "Blocked: boundary"
                        self._refresh_status()
                        self.state_changed.emit(self.mode_text)
                else:
                    self.continuous_direction = None
                    self.mode_text = "Blocked: collision"
                    self._refresh_status()
                    self.state_changed.emit(self.mode_text)
            if self.motion_target is not None:
                alpha = min(1.0, max(0.0, (now - self.motion_started_at) / max(1e-6, self.motion_duration)))
                self.ee_target = [
                    float(start + (target - start) * alpha)
                    for start, target in zip(self.motion_start, self.motion_target)
                ]
                if alpha >= 1.0:
                    self.ee_target = self.motion_target[:]
                    self.motion_target = None
                    self.motion_until = 0.0
                    self.motion_duration = 0.0
            if self.grabbed_object_index is not None:
                self.object_positions[self.grabbed_object_index] = self._grasp_anchor().tolist()
            if self.pending_grab:
                self._maybe_grab_object()
            self._update_gl_items()
            self._refresh_status()

        def _arm_points(self) -> np.ndarray:
            target = np.asarray(self.ee_target, dtype=np.float32)
            base = np.asarray([0.0, 0.0, 0.25], dtype=np.float32)
            shoulder = np.asarray([0.9, 0.0, 1.8], dtype=np.float32)
            elbow = np.asarray([
                1.2 + (float(target[0]) - 1.2) * 0.42,
                float(target[1]) * 0.22,
                max(2.4, float(target[2]) + 1.2),
            ], dtype=np.float32)
            wrist = np.asarray([
                1.6 + (float(target[0]) - 1.6) * 0.72,
                float(target[1]) * 0.68,
                float(target[2]) + 0.65,
            ], dtype=np.float32)
            tool = target + np.asarray([0.0, 0.0, 0.35], dtype=np.float32)
            return np.asarray([base, shoulder, elbow, wrist, tool, target], dtype=np.float32)

        def _gripper_geometry(self, root_position: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            root = np.asarray(root_position, dtype=float)
            side = np.array([0.0, 1.0, 0.0], dtype=float)
            down = np.array([0.0, 0.0, -1.0], dtype=float)
            open_amount = max(0.0, min(1.0, self.finger_target / 0.35))
            open_span = 0.25 + 0.55 * open_amount
            knuckle_span = open_span * 0.55
            jaw_depth = 1.15
            left_knuckle = root + side * knuckle_span
            right_knuckle = root - side * knuckle_span
            left_tip = root + side * open_span + down * jaw_depth
            right_tip = root - side * open_span + down * jaw_depth
            crossbar = np.asarray([left_knuckle, root, right_knuckle], dtype=np.float32)
            left = np.asarray([root, left_knuckle, left_tip], dtype=np.float32)
            right = np.asarray([root, right_knuckle, right_tip], dtype=np.float32)
            grasp_anchor = root + down * (jaw_depth * 0.82)
            return crossbar, left, right, grasp_anchor

        def _grasp_anchor(self) -> np.ndarray:
            _crossbar, _left, _right, anchor = self._gripper_geometry(self.ee_target)
            return np.asarray(anchor, dtype=float)

        def _maybe_grab_object(self) -> None:
            if self.grabbed_object_index is not None:
                self.pending_grab = False
                return
            # A grab command applies only at the current position.
            self.pending_grab = False
            anchor = self._grasp_anchor()
            best_index: Optional[int] = None
            best_score = float("inf")
            for index, position in enumerate(self.object_positions):
                pos = np.asarray(position, dtype=float)
                xy_distance = float(np.linalg.norm(anchor[:2] - pos[:2]))
                z_distance = abs(float(anchor[2] - pos[2]))
                if xy_distance > 0.85 or z_distance > 0.90:
                    continue
                score = xy_distance + z_distance * 0.55
                if score < best_score:
                    best_score = score
                    best_index = index
            if best_index is None:
                self.mode_text = "Grab missed"
                self.state_changed.emit(self.mode_text)
                return
            self.grabbed_object_index = best_index
            self.object_positions[best_index] = anchor.tolist()
            self.pending_grab = False
            self.mode_text = "Grabbed"
            self.state_changed.emit(self.mode_text)

        def _release_object(self) -> None:
            if self.grabbed_object_index is not None:
                self._settle_released_object(self.grabbed_object_index)
            self.grabbed_object_index = None
            self.preferred_drop_zone_index = None

        def _settle_released_object(self, object_index: int) -> None:
            """Drop vertically onto the highest support without changing X/Y."""
            if object_index < 0 or object_index >= len(self.object_positions):
                return
            position = self.object_positions[object_index]
            object_size = self.object_specs[object_index]["size"]
            x, y, current_z = [float(value) for value in position]
            half_x = float(object_size[0]) / 2.0
            half_y = float(object_size[1]) / 2.0
            half_z = float(object_size[2]) / 2.0
            support_top = float(self.floor_z)

            for zone, size in zip(self.target_zones, self.target_sizes):
                overlaps_x = abs(x - float(zone[0])) < half_x + float(size[0]) / 2.0
                overlaps_y = abs(y - float(zone[1])) < half_y + float(size[1]) / 2.0
                if overlaps_x and overlaps_y:
                    support_top = max(support_top, float(zone[2]) + float(size[2]) / 2.0)

            for other_index, other_position in enumerate(self.object_positions):
                if other_index == object_index:
                    continue
                other_size = self.object_specs[other_index]["size"]
                overlaps_x = abs(x - float(other_position[0])) < half_x + float(other_size[0]) / 2.0
                overlaps_y = abs(y - float(other_position[1])) < half_y + float(other_size[1]) / 2.0
                other_top = float(other_position[2]) + float(other_size[2]) / 2.0
                if overlaps_x and overlaps_y:
                    support_top = max(support_top, other_top)

            self.object_positions[object_index] = [x, y, support_top + half_z]

        @staticmethod
        def _boxes_overlap(
            center_a: Sequence[float],
            size_a: Sequence[float],
            center_b: Sequence[float],
            size_b: Sequence[float],
            clearance: float = -0.01,
        ) -> bool:
            return all(
                abs(float(center_a[axis]) - float(center_b[axis]))
                < (float(size_a[axis]) + float(size_b[axis])) / 2.0 + float(clearance)
                for axis in range(3)
            )

        def _grasp_anchor_for_target(self, target: Sequence[float]) -> np.ndarray:
            _crossbar, _left, _right, anchor = self._gripper_geometry(target)
            return np.asarray(anchor, dtype=float)

        def _gripper_collision_boxes(
            self,
            target: Sequence[float],
            finger_target: Optional[float] = None,
        ) -> List[Tuple[np.ndarray, Tuple[float, float, float]]]:
            root = np.asarray(target, dtype=float)
            finger_value = self.finger_target if finger_target is None else float(finger_target)
            open_amount = max(0.0, min(1.0, finger_value / 0.35))
            open_span = 0.25 + 0.55 * open_amount
            jaw_depth = 1.15
            crossbar = (root, (0.22, max(0.58, open_span * 1.25), 0.16))
            left_finger = (
                root + np.asarray([0.0, open_span, -jaw_depth / 2.0]),
                (0.16, 0.12, jaw_depth),
            )
            right_finger = (
                root + np.asarray([0.0, -open_span, -jaw_depth / 2.0]),
                (0.16, 0.12, jaw_depth),
            )
            return [crossbar, left_finger, right_finger]

        def _opening_gripper_blocked(self) -> bool:
            root = np.asarray(self.ee_target, dtype=float)
            ignored_index = self.grabbed_object_index
            jaw_depth = 1.15
            closed_span = 0.25
            open_span = 0.80
            sweep_center = (closed_span + open_span) / 2.0
            sweep_width = open_span - closed_span + 0.12
            swept_fingers = [
                (root + np.asarray([0.0, sweep_center, -jaw_depth / 2.0]), (0.16, sweep_width, jaw_depth)),
                (root + np.asarray([0.0, -sweep_center, -jaw_depth / 2.0]), (0.16, sweep_width, jaw_depth)),
            ]
            components = self._gripper_collision_boxes(root, finger_target=0.35) + swept_fingers
            for component_center, component_size in components:
                if float(component_center[2]) - float(component_size[2]) / 2.0 < self.floor_z - 0.01:
                    return True
                for object_index, position in enumerate(self.object_positions):
                    if object_index == ignored_index:
                        continue
                    if self._boxes_overlap(
                        component_center,
                        component_size,
                        position,
                        self.object_specs[object_index]["size"],
                        clearance=-0.06,
                    ):
                        return True
            return False

        def _movement_target_blocked(self, target: Sequence[float]) -> bool:
            target = self._clamp_target(target)
            if self.grabbed_object_index is not None:
                moving_index = self.grabbed_object_index
                moving_center = self._grasp_anchor_for_target(target)
                moving_size = self.object_specs[moving_index]["size"]
                if float(moving_center[2]) - float(moving_size[2]) / 2.0 < self.floor_z:
                    return True
                for other_index, other_position in enumerate(self.object_positions):
                    if other_index == moving_index:
                        continue
                    if self._boxes_overlap(
                        moving_center,
                        moving_size,
                        other_position,
                        self.object_specs[other_index]["size"],
                    ):
                        return True
                return False

            for component_center, component_size in self._gripper_collision_boxes(target):
                if float(component_center[2]) - float(component_size[2]) / 2.0 < self.floor_z - 0.01:
                    return True
                for index, position in enumerate(self.object_positions):
                    if self._boxes_overlap(
                        component_center,
                        component_size,
                        position,
                        self.object_specs[index]["size"],
                        clearance=-0.06,
                    ):
                        return True
            return False

        def _update_gl_items(self) -> None:
            points = self._arm_points()
            self.arm_line.setData(pos=points)
            crossbar, left, right, _anchor = self._gripper_geometry(points[-1])
            self.gripper_crossbar.setData(pos=crossbar)
            self.left_gripper.setData(pos=left)
            self.right_gripper.setData(pos=right)
            for position, box, spec in zip(self.object_positions, self.object_boxes, self.object_specs):
                self._set_box_transform(box, position, spec["size"])

        def _refresh_status(self) -> None:
            attach = "已抓取" if self.grabbed_object_index is not None else "未抓取"
            motion_left = max(0.0, self.motion_until - time.perf_counter())
            motion_text = "持续运动中" if self.continuous_direction is not None else ("执行中" if motion_left > 0.0 else "停止")
            mode_text = {
                "Ready": "待命",
                "Move up": "上移",
                "Move down": "下移",
                "Move left": "左移",
                "Move right": "右移",
                "Move forward": "前移",
                "Move backward": "后移",
                "Grab": "抓取",
                "Grabbed": "已抓取",
                "Release": "释放",
                "Grab missed": "未抓到物体",
                "Blocked: collision": "检测到碰撞，移动已阻止",
                "Blocked: boundary": "到达场景边界，移动已停止",
                "Blocked: release collision": "开爪路径发生碰撞，释放已阻止",
            }.get(self.mode_text, self.mode_text)
            self.state_badge.setText(mode_text)
            scene = self.TASK_SCENES[self.active_task]
            completed, total, task_done = self._task_progress()
            if self.active_task == "free":
                progress_text = "自由操作"
            elif task_done:
                progress_text = f"任务完成 {completed}/{total}"
            else:
                progress_text = f"进度 {completed}/{total}"
            self.task_goal_label.setText(f"{scene['name']} | {scene['goal']} | {progress_text}")
            self.status_label.setText(
                f"3D机械臂：{mode_text} | 坐标：{self._axis_state_text()} | 方块：{attach} | 运动：{motion_text}"
            )

        def _axis_state_text(self) -> str:
            x, y, z = [float(value) for value in self.ee_target]
            if x >= 6.0:
                x_text = "X+"
            elif x <= 2.0:
                x_text = "X-"
            else:
                x_text = "X0"
            if y >= 2.0:
                y_text = "Y+"
            elif y <= -2.0:
                y_text = "Y-"
            else:
                y_text = "Y0"
            if z >= 2.5:
                z_text = "Z+"
            elif z <= 1.5:
                z_text = "Z-"
            else:
                z_text = "Z0"
            return f"X={int(round(x))} / Y={int(round(y))} / Z={int(round(z))}"

    class ConfigPage(QtWidgets.QWidget):
        config_changed = QtCore.pyqtSignal()
        DEFAULTS_PATH = os.path.join(APP_DIR, "config", "gui_defaults.json")

        def __init__(self, parent=None):
            super().__init__(parent)
            self.defaults_migration_notes: Tuple[str, ...] = tuple()
            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel(CONFIG_PAGE_TITLE)
            title.setObjectName("title")
            layout.addWidget(title)
            desc = QtWidgets.QLabel(CONFIG_PAGE_DESC)
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            layout.addWidget(desc)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            content = QtWidgets.QWidget()
            content_layout = QtWidgets.QVBoxLayout(content)
            self.setup_tabs = QtWidgets.QTabWidget()
            content_layout.addWidget(self.setup_tabs)
            collection_page = QtWidgets.QWidget()
            collection_layout = QtWidgets.QVBoxLayout(collection_page)
            model_page = QtWidgets.QWidget()
            model_layout = QtWidgets.QVBoxLayout(model_page)
            runtime_page = QtWidgets.QWidget()
            runtime_layout = QtWidgets.QVBoxLayout(runtime_page)
            for tab, label in ((collection_page, "采集与类别"), (model_page, "模型训练"), (runtime_page, "实时与硬件")):
                self.setup_tabs.addTab(tab, label)

            collection_card = Card()
            collection_form = QtWidgets.QFormLayout(collection_card)
            self.collection_mode_combo = QtWidgets.QComboBox()
            self.collection_mode_combo.addItems(["BrainVision LSL（EEG+肌电）"])
            collection_form.addRow("采集方式", self.collection_mode_combo)
            self.paradigm_combo = QtWidgets.QComboBox()
            self.paradigm_combo.addItem("原八分类范式", "eeg8")
            self.paradigm_combo.addItem("纯 MI 二分类（左手=1 / 右手=0）", "pure_mi_binary")
            self.paradigm_combo.addItem("自定义 MI 分类（2–16 类）", "custom_mi")
            self.paradigm_combo.setToolTip("该选择同时用于数据采集、模型训练和在线解码。")
            collection_form.addRow("采集/在线范式", self.paradigm_combo)
            self.class_count_spin = QtWidgets.QSpinBox()
            self.class_count_spin.setRange(2, 16)
            self.class_count_spin.setValue(8)
            collection_form.addRow("类别数量", self.class_count_spin)
            self.category_table = QtWidgets.QTableWidget(8, 2)
            self.category_table.setHorizontalHeaderLabels(["编号", "类别名称"])
            self.category_table.verticalHeader().setVisible(False)
            self.category_table.setMaximumHeight(250)
            self.category_table.horizontalHeader().setStretchLastSection(True)
            for row, name in enumerate(CLASS_NAMES.values()):
                self.category_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"C{row + 1}"))
                self.category_table.setItem(row, 1, QtWidgets.QTableWidgetItem(name))
            collection_form.addRow("类别定义", self.category_table)
            self.class_count_spin.valueChanged.connect(self.sync_category_rows)
            self.collection_out_edit = QtWidgets.QLineEdit(RECORDINGS_DIR)
            collection_form.addRow("采集归档目录", self.path_row(self.collection_out_edit, self.pick_collection_out, "选择"))
            self.fs_spin = QtWidgets.QSpinBox()
            self.fs_spin.setRange(128, 2000)
            self.fs_spin.setValue(250)
            collection_form.addRow("采样率 Hz", self.fs_spin)
            self.trials_spin = QtWidgets.QSpinBox()
            self.trials_spin.setRange(1, 60)
            self.trials_spin.setValue(12)
            collection_form.addRow("每类 Trial 数", self.trials_spin)
            self.epoch_spin = QtWidgets.QDoubleSpinBox()
            self.epoch_spin.setRange(0.5, 8.0)
            self.epoch_spin.setSingleStep(0.1)
            self.epoch_spin.setValue(4.0)
            collection_form.addRow("采集/执行秒数", self.epoch_spin)
            self.pre_rest_spin = QtWidgets.QDoubleSpinBox()
            self.pre_rest_spin.setRange(0.0, 10.0)
            self.pre_rest_spin.setSingleStep(0.5)
            self.pre_rest_spin.setValue(8.0)
            collection_form.addRow("静息/基线秒数", self.pre_rest_spin)
            self.cue_spin = QtWidgets.QDoubleSpinBox()
            self.cue_spin.setRange(0.0, 5.0)
            self.cue_spin.setSingleStep(0.5)
            self.cue_spin.setValue(1.0)
            collection_form.addRow("提示秒数", self.cue_spin)
            self.cue_rest_spin = QtWidgets.QDoubleSpinBox()
            self.cue_rest_spin.setRange(0.0, 10.0)
            self.cue_rest_spin.setSingleStep(0.5)
            self.cue_rest_spin.setValue(3.0)
            collection_form.addRow("提示后静息", self.cue_rest_spin)
            self.post_rest_spin = QtWidgets.QDoubleSpinBox()
            self.post_rest_spin.setRange(0.0, 10.0)
            self.post_rest_spin.setSingleStep(0.5)
            self.post_rest_spin.setValue(3.0)
            collection_form.addRow("采集后静息", self.post_rest_spin)
            collection_layout.addWidget(collection_card)

            io_card = Card()
            io_layout = QtWidgets.QVBoxLayout(io_card)
            io_title = QtWidgets.QLabel("硬件接口")
            io_title.setObjectName("vrTitle")
            io_layout.addWidget(io_title)
            lsl_hint = QtWidgets.QLabel("BrainVision LSL 自动发现 LiveAmp / EEG 流；采集与 VR 在线默认读取前 32 路 EEG，多余通道不参与八分类控制。")
            lsl_hint.setWordWrap(True)
            lsl_hint.setObjectName("hint")
            io_layout.addWidget(lsl_hint)
            runtime_options = QtWidgets.QHBoxLayout()
            self.use_model_check = QtWidgets.QCheckBox("使用当前模型/权重")
            self.use_model_check.setChecked(True)
            self.use_weight_check = QtWidgets.QCheckBox("使用 Config 阈值")
            self.use_weight_check.setChecked(True)
            runtime_options.addWidget(self.use_model_check)
            runtime_options.addWidget(self.use_weight_check)
            runtime_options.addStretch(1)
            io_layout.addLayout(runtime_options)
            self.rda_host_edit = QtWidgets.QLineEdit("auto")
            self.rda_port_spin = QtWidgets.QSpinBox()
            self.rda_port_spin.setRange(1, 65535)
            self.rda_port_spin.setValue(51244)
            self.rda_channels_spin = QtWidgets.QSpinBox()
            self.rda_channels_spin.setRange(1, 128)
            self.rda_channels_spin.setValue(32)
            self.aux_channels_spin = QtWidgets.QSpinBox()
            self.aux_channels_spin.setRange(0, 16)
            self.aux_channels_spin.setValue(4)
            self.parallel_check = QtWidgets.QCheckBox("发送并口 trigger")
            self.parallel_check.setChecked(True)
            self.trigger_port_edit = QtWidgets.QLineEdit("0x0378")

            paradigm_title = QtWidgets.QLabel("标准动作参考")
            paradigm_title.setObjectName("vrTitle")
            io_layout.addWidget(paradigm_title)
            paradigm_text = QtWidgets.QTextEdit()
            paradigm_text.setReadOnly(True)
            paradigm_text.setMinimumHeight(210)
            lines = []
            for class_id in range(1, 9):
                task = eeg8_task_map()[class_id]
                lines.append(
                    f"C{class_id} {task['class_name']} -> {CLASS_ARM_LABELS[class_id]} "
                    f"({CLASS_TO_ARM[class_id]}) | {task.get('cue_title', '')}"
                )
            paradigm_text.setPlainText("\n".join(lines))
            io_layout.addWidget(paradigm_text, 1)

            self.save_defaults_button = QtWidgets.QPushButton("保存为默认配置")
            self.save_defaults_button.setObjectName("primaryButton")
            self.save_defaults_button.clicked.connect(self.save_defaults)
            io_layout.addWidget(self.save_defaults_button)
            self.defaults_status_label = QtWidgets.QLabel("默认配置：未保存")
            self.defaults_status_label.setObjectName("hint")
            self.defaults_status_label.setWordWrap(True)
            io_layout.addWidget(self.defaults_status_label)
            runtime_layout.addWidget(io_card)

            model_card = Card()
            model_form = QtWidgets.QFormLayout(model_card)
            self.train_data_edit = QtWidgets.QLineEdit("")
            self.train_data_edit.setPlaceholderText("在模型训练页勾选真实采集记录")
            model_form.addRow("最近训练记录", self.path_row(self.train_data_edit, self.pick_train_data, "选择"))
            self.test_data_edit = QtWidgets.QLineEdit("")
            self.test_data_edit.setPlaceholderText("在模型训练页勾选独立测试记录")
            model_form.addRow("最近测试记录", self.path_row(self.test_data_edit, self.pick_test_data, "选择"))
            self.model_edit = QtWidgets.QLineEdit(os.path.join(APP_DIR, "models", "arm_bci_gui.joblib"))
            model_form.addRow("模型/权重", self.path_row(self.model_edit, self.pick_model_out, "选择"))
            sync_data = QtWidgets.QPushButton("采集输出 -> 训练/测试")
            sync_data.clicked.connect(self.sync_data_paths)
            model_form.addRow(sync_data)
            self.validation_strategy_combo = QtWidgets.QComboBox()
            self.validation_strategy_combo.addItem("分层 K 折交叉验证", "stratified_cv")
            self.validation_strategy_combo.addItem("仅训练最终模型", "final_fit")
            self.validation_strategy_combo.addItem("训练后立即独立测试", "train_then_test")
            model_form.addRow("验证/测试策略", self.validation_strategy_combo)
            self.feature_strategy_combo = QtWidgets.QComboBox()
            self.feature_strategy_combo.addItem("FilterBank + CSP + BandPower", "fbcsp")
            self.feature_strategy_combo.addItem("FilterBank BandPower（不使用 CSP）", "bandpower")
            model_form.addRow("特征策略", self.feature_strategy_combo)
            self.eeg_weight = QtWidgets.QDoubleSpinBox()
            self.eeg_weight.setRange(0.35, 0.85)
            self.eeg_weight.setSingleStep(0.05)
            self.eeg_weight.setValue(0.55)
            self.eeg_weight.setEnabled(False)
            self.conf_spin = QtWidgets.QDoubleSpinBox()
            self.conf_spin.setRange(0.40, 0.95)
            self.conf_spin.setSingleStep(0.02)
            self.conf_spin.setValue(0.62)
            self.margin_spin = QtWidgets.QDoubleSpinBox()
            self.margin_spin.setRange(0.00, 0.50)
            self.margin_spin.setSingleStep(0.01)
            self.margin_spin.setValue(0.08)
            self.classifier_combo = QtWidgets.QComboBox()
            self.classifier_combo.addItem("EEGNet + Shallow MI 概率融合（6 个成员）", "best_mi_fusion")
            self.classifier_combo.setToolTip("基于本项目采集的 C1–C8 EEG 数据重新训练；不使用外部二分类权重。")
            model_form.addRow("运动想象算法", self.classifier_combo)
            self.csp_check = QtWidgets.QCheckBox("启用 FBCSP 特征")
            self.csp_check.setChecked(True)
            self.csp_check.setVisible(False)
            model_form.addRow(self.csp_check)
            model_layout.addWidget(model_card)

            realtime_card = Card()
            realtime_form = QtWidgets.QFormLayout(realtime_card)
            self.vr_model_edit = QtWidgets.QLineEdit(os.path.join(APP_DIR, "models", "arm_bci_gui.joblib"))
            realtime_form.addRow("VR 模型/权重", self.path_row(self.vr_model_edit, self.pick_vr_model, "选择"))
            sync_model = QtWidgets.QPushButton("模型输出 -> VR")
            sync_model.clicked.connect(lambda: self.set_model_path(self.model_edit.text().strip()))
            realtime_form.addRow(sync_model)
            self.online_mode_combo = QtWidgets.QComboBox()
            self.online_mode_combo.addItems(["BrainVision LSL（32导EEG+AUX）"])
            realtime_form.addRow("实时采集方式", self.online_mode_combo)
            self.runtime_strategy_combo = QtWidgets.QComboBox()
            self.runtime_strategy_combo.addItem("纯 EEG（不读取 AUX；安全阈值后单次执行）", EEG8_RUNTIME_STRATEGY_EEG_ONLY)
            self.runtime_strategy_combo.addItem("EEG + 咬肌确认（旧路径）", EEG8_RUNTIME_STRATEGY_MASSETER)
            self.runtime_strategy_combo.addItem("仅 VR 分类展示（禁止机械臂输出）", "vr_only")
            realtime_form.addRow("实时运行策略", self.runtime_strategy_combo)
            self.decode_window_spin = QtWidgets.QDoubleSpinBox()
            self.decode_window_spin.setRange(0.5, 60.0)
            self.decode_window_spin.setSingleStep(0.05)
            self.decode_window_spin.setDecimals(2)
            self.decode_window_spin.setValue(2.0)
            self.decode_window_spin.setSuffix(" s")
            realtime_form.addRow("窗口1：EEG+EMG 联合时长", self.decode_window_spin)
            self.decode_step_spin = QtWidgets.QDoubleSpinBox()
            self.decode_step_spin.setRange(0.1, 60.0)
            self.decode_step_spin.setSingleStep(0.05)
            self.decode_step_spin.setDecimals(2)
            self.decode_step_spin.setValue(2.0)
            self.decode_step_spin.setSuffix(" s")
            self.decode_step_spin.setVisible(False)
            self.decode_window_spin.valueChanged.connect(self.decode_step_spin.setValue)
            self.confirm_window_spin = QtWidgets.QDoubleSpinBox()
            self.confirm_window_spin.setRange(0.2, 60.0)
            self.confirm_window_spin.setSingleStep(0.05)
            self.confirm_window_spin.setDecimals(2)
            self.confirm_window_spin.setValue(1.0)
            self.confirm_window_spin.setSuffix(" s")
            realtime_form.addRow("窗口2：咬肌确认时长", self.confirm_window_spin)
            self.confirm_masseter_count_spin = QtWidgets.QSpinBox()
            self.confirm_masseter_count_spin.setRange(1, 999)
            self.confirm_masseter_count_spin.setValue(1)
            self.confirm_masseter_count_spin.setToolTip("窗口2内左右咬肌执行计数之和达到该值，才执行窗口1结果。")
            realtime_form.addRow("窗口2：确认咬肌次数", self.confirm_masseter_count_spin)
            self.continuous_move_speed_spin = QtWidgets.QDoubleSpinBox()
            self.continuous_move_speed_spin.setRange(0.1, 100.0)
            self.continuous_move_speed_spin.setSingleStep(0.05)
            self.continuous_move_speed_spin.setDecimals(2)
            self.continuous_move_speed_spin.setValue(0.5)
            self.continuous_move_speed_spin.setSuffix(" 格/秒")
            realtime_form.addRow("连续移动速度", self.continuous_move_speed_spin)
            self.continuous_blink_window_spin = QtWidgets.QDoubleSpinBox()
            self.continuous_blink_window_spin.setRange(0.5, 60.0)
            self.continuous_blink_window_spin.setSingleStep(0.05)
            self.continuous_blink_window_spin.setDecimals(2)
            self.continuous_blink_window_spin.setValue(2.0)
            self.continuous_blink_window_spin.setSuffix(" s")
            realtime_form.addRow("窗口3：持续运动眨眼时长", self.continuous_blink_window_spin)
            self.continuous_blink_count_spin = QtWidgets.QSpinBox()
            self.continuous_blink_count_spin.setRange(1, 999)
            self.continuous_blink_count_spin.setValue(3)
            realtime_form.addRow("窗口3：停止眨眼次数", self.continuous_blink_count_spin)
            self.target_edit = QtWidgets.QLineEdit("target_1")
            realtime_form.addRow("VR 目标区域", self.target_edit)
            self.robot_enabled_check = QtWidgets.QCheckBox("启用真实机械臂 TCP 输出")
            self.robot_enabled_check.setChecked(False)
            realtime_form.addRow("真实机械臂", self.robot_enabled_check)
            self.robot_host_edit = QtWidgets.QLineEdit("192.168.1.100")
            realtime_form.addRow("机械臂接收端 IP", self.robot_host_edit)
            self.robot_port_spin = QtWidgets.QSpinBox()
            self.robot_port_spin.setRange(1, 65535)
            self.robot_port_spin.setValue(9527)
            realtime_form.addRow("机械臂接收端端口", self.robot_port_spin)
            self.robot_speed_spin = QtWidgets.QDoubleSpinBox()
            self.robot_speed_spin.setRange(0.01, 10000.0)
            self.robot_speed_spin.setSingleStep(0.1)
            self.robot_speed_spin.setDecimals(2)
            self.robot_speed_spin.setValue(50.0)
            self.robot_speed_spin.setSuffix(" (协议速度)")
            realtime_form.addRow("机械臂移动速度", self.robot_speed_spin)
            self.refractory_spin = QtWidgets.QDoubleSpinBox()
            self.refractory_spin.setRange(0.0, 60.0)
            self.refractory_spin.setSingleStep(0.01)
            self.refractory_spin.setDecimals(2)
            self.refractory_spin.setValue(0.25)
            realtime_form.addRow("指令冷却秒数", self.refractory_spin)
            self.command_duration_spin = QtWidgets.QDoubleSpinBox()
            self.command_duration_spin.setRange(0.01, 60.0)
            self.command_duration_spin.setSingleStep(0.01)
            self.command_duration_spin.setDecimals(2)
            self.command_duration_spin.setValue(0.55)
            realtime_form.addRow("VR 指令保持秒数", self.command_duration_spin)
            self.aux_blink_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.aux_blink_ratio_spin.setRange(1.2, 30.0)
            self.aux_blink_ratio_spin.setSingleStep(0.2)
            self.aux_blink_ratio_spin.setValue(4.0)
            realtime_form.addRow("AUX 眨眼阈值", self.aux_blink_ratio_spin)
            self.aux_bite_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.aux_bite_ratio_spin.setRange(1.2, 30.0)
            self.aux_bite_ratio_spin.setSingleStep(0.2)
            self.aux_bite_ratio_spin.setValue(3.0)
            realtime_form.addRow("AUX 咬肌阈值", self.aux_bite_ratio_spin)
            self.aux_side_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.aux_side_ratio_spin.setRange(1.0, 5.0)
            self.aux_side_ratio_spin.setSingleStep(0.05)
            self.aux_side_ratio_spin.setValue(1.35)
            realtime_form.addRow("AUX 左右区分", self.aux_side_ratio_spin)
            self.aux_blink_count_spin = QtWidgets.QSpinBox()
            self.aux_blink_count_spin.setRange(1, 999)
            self.aux_blink_count_spin.setValue(3)
            realtime_form.addRow("状态切换眨眼次数", self.aux_blink_count_spin)
            realtime_form.addRow("双咬判定", QtWidgets.QLabel("左咬、右咬计数均达到各自门槛（无独立双咬计数）"))
            self.aux_left_count_spin = QtWidgets.QSpinBox()
            self.aux_left_count_spin.setRange(1, 999)
            self.aux_left_count_spin.setValue(1)
            realtime_form.addRow("窗口1：左咬映射次数", self.aux_left_count_spin)
            self.aux_right_count_spin = QtWidgets.QSpinBox()
            self.aux_right_count_spin.setRange(1, 999)
            self.aux_right_count_spin.setValue(1)
            realtime_form.addRow("窗口1：右咬映射次数", self.aux_right_count_spin)
            self.masseter_detector_combo = QtWidgets.QComboBox()
            self.masseter_detector_combo.addItem("NCC 模板匹配", "ncc")
            self.masseter_detector_combo.addItem("RMS 阈值检测", "threshold")
            self.masseter_detector_combo.addItem("精准模式（逐状态选择）", "precise")
            realtime_form.addRow("咬肌识别模式", self.masseter_detector_combo)
            self.masseter_state_detector_combos = {}
            state_labels = {
                "00": "00：无咬肌 / C3-C4",
                "01": "01：右咬肌 / C1-C2",
                "10": "10：左咬肌 / C5-C6",
                "11": "11：双侧咬肌 / C7-C8",
            }
            for state, label in state_labels.items():
                combo = QtWidgets.QComboBox()
                combo.addItem("通道阈值", "threshold")
                combo.addItem("NCC 模板匹配", "ncc")
                combo.setCurrentIndex(0 if state == "00" else 1)
                self.masseter_state_detector_combos[state] = combo
                realtime_form.addRow(f"精准 {label}", combo)
            self.masseter_template_edit = QtWidgets.QLineEdit(os.path.join(APP_DIR, "models", "masseter_ncc_templates.json"))
            realtime_form.addRow("NCC 眼电/肌电模板", self.path_row(self.masseter_template_edit, self.pick_masseter_template, "选择"))
            self.masseter_ncc_score_spin = QtWidgets.QDoubleSpinBox()
            self.masseter_ncc_score_spin.setRange(0.0, 1.0)
            self.masseter_ncc_score_spin.setSingleStep(0.02)
            self.masseter_ncc_score_spin.setDecimals(2)
            self.masseter_ncc_score_spin.setValue(0.45)
            realtime_form.addRow("NCC 最小分数", self.masseter_ncc_score_spin)
            self.masseter_ncc_margin_spin = QtWidgets.QDoubleSpinBox()
            self.masseter_ncc_margin_spin.setRange(0.0, 1.0)
            self.masseter_ncc_margin_spin.setSingleStep(0.02)
            self.masseter_ncc_margin_spin.setDecimals(2)
            self.masseter_ncc_margin_spin.setValue(0.08)
            realtime_form.addRow("NCC 最小边距", self.masseter_ncc_margin_spin)
            self.masseter_outside_weight_spin = QtWidgets.QDoubleSpinBox()
            self.masseter_outside_weight_spin.setRange(0.0, 1.0)
            self.masseter_outside_weight_spin.setSingleStep(0.05)
            self.masseter_outside_weight_spin.setDecimals(2)
            self.masseter_outside_weight_spin.setValue(0.30)
            realtime_form.addRow("映射外权重", self.masseter_outside_weight_spin)
            self.candidate_random_margin_spin = QtWidgets.QDoubleSpinBox()
            self.candidate_random_margin_spin.setRange(0.0, 1.0)
            self.candidate_random_margin_spin.setSingleStep(0.01)
            self.candidate_random_margin_spin.setDecimals(2)
            self.candidate_random_margin_spin.setValue(0.05)
            self.candidate_random_margin_spin.setToolTip("映射候选内两类概率差不超过该值时，在两类中随机选择；超过时选择概率较大者。")
            realtime_form.addRow("二类随机差值", self.candidate_random_margin_spin)
            runtime_layout.addWidget(realtime_card)
            collection_layout.addStretch(1)
            model_layout.addStretch(1)
            runtime_layout.addStretch(1)

            scroll.setWidget(content)
            layout.addWidget(scroll, 1)
            self.bind_change_signals()
            self.load_defaults()
            self.update_mode_fields()

        def path_row(self, edit: QtWidgets.QLineEdit, callback, text: str) -> QtWidgets.QWidget:
            box = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(box)
            row.setContentsMargins(0, 0, 0, 0)
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(edit)
            row.addWidget(button)
            return box

        def sync_category_rows(self, count: int) -> None:
            count = max(2, min(16, int(count)))
            previous = [
                self.category_table.item(row, 1).text().strip()
                if self.category_table.item(row, 1) else ""
                for row in range(self.category_table.rowCount())
            ]
            self.category_table.setRowCount(count)
            for row in range(count):
                self.category_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"C{row + 1}"))
                if not self.category_table.item(row, 1):
                    fallback = CLASS_NAMES.get(row + 1, f"类别 {row + 1}")
                    value = previous[row] if row < len(previous) and previous[row] else fallback
                    self.category_table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))

        def category_definitions(self) -> List[Dict[str, Any]]:
            definitions = []
            for row in range(self.category_table.rowCount()):
                item = self.category_table.item(row, 1)
                name = item.text().strip() if item else ""
                if not name:
                    raise ValueError(f"请填写 C{row + 1} 的类别名称")
                definitions.append({
                    "class_id": row + 1, "class_name": name, "class_name_en": name,
                    "cue_title": name, "cue_subtitle": "根据提示进行运动想象",
                    "instruction": f"想象与“{name}”对应的动作，不要实际运动。",
                    "forbidden": "不要实际运动或改变身体姿势。",
                })
            return definitions

        def bind_change_signals(self) -> None:
            widgets = [
                self.collection_mode_combo,
                self.paradigm_combo,
                self.class_count_spin,
                self.collection_out_edit,
                self.fs_spin,
                self.trials_spin,
                self.epoch_spin,
                self.pre_rest_spin,
                self.cue_spin,
                self.cue_rest_spin,
                self.post_rest_spin,
                self.rda_host_edit,
                self.rda_port_spin,
                self.rda_channels_spin,
                self.aux_channels_spin,
                self.parallel_check,
                self.trigger_port_edit,
                self.use_model_check,
                self.use_weight_check,
                self.train_data_edit,
                self.test_data_edit,
                self.model_edit,
                self.validation_strategy_combo,
                self.feature_strategy_combo,
                self.eeg_weight,
                self.conf_spin,
                self.margin_spin,
                self.classifier_combo,
                self.csp_check,
                self.vr_model_edit,
                self.online_mode_combo,
                self.runtime_strategy_combo,
                self.decode_window_spin,
                self.decode_step_spin,
                self.confirm_window_spin,
                self.confirm_masseter_count_spin,
                self.continuous_move_speed_spin,
                self.continuous_blink_window_spin,
                self.continuous_blink_count_spin,
                self.target_edit,
                self.robot_enabled_check,
                self.robot_host_edit,
                self.robot_port_spin,
                self.robot_speed_spin,
                self.refractory_spin,
                self.command_duration_spin,
                self.aux_blink_ratio_spin,
                self.aux_bite_ratio_spin,
                self.aux_side_ratio_spin,
                self.aux_blink_count_spin,
                self.aux_left_count_spin,
                self.aux_right_count_spin,
                self.masseter_detector_combo,
                *self.masseter_state_detector_combos.values(),
                self.masseter_template_edit,
                self.masseter_ncc_score_spin,
                self.masseter_ncc_margin_spin,
                self.masseter_outside_weight_spin,
                self.candidate_random_margin_spin,
            ]
            for widget in widgets:
                if isinstance(widget, QtWidgets.QLineEdit):
                    widget.textChanged.connect(self.config_changed.emit)
                elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    widget.valueChanged.connect(self.config_changed.emit)
                elif isinstance(widget, QtWidgets.QComboBox):
                    widget.currentTextChanged.connect(self.config_changed.emit)
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.toggled.connect(self.config_changed.emit)
            self.category_table.itemChanged.connect(lambda _item: self.config_changed.emit())
            self.collection_mode_combo.currentTextChanged.connect(self.update_mode_fields)
            self.paradigm_combo.currentTextChanged.connect(self.update_mode_fields)
            self.online_mode_combo.currentTextChanged.connect(self.update_mode_fields)
            self.feature_strategy_combo.currentTextChanged.connect(self.update_mode_fields)
            self.masseter_detector_combo.currentTextChanged.connect(self.update_mode_fields)

        def _combo_set_text(self, combo: QtWidgets.QComboBox, text: str) -> None:
            index = combo.findData(str(text))
            if index < 0:
                index = combo.findText(str(text))
            if index >= 0:
                combo.setCurrentIndex(index)

        def _combo_value(self, combo: QtWidgets.QComboBox) -> str:
            data = combo.currentData()
            return str(data if data is not None else combo.currentText())

        def _spin_set_value(self, spin, value) -> None:
            try:
                spin.setValue(type(spin.value())(value))
            except Exception:
                try:
                    spin.setValue(float(value))
                except Exception:
                    pass

        def export_defaults(self) -> Dict:
            realtime = dict(self.realtime_config())
            decoder = realtime.get("decoder")
            if isinstance(decoder, DecoderConfig):
                realtime["decoder"] = asdict(decoder)
            return {
                "collection": self.collection_config(),
                "paths": {
                    "train_data_path": self.train_data_edit.text().strip(),
                    "test_data_path": self.test_data_edit.text().strip(),
                    "model_path": self.model_edit.text().strip(),
                    "vr_model_path": self.vr_model_edit.text().strip(),
                },
                "model": {
                    "validation_strategy": self._combo_value(self.validation_strategy_combo),
                    "feature_strategy": self._combo_value(self.feature_strategy_combo),
                    "classifier": self._combo_value(self.classifier_combo),
                    "use_csp": self._combo_value(self.feature_strategy_combo) == "fbcsp",
                },
                "realtime": realtime,
            }

        def apply_defaults(self, data: Dict) -> None:
            if not isinstance(data, dict):
                return
            old_state = self.blockSignals(True)
            try:
                collection = dict(data.get("collection", {}) or {})
                self._combo_set_text(self.collection_mode_combo, collection.get("mode", self.collection_mode_combo.currentText()))
                self._combo_set_text(
                    self.paradigm_combo,
                    collection.get(
                        "paradigm_key",
                        dict(data.get("realtime", {}) or {}).get("paradigm_key", "eeg8"),
                    ),
                )
                definitions = collection.get("class_definitions", [])
                if definitions:
                    self.class_count_spin.setValue(len(definitions))
                    self.sync_category_rows(len(definitions))
                    for row, definition in enumerate(definitions):
                        self.category_table.item(row, 1).setText(str(definition.get("class_name", f"类别 {row + 1}")))
                if collection.get("out_path"):
                    self.collection_out_edit.setText(str(collection["out_path"]))
                for key, widget in [
                    ("fs", self.fs_spin),
                    ("trials_per_class", self.trials_spin),
                    ("epoch_seconds", self.epoch_spin),
                    ("pre_rest_seconds", self.pre_rest_spin),
                    ("cue_seconds", self.cue_spin),
                    ("cue_rest_seconds", self.cue_rest_spin),
                    ("post_rest_seconds", self.post_rest_spin),
                    ("port", self.rda_port_spin),
                    ("expected_channels", self.rda_channels_spin),
                    ("aux_channels", self.aux_channels_spin),
                ]:
                    if key in collection:
                        self._spin_set_value(widget, collection[key])
                if collection.get("host"):
                    self.rda_host_edit.setText(str(collection["host"]))
                if "use_parallel_trigger" in collection:
                    self.parallel_check.setChecked(bool(collection["use_parallel_trigger"]))
                if collection.get("parallel_port") is not None:
                    self.trigger_port_edit.setText(str(collection["parallel_port"]))

                paths = dict(data.get("paths", {}) or {})
                self.train_data_edit.setText(str(paths.get("train_data_path", self.train_data_edit.text())))
                self.test_data_edit.setText(str(paths.get("test_data_path", self.test_data_edit.text())))
                self.model_edit.setText(str(paths.get("model_path", self.model_edit.text())))
                self.vr_model_edit.setText(str(paths.get("vr_model_path", self.vr_model_edit.text())))

                model = dict(data.get("model", {}) or {})
                self._combo_set_text(self.classifier_combo, model.get("classifier", self.classifier_combo.currentText()))
                self._combo_set_text(self.validation_strategy_combo, model.get("validation_strategy", self._combo_value(self.validation_strategy_combo)))
                self._combo_set_text(self.feature_strategy_combo, model.get("feature_strategy", self._combo_value(self.feature_strategy_combo)))
                if "use_csp" in model:
                    self.csp_check.setChecked(bool(model["use_csp"]))
                    self._combo_set_text(self.feature_strategy_combo, "fbcsp" if bool(model["use_csp"]) else "bandpower")
                realtime, migration_notes = migrate_legacy_eeg8_realtime_config(
                    data.get("realtime", {}) or {}
                )
                self.defaults_migration_notes = migration_notes
                self._combo_set_text(self.online_mode_combo, realtime.get("mode", self.online_mode_combo.currentText()))
                self._combo_set_text(self.runtime_strategy_combo, realtime.get("runtime_strategy", self._combo_value(self.runtime_strategy_combo)))
                if "use_current_model" in realtime:
                    self.use_model_check.setChecked(bool(realtime["use_current_model"]))
                if "use_config_thresholds" in realtime:
                    self.use_weight_check.setChecked(bool(realtime["use_config_thresholds"]))
                if realtime.get("target"):
                    self.target_edit.setText(str(realtime["target"]))
                if "robot_enabled" in realtime:
                    self.robot_enabled_check.setChecked(bool(realtime["robot_enabled"]))
                if realtime.get("robot_host"):
                    self.robot_host_edit.setText(str(realtime["robot_host"]))
                if "robot_port" in realtime:
                    self._spin_set_value(self.robot_port_spin, realtime["robot_port"])
                if "robot_speed" in realtime:
                    self._spin_set_value(self.robot_speed_spin, realtime["robot_speed"])
                for key, widget in [
                    ("decode_window_seconds", self.decode_window_spin),
                    ("confirm_window_seconds", self.confirm_window_spin),
                    ("confirmation_masseter_count", self.confirm_masseter_count_spin),
                    ("continuous_move_speed", self.continuous_move_speed_spin),
                    ("continuous_blink_window_seconds", self.continuous_blink_window_spin),
                    ("continuous_blink_count", self.continuous_blink_count_spin),
                    ("aux_blink_ratio", self.aux_blink_ratio_spin),
                    ("aux_bite_ratio", self.aux_bite_ratio_spin),
                    ("aux_side_ratio", self.aux_side_ratio_spin),
                    ("aux_blink_count", self.aux_blink_count_spin),
                    ("aux_left_count", self.aux_left_count_spin),
                    ("aux_right_count", self.aux_right_count_spin),
                    ("masseter_ncc_min_score", self.masseter_ncc_score_spin),
                    ("masseter_ncc_min_margin", self.masseter_ncc_margin_spin),
                    ("masseter_outside_candidate_weight", self.masseter_outside_weight_spin),
                    ("candidate_random_margin", self.candidate_random_margin_spin),
                ]:
                    if key in realtime:
                        self._spin_set_value(widget, realtime[key])
                if realtime.get("masseter_template_path"):
                    self.masseter_template_edit.setText(str(realtime["masseter_template_path"]))
                decoder = realtime.get("decoder")
                if isinstance(decoder, DecoderConfig):
                    self.conf_spin.setValue(float(getattr(decoder, "command_confidence", self.conf_spin.value())))
                    self.margin_spin.setValue(float(getattr(decoder, "min_margin", self.margin_spin.value())))
                    self.refractory_spin.setValue(float(decoder.refractory_seconds))
                    self.command_duration_spin.setValue(float(decoder.command_duration_seconds))
                    self.confirm_window_spin.setValue(float(getattr(decoder, "confirmation_window_seconds", self.confirm_window_spin.value())))
                    self.confirm_masseter_count_spin.setValue(int(getattr(decoder, "confirmation_masseter_count", self.confirm_masseter_count_spin.value())))
                    self.continuous_move_speed_spin.setValue(float(getattr(decoder, "continuous_move_speed", self.continuous_move_speed_spin.value())))
                    self.continuous_blink_window_spin.setValue(float(getattr(decoder, "continuous_blink_window_seconds", self.continuous_blink_window_spin.value())))
                    self.continuous_blink_count_spin.setValue(int(getattr(decoder, "continuous_blink_count", self.continuous_blink_count_spin.value())))
                    if getattr(decoder, "masseter_template_path", ""):
                        self.masseter_template_edit.setText(str(decoder.masseter_template_path))
                    self._combo_set_text(self.masseter_detector_combo, getattr(decoder, "masseter_detector", "ncc"))
                    for state, mode in dict(getattr(decoder, "masseter_state_detectors", {}) or {}).items():
                        if state in self.masseter_state_detector_combos:
                            self._combo_set_text(self.masseter_state_detector_combos[state], mode)
                    self.masseter_ncc_score_spin.setValue(float(getattr(decoder, "masseter_ncc_min_score", self.masseter_ncc_score_spin.value())))
                    self.masseter_ncc_margin_spin.setValue(float(getattr(decoder, "masseter_ncc_min_margin", self.masseter_ncc_margin_spin.value())))
                    self.candidate_random_margin_spin.setValue(float(getattr(decoder, "candidate_random_margin", self.candidate_random_margin_spin.value())))
                elif isinstance(decoder, dict):
                    if "command_confidence" in decoder:
                        self.conf_spin.setValue(float(decoder["command_confidence"]))
                    if "min_margin" in decoder:
                        self.margin_spin.setValue(float(decoder["min_margin"]))
                    if "refractory_seconds" in decoder:
                        self.refractory_spin.setValue(float(decoder["refractory_seconds"]))
                    if "command_duration_seconds" in decoder:
                        self.command_duration_spin.setValue(float(decoder["command_duration_seconds"]))
                    if "confirmation_window_seconds" in decoder:
                        self.confirm_window_spin.setValue(float(decoder["confirmation_window_seconds"]))
                    if "confirmation_masseter_count" in decoder:
                        self.confirm_masseter_count_spin.setValue(int(decoder["confirmation_masseter_count"]))
                    if "continuous_move_speed" in decoder:
                        self.continuous_move_speed_spin.setValue(float(decoder["continuous_move_speed"]))
                    if "continuous_blink_window_seconds" in decoder:
                        self.continuous_blink_window_spin.setValue(float(decoder["continuous_blink_window_seconds"]))
                    if "continuous_blink_count" in decoder:
                        self.continuous_blink_count_spin.setValue(int(decoder["continuous_blink_count"]))
                    if decoder.get("masseter_template_path"):
                        self.masseter_template_edit.setText(str(decoder["masseter_template_path"]))
                    if decoder.get("masseter_detector"):
                        self._combo_set_text(self.masseter_detector_combo, decoder["masseter_detector"])
                    for state, mode in dict(decoder.get("masseter_state_detectors", {}) or {}).items():
                        if state in self.masseter_state_detector_combos:
                            self._combo_set_text(self.masseter_state_detector_combos[state], mode)
                    if "masseter_ncc_min_score" in decoder:
                        self.masseter_ncc_score_spin.setValue(float(decoder["masseter_ncc_min_score"]))
                    if "masseter_ncc_min_margin" in decoder:
                        self.masseter_ncc_margin_spin.setValue(float(decoder["masseter_ncc_min_margin"]))
                    if "masseter_outside_candidate_weight" in decoder:
                        self.masseter_outside_weight_spin.setValue(float(decoder["masseter_outside_candidate_weight"]))
                    if "candidate_random_margin" in decoder:
                        self.candidate_random_margin_spin.setValue(float(decoder["candidate_random_margin"]))
            finally:
                self.blockSignals(old_state)
            # Window1 uses non-overlapping segments; its step always equals its duration.
            self.decode_step_spin.setValue(self.decode_window_spin.value())
            self.update_mode_fields()
            self.config_changed.emit()

        def load_defaults(self) -> None:
            path = self.DEFAULTS_PATH
            if not os.path.exists(path):
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.apply_defaults(data)
                if self.defaults_migration_notes:
                    self.defaults_status_label.setText(
                        f"默认配置：已加载并迁移 {path}。{self.defaults_migration_notes[0]}"
                    )
                else:
                    self.defaults_status_label.setText(f"默认配置：已加载 {path}")
            except Exception as exc:
                self.defaults_status_label.setText(f"默认配置加载失败：{exc}")

        def save_defaults(self) -> None:
            path = self.DEFAULTS_PATH
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                data = self.export_defaults()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                self.defaults_status_label.setText(f"默认配置：已保存 {path}")
                QtWidgets.QMessageBox.information(self, "已保存", f"当前选择已保存为默认配置：\n{path}")
            except Exception as exc:
                self.defaults_status_label.setText(f"默认配置保存失败：{exc}")
                QtWidgets.QMessageBox.critical(self, "保存失败", str(exc))

        def update_mode_fields(self) -> None:
            is_collection_rda = self.collection_mode_combo.currentText().startswith("64导")
            is_online_rda = self.online_mode_combo.currentText().startswith("64导")
            is_any_rda = is_collection_rda or is_online_rda
            for widget in [self.rda_host_edit, self.rda_port_spin, self.rda_channels_spin, self.aux_channels_spin]:
                widget.setEnabled(is_any_rda)
            for widget in [self.parallel_check, self.trigger_port_edit]:
                widget.setEnabled(is_collection_rda)
            self.fs_spin.setEnabled(not is_collection_rda)
            self.csp_check.setChecked(self._combo_value(self.feature_strategy_combo) == "fbcsp")
            detector_mode = self._combo_value(self.masseter_detector_combo)
            use_ncc = detector_mode == "ncc" or (
                detector_mode == "precise" and any(
                    self._combo_value(combo) == "ncc"
                    for combo in self.masseter_state_detector_combos.values()
                )
            )
            for widget in [self.masseter_template_edit, self.masseter_ncc_score_spin, self.masseter_ncc_margin_spin]:
                widget.setEnabled(use_ncc)
            for combo in self.masseter_state_detector_combos.values():
                combo.setEnabled(detector_mode == "precise")
            custom_categories = self._combo_value(self.paradigm_combo) == "custom_mi"
            self.class_count_spin.setEnabled(custom_categories)
            self.category_table.setEnabled(custom_categories)
            if custom_categories:
                vr_index = self.runtime_strategy_combo.findData("vr_only")
                if vr_index >= 0 and self.runtime_strategy_combo.currentIndex() != vr_index:
                    self.runtime_strategy_combo.setCurrentIndex(vr_index)
                self.robot_enabled_check.setChecked(False)
                self.robot_enabled_check.setEnabled(False)
            else:
                self.robot_enabled_check.setEnabled(True)

        def pick_collection_out(self) -> None:
            start = self.collection_out_edit.text().strip() or RECORDINGS_DIR
            if os.path.splitext(start)[1].lower() == ".npz":
                start = os.path.dirname(start)
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择采集归档目录", start)
            if path:
                self.collection_out_edit.setText(path)

        def pick_train_data(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择训练数据", APP_DIR, "NPZ Files (*.npz)")
            if path:
                self.train_data_edit.setText(path)

        def pick_test_data(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择测试数据", APP_DIR, "NPZ Files (*.npz)")
            if path:
                self.test_data_edit.setText(path)

        def pick_model_out(self) -> None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "选择模型/权重", self.model_edit.text(), "Joblib Files (*.joblib)")
            if path:
                self.set_model_path(path)

        def pick_vr_model(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 VR 模型/权重", APP_DIR, "Joblib Files (*.joblib)")
            if path:
                self.vr_model_edit.setText(path)

        def pick_masseter_template(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 NCC 肌电模板", APP_DIR, "Template Files (*.json *.npz);;All Files (*)")
            if path:
                self.masseter_template_edit.setText(path)

        def sync_data_paths(self) -> None:
            path = self.collection_out_edit.text().strip()
            self.set_training_data_path(path)

        def set_training_data_path(self, path: str) -> None:
            if path:
                self.train_data_edit.setText(path)
                self.test_data_edit.setText(path)

        def set_model_path(self, path: str) -> None:
            if path:
                self.model_edit.setText(path)
                self.vr_model_edit.setText(path)

        def parse_parallel_port(self) -> int:
            text = self.trigger_port_edit.text().strip()
            try:
                return int(text, 0)
            except ValueError as exc:
                raise ValueError(f"并口地址格式错误：{text}，可填写 0x0378 或十进制地址") from exc

        def collection_config(self) -> Dict:
            return {
                "paradigm_key": self._combo_value(self.paradigm_combo),
                "class_definitions": self.category_definitions(),
                "mode": self.collection_mode_combo.currentText(),
                "out_path": self.collection_out_edit.text().strip(),
                "fs": int(self.fs_spin.value()),
                "trials_per_class": int(self.trials_spin.value()),
                "epoch_seconds": float(self.epoch_spin.value()),
                "pre_rest_seconds": float(self.pre_rest_spin.value()),
                "cue_seconds": float(self.cue_spin.value()),
                "cue_rest_seconds": float(self.cue_rest_spin.value()),
                "post_rest_seconds": float(self.post_rest_spin.value()),
                "host": self.rda_host_edit.text().strip() or "127.0.0.1",
                "port": int(self.rda_port_spin.value()),
                "expected_channels": int(self.rda_channels_spin.value()),
                "aux_channels": int(self.aux_channels_spin.value()),
                "use_parallel_trigger": self.parallel_check.isChecked(),
                "parallel_port": self.parse_parallel_port() if self.parallel_check.isChecked() else 0x0378,
            }

        def decoder_config(self) -> DecoderConfig:
            return DecoderConfig(
                eeg_weight=0.55,
                # Kept at zero only for compatibility with older model/config files;
                # online classification no longer uses either value as a gate.
                command_confidence=0.0,
                min_margin=0.0,
                stable_windows=1,
                history_windows=1,
                refractory_seconds=float(self.refractory_spin.value()),
                command_duration_seconds=float(self.command_duration_spin.value()),
                masseter_gate_enabled=True,
                masseter_left_count=int(self.aux_left_count_spin.value()),
                masseter_right_count=int(self.aux_right_count_spin.value()),
                masseter_bite_ratio=float(self.aux_bite_ratio_spin.value()),
                masseter_side_ratio=float(self.aux_side_ratio_spin.value()),
                masseter_detector=self._combo_value(self.masseter_detector_combo),
                masseter_state_detectors={
                    state: self._combo_value(combo)
                    for state, combo in self.masseter_state_detector_combos.items()
                },
                masseter_template_path=self.masseter_template_edit.text().strip(),
                masseter_ncc_min_score=float(self.masseter_ncc_score_spin.value()),
                masseter_ncc_min_margin=float(self.masseter_ncc_margin_spin.value()),
                confirmation_window_seconds=float(self.confirm_window_spin.value()),
                confirmation_masseter_count=int(self.confirm_masseter_count_spin.value()),
                continuous_move_speed=float(self.continuous_move_speed_spin.value()),
                continuous_blink_window_seconds=float(self.continuous_blink_window_spin.value()),
                continuous_blink_count=int(self.continuous_blink_count_spin.value()),
                masseter_outside_candidate_weight=float(self.masseter_outside_weight_spin.value()),
                candidate_random_margin=float(self.candidate_random_margin_spin.value()),
            )

        def train_config(self) -> Dict:
            feature_strategy = self._combo_value(self.feature_strategy_combo)
            validation_strategy = self._combo_value(self.validation_strategy_combo)
            return {
                "paradigm_key": self._combo_value(self.paradigm_combo),
                "data_path": self.train_data_edit.text().strip(),
                "test_data_path": self.test_data_edit.text().strip(),
                "model_path": self.model_edit.text().strip(),
                "label_contract": eeg8_action_label_info(
                    [item["class_name"] for item in self.category_definitions()]
                    if self._combo_value(self.paradigm_combo) == "custom_mi" else None,
                    self._combo_value(self.paradigm_combo) != "custom_mi",
                ),
                "feature_strategy": feature_strategy,
                "validation_strategy": validation_strategy,
                "classifier": self._combo_value(self.classifier_combo),
                "classifier_label": self.classifier_combo.currentText(),
                "use_csp": feature_strategy == "fbcsp",
                # Persist the window contract used by online decoding.
                "decode_window_seconds": float(self.decode_window_spin.value()),
                "decode_step_seconds": float(self.decode_window_spin.value()),
                "decoder": self.decoder_config(),
            }

        def realtime_config(self) -> Dict:
            selected_strategy = self._combo_value(self.runtime_strategy_combo)
            runtime_strategy = EEG8_RUNTIME_STRATEGY_EEG_ONLY if selected_strategy == "vr_only" else selected_strategy
            decoder = self.decoder_config()
            custom_categories = self._combo_value(self.paradigm_combo) == "custom_mi"
            if runtime_strategy == "eeg8_two_phase_eeg_only":
                decoder.masseter_gate_enabled = False
            return {
                "paradigm_key": self._combo_value(self.paradigm_combo),
                "class_definitions": self.category_definitions(),
                "model_path": self.vr_model_edit.text().strip(),
                "mode": self.online_mode_combo.currentText(),
                "runtime_strategy": runtime_strategy,
                "vr_only": selected_strategy == "vr_only" or self._combo_value(self.paradigm_combo) == "custom_mi",
                "use_current_model": self.use_model_check.isChecked(),
                "use_config_thresholds": self.use_weight_check.isChecked(),
                "host": self.rda_host_edit.text().strip() or "127.0.0.1",
                "port": int(self.rda_port_spin.value()),
                "expected_channels": int(self.rda_channels_spin.value()),
                "aux_channels": int(self.aux_channels_spin.value()),
                "target": self.target_edit.text().strip() or "target_1",
                "robot_enabled": self.robot_enabled_check.isChecked() and not custom_categories and selected_strategy != "vr_only",
                "robot_host": self.robot_host_edit.text().strip() or "192.168.1.100",
                "robot_port": int(self.robot_port_spin.value()),
                "robot_speed": float(self.robot_speed_spin.value()),
                "decode_window_seconds": float(self.decode_window_spin.value()),
                "decode_step_seconds": float(self.decode_window_spin.value()),
                "confirm_window_seconds": float(self.confirm_window_spin.value()),
                "confirmation_masseter_count": int(self.confirm_masseter_count_spin.value()),
                "continuous_move_speed": float(self.continuous_move_speed_spin.value()),
                "continuous_blink_window_seconds": float(self.continuous_blink_window_spin.value()),
                "continuous_blink_count": int(self.continuous_blink_count_spin.value()),
                "decoder": decoder,
                "masseter_template_path": self.masseter_template_edit.text().strip(),
                "masseter_ncc_min_score": float(self.masseter_ncc_score_spin.value()),
                "masseter_ncc_min_margin": float(self.masseter_ncc_margin_spin.value()),
                "candidate_random_margin": float(self.candidate_random_margin_spin.value()),
                "aux_blink_ratio": float(self.aux_blink_ratio_spin.value()),
                "aux_bite_ratio": float(self.aux_bite_ratio_spin.value()),
                "aux_side_ratio": float(self.aux_side_ratio_spin.value()),
                "aux_blink_count": int(self.aux_blink_count_spin.value()),
                "aux_left_count": int(self.aux_left_count_spin.value()),
                "aux_right_count": int(self.aux_right_count_spin.value()),
            }

    class ModelTrainingWorker(QtCore.QThread):
        progress_signal = QtCore.pyqtSignal(int, str)
        log_signal = QtCore.pyqtSignal(str)
        result_signal = QtCore.pyqtSignal(dict)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(
            self,
            data_paths: List[str],
            test_paths: List[str],
            model_path: str,
            cfg: Dict,
            ui_fs: int,
            parent=None,
        ):
            super().__init__(parent)
            self.data_paths = list(data_paths)
            self.test_paths = list(test_paths)
            self.model_path = str(model_path)
            self.cfg = dict(cfg)
            self.ui_fs = int(ui_fs)

        @staticmethod
        def _metrics_for_report(metrics: Dict) -> Dict:
            return {k: v for k, v in dict(metrics).items() if not str(k).startswith("_")}

        def _emit_progress(self, percent: int, stage: str) -> None:
            self.progress_signal.emit(max(0, min(100, int(percent))), stage)

        def run(self) -> None:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.model_path)) or ".", exist_ok=True)
                self._emit_progress(5, "读取训练记录")
                eeg, y_raw, aux, file_fs, channel_names, data_contract = load_eeg8_recordings(
                    self.data_paths,
                    log=self.log_signal.emit,
                    return_metadata=True,
                )
                validate_eeg8_training_data_contract(data_contract)
                decode_window = max(0.5, float(self.cfg.get("decode_window_seconds", 2.0) or 2.0))
                decode_step = max(0.1, float(self.cfg.get("decode_step_seconds", decode_window) or decode_window))
                signal_cfg = SignalConfig(
                    fs=file_fs or self.ui_fs,
                    window_seconds=decode_window,
                    step_seconds=decode_step,
                    aux_channel_names=list(data_contract.get("aux_channel_names", [])),
                    masseter_gate_capable=bool(data_contract.get("masseter_gate_capable", False)),
                )
                signal_cfg.eeg_channels = channel_names
                signal_cfg.use_csp = bool(self.cfg["use_csp"])
                decoder_cfg = self.cfg["decoder"]
                self.log_signal.emit(
                    f"训练窗口契约：{signal_cfg.window_seconds:g}s 窗口 / {signal_cfg.step_seconds:g}s 间隔；"
                    f"{data_contract.get('masseter_gate_status', '')}"
                )
                # Automatic model selection always needs the same held-out
                # validation evidence for every candidate.
                run_cv = True
                classifier_candidates = [(
                    str(self.cfg.get("classifier", "best_mi_fusion")),
                    str(self.cfg.get("classifier_label", "EEGNet + Shallow MI 概率融合")),
                )]
                comparison = []
                winner = None
                winner_score = None
                for candidate_index, (classifier_key, classifier_label) in enumerate(
                    classifier_candidates
                ):
                    if self.isInterruptionRequested():
                        return
                    start_progress = 15 + int(68 * candidate_index / len(classifier_candidates))
                    span = max(1, int(68 / len(classifier_candidates)))
                    self._emit_progress(
                        start_progress,
                        f"模型竞赛 {candidate_index + 1}/{len(classifier_candidates)}：{classifier_label}",
                    )

                    def candidate_progress(percent, stage, start=start_progress, width=span, label=classifier_label):
                        scaled = start + int(width * max(0, min(100, int(percent))) / 100)
                        self._emit_progress(scaled, f"{label}：{stage}")

                    candidate_signal_cfg = copy.deepcopy(signal_cfg)
                    if classifier_key == "fbcsp_lda":
                        candidate_signal_cfg.use_csp = True
                    candidate_model, candidate_metrics = train_eeg8_model(
                        eeg,
                        y_raw,
                        aux,
                        candidate_signal_cfg,
                        decoder_cfg,
                        classifier_key,
                        paradigm_key=self.cfg.get("paradigm_key", "eeg8"),
                        run_cv=run_cv,
                        progress_callback=candidate_progress,
                        data_contract=data_contract,
                    )
                    accuracy = float(
                        candidate_metrics.get(
                            "eeg_accuracy",
                            candidate_metrics.get("overall_accuracy", 0.0),
                        )
                        or 0.0
                    )
                    kappa = float(
                        candidate_metrics.get(
                            "eeg_kappa", candidate_metrics.get("kappa", 0.0)
                        )
                        or 0.0
                    )
                    class_accuracy = [
                        float(value)
                        for value in candidate_metrics.get("class_accuracy", [])
                    ]
                    minimum_class_accuracy = min(class_accuracy or [0.0])
                    row = {
                        "classifier": classifier_key,
                        "classifier_label": classifier_label,
                        "feature": (
                            "fbcsp"
                            if bool(candidate_model.signal_config.use_csp)
                            else "bandpower"
                        ),
                        "accuracy": accuracy,
                        "kappa": kappa,
                        "minimum_class_accuracy": minimum_class_accuracy,
                        "class_accuracy": class_accuracy,
                        "fold_accuracy": list(candidate_metrics.get("fold_accuracy", [])),
                        "confusion_matrix": candidate_metrics.get("confusion_matrix", []),
                    }
                    comparison.append(row)
                    self.log_signal.emit(
                        f"{classifier_label}：准确率 {accuracy:.1%}，"
                        f"Kappa {kappa:.3f}，最低类别准确率 {minimum_class_accuracy:.1%}"
                    )
                    score = (accuracy, kappa, minimum_class_accuracy)
                    if winner_score is None or score > winner_score:
                        winner_score = score
                        winner = (
                            candidate_model,
                            candidate_metrics,
                            classifier_key,
                            classifier_label,
                        )

                if winner is None:
                    raise RuntimeError("所有候选模型训练均未产生有效结果")
                model, metrics, winner_key, winner_label = winner
                comparison.sort(
                    key=lambda row: (
                        row["accuracy"],
                        row["kappa"],
                        row["minimum_class_accuracy"],
                    ),
                    reverse=True,
                )
                metrics["model_comparison"] = comparison
                metrics["selected_classifier"] = winner_key
                metrics["selected_classifier_label"] = winner_label
                model.training_strategy_["model_comparison"] = comparison
                model.training_strategy_["selected_classifier"] = winner_key
                feature_name = "FilterBank+CSP" if model.signal_config.use_csp else "FilterBank BandPower"
                algorithm = f"{feature_name} + {winner_label}（8模型自动优选）"

                self._emit_progress(90, "保存模型和训练报告")
                metrics["data_contract"] = data_contract
                metrics["masseter_gate_capable"] = bool(data_contract.get("masseter_gate_capable", False))
                if not data_contract.get("masseter_gate_capable", False):
                    metrics["aux_warning"] = str(data_contract.get("masseter_gate_status", "不具备咬肌门控条件"))
                model.training_strategy_["data_contract"] = data_contract
                model.metrics_ = self._metrics_for_report(metrics)
                model.save(self.model_path)
                report_path = os.path.splitext(self.model_path)[0] + "_report.json"
                report_payload = {
                    "model": self.model_path,
                    "data": self.data_paths,
                    "algorithm": algorithm,
                    "strategies": {
                        "label_contract": self.cfg.get("label_contract", {}),
                        "feature": self.cfg.get("feature_strategy"),
                        "validation": self.cfg.get("validation_strategy"),
                        "classifier": winner_key,
                        "classifier_selection": "automatic_best_of_all_gui_models",
                        "model_comparison": comparison,
                        "runtime_compatible": bool(metrics.get("runtime_compatible")),
                    },
                    "signal_config": asdict(signal_cfg),
                    "decoder_config": asdict(decoder_cfg),
                    "metrics": self._metrics_for_report(metrics),
                    "class_names": CLASS_NAMES,
                }
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report_payload, f, ensure_ascii=False, indent=2)

                result = {
                    "model_path": self.model_path,
                    "report_path": report_path,
                    "train_data_path": self.data_paths[0] if self.data_paths else "",
                    "data_paths": self.data_paths,
                    "trial_count": len(y_raw),
                    "metrics": metrics,
                    "plot_metrics": metrics,
                    "plot_title": "训练验证准确率",
                    "algorithm": algorithm,
                    "signal_config": signal_cfg,
                    "image_prefix": os.path.splitext(self.model_path)[0],
                }

                if self.cfg.get("validation_strategy") == "train_then_test":
                    if self.test_paths:
                        self._emit_progress(94, "加载独立测试记录")
                        eeg_t, y_t, _aux_t, _fs_t, _names_t = load_eeg8_recordings(
                            self.test_paths,
                            log=self.log_signal.emit,
                        )
                        self._emit_progress(97, "评估独立测试集")
                        test_metrics = eval_eeg8_model(
                            model,
                            eeg_t,
                            y_t,
                            model.signal_config,
                        )
                        test_report_path = os.path.splitext(self.model_path)[0] + "_train_then_test_eval.json"
                        with open(test_report_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "model": self.model_path,
                                    "data": self.test_paths,
                                    "strategies": {
                                        "label_contract": self.cfg.get("label_contract", {}),
                                        "feature": self.cfg.get("feature_strategy"),
                                        "validation": self.cfg.get("validation_strategy"),
                                        "classifier": winner_key,
                                    },
                                    "metrics": self._metrics_for_report(test_metrics),
                                },
                                f,
                                ensure_ascii=False,
                                indent=2,
                            )
                        result.update({
                            "test_paths": self.test_paths,
                            "test_trial_count": len(y_t),
                            "test_metrics": test_metrics,
                            "test_report_path": test_report_path,
                            "plot_metrics": test_metrics,
                            "plot_title": "训练后独立测试准确率",
                        })
                    else:
                        result["test_warning"] = "验证策略为训练后独立测试，但未勾选测试记录；已跳过自动测试。"

                self._emit_progress(100, "训练完成")
                self.result_signal.emit(result)
            except Exception as exc:
                self.error_signal.emit(f"{exc}\n{traceback.format_exc()}")

    class RecordingScanWorker(QtCore.QThread):
        progress_signal = QtCore.pyqtSignal(int, str)
        result_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def run(self) -> None:
            try:
                paths = recording_npz_files(RECORDINGS_DIR)
                total = max(1, len(paths))
                records = []
                warnings = []
                for index, path in enumerate(paths, start=1):
                    try:
                        records.append(inspect_recording_npz(path))
                    except Exception as exc:
                        warnings.append(f"跳过无效记录：{path} | {exc}")
                    self.progress_signal.emit(int(index * 100 / total), f"扫描采集记录 {index}/{len(paths)}")
                self.result_signal.emit({"records": records, "warnings": warnings})
            except Exception as exc:
                self.error_signal.emit(str(exc))

    class ModelEvaluationWorker(QtCore.QThread):
        progress_signal = QtCore.pyqtSignal(int, str)
        result_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(self, model_path: str, data_paths: List[str], cfg: Dict, parent=None):
            super().__init__(parent)
            self.model_path = model_path
            self.data_paths = list(data_paths)
            self.cfg = cfg

        def run(self) -> None:
            try:
                self.progress_signal.emit(10, "读取模型")
                model = EEG8Model.load(self.model_path)
                self.progress_signal.emit(35, "读取独立测试记录")
                eeg, y_raw, aux, _file_fs, _channel_names = load_eeg8_recordings(self.data_paths)
                self.progress_signal.emit(65, "计算离线评估指标")
                metrics = eval_eeg8_model(model, eeg, y_raw, model.signal_config)
                report_path = os.path.splitext(self.model_path)[0] + "_offline_eval.json"
                self.progress_signal.emit(90, "保存评估报告")
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": self.model_path,
                            "data": self.data_paths,
                            "strategies": {
                                "label_contract": self.cfg.get("label_contract", {}),
                                "feature": self.cfg.get("feature_strategy"),
                                "validation": "offline_eval",
                                "classifier": self.cfg.get("classifier"),
                            },
                            "metrics": {key: value for key, value in dict(metrics).items() if not str(key).startswith("_")},
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.progress_signal.emit(100, "离线评估完成")
                self.result_signal.emit({
                    "metrics": metrics,
                    "report_path": report_path,
                    "trial_count": len(y_raw),
                    "data_paths": self.data_paths,
                })
            except Exception as exc:
                self.error_signal.emit(f"{exc}\n{traceback.format_exc()}")

    class TrainTunePage(QtWidgets.QWidget):
        model_ready = QtCore.pyqtSignal(str)

        def __init__(self, config: ConfigPage, parent=None):
            super().__init__(parent)
            self.config = config
            self.last_metrics: Dict = {}
            self.training_worker: Optional[QtCore.QThread] = None
            self.scan_worker: Optional[QtCore.QThread] = None
            self.evaluation_worker: Optional[QtCore.QThread] = None
            root_layout = QtWidgets.QVBoxLayout(self)
            root_layout.setContentsMargins(0, 0, 0, 0)
            root_layout.setSpacing(0)
            page_scroll = QtWidgets.QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            page_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            page_content = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page_content)
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(12)
            page_scroll.setWidget(page_content)
            root_layout.addWidget(page_scroll)
            title = QtWidgets.QLabel("模型训练和测试")
            title.setObjectName("title")
            layout.addWidget(title)
            desc = QtWidgets.QLabel("扫描按时间归档的采集记录。逐条勾选训练集或独立测试集；表格明确显示每个文件实际包含的 32 路 EEG、AUX/肌电和 Trial 数。")
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            layout.addWidget(desc)

            card = Card()
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setSpacing(8)
            toolbar = QtWidgets.QHBoxLayout()
            self.recording_root_label = QtWidgets.QLabel(f"真实采集目录：{RECORDINGS_DIR}")
            self.recording_root_label.setObjectName("hint")
            self.recording_root_label.setWordWrap(True)
            self.refresh_button = QtWidgets.QPushButton("刷新记录")
            self.refresh_button.clicked.connect(self.refresh_recordings)
            toolbar.addWidget(self.recording_root_label, 1)
            toolbar.addWidget(self.refresh_button)
            card_layout.addLayout(toolbar)

            self.recording_table = QtWidgets.QTableWidget(0, 8)
            self.recording_table.setHorizontalHeaderLabels(
                ["训练", "测试", "采集时间", "Trial", "EEG", "AUX", "采样率", "文件"]
            )
            self.recording_table.setAlternatingRowColors(True)
            self.recording_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.recording_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.recording_table.verticalHeader().setVisible(False)
            header = self.recording_table.horizontalHeader()
            for col in range(7):
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeMode.Stretch)
            self.recording_table.setMinimumHeight(150)
            self.recording_table.setMaximumHeight(240)
            card_layout.addWidget(self.recording_table)

            options = QtWidgets.QHBoxLayout()
            self.test_model_check = QtWidgets.QCheckBox("离线测试使用当前模型/权重")
            self.test_model_check.setChecked(True)
            options.addWidget(self.test_model_check)
            self.strategy_label = QtWidgets.QLabel("")
            self.strategy_label.setObjectName("hint")
            self.strategy_label.setWordWrap(True)
            options.addWidget(self.strategy_label, 1)
            options.addStretch(1)
            card_layout.addLayout(options)
            self.path_label = QtWidgets.QLabel("")
            self.path_label.setWordWrap(True)
            self.path_label.setObjectName("hint")
            self.path_label.setMinimumHeight(34)
            card_layout.addWidget(self.path_label)
            controls = QtWidgets.QHBoxLayout()
            self.train_button = QtWidgets.QPushButton("开始训练")
            self.train_button.clicked.connect(self.train_model)
            self.eval_button = QtWidgets.QPushButton("离线测试并绘图")
            self.eval_button.clicked.connect(self.evaluate_model)
            controls.addWidget(self.train_button)
            controls.addWidget(self.eval_button)
            card_layout.addLayout(controls)
            self.train_progress = QtWidgets.QProgressBar()
            self.train_progress.setRange(0, 100)
            self.train_progress.setValue(0)
            self.train_progress.setFormat("训练进度：%p%")
            card_layout.addWidget(self.train_progress)
            layout.addWidget(card)

            self.summary_label = QtWidgets.QLabel("准确率：-")
            self.summary_label.setObjectName("phaseBadge")
            self.summary_label.setWordWrap(True)
            layout.addWidget(self.summary_label)
            self.kappa_label = QtWidgets.QLabel("Kappa 系数：-")
            self.kappa_label.setObjectName("phaseBadge")
            self.kappa_label.setWordWrap(True)
            layout.addWidget(self.kappa_label)
            self.accuracy_pixmap = QtGui.QPixmap()
            self.kappa_pixmap = QtGui.QPixmap()
            self.accuracy_image_label = QtWidgets.QLabel("训练完成后显示准确率报告图")
            self.accuracy_image_label.setObjectName("metricImage")
            self.accuracy_image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.accuracy_image_label.setScaledContents(False)
            self.accuracy_image_label.setMinimumHeight(300)
            self.accuracy_image_label.setMaximumHeight(420)
            layout.addWidget(self.accuracy_image_label, 1)
            self.kappa_image_label = QtWidgets.QLabel("训练完成后显示 Kappa 报告图")
            self.kappa_image_label.setObjectName("metricImage")
            self.kappa_image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.kappa_image_label.setScaledContents(False)
            self.kappa_image_label.setMinimumHeight(170)
            self.kappa_image_label.setMaximumHeight(260)
            layout.addWidget(self.kappa_image_label, 1)

            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMinimumHeight(130)
            self.log.setMaximumHeight(220)
            layout.addWidget(self.log, 1)
            self.config.config_changed.connect(self.refresh_paths)
            self.refresh_recordings()

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            QtCore.QTimer.singleShot(0, self._rescale_metric_images)

        def _rescale_metric_images(self) -> None:
            if hasattr(self, "accuracy_pixmap") and not self.accuracy_pixmap.isNull():
                self._set_metric_image(self.accuracy_image_label, self.accuracy_pixmap)
            if hasattr(self, "kappa_pixmap") and not self.kappa_pixmap.isNull():
                self._set_metric_image(self.kappa_image_label, self.kappa_pixmap)

        def refresh_paths(self) -> None:
            cfg = self.config.train_config()
            train_count = len(self.selected_recordings("train"))
            test_count = len(self.selected_recordings("test"))
            self.path_label.setText(
                f"已选训练记录 {train_count} 个；测试记录 {test_count} 个；模型/权重：{cfg['model_path']}"
            )
            label_contract = cfg.get("label_contract", {})
            self.strategy_label.setText(
                f"标签：{label_contract.get('name', 'EEG8 C1-C8 动作分类')} | "
                f"{cfg.get('classifier_label', cfg.get('classifier'))} | "
                f"{cfg.get('feature_strategy')} | {cfg.get('validation_strategy')} | 可用于实时运行"
            )

        def selected_recordings(self, role: str) -> List[str]:
            column = 0 if role == "train" else 1
            paths: List[str] = []
            for row in range(self.recording_table.rowCount()):
                holder = self.recording_table.cellWidget(row, column)
                check = holder.findChild(QtWidgets.QCheckBox) if holder is not None else None
                path_item = self.recording_table.item(row, 7)
                if check is not None and check.isChecked() and path_item is not None:
                    paths.append(path_item.data(QtCore.Qt.ItemDataRole.UserRole) or path_item.text())
            return paths

        def _dataset_role_changed(self, path: str, role: str, checked: bool) -> None:
            if checked:
                opposite = 1 if role == "train" else 0
                for row in range(self.recording_table.rowCount()):
                    item = self.recording_table.item(row, 7)
                    if item is not None and (item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()) == path:
                        holder = self.recording_table.cellWidget(row, opposite)
                        other = holder.findChild(QtWidgets.QCheckBox) if holder is not None else None
                        if other is not None:
                            other.blockSignals(True)
                            other.setChecked(False)
                            other.blockSignals(False)
                        break
            self.refresh_paths()

        def refresh_recordings(self, preferred_train_path: str = "") -> None:
            if self.scan_worker is not None and self.scan_worker.isRunning():
                return
            previous_train = set(self.selected_recordings("train")) if hasattr(self, "recording_table") else set()
            previous_test = set(self.selected_recordings("test")) if hasattr(self, "recording_table") else set()
            if isinstance(preferred_train_path, str) and preferred_train_path:
                previous_train.add(os.path.abspath(preferred_train_path))
                previous_test.discard(os.path.abspath(preferred_train_path))
            self._scan_previous_train = previous_train
            self._scan_previous_test = previous_test
            self.recording_table.setRowCount(0)
            self.refresh_button.setEnabled(False)
            self.train_progress.setValue(0)
            self.train_progress.setFormat("扫描采集记录：%p%")
            worker = RecordingScanWorker(self)
            worker.progress_signal.connect(self._on_scan_progress)
            worker.result_signal.connect(self._on_scan_finished)
            worker.error_signal.connect(self._on_scan_error)
            worker.finished.connect(worker.deleteLater)
            self.scan_worker = worker
            worker.start()

        def _on_scan_progress(self, percent: int, stage: str) -> None:
            self.train_progress.setValue(percent)
            self.train_progress.setFormat(f"{stage}：%p%")

        def _on_scan_finished(self, payload: Dict) -> None:
            previous_train = set(getattr(self, "_scan_previous_train", set()))
            previous_test = set(getattr(self, "_scan_previous_test", set()))
            records = list(payload.get("records", []))
            for warning in payload.get("warnings", []):
                self.log.append(str(warning))
            for meta in records:
                row = self.recording_table.rowCount()
                self.recording_table.insertRow(row)
                abs_path = str(meta["path"])
                for col, role in [(0, "train"), (1, "test")]:
                    check = QtWidgets.QCheckBox()
                    check.setToolTip("训练与测试互斥；同一记录不能同时勾选。")
                    checked = abs_path in (previous_train if role == "train" else previous_test)
                    check.setChecked(checked)
                    check.toggled.connect(lambda state, p=abs_path, r=role: self._dataset_role_changed(p, r, state))
                    holder = QtWidgets.QWidget()
                    holder_layout = QtWidgets.QHBoxLayout(holder)
                    holder_layout.setContentsMargins(0, 0, 0, 0)
                    holder_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    holder_layout.addWidget(check)
                    self.recording_table.setCellWidget(row, col, holder)
                values = [
                    str(meta["time"]),
                    str(meta["trials"]),
                    str(meta["eeg_channels"]),
                    str(meta["aux_channels"]),
                    f"{meta['fs']} Hz",
                    os.path.relpath(abs_path, RECORDINGS_DIR),
                ]
                for col, value in enumerate(values, start=2):
                    item = QtWidgets.QTableWidgetItem(value)
                    if col == 7:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, abs_path)
                        item.setToolTip(abs_path)
                    self.recording_table.setItem(row, col, item)
            valid_count = len(records)
            if valid_count and not previous_train:
                first = self.recording_table.cellWidget(0, 0)
                check = first.findChild(QtWidgets.QCheckBox) if first is not None else None
                if check is not None:
                    check.setChecked(True)
            self.recording_root_label.setText(f"采集目录：{RECORDINGS_DIR} | 找到 {valid_count} 条可训练记录")
            self.train_progress.setValue(100)
            self.train_progress.setFormat("记录扫描完成：%p%")
            self.refresh_button.setEnabled(True)
            self.scan_worker = None
            self.refresh_paths()

        def _on_scan_error(self, message: str) -> None:
            self.log.append(f"扫描采集记录失败：{message}")
            self.train_progress.setValue(0)
            self.train_progress.setFormat("记录扫描失败")
            self.refresh_button.setEnabled(True)
            self.scan_worker = None

        def recording_added(self, path: str) -> None:
            self.refresh_recordings(path)
            self.log.append(f"已加入最新真实采集记录：{path}")

        def _metric_float(self, metrics: Dict, *keys: str) -> Optional[float]:
            for key in keys:
                value = metrics.get(key)
                if value is None:
                    continue
                try:
                    score = float(value)
                except Exception:
                    continue
                if np.isfinite(score):
                    return score
            return None

        def _set_metric_image(self, label: QtWidgets.QLabel, pixmap: QtGui.QPixmap) -> None:
            if pixmap.isNull():
                label.setText("暂无可显示的指标图")
                return
            label.setText("")
            rect = label.contentsRect()
            target_width = max(260, rect.width() - 16)
            target_height = max(120, rect.height() - 16)
            scaled = pixmap.scaled(
                target_width,
                target_height,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(scaled)

        def _render_metric_report(
            self,
            title: str,
            rows: List[Tuple[str, float]],
            value_suffix: str,
            value_min: float,
            value_max: float,
            color: str,
            diverging: bool = False,
        ) -> QtGui.QPixmap:
            width = 1280
            row_count = max(1, len(rows))
            row_h = 40 if row_count <= 8 else 34
            title_top = 30
            title_h = 42
            top = title_top + title_h + 34
            bottom = 52
            height = max(190, top + row_h * row_count + bottom)
            pixmap = QtGui.QPixmap(width, height)
            pixmap.fill(QtGui.QColor("#ffffff"))
            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

            title_font = QtGui.QFont()
            title_font.setPointSize(18)
            title_font.setBold(True)
            while title_font.pointSize() > 11:
                metrics = QtGui.QFontMetrics(title_font)
                if metrics.horizontalAdvance(title) <= width - 112:
                    break
                title_font.setPointSize(title_font.pointSize() - 1)
            painter.setFont(title_font)
            painter.setPen(QtGui.QColor("#123042"))
            painter.drawText(
                QtCore.QRectF(56, title_top, width - 112, title_h),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                title,
            )

            label_font = QtGui.QFont()
            label_font.setPointSize(10)
            value_font = QtGui.QFont()
            value_font.setPointSize(10)
            value_font.setBold(True)
            left = 190
            right = 116
            bar_w = width - left - right - 30
            track_color = QtGui.QColor("#e9eef3")
            fill_color = QtGui.QColor(color)
            negative_color = QtGui.QColor("#c44d58")
            text_color = QtGui.QColor("#263746")
            max_span = max(1e-9, value_max - value_min)

            for row, (name, raw_value) in enumerate(rows):
                y = top + row * row_h
                value = max(value_min, min(value_max, float(raw_value)))
                painter.setFont(label_font)
                painter.setPen(text_color)
                painter.drawText(QtCore.QRectF(22, y - 1, left - 36, 24), QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, str(name))

                track_rect = QtCore.QRectF(left, y + 5, bar_w, 14)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(track_color)
                painter.drawRoundedRect(track_rect, 7, 7)

                if diverging:
                    zero_x = left + ((0.0 - value_min) / max_span) * bar_w
                    value_x = left + ((value - value_min) / max_span) * bar_w
                    fill_x = min(zero_x, value_x)
                    fill_w = abs(value_x - zero_x)
                    painter.setBrush(fill_color if value >= 0 else negative_color)
                    if fill_w >= 2:
                        painter.drawRoundedRect(QtCore.QRectF(fill_x, y + 5, fill_w, 14), 7, 7)
                    painter.setPen(QtGui.QPen(QtGui.QColor("#94a3af"), 1))
                    painter.drawLine(QtCore.QPointF(zero_x, y + 1), QtCore.QPointF(zero_x, y + 23))
                else:
                    fill_w = ((value - value_min) / max_span) * bar_w
                    painter.setBrush(fill_color)
                    if fill_w >= 2:
                        painter.drawRoundedRect(QtCore.QRectF(left, y + 5, fill_w, 14), 7, 7)

                painter.setFont(value_font)
                painter.setPen(text_color)
                if value_suffix == "%":
                    value_text = f"{raw_value:.1f}%"
                else:
                    value_text = f"{raw_value:.3f}"
                painter.drawText(QtCore.QRectF(left + bar_w + 12, y - 1, right, 24), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, value_text)

            painter.setFont(label_font)
            painter.setPen(QtGui.QColor("#6a7885"))
            note = "训练报告图：左侧为类别/指标，中间为结果条，右侧为数值。"
            painter.drawText(QtCore.QRectF(40, height - 36, width - 80, 24), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, note)
            painter.end()
            return pixmap

        def _plot_kappa(self, metrics: Dict, title: str = "Kappa 系数") -> None:
            items: List[Tuple[str, float]] = []
            eeg_kappa = self._metric_float(metrics, "eeg_only_kappa", "eeg_kappa")
            fusion_kappa = self._metric_float(metrics, "fusion_kappa")
            current_kappa = self._metric_float(metrics, "kappa")
            if eeg_kappa is not None:
                items.append(("EEG", eeg_kappa))
            if fusion_kappa is not None:
                items.append(("融合", fusion_kappa))
            elif current_kappa is not None:
                items.append(("Kappa", current_kappa))
            if not items:
                self.kappa_pixmap = self._render_metric_report("Kappa 系数：暂无结果", [("Kappa", 0.0)], "", -1.0, 1.0, "#1787a6", True)
                self._set_metric_image(self.kappa_image_label, self.kappa_pixmap)
                return
            rows = [(name, max(-1.0, min(1.0, float(score)))) for name, score in items]
            self.kappa_pixmap = self._render_metric_report(title, rows, "", -1.0, 1.0, "#1787a6", True)
            self._set_metric_image(self.kappa_image_label, self.kappa_pixmap)

        def export_metric_images(self, output_prefix: str) -> Tuple[str, str]:
            accuracy_path = output_prefix + "_accuracy.png"
            kappa_path = output_prefix + "_kappa.png"
            if self.accuracy_pixmap.isNull() or self.kappa_pixmap.isNull():
                raise RuntimeError("暂无可导出的指标图片")
            self.accuracy_pixmap.save(accuracy_path, "PNG")
            self.kappa_pixmap.save(kappa_path, "PNG")
            return accuracy_path, kappa_path

        def _log_exported_metric_images(self, output_prefix: str) -> None:
            try:
                accuracy_path, kappa_path = self.export_metric_images(output_prefix)
                self.log.append(f"准确率图片已保存：{accuracy_path}")
                self.log.append(f"Kappa 图片已保存：{kappa_path}")
            except Exception as exc:
                self.log.append(f"导出准确率/Kappa 图片失败：{exc}")

        def update_class_accuracy_plot(self, metrics: Dict) -> None:
            labels = list(metrics.get("class_labels", []))
            values = [float(x) * 100.0 for x in metrics.get("class_accuracy", [])]
            if not labels or not values:
                self.summary_label.setText("准确率：暂无分类评估结果")
                self.kappa_label.setText("Kappa 系数：-")
                self.accuracy_pixmap = self._render_metric_report("准确率：暂无分类评估结果", [("暂无数据", 0.0)], "%", 0.0, 100.0, "#1787a6")
                self.kappa_pixmap = self._render_metric_report("Kappa 系数：暂无结果", [("暂无数据", 0.0)], "", -1.0, 1.0, "#1787a6", True)
                self._set_metric_image(self.accuracy_image_label, self.accuracy_pixmap)
                self._set_metric_image(self.kappa_image_label, self.kappa_pixmap)
                return
            class_count = len(labels)
            strategy_name = dict(metrics.get("label_contract", {}) or {}).get("name", f"{class_count} 类")
            rows = list(zip(labels, values))
            self.accuracy_pixmap = self._render_metric_report(
                f"{strategy_name} 准确率",
                rows,
                "%",
                0.0,
                100.0,
                "#20a4b8" if metrics.get("emg_added") else "#1787a6",
            )
            self._set_metric_image(self.accuracy_image_label, self.accuracy_pixmap)
            mode = "EEG + 肌电融合" if metrics.get("emg_added") else "仅 EEG"
            mean_acc = float(metrics.get("overall_accuracy", np.mean(values) / 100.0)) * 100.0
            self.summary_label.setText(f"准确率：{mode} {strategy_name} 平均 {mean_acc:.1f}%")
            kappa = metrics.get("kappa", metrics.get("fusion_kappa", metrics.get("eeg_only_kappa")))
            try:
                kappa_text = f"{float(kappa):.3f}"
            except Exception:
                kappa_text = "-"
            self.kappa_label.setText(f"Kappa 系数：{kappa_text}")
            self._plot_kappa(metrics, "Kappa 系数（离线多分类一致性）")

        def _set_training_busy(self, busy: bool) -> None:
            self.train_button.setEnabled(not busy)
            self.eval_button.setEnabled(not busy)
            self.refresh_button.setEnabled(not busy)
            self.recording_table.setEnabled(not busy)
            if busy:
                self.train_button.setText("训练中...")
            else:
                self.train_button.setText("开始训练")

        def _on_training_progress(self, percent: int, stage: str) -> None:
            self.train_progress.setValue(max(0, min(100, int(percent))))
            self.train_progress.setFormat(f"{stage}：%p%")
            if stage:
                self.summary_label.setText(f"训练状态：{stage}")

        def _on_training_log(self, message: str) -> None:
            if message:
                self.log.append(str(message))

        def _on_training_finished(self, result: Dict) -> None:
            self._set_training_busy(False)
            self.training_worker = None
            metrics = dict(result.get("metrics", {}) or {})
            plot_metrics = dict(result.get("plot_metrics", metrics) or {})
            self.last_metrics = plot_metrics
            model_path = str(result.get("model_path", ""))
            report_path = str(result.get("report_path", ""))
            train_data_path = str(result.get("train_data_path", ""))
            if model_path:
                self.config.set_model_path(model_path)
            if train_data_path:
                self.config.train_data_edit.setText(train_data_path)
            self.log.append(
                f"自动模型竞赛完成：{len(result.get('data_paths', []))} 条记录，"
                f"{int(result.get('trial_count', 0))} trials, {metrics.get('n_classes', 0)} 类"
            )
            self.log.append(f"自动选中的最佳模型：{result.get('algorithm', '-')}")
            comparison = list(metrics.get("model_comparison", []) or [])
            if comparison:
                self.log.append("全部候选模型排名：")
                for rank, row in enumerate(comparison, start=1):
                    self.log.append(
                        f"{rank}. {row.get('classifier_label', row.get('classifier', '-'))} | "
                        f"准确率 {float(row.get('accuracy', 0.0)):.1%} | "
                        f"Kappa {float(row.get('kappa', 0.0)):.3f} | "
                        f"最低类别准确率 {float(row.get('minimum_class_accuracy', 0.0)):.1%}"
                    )
            self.log.append(
                f"标签契约：{self.config.train_config().get('label_contract', {}).get('name', 'EEG8 C1-C8 动作分类')}"
            )
            if model_path:
                self.log.append(f"最佳模型已保存并同步到在线模型/权重路径：{model_path}")
            if report_path:
                self.log.append(f"报告已保存：{report_path}")
            if metrics:
                self.log.append(json.dumps(metrics_summary(metrics), ensure_ascii=False, indent=2))
            if result.get("test_warning"):
                self.log.append(str(result["test_warning"]))
            if result.get("test_metrics"):
                test_metrics = dict(result.get("test_metrics") or {})
                self.log.append(f"训练后独立测试完成：{result.get('test_report_path')}")
                self.log.append(json.dumps(metrics_summary(test_metrics), ensure_ascii=False, indent=2))
            self.update_accuracy_plot(plot_metrics, str(result.get("plot_title", "训练验证准确率")))
            image_prefix = str(result.get("image_prefix", os.path.splitext(model_path)[0] if model_path else ""))
            if image_prefix:
                self._log_exported_metric_images(image_prefix)
            if model_path:
                self.model_ready.emit(model_path)
            self._on_training_progress(100, "训练完成")

        def _on_training_error(self, message: str) -> None:
            self._set_training_busy(False)
            self.training_worker = None
            short_message = str(message).splitlines()[0] if message else "未知错误"
            QtWidgets.QMessageBox.critical(self, "训练失败", short_message)
            self.log.append(f"训练失败：{short_message}")
            if message:
                self.log.append(str(message))
            self.train_progress.setFormat("训练失败：%p%")

        def train_model(self) -> None:
            cfg = self.config.train_config()
            data_paths = self.selected_recordings("train")
            test_paths = self.selected_recordings("test")
            model_path = cfg["model_path"]
            if self.training_worker is not None and self.training_worker.isRunning():
                QtWidgets.QMessageBox.information(self, "训练正在运行", "后台训练还未结束，请等待进度完成。")
                return
            if not data_paths:
                QtWidgets.QMessageBox.warning(self, "未勾选训练数据", "请在记录表的“训练”列至少勾选一条真实采集记录。")
                return
            overlap = sorted(
                set(map(os.path.abspath, data_paths))
                & set(map(os.path.abspath, test_paths))
            )
            if overlap:
                QtWidgets.QMessageBox.warning(
                    self,
                    "训练/测试数据重复",
                    "同一采集记录不能同时用于训练和独立测试。请取消其中一列：\n"
                    + "\n".join(overlap),
                )
                return
            if not model_path:
                QtWidgets.QMessageBox.warning(self, "缺少模型路径", "请先在 Config 页设置模型/权重输出路径。")
                return
            self.log.append("后台训练已启动：页面只接收进度、日志和最终结果。")
            self.train_progress.setValue(0)
            self._set_training_busy(True)
            worker = ModelTrainingWorker(
                data_paths,
                test_paths,
                model_path,
                cfg,
                int(self.config.fs_spin.value()),
                self,
            )
            worker.progress_signal.connect(self._on_training_progress)
            worker.log_signal.connect(self._on_training_log)
            worker.result_signal.connect(self._on_training_finished)
            worker.error_signal.connect(self._on_training_error)
            worker.finished.connect(worker.deleteLater)
            self.training_worker = worker
            worker.start()

        def evaluate_model(self) -> None:
            if self.evaluation_worker is not None and self.evaluation_worker.isRunning():
                return
            if not self.test_model_check.isChecked():
                QtWidgets.QMessageBox.warning(self, "未勾选模型", "离线测试需要使用当前模型/权重。")
                return
            cfg = self.config.train_config()
            model_path = cfg["model_path"]
            data_paths = self.selected_recordings("test")
            if not model_path or not os.path.exists(model_path):
                QtWidgets.QMessageBox.warning(self, "模型不存在", f"找不到模型/权重：{model_path}")
                return
            if not data_paths:
                QtWidgets.QMessageBox.warning(self, "未勾选测试数据", "请在记录表的“测试”列勾选独立采集记录。")
                return
            self.train_button.setEnabled(False)
            self.eval_button.setEnabled(False)
            self.train_progress.setValue(0)
            worker = ModelEvaluationWorker(model_path, data_paths, cfg, self)
            worker.progress_signal.connect(self._on_evaluation_progress)
            worker.result_signal.connect(self._on_evaluation_finished)
            worker.error_signal.connect(self._on_evaluation_error)
            worker.finished.connect(worker.deleteLater)
            self.evaluation_worker = worker
            worker.start()

        def _on_evaluation_progress(self, percent: int, stage: str) -> None:
            self.train_progress.setValue(percent)
            self.train_progress.setFormat(f"{stage}：%p%")

        def _on_evaluation_finished(self, payload: Dict) -> None:
            metrics = dict(payload["metrics"])
            data_paths = list(payload["data_paths"])
            self.last_metrics = metrics
            self.config.test_data_edit.setText(data_paths[0])
            self.log.append(f"离线测试完成：{len(data_paths)} 条独立记录，{payload['trial_count']} 个 trial")
            self.log.append(f"离线测试报告：{payload['report_path']}")
            self.log.append(json.dumps(metrics_summary(metrics), ensure_ascii=False, indent=2))
            self.update_accuracy_plot(metrics, "离线测试准确率")
            self._log_exported_metric_images(os.path.splitext(self.config.train_config()["model_path"])[0] + "_offline_eval")
            self.train_progress.setValue(100)
            self.train_progress.setFormat("离线评估完成：%p%")
            self.train_button.setEnabled(True)
            self.eval_button.setEnabled(True)
            self.evaluation_worker = None

        def _on_evaluation_error(self, message: str) -> None:
            self.log.append(f"离线测试失败：{message}")
            self.train_progress.setValue(0)
            self.train_progress.setFormat("离线评估失败")
            self.train_button.setEnabled(True)
            self.eval_button.setEnabled(True)
            self.evaluation_worker = None

        def update_accuracy_plot(self, metrics: Dict, title: str) -> None:
            if metrics.get("class_accuracy") is not None:
                self.update_class_accuracy_plot(metrics)
                return
            items = []
            if metrics.get("eeg_accuracy") is not None:
                items.append(("EEG", metrics.get("eeg_accuracy")))
            else:
                items.append(("EEG", metrics.get("eeg_only_accuracy")))
            items.extend([
                ("AUX", metrics.get("aux_only_accuracy")),
                ("Fusion", metrics.get("fusion_accuracy")),
            ])
            labels = []
            values = []
            for name, value in items:
                if value is None:
                    continue
                try:
                    score = float(value)
                except Exception:
                    continue
                if not np.isfinite(score):
                    continue
                labels.append(name)
                values.append(score * 100.0 if score <= 1.0 else score)
            if not values:
                self.summary_label.setText("准确率：没有可绘制的验证结果")
                self.kappa_label.setText("Kappa 系数：-")
                self.accuracy_pixmap = self._render_metric_report("准确率：没有可绘制的验证结果", [("暂无数据", 0.0)], "%", 0.0, 100.0, "#1787a6")
                self.kappa_pixmap = self._render_metric_report("Kappa 系数：暂无结果", [("暂无数据", 0.0)], "", -1.0, 1.0, "#1787a6", True)
                self._set_metric_image(self.accuracy_image_label, self.accuracy_pixmap)
                self._set_metric_image(self.kappa_image_label, self.kappa_pixmap)
                return
            self.accuracy_pixmap = self._render_metric_report(
                title,
                list(zip(labels, values)),
                "%",
                0.0,
                100.0,
                "#1787a6",
            )
            self._set_metric_image(self.accuracy_image_label, self.accuracy_pixmap)
            best = max(values)
            kappa = metrics.get("kappa", metrics.get("fusion_kappa", metrics.get("eeg_kappa", metrics.get("eeg_only_kappa"))))
            suffix = ""
            if kappa is not None:
                try:
                    suffix = f"，Kappa {float(kappa):.3f}"
                except Exception:
                    suffix = ""
            self.summary_label.setText(f"准确率：{title} 最高 {best:.1f}%{suffix}")
            kappa_text = "-"
            if kappa is not None:
                try:
                    kappa_text = f"{float(kappa):.3f}"
                except Exception:
                    kappa_text = "-"
            self.kappa_label.setText(f"Kappa 系数：{kappa_text}")
            self._plot_kappa(metrics, "Kappa 系数（离线多分类一致性）")

    class RealtimeStartupWorker(QtCore.QThread):
        progress_signal = QtCore.pyqtSignal(int, str)
        log_signal = QtCore.pyqtSignal(str)
        ready_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(self, cfg: Dict, use_model: bool, use_config_thresholds: bool, parent=None):
            super().__init__(parent)
            self.cfg = cfg
            self.use_model = bool(use_model)
            self.use_config_thresholds = bool(use_config_thresholds)

        def _progress(self, value: int, text: str) -> None:
            self.progress_signal.emit(max(0, min(100, int(value))), str(text))

        def run(self) -> None:
            source = None
            try:
                if not self.use_model:
                    raise RuntimeError("请在准备配置页启用“使用当前模型/权重”")
                model_path = str(self.cfg.get("model_path", "") or "").strip()
                if not model_path or not os.path.exists(model_path):
                    raise RuntimeError(f"找不到模型/权重：{model_path}")
                self._progress(10, "读取模型")
                model = EEG8Model.load(model_path)
                if self.isInterruptionRequested():
                    return

                selected_paradigm = str(self.cfg.get("paradigm_key", "eeg8") or "eeg8")
                self._progress(35, "校验解码模型")
                if model.eeg_classifier is None or model.eeg_features is None:
                    raise RuntimeError("当前模型没有 EEG 分类器，请先训练所选范式模型")
                classes = {int(value) for value in np.asarray(model.eeg_classifier.classes_).reshape(-1)}
                model_paradigm = str(getattr(model, "paradigm_key", "eeg8") or "eeg8")
                strategy = dict(getattr(model, "training_strategy_", {}) or {})
                if selected_paradigm == "pure_mi_binary":
                    expected = {0, 1}
                elif selected_paradigm == "custom_mi":
                    definitions = list(self.cfg.get("class_definitions", []) or [])
                    expected = set(range(len(definitions)))
                    saved_names = list(strategy.get("class_names", []) or [])
                    current_names = [str(item.get("class_name", "")).strip() for item in definitions]
                    if saved_names != current_names:
                        raise RuntimeError("自定义类别名称与模型训练时不一致，请恢复类别名称或重新训练。")
                else:
                    expected = set(range(8))
                if classes != expected or model_paradigm != selected_paradigm:
                    raise RuntimeError(
                        f"模型与所选范式不匹配：选择={selected_paradigm}，"
                        f"模型={model_paradigm}，分类器类别={sorted(classes)}"
                    )

                # Freeze settings after model/window validation.
                runtime_session = build_runtime_session_config(
                    self.cfg,
                    model,
                    use_config_thresholds=self.use_config_thresholds,
                )
                model.decoder_config = runtime_session.decoder_config
                apply_runtime_control_config(model, runtime_session.decoder_config)
                runtime_strategy = runtime_session.runtime_strategy

                self._progress(55, "读取肌电模板")
                template_bank = None
                template_path = str(
                    runtime_session.decoder_config.masseter_template_path or ""
                ).strip()
                if template_path:
                    template_bank = load_masseter_template_bank(template_path)
                validate_eeg8_ncc_confirmation_timing(
                    runtime_session.runtime_strategy,
                    runtime_session.decoder_config,
                    template_bank,
                    classification_window_seconds=runtime_session.signal_window_seconds,
                )
                if self.isInterruptionRequested():
                    return

                self._progress(70, "搜索 BrainVision LSL 数据流")
                source = LslEeg8OnlineSource(
                    model.signal_config,
                    log_callback=self.log_signal.emit,
                    require_masseter_gate=runtime_session.requires_aux,
                )
                if self.isInterruptionRequested():
                    source.close()
                    return
                self._progress(100, "实时控制已就绪")
                self.ready_signal.emit(build_eeg8_realtime_startup_payload(
                    model, source, template_bank, runtime_strategy, runtime_session
                ))
                source = None
            except Exception as exc:
                if not self.isInterruptionRequested():
                    self.error_signal.emit(str(exc))
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception:
                        pass

    class VRRealtimeControlBasePage(QtWidgets.QWidget):
        CONFIRMATION_CANCEL_LIMIT = 5
        ONLINE_CONTROL_DEFAULTS_PATH = os.path.join(
            APP_DIR, "config", "online_control_defaults.json"
        )

        def __init__(self, bus: CommandBus, config: ConfigPage, parent=None):
            super().__init__(parent)
            self.bus = bus
            self.config = config
            self.model: Optional[EEG8Model] = None
            self.source: Optional[SignalSource] = None
            self.robot = GuiRobotArm(bus, self.append_log)
            self.grip_controller = GripStateController(self.robot)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick_online)
            self.sequence = 0
            self.eeg8_history: deque = deque(maxlen=1)
            self.last_command_time = 0.0
            self.transport_demo_running = False
            self.transport_demo_index = 0
            self.transport_demo_step_seconds = 5.0
            self.is_test_page = False
            self.decode_history: deque = deque(maxlen=200)
            self.decode_window_index = 0
            self.decode_log_path = ""
            self.pending_start_mode: Optional[str] = None
            self.active_runtime_strategy = DEFAULT_EEG8_RUNTIME_STRATEGY
            self.pure_mi_decoder = PureMISequenceDecoder()
            self.pure_mi_confidences: List[float] = []
            # Active-session settings from the startup worker.
            self.runtime_session: Optional[RuntimeSessionConfig] = None
            self.pending_confirmation: Optional[Dict] = None
            self.continuous_move_intent: Optional[str] = None
            self.continuous_move_started_at = 0.0
            self.confirmation_cancel_count = 0
            self.last_aux_timestamp = 0.0
            self.masseter_template_bank: Optional[MasseterTemplateBank] = None
            self.control_gate_state = "rest"
            self.gate_hold_start: Optional[float] = None
            self.gate_hold_seconds = 3.0
            # Shared phase-local AUX counter.
            self.phase_emg_events = PhaseEmgEventEngine()
            self.last_window_emg_counts = {"blink": 0, "left_jaw": 0, "right_jaw": 0}
            self.pending_record_emg_counts: Optional[Dict[str, int]] = None
            self.last_emg_assist_text = "肌电辅助：-"
            self.aux_baseline_rms: Optional[np.ndarray] = None
            self.aux_baseline_values: List[np.ndarray] = []
            self.aux_baseline_until = 0.0
            self.runtime_phase = "baseline"
            self.runtime_phase_started_at = 0.0
            self.operation_start_time = 0.0
            self.operation_elapsed_timer = QtCore.QTimer(self)
            self.operation_elapsed_timer.timeout.connect(self._update_operation_elapsed)
            self.manual_simulation_timer = QtCore.QTimer(self)
            self.manual_simulation_timer.setInterval(100)
            self.manual_simulation_timer.timeout.connect(
                self._update_manual_simulation_clock
            )
            self.startup_worker: Optional[QtCore.QThread] = None
            self.intent_buttons: Dict[str, QtWidgets.QPushButton] = {}
            self.manual_simulation_phase = "window1"
            self.manual_simulation_pending: Optional[Dict[str, object]] = None
            self.manual_simulation_token = 0
            self.manual_simulation_deadline_monotonic = 0.0
            self.manual_confirmation_count = 0
            self.manual_blink_count = 0
            self.manual_stop_pending = False
            # Open-loop XYZ estimate in millimetres relative to the last Home.
            self.translation_position_mm = {"x": 0.0, "y": 0.0, "z": 0.0}
            self.translation_tracking_intent: Optional[str] = None
            self.translation_tracking_started_at = 0.0
            self.translation_limit_timer = QtCore.QTimer(self)
            self.translation_limit_timer.setInterval(20)
            self.translation_limit_timer.timeout.connect(
                self._update_translation_soft_limit
            )
            self.bus.command_emitted.connect(self._observe_translation_command)
            self.manual_orientation_degrees = {
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
            }
            self.console_frame_width = 1380
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

            page_layout = QtWidgets.QVBoxLayout(self)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(0)
            page_scroll = QtWidgets.QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            page_content = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page_content)
            self.vr_layout = layout
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(12)
            title = QtWidgets.QLabel("真实机械臂八分类在线控制")
            title.setObjectName("title")
            self.title_label = title
            layout.addWidget(title)
            desc = QtWidgets.QLabel("三窗口流程：窗口1用 EEG+EMG 联合分类；窗口2只做短时咬肌计数确认；C1-C6 执行后进入窗口3，以分段眨眼计数停止持续移动。")
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            self.desc_label = desc
            layout.addWidget(desc)

            demo_card = QtWidgets.QFrame()
            self.demo_card = demo_card
            demo_card.setObjectName("vrControls")
            demo_layout = QtWidgets.QVBoxLayout(demo_card)
            demo_layout.setContentsMargins(12, 10, 12, 10)
            demo_title = QtWidgets.QLabel("机械臂流程")
            demo_title.setObjectName("vrTitle")
            demo_layout.addWidget(demo_title)
            self.demo_plan = QtWidgets.QTextEdit()
            self.demo_plan.setReadOnly(True)
            self.demo_plan.setMaximumHeight(112)
            self.demo_plan.setPlainText(
                "参数：2.0s EEG 解码后进入 1.0s 肌电确认；确认后才执行或返回下一轮。\n"
                "搬运流程：静息待命 -> 连续眨眼三次切换 -> EEG 八分类控制 -> 右下角方块 -> 中心目标区 -> 左下角贴边目标区。\n"
                "指令序列：静息待命 -> 门控切换 -> C3 前移 -> C2 右移 -> C6 下移 -> C7 抓取 -> "
                "C5 上移 -> C4/C1 到中心 -> C4/C1 到左下角 -> C8 释放。"
                "\nAUX1/2：左眼/右眼，连续眨眼三次切换状态。"
                "\nAUX3/4：左/右咬肌计数形成 00/01/10/11；候选映射为 00=C3/C4，01=C1/C2，10=C5/C6，11=C7/C8。"
            )
            demo_layout.addWidget(self.demo_plan)
            layout.addWidget(demo_card)

            runtime = QtWidgets.QFrame()
            self.runtime_panel = runtime
            runtime.setObjectName("vrControls")
            runtime_layout = QtWidgets.QHBoxLayout(runtime)
            runtime_layout.setContentsMargins(12, 10, 12, 10)
            runtime_layout.setSpacing(10)
            left_panel = QtWidgets.QWidget()
            left_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            left_panel.setMinimumWidth(0)
            left_panel.setMaximumWidth(980)
            left_grid = QtWidgets.QGridLayout(left_panel)
            left_grid.setContentsMargins(0, 0, 0, 0)
            left_grid.setHorizontalSpacing(8)
            left_grid.setVerticalSpacing(4)
            self.use_model_check = self.config.use_model_check
            self.use_weight_check = self.config.use_weight_check
            self.pure_emg_control_check = QtWidgets.QCheckBox("肌电直接控制（停用）")
            self.pure_emg_control_check.setChecked(False)
            self.pure_emg_control_check.setVisible(False)
            self.path_label = QtWidgets.QLabel("")
            self.path_label.setObjectName("vrInfo")
            self.path_label.setWordWrap(True)
            self.startup_progress = QtWidgets.QProgressBar()
            self.startup_progress.setRange(0, 100)
            self.startup_progress.setValue(0)
            self.startup_progress.setFormat("按回车或空格启动实时控制")
            self.startup_progress.setFixedHeight(30)
            self.stop_button = QtWidgets.QPushButton("停止")
            self.stop_button.clicked.connect(self.stop_online)
            self.reset_button = QtWidgets.QPushButton("回零")
            self.reset_button.clicked.connect(self.reset_task)
            self.gate_toggle_button = QtWidgets.QPushButton("手动切换：进入运动态")
            self.gate_toggle_button.clicked.connect(self.manual_toggle_control_gate)
            self.demo_button = QtWidgets.QPushButton("开始搬运")
            self.demo_button.setObjectName("primaryButton")
            self.demo_button.clicked.connect(self.arm_demo_start)
            self.window_label_base_text = "窗口1：EEG+EMG 2.0 s | 窗口2：咬肌确认 1.0 s"
            self.window_label = QtWidgets.QLabel(self.window_label_base_text)
            self.window_label.setObjectName("vrStatus")
            self.window_label.setWordWrap(False)
            self.gate_label = QtWidgets.QLabel("当前状态：静息态 | 门控：眨眼三次切换")
            self.gate_label.setObjectName("vrStatus")
            self.gate_label.setWordWrap(False)
            self.elapsed_label = QtWidgets.QLabel("操作计时：00.0 s")
            self.elapsed_label.setObjectName("vrStatus")
            self.elapsed_label.setWordWrap(False)
            self.gate_counter_label = QtWidgets.QLabel("当前时间段计数：眨眼 0 | 左咬肌 0 | 右咬肌 0 | AUX：未接入")
            self.gate_counter_label.setObjectName("vrStatus")
            self.gate_counter_label.setWordWrap(False)
            self.decode_result_label = QtWidgets.QLabel("实时解码结果：等待开始")
            self.decode_result_label.setObjectName("vrStatus")
            self.decode_result_label.setWordWrap(False)
            self.step_counter_label = QtWidgets.QLabel("进度：当前第 0 步 | 解码状态：待开始")
            self.step_counter_label.setObjectName("vrStatus")
            self.step_counter_label.setWordWrap(False)
            self.result_card = QtWidgets.QFrame()
            self.result_card.setObjectName("vrResultCard")
            self.result_card.setFixedWidth(200)
            self.result_card.setFixedHeight(118)
            self.result_card.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            result_layout = QtWidgets.QVBoxLayout(self.result_card)
            result_layout.setContentsMargins(12, 10, 12, 10)
            result_layout.setSpacing(2)
            result_title = QtWidgets.QLabel("当前时间段结果")
            result_title.setObjectName("vrTitle")
            result_layout.addWidget(result_title)
            self.current_result_label = QtWidgets.QLabel("")
            self.current_result_label.setObjectName("vrResultText")
            self.current_result_label.setWordWrap(False)
            self.current_result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.current_result_label.setFixedHeight(26)
            result_layout.addWidget(self.current_result_label)
            self.current_result_detail = QtWidgets.QLabel("")
            self.current_result_detail.setObjectName("vrStatus")
            self.current_result_detail.setWordWrap(False)
            self.current_result_detail.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.current_result_detail.setFixedHeight(20)
            result_layout.addWidget(self.current_result_detail)
            for label in [
                self.window_label,
                self.gate_label,
                self.elapsed_label,
                self.gate_counter_label,
                self.decode_result_label,
                self.step_counter_label,
            ]:
                label.setFixedHeight(24)
                label.setMinimumWidth(0)
                label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            left_grid.addWidget(self.pure_emg_control_check, 0, 2)
            left_grid.addWidget(self.path_label, 0, 3, 1, 2)
            left_grid.addWidget(self.gate_label, 1, 0, 1, 3)
            left_grid.addWidget(self.elapsed_label, 1, 3, 1, 2)
            # gate_counter_label remains as an internal counter sink, but its
            # duplicated "当前时间段计数" row is intentionally not displayed.
            left_grid.addWidget(self.decode_result_label, 2, 0, 1, 5)
            left_grid.addWidget(self.step_counter_label, 3, 0, 1, 5)
            left_grid.addWidget(self.startup_progress, 4, 0, 1, 5)
            left_grid.addWidget(self.stop_button, 5, 0, 1, 2)
            left_grid.addWidget(self.reset_button, 5, 2, 1, 3)
            left_grid.addWidget(self.gate_toggle_button, 6, 0, 1, 5)
            left_grid.setColumnStretch(0, 1)
            left_grid.setColumnStretch(1, 1)
            left_grid.setColumnStretch(2, 1)
            left_grid.setColumnStretch(3, 1)
            left_grid.setColumnStretch(4, 1)
            runtime_layout.addWidget(left_panel, 1)
            runtime_layout.addWidget(self.result_card, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            runtime_layout.setStretch(0, 1)
            runtime_layout.setStretch(1, 0)
            runtime.setFixedWidth(self.console_frame_width)
            runtime.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            runtime.setFixedHeight(230)
            layout.addWidget(runtime, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            self._set_current_result_card("collect", "待采集", "采集阶段：等待当前时间段数据")
            self.demo_card.setVisible(False)
            self.path_label.setVisible(False)
            self.window_label.setVisible(False)
            self.gate_toggle_button.setEnabled(False)

            # 真实机械臂配置覆盖原 VR 三维场景区域。上方运行状态区保持不变。
            robot_config = QtWidgets.QFrame()
            self.robot_config_panel = robot_config
            robot_config.setObjectName("vrControls")
            robot_config.setFixedWidth(self.console_frame_width)
            robot_config.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            robot_config_layout = QtWidgets.QVBoxLayout(robot_config)
            robot_config_layout.setContentsMargins(12, 10, 12, 10)
            robot_config_title = QtWidgets.QLabel("真实机械臂连接与动作参数")
            robot_config_title.setObjectName("vrTitle")
            robot_config_layout.addWidget(robot_config_title)
            robot_config_hint = QtWidgets.QLabel(
                "Ubuntu 先运行 receiver_arm.py。协议：TCP + 4 字节大端长度 + UTF-8 JSON；"
                "速度 1≈1 mm/s 或 0.005 rad/s，100≈10 cm/s 或 0.5 rad/s。"
            )
            robot_config_hint.setWordWrap(True)
            robot_config_hint.setObjectName("vrInfo")
            robot_config_layout.addWidget(robot_config_hint)
            robot_form = QtWidgets.QGridLayout()
            robot_form.setHorizontalSpacing(10)
            robot_form.setVerticalSpacing(8)
            self.online_robot_enabled_check = QtWidgets.QCheckBox("启用真实机械臂 TCP 输出")
            self.online_robot_enabled_check.setChecked(self.config.robot_enabled_check.isChecked())
            self.online_input_mode_combo = QtWidgets.QComboBox()
            self.online_input_mode_combo.addItem(
                "人工按钮三窗口测试（不连接 LSL）", "manual"
            )
            self.online_input_mode_combo.addItem(
                "BrainVision LSL 实时解码", "lsl"
            )
            self.online_input_mode_combo.setToolTip(
                "人工模式不加载模型、不搜索 LSL；LSL 模式才启动正式脑电解码。"
            )
            self.online_robot_host_edit = QtWidgets.QLineEdit(self.config.robot_host_edit.text())
            self.online_robot_port_spin = QtWidgets.QSpinBox()
            self.online_robot_port_spin.setRange(1, 65535)
            self.online_robot_port_spin.setValue(self.config.robot_port_spin.value())
            self.online_robot_speed_spin = QtWidgets.QDoubleSpinBox()
            self.online_robot_speed_spin.setRange(0.01, 200.0)
            self.online_robot_speed_spin.setDecimals(2)
            self.online_robot_speed_spin.setSingleStep(1.0)
            self.online_robot_speed_spin.setValue(self.config.robot_speed_spin.value())
            self.online_robot_rotation_speed_spin = QtWidgets.QDoubleSpinBox()
            self.online_robot_rotation_speed_spin.setRange(0.01, 200.0)
            self.online_robot_rotation_speed_spin.setDecimals(2)
            self.online_robot_rotation_speed_spin.setSingleStep(1.0)
            self.online_robot_rotation_speed_spin.setValue(20.0)
            self.online_robot_rotation_speed_spin.setToolTip(
                "仅用于 Pitch/Yaw/Roll；speed=100 约等于 0.5 rad/s。"
            )
            robot_form.addWidget(self.online_robot_enabled_check, 0, 0, 1, 2)
            robot_form.addWidget(QtWidgets.QLabel("输入模式"), 0, 2)
            robot_form.addWidget(self.online_input_mode_combo, 0, 3)
            robot_form.addWidget(QtWidgets.QLabel("Ubuntu 接收端 IP"), 1, 0)
            robot_form.addWidget(self.online_robot_host_edit, 1, 1)
            robot_form.addWidget(QtWidgets.QLabel("TCP 端口"), 1, 2)
            robot_form.addWidget(self.online_robot_port_spin, 1, 3)
            robot_form.addWidget(QtWidgets.QLabel("平移 speed"), 2, 0)
            robot_form.addWidget(self.online_robot_speed_spin, 2, 1)
            robot_form.addWidget(QtWidgets.QLabel("姿态旋转 speed"), 2, 2)
            robot_form.addWidget(self.online_robot_rotation_speed_spin, 2, 3)
            for column in (1, 3):
                robot_form.setColumnStretch(column, 1)
            robot_config_layout.addLayout(robot_form)
            self.online_robot_enabled_check.toggled.connect(self.config.robot_enabled_check.setChecked)
            self.online_robot_host_edit.textChanged.connect(self.config.robot_host_edit.setText)
            self.online_robot_port_spin.valueChanged.connect(self.config.robot_port_spin.setValue)
            self.online_robot_speed_spin.valueChanged.connect(self.config.robot_speed_spin.setValue)
            self.config.robot_enabled_check.toggled.connect(self.online_robot_enabled_check.setChecked)
            self.config.robot_host_edit.textChanged.connect(self.online_robot_host_edit.setText)
            self.config.robot_port_spin.valueChanged.connect(self.online_robot_port_spin.setValue)
            self.config.robot_speed_spin.valueChanged.connect(self.online_robot_speed_spin.setValue)
            self.online_input_mode_combo.currentIndexChanged.connect(
                self._online_input_mode_changed
            )
            layout.addWidget(robot_config, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

            manual = QtWidgets.QFrame()
            self.manual_panel = manual
            manual.setObjectName("vrControls")
            manual.setFixedWidth(self.console_frame_width)
            manual.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            row = QtWidgets.QGridLayout(manual)
            row.setContentsMargins(12, 10, 12, 10)
            row.setHorizontalSpacing(8)
            row.setVerticalSpacing(8)
            for button_index, (label, intent) in enumerate(
                [
                    ("前移", "forward"),
                    ("后移", "backward"),
                    ("左移", "left"),
                    ("右移", "right"),
                    ("上移", "up"),
                    ("下移", "down"),
                    ("抓取", "grab"),
                    ("释放", "release"),
                ]
            ):
                btn = QtWidgets.QPushButton(label)
                btn.setObjectName("vrCmd")
                btn.setProperty("active", False)
                btn.setFixedHeight(48)
                btn.clicked.connect(
                    lambda _=False, i=intent: self.simulate_manual_classification(i)
                )
                self.intent_buttons[intent] = btn
                row.addWidget(btn, button_index // 4, button_index % 4)
            for column in range(4):
                row.setColumnStretch(column, 1)
            orientation_hint = QtWidgets.QLabel(
                "姿态微调（°）：正数=增加，负数=减小，0=不动作"
            )
            orientation_hint.setObjectName("vrInfo")
            row.addWidget(orientation_hint, 2, 0, 1, 4)
            self.manual_orientation_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
            for column, (axis_label, intent_prefix) in enumerate(
                [("Pitch", "pitch"), ("Yaw", "yaw"), ("Roll", "roll")]
            ):
                field = QtWidgets.QDoubleSpinBox()
                field.setRange(-360.0, 360.0)
                field.setDecimals(2)
                field.setSingleStep(1.0)
                field.setSuffix(" °")
                field.setValue(0.0)
                field.setToolTip(
                    f"填写 {axis_label} 相对调整角度；正数增加，负数减小。"
                )
                self.manual_orientation_spins[intent_prefix] = field
                field_box = QtWidgets.QWidget()
                field_layout = QtWidgets.QHBoxLayout(field_box)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.addWidget(QtWidgets.QLabel(axis_label))
                field_layout.addWidget(field)
                row.addWidget(field_box, 3, column)
            self.orientation_apply_button = QtWidgets.QPushButton("执行姿态数值")
            self.orientation_apply_button.setObjectName("vrCmd")
            self.orientation_apply_button.setFixedHeight(48)
            self.orientation_apply_button.clicked.connect(
                self.execute_manual_orientation_values
            )
            row.addWidget(self.orientation_apply_button, 3, 3)
            self.manual_simulation_label = QtWidgets.QLabel(
                "人工测试状态：窗口1——请选择一个 C1-C8 分类结果"
            )
            self.manual_simulation_label.setObjectName("vrStatus")
            self.manual_simulation_label.setWordWrap(True)
            row.addWidget(self.manual_simulation_label, 4, 0, 1, 4)
            self.manual_confirm_button = QtWidgets.QPushButton(
                "确认执行"
            )
            self.manual_confirm_button.setObjectName("vrCmd")
            self.manual_confirm_button.setProperty("active", False)
            self.manual_confirm_button.setFixedHeight(48)
            self.manual_confirm_button.setEnabled(False)
            self.manual_confirm_button.clicked.connect(
                self.confirm_manual_simulation
            )
            row.addWidget(self.manual_confirm_button, 5, 0, 1, 2)
            self.manual_move_stop_button = QtWidgets.QPushButton(
                "移动停止"
            )
            self.manual_move_stop_button.setObjectName("vrCmd")
            self.manual_move_stop_button.setProperty("active", False)
            self.manual_move_stop_button.setFixedHeight(48)
            self.manual_move_stop_button.setEnabled(False)
            self.manual_move_stop_button.clicked.connect(
                self.stop_manual_simulated_motion
            )
            row.addWidget(self.manual_move_stop_button, 5, 2, 1, 2)
            self.config.paradigm_combo.currentIndexChanged.connect(
                lambda _index: (
                    self._refresh_pure_mi_counter(),
                    self._sync_gate_toggle_button(),
                )
            )
            self._refresh_pure_mi_counter()
            self._sync_gate_toggle_button()
            bus.command_emitted.connect(self._on_command_visual)
            layout.addWidget(manual, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

            history_panel = QtWidgets.QFrame()
            self.decode_history_panel = history_panel
            history_panel.setObjectName("vrControls")
            history_panel.setFixedWidth(self.console_frame_width)
            history_panel.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            history_layout = QtWidgets.QVBoxLayout(history_panel)
            history_layout.setContentsMargins(12, 10, 12, 10)
            history_title = QtWidgets.QLabel("在线测试分类结果")
            history_title.setObjectName("vrTitle")
            history_layout.addWidget(history_title)
            self.decode_history_table = QtWidgets.QTableWidget(0, 8)
            self.decode_history_table.setHorizontalHeaderLabels(
                ["#", "时间", "真实", "预测", "动作", "置信度", "Margin", "眨眼 / 左咬肌 / 右咬肌"]
            )
            self.decode_history_table.setAlternatingRowColors(True)
            self.decode_history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.decode_history_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.decode_history_table.verticalHeader().setVisible(False)
            self.decode_history_table.setMinimumHeight(170)
            self.decode_history_table.setMaximumHeight(260)
            history_header = self.decode_history_table.horizontalHeader()
            for col in range(7):
                history_header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            history_header.setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeMode.Stretch)
            history_layout.addWidget(self.decode_history_table)
            layout.addWidget(history_panel, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

            self.status_label = QtWidgets.QLabel("状态：未加载模型")
            self.status_label.setObjectName("vrStatus")
            self.status_label.setFixedWidth(self.console_frame_width)
            layout.addWidget(self.status_label, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(128)
            self.log.setFixedWidth(self.console_frame_width)
            layout.addWidget(self.log, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

            online_timing_panel = QtWidgets.QFrame()
            self.online_timing_panel = online_timing_panel
            online_timing_panel.setObjectName("vrControls")
            online_timing_panel.setFixedWidth(self.console_frame_width)
            online_timing_layout = QtWidgets.QFormLayout(online_timing_panel)
            online_timing_layout.setContentsMargins(12, 10, 12, 10)
            online_timing_title = QtWidgets.QLabel(
                "三窗口参数（LSL 与人工按钮模式共用）"
            )
            online_timing_title.setObjectName("vrTitle")
            online_timing_layout.addRow(online_timing_title)

            self.online_window1_seconds_spin = QtWidgets.QDoubleSpinBox()
            self.online_window1_seconds_spin.setRange(0.5, 60.0)
            self.online_window1_seconds_spin.setSingleStep(0.05)
            self.online_window1_seconds_spin.setDecimals(2)
            self.online_window1_seconds_spin.setSuffix(" s")
            self.online_window1_seconds_spin.setValue(
                self.config.decode_window_spin.value()
            )
            online_timing_layout.addRow(
                "窗口1：EEG+EMG 联合时长",
                self.online_window1_seconds_spin,
            )

            self.online_window2_seconds_spin = QtWidgets.QDoubleSpinBox()
            self.online_window2_seconds_spin.setRange(0.2, 60.0)
            self.online_window2_seconds_spin.setSingleStep(0.05)
            self.online_window2_seconds_spin.setDecimals(2)
            self.online_window2_seconds_spin.setSuffix(" s")
            self.online_window2_seconds_spin.setValue(
                self.config.confirm_window_spin.value()
            )
            online_timing_layout.addRow(
                "窗口2：咬肌确认时长",
                self.online_window2_seconds_spin,
            )

            self.online_window2_count_spin = QtWidgets.QSpinBox()
            self.online_window2_count_spin.setRange(1, 999)
            self.online_window2_count_spin.setValue(
                self.config.confirm_masseter_count_spin.value()
            )
            online_timing_layout.addRow(
                "窗口2：确认咬肌次数",
                self.online_window2_count_spin,
            )

            self.online_continuous_speed_spin = QtWidgets.QDoubleSpinBox()
            self.online_continuous_speed_spin.setRange(0.1, 100.0)
            self.online_continuous_speed_spin.setSingleStep(0.05)
            self.online_continuous_speed_spin.setDecimals(2)
            self.online_continuous_speed_spin.setSuffix(" 格/秒")
            self.online_continuous_speed_spin.setValue(
                self.config.continuous_move_speed_spin.value()
            )
            online_timing_layout.addRow(
                "连续移动速度",
                self.online_continuous_speed_spin,
            )

            self.online_window3_seconds_spin = QtWidgets.QDoubleSpinBox()
            self.online_window3_seconds_spin.setRange(0.5, 60.0)
            self.online_window3_seconds_spin.setSingleStep(0.05)
            self.online_window3_seconds_spin.setDecimals(2)
            self.online_window3_seconds_spin.setSuffix(" s")
            self.online_window3_seconds_spin.setValue(
                self.config.continuous_blink_window_spin.value()
            )
            online_timing_layout.addRow(
                "窗口3：持续运动眨眼时长",
                self.online_window3_seconds_spin,
            )

            self.online_window3_count_spin = QtWidgets.QSpinBox()
            self.online_window3_count_spin.setRange(1, 999)
            self.online_window3_count_spin.setValue(
                self.config.continuous_blink_count_spin.value()
            )
            online_timing_layout.addRow(
                "窗口3：停止眨眼次数",
                self.online_window3_count_spin,
            )

            orientation_limit_title = QtWidgets.QLabel(
                "姿态软限位（相对回零点，上下限可在 -360°～360° 内填写）"
            )
            orientation_limit_title.setObjectName("vrTitle")
            online_timing_layout.addRow(orientation_limit_title)
            self.orientation_min_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
            self.orientation_max_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
            for axis, label in [
                ("pitch", "Pitch"),
                ("yaw", "Yaw"),
                ("roll", "Roll"),
            ]:
                limit_box = QtWidgets.QWidget()
                limit_layout = QtWidgets.QHBoxLayout(limit_box)
                limit_layout.setContentsMargins(0, 0, 0, 0)
                minimum_spin = QtWidgets.QDoubleSpinBox()
                minimum_spin.setRange(-360.0, 360.0)
                minimum_spin.setDecimals(2)
                minimum_spin.setSingleStep(1.0)
                minimum_spin.setSuffix(" °")
                minimum_spin.setValue(-360.0)
                maximum_spin = QtWidgets.QDoubleSpinBox()
                maximum_spin.setRange(-360.0, 360.0)
                maximum_spin.setDecimals(2)
                maximum_spin.setSingleStep(1.0)
                maximum_spin.setSuffix(" °")
                maximum_spin.setValue(360.0)
                minimum_spin.setToolTip("相对回零点的软件最小角度。")
                maximum_spin.setToolTip("相对回零点的软件最大角度。")
                minimum_spin.valueChanged.connect(
                    lambda _value, a=axis: self._refresh_orientation_limit_status(a)
                )
                maximum_spin.valueChanged.connect(
                    lambda _value, a=axis: self._refresh_orientation_limit_status(a)
                )
                self.orientation_min_spins[axis] = minimum_spin
                self.orientation_max_spins[axis] = maximum_spin
                limit_layout.addWidget(QtWidgets.QLabel("下限"))
                limit_layout.addWidget(minimum_spin)
                limit_layout.addWidget(QtWidgets.QLabel("上限"))
                limit_layout.addWidget(maximum_spin)
                online_timing_layout.addRow(label, limit_box)
            self.orientation_limit_status_label = QtWidgets.QLabel("")
            self.orientation_limit_status_label.setObjectName("vrStatus")
            self.orientation_limit_status_label.setWordWrap(True)
            online_timing_layout.addRow(
                "当前软件估算角度",
                self.orientation_limit_status_label,
            )
            self._refresh_orientation_limit_status()

            translation_limit_title = QtWidgets.QLabel(
                "平移软限位（相对回零点，各方向最大距离）"
            )
            translation_limit_title.setObjectName("vrTitle")
            online_timing_layout.addRow(translation_limit_title)
            self.translation_limit_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
            for first_key, first_label, second_key, second_label in [
                ("forward", "前移", "backward", "后移"),
                ("left", "左移", "right", "右移"),
                ("up", "上移", "down", "下移"),
            ]:
                limit_box = QtWidgets.QWidget()
                limit_layout = QtWidgets.QHBoxLayout(limit_box)
                limit_layout.setContentsMargins(0, 0, 0, 0)
                for key, label in [
                    (first_key, first_label),
                    (second_key, second_label),
                ]:
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(0.0, 100000.0)
                    spin.setDecimals(1)
                    spin.setSingleStep(10.0)
                    spin.setSuffix(" mm")
                    spin.setValue(1000.0)
                    spin.setToolTip("相对最近一次回零点允许移动的最大距离。")
                    self.translation_limit_spins[key] = spin
                    limit_layout.addWidget(QtWidgets.QLabel(label))
                    limit_layout.addWidget(spin)
                online_timing_layout.addRow(
                    f"{first_label}/{second_label}", limit_box
                )
            self.translation_limit_status_label = QtWidgets.QLabel("")
            self.translation_limit_status_label.setObjectName("vrStatus")
            online_timing_layout.addRow(
                "当前软件估算位置", self.translation_limit_status_label
            )
            self._refresh_translation_limit_status()

            settings_buttons = QtWidgets.QWidget()
            settings_buttons_layout = QtWidgets.QHBoxLayout(settings_buttons)
            settings_buttons_layout.setContentsMargins(0, 4, 0, 0)
            self.save_online_control_settings_button = QtWidgets.QPushButton(
                "保存在线控制参数"
            )
            self.save_online_control_settings_button.setObjectName("primaryButton")
            self.save_online_control_settings_button.clicked.connect(
                self._save_online_control_settings
            )
            self.online_control_settings_status = QtWidgets.QLabel("")
            self.online_control_settings_status.setObjectName("hint")
            settings_buttons_layout.addWidget(
                self.save_online_control_settings_button
            )
            settings_buttons_layout.addWidget(
                self.online_control_settings_status, 1
            )
            online_timing_layout.addRow(settings_buttons)

            timing_pairs = [
                (self.online_window1_seconds_spin, self.config.decode_window_spin),
                (self.online_window2_seconds_spin, self.config.confirm_window_spin),
                (self.online_window2_count_spin, self.config.confirm_masseter_count_spin),
                (self.online_continuous_speed_spin, self.config.continuous_move_speed_spin),
                (self.online_window3_seconds_spin, self.config.continuous_blink_window_spin),
                (self.online_window3_count_spin, self.config.continuous_blink_count_spin),
            ]
            for online_widget, config_widget in timing_pairs:
                online_widget.valueChanged.connect(config_widget.setValue)
                config_widget.valueChanged.connect(online_widget.setValue)
            self._load_online_control_settings()

            layout.addWidget(
                online_timing_panel,
                0,
                QtCore.Qt.AlignmentFlag.AlignLeft,
            )
            page_scroll.setWidget(page_content)
            page_layout.addWidget(page_scroll)

            for shortcut in ["Return", "Enter", "Space"]:
                QtGui.QShortcut(QtGui.QKeySequence(shortcut), self, activated=self.begin_armed_start)
            self.config.config_changed.connect(self.refresh_paths)
            self.refresh_paths()

        def _online_control_settings(self) -> Dict[str, object]:
            return {
                "robot_enabled": self.online_robot_enabled_check.isChecked(),
                "input_mode": self.online_input_mode_combo.currentData(),
                "robot_host": self.online_robot_host_edit.text().strip(),
                "robot_port": int(self.online_robot_port_spin.value()),
                "translation_speed": float(self.online_robot_speed_spin.value()),
                "rotation_speed": float(
                    self.online_robot_rotation_speed_spin.value()
                ),
                "window1_seconds": float(
                    self.online_window1_seconds_spin.value()
                ),
                "window2_seconds": float(
                    self.online_window2_seconds_spin.value()
                ),
                "window2_count": int(self.online_window2_count_spin.value()),
                "continuous_speed": float(
                    self.online_continuous_speed_spin.value()
                ),
                "window3_seconds": float(
                    self.online_window3_seconds_spin.value()
                ),
                "window3_count": int(self.online_window3_count_spin.value()),
                "orientation_min": {
                    axis: float(spin.value())
                    for axis, spin in self.orientation_min_spins.items()
                },
                "orientation_max": {
                    axis: float(spin.value())
                    for axis, spin in self.orientation_max_spins.items()
                },
                "translation_limits_mm": {
                    direction: float(spin.value())
                    for direction, spin in self.translation_limit_spins.items()
                },
            }

        def _save_online_control_settings(self) -> None:
            path = self.ONLINE_CONTROL_DEFAULTS_PATH
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(
                        self._online_control_settings(),
                        stream,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.online_control_settings_status.setText(
                    "已保存，下一次启动自动加载"
                )
                self.append_log(f"在线控制参数已保存：{path}")
            except Exception as exc:
                self.online_control_settings_status.setText(f"保存失败：{exc}")
                QtWidgets.QMessageBox.critical(self, "保存失败", str(exc))

        def _load_online_control_settings(self) -> None:
            path = self.ONLINE_CONTROL_DEFAULTS_PATH
            if not os.path.exists(path):
                self.online_control_settings_status.setText("尚未保存")
                return
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    data = json.load(stream)
                if not isinstance(data, dict):
                    raise ValueError("配置文件不是 JSON 对象")
                self.online_robot_enabled_check.setChecked(
                    bool(data.get("robot_enabled", False))
                )
                mode = str(data.get("input_mode", "manual"))
                index = self.online_input_mode_combo.findData(mode)
                if index >= 0:
                    self.online_input_mode_combo.setCurrentIndex(index)
                self.online_robot_host_edit.setText(
                    str(data.get("robot_host", self.online_robot_host_edit.text()))
                )
                values = [
                    ("robot_port", self.online_robot_port_spin),
                    ("translation_speed", self.online_robot_speed_spin),
                    ("rotation_speed", self.online_robot_rotation_speed_spin),
                    ("window1_seconds", self.online_window1_seconds_spin),
                    ("window2_seconds", self.online_window2_seconds_spin),
                    ("window2_count", self.online_window2_count_spin),
                    ("continuous_speed", self.online_continuous_speed_spin),
                    ("window3_seconds", self.online_window3_seconds_spin),
                    ("window3_count", self.online_window3_count_spin),
                ]
                for key, widget in values:
                    if key in data:
                        widget.setValue(data[key])
                for group_key, widgets in [
                    ("orientation_min", self.orientation_min_spins),
                    ("orientation_max", self.orientation_max_spins),
                    ("translation_limits_mm", self.translation_limit_spins),
                ]:
                    saved = data.get(group_key, {})
                    if isinstance(saved, dict):
                        for key, widget in widgets.items():
                            if key in saved:
                                widget.setValue(float(saved[key]))
                self.online_control_settings_status.setText("已自动加载保存值")
                self._refresh_orientation_limit_status()
                self._refresh_translation_limit_status()
            except Exception as exc:
                self.online_control_settings_status.setText(f"加载失败：{exc}")

        def refresh_paths(self) -> None:
            cfg = self.config.realtime_config()
            self.path_label.setText(f"{cfg['mode']} | 模型/权重：{cfg['model_path']} | 目标：{cfg['target']}")

        def _online_input_mode(self) -> str:
            if not hasattr(self, "online_input_mode_combo"):
                return "manual"
            return str(self.online_input_mode_combo.currentData() or "manual")

        def _online_input_mode_changed(self) -> None:
            if self._online_input_mode() == "manual":
                self.startup_progress.setFormat(
                    "人工按钮模式 | 按回车或空格开始（不连接 LSL）"
                )
                self.status_label.setText(
                    "状态：人工按钮模式 | 不加载模型、不连接 LSL"
                )
            else:
                self.startup_progress.setFormat(
                    "LSL 模式 | 按回车或空格加载模型并订阅 LSL"
                )
                self.status_label.setText(
                    "状态：LSL 实时解码模式 | 等待启动"
                )

        def set_model_path(self, path: str) -> None:
            self.config.set_model_path(path)
            self.refresh_paths()

        def append_log(self, text: str) -> None:
            self.log.append(text)

        def _start_decode_result_session(self) -> None:
            self.decode_history.clear()
            self.decode_window_index = 0
            if hasattr(self, "decode_history_table"):
                self.decode_history_table.setRowCount(0)
            out_dir = os.path.join(APP_DIR, "data", "experiments", "online_decode_results")
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.decode_log_path = os.path.join(out_dir, f"online_decode_{stamp}.jsonl")
            self.append_log(f"在线分类结果将保存：{self.decode_log_path}")

        def _append_decode_result(
            self,
            decision: Decision,
            true_label: str,
            pred_text: str,
            emg_stats: str,
        ) -> None:
            self.decode_window_index += 1
            count_snapshot = getattr(self, "pending_record_emg_counts", None)
            self.pending_record_emg_counts = None
            if count_snapshot is None:
                count_snapshot = dict(getattr(
                    self,
                    "last_window_emg_counts",
                    {"blink": 0, "left_jaw": 0, "right_jaw": 0},
                ))
            blink_count = int(count_snapshot.get("blink", 0))
            left_count = int(count_snapshot.get("left_jaw", 0))
            right_count = int(count_snapshot.get("right_jaw", 0))
            record = {
                "index": int(self.decode_window_index),
                "timestamp": datetime.fromtimestamp(float(decision.timestamp)).isoformat(timespec="milliseconds"),
                "true_label": true_label or "-",
                "prediction": pred_text,
                "intent": str(decision.intent),
                "state": str(decision.state),
                "confidence": float(decision.confidence),
                "margin": float(decision.margin),
                "accepted": bool(decision.accepted),
                "command": str(decision.command or ""),
                "reason": str(decision.reason),
                # Store the completed window counters.
                "emg_stats": self._emg_stats_from_counts(count_snapshot),
                "emg_counts": {
                    "blink": blink_count,
                    "left_jaw": left_count,
                    "right_jaw": right_count,
                },
                "probabilities": dict(decision.probabilities or {}),
            }
            self.decode_history.append(record)
            if self.decode_log_path:
                try:
                    with open(self.decode_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except Exception as exc:
                    self.append_log(f"保存在线分类结果失败：{exc}")
            if not hasattr(self, "decode_history_table"):
                return
            table = self.decode_history_table
            row = table.rowCount()
            table.insertRow(row)
            values = [
                record["index"],
                record["timestamp"][11:19],
                record["true_label"],
                record["prediction"],
                record["intent"],
                f"{record['confidence']:.3f}",
                f"{record['margin']:.3f}",
                f"眨眼 {blink_count} | 左 {left_count} | 右 {right_count}",
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if col in (0, 5, 6):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                if record["accepted"]:
                    item.setBackground(QtGui.QColor("#e7f6ee"))
                elif record["intent"] in {"rest", "artifact", "pure_emg_wait"}:
                    item.setBackground(QtGui.QColor("#f4f7fa"))
                table.setItem(row, col, item)
            while table.rowCount() > 200:
                table.removeRow(0)
            table.scrollToBottom()

        def _sync_gate_toggle_button(self) -> None:
            if not hasattr(self, "gate_toggle_button"):
                return
            pure_mi = (
                self.config._combo_value(self.config.paradigm_combo)
                == "pure_mi_binary"
            )
            self.gate_toggle_button.setVisible(not pure_mi)
            self.gate_toggle_button.setEnabled(not pure_mi)
            if pure_mi:
                if hasattr(self, "gate_label"):
                    state_text = "静息态" if self.control_gate_state == "rest" else "运动态"
                    self.gate_label.setText(
                        f"当前状态：{state_text} | 纯 MI：三次眨眼双向切换"
                    )
                return
            if self.control_gate_state == "rest":
                self.gate_toggle_button.setText("手动切换：进入运动态")
            else:
                self.gate_toggle_button.setText("手动切换：回到静息态")

        def manual_toggle_control_gate(self) -> None:
            if self.config._combo_value(self.config.paradigm_combo) == "pure_mi_binary":
                self._sync_gate_toggle_button()
                return
            now = time.time()
            if self.control_gate_state == "rest":
                self.control_gate_state = "active"
                state_text = "运动控制态"
                reason = "manual_gate_switch_to_active"
                label = "手动切换：进入运动态，八分类动作输出已解锁"
            else:
                self.control_gate_state = "rest"
                state_text = "静息态"
                reason = "manual_gate_switch_to_rest"
                label = "手动切换：回到静息态，八分类动作输出已锁定"
            self.gate_hold_start = None
            # Start a new event phase after a manual gate change.
            self.pending_record_emg_counts = dict(self.last_window_emg_counts)
            self._set_runtime_phase(self.runtime_phase, now)
            self.eeg8_history.clear()
            text = f"当前状态：{state_text} | 门控：{label}"
            if hasattr(self, "gate_label"):
                self.gate_label.setText(text)
            if hasattr(self, "decode_result_label"):
                self.decode_result_label.setText(f"实时解码结果：{label}")
            self._update_fullscreen_state(text)
            self._update_gate_counter(None, 0.0)
            self._sync_gate_toggle_button()
            decision = Decision(now, "gate_switch", 1.0, 0.0, None, False, reason, "gate_switch", {})
            self._append_decode_result(decision, "-", label, self._window_emg_stats_text())
            self.append_log(label)

        def load_model(self) -> None:
            self.model = None
            self.runtime_session = None
            self.controller = None
            if not self.use_model_check.isChecked():
                QtWidgets.QMessageBox.warning(self, "未勾选模型", "请勾选模型/权重后再启动 VR 实时测试。")
                return
            cfg = self.config.realtime_config()
            model_path = cfg["model_path"]
            if not model_path or not os.path.exists(model_path):
                QtWidgets.QMessageBox.warning(self, "模型不存在", f"找不到模型/权重：{model_path}")
                return
            try:
                self.model = EEG8Model.load(model_path)
                self._validate_eeg8_model()
                preview_session = build_runtime_session_config(
                    cfg,
                    self.model,
                    use_config_thresholds=self.use_weight_check.isChecked(),
                )
                runtime_strategy = preview_session.runtime_strategy
                self.active_runtime_strategy = runtime_strategy
                self.model.decoder_config = preview_session.decoder_config
                apply_runtime_control_config(self.model, preview_session.decoder_config)
                self.masseter_template_bank = None
                template_path = preview_session.decoder_config.masseter_template_path
                if template_path:
                    try:
                        template_bank = load_masseter_template_bank(template_path)
                    except Exception as exc:
                        self.append_log(f"NCC 肌电模板加载失败；NCC 映射将安全拒绝，不回退阈值：{exc}")
                    else:
                        # Window 1 uses NCC for mapping; window 2 only counts jaw events.
                        validate_eeg8_ncc_confirmation_timing(
                            preview_session.runtime_strategy,
                            preview_session.decoder_config,
                            template_bank,
                            classification_window_seconds=preview_session.signal_window_seconds,
                        )
                        self.masseter_template_bank = template_bank
                        if template_bank is not None:
                            self.append_log(f"NCC 肌电模板已加载：{template_path}")
                self.grip_controller = GripStateController(
                    self.robot,
                    target=preview_session.target,
                    duration_seconds=preview_session.decoder_config.command_duration_seconds,
                )
                self._reset_eeg8_decoder_state()
                confirm_window = preview_session.confirmation_window_seconds
                if is_eeg_only_runtime_strategy(runtime_strategy):
                    self.window_label_base_text = (
                        f"分类窗口：{self.model.signal_config.window_seconds:.1f} s | "
                        "EEG-only：通过安全阈值后直接执行"
                    )
                else:
                    self.window_label_base_text = (
                        f"窗口1：EEG+EMG {self.model.signal_config.window_seconds:.1f} s | "
                        f"窗口2：咬肌确认 {confirm_window:.1f} s"
                    )
                self._update_window_countdown(None)
                mode_text = (
                    f"{self.model.signal_config.window_seconds:.1f}s EEG-only 直接执行"
                    if is_eeg_only_runtime_strategy(runtime_strategy)
                    else f"{self.model.signal_config.window_seconds:.1f}s解码 + {confirm_window:.1f}s确认"
                )
                self.status_label.setText(f"状态：八分类模型已加载 | {mode_text}")
                self.append_log(f"八分类模型已加载：{model_path}（预览配置：{preview_session.decoder_config_source}）")
                if is_eeg_only_runtime_strategy(runtime_strategy):
                    self.append_log(
                        f"在线策略：{runtime_strategy}；{self.model.signal_config.window_seconds:.1f}s EEG "
                        "仅检查当前窗口的置信度和边距，随后发送一次有时长限制的命令，不读取 AUX。"
                    )
                else:
                    self.append_log(
                        f"在线策略：{runtime_strategy}；窗口1为单个 {self.model.signal_config.window_seconds:.1f}s "
                        f"EEG+EMG 联合分类，随后进入窗口2（{confirm_window:.1f}s 咬肌计数确认）。"
                    )
                self.append_log("C1左移 C2右移 C3前移 C4后移 C5上移 C6下移 C7抓取 C8释放")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
                self.append_log(f"加载失败：{exc}")

        def _validate_eeg8_model(self) -> None:
            if self.model is None or self.model.eeg_classifier is None or self.model.eeg_features is None:
                raise RuntimeError("当前模型没有 EEG 分类器，请先在模型训练页训练八分类 EEG 模型。")
            classes = {int(c) for c in np.asarray(self.model.eeg_classifier.classes_).reshape(-1)}
            runtime_cfg = self.config.realtime_config()
            selected = str(runtime_cfg.get("paradigm_key", "eeg8"))
            if selected == "pure_mi_binary":
                expected = {0, 1}
            elif selected == "custom_mi":
                definitions = list(runtime_cfg.get("class_definitions", []) or [])
                expected = set(range(len(definitions)))
                strategy = dict(getattr(self.model, "training_strategy_", {}) or {})
                saved_names = list(strategy.get("class_names", []) or [])
                current_names = [str(item.get("class_name", "")).strip() for item in definitions]
                if saved_names != current_names:
                    raise RuntimeError("自定义类别名称与模型训练时不一致，请恢复类别名称或重新训练。")
            else:
                expected = set(range(8))
            model_paradigm = str(getattr(self.model, "paradigm_key", "eeg8") or "eeg8")
            if classes != expected or model_paradigm != selected:
                raise RuntimeError(
                    f"模型与所选范式不匹配：选择={selected}，模型={model_paradigm}，"
                    f"分类器类别={sorted(classes)}。请使用所选范式采集数据重新训练。"
                )

        def _reset_eeg8_decoder_state(self) -> None:
            self.eeg8_history = deque(maxlen=1)
            self.pure_mi_decoder.clear()
            self.pure_mi_confidences.clear()
            self._refresh_pure_mi_counter(event_text="等待第 1 个二分类窗口")
            self.last_command_time = 0.0
            pure_mi = (
                self.config._combo_value(self.config.paradigm_combo)
                == "pure_mi_binary"
            )
            self.control_gate_state = "rest"
            self.pending_confirmation = None
            self.continuous_move_intent = None
            self.continuous_move_started_at = 0.0
            self.confirmation_cancel_count = 0
            self.gate_hold_start = None
            self.last_window_emg_counts = {"blink": 0, "left_jaw": 0, "right_jaw": 0}
            self.aux_baseline_rms = None
            self.aux_baseline_values = []
            self.aux_baseline_until = 0.0
            self._set_runtime_phase("baseline", 0.0)
            if hasattr(self, "gate_label"):
                self.gate_label.setText(
                    "当前状态：静息态 | 纯 MI：只检测眼电，三次眨眼进入运动态"
                    if pure_mi
                    else "当前状态：静息态 | 门控：眨眼三次切换"
                )
            self._sync_gate_toggle_button()
            self._update_gate_counter(None, 0.0)

        def _eeg8_proba_dict(self, probabilities: np.ndarray) -> Dict[str, float]:
            return {
                f"C{idx + 1}_{CLASS_ARM_LABELS.get(idx + 1, '')}": float(probabilities[idx])
                for idx in range(len(probabilities))
            }

        def _aux_group_score(self, aux: Optional[np.ndarray], indices: Sequence[int]) -> float:
            if aux is None or np.size(aux) == 0:
                return 0.0
            try:
                aux_arr = np.asarray(aux, dtype=np.float64)
                if aux_arr.ndim == 1:
                    aux_arr = aux_arr.reshape(1, -1)
                valid = [idx for idx in indices if 0 <= int(idx) < aux_arr.shape[0]]
                if not valid:
                    return 0.0
                baseline = getattr(self, "aux_baseline_rms", None)
                if baseline is not None:
                    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
                    if base.size >= aux_arr.shape[0]:
                        centered = aux_arr - np.median(aux_arr, axis=1, keepdims=True)
                        rms = np.sqrt(np.mean(np.square(centered), axis=1))
                        ratio = rms / np.maximum(base[: aux_arr.shape[0]], 1e-6)
                        values = [float(ratio[int(idx)]) for idx in valid if int(idx) < ratio.size and np.isfinite(ratio[int(idx)])]
                        return max(values) if values else 0.0
                aux_level = float(np.percentile(np.abs(aux_arr[valid, :]), 95))
                center = float(getattr(self.model, "aux_abs_center_", 0.0)) if self.model is not None else 0.0
                mad = max(float(getattr(self.model, "aux_abs_mad_", 1.0)) if self.model is not None else 1.0, 1e-6)
                return max(0.0, (aux_level - center) / mad)
            except Exception:
                return 0.0

        def _recent_aux_detection_window(self, aux: Optional[np.ndarray], seconds: float = 0.18) -> Optional[np.ndarray]:
            if aux is None or np.size(aux) == 0:
                return None
            arr = np.asarray(aux, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] == 0:
                return None
            fs = (
                float(self.runtime_session.signal_fs)
                if self.runtime_session is not None
                else float(getattr(self.model.signal_config, "fs", 250.0) if self.model is not None else 250.0)
            )
            rows = max(8, int(round(max(0.05, float(seconds)) * fs)))
            return arr[:, -min(rows, arr.shape[1]) :]

        def _update_aux_baseline(self, aux: Optional[np.ndarray], now: float) -> bool:
            if aux is None or np.size(aux) == 0:
                return False
            try:
                aux_arr = np.asarray(aux, dtype=np.float64)
                if aux_arr.ndim == 1:
                    aux_arr = aux_arr.reshape(1, -1)
                centered = aux_arr - np.median(aux_arr, axis=1, keepdims=True)
                rms = np.sqrt(np.mean(np.square(centered), axis=1))
            except Exception:
                return False
            if self.aux_baseline_rms is None:
                if self.aux_baseline_until <= 0.0:
                    self.aux_baseline_until = float(now) + 2.0
                self.aux_baseline_values.append(rms)
                if now < self.aux_baseline_until:
                    return False
                baseline = np.median(np.vstack(self.aux_baseline_values), axis=0)
                self.aux_baseline_rms = np.maximum(baseline, 1e-6)
                self.append_log(f"AUX 基线完成 RMS={np.array2string(self.aux_baseline_rms, precision=3)}")
                self._begin_next_eeg_window(float(now))
                return False
            return True

        def _set_runtime_phase(self, phase: str, started_at: float) -> None:
            self.runtime_phase = str(phase)
            self.runtime_phase_started_at = float(started_at)
            blink_thr, bite_thr, side_ratio = self._emg_thresholds()
            fs = (
                float(self.runtime_session.signal_fs)
                if self.runtime_session is not None
                else float(getattr(self.model.signal_config, "fs", 250.0) if self.model is not None else 250.0)
            )
            # Reset counters and source slicing at phase boundaries.
            self.phase_emg_events = PhaseEmgEventEngine(
                fs=fs,
                baseline_rms=self.aux_baseline_rms,
                blink_ratio=blink_thr,
                bite_ratio=bite_thr,
                side_ratio=side_ratio,
                rms_window_seconds=0.18,
                rms_hop_seconds=0.09,
                blink_min_gap_seconds=0.18,
                masseter_min_gap_seconds=0.25,
                # Window 1, 2 and each window 3 segment own independent counts.
                event_window_seconds={},
            )
            snapshot = self.phase_emg_events.reset(self.runtime_phase, self.runtime_phase_started_at)
            self._set_last_window_emg_counts(
                snapshot.blink, snapshot.left_jaw, snapshot.right_jaw
            )
            self._update_gate_counter(None, 0.0)

        def _emg_count_targets(self) -> Dict[str, int]:
            if self.runtime_session is not None:
                return {
                    "blink": self.runtime_session.aux_blink_count,
                    "left_jaw": self.runtime_session.aux_left_count,
                    "right_jaw": self.runtime_session.aux_right_count,
                }
            cfg = self.config.realtime_config()
            return {
                "blink": max(1, int(cfg.get("aux_blink_count", 3))),
                "left_jaw": max(1, int(cfg.get("aux_left_count", 1))),
                "right_jaw": max(1, int(cfg.get("aux_right_count", 1))),
            }

        def _emg_thresholds(self) -> Tuple[float, float, float]:
            if self.runtime_session is not None:
                return (
                    self.runtime_session.aux_blink_ratio,
                    self.runtime_session.aux_bite_ratio,
                    self.runtime_session.aux_side_ratio,
                )
            cfg = self.config.realtime_config()
            blink_thr = max(1.0, float(cfg.get("aux_blink_ratio", 4.0)))
            bite_thr = max(1.0, float(cfg.get("aux_bite_ratio", 3.0)))
            side_ratio = max(1.0, float(cfg.get("aux_side_ratio", 1.35)))
            return blink_thr, bite_thr, side_ratio

        def _update_gate_counter(self, aux: Optional[np.ndarray], fist_elapsed: float = 0.0) -> None:
            if not hasattr(self, "gate_counter_label"):
                return
            aux_connected = (aux is not None and np.size(aux) > 0) or self.source is not None
            aux_text = "已接入" if aux_connected else "未接入"
            counts = dict(getattr(
                self,
                "last_window_emg_counts",
                {"blink": 0, "left_jaw": 0, "right_jaw": 0},
            ))
            text = (
                f"当前时间段计数：眨眼 {int(counts.get('blink', 0))} | "
                f"左咬肌 {int(counts.get('left_jaw', 0))} | "
                f"右咬肌 {int(counts.get('right_jaw', 0))} | "
                f"AUX：{aux_text}"
            )
            self.gate_counter_label.setText(text)

        def _on_arm_motion_state(self, state: str) -> None:
            if self.continuous_move_intent is None:
                return
            if str(state) not in {"Blocked: collision", "Blocked: boundary"}:
                return
            reason = "检测到碰撞" if str(state) == "Blocked: collision" else "到达场景边界"
            self.continuous_move_intent = None
            self.continuous_move_started_at = 0.0
            now = float(self.last_aux_timestamp or time.time())
            self._begin_next_eeg_window(now)
            self.gate_label.setText(f"连续移动已停止：{reason} | 进入 EEG 解码窗口")
            self._set_current_result_card("rejected", "连续移动已停止", reason)
            self.append_log(f"连续移动停止：{reason}")

        def _set_last_window_emg_counts(
            self,
            blink_count: Optional[int] = None,
            left_count: Optional[int] = None,
            right_count: Optional[int] = None,
        ) -> None:
            counts = dict(getattr(self, "last_window_emg_counts", {"blink": 0, "left_jaw": 0, "right_jaw": 0}))
            if blink_count is not None:
                counts["blink"] = int(blink_count)
            if left_count is not None:
                counts["left_jaw"] = int(left_count)
            if right_count is not None:
                counts["right_jaw"] = int(right_count)
            self.last_window_emg_counts = counts

        def _active_emg_snapshot(self) -> EmgEventSnapshot:
            engine = getattr(self, "phase_emg_events", None)
            if engine is None:
                return EmgEventSnapshot(str(getattr(self, "runtime_phase", "idle")), reason="engine_not_ready")
            return engine.snapshot()

        def _update_phase_emg_events(self, aux: Optional[np.ndarray], now: float) -> EmgEventSnapshot:
            """Update the one phase-local AUX counter and mirror its snapshot to UI state."""
            engine = getattr(self, "phase_emg_events", None)
            if engine is None:
                self._set_runtime_phase(str(getattr(self, "runtime_phase", "idle")), float(now))
                engine = self.phase_emg_events
            # Refresh the baseline after calibration.
            engine.set_baseline(self.aux_baseline_rms)
            snapshot = engine.update(aux, float(now))
            self._set_last_window_emg_counts(
                snapshot.blink, snapshot.left_jaw, snapshot.right_jaw
            )
            return snapshot

        def _window_emg_stats_text(self) -> str:
            counts = dict(getattr(self, "last_window_emg_counts", {"blink": 0, "left_jaw": 0, "right_jaw": 0}))
            emg_text = self._emg_stats_from_counts(counts)
            pure_mi_text = str(getattr(self, "pure_mi_counter_text", "") or "")
            return f"{pure_mi_text} | {emg_text}" if pure_mi_text else emg_text

        @staticmethod
        def _emg_stats_from_counts(counts: Dict[str, int]) -> str:
            return (
                f"肌电统计：眨眼 {int(counts.get('blink', 0))} "
                f"左咬肌 {int(counts.get('left_jaw', 0))} "
                f"右咬肌 {int(counts.get('right_jaw', 0))}"
            )

        def _set_current_result_card(self, phase: str, title: str, detail: str = "") -> None:
            if not hasattr(self, "current_result_label"):
                return
            colors = {
                "collect": ("#f8fafc", "#cbd5e1", "#0f172a"),
                "decode": ("#eef6ff", "#60a5fa", "#0f3a63"),
                "confirm": ("#fff1cc", "#f59e0b", "#5f370e"),
                "window1": ("#e8f4ff", "#3b82f6", "#123f6d"),
                "window2": ("#fff1cc", "#f59e0b", "#5f370e"),
                "window3": ("#f3e8ff", "#a855f7", "#581c87"),
                "accepted": ("#dcfce7", "#22c55e", "#14532d"),
                "rejected": ("#fee2e2", "#ef4444", "#7f1d1d"),
            }
            bg, border, fg = colors.get(str(phase), colors["collect"])
            self.result_card.setStyleSheet(
                "QFrame#vrResultCard { "
                f"background: {bg}; border: 2px solid {border}; border-radius: 7px; "
                "} "
                "QLabel#vrResultText { font-size: 22px; font-weight: 700; }"
            )
            self.current_result_label.setStyleSheet(f"color: {fg};")
            self.current_result_label.setText(title)
            self.current_result_detail.setText(detail)

        def _update_window_countdown(self, remaining_seconds: Optional[float] = None) -> None:
            if not hasattr(self, "window_label"):
                return
            if self.continuous_move_intent is not None:
                cfg = self._runtime_decoder_config()
                if self._is_pure_mi_runtime():
                    move_label = {
                        "left": "左移", "right": "右移", "forward": "前移",
                        "backward": "后移", "up": "上移", "down": "下移",
                    }.get(self.continuous_move_intent, self.continuous_move_intent)
                    self.window_label.setText(
                        f"纯 MI 运动状态：持续{move_label} | 双侧咬肌迅速停止"
                    )
                    self._set_current_result_card(
                        "window3", f"持续{move_label}", "等待双侧咬肌停止"
                    )
                    return
                window = max(0.5, float(getattr(cfg, "continuous_blink_window_seconds", 2.0)))
                required_blinks = max(1, int(getattr(cfg, "continuous_blink_count", 3) or 3))
                phase_now = float(getattr(self, "last_aux_timestamp", 0.0) or 0.0)
                phase_started = float(getattr(self, "runtime_phase_started_at", 0.0) or 0.0)
                elapsed = max(0.0, phase_now - phase_started) if phase_now > 0.0 and phase_started > 0.0 else 0.0
                remaining = max(0.0, window - elapsed)
                move_label = {
                    "left": "左移", "right": "右移", "forward": "前移",
                    "backward": "后移", "up": "上移", "down": "下移",
                }.get(self.continuous_move_intent, self.continuous_move_intent)
                self.window_label.setText(
                    f"窗口3：连续{self.continuous_move_intent} | 本 {window:.1f}s 窗口计数 | "
                    f"眨眼 {required_blinks} 次停止"
                )
                self._set_current_result_card(
                    "window3",
                    f"运动方向：{move_label}",
                    f"剩余倒计时：{remaining:.1f} s",
                )
                return
            if self.pending_confirmation is not None:
                cfg = self._runtime_decoder_config()
                confirm_window = max(0.2, float(getattr(cfg, "confirmation_window_seconds", 1.0) or 1.0))
                required_count = max(1, int(getattr(cfg, "confirmation_masseter_count", 1) or 1))
                pending = dict(self.pending_confirmation or {})
                class_id = int(pending.get("class_id", 0) or 0)
                intent = str(pending.get("intent", "rest"))
                remaining = confirm_window if remaining_seconds is None else max(0.0, float(remaining_seconds))
                self.window_label.setText(
                    f"窗口2：咬肌确认 {confirm_window:.1f} s | 剩余：{remaining:.1f} s | "
                    f"当前分类结果：C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)}"
                )
                self._set_current_result_card(
                    "window2",
                    f"预测动作：{CLASS_ARM_LABELS.get(class_id, intent)}",
                    f"剩余倒计时：{remaining:.1f} s",
                )
                return
            eeg_window = (
                self.runtime_session.signal_window_seconds
                if self.runtime_session is not None
                else (float(getattr(self.model.signal_config, "window_seconds", 2.0)) if self.model is not None else 2.0)
            )
            base = str(getattr(self, "window_label_base_text", f"EEG 采集：{eeg_window:.1f} s"))
            if remaining_seconds is None:
                self.window_label.setText(base)
                self._set_current_result_card(
                    "window1", "模型解码中：", f"剩余倒计时：{eeg_window:.1f} s"
                )
                return
            self.window_label.setText(f"{base} | 本窗口倒计时：{max(0.0, float(remaining_seconds)):.1f} s")
            self._set_current_result_card(
                "window1",
                "模型解码中：",
                f"剩余倒计时：{max(0.0, float(remaining_seconds)):.1f} s",
            )

        def _update_live_emg_counts_from_source(self) -> None:
            if self.source is None or not hasattr(self.source, "latest_aux_window"):
                return
            try:
                if self.runtime_phase_started_at > 0.0 and hasattr(self.source, "latest_aux_window_since"):
                    aux, now = self.source.latest_aux_window_since(self.runtime_phase_started_at)
                else:
                    aux, now = self.source.latest_aux_window(self._event_memory_seconds())
            except Exception:
                return
            if aux is None or np.size(aux) == 0:
                self._update_gate_counter(None, 0.0)
                return
            self.last_aux_timestamp = float(now)
            if not self._update_aux_baseline(aux, now):
                remaining = max(0.0, self.aux_baseline_until - float(now))
                text = f"实时解码结果：AUX 基线采集中 {remaining:.1f}s | {self._window_emg_stats_text()}"
                if hasattr(self, "decode_result_label"):
                    self.decode_result_label.setText(text)
                self._update_gate_counter(aux, 0.0)
                return
            self._update_phase_emg_events(aux, now)
            self._update_gate_counter(aux, 0.0)
            if hasattr(self, "decode_result_label") and self.timer.isActive():
                window_seconds = (
                    self.runtime_session.signal_window_seconds
                    if self.runtime_session is not None
                    else (float(getattr(self.model.signal_config, "window_seconds", 2.0)) if self.model is not None else 2.0)
                )
                if self.runtime_phase == "confirm":
                    confirm_seconds = float(getattr(self._runtime_decoder_config(), "confirmation_window_seconds", 1.0))
                    phase_text = f"肌电确认 {confirm_seconds:.1f}s"
                elif self.runtime_phase == "continuous_move":
                    phase_text = f"连续{self.continuous_move_intent or '移动'}中"
                else:
                    phase_text = f"窗口1 EEG+EMG 采集 {window_seconds:.1f}s"
                self.decode_result_label.setText(f"实时解码结果：{phase_text} | {self._window_emg_stats_text()}")

        def _event_memory_seconds(self) -> float:
            """Fallback source slice length when a source lacks phase timestamps.

            The normal LSL source implements ``latest_aux_window_since``. This
            fallback only defines the physical size of a fixed phase window;
            counter state still belongs solely to ``PhaseEmgEventEngine``.
            """
            session = self.runtime_session
            if self.runtime_phase == "continuous_move":
                if session is not None:
                    return session.continuous_blink_window_seconds
                runtime_cfg = self.config.realtime_config()
                return max(0.5, float(runtime_cfg.get("continuous_blink_window_seconds", 2.0)))
            if self.runtime_phase == "confirm":
                if session is not None:
                    return session.confirmation_window_seconds
                runtime_cfg = self.config.realtime_config()
                return max(0.2, float(runtime_cfg.get("confirm_window_seconds", 1.0)))
            if session is not None:
                return max(0.5, session.signal_window_seconds)
            if self.model is not None:
                return max(0.5, float(getattr(self.model.signal_config, "window_seconds", 2.0)))
            runtime_cfg = self.config.realtime_config()
            return max(0.5, float(runtime_cfg.get("decode_window_seconds", 2.0)))

        def _update_jaw_event_counts(self, aux: Optional[np.ndarray], now: float) -> Tuple[int, int, int]:
            snapshot = self._update_phase_emg_events(aux, now)
            return snapshot.left_jaw, snapshot.right_jaw, snapshot.both_jaw

        def _emg_assist_text(self, aux: Optional[np.ndarray]) -> str:
            left_jaw = self._aux_group_score(aux, [2])
            right_jaw = self._aux_group_score(aux, [3])
            _blink_thr, threshold, _side_ratio = self._emg_thresholds()
            left_on = left_jaw >= threshold
            right_on = right_jaw >= threshold
            if left_on and right_on:
                return "肌电辅助：11 候选 C7/C8"
            if left_on:
                return "肌电辅助：10 候选 C5/C6"
            if right_on:
                return "肌电辅助：01 候选 C1/C2"
            return "肌电辅助：00 候选 C3/C4"

        def _gate_blink_detected(self, aux: Optional[np.ndarray], now: float, _true_label: Optional[str]) -> int:
            """Count only real AUX blink events in the online runtime."""
            snapshot = self._update_phase_emg_events(aux, now)
            return snapshot.blink

        def _is_both_fist_gate(self, class_id: Optional[int], true_label: Optional[str] = None) -> bool:
            text = str(true_label or "").strip().lower()
            if text and any(key in text for key in ["双手握拳", "both_fist", "both-fist", "both hands", "both_hands"]):
                return True
            if class_id is None:
                return False
            try:
                cid = int(class_id)
            except Exception:
                return False
            label = str(CLASS_NAMES.get(cid, "")).lower()
            return cid == 7 or "双手握拳" in label or "both" in label

        def _rest_gate_decision(self, aux: Optional[np.ndarray], now: float, true_label: Optional[str] = None) -> Optional[Decision]:
            if self.control_gate_state != "rest":
                return None
            blink_count = self._gate_blink_detected(aux, now, true_label)
            snapshot = self._active_emg_snapshot()
            left_count, right_count = snapshot.left_jaw, snapshot.right_jaw
            targets = self._emg_count_targets()
            self._set_last_window_emg_counts(blink_count, left_count, right_count)
            self._update_gate_counter(aux, 0.0)
            if blink_count >= targets["blink"]:
                self.control_gate_state = "active"
                self.gate_hold_start = None
                self.phase_emg_events.clear_events("blink")
                self._set_last_window_emg_counts(0, left_count, right_count)
                if hasattr(self, "gate_label"):
                    self.gate_label.setText("当前状态：运动控制态 | 门控：连续眨眼三次完成，八分类动作输出已解锁")
                self._update_fullscreen_state("当前状态：运动控制态 | 门控：连续眨眼三次已完成")
                return Decision(now, "gate_switch", 1.0, 1.0, None, False, "连续眨眼三次，切换到控制态", "gate_switch", {})
            if hasattr(self, "gate_label"):
                self.gate_label.setText(f"当前状态：静息态 | 门控：眨眼 {blink_count}/{targets['blink']}")
            self._update_fullscreen_state(f"当前状态：静息态 | 门控：连续眨眼 {blink_count}/{targets['blink']}")
            return None

        def _both_fist_gate_decision(self, class_id: int, confidence: float, now: float, true_label: Optional[str] = None, aux: Optional[np.ndarray] = None) -> Optional[Decision]:
            if self.control_gate_state != "rest":
                return None
            if not self._is_both_fist_gate(class_id, true_label):
                self.gate_hold_start = None
                text = f"当前状态：静息态 | 门控：等待眨眼三次"
                self.gate_label.setText(text)
                self._update_fullscreen_state(text)
                return Decision(now, "rest", 1.0, 0.0, None, False, "rest_gate_idle_no_8class_output", "rest_gate", {})
            if self.gate_hold_start is None:
                self.gate_hold_start = now
            step_seconds = (
                self.runtime_session.signal_step_seconds
                if self.runtime_session is not None
                else float(self.model.signal_config.step_seconds if self.model is not None else 0.5)
            )
            elapsed = min(self.gate_hold_seconds, max(0.0, now - self.gate_hold_start) + step_seconds)
            if elapsed >= self.gate_hold_seconds:
                self.control_gate_state = "active"
                self.gate_hold_start = None
                self._update_gate_counter(aux, self.gate_hold_seconds)
                text = "当前状态：运动控制态 | 门控：眨眼三次完成，八分类动作输出已解锁"
                self.gate_label.setText(text)
                self._update_fullscreen_state(text)
                return Decision(now, "gate_switch", 1.0, 1.0, None, False, "眨眼三次，切换到控制态", "gate_switch", {})
            self._update_gate_counter(aux, elapsed)
            text = f"当前状态：静息态 | 门控：等待眨眼三次 | conf={confidence:.2f}"
            self.gate_label.setText(text)
            self._update_fullscreen_state(text)
            return Decision(now, "rest", confidence, 0.0, None, False, "both_fist_gate_holding", "rest_gate", {})


        def _eeg8_display(self, decision: Decision) -> str:
            if str(decision.state) == "MI1":
                return "左手(1)"
            if str(decision.state) == "MI0":
                return "右手(0)"
            if str(decision.state).startswith("custom_mi:"):
                return str(decision.intent)
            if not str(decision.state).startswith("C"):
                return decision.intent
            try:
                class_id = int(str(decision.state)[1:])
            except Exception:
                return decision.intent
            return f"C{class_id} {CLASS_ARM_LABELS.get(class_id, decision.intent)}"

        def _pure_emg_enabled(self) -> bool:
            return False

        def _is_eeg_only_runtime(self) -> bool:
            """Use the strategy frozen when this online session was started."""
            if self.runtime_session is not None:
                return not self.runtime_session.requires_aux
            return is_eeg_only_runtime_strategy(self.active_runtime_strategy)

        def _runtime_decoder_config(self):
            """Return the immutable session decoder while online is active."""
            if self.runtime_session is not None:
                return self.runtime_session.decoder_config
            if self.model is not None:
                return self.model.decoder_config
            return self.config.decoder_config()

        def _runtime_target(self) -> str:
            if self.runtime_session is not None:
                return self.runtime_session.target
            return self.config.realtime_config()["target"]

        def _runtime_continuous_move_speed(self) -> float:
            if self.runtime_session is not None:
                return self.runtime_session.continuous_move_speed
            return max(0.1, float(self.config.realtime_config().get("continuous_move_speed", 0.5)))

        def _build_motion_command(
            self,
            intent: str,
            confidence: float,
            now: float,
            cfg: Optional[object] = None,
            continuous_override: Optional[bool] = None,
        ) -> RobotCommand:
            runtime_cfg = cfg or self._runtime_decoder_config()
            self.sequence += 1
            continuous = intent in CONTINUOUS_MOVE_VECTORS
            if continuous_override is not None:
                continuous = bool(continuous_override) and intent in CONTINUOUS_MOVE_VECTORS
            return RobotCommand(
                name=ACTION_COMMANDS[intent],
                intent=intent,
                target=self._runtime_target(),
                confidence=float(confidence),
                sequence_id=self.sequence,
                timestamp=float(now),
                duration_seconds=float(runtime_cfg.command_duration_seconds),
                continuous=continuous,
                move_speed=self._runtime_continuous_move_speed() if continuous else 0.0,
                velocity=CONTINUOUS_MOVE_VECTORS.get(intent, (0.0, 0.0, 0.0)) if continuous else (0.0, 0.0, 0.0),
            )

        def _send_vr_command(self, intent: str, confidence: float, now: float, cfg: Optional[DecoderConfig] = None) -> Tuple[Optional[str], bool, str]:
            runtime_cfg = cfg or self._runtime_decoder_config()
            if now - self.last_command_time < float(runtime_cfg.refractory_seconds):
                return None, False, "refractory"
            if intent in CONTINUOUS_MOVE_VECTORS and not self._translation_can_start(
                intent
            ):
                return None, False, "translation_soft_limit"
            command = self._build_motion_command(intent, confidence, now, runtime_cfg)
            self.robot.execute(command)
            if command.continuous:
                self.continuous_move_intent = intent
                self.continuous_move_started_at = float(now)
                self._start_translation_limit_tracking(intent)
                self._set_runtime_phase("continuous_move", float(now))
            self.last_command_time = now
            return command.name, True, "accepted"

        def _pure_emg_decision(self, aux: Optional[np.ndarray], now: float, true_label: Optional[str] = None) -> Decision:
            if not self._update_aux_baseline(aux, now):
                self._update_gate_counter(aux, 0.0)
                self.last_emg_assist_text = "肌电触发：AUX 基线采集中"
                return Decision(now, "pure_emg_baseline", 0.0, 0.0, None, False, "aux_baseline_collecting", "pure_emg", {})
            blink_count = self._gate_blink_detected(aux, now, true_label)
            snapshot = self._active_emg_snapshot()
            left_count, right_count, both_count = (
                snapshot.left_jaw,
                snapshot.right_jaw,
                snapshot.both_jaw,
            )
            targets = self._emg_count_targets()
            self._set_last_window_emg_counts(blink_count, left_count, right_count)
            self._update_gate_counter(aux, 0.0)
            left_jaw = self._aux_group_score(aux, [2])
            right_jaw = self._aux_group_score(aux, [3])
            _blink_thr, threshold, _side_ratio = self._emg_thresholds()
            intent: Optional[str] = None
            class_id: Optional[int] = None
            reason = "pure_emg_waiting"
            confidence = 0.0
            if blink_count >= targets["blink"]:
                intent = "forward"
                class_id = 3
                confidence = 1.0
                reason = "pure_emg_blink3_forward"
                self.phase_emg_events.clear_events("blink")
            elif left_count >= targets["left_jaw"] and right_count < targets["right_jaw"]:
                intent = "left"
                class_id = 1
                confidence = min(1.0, max(left_jaw, threshold) / max(threshold * 1.8, 1e-6))
                reason = "pure_emg_left_masseter"
                self.phase_emg_events.clear_events("left_jaw")
            elif right_count >= targets["right_jaw"] and left_count < targets["left_jaw"]:
                intent = "right"
                class_id = 2
                confidence = min(1.0, max(right_jaw, threshold) / max(threshold * 1.8, 1e-6))
                reason = "pure_emg_right_masseter"
                self.phase_emg_events.clear_events("right_jaw")
            if intent is None or class_id is None:
                text = (
                    f"当前状态：肌电直接控制停用 | 眨眼 {blink_count}/{targets['blink']} | "
                f"双咬肌 {both_count} | "
                    f"左咬 {left_count}/{targets['left_jaw']} | 右咬 {right_count}/{targets['right_jaw']}"
                )
                if hasattr(self, "gate_label"):
                    self.gate_label.setText(text)
                self._update_fullscreen_state(text)
                self.last_emg_assist_text = self._emg_assist_text(aux)
                return Decision(now, "pure_emg_wait", 0.0, 0.0, None, False, reason, "pure_emg", {})
            command_name, accepted, send_reason = self._send_vr_command(intent, confidence, now)
            if send_reason != "accepted":
                reason = f"{reason}_{send_reason}"
            text = f"当前状态：肌电直接控制停用 | 输出：C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)}"
            if hasattr(self, "gate_label"):
                self.gate_label.setText(text)
            self._update_fullscreen_state(text)
            self.last_emg_assist_text = self._emg_assist_text(aux)
            return Decision(
                now,
                intent,
                confidence,
                0.0,
                command_name,
                accepted,
                reason,
                f"C{class_id}",
                {f"C{class_id}_{CLASS_ARM_LABELS.get(class_id, '')}": confidence},
            )

        def _masseter_direct_decision(self, aux: Optional[np.ndarray], now: float) -> Optional[Decision]:
            return None

        def _masseter_gate_for_window(self, aux: Optional[np.ndarray]) -> MasseterGateResult:
            cfg = self._runtime_decoder_config()
            snapshot = self._active_emg_snapshot()
            if not bool(getattr(cfg, "masseter_gate_enabled", True)):
                gate = MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), "masseter_gate_disabled")
                self._set_last_window_emg_counts(snapshot.blink, 0, 0)
                self.last_emg_assist_text = "肌电辅助：已关闭左右咬肌候选门控"
                return gate
            if aux is None or np.size(aux) == 0:
                self._set_last_window_emg_counts(snapshot.blink, 0, 0)
                self.last_emg_assist_text = "肌电辅助：确认输入缺失，禁止执行"
                return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), "no_aux")
            try:
                aux_arr = np.asarray(aux, dtype=np.float64)
                if aux_arr.ndim == 1:
                    aux_arr = aux_arr.reshape(1, -1)
                if aux_arr.ndim != 2 or aux_arr.shape[0] < 4:
                    raise ValueError("missing_masseter_channels")
                if aux_arr.shape[1] < 8 or not np.all(np.isfinite(aux_arr)):
                    raise ValueError("invalid_masseter_aux")
            except (TypeError, ValueError) as exc:
                reason = str(exc) or "invalid_masseter_aux"
                self._set_last_window_emg_counts(snapshot.blink, 0, 0)
                self.last_emg_assist_text = "肌电辅助：确认输入无效，禁止执行"
                return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), reason)
            targets = self._emg_count_targets()
            left_count = int(snapshot.left_jaw)
            right_count = int(snapshot.right_jaw)
            both_count = int(left_count >= targets["left_jaw"] and right_count >= targets["right_jaw"])
            self._set_last_window_emg_counts(
                snapshot.blink, left_count, right_count
            )
            # UI and confirmation use the same event snapshot.
            counts = dict(self.last_window_emg_counts)
            left_count = int(counts["left_jaw"])
            right_count = int(counts["right_jaw"])
            both_count = int(
                left_count >= targets["left_jaw"] and right_count >= targets["right_jaw"]
            )
            both_reached = bool(both_count)
            side = masseter_binary_state(
                left_count,
                right_count,
                targets["left_jaw"],
                targets["right_jaw"],
                both_count=1 if both_reached else 0,
            )
            left_ratio = float(snapshot.left_peak_ratio)
            right_ratio = float(snapshot.right_peak_ratio)
            channel_gate = MasseterGateResult(
                side,
                left_count,
                right_count,
                both_count,
                left_ratio,
                right_ratio,
                masseter_candidate_indices(side),
                f"incremental_masseter_{side}",
            )
            detector_mode = str(getattr(cfg, "masseter_detector", "threshold") or "threshold").lower()
            if detector_mode == "threshold":
                gate = channel_gate
            else:
                ncc_gate = match_masseter_ncc(
                    # NCC uses the complete phase-local AUX window.
                    aux_arr,
                    fs=(
                        float(self.runtime_session.signal_fs)
                        if self.runtime_session is not None
                        else float(getattr(self.model.signal_config, "fs", 250) if self.model is not None else 250)
                    ),
                    template_bank=self.masseter_template_bank,
                    min_score=float(getattr(cfg, "masseter_ncc_min_score", 0.45)),
                    min_margin=float(getattr(cfg, "masseter_ncc_min_margin", 0.08)),
                )
                gate = resolve_gui_masseter_detector_gate(
                    detector_mode,
                    channel_gate,
                    ncc_gate,
                    dict(getattr(cfg, "masseter_state_detectors", {}) or {}),
                )
            self._update_gate_counter(aux, 0.0)
            if gate.side == "00":
                self.last_emg_assist_text = "肌电辅助：00 候选 C3/C4"
            elif gate.side == "01":
                self.last_emg_assist_text = "肌电辅助：01 候选 C1/C2"
            elif gate.side == "10":
                self.last_emg_assist_text = "肌电辅助：10 候选 C5/C6"
            elif gate.side == "11":
                self.last_emg_assist_text = "肌电辅助：11 候选 C7/C8"
            else:
                self.last_emg_assist_text = "肌电辅助：未形成有效二进制门控"
            return gate

        def _apply_masseter_gate(self, probabilities: np.ndarray, gate: MasseterGateResult) -> Tuple[np.ndarray, str]:
            proba = np.asarray(probabilities, dtype=np.float64).reshape(-1)
            if proba.size < 8:
                return proba, gate.reason
            cfg = self._runtime_decoder_config()
            weight = float(getattr(cfg, "masseter_outside_candidate_weight", 0.3))
            return apply_masseter_candidate_weight(proba, gate.candidates, weight), gate.reason

        def _select_masseter_mapped_class(self, probabilities: np.ndarray, gate: MasseterGateResult) -> Tuple[Optional[int], int, str]:
            cfg = self._runtime_decoder_config()
            return select_gui_eeg8_confirmation_candidate(
                probabilities,
                gate.candidates,
                gate.reason,
                float(getattr(cfg, "candidate_random_margin", 0.05)),
            )

        def _store_pending_confirmation(
            self,
            class_id: int,
            intent: str,
            confidence: float,
            margin: float,
            probabilities: np.ndarray,
            gate_reason: str,
            now: float,
        ) -> Decision:
            if self._is_eeg_only_runtime():
                cfg = self._runtime_decoder_config()
                return self._execute_eeg8_action(
                    class_id,
                    intent,
                    confidence,
                    margin,
                    self._eeg8_proba_dict(probabilities),
                    now,
                    cfg,
                    "eeg_only_direct_execute",
                )
            # Save EEG-window counters before entering confirmation.
            self.pending_record_emg_counts = dict(self.last_window_emg_counts)
            self._set_runtime_phase("confirm", float(now))
            self.pending_confirmation = {
                "class_id": int(class_id),
                "intent": str(intent),
                "confidence": float(confidence),
                "margin": float(margin),
                "probabilities": self._eeg8_proba_dict(probabilities),
                "gate_reason": str(gate_reason),
                "timestamp": float(now),
                # LSL slices samples; monotonic time controls the deadline.
                "confirmation_started_at": float(now),
                "confirmation_started_monotonic": time.monotonic(),
            }

        def _is_pure_mi_runtime(self) -> bool:
            return bool(
                self.model is not None
                and str(getattr(self.model, "paradigm_key", "eeg8") or "eeg8")
                == "pure_mi_binary"
            )

        def _is_custom_mi_runtime(self) -> bool:
            return bool(
                self.model is not None
                and str(getattr(self.model, "paradigm_key", "eeg8") or "eeg8")
                == "custom_mi"
            )

        def _handle_pure_mi_blink_switch(
            self, now: float
        ) -> Optional[Decision]:
            """Use three blinks as a bidirectional rest/motion state switch."""
            snapshot = self._active_emg_snapshot()
            required = self._emg_count_targets()["blink"]
            blink_count = int(snapshot.blink)
            if self.control_gate_state == "rest":
                self._set_last_window_emg_counts(blink_count, 0, 0)
            else:
                self._set_last_window_emg_counts(
                    blink_count, snapshot.left_jaw, snapshot.right_jaw
                )
            self._update_gate_counter(None, 0.0)
            if blink_count < required:
                if self.control_gate_state == "rest":
                    self.gate_label.setText(
                        f"当前状态：静息态 | 只检测眼电 | 眨眼 {blink_count}/{required}"
                    )
                return None

            entering_active = self.control_gate_state == "rest"
            self.control_gate_state = "active" if entering_active else "rest"
            self.pure_mi_decoder.clear()
            self.pure_mi_confidences.clear()
            self.phase_emg_events.clear_events()
            self._refresh_pure_mi_counter(
                event_text="三次眨眼：进入运动态" if entering_active else "三次眨眼：进入静息态"
            )
            if not entering_active and self.continuous_move_intent is not None:
                self.bus.command_emitted.emit(
                    {"cmd": "stop_motion", "intent": "stop_motion"}
                )
                self.continuous_move_intent = None
                self.continuous_move_started_at = 0.0
            self._begin_next_eeg_window(now)
            text = (
                "当前状态：运动态 | 三次眨眼切换完成，开始纯 MI 解码"
                if entering_active
                else "当前状态：静息态 | 三次眨眼切换完成，只检测眼电"
            )
            self.gate_label.setText(text)
            self._update_fullscreen_state(text)
            return Decision(
                now,
                "gate_switch",
                1.0,
                1.0,
                None,
                False,
                "pure_mi_blink3_to_active"
                if entering_active
                else "pure_mi_blink3_to_rest",
                "gate_switch",
                {},
            )

        def _refresh_pure_mi_counter(
            self,
            event_text: str = "",
        ) -> None:
            selected = (
                self.config._combo_value(self.config.paradigm_combo)
                == "pure_mi_binary"
            )
            if not selected:
                self.pure_mi_counter_text = ""
                self._update_gate_counter(None, 0.0)
                return
            code = self.pure_mi_decoder.completed_code
            count = len(code)
            event = f" | {event_text}" if event_text else ""
            self.pure_mi_counter_text = (
                f"纯 MI：已记录 {count}，当前编码 {code or '空'}"
                f"{event}"
            )
            self._update_gate_counter(None, 0.0)

        def _update_pure_mi_intent(
            self, eeg: np.ndarray, aux: Optional[np.ndarray], now: float
        ) -> Decision:
            """Decode one binary MI window while reusing the existing output path."""
            if self.model is None:
                raise RuntimeError("纯 MI 模型未加载")
            cfg = self._runtime_decoder_config()
            self._update_phase_emg_events(aux, now)
            gate_decision = self._handle_pure_mi_blink_switch(now)
            if gate_decision is not None:
                return gate_decision
            if self.control_gate_state == "rest":
                return Decision(
                    now,
                    "rest",
                    0.0,
                    0.0,
                    None,
                    False,
                    "pure_mi_rest_eye_only",
                    "rest_gate",
                    {},
                )
            probabilities = np.asarray(self.model.predict_eeg_proba(eeg)[0], dtype=np.float64)
            if probabilities.size < 2:
                raise RuntimeError(f"纯 MI 模型输出类别数不足 2：{probabilities.size}")
            selected_idx = int(np.argmax(probabilities[:2]))
            second_idx = 1 - selected_idx
            confidence = float(probabilities[selected_idx])
            margin = abs(float(probabilities[selected_idx] - probabilities[second_idx]))
            bit = "1" if selected_idx == 0 else "0"
            proba_dict = {"MI_1_左手": float(probabilities[0]), "MI_0_右手": float(probabilities[1])}
            snapshot = self._active_emg_snapshot()
            left = int(snapshot.left_jaw) > 0
            right = int(snapshot.right_jaw) > 0
            if left or right:
                return self._handle_pure_mi_masseter_now(
                    left, right, now, confidence, margin, proba_dict
                )

            cancel_previous_three = len(self.pure_mi_decoder.completed_code) == 3
            code = self.pure_mi_decoder.feed_completed_window(bit)
            if cancel_previous_three:
                self.pure_mi_confidences.clear()
                event_text = "未执行咬肌"
                self.last_emg_assist_text = "纯 MI：未执行咬肌，已清空目前全部计数"
            else:
                self.pure_mi_confidences.append(confidence)
                event_text = "窗口计数 +1"
                self.last_emg_assist_text = f"纯 MI：已记录 {bit}，当前编码 {code}"
            self._refresh_pure_mi_counter(
                event_text=event_text,
            )
            return Decision(
                now,
                "编码缓存",
                confidence,
                margin,
                None,
                False,
                f"pure_mi_bit_{bit}_buffered",
                f"MI{bit}",
                proba_dict,
            )

        def _handle_pure_mi_masseter_now(
            self,
            left: bool,
            right: bool,
            now: float,
            confidence: Optional[float] = None,
            margin: float = 0.0,
            probabilities: Optional[Dict[str, float]] = None,
        ) -> Decision:
            """Immediately consume a jaw event and discard the active EEG window."""
            cfg = self._runtime_decoder_config()
            saved_confidence = (
                min(self.pure_mi_confidences)
                if self.pure_mi_confidences
                else float(confidence if confidence is not None else 1.0)
            )
            decoded = self.pure_mi_decoder.handle_masseter(left=left, right=right)
            proba_dict = dict(probabilities or {})

            if decoded.intent == "stop_motion":
                self.pure_mi_confidences.clear()
                self.bus.command_emitted.emit({"cmd": "stop_motion", "intent": "stop_motion"})
                self.continuous_move_intent = None
                self._begin_next_eeg_window(now)
                self._refresh_pure_mi_counter(event_text="双侧咬肌：实时迅速停止，当前窗口已丢弃")
                return Decision(now, "stop_motion", saved_confidence, margin, "stop_motion", True,
                                decoded.reason, "双侧咬肌实时急停", proba_dict)
            if decoded.action == "rollback":
                if self.pure_mi_confidences:
                    self.pure_mi_confidences.pop()
                self._begin_next_eeg_window(now)
                self._refresh_pure_mi_counter()
                return Decision(now, "rest", saved_confidence, margin, None, False, decoded.reason,
                                f"回退后编码={decoded.code or '空'}", proba_dict)
            if decoded.intent is None:
                self._begin_next_eeg_window(now)
                self._refresh_pure_mi_counter(event_text="左咬肌：暂无可提交结果，当前窗口已丢弃")
                return Decision(now, "rest", saved_confidence, margin, None, False, decoded.reason,
                                "等待已完成窗口", proba_dict)

            submitted_code = decoded.code
            self.pure_mi_confidences.clear()
            intent_to_class = {
                "left": 1, "right": 2, "forward": 3, "backward": 4,
                "up": 5, "down": 6, "grab": 7, "release": 8,
            }
            class_id = intent_to_class[decoded.intent]
            padded = np.zeros(8, dtype=np.float64)
            padded[class_id - 1] = saved_confidence
            self._refresh_pure_mi_counter(
                event_text=f"左咬肌实时提交：{submitted_code}，当前窗口已丢弃"
            )
            if decoded.requires_confirmation:
                return self._store_pending_confirmation(
                    class_id, decoded.intent, saved_confidence, margin, padded,
                    f"pure_mi_code_{submitted_code}", now,
                )
            return self._execute_eeg8_action(
                class_id, decoded.intent, saved_confidence, margin, proba_dict,
                now, cfg, f"pure_mi_code_{submitted_code}",
            )
            cfg = self._runtime_decoder_config()
            confirm_window = max(0.2, float(getattr(cfg, "confirmation_window_seconds", 1.0) or 1.0))
            required_count = max(
                1,
                int(
                    getattr(
                        cfg,
                        "confirmation_masseter_count",
                        EEG8_CONFIRMATION_MASSETER_EXECUTIONS,
                    )
                    or EEG8_CONFIRMATION_MASSETER_EXECUTIONS
                ),
            )
            text = (
                f"确认执行？C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)} | "
                f"窗口2 {confirm_window:.1f}s：咬肌执行至少 {required_count} 次"
            )
            self._set_active_intent(intent)
            if hasattr(self, "gate_label"):
                self.gate_label.setText(text)
            self._update_fullscreen_state(text)
            self.last_emg_assist_text = (
                f"窗口2：只确认窗口1结果，咬肌计数需达到 {required_count}"
            )
            self._update_window_countdown(confirm_window)
            return Decision(
                now,
                intent,
                confidence,
                margin,
                None,
                False,
                f"awaiting_masseter_confirmation_{gate_reason}",
                f"C{class_id}",
                self._eeg8_proba_dict(probabilities),
            )

        def _execute_eeg8_action(
            self,
            class_id: int,
            intent: str,
            confidence: float,
            margin: float,
            probabilities: Dict[str, float],
            now: float,
            cfg: object,
            execution_reason: str,
        ) -> Decision:
            """Dispatch one accepted EEG action through the shared RobotCommand path."""
            self.pending_record_emg_counts = dict(self.last_window_emg_counts)
            self.confirmation_cancel_count = 0
            # EEG-only directional actions are bounded one-shot commands.
            continuous_move = eeg8_command_is_continuous(intent, self.active_runtime_strategy)
            # Consume confirmation before publishing the command.
            self.pending_confirmation = None
            command = self._build_motion_command(
                intent,
                confidence,
                now,
                cfg,
                continuous_override=continuous_move,
            )
            command_name = command.name
            self.robot.execute(command)
            if continuous_move:
                self.continuous_move_intent = intent
                self.continuous_move_started_at = float(now)
                self._set_runtime_phase("continuous_move", float(now))
            else:
                self._begin_next_eeg_window(now)
            self.last_command_time = now
            self.eeg8_history.clear()
            self.last_emg_assist_text = "EEG-only 结果：已执行" if self._is_eeg_only_runtime() else "确认结果：已执行"
            if continuous_move:
                if self._is_pure_mi_runtime():
                    text = (
                        f"持续{CLASS_ARM_LABELS.get(class_id, intent)}中 | "
                        "纯 MI 运动状态：双侧咬肌迅速停止"
                    )
                else:
                    stop_window = float(getattr(cfg, "continuous_blink_window_seconds", 2.0))
                    required_blinks = max(1, int(getattr(cfg, "continuous_blink_count", 3) or 3))
                    text = (
                        f"持续{CLASS_ARM_LABELS.get(class_id, intent)}中 | "
                        f"窗口3每 {stop_window:.1f}s 独立计数，眨眼 {required_blinks} 次停止"
                    )
            elif self._is_eeg_only_runtime():
                text = f"EEG-only 已执行：C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)}"
            else:
                text = f"已确认执行：C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)}"
            if hasattr(self, "gate_label"):
                self.gate_label.setText(text)
            if continuous_move:
                self._update_window_countdown(None)
            else:
                self._set_current_result_card("accepted", f"C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)}", "已执行")
            self._update_fullscreen_state(text)
            return Decision(
                now,
                intent,
                confidence,
                margin,
                command_name,
                True,
                execution_reason,
                f"C{class_id}",
                probabilities,
            )

        def _cancel_pending_confirmation(
            self,
            class_id: int,
            intent: str,
            confidence: float,
            margin: float,
            probabilities: Dict[str, float],
            now: float,
            reason: str,
        ) -> Decision:
            """Clear a pending confirmation without emitting a robot command."""
            self.pending_record_emg_counts = dict(self.last_window_emg_counts)
            self.pending_confirmation = None
            self.confirmation_cancel_count += 1
            self._begin_next_eeg_window(now)
            self.eeg8_history.clear()
            self.last_emg_assist_text = "确认结果：未执行"
            if self.confirmation_cancel_count >= self.CONFIRMATION_CANCEL_LIMIT:
                self.confirmation_cancel_count = 0
                if self._is_pure_mi_runtime():
                    self.control_gate_state = "active"
                    text = "连续五次确认取消：纯 MI 保持运动状态，继续下一个二分类窗口"
                else:
                    self.control_gate_state = "rest"
                    text = "连续五次确认取消：已回到静息态，需要连续眨眼三次重新解锁"
            else:
                text = (
                    f"未执行：C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)} | 重新解码 | "
                    f"取消 {self.confirmation_cancel_count}/{self.CONFIRMATION_CANCEL_LIMIT}"
                )
            if hasattr(self, "gate_label"):
                self.gate_label.setText(text)
            self._set_current_result_card("rejected", f"C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)}", "未执行")
            self._update_fullscreen_state(text)
            return Decision(now, "rest", confidence, margin, None, False, reason, f"C{class_id}", probabilities)

        def _confirmation_decision(
            self,
            aux: Optional[np.ndarray],
            now: float,
            confirmation_now_monotonic: Optional[float] = None,
            gate_observed_before_deadline: bool = False,
        ) -> Decision:
            pending = dict(self.pending_confirmation or {})
            if not pending:
                return Decision(now, "rest", 0.0, 0.0, None, False, "no_pending_confirmation", "confirm", {})
            cfg = self._runtime_decoder_config()
            # Window 2 only confirms window 1's frozen result. It never uses
            # jaw side to remap the class selected by window 1.
            snapshot = self._update_phase_emg_events(aux, now)
            masseter_executions = int(snapshot.jaw_executions)
            required_count = max(
                1,
                int(
                    getattr(
                        cfg,
                        "confirmation_masseter_count",
                        EEG8_CONFIRMATION_MASSETER_EXECUTIONS,
                    )
                    or EEG8_CONFIRMATION_MASSETER_EXECUTIONS
                ),
            )
            confirmation_reason = (
                f"window2_masseter_count_{masseter_executions}_"
                f"left_{int(snapshot.left_jaw)}_right_{int(snapshot.right_jaw)}"
            )
            class_id = int(pending.get("class_id", 0) or 0)
            intent = str(pending.get("intent", "rest"))
            confidence = float(pending.get("confidence", 0.0) or 0.0)
            margin = float(pending.get("margin", 0.0) or 0.0)
            probabilities = dict(pending.get("probabilities", {}) or {})
            action, reason = resolve_eeg8_confirmation_action(
                self.active_runtime_strategy,
                True,
                pending.get("confirmation_started_monotonic"),
                time.monotonic() if confirmation_now_monotonic is None else confirmation_now_monotonic,
                float(getattr(cfg, "confirmation_window_seconds", 1.0) or 1.0),
                "count",
                confirmation_reason,
                gate_observed_before_deadline=gate_observed_before_deadline,
                masseter_execution_count=masseter_executions,
                required_masseter_executions=required_count,
            )
            if action == "execute":
                return self._execute_eeg8_action(
                    class_id, intent, confidence, margin, probabilities, now, cfg, reason
                )
            if action == "cancel":
                return self._cancel_pending_confirmation(
                    class_id, intent, confidence, margin, probabilities, now, reason
                )
            self.last_emg_assist_text = (
                f"窗口2确认中：咬肌 {masseter_executions}/"
                f"{required_count}"
            )
            text = (
                f"确认执行？C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)} | "
                f"窗口2咬肌计数 {masseter_executions}/"
                f"{required_count}"
            )
            if hasattr(self, "gate_label"):
                self.gate_label.setText(text)
            self._update_fullscreen_state(text)
            return Decision(now, intent, confidence, margin, None, False, reason, f"C{class_id}", probabilities)

        def _begin_next_eeg_window(self, timestamp: float) -> None:
            self._set_runtime_phase("eeg", float(timestamp))
            if self.source is not None and hasattr(self.source, "begin_next_eeg_window"):
                self.source.begin_next_eeg_window(float(timestamp))

        def _update_eeg8_intent(self, eeg: np.ndarray, aux: Optional[np.ndarray], now: float, true_label: Optional[str] = None) -> Decision:
            if self.model is None:
                raise RuntimeError("八分类模型未加载")
            cfg = self._runtime_decoder_config()
            if self.pending_confirmation is not None:
                return self._confirmation_decision(aux, now)
            # A completed action remains highlighted until the next EEG window
            # is actually decoded.
            self._set_active_intent(None)
            if not self._is_eeg_only_runtime():
                self._update_phase_emg_events(aux, now)
            if not self._is_eeg_only_runtime():
                gate_decision = self._rest_gate_decision(aux, now, true_label)
                if gate_decision is not None:
                    self.last_emg_assist_text = self._emg_assist_text(aux)
                    return gate_decision
            # EEG-only skips AUX checks; EEG artifact checks still apply.
            artifact_aux = None if self._is_eeg_only_runtime() else aux
            artifact, artifact_reason = self.model.detect_artifact(eeg, artifact_aux)
            if artifact:
                return Decision(now, "artifact", 0.0, 0.0, None, False, artifact_reason, "artifact", {})
            raw_probabilities = np.asarray(self.model.predict_eeg_proba(eeg)[0], dtype=np.float64)
            if self._is_custom_mi_runtime():
                strategy = dict(getattr(self.model, "training_strategy_", {}) or {})
                labels = list(strategy.get("class_labels", []) or [])
                names = list(strategy.get("class_names", []) or [])
                if len(raw_probabilities) != len(names) or len(labels) != len(names):
                    raise RuntimeError("自定义 MI 模型的类别元数据与输出维度不匹配")
                order = np.argsort(raw_probabilities)[::-1]
                top_idx = int(order[0])
                second_idx = int(order[1]) if order.size > 1 else top_idx
                confidence = float(raw_probabilities[top_idx])
                margin = float(raw_probabilities[top_idx] - raw_probabilities[second_idx])
                probability_map = {
                    str(labels[index]): float(raw_probabilities[index])
                    for index in range(len(labels))
                }
                display_label = str(labels[top_idx])
                self.eeg8_history.clear()
                self._set_active_intent(None)
                self._update_fullscreen_state(
                    f"VR 分类展示 | {display_label} | 置信度 {confidence:.1%} | 机械臂输出关闭"
                )
                if hasattr(self, "gate_label"):
                    self.gate_label.setText("当前状态：自定义 MI 分类展示 | 仅更新 VR，不发送机械臂指令")
                return Decision(
                    now,
                    display_label,
                    confidence,
                    margin,
                    None,
                    False,
                    "custom_mi_vr_display_only",
                    f"custom_mi:{top_idx + 1}",
                    probability_map,
                )
            if raw_probabilities.size < 8:
                raise RuntimeError(f"模型输出类别数不足 8：{raw_probabilities.size}")
            if self._is_eeg_only_runtime():
                probabilities = raw_probabilities
                raw_order = np.argsort(probabilities)[::-1]
                selected_idx = int(raw_order[0])
                second_idx = int(raw_order[1]) if raw_order.size > 1 else selected_idx
                gate_reason = f"eeg_only_raw_eeg_top_C{selected_idx + 1}"
            else:
                gate = self._masseter_gate_for_window(aux)
                probabilities, gate_reason = self._apply_masseter_gate(raw_probabilities, gate)
                selected_idx, second_idx, gate_reason = self._select_masseter_mapped_class(probabilities, gate)
                raw_order = np.argsort(probabilities)[::-1]
                raw_top_idx = int(raw_order[0])
                if selected_idx is None:
                    confidence = float(probabilities[raw_top_idx])
                    margin = 0.0
                    self.eeg8_history.clear()
                    self._update_fullscreen_state(f"当前状态：映射未匹配 | EEG Top=C{raw_top_idx + 1} | {gate.reason}")
                    return Decision(now, "rest", confidence, margin, None, False, gate_reason, f"C{raw_top_idx + 1}", self._eeg8_proba_dict(probabilities))
            top_idx = int(selected_idx)
            class_id = top_idx + 1
            confidence = float(probabilities[top_idx])
            margin = abs(float(probabilities[top_idx] - probabilities[second_idx]))
            if class_id > 8:
                self.eeg8_history.clear()
                self._update_fullscreen_state("当前状态：静息/基线 | 机械臂保持当前位置")
                return Decision(now, "rest", confidence, margin, None, False, "rest_baseline", f"C{class_id}", self._eeg8_proba_dict(probabilities))
            intent = CLASS_TO_ARM[class_id]
            if not self._is_eeg_only_runtime() and self.control_gate_state == "rest":
                return Decision(now, "rest", confidence, margin, None, False, "rest_gate_waiting_for_blink3", "rest_gate", self._eeg8_proba_dict(probabilities))
            self.eeg8_history.append((intent, confidence, margin, now))
            eligible, safety_reason = eeg8_command_eligibility(self.eeg8_history, intent, cfg)
            if not eligible:
                if safety_reason == "invalid_eeg8_history_record":
                    # Discard the rejected single-window record.
                    self.eeg8_history.clear()
                self._update_fullscreen_state(
                    f"当前状态：EEG 决策未通过 | C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)} | {safety_reason}"
                )
                return Decision(
                    now,
                    "rest",
                    confidence,
                    margin,
                    None,
                    False,
                    safety_reason,
                    f"C{class_id}",
                    self._eeg8_proba_dict(probabilities),
                )
            if self._is_eeg_only_runtime():
                self._update_fullscreen_state(
                    f"当前状态：EEG-only 安全执行 | C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)}"
                )
                decision = self._execute_eeg8_action(
                    class_id,
                    intent,
                    confidence,
                    margin,
                    self._eeg8_proba_dict(probabilities),
                    now,
                    cfg,
                    "eeg_only_direct_execute",
                )
            else:
                if hasattr(self, "gate_label"):
                    self.gate_label.setText(f"当前状态：运动控制态 | EEG 八分类 + 咬肌门控 | {gate.reason}")
                self._update_fullscreen_state(f"当前状态：运动控制态 | 待确认：C{class_id} {CLASS_ARM_LABELS.get(class_id, intent)} | {gate.reason}")
                decision = self._store_pending_confirmation(class_id, intent, confidence, margin, probabilities, gate_reason, now)
            self.eeg8_history.clear()
            return decision

        def _arm_start_mode(self, mode: str, label: str) -> None:
            if self.timer.isActive() or self.transport_demo_running:
                return
            self.pending_start_mode = str(mode)
            text = f"{label}已就绪 | 按回车或空格开始"
            self.status_label.setText(f"状态：{text}")
            if hasattr(self, "decode_result_label"):
                self.decode_result_label.setText(f"实时解码结果：{text}")
            self.append_log(text)
            self.setFocus()

        def arm_realtime_start(self) -> None:
            self._arm_start_mode("realtime", "实时采集")

        def arm_demo_start(self) -> None:
            self._arm_start_mode("demo", "搬运")

        def begin_armed_start(self) -> None:
            if self.timer.isActive() or self.transport_demo_running:
                return
            mode = self.pending_start_mode
            if not mode:
                mode = "realtime"
            self.pending_start_mode = None
            if mode == "realtime":
                if self._online_input_mode() == "manual":
                    self.start_manual_button_test()
                else:
                    self.start_online()
            elif mode == "demo":
                self.start_transport_demo()

        def start_manual_button_test(self) -> None:
            """Start the three-window button workflow without model or LSL."""
            if self.timer.isActive() or (
                self.startup_worker is not None and self.startup_worker.isRunning()
            ):
                self.append_log(
                    "人工按钮模式未启动：请先停止正在运行的 LSL 解码或后台启动。"
                )
                return
            self.pending_start_mode = None
            self.manual_simulation_token += 1
            self.manual_simulation_pending = None
            self.manual_simulation_deadline_monotonic = 0.0
            self.manual_confirmation_count = 0
            self.manual_blink_count = 0
            self.manual_stop_pending = False
            self.manual_simulation_timer.start()
            self.runtime_session = None
            self.model = None
            if self.source is not None and hasattr(self.source, "close"):
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭旧数据源失败：{exc}")
            self.source = None
            self._start_decode_result_session()
            self.operation_start_time = time.perf_counter()
            self._update_operation_elapsed()
            self.operation_elapsed_timer.start(100)
            self._set_manual_simulation_phase(
                "window1", "窗口1——请选择一个 C1-C8 分类结果"
            )
            self.startup_progress.setValue(100)
            self.startup_progress.setFormat(
                "人工按钮三窗口测试运行中 | 未连接 LSL"
            )
            self.gate_label.setText(
                "当前状态：人工测试窗口1 | 等待按钮输入 C1-C8"
            )
            self.decode_result_label.setText(
                "实时解码结果：人工按钮模式 | 未连接 LSL"
            )
            self._set_current_result_card(
                "collect", "等待按钮输入", "窗口1"
            )
            self.status_label.setText(
                "状态：人工按钮三窗口测试运行中 | LSL：未连接"
            )
            self.append_log(
                "人工按钮三窗口测试已启动：未加载模型、未搜索或连接 LSL。"
            )

        def start_online(self) -> None:
            if self.timer.isActive() or self.transport_demo_running:
                return
            self.start_realtime_online(force_lsl=True)

        def start_realtime_online(self, force_lsl: bool = False) -> None:
            if self.timer.isActive() or self.transport_demo_running or (
                self.startup_worker is not None and self.startup_worker.isRunning()
            ):
                return
            self.pending_start_mode = None
            cfg = self.config.realtime_config()
            self.startup_progress.setValue(0)
            self.startup_progress.setFormat("准备后台启动：%p%")
            self.status_label.setText("状态：正在后台加载模型并订阅系统 LSL")
            self.decode_result_label.setText("实时解码结果：启动准备中")
            worker = RealtimeStartupWorker(
                cfg,
                use_model=self.use_model_check.isChecked(),
                use_config_thresholds=self.use_weight_check.isChecked(),
                parent=self,
            )
            self.startup_worker = worker
            worker.progress_signal.connect(self._on_startup_progress)
            worker.log_signal.connect(self.append_log)
            worker.ready_signal.connect(self._on_startup_ready)
            worker.error_signal.connect(self._on_startup_error)
            worker.finished.connect(self._on_startup_worker_finished)
            worker.start()

        def _on_startup_progress(self, percent: int, stage: str) -> None:
            self.startup_progress.setValue(max(0, min(100, int(percent))))
            self.startup_progress.setFormat(f"{stage}：%p%")
            self.status_label.setText(f"状态：{stage}")

        def _on_startup_ready(self, payload: Dict) -> None:
            worker = self.startup_worker
            if worker is None or worker.isInterruptionRequested():
                source = payload.get("source")
                if source is not None and hasattr(source, "close"):
                    source.close()
                return
            self.model = payload["model"]
            self.source = payload["source"]
            self.masseter_template_bank = payload.get("template_bank")
            session = payload.get("runtime_session")
            if not isinstance(session, RuntimeSessionConfig):
                if self.source is not None and hasattr(self.source, "close"):
                    self.source.close()
                self.model = None
                self.source = None
                self._on_startup_error("启动结果缺少不可变 RuntimeSessionConfig，已安全取消")
                return
            self.runtime_session = session
            self.active_runtime_strategy = session.runtime_strategy
            signal_cfg = self.model.signal_config
            decoder_cfg = session.decoder_config
            self.operation_start_time = time.perf_counter()
            self._update_operation_elapsed()
            self.operation_elapsed_timer.start(100)
            self._start_decode_result_session()
            self.grip_controller = GripStateController(
                self.robot,
                target=session.target,
                duration_seconds=decoder_cfg.command_duration_seconds,
            )
            self._reset_eeg8_decoder_state()
            self.timer.start(int(round(EEG8_ONLINE_TIMER_INTERVAL_SECONDS * 1000.0)))
            self.gate_toggle_button.setEnabled(True)
            self.startup_progress.setValue(100)
            self.startup_progress.setFormat("实时控制运行中：%p%")
            self.append_log(f"运行期配置已冻结：{session.audit_summary()}")
            self.append_log(f"融合策略：{session.fusion_policy.description}")
            if self._is_pure_mi_runtime():
                self.status_label.setText(
                    f"状态：纯 MI 静息态 | 仅检测眼电，三次眨眼进入运动态 | "
                    f"{signal_cfg.window_seconds:.1f}s/窗口 | {session.decoder_config_source} 阈值"
                )
                self.append_log(
                    "纯 MI 已在静息态启动：静息态只检测眼电；三次眨眼双向切换"
                    "静息态/运动态。运动态下左咬肌提交、右咬肌回退一窗、"
                    "双侧咬肌迅速停止。"
                )
            elif is_eeg_only_runtime_strategy(self.active_runtime_strategy):
                self.status_label.setText(
                    f"状态：BrainVision LSL 32导 EEG | {signal_cfg.window_seconds:.1f}s EEG-only 安全直接执行 | {session.decoder_config_source} 阈值"
                )
                self.append_log("BrainVision LSL EEG-only 在线读取已启动：只取前 32 路 EEG。")
                self.append_log("实时分类算法：单个 EEG 窗口 -> FBCSP/分类器 -> C1-C8 -> 单窗口安全阈值 -> 一次性命令。")
            else:
                self.status_label.setText(
                    f"状态：BrainVision LSL 32导 EEG+AUX | {signal_cfg.window_seconds:.1f}s解码 + "
                    f"{float(getattr(decoder_cfg, 'confirmation_window_seconds', 1.0)):.1f}s确认 | {session.decoder_config_source} 阈值"
                )
                self.append_log("BrainVision LSL 在线读取已启动：窗口1同步读取前32路 EEG 与 AUX1-4。")
                self.append_log(
                    "三窗口算法：窗口1咬肌映射并对映射外概率降权、EEG选类；"
                    "窗口2累计咬肌确认；C1-C6进入窗口3持续运动并以三次眨眼停止，C7-C8单次抓取/释放。"
                )
            self.setFocus()

        def _on_startup_error(self, message: str) -> None:
            self.source = None
            self.model = None
            self.runtime_session = None
            self.startup_progress.setValue(0)
            self.startup_progress.setFormat("未订阅系统 LSL | 按回车或空格重试")
            self.gate_label.setText("当前状态：离线预览 | BCI 控制未启动")
            self.gate_counter_label.setText("当前时间段计数：-- | AUX：未连接")
            self.decode_result_label.setText("实时解码结果：未开始（LSL 未连接）")
            self._set_current_result_card("collect", "未连接", "等待 BrainVision LSL 数据流")
            self.gate_toggle_button.setEnabled(False)
            self.status_label.setText("状态：未连接 BrainVision LSL | 机械臂手动演示仍可使用")
            self.append_log(f"未连接 BrainVision LSL：{message}")

        def _on_startup_worker_finished(self) -> None:
            worker = self.sender()
            if worker is self.startup_worker:
                if worker.isInterruptionRequested():
                    self.startup_progress.setValue(0)
                    self.startup_progress.setFormat("按回车或空格启动实时控制")
                self.startup_worker = None

        def stop_online(self) -> None:
            self.transport_demo_running = False
            self.pending_start_mode = None
            if self.continuous_move_intent is not None:
                self.bus.command_emitted.emit({"cmd": "stop_motion", "intent": "stop_motion"})
            self.continuous_move_intent = None
            self.manual_simulation_token += 1
            self.manual_simulation_pending = None
            self.manual_simulation_deadline_monotonic = 0.0
            self.manual_confirmation_count = 0
            self.manual_blink_count = 0
            self.manual_stop_pending = False
            self.manual_simulation_timer.stop()
            self._set_manual_simulation_phase(
                "window1", "窗口1——请选择一个 C1-C8 分类结果"
            )
            if self.startup_worker is not None and self.startup_worker.isRunning():
                self.startup_worker.requestInterruption()
                self.startup_progress.setValue(0)
                self.startup_progress.setFormat("正在取消后台启动")
            self.operation_elapsed_timer.stop()
            self.operation_start_time = 0.0
            self.timer.stop()
            if self.source is not None and hasattr(self.source, "close"):
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭在线源失败：{exc}")
            self.source = None
            # Release session settings after stop_motion is dispatched.
            self.runtime_session = None
            self.gate_toggle_button.setEnabled(False)
            if self.startup_worker is None or not self.startup_worker.isRunning():
                self.startup_progress.setValue(0)
                self._online_input_mode_changed()
            self.status_label.setText("状态：已停止")
            self._update_window_countdown(None)
            self._update_transport_counter(0, len(self._transport_demo_steps()), "待开始", "待开始")
            self.append_log("实时分类已停止。")
            if self.decode_log_path:
                self.append_log(f"本次在线分类结果文件：{self.decode_log_path}")

        def reset_task(self) -> None:
            self.pending_start_mode = None
            self.manual_simulation_token += 1
            self.manual_simulation_pending = None
            self._set_manual_simulation_phase(
                "window1", "窗口1——请选择一个 C1-C8 分类结果"
            )
            self._reset_eeg8_decoder_state()
            self.operation_elapsed_timer.stop()
            self.elapsed_label.setText("操作计时：00.0 s")
            self.grip_controller.reset()
            for axis in self.manual_orientation_degrees:
                self.manual_orientation_degrees[axis] = 0.0
            self._refresh_orientation_limit_status()
            payload = {
                "cmd": "home",
                "intent": "home",
                "target": self._runtime_target(),
                "confidence": 1.0,
                "sequence_id": 0,
                "timestamp": time.time(),
                "duration_seconds": 0.0,
            }
            self.bus.command_emitted.emit(payload)
            self.append_log("真实机械臂已发送回零指令。")
            self._set_active_intent(None)
            self._update_window_countdown(None)
            self._update_fullscreen_state("当前状态：静息态 | 门控：等待眨眼三次")
            self._sync_gate_toggle_button()
            self._update_transport_counter(0, len(self._transport_demo_steps()), "待开始", "待开始")

        def _update_operation_elapsed(self) -> None:
            if self.operation_start_time <= 0:
                self.elapsed_label.setText("操作计时：00.0 s")
                return
            elapsed = max(0.0, time.perf_counter() - self.operation_start_time)
            self.elapsed_label.setText(f"操作计时：{elapsed:04.1f} s")

        def _update_fullscreen_state(self, text: Optional[str] = None) -> None:
            """Compatibility no-op after removal of online fullscreen mode."""

        def _transport_decode_state(self, intent: str) -> str:
            return {
                "home": "静息/回零",
                "gate_switch": "状态切换",
                "left": "C1 左移",
                "right": "C2 右移",
                "forward": "C3 前移",
                "backward": "C4 后移",
                "up": "C5 上移",
                "down": "C6 下移",
                "grab": "C7 抓取",
                "release": "C8 释放",
            }.get(str(intent), str(intent or "待开始"))

        def _update_transport_counter(self, current: int, total: int, label: str = "", decode_state: str = "") -> None:
            total = max(1, int(total))
            current = max(0, min(total, int(current)))
            state = decode_state or ("待开始" if current <= 0 else (label or "进行中"))
            text = f"进度：当前第 {current} 步 | 解码状态：{state}"
            if hasattr(self, "step_counter_label"):
                self.step_counter_label.setText(text)

        def _refresh_vr_button(self, button: QtWidgets.QPushButton) -> None:
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

        def _set_active_intent(self, intent: Optional[str]) -> None:
            for mapping in [self.intent_buttons]:
                for key, button in mapping.items():
                    button.setProperty("active", bool(intent and key == intent))
                    self._refresh_vr_button(button)

        def _on_command_visual(self, payload: Dict) -> None:
            intent = str(payload.get("intent", ""))
            if intent in self.intent_buttons:
                self._set_active_intent(intent)
            elif intent in {"home", "stop_motion"}:
                self._set_active_intent(None)
            # Pitch/Yaw/Roll are independent pose adjustments. They must not
            # overwrite the current C1-C8/gripper display state.

        def tick_online(self) -> None:
            if self.source is None or (self.model is None and not self._pure_emg_enabled()):
                self.stop_online()
                return
            if (
                self.runtime_session is not None
                and self.model is not None
                and not runtime_signal_contract_matches(self.runtime_session, self.model.signal_config)
            ):
                message = "模型 SignalConfig 与本次运行快照不一致，已安全停止；请重新加载并校验窗口1模型。"
                self.append_log(message)
                self.stop_online()
                self.status_label.setText(f"状态：{message}")
                self.decode_result_label.setText("实时解码结果：已因 SignalConfig 漂移安全停止")
                return
            if self.continuous_move_intent is not None and self.model is not None:
                self._update_live_emg_counts_from_source()
                cfg = self._runtime_decoder_config()
                stop_window = max(0.5, float(getattr(cfg, "continuous_blink_window_seconds", 2.0)))
                required_blinks = max(1, int(getattr(cfg, "continuous_blink_count", 3) or 3))
                now = float(self.last_aux_timestamp or time.time())
                watchdog_now = time.monotonic()
                if hasattr(self.source, "last_new_sample_monotonic") and aux_stream_watchdog_expired(
                    getattr(self.source, "last_new_sample_monotonic", 0.0),
                    watchdog_now,
                    0.5,
                ):
                    self.bus.command_emitted.emit({"cmd": "stop_motion", "intent": "stop_motion"})
                    self.continuous_move_intent = None
                    self.continuous_move_started_at = 0.0
                    self.control_gate_state = "rest"
                    self._begin_next_eeg_window(now)
                    text = "AUX 数据超过 0.5s 未更新：已急停并回到静息态"
                    self.gate_label.setText(text)
                    self._set_current_result_card("rejected", "连续移动已急停", "AUX 数据中断")
                    self._update_fullscreen_state(text)
                    self.append_log(text)
                    return
                self._update_window_countdown(None)
                snapshot = self._active_emg_snapshot()
                if self._is_pure_mi_runtime():
                    gate_decision = self._handle_pure_mi_blink_switch(now)
                    if gate_decision is not None:
                        self._append_decode_result(
                            gate_decision,
                            "-",
                            self._eeg8_display(gate_decision),
                            self._window_emg_stats_text(),
                        )
                        self.append_log(gate_decision.reason)
                        return
                if (
                    self._is_pure_mi_runtime()
                    and int(snapshot.left_jaw) > 0
                    and int(snapshot.right_jaw) > 0
                ):
                    self.bus.command_emitted.emit({"cmd": "stop_motion", "intent": "stop_motion"})
                    self.continuous_move_intent = None
                    self.continuous_move_started_at = 0.0
                    self.pure_mi_decoder.clear()
                    self._refresh_pure_mi_counter(event_text="双侧咬肌：迅速停止并清空编码")
                    self._begin_next_eeg_window(now)
                    text = "纯 MI：检测到双侧咬肌，持续移动已迅速停止"
                    self.gate_label.setText(text)
                    self._update_fullscreen_state(text)
                    self.append_log(text)
                    return
                if self._is_pure_mi_runtime():
                    self.gate_label.setText(
                        "纯 MI 运动态：持续移动中 | 三次眨眼切回静息态 | 双侧咬肌迅速停止"
                    )
                    return
                phase_started_at = float(self.runtime_phase_started_at or now)
                elapsed = max(0.0, now - phase_started_at)
                # Qt may tick slightly after the segment deadline. Re-evaluate
                # the completed segment using an exact LSL-time slice so samples
                # after the boundary cannot leak backward into this decision.
                if elapsed >= stop_window and hasattr(self.source, "latest_aux_window_between"):
                    segment_ended_at = phase_started_at + stop_window
                    try:
                        bounded_aux, _bounded_now = self.source.latest_aux_window_between(
                            phase_started_at,
                            segment_ended_at,
                        )
                        self._set_runtime_phase("continuous_move", phase_started_at)
                        snapshot = self._update_phase_emg_events(
                            bounded_aux,
                            segment_ended_at,
                        )
                    except Exception as exc:
                        self.append_log(f"窗口3边界切片失败，按当前安全计数结算：{exc}")
                window3_action, window3_reason = resolve_continuous_stop_window(
                    snapshot.blink,
                    required_blinks,
                    elapsed,
                    stop_window,
                )
                if window3_action == "stop":
                    self.bus.command_emitted.emit({"cmd": "stop_motion", "intent": "stop_motion"})
                    self.continuous_move_intent = None
                    self.continuous_move_started_at = 0.0
                    self._begin_next_eeg_window(now)
                    self.gate_label.setText(
                        f"连续移动已停止：窗口3眨眼 {int(snapshot.blink)}/{required_blinks} | 进入窗口1"
                    )
                    self.append_log(f"窗口3停止：{window3_reason}")
                elif window3_action == "next_window":
                    # Window 3 is fixed and non-overlapping. If this segment
                    # did not reach the blink target, clear it and continue.
                    self._set_runtime_phase(
                        "continuous_move",
                        phase_started_at + stop_window,
                    )
                    self.gate_label.setText(
                        f"连续移动中：窗口3眨眼 {int(snapshot.blink)}/{required_blinks}，"
                        f"开始下一个 {stop_window:.1f}s 窗口"
                    )
                    self.append_log(f"窗口3换段：{window3_reason}")
                elif int(snapshot.blink) >= required_blinks:
                    remaining = max(0.0, stop_window - elapsed)
                    self.gate_label.setText(
                        f"连续移动中：窗口3眨眼 {int(snapshot.blink)}/{required_blinks}，"
                        f"停止条件已满足，等待本窗口结束 {remaining:.1f}s"
                    )
                return
            if self.pending_confirmation is not None and self.model is not None:
                cfg = self._runtime_decoder_config()
                confirm_window = max(0.2, float(getattr(cfg, "confirmation_window_seconds", 1.0) or 1.0))
                pending = dict(self.pending_confirmation or {})
                started_at = float(pending.get("confirmation_started_at", 0.0) or 0.0)
                started_monotonic = float(pending.get("confirmation_started_monotonic", 0.0) or 0.0)
                now_monotonic = time.monotonic()
                elapsed_monotonic = max(0.0, now_monotonic - started_monotonic)
                bounded_to_deadline = False
                aux, aux_now = None, time.time()
                if hasattr(self.source, "latest_aux_window"):
                    try:
                        if (
                            started_at > 0.0
                            and elapsed_monotonic >= confirm_window
                            and hasattr(self.source, "latest_aux_window_between")
                        ):
                            aux, aux_now = self.source.latest_aux_window_between(
                                started_at,
                                started_at + confirm_window,
                            )
                            bounded_to_deadline = True
                        elif started_at > 0.0 and hasattr(self.source, "latest_aux_window_since"):
                            aux, aux_now = self.source.latest_aux_window_since(started_at)
                        else:
                            aux, aux_now = self.source.latest_aux_window(confirm_window)
                    except Exception:
                        aux, aux_now = None, time.time()
                # Use monotonic time for the deadline and LSL time for samples.
                now_for_confirm = float(aux_now or time.time())
                try:
                    # Resolve each timer tick; deadline samples are clipped above.
                    decision = self._confirmation_decision(
                        aux,
                        now_for_confirm,
                        now_monotonic,
                        gate_observed_before_deadline=bounded_to_deadline,
                    )
                except Exception as exc:
                    self.append_log(f"确认阶段肌电解码失败：{exc}")
                    QtWidgets.QMessageBox.critical(self, "确认阶段失败", str(exc))
                    self.stop_online()
                    return
                if self.pending_confirmation is not None:
                    # Record a result only after execute or cancel.
                    elapsed = max(0.0, now_monotonic - started_monotonic)
                    self._update_window_countdown(max(0.0, confirm_window - elapsed))
                    return
                pred_text = self._eeg8_display(decision)
                self._append_decode_result(decision, "-", pred_text, self._window_emg_stats_text())
                self._update_window_countdown(None)
                return
            try:
                eeg, aux, now, true_label = self.source.read_window()
            except BlockingIOError:
                self._update_live_emg_counts_from_source()
                if self._is_pure_mi_runtime():
                    event_now = float(self.last_aux_timestamp or time.time())
                    gate_decision = self._handle_pure_mi_blink_switch(event_now)
                    if gate_decision is not None:
                        pred_text = self._eeg8_display(gate_decision)
                        self._append_decode_result(
                            gate_decision,
                            "-",
                            pred_text,
                            self._window_emg_stats_text(),
                        )
                        self.append_log(gate_decision.reason)
                        self._update_window_countdown(None)
                        return
                    if self.control_gate_state == "rest":
                        # Rest state consumes only AUX eye events. EEG, jaw
                        # commands and binary sequence updates remain disabled.
                        self.decode_result_label.setText(
                            "实时解码结果：静息态仅检测眼电 | "
                            + self._window_emg_stats_text()
                        )
                        return
                    snapshot = self._active_emg_snapshot()
                    left_now = int(snapshot.left_jaw) > 0
                    right_now = int(snapshot.right_jaw) > 0
                    if left_now or right_now:
                        decision = self._handle_pure_mi_masseter_now(
                            left_now,
                            right_now,
                            event_now,
                        )
                        pred_text = self._eeg8_display(decision)
                        emg_stats = self._window_emg_stats_text()
                        self._append_decode_result(
                            decision, "-", pred_text, emg_stats
                        )
                        self.decode_result_label.setText(
                            f"实时解码结果：{pred_text} | 当前 EEG 窗口已截断且未保存"
                        )
                        self.status_label.setText(
                            f"状态：运动状态 | {decision.reason} | 当前窗口已丢弃"
                        )
                        self.append_log(
                            f"纯 MI 实时咬肌：{decision.reason}；"
                            "当前 EEG 窗口已立即结束且未保存分类结果。"
                        )
                        self._update_window_countdown(None)
                        return
                remaining = None
                if self.source is not None and hasattr(self.source, "seconds_until_next_window"):
                    try:
                        remaining = float(self.source.seconds_until_next_window())
                    except Exception:
                        remaining = None
                self._update_window_countdown(remaining)
                return
            except StopIteration:
                self.stop_online()
                self.append_log("在线流结束。")
                return
            except Exception as exc:
                self.append_log(f"在线读取失败：{exc}")
                QtWidgets.QMessageBox.critical(self, "在线读取失败", str(exc))
                self.stop_online()
                return
            try:
                decision = (
                    self._update_pure_mi_intent(eeg, aux, now)
                    if self._is_pure_mi_runtime()
                    else self._update_eeg8_intent(eeg, aux, now, true_label)
                )
            except Exception as exc:
                decoder_name = "纯 MI" if self._is_pure_mi_runtime() else "八分类"
                self.append_log(f"{decoder_name}在线解码失败：{exc}")
                QtWidgets.QMessageBox.critical(self, f"{decoder_name}在线解码失败", str(exc))
                self.stop_online()
                return
            completed_counts = dict(getattr(
                self,
                "last_window_emg_counts",
                {"blink": 0, "left_jaw": 0, "right_jaw": 0},
            ))
            if self.pending_confirmation is None and self.continuous_move_intent is None:
                # Start a fresh counter phase after each completed EEG window.
                self.pending_record_emg_counts = completed_counts
                self._begin_next_eeg_window(float(now))
            label_text = true_label or "-"
            pred_text = self._eeg8_display(decision)
            next_window = (
                self.runtime_session.signal_step_seconds
                if self.runtime_session is not None
                else (float(getattr(self.model.signal_config, "step_seconds", 2.0) or 2.0) if self.model is not None else 2.0)
            )
            self._update_window_countdown(next_window)
            # Live counters show the active phase.
            emg_stats = self._window_emg_stats_text()
            self._append_decode_result(decision, label_text, pred_text, emg_stats)
            if hasattr(self, "decode_result_label"):
                self.decode_result_label.setText(
                    f"实时解码结果：pred={pred_text} | conf={decision.confidence:.3f} | "
                    f"accepted={decision.accepted} | {emg_stats}"
                )
            self.status_label.setText(
                f"状态：true={label_text} pred={pred_text} conf={decision.confidence:.3f} "
                f"accepted={decision.accepted} reason={decision.reason} | {self.last_emg_assist_text} | {emg_stats}"
            )
            self.append_log(
                f"true={label_text:9s} pred={pred_text:9s} intent={decision.intent:8s} conf={decision.confidence:.3f} "
                f"margin={decision.margin:.3f} accepted={decision.accepted} reason={decision.reason} | {self.last_emg_assist_text} | {emg_stats}"
            )

        def manual_command(
            self,
            intent: str,
            duration_seconds: Optional[float] = None,
            target_position: Optional[Sequence[float]] = None,
            drop_zone_index: Optional[int] = None,
            continuous: Optional[bool] = None,
            protocol_speed: Optional[float] = None,
        ) -> None:
            target = self._runtime_target()
            if intent == "home":
                if self.continuous_move_intent is not None:
                    self.bus.command_emitted.emit({"cmd": "stop_motion", "intent": "stop_motion"})
                    self.continuous_move_intent = None
                    self.continuous_move_started_at = 0.0
                payload = {"cmd": "home", "intent": "home", "target": target, "confidence": 1.0, "sequence_id": 0, "timestamp": time.time(), "duration_seconds": 0.0}
                self._stop_translation_limit_tracking()
                self.translation_position_mm = {"x": 0.0, "y": 0.0, "z": 0.0}
                self._refresh_translation_limit_status()
            else:
                requested_continuous = (
                    intent in CONTINUOUS_MOVE_VECTORS
                    if continuous is None
                    else bool(continuous) and intent in CONTINUOUS_MOVE_VECTORS
                )
                if requested_continuous and not self._translation_can_start(intent):
                    self.append_log(
                        f"{CLASS_ARM_LABELS.get(self._manual_class_id(intent), intent)}"
                        " 已到达平移软限位，未发送 move_start。"
                    )
                    return
                command = self._build_motion_command(
                    intent,
                    confidence=1.0,
                    now=time.time(),
                    continuous_override=continuous,
                )
                payload = asdict(command)
                payload["cmd"] = payload.pop("name")
                payload["duration_seconds"] = float(
                    self._runtime_decoder_config().command_duration_seconds
                    if duration_seconds is None and self.runtime_session is not None
                    else (self.config.command_duration_spin.value() if duration_seconds is None else duration_seconds)
                )
                use_continuous = command.continuous
                if use_continuous:
                    self.continuous_move_intent = intent
                    self.continuous_move_started_at = float(payload["timestamp"])
                    if self.model is not None and self.source is not None:
                        self._set_runtime_phase("continuous_move", float(payload["timestamp"]))
                        self._update_window_countdown(None)
                elif self.continuous_move_intent is not None:
                    self.bus.command_emitted.emit({"cmd": "stop_motion", "intent": "stop_motion"})
                    self.continuous_move_intent = None
                    self.continuous_move_started_at = 0.0
            if target_position is not None:
                payload["target_position"] = [float(value) for value in target_position]
            if drop_zone_index is not None:
                payload["drop_zone_index"] = int(drop_zone_index)
            if protocol_speed is not None:
                payload["protocol_speed"] = max(0.0, float(protocol_speed))
            self.bus.command_emitted.emit(payload)
            if payload.get("continuous") and intent in CONTINUOUS_MOVE_VECTORS:
                self._start_translation_limit_tracking(intent)
            self.append_log(f"手动发送：{payload['cmd']} / {payload['intent']}")

        def _set_manual_simulation_phase(self, phase: str, text: str) -> None:
            self.manual_simulation_phase = str(phase)
            if phase == "window1":
                self._set_active_intent(None)
            if hasattr(self, "manual_simulation_label"):
                self.manual_simulation_label.setText(f"人工测试状态：{text}")
            window1_ready = phase == "window1"
            for button in self.intent_buttons.values():
                button.setEnabled(window1_ready)
            self.manual_confirm_button.setEnabled(phase == "window2")
            self.manual_move_stop_button.setEnabled(phase == "window3")
            self.manual_confirm_button.setProperty("active", phase == "window2")
            self.manual_move_stop_button.setProperty("active", phase == "window3")
            self._refresh_vr_button(self.manual_confirm_button)
            self._refresh_vr_button(self.manual_move_stop_button)

        def _update_manual_simulation_clock(self) -> None:
            phase = self.manual_simulation_phase
            if phase not in {"window1_wait", "window2", "window3"}:
                return
            remaining = max(
                0.0,
                float(self.manual_simulation_deadline_monotonic)
                - time.monotonic(),
            )
            pending = dict(self.manual_simulation_pending or {})
            class_id = int(pending.get("class_id", 0) or 0)
            intent = str(pending.get("intent", ""))
            label = CLASS_ARM_LABELS.get(class_id, intent)
            if phase == "window1_wait":
                text = (
                    f"窗口1倒计时 {remaining:.1f}s | "
                    f"已人工注入 C{class_id} {label}"
                )
            elif phase == "window2":
                required = max(
                    1, int(self.config.confirm_masseter_count_spin.value())
                )
                text = (
                    f"窗口2倒计时 {remaining:.1f}s | "
                    f"咬肌确认 {self.manual_confirmation_count}/{required}"
                )
                if remaining <= 0.0:
                    self.manual_simulation_token += 1
                    self.manual_simulation_pending = None
                    self.manual_confirmation_count = 0
                    self._set_manual_simulation_phase(
                        "window1",
                        "窗口2确认超时，未执行机械臂指令，返回窗口1",
                    )
                    self.gate_label.setText(
                        "人工窗口2确认超时 | 未执行 | 返回窗口1"
                    )
                    self._set_current_result_card(
                        "rejected", f"C{class_id} {label}", "确认超时，未执行"
                    )
                    self.append_log(
                        f"人工窗口2超时：C{class_id} {label} 未执行，返回窗口1。"
                    )
                    return
            else:
                required = max(
                    1, int(self.config.continuous_blink_count_spin.value())
                )
                if self.manual_stop_pending:
                    text = (
                        f"窗口3本段倒计时 {remaining:.1f}s | "
                        f"停止条件已满足（眨眼 {required}/{required}）| "
                        f"等待本窗口结束，机械臂继续{label}"
                    )
                else:
                    text = (
                        f"窗口3本段倒计时 {remaining:.1f}s | "
                        f"眨眼 {self.manual_blink_count}/{required} | 持续{label}"
                    )
                if remaining <= 0.0:
                    if self.manual_stop_pending:
                        self._complete_manual_simulated_stop()
                        return
                    window_seconds = max(
                        0.5,
                        float(self.config.continuous_blink_window_spin.value()),
                    )
                    self.manual_blink_count = 0
                    self.manual_simulation_deadline_monotonic = (
                        time.monotonic() + window_seconds
                    )
                    text = (
                        f"窗口3上一计时段未达到 {required} 次，计数清零；"
                        f"新计时段 {window_seconds:.1f}s | 眨眼 0/{required}"
                    )
                    self.append_log(
                        f"人工窗口3计时段结束：未达到 {required} 次眨眼，"
                        "计数清零并开始下一段，机械臂继续移动。"
                    )
            self.manual_simulation_label.setText(f"人工测试状态：{text}")

        def _manual_class_id(self, intent: str) -> int:
            return next(
                (
                    int(class_id)
                    for class_id, mapped_intent in CLASS_TO_ARM.items()
                    if mapped_intent == intent
                ),
                0,
            )

        def simulate_manual_classification(self, intent: str) -> None:
            """Inject one C1-C8 result and follow the real three-window workflow."""
            if self.timer.isActive():
                self.append_log(
                    "人工三窗口测试未启动：请先停止真实 LSL 在线解码。"
                )
                return
            if self.operation_start_time <= 0.0:
                self.start_manual_button_test()
            if self.manual_simulation_phase != "window1":
                self.append_log(
                    f"人工分类未注入：当前处于 {self.manual_simulation_phase}。"
                )
                return
            class_id = self._manual_class_id(intent)
            if class_id <= 0:
                self.append_log(f"人工分类不支持：{intent}")
                return
            window_seconds = max(
                0.1, float(self.config.decode_window_spin.value())
            )
            self.manual_simulation_token += 1
            token = self.manual_simulation_token
            self.manual_simulation_pending = {
                "class_id": class_id,
                "intent": intent,
                "started_at": time.time(),
            }
            self._set_active_intent(intent)
            self.manual_simulation_deadline_monotonic = (
                time.monotonic() + window_seconds
            )
            self.manual_simulation_timer.start()
            label = CLASS_ARM_LABELS.get(class_id, intent)
            self._set_manual_simulation_phase(
                "window1_wait",
                f"窗口1采集中——已人工注入 C{class_id} {label}，"
                f"{window_seconds:.1f}s 后输出分类结果",
            )
            self.gate_label.setText(
                f"当前状态：人工测试窗口1 | 待输出 C{class_id} {label}"
            )
            self.decode_result_label.setText(
                f"实时解码结果：窗口1采集中 | 人工输入 C{class_id} {label}"
            )
            self._set_current_result_card(
                "collect", f"C{class_id} {label}", "窗口1采集中"
            )
            self.append_log(
                f"人工测试窗口1开始：注入 C{class_id} {label}，"
                f"等待 {window_seconds:.1f}s。"
            )

            def finish_window1() -> None:
                if (
                    token != self.manual_simulation_token
                    or self.manual_simulation_phase != "window1_wait"
                    or not self.manual_simulation_pending
                ):
                    return
                decision = Decision(
                    time.time(),
                    intent,
                    1.0,
                    1.0,
                    None,
                    False,
                    "manual_window1_classified",
                    f"C{class_id}",
                    {f"C{class_id}_{label}": 1.0},
                )
                self.pending_record_emg_counts = {
                    "blink": 0,
                    "left_jaw": 0,
                    "right_jaw": 0,
                }
                self._append_decode_result(
                    decision,
                    f"人工 C{class_id}",
                    f"C{class_id} {label}",
                    "眨眼 0 | 左咬肌 0 | 右咬肌 0",
                )
                confirm_seconds = max(
                    0.2, float(self.config.confirm_window_spin.value())
                )
                self.manual_confirmation_count = 0
                self.manual_simulation_deadline_monotonic = (
                    time.monotonic() + confirm_seconds
                )
                self._set_manual_simulation_phase(
                    "window2",
                    f"窗口2确认——C{class_id} {label} 为窗口1待确认结果，"
                    f"{confirm_seconds:.1f}s 内点击确认按钮模拟咬肌",
                )
                self.gate_label.setText(
                    f"确认执行？C{class_id} {label} | 人工测试窗口2"
                )
                self.decode_result_label.setText(
                    f"实时解码结果：C{class_id} {label} | 等待窗口2确认"
                )
                self._set_current_result_card(
                    "collect", f"C{class_id} {label}", "等待窗口2确认"
                )
                self.append_log(
                    f"人工测试窗口1结束：分类为 C{class_id} {label}，进入窗口2。"
                )

            QtCore.QTimer.singleShot(
                int(round(window_seconds * 1000.0)), finish_window1
            )

        def confirm_manual_simulation(self) -> None:
            if (
                self.manual_simulation_phase != "window2"
                or not self.manual_simulation_pending
            ):
                return
            if time.monotonic() >= self.manual_simulation_deadline_monotonic:
                self._update_manual_simulation_clock()
                return
            class_id = int(self.manual_simulation_pending["class_id"])
            intent = str(self.manual_simulation_pending["intent"])
            label = CLASS_ARM_LABELS.get(class_id, intent)
            required_confirmations = max(
                1, int(self.config.confirm_masseter_count_spin.value())
            )
            self.manual_confirmation_count += 1
            if self.manual_confirmation_count < required_confirmations:
                remaining = max(
                    0.0,
                    self.manual_simulation_deadline_monotonic - time.monotonic(),
                )
                self._set_manual_simulation_phase(
                    "window2",
                    f"窗口2倒计时 {remaining:.1f}s | 咬肌确认 "
                    f"{self.manual_confirmation_count}/{required_confirmations}",
                )
                self.append_log(
                    f"人工窗口2：模拟咬肌确认 "
                    f"{self.manual_confirmation_count}/{required_confirmations}。"
                )
                return
            continuous = intent in CONTINUOUS_MOVE_VECTORS
            self.manual_command(intent, continuous=continuous)
            if continuous:
                window3_seconds = max(
                    0.5,
                    float(self.config.continuous_blink_window_spin.value()),
                )
                self.manual_blink_count = 0
                self.manual_stop_pending = False
                self.manual_simulation_deadline_monotonic = (
                    time.monotonic() + window3_seconds
                )
                self._set_manual_simulation_phase(
                    "window3",
                    f"窗口3持续移动——C{class_id} {label} 已执行；"
                    f"停止按钮模拟达到停止次数，本段 {window3_seconds:.1f}s",
                )
                self.gate_label.setText(
                    f"持续{label}中 | 人工测试窗口3等待3次眨眼停止"
                )
                self.decode_result_label.setText(
                    f"实时解码结果：C{class_id} {label} 已确认并持续执行"
                )
                self._set_current_result_card(
                    "accepted", f"C{class_id} {label}", "窗口3持续移动"
                )
                self.append_log(
                    f"人工窗口2确认：已发送 C{class_id} {label} move_start，"
                    "进入窗口3。"
                )
            else:
                self.manual_simulation_pending = None
                self.manual_confirmation_count = 0
                self.manual_simulation_deadline_monotonic = 0.0
                self._set_manual_simulation_phase(
                    "window1",
                    f"C{class_id} {label} 已单次执行，返回窗口1",
                )
                self.gate_label.setText(
                    f"C{class_id} {label} 已确认执行 | 返回人工测试窗口1"
                )
                self.decode_result_label.setText(
                    f"实时解码结果：C{class_id} {label} 已执行"
                )
                self._set_current_result_card(
                    "accepted", f"C{class_id} {label}", "已执行并返回窗口1"
                )
                self.append_log(
                    f"人工窗口2确认：已发送 C{class_id} {label}，返回窗口1。"
                )

        def stop_manual_simulated_motion(self) -> None:
            if self.manual_simulation_phase != "window3":
                return
            pending = dict(self.manual_simulation_pending or {})
            class_id = int(pending.get("class_id", 0) or 0)
            intent = str(pending.get("intent", ""))
            label = CLASS_ARM_LABELS.get(class_id, intent or "移动")
            required_blinks = max(
                1, int(self.config.continuous_blink_count_spin.value())
            )
            # 一次点击表示本段已达到配置次数；到本计时段边界才停。
            self.manual_blink_count = required_blinks
            self.manual_stop_pending = True
            remaining = max(
                0.0,
                self.manual_simulation_deadline_monotonic - time.monotonic(),
            )
            self.manual_move_stop_button.setEnabled(False)
            self.manual_simulation_label.setText(
                "人工测试状态："
                f"窗口3停止条件已满足（眨眼 {required_blinks}/{required_blinks}）| "
                f"等待当前窗口结束 {remaining:.1f}s"
            )
            self.gate_label.setText(
                f"窗口3已达到{required_blinks}次眨眼 | "
                f"等待本窗口结束后停止{label}"
            )
            self.decode_result_label.setText(
                f"实时解码结果：停止条件已满足 | "
                f"窗口结束后发送 move_stop"
            )
            self._set_current_result_card(
                "collect",
                f"等待窗口结束 {remaining:.1f}s",
                f"眨眼 {required_blinks}/{required_blinks}",
            )
            self.append_log(
                f"人工窗口3：已模拟达到{required_blinks}次眨眼；"
                f"不立即停止，等待当前窗口剩余 {remaining:.1f}s。"
            )

        def _complete_manual_simulated_stop(self) -> None:
            if (
                self.manual_simulation_phase != "window3"
                or not self.manual_stop_pending
            ):
                return
            pending = dict(self.manual_simulation_pending or {})
            class_id = int(pending.get("class_id", 0) or 0)
            intent = str(pending.get("intent", ""))
            label = CLASS_ARM_LABELS.get(class_id, intent or "移动")
            required_blinks = max(
                1, int(self.config.continuous_blink_count_spin.value())
            )
            self.bus.command_emitted.emit(
                {
                    "cmd": "stop_motion",
                    "intent": "stop_motion",
                    "target": self._runtime_target(),
                    "confidence": 1.0,
                    "sequence_id": 0,
                    "timestamp": time.time(),
                    "duration_seconds": 0.0,
                }
            )
            self.continuous_move_intent = None
            self.continuous_move_started_at = 0.0
            self.pending_record_emg_counts = {
                "blink": required_blinks,
                "left_jaw": 0,
                "right_jaw": 0,
            }
            self.manual_simulation_pending = None
            self.manual_blink_count = 0
            self.manual_stop_pending = False
            self.manual_simulation_deadline_monotonic = 0.0
            self._set_manual_simulation_phase(
                "window1",
                f"窗口3检测到{required_blinks}次眨眼，{label}已停止，返回窗口1",
            )
            self.gate_label.setText(
                f"窗口3达到{required_blinks}次眨眼：{label}已停止 | 返回窗口1"
            )
            self.decode_result_label.setText(
                f"实时解码结果：move_stop | 模拟眨眼 {required_blinks} 次"
            )
            self._set_current_result_card(
                "accepted",
                "移动已停止",
                f"窗口3眨眼 {required_blinks} 次",
            )
            self.append_log(
                f"人工窗口3：达到{required_blinks}次眨眼，已发送 move_stop，"
                f"停止{label}并返回窗口1。"
            )

        def _translation_axis_and_sign(self, intent: str) -> Tuple[str, float]:
            return {
                "forward": ("x", 1.0),
                "backward": ("x", -1.0),
                "left": ("y", 1.0),
                "right": ("y", -1.0),
                "up": ("z", 1.0),
                "down": ("z", -1.0),
            }[intent]

        def _observe_translation_command(self, payload: Dict[str, object]) -> None:
            """Keep the open-loop estimate aligned with every shared command path."""
            if not isinstance(payload, dict):
                return
            intent = str(payload.get("intent", ""))
            command = str(payload.get("cmd", ""))
            if command == "stop_motion" or intent == "stop_motion":
                self._stop_translation_limit_tracking()
            elif command == "home" or intent == "home":
                self._stop_translation_limit_tracking()
                self.translation_position_mm = {"x": 0.0, "y": 0.0, "z": 0.0}
                self._refresh_translation_limit_status()
            elif bool(payload.get("continuous")) and intent in CONTINUOUS_MOVE_VECTORS:
                self._start_translation_limit_tracking(intent)

        def _translation_bounds_mm(self, axis: str) -> Tuple[float, float]:
            direction_pair = {
                "x": ("backward", "forward"),
                "y": ("right", "left"),
                "z": ("down", "up"),
            }[axis]
            return (
                -float(self.translation_limit_spins[direction_pair[0]].value()),
                float(self.translation_limit_spins[direction_pair[1]].value()),
            )

        def _translation_can_start(self, intent: str) -> bool:
            if intent not in CONTINUOUS_MOVE_VECTORS:
                return True
            axis, sign = self._translation_axis_and_sign(intent)
            current = float(self.translation_position_mm[axis])
            minimum, maximum = self._translation_bounds_mm(axis)
            allowed = current < maximum - 1e-6 if sign > 0 else current > minimum + 1e-6
            if not allowed:
                self.append_log(
                    f"平移软限位：{intent} 未发送；{axis.upper()}={current:.1f} mm，"
                    f"允许范围 {minimum:.1f}～{maximum:.1f} mm。"
                )
            return allowed

        def _start_translation_limit_tracking(self, intent: str) -> None:
            self._stop_translation_limit_tracking()
            self.translation_tracking_intent = intent
            self.translation_tracking_started_at = time.monotonic()
            self.translation_limit_timer.start()

        def _stop_translation_limit_tracking(self) -> None:
            if self.translation_tracking_intent:
                self._integrate_translation_position(time.monotonic())
            self.translation_tracking_intent = None
            self.translation_tracking_started_at = 0.0
            self.translation_limit_timer.stop()

        def _integrate_translation_position(self, now: float) -> None:
            intent = self.translation_tracking_intent
            if not intent or self.translation_tracking_started_at <= 0.0:
                return
            elapsed = max(0.0, float(now) - self.translation_tracking_started_at)
            axis, sign = self._translation_axis_and_sign(intent)
            speed_mm_s = max(0.0, float(self.online_robot_speed_spin.value()))
            minimum, maximum = self._translation_bounds_mm(axis)
            position = float(self.translation_position_mm[axis])
            position += sign * speed_mm_s * elapsed
            self.translation_position_mm[axis] = min(max(position, minimum), maximum)
            self.translation_tracking_started_at = float(now)
            self._refresh_translation_limit_status()

        def _update_translation_soft_limit(self) -> None:
            intent = self.translation_tracking_intent
            if not intent:
                self.translation_limit_timer.stop()
                return
            self._integrate_translation_position(time.monotonic())
            axis, sign = self._translation_axis_and_sign(intent)
            minimum, maximum = self._translation_bounds_mm(axis)
            current = float(self.translation_position_mm[axis])
            reached = current >= maximum - 1e-6 if sign > 0 else current <= minimum + 1e-6
            if not reached:
                return
            self.translation_tracking_intent = None
            self.translation_tracking_started_at = 0.0
            self.translation_limit_timer.stop()
            self.bus.command_emitted.emit(
                {"cmd": "stop_motion", "intent": "stop_motion"}
            )
            self.continuous_move_intent = None
            self.continuous_move_started_at = 0.0
            self.append_log(
                f"安全停止：{axis.upper()} 已到平移软限位 {current:.1f} mm，"
                "已立即发送 move_stop。"
            )
            if self.manual_simulation_phase == "window3":
                self.manual_simulation_pending = None
                self.manual_stop_pending = False
                self.manual_simulation_deadline_monotonic = 0.0
                self._set_manual_simulation_phase(
                    "window1", "到达平移软限位，已安全停止并返回窗口1"
                )

        def _refresh_translation_limit_status(self) -> None:
            if not hasattr(self, "translation_limit_status_label"):
                return
            position = self.translation_position_mm
            self.translation_limit_status_label.setText(
                f"X={position['x']:.1f} mm | Y={position['y']:.1f} mm | "
                f"Z={position['z']:.1f} mm（回零后为 0）"
            )

        def _refresh_orientation_limit_status(
            self, _axis: Optional[str] = None
        ) -> None:
            if not hasattr(self, "orientation_limit_status_label"):
                return
            parts = []
            for axis, label in [
                ("pitch", "Pitch"),
                ("yaw", "Yaw"),
                ("roll", "Roll"),
            ]:
                current = float(self.manual_orientation_degrees.get(axis, 0.0))
                minimum = float(self.orientation_min_spins[axis].value())
                maximum = float(self.orientation_max_spins[axis].value())
                state = (
                    "配置错误：下限大于上限"
                    if minimum > maximum
                    else f"{minimum:.2f}～{maximum:.2f}°"
                )
                parts.append(f"{label}={current:.2f}°（{state}）")
            self.orientation_limit_status_label.setText(" | ".join(parts))

        def execute_manual_orientation_values(self) -> None:
            """Convert signed RPY degree inputs to sequential receiver movements."""
            if not hasattr(self, "manual_orientation_spins"):
                return
            speed = max(
                0.01, float(self.online_robot_rotation_speed_spin.value())
            )
            angular_speed = speed * 0.005  # Manipulator-communication: rad/s
            moves = []
            for axis in ("pitch", "yaw", "roll"):
                requested = float(self.manual_orientation_spins[axis].value())
                if abs(requested) < 1e-9:
                    continue
                current = float(self.manual_orientation_degrees.get(axis, 0.0))
                minimum = float(self.orientation_min_spins[axis].value())
                maximum = float(self.orientation_max_spins[axis].value())
                if minimum > maximum:
                    self.append_log(
                        f"{axis.title()} 软限位配置错误：下限 {minimum:.2f}° "
                        f"大于上限 {maximum:.2f}°，已拒绝本次姿态运动。"
                    )
                    continue
                if requested > 0.0:
                    allowed = min(requested, max(0.0, maximum - current))
                else:
                    allowed = max(requested, min(0.0, minimum - current))
                target = current + allowed
                if abs(allowed) < 1e-9:
                    boundary = "上限" if requested > 0 else "下限"
                    self.append_log(
                        f"{axis.title()} 已达到软件{boundary}："
                        f"当前 {current:.2f}°，范围 {minimum:.2f}～{maximum:.2f}°，"
                        "未发送指令。"
                    )
                    continue
                if not math.isclose(allowed, requested, abs_tol=1e-9):
                    self.append_log(
                        f"{axis.title()} 输入 {requested:+.2f}° 超出软限位，"
                        f"已 Clamp 为 {allowed:+.2f}°，目标 {target:.2f}°。"
                    )
                direction = "increase" if allowed > 0 else "decrease"
                duration = math.radians(abs(allowed)) / angular_speed
                moves.append(
                    (f"{axis}_{direction}", axis, allowed, duration, target)
                )
            if not moves:
                self.append_log("没有可执行的姿态数值，未发送机械臂指令。")
                return
            self.orientation_apply_button.setEnabled(False)

            def send_next(index: int = 0) -> None:
                if index >= len(moves):
                    for field in self.manual_orientation_spins.values():
                        field.setValue(0.0)
                    self.orientation_apply_button.setEnabled(True)
                    self.append_log("Pitch/Yaw/Roll 姿态数值执行完成。")
                    return
                intent, axis, degrees, duration, target = moves[index]
                self.manual_command(
                    intent,
                    duration_seconds=duration,
                    continuous=False,
                    protocol_speed=speed,
                )
                self.append_log(
                    f"姿态微调：{intent} {degrees:+.2f}°，"
                    f"speed={speed:.2f}，预计 {duration:.2f}s"
                )

                def finish_and_continue() -> None:
                    self.manual_orientation_degrees[axis] = float(target)
                    self._refresh_orientation_limit_status(axis)
                    send_next(index + 1)

                QtCore.QTimer.singleShot(
                    max(50, int(round((duration + 0.15) * 1000.0))),
                    finish_and_continue,
                )

            send_next()

        def _transport_demo_steps(self) -> List[Tuple[str, str, float, Optional[List[float]], Optional[int]]]:
            s = float(self.transport_demo_step_seconds)
            return [
                ("home", "静息态：等待三次眨眼切换，场景回零", s, None, None),
                ("gate_switch", "状态切换：连续眨眼三次进入八分类控制态", s, None, None),
                ("forward", "C3 前移：X+ 一格，对齐右下方块 X=5", s, None, None),
                ("right", "C2 右移：Y- 一格，对齐右下方块 Y=-1", s, None, None),
                ("down", "C6 下移：Z- 一格，下降到抓取高度 Z=2", s, None, None),
                ("grab", "C7 抓取：抓取右下方块", s, None, None),
                ("up", "C5 上移：Z+ 一格，抬起方块 Z=3", s, None, None),
                ("backward", "C4 后移：X- 一格，回到中心 X=4", s, None, None),
                ("left", "C1 左移：Y+ 一格，回到中心 Y=0", s, None, None),
                ("backward", "C4 后移：X- 一格，到左下放置点 X=3", s, None, None),
                ("left", "C1 左移：Y+ 一格，到左下放置点 Y=1", s, None, None),
                ("release", "C8 释放：方块完整放置到左下托盘中心", s, None, 1),
            ]

        def start_transport_demo(self) -> None:
            if self.transport_demo_running:
                return
            self.pending_start_mode = None
            self.timer.stop()
            if self.source is not None and hasattr(self.source, "close"):
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭在线源失败：{exc}")
            self.source = None
            self.transport_demo_running = True
            self.transport_demo_index = 0
            self.control_gate_state = "rest"
            self._sync_gate_toggle_button()
            self.gate_hold_start = None
            self.phase_emg_events.reset("demo", time.time())
            self._set_last_window_emg_counts(0, 0, 0)
            self.operation_start_time = time.perf_counter()
            self._update_operation_elapsed()
            self.operation_elapsed_timer.start(100)
            self.gate_label.setText("当前状态：静息态 | 门控：眨眼三次切换")
            self._update_fullscreen_state("当前状态：静息态 | 门控：等待眨眼三次")
            self._update_transport_counter(0, len(self._transport_demo_steps()), "待开始", "待开始")
            self.append_log("机械臂搬运启动：每 5.0s 输出一个结果；每个结果移动 1 格，动画 0.5s。")
            self._run_next_transport_step()

        def _run_next_transport_step(self) -> None:
            if not self.transport_demo_running:
                return
            steps = self._transport_demo_steps()
            if self.transport_demo_index >= len(steps):
                self.transport_demo_running = False
                self.operation_elapsed_timer.stop()
                self._update_operation_elapsed()
                self._set_active_intent(None)
                self.gate_label.setText("当前状态：静息态 | 搬运完成，回到静息待命")
                self._sync_gate_toggle_button()
                self._update_fullscreen_state("当前状态：静息态 | 搬运完成，回到静息待命")
                self._update_transport_counter(len(steps), len(steps), "完成", "完成")
                self.status_label.setText("状态：搬运完成 | 方块：右下角 -> 中心 -> 左下角")
                if hasattr(self, "decode_result_label"):
                    self.decode_result_label.setText("实时解码结果：搬运完成 | 解码状态：完成")
                self.append_log("机械臂搬运完成。")
                return
            intent, label, duration, target_position, drop_zone_index = steps[self.transport_demo_index]
            step_no = self.transport_demo_index + 1
            decode_state = self._transport_decode_state(intent)
            self._update_transport_counter(step_no, len(steps), label, decode_state)
            if hasattr(self, "decode_result_label"):
                self.decode_result_label.setText(
                    f"实时解码结果：当前第 {step_no} 步 | 解码状态：{decode_state}"
                )
            self.status_label.setText(
                f"步骤 {step_no}/{len(steps)}：{label} | 解码状态：{decode_state} | 分类窗口 5.0s，每5.0s输出一次，每个结果移动 1 格"
            )
            self.append_log(f"步骤 {step_no}/{len(steps)}：{label}")
            if intent == "gate_switch":
                self._set_active_intent(None)
                self.control_gate_state = "active"
                self._sync_gate_toggle_button()
                self.gate_label.setText("当前状态：运动控制态 | 门控完成，八分类动作输出已解锁")
                self._update_fullscreen_state("当前状态：运动控制态 | 门控完成，八分类动作输出已解锁")
            elif intent == "home" and step_no == 1:
                self._set_active_intent(None)
                self.gate_label.setText("当前状态：静息态 | 门控：等待眨眼三次")
                self._update_fullscreen_state("当前状态：静息态 | 门控：等待眨眼三次")
                self.manual_command(intent, duration, target_position, drop_zone_index, continuous=False)
            else:
                self.gate_label.setText("当前状态：运动控制态 | 八分类动作输出中")
                action_text = {
                    "left": "C1 左移",
                    "right": "C2 右移",
                    "forward": "C3 前移",
                    "backward": "C4 后移",
                    "up": "C5 上移",
                    "down": "C6 下移",
                    "grab": "C7 抓取",
                    "release": "C8 释放",
                    "home": "回零",
                }.get(intent, intent)
                self._update_fullscreen_state(f"当前状态：运动控制态 | 当前输出：{action_text}")
                self.manual_command(intent, duration, target_position, drop_zone_index, continuous=False)
            self.transport_demo_index += 1
            QtCore.QTimer.singleShot(int(self.transport_demo_step_seconds * 1000), self._run_next_transport_step)

        def closeEvent(self, event) -> None:
            self.stop_online()
            super().closeEvent(event)

    class VRRealtimeControlPage(VRRealtimeControlBasePage):
        def __init__(self, bus: CommandBus, config: ConfigPage, parent=None):
            super().__init__(bus, config, parent)
            self.is_test_page = True
            self.title_label.setText("实时脑电机械臂控制")
            self.desc_label.setText(
                "接入 BrainVision LSL 后，每 2s 读取 EEG 窗口并进入 1s AUX 确认；EEG 输出 C1-C8 概率，"
                "AUX1/2 三次眨眼用于阶段切换，AUX3/4 左右咬肌计数用于候选门控。"
            )
            self.title_label.setVisible(False)
            self.desc_label.setVisible(False)
            self.demo_button.setVisible(False)
            self.demo_card.setVisible(False)
            self.pure_emg_control_check.setVisible(False)
            self.step_counter_label.setVisible(False)
            self.demo_plan.setPlainText(
                "实时参数：2.0s EEG 解码后进入 1.0s 肌电确认；确认后才执行或返回下一轮。\n"
                "AUX1/2：左眼/右眼，仅用于连续眨眼三次切换阶段。\n"
                "AUX3/4：左/右咬肌计数形成 00/01/10/11；候选映射为 00=C3/C4，01=C1/C2，10=C5/C6，11=C7/C8。\n"
                "静息态不输出八分类动作；进入运动控制态后只输出 C1-C8 八分类动作。\n"
                "咬肌不改写 8 类 softmax；系统按概率排序，输出第一个命中当前映射表的类别。"
            )
            self.startup_progress.setFormat("离线预览 | 按回车或空格订阅系统 LSL")
            self.gate_label.setText("当前状态：离线预览 | BCI 控制未启动")
            self.gate_counter_label.setText("当前时间段计数：-- | AUX：未连接")
            self.decode_result_label.setText("实时解码结果：未开始（LSL 未连接）")
            self._set_current_result_card("collect", "未连接", "等待 BrainVision LSL 数据流")
            self.status_label.setText("状态：未连接 BrainVision LSL | 机械臂手动演示可用")
            self._online_input_mode_changed()

        def _validate_eeg8_model(self) -> None:
            super()._validate_eeg8_model()
            if self.model is None:
                raise RuntimeError("模型未加载")
            kind = str(getattr(self.model, "classifier_kind", "")).lower()
            if kind and kind not in {"lda", "fbcsp_lda", "fbcsp-lda", "vote", "vote-ml-best", "voting", "logreg", "svm", "rbf_svm", "extra_trees", "random_forest", "gaussian_nb"}:
                self.append_log(f"提示：当前权重分类器为 {kind}，test 将按模型自身配置解码。")
            csp_models = list(getattr(self.model.eeg_features, "csp_models", []) or [])
            if not any(model is not None for model in csp_models):
                self.append_log("提示：当前权重没有可用 CSP 子模型，将使用权重内已有 EEG 特征/分类器实时解码。")

        def start_online(self) -> None:
            self.start_realtime_online(force_lsl=True)

    class LslFourEmgSource:
        def __init__(
            self,
            timeout: float = 8.0,
            expected_eeg_channels: Optional[Sequence[str]] = None,
        ):
            try:
                import pylsl
            except ImportError as exc:
                raise RuntimeError("pylsl 未安装，请在当前环境运行：python -m pip install pylsl") from exc

            self.pylsl = pylsl
            self.stream_name = ""
            self.stream_type = "EEG"
            self.emg_start = 32
            self.indices: List[int] = []
            self.timeout = float(timeout)
            self.inlet = None
            self.info = None
            self.fs = 250.0
            self.channel_count = 0
            self.channel_names: List[str] = []
            self.reported_channel_names: List[str] = []
            self.expected_eeg_channels = (
                tuple(expected_eeg_channels) if expected_eeg_channels is not None else None
            )
            self.last_full_samples: Optional[np.ndarray] = None
            self.last_full_timestamps: Optional[np.ndarray] = None
            self._connect()

        def _resolve_all_streams(self) -> List[object]:
            try:
                return list(self.pylsl.resolve_streams(wait_time=self.timeout))
            except TypeError:
                return list(self.pylsl.resolve_streams(self.timeout))

        def _resolve_byprop(self, prop: str, value: str):
            try:
                return list(self.pylsl.resolve_byprop(prop, value, minimum=1, timeout=self.timeout))
            except AttributeError:
                return list(self.pylsl.resolve_stream(prop, value, timeout=self.timeout))

        def _choose_stream(self, streams: List[object]) -> object:
            if not streams:
                raise RuntimeError(
                    "未发现任何 LSL 流。请确认 BrainVision LSL Connector 已点击 Link，"
                    "并且窗口标题显示正在从放大器发送数据。"
                )

            def score(info) -> int:
                name = str(info.name()).lower()
                stream_type = str(info.type()).lower()
                source_id = str(info.source_id()).lower()
                value = 0
                if stream_type == "eeg":
                    value += 100
                if "liveamp" in name or "liveamp" in source_id:
                    value += 50
                if int(info.channel_count()) >= 36:
                    value += 20
                if "brainvision" in name:
                    value += 10
                return value

            return sorted(streams, key=score, reverse=True)[0]

        def _connect(self) -> None:
            streams = self._resolve_all_streams()
            if not streams:
                streams = self._resolve_byprop("type", self.stream_type)
            self.info = self._choose_stream(streams)
            processing_flags = int(getattr(self.pylsl, "proc_clocksync", 0)) | int(
                getattr(self.pylsl, "proc_dejitter", 0)
            )
            try:
                self.inlet = self.pylsl.StreamInlet(
                    self.info,
                    max_buflen=30,
                    processing_flags=processing_flags,
                )
            except TypeError:
                self.inlet = self.pylsl.StreamInlet(self.info, max_buflen=30)
            try:
                # Resolver results may omit the XML channel description.  Ask
                # the inlet for the full StreamInfo before reading labels.
                try:
                    full_info = self.inlet.info(timeout=self.timeout)
                except TypeError:
                    full_info = self.inlet.info(self.timeout)
                if full_info is not None:
                    self.info = full_info
                self.channel_count = int(self.info.channel_count())
                self.indices = brainvision_lsl_aux_indices(self.channel_count)
                # EEG-only streams have no AUX indices.
                self.emg_start = self.indices[0] if self.indices else self.channel_count
                self.fs = float(self.info.nominal_srate() or 250.0)
                self.stream_name = str(self.info.name())
                self.stream_type = str(self.info.type())
                self.reported_channel_names = self._read_channel_labels()
                self.channel_names = self._mapped_channel_names(self.reported_channel_names)
                if self.expected_eeg_channels is not None:
                    validate_eeg8_online_channel_contract(
                        self.reported_channel_names, self.expected_eeg_channels
                    )
            except Exception:
                try:
                    self.inlet.close_stream()
                finally:
                    self.inlet = None
                raise

        def _read_channel_labels(self) -> List[str]:
            labels: List[str] = []
            try:
                node = self.info.desc().child("channels").child("channel")
                for idx in range(self.channel_count):
                    label = str(node.child_value("label") or "").strip()
                    labels.append(label or f"CH{idx + 1}")
                    node = node.next_sibling()
            except Exception:
                labels = []
            if len(labels) != self.channel_count:
                labels = [f"CH{idx + 1}" for idx in range(self.channel_count)]
            return labels

        def _mapped_channel_names(self, reported: Sequence[str]) -> List[str]:
            """Keep LSL labels verbatim for recordings and decoder checks.

            A presentation page may render friendly AUX labels from physical
            indices, but it must never replace missing EEG metadata with the
            default montage.  ``channel_names`` is persisted into calibration
            recordings, so changing it would corrupt the training contract.
            """
            return [str(name) for name in reported]

        def clock_now(self) -> float:
            return float(self.pylsl.local_clock())

        @property
        def description(self) -> str:
            if self.info is None:
                return "-"
            selected = f"{self.indices[0] + 1}-{self.indices[-1] + 1}" if self.indices else "-"
            return (
                f"{self.info.name()} | type={self.info.type()} | "
                f"channels={self.channel_count} | fs={self.fs:g}Hz | EEG=1-{min(32, self.channel_count)} | AUX={selected}"
            )

        def read_all_available(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            if self.inlet is None:
                return None, None
            chunks = []
            timestamps = []
            while True:
                samples, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=512)
                if not samples or not ts:
                    break
                arr = np.asarray(samples, dtype=np.float32)
                chunks.append(arr)
                timestamps.append(np.asarray(ts, dtype=np.float64))
            if not chunks:
                return None, None
            self.last_full_samples = np.vstack(chunks)
            self.last_full_timestamps = np.concatenate(timestamps)
            return self.last_full_samples, self.last_full_timestamps

        def read_available(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            samples, timestamps = self.read_all_available()
            if samples is None or timestamps is None:
                return None, None
            return samples[:, self.indices], timestamps

        def close(self) -> None:
            inlet = self.inlet
            self.inlet = None
            if inlet is not None:
                try:
                    inlet.close_stream()
                except Exception:
                    pass

    class SystemLslLease:
        def __init__(self, manager: "SystemLslManager", owner: str):
            self._manager = manager
            self._owner = str(owner or "page")
            self._cursor = manager.seq_end
            self._closed = False

        def __getattr__(self, name: str):
            return getattr(self._manager.source, name)

        def read_all_available(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            return self._manager.read_all_available(self)

        def read_available(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            samples, timestamps = self.read_all_available()
            if samples is None or timestamps is None:
                return None, None
            return samples[:, self.indices], timestamps

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            self._manager.release(self)

    class SystemLslManager:
        """One system-level BrainVision LSL inlet with per-page read cursors."""

        def __init__(self):
            self.source: Optional[LslFourEmgSource] = None
            self.lock = threading.RLock()
            self.connect_lock = threading.Lock()
            self.closing = False
            self.leases: List[SystemLslLease] = []
            self.samples = np.zeros((0, 0), dtype=np.float32)
            self.timestamps = np.zeros((0,), dtype=np.float64)
            self.seq_start = 0
            self.seq_end = 0
            self.max_rows = 250 * 120

        def acquire(
            self,
            timeout: float = 8.0,
            owner: str = "page",
            expected_eeg_channels: Optional[Sequence[str]] = None,
        ) -> SystemLslLease:
            with self.connect_lock:
                with self.lock:
                    existing = self.source if self.source is not None and self.source.inlet is not None else None
                    if self.closing:
                        raise RuntimeError("系统正在退出，已取消 LSL 订阅")
                if existing is None:
                    new_source = LslFourEmgSource(
                        timeout=timeout,
                        expected_eeg_channels=expected_eeg_channels,
                    )
                    with self.lock:
                        if self.closing:
                            new_source.close()
                            raise RuntimeError("系统正在退出，已取消 LSL 订阅")
                        if self.source is None or self.source.inlet is None:
                            self.source = new_source
                            self.samples = np.zeros((0, int(self.source.channel_count)), dtype=np.float32)
                            self.timestamps = np.zeros((0,), dtype=np.float64)
                            self.seq_start = 0
                            self.seq_end = 0
                            self.max_rows = max(int(float(self.source.fs or 250.0) * 180), 4096)
                        else:
                            new_source.close()
                with self.lock:
                    if self.source is None or self.source.inlet is None:
                        raise RuntimeError("系统 LSL 连接不可用")
                    if expected_eeg_channels is not None:
                        # Recheck labels when reusing the shared inlet.
                        validate_eeg8_online_channel_contract(
                            self.source.reported_channel_names, expected_eeg_channels
                        )
                lease = SystemLslLease(self, owner)
                self.leases.append(lease)
                return lease

        def is_connected(self) -> bool:
            with self.lock:
                return self.source is not None and self.source.inlet is not None

        def _drain_locked(self) -> None:
            if self.source is None:
                return
            samples, timestamps = self.source.read_all_available()
            if samples is None or timestamps is None:
                return
            samples = np.asarray(samples, dtype=np.float32)
            timestamps = np.asarray(timestamps, dtype=np.float64)
            if samples.ndim != 2 or timestamps.ndim != 1 or samples.shape[0] != timestamps.shape[0]:
                return
            if self.samples.size:
                self.samples = np.vstack([self.samples, samples])
                self.timestamps = np.concatenate([self.timestamps, timestamps])
            else:
                self.samples = samples
                self.timestamps = timestamps
            self.seq_end += int(samples.shape[0])
            if self.samples.shape[0] > self.max_rows:
                drop = int(self.samples.shape[0] - self.max_rows)
                self.samples = self.samples[drop:, :]
                self.timestamps = self.timestamps[drop:]
                self.seq_start += drop

        def read_all_available(self, lease: SystemLslLease) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            with self.lock:
                self._drain_locked()
                if lease._cursor < self.seq_start:
                    lease._cursor = self.seq_start
                start = int(lease._cursor - self.seq_start)
                end = int(self.seq_end - self.seq_start)
                if end <= start:
                    return None, None
                samples = self.samples[start:end, :].copy()
                timestamps = self.timestamps[start:end].copy()
                lease._cursor = self.seq_end
                return samples, timestamps

        def release(self, lease: SystemLslLease) -> None:
            with self.lock:
                self.leases = [item for item in self.leases if item is not lease]

        def close_system(self) -> None:
            with self.lock:
                self.closing = True
                source = self.source
                self.source = None
                self.leases.clear()
                self.samples = np.zeros((0, 0), dtype=np.float32)
                self.timestamps = np.zeros((0,), dtype=np.float64)
                self.seq_start = 0
                self.seq_end = 0
            if source is not None:
                source.close()

    _SYSTEM_LSL_MANAGER = SystemLslManager()

    def acquire_system_lsl_source(
        timeout: float = 8.0,
        owner: str = "page",
        expected_eeg_channels: Optional[Sequence[str]] = None,
    ) -> SystemLslLease:
        return _SYSTEM_LSL_MANAGER.acquire(
            timeout=timeout,
            owner=owner,
            expected_eeg_channels=expected_eeg_channels,
        )

    class LslAcquireWorker(QtCore.QThread):
        progress_signal = QtCore.pyqtSignal(int, str)
        ready_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(self, owner: str, timeout: float = 8.0, parent=None):
            super().__init__(parent)
            self.owner = owner
            self.timeout = float(timeout)

        def run(self) -> None:
            source = None
            try:
                already_connected = _SYSTEM_LSL_MANAGER.is_connected()
                self.progress_signal.emit(10, "准备订阅系统 LSL")
                self.progress_signal.emit(35, "复用系统 LSL" if already_connected else "首次搜索系统 EEG 数据流")
                source = acquire_system_lsl_source(timeout=self.timeout, owner=self.owner)
                if self.isInterruptionRequested():
                    source.close()
                    return
                self.progress_signal.emit(85, "读取通道信息")
                self.progress_signal.emit(100, "已订阅系统 LSL")
                self.ready_signal.emit(source)
                source = None
            except Exception as exc:
                if not self.isInterruptionRequested():
                    self.error_signal.emit(str(exc))
            finally:
                if source is not None:
                    try:
                        source.close()
                    except Exception:
                        pass

    class LslEeg8OnlineSource(SignalSource):
        """BrainVision LSL online source for EEG8 VR control.

        The first 32 EEG channels are used for classification. LSL channels
        33-36 are returned as AUX1-4 for blink/masseter gating.
        """

        def __init__(self, cfg: SignalConfig, log_callback=None, require_masseter_gate: bool = True):
            self.cfg = cfg
            self.log_callback = log_callback
            self.source = acquire_system_lsl_source(
                timeout=8.0,
                owner="online_eeg8",
                expected_eeg_channels=cfg.eeg_channels,
            )
            self.expected_channels = 32
            try:
                if int(self.source.channel_count) < self.expected_channels:
                    raise RuntimeError(
                        f"LSL 通道不足：EEG8 在线解码需要至少 {self.expected_channels} 路 EEG，"
                        f"当前 {self.source.channel_count} 路。"
                    )
                if abs(float(self.source.fs) - float(cfg.fs)) > 0.1:
                    raise RuntimeError(
                        f"LSL 采样率与模型不一致：LSL={self.source.fs:g}Hz，模型={cfg.fs:g}Hz。"
                        "请使用相同采样率采集/训练，系统不会静默重采样。"
                    )
                self.buffer = np.zeros((0, self.expected_channels), dtype=np.float32)
                self.aux_indices = [idx for idx in self.source.indices[:4] if idx < int(self.source.channel_count)]
                expected_aux_indices = [32, 33, 34, 35]
                if require_masseter_gate and self.aux_indices != expected_aux_indices:
                    available = ", ".join(f"AUX{idx - 31}" for idx in self.aux_indices) or "无"
                    raise RuntimeError(
                        "当前实时策略需要 AUX1-AUX4（AUX3/AUX4 为左右咬肌），"
                        f"但仅检测到 {available}。请补齐 32 EEG + 4 AUX，或显式选择 EEG-only 策略。"
                    )
                self.aux_buffer = np.zeros((0, len(self.aux_indices)), dtype=np.float32)
                self.timestamps = np.zeros((0,), dtype=np.float64)
                self.max_rows = max(int(cfg.window_samples * 6), int(cfg.window_samples + cfg.fs * 5))
                self.last_window_emit_ts = 0.0
                self.last_new_sample_monotonic = 0.0
            except Exception:
                self.source.close()
                raise
            if self.log_callback:
                self.log_callback(f"LSL 在线源：{self.source.description}")
                aux_text = ",".join(str(idx + 1) for idx in self.aux_indices) or "-"
                self.log_callback(f"在线 AUX 门控通道：{aux_text}（AUX1/2=眨眼，AUX3/4=咬肌）")

        def _drain_available(self) -> None:
            chunk, ts = self.source.read_all_available()
            if chunk is not None and ts is not None:
                eeg = np.asarray(chunk[:, : self.expected_channels], dtype=np.float32)
                aux = (
                    np.asarray(chunk[:, self.aux_indices], dtype=np.float32)
                    if self.aux_indices
                    else np.zeros((chunk.shape[0], 0), dtype=np.float32)
                )
                self.buffer = np.vstack([self.buffer, eeg]) if self.buffer.size else eeg
                self.aux_buffer = np.vstack([self.aux_buffer, aux]) if self.aux_buffer.size else aux
                self.timestamps = np.concatenate([self.timestamps, np.asarray(ts, dtype=np.float64)])
                if chunk.shape[0] > 0:
                    self.last_new_sample_monotonic = time.monotonic()
                if self.buffer.shape[0] > self.max_rows:
                    self.buffer = self.buffer[-self.max_rows :, :]
                    self.aux_buffer = self.aux_buffer[-self.max_rows :, :]
                    self.timestamps = self.timestamps[-self.max_rows :]

        def read_window(self) -> Tuple[np.ndarray, Optional[np.ndarray], float, Optional[str]]:
            self._drain_available()
            need = int(self.cfg.window_samples)
            if self.buffer.shape[0] < need:
                raise BlockingIOError("等待足够的 LSL EEG 样本")
            now = float(self.timestamps[-1]) if self.timestamps.size else time.time()
            step_seconds = max(0.1, float(getattr(self.cfg, "step_seconds", 2.0) or 2.0))
            if self.last_window_emit_ts > 0.0 and now - self.last_window_emit_ts < step_seconds:
                raise BlockingIOError(f"等待下一个 {step_seconds:g}s 解码窗口")
            self.last_window_emit_ts = now
            eeg_window = self.buffer[-need:, :].T.astype(np.float32)
            aux_window = self.aux_buffer[-need:, :].T.astype(np.float32) if self.aux_buffer.shape[1] else None
            return eeg_window, aux_window, now, None

        def seconds_until_next_window(self) -> float:
            self._drain_available()
            need = int(self.cfg.window_samples)
            fs = max(1.0, float(getattr(self.cfg, "fs", 250) or 250))
            step_seconds = max(0.1, float(getattr(self.cfg, "step_seconds", 2.0) or 2.0))
            if self.buffer.shape[0] < need:
                return max(0.0, float(need - self.buffer.shape[0]) / fs)
            now = float(self.timestamps[-1]) if self.timestamps.size else time.time()
            if self.last_window_emit_ts <= 0.0:
                return 0.0
            return max(0.0, step_seconds - max(0.0, now - self.last_window_emit_ts))

        def begin_next_eeg_window(self, timestamp: float) -> None:
            """Start a fresh EEG window after the confirmation interval."""
            self.last_window_emit_ts = float(timestamp)

        def latest_aux_window(self, seconds: Optional[float] = None) -> Tuple[Optional[np.ndarray], float]:
            self._drain_available()
            if self.aux_buffer.shape[1] == 0 or self.aux_buffer.shape[0] == 0:
                return None, time.time()
            if seconds is None:
                rows = self.aux_buffer.shape[0]
            else:
                rows = min(self.aux_buffer.shape[0], max(1, int(round(float(seconds) * float(self.cfg.fs)))))
            now = float(self.timestamps[-1]) if self.timestamps.size else time.time()
            return self.aux_buffer[-rows:, :].T.astype(np.float32), now

        def latest_aux_window_since(self, started_at: float) -> Tuple[Optional[np.ndarray], float]:
            self._drain_available()
            if self.aux_buffer.shape[1] == 0 or self.aux_buffer.shape[0] == 0 or self.timestamps.size == 0:
                return None, time.time()
            start_index = int(np.searchsorted(self.timestamps, float(started_at), side="right"))
            if start_index >= self.aux_buffer.shape[0]:
                return None, float(self.timestamps[-1])
            return self.aux_buffer[start_index:, :].T.astype(np.float32), float(self.timestamps[-1])

        def latest_aux_window_between(
            self,
            started_at: float,
            ended_at: float,
        ) -> Tuple[Optional[np.ndarray], float]:
            """Return only AUX samples inside one fixed LSL-time interval."""
            self._drain_available()
            if self.aux_buffer.shape[1] == 0 or self.aux_buffer.shape[0] == 0 or self.timestamps.size == 0:
                return None, float(ended_at)
            start_index = int(np.searchsorted(self.timestamps, float(started_at), side="right"))
            end_index = int(np.searchsorted(self.timestamps, float(ended_at), side="right"))
            if end_index <= start_index:
                return None, float(ended_at)
            return self.aux_buffer[start_index:end_index, :].T.astype(np.float32), float(ended_at)

        def aux_sample_age_seconds(self, now_monotonic: Optional[float] = None) -> float:
            self._drain_available()
            now = time.monotonic() if now_monotonic is None else float(now_monotonic)
            if self.last_new_sample_monotonic <= 0.0:
                return float("inf")
            return max(0.0, now - float(self.last_new_sample_monotonic))

        def close(self) -> None:
            self.source.close()

    class EmgTestPage(QtWidgets.QWidget):
        CHANNEL_NAMES = ("Aux1", "Aux2", "Aux3", "Aux4")
        ROLE_OPTIONS = (
            ("左眼", "left_eye"),
            ("右眼", "right_eye"),
            ("左咬肌", "left_masseter"),
            ("右咬肌", "right_masseter"),
            ("仅显示", "display"),
            ("不使用", "unused"),
        )
        DEFAULT_ROLES = ("left_eye", "right_eye", "left_masseter", "right_masseter")
        ROLE_DISPLAY = {key: text for text, key in ROLE_OPTIONS}

        def __init__(self, parent=None):
            super().__init__(parent)
            self.source: Optional[LslFourEmgSource] = None
            self.connect_worker: Optional[QtCore.QThread] = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.samples = np.zeros((0, 4), dtype=np.float32)
            self.timestamps = np.zeros((0,), dtype=np.float64)
            self.eeg_samples = np.zeros((0, 32), dtype=np.float32)
            self.eeg_timestamps = np.zeros((0,), dtype=np.float64)
            self.fs = 250.0
            self.baseline_rms: Optional[np.ndarray] = None
            self.baseline_values: List[np.ndarray] = []
            self.baseline_until = 0.0
            self.blink_times: deque = deque()
            self.blink_active = False
            self.test_emg_events: Optional[PhaseEmgEventEngine] = None
            self.test_event_buffer_started_at = 0.0
            self.test_event_config: Optional[Tuple[object, ...]] = None
            self.hit_counts = {
                "blink3": 0,
                "both_masseter": 0,
                "left_masseter": 0,
                "right_masseter": 0,
            }
            self.held_result = ""
            self.hold_until = 0.0
            self.last_result = ""
            self.current_result = "静息/未识别"
            self.last_classify_ts = 0.0
            self.classify_step_seconds = 0.30
            self.last_sample_wall = 0.0
            self.arming_until = 0.0
            self.plot_page_seconds = 30.0
            self.plot_page_start_ts: Optional[float] = None
            self.eeg_plot_page_start_ts: Optional[float] = None
            self.last_plot_wall = 0.0
            self.plot_update_interval = 0.20
            self.raw_uv_per_row = 75.0
            self.channel_checks: List[QtWidgets.QCheckBox] = []
            self.role_combos: List[QtWidgets.QComboBox] = []
            self.channel_scale_spins: List[QtWidgets.QDoubleSpinBox] = []
            self.physical_labels: List[QtWidgets.QLabel] = []
            self.channel_value_labels: List[QtWidgets.QLabel] = []
            self.speaker = SpeechAnnouncer()

            outer_layout = QtWidgets.QVBoxLayout(self)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            page_scroll = QtWidgets.QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            page_content = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page_content)
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(10)
            page_scroll.setWidget(page_content)
            outer_layout.addWidget(page_scroll)

            title = QtWidgets.QLabel("EEG + 4通道肌电信号监测")
            title.setObjectName("title")
            layout.addWidget(title)
            desc = QtWidgets.QLabel("直接读取 BrainVision LiveAmp 的真实 LSL 数据：LSL 1-32 显示为 EEG，33-36 显示为 AUX/肌电。当前动作检测只使用四路肌电，但 32 路 EEG 会同步读取、显示并在范式采集时保存。")
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            layout.addWidget(desc)

            controls = Card()
            controls_layout = QtWidgets.QGridLayout(controls)
            controls_layout.setHorizontalSpacing(10)
            controls_layout.setVerticalSpacing(8)
            self.baseline_spin = QtWidgets.QDoubleSpinBox()
            self.baseline_spin.setRange(0.5, 5.0)
            self.baseline_spin.setSingleStep(0.5)
            self.baseline_spin.setValue(2.0)
            self.blink_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.blink_ratio_spin.setRange(1.2, 30.0)
            self.blink_ratio_spin.setSingleStep(0.2)
            self.blink_ratio_spin.setValue(4.0)
            self.bite_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.bite_ratio_spin.setRange(1.2, 30.0)
            self.bite_ratio_spin.setSingleStep(0.2)
            self.bite_ratio_spin.setValue(3.0)
            self.side_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.side_ratio_spin.setRange(1.0, 5.0)
            self.side_ratio_spin.setSingleStep(0.05)
            self.side_ratio_spin.setValue(1.35)
            self.side_ratio_spin.setToolTip("单侧咬肌判定需要一侧强度至少达到另一侧的这个倍数；值越大，左右区分越严格。")
            self.plot_seconds_spin = QtWidgets.QDoubleSpinBox()
            self.plot_seconds_spin.setRange(5.0, 120.0)
            self.plot_seconds_spin.setSingleStep(5.0)
            self.plot_seconds_spin.setValue(self.plot_page_seconds)
            self.plot_seconds_spin.setSuffix(" s")
            self.plot_seconds_spin.setToolTip("下方波形每页显示的时长；到达该时长后自动翻到下一页。")
            self.auto_scale_check = QtWidgets.QCheckBox("自动量程")
            self.auto_scale_check.setChecked(True)
            self.auto_scale_check.setToolTip("自动根据当前页原始信号幅度分别设置每个通道量程；关闭后使用每行的手动量程。")
            self.speech_check = QtWidgets.QCheckBox("语音播报")
            self.speech_check.setChecked(True)
            self.speech_check.setToolTip("识别到有效肌电分类时播报结果；静息/未识别不会重复播报。")
            self.filter_check = QtWidgets.QCheckBox("启用滤波")
            self.filter_check.setChecked(True)
            self.filter_check.setToolTip(
                "使用 6 阶 Butterworth 带通滤波显示波形和数值；事件计数始终使用与正式在线控制一致的基线归一化 AUX RMS。"
            )
            self.emg_low_spin = QtWidgets.QDoubleSpinBox()
            self.emg_high_spin = QtWidgets.QDoubleSpinBox()
            self.eog_low_spin = QtWidgets.QDoubleSpinBox()
            self.eog_high_spin = QtWidgets.QDoubleSpinBox()
            for spin in [self.emg_low_spin, self.emg_high_spin, self.eog_low_spin, self.eog_high_spin]:
                spin.setRange(0.1, 500.0)
                spin.setDecimals(1)
                spin.setSingleStep(1.0)
                spin.setSuffix(" Hz")
                spin.setMinimumWidth(110)
                spin.valueChanged.connect(self._filter_config_changed)
            self.emg_low_spin.setValue(EMG_BAND_LOW_HZ)
            self.emg_high_spin.setValue(EMG_BAND_HIGH_HZ)
            self.eog_low_spin.setValue(EOG_BAND_LOW_HZ)
            self.eog_high_spin.setValue(EOG_BAND_HIGH_HZ)
            self.filter_check.toggled.connect(self._filter_config_changed)

            self.auto_lsl_label = QtWidgets.QLabel("BrainVision LSL 自动发现：优先连接 LiveAmp / EEG 流；43 通道流读取 EEG 1-32 与 AUX 33-36，无模拟数据回退。")
            self.auto_lsl_label.setWordWrap(True)
            self.auto_lsl_label.setObjectName("hint")
            controls_layout.addWidget(self.auto_lsl_label, 0, 0, 1, 6)
            for col, text in enumerate(["物理输入", "启用", "当前位置/用途", "本通道量程", "", ""]):
                header = QtWidgets.QLabel(text)
                header.setObjectName("hint")
                controls_layout.addWidget(header, 1, col)
            for idx, name in enumerate(self.CHANNEL_NAMES):
                row = 2 + idx
                physical = QtWidgets.QLabel(f"{name} / LSL {33 + idx}")
                physical.setObjectName("hint")
                self.physical_labels.append(physical)
                check = QtWidgets.QCheckBox()
                check.setChecked(True)
                check.toggled.connect(self._channel_config_changed)
                self.channel_checks.append(check)
                role_combo = QtWidgets.QComboBox()
                for role_text, role_key in self.ROLE_OPTIONS:
                    role_combo.addItem(role_text, role_key)
                role_combo.setCurrentIndex(max(0, role_combo.findData(self.DEFAULT_ROLES[idx])))
                role_combo.currentIndexChanged.connect(self._channel_config_changed)
                self.role_combos.append(role_combo)
                scale_spin = QtWidgets.QDoubleSpinBox()
                scale_spin.setRange(1.0, 50000000.0)
                scale_spin.setDecimals(2)
                scale_spin.setSingleStep(25.0)
                scale_spin.setValue(self.raw_uv_per_row)
                scale_spin.setSuffix(" uV")
                scale_spin.setToolTip("只影响这一行原始波形的显示高度，不影响分类。")
                scale_spin.valueChanged.connect(self._update_raw_scale)
                self.channel_scale_spins.append(scale_spin)
                controls_layout.addWidget(physical, row, 0)
                controls_layout.addWidget(check, row, 1)
                controls_layout.addWidget(role_combo, row, 2)
                controls_layout.addWidget(scale_spin, row, 3, 1, 3)
            controls_layout.addWidget(QtWidgets.QLabel("基线秒数"), 6, 0)
            controls_layout.addWidget(self.baseline_spin, 6, 1)
            controls_layout.addWidget(QtWidgets.QLabel("眨眼阈值倍数"), 6, 2)
            controls_layout.addWidget(self.blink_ratio_spin, 6, 3)
            controls_layout.addWidget(QtWidgets.QLabel("咬肌阈值倍数"), 6, 4)
            controls_layout.addWidget(self.bite_ratio_spin, 6, 5)
            side_ratio_label = QtWidgets.QLabel("左右区分倍数")
            side_ratio_label.setToolTip("单侧咬肌判定需要一侧强度至少达到另一侧的这个倍数；值越大，左右区分越严格。")
            controls_layout.addWidget(side_ratio_label, 7, 0)
            controls_layout.addWidget(self.side_ratio_spin, 7, 1)
            controls_layout.addWidget(QtWidgets.QLabel("波形页时长"), 7, 2)
            controls_layout.addWidget(self.plot_seconds_spin, 7, 3)
            controls_layout.addWidget(self.auto_scale_check, 7, 4, 1, 2)
            controls_layout.addWidget(self.filter_check, 8, 0)
            controls_layout.addWidget(self.speech_check, 9, 0)
            controls_layout.addWidget(QtWidgets.QLabel("咬肌带通"), 8, 1)
            controls_layout.addWidget(self.emg_low_spin, 8, 2)
            controls_layout.addWidget(self.emg_high_spin, 8, 3)
            controls_layout.addWidget(QtWidgets.QLabel("眼电带通"), 9, 1)
            controls_layout.addWidget(self.eog_low_spin, 9, 2)
            controls_layout.addWidget(self.eog_high_spin, 9, 3)

            self.start_button = QtWidgets.QPushButton("启用信号监测")
            self.start_button.clicked.connect(self.start_stream)
            self.stop_button = QtWidgets.QPushButton("停止")
            self.stop_button.clicked.connect(self.stop_stream)
            self.reset_button = QtWidgets.QPushButton("重置基线")
            self.reset_button.clicked.connect(self.reset_baseline)
            controls_layout.addWidget(self.start_button, 9, 4)
            controls_layout.addWidget(self.stop_button, 9, 5)
            controls_layout.addWidget(self.reset_button, 8, 4, 1, 2)
            layout.addWidget(controls)

            self.result_label = QtWidgets.QLabel("结果：未开始")
            self.result_label.setObjectName("vrBadge")
            layout.addWidget(self.result_label)
            self.status_label = QtWidgets.QLabel("状态：等待连接")
            self.status_label.setObjectName("vrStatus")
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label)
            self.connect_progress = QtWidgets.QProgressBar()
            self.connect_progress.setRange(0, 100)
            self.connect_progress.setValue(0)
            self.connect_progress.setFormat("未订阅系统 LSL")
            layout.addWidget(self.connect_progress)

            values = Card()
            values.setMinimumHeight(270)
            values_layout = QtWidgets.QGridLayout(values)
            values_layout.setHorizontalSpacing(10)
            values_layout.setVerticalSpacing(8)
            values_layout.setColumnStretch(0, 1)
            values_layout.setColumnStretch(1, 2)
            values_layout.setColumnStretch(2, 5)
            values_layout.setColumnStretch(3, 1)
            values_layout.setColumnStretch(4, 10)
            self.raw_labels: List[QtWidgets.QLabel] = []
            self.rms_labels: List[QtWidgets.QLabel] = []
            self.ratio_labels: List[QtWidgets.QLabel] = []
            self.bars: List[QtWidgets.QProgressBar] = []
            for col, text in enumerate(["通道/位置", "最新值", "RMS", "基线倍数", "强度"]):
                header = QtWidgets.QLabel(text)
                header.setObjectName("hint")
                header.setMinimumHeight(30)
                header.setStyleSheet("font-size: 16px;")
                values_layout.addWidget(header, 0, col)
            for row, name in enumerate(self.channel_display_names(), start=1):
                values_layout.setRowMinimumHeight(row, 48)
                channel_label = QtWidgets.QLabel(name)
                channel_label.setMinimumHeight(44)
                channel_label.setWordWrap(True)
                channel_label.setStyleSheet("font-size: 18px; font-weight: 700;")
                self.channel_value_labels.append(channel_label)
                values_layout.addWidget(channel_label, row, 0)
                raw = QtWidgets.QLabel("-")
                rms = QtWidgets.QLabel("-")
                ratio = QtWidgets.QLabel("-")
                bar = QtWidgets.QProgressBar()
                bar.setRange(0, 1000)
                bar.setValue(0)
                for label in [raw, rms, ratio]:
                    label.setMinimumHeight(44)
                    label.setStyleSheet("font-size: 20px; font-weight: 700;")
                rms.setStyleSheet("font-size: 18px; font-weight: 700;")
                bar.setMinimumHeight(42)
                bar.setStyleSheet("QProgressBar { font-size: 18px; font-weight: 700; }")
                self.raw_labels.append(raw)
                self.rms_labels.append(rms)
                self.ratio_labels.append(ratio)
                self.bars.append(bar)
                values_layout.addWidget(raw, row, 1)
                values_layout.addWidget(rms, row, 2)
                values_layout.addWidget(ratio, row, 3)
                values_layout.addWidget(bar, row, 4)
            layout.addWidget(values)

            raw_title = QtWidgets.QLabel("真实原始信号：横坐标=时间（秒），每行一个物理通道。EEG 与 AUX 来自同一次 LSL 拉取，不生成模拟波形。")
            raw_title.setObjectName("hint")
            raw_title.setStyleSheet("font-size: 16px;")
            layout.addWidget(raw_title)

            self.trace_tabs = QtWidgets.QTabWidget()
            self.trace_tabs.setMinimumHeight(820)

            eeg_tab = QtWidgets.QWidget()
            eeg_tab_layout = QtWidgets.QVBoxLayout(eeg_tab)
            eeg_controls = QtWidgets.QHBoxLayout()
            eeg_controls.addWidget(QtWidgets.QLabel("EEG 每通道量程"))
            self.eeg_scale_spin = QtWidgets.QDoubleSpinBox()
            self.eeg_scale_spin.setRange(1.0, 50000000.0)
            self.eeg_scale_spin.setDecimals(1)
            self.eeg_scale_spin.setValue(75.0)
            self.eeg_scale_spin.setSuffix(" uV")
            self.eeg_scale_spin.valueChanged.connect(lambda _value: self._maybe_update_plot(force=True))
            self.eeg_auto_scale_check = QtWidgets.QCheckBox("EEG 自动量程")
            self.eeg_auto_scale_check.setChecked(False)
            self.eeg_auto_scale_check.toggled.connect(lambda _checked: self._maybe_update_plot(force=True))
            eeg_controls.addWidget(self.eeg_scale_spin)
            eeg_controls.addWidget(self.eeg_auto_scale_check)
            eeg_controls.addStretch(1)
            eeg_tab_layout.addLayout(eeg_controls)
            eeg_scroll = QtWidgets.QScrollArea()
            eeg_scroll.setWidgetResizable(True)
            eeg_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            self.eeg_trace = RawTraceWidget(DEFAULT_EEG_CHANNELS, row_height=64)
            self.eeg_trace.set_config(self.plot_page_seconds, float(self.eeg_scale_spin.value()))
            eeg_scroll.setWidget(self.eeg_trace)
            eeg_tab_layout.addWidget(eeg_scroll, 1)

            emg_tab = QtWidgets.QWidget()
            emg_tab_layout = QtWidgets.QVBoxLayout(emg_tab)
            self.raw_trace = RawTraceWidget(self.CHANNEL_NAMES)
            self.raw_trace.set_config(self.plot_page_seconds, self.raw_uv_per_row)
            self._update_channel_labels()
            emg_tab_layout.addWidget(self.raw_trace)
            self.trace_tabs.addTab(eeg_tab, "32 路 EEG")
            self.trace_tabs.addTab(emg_tab, "4 路 AUX / 肌电")
            layout.addWidget(self.trace_tabs, 3)
            self.plot_seconds_spin.valueChanged.connect(self.update_plot_duration)
            self.auto_scale_check.toggled.connect(self._auto_scale_changed)
            self._auto_scale_changed(self.auto_scale_check.isChecked())

            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(70)
            layout.addWidget(self.log)

        def append_log(self, text: str) -> None:
            self.log.append(text)

        def _reset_hit_counts(self) -> None:
            for key in self.hit_counts:
                self.hit_counts[key] = 0
            self.blink_times.clear()
            self.blink_active = False
            self.test_emg_events = None
            self.test_event_buffer_started_at = 0.0
            self.test_event_config = None

        def _role_key(self, idx: int) -> str:
            if idx >= len(self.role_combos):
                return self.DEFAULT_ROLES[idx] if idx < len(self.DEFAULT_ROLES) else "display"
            value = self.role_combos[idx].currentData()
            return str(value) if value is not None else "display"

        def channel_roles(self) -> List[str]:
            return [self._role_key(idx) for idx in range(4)]

        def _physical_name(self, idx: int) -> str:
            if self.source is not None and idx < len(self.source.indices):
                lsl_ch = self.source.indices[idx] + 1
                aux_ch = lsl_ch - 32 if self.source.channel_count >= 36 else idx + 1
                return f"Aux{aux_ch} / LSL {lsl_ch}"
            return f"Aux{idx + 1} / LSL {33 + idx}"

        def channel_display_names(self) -> List[str]:
            names = []
            for idx in range(4):
                role = self._role_key(idx)
                role_text = self.ROLE_DISPLAY.get(role, "仅显示")
                names.append(f"{self._physical_name(idx)} -> {role_text}")
            return names

        def _manual_channel_scales(self) -> np.ndarray:
            if not self.channel_scale_spins:
                return np.full(4, max(1e-6, float(self.raw_uv_per_row)), dtype=np.float64)
            return np.asarray([max(1e-6, float(spin.value())) for spin in self.channel_scale_spins], dtype=np.float64)

        def _set_channel_scales(self, scales: Sequence[float]) -> None:
            arr = np.asarray(scales, dtype=np.float64).reshape(-1)
            if arr.size < 4:
                arr = np.pad(arr, (0, 4 - arr.size), constant_values=max(1e-6, float(self.raw_uv_per_row)))
            arr = np.maximum(arr[:4], 1e-6)
            self.raw_uv_per_row = float(np.median(arr))
            for idx, spin in enumerate(self.channel_scale_spins):
                spin.blockSignals(True)
                spin.setValue(float(arr[idx]))
                spin.blockSignals(False)

        def _update_channel_labels(self) -> None:
            names = self.channel_display_names()
            for idx, label in enumerate(self.physical_labels):
                label.setText(self._physical_name(idx))
            for idx, label in enumerate(self.channel_value_labels):
                label.setText(names[idx])
            if hasattr(self, "raw_trace"):
                self.raw_trace.set_channel_names(names)

        def _channel_config_changed(self, *_args) -> None:
            self._update_channel_labels()
            if self.baseline_rms is not None:
                self.baseline_values = []
                self.baseline_rms = None
                self.baseline_until = time.time() + float(self.baseline_spin.value())
                self.blink_times.clear()
                self.blink_active = False
                self._reset_hit_counts()
                self.result_label.setText("结果：通道设置已变化，重新采集基线中")
                self.append_log("通道位置/启用状态已变化，重新采集基线。")
            self._maybe_update_plot(self.selected_mask(), force=True)

        def _auto_scale_changed(self, checked: bool) -> None:
            for spin in self.channel_scale_spins:
                spin.setEnabled(not checked)
            self._maybe_update_plot(self.selected_mask(), force=True)

        def filter_config(self) -> Tuple[bool, float, float, float, float]:
            enabled = bool(getattr(self, "filter_check", None) is not None and self.filter_check.isChecked())
            emg_low = float(self.emg_low_spin.value()) if hasattr(self, "emg_low_spin") else EMG_BAND_LOW_HZ
            emg_high = float(self.emg_high_spin.value()) if hasattr(self, "emg_high_spin") else EMG_BAND_HIGH_HZ
            eog_low = float(self.eog_low_spin.value()) if hasattr(self, "eog_low_spin") else EOG_BAND_LOW_HZ
            eog_high = float(self.eog_high_spin.value()) if hasattr(self, "eog_high_spin") else EOG_BAND_HIGH_HZ
            if emg_high <= emg_low:
                emg_high = emg_low + 0.1
            if eog_high <= eog_low:
                eog_high = eog_low + 0.1
            return enabled, emg_low, emg_high, eog_low, eog_high

        def _filter_config_changed(self, *_args) -> None:
            if self.baseline_rms is not None:
                self.baseline_values = []
                self.baseline_rms = None
                self.baseline_until = time.time() + float(self.baseline_spin.value())
                self.blink_times.clear()
                self.blink_active = False
                self._reset_hit_counts()
                self.result_label.setText("结果：滤波设置已变化，重新采集基线中")
                if hasattr(self, "log"):
                    self.append_log("滤波设置已变化，重新采集基线。")
            self._maybe_update_plot(self.selected_mask(), force=True)

        def _filtered_samples(self, data: np.ndarray) -> np.ndarray:
            enabled, emg_low, emg_high, eog_low, eog_high = self.filter_config()
            return filter_aux_samples_by_role(
                data,
                self.fs,
                self.channel_roles(),
                enabled=enabled,
                emg_low=emg_low,
                emg_high=emg_high,
                eog_low=eog_low,
                eog_high=eog_high,
            )

        def selected_mask(self) -> np.ndarray:
            if not self.channel_checks:
                return np.ones(4, dtype=bool)
            roles = self.channel_roles()
            return np.asarray(
                [check.isChecked() and roles[idx] != "unused" for idx, check in enumerate(self.channel_checks)],
                dtype=bool,
            )

        def current_plot_seconds(self) -> float:
            spin = getattr(self, "plot_seconds_spin", None)
            if spin is None:
                return float(self.plot_page_seconds)
            return max(1.0, float(spin.value()))

        def update_plot_duration(self, *_args) -> None:
            self.plot_page_seconds = self.current_plot_seconds()
            self.plot_page_start_ts = None
            self.eeg_plot_page_start_ts = None
            if hasattr(self, "raw_trace"):
                self.raw_trace.set_config(self.plot_page_seconds, self._manual_channel_scales())
            if hasattr(self, "eeg_trace"):
                self.eeg_trace.set_config(self.plot_page_seconds, float(self.eeg_scale_spin.value()))

        def _update_raw_scale(self, *_args) -> None:
            scales = self._manual_channel_scales()
            self.raw_uv_per_row = float(np.median(scales))
            if hasattr(self, "raw_trace"):
                self.raw_trace.set_config(self.current_plot_seconds(), scales)

        def _connect_worker_running(self) -> bool:
            worker = self.connect_worker
            if worker is None:
                return False
            try:
                running = bool(worker.isRunning())
                if not running:
                    self.connect_worker = None
                    self.start_button.setEnabled(True)
                    worker.deleteLater()
                return running
            except RuntimeError:
                self.connect_worker = None
                return False

        def _on_connect_worker_finished(self) -> None:
            worker = self.sender()
            if worker is self.connect_worker:
                self.connect_worker = None
            self.start_button.setEnabled(True)
            worker.deleteLater()

        def start_stream(self) -> None:
            if self.timer.isActive() or self._connect_worker_running():
                return
            self.stop_stream()
            self.start_button.setEnabled(False)
            worker = LslAcquireWorker("signal_monitor", 8.0, self)
            worker.progress_signal.connect(self._on_connect_progress)
            worker.ready_signal.connect(self._on_connect_ready)
            worker.error_signal.connect(self._on_connect_error)
            worker.finished.connect(self._on_connect_worker_finished)
            self.connect_worker = worker
            worker.start()

        def _on_connect_progress(self, percent: int, stage: str) -> None:
            self.connect_progress.setValue(percent)
            self.connect_progress.setFormat(f"{stage}：%p%")
            self.status_label.setText(f"状态：{stage}")

        def _on_connect_ready(self, source) -> None:
            self.source = source
            try:
                self.fs = max(1.0, float(self.source.fs))
                self._update_channel_labels()
                eeg_count = min(32, self.source.channel_count)
                self.eeg_trace.set_channel_names(self.source.channel_names[:eeg_count])
                self.samples = np.zeros((0, 4), dtype=np.float32)
                self.timestamps = np.zeros((0,), dtype=np.float64)
                self.eeg_samples = np.zeros((0, eeg_count), dtype=np.float32)
                self.eeg_timestamps = np.zeros((0,), dtype=np.float64)
                self.last_sample_wall = time.time()
                self.baseline_values = []
                self.baseline_rms = None
                self.blink_times.clear()
                self.blink_active = False
                self._reset_hit_counts()
                self.held_result = ""
                self.hold_until = 0.0
                self.last_result = ""
                self.current_result = "静息/未识别"
                self.last_classify_ts = 0.0
                self.arming_until = time.time() + 2.0
                self.plot_page_start_ts = None
                self.eeg_plot_page_start_ts = None
                self.update_plot_duration()
                self.timer.start(40)
                self.status_label.setText(f"状态：已连接 {self.source.description}")
                self.result_label.setText("提示：2.0 秒后开始采集，请准备并保持放松")
                self.append_log(f"LSL 已连接：{self.source.description}")
                self.append_log("提示：2 秒后开始采集肌电。")
                self.connect_progress.setValue(100)
                self.connect_progress.setFormat("已订阅系统 LSL：%p%")
            except Exception as exc:
                self.source = None
                self.status_label.setText(f"状态：连接失败：{exc}")
                self.append_log(f"连接失败：{exc}")
            finally:
                self.start_button.setEnabled(True)

        def _on_connect_error(self, message: str) -> None:
            self.source = None
            self.connect_progress.setValue(0)
            self.connect_progress.setFormat("系统 LSL 订阅失败")
            self.status_label.setText(f"状态：未订阅系统 LSL | {message}")
            self.append_log(f"系统 LSL 订阅失败：{message}")
            self.start_button.setEnabled(True)

        def stop_stream(self) -> None:
            if self._connect_worker_running():
                self.connect_worker.requestInterruption()
            self.timer.stop()
            if self.source is not None:
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭 LSL 失败：{exc}")
            self.source = None
            self.arming_until = 0.0
            self.status_label.setText("状态：已停止")
            self.connect_progress.setValue(0)
            self.connect_progress.setFormat("未订阅系统 LSL")
            self.start_button.setEnabled(True)

        def reset_baseline(self) -> None:
            if not np.any(self.selected_mask()):
                QtWidgets.QMessageBox.warning(self, "未选择通道", "请至少选择 1 个肌电通道。")
                return
            self.baseline_values = []
            self.baseline_rms = None
            self.baseline_until = time.time() + float(self.baseline_spin.value())
            self.arming_until = 0.0
            self.blink_times.clear()
            self.blink_active = False
            self._reset_hit_counts()
            self.held_result = ""
            self.hold_until = 0.0
            self.last_result = ""
            self.current_result = "静息/未识别"
            self.last_classify_ts = 0.0
            self.result_label.setText("结果：采集基线中，请保持放松")
            self.append_log("开始采集基线，请保持放松。")

        def _append_samples(self, chunk: np.ndarray, ts: np.ndarray) -> None:
            max_rows = max(250, int(round(self.fs * max(8.0, self.current_plot_seconds() + 2.0))))
            chunk_arr = np.asarray(chunk, dtype=np.float32)
            ts_arr = np.asarray(ts, dtype=np.float64)
            if chunk_arr.shape[0] > max_rows:
                chunk_arr = chunk_arr[-max_rows:, :]
                ts_arr = ts_arr[-max_rows:]
            keep_rows = max(0, max_rows - chunk_arr.shape[0])
            if keep_rows and self.samples.shape[0] > keep_rows:
                self.samples = self.samples[-keep_rows:, :]
                self.timestamps = self.timestamps[-keep_rows:]
            if self.samples.size:
                self.samples = np.vstack([self.samples, chunk_arr])
                self.timestamps = np.concatenate([self.timestamps, ts_arr])
            else:
                self.samples = chunk_arr
                self.timestamps = ts_arr

        def _append_eeg_samples(self, chunk: np.ndarray, ts: np.ndarray) -> None:
            chunk_arr = np.asarray(chunk, dtype=np.float32)
            ts_arr = np.asarray(ts, dtype=np.float64)
            if chunk_arr.ndim != 2 or chunk_arr.shape[1] == 0:
                return
            max_rows = max(250, int(round(self.fs * max(8.0, self.current_plot_seconds() + 2.0))))
            if chunk_arr.shape[0] > max_rows:
                chunk_arr = chunk_arr[-max_rows:, :]
                ts_arr = ts_arr[-max_rows:]
            keep_rows = max(0, max_rows - chunk_arr.shape[0])
            if keep_rows and self.eeg_samples.shape[0] > keep_rows:
                self.eeg_samples = self.eeg_samples[-keep_rows:, :]
                self.eeg_timestamps = self.eeg_timestamps[-keep_rows:]
            if self.eeg_samples.size and self.eeg_samples.shape[1] == chunk_arr.shape[1]:
                self.eeg_samples = np.vstack([self.eeg_samples, chunk_arr])
                self.eeg_timestamps = np.concatenate([self.eeg_timestamps, ts_arr])
            else:
                self.eeg_samples = chunk_arr
                self.eeg_timestamps = ts_arr

        def _feature_window(self) -> Optional[np.ndarray]:
            if self.samples.shape[0] < 8:
                return None
            rows = max(8, int(round(self.fs * 0.30)))
            return self.samples[-min(rows, self.samples.shape[0]) :, :]

        def _compute_features(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            filtered = self._filtered_samples(window)
            # Event counting uses raw-AUX RMS; filters affect display only.
            centered = window - np.median(window, axis=0, keepdims=True)
            rms = np.sqrt(np.mean(np.square(centered), axis=0))
            p2p = np.ptp(filtered, axis=0)
            latest = filtered[-1, :]
            return latest, rms, p2p

        def _update_baseline(self, rms: np.ndarray, now: float) -> bool:
            if self.baseline_rms is None:
                self.baseline_values.append(rms)
                if now < self.baseline_until:
                    return False
                baseline = np.median(np.vstack(self.baseline_values), axis=0)
                self.baseline_rms = np.maximum(baseline, 1e-6)
                self.append_log(f"基线完成 RMS={np.array2string(self.baseline_rms, precision=3)}")
            return True

        def _canonical_test_aux(
            self,
            selected: np.ndarray,
            row_mask: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Map configurable monitor channels to the runtime AUX1-4 contract."""
            roles = self.channel_roles()
            role_order = ("left_eye", "right_eye", "left_masseter", "right_masseter")
            canonical = np.zeros((4, int(np.count_nonzero(row_mask))), dtype=np.float64)
            baseline = np.ones(4, dtype=np.float64)
            source_baseline = np.asarray(self.baseline_rms, dtype=np.float64).reshape(-1)
            for target_idx, role in enumerate(role_order):
                source_idx = next(
                    (
                        idx
                        for idx, configured_role in enumerate(roles)
                        if idx < len(selected) and bool(selected[idx]) and configured_role == role
                    ),
                    None,
                )
                if source_idx is None:
                    continue
                canonical[target_idx, :] = np.asarray(
                    self.samples[row_mask, source_idx], dtype=np.float64
                )
                if source_idx < source_baseline.size:
                    baseline[target_idx] = max(1e-6, float(source_baseline[source_idx]))
            return canonical, baseline

        def _test_event_snapshot(self, selected: np.ndarray) -> EmgEventSnapshot:
            if self.baseline_rms is None or self.timestamps.size == 0:
                return EmgEventSnapshot("monitor", reason="missing_aux_baseline")
            latest_ts = float(self.timestamps[-1])
            roles = tuple(self.channel_roles())
            signature = (
                roles,
                tuple(bool(value) for value in selected),
                round(float(self.fs), 6),
                round(float(self.blink_ratio_spin.value()), 6),
                round(float(self.bite_ratio_spin.value()), 6),
                round(float(self.side_ratio_spin.value()), 6),
            )
            if self.test_emg_events is None or signature != self.test_event_config:
                initial_mask = self.timestamps > latest_ts
                _unused, baseline = self._canonical_test_aux(selected, initial_mask)
                self.test_emg_events = PhaseEmgEventEngine(
                    fs=self.fs,
                    baseline_rms=baseline,
                    blink_ratio=float(self.blink_ratio_spin.value()),
                    bite_ratio=float(self.bite_ratio_spin.value()),
                    side_ratio=float(self.side_ratio_spin.value()),
                    rms_window_seconds=0.18,
                    rms_hop_seconds=0.09,
                    blink_min_gap_seconds=0.18,
                    masseter_min_gap_seconds=0.25,
                    event_window_seconds={
                        "blink": 2.0,
                        "left_jaw": 1.0,
                        "right_jaw": 1.0,
                        "both_jaw": 1.0,
                    },
                )
                self.test_emg_events.reset("monitor", latest_ts)
                self.test_event_buffer_started_at = latest_ts
                self.test_event_config = signature
                return self.test_emg_events.snapshot()

            engine = self.test_emg_events
            if latest_ts - self.test_event_buffer_started_at >= 2.0:
                overlap = max(0.05, float(engine.rms_window_samples) / max(1.0, float(engine.fs)))
                self.test_event_buffer_started_at = latest_ts - overlap
                engine.roll_buffer(self.test_event_buffer_started_at)
            row_mask = self.timestamps > self.test_event_buffer_started_at
            if not np.any(row_mask):
                return engine.update(None, latest_ts)
            canonical, baseline = self._canonical_test_aux(selected, row_mask)
            engine.set_baseline(baseline)
            return engine.update(canonical, latest_ts)

        def _classify(self, ratio: np.ndarray, now: float, selected: np.ndarray) -> str:
            if now < self.hold_until and self.held_result:
                return self.held_result
            roles = self.channel_roles()

            def role_indices(*role_keys: str) -> List[int]:
                keys = set(role_keys)
                return [idx for idx, role in enumerate(roles) if idx < len(selected) and selected[idx] and role in keys]

            eye_indices = role_indices("left_eye", "right_eye")
            left_indices = role_indices("left_masseter")
            right_indices = role_indices("right_masseter")
            snapshot = self._test_event_snapshot(selected)
            self.hit_counts["blink3"] = int(snapshot.blink)
            self.hit_counts["left_masseter"] = int(snapshot.left_jaw)
            self.hit_counts["right_masseter"] = int(snapshot.right_jaw)
            self.hit_counts["both_masseter"] = int(snapshot.both_jaw)
            if snapshot.blink >= 3:
                self.held_result = "双眼眨眼三次"
                self.hold_until = now + 1.2
                self._reset_hit_counts()
                return self.held_result
            if snapshot.both_jaw >= 1:
                self.held_result = "咬两边咬肌"
                self.hold_until = now + 1.0
                self._reset_hit_counts()
                return self.held_result
            if snapshot.left_jaw >= 1 and snapshot.right_jaw == 0:
                return "左咬候选"
            if snapshot.right_jaw >= 1 and snapshot.left_jaw == 0:
                return "右咬候选"
            if snapshot.left_jaw or snapshot.right_jaw:
                return "动作不清晰/等待稳定"
            if not eye_indices and not left_indices and not right_indices:
                return "所选通道无可判定动作"
            return "静息/未识别"

        def _update_values(self, latest: np.ndarray, rms: np.ndarray, p2p: np.ndarray, ratio: np.ndarray, selected: Optional[np.ndarray] = None) -> None:
            if selected is None:
                selected = np.ones(4, dtype=bool)
            for idx in range(4):
                if not selected[idx]:
                    self.raw_labels[idx].setText("未选")
                    self.rms_labels[idx].setText("-")
                    self.ratio_labels[idx].setText("-")
                    self.bars[idx].setValue(0)
                    continue
                self.raw_labels[idx].setText(f"{latest[idx]:.3f}")
                self.rms_labels[idx].setText(f"{rms[idx]:.3f} / p2p {p2p[idx]:.3f}")
                self.ratio_labels[idx].setText(f"{ratio[idx]:.2f}x")
                self.bars[idx].setValue(int(min(1000, max(0, ratio[idx] * 120))))

        def _update_plot(self, selected: Optional[np.ndarray] = None) -> None:
            if self.samples.shape[0] < 2 or self.timestamps.shape[0] < 2:
                return
            if selected is None:
                selected = np.ones(4, dtype=bool)
            page_seconds = self.current_plot_seconds()
            latest_ts = float(self.timestamps[-1])
            if self.plot_page_start_ts is None:
                self.plot_page_start_ts = latest_ts
            while latest_ts - self.plot_page_start_ts >= page_seconds:
                self.plot_page_start_ts += page_seconds
            page_mask = self.timestamps >= self.plot_page_start_ts
            if not np.any(page_mask):
                return
            data = self._filtered_samples(self.samples[page_mask, :])
            t = self.timestamps[page_mask] - self.plot_page_start_ts
            scales = self._manual_channel_scales()
            if hasattr(self, "auto_scale_check") and self.auto_scale_check.isChecked():
                centered = data - np.median(data, axis=0, keepdims=True)
                valid_cols = [idx for idx in range(centered.shape[1]) if idx < len(selected) and selected[idx]]
                if valid_cols:
                    for col in valid_cols:
                        abs_values = np.abs(centered[:, col])
                        finite = abs_values[np.isfinite(abs_values)]
                        if finite.size:
                            suggested = float(np.percentile(finite, 99) / 0.22)
                            if suggested > 0:
                                exponent = math.floor(math.log10(suggested))
                                base = 10 ** exponent
                                nice = 10 * base
                                for multiplier in [1, 2, 5, 10]:
                                    candidate = multiplier * base
                                    if candidate >= suggested:
                                        nice = candidate
                                        break
                                scales[col] = nice
                    self._set_channel_scales(scales)
            if hasattr(self, "raw_trace"):
                self.raw_trace.set_config(page_seconds, scales)
                self.raw_trace.set_trace_data(t, data, selected)

        def _update_eeg_plot(self) -> None:
            if self.eeg_samples.shape[0] < 2 or self.eeg_timestamps.shape[0] < 2:
                return
            page_seconds = self.current_plot_seconds()
            latest_ts = float(self.eeg_timestamps[-1])
            if self.eeg_plot_page_start_ts is None:
                self.eeg_plot_page_start_ts = latest_ts
            while latest_ts - self.eeg_plot_page_start_ts >= page_seconds:
                self.eeg_plot_page_start_ts += page_seconds
            page_mask = self.eeg_timestamps >= self.eeg_plot_page_start_ts
            if not np.any(page_mask):
                return
            data = self.eeg_samples[page_mask, :]
            t = self.eeg_timestamps[page_mask] - self.eeg_plot_page_start_ts
            scales = np.full(data.shape[1], max(1.0, float(self.eeg_scale_spin.value())), dtype=np.float64)
            if self.eeg_auto_scale_check.isChecked():
                centered = data - np.median(data, axis=0, keepdims=True)
                robust = np.nanpercentile(np.abs(centered), 99, axis=0) / 0.22
                scales = np.maximum(1.0, robust)
            selected = np.ones(data.shape[1], dtype=bool)
            self.eeg_trace.set_config(page_seconds, scales)
            self.eeg_trace.set_trace_data(t, data, selected)

        def _maybe_update_plot(self, selected: Optional[np.ndarray] = None, force: bool = False) -> None:
            now = time.time()
            if force or now - self.last_plot_wall >= self.plot_update_interval:
                self.last_plot_wall = now
                self._update_plot(selected)
                self._update_eeg_plot()

        def tick(self) -> None:
            if self.source is None:
                self.stop_stream()
                return
            selected = self.selected_mask()
            if not np.any(selected):
                self.result_label.setText("结果：未启用肌电分类，仅监测 32 路 EEG")
            try:
                chunk, ts = self.source.read_available()
            except Exception as exc:
                self.append_log(f"LSL 读取失败：{exc}")
                QtWidgets.QMessageBox.critical(self, "LSL 读取失败", str(exc))
                self.stop_stream()
                return
            now = time.time()
            if self.arming_until > 0.0:
                remaining = self.arming_until - now
                if remaining > 0.0:
                    self.samples = np.zeros((0, 4), dtype=np.float32)
                    self.timestamps = np.zeros((0,), dtype=np.float64)
                    self.eeg_samples = np.zeros((0, self.eeg_samples.shape[1]), dtype=np.float32)
                    self.eeg_timestamps = np.zeros((0,), dtype=np.float64)
                    self.result_label.setText(f"提示：{remaining:.1f} 秒后开始采集，请准备并保持放松")
                    if self.source is not None:
                        self.status_label.setText(f"状态：已连接，等待开始 | {self.source.description}")
                    return
                self.arming_until = 0.0
                self.samples = np.zeros((0, 4), dtype=np.float32)
                self.timestamps = np.zeros((0,), dtype=np.float64)
                self.eeg_samples = np.zeros((0, self.eeg_samples.shape[1]), dtype=np.float32)
                self.eeg_timestamps = np.zeros((0,), dtype=np.float64)
                self.last_sample_wall = now
                self.plot_page_start_ts = None
                self.eeg_plot_page_start_ts = None
                self.reset_baseline()
                return
            if chunk is not None and ts is not None:
                self._append_samples(chunk, ts)
                full = self.source.last_full_samples
                full_ts = self.source.last_full_timestamps
                if full is not None and full_ts is not None:
                    self._append_eeg_samples(full[:, : min(32, full.shape[1])], full_ts)
                self.last_sample_wall = now
            if not np.any(selected):
                self._maybe_update_plot(selected)
                self.result_label.setText("结果：未启用肌电分类，仅监测 32 路 EEG")
                self.status_label.setText(f"状态：实时接收 EEG | {self.source.description}")
                return

            window = self._feature_window()
            if window is None:
                if now - self.last_sample_wall > 2.0:
                    self.status_label.setText("状态：已连接，但暂时没有收到样本")
                return
            latest, rms, p2p = self._compute_features(window)
            if not self._update_baseline(rms, now):
                ratio = np.zeros(4, dtype=np.float32)
                self._update_values(latest, rms, p2p, ratio, selected)
                self._maybe_update_plot(selected)
                self.result_label.setText("结果：采集基线中，请保持放松")
                return
            ratio = rms / np.maximum(self.baseline_rms, 1e-6)
            latest_ts = float(self.timestamps[-1]) if self.timestamps.shape[0] else now
            if self.last_classify_ts <= 0.0 or latest_ts - self.last_classify_ts >= self.classify_step_seconds:
                self.last_classify_ts = latest_ts
                self.current_result = self._classify(ratio, now, selected)
            result = self.current_result
            blink_count = int(self.hit_counts.get("blink3", 0))
            self._update_values(latest, rms, p2p, ratio, selected)
            self._maybe_update_plot(selected)
            self.result_label.setText(
                "结果：{result}    计数：眼 {eye} | 双咬肌 {both} | 左咬肌 {left} | 右咬肌 {right}".format(
                    result=result,
                    eye=blink_count,
                    both=self.hit_counts.get("both_masseter", 0),
                    left=self.hit_counts.get("left_masseter", 0),
                    right=self.hit_counts.get("right_masseter", 0),
                )
            )
            if self.source is not None:
                self.status_label.setText(f"状态：实时接收中 | {self.source.description}")
            if result != self.last_result:
                if result not in {"静息/未识别", "左咬候选", "右咬候选", "双咬候选（左右各到1）", "动作不清晰/等待稳定"}:
                    self.append_log(f"{time.strftime('%H:%M:%S')} 识别：{result} ratio={np.array2string(ratio, precision=2)}")
                    if result != "所选通道无可判定动作":
                        self.speaker.speak(result, enabled=self.speech_check.isChecked())
                self.last_result = result

        def closeEvent(self, event) -> None:
            self.stop_stream()
            self.speaker.close()
            super().closeEvent(event)

    class ActiveEEG8CollectionWorker(QtCore.QThread):
        """八分类采集后台线程"""
        log_signal = QtCore.pyqtSignal(str)
        status_signal = QtCore.pyqtSignal(str)
        event_signal = QtCore.pyqtSignal(dict)
        finished_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(self, config, trial_plan, output_path, parent=None):
            super().__init__(parent)
            self.config = dict(config)
            self.trial_plan = list(trial_plan)
            self.output_path = output_path
            self.stop_event = threading.Event()
            self.raw_chunks = []
            self.timestamp_chunks = []
            self.event_logs = []
            self.trial_records = []
            self.source = None
            self.session_start_perf = 0.0
            self.session_id = str(uuid.uuid4())
            self._completed = 0

        def stop(self): self.stop_event.set()

        def _pull_samples(self):
            if self.source is None: return
            try:
                chunk, ts = self.source.read_all_available()
                if chunk is not None and ts is not None:
                    self.raw_chunks.append(np.asarray(chunk, dtype=np.float32))
                    self.timestamp_chunks.append(np.asarray(ts, dtype=np.float64))
            except Exception: pass

        def _wait(self, seconds):
            end = time.perf_counter() + max(0.0, float(seconds))
            while time.perf_counter() < end and not self.stop_event.is_set():
                self._pull_samples()
                time.sleep(0.01)
            self._pull_samples()

        def _event_time_ms(self):
            return int(round((time.perf_counter() - self.session_start_perf) * 1000.0))

        def _log(self, event_type, trial=None):
            trial = trial or {}
            rec = {"timestamp_ms": self._event_time_ms(), "event_type": event_type,
                   "block_id": trial.get("block_id", 0),
                   "trial_id": trial.get("trial_id", 0),
                   "global_trial_id": trial.get("global_trial_id", 0),
                   "class_id": trial.get("class_id", 0)}
            phase_labels = {
                "baseline_start": "静息/基线",
                "cue_onset": "提示",
                "task_start": "采集/执行",
                "task_end": "采集/执行结束",
                "rest_start": "采集后静息",
                "trial_end": "Trial结束",
            }
            if event_type in phase_labels:
                rec["phase_label"] = phase_labels[event_type]
            for key in (
                "class_name", "class_name_en", "cue_title", "cue_subtitle",
                "instruction", "forbidden", "slot", "is_practice",
                "baseline_label", "baseline_duration_s",
            ):
                if key in trial:
                    rec[key] = trial.get(key)
            self.event_logs.append(rec)
            self.event_signal.emit(rec)

        def _capture_trial(self, spec):
            rec = {"block_id": spec.get("block_id", 0), "trial_id": spec.get("trial_id", 0),
                   "global_trial_id": spec.get("global_trial_id", 0),
                   "class_id": spec["class_id"], "class_name": spec["class_name"],
                   "cue_title": spec.get("cue_title", ""), "cue_subtitle": spec.get("cue_subtitle", ""),
                   "class_name_en": spec.get("class_name_en", ""),
                   "instruction": spec.get("instruction", ""), "forbidden": spec.get("forbidden", ""),
                   "slot": spec.get("slot", 0),
                   "is_practice": spec.get("is_practice", False), "bad_trial": False}
            bl = float(self.config.get("baseline_seconds", 2.0))
            cu = float(self.config.get("cue_seconds", 1.0))
            ta = float(self.config.get("task_seconds", 5.0))
            rs = float(self.config.get("rest_seconds", 2.0))
            rec["baseline_label"] = "静息"
            rec["baseline_duration_s"] = bl
            self._log("trial_start", rec)
            # 静息/基线
            rec["baseline_start_ms"] = self._event_time_ms()
            self._log("baseline_start", rec); self._wait(bl)
            # 提示
            rec["cue_onset_ms"] = self._event_time_ms()
            self._log("cue_onset", rec); self._wait(cu)
            # 执行
            rec["task_start_ms"] = self._event_time_ms()
            self._log("task_start", rec); self._wait(ta)
            rec["task_end_ms"] = self._event_time_ms()
            self._log("task_end", rec)
            # 休息
            rec["rest_start_ms"] = rec["task_end_ms"]
            self._log("rest_start", rec); self._wait(rs)
            rec["trial_end_ms"] = self._event_time_ms()
            self._log("trial_end", rec)
            self.trial_records.append(rec); self._completed += 1
            return rec

        def _save(self):
            self._pull_samples()
            raw = np.vstack(self.raw_chunks).astype(np.float32) if self.raw_chunks else np.zeros((0, 0), dtype=np.float32)
            ts = np.concatenate(self.timestamp_chunks).astype(np.float64) if self.timestamp_chunks else np.zeros((0,), dtype=np.float64)
            ch_names = list(self.source.channel_names) if self.source is not None else []
            ec = min(32, raw.shape[1]) if raw.ndim == 2 and raw.size else 0
            eeg = raw[:, :ec].T.astype(np.float32) if ec else np.zeros((0, 0), dtype=np.float32)
            source_aux_indices = list(getattr(self.source, "indices", []) or []) if self.source is not None else []
            aux_indices = [
                int(index)
                for index in source_aux_indices[:4]
                if raw.ndim == 2 and 32 <= int(index) < raw.shape[1]
            ]
            if not aux_indices and raw.ndim == 2 and raw.shape[1] > 32:
                # Fallback for a disconnected source.
                aux_indices = list(range(32, min(36, raw.shape[1])))
            aux = raw[:, aux_indices].T.astype(np.float32) if aux_indices else np.zeros((0, 0), dtype=np.float32)
            aux_names = [ch_names[idx] if idx < len(ch_names) else f"Aux{idx - 31}" for idx in aux_indices]
            gate_capable = aux_indices == [32, 33, 34, 35]
            payload = {"eeg": eeg, "y": np.asarray([r["class_id"] for r in self.trial_records], dtype=np.int64),
                       "fs": np.asarray(int(round(self.source.fs if self.source is not None else 250)), dtype=np.int64),
                       "recording_contract_version": np.asarray(2, dtype=np.int64),
                       "eeg_indices": np.asarray(list(range(ec)), dtype=np.int64),
                       "eeg_channel_count": np.asarray(ec, dtype=np.int64),
                       "channel_names": np.asarray(ch_names[:ec], dtype=object),
                       "aux": aux,
                       "aux_channel_names": np.asarray(aux_names, dtype=object),
                       "aux_channel_count": np.asarray(len(aux_indices), dtype=np.int64),
                       "masseter_gate_capable": np.asarray(gate_capable, dtype=bool),
                       "raw_stream": raw, "raw_timestamps": ts,
                       "raw_channel_names": np.asarray(ch_names, dtype=object),
                       "raw_channel_count": np.asarray(int(raw.shape[1]) if raw.ndim == 2 else 0, dtype=np.int64),
                       "aux_indices": np.asarray(aux_indices, dtype=np.int64),
                       "mode": "brainvision_lsl" if self.source is not None else "no_lsl",
                       "completed": np.asarray(not self.stop_event.is_set(), dtype=bool),
                       "session_id": self.session_id,
                       "experiment_config_json": json.dumps(self.config, ensure_ascii=False),
                       "task_definitions_json": json.dumps(
                           self.config.get("class_definitions") or EEG8_TASKS,
                           ensure_ascii=False,
                       ),
                       "trial_records_json": json.dumps(self.trial_records, ensure_ascii=False),
                       "event_logs_json": json.dumps(self.event_logs, ensure_ascii=False)}
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)) or ".", exist_ok=True)
            np.savez_compressed(self.output_path, **payload)
            return {"path": self.output_path, "trial_records": self.trial_records,
                    "event_logs": self.event_logs, "session_id": self.session_id}

        def run(self):
            try:
                self.session_start_perf = time.perf_counter()
                self._log("experiment_start")
                try:
                    self.source = acquire_system_lsl_source(timeout=8.0, owner="active_eeg8_collection")
                    self.status_signal.emit(f"LSL 已连接：{self.source.description}")
                    self.log_signal.emit(f"LSL 已连接：{self.source.description}")
                except Exception as exc:
                    self.source = None
                    self.log_signal.emit(f"LSL 未接通：{exc}")
                self.log_signal.emit("请确认关闭无关软件，正式任务期只看屏幕中央十字。")
                self._wait(2.0)
                for spec in self.trial_plan:
                    if self.stop_event.is_set(): break
                    self._capture_trial(spec)
                result = self._save()
                self.finished_signal.emit(result)
            except Exception as exc:
                self.error_signal.emit(str(exc))
            finally:
                if self.source is not None: self.source.close()

    class ActiveEEG8CollectionPage(QtWidgets.QWidget):
        """八分类主动采集页面"""
        finished_signal = QtCore.pyqtSignal(str)

        def __init__(self, config, parent=None):
            super().__init__(parent)
            self.config = config
            self.worker: Optional[ActiveEEG8CollectionWorker] = None
            self.session_bundle: Dict = {}
            self.unsaved_changes = False
            self.current_phase = "idle"
            self.fs_dialog = None
            self.fs_stage_frame = None
            self.fs_run_stage = None
            self.fs_progress = None
            self.fs_cue_title = None
            self.fs_cue_subtitle = None
            self.fs_status_label = None
            self.fs_start_btn = None
            self.fs_stop_btn = None
            self.fs_exit_btn = None
            self.fs_shortcuts = []
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel("主动 EEG / 纯 MI 采集"); title.setObjectName("title"); layout.addWidget(title)
            desc = QtWidgets.QLabel("范式在“准备配置 → 采集/在线范式”中统一选择；纯 MI 模式只采集左手和右手运动想象。")
            desc.setWordWrap(True); desc.setObjectName("desc"); layout.addWidget(desc)
            self.paradigm_status_label = QtWidgets.QLabel()
            self.paradigm_status_label.setObjectName("vrStatus")
            self.paradigm_status_label.setText(
                "当前范式：" + (
                    "纯 MI 二分类（左手=1 / 右手=0）"
                    if self.config.collection_config().get("paradigm_key") == "pure_mi_binary"
                    else "原八分类范式"
                )
            )
            layout.addWidget(self.paradigm_status_label)

            top = QtWidgets.QHBoxLayout()
            self.fullscreen_btn = QtWidgets.QPushButton("全屏采集"); self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)
            self.start_btn = QtWidgets.QPushButton("开始采集"); self.start_btn.setObjectName("primaryButton"); self.start_btn.clicked.connect(self.start_collection)
            self.stop_btn = QtWidgets.QPushButton("停止"); self.stop_btn.setObjectName("dangerButton"); self.stop_btn.clicked.connect(self.stop_collection); self.stop_btn.setEnabled(False)
            for b in [self.fullscreen_btn, self.start_btn, self.stop_btn]: top.addWidget(b)
            top.addStretch(1); layout.addLayout(top)

            # 设置
            cfg_card = Card(); cfg_form = QtWidgets.QFormLayout(cfg_card)
            self.blocks_spin = QtWidgets.QSpinBox(); self.blocks_spin.setRange(1, 24); self.blocks_spin.setValue(4)
            cfg_form.addRow("Block 数量", self.blocks_spin)
            self.trials_spin = QtWidgets.QSpinBox(); self.trials_spin.setRange(1, 20); self.trials_spin.setValue(5)
            cfg_form.addRow("每类 Trial 数", self.trials_spin)
            self.seed_spin = QtWidgets.QSpinBox(); self.seed_spin.setRange(1, 999999); self.seed_spin.setValue(42)
            cfg_form.addRow("随机种子", self.seed_spin)
            layout.addWidget(cfg_card)

            # 采集舞台
            stage = QtWidgets.QFrame(); stage.setObjectName("paradigmStage")
            self.stage_frame = stage
            sl = QtWidgets.QVBoxLayout(stage); sl.setContentsMargins(24, 22, 24, 22)
            hdr = QtWidgets.QHBoxLayout()
            self.run_stage = QtWidgets.QLabel("等待开始"); self.run_stage.setObjectName("phaseBadge")
            self.progress = QtWidgets.QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
            hdr.addWidget(self.run_stage); hdr.addStretch(1); hdr.addWidget(self.progress)
            sl.addLayout(hdr)
            self.cue_title = QtWidgets.QLabel("+"); self.cue_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.cue_title.setWordWrap(True)
            self.cue_title.setObjectName("cueWord"); sl.addWidget(self.cue_title, 2)
            self.cue_subtitle = QtWidgets.QLabel(""); self.cue_subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.cue_subtitle.setWordWrap(True)
            self.cue_subtitle.setObjectName("phaseInstruction"); sl.addWidget(self.cue_subtitle)
            layout.addWidget(stage, 2)

            self.status_label = QtWidgets.QLabel("状态：等待开始"); self.status_label.setObjectName("vrStatus"); layout.addWidget(self.status_label)
            self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(120); layout.addWidget(self.log)
            self._set_phase_visual("idle")

        def has_unsaved_changes(self): return bool(self.unsaved_changes)

        def eventFilter(self, obj, event):
            if obj is self.fs_dialog and event.type() == QtCore.QEvent.Type.Close:
                self.fullscreen_btn.setText("全屏采集")
            return super().eventFilter(obj, event)

        def _toggle_fullscreen(self):
            if self.fs_dialog is not None and self.fs_dialog.isVisible():
                self._close_collection_fullscreen()
            else:
                self._open_collection_fullscreen()

        def _open_collection_fullscreen(self):
            self._ensure_collection_fullscreen()
            self._sync_fullscreen_stage()
            self.fs_dialog.showFullScreen()
            self.fs_dialog.raise_()
            self.fs_dialog.activateWindow()
            self.fullscreen_btn.setText("退出全屏")

        def _close_collection_fullscreen(self):
            if self.fs_dialog is not None:
                self.fs_dialog.hide()
            self.fullscreen_btn.setText("全屏采集")

        def _ensure_collection_fullscreen(self):
            if self.fs_dialog is not None:
                return
            dialog = QtWidgets.QDialog(self.window())
            dialog.setObjectName("collectionFullscreen")
            dialog.setWindowTitle("八分类主动 EEG 采集")
            dialog.setWindowFlag(QtCore.Qt.WindowType.Window, True)
            dialog.setWindowFlag(QtCore.Qt.WindowType.FramelessWindowHint, True)
            dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
            dialog.installEventFilter(self)
            dialog.setStyleSheet("""
                QDialog#collectionFullscreen { background: #05080c; }
                QFrame#collectionFullscreenStage { border-radius: 0px; }
                QLabel#collectionFullscreenCue { letter-spacing: 0px; }
                QLabel#collectionFullscreenSubtitle { letter-spacing: 0px; }
                QLabel#collectionFullscreenStatus {
                    color: #d7e2ee;
                    font-size: 18px;
                    font-weight: 700;
                }
                QProgressBar {
                    min-height: 16px;
                    max-height: 16px;
                    border: 1px solid #334155;
                    border-radius: 3px;
                    background: #111827;
                    color: transparent;
                }
                QProgressBar::chunk {
                    border-radius: 2px;
                    background: #38bdf8;
                }
                QPushButton {
                    min-width: 132px;
                    min-height: 42px;
                    padding: 8px 18px;
                    border: 1px solid #3f5268;
                    border-radius: 6px;
                    background: #172033;
                    color: #f8fafc;
                    font-size: 18px;
                    font-weight: 800;
                }
                QPushButton:hover { background: #22304a; }
                QPushButton:disabled {
                    color: #7f8ea3;
                    background: #101722;
                    border-color: #263244;
                }
                QPushButton#primaryButton {
                    background: #0f766e;
                    border-color: #14b8a6;
                }
                QPushButton#primaryButton:hover { background: #0d9488; }
                QPushButton#dangerButton {
                    background: #991b1b;
                    border-color: #ef4444;
                }
                QPushButton#dangerButton:hover { background: #b91c1c; }
            """)

            root = QtWidgets.QVBoxLayout(dialog)
            root.setContentsMargins(34, 28, 34, 28)
            root.setSpacing(18)

            header = QtWidgets.QHBoxLayout()
            self.fs_run_stage = QtWidgets.QLabel("等待开始")
            self.fs_run_stage.setObjectName("phaseBadge")
            self.fs_progress = QtWidgets.QProgressBar()
            self.fs_progress.setRange(0, 100)
            self.fs_progress.setValue(0)
            self.fs_progress.setFixedWidth(360)
            self.fs_status_label = QtWidgets.QLabel("状态：等待开始")
            self.fs_status_label.setObjectName("collectionFullscreenStatus")
            self.fs_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            header.addWidget(self.fs_run_stage)
            header.addSpacing(18)
            header.addWidget(self.fs_progress)
            header.addStretch(1)
            header.addWidget(self.fs_status_label, 1)
            root.addLayout(header)

            self.fs_stage_frame = QtWidgets.QFrame()
            self.fs_stage_frame.setObjectName("collectionFullscreenStage")
            stage_layout = QtWidgets.QVBoxLayout(self.fs_stage_frame)
            stage_layout.setContentsMargins(44, 34, 44, 34)
            stage_layout.setSpacing(26)
            stage_layout.addStretch(1)
            self.fs_cue_title = QtWidgets.QLabel("+")
            self.fs_cue_title.setObjectName("collectionFullscreenCue")
            self.fs_cue_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.fs_cue_title.setWordWrap(True)
            self.fs_cue_title.setMinimumHeight(180)
            self.fs_cue_subtitle = QtWidgets.QLabel("")
            self.fs_cue_subtitle.setObjectName("collectionFullscreenSubtitle")
            self.fs_cue_subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.fs_cue_subtitle.setWordWrap(True)
            self.fs_cue_subtitle.setMinimumHeight(90)
            stage_layout.addWidget(self.fs_cue_title)
            stage_layout.addWidget(self.fs_cue_subtitle)
            stage_layout.addStretch(1)
            root.addWidget(self.fs_stage_frame, 1)

            controls = QtWidgets.QHBoxLayout()
            controls.addStretch(1)
            self.fs_start_btn = QtWidgets.QPushButton("开始采集")
            self.fs_start_btn.setObjectName("primaryButton")
            self.fs_start_btn.clicked.connect(self.start_collection)
            self.fs_stop_btn = QtWidgets.QPushButton("停止")
            self.fs_stop_btn.setObjectName("dangerButton")
            self.fs_stop_btn.clicked.connect(self.stop_collection)
            self.fs_exit_btn = QtWidgets.QPushButton("退出全屏")
            self.fs_exit_btn.clicked.connect(self._close_collection_fullscreen)
            controls.addWidget(self.fs_start_btn)
            controls.addWidget(self.fs_stop_btn)
            controls.addWidget(self.fs_exit_btn)
            controls.addStretch(1)
            root.addLayout(controls)

            self.fs_shortcuts = [
                QtGui.QShortcut(QtGui.QKeySequence("Esc"), dialog, activated=self._close_collection_fullscreen),
                QtGui.QShortcut(QtGui.QKeySequence("F11"), dialog, activated=self._close_collection_fullscreen),
            ]
            self.fs_dialog = dialog

        def _set_status_text(self, text):
            self.status_label.setText(text)
            if self.fs_status_label is not None:
                self.fs_status_label.setText(text)

        def _sync_collection_buttons(self):
            if self.fs_start_btn is not None:
                self.fs_start_btn.setEnabled(self.start_btn.isEnabled())
            if self.fs_stop_btn is not None:
                self.fs_stop_btn.setEnabled(self.stop_btn.isEnabled())

        def _sync_fullscreen_stage(self):
            if self.fs_dialog is None:
                return
            self.fs_run_stage.setText(self.run_stage.text())
            self.fs_progress.setValue(self.progress.value())
            self.fs_cue_title.setText(self.cue_title.text())
            self.fs_cue_subtitle.setText(self.cue_subtitle.text())
            self.fs_status_label.setText(self.status_label.text())
            self._sync_collection_buttons()
            self._apply_phase_visual(
                self.fs_stage_frame,
                self.fs_cue_title,
                self.fs_cue_subtitle,
                self.fs_run_stage,
                self.current_phase,
                fullscreen=True,
            )

        def _make_path(self):
            root = self.config.collection_out_edit.text().strip() or RECORDINGS_DIR
            if os.path.splitext(root)[1].lower() == ".npz": root = os.path.dirname(root)
            suffix = {"pure_mi_binary": "pure_mi", "custom_mi": "custom_mi"}.get(self.config.collection_config().get("paradigm_key"), "eeg8")
            stamp = datetime.now().strftime(f"%Y-%m-%d/%H-%M-%S_{suffix}")
            return os.path.join(root, stamp + ".npz")

        def _build_plan(self):
            blocks = int(self.blocks_spin.value()); tpc = int(self.trials_spin.value()); seed = int(self.seed_spin.value())
            paradigm_key = self.config.collection_config().get("paradigm_key", "eeg8")
            definitions = self.config.category_definitions()
            class_ids = [1, 2] if paradigm_key == "pure_mi_binary" else [int(item["class_id"]) for item in definitions]
            task_definitions = definitions if paradigm_key == "custom_mi" else None
            plan = []; gid = 0
            for bid in range(1, blocks + 1):
                pool = generate_balanced_block(class_ids, tpc, seed + bid - 1, task_definitions=task_definitions)
                for tid, item in enumerate(pool, 1):
                    gid += 1
                    plan.append({**item, "block_id": bid, "trial_id": tid, "global_trial_id": gid,
                                 "is_practice": False, "use_for_training": True})
            return plan

        def start_collection(self):
            if self.worker is not None and self.worker.isRunning(): return
            cfg = self.config.collection_config()
            paradigm_key = cfg.get("paradigm_key", "eeg8")
            paradigm_name = {"pure_mi_binary": "纯 MI 二分类（左手=1 / 右手=0）", "custom_mi": f"自定义 MI（{self.config.class_count_spin.value()} 类）"}.get(paradigm_key, "原八分类范式")
            self.paradigm_status_label.setText(f"当前范式：{paradigm_name}")
            plan = self._build_plan(); path = self._make_path()
            preview = "，".join(
                f"B{p.get('block_id')}T{p.get('trial_id')}:{p.get('cue_title', p.get('class_name', ''))}"
                for p in plan[:8]
            )
            self.log.clear(); self.run_stage.setText("正在连接 LSL")
            self._set_phase_visual("idle")
            self.cue_title.setText("准备采集")
            self.cue_subtitle.setText(
                f"已生成随机范式：{len(plan)} 个 trial。请被试看屏幕中央，等待正式开始。"
            )
            self.progress.setValue(0); self._set_status_text("状态：正在连接 LSL...")
            self.log.append(f"随机范式已生成：{len(plan)} 个 trial，seed={int(self.seed_spin.value())}")
            if preview:
                self.log.append(f"前 8 个任务：{preview}")
            self.worker = ActiveEEG8CollectionWorker(
                {"paradigm_key": cfg.get("paradigm_key", "eeg8"),
                 "class_definitions": cfg.get("class_definitions", []),
                 "baseline_seconds": float(cfg.get("pre_rest_seconds", 2.0)),
                 "cue_seconds": float(cfg.get("cue_seconds", 1.0)),
                 "task_seconds": float(cfg.get("epoch_seconds", 5.0)),
                 "rest_seconds": float(cfg.get("post_rest_seconds", 2.0))},
                plan, path, parent=self)
            self.worker.log_signal.connect(self.log.append)
            self.worker.status_signal.connect(self._set_status_text)
            self.worker.event_signal.connect(self._on_event)
            self.worker.finished_signal.connect(self._on_finished)
            self.worker.error_signal.connect(self._on_error)
            self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True); self._sync_collection_buttons()
            self._sync_fullscreen_stage()
            self.worker.start()

        def stop_collection(self):
            if self.worker is not None: self.worker.stop()
            self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self._sync_collection_buttons()

        def _trial_label(self, payload):
            bid = int(payload.get("block_id", 0)); tid = int(payload.get("trial_id", 0)); cid = int(payload.get("class_id", 0))
            name = str(payload.get("class_name", "") or "")
            return f"Block {bid} Trial {tid} C{cid}" + (f" {name}" if name else "")

        def _task_title(self, payload):
            return str(payload.get("cue_title") or payload.get("class_name") or "+")

        def _task_action(self, payload):
            title = self._task_title(payload)
            detail = str(payload.get("cue_subtitle") or "").strip()
            instruction = str(payload.get("instruction") or "").strip()
            if instruction:
                return instruction
            if detail:
                return f"{title}：{detail}"
            return "按屏幕提示持续执行当前任务。"

        def _phase_style(self, phase):
            styles = {
                "idle": ("#0b1014", "#356072", "#f7fbff", "#bfe8ff", "#182933", "#356072", "#d9f2ff"),
                "baseline": ("#07131f", "#2f7da8", "#d8f2ff", "#a9d7ef", "#113047", "#2f7da8", "#d8f2ff"),
                "cue": ("#201305", "#f59e0b", "#ffd166", "#fff0c2", "#3b2406", "#f59e0b", "#fff3c4"),
                "task": ("#061b10", "#22c55e", "#62ff9a", "#c5ffd8", "#0d3a22", "#22c55e", "#ddffe8"),
                "rest": ("#101827", "#64748b", "#e2e8f0", "#cbd5e1", "#1e293b", "#64748b", "#e2e8f0"),
                "done": ("#071a16", "#14b8a6", "#99f6e4", "#ccfbf1", "#0f3b35", "#14b8a6", "#ccfbf1"),
                "error": ("#210c10", "#ef4444", "#fecaca", "#fecaca", "#3b1118", "#ef4444", "#fee2e2"),
            }
            return styles.get(phase, styles["idle"])

        def _apply_phase_visual(self, stage_frame, cue_title, cue_subtitle, run_stage, phase, fullscreen=False):
            stage_bg, border, title, subtitle, badge_bg, badge_border, badge_text = self._phase_style(phase)
            stage_object = "collectionFullscreenStage" if fullscreen else "paradigmStage"
            title_size = 92 if fullscreen else 56
            subtitle_size = 30 if fullscreen else 20
            badge_size = 18 if fullscreen else 10
            badge_padding = "8px 16px" if fullscreen else "6px 12px"
            stage_frame.setStyleSheet(
                f"QFrame#{stage_object} {{ background: {stage_bg}; border: 3px solid {border}; border-radius: 7px; }}"
            )
            cue_title.setStyleSheet(
                f"background: transparent; font-size: {title_size}px; font-weight: 900; color: {title};"
            )
            cue_subtitle.setStyleSheet(
                f"background: transparent; font-size: {subtitle_size}px; font-weight: 700; color: {subtitle};"
            )
            run_stage.setStyleSheet(
                f"background: {badge_bg}; border: 2px solid {badge_border}; border-radius: 5px; "
                f"color: {badge_text}; padding: {badge_padding}; font-size: {badge_size}px; font-weight: 800;"
            )

        def _set_phase_visual(self, phase):
            self.current_phase = phase
            self._apply_phase_visual(self.stage_frame, self.cue_title, self.cue_subtitle, self.run_stage, phase)
            if self.fs_dialog is not None:
                self._apply_phase_visual(
                    self.fs_stage_frame,
                    self.fs_cue_title,
                    self.fs_cue_subtitle,
                    self.fs_run_stage,
                    phase,
                    fullscreen=True,
                )

        def _progress_for_event(self, payload, event_type):
            total = max(1, len(self.worker.trial_plan) if self.worker else 1)
            gid = int(payload.get("global_trial_id", 0) or 0)
            if gid <= 0:
                completed = len(self.worker.trial_records) if self.worker else 0
                return int(min(100, 100 * completed / total))
            phase_pos = {
                "trial_start": 0.0,
                "baseline_start": 0.1,
                "cue_onset": 0.35,
                "task_start": 0.55,
                "rest_start": 0.85,
                "trial_end": 1.0,
            }.get(event_type, 0.0)
            return int(min(100, 100 * ((gid - 1) + phase_pos) / total))

        def _on_event(self, payload):
            et = str(payload.get("event_type", ""))
            self.progress.setValue(self._progress_for_event(payload, et))
            label = self._trial_label(payload)
            task_title = self._task_title(payload)
            task_action = self._task_action(payload)
            if et == "trial_start":
                self._set_phase_visual("idle")
                self.run_stage.setText(f"即将开始 - {label}")
                self.cue_title.setText("+")
                self.cue_subtitle.setText(f"本 trial 任务：{task_title}。先保持放松，等待提示。")
                self._set_status_text(f"状态：即将开始 {label}")
            elif et == "baseline_start":
                self._set_phase_visual("baseline")
                self.run_stage.setText(f"静息/基线 - {label}"); self.cue_title.setText("+")
                self.cue_subtitle.setText(f"静息阶段：注视十字，保持放松。下一任务：{task_title}")
                self._set_status_text(f"状态：静息/基线 {label}")
            elif et == "cue_onset":
                self._set_phase_visual("cue")
                self.run_stage.setText(f"提示阶段 - {label}")
                self.cue_title.setText(f"提示阶段\n{task_title}")
                detail = str(payload.get("cue_subtitle") or "").strip()
                self.cue_subtitle.setText(f"{detail or task_action}。看清任务，暂时不要开始。")
                self._set_status_text(f"状态：提示阶段 {label}")
            elif et == "task_start":
                self._set_phase_visual("task")
                self.run_stage.setText(f"采集阶段 - {label}")
                self.cue_title.setText(f"采集阶段\n{task_title}")
                self.cue_subtitle.setText(f"现在开始执行。{task_action}")
                self._set_status_text(f"状态：正在采集 {label}")
            elif et == "rest_start":
                self._set_phase_visual("rest")
                self.run_stage.setText(f"采集后静息 - {label}"); self.cue_title.setText("+")
                self.cue_subtitle.setText("采集后静息：本次任务结束，放松身体，等待下一 trial。")
                self._set_status_text(f"状态：采集后静息 {label}")
            elif et == "trial_end":
                self._set_status_text(f"状态：已完成 {label}")
            self._sync_fullscreen_stage()

        def _on_finished(self, bundle):
            self.session_bundle = dict(bundle); self.unsaved_changes = False
            self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self._sync_collection_buttons()
            self._set_phase_visual("done")
            self.run_stage.setText("采集完成"); self.cue_title.setText("✓"); self.cue_subtitle.setText("")
            self.progress.setValue(100); self._set_status_text("状态：采集完成")
            self._sync_fullscreen_stage()
            path = bundle.get("path", "")
            self.log.append(f"数据已保存：{path}"); self.finished_signal.emit(path)

        def _on_error(self, msg):
            self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False); self._sync_collection_buttons()
            self._set_phase_visual("error")
            self._set_status_text(f"状态：采集失败 - {msg}")
            self._sync_fullscreen_stage()
            self.log.append(f"采集失败：{msg}")

    class EmgTemplateTrainPage(QtWidgets.QWidget):
        STATES = (
            ("00", "静息"),
            ("blink", "眨眼"),
            ("01", "右咬肌"),
            ("10", "左咬肌"),
            ("11", "双咬肌"),
        )
        EYE_STATES = {"blink"}

        def __init__(self, config: ConfigPage, parent=None):
            super().__init__(parent)
            self.config = config
            self.source = None
            self.connect_worker: Optional[QtCore.QThread] = None
            self.buffer = np.zeros((0, 4), dtype=np.float32)
            self.timestamps = np.zeros((0,), dtype=np.float64)
            self.samples: Dict[str, List[np.ndarray]] = {key: [] for key, _label in self.STATES}
            self.pending_capture_state: Optional[str] = None
            self.pending_capture_start_row = 0
            self.pending_capture_target_rows = 0
            self.pending_capture_fs = 0.0
            self.pending_capture_start_ts = 0.0
            self.capture_highlights: List[Tuple[float, float]] = []
            self.capture_buttons: Dict[str, QtWidgets.QPushButton] = {}
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._poll)

            outer_layout = QtWidgets.QVBoxLayout(self)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            content = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(content)
            layout.setContentsMargins(32, 24, 32, 24)
            scroll.setWidget(content)
            outer_layout.addWidget(scroll)
            title = QtWidgets.QLabel("眼电/肌电模板训练")
            title.setObjectName("title")
            layout.addWidget(title)
            desc = QtWidgets.QLabel("")
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            layout.addWidget(desc)

            card = Card()
            form = QtWidgets.QFormLayout(card)
            self.template_seconds_spin = QtWidgets.QDoubleSpinBox()
            self.template_seconds_spin.setRange(0.2, 3.0)
            self.template_seconds_spin.setSingleStep(0.1)
            self.template_seconds_spin.setValue(1.0)
            self.template_seconds_spin.setSuffix(" s")
            form.addRow("采集搜索窗口", self.template_seconds_spin)
            self.ncc_snippet_seconds_spin = QtWidgets.QDoubleSpinBox()
            self.ncc_snippet_seconds_spin.setRange(0.2, 0.8)
            self.ncc_snippet_seconds_spin.setSingleStep(0.05)
            self.ncc_snippet_seconds_spin.setValue(0.4)
            self.ncc_snippet_seconds_spin.setSuffix(" s")
            form.addRow("NCC 动作片段", self.ncc_snippet_seconds_spin)
            self.path_edit = QtWidgets.QLineEdit(self.config.masseter_template_edit.text().strip() or os.path.join(APP_DIR, "models", "masseter_ncc_templates.json"))
            form.addRow("保存路径", self.path_edit)
            buttons = QtWidgets.QHBoxLayout()
            self.connect_button = QtWidgets.QPushButton("启用模板采集")
            self.connect_button.clicked.connect(self.start)
            self.stop_button = QtWidgets.QPushButton("停止")
            self.stop_button.clicked.connect(self.stop)
            self.save_button = QtWidgets.QPushButton("保存模板")
            self.save_button.setObjectName("primaryButton")
            self.save_button.clicked.connect(self.save_templates)
            for button in [self.connect_button, self.stop_button, self.save_button]:
                buttons.addWidget(button)
            form.addRow(buttons)
            self.connect_progress = QtWidgets.QProgressBar()
            self.connect_progress.setRange(0, 100)
            self.connect_progress.setValue(0)
            self.connect_progress.setFormat("未订阅系统 LSL")
            form.addRow(self.connect_progress)
            layout.addWidget(card)

            grid_card = Card()
            grid = QtWidgets.QGridLayout(grid_card)
            self.count_labels: Dict[str, QtWidgets.QLabel] = {}
            self.undo_buttons: Dict[str, QtWidgets.QPushButton] = {}
            self.clear_buttons: Dict[str, QtWidgets.QPushButton] = {}
            for idx, (state, label) in enumerate(self.STATES):
                button = QtWidgets.QPushButton(f"采集 {label}")
                button.setFixedWidth(210)
                button.clicked.connect(lambda _=False, s=state: self.capture_state(s))
                count = QtWidgets.QLabel("0 段")
                count.setFixedWidth(56)
                self.count_labels[state] = count
                self.capture_buttons[state] = button
                undo_button = QtWidgets.QPushButton("撤销上次")
                undo_button.setFixedWidth(92)
                undo_button.clicked.connect(lambda _=False, s=state: self.undo_last_state(s))
                clear_button = QtWidgets.QPushButton("清空当前类")
                clear_button.setFixedWidth(104)
                clear_button.clicked.connect(lambda _=False, s=state: self.clear_state(s))
                self.undo_buttons[state] = undo_button
                self.clear_buttons[state] = clear_button
                grid.addWidget(button, idx, 0)
                grid.addWidget(count, idx, 1)
                grid.addWidget(undo_button, idx, 2)
                grid.addWidget(clear_button, idx, 3)
            layout.addWidget(grid_card)

            scale_controls = QtWidgets.QHBoxLayout()
            self.trace_auto_scale_check = QtWidgets.QCheckBox("自动量程")
            self.trace_auto_scale_check.setChecked(True)
            self.trace_auto_scale_check.setToolTip("分别根据 AUX1-AUX4 当前显示数据自动设置纵向量程。")
            self.trace_fixed_scale_spin = QtWidgets.QDoubleSpinBox()
            self.trace_fixed_scale_spin.setRange(1.0, 50000000.0)
            self.trace_fixed_scale_spin.setDecimals(1)
            self.trace_fixed_scale_spin.setValue(500.0)
            self.trace_fixed_scale_spin.setSuffix(" uV")
            self.trace_fixed_scale_spin.setToolTip("关闭自动量程后，四个通道使用此固定量程。")
            self.trace_fixed_scale_spin.setEnabled(False)
            self.trace_auto_scale_check.toggled.connect(self._trace_scale_mode_changed)
            self.trace_fixed_scale_spin.valueChanged.connect(self._refresh_trace_scales)
            scale_controls.addWidget(self.trace_auto_scale_check)
            scale_controls.addWidget(QtWidgets.QLabel("固定量程"))
            scale_controls.addWidget(self.trace_fixed_scale_spin)
            scale_controls.addStretch(1)
            layout.addLayout(scale_controls)

            waveform_tabs = QtWidgets.QTabWidget()
            live_tab = QtWidgets.QWidget()
            live_layout = QtWidgets.QVBoxLayout(live_tab)
            live_layout.setContentsMargins(0, 0, 0, 0)
            self.live_trace = RawTraceWidget(["AUX1 左眼", "AUX2 右眼", "AUX3 左咬肌", "AUX4 右咬肌"], row_height=64)
            self.live_trace.set_config(10.0, 1.0)
            live_layout.addWidget(self.live_trace)
            waveform_tabs.addTab(live_tab, "实时 AUX 波形")

            preview_tab = QtWidgets.QWidget()
            preview_layout = QtWidgets.QVBoxLayout(preview_tab)
            preview_layout.setContentsMargins(0, 0, 0, 0)
            self.template_preview_label = QtWidgets.QLabel("尚未截取模板")
            self.template_preview_label.setObjectName("vrStatus")
            preview_layout.addWidget(self.template_preview_label)
            self.template_trace = RawTraceWidget(["AUX1 左眼", "AUX2 右眼", "AUX3 左咬肌", "AUX4 右咬肌"], row_height=64)
            self.template_trace.set_config(float(self.template_seconds_spin.value()), 1.0)
            preview_layout.addWidget(self.template_trace)
            waveform_tabs.addTab(preview_tab, "已截取模板")
            self.waveform_tabs = waveform_tabs
            waveform_tabs.setMinimumHeight(440)
            layout.addWidget(waveform_tabs, 1)

            hint = QtWidgets.QLabel("")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            self.status_label = QtWidgets.QLabel("状态：未连接")
            self.status_label.setObjectName("vrStatus")
            layout.addWidget(self.status_label)
            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(100)
            layout.addWidget(self.log)
            layout.addStretch(1)

        def _append_log(self, text: str) -> None:
            self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

        def _refresh_count(self, state: str) -> None:
            self.count_labels[str(state)].setText(f"{len(self.samples[str(state)])} 段")

        def _set_capture_buttons_enabled(self, enabled: bool) -> None:
            for button in self.capture_buttons.values():
                button.setEnabled(enabled)

        def _reset_pending_capture(self) -> None:
            self.pending_capture_state = None
            self.pending_capture_start_row = 0
            self.pending_capture_target_rows = 0
            self.pending_capture_fs = 0.0
            self.pending_capture_start_ts = 0.0
            self._set_capture_buttons_enabled(True)

        def _connect_worker_running(self) -> bool:
            worker = self.connect_worker
            if worker is None:
                return False
            try:
                running = bool(worker.isRunning())
                if not running:
                    self.connect_worker = None
                    self.connect_button.setEnabled(True)
                    worker.deleteLater()
                return running
            except RuntimeError:
                self.connect_worker = None
                return False

        def _on_connect_worker_finished(self) -> None:
            worker = self.sender()
            if worker is self.connect_worker:
                self.connect_worker = None
            self.connect_button.setEnabled(True)
            worker.deleteLater()

        def start(self) -> None:
            if self.source is not None or self._connect_worker_running():
                return
            self.connect_button.setEnabled(False)
            worker = LslAcquireWorker("emg_template_train", 8.0, self)
            worker.progress_signal.connect(self._on_connect_progress)
            worker.ready_signal.connect(self._on_connect_ready)
            worker.error_signal.connect(self._on_connect_error)
            worker.finished.connect(self._on_connect_worker_finished)
            self.connect_worker = worker
            worker.start()

        def _on_connect_progress(self, percent: int, stage: str) -> None:
            self.connect_progress.setValue(percent)
            self.connect_progress.setFormat(f"{stage}：%p%")
            self.status_label.setText(f"状态：{stage}")

        def _on_connect_ready(self, source) -> None:
            self.source = source
            self.timer.start(40)
            self.connect_progress.setValue(100)
            self.connect_progress.setFormat("已订阅系统 LSL：%p%")
            self.status_label.setText("状态：已订阅系统 LSL；可开始采集模板")
            self._append_log(f"已订阅系统 LSL：{self.source.description}")
            self.connect_button.setEnabled(True)

        def _on_connect_error(self, message: str) -> None:
            self.source = None
            self.connect_progress.setValue(0)
            self.connect_progress.setFormat("系统 LSL 订阅失败")
            self.status_label.setText(f"状态：未订阅系统 LSL | {message}")
            self._append_log(f"系统 LSL 订阅失败：{message}")
            self.connect_button.setEnabled(True)

        def stop(self) -> None:
            if self._connect_worker_running():
                self.connect_worker.requestInterruption()
            self.timer.stop()
            self._reset_pending_capture()
            if self.source is not None:
                try:
                    self.source.close()
                except Exception:
                    pass
            self.source = None
            self.status_label.setText("状态：已停止")
            self.connect_progress.setValue(0)
            self.connect_progress.setFormat("未订阅系统 LSL")
            self.connect_button.setEnabled(True)

        def _update_pending_capture_status(self) -> None:
            if self.pending_capture_state is None:
                return
            have_rows = max(0, self.buffer.shape[0] - self.pending_capture_start_row)
            remaining_rows = max(0, self.pending_capture_target_rows - have_rows)
            remaining_sec = remaining_rows / self.pending_capture_fs if self.pending_capture_fs > 0 else 0.0
            self.status_label.setText(f"状态：正在采集 {self.pending_capture_state}，剩余 {remaining_sec:.1f} s")

        def _complete_pending_capture_if_ready(self) -> None:
            if self.pending_capture_state is None:
                return
            have_rows = self.buffer.shape[0] - self.pending_capture_start_row
            if have_rows < self.pending_capture_target_rows:
                self._update_pending_capture_status()
                return
            end_row = self.pending_capture_start_row + self.pending_capture_target_rows
            state = str(self.pending_capture_state)
            aux = self.buffer[self.pending_capture_start_row:end_row, :].T.astype(np.float32)
            self.samples[state].append(aux)
            start_ts = float(self.timestamps[self.pending_capture_start_row]) if self.timestamps.size > self.pending_capture_start_row else self.pending_capture_start_ts
            end_index = min(max(0, end_row - 1), max(0, self.timestamps.size - 1))
            end_ts = float(self.timestamps[end_index]) if self.timestamps.size else start_ts + self.pending_capture_target_rows / max(self.pending_capture_fs, 1.0)
            self.capture_highlights.append((start_ts, end_ts))
            preview_data = aux.T
            preview_t = np.arange(preview_data.shape[0], dtype=np.float64) / max(self.pending_capture_fs, 1.0)
            seconds = max(float(self.template_seconds_spin.value()), float(preview_t[-1]) if preview_t.size else 1.0)
            self.template_trace.set_config(seconds, self._trace_scales(preview_data))
            self.template_trace.set_trace_data(preview_t, preview_data, np.ones(4, dtype=bool))
            self.template_trace.set_highlight_ranges([(0.0, seconds)])
            label = dict(self.STATES).get(state, state)
            self.template_preview_label.setText(f"最近截取：{state} {label} | {preview_data.shape[0]} 点 | {seconds:.2f}s")
            self.waveform_tabs.setCurrentIndex(1)
            self._refresh_count(state)
            self._append_log(f"已采集 {state} 模板段：采集的是点击后的窗口。")
            self._reset_pending_capture()
            self.status_label.setText("状态：采集完成；可继续采集、撤销或保存")

        def _poll(self) -> None:
            if self.source is None:
                return
            chunk, ts = self.source.read_all_available()
            if chunk is None or ts is None:
                return
            aux_indices = [idx for idx in self.source.indices[:4] if idx < int(self.source.channel_count)]
            if not aux_indices:
                return
            aux = np.asarray(chunk[:, aux_indices], dtype=np.float32)
            if aux.shape[1] < 4:
                padded = np.zeros((aux.shape[0], 4), dtype=np.float32)
                padded[:, : aux.shape[1]] = aux
                aux = padded
            self.buffer = np.vstack([self.buffer, aux]) if self.buffer.size else aux
            self.timestamps = np.concatenate([self.timestamps, np.asarray(ts, dtype=np.float64)])
            max_rows = int(max(10.0, float(self.template_seconds_spin.value()) * 8.0) * float(getattr(self.source, "fs", 250.0)))
            if self.buffer.shape[0] > max_rows:
                drop_rows = self.buffer.shape[0] - max_rows
                self.buffer = self.buffer[-max_rows:, :]
                self.timestamps = self.timestamps[-max_rows:]
                if self.pending_capture_state is not None:
                    self.pending_capture_start_row = max(0, self.pending_capture_start_row - drop_rows)
            self._complete_pending_capture_if_ready()
            self._refresh_waveforms()

        def _trace_scales(self, data: np.ndarray) -> np.ndarray:
            if not self.trace_auto_scale_check.isChecked():
                return np.full(4, float(self.trace_fixed_scale_spin.value()), dtype=np.float64)
            if data.ndim != 2 or data.shape[0] == 0:
                return np.ones(4, dtype=np.float64)
            centered = data - np.median(data, axis=0, keepdims=True)
            # Add headroom to keep peaks inside each trace row.
            robust_peak = np.nanpercentile(np.abs(centered), 99, axis=0)
            return np.maximum(robust_peak / 0.42, 1.0)

        def _trace_scale_mode_changed(self, automatic: bool) -> None:
            self.trace_fixed_scale_spin.setEnabled(not automatic)
            self._refresh_trace_scales()

        def _refresh_trace_scales(self, *_args) -> None:
            for trace in (self.live_trace, self.template_trace):
                data = np.asarray(trace.data, dtype=np.float32)
                if data.ndim == 2 and data.shape[0] > 0:
                    trace.set_config(trace.page_seconds, self._trace_scales(data))

        def _refresh_waveforms(self) -> None:
            if self.buffer.shape[0] == 0 or self.timestamps.size == 0:
                return
            page_seconds = 10.0
            end_ts = float(self.timestamps[-1])
            start_ts = end_ts - page_seconds
            start_row = int(np.searchsorted(self.timestamps, start_ts, side="left"))
            data = self.buffer[start_row:, :]
            t = self.timestamps[start_row:] - start_ts
            self.live_trace.set_config(page_seconds, self._trace_scales(data))
            self.live_trace.set_trace_data(t, data, np.ones(4, dtype=bool))
            highlights = [(start - start_ts, end - start_ts) for start, end in self.capture_highlights]
            if self.pending_capture_state is not None and self.pending_capture_start_ts > 0.0:
                highlights.append((self.pending_capture_start_ts - start_ts, end_ts - start_ts))
            self.live_trace.set_highlight_ranges(highlights)

        def capture_state(self, state: str) -> None:
            if self.source is None:
                self._append_log("请先启用模板采集并等待系统 LSL 订阅完成。")
                return
            self._poll()
            if self.pending_capture_state is not None:
                self._append_log("已有模板采集在进行中，请先等待完成。")
                return
            fs = float(getattr(self.source, "fs", self.config.fs_spin.value()) if self.source is not None else self.config.fs_spin.value())
            rows = max(8, int(round(float(self.template_seconds_spin.value()) * fs)))
            self.pending_capture_state = str(state)
            self.pending_capture_start_row = self.buffer.shape[0]
            self.pending_capture_target_rows = rows
            self.pending_capture_fs = fs
            self.pending_capture_start_ts = float(self.timestamps[-1]) if self.timestamps.size else 0.0
            self._set_capture_buttons_enabled(False)
            self.status_label.setText(f"状态：正在采集 {state}，请现在执行，剩余 {float(self.template_seconds_spin.value()):.1f} s")
            self._append_log(f"开始采集 {state} 模板：采集按钮点击后的窗口。")
            self._complete_pending_capture_if_ready()

        def undo_last_state(self, state: str) -> None:
            state = str(state)
            if not self.samples[state]:
                self._append_log(f"{state} 没有可撤销的模板段。")
                return
            self.samples[state].pop()
            self._refresh_count(state)
            self.status_label.setText(f"状态：已撤销 {state} 最后一段")
            self._append_log(f"已撤销 {state} 最后一段模板。")

        def clear_state(self, state: str) -> None:
            state = str(state)
            self.samples[state].clear()
            self._refresh_count(state)
            self.status_label.setText(f"状态：已清空 {state} 当前类")
            self._append_log(f"已清空 {state} 当前类模板。")

        def save_templates(self) -> None:
            fs = float(getattr(self.source, "fs", self.config.fs_spin.value()) if self.source is not None else self.config.fs_spin.value())
            capture_seconds = float(self.template_seconds_spin.value())
            snippet_seconds = float(self.ncc_snippet_seconds_spin.value())
            states = {}
            for state, windows in self.samples.items():
                if not windows:
                    continue
                if state in self.EYE_STATES:
                    states[state] = make_eye_template(windows, fs=fs, template_seconds=snippet_seconds)
                else:
                    states[state] = make_masseter_template(windows, fs=fs, template_seconds=snippet_seconds)
            if not states:
                QtWidgets.QMessageBox.warning(self, "没有模板", "请至少采集一个模板段。")
                return
            path = save_masseter_template_bank(
                self.path_edit.text().strip(),
                MasseterTemplateBank(fs=fs, template_seconds=snippet_seconds, states=states),
            )
            self.config.masseter_template_edit.setText(path)
            self._append_log(
                f"模板已保存：{path} | 搜索窗口 {capture_seconds:.2f}s | NCC 片段 {snippet_seconds:.2f}s"
            )
            QtWidgets.QMessageBox.information(self, "已保存", f"NCC 眼电/肌电模板已保存：\n{path}")

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(CONSOLE_WINDOW_TITLE)
            self.resize(1680, 1020)
            self.setMinimumSize(1280, 800)
            self.bus = CommandBus()
            self.test_bus = CommandBus()
            self.page_names = list(CONSOLE_PAGE_NAMES)
            self.page_names = [name for name in self.page_names if name not in {"肌电模板训练", "眼电模板训练"}]
            if "眼电/肌电模板训练" not in self.page_names:
                self.page_names.insert(2, "眼电/肌电模板训练")
            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)

            sidebar = QtWidgets.QFrame()
            sidebar.setObjectName("sidebar")
            side_layout = QtWidgets.QVBoxLayout(sidebar)
            side_layout.setContentsMargins(16, 18, 16, 18)
            brand = QtWidgets.QLabel(CONSOLE_BRAND)
            brand.setObjectName("brand")
            side_layout.addWidget(brand)
            sub = QtWidgets.QLabel(CONSOLE_SUBTITLE)
            sub.setObjectName("subbrand")
            side_layout.addWidget(sub)
            side_layout.addSpacing(8)
            self.buttons = []
            nav_icons = list(CONSOLE_NAV_CODES)
            if len(nav_icons) < len(self.page_names):
                nav_icons = [f"{idx + 1:02d}" for idx in range(len(self.page_names))]
            for idx, name in enumerate(self.page_names):
                btn = QtWidgets.QPushButton(f"  {nav_icons[idx]}  {name}")
                btn.setObjectName("nav")
                btn.clicked.connect(lambda _=False, i=idx: self.switch_page(i))
                side_layout.addWidget(btn)
                self.buttons.append(btn)
            side_layout.addStretch(1)
            workflow = QtWidgets.QLabel(CONSOLE_WORKFLOW_TEXT)
            workflow.setObjectName("workflow")
            side_layout.addWidget(workflow)
            hint = QtWidgets.QLabel(CONSOLE_HINT_TEXT)
            hint.setWordWrap(True)
            hint.setObjectName("hint")
            side_layout.addWidget(hint)
            layout.addWidget(sidebar)

            content_shell = QtWidgets.QWidget()
            content_layout = QtWidgets.QVBoxLayout(content_shell)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(0)
            topbar = QtWidgets.QFrame()
            topbar.setObjectName("topbar")
            topbar_layout = QtWidgets.QHBoxLayout(topbar)
            topbar_layout.setContentsMargins(22, 10, 22, 10)
            self.page_title = QtWidgets.QLabel(self.page_names[0])
            self.page_title.setObjectName("pageTitle")
            topbar_layout.addWidget(self.page_title)
            topbar_layout.addStretch(1)
            content_layout.addWidget(topbar)

            self.stack = QtWidgets.QStackedWidget()
            self.config_page = ConfigPage()
            self.emg_test_page = EmgTestPage()
            self.emg_template_page = EmgTemplateTrainPage(self.config_page)
            self.eeg8_page = ActiveEEG8CollectionPage(self.config_page)
            self.train_page = TrainTunePage(self.config_page)
            self.test_page = VRRealtimeControlPage(self.test_bus, self.config_page)
            self.robot_forwarder = NetworkRobotForwarder(
                self.config_page,
                session_provider=lambda: self.test_page.runtime_session,
                parent=self,
            )
            self.test_command_fanout = CommandPayloadFanout(
                self.robot_forwarder.execute_payload,
            )
            self.test_bus.command_emitted.connect(self.test_command_fanout.dispatch)
            self.eeg8_page.finished_signal.connect(self.train_page.recording_added)
            self.train_page.model_ready.connect(self.config_page.set_model_path)
            self.train_page.model_ready.connect(self.test_page.set_model_path)
            for page in [self.config_page, self.emg_test_page, self.emg_template_page, self.eeg8_page, self.train_page, self.test_page]:
                self.stack.addWidget(page)
            content_layout.addWidget(self.stack, 1)
            layout.addWidget(content_shell, 1)
            self.hide_hint_labels()
            self.switch_page(0)
            self.apply_style()

        def closeEvent(self, event) -> None:
            self.robot_forwarder.close()
            super().closeEvent(event)

        def switch_page(self, index: int) -> None:
            self.stack.setCurrentIndex(index)
            self.page_title.setText(self.page_names[index])
            for i, btn in enumerate(self.buttons):
                btn.setProperty("active", i == index)
                btn.style().unpolish(btn)
                btn.style().polish(btn)

        def hide_hint_labels(self) -> None:
            for label in self.findChildren(QtWidgets.QLabel):
                if label.objectName() in {"desc", "hint", "workflow", "subbrand", "vrInfo"}:
                    label.setVisible(False)

        def apply_style(self) -> None:
            self.setStyleSheet(
                """
                QWidget { background: #f4f7fa; color: #1f2a35; font-family: "Microsoft YaHei UI", "Segoe UI", Arial; font-size: 14px; }
                #sidebar { background: #ffffff; border-right: 1px solid #dbe3ea; min-width: 218px; max-width: 218px; }
                #brand { font-size: 26px; font-weight: 800; color: #123042; }
                #subbrand, #desc, #hint { color: #647382; }
                #hint { font-size: 12px; }
                #workflow { color: #314253; background: #f2f6f9; border: 1px solid #d7e1ea; border-radius: 6px; padding: 10px; line-height: 1.6; }
                #topbar { background: #ffffff; border-bottom: 1px solid #dbe3ea; min-height: 50px; max-height: 50px; }
                #pageTitle { font-size: 16px; font-weight: 800; color: #172635; }
                #title { font-size: 25px; font-weight: 800; color: #172635; margin: 5px 0; }
                #sectionTitle { font-size: 15px; font-weight: 800; color: #243444; }
                QPushButton#nav { text-align: left; min-height: 42px; padding: 0 13px; border-radius: 6px; border: 1px solid transparent; background: transparent; color: #617182; font-weight: 700; }
                QPushButton#nav:hover { background: #eef4f8; color: #172635; }
                QPushButton#nav[active="true"] { background: #e5f3f8; color: #0b5e76; border-color: #b9ddea; border-left: 4px solid #1787a6; }
                QPushButton { background: #ffffff; border: 1px solid #cbd7e2; color: #23313d; padding: 8px 13px; border-radius: 6px; min-height: 20px; font-weight: 600; }
                QPushButton:hover { background: #f1f6fa; border-color: #9fb5c8; }
                QPushButton#primaryButton { background: #137fa2; border-color: #137fa2; color: #ffffff; font-weight: 800; }
                QPushButton#primaryButton:hover { background: #0f6d8d; border-color: #0f6d8d; }
                QPushButton#dangerButton { background: #fff1f1; border-color: #efb1b1; color: #9b1c1c; }
                QPushButton#dangerButton:hover { background: #ffe3e3; border-color: #df8d8d; }
                QPushButton:disabled { background: #eef2f6; color: #9aa7b3; border-color: #d8e0e7; }
                QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #ffffff; border: 1px solid #cbd7e2; border-radius: 5px; padding: 7px 9px; color: #1f2a35; selection-background-color: #b9e4ef; }
                QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #1787a6; }
                QCheckBox { color: #243444; spacing: 8px; }
                QCheckBox::indicator { width: 16px; height: 16px; }
                #card { background: #ffffff; border: 1px solid #dbe3ea; border-radius: 7px; padding: 10px; }
                #paradigmStage { background: #eef6f9; border: 1px solid #b9ddea; border-radius: 7px; }
                #cueWord { font-size: 56px; font-weight: 900; color: #123042; }
                #phaseInstruction { font-size: 18px; color: #356172; }
                #phaseBadge { background: #e7f4f8; border: 1px solid #b9ddea; border-radius: 5px; color: #174f62; padding: 6px 12px; }
                #metricImage { background: #ffffff; border: 1px solid #dbe3ea; border-radius: 7px; padding: 6px; color: #647382; }
                #countdown { font-size: 24px; font-weight: 800; color: #123042; background: transparent; }
                QProgressBar { background: #edf2f6; border: 1px solid #cbd7e2; border-radius: 5px; min-height: 13px; text-align: center; color: #23313d; }
                QProgressBar::chunk { background: #1787a6; border-radius: 4px; }
                QScrollArea { background: transparent; border: 0; }
                #vrTitle { font-size: 20px; font-weight: 800; color: #172635; }
                #vrBadge { background: #e7f4f8; border: 1px solid #b9ddea; border-radius: 5px; color: #174f62; padding: 7px 12px; }
                #vrStage { background: #091016; border: 1px solid #2f414d; border-radius: 7px; }
                #vrControls { background: #ffffff; border: 1px solid #dbe3ea; border-radius: 7px; }
                QPushButton#vrCmd { background: #eef6f9; border: 1px solid #b9ddea; border-radius: 5px; min-height: 32px; font-size: 14px; font-weight: 700; color: #174f62; }
                QPushButton#vrCmd:hover { background: #dff0f6; }
                QPushButton#vrCmd[active="true"] { background: #137fa2; border-color: #137fa2; color: #ffffff; }
                #vrStatus { color: #245367; background: transparent; padding: 4px 8px; font-weight: 600; }
                QTextEdit { font-family: Consolas, "Cascadia Mono"; }
                QToolTip { background: #ffffff; color: #172635; border: 1px solid #9fb5c8; padding: 5px; }
                """
            )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.aboutToQuit.connect(_SYSTEM_LSL_MANAGER.close_system)
    if sys.platform.startswith("win"):
        for font_path in (
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
        ):
            if os.path.exists(font_path):
                QtGui.QFontDatabase.addApplicationFont(font_path)
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
