# -*- coding: utf-8 -*-
"""PyQt GUI for the arm BCI system."""

from __future__ import annotations

from .commands import apply_runtime_control_config, metrics_summary
from .core import *

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
    except Exception as exc:
        print(f"ERROR: pybullet is required for the 3D physics VR page: {exc}")
        print("Install it in your environment, for example:")
        print("  conda install -n bciwork_1 -c conda-forge pybullet=3.25")
        return 1
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
            payload = asdict(command)
            self.bus.command_emitted.emit(payload)
            if self.log_callback:
                self.log_callback(
                    f"机械臂指令 seq={command.sequence_id} cmd={command.name} "
                    f"intent={command.intent} conf={command.confidence:.3f} duration={command.duration_seconds:.2f}s"
                )

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

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
            rect = self.rect()
            painter.fillRect(rect, QtGui.QColor("#fbf6dd"))

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

            painter.fillRect(QtCore.QRect(0, 0, label_w, rect.height()), QtGui.QColor("#f2f2f2"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#c8d0d8"), 1))
            painter.drawLine(label_w, 0, label_w, rect.height())

            minor_step = 0.5 if self.page_seconds <= 30 else 1.0
            major_step = 5.0 if self.page_seconds >= 10 else 1.0
            x = 0.0
            while x <= self.page_seconds + 1e-9:
                px = plot_left + int(round((x / self.page_seconds) * plot_w))
                is_major = abs((x / major_step) - round(x / major_step)) < 1e-6
                color = QtGui.QColor("#8bd36b") if is_major else QtGui.QColor("#b8dda2")
                pen = QtGui.QPen(color, 1)
                if not is_major:
                    pen.setStyle(QtCore.Qt.PenStyle.DotLine)
                painter.setPen(pen)
                painter.drawLine(px, plot_top, px, plot_bottom)
                if is_major:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#333333"), 1))
                    painter.drawText(px - 12, plot_bottom + 18, 28, 18, QtCore.Qt.AlignmentFlag.AlignCenter, f"{int(x)}")
                x += minor_step

            painter.setPen(QtGui.QPen(QtGui.QColor("#d1d1bd"), 1))
            for idx in range(n_channels + 1):
                y = int(round(plot_top + idx * row_h))
                painter.drawLine(0, y, plot_right, y)

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

    class RDA64CollectionWorker(QtCore.QThread):
        log_signal = QtCore.pyqtSignal(str)
        phase_signal = QtCore.pyqtSignal(str, str, int, int, float)
        finished_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(
            self,
            out_path: str,
            trials_per_class: int,
            epoch_seconds: float,
            pre_rest_seconds: float,
            cue_seconds: float,
            cue_rest_seconds: float,
            post_rest_seconds: float,
            host: str,
            port: int,
            expected_channels: int,
            aux_channels: int,
            use_parallel_trigger: bool,
            parallel_port: int,
            parent=None,
        ):
            super().__init__(parent)
            self.out_path = out_path
            self.trials_per_class = trials_per_class
            self.epoch_seconds = epoch_seconds
            self.pre_rest_seconds = pre_rest_seconds
            self.cue_seconds = cue_seconds
            self.cue_rest_seconds = cue_rest_seconds
            self.post_rest_seconds = post_rest_seconds
            self.host = host
            self.port = port
            self.expected_channels = expected_channels
            self.aux_channels = aux_channels
            self.use_parallel_trigger = use_parallel_trigger
            self.parallel_port = parallel_port
            self.stop_event = threading.Event()

        def stop(self) -> None:
            self.stop_event.set()

        def run(self) -> None:
            try:
                result = collect_rda64_mi8_calibration(
                    out_path=self.out_path,
                    trials_per_class=self.trials_per_class,
                    epoch_seconds=self.epoch_seconds,
                    pre_rest_seconds=self.pre_rest_seconds,
                    cue_seconds=self.cue_seconds,
                    cue_rest_seconds=self.cue_rest_seconds,
                    post_rest_seconds=self.post_rest_seconds,
                    host=self.host,
                    port=self.port,
                    expected_channels=self.expected_channels,
                    aux_channels=self.aux_channels,
                    use_parallel_trigger=self.use_parallel_trigger,
                    parallel_port=self.parallel_port,
                    log_callback=self.log_signal.emit,
                    cue_callback=lambda label, phase, idx, total, seconds: self.phase_signal.emit(
                        str(label), str(phase), int(idx), int(total), float(seconds)
                    ),
                    stop_event=self.stop_event,
                )
                self.finished_signal.emit(result)
            except Exception as exc:
                self.error_signal.emit(str(exc))

    class SimulatedCollectionWorker(QtCore.QThread):
        log_signal = QtCore.pyqtSignal(str)
        phase_signal = QtCore.pyqtSignal(str, str, int, int, float)
        finished_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(
            self,
            out_path: str,
            fs: int,
            trials_per_class: int,
            epoch_seconds: float,
            pre_rest_seconds: float,
            cue_seconds: float,
            cue_rest_seconds: float,
            post_rest_seconds: float,
            parent=None,
        ):
            super().__init__(parent)
            self.out_path = out_path
            self.fs = int(fs)
            self.trials_per_class = int(trials_per_class)
            self.epoch_seconds = float(epoch_seconds)
            self.pre_rest_seconds = float(pre_rest_seconds)
            self.cue_seconds = float(cue_seconds)
            self.cue_rest_seconds = float(cue_rest_seconds)
            self.post_rest_seconds = float(post_rest_seconds)
            self.stop_event = threading.Event()

        def stop(self) -> None:
            self.stop_event.set()

        def _wait(self, seconds: float) -> None:
            start = time.perf_counter()
            while time.perf_counter() - start < seconds:
                if self.stop_event.is_set():
                    break
                time.sleep(0.02)

        def run(self) -> None:
            try:
                trial_order = list(LABELS) * self.trials_per_class
                random.shuffle(trial_order)
                trial_table: List[Dict] = []
                phase_plan = [
                    ("pre_rest", self.pre_rest_seconds, "休息"),
                    ("cue", self.cue_seconds, "提示"),
                    ("cue_rest", self.cue_rest_seconds, "休息"),
                    ("imagery", self.epoch_seconds, "采集/执行"),
                    ("post_rest", self.post_rest_seconds, "休息"),
                ]
                total = len(trial_order)
                self.log_signal.emit(f"开始执行模拟 6类咬肌+MI 范式：{total} 个 trial")
                for idx, label in enumerate(trial_order, start=1):
                    if self.stop_event.is_set():
                        self.log_signal.emit("模拟范式已停止，未生成数据。")
                        return
                    display = LABEL_DISPLAY_NAMES.get(label, label)
                    info = paradigm_info(label)
                    row = {"trial": idx, "label": label, "display": display, "paradigm": info, "events": []}
                    self.log_signal.emit(f"Trial {idx}/{total}: {display} ({label}) | {info['modality']} {info['region']} | {info['subject_task']}")
                    for phase, seconds, phase_display in phase_plan:
                        trigger = MODEL_TRIGGER_CODES[label] if phase == "imagery" else (REST_TRIGGER_CODE if phase.endswith("rest") else None)
                        row["events"].append(
                            {
                                "phase": phase,
                                "display": info["subject_task"] if phase in {"cue", "imagery"} else phase_display,
                                "seconds": float(seconds),
                                "trigger": trigger,
                            }
                        )
                        self.phase_signal.emit(str(label), str(phase), int(idx), int(total), float(seconds))
                        self._wait(float(seconds))
                        if self.stop_event.is_set():
                            self.log_signal.emit("模拟范式已停止，未生成数据。")
                            return
                    trial_table.append(row)

                cfg = SignalConfig(fs=self.fs, calibration_trial_seconds=self.epoch_seconds)
                eeg, y, aux = build_demo_calibration(cfg, trials_per_class=self.trials_per_class)
                os.makedirs(os.path.dirname(os.path.abspath(self.out_path)) or ".", exist_ok=True)
                np.savez_compressed(
                    self.out_path,
                    eeg=eeg,
                    y=y,
                    aux=aux,
                    fs=cfg.fs,
                    channel_names=np.asarray(cfg.eeg_channels, dtype=object),
                    aux_channel_names=np.asarray(["LeftMasseter", "RightMasseter", "VirtualBlink"], dtype=object),
                    timeline_seconds=np.asarray(
                        [self.pre_rest_seconds, self.cue_seconds, self.cue_rest_seconds, self.epoch_seconds, self.post_rest_seconds],
                        dtype=np.float32,
                    ),
                    timeline_labels=np.asarray(["rest", "cue", "rest", "imagery", "rest"], dtype=object),
                    trial_order=np.asarray(trial_order, dtype=object),
                    trial_table_json=json.dumps(trial_table, ensure_ascii=False),
                    paradigm_name=PARADIGM_NAME,
                    paradigm_json=json.dumps(MI8_PARADIGM, ensure_ascii=False),
                    mode="simulated",
                )
                self.finished_signal.emit(
                    {"path": self.out_path, "fs": cfg.fs, "eeg_shape": eeg.shape, "aux_shape": aux.shape, "labels": y.tolist()}
                )
            except Exception as exc:
                self.error_signal.emit(str(exc))

    class LslCollectionWorker(QtCore.QThread):
        log_signal = QtCore.pyqtSignal(str)
        phase_signal = QtCore.pyqtSignal(str, str, int, int, float)
        stream_signal = QtCore.pyqtSignal(object)
        finished_signal = QtCore.pyqtSignal(object)
        error_signal = QtCore.pyqtSignal(str)

        def __init__(
            self,
            out_path: str,
            trials_per_class: int,
            epoch_seconds: float,
            pre_rest_seconds: float,
            cue_seconds: float,
            cue_rest_seconds: float,
            post_rest_seconds: float,
            expected_eeg_channels: int = 32,
            aux_channels: int = 4,
            parent=None,
        ):
            super().__init__(parent)
            self.out_path = out_path
            self.trials_per_class = int(trials_per_class)
            self.epoch_seconds = float(epoch_seconds)
            self.pre_rest_seconds = float(pre_rest_seconds)
            self.cue_seconds = float(cue_seconds)
            self.cue_rest_seconds = float(cue_rest_seconds)
            self.post_rest_seconds = float(post_rest_seconds)
            self.expected_eeg_channels = int(expected_eeg_channels)
            self.aux_channels = int(aux_channels)
            self.stop_event = threading.Event()
            self.phase_displayed_event = threading.Event()
            self.raw_chunks: List[np.ndarray] = []
            self.timestamp_chunks: List[np.ndarray] = []

        def stop(self) -> None:
            self.stop_event.set()
            self.phase_displayed_event.set()

        def acknowledge_phase_displayed(self) -> None:
            self.phase_displayed_event.set()

        def _show_phase(self, label: str, phase: str, idx: int, total: int, seconds: float) -> None:
            self.phase_displayed_event.clear()
            self.phase_signal.emit(str(label), str(phase), int(idx), int(total), float(seconds))
            deadline = time.perf_counter() + 2.0
            while not self.stop_event.is_set() and time.perf_counter() < deadline:
                if self.phase_displayed_event.wait(0.02):
                    return
            if not self.stop_event.is_set():
                self.log_signal.emit(f"界面阶段确认超时：{phase}；仍按配置时长 {seconds:.1f}s 采集。")

        def _record_for(self, source, seconds: float) -> Tuple[float, float]:
            phase_start = source.clock_now()
            wall_start = time.perf_counter()
            deadline = wall_start + max(0.0, float(seconds))
            while time.perf_counter() < deadline and not self.stop_event.is_set():
                chunk, timestamps = source.read_all_available()
                if chunk is not None and timestamps is not None:
                    self.raw_chunks.append(chunk)
                    self.timestamp_chunks.append(timestamps)
                time.sleep(0.01)
            return float(phase_start), float(time.perf_counter() - wall_start)

        def _save_recording(
            self,
            source,
            labels: Sequence[str],
            imagery_starts: Sequence[float],
            trial_table: Sequence[Dict],
            completed: bool,
            total_trials: int,
        ) -> Dict:
            chunk, timestamps = source.read_all_available()
            if chunk is not None and timestamps is not None:
                self.raw_chunks.append(chunk)
                self.timestamp_chunks.append(timestamps)
            if not self.raw_chunks or not imagery_starts:
                raise RuntimeError("LSL 已连接，但没有形成可保存的完整采集试次")
            raw = np.vstack(self.raw_chunks).astype(np.float32)
            ts = np.concatenate(self.timestamp_chunks).astype(np.float64)
            order = np.argsort(ts, kind="stable")
            raw = raw[order]
            ts = ts[order]
            unique = np.concatenate(([True], np.diff(ts) > 0))
            raw = raw[unique]
            ts = ts[unique]
            epochs = extract_timestamped_epochs(raw, ts, imagery_starts, source.fs, self.epoch_seconds)
            eeg_count = min(self.expected_eeg_channels, source.channel_count)
            eeg = epochs[:, :eeg_count, :]
            aux_indices = [idx for idx in source.indices[: self.aux_channels] if idx < source.channel_count]
            aux = epochs[:, aux_indices, :] if aux_indices else np.zeros((len(epochs), 0, epochs.shape[-1]), dtype=np.float32)
            channel_names = list(source.channel_names)
            reported_names = list(source.reported_channel_names)
            os.makedirs(os.path.dirname(os.path.abspath(self.out_path)) or ".", exist_ok=True)
            np.savez_compressed(
                self.out_path,
                eeg=eeg,
                aux=aux,
                y=np.asarray(labels, dtype=object),
                fs=np.asarray(int(round(source.fs)), dtype=np.int64),
                channel_names=np.asarray(channel_names[:eeg_count], dtype=object),
                aux_channel_names=np.asarray([channel_names[idx] for idx in aux_indices], dtype=object),
                raw_stream=raw,
                raw_timestamps=ts,
                raw_channel_names=np.asarray(channel_names, dtype=object),
                reported_channel_names=np.asarray(reported_names, dtype=object),
                imagery_start_timestamps=np.asarray(imagery_starts, dtype=np.float64),
                trial_table_json=json.dumps(list(trial_table), ensure_ascii=False),
                paradigm_name=PARADIGM_NAME,
                paradigm_json=json.dumps(MI8_PARADIGM, ensure_ascii=False),
                stream_name=source.stream_name,
                stream_type=source.stream_type,
                source_id=str(source.info.source_id()) if source.info is not None else "",
                mode="brainvision_lsl",
                completed=np.asarray(bool(completed)),
                planned_trials=np.asarray(int(total_trials), dtype=np.int64),
                completed_trials=np.asarray(len(labels), dtype=np.int64),
                used_default_montage=np.asarray(bool(source.using_default_montage)),
            )
            return {
                "path": self.out_path,
                "fs": int(round(source.fs)),
                "raw_shape": raw.shape,
                "eeg_shape": eeg.shape,
                "aux_shape": aux.shape,
                "labels": list(labels),
                "completed": bool(completed),
                "completed_trials": len(labels),
                "total_trials": int(total_trials),
            }

        def run(self) -> None:
            source = None
            labels: List[str] = []
            imagery_starts: List[float] = []
            trial_table: List[Dict] = []
            try:
                source = LslFourEmgSource(timeout=8.0)
                self.log_signal.emit(f"LSL 已连接：{source.description}")
                self.stream_signal.emit(
                    {
                        "name": source.stream_name,
                        "type": source.stream_type,
                        "channels": source.channel_count,
                        "fs": source.fs,
                        "eeg_count": min(self.expected_eeg_channels, source.channel_count),
                        "aux_channels": [idx + 1 for idx in source.indices[: self.aux_channels]],
                    }
                )
                if source.using_default_montage:
                    self.log_signal.emit("LSL EEG 标签为通用编号，已按系统默认 32 导顺序映射；请确认电极帽接线顺序一致。")
                source.read_all_available()
                trial_order = list(LABELS) * self.trials_per_class
                random.shuffle(trial_order)
                phase_plan = [
                    ("pre_rest", self.pre_rest_seconds, "休息"),
                    ("cue", self.cue_seconds, "提示"),
                    ("cue_rest", self.cue_rest_seconds, "休息"),
                    ("imagery", self.epoch_seconds, "采集/执行"),
                    ("post_rest", self.post_rest_seconds, "休息"),
                ]
                total = len(trial_order)
                self.log_signal.emit(f"开始 LSL 同步采集：{total} 个 trial，保存 EEG + AUX/EMG + 原始时间戳")
                self._show_phase(REST_LABEL, "arming", 0, total, 2.0)
                self._record_for(source, 2.0)
                for idx, label in enumerate(trial_order, start=1):
                    if self.stop_event.is_set():
                        break
                    info = paradigm_info(label)
                    row = {"trial": idx, "label": label, "paradigm": info, "events": []}
                    trial_complete = True
                    for phase, seconds, phase_display in phase_plan:
                        self._show_phase(label, phase, idx, total, float(seconds))
                        start, actual_seconds = self._record_for(source, seconds)
                        row["events"].append(
                            {
                                "phase": phase,
                                "display": info["subject_task"] if phase in {"cue", "imagery"} else phase_display,
                                "seconds": float(seconds),
                                "actual_seconds": float(actual_seconds),
                                "start_timestamp": float(start),
                                "trigger": MODEL_TRIGGER_CODES[label] if phase == "imagery" else None,
                            }
                        )
                        if phase == "imagery":
                            row["imagery_start_timestamp"] = float(start)
                        if self.stop_event.is_set():
                            trial_complete = False
                            break
                    if trial_complete:
                        labels.append(label)
                        imagery_starts.append(float(row["imagery_start_timestamp"]))
                        trial_table.append(row)
                        self.log_signal.emit(f"已完成 {len(labels)}/{total} trial")
                result = self._save_recording(
                    source,
                    labels,
                    imagery_starts,
                    trial_table,
                    completed=not self.stop_event.is_set() and len(labels) == total,
                    total_trials=total,
                )
                self.finished_signal.emit(result)
            except Exception as exc:
                self.error_signal.emit(str(exc))
            finally:
                if source is not None:
                    source.close()

    class VirtualArmWidget(QtWidgets.QWidget):
        state_changed = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(560, 420)
            self.current = self.pose_points("home")
            self.target = [p[:] for p in self.current]
            self.gripper = 1.0
            self.gripper_target = 1.0
            self.object_home = QtCore.QPointF(0.25, 0.73)
            self.object_target = QtCore.QPointF(0.76, 0.73)
            self.object_pos = QtCore.QPointF(self.object_home)
            self.object_grabbed = False
            self.pending_effect = ""
            self.mode_text = "Ready"
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.animate)
            self.timer.start(16)

        def pose_points(self, name: str) -> List[List[float]]:
            poses = {
                "home": [[0.50, 0.75], [0.50, 0.50], [0.62, 0.37], [0.70, 0.42]],
            }
            return [p[:] for p in poses[name]]

        def execute_payload(self, payload: dict) -> None:
            self.execute_command(payload.get("cmd", ""), payload.get("intent", ""))

        def execute_command(self, command: str, intent: str = "") -> None:
            if command == "move_up" or intent == "up":
                self.shift_target(0.0, -0.08)
                self.pending_effect = ""
                self.mode_text = "Move up"
            elif command == "move_down" or intent == "down":
                self.shift_target(0.0, 0.08)
                self.pending_effect = ""
                self.mode_text = "Move down"
            elif command == "move_left" or intent == "left":
                self.shift_target(-0.08, 0.0)
                self.pending_effect = ""
                self.mode_text = "Move left"
            elif command == "move_right" or intent == "right":
                self.shift_target(0.08, 0.0)
                self.pending_effect = ""
                self.mode_text = "Move right"
            elif command == "move_forward" or intent == "forward":
                self.shift_target(0.04, -0.035)
                self.pending_effect = ""
                self.mode_text = "Move forward"
            elif command == "move_backward" or intent == "backward":
                self.shift_target(-0.04, 0.035)
                self.pending_effect = ""
                self.mode_text = "Move backward"
            elif command == "close_gripper" or intent == "grab":
                self.gripper_target = 0.0
                self.pending_effect = "grab"
                self.mode_text = "Grab"
            elif command == "open_gripper" or intent == "release":
                self.gripper_target = 1.0
                self.pending_effect = "release"
                self.mode_text = "Release"
            elif command == "home":
                self.target = self.pose_points("home")
                self.gripper_target = 1.0
                self.pending_effect = ""
                self.mode_text = "Home"
            self.state_changed.emit(self.mode_text)

        def shift_target(self, dx: float, dy: float) -> None:
            shifted = [p[:] for p in self.target]
            for point in shifted[1:]:
                point[0] = min(0.88, max(0.12, point[0] + dx))
                point[1] = min(0.76, max(0.18, point[1] + dy))
            self.target = shifted

        def reset_scene(self) -> None:
            self.current = self.pose_points("home")
            self.target = self.pose_points("home")
            self.gripper = 1.0
            self.gripper_target = 1.0
            self.object_grabbed = False
            self.object_pos = QtCore.QPointF(self.object_home)
            self.pending_effect = ""
            self.mode_text = "Ready"
            self.state_changed.emit(self.mode_text)
            self.update()

        def animate(self) -> None:
            done = True
            for i, point in enumerate(self.current):
                for j in (0, 1):
                    delta = self.target[i][j] - point[j]
                    point[j] += delta * 0.16
                    done = done and abs(delta) < 0.003
            grip_delta = self.gripper_target - self.gripper
            self.gripper += grip_delta * 0.18
            done = done and abs(grip_delta) < 0.01
            if self.object_grabbed:
                grip = self.to_screen(self.current[-1])
                self.object_pos = QtCore.QPointF(grip.x(), grip.y() + 18)
            if done and self.pending_effect == "grab":
                self.object_grabbed = True
                self.pending_effect = ""
            elif done and self.pending_effect == "release":
                self.object_grabbed = False
                grip = self.to_screen(self.current[-1])
                self.object_pos = QtCore.QPointF(grip.x(), grip.y() + 18)
                self.pending_effect = ""
            self.update()

        def to_screen(self, normalized: Sequence[float]) -> QtCore.QPointF:
            return QtCore.QPointF(normalized[0] * self.width(), normalized[1] * self.height())

        def to_screen_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
            return QtCore.QPointF(point.x() * self.width(), point.y() * self.height())

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            painter.fillRect(rect, QtGui.QColor("#101820"))

            floor_y = int(self.height() * 0.80)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#1c2a35"))
            painter.drawRect(0, floor_y, self.width(), self.height() - floor_y)

            self.draw_grid(painter, floor_y)
            self.draw_target_zone(painter)
            self.draw_object(painter)
            self.draw_arm(painter)
            self.draw_status(painter)

        def draw_grid(self, painter, floor_y: int) -> None:
            pen = QtGui.QPen(QtGui.QColor(56, 76, 92, 110), 1)
            painter.setPen(pen)
            for x in range(-self.width(), self.width() * 2, 46):
                painter.drawLine(x, floor_y, x + int(self.width() * 0.35), self.height())
            for y in range(floor_y, self.height(), 28):
                painter.drawLine(0, y, self.width(), y)

        def draw_target_zone(self, painter) -> None:
            pos = self.to_screen_point(self.object_target)
            zone = QtCore.QRectF(pos.x() - 46, pos.y() + 10, 92, 28)
            painter.setPen(QtGui.QPen(QtGui.QColor("#76d275"), 2))
            painter.setBrush(QtGui.QColor(118, 210, 117, 34))
            painter.drawRoundedRect(zone, 6, 6)
            painter.setPen(QtGui.QColor("#9edb9d"))
            painter.drawText(zone.adjusted(0, 32, 0, 54), QtCore.Qt.AlignmentFlag.AlignCenter, "目标区")

        def draw_object(self, painter) -> None:
            if not self.object_grabbed and isinstance(self.object_pos, QtCore.QPointF) and self.object_pos.x() <= 1.0:
                pos = self.to_screen_point(self.object_home)
            else:
                pos = self.object_pos
            body = QtCore.QRectF(pos.x() - 18, pos.y() - 34, 36, 44)
            painter.setPen(QtGui.QPen(QtGui.QColor("#d8eefc"), 2))
            painter.setBrush(QtGui.QColor("#66c0f4"))
            painter.drawRoundedRect(body, 6, 6)
            cap = QtCore.QRectF(pos.x() - 10, pos.y() - 44, 20, 10)
            painter.setBrush(QtGui.QColor("#c8e6f5"))
            painter.drawRoundedRect(cap, 4, 4)

        def draw_arm(self, painter) -> None:
            points = [self.to_screen(p) for p in self.current]
            base = points[0]
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#263849"))
            painter.drawEllipse(QtCore.QPointF(base.x(), base.y() + 8), 68, 22)
            painter.setBrush(QtGui.QColor("#44596c"))
            painter.drawRoundedRect(QtCore.QRectF(base.x() - 42, base.y() - 16, 84, 32), 8, 8)

            shadow_pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 80), 24, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
            arm_pen = QtGui.QPen(QtGui.QColor("#d7dde4"), 18, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
            edge_pen = QtGui.QPen(QtGui.QColor("#8fa6b9"), 2)
            for a, b in zip(points, points[1:]):
                painter.setPen(shadow_pen)
                painter.drawLine(a + QtCore.QPointF(5, 7), b + QtCore.QPointF(5, 7))
                painter.setPen(arm_pen)
                painter.drawLine(a, b)
                painter.setPen(edge_pen)
                painter.drawLine(a, b)

            for p in points[:-1]:
                painter.setPen(QtGui.QPen(QtGui.QColor("#e9f1f8"), 2))
                painter.setBrush(QtGui.QColor("#526a7e"))
                painter.drawEllipse(p, 16, 16)

            grip = points[-1]
            open_span = 18 + 18 * self.gripper
            painter.setPen(QtGui.QPen(QtGui.QColor("#f1d47b"), 6, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
            painter.drawLine(grip, QtCore.QPointF(grip.x() - open_span, grip.y() + 23))
            painter.drawLine(grip, QtCore.QPointF(grip.x() + open_span, grip.y() + 23))

        def draw_status(self, painter) -> None:
            painter.setPen(QtGui.QColor("#eef3f8"))
            font = painter.font()
            font.setPointSize(13)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(18, 30, f"Virtual arm: {self.mode_text}")

    class BulletArmWidget(QtWidgets.QWidget):
        state_changed = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(720, 480)
            self.setAutoFillBackground(False)
            self.client_id = -1
            self.robot_id = -1
            self.object_id = -1
            self.grab_constraint = None
            self.arm_joint_indices: List[int] = []
            self.finger_joint_indices: List[int] = []
            self.render_link_indices: List[int] = []
            self.simple_joint_bodies: List[int] = []
            self.simple_segment_bodies: List[int] = []
            self.end_effector_link = 11
            self.home_target = [0.46, 0.0, 0.36]
            self.ee_target = self.home_target[:]
            self.object_home = [0.62, -0.22, 0.065]
            self.target_zone = [0.62, 0.24, 0.018]
            self.ee_orientation = bullet.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2.0])
            self.finger_target = 0.04
            self.pending_grab = False
            self.mode_text = "Ready"
            self.render_image = QtGui.QImage()
            self.frame_counter = 0
            self.motion_velocity = [0.0, 0.0, 0.0]
            self.motion_until = 0.0
            self.motion_duration_seconds = 0.55
            self.motion_speed = 0.22
            self.last_tick_time = time.perf_counter()
            self.last_render_time = 0.0
            self.render_interval_seconds = 1.0 / 10.0
            self.render_scale = 0.35
            self.max_render_size = (420, 280)
            self.renderer = bullet.ER_BULLET_HARDWARE_OPENGL
            self.physics_substeps = 2
            self._init_physics()
            self._render_scene()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(16)

        def _init_physics(self) -> None:
            if self.client_id >= 0:
                bullet.disconnect(self.client_id)
            self.client_id = bullet.connect(bullet.DIRECT)
            bullet.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            bullet.resetSimulation(physicsClientId=self.client_id)
            bullet.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
            bullet.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)
            bullet.setPhysicsEngineParameter(
                numSolverIterations=80,
                fixedTimeStep=1.0 / 240.0,
                physicsClientId=self.client_id,
            )
            self._create_scene()
            self._load_robot()
            self._set_home_pose()
            self._create_simple_arm_visuals()
            for _ in range(120):
                self._drive_arm()
                bullet.stepSimulation(physicsClientId=self.client_id)

        def _create_scene(self) -> None:
            bullet.loadURDF("plane.urdf", physicsClientId=self.client_id)
            floor_visual = bullet.createVisualShape(
                bullet.GEOM_BOX,
                halfExtents=[1.25, 0.9, 0.015],
                rgbaColor=[0.08, 0.13, 0.16, 1.0],
                physicsClientId=self.client_id,
            )
            floor_collision = bullet.createCollisionShape(
                bullet.GEOM_BOX,
                halfExtents=[1.25, 0.9, 0.015],
                physicsClientId=self.client_id,
            )
            bullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=floor_collision,
                baseVisualShapeIndex=floor_visual,
                basePosition=[0.45, 0.0, -0.015],
                physicsClientId=self.client_id,
            )
            pad_visual = bullet.createVisualShape(
                bullet.GEOM_BOX,
                halfExtents=[0.13, 0.09, 0.012],
                rgbaColor=[0.18, 0.85, 0.33, 0.85],
                physicsClientId=self.client_id,
            )
            pad_collision = bullet.createCollisionShape(
                bullet.GEOM_BOX,
                halfExtents=[0.13, 0.09, 0.012],
                physicsClientId=self.client_id,
            )
            bullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=pad_collision,
                baseVisualShapeIndex=pad_visual,
                basePosition=self.target_zone,
                physicsClientId=self.client_id,
            )
            object_visual = bullet.createVisualShape(
                bullet.GEOM_BOX,
                halfExtents=[0.04, 0.04, 0.055],
                rgbaColor=[0.18, 0.75, 0.95, 1.0],
                physicsClientId=self.client_id,
            )
            object_collision = bullet.createCollisionShape(
                bullet.GEOM_BOX,
                halfExtents=[0.04, 0.04, 0.055],
                physicsClientId=self.client_id,
            )
            self.object_id = bullet.createMultiBody(
                baseMass=0.08,
                baseCollisionShapeIndex=object_collision,
                baseVisualShapeIndex=object_visual,
                basePosition=self.object_home,
                physicsClientId=self.client_id,
            )
            bullet.changeDynamics(
                self.object_id,
                -1,
                lateralFriction=0.9,
                rollingFriction=0.02,
                spinningFriction=0.02,
                restitution=0.03,
                physicsClientId=self.client_id,
            )

        def _load_robot(self) -> None:
            self.robot_id = bullet.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=[0.0, 0.0, 0.0],
                baseOrientation=bullet.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                useFixedBase=True,
                flags=bullet.URDF_USE_SELF_COLLISION,
                physicsClientId=self.client_id,
            )
            self.arm_joint_indices = []
            self.finger_joint_indices = []
            num_joints = bullet.getNumJoints(self.robot_id, physicsClientId=self.client_id)
            for idx in range(num_joints):
                info = bullet.getJointInfo(self.robot_id, idx, physicsClientId=self.client_id)
                joint_name = info[1].decode("utf-8")
                link_name = info[12].decode("utf-8")
                if joint_name.startswith("panda_joint") and info[2] == bullet.JOINT_REVOLUTE:
                    self.arm_joint_indices.append(idx)
                elif "finger_joint" in joint_name:
                    self.finger_joint_indices.append(idx)
                if link_name == "panda_grasptarget":
                    self.end_effector_link = idx
            if not self.arm_joint_indices:
                raise RuntimeError("No Panda arm joints were found in the pybullet URDF.")
            self.render_link_indices = [0, 1, 2, 3, 4, 5, 6, 8, self.end_effector_link]
            for link_idx in range(-1, num_joints):
                try:
                    bullet.changeVisualShape(
                        self.robot_id,
                        link_idx,
                        rgbaColor=[0.0, 0.0, 0.0, 0.0],
                        physicsClientId=self.client_id,
                    )
                except Exception:
                    pass

        def _render_points(self) -> List[np.ndarray]:
            points: List[np.ndarray] = []
            for link_idx in self.render_link_indices:
                state = bullet.getLinkState(
                    self.robot_id,
                    link_idx,
                    computeForwardKinematics=True,
                    physicsClientId=self.client_id,
                )
                points.append(np.array(state[4], dtype=float))
            return points

        def _quat_from_z_axis(self, vector: np.ndarray) -> List[float]:
            length = float(np.linalg.norm(vector))
            if length <= 1e-6:
                return [0.0, 0.0, 0.0, 1.0]
            direction = vector / length
            z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
            if dot > 0.9999:
                return [0.0, 0.0, 0.0, 1.0]
            if dot < -0.9999:
                return [1.0, 0.0, 0.0, 0.0]
            axis = np.cross(z_axis, direction)
            axis /= max(1e-6, float(np.linalg.norm(axis)))
            angle = math.acos(dot)
            half = angle * 0.5
            scale = math.sin(half)
            return [float(axis[0] * scale), float(axis[1] * scale), float(axis[2] * scale), float(math.cos(half))]

        def _create_simple_arm_visuals(self) -> None:
            points = self._render_points()
            joint_visual = bullet.createVisualShape(
                bullet.GEOM_SPHERE,
                radius=0.035,
                rgbaColor=[0.78, 0.86, 0.92, 1.0],
                physicsClientId=self.client_id,
            )
            self.simple_joint_bodies = [
                bullet.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=joint_visual,
                    basePosition=point.tolist(),
                    physicsClientId=self.client_id,
                )
                for point in points
            ]
            self.simple_segment_bodies = []
            for start, end in zip(points, points[1:]):
                length = max(0.04, float(np.linalg.norm(end - start)))
                segment_visual = bullet.createVisualShape(
                    bullet.GEOM_CAPSULE,
                    radius=0.024,
                    length=length,
                    rgbaColor=[0.55, 0.70, 0.82, 1.0],
                    physicsClientId=self.client_id,
                )
                midpoint = ((start + end) * 0.5).tolist()
                orientation = self._quat_from_z_axis(end - start)
                body = bullet.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=segment_visual,
                    basePosition=midpoint,
                    baseOrientation=orientation,
                    physicsClientId=self.client_id,
                )
                self.simple_segment_bodies.append(body)

        def _update_simple_arm_visuals(self) -> None:
            if not self.simple_joint_bodies:
                return
            points = self._render_points()
            for body, point in zip(self.simple_joint_bodies, points):
                bullet.resetBasePositionAndOrientation(
                    body,
                    point.tolist(),
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client_id,
                )
            for body, start, end in zip(self.simple_segment_bodies, points, points[1:]):
                midpoint = ((start + end) * 0.5).tolist()
                orientation = self._quat_from_z_axis(end - start)
                bullet.resetBasePositionAndOrientation(
                    body,
                    midpoint,
                    orientation,
                    physicsClientId=self.client_id,
                )

        def _set_home_pose(self) -> None:
            self.ee_target = self.home_target[:]
            self.finger_target = 0.04
            self.motion_velocity = [0.0, 0.0, 0.0]
            self.motion_until = 0.0
            rest_pose = [0.0, -0.55, 0.0, -2.35, 0.0, 1.8, 0.78]
            for joint_idx, joint_value in zip(self.arm_joint_indices, rest_pose):
                bullet.resetJointState(self.robot_id, joint_idx, joint_value, physicsClientId=self.client_id)
            for joint_idx in self.finger_joint_indices:
                bullet.resetJointState(self.robot_id, joint_idx, self.finger_target, physicsClientId=self.client_id)

        def execute_payload(self, payload: dict) -> None:
            duration = payload.get("duration_seconds", payload.get("duration"))
            self.execute_command(payload.get("cmd", ""), payload.get("intent", ""), duration)

        def execute_command(self, command: str, intent: str = "", duration_seconds: Optional[float] = None) -> None:
            if command == "move_up" or intent == "up":
                self.start_motion(0.0, 0.0, 1.0, "Move up", duration_seconds)
            elif command == "move_down" or intent == "down":
                self.start_motion(0.0, 0.0, -1.0, "Move down", duration_seconds)
            elif command == "move_left" or intent == "left":
                self.start_motion(0.0, 1.0, 0.0, "Move left", duration_seconds)
            elif command == "move_right" or intent == "right":
                self.start_motion(0.0, -1.0, 0.0, "Move right", duration_seconds)
            elif command == "move_forward" or intent == "forward":
                self.start_motion(1.0, 0.0, 0.0, "Move forward", duration_seconds)
            elif command == "move_backward" or intent == "backward":
                self.start_motion(-1.0, 0.0, 0.0, "Move backward", duration_seconds)
            elif command == "close_gripper" or intent == "grab":
                self.stop_motion()
                self.finger_target = 0.0
                self.pending_grab = True
                self.mode_text = "Grab"
            elif command == "open_gripper" or intent == "release":
                self.stop_motion()
                self.finger_target = 0.04
                self.pending_grab = False
                self._release_object()
                self.mode_text = "Release"
            elif command == "home" or intent == "home":
                self.reset_scene()
                return
            self.state_changed.emit(self.mode_text)

        def start_motion(self, dx: float, dy: float, dz: float, mode_text: str, duration_seconds: Optional[float] = None) -> None:
            duration = self.motion_duration_seconds if duration_seconds is None else float(duration_seconds)
            duration = min(2.0, max(0.05, duration))
            self.motion_velocity = [
                float(dx) * self.motion_speed,
                float(dy) * self.motion_speed,
                float(dz) * self.motion_speed,
            ]
            self.motion_until = time.perf_counter() + duration
            self.mode_text = mode_text
            self.last_render_time = 0.0

        def stop_motion(self) -> None:
            self.motion_velocity = [0.0, 0.0, 0.0]
            self.motion_until = 0.0

        def shift_target(self, dx: float, dy: float, dz: float) -> None:
            self.ee_target[0] = min(0.78, max(0.22, self.ee_target[0] + dx))
            self.ee_target[1] = min(0.42, max(-0.42, self.ee_target[1] + dy))
            self.ee_target[2] = min(0.72, max(0.16, self.ee_target[2] + dz))

        def reset_scene(self) -> None:
            self._release_object()
            self.pending_grab = False
            self.mode_text = "Ready"
            self._set_home_pose()
            bullet.resetBasePositionAndOrientation(
                self.object_id,
                self.object_home,
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client_id,
            )
            bullet.resetBaseVelocity(
                self.object_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.client_id,
            )
            self.state_changed.emit(self.mode_text)
            self._render_scene()
            self.update()

        def tick(self) -> None:
            if self.client_id < 0:
                return
            now = time.perf_counter()
            dt = min(0.05, max(0.0, now - self.last_tick_time))
            self.last_tick_time = now
            self._apply_motion(dt, now)
            for _ in range(self.physics_substeps):
                self._drive_arm()
                self._maybe_grab_object()
                bullet.stepSimulation(physicsClientId=self.client_id)
            self.frame_counter += 1
            if now - self.last_render_time >= self.render_interval_seconds:
                self._render_scene()
                self.last_render_time = now
                self.update()

        def _apply_motion(self, dt: float, now: float) -> None:
            if now > self.motion_until:
                self.stop_motion()
                return
            vx, vy, vz = self.motion_velocity
            if vx == 0.0 and vy == 0.0 and vz == 0.0:
                return
            self.shift_target(vx * dt, vy * dt, vz * dt)

        def _drive_arm(self) -> None:
            joint_targets = bullet.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                self.ee_target,
                targetOrientation=self.ee_orientation,
                maxNumIterations=80,
                residualThreshold=1e-4,
                physicsClientId=self.client_id,
            )
            for offset, joint_idx in enumerate(self.arm_joint_indices):
                target_pos = joint_targets[joint_idx] if joint_idx < len(joint_targets) else joint_targets[offset]
                bullet.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    bullet.POSITION_CONTROL,
                    targetPosition=float(target_pos),
                    force=90.0,
                    positionGain=0.08,
                    velocityGain=1.0,
                    maxVelocity=1.6,
                    physicsClientId=self.client_id,
                )
            for joint_idx in self.finger_joint_indices:
                bullet.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    bullet.POSITION_CONTROL,
                    targetPosition=self.finger_target,
                    force=35.0,
                    positionGain=0.35,
                    velocityGain=1.0,
                    maxVelocity=0.25,
                    physicsClientId=self.client_id,
                )

        def _maybe_grab_object(self) -> None:
            if not self.pending_grab or self.grab_constraint is not None:
                return
            hand_state = bullet.getLinkState(
                self.robot_id,
                self.end_effector_link,
                computeForwardKinematics=True,
                physicsClientId=self.client_id,
            )
            hand_pos = np.array(hand_state[4], dtype=float)
            hand_orn = hand_state[5]
            object_pos, object_orn = bullet.getBasePositionAndOrientation(self.object_id, physicsClientId=self.client_id)
            if np.linalg.norm(hand_pos - np.array(object_pos, dtype=float)) > 0.16:
                return
            inv_pos, inv_orn = bullet.invertTransform(hand_pos.tolist(), hand_orn)
            rel_pos, rel_orn = bullet.multiplyTransforms(inv_pos, inv_orn, object_pos, object_orn)
            self.grab_constraint = bullet.createConstraint(
                self.robot_id,
                self.end_effector_link,
                self.object_id,
                -1,
                bullet.JOINT_FIXED,
                [0.0, 0.0, 0.0],
                rel_pos,
                [0.0, 0.0, 0.0],
                rel_orn,
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client_id,
            )
            bullet.changeConstraint(self.grab_constraint, maxForce=80.0, physicsClientId=self.client_id)
            self.pending_grab = False
            self.mode_text = "Grabbed"
            self.state_changed.emit(self.mode_text)

        def _release_object(self) -> None:
            if self.grab_constraint is not None:
                bullet.removeConstraint(self.grab_constraint, physicsClientId=self.client_id)
                self.grab_constraint = None

        def _render_scene(self) -> None:
            if self.client_id < 0 or self.width() <= 0 or self.height() <= 0:
                return
            self._update_simple_arm_visuals()
            widget_aspect = max(0.5, min(2.4, float(self.width()) / float(max(1, self.height()))))
            render_w = int(max(300, min(self.max_render_size[0], self.width() * self.render_scale)))
            render_h = int(max(200, round(render_w / widget_aspect)))
            if render_h > self.max_render_size[1]:
                render_h = int(self.max_render_size[1])
                render_w = int(max(300, round(render_h * widget_aspect)))
            view = bullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.46, 0.0, 0.28],
                distance=1.42,
                yaw=42.0,
                pitch=-31.0,
                roll=0.0,
                upAxisIndex=2,
            )
            projection = bullet.computeProjectionMatrixFOV(
                fov=48.0,
                aspect=float(render_w) / float(render_h),
                nearVal=0.02,
                farVal=5.0,
            )
            try:
                _, _, rgba, _, _ = bullet.getCameraImage(
                    render_w,
                    render_h,
                    view,
                    projection,
                    renderer=self.renderer,
                    shadow=0,
                    physicsClientId=self.client_id,
                )
            except Exception:
                self.renderer = bullet.ER_TINY_RENDERER
                _, _, rgba, _, _ = bullet.getCameraImage(
                    render_w,
                    render_h,
                    view,
                    projection,
                    renderer=self.renderer,
                    shadow=0,
                    physicsClientId=self.client_id,
                )
            frame = np.asarray(rgba, dtype=np.uint8).reshape(render_h, render_w, 4)
            self.render_image = QtGui.QImage(
                frame.data,
                render_w,
                render_h,
                4 * render_w,
                QtGui.QImage.Format.Format_RGBA8888,
            ).copy()

        def paintEvent(self, event) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QtGui.QColor("#101820"))
            if not self.render_image.isNull():
                pixmap = QtGui.QPixmap.fromImage(self.render_image)
                scaled = pixmap.scaled(
                    self.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    QtCore.Qt.TransformationMode.FastTransformation,
                )
                x = int((self.width() - scaled.width()) / 2)
                y = int((self.height() - scaled.height()) / 2)
                painter.drawPixmap(x, y, scaled)
            else:
                painter.setPen(QtGui.QColor("#eef3f8"))
                painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Starting Bullet 3D scene...")
            self._draw_status(painter)

        def _draw_status(self, painter) -> None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(10, 16, 22, 180))
            painter.drawRoundedRect(QtCore.QRectF(14, 14, 430, 78), 8, 8)
            painter.setPen(QtGui.QColor("#eef3f8"))
            font = painter.font()
            font.setPointSize(13)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(30, 42, f"Bullet 3D arm: {self.mode_text}")
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)
            attach = "attached" if self.grab_constraint is not None else "free"
            motion_left = max(0.0, self.motion_until - time.perf_counter())
            speed = math.sqrt(sum(value * value for value in self.motion_velocity))
            motion = f"motion={motion_left:.2f}s @ {speed:.2f}m/s" if motion_left > 0.0 else "motion=idle"
            painter.setPen(QtGui.QColor("#b9c7d5"))
            painter.drawText(
                30,
                70,
                f"EE target xyz=({self.ee_target[0]:.2f}, {self.ee_target[1]:.2f}, {self.ee_target[2]:.2f})  object={attach}  {motion}",
            )

        def closeEvent(self, event) -> None:
            self._disconnect_physics()
            super().closeEvent(event)

        def _disconnect_physics(self) -> None:
            if self.client_id >= 0:
                try:
                    bullet.disconnect(self.client_id)
                except Exception:
                    pass
                self.client_id = -1

        def __del__(self):
            self._disconnect_physics()

    class BulletOpenGLArmWidget(QtWidgets.QWidget):
        state_changed = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumSize(720, 480)
            self.client_id = -1
            self.robot_id = -1
            self.object_id = -1
            self.object_ids: List[int] = []
            self.grabbed_object_id: Optional[int] = None
            self.grab_constraint = None
            self.arm_joint_indices: List[int] = []
            self.finger_joint_indices: List[int] = []
            self.render_link_indices: List[int] = []
            self.end_effector_link = 11
            self.home_target = [0.46, 0.0, 0.36]
            self.ee_target = self.home_target[:]
            self.object_specs = [
                {"home": [0.62, -0.22, 0.065], "size": (0.08, 0.08, 0.11), "color": (0.10, 0.67, 0.92, 0.96), "mass": 0.08},
                {"home": [0.42, -0.08, 0.055], "size": (0.07, 0.07, 0.09), "color": (0.93, 0.38, 0.24, 0.96), "mass": 0.07},
                {"home": [0.72, 0.04, 0.075], "size": (0.10, 0.07, 0.13), "color": (0.93, 0.72, 0.18, 0.96), "mass": 0.09},
                {"home": [0.30, 0.12, 0.045], "size": (0.06, 0.10, 0.08), "color": (0.58, 0.42, 0.90, 0.96), "mass": 0.06},
            ]
            self.object_home = list(self.object_specs[0]["home"])
            self.object_size = tuple(self.object_specs[0]["size"])
            self.target_zones = [
                [0.62, 0.28, 0.018],
                [0.35, 0.30, 0.018],
            ]
            self.target_zone = self.target_zones[0]
            self.target_size = (0.24, 0.17, 0.025)
            self.target_sizes = [self.target_size, (0.19, 0.16, 0.025)]
            self.ee_orientation = bullet.getQuaternionFromEuler([math.pi, 0.0, math.pi / 2.0])
            self.finger_target = 0.04
            self.pending_grab = False
            self.mode_text = "Ready"
            self.motion_velocity = [0.0, 0.0, 0.0]
            self.motion_until = 0.0
            self.motion_duration_seconds = 0.55
            self.motion_speed = 0.22
            self.last_tick_time = time.perf_counter()
            self.physics_substeps = 2

            outer_layout = QtWidgets.QVBoxLayout(self)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            page_scroll = QtWidgets.QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            page_content = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page_content)
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(12)
            page_scroll.setWidget(page_content)
            outer_layout.addWidget(page_scroll)
            header = QtWidgets.QHBoxLayout()
            title_new = QtWidgets.QLabel("VR 演示")
            title_new.setObjectName("vrTitle")
            header.addWidget(title_new)
            header.addStretch(1)
            self.state_badge = QtWidgets.QLabel("Ready")
            self.state_badge.setObjectName("vrBadge")
            header.addWidget(self.state_badge)
            layout.addLayout(header)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self.status_label = QtWidgets.QLabel(self)
            self.status_label.setObjectName("vrStatus")
            self.status_label.setMinimumHeight(28)
            self.status_label.setVisible(False)
            self.view = gl.GLViewWidget()
            self.view.setBackgroundColor(QtGui.QColor("#0a1118"))
            self.view.setCameraPosition(distance=2.15, elevation=22, azimuth=42)
            self.view.opts["center"] = QtGui.QVector3D(0.42, 0.0, 0.24)
            layout.addWidget(self.view, 1)

            self.arm_line = gl.GLLinePlotItem(color=(0.52, 0.80, 0.95, 1.0), width=8, antialias=True)
            self.joint_points = gl.GLScatterPlotItem(color=(0.90, 0.97, 1.0, 1.0), size=12, pxMode=True)
            self.gripper_crossbar = gl.GLLinePlotItem(color=(1.0, 0.86, 0.22, 1.0), width=6, antialias=True)
            self.left_gripper = gl.GLLinePlotItem(color=(1.0, 0.80, 0.18, 1.0), width=6, antialias=True)
            self.right_gripper = gl.GLLinePlotItem(color=(1.0, 0.80, 0.18, 1.0), width=6, antialias=True)
            self.object_boxes = [
                self._make_box_item(spec["size"], spec["color"], "opaque")
                for spec in self.object_specs
            ]
            self.object_box = self.object_boxes[0]
            self.target_boxes = [
                self._make_box_item(size, color, "translucent")
                for size, color in zip(
                    self.target_sizes,
                    [(0.18, 0.88, 0.36, 0.42), (0.20, 0.56, 0.98, 0.36)],
                )
            ]
            self.target_box = self.target_boxes[0]
            self.grid = gl.GLGridItem(size=QtGui.QVector3D(1.45, 1.05, 1.0), color=(56, 96, 118, 105))
            self.grid.translate(0.45, 0.0, 0.0)
            for item in [self.grid, *self.target_boxes, *self.object_boxes, self.arm_line, self.joint_points, self.gripper_crossbar, self.left_gripper, self.right_gripper]:
                self.view.addItem(item)

            self._init_physics()
            for box, zone, size in zip(self.target_boxes, self.target_zones, self.target_sizes):
                self._set_box_transform(box, zone, size)
            self._update_gl_items()
            self._refresh_status()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(16)

        def _make_box_item(self, size: Sequence[float], color: Tuple[float, float, float, float], gl_options: str):
            sx, sy, sz = [float(value) for value in size]
            vertices = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [sx, 0.0, 0.0],
                    [sx, sy, 0.0],
                    [0.0, sy, 0.0],
                    [0.0, 0.0, sz],
                    [sx, 0.0, sz],
                    [sx, sy, sz],
                    [0.0, sy, sz],
                ],
                dtype=np.float32,
            )
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [4, 7, 6],
                    [4, 6, 5],
                    [0, 4, 5],
                    [0, 5, 1],
                    [1, 5, 6],
                    [1, 6, 2],
                    [2, 6, 7],
                    [2, 7, 3],
                    [3, 7, 4],
                    [3, 4, 0],
                ],
                dtype=np.int32,
            )
            mesh = gl.MeshData(vertexes=vertices, faces=faces)
            return gl.GLMeshItem(meshdata=mesh, color=color, smooth=False, shader="shaded", glOptions=gl_options)

        def _init_physics(self) -> None:
            if self.client_id >= 0:
                bullet.disconnect(self.client_id)
            self.client_id = bullet.connect(bullet.DIRECT)
            bullet.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            bullet.resetSimulation(physicsClientId=self.client_id)
            bullet.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
            bullet.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)
            bullet.setPhysicsEngineParameter(numSolverIterations=60, fixedTimeStep=1.0 / 240.0, physicsClientId=self.client_id)
            bullet.loadURDF("plane.urdf", physicsClientId=self.client_id)
            for zone, size in zip(self.target_zones, self.target_sizes):
                pad_collision = bullet.createCollisionShape(
                    bullet.GEOM_BOX,
                    halfExtents=[float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0],
                    physicsClientId=self.client_id,
                )
                bullet.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=pad_collision,
                    baseVisualShapeIndex=-1,
                    basePosition=zone,
                    physicsClientId=self.client_id,
                )

            self.object_ids = []
            for spec in self.object_specs:
                size = spec["size"]
                object_visual = bullet.createVisualShape(
                    bullet.GEOM_BOX,
                    halfExtents=[float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0],
                    rgbaColor=list(spec["color"]),
                    physicsClientId=self.client_id,
                )
                object_collision = bullet.createCollisionShape(
                    bullet.GEOM_BOX,
                    halfExtents=[float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0],
                    physicsClientId=self.client_id,
                )
                object_id = bullet.createMultiBody(
                    baseMass=float(spec["mass"]),
                    baseCollisionShapeIndex=object_collision,
                    baseVisualShapeIndex=object_visual,
                    basePosition=spec["home"],
                    physicsClientId=self.client_id,
                )
                bullet.changeDynamics(
                    object_id,
                    -1,
                    lateralFriction=0.95,
                    rollingFriction=0.02,
                    spinningFriction=0.02,
                    restitution=0.03,
                    physicsClientId=self.client_id,
                )
                self.object_ids.append(object_id)
            self.object_id = self.object_ids[0] if self.object_ids else -1
            self.grabbed_object_id = None
            self._load_robot()
            self._set_home_pose()
            for _ in range(80):
                self._drive_arm()
                bullet.stepSimulation(physicsClientId=self.client_id)

        def _load_robot(self) -> None:
            self.robot_id = bullet.loadURDF(
                "franka_panda/panda.urdf",
                basePosition=[0.0, 0.0, 0.0],
                useFixedBase=True,
                flags=bullet.URDF_USE_SELF_COLLISION,
                physicsClientId=self.client_id,
            )
            self.arm_joint_indices = []
            self.finger_joint_indices = []
            num_joints = bullet.getNumJoints(self.robot_id, physicsClientId=self.client_id)
            for idx in range(num_joints):
                info = bullet.getJointInfo(self.robot_id, idx, physicsClientId=self.client_id)
                joint_name = info[1].decode("utf-8")
                link_name = info[12].decode("utf-8")
                if joint_name.startswith("panda_joint") and info[2] == bullet.JOINT_REVOLUTE:
                    self.arm_joint_indices.append(idx)
                elif "finger_joint" in joint_name:
                    self.finger_joint_indices.append(idx)
                if link_name == "panda_grasptarget":
                    self.end_effector_link = idx
            self.render_link_indices = [0, 1, 2, 3, 4, 5, 6, 8, self.end_effector_link]
            for link_idx in range(-1, num_joints):
                try:
                    bullet.changeVisualShape(self.robot_id, link_idx, rgbaColor=[0.0, 0.0, 0.0, 0.0], physicsClientId=self.client_id)
                except Exception:
                    pass

        def _set_home_pose(self) -> None:
            self.ee_target = self.home_target[:]
            self.finger_target = 0.04
            self.stop_motion()
            rest_pose = [0.0, -0.55, 0.0, -2.35, 0.0, 1.8, 0.78]
            for joint_idx, joint_value in zip(self.arm_joint_indices, rest_pose):
                bullet.resetJointState(self.robot_id, joint_idx, joint_value, physicsClientId=self.client_id)
            for joint_idx in self.finger_joint_indices:
                bullet.resetJointState(self.robot_id, joint_idx, self.finger_target, physicsClientId=self.client_id)

        def _render_points(self) -> np.ndarray:
            points = []
            for link_idx in self.render_link_indices:
                state = bullet.getLinkState(
                    self.robot_id,
                    link_idx,
                    computeForwardKinematics=True,
                    physicsClientId=self.client_id,
                )
                points.append(state[4])
            return np.asarray(points, dtype=np.float32)

        def _set_box_transform(self, item, position: Sequence[float], size: Sequence[float]) -> None:
            item.resetTransform()
            item.translate(
                float(position[0]) - float(size[0]) / 2.0,
                float(position[1]) - float(size[1]) / 2.0,
                float(position[2]) - float(size[2]) / 2.0,
            )

        def execute_payload(self, payload: dict) -> None:
            duration = payload.get("duration_seconds", payload.get("duration"))
            self.execute_command(payload.get("cmd", ""), payload.get("intent", ""), duration)

        def execute_command(self, command: str, intent: str = "", duration_seconds: Optional[float] = None) -> None:
            if command == "move_up" or intent == "up":
                self.start_motion(0.0, 0.0, 1.0, "Move up", duration_seconds)
            elif command == "move_down" or intent == "down":
                self.start_motion(0.0, 0.0, -1.0, "Move down", duration_seconds)
            elif command == "move_left" or intent == "left":
                self.start_motion(0.0, 1.0, 0.0, "Move left", duration_seconds)
            elif command == "move_right" or intent == "right":
                self.start_motion(0.0, -1.0, 0.0, "Move right", duration_seconds)
            elif command == "move_forward" or intent == "forward":
                self.start_motion(1.0, 0.0, 0.0, "Move forward", duration_seconds)
            elif command == "move_backward" or intent == "backward":
                self.start_motion(-1.0, 0.0, 0.0, "Move backward", duration_seconds)
            elif command == "close_gripper" or intent == "grab":
                self.stop_motion()
                self.finger_target = 0.0
                self.pending_grab = True
                self.mode_text = "Grab"
            elif command == "open_gripper" or intent == "release":
                self.stop_motion()
                self.finger_target = 0.04
                self.pending_grab = False
                self._release_object()
                self.mode_text = "Release"
            elif command == "home" or intent == "home":
                self.reset_scene()
                return
            self._refresh_status()
            self.state_changed.emit(self.mode_text)

        def start_motion(self, dx: float, dy: float, dz: float, mode_text: str, duration_seconds: Optional[float] = None) -> None:
            duration = self.motion_duration_seconds if duration_seconds is None else float(duration_seconds)
            duration = min(2.0, max(0.05, duration))
            self.motion_velocity = [
                float(dx) * self.motion_speed,
                float(dy) * self.motion_speed,
                float(dz) * self.motion_speed,
            ]
            self.motion_until = time.perf_counter() + duration
            self.mode_text = mode_text

        def stop_motion(self) -> None:
            self.motion_velocity = [0.0, 0.0, 0.0]
            self.motion_until = 0.0

        def shift_target(self, dx: float, dy: float, dz: float) -> None:
            self.ee_target[0] = min(0.78, max(0.22, self.ee_target[0] + dx))
            self.ee_target[1] = min(0.42, max(-0.42, self.ee_target[1] + dy))
            self.ee_target[2] = min(0.72, max(0.16, self.ee_target[2] + dz))

        def reset_scene(self) -> None:
            self._release_object()
            self.pending_grab = False
            self.mode_text = "Ready"
            self._set_home_pose()
            for object_id, spec in zip(self.object_ids, self.object_specs):
                bullet.resetBasePositionAndOrientation(
                    object_id,
                    spec["home"],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.client_id,
                )
                bullet.resetBaseVelocity(
                    object_id,
                    linearVelocity=[0.0, 0.0, 0.0],
                    angularVelocity=[0.0, 0.0, 0.0],
                    physicsClientId=self.client_id,
                )
            self._update_gl_items()
            self._refresh_status()
            self.state_changed.emit(self.mode_text)
        def tick(self) -> None:
            if self.client_id < 0:
                return
            now = time.perf_counter()
            dt = min(0.05, max(0.0, now - self.last_tick_time))
            self.last_tick_time = now
            self._apply_motion(dt, now)
            for _ in range(self.physics_substeps):
                self._drive_arm()
                self._maybe_grab_object()
                bullet.stepSimulation(physicsClientId=self.client_id)
            self._update_gl_items()
            self._refresh_status()

        def _apply_motion(self, dt: float, now: float) -> None:
            if now > self.motion_until:
                self.stop_motion()
                return
            vx, vy, vz = self.motion_velocity
            if vx == 0.0 and vy == 0.0 and vz == 0.0:
                return
            self.shift_target(vx * dt, vy * dt, vz * dt)

        def _drive_arm(self) -> None:
            joint_targets = bullet.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                self.ee_target,
                targetOrientation=self.ee_orientation,
                maxNumIterations=60,
                residualThreshold=1e-4,
                physicsClientId=self.client_id,
            )
            for offset, joint_idx in enumerate(self.arm_joint_indices):
                target_pos = joint_targets[joint_idx] if joint_idx < len(joint_targets) else joint_targets[offset]
                bullet.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    bullet.POSITION_CONTROL,
                    targetPosition=float(target_pos),
                    force=90.0,
                    positionGain=0.08,
                    velocityGain=1.0,
                    maxVelocity=1.6,
                    physicsClientId=self.client_id,
                )
            for joint_idx in self.finger_joint_indices:
                bullet.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    bullet.POSITION_CONTROL,
                    targetPosition=self.finger_target,
                    force=35.0,
                    positionGain=0.35,
                    velocityGain=1.0,
                    maxVelocity=0.25,
                    physicsClientId=self.client_id,
                )

        def _gripper_geometry(self, root_position: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            root = np.asarray(root_position, dtype=float)
            side = np.array([0.0, 1.0, 0.0], dtype=float)
            down = np.array([0.0, 0.0, -1.0], dtype=float)
            open_amount = max(0.0, min(1.0, self.finger_target / 0.04))
            open_span = 0.025 + 0.055 * open_amount
            knuckle_span = open_span * 0.55
            jaw_depth = 0.12
            left_knuckle = root + side * knuckle_span
            right_knuckle = root - side * knuckle_span
            left_tip = root + side * open_span + down * jaw_depth
            right_tip = root - side * open_span + down * jaw_depth
            crossbar = np.asarray([left_knuckle, root, right_knuckle], dtype=np.float32)
            left = np.asarray([root, left_knuckle, left_tip], dtype=np.float32)
            right = np.asarray([root, right_knuckle, right_tip], dtype=np.float32)
            grasp_anchor = root + down * (jaw_depth * 0.82)
            return crossbar, left, right, grasp_anchor

        def _maybe_grab_object(self) -> None:
            if not self.pending_grab or self.grab_constraint is not None:
                return
            hand_state = bullet.getLinkState(self.robot_id, self.end_effector_link, computeForwardKinematics=True, physicsClientId=self.client_id)
            hand_pos = np.array(hand_state[4], dtype=float)
            hand_orn = hand_state[5]
            _crossbar, _left, _right, grasp_anchor = self._gripper_geometry(hand_pos)
            nearest_object_id = None
            nearest_object_orn = None
            nearest_score = float("inf")
            for object_id in self.object_ids:
                object_pos, object_orn = bullet.getBasePositionAndOrientation(object_id, physicsClientId=self.client_id)
                object_pos_arr = np.array(object_pos, dtype=float)
                xy_distance = float(np.linalg.norm(grasp_anchor[:2] - object_pos_arr[:2]))
                z_distance = abs(float(grasp_anchor[2] - object_pos_arr[2]))
                if xy_distance > 0.17 or z_distance > 0.15:
                    continue
                score = xy_distance + z_distance * 0.55
                if score < nearest_score:
                    nearest_object_id = object_id
                    nearest_object_orn = object_orn
                    nearest_score = score
            if nearest_object_id is None or nearest_object_orn is None:
                return
            object_pos = grasp_anchor.tolist()
            bullet.resetBasePositionAndOrientation(
                nearest_object_id,
                object_pos,
                nearest_object_orn,
                physicsClientId=self.client_id,
            )
            bullet.resetBaseVelocity(
                nearest_object_id,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=self.client_id,
            )
            inv_pos, inv_orn = bullet.invertTransform(hand_pos.tolist(), hand_orn)
            rel_pos, rel_orn = bullet.multiplyTransforms(inv_pos, inv_orn, object_pos, nearest_object_orn)
            self.grab_constraint = bullet.createConstraint(
                self.robot_id,
                self.end_effector_link,
                nearest_object_id,
                -1,
                bullet.JOINT_FIXED,
                [0.0, 0.0, 0.0],
                rel_pos,
                [0.0, 0.0, 0.0],
                rel_orn,
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client_id,
            )
            bullet.changeConstraint(self.grab_constraint, maxForce=80.0, physicsClientId=self.client_id)
            self.grabbed_object_id = nearest_object_id
            self.pending_grab = False
            self.mode_text = "Grabbed"
            self.state_changed.emit(self.mode_text)

        def _release_object(self) -> None:
            if self.grab_constraint is not None:
                bullet.removeConstraint(self.grab_constraint, physicsClientId=self.client_id)
                self.grab_constraint = None
            self.grabbed_object_id = None

        def _update_gl_items(self) -> None:
            points = self._render_points()
            self.arm_line.setData(pos=points)
            self.joint_points.setData(pos=points)
            crossbar, left, right, _grasp_anchor = self._gripper_geometry(points[-1])
            self.gripper_crossbar.setData(pos=crossbar)
            self.left_gripper.setData(pos=left)
            self.right_gripper.setData(pos=right)
            for object_id, box, spec in zip(self.object_ids, self.object_boxes, self.object_specs):
                object_pos, _object_orn = bullet.getBasePositionAndOrientation(object_id, physicsClientId=self.client_id)
                self._set_box_transform(box, object_pos, spec["size"])

        def _refresh_status(self) -> None:
            attach = "attached" if self.grab_constraint is not None else "free"
            motion_left = max(0.0, self.motion_until - time.perf_counter())
            if hasattr(self, "state_badge"):
                self.state_badge.setText(self.mode_text)
            self.status_label.setText(
                f"OpenGL 3D arm: {self.mode_text} | xyz=({self.ee_target[0]:.2f}, {self.ee_target[1]:.2f}, {self.ee_target[2]:.2f}) | object={attach} | motion={motion_left:.2f}s"
            )

        def closeEvent(self, event) -> None:
            self._disconnect_physics()
            super().closeEvent(event)

        def _disconnect_physics(self) -> None:
            if self.client_id >= 0:
                try:
                    bullet.disconnect(self.client_id)
                except Exception:
                    pass
                self.client_id = -1

        def __del__(self):
            self._disconnect_physics()

    class ConfigPage(QtWidgets.QWidget):
        config_changed = QtCore.pyqtSignal()

        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel("系统设置")
            title.setObjectName("title")
            layout.addWidget(title)
            desc = QtWidgets.QLabel("集中设置采集、训练、离线测试和 VR 实时测试参数。")
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            layout.addWidget(desc)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            content = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(content)
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(14)

            collection_card = Card()
            collection_form = QtWidgets.QFormLayout(collection_card)
            self.collection_mode_combo = QtWidgets.QComboBox()
            self.collection_mode_combo.addItems(["BrainVision LSL（EEG+肌电）", "模拟数据"])
            collection_form.addRow("采集方式", self.collection_mode_combo)
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
            self.pre_rest_spin.setValue(3.0)
            collection_form.addRow("想象前休息", self.pre_rest_spin)
            self.cue_spin = QtWidgets.QDoubleSpinBox()
            self.cue_spin.setRange(0.0, 5.0)
            self.cue_spin.setSingleStep(0.5)
            self.cue_spin.setValue(1.0)
            collection_form.addRow("提示秒数", self.cue_spin)
            self.cue_rest_spin = QtWidgets.QDoubleSpinBox()
            self.cue_rest_spin.setRange(0.0, 10.0)
            self.cue_rest_spin.setSingleStep(0.5)
            self.cue_rest_spin.setValue(3.0)
            collection_form.addRow("提示后休息", self.cue_rest_spin)
            self.post_rest_spin = QtWidgets.QDoubleSpinBox()
            self.post_rest_spin.setRange(0.0, 10.0)
            self.post_rest_spin.setSingleStep(0.5)
            self.post_rest_spin.setValue(3.0)
            collection_form.addRow("想象后休息", self.post_rest_spin)
            grid.addWidget(collection_card, 0, 0)

            io_card = Card()
            io_form = QtWidgets.QFormLayout(io_card)
            lsl_hint = QtWidgets.QLabel("BrainVision LSL 自动发现 LiveAmp / EEG 流；无需填写 IP、端口或流名。当前四路肌电默认使用 AUX1-AUX4。")
            lsl_hint.setWordWrap(True)
            lsl_hint.setObjectName("hint")
            io_form.addRow("硬件接口", lsl_hint)
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
            grid.addWidget(io_card, 0, 1)

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
            self.eeg_weight = QtWidgets.QDoubleSpinBox()
            self.eeg_weight.setRange(0.35, 0.85)
            self.eeg_weight.setSingleStep(0.05)
            self.eeg_weight.setValue(0.55)
            model_form.addRow("EEG/MI 融合权重", self.eeg_weight)
            self.conf_spin = QtWidgets.QDoubleSpinBox()
            self.conf_spin.setRange(0.40, 0.95)
            self.conf_spin.setSingleStep(0.02)
            self.conf_spin.setValue(0.62)
            model_form.addRow("指令置信度阈值", self.conf_spin)
            self.margin_spin = QtWidgets.QDoubleSpinBox()
            self.margin_spin.setRange(0.00, 0.50)
            self.margin_spin.setSingleStep(0.01)
            self.margin_spin.setValue(0.08)
            model_form.addRow("概率间隔阈值", self.margin_spin)
            self.stable_spin = QtWidgets.QSpinBox()
            self.stable_spin.setRange(1, 8)
            self.stable_spin.setValue(2)
            model_form.addRow("连续稳定窗口", self.stable_spin)
            self.classifier_combo = QtWidgets.QComboBox()
            self.classifier_combo.addItems(["logreg", "lda"])
            model_form.addRow("分类器", self.classifier_combo)
            self.csp_check = QtWidgets.QCheckBox("启用 CSP")
            self.csp_check.setChecked(True)
            model_form.addRow(self.csp_check)
            grid.addWidget(model_card, 1, 0)

            realtime_card = Card()
            realtime_form = QtWidgets.QFormLayout(realtime_card)
            self.vr_model_edit = QtWidgets.QLineEdit(os.path.join(APP_DIR, "models", "arm_bci_gui.joblib"))
            realtime_form.addRow("VR 模型/权重", self.path_row(self.vr_model_edit, self.pick_vr_model, "选择"))
            sync_model = QtWidgets.QPushButton("模型输出 -> VR")
            sync_model.clicked.connect(lambda: self.set_model_path(self.model_edit.text().strip()))
            realtime_form.addRow(sync_model)
            self.online_mode_combo = QtWidgets.QComboBox()
            self.online_mode_combo.addItems(["模拟在线", "BrainVision LSL肌电"])
            realtime_form.addRow("实时采集方式", self.online_mode_combo)
            self.target_edit = QtWidgets.QLineEdit("target_1")
            realtime_form.addRow("VR 目标区域", self.target_edit)
            self.refractory_spin = QtWidgets.QDoubleSpinBox()
            self.refractory_spin.setRange(0.0, 3.0)
            self.refractory_spin.setSingleStep(0.05)
            self.refractory_spin.setValue(0.25)
            realtime_form.addRow("指令冷却秒数", self.refractory_spin)
            self.command_duration_spin = QtWidgets.QDoubleSpinBox()
            self.command_duration_spin.setRange(0.05, 2.0)
            self.command_duration_spin.setSingleStep(0.05)
            self.command_duration_spin.setValue(0.55)
            realtime_form.addRow("VR 指令保持秒数", self.command_duration_spin)
            grid.addWidget(realtime_card, 1, 1)

            paradigm_card = Card()
            paradigm_layout = QtWidgets.QVBoxLayout(paradigm_card)
            paradigm_title = QtWidgets.QLabel("范式映射")
            paradigm_title.setObjectName("vrTitle")
            paradigm_layout.addWidget(paradigm_title)
            paradigm_text = QtWidgets.QTextEdit()
            paradigm_text.setReadOnly(True)
            paradigm_text.setMaximumHeight(220)
            lines = []
            for label in ["left", "right", "forward", "backward", "up", "down", GRIP_TOGGLE_LABEL]:
                info = MI8_PARADIGM[label]
                lines.append(f"{info['action']} | {info['modality']} | {info['region']} | {info['subject_task']}")
            paradigm_text.setPlainText("\n".join(lines))
            paradigm_layout.addWidget(paradigm_text)
            grid.addWidget(paradigm_card, 2, 0, 1, 2)

            scroll.setWidget(content)
            layout.addWidget(scroll, 1)
            self.bind_change_signals()
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

        def bind_change_signals(self) -> None:
            widgets = [
                self.collection_mode_combo,
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
                self.train_data_edit,
                self.test_data_edit,
                self.model_edit,
                self.eeg_weight,
                self.conf_spin,
                self.margin_spin,
                self.stable_spin,
                self.classifier_combo,
                self.csp_check,
                self.vr_model_edit,
                self.online_mode_combo,
                self.target_edit,
                self.refractory_spin,
                self.command_duration_spin,
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
            self.collection_mode_combo.currentTextChanged.connect(self.update_mode_fields)
            self.online_mode_combo.currentTextChanged.connect(self.update_mode_fields)

        def update_mode_fields(self) -> None:
            is_collection_rda = self.collection_mode_combo.currentText().startswith("64导")
            is_online_rda = self.online_mode_combo.currentText().startswith("64导")
            is_any_rda = is_collection_rda or is_online_rda
            for widget in [self.rda_host_edit, self.rda_port_spin, self.rda_channels_spin, self.aux_channels_spin]:
                widget.setEnabled(is_any_rda)
            for widget in [self.parallel_check, self.trigger_port_edit]:
                widget.setEnabled(is_collection_rda)
            self.fs_spin.setEnabled(not is_collection_rda)

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
                eeg_weight=float(self.eeg_weight.value()),
                command_confidence=float(self.conf_spin.value()),
                min_margin=float(self.margin_spin.value()),
                stable_windows=int(self.stable_spin.value()),
                refractory_seconds=float(self.refractory_spin.value()),
                command_duration_seconds=float(self.command_duration_spin.value()),
            )

        def train_config(self) -> Dict:
            return {
                "data_path": self.train_data_edit.text().strip(),
                "test_data_path": self.test_data_edit.text().strip(),
                "model_path": self.model_edit.text().strip(),
                "classifier": self.classifier_combo.currentText(),
                "use_csp": self.csp_check.isChecked(),
                "decoder": self.decoder_config(),
            }

        def realtime_config(self) -> Dict:
            return {
                "model_path": self.vr_model_edit.text().strip(),
                "mode": self.online_mode_combo.currentText(),
                "host": self.rda_host_edit.text().strip() or "127.0.0.1",
                "port": int(self.rda_port_spin.value()),
                "expected_channels": int(self.rda_channels_spin.value()),
                "aux_channels": int(self.aux_channels_spin.value()),
                "target": self.target_edit.text().strip() or "target_1",
                "decoder": self.decoder_config(),
            }

    class CollectionPage(QtWidgets.QWidget):
        recording_saved = QtCore.pyqtSignal(str)

        def __init__(self, config: ConfigPage, parent=None):
            super().__init__(parent)
            self.config = config
            self.worker: Optional[QtCore.QThread] = None
            self.phase_end_time = 0.0
            self.phase_total_seconds = 0.0
            self.countdown_timer = QtCore.QTimer(self)
            self.countdown_timer.timeout.connect(self.update_countdown)
            self.phase_index = {"pre_rest": 0, "cue": 1, "cue_rest": 2, "imagery": 3, "post_rest": 4}

            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel("数据采集")
            title.setObjectName("title")
            layout.addWidget(title)
            self.desc_label = QtWidgets.QLabel("")
            self.desc_label.setWordWrap(True)
            self.desc_label.setObjectName("desc")
            layout.addWidget(self.desc_label)

            stage = QtWidgets.QFrame()
            stage.setObjectName("paradigmStage")
            stage_layout = QtWidgets.QVBoxLayout(stage)
            stage_layout.setContentsMargins(24, 22, 24, 22)
            header = QtWidgets.QHBoxLayout()
            self.trial_label = QtWidgets.QLabel("Trial 0/0")
            self.trial_label.setObjectName("phaseBadge")
            self.phase_label = QtWidgets.QLabel("准备")
            self.phase_label.setObjectName("phaseBadge")
            self.countdown_label = QtWidgets.QLabel("0.0s")
            self.countdown_label.setObjectName("countdown")
            header.addWidget(self.trial_label)
            header.addWidget(self.phase_label)
            header.addStretch(1)
            header.addWidget(self.countdown_label)
            stage_layout.addLayout(header)
            self.stream_label = QtWidgets.QLabel("数据源：等待连接 | 计划读取 EEG 1-32 + AUX 33-36")
            self.stream_label.setObjectName("hint")
            self.stream_label.setWordWrap(True)
            stage_layout.addWidget(self.stream_label)
            self.cue_label = QtWidgets.QLabel("休息")
            self.cue_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.cue_label.setObjectName("cueWord")
            self.cue_label.setWordWrap(True)
            stage_layout.addWidget(self.cue_label, 1)
            self.instruction_label = QtWidgets.QLabel("准备开始")
            self.instruction_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.instruction_label.setObjectName("phaseInstruction")
            stage_layout.addWidget(self.instruction_label)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            stage_layout.addWidget(self.progress)
            layout.addWidget(stage, 2)

            controls = QtWidgets.QVBoxLayout()
            button_row = QtWidgets.QHBoxLayout()
            self.start_button = QtWidgets.QPushButton("开始范式采集")
            self.start_button.setObjectName("primaryButton")
            self.start_button.clicked.connect(self.start_collection)
            self.stop_button = QtWidgets.QPushButton("停止")
            self.stop_button.setObjectName("dangerButton")
            self.stop_button.clicked.connect(self.stop_collection)
            self.stop_button.setEnabled(False)
            self.config_label = QtWidgets.QLabel("")
            self.config_label.setObjectName("hint")
            self.config_label.setWordWrap(True)
            button_row.addWidget(self.start_button)
            button_row.addWidget(self.stop_button)
            button_row.addStretch(1)
            controls.addLayout(button_row)
            controls.addWidget(self.config_label)
            layout.addLayout(controls)

            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log, 1)
            self.config.config_changed.connect(self.refresh_config_label)
            self.refresh_config_label()

        def refresh_config_label(self) -> None:
            try:
                cfg = self.config.collection_config()
            except Exception as exc:
                self.config_label.setText(f"Config 参数错误：{exc}")
                return
            self.config_label.setText(
                f"{cfg['mode']} | 输出 {cfg['out_path']} | 每类 {cfg['trials_per_class']} trial | "
                f"保存 EEG 1-{cfg['expected_channels']} + AUX 33-{32 + cfg['aux_channels']} + 原始 LSL 流 | "
                f"阶段 {cfg['pre_rest_seconds']:.1f}s/{cfg['cue_seconds']:.1f}s/{cfg['cue_rest_seconds']:.1f}s/{cfg['epoch_seconds']:.1f}s/{cfg['post_rest_seconds']:.1f}s"
            )
            trial_seconds = sum(
                float(cfg[key])
                for key in ["pre_rest_seconds", "cue_seconds", "cue_rest_seconds", "epoch_seconds", "post_rest_seconds"]
            )
            total_trials = len(LABELS) * int(cfg["trials_per_class"])
            total_seconds = 2.0 + trial_seconds * total_trials
            total_whole_seconds = int(round(total_seconds))
            total_minutes, total_remaining_seconds = divmod(total_whole_seconds, 60)
            self.desc_label.setText(
                "当前范式：想象前休息 {pre:.1f}s -> 提示 {cue:.1f}s -> 提示后休息 {cue_rest:.1f}s -> "
                "采集/执行 {epoch:.1f}s -> 想象后休息 {post:.1f}s。每 Trial {trial:.1f}s，共 {count} 个 Trial，"
                "预计 {minutes:d}分{seconds:02d}秒（含连接后的2秒准备）。".format(
                    pre=cfg["pre_rest_seconds"],
                    cue=cfg["cue_seconds"],
                    cue_rest=cfg["cue_rest_seconds"],
                    epoch=cfg["epoch_seconds"],
                    post=cfg["post_rest_seconds"],
                    trial=trial_seconds,
                    count=total_trials,
                    minutes=total_minutes,
                    seconds=total_remaining_seconds,
                )
            )

        def start_collection(self) -> None:
            if self.worker is not None and self.worker.isRunning():
                QtWidgets.QMessageBox.information(self, "正在采集", "范式采集正在运行。")
                return
            try:
                cfg = self.config.collection_config()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Config 参数错误", str(exc))
                return
            path = timestamped_recording_path(cfg["out_path"] or RECORDINGS_DIR)
            self.log.append(f"本次记录将归档到：{path}")
            if cfg["mode"].startswith("8通道"):
                QtWidgets.QMessageBox.information(self, "8通道待接入", "8通道采集接口先保留；等你给设备 SDK/采集脚本后再接入。")
                self.log.append("8通道采集入口已保留，当前尚未绑定具体设备。")
                return

            self.reset_stage()
            if "LSL" in cfg["mode"]:
                self.stream_label.setText("数据源：正在发现 BrainVision LSL | 将读取 EEG 1-32 + AUX 33-36")
                self.log.append("正在自动发现 BrainVision LSL；连接后等待 2 秒再开始范式。")
                self.worker = LslCollectionWorker(
                    out_path=path,
                    trials_per_class=cfg["trials_per_class"],
                    epoch_seconds=cfg["epoch_seconds"],
                    pre_rest_seconds=cfg["pre_rest_seconds"],
                    cue_seconds=cfg["cue_seconds"],
                    cue_rest_seconds=cfg["cue_rest_seconds"],
                    post_rest_seconds=cfg["post_rest_seconds"],
                    expected_eeg_channels=cfg["expected_channels"],
                    aux_channels=cfg["aux_channels"],
                    parent=self,
                )
            elif cfg["mode"].startswith("64导"):
                self.log.append("开始64导 BrainVision RDA 范式采集...")
                self.log.append("请确认 BrainVision Recorder 已开启 RDA 输出，32-bit 端口通常为 51244。")
                self.worker = RDA64CollectionWorker(
                    out_path=path,
                    trials_per_class=cfg["trials_per_class"],
                    epoch_seconds=cfg["epoch_seconds"],
                    pre_rest_seconds=cfg["pre_rest_seconds"],
                    cue_seconds=cfg["cue_seconds"],
                    cue_rest_seconds=cfg["cue_rest_seconds"],
                    post_rest_seconds=cfg["post_rest_seconds"],
                    host=cfg["host"],
                    port=cfg["port"],
                    expected_channels=cfg["expected_channels"],
                    aux_channels=cfg["aux_channels"],
                    use_parallel_trigger=cfg["use_parallel_trigger"],
                    parallel_port=cfg["parallel_port"],
                    parent=self,
                )
            else:
                self.stream_label.setText("数据源：模拟数据（仅在系统设置明确选择模拟时启用）")
                self.log.append("开始模拟 6类咬肌+MI 范式显示，结束后生成模拟校准数据...")
                self.worker = SimulatedCollectionWorker(
                    out_path=path,
                    fs=cfg["fs"],
                    trials_per_class=cfg["trials_per_class"],
                    epoch_seconds=cfg["epoch_seconds"],
                    pre_rest_seconds=cfg["pre_rest_seconds"],
                    cue_seconds=cfg["cue_seconds"],
                    cue_rest_seconds=cfg["cue_rest_seconds"],
                    post_rest_seconds=cfg["post_rest_seconds"],
                    parent=self,
                )
            self.worker.log_signal.connect(self.log.append)
            self.worker.phase_signal.connect(self.on_phase)
            if hasattr(self.worker, "stream_signal"):
                self.worker.stream_signal.connect(self.on_stream_connected)
            self.worker.finished_signal.connect(self.on_collection_finished)
            self.worker.error_signal.connect(self.on_collection_error)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.worker.start()

        def reset_stage(self) -> None:
            self.countdown_timer.stop()
            self.trial_label.setText("Trial 0/0")
            self.phase_label.setText("准备")
            self.countdown_label.setText("0.0s")
            self.cue_label.setText("休息")
            self.instruction_label.setText("准备开始")
            self.progress.setValue(0)

        def on_stream_connected(self, info: Dict) -> None:
            aux_text = ",".join(str(value) for value in info.get("aux_channels", [])) or "-"
            self.stream_label.setText(
                f"真实 LSL：{info.get('name', '-')} | {info.get('channels', 0)} 通道 | {info.get('fs', 0):g} Hz | "
                f"本次读取 EEG 1-{info.get('eeg_count', 0)} + AUX {aux_text}"
            )
            self.log.append(
                f"已确认真实数据通道：EEG={info.get('eeg_count', 0)} 路，AUX/EMG={len(info.get('aux_channels', []))} 路。"
            )

        def _acknowledge_phase_display(self) -> None:
            worker = self.worker
            if worker is not None and hasattr(worker, "acknowledge_phase_displayed"):
                worker.acknowledge_phase_displayed()

        def on_phase(self, label: str, phase: str, idx: int, total: int, seconds: float) -> None:
            if phase == "arming":
                self.trial_label.setText(f"连接成功 | 共 {total} 个 Trial")
                self.phase_label.setText("准备")
                self.cue_label.setText("请保持放松")
                self.instruction_label.setText("2 秒后开始同步采集 EEG 与肌电")
                self.phase_total_seconds = max(0.0, float(seconds))
                self.phase_end_time = time.perf_counter() + self.phase_total_seconds
                self.countdown_label.setText(f"{self.phase_total_seconds:.1f}s")
                self.countdown_timer.start(50)
                self.log.append("LSL 连接成功，2 秒后开始采集。")
                self._acknowledge_phase_display()
                return
            display = LABEL_DISPLAY_NAMES.get(label, label)
            info = paradigm_info(label)
            phase_names = {
                "pre_rest": "休息",
                "cue": "提示",
                "cue_rest": "提示后休息",
                "imagery": "采集/执行",
                "post_rest": "采集后休息",
            }
            instructions = {
                "pre_rest": "放松，保持静息",
                "cue": "看提示，准备对应任务",
                "cue_rest": "保持静息，不要提前想象",
                "imagery": "开始采集，持续执行当前任务",
                "post_rest": "放松，等待下一轮",
            }
            shown = f"{info['action']}\n{info['subject_task']}" if phase in {"cue", "imagery"} else "休息"
            if phase in {"cue", "imagery"}:
                task_prefix = "咬肌+运动想象" if info["modality"] != "REST" else "静息"
                instruction = f"{task_prefix} | {info['region']} | {info['feature']}"
            else:
                instruction = instructions.get(phase, "")
            self.trial_label.setText(f"Trial {idx}/{total}  目标：{info['action']}  来源：{info['modality']}")
            self.phase_label.setText(phase_names.get(phase, phase))
            self.cue_label.setText(shown)
            self.instruction_label.setText(instruction)
            self.phase_total_seconds = max(0.0, float(seconds))
            self.phase_end_time = time.perf_counter() + self.phase_total_seconds
            self.countdown_label.setText(f"{self.phase_total_seconds:.1f}s")
            if self.phase_total_seconds > 0:
                self.countdown_timer.start(50)
            phase_number = self.phase_index.get(phase, 0)
            total_steps = max(1, total * 5)
            current_step = min(total_steps, (idx - 1) * 5 + phase_number)
            self.progress.setValue(int(100 * current_step / total_steps))
            self.log.append(f"Cue: {idx}/{total} {phase_names.get(phase, phase)} {info['action']} | {info['subject_task']} | {seconds:.1f}s")
            self._acknowledge_phase_display()

        def update_countdown(self) -> None:
            remaining = max(0.0, self.phase_end_time - time.perf_counter())
            self.countdown_label.setText(f"{remaining:.1f}s")
            if remaining <= 0.0:
                self.countdown_timer.stop()

        def stop_collection(self) -> None:
            if self.worker is not None and self.worker.isRunning():
                self.log.append("正在停止范式采集...")
                self.stop_button.setEnabled(False)
                self.worker.stop()

        def on_collection_finished(self, result: Dict) -> None:
            self.countdown_timer.stop()
            path = str(result.get("path"))
            completed = bool(result.get("completed", True))
            completed_trials = int(result.get("completed_trials", len(result.get("labels", []))))
            total_trials = max(1, int(result.get("total_trials", completed_trials or 1)))
            self.progress.setValue(100 if completed else int(100 * completed_trials / total_trials))
            self.trial_label.setText(f"已保存 {completed_trials}/{total_trials} 个完整 Trial")
            self.phase_label.setText("完成" if completed else "已停止")
            self.cue_label.setText("采集完成" if completed else "已停止并保存")
            self.instruction_label.setText("全部数据已保存" if completed else "仅保存停止前完成的 Trial")
            self.log.append(f"采集数据已保存：{path}")
            if not completed:
                self.log.append(f"采集由用户提前停止：计划 {total_trials} 个 Trial，实际保存 {completed_trials} 个完整 Trial。")
            if result.get("raw_shape"):
                self.log.append(f"原始流 raw: {result.get('raw_shape')}，截取 epoch: {result.get('eeg_shape')}，fs={result.get('fs')}")
            else:
                self.log.append(f"EEG: {result.get('eeg_shape')}, AUX: {result.get('aux_shape')}, labels: {len(result.get('labels', []))}")
            self.config.set_training_data_path(path)
            self.recording_saved.emit(path)
            self.worker = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        def on_collection_error(self, message: str) -> None:
            self.countdown_timer.stop()
            self.log.append(f"采集失败：{message}")
            QtWidgets.QMessageBox.critical(self, "采集失败", message)
            self.worker = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    class TrainTunePage(QtWidgets.QWidget):
        model_ready = QtCore.pyqtSignal(str)

        def __init__(self, config: ConfigPage, parent=None):
            super().__init__(parent)
            self.config = config
            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel("模型训练和测试")
            title.setObjectName("title")
            layout.addWidget(title)
            desc = QtWidgets.QLabel("扫描按时间归档的采集记录。逐条勾选训练集或独立测试集；表格明确显示每个文件实际包含的 32 路 EEG、AUX/肌电、Trial 数和来源。")
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            layout.addWidget(desc)

            card = Card()
            card_layout = QtWidgets.QVBoxLayout(card)
            toolbar = QtWidgets.QHBoxLayout()
            self.recording_root_label = QtWidgets.QLabel(f"真实采集目录：{RECORDINGS_DIR}")
            self.recording_root_label.setObjectName("hint")
            self.recording_root_label.setWordWrap(True)
            self.refresh_button = QtWidgets.QPushButton("刷新记录")
            self.refresh_button.clicked.connect(self.refresh_recordings)
            toolbar.addWidget(self.recording_root_label, 1)
            toolbar.addWidget(self.refresh_button)
            card_layout.addLayout(toolbar)

            self.recording_table = QtWidgets.QTableWidget(0, 9)
            self.recording_table.setHorizontalHeaderLabels(
                ["训练", "测试", "采集时间", "来源", "Trial", "EEG", "AUX", "采样率", "文件"]
            )
            self.recording_table.setAlternatingRowColors(True)
            self.recording_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.recording_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.recording_table.verticalHeader().setVisible(False)
            header = self.recording_table.horizontalHeader()
            for col in range(8):
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(8, QtWidgets.QHeaderView.ResizeMode.Stretch)
            self.recording_table.setMinimumHeight(280)
            card_layout.addWidget(self.recording_table)

            options = QtWidgets.QHBoxLayout()
            self.test_model_check = QtWidgets.QCheckBox("离线测试使用当前模型/权重")
            self.test_model_check.setChecked(True)
            options.addWidget(self.test_model_check)
            options.addStretch(1)
            card_layout.addLayout(options)
            self.path_label = QtWidgets.QLabel("")
            self.path_label.setWordWrap(True)
            self.path_label.setObjectName("hint")
            card_layout.addWidget(self.path_label)
            controls = QtWidgets.QHBoxLayout()
            self.train_button = QtWidgets.QPushButton("开始训练")
            self.train_button.clicked.connect(self.train_model)
            self.eval_button = QtWidgets.QPushButton("离线测试并绘图")
            self.eval_button.clicked.connect(self.evaluate_model)
            controls.addWidget(self.train_button)
            controls.addWidget(self.eval_button)
            card_layout.addLayout(controls)
            layout.addWidget(card)

            self.summary_label = QtWidgets.QLabel("准确率：-")
            self.summary_label.setObjectName("phaseBadge")
            layout.addWidget(self.summary_label)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setObjectName("accuracyPlot")
            self.plot_widget.setMinimumHeight(220)
            self.plot_widget.setBackground("#0d131a")
            self.plot_widget.getPlotItem().showGrid(y=True, alpha=0.25)
            self.plot_widget.setLabel("left", "Accuracy (%)")
            self.plot_widget.setLabel("bottom", "Metric")
            layout.addWidget(self.plot_widget, 1)

            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log, 1)
            self.config.config_changed.connect(self.refresh_paths)
            self.refresh_recordings()

        def refresh_paths(self) -> None:
            cfg = self.config.train_config()
            train_count = len(self.selected_recordings("train"))
            test_count = len(self.selected_recordings("test"))
            self.path_label.setText(
                f"已选训练记录 {train_count} 个；测试记录 {test_count} 个；模型/权重：{cfg['model_path']}"
            )

        def selected_recordings(self, role: str) -> List[str]:
            column = 0 if role == "train" else 1
            paths: List[str] = []
            for row in range(self.recording_table.rowCount()):
                holder = self.recording_table.cellWidget(row, column)
                check = holder.findChild(QtWidgets.QCheckBox) if holder is not None else None
                path_item = self.recording_table.item(row, 8)
                if check is not None and check.isChecked() and path_item is not None:
                    paths.append(path_item.data(QtCore.Qt.ItemDataRole.UserRole) or path_item.text())
            return paths

        def _dataset_role_changed(self, path: str, role: str, checked: bool) -> None:
            if checked:
                opposite = 1 if role == "train" else 0
                for row in range(self.recording_table.rowCount()):
                    item = self.recording_table.item(row, 8)
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
            previous_train = set(self.selected_recordings("train")) if hasattr(self, "recording_table") else set()
            previous_test = set(self.selected_recordings("test")) if hasattr(self, "recording_table") else set()
            if preferred_train_path:
                previous_train.add(os.path.abspath(preferred_train_path))
                previous_test.discard(os.path.abspath(preferred_train_path))
            self.recording_table.setRowCount(0)
            valid_count = 0
            for path in recording_npz_files(RECORDINGS_DIR):
                try:
                    meta = inspect_recording_npz(path)
                except Exception as exc:
                    self.log.append(f"跳过无效记录：{path} | {exc}")
                    continue
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
                mode = "真实 LSL" if meta["mode"] == "brainvision_lsl" else str(meta["mode"])
                values = [
                    str(meta["time"]),
                    mode + ("" if meta["completed"] else "（未完成）"),
                    str(meta["trials"]),
                    str(meta["eeg_channels"]),
                    str(meta["aux_channels"]),
                    f"{meta['fs']} Hz",
                    os.path.relpath(abs_path, RECORDINGS_DIR),
                ]
                for col, value in enumerate(values, start=2):
                    item = QtWidgets.QTableWidgetItem(value)
                    if col == 8:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, abs_path)
                        item.setToolTip(abs_path)
                    self.recording_table.setItem(row, col, item)
                valid_count += 1
            if valid_count and not previous_train and not preferred_train_path:
                first = self.recording_table.cellWidget(0, 0)
                check = first.findChild(QtWidgets.QCheckBox) if first is not None else None
                if check is not None:
                    check.setChecked(True)
            self.recording_root_label.setText(f"采集目录：{RECORDINGS_DIR} | 找到 {valid_count} 条可训练记录")
            self.refresh_paths()

        def recording_added(self, path: str) -> None:
            self.refresh_recordings(path)
            self.log.append(f"已加入最新真实采集记录：{path}")

        def train_model(self) -> None:
            cfg = self.config.train_config()
            data_paths = self.selected_recordings("train")
            model_path = cfg["model_path"]
            if not data_paths:
                QtWidgets.QMessageBox.warning(self, "未勾选训练数据", "请在记录表的“训练”列至少勾选一条真实采集记录。")
                return
            if not model_path:
                QtWidgets.QMessageBox.warning(self, "缺少模型路径", "请先在 Config 页设置模型/权重输出路径。")
                return
            self.train_button.setEnabled(False)
            self.eval_button.setEnabled(False)
            QtWidgets.QApplication.processEvents()
            try:
                eeg, y, aux, file_fs, channel_names = load_calibration_npz_many(data_paths)
                signal_cfg = SignalConfig(fs=file_fs or int(self.config.fs_spin.value()))
                signal_cfg.eeg_channels = channel_names
                signal_cfg.use_csp = bool(cfg["use_csp"])
                decoder_cfg = cfg["decoder"]
                model = HybridBCIModel(signal_cfg, decoder_cfg)
                self.log.append(f"开始训练咬肌+MI 融合模型：{len(data_paths)} 条记录，{len(y)} 个 trial，EEG={eeg.shape[1]}，AUX={0 if aux is None else aux.shape[1]}")
                metrics = model.fit(eeg, y, aux=aux, classifier_kind=cfg["classifier"])
                model.save(model_path)
                report_path = os.path.splitext(model_path)[0] + "_report.json"
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": model_path,
                            "data": data_paths,
                            "signal_config": asdict(signal_cfg),
                            "decoder_config": asdict(decoder_cfg),
                            "metrics": metrics,
                            "paradigm": MI8_PARADIGM,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.config.set_model_path(model_path)
                self.config.train_data_edit.setText(data_paths[0])
                self.log.append(f"模型已保存：{model_path}")
                self.log.append(f"报告已保存：{report_path}")
                self.log.append(json.dumps(metrics_summary(metrics), ensure_ascii=False, indent=2))
                self.update_accuracy_plot(metrics, "训练验证准确率")
                self.model_ready.emit(model_path)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "训练失败", str(exc))
                self.log.append(f"训练失败：{exc}")
            finally:
                self.train_button.setEnabled(True)
                self.eval_button.setEnabled(True)

        def evaluate_model(self) -> None:
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
            QtWidgets.QApplication.processEvents()
            try:
                model = HybridBCIModel.load(model_path)
                eeg, y, aux, _file_fs, _channel_names = load_calibration_npz_many(data_paths)
                y_idx = labels_to_indices(y)
                eeg_windows, y_windows, aux_windows = make_windows_from_trials(eeg, y_idx, model.signal_config, aux)
                metrics = model.evaluate_windows(eeg_windows, y_windows, aux_windows)
                report_path = os.path.splitext(model_path)[0] + "_offline_eval.json"
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "model": model_path,
                            "data": data_paths,
                            "metrics": metrics,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                self.config.test_data_edit.setText(data_paths[0])
                self.log.append(f"离线测试完成：{len(data_paths)} 条独立记录，{len(y)} 个 trial")
                self.log.append(f"离线测试报告：{report_path}")
                self.log.append(json.dumps(metrics_summary(metrics), ensure_ascii=False, indent=2))
                self.update_accuracy_plot(metrics, "离线测试准确率")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "离线测试失败", str(exc))
                self.log.append(f"离线测试失败：{exc}")
            finally:
                self.train_button.setEnabled(True)
                self.eval_button.setEnabled(True)

        def update_accuracy_plot(self, metrics: Dict, title: str) -> None:
            self.plot_widget.clear()
            items = [
                ("EEG", metrics.get("eeg_only_accuracy")),
                ("AUX", metrics.get("aux_only_accuracy")),
                ("Fusion", metrics.get("fusion_accuracy")),
            ]
            labels = [name for name, value in items if value is not None]
            values = [float(value) * 100.0 for _name, value in items if value is not None]
            if not values:
                self.summary_label.setText("准确率：没有可绘制的验证结果")
                return
            x = np.arange(len(values), dtype=float)
            bars = pg.BarGraphItem(x=x, height=values, width=0.56, brush=pg.mkBrush("#2e8fbd"))
            self.plot_widget.addItem(bars)
            self.plot_widget.getAxis("bottom").setTicks([[(float(i), labels[i]) for i in range(len(labels))]])
            self.plot_widget.setYRange(0, 100)
            self.plot_widget.setTitle(title, color="#eef3f8")
            for i, value in enumerate(values):
                text = pg.TextItem(f"{value:.1f}%", color="#eef3f8", anchor=(0.5, 1.0))
                text.setPos(float(i), min(99.0, value + 5.0))
                self.plot_widget.addItem(text)
            best = max(values)
            self.summary_label.setText(f"准确率：{title} 最高 {best:.1f}%")

    class UsePage(QtWidgets.QWidget):
        def __init__(self, bus: CommandBus, parent=None):
            super().__init__(parent)
            self.bus = bus
            self.model: Optional[HybridBCIModel] = None
            self.controller: Optional[AsyncIntentController] = None
            self.source: Optional[SignalSource] = None
            self.robot = GuiRobotArm(bus, self.append_log)
            self.grip_controller = GripStateController(self.robot)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick_online)
            self.sequence = 0

            layout = QtWidgets.QVBoxLayout(self)
            title = QtWidgets.QLabel("实际使用页面")
            title.setObjectName("title")
            layout.addWidget(title)
            desc = QtWidgets.QLabel("加载模型后运行在线测试。系统连续读取 EEG/AUX，六类移动由咬肌+左右手想象分类；三次眨眼单独切换抓/放状态。")
            desc.setWordWrap(True)
            desc.setObjectName("desc")
            layout.addWidget(desc)

            card = Card()
            form = QtWidgets.QFormLayout(card)
            self.model_edit = QtWidgets.QLineEdit(os.path.join(APP_DIR, "models", "arm_bci_gui.joblib"))
            form.addRow("模型文件", self.path_row(self.model_edit, self.pick_model))
            self.target_edit = QtWidgets.QLineEdit("target_1")
            form.addRow("目标区域", self.target_edit)
            self.online_mode_combo = QtWidgets.QComboBox()
            self.online_mode_combo.addItems(["模拟在线", "64导 BrainVision RDA在线", "8通道（待接入）"])
            self.online_mode_combo.currentTextChanged.connect(self.update_online_mode_fields)
            form.addRow("在线模式", self.online_mode_combo)
            self.rda_host_edit = QtWidgets.QLineEdit("127.0.0.1")
            form.addRow("RDA 主机", self.rda_host_edit)
            self.rda_port_spin = QtWidgets.QSpinBox()
            self.rda_port_spin.setRange(1, 65535)
            self.rda_port_spin.setValue(51244)
            form.addRow("RDA 端口", self.rda_port_spin)
            self.rda_channels_spin = QtWidgets.QSpinBox()
            self.rda_channels_spin.setRange(1, 128)
            self.rda_channels_spin.setValue(64)
            form.addRow("EEG 通道数", self.rda_channels_spin)
            controls = QtWidgets.QHBoxLayout()
            self.load_button = QtWidgets.QPushButton("加载模型")
            self.load_button.clicked.connect(self.load_model)
            self.start_button = QtWidgets.QPushButton("启动在线")
            self.start_button.clicked.connect(self.start_online)
            self.stop_button = QtWidgets.QPushButton("停止")
            self.stop_button.clicked.connect(self.stop_online)
            self.reset_button = QtWidgets.QPushButton("重置任务")
            self.reset_button.clicked.connect(self.reset_task)
            for button in [self.load_button, self.start_button, self.stop_button, self.reset_button]:
                controls.addWidget(button)
            form.addRow(controls)
            layout.addWidget(card)

            manual = Card()
            manual_layout = QtWidgets.QVBoxLayout(manual)
            manual_layout.addWidget(QtWidgets.QLabel("手动指令测试"))
            row = QtWidgets.QHBoxLayout()
            for label, intent in [
                ("上", "up"),
                ("下", "down"),
                ("左", "left"),
                ("右", "right"),
                ("前", "forward"),
                ("后", "backward"),
                ("眨眼三次", GRIP_TOGGLE_LABEL),
                ("回零", "home"),
            ]:
                btn = QtWidgets.QPushButton(label)
                btn.clicked.connect(lambda _=False, i=intent: self.manual_command(i))
                row.addWidget(btn)
            manual_layout.addLayout(row)
            layout.addWidget(manual)

            self.status_label = QtWidgets.QLabel("状态：未加载模型")
            layout.addWidget(self.status_label)
            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log, 1)
            self.update_online_mode_fields()

        def path_row(self, edit: QtWidgets.QLineEdit, callback) -> QtWidgets.QWidget:
            box = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(box)
            row.setContentsMargins(0, 0, 0, 0)
            button = QtWidgets.QPushButton("浏览")
            button.clicked.connect(callback)
            row.addWidget(edit)
            row.addWidget(button)
            return box

        def pick_model(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择模型", APP_DIR, "Joblib Files (*.joblib)")
            if path:
                self.model_edit.setText(path)

        def set_model_path(self, path: str) -> None:
            self.model_edit.setText(path)

        def append_log(self, text: str) -> None:
            self.log.append(text)

        def update_online_mode_fields(self) -> None:
            is_rda = self.online_mode_combo.currentText().startswith("64导")
            for widget in [self.rda_host_edit, self.rda_port_spin, self.rda_channels_spin]:
                widget.setEnabled(is_rda)

        def load_model(self) -> None:
            try:
                self.model = HybridBCIModel.load(self.model_edit.text().strip())
                apply_runtime_control_config(self.model)
                self.controller = AsyncIntentController(self.model, self.robot, target=self.target_edit.text().strip())
                self.grip_controller = GripStateController(
                    self.robot,
                    target=self.target_edit.text().strip(),
                    duration_seconds=self.model.decoder_config.command_duration_seconds,
                )
                self.status_label.setText("状态：模型已加载")
                self.append_log(f"模型已加载：{self.model_edit.text().strip()}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))

        def start_online(self) -> None:
            if self.model is None:
                self.load_model()
            if self.model is None:
                return
            mode = self.online_mode_combo.currentText()
            if mode.startswith("8通道"):
                QtWidgets.QMessageBox.information(self, "8通道待接入", "8通道在线接口先保留；等你给设备 SDK/采集脚本后我再接入。")
                return
            self.controller = AsyncIntentController(self.model, self.robot, target=self.target_edit.text().strip())
            self.grip_controller = GripStateController(
                self.robot,
                target=self.target_edit.text().strip(),
                duration_seconds=self.model.decoder_config.command_duration_seconds,
            )
            try:
                if mode.startswith("64导"):
                    self.source = RDA64OnlineSource(
                        self.model.signal_config,
                        host=self.rda_host_edit.text().strip() or "127.0.0.1",
                        port=int(self.rda_port_spin.value()),
                        expected_channels=int(self.rda_channels_spin.value()),
                        log_callback=self.append_log,
                    )
                    self.source.start()
                    self.timer.start(50)
                    self.status_label.setText("状态：64导 RDA 在线运行中")
                    self.append_log("64导 RDA 在线读取已启动。")
                else:
                    self.source = SimulatedSignalSource(self.model.signal_config, cycles=3)
                    self.timer.start(int(self.model.signal_config.step_seconds * 1000))
                    self.status_label.setText("状态：模拟在线运行中")
                    self.append_log("模拟在线流已启动。")
            except Exception as exc:
                self.source = None
                QtWidgets.QMessageBox.critical(self, "启动在线失败", str(exc))
                self.append_log(f"启动在线失败：{exc}")

        def stop_online(self) -> None:
            self.timer.stop()
            if self.source is not None and hasattr(self.source, "close"):
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭在线源失败：{exc}")
            self.source = None
            self.status_label.setText("状态：已停止")
            self.append_log("在线流已停止。")

        def reset_task(self) -> None:
            if self.controller:
                self.controller.reset_task()
            self.grip_controller.reset()
            self.bus.command_emitted.emit({"cmd": "home", "intent": "home", "target": self.target_edit.text(), "confidence": 1.0, "sequence_id": 0, "timestamp": time.time()})
            self.append_log("任务状态已重置。")

        def tick_online(self) -> None:
            if self.source is None or self.controller is None:
                self.stop_online()
                return
            try:
                eeg, aux, now, true_label = self.source.read_window()
            except BlockingIOError:
                return
            except StopIteration:
                self.stop_online()
                self.append_log("模拟在线流完成。")
                return
            except Exception as exc:
                self.append_log(f"在线读取失败：{exc}")
                QtWidgets.QMessageBox.critical(self, "在线读取失败", str(exc))
                self.stop_online()
                return
            if is_grip_toggle_event(true_label):
                command = self.grip_controller.toggle(now=now, confidence=1.0)
                if command is not None:
                    self.status_label.setText(f"状态：三次眨眼 -> 抓握状态：{self.grip_controller.state_text}")
                    self.append_log(f"三次眨眼切换抓握：{command.name} / 状态={self.grip_controller.state_text}")
                return
            decision = self.controller.update(eeg, aux, now)
            label_text = true_label or "-"
            self.status_label.setText(
                f"状态：true={label_text} pred={decision.intent} conf={decision.confidence:.3f} "
                f"accepted={decision.accepted}"
            )
            self.append_log(
                f"true={label_text:9s} pred={decision.intent:9s} conf={decision.confidence:.3f} "
                f"margin={decision.margin:.3f} accepted={decision.accepted} reason={decision.reason}"
            )

        def manual_command(self, intent: str) -> None:
            if intent == "home":
                payload = {"cmd": "home", "intent": "home", "target": self.target_edit.text(), "confidence": 1.0, "sequence_id": 0, "timestamp": time.time(), "duration_seconds": 0.0}
            elif is_grip_toggle_event(intent):
                command = self.grip_controller.toggle(now=time.time(), confidence=1.0, force=True)
                if command is not None:
                    self.append_log(f"手动三次眨眼：{command.name} / 状态={self.grip_controller.state_text}")
                return
            else:
                self.sequence += 1
                payload = {
                    "cmd": ACTION_COMMANDS[intent],
                    "intent": intent,
                    "target": self.target_edit.text(),
                    "confidence": 1.0,
                    "sequence_id": self.sequence,
                    "timestamp": time.time(),
                    "duration_seconds": 0.55,
                }
            self.bus.command_emitted.emit(payload)
            self.append_log(f"手动发送：{payload['cmd']} / {payload['intent']}")

    class VRDemoPage(QtWidgets.QWidget):
        def __init__(self, bus: CommandBus, config: ConfigPage, parent=None):
            super().__init__(parent)
            self.bus = bus
            self.config = config
            self.model: Optional[HybridBCIModel] = None
            self.controller: Optional[AsyncIntentController] = None
            self.source: Optional[SignalSource] = None
            self.robot = GuiRobotArm(bus, self.append_log)
            self.grip_controller = GripStateController(self.robot)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick_online)
            self.sequence = 0
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(24, 18, 24, 18)
            layout.setSpacing(12)
            title = QtWidgets.QLabel("VR 测试")
            title.setObjectName("title")
            layout.addWidget(title)

            runtime = QtWidgets.QFrame()
            runtime.setObjectName("vrControls")
            runtime_layout = QtWidgets.QGridLayout(runtime)
            runtime_layout.setContentsMargins(12, 10, 12, 10)
            self.use_model_check = QtWidgets.QCheckBox("模型/权重")
            self.use_model_check.setChecked(True)
            self.use_weight_check = QtWidgets.QCheckBox("Config 权重/阈值")
            self.use_weight_check.setChecked(True)
            self.path_label = QtWidgets.QLabel("")
            self.path_label.setObjectName("hint")
            self.path_label.setWordWrap(True)
            self.load_button = QtWidgets.QPushButton("加载模型")
            self.load_button.clicked.connect(self.load_model)
            self.start_button = QtWidgets.QPushButton("开始实时分类")
            self.start_button.clicked.connect(self.start_online)
            self.stop_button = QtWidgets.QPushButton("停止")
            self.stop_button.clicked.connect(self.stop_online)
            self.reset_button = QtWidgets.QPushButton("回零")
            self.reset_button.clicked.connect(self.reset_task)
            runtime_layout.addWidget(self.use_model_check, 0, 0)
            runtime_layout.addWidget(self.use_weight_check, 0, 1)
            runtime_layout.addWidget(self.path_label, 0, 2, 1, 3)
            runtime_layout.addWidget(self.load_button, 1, 0)
            runtime_layout.addWidget(self.start_button, 1, 1)
            runtime_layout.addWidget(self.stop_button, 1, 2)
            runtime_layout.addWidget(self.reset_button, 1, 3)
            layout.addWidget(runtime)

            self.arm = BulletOpenGLArmWidget()
            self.arm.setObjectName("vrStage")
            bus.command_emitted.connect(self.arm.execute_payload)
            layout.addWidget(self.arm, 1)

            manual = QtWidgets.QFrame()
            manual.setObjectName("vrControls")
            row = QtWidgets.QGridLayout(manual)
            row.setContentsMargins(12, 10, 12, 10)
            row.setHorizontalSpacing(8)
            row.setVerticalSpacing(8)
            for button_index, (label, intent) in enumerate(
                [
                    ("上", "up"),
                    ("下", "down"),
                    ("左", "left"),
                    ("右", "right"),
                    ("前", "forward"),
                    ("后", "backward"),
                    ("眨眼三次", GRIP_TOGGLE_LABEL),
                    ("回零", "home"),
                ]
            ):
                btn = QtWidgets.QPushButton(label)
                btn.setObjectName("vrCmd")
                btn.clicked.connect(lambda _=False, i=intent: self.manual_command(i))
                row.addWidget(btn, button_index // 5, button_index % 5)
            layout.addWidget(manual)

            self.status_label = QtWidgets.QLabel("状态：未加载模型")
            self.status_label.setObjectName("vrStatus")
            layout.addWidget(self.status_label)
            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(128)
            layout.addWidget(self.log)

            for shortcut in ["Return", "Enter", "Space"]:
                QtGui.QShortcut(QtGui.QKeySequence(shortcut), self, activated=self.start_online)
            self.config.config_changed.connect(self.refresh_paths)
            self.refresh_paths()

        def refresh_paths(self) -> None:
            cfg = self.config.realtime_config()
            self.path_label.setText(f"{cfg['mode']} | 模型/权重：{cfg['model_path']} | 目标：{cfg['target']}")

        def set_model_path(self, path: str) -> None:
            self.config.set_model_path(path)
            self.refresh_paths()

        def append_log(self, text: str) -> None:
            self.log.append(text)

        def load_model(self) -> None:
            self.model = None
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
                self.model = HybridBCIModel.load(model_path)
                if self.use_weight_check.isChecked():
                    self.model.decoder_config = cfg["decoder"]
                apply_runtime_control_config(self.model)
                self.controller = AsyncIntentController(self.model, self.robot, target=cfg["target"])
                self.grip_controller = GripStateController(
                    self.robot,
                    target=cfg["target"],
                    duration_seconds=self.model.decoder_config.command_duration_seconds,
                )
                self.status_label.setText("状态：模型已加载")
                self.append_log(f"模型已加载：{model_path}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
                self.append_log(f"加载失败：{exc}")

        def start_online(self) -> None:
            if self.timer.isActive():
                return
            self.load_model()
            if self.model is None or self.controller is None:
                return
            cfg = self.config.realtime_config()
            mode = cfg["mode"]
            if "LSL" in mode:
                QtWidgets.QMessageBox.information(self, "BrainVision LSL", "当前四路肌电分类已内置在左侧“肌电测试”页；请在那里点击“连接并开始”。")
                self.append_log("BrainVision LSL 硬件实时测试请使用“肌电测试”页，VR 页不再使用 RDA/IP 连接。")
                return
            if mode.startswith("8通道"):
                QtWidgets.QMessageBox.information(self, "8通道待接入", "8通道在线接口先保留；等你给设备 SDK/采集脚本后再接入。")
                return
            self.controller = AsyncIntentController(self.model, self.robot, target=cfg["target"])
            self.grip_controller = GripStateController(
                self.robot,
                target=cfg["target"],
                duration_seconds=self.model.decoder_config.command_duration_seconds,
            )
            try:
                if mode.startswith("64导"):
                    self.source = RDA64OnlineSource(
                        self.model.signal_config,
                        host=cfg["host"],
                        port=cfg["port"],
                        expected_channels=cfg["expected_channels"],
                        aux_channels=cfg["aux_channels"],
                        log_callback=self.append_log,
                    )
                    self.source.start()
                    self.timer.start(50)
                    self.status_label.setText("状态：64导 RDA 实时分类中")
                    self.append_log("64导 RDA 在线读取已启动，分类结果将映射到 VR 操作。")
                else:
                    self.source = SimulatedSignalSource(self.model.signal_config, cycles=3)
                    self.timer.start(int(self.model.signal_config.step_seconds * 1000))
                    self.status_label.setText("状态：模拟实时分类中")
                    self.append_log("模拟在线流已启动，分类结果将映射到 VR 操作。")
                self.setFocus()
            except Exception as exc:
                self.source = None
                QtWidgets.QMessageBox.critical(self, "启动实时分类失败", str(exc))
                self.append_log(f"启动实时分类失败：{exc}")

        def stop_online(self) -> None:
            self.timer.stop()
            if self.source is not None and hasattr(self.source, "close"):
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭在线源失败：{exc}")
            self.source = None
            self.status_label.setText("状态：已停止")
            self.append_log("实时分类已停止。")

        def reset_task(self) -> None:
            if self.controller:
                self.controller.reset_task()
            self.grip_controller.reset()
            payload = {
                "cmd": "home",
                "intent": "home",
                "target": self.config.realtime_config()["target"],
                "confidence": 1.0,
                "sequence_id": 0,
                "timestamp": time.time(),
                "duration_seconds": 0.0,
            }
            self.bus.command_emitted.emit(payload)
            self.append_log("VR 场景已回零。")

        def tick_online(self) -> None:
            if self.source is None or self.controller is None:
                self.stop_online()
                return
            try:
                eeg, aux, now, true_label = self.source.read_window()
            except BlockingIOError:
                return
            except StopIteration:
                self.stop_online()
                self.append_log("模拟在线流完成。")
                return
            except Exception as exc:
                self.append_log(f"在线读取失败：{exc}")
                QtWidgets.QMessageBox.critical(self, "在线读取失败", str(exc))
                self.stop_online()
                return
            if is_grip_toggle_event(true_label):
                command = self.grip_controller.toggle(now=now, confidence=1.0)
                if command is not None:
                    self.status_label.setText(f"状态：三次眨眼 -> 抓握状态：{self.grip_controller.state_text}")
                    self.append_log(f"三次眨眼切换抓握：{command.name} / 状态={self.grip_controller.state_text}")
                return
            decision = self.controller.update(eeg, aux, now)
            label_text = true_label or "-"
            self.status_label.setText(
                f"状态：true={label_text} pred={decision.intent} conf={decision.confidence:.3f} "
                f"accepted={decision.accepted} reason={decision.reason}"
            )
            self.append_log(
                f"true={label_text:9s} pred={decision.intent:9s} conf={decision.confidence:.3f} "
                f"margin={decision.margin:.3f} accepted={decision.accepted} reason={decision.reason}"
            )

        def manual_command(self, intent: str) -> None:
            target = self.config.realtime_config()["target"]
            if intent == "home":
                payload = {"cmd": "home", "intent": "home", "target": target, "confidence": 1.0, "sequence_id": 0, "timestamp": time.time(), "duration_seconds": 0.0}
            elif is_grip_toggle_event(intent):
                self.grip_controller.target = target
                self.grip_controller.duration_seconds = float(self.config.command_duration_spin.value())
                command = self.grip_controller.toggle(now=time.time(), confidence=1.0, force=True)
                if command is not None:
                    self.append_log(f"手动三次眨眼：{command.name} / 状态={self.grip_controller.state_text}")
                return
            else:
                self.sequence += 1
                payload = {
                    "cmd": ACTION_COMMANDS[intent],
                    "intent": intent,
                    "target": target,
                    "confidence": 1.0,
                    "sequence_id": self.sequence,
                    "timestamp": time.time(),
                    "duration_seconds": float(self.config.command_duration_spin.value()),
                }
            self.bus.command_emitted.emit(payload)
            self.append_log(f"手动发送：{payload['cmd']} / {payload['intent']}")

        def closeEvent(self, event) -> None:
            self.stop_online()
            super().closeEvent(event)

    class LslFourEmgSource:
        def __init__(self, timeout: float = 8.0):
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
            self.using_default_montage = False
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
            self.channel_count = int(self.info.channel_count())
            self.indices = brainvision_lsl_aux_indices(self.channel_count)
            self.emg_start = self.indices[0]
            self.fs = float(self.info.nominal_srate() or 250.0)
            self.stream_name = str(self.info.name())
            self.stream_type = str(self.info.type())
            self.reported_channel_names = self._read_channel_labels()
            self.channel_names = self._mapped_channel_names(self.reported_channel_names)
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
            names = [str(name) for name in reported]
            eeg_count = min(32, self.channel_count)
            semantic = {self._channel_key(name) for name in DEFAULT_EEG_CHANNELS}
            recognized = sum(self._channel_key(name) in semantic for name in names[:eeg_count])
            if eeg_count == 32 and recognized < 8:
                names[:32] = list(DEFAULT_EEG_CHANNELS)
                self.using_default_montage = True
            for idx in range(32, min(40, len(names))):
                if self._channel_key(names[idx]) in {f"CH{idx + 1}", f"EEG{idx + 1}"}:
                    names[idx] = f"Aux{idx - 31}"
            return names

        @staticmethod
        def _channel_key(name: str) -> str:
            return "".join(char for char in str(name).upper() if char.isalnum())

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
            self.filter_check.setToolTip("使用 6 阶 Butterworth 带通滤波；关闭后显示和分类都使用原始 LSL 数据。")
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

            self.start_button = QtWidgets.QPushButton("连接 EEG + 肌电")
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

        def start_stream(self) -> None:
            if self.timer.isActive():
                return
            self.stop_stream()
            try:
                self.source = LslFourEmgSource(timeout=8.0)
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
            except Exception as exc:
                self.source = None
                QtWidgets.QMessageBox.critical(self, "LSL 连接失败", str(exc))
                self.status_label.setText(f"状态：连接失败：{exc}")
                self.append_log(f"连接失败：{exc}")

        def stop_stream(self) -> None:
            self.timer.stop()
            if self.source is not None:
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭 LSL 失败：{exc}")
            self.source = None
            self.arming_until = 0.0
            self.status_label.setText("状态：已停止")

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
            centered = filtered - np.median(filtered, axis=0, keepdims=True)
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

        def _classify(self, ratio: np.ndarray, now: float, selected: np.ndarray) -> str:
            if now < self.hold_until and self.held_result:
                return self.held_result
            blink_thr = float(self.blink_ratio_spin.value())
            bite_thr = float(self.bite_ratio_spin.value())
            side_ratio = max(1.0, float(self.side_ratio_spin.value()))
            roles = self.channel_roles()

            def role_indices(*role_keys: str) -> List[int]:
                keys = set(role_keys)
                return [idx for idx, role in enumerate(roles) if idx < len(selected) and selected[idx] and role in keys]

            def max_ratio(indices: Sequence[int]) -> float:
                if not indices:
                    return 0.0
                values = [float(ratio[idx]) for idx in indices if idx < len(ratio) and np.isfinite(ratio[idx])]
                return max(values) if values else 0.0

            eye_indices = role_indices("left_eye", "right_eye")
            left_indices = role_indices("left_masseter")
            right_indices = role_indices("right_masseter")
            eye_on = bool(eye_indices and max_ratio(eye_indices) >= blink_thr)
            if eye_on and not self.blink_active:
                if not self.blink_times or now - self.blink_times[-1] > 0.18:
                    self.blink_times.append(now)
                self.blink_active = True
            elif not eye_on:
                self.blink_active = False
            while self.blink_times and now - self.blink_times[0] > 2.5:
                self.blink_times.popleft()
            self.hit_counts["blink3"] = len(self.blink_times)
            if len(self.blink_times) >= 3:
                self.held_result = "双眼眨眼三次"
                self.hold_until = now + 1.2
                self._reset_hit_counts()
                return self.held_result

            left_face = max_ratio(left_indices)
            right_face = max_ratio(right_indices)
            left_on = bool(left_indices and left_face >= bite_thr)
            right_on = bool(right_indices and right_face >= bite_thr)

            if left_indices and right_indices and left_on and right_on:
                self.hit_counts["both_masseter"] += 1
                self.hit_counts["left_masseter"] = 0
                self.hit_counts["right_masseter"] = 0
                if self.hit_counts["both_masseter"] >= 2:
                    self.held_result = "咬两边咬肌"
                    self.hold_until = now + 1.0
                    self._reset_hit_counts()
                    return self.held_result
                return "双咬候选"
            if left_on and (not right_on) and left_face >= right_face * side_ratio:
                self.hit_counts["left_masseter"] += 1
                self.hit_counts["both_masseter"] = 0
                self.hit_counts["right_masseter"] = 0
                if self.hit_counts["left_masseter"] >= 3:
                    self.held_result = "咬左边咬肌"
                    self.hold_until = now + 1.0
                    self._reset_hit_counts()
                    return self.held_result
                return "左咬候选"
            if right_on and (not left_on) and right_face >= left_face * side_ratio:
                self.hit_counts["right_masseter"] += 1
                self.hit_counts["both_masseter"] = 0
                self.hit_counts["left_masseter"] = 0
                if self.hit_counts["right_masseter"] >= 3:
                    self.held_result = "咬右边咬肌"
                    self.hold_until = now + 1.0
                    self._reset_hit_counts()
                    return self.held_result
                return "右咬候选"
            self.hit_counts["both_masseter"] = 0
            self.hit_counts["left_masseter"] = 0
            self.hit_counts["right_masseter"] = 0
            if left_on or right_on:
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
            blink_count = len(self.blink_times)
            self._update_values(latest, rms, p2p, ratio, selected)
            self._maybe_update_plot(selected)
            self.result_label.setText(
                "结果：{result}    计数：眼={eye}/3 双咬={both}/2 左={left}/3 右={right}/3".format(
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
                if result not in {"静息/未识别", "左咬候选", "右咬候选", "双咬候选", "动作不清晰/等待稳定"}:
                    self.append_log(f"{time.strftime('%H:%M:%S')} 识别：{result} ratio={np.array2string(ratio, precision=2)}")
                    if result != "所选通道无可判定动作":
                        self.speaker.speak(result, enabled=self.speech_check.isChecked())
                self.last_result = result

        def closeEvent(self, event) -> None:
            self.stop_stream()
            self.speaker.close()
            super().closeEvent(event)

    class EmgVRPage(QtWidgets.QWidget):
        EVENT_NAMES = {
            "left_masseter": "左咬肌",
            "right_masseter": "右咬肌",
            "both_masseter": "双咬肌",
            "blink3": "眨眼三次",
        }
        DEFAULT_EVENT_INTENTS = {
            "left_masseter": "left",
            "right_masseter": "right",
            "both_masseter": "forward",
            "blink3": "backward",
        }
        DIRECTION_OPTIONS = (
            ("前", "forward"),
            ("后", "backward"),
            ("左", "left"),
            ("右", "right"),
            ("上", "up"),
            ("下", "down"),
        )
        DIRECTION_TEXT = {intent: text for text, intent in DIRECTION_OPTIONS}

        def __init__(self, config: ConfigPage, parent=None):
            super().__init__(parent)
            self.config = config
            self.bus = CommandBus()
            self.source: Optional[LslFourEmgSource] = None
            self.mi_model: Optional[BinaryMIDeploymentModel] = None
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.sequence = 0
            self.samples = np.zeros((0, 4), dtype=np.float32)
            self.timestamps = np.zeros((0,), dtype=np.float64)
            self.full_samples = np.zeros((0, 0), dtype=np.float32)
            self.full_timestamps = np.zeros((0,), dtype=np.float64)
            self.pending_hybrid_event: Optional[Dict[str, object]] = None
            self.gripper_closed = False
            self.fs = 250.0
            self.baseline_rms: Optional[np.ndarray] = None
            self.baseline_values: List[np.ndarray] = []
            self.baseline_until = 0.0
            self.arming_until = 0.0
            self.last_sample_wall = 0.0
            self.blink_times: deque = deque()
            self.blink_active = False
            self.last_event_key = ""
            self.last_command_at = 0.0
            self.last_classify_ts = 0.0
            self.waiting_for_release = False
            self.latest_ratio = np.zeros(4, dtype=np.float32)
            self.hit_counts = {
                "blink3": 0,
                "both_masseter": 0,
                "left_masseter": 0,
                "right_masseter": 0,
            }
            self.channel_checks: List[QtWidgets.QCheckBox] = []
            self.role_combos: List[QtWidgets.QComboBox] = []
            self.physical_labels: List[QtWidgets.QLabel] = []
            self.metric_channel_labels: List[QtWidgets.QLabel] = []
            self.ratio_labels: List[QtWidgets.QLabel] = []
            self.bars: List[QtWidgets.QProgressBar] = []
            self.event_mapping_combos: Dict[str, QtWidgets.QComboBox] = {}
            self.speaker = SpeechAnnouncer()

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
            self.refractory_spin = QtWidgets.QDoubleSpinBox()
            self.refractory_spin.setRange(0.05, 2.0)
            self.refractory_spin.setSingleStep(0.05)
            self.refractory_spin.setValue(0.35)
            self.duration_spin = QtWidgets.QDoubleSpinBox()
            self.duration_spin.setRange(0.05, 2.0)
            self.duration_spin.setSingleStep(0.05)
            self.duration_spin.setValue(float(self.config.command_duration_spin.value()))
            self.window_seconds_spin = QtWidgets.QDoubleSpinBox()
            self.window_seconds_spin.setRange(0.10, 2.00)
            self.window_seconds_spin.setDecimals(2)
            self.window_seconds_spin.setSingleStep(0.05)
            self.window_seconds_spin.setValue(0.30)
            self.window_seconds_spin.setSuffix(" s")
            self.release_ratio_spin = QtWidgets.QDoubleSpinBox()
            self.release_ratio_spin.setRange(1.0, 5.0)
            self.release_ratio_spin.setDecimals(2)
            self.release_ratio_spin.setSingleStep(0.05)
            self.release_ratio_spin.setValue(1.35)
            self.require_release_check = QtWidgets.QCheckBox("动作后等待静息")
            self.require_release_check.setChecked(True)
            self.require_release_check.setToolTip("发出一次命令后，必须等所有启用通道低于回静息倍数，才允许下一次分类。")
            self.speech_check = QtWidgets.QCheckBox("语音播报")
            self.speech_check.setChecked(True)
            self.speech_check.setToolTip("发出肌电 VR 指令时播报分类结果和方向。")
            self.filter_check = QtWidgets.QCheckBox("启用滤波")
            self.filter_check.setChecked(True)
            self.filter_check.setToolTip("使用 6 阶 Butterworth 带通滤波；关闭后使用原始 LSL 数据。")
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
            for spin in [
                self.baseline_spin,
                self.blink_ratio_spin,
                self.bite_ratio_spin,
                self.side_ratio_spin,
                self.refractory_spin,
                self.duration_spin,
                self.window_seconds_spin,
                self.release_ratio_spin,
            ]:
                spin.setMinimumWidth(110)

            root_layout = QtWidgets.QVBoxLayout(self)
            root_layout.setContentsMargins(24, 18, 24, 18)
            root_layout.setSpacing(10)
            title = QtWidgets.QLabel("肌电 + 群 CSP 混合控制")
            title.setObjectName("title")
            root_layout.addWidget(title)
            subtitle = QtWidgets.QLabel("左/右/双咬肌选择动作组，群 CSP 判别左/右手运动想象并输出六方向；关闭群 CSP 后可单独测试肌电。")
            subtitle.setObjectName("desc")
            subtitle.setWordWrap(True)
            root_layout.addWidget(subtitle)

            body = QtWidgets.QHBoxLayout()
            body.setSpacing(14)
            root_layout.addLayout(body, 1)

            left_scroll = QtWidgets.QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            left_scroll.setMinimumWidth(430)
            left_scroll.setMaximumWidth(500)
            left_content = QtWidgets.QWidget()
            left_layout = QtWidgets.QVBoxLayout(left_content)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(10)
            left_scroll.setWidget(left_content)
            body.addWidget(left_scroll, 0)

            connection_card = QtWidgets.QFrame()
            connection_card.setObjectName("vrControls")
            connection_layout = QtWidgets.QVBoxLayout(connection_card)
            connection_layout.setContentsMargins(12, 10, 12, 10)
            connection_layout.setSpacing(8)
            connection_title = QtWidgets.QLabel("BrainVision LSL 自动连接")
            connection_title.setObjectName("hint")
            connection_layout.addWidget(connection_title)
            button_grid = QtWidgets.QGridLayout()
            button_grid.setHorizontalSpacing(8)
            button_grid.setVerticalSpacing(8)
            self.start_button = QtWidgets.QPushButton("连接并开始")
            self.start_button.setObjectName("primaryButton")
            self.start_button.clicked.connect(self.start_stream)
            self.stop_button = QtWidgets.QPushButton("停止")
            self.stop_button.setObjectName("dangerButton")
            self.stop_button.clicked.connect(self.stop_stream)
            self.reset_baseline_button = QtWidgets.QPushButton("重置基线")
            self.reset_baseline_button.clicked.connect(self.reset_baseline)
            self.reset_scene_button = QtWidgets.QPushButton("回零")
            self.reset_scene_button.clicked.connect(self.reset_scene)
            for btn in [self.reset_baseline_button, self.reset_scene_button]:
                btn.setObjectName("vrCmd")
                btn.setMinimumHeight(34)
            self.start_button.setMinimumHeight(34)
            self.stop_button.setMinimumHeight(34)
            button_grid.addWidget(self.start_button, 0, 0)
            button_grid.addWidget(self.stop_button, 0, 1)
            button_grid.addWidget(self.reset_baseline_button, 1, 0)
            button_grid.addWidget(self.reset_scene_button, 1, 1)
            connection_layout.addLayout(button_grid)
            left_layout.addWidget(connection_card)

            model_card = QtWidgets.QFrame()
            model_card.setObjectName("vrControls")
            model_layout = QtWidgets.QGridLayout(model_card)
            model_layout.setContentsMargins(12, 10, 12, 10)
            model_layout.setHorizontalSpacing(8)
            model_layout.setVerticalSpacing(8)
            model_title = QtWidgets.QLabel("二分类 MI 解码器（优先 32 导）")
            model_title.setObjectName("sectionTitle")
            self.group_csp_check = QtWidgets.QCheckBox("启用反射群 CSP + LDA")
            self.group_csp_check.setChecked(True)
            self.group_model_edit = QtWidgets.QLineEdit(DEFAULT_GROUP_CSP32_MODEL_PATH)
            self.group_model_edit.setReadOnly(True)
            self.group_model_button = QtWidgets.QPushButton("选择")
            self.group_model_button.clicked.connect(self.pick_group_model)
            self.group_model_status = QtWidgets.QLabel("等待加载 32 导 MI 模型")
            self.group_model_status.setObjectName("hint")
            model_layout.addWidget(model_title, 0, 0, 1, 2)
            model_layout.addWidget(self.group_csp_check, 1, 0, 1, 2)
            model_layout.addWidget(self.group_model_edit, 2, 0)
            model_layout.addWidget(self.group_model_button, 2, 1)
            model_layout.addWidget(self.group_model_status, 3, 0, 1, 2)
            left_layout.addWidget(model_card)

            settings_card = QtWidgets.QFrame()
            settings_card.setObjectName("vrControls")
            settings_layout = QtWidgets.QGridLayout(settings_card)
            settings_layout.setContentsMargins(12, 10, 12, 10)
            settings_layout.setHorizontalSpacing(10)
            settings_layout.setVerticalSpacing(6)
            settings_items = [
                ("基线秒数", self.baseline_spin),
                ("眨眼阈值", self.blink_ratio_spin),
                ("咬肌阈值", self.bite_ratio_spin),
                ("左右区分", self.side_ratio_spin),
                ("冷却秒数", self.refractory_spin),
                ("保持秒数", self.duration_spin),
                ("分类窗口", self.window_seconds_spin),
                ("回静息倍数", self.release_ratio_spin),
                ("回静息门控", self.require_release_check),
                ("语音播报", self.speech_check),
                ("滤波", self.filter_check),
                ("咬肌低频", self.emg_low_spin),
                ("咬肌高频", self.emg_high_spin),
                ("眼电低频", self.eog_low_spin),
                ("眼电高频", self.eog_high_spin),
            ]
            for idx, (label_text, widget) in enumerate(settings_items):
                settings_layout.addWidget(QtWidgets.QLabel(label_text), idx, 0)
                settings_layout.addWidget(widget, idx, 1)
            left_layout.addWidget(settings_card)

            channel_card = QtWidgets.QFrame()
            channel_card.setObjectName("vrControls")
            channel_layout = QtWidgets.QGridLayout(channel_card)
            channel_layout.setContentsMargins(12, 10, 12, 10)
            channel_layout.setHorizontalSpacing(10)
            channel_layout.setVerticalSpacing(6)
            channel_layout.addWidget(QtWidgets.QLabel("物理输入"), 0, 0)
            channel_layout.addWidget(QtWidgets.QLabel("启用"), 0, 1)
            channel_layout.addWidget(QtWidgets.QLabel("当前位置/用途"), 0, 2)
            for idx, name in enumerate(EmgTestPage.CHANNEL_NAMES):
                physical = QtWidgets.QLabel(f"{name} / LSL {33 + idx}")
                physical.setObjectName("hint")
                physical.setMinimumWidth(150)
                self.physical_labels.append(physical)
                check = QtWidgets.QCheckBox()
                check.setChecked(True)
                check.toggled.connect(self._channel_config_changed)
                self.channel_checks.append(check)
                role_combo = QtWidgets.QComboBox()
                for role_text, role_key in EmgTestPage.ROLE_OPTIONS:
                    role_combo.addItem(role_text, role_key)
                role_combo.setCurrentIndex(max(0, role_combo.findData(EmgTestPage.DEFAULT_ROLES[idx])))
                role_combo.currentIndexChanged.connect(self._channel_config_changed)
                role_combo.setMinimumWidth(180)
                self.role_combos.append(role_combo)
                row = idx + 1
                channel_layout.addWidget(physical, row, 0)
                channel_layout.addWidget(check, row, 1)
                channel_layout.addWidget(role_combo, row, 2)
            left_layout.addWidget(channel_card)

            mapping_card = QtWidgets.QFrame()
            mapping_card.setObjectName("vrControls")
            mapping_layout = QtWidgets.QGridLayout(mapping_card)
            mapping_layout.setContentsMargins(12, 10, 12, 10)
            mapping_layout.setHorizontalSpacing(10)
            mapping_layout.setVerticalSpacing(6)
            mapping_layout.addWidget(QtWidgets.QLabel("纯肌电动作"), 0, 0)
            mapping_layout.addWidget(QtWidgets.QLabel("方向"), 0, 1)
            for row, (event_key, event_text) in enumerate(self.EVENT_NAMES.items(), start=1):
                combo = QtWidgets.QComboBox()
                for direction_text, intent in self.DIRECTION_OPTIONS:
                    combo.addItem(direction_text, intent)
                combo.setCurrentIndex(max(0, combo.findData(self.DEFAULT_EVENT_INTENTS[event_key])))
                combo.setMinimumWidth(120)
                self.event_mapping_combos[event_key] = combo
                mapping_layout.addWidget(QtWidgets.QLabel(event_text), row, 0)
                mapping_layout.addWidget(combo, row, 1)
            left_layout.addWidget(mapping_card)
            left_layout.addStretch(1)

            right_panel = QtWidgets.QWidget()
            right_layout = QtWidgets.QVBoxLayout(right_panel)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(10)
            body.addWidget(right_panel, 1)

            self.result_label = QtWidgets.QLabel("结果：未开始")
            self.result_label.setObjectName("vrBadge")
            right_layout.addWidget(self.result_label)
            self.status_label = QtWidgets.QLabel("状态：等待连接")
            self.status_label.setObjectName("vrStatus")
            right_layout.addWidget(self.status_label)

            self.arm = BulletOpenGLArmWidget()
            self.arm.setObjectName("vrStage")
            self.bus.command_emitted.connect(self.arm.execute_payload)
            right_layout.addWidget(self.arm, 1)

            metrics = QtWidgets.QFrame()
            metrics.setObjectName("vrControls")
            metrics_layout = QtWidgets.QGridLayout(metrics)
            metrics_layout.setContentsMargins(12, 8, 12, 8)
            metrics_layout.setHorizontalSpacing(10)
            for col, text in enumerate(["通道", "基线倍数", "强度"]):
                label = QtWidgets.QLabel(text)
                label.setObjectName("hint")
                metrics_layout.addWidget(label, 0, col)
            for idx in range(4):
                name = QtWidgets.QLabel(self._physical_name(idx))
                ratio = QtWidgets.QLabel("-")
                bar = QtWidgets.QProgressBar()
                bar.setRange(0, 1000)
                bar.setValue(0)
                self.metric_channel_labels.append(name)
                self.ratio_labels.append(ratio)
                self.bars.append(bar)
                metrics_layout.addWidget(name, idx + 1, 0)
                metrics_layout.addWidget(ratio, idx + 1, 1)
                metrics_layout.addWidget(bar, idx + 1, 2)
            right_layout.addWidget(metrics)

            manual = QtWidgets.QFrame()
            manual.setObjectName("vrControls")
            manual_layout = QtWidgets.QGridLayout(manual)
            manual_layout.setContentsMargins(12, 8, 12, 8)
            for idx, (label, intent) in enumerate(
                [("前", "forward"), ("后", "backward"), ("左", "left"), ("右", "right"), ("上", "up"), ("下", "down"), ("回零", "home")]
            ):
                btn = QtWidgets.QPushButton(label)
                btn.setObjectName("vrCmd")
                btn.clicked.connect(lambda _=False, i=intent: self.manual_command(i))
                manual_layout.addWidget(btn, idx // 4, idx % 4)
            right_layout.addWidget(manual)

            self.log = QtWidgets.QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(76)
            right_layout.addWidget(self.log)

        def append_log(self, text: str) -> None:
            self.log.append(text)

        def pick_group_model(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择 32 导或群 CSP 模型",
                self.group_model_edit.text() or os.path.join(APP_DIR, "models"),
                "Joblib Files (*.joblib)",
            )
            if path:
                self.group_model_edit.setText(path)
                self.mi_model = None
                self.group_model_status.setText("已选择，连接时加载")

        def _load_group_model(self) -> None:
            self.mi_model = None
            if not self.group_csp_check.isChecked():
                self.group_model_status.setText("纯肌电模式")
                return
            selected_path = self.group_model_edit.text().strip()
            candidates = group_csp_model_candidates(selected_path)
            errors = []
            for path in candidates:
                if not os.path.exists(path):
                    errors.append(f"{path}: 文件不存在")
                    continue
                try:
                    model = BinaryMIDeploymentModel.load(path)
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
                    continue
                self.mi_model = model
                self.group_model_edit.setText(path)
                self.group_model_status.setText(
                    f"已加载 | {model.bundle.get('model_name', model.model_type)} | "
                    f"{len(model.input_channel_names)} 通道"
                )
                if selected_path and os.path.abspath(path) != os.path.abspath(selected_path):
                    self.append_log(f"所选模型不兼容，已自动回退：{path}")
                return
            raise RuntimeError("无法加载二分类群 CSP 模型：\n" + "\n".join(errors))

        def _reset_hit_counts(self) -> None:
            self.latest_ratio = np.zeros(4, dtype=np.float32)
            for key in self.hit_counts:
                self.hit_counts[key] = 0
            self.blink_times.clear()
            self.blink_active = False
            self.last_event_key = ""

        def current_window_seconds(self) -> float:
            if hasattr(self, "window_seconds_spin"):
                return max(0.05, float(self.window_seconds_spin.value()))
            return 0.30

        def _role_key(self, idx: int) -> str:
            if idx >= len(self.role_combos):
                return EmgTestPage.DEFAULT_ROLES[idx] if idx < len(EmgTestPage.DEFAULT_ROLES) else "display"
            value = self.role_combos[idx].currentData()
            return str(value) if value is not None else "display"

        def channel_roles(self) -> List[str]:
            return [self._role_key(idx) for idx in range(4)]

        def selected_mask(self) -> np.ndarray:
            roles = self.channel_roles()
            return np.asarray(
                [check.isChecked() and roles[idx] != "unused" for idx, check in enumerate(self.channel_checks)],
                dtype=bool,
            )

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
                if hasattr(self, "result_label"):
                    self.result_label.setText("结果：滤波设置已变化，重新采集基线中")
                if hasattr(self, "log"):
                    self.append_log("滤波设置已变化，重新采集基线。")

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

        def _physical_name(self, idx: int) -> str:
            if self.source is not None and idx < len(self.source.indices):
                lsl_ch = self.source.indices[idx] + 1
                aux_ch = lsl_ch - 32 if self.source.channel_count >= 36 else idx + 1
                role_text = EmgTestPage.ROLE_DISPLAY.get(self._role_key(idx), "仅显示")
                return f"Aux{aux_ch} / LSL {lsl_ch} -> {role_text}"
            role_text = EmgTestPage.ROLE_DISPLAY.get(self._role_key(idx), "仅显示")
            return f"Aux{idx + 1} / LSL {33 + idx} -> {role_text}"

        def _channel_config_changed(self, *_args) -> None:
            for idx, label in enumerate(self.physical_labels):
                label.setText(self._physical_name(idx))
            for idx, label in enumerate(self.metric_channel_labels):
                label.setText(self._physical_name(idx))
            self._reset_hit_counts()
            if self.baseline_rms is not None:
                self.reset_baseline()

        def start_stream(self) -> None:
            if self.timer.isActive():
                return
            if not np.any(self.selected_mask()):
                QtWidgets.QMessageBox.warning(self, "未选择通道", "请至少启用 1 个通道。")
                return
            self.stop_stream()
            try:
                self.source = LslFourEmgSource(timeout=8.0)
                self.fs = max(1.0, float(self.source.fs))
                model_error = None
                try:
                    self._load_group_model()
                except Exception as exc:
                    self.mi_model = None
                    model_error = str(exc)
                    self.group_model_status.setText("模型不可用 | 当前仅肌电")
                    self.append_log(f"MI 模型加载失败，LSL 保持连接：{model_error}")
                self.samples = np.zeros((0, 4), dtype=np.float32)
                self.timestamps = np.zeros((0,), dtype=np.float64)
                self.full_samples = np.zeros((0, self.source.channel_count), dtype=np.float32)
                self.full_timestamps = np.zeros((0,), dtype=np.float64)
                self.pending_hybrid_event = None
                self.baseline_values = []
                self.baseline_rms = None
                self.blink_times.clear()
                self.blink_active = False
                self._reset_hit_counts()
                self.last_command_at = 0.0
                self.last_classify_ts = 0.0
                self.waiting_for_release = False
                self.last_sample_wall = time.time()
                self.arming_until = time.time() + 2.0
                for idx, label in enumerate(self.physical_labels):
                    label.setText(self._physical_name(idx))
                for idx, label in enumerate(self.metric_channel_labels):
                    label.setText(self._physical_name(idx))
                self.timer.start(40)
                self.status_label.setText(f"状态：已连接，2 秒后开始 | {self.source.description}")
                self.result_label.setText("提示：2.0 秒后开始采集，请保持放松")
                self.append_log(f"LSL 已连接：{self.source.description}")
                if model_error:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "LSL 已连接，MI 模型未加载",
                        model_error + "\n\n当前仍可接收 EEG+肌电数据，并以纯肌电模式测试。",
                    )
                if self.source.using_default_montage:
                    self.append_log("注意：LSL 未提供标准电极名，群 CSP 使用系统默认 32 导顺序映射。")
            except Exception as exc:
                self.source = None
                QtWidgets.QMessageBox.critical(self, "LSL 连接失败", str(exc))
                self.status_label.setText(f"状态：连接失败：{exc}")
                self.append_log(f"连接失败：{exc}")

        def stop_stream(self) -> None:
            self.timer.stop()
            if self.source is not None:
                try:
                    self.source.close()
                except Exception as exc:
                    self.append_log(f"关闭 LSL 失败：{exc}")
            self.source = None
            self.arming_until = 0.0
            self.pending_hybrid_event = None
            self.status_label.setText("状态：已停止")

        def reset_baseline(self) -> None:
            if not np.any(self.selected_mask()):
                QtWidgets.QMessageBox.warning(self, "未选择通道", "请至少启用 1 个通道。")
                return
            self.baseline_values = []
            self.baseline_rms = None
            self.baseline_until = time.time() + float(self.baseline_spin.value())
            self.blink_times.clear()
            self.blink_active = False
            self._reset_hit_counts()
            self.last_classify_ts = 0.0
            self.waiting_for_release = False
            self.pending_hybrid_event = None
            self.result_label.setText("结果：采集基线中，请保持放松")
            self.append_log("开始采集肌电 VR 基线。")

        def reset_scene(self) -> None:
            self.manual_command("home")

        def _append_samples(self, chunk: np.ndarray, ts: np.ndarray) -> None:
            max_rows = max(250, int(round(self.fs * 8.0)))
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

        def _append_full_samples(self, chunk: np.ndarray, ts: np.ndarray) -> None:
            chunk_arr = np.asarray(chunk, dtype=np.float32)
            ts_arr = np.asarray(ts, dtype=np.float64)
            max_rows = max(250, int(round(self.fs * 8.0)))
            if chunk_arr.shape[0] > max_rows:
                chunk_arr = chunk_arr[-max_rows:, :]
                ts_arr = ts_arr[-max_rows:]
            keep_rows = max(0, max_rows - chunk_arr.shape[0])
            if keep_rows and self.full_samples.shape[0] > keep_rows:
                self.full_samples = self.full_samples[-keep_rows:, :]
                self.full_timestamps = self.full_timestamps[-keep_rows:]
            if self.full_samples.size:
                self.full_samples = np.vstack([self.full_samples, chunk_arr])
                self.full_timestamps = np.concatenate([self.full_timestamps, ts_arr])
            else:
                self.full_samples = chunk_arr
                self.full_timestamps = ts_arr

        def _feature_window(self) -> Optional[np.ndarray]:
            if self.samples.shape[0] < 8:
                return None
            rows = max(8, int(round(self.fs * self.current_window_seconds())))
            return self.samples[-min(rows, self.samples.shape[0]) :, :]

        def _compute_features(self, window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            filtered = self._filtered_samples(window)
            centered = filtered - np.median(filtered, axis=0, keepdims=True)
            rms = np.sqrt(np.mean(np.square(centered), axis=0))
            latest = filtered[-1, :]
            return latest, rms

        def _update_baseline(self, rms: np.ndarray, now: float) -> bool:
            if self.baseline_rms is None:
                self.baseline_values.append(rms)
                if now < self.baseline_until:
                    return False
                baseline = np.median(np.vstack(self.baseline_values), axis=0)
                self.baseline_rms = np.maximum(baseline, 1e-6)
                self.append_log(f"基线完成 RMS={np.array2string(self.baseline_rms, precision=3)}")
            return True

        def _role_indices(self, selected: np.ndarray, *role_keys: str) -> List[int]:
            keys = set(role_keys)
            roles = self.channel_roles()
            return [idx for idx, role in enumerate(roles) if idx < len(selected) and selected[idx] and role in keys]

        def _max_ratio(self, ratio: np.ndarray, indices: Sequence[int]) -> float:
            if not indices:
                return 0.0
            values = [float(ratio[idx]) for idx in indices if idx < len(ratio) and np.isfinite(ratio[idx])]
            return max(values) if values else 0.0

        def _classify_event(self, ratio: np.ndarray, now: float, selected: np.ndarray) -> Tuple[str, str, float]:
            blink_thr = float(self.blink_ratio_spin.value())
            bite_thr = float(self.bite_ratio_spin.value())
            quiet_thr = float(self.release_ratio_spin.value()) if hasattr(self, "release_ratio_spin") else 1.35
            side_ratio = float(self.side_ratio_spin.value()) if hasattr(self, "side_ratio_spin") else 1.35
            side_ratio = max(1.0, side_ratio)
            left_eye_indices = self._role_indices(selected, "left_eye")
            right_eye_indices = self._role_indices(selected, "right_eye")
            left_indices = self._role_indices(selected, "left_masseter")
            right_indices = self._role_indices(selected, "right_masseter")

            left_eye_strength = self._max_ratio(ratio, left_eye_indices)
            right_eye_strength = self._max_ratio(ratio, right_eye_indices)
            left_strength = self._max_ratio(ratio, left_indices)
            right_strength = self._max_ratio(ratio, right_indices)

            eye_pair_on = bool(
                left_eye_indices
                and right_eye_indices
                and left_eye_strength >= blink_thr
                and right_eye_strength >= blink_thr
            )
            left_on = bool(left_indices and left_strength >= bite_thr)
            right_on = bool(right_indices and right_strength >= bite_thr)
            left_quiet = bool((not left_indices) or left_strength <= quiet_thr)
            right_quiet = bool((not right_indices) or right_strength <= quiet_thr)
            left_dominant = bool(left_on and right_quiet and left_strength >= right_strength * side_ratio)
            right_dominant = bool(right_on and left_quiet and right_strength >= left_strength * side_ratio)

            if eye_pair_on and not self.blink_active:
                self.blink_times.append(now)
                self.blink_active = True
            elif not eye_pair_on:
                self.blink_active = False
            while self.blink_times and now - float(self.blink_times[0]) > 2.5:
                self.blink_times.popleft()
            self.hit_counts["blink3"] = len(self.blink_times)

            masseter_state = "静息/未识别"
            if left_strength > quiet_thr or right_strength > quiet_thr:
                masseter_state = "动作不清晰/等待稳定"

            if left_indices and right_indices and left_on and right_on:
                self.hit_counts["both_masseter"] = self.hit_counts.get("both_masseter", 0) + 1
                self.hit_counts["left_masseter"] = 0
                self.hit_counts["right_masseter"] = 0
                masseter_state = "双咬候选"
            elif left_dominant:
                self.hit_counts["left_masseter"] = self.hit_counts.get("left_masseter", 0) + 1
                self.hit_counts["both_masseter"] = 0
                self.hit_counts["right_masseter"] = 0
                masseter_state = "左咬候选"
            elif right_dominant:
                self.hit_counts["right_masseter"] = self.hit_counts.get("right_masseter", 0) + 1
                self.hit_counts["both_masseter"] = 0
                self.hit_counts["left_masseter"] = 0
                masseter_state = "右咬候选"
            else:
                # Ambiguous windows, such as one side above action threshold while the
                # other side is also elevated, are ignored instead of sent as left/right.
                self.hit_counts["both_masseter"] = 0
                self.hit_counts["left_masseter"] = 0
                self.hit_counts["right_masseter"] = 0
                if not eye_pair_on:
                    masseter_state = "静息/未识别" if left_quiet and right_quiet else "动作不清晰/等待稳定"

            if not eye_pair_on and not self.blink_times:
                self.hit_counts["blink3"] = 0

            event_specs = (
                ("blink3", 3, "双眼达标三次", max(left_eye_strength, right_eye_strength)),
                ("both_masseter", 2, "双咬肌达标两次", max(left_strength, right_strength)),
                ("left_masseter", 3, "左咬肌达标三次", left_strength),
                ("right_masseter", 3, "右咬肌达标三次", right_strength),
            )
            for key, required_count, text, score in event_specs:
                if self.hit_counts.get(key, 0) >= required_count:
                    if key == "blink3":
                        self.blink_times.clear()
                        self.blink_active = False
                    self._reset_hit_counts()
                    return key, text, score

            best_score = max(left_eye_strength, right_eye_strength, left_strength, right_strength)
            return "", masseter_state, best_score

        def _intent_for_event(self, event_key: str) -> str:
            combo = self.event_mapping_combos.get(event_key)
            if combo is None:
                return self.DEFAULT_EVENT_INTENTS.get(event_key, "forward")
            value = combo.currentData()
            return str(value) if value is not None else self.DEFAULT_EVENT_INTENTS.get(event_key, "forward")

        def _emit_intent(self, intent: str, confidence: float, now: float) -> None:
            self.sequence += 1
            payload = {
                "cmd": ACTION_COMMANDS[intent],
                "intent": intent,
                "target": self.config.realtime_config()["target"],
                "confidence": float(confidence),
                "sequence_id": self.sequence,
                "timestamp": now,
                "duration_seconds": float(self.duration_spin.value()),
            }
            self.bus.command_emitted.emit(payload)
            self.last_command_at = now

        def _toggle_gripper(self, event_text: str, score: float, now: float) -> None:
            intent = "release" if self.gripper_closed else "grab"
            self.gripper_closed = not self.gripper_closed
            threshold = float(self.blink_ratio_spin.value())
            confidence = min(1.0, max(0.0, score / max(threshold * 1.8, 1e-6)))
            self._emit_intent(intent, confidence, now)
            action_text = "松开" if intent == "release" else "抓取"
            self.append_log(f"{time.strftime('%H:%M:%S')} {event_text} -> {action_text}")
            self.speaker.speak(action_text, enabled=self.speech_check.isChecked())
            if self.require_release_check.isChecked():
                self.waiting_for_release = True

        def _queue_hybrid_event(
            self,
            event_key: str,
            event_text: str,
            score: float,
            lsl_timestamp: float,
        ) -> None:
            required = 2 if event_key == "both_masseter" else 3
            onset = float(lsl_timestamp) - self.current_window_seconds() * max(0, required - 1)
            self.pending_hybrid_event = {
                "event_key": event_key,
                "event_text": event_text,
                "score": float(score),
                "onset": onset,
            }
            self.result_label.setText(f"结果：{event_text}，正在采集运动想象窗口")

        def _try_complete_hybrid_event(self, latest_ts: float, now: float) -> bool:
            pending = self.pending_hybrid_event
            if pending is None or self.mi_model is None or self.source is None:
                return False
            duration = self.mi_model.window_samples / float(self.mi_model.fs) if self.mi_model.window_samples else 3.0
            onset = float(pending["onset"])
            elapsed = latest_ts - onset
            if elapsed < duration:
                self.result_label.setText(
                    f"结果：{pending['event_text']}，MI 采集中 {max(0.0, elapsed):.1f}/{duration:.1f}s"
                )
                return True
            try:
                eeg_epoch = extract_timestamped_epochs(
                    self.full_samples,
                    self.full_timestamps,
                    [onset],
                    self.fs,
                    duration,
                )[0]
                intent, mi_result = self.mi_model.predict_intent(
                    eeg_epoch,
                    self.source.channel_names,
                    self.fs,
                    str(pending["event_key"]),
                )
            except Exception as exc:
                self.append_log(f"群 CSP 分类失败：{exc}")
                self.result_label.setText(f"结果：群 CSP 分类失败：{exc}")
                self.pending_hybrid_event = None
                return True
            threshold = float(self.bite_ratio_spin.value())
            emg_confidence = min(1.0, max(0.0, float(pending["score"]) / max(threshold * 1.8, 1e-6)))
            confidence = 0.5 * emg_confidence + 0.5 * mi_result.confidence
            self._emit_intent(intent, confidence, now)
            self.last_event_key = str(pending["event_key"])
            if self.require_release_check.isChecked():
                self.waiting_for_release = True
            direction_text = self.DIRECTION_TEXT.get(intent, intent)
            hand_text = "左手想象" if mi_result.hand == "left_hand_mi" else "右手想象"
            self.result_label.setText(
                f"结果：{pending['event_text']} + {hand_text} -> {direction_text} | MI {mi_result.confidence:.0%}"
            )
            self.append_log(
                f"{time.strftime('%H:%M:%S')} {pending['event_text']} + {hand_text} -> {direction_text} "
                f"mi={mi_result.confidence:.3f} fused={confidence:.3f}"
            )
            self.speaker.speak(
                f"{hand_text}，向{direction_text}",
                enabled=self.speech_check.isChecked(),
            )
            self.pending_hybrid_event = None
            return True

        def _send_event_command(
            self,
            event_key: str,
            event_text: str,
            score: float,
            now: float,
            lsl_timestamp: Optional[float] = None,
        ) -> None:
            if not event_key:
                return
            refractory = float(self.refractory_spin.value())
            if now - self.last_command_at < refractory:
                return
            if self.group_csp_check.isChecked() and self.mi_model is not None:
                if event_key == "blink3":
                    self._toggle_gripper(event_text, score, now)
                elif self.pending_hybrid_event is None:
                    self._queue_hybrid_event(event_key, event_text, score, lsl_timestamp or now)
                return
            intent = self._intent_for_event(event_key)
            threshold = float(self.blink_ratio_spin.value()) if event_key == "blink3" else float(self.bite_ratio_spin.value())
            confidence = min(1.0, max(0.0, score / max(threshold * 1.8, 1e-6)))
            self._emit_intent(intent, confidence, now)
            self.last_event_key = event_key
            if self.require_release_check.isChecked():
                self.waiting_for_release = True
            direction_text = self.DIRECTION_TEXT.get(intent, intent)
            self.append_log(f"{time.strftime('%H:%M:%S')} {event_text} -> {direction_text} score={score:.2f}")
            self.speaker.speak(f"{event_text}，向{direction_text}", enabled=self.speech_check.isChecked())

        def _update_values(self, ratio: np.ndarray, selected: np.ndarray) -> None:
            for idx in range(4):
                if idx >= len(selected) or not selected[idx]:
                    self.ratio_labels[idx].setText("未选")
                    self.bars[idx].setValue(0)
                    continue
                value = float(ratio[idx]) if idx < len(ratio) and np.isfinite(ratio[idx]) else 0.0
                self.ratio_labels[idx].setText(f"{value:.2f}x")
                self.bars[idx].setValue(int(min(1000, max(0, value * 120))))

        def tick(self) -> None:
            if self.source is None:
                self.stop_stream()
                return
            selected = self.selected_mask()
            if not np.any(selected):
                self.result_label.setText("结果：未选择通道")
                return
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
                    self.result_label.setText(f"提示：{remaining:.1f} 秒后开始采集，请保持放松")
                    if self.source is not None:
                        self.status_label.setText(f"状态：已连接，等待开始 | {self.source.description}")
                    return
                self.arming_until = 0.0
                self.samples = np.zeros((0, 4), dtype=np.float32)
                self.timestamps = np.zeros((0,), dtype=np.float64)
                self.full_samples = np.zeros((0, self.source.channel_count), dtype=np.float32)
                self.full_timestamps = np.zeros((0,), dtype=np.float64)
                self.last_sample_wall = now
                self.reset_baseline()
                return
            if chunk is not None and ts is not None:
                self._append_samples(chunk, ts)
                if self.source.last_full_samples is not None and self.source.last_full_timestamps is not None:
                    self._append_full_samples(
                        self.source.last_full_samples,
                        self.source.last_full_timestamps,
                    )
                self.last_sample_wall = now

            if self.timestamps.shape[0] < 2:
                return
            latest_ts = float(self.timestamps[-1])
            window_seconds = self.current_window_seconds()
            if self.last_classify_ts > 0.0 and latest_ts - self.last_classify_ts < window_seconds:
                if now - self.last_sample_wall > 2.0:
                    self.status_label.setText("状态：已连接，但暂时没有收到样本")
                return

            window = self._feature_window()
            if window is None:
                if now - self.last_sample_wall > 2.0:
                    self.status_label.setText("状态：已连接，但暂时没有收到样本")
                return
            self.last_classify_ts = latest_ts
            _latest, rms = self._compute_features(window)
            if not self._update_baseline(rms, now):
                ratio = np.zeros(4, dtype=np.float32)
                self.latest_ratio = ratio.copy()
                self._update_values(ratio, selected)
                self.result_label.setText("结果：采集基线中，请保持放松")
                return
            ratio = rms / np.maximum(self.baseline_rms, 1e-6)
            self.latest_ratio = np.asarray(ratio, dtype=np.float32).reshape(4)
            self._update_values(ratio, selected)

            if self.pending_hybrid_event is not None:
                self._try_complete_hybrid_event(latest_ts, now)
                self.status_label.setText(
                    f"状态：群 CSP 混合控制中 | EMG {window_seconds:.2f}s窗 | {self.source.description}"
                )
                return

            if self.waiting_for_release and self.require_release_check.isChecked():
                selected_values = self.latest_ratio[selected] if np.any(selected) else self.latest_ratio
                release_ok = bool(selected_values.size and np.all(selected_values < float(self.release_ratio_spin.value())))
                if release_ok:
                    self.waiting_for_release = False
                    self._reset_hit_counts()
                    self.result_label.setText("结果：已回静息，可以识别下一次动作")
                else:
                    self._reset_hit_counts()
                    self.result_label.setText(
                        "结果：等待回静息    倍数={values}".format(
                            values=np.array2string(self.latest_ratio, precision=2)
                        )
                    )
                if self.source is not None:
                    self.status_label.setText(
                        f"状态：纯肌电 VR 实时接收中 | 每 {window_seconds:.2f}s 分类一次 | {self.source.description}"
                    )
                return

            event_key, event_text, score = self._classify_event(ratio, now, selected)
            if event_key:
                if self.group_csp_check.isChecked() and self.mi_model is not None:
                    if event_key == "blink3":
                        self.result_label.setText(f"结果：{event_text} -> 抓/放切换")
                    else:
                        self.result_label.setText(f"结果：{event_text}，准备群 CSP 分类")
                else:
                    intent = self._intent_for_event(event_key)
                    self.result_label.setText(f"结果：{event_text} -> {self.DIRECTION_TEXT.get(intent, intent)}")
                self._send_event_command(event_key, event_text, score, now, latest_ts)
            else:
                self.result_label.setText(
                    "结果：{text}    计数：眼={eye}/3 双咬={both}/2 左={left}/3 右={right}/3".format(
                        text=event_text,
                        eye=self.hit_counts.get("blink3", 0),
                        both=self.hit_counts.get("both_masseter", 0),
                        left=self.hit_counts.get("left_masseter", 0),
                        right=self.hit_counts.get("right_masseter", 0),
                    )
                )
            if self.source is not None:
                mode_text = "群 CSP 混合控制" if self.group_csp_check.isChecked() and self.mi_model is not None else "纯肌电控制"
                self.status_label.setText(
                    f"状态：{mode_text}中 | 每 {window_seconds:.2f}s 检测一次 | {self.source.description}"
                )

        def manual_command(self, intent: str) -> None:
            if intent == "home":
                payload = {
                    "cmd": "home",
                    "intent": "home",
                    "target": self.config.realtime_config()["target"],
                    "confidence": 1.0,
                    "sequence_id": 0,
                    "timestamp": time.time(),
                    "duration_seconds": 0.0,
                }
            else:
                self.sequence += 1
                payload = {
                    "cmd": ACTION_COMMANDS[intent],
                    "intent": intent,
                    "target": self.config.realtime_config()["target"],
                    "confidence": 1.0,
                    "sequence_id": self.sequence,
                    "timestamp": time.time(),
                    "duration_seconds": float(self.duration_spin.value()),
                }
            self.bus.command_emitted.emit(payload)
            self.append_log(f"手动发送：{payload['cmd']} / {payload['intent']}")

        def closeEvent(self, event) -> None:
            self.stop_stream()
            self.speaker.close()
            super().closeEvent(event)

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("机械臂应用赛 BCI 闭环系统")
            self.resize(1680, 1020)
            self.setMinimumSize(1280, 800)
            self.bus = CommandBus()
            self.page_names = ["系统设置", "EEG/肌电监测", "混合控制", "范式采集", "模型训练", "机械臂仿真"]
            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)

            sidebar = QtWidgets.QFrame()
            sidebar.setObjectName("sidebar")
            side_layout = QtWidgets.QVBoxLayout(sidebar)
            side_layout.setContentsMargins(16, 18, 16, 18)
            brand = QtWidgets.QLabel("BCI Arm")
            brand.setObjectName("brand")
            side_layout.addWidget(brand)
            sub = QtWidgets.QLabel("EEG + EMG 闭环机械臂")
            sub.setObjectName("subbrand")
            side_layout.addWidget(sub)
            self.buttons = []
            for idx, name in enumerate(self.page_names):
                btn = QtWidgets.QPushButton(name)
                btn.setObjectName("nav")
                btn.clicked.connect(lambda _=False, i=idx: self.switch_page(i))
                side_layout.addWidget(btn)
                self.buttons.append(btn)
            side_layout.addStretch(1)
            workflow = QtWidgets.QLabel("工作流\n01 连接信号\n02 范式采集\n03 训练/加载模型\n04 混合控制")
            workflow.setObjectName("workflow")
            side_layout.addWidget(workflow)
            hint = QtWidgets.QLabel("LSL 自动发现 LiveAmp。采集文件同时保存 EEG、AUX/EMG、原始流和时间戳。")
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
            self.app_status = QtWidgets.QLabel("SYSTEM READY")
            self.app_status.setObjectName("appStatus")
            topbar_layout.addWidget(self.page_title)
            topbar_layout.addStretch(1)
            topbar_layout.addWidget(self.app_status)
            content_layout.addWidget(topbar)

            self.stack = QtWidgets.QStackedWidget()
            self.config_page = ConfigPage()
            self.emg_test_page = EmgTestPage()
            self.emg_vr_page = EmgVRPage(self.config_page)
            self.collection_page = CollectionPage(self.config_page)
            self.train_page = TrainTunePage(self.config_page)
            self.vr_page = VRDemoPage(self.bus, self.config_page)
            self.collection_page.recording_saved.connect(self.train_page.recording_added)
            self.train_page.model_ready.connect(self.config_page.set_model_path)
            self.train_page.model_ready.connect(self.vr_page.set_model_path)
            for page in [self.config_page, self.emg_test_page, self.emg_vr_page, self.collection_page, self.train_page, self.vr_page]:
                self.stack.addWidget(page)
            content_layout.addWidget(self.stack, 1)
            layout.addWidget(content_shell, 1)
            self.switch_page(0)
            self.apply_style()

        def switch_page(self, index: int) -> None:
            self.stack.setCurrentIndex(index)
            self.page_title.setText(self.page_names[index])
            for i, btn in enumerate(self.buttons):
                btn.setProperty("active", i == index)
                btn.style().unpolish(btn)
                btn.style().polish(btn)

        def apply_style(self) -> None:
            self.setStyleSheet(
                """
                QWidget { background: #101419; color: #edf2f5; font-family: "Microsoft YaHei UI", "Segoe UI", Arial; font-size: 14px; }
                #sidebar { background: #171c22; border-right: 1px solid #303842; min-width: 206px; max-width: 206px; }
                #brand { font-size: 25px; font-weight: 700; color: #ffffff; }
                #subbrand, #desc, #hint { color: #9eabb6; }
                #hint { font-size: 12px; }
                #workflow { color: #c2ccd4; background: #20262d; border: 1px solid #343d46; border-radius: 6px; padding: 10px; line-height: 1.6; }
                #topbar { background: #14191f; border-bottom: 1px solid #2d353e; min-height: 46px; max-height: 46px; }
                #pageTitle { font-size: 16px; font-weight: 700; color: #f7f9fa; }
                #appStatus { color: #8ee3b2; background: #183126; border: 1px solid #28533c; border-radius: 5px; padding: 5px 10px; font-size: 11px; font-weight: 700; }
                #title { font-size: 26px; font-weight: 700; color: #ffffff; margin: 5px 0; }
                #sectionTitle { font-size: 15px; font-weight: 700; color: #f3f6f8; }
                QPushButton#nav { text-align: left; min-height: 42px; padding: 0 13px; border-radius: 6px; border: 1px solid transparent; background: transparent; color: #adb8c2; }
                QPushButton#nav:hover { background: #222930; color: #f5f7f8; }
                QPushButton#nav[active="true"] { background: #25323b; color: #ffffff; border-color: #3b5969; border-left: 3px solid #4fc3dc; }
                QPushButton { background: #29323a; border: 1px solid #46515c; color: #f7f9fa; padding: 8px 13px; border-radius: 6px; min-height: 20px; }
                QPushButton:hover { background: #35414b; border-color: #65727e; }
                QPushButton#primaryButton { background: #087fa3; border-color: #25a6c8; font-weight: 700; }
                QPushButton#primaryButton:hover { background: #1195bb; }
                QPushButton#dangerButton { background: #622d35; border-color: #98505b; }
                QPushButton#dangerButton:hover { background: #7a3641; }
                QPushButton:disabled { background: #252b31; color: #707a83; border-color: #343b42; }
                QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #0d1115; border: 1px solid #3a444e; border-radius: 5px; padding: 7px 9px; color: #edf2f5; selection-background-color: #167f9f; }
                QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #45b6cf; }
                QCheckBox { color: #edf2f5; spacing: 8px; }
                QCheckBox::indicator { width: 16px; height: 16px; }
                #card { background: #191f25; border: 1px solid #343d46; border-radius: 7px; padding: 10px; }
                #paradigmStage { background: #0b1014; border: 1px solid #356072; border-radius: 7px; }
                #cueWord { font-size: 44px; font-weight: 800; color: #f7fbff; }
                #phaseInstruction { font-size: 18px; color: #bfe8ff; }
                #phaseBadge { background: #182933; border: 1px solid #356072; border-radius: 5px; color: #d9f2ff; padding: 6px 12px; }
                #countdown { font-size: 24px; font-weight: 800; color: #ffffff; background: transparent; }
                QProgressBar { background: #0d1115; border: 1px solid #38434d; border-radius: 5px; min-height: 13px; text-align: center; }
                QProgressBar::chunk { background: #1e9fc1; border-radius: 4px; }
                QScrollArea { background: transparent; border: 0; }
                #vrTitle { font-size: 20px; font-weight: 700; color: #f4f8fb; }
                #vrBadge { background: #17303a; border: 1px solid #376579; border-radius: 5px; color: #c8edf5; padding: 7px 12px; }
                #vrStage { background: #091016; border: 1px solid #2f414d; border-radius: 7px; }
                #vrControls { background: #191f25; border: 1px solid #343d46; border-radius: 7px; }
                QPushButton#vrCmd { background: #28414d; border: 1px solid #3f7185; border-radius: 5px; min-height: 32px; font-size: 14px; font-weight: 600; }
                QPushButton#vrCmd:hover { background: #315464; }
                #vrStatus { color: #a9d7ef; background: transparent; padding: 4px 8px; }
                QTextEdit { font-family: Consolas, "Cascadia Mono"; }
                QToolTip { background: #f2f5f7; color: #172027; border: 1px solid #83909b; padding: 5px; }
                """
            )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
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


