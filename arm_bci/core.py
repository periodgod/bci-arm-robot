# -*- coding: utf-8 -*-
"""
Hybrid masseter-plus-MI system for robotic arm direct control.

Main workflow:
  1. Collect calibration trials for rest plus six movement commands.
  2. Train a masseter-plus-MI filter-bank decoder with optional virtual/real auxiliary signals.
  3. Run asynchronous online decoding with false-trigger suppression.
  4. Toggle gripper state from a triple-blink event, then send guarded robot-arm commands.

Expected calibration .npz format:
  eeg: (trials, eeg_channels, samples)
  y:   (trials,), labels as rest plus 6 movement action names, numeric action ids, or Chinese task names
  aux: optional (trials, aux_channels, samples), virtual/real masseter EMG/EOG/etc.
  fs:  optional scalar sampling rate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence, Tuple

# Directly launching a Conda environment's python.exe does not always add
# Library/bin to the Windows DLL search path. NumPy/SciPy linear algebra then
# fails at the native-DLL level instead of raising a catchable Python error.
_DLL_DIRECTORY_HANDLES = []
if sys.platform.startswith("win"):
    _conda_dll_dir = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
    if os.path.isdir(_conda_dll_dir):
        os.environ["PATH"] = _conda_dll_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(_conda_dll_dir))
        except (AttributeError, OSError):
            pass

import joblib
import numpy as np

try:
    from scipy.signal import butter, iirnotch, resample_poly, sosfilt, sosfiltfilt, filtfilt

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    from mne.decoding import CSP

    HAS_MNE_CSP = True
except Exception:
    CSP = None
    HAS_MNE_CSP = False

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(APP_DIR, "data", "recordings")
DEFAULT_GROUP_CSP_MODEL_PATH = os.path.join(APP_DIR, "models", "group_csp_left_right_mi.joblib")
DEFAULT_GROUP_CSP32_MODEL_PATH = os.path.join(APP_DIR, "models", "group_csp_32ch_left_right_mi.joblib")
LABELS = ("up", "down", "left", "right", "forward", "backward")
REST_LABEL = "rest"
MODEL_LABELS = (REST_LABEL,) + LABELS
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(MODEL_LABELS)}
GRIP_TOGGLE_LABEL = "blink3"
GRIP_TOGGLE_ALIASES = {"blink3", "triple_blink", "triple-blink", "blink_toggle", "三次眨眼", "眨眼三次", "眨眼", "blink"}
PARADIGM_NAME = "MI6_masseter3xMI2_blink_grip_toggle"

BINARY_MI_FUSION_MAP = {
    ("right_masseter", "left_hand_mi"): "forward",
    ("right_masseter", "right_hand_mi"): "backward",
    ("left_masseter", "left_hand_mi"): "left",
    ("left_masseter", "right_hand_mi"): "right",
    ("both_masseter", "left_hand_mi"): "up",
    ("both_masseter", "right_hand_mi"): "down",
}

LABEL_ALIASES = {
    "0": "up",
    "up": "up",
    "raise": "up",
    "上": "up",
    "向上": "up",
    "上升": "up",
    "1": "down",
    "down": "down",
    "lower": "down",
    "下": "down",
    "向下": "down",
    "下降": "down",
    "2": "left",
    "left": "left",
    "左": "left",
    "向左": "left",
    "左移": "left",
    "3": "right",
    "right": "right",
    "右": "right",
    "向右": "right",
    "右移": "right",
    "4": "forward",
    "forward": "forward",
    "front": "forward",
    "前": "forward",
    "向前": "forward",
    "前进": "forward",
    "5": "backward",
    "backward": "backward",
    "back": "backward",
    "后": "backward",
    "向后": "backward",
    "后退": "backward",
    "6": "rest",
    "rest": "rest",
    "idle": "rest",
    "relax": "rest",
    "休息": "rest",
    "静息": "rest",
    "空闲": "rest",
}

ACTION_COMMANDS = {
    "up": "move_up",
    "down": "move_down",
    "left": "move_left",
    "right": "move_right",
    "forward": "move_forward",
    "backward": "move_backward",
    "grab": "close_gripper",
    "release": "open_gripper",
}

MI8_TRIGGER_CODES = {
    "up": 11,
    "down": 12,
    "left": 13,
    "right": 14,
    "forward": 15,
    "backward": 16,
}
REST_TRIGGER_CODE = 10
GRIP_TOGGLE_TRIGGER_CODE = 19
MODEL_TRIGGER_CODES = {REST_LABEL: REST_TRIGGER_CODE, **MI8_TRIGGER_CODES}
TRIGGER_TO_LABEL = {code: label for label, code in MODEL_TRIGGER_CODES.items()}
LABEL_DISPLAY_NAMES = {
    "rest": "休息",
    "up": "上",
    "down": "下",
    "left": "左",
    "right": "右",
    "forward": "前",
    "backward": "后",
    "grab": "抓",
    "release": "放",
    GRIP_TOGGLE_LABEL: "眨眼三次",
}

MI6_PARADIGM = {
    "forward": {
        "action": "前进",
        "modality": "咬肌+MI",
        "region": "右侧咬肌 + C4",
        "masseter_task": "咬住右边后槽牙",
        "mi_task": "想象左手握拳/用力",
        "subject_task": "咬住右边，同时想象左手握拳；对应机械臂前进",
        "feature": "右咬肌选择前后轴，左手 MI 选择前进方向",
    },
    "backward": {
        "action": "后退",
        "modality": "咬肌+MI",
        "region": "右侧咬肌 + C3",
        "masseter_task": "咬住右边后槽牙",
        "mi_task": "想象右手握拳/用力",
        "subject_task": "咬住右边，同时想象右手握拳；对应机械臂后退",
        "feature": "右咬肌选择前后轴，右手 MI 选择后退方向",
    },
    "right": {
        "action": "向右",
        "modality": "咬肌+MI",
        "region": "左侧咬肌 + C3",
        "masseter_task": "咬住左边后槽牙",
        "mi_task": "想象右手握拳/用力",
        "subject_task": "咬住左边，同时想象右手握拳；对应机械臂向右",
        "feature": "左咬肌选择左右轴，右手 MI 选择向右方向",
    },
    "left": {
        "action": "向左",
        "modality": "咬肌+MI",
        "region": "左侧咬肌 + C4",
        "masseter_task": "咬住左边后槽牙",
        "mi_task": "想象左手握拳/用力",
        "subject_task": "咬住左边，同时想象左手握拳；对应机械臂向左",
        "feature": "左咬肌选择左右轴，左手 MI 选择向左方向",
    },
    "up": {
        "action": "向上",
        "modality": "咬肌+MI",
        "region": "双侧咬肌 + C4",
        "masseter_task": "同时咬住左右两边后槽牙",
        "mi_task": "想象左手握拳/用力",
        "subject_task": "同时咬住两边，同时想象左手握拳；对应机械臂向上",
        "feature": "双侧咬肌选择上下轴，左手 MI 选择向上方向",
    },
    "down": {
        "action": "向下",
        "modality": "咬肌+MI",
        "region": "双侧咬肌 + C3",
        "masseter_task": "同时咬住左右两边后槽牙",
        "mi_task": "想象右手握拳/用力",
        "subject_task": "同时咬住两边，同时想象右手握拳；对应机械臂向下",
        "feature": "双侧咬肌选择上下轴，右手 MI 选择向下方向",
    },
    GRIP_TOGGLE_LABEL: {
        "action": "抓/放状态切换",
        "modality": "EOG/虚拟眨眼",
        "region": "眼电/眨眼事件",
        "masseter_task": "-",
        "mi_task": "-",
        "subject_task": "连续眨眼三次，机械臂在抓取状态和释放状态之间切换",
        "feature": "三次眨眼事件只控制抓握状态，不进入六类移动模型训练",
    },
    "rest": {
        "action": "休息",
        "modality": "REST",
        "region": "-",
        "masseter_task": "-",
        "mi_task": "-",
        "subject_task": "放松，不要执行运动想象、咬肌动作或眨眼指令",
        "feature": "静息状态",
    },
}
MI8_PARADIGM = MI6_PARADIGM
EMG_LABELS: set = set()


def is_grip_toggle_event(label) -> bool:
    return str(label).strip().lower() in GRIP_TOGGLE_ALIASES


def paradigm_info(label: str) -> Dict[str, str]:
    key = str(label).strip().lower()
    if key in GRIP_TOGGLE_ALIASES:
        return MI8_PARADIGM[GRIP_TOGGLE_LABEL]
    try:
        return MI8_PARADIGM.get(normalize_label(label), MI8_PARADIGM["rest"])
    except ValueError:
        return MI8_PARADIGM["rest"]


def group_csp_model_candidates(preferred_path: str = "") -> List[str]:
    candidates: List[str] = []
    for path in [preferred_path, DEFAULT_GROUP_CSP32_MODEL_PATH, DEFAULT_GROUP_CSP_MODEL_PATH]:
        if path and os.path.abspath(path) not in [os.path.abspath(item) for item in candidates]:
            candidates.append(path)
    return candidates

DEFAULT_EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "FC5",
    "FC6",
    "CP5",
    "CP6",
    "AF3",
    "AF4",
    "PO3",
    "PO4",
]


def brainvision_lsl_aux_indices(channel_count: int, eeg_channels: int = 32, aux_channels: int = 4) -> List[int]:
    total = int(channel_count)
    required = int(eeg_channels) + int(aux_channels)
    if total < required:
        raise RuntimeError(
            f"当前 LSL 流只有 {total} 个通道；EEG+肌电模式至少需要 {required} 个通道"
            f"（前 {eeg_channels} 路 EEG + 后 {aux_channels} 路 AUX/肌电）。"
        )
    return list(range(int(eeg_channels), required))


@dataclass
class SignalConfig:
    fs: int = 250
    eeg_channels: List[str] = field(default_factory=lambda: list(DEFAULT_EEG_CHANNELS))
    window_seconds: float = 1.0
    step_seconds: float = 0.2
    calibration_trial_seconds: float = 4.0
    max_windows_per_trial: int = 8
    notch_hz: float = 50.0
    use_csp: bool = True
    csp_components: int = 4
    filter_bands: Tuple[Tuple[float, float], ...] = (
        (8.0, 12.0),
        (12.0, 16.0),
        (16.0, 20.0),
        (20.0, 24.0),
        (24.0, 30.0),
        (30.0, 40.0),
    )

    @property
    def window_samples(self) -> int:
        return max(1, int(round(self.window_seconds * self.fs)))

    @property
    def step_samples(self) -> int:
        return max(1, int(round(self.step_seconds * self.fs)))


@dataclass
class DecoderConfig:
    eeg_weight: float = 0.55
    aux_weight_cap: float = 0.45
    command_confidence: float = 0.62
    min_margin: float = 0.08
    stable_windows: int = 2
    history_windows: int = 5
    refractory_seconds: float = 0.25
    command_duration_seconds: float = 0.55
    artifact_z_threshold: float = 9.0

    def validated_eeg_weight(self) -> float:
        return min(0.85, max(0.35, float(self.eeg_weight)))


@dataclass
class Decision:
    timestamp: float
    intent: str
    confidence: float
    margin: float
    command: Optional[str]
    accepted: bool
    reason: str
    state: str
    probabilities: Dict[str, float]


@dataclass
class RobotCommand:
    name: str
    intent: str
    target: str
    confidence: float
    sequence_id: int
    timestamp: float
    duration_seconds: float = 0.55


def normalize_label(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, np.integer)):
        if 0 <= int(value) < len(LABELS):
            return LABELS[int(value)]
        if int(value) == len(LABELS):
            return REST_LABEL
    key = str(value).strip().lower()
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    raise ValueError(f"Unsupported label {value!r}; expected one of {MODEL_LABELS}")


def labels_to_indices(y: Sequence) -> np.ndarray:
    return np.asarray([LABEL_TO_INDEX[normalize_label(v)] for v in y], dtype=np.int64)


def indices_to_label_names(y: Sequence[int]) -> List[str]:
    return [MODEL_LABELS[int(i)] for i in y]


def ensure_3d_epochs(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (trials, channels, samples)")
    return arr


def ensure_2d_window(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (channels, samples)")
    return arr


def safe_sos_filter(data: np.ndarray, sos) -> np.ndarray:
    if not HAS_SCIPY:
        return data
    if data.shape[-1] < 18:
        return data
    try:
        return sosfiltfilt(sos, data, axis=-1).astype(np.float32)
    except Exception:
        return data


def bandpass(data: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    if not HAS_SCIPY:
        return data
    nyq = fs / 2.0
    low = max(0.1, min(low, nyq - 1.0))
    high = max(low + 0.1, min(high, nyq - 1.0))
    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return safe_sos_filter(data, sos)


def notch(data: np.ndarray, fs: int, freq: float) -> np.ndarray:
    if not HAS_SCIPY or freq <= 0 or freq >= fs / 2.0 or data.shape[-1] < 18:
        return data
    try:
        b, a = iirnotch(freq / (fs / 2.0), Q=30.0)
        return filtfilt(b, a, data, axis=-1).astype(np.float32)
    except Exception:
        return data


EMG_FILTER_ORDER = 6
EMG_BAND_LOW_HZ = 20.0
EMG_BAND_HIGH_HZ = 100.0
EOG_BAND_LOW_HZ = 0.5
EOG_BAND_HIGH_HZ = 15.0


def _butter_bandpass_samples(data: np.ndarray, fs: float, low: float, high: float, order: int = EMG_FILTER_ORDER) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if not HAS_SCIPY or arr.ndim != 2 or arr.shape[0] < 24:
        return arr
    nyq = max(1.0, float(fs) / 2.0)
    low = max(0.05, min(float(low), nyq - 1.0))
    high = max(low + 0.1, min(float(high), nyq - 1.0))
    if not (0.0 < low < high < nyq):
        return arr
    try:
        sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        filtered = sosfilt(sos, arr, axis=0)
        return np.asarray(filtered, dtype=np.float32)
    except Exception:
        return arr


def filter_aux_samples_by_role(
    data: np.ndarray,
    fs: float,
    roles: Optional[Sequence[str]] = None,
    enabled: bool = True,
    emg_low: float = EMG_BAND_LOW_HZ,
    emg_high: float = EMG_BAND_HIGH_HZ,
    eog_low: float = EOG_BAND_LOW_HZ,
    eog_high: float = EOG_BAND_HIGH_HZ,
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if not enabled or arr.ndim != 2 or arr.shape[0] < 24:
        return arr
    if roles is None:
        roles = ["left_masseter"] * arr.shape[1]
    output = np.empty_like(arr)
    for idx in range(arr.shape[1]):
        role = str(roles[idx]) if idx < len(roles) else "display"
        if role in {"left_eye", "right_eye"}:
            output[:, idx : idx + 1] = _butter_bandpass_samples(
                arr[:, idx : idx + 1], fs, eog_low, eog_high
            )
        else:
            output[:, idx : idx + 1] = _butter_bandpass_samples(
                arr[:, idx : idx + 1], fs, emg_low, emg_high
            )
    return output


class SpeechAnnouncer:
    def __init__(self):
        self.queue: Queue = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.last_text = ""
        self.last_time = 0.0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def speak(self, text: str, enabled: bool = True, min_interval: float = 0.8) -> None:
        if not enabled:
            return
        text = str(text).strip()
        if not text:
            return
        now = time.time()
        if text == self.last_text and now - self.last_time < min_interval:
            return
        self.last_text = text
        self.last_time = now
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass
        try:
            self.queue.put_nowait(text)
        except Exception:
            pass

    def close(self) -> None:
        self.stop_event.set()
        try:
            self.queue.put_nowait("")
        except Exception:
            pass

    def _worker(self) -> None:
        sapi_voice = None
        if sys.platform.startswith("win"):
            try:
                import win32com.client  # type: ignore

                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Rate = 0
                sapi_voice.Volume = 100
            except Exception:
                sapi_voice = None
        while not self.stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            if self.stop_event.is_set() or not text:
                continue
            try:
                if sapi_voice is not None:
                    sapi_voice.Speak(text)
                elif sys.platform.startswith("win"):
                    script = (
                        "Add-Type -AssemblyName System.Speech;"
                        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                        "$s.Rate=0;$s.Volume=100;"
                        "$s.Speak([Console]::In.ReadToEnd())"
                    )
                    subprocess.run(
                        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                        input=text,
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=8,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
            except Exception:
                pass


def common_average_reference(eeg: np.ndarray) -> np.ndarray:
    return eeg - np.mean(eeg, axis=-2, keepdims=True)


def standardize_epochs(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def preprocess_eeg(eeg: np.ndarray, cfg: SignalConfig) -> np.ndarray:
    arr = np.asarray(eeg, dtype=np.float32)
    arr = common_average_reference(arr)
    arr = notch(arr, cfg.fs, cfg.notch_hz)
    arr = bandpass(arr, cfg.fs, 1.0, min(45.0, cfg.fs / 2.0 - 1.0))
    return standardize_epochs(arr)


@dataclass
class BinaryMIResult:
    hand: str
    confidence: float
    probabilities: Dict[str, float]


def normalize_binary_mi_hand(value: str) -> str:
    key = str(value).strip().lower()
    aliases = {
        "left": "left_hand_mi",
        "left_hand": "left_hand_mi",
        "left_hand_mi": "left_hand_mi",
        "左手": "left_hand_mi",
        "左手想象": "left_hand_mi",
        "right": "right_hand_mi",
        "right_hand": "right_hand_mi",
        "right_hand_mi": "right_hand_mi",
        "右手": "right_hand_mi",
        "右手想象": "right_hand_mi",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported binary MI hand label: {value!r}")
    return aliases[key]


def fuse_masseter_and_binary_mi(masseter_event: str, mi_hand: str) -> str:
    event_aliases = {
        "left": "left_masseter",
        "left_masseter": "left_masseter",
        "左咬肌": "left_masseter",
        "right": "right_masseter",
        "right_masseter": "right_masseter",
        "右咬肌": "right_masseter",
        "both": "both_masseter",
        "both_masseter": "both_masseter",
        "双咬肌": "both_masseter",
    }
    event_key = event_aliases.get(str(masseter_event).strip().lower())
    hand_key = normalize_binary_mi_hand(mi_hand)
    if event_key is None or (event_key, hand_key) not in BINARY_MI_FUSION_MAP:
        raise ValueError(f"Unsupported EMG/MI fusion input: {masseter_event!r}, {mi_hand!r}")
    return BINARY_MI_FUSION_MAP[(event_key, hand_key)]


class BinaryMIDeploymentModel:
    """Runtime wrapper for the deployable two-class CSP/MI model bundle."""

    def __init__(self, bundle: Dict):
        if not isinstance(bundle, dict):
            raise TypeError(
                f"所选文件是 {type(bundle).__name__}，不是二分类群 CSP 部署包；"
                "请选择 group_csp_left_right_mi.joblib。"
            )
        supported_types = {
            "binary_mi_compact_csp_lda",
            "binary_mi_reflection_group_csp_lda",
        }
        self.model_type = str(bundle.get("model_type", ""))
        if self.model_type not in supported_types:
            raise TypeError(f"不支持的二分类 MI 模型类型：{self.model_type or '缺少 model_type'}")
        self.bundle = bundle
        self.model = bundle.get("model")
        self.spatial_filters = np.asarray(bundle.get("spatial_filters", []), dtype=np.float64)
        self.classifier = bundle.get("classifier")
        self.fs = int(bundle["sampling_rate_hz"])
        self.input_channel_names = [str(name) for name in bundle["input_channel_names"]]
        self.selected_indices = np.asarray(
            bundle.get("selected_channel_indices", np.arange(len(self.input_channel_names))),
            dtype=np.int64,
        )
        self.window_samples = int(bundle.get("window_samples", bundle.get("samples_per_trial", 0)))
        self.labels = [str(label) for label in bundle.get("labels", ["left_hand_mi", "right_hand_mi"])]
        if self.model_type == "binary_mi_reflection_group_csp_lda":
            if self.spatial_filters.ndim != 2 or self.classifier is None:
                raise ValueError("Group CSP bundle is missing spatial_filters or classifier")

    @staticmethod
    def load(path: str) -> "BinaryMIDeploymentModel":
        return BinaryMIDeploymentModel(joblib.load(path))

    @staticmethod
    def _channel_key(name: str) -> str:
        return "".join(char for char in str(name).upper() if char.isalnum())

    def predict(self, eeg_window: np.ndarray, channel_names: Sequence[str], fs: float) -> BinaryMIResult:
        eeg = ensure_2d_window(eeg_window, "eeg_window")
        if eeg.shape[0] != len(channel_names):
            raise ValueError("channel_names length must match eeg_window channels")
        lookup = {self._channel_key(name): idx for idx, name in enumerate(channel_names)}
        missing = [name for name in self.input_channel_names if self._channel_key(name) not in lookup]
        if missing:
            raise ValueError(f"Binary MI input is missing channels: {missing}")
        order = [lookup[self._channel_key(name)] for name in self.input_channel_names]
        x = eeg[np.asarray(order, dtype=np.int64), :][None, :, :]

        source_fs = max(1, int(round(float(fs))))
        if source_fs != self.fs:
            if not HAS_SCIPY:
                raise RuntimeError("SciPy is required to resample binary MI input")
            divisor = math.gcd(source_fs, self.fs)
            x = resample_poly(x, self.fs // divisor, source_fs // divisor, axis=-1).astype(np.float32)

        if self.window_samples > 0:
            x = crop_or_pad_window(x, self.window_samples)

        x = common_average_reference(x)
        x = bandpass(x, self.fs, 8.0, 30.0)
        x = x[:, self.selected_indices, :]
        if self.model_type == "binary_mi_reflection_group_csp_lda":
            features = csp_log_variance_features(x, self.spatial_filters)
            probabilities = np.asarray(self.classifier.predict_proba(features)[0], dtype=np.float64)
            classes = np.asarray(self.classifier.classes_, dtype=np.int64)
        else:
            probabilities = np.asarray(self.model.predict_proba(x)[0], dtype=np.float64)
            classes = np.asarray(self.model.classes_, dtype=np.int64)
        best_pos = int(np.argmax(probabilities))
        class_index = int(classes[best_pos])
        if not 0 <= class_index < len(self.labels):
            raise RuntimeError(f"Unexpected binary MI class index: {class_index}")
        probability_map = {
            self.labels[int(cls)]: float(probabilities[pos])
            for pos, cls in enumerate(classes)
            if 0 <= int(cls) < len(self.labels)
        }
        return BinaryMIResult(
            hand=self.labels[class_index],
            confidence=float(probabilities[best_pos]),
            probabilities=probability_map,
        )

    def predict_intent(
        self,
        eeg_window: np.ndarray,
        channel_names: Sequence[str],
        fs: float,
        masseter_event: str,
    ) -> Tuple[str, BinaryMIResult]:
        result = self.predict(eeg_window, channel_names, fs)
        return fuse_masseter_and_binary_mi(masseter_event, result.hand), result


def csp_log_variance_features(eeg: np.ndarray, spatial_filters: np.ndarray) -> np.ndarray:
    epochs = ensure_3d_epochs(eeg, "eeg")
    filters = np.asarray(spatial_filters, dtype=np.float64)
    if filters.ndim != 2 or filters.shape[1] != epochs.shape[1]:
        raise ValueError("spatial_filters must have shape (components, channels)")
    projected = np.einsum("kc,nct->nkt", filters, epochs, optimize=True)
    variance = np.var(projected, axis=-1)
    normalized = variance / np.maximum(np.sum(variance, axis=1, keepdims=True), 1e-12)
    return np.log(np.maximum(normalized, 1e-12)).astype(np.float64)


def extract_timestamped_epochs(
    samples: np.ndarray,
    timestamps: np.ndarray,
    epoch_starts: Sequence[float],
    fs: float,
    duration_seconds: float,
) -> np.ndarray:
    """Interpolate a timestamped sample stream into fixed-length channel-first epochs."""
    data = np.asarray(samples, dtype=np.float32)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if data.ndim != 2 or data.shape[0] != ts.size:
        raise ValueError("samples must be (samples, channels) and match timestamps")
    if ts.size < 2 or np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    sample_count = max(1, int(round(float(fs) * float(duration_seconds))))
    offsets = np.arange(sample_count, dtype=np.float64) / float(fs)
    epochs = []
    for start in epoch_starts:
        target = float(start) + offsets
        if target[0] < ts[0] or target[-1] > ts[-1]:
            raise ValueError(f"Epoch {start:.6f} is outside the recorded LSL time range")
        epoch = np.stack(
            [np.interp(target, ts, data[:, channel]) for channel in range(data.shape[1])],
            axis=0,
        )
        epochs.append(epoch.astype(np.float32))
    if not epochs:
        return np.zeros((0, data.shape[1], sample_count), dtype=np.float32)
    return np.stack(epochs)


def crop_or_pad_window(x: np.ndarray, samples: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[-1] == samples:
        return arr
    if arr.shape[-1] > samples:
        return arr[..., :samples]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, samples - arr.shape[-1])
    return np.pad(arr, pad_width, mode="constant")


def make_windows_from_trials(
    eeg: np.ndarray,
    y: np.ndarray,
    cfg: SignalConfig,
    aux: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    eeg = ensure_3d_epochs(eeg, "eeg")
    aux = ensure_3d_epochs(aux, "aux") if aux is not None else None
    y = np.asarray(y, dtype=np.int64)
    win = cfg.window_samples
    step = cfg.step_samples

    eeg_windows: List[np.ndarray] = []
    aux_windows: List[np.ndarray] = []
    y_windows: List[int] = []

    for i in range(eeg.shape[0]):
        samples = eeg.shape[-1]
        if samples <= win:
            starts = [0]
        else:
            starts = list(range(0, samples - win + 1, step))
            if not starts or starts[-1] != samples - win:
                starts.append(samples - win)
            if len(starts) > cfg.max_windows_per_trial:
                idx = np.linspace(0, len(starts) - 1, cfg.max_windows_per_trial).round().astype(int)
                starts = [starts[j] for j in sorted(set(idx.tolist()))]

        for start in starts:
            end = start + win
            eeg_windows.append(crop_or_pad_window(eeg[i, :, start:end], win))
            if aux is not None:
                aux_windows.append(crop_or_pad_window(aux[i, :, start:end], win))
            y_windows.append(int(y[i]))

    eeg_arr = np.stack(eeg_windows).astype(np.float32)
    aux_arr = np.stack(aux_windows).astype(np.float32) if aux is not None else None
    return eeg_arr, np.asarray(y_windows, dtype=np.int64), aux_arr


def log_band_power(x: np.ndarray) -> np.ndarray:
    variance = np.var(x, axis=-1)
    return np.log(variance + 1e-6).astype(np.float32)


class FilterBankEEGFeatures:
    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self.csp_models: List[Optional[object]] = []
        self.use_csp = bool(cfg.use_csp and HAS_MNE_CSP)

    def fit(self, eeg: np.ndarray, y: np.ndarray) -> "FilterBankEEGFeatures":
        eeg = preprocess_eeg(ensure_3d_epochs(eeg, "eeg"), self.cfg)
        y = np.asarray(y, dtype=np.int64)
        self.csp_models = []

        for low, high in self.cfg.filter_bands:
            filtered = bandpass(eeg, self.cfg.fs, low, high)
            model = None
            if self.use_csp:
                n_components = min(self.cfg.csp_components, filtered.shape[1], max(2, len(np.unique(y))))
                try:
                    model = CSP(
                        n_components=n_components,
                        reg="ledoit_wolf",
                        log=True,
                        norm_trace=False,
                    )
                    model.fit(filtered, y)
                except Exception:
                    model = None
            self.csp_models.append(model)
        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        eeg = preprocess_eeg(ensure_3d_epochs(eeg, "eeg"), self.cfg)
        parts: List[np.ndarray] = []

        for band_idx, (low, high) in enumerate(self.cfg.filter_bands):
            filtered = bandpass(eeg, self.cfg.fs, low, high)
            parts.append(log_band_power(filtered))
            if band_idx < len(self.csp_models) and self.csp_models[band_idx] is not None:
                try:
                    parts.append(self.csp_models[band_idx].transform(filtered).astype(np.float32))
                except Exception:
                    pass

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(self, eeg: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(eeg, y).transform(eeg)


class HeadFaceAuxFeatures:
    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg

    def fit(self, aux: np.ndarray, y: np.ndarray) -> "HeadFaceAuxFeatures":
        return self

    def transform(self, aux: np.ndarray) -> np.ndarray:
        aux = ensure_3d_epochs(aux, "aux")
        x = notch(aux, self.cfg.fs, self.cfg.notch_hz)
        x = standardize_epochs(x)

        rms = np.sqrt(np.mean(np.square(x), axis=-1) + 1e-6)
        mav = np.mean(np.abs(x), axis=-1)
        waveform_len = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
        peak = np.percentile(np.abs(x), 95, axis=-1)
        parts = [rms, mav, waveform_len, peak]

        if self.cfg.fs >= 200:
            for low, high in ((20.0, 45.0), (45.0, min(90.0, self.cfg.fs / 2.0 - 1.0))):
                if high > low + 1.0:
                    parts.append(log_band_power(bandpass(x, self.cfg.fs, low, high)))

        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(self, aux: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(aux, y).transform(aux)


def make_classifier(kind: str = "logreg"):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )


def can_stratified_split(y: np.ndarray) -> bool:
    counts = Counter(y.tolist())
    return len(counts) >= 2 and min(counts.values()) >= 2


def proba_dict(proba: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(proba, dtype=float).reshape(-1)
    return {label: float(arr[idx]) for idx, label in enumerate(MODEL_LABELS) if idx < len(arr)}


class HybridBCIModel:
    def __init__(
        self,
        signal_config: Optional[SignalConfig] = None,
        decoder_config: Optional[DecoderConfig] = None,
    ):
        self.signal_config = signal_config or SignalConfig()
        self.decoder_config = decoder_config or DecoderConfig()
        self.eeg_features: Optional[FilterBankEEGFeatures] = None
        self.aux_features: Optional[HeadFaceAuxFeatures] = None
        self.eeg_classifier = None
        self.aux_classifier = None
        self.classifier_kind = "logreg"
        self.metrics_: Dict = {}
        self.eeg_abs_center_: float = 0.0
        self.eeg_abs_mad_: float = 1.0
        self.aux_abs_center_: float = 0.0
        self.aux_abs_mad_: float = 1.0

    def fit(
        self,
        eeg: np.ndarray,
        y: Sequence,
        aux: Optional[np.ndarray] = None,
        classifier_kind: str = "logreg",
    ) -> Dict:
        self.classifier_kind = classifier_kind
        y_idx = labels_to_indices(y)
        eeg = ensure_3d_epochs(eeg, "eeg")
        aux = ensure_3d_epochs(aux, "aux") if aux is not None else None

        if eeg.shape[0] != len(y_idx):
            raise ValueError("eeg and y must have the same trial count")
        if aux is not None and aux.shape[0] != len(y_idx):
            raise ValueError("aux and y must have the same trial count")

        metrics: Dict = {}
        if can_stratified_split(y_idx):
            n_classes = len(np.unique(y_idx))
            test_size = max(0.25, n_classes / float(len(y_idx)))
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            train_idx, val_idx = next(splitter.split(np.arange(len(y_idx)), y_idx))
            train_eeg, train_y, train_aux = make_windows_from_trials(
                eeg[train_idx], y_idx[train_idx], self.signal_config, aux[train_idx] if aux is not None else None
            )
            val_eeg, val_y, val_aux = make_windows_from_trials(
                eeg[val_idx], y_idx[val_idx], self.signal_config, aux[val_idx] if aux is not None else None
            )
            validator = HybridBCIModel(self.signal_config, self.decoder_config)
            validator.classifier_kind = classifier_kind
            validator._fit_core(train_eeg, train_y, train_aux)
            metrics = validator.evaluate_windows(val_eeg, val_y, val_aux)

        all_eeg, all_y, all_aux = make_windows_from_trials(eeg, y_idx, self.signal_config, aux)
        self._fit_core(all_eeg, all_y, all_aux)
        self.metrics_ = metrics
        return metrics

    def _fit_core(self, eeg_windows: np.ndarray, y_windows: np.ndarray, aux_windows: Optional[np.ndarray]) -> None:
        self._fit_artifact_baseline(eeg_windows, aux_windows)

        self.eeg_features = FilterBankEEGFeatures(self.signal_config)
        eeg_x = self.eeg_features.fit_transform(eeg_windows, y_windows)
        self.eeg_classifier = make_classifier(self.classifier_kind)
        self.eeg_classifier.fit(eeg_x, y_windows)

        if aux_windows is not None:
            self.aux_features = HeadFaceAuxFeatures(self.signal_config)
            aux_x = self.aux_features.fit_transform(aux_windows, y_windows)
            self.aux_classifier = make_classifier("lda")
            self.aux_classifier.fit(aux_x, y_windows)
        else:
            self.aux_features = None
            self.aux_classifier = None

    def _fit_artifact_baseline(self, eeg_windows: np.ndarray, aux_windows: Optional[np.ndarray]) -> None:
        eeg_abs = np.percentile(np.abs(eeg_windows), 95, axis=(1, 2))
        self.eeg_abs_center_, self.eeg_abs_mad_ = robust_center_mad(eeg_abs)
        if aux_windows is not None:
            aux_abs = np.percentile(np.abs(aux_windows), 95, axis=(1, 2))
            self.aux_abs_center_, self.aux_abs_mad_ = robust_center_mad(aux_abs)

    def detect_artifact(self, eeg_window: np.ndarray, aux_window: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        eeg_window = ensure_2d_window(eeg_window, "eeg_window")
        eeg_score = (np.percentile(np.abs(eeg_window), 95) - self.eeg_abs_center_) / self.eeg_abs_mad_
        if eeg_score > self.decoder_config.artifact_z_threshold:
            return True, f"eeg_artifact_z={eeg_score:.1f}"

        if aux_window is not None and self.aux_features is not None:
            aux_window = ensure_2d_window(aux_window, "aux_window")
            aux_score = (np.percentile(np.abs(aux_window), 95) - self.aux_abs_center_) / self.aux_abs_mad_
            if aux_score > self.decoder_config.artifact_z_threshold:
                return True, f"aux_artifact_z={aux_score:.1f}"

        return False, "clean"

    def predict_eeg_proba(self, eeg_window_or_batch: np.ndarray) -> np.ndarray:
        if self.eeg_features is None or self.eeg_classifier is None:
            raise RuntimeError("Model is not fitted")
        eeg = ensure_3d_epochs(eeg_window_or_batch, "eeg")
        eeg = crop_or_pad_window(eeg, self.signal_config.window_samples)
        x = self.eeg_features.transform(eeg)
        return align_proba(self.eeg_classifier.predict_proba(x), self.eeg_classifier.classes_)

    def predict_aux_proba(self, aux_window_or_batch: np.ndarray) -> Optional[np.ndarray]:
        if self.aux_features is None or self.aux_classifier is None or aux_window_or_batch is None:
            return None
        aux = ensure_3d_epochs(aux_window_or_batch, "aux")
        aux = crop_or_pad_window(aux, self.signal_config.window_samples)
        x = self.aux_features.transform(aux)
        return align_proba(self.aux_classifier.predict_proba(x), self.aux_classifier.classes_)

    def predict_proba(self, eeg_window: np.ndarray, aux_window: Optional[np.ndarray] = None) -> np.ndarray:
        eeg_proba = self.predict_eeg_proba(eeg_window)
        aux_proba = self.predict_aux_proba(aux_window) if aux_window is not None else None
        if aux_proba is None:
            return eeg_proba

        eeg_weight = self.decoder_config.validated_eeg_weight()
        aux_weight = min(self.decoder_config.aux_weight_cap, 1.0 - eeg_weight)
        if aux_weight <= 0:
            fused = eeg_proba.copy()
        else:
            fused = eeg_weight * eeg_proba + aux_weight * aux_proba
        fused = fused / np.sum(fused, axis=1, keepdims=True)
        return fused

    def evaluate_windows(
        self,
        eeg_windows: np.ndarray,
        y_windows: np.ndarray,
        aux_windows: Optional[np.ndarray] = None,
    ) -> Dict:
        y_windows = np.asarray(y_windows, dtype=np.int64)
        eeg_proba = self.predict_eeg_proba(eeg_windows)
        eeg_pred = np.argmax(eeg_proba, axis=1)
        result = {
            "window_count": int(len(y_windows)),
            "eeg_only_accuracy": float(accuracy_score(y_windows, eeg_pred)),
            "eeg_only_kappa": float(cohen_kappa_score(y_windows, eeg_pred)),
            "labels": list(MODEL_LABELS),
            "eeg_weight": self.decoder_config.validated_eeg_weight(),
        }

        aux_proba = self.predict_aux_proba(aux_windows) if aux_windows is not None else None
        if aux_proba is not None:
            aux_pred = np.argmax(aux_proba, axis=1)
            result["aux_only_accuracy"] = float(accuracy_score(y_windows, aux_pred))
            result["aux_only_kappa"] = float(cohen_kappa_score(y_windows, aux_pred))

        fused_proba = self.predict_proba(eeg_windows, aux_windows)
        fused_pred = np.argmax(fused_proba, axis=1)
        result.update(
            {
                "fusion_accuracy": float(accuracy_score(y_windows, fused_pred)),
                "fusion_kappa": float(cohen_kappa_score(y_windows, fused_pred)),
                "confusion_matrix": confusion_matrix(y_windows, fused_pred, labels=list(range(len(MODEL_LABELS)))).tolist(),
                "classification_report": classification_report(
                    y_windows,
                    fused_pred,
                    labels=list(range(len(MODEL_LABELS))),
                    target_names=list(MODEL_LABELS),
                    zero_division=0,
                    output_dict=True,
                ),
            }
        )
        return result

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HybridBCIModel":
        model = joblib.load(path)
        if not isinstance(model, HybridBCIModel):
            raise TypeError(f"{path} does not contain a HybridBCIModel")
        return model


def robust_center_mad(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)) * 1.4826)
    return center, max(mad, 1e-6)


def align_proba(proba: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(MODEL_LABELS)), dtype=np.float32)
    for src_idx, cls in enumerate(classes):
        if 0 <= int(cls) < len(MODEL_LABELS):
            out[:, int(cls)] = proba[:, src_idx]
    row_sum = np.sum(out, axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return out / row_sum


class RobotArmClient:
    def execute(self, command: RobotCommand) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockRobotArm(RobotArmClient):
    def __init__(self, action_delay: float = 0.15):
        self.action_delay = action_delay
        self.commands: List[RobotCommand] = []

    def execute(self, command: RobotCommand) -> None:
        self.commands.append(command)
        print(
            f"[MOCK_ROBOT] seq={command.sequence_id} cmd={command.name} "
            f"intent={command.intent} target={command.target} conf={command.confidence:.3f} "
            f"duration={command.duration_seconds:.2f}s"
        )
        if self.action_delay > 0:
            time.sleep(self.action_delay)


class SocketRobotArmClient(RobotArmClient):
    def __init__(self, host: str, port: int, timeout: float = 1.0):
        self.host = host
        self.port = int(port)
        self.timeout = timeout

    def execute(self, command: RobotCommand) -> None:
        payload = {
            "cmd": command.name,
            "intent": command.intent,
            "target": command.target,
            "confidence": command.confidence,
            "sequence_id": command.sequence_id,
            "timestamp": command.timestamp,
            "duration_seconds": command.duration_seconds,
        }
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.sendall(data)
            sock.settimeout(self.timeout)
            try:
                ack = sock.recv(256)
                if ack:
                    print(f"[ROBOT_ACK] {ack.decode('utf-8', errors='replace').strip()}")
            except socket.timeout:
                pass


class AsyncIntentController:
    def __init__(
        self,
        model: HybridBCIModel,
        robot: RobotArmClient,
        target: str = "target_1",
        cfg: Optional[DecoderConfig] = None,
    ):
        self.model = model
        self.robot = robot
        self.cfg = cfg or model.decoder_config
        if not hasattr(self.cfg, "refractory_seconds"):
            self.cfg.refractory_seconds = 0.25
        if not hasattr(self.cfg, "command_duration_seconds"):
            self.cfg.command_duration_seconds = 0.55
        self.target = target
        self.state = "ready"
        self.sequence_id = 0
        self.last_command_time = 0.0
        self.history: deque = deque(maxlen=max(self.cfg.history_windows, self.cfg.stable_windows))

    def reset_task(self) -> None:
        self.state = "ready"
        self.history.clear()

    def update(self, eeg_window: np.ndarray, aux_window: Optional[np.ndarray] = None, now: Optional[float] = None) -> Decision:
        now = time.time() if now is None else float(now)
        artifact, artifact_reason = self.model.detect_artifact(eeg_window, aux_window)
        if artifact:
            uniform = np.ones(len(MODEL_LABELS), dtype=np.float32) / float(len(MODEL_LABELS))
            return self._decision(now, "artifact", 0.0, 0.0, None, False, artifact_reason, uniform)

        probabilities = self.model.predict_proba(eeg_window, aux_window)[0]
        order = np.argsort(probabilities)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        intent = MODEL_LABELS[top_idx]
        confidence = float(probabilities[top_idx])
        margin = float(probabilities[top_idx] - probabilities[second_idx])

        if intent == REST_LABEL:
            self.state = REST_LABEL
            self.history.clear()
            return self._decision(now, intent, confidence, margin, None, False, "rest_state", probabilities)

        self.history.append((intent, confidence, margin, now))

        if now - self.last_command_time < self.cfg.refractory_seconds:
            return self._decision(now, intent, confidence, margin, None, False, "refractory", probabilities)

        if not self._is_stable(intent):
            return self._decision(now, intent, confidence, margin, None, False, "waiting_for_stability", probabilities)

        if confidence < self.cfg.command_confidence or margin < self.cfg.min_margin:
            return self._decision(now, intent, confidence, margin, None, False, "low_confidence_or_margin", probabilities)

        command_name = ACTION_COMMANDS[intent]
        self.sequence_id += 1
        command = RobotCommand(
            name=command_name,
            intent=intent,
            target=self.target,
            confidence=confidence,
            sequence_id=self.sequence_id,
            timestamp=now,
            duration_seconds=self.cfg.command_duration_seconds,
        )
        self.robot.execute(command)
        self.last_command_time = now
        self.state = intent
        self.history.clear()

        return self._decision(now, intent, confidence, margin, command_name, True, "accepted", probabilities)

    def _is_stable(self, intent: str) -> bool:
        recent = list(self.history)[-self.cfg.stable_windows :]
        if len(recent) < self.cfg.stable_windows:
            return False
        same = [item for item in recent if item[0] == intent]
        if len(same) < self.cfg.stable_windows:
            return False
        mean_conf = float(np.mean([item[1] for item in same]))
        mean_margin = float(np.mean([item[2] for item in same]))
        return mean_conf >= self.cfg.command_confidence and mean_margin >= self.cfg.min_margin

    def _decision(
        self,
        now: float,
        intent: str,
        confidence: float,
        margin: float,
        command: Optional[str],
        accepted: bool,
        reason: str,
        probabilities: np.ndarray,
    ) -> Decision:
        return Decision(
            timestamp=now,
            intent=intent,
            confidence=float(confidence),
            margin=float(margin),
            command=command,
            accepted=accepted,
            reason=reason,
            state=self.state,
            probabilities=proba_dict(probabilities),
        )


class GripStateController:
    def __init__(
        self,
        robot: RobotArmClient,
        target: str = "target_1",
        duration_seconds: float = 0.35,
        refractory_seconds: float = 1.5,
    ):
        self.robot = robot
        self.target = target
        self.duration_seconds = float(duration_seconds)
        self.refractory_seconds = float(refractory_seconds)
        self.closed = False
        self.sequence_id = 0
        self.last_toggle_time = 0.0

    def reset(self) -> None:
        self.closed = False
        self.last_toggle_time = 0.0

    def toggle(
        self,
        now: Optional[float] = None,
        confidence: float = 1.0,
        force: bool = False,
    ) -> Optional[RobotCommand]:
        now = time.time() if now is None else float(now)
        if not force and now - self.last_toggle_time < self.refractory_seconds:
            return None
        self.closed = not self.closed
        intent = "grab" if self.closed else "release"
        self.sequence_id += 1
        command = RobotCommand(
            name=ACTION_COMMANDS[intent],
            intent=intent,
            target=self.target,
            confidence=float(confidence),
            sequence_id=self.sequence_id,
            timestamp=now,
            duration_seconds=self.duration_seconds,
        )
        self.robot.execute(command)
        self.last_toggle_time = now
        return command

    @property
    def state_text(self) -> str:
        return "抓取" if self.closed else "释放"


class SignalSource:
    def read_window(self) -> Tuple[np.ndarray, Optional[np.ndarray], float, Optional[str]]:
        raise NotImplementedError


@dataclass
class RdaMarker:
    position: int = 0
    points: int = 0
    channel: int = -1
    type: str = ""
    description: str = ""


class ParallelPortTrigger:
    def __init__(self, port: int = 0x0378, pulse_seconds: float = 0.006, clear_seconds: float = 0.010):
        self.port = int(port)
        self.pulse_seconds = pulse_seconds
        self.clear_seconds = clear_seconds
        try:
            from ctypes import windll

            self.driver = windll.inpoutx64
        except Exception as exc:
            raise RuntimeError(f"无法加载 inpoutx64.dll，并口 trigger 不可用：{exc}") from exc

    def send(self, code: int) -> None:
        self.driver.Out32(self.port, int(code))
        time.sleep(self.pulse_seconds)
        self.driver.Out32(self.port, 0)
        time.sleep(self.clear_seconds)


def recv_exact(sock: socket.socket, requested_size: int) -> bytes:
    chunks = []
    remaining = requested_size
    while remaining > 0:
        data = sock.recv(remaining)
        if not data:
            raise RuntimeError("RDA connection broken")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def split_rda_strings(raw: bytes) -> List[str]:
    out: List[str] = []
    buf = bytearray()
    for value in raw:
        if value:
            buf.append(value)
        else:
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf.clear()
    if buf:
        out.append(buf.decode("utf-8", errors="replace"))
    return out


def parse_rda_properties(rawdata: bytes) -> Tuple[int, float, List[float], List[str]]:
    channel_count, sampling_interval = struct.unpack("<Ld", rawdata[:12])
    resolutions = []
    for channel_idx in range(channel_count):
        start = 12 + channel_idx * 8
        resolutions.append(struct.unpack("<d", rawdata[start : start + 8])[0])
    channel_names = split_rda_strings(rawdata[12 + 8 * channel_count :])
    return int(channel_count), float(sampling_interval), resolutions, channel_names


def parse_rda_data(rawdata: bytes, channel_count: int) -> Tuple[int, int, int, np.ndarray, List[RdaMarker]]:
    block, points, marker_count = struct.unpack("<LLL", rawdata[:12])
    data_count = int(points) * int(channel_count)
    data_start = 12
    data_end = data_start + 4 * data_count
    eeg = np.frombuffer(rawdata[data_start:data_end], dtype="<f4", count=data_count).astype(np.float32)
    eeg = eeg.reshape(int(points), int(channel_count)).T

    markers: List[RdaMarker] = []
    index = data_end
    for _ in range(marker_count):
        marker_size = struct.unpack("<L", rawdata[index : index + 4])[0]
        marker = RdaMarker()
        marker.position, marker.points, marker.channel = struct.unpack("<LLl", rawdata[index + 4 : index + 16])
        typedesc = split_rda_strings(rawdata[index + 16 : index + marker_size])
        if typedesc:
            marker.type = typedesc[0]
        if len(typedesc) > 1:
            marker.description = typedesc[1]
        markers.append(marker)
        index += marker_size
    return int(block), int(points), int(marker_count), eeg, markers


class BrainVisionRDAClient:
    START = 1
    STOP = 3
    DATA32 = 4

    def __init__(self, host: str = "127.0.0.1", port: int = 51244, timeout: float = 5.0):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.channel_count = 0
        self.sampling_interval_us = 0.0
        self.fs = 250
        self.resolutions: List[float] = []
        self.channel_names: List[str] = []
        self.last_block = -1

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def read_message(self) -> Tuple[str, object]:
        if self.sock is None:
            raise RuntimeError("RDA client is not connected")
        raw_header = recv_exact(self.sock, 24)
        _id1, _id2, _id3, _id4, msg_size, msg_type = struct.unpack("<llllLL", raw_header)
        raw_data = recv_exact(self.sock, int(msg_size) - 24)

        if msg_type == self.START:
            channel_count, sampling_interval, resolutions, channel_names = parse_rda_properties(raw_data)
            self.channel_count = channel_count
            self.sampling_interval_us = sampling_interval
            self.fs = int(round(1_000_000.0 / sampling_interval)) if sampling_interval > 0 else 250
            self.resolutions = resolutions
            self.channel_names = channel_names
            self.last_block = -1
            return "info", {
                "channel_count": self.channel_count,
                "sampling_interval_us": self.sampling_interval_us,
                "fs": self.fs,
                "resolutions": self.resolutions,
                "channel_names": self.channel_names,
            }

        if msg_type == self.DATA32:
            block, points, marker_count, eeg, markers = parse_rda_data(raw_data, self.channel_count)
            overflow = self.last_block != -1 and block > self.last_block + 1
            self.last_block = block
            return "data", {
                "block": block,
                "points": points,
                "marker_count": marker_count,
                "eeg": eeg,
                "markers": markers,
                "overflow": overflow,
            }

        if msg_type == self.STOP:
            return "stop", {}

        return "unknown", {"msg_type": msg_type}


class RDA64Recorder:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 51244,
        expected_channels: int = 64,
        timeout: float = 5.0,
        log_callback=None,
    ):
        self.client = BrainVisionRDAClient(host, port, timeout)
        self.expected_channels = int(expected_channels)
        self.log_callback = log_callback
        self.stop_event = threading.Event()
        self.trigger_queue: Queue[int] = Queue()
        self.packet_queue: Queue[np.ndarray] = Queue()
        self.info: Dict = {}
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None

    def log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    def start(self) -> None:
        self.stop_event.clear()
        self.client.connect()
        self.thread = threading.Thread(target=self._read_loop, name="RDA64Reader", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.client.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def push_trigger(self, code: int) -> None:
        self.trigger_queue.put(int(code))

    def _read_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                form, payload = self.client.read_message()
                if form == "info":
                    self.info = payload
                    channel_count = int(payload.get("channel_count", 0))
                    self.log(f"RDA connected: channels={channel_count}, fs={payload.get('fs')}")
                    if channel_count < self.expected_channels:
                        self.log(f"WARNING: RDA channel count {channel_count} < expected {self.expected_channels}")
                elif form == "data":
                    eeg = payload["eeg"]
                    if eeg.shape[0] > self.expected_channels:
                        eeg = eeg[: self.expected_channels, :]
                    elif eeg.shape[0] < self.expected_channels:
                        pad = np.zeros((self.expected_channels - eeg.shape[0], eeg.shape[1]), dtype=np.float32)
                        eeg = np.concatenate([eeg, pad], axis=0)
                    trigger_row = np.zeros((1, eeg.shape[1]), dtype=np.float32)
                    try:
                        trigger_row[0, 0] = float(self.trigger_queue.get_nowait())
                    except Empty:
                        pass
                    packet = np.concatenate([eeg.astype(np.float32), trigger_row], axis=0)
                    self.packet_queue.put(packet)
                    if payload.get("overflow"):
                        self.log(f"WARNING: RDA block overflow near block {payload.get('block')}")
                elif form == "stop":
                    self.log("RDA stream stopped by recorder")
                    break
        except Exception as exc:
            self.error = exc
            self.log(f"RDA read error: {exc}")
        finally:
            self.client.close()

    def collect_packets(self) -> List[np.ndarray]:
        packets: List[np.ndarray] = []
        while True:
            try:
                packets.append(self.packet_queue.get_nowait())
            except Empty:
                break
        return packets


class RDA64OnlineSource(SignalSource):
    def __init__(
        self,
        cfg: SignalConfig,
        host: str = "127.0.0.1",
        port: int = 51244,
        expected_channels: int = 64,
        aux_channels: int = 0,
        log_callback=None,
    ):
        self.cfg = cfg
        self.expected_channels = int(expected_channels)
        self.aux_channels = max(0, int(aux_channels))
        self.total_channels = self.expected_channels + self.aux_channels
        self.recorder = RDA64Recorder(host=host, port=port, expected_channels=self.total_channels, log_callback=log_callback)
        self.buffer = np.zeros((self.total_channels, 0), dtype=np.float32)
        self.started = False

    def start(self) -> None:
        if not self.started:
            self.recorder.start()
            self.started = True

    def close(self) -> None:
        self.recorder.stop()
        self.started = False

    def _pull_packets(self) -> None:
        packets = self.recorder.collect_packets()
        if not packets:
            return
        signal_packets = [packet[: self.total_channels, :] for packet in packets if packet.shape[1] > 0]
        if signal_packets:
            self.buffer = np.concatenate([self.buffer] + signal_packets, axis=1)

    def read_window(self) -> Tuple[np.ndarray, Optional[np.ndarray], float, Optional[str]]:
        if not self.started:
            self.start()
        if self.recorder.error:
            raise RuntimeError(f"RDA在线读取失败：{self.recorder.error}") from self.recorder.error
        self._pull_packets()
        win = self.cfg.window_samples
        step = self.cfg.step_samples
        if self.buffer.shape[1] < win:
            raise BlockingIOError("waiting_for_rda_samples")
        window = self.buffer[:, :win].astype(np.float32)
        self.buffer = self.buffer[:, min(step, self.buffer.shape[1]) :]
        eeg = window[: self.expected_channels, :]
        aux = window[self.expected_channels : self.expected_channels + self.aux_channels, :] if self.aux_channels else None
        return eeg, aux, time.time(), None


def extract_epochs_from_triggered_raw(
    raw_with_trigger: np.ndarray,
    fs: int,
    trigger_to_label: Dict[int, str],
    epoch_seconds: float,
    epoch_offset_seconds: float = 0.0,
    n_eeg_channels: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(raw_with_trigger, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("raw_with_trigger must have shape (channels + trigger, samples)")
    trigger = arr[-1, :]
    eeg = arr[:n_eeg_channels, :] if n_eeg_channels else arr[:-1, :]
    starts = np.where(trigger != 0)[0]
    samples = int(round(epoch_seconds * fs))
    offset = int(round(epoch_offset_seconds * fs))
    epochs: List[np.ndarray] = []
    labels: List[str] = []
    used_trigger_samples: List[int] = []
    for pos in starts:
        code = int(round(float(trigger[pos])))
        label = trigger_to_label.get(code)
        if label is None:
            continue
        start = int(pos) + offset
        end = start + samples
        if start < 0 or end > eeg.shape[1]:
            continue
        epochs.append(eeg[:, start:end].astype(np.float32))
        labels.append(label)
        used_trigger_samples.append(int(pos))
    if not epochs:
        raise RuntimeError("没有从 trigger 行中截取到有效 epoch，请检查 trigger 是否进入数据流")
    return np.stack(epochs), np.asarray(labels, dtype=object), np.asarray(used_trigger_samples, dtype=np.int64)


def collect_rda64_mi8_calibration(
    out_path: str,
    trials_per_class: int,
    epoch_seconds: float,
    rest_seconds: float = 2.0,
    pre_rest_seconds: float = 3.0,
    cue_seconds: float = 1.0,
    cue_rest_seconds: float = 3.0,
    post_rest_seconds: float = 3.0,
    host: str = "127.0.0.1",
    port: int = 51244,
    expected_channels: int = 64,
    aux_channels: int = 0,
    use_parallel_trigger: bool = True,
    parallel_port: int = 0x0378,
    log_callback=None,
    cue_callback=None,
    stop_event: Optional[threading.Event] = None,
) -> Dict:
    stop_event = stop_event or threading.Event()
    aux_channels = max(0, int(aux_channels))
    signal_channels = int(expected_channels) + aux_channels
    recorder = RDA64Recorder(host=host, port=port, expected_channels=signal_channels, log_callback=log_callback)
    trigger_sender = ParallelPortTrigger(port=parallel_port) if use_parallel_trigger else None
    labels = list(LABELS) * int(trials_per_class)
    random.shuffle(labels)
    packets: List[np.ndarray] = []
    trial_table: List[Dict] = []

    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    def emit_trigger(label: str) -> None:
        code = MODEL_TRIGGER_CODES[label]
        recorder.push_trigger(code)
        if trigger_sender:
            trigger_sender.send(code)

    def wait_phase(seconds: float) -> None:
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            if stop_event.is_set():
                break
            time.sleep(0.02)
        packets.extend(recorder.collect_packets())

    try:
        recorder.start()
        time.sleep(1.0)
        for idx, label in enumerate(labels, start=1):
            if stop_event.is_set():
                break
            display = LABEL_DISPLAY_NAMES.get(label, label)
            info = paradigm_info(label)
            log(f"Trial {idx}/{len(labels)}: {display} ({label})")
            row = {"trial": idx, "label": label, "display": display, "paradigm": info, "events": []}

            if cue_callback:
                cue_callback(label, "pre_rest", idx, len(labels), float(pre_rest_seconds))
            row["events"].append({"phase": "pre_rest", "display": "休息", "seconds": float(pre_rest_seconds), "trigger": REST_TRIGGER_CODE})
            emit_trigger(REST_LABEL)
            wait_phase(pre_rest_seconds)
            if stop_event.is_set():
                break

            if cue_callback:
                cue_callback(label, "cue", idx, len(labels), float(cue_seconds))
            row["events"].append({"phase": "cue", "display": info["subject_task"], "seconds": float(cue_seconds), "trigger": None})
            wait_phase(cue_seconds)
            if stop_event.is_set():
                break

            if cue_callback:
                cue_callback(label, "cue_rest", idx, len(labels), float(cue_rest_seconds))
            row["events"].append({"phase": "cue_rest", "display": "休息", "seconds": float(cue_rest_seconds), "trigger": REST_TRIGGER_CODE})
            emit_trigger(REST_LABEL)
            wait_phase(cue_rest_seconds)
            if stop_event.is_set():
                break

            if cue_callback:
                cue_callback(label, "imagery", idx, len(labels), float(epoch_seconds))
            row["events"].append({"phase": "imagery", "display": info["subject_task"], "seconds": float(epoch_seconds), "trigger": MODEL_TRIGGER_CODES[label]})
            emit_trigger(label)
            wait_phase(epoch_seconds)
            if stop_event.is_set():
                break

            if cue_callback:
                cue_callback(label, "post_rest", idx, len(labels), float(post_rest_seconds if post_rest_seconds is not None else rest_seconds))
            row["events"].append(
                {
                    "phase": "post_rest",
                    "display": "休息",
                    "seconds": float(post_rest_seconds if post_rest_seconds is not None else rest_seconds),
                    "trigger": REST_TRIGGER_CODE,
                }
            )
            emit_trigger(REST_LABEL)
            wait_phase(post_rest_seconds if post_rest_seconds is not None else rest_seconds)
            trial_table.append(row)
        time.sleep(0.5)
        packets.extend(recorder.collect_packets())
    finally:
        recorder.stop()

    if recorder.error:
        raise RuntimeError(f"RDA采集失败：{recorder.error}") from recorder.error
    if not packets:
        raise RuntimeError("没有收到 RDA 数据包，请确认 BrainVision Recorder RDA 输出已开启")

    raw = np.concatenate(packets, axis=1).astype(np.float32)
    fs = int(recorder.info.get("fs") or 250)
    eeg, y, trigger_samples = extract_epochs_from_triggered_raw(
        raw,
        fs=fs,
        trigger_to_label=TRIGGER_TO_LABEL,
        epoch_seconds=epoch_seconds,
        n_eeg_channels=expected_channels,
    )
    aux = None
    if aux_channels:
        aux_raw = np.concatenate([raw[expected_channels : expected_channels + aux_channels, :], raw[-1:, :]], axis=0)
        aux, aux_y, _aux_trigger_samples = extract_epochs_from_triggered_raw(
            aux_raw,
            fs=fs,
            trigger_to_label=TRIGGER_TO_LABEL,
            epoch_seconds=epoch_seconds,
            n_eeg_channels=aux_channels,
        )
        if len(aux_y) != len(y):
            log(f"WARNING: AUX epoch count {len(aux_y)} differs from EEG epoch count {len(y)}")
    channel_names = recorder.info.get("channel_names") or [f"Ch{i + 1}" for i in range(eeg.shape[1])]
    if len(channel_names) < signal_channels:
        channel_names = list(channel_names) + [f"Ch{i + 1}" for i in range(len(channel_names), signal_channels)]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    save_payload = {
        "eeg": eeg,
        "y": y,
        "fs": fs,
        "raw": raw,
        "trigger_samples": trigger_samples,
        "channel_names": np.asarray(channel_names[: eeg.shape[1]], dtype=object),
        "trigger_codes": np.asarray([MODEL_TRIGGER_CODES[str(label)] for label in y], dtype=np.int64),
        "timeline_seconds": np.asarray(
            [pre_rest_seconds, cue_seconds, cue_rest_seconds, epoch_seconds, post_rest_seconds],
            dtype=np.float32,
        ),
        "timeline_labels": np.asarray(["rest", "cue", "rest", "imagery", "rest"], dtype=object),
        "trial_order": np.asarray(labels, dtype=object),
        "trial_table_json": json.dumps(trial_table, ensure_ascii=False),
        "paradigm_name": PARADIGM_NAME,
        "paradigm_json": json.dumps(MI8_PARADIGM, ensure_ascii=False),
        "mode": "rda64",
    }
    if aux is not None:
        save_payload["aux"] = aux
        save_payload["aux_channel_names"] = np.asarray(
            channel_names[expected_channels : expected_channels + aux_channels],
            dtype=object,
        )
    np.savez_compressed(out_path, **save_payload)
    return {
        "path": out_path,
        "fs": fs,
        "raw_shape": raw.shape,
        "eeg_shape": eeg.shape,
        "aux_shape": aux.shape if aux is not None else None,
        "labels": y.tolist(),
        "channel_names": channel_names[: eeg.shape[1]],
    }


class NpzReplaySource(SignalSource):
    def __init__(self, path: str, cfg: SignalConfig):
        data = np.load(path, allow_pickle=True)
        self.cfg = cfg
        self.eeg = np.asarray(data["eeg"], dtype=np.float32)
        if self.eeg.ndim == 3:
            self.eeg = np.concatenate(list(self.eeg), axis=-1)
        self.aux = np.asarray(data["aux"], dtype=np.float32) if "aux" in data else None
        if self.aux is not None and self.aux.ndim == 3:
            self.aux = np.concatenate(list(self.aux), axis=-1)
        self.labels = data["labels"] if "labels" in data else (data["y"] if "y" in data else None)
        self.pos = 0

    def read_window(self) -> Tuple[np.ndarray, Optional[np.ndarray], float, Optional[str]]:
        win = self.cfg.window_samples
        if self.pos + win > self.eeg.shape[-1]:
            raise StopIteration
        start = self.pos
        end = start + win
        eeg = self.eeg[:, start:end]
        aux = self.aux[:, start:end] if self.aux is not None else None
        self.pos += self.cfg.step_samples
        label = None
        if self.labels is not None:
            label_idx = min(len(self.labels) - 1, int(start / self.cfg.step_samples))
            raw_label = self.labels[label_idx]
            label = GRIP_TOGGLE_LABEL if is_grip_toggle_event(raw_label) else normalize_label(raw_label)
        return eeg, aux, time.time(), label


class SimulatedSignalSource(SignalSource):
    def __init__(self, cfg: SignalConfig, cycles: int = 2, noise: float = 0.8):
        self.cfg = cfg
        self.noise = noise
        self.index = 0
        self.start_time = time.time()
        cycle = (
            ["rest"] * 5
            + ["up"] * 7
            + ["rest"] * 5
            + ["right"] * 7
            + ["rest"] * 5
            + ["forward"] * 7
            + ["rest"] * 5
            + [GRIP_TOGGLE_LABEL] * 7
            + ["rest"] * 5
            + ["left"] * 7
            + ["rest"] * 5
            + ["down"] * 7
            + ["rest"] * 5
            + ["backward"] * 7
            + ["rest"] * 5
            + [GRIP_TOGGLE_LABEL] * 7
        )
        self.schedule = cycle * cycles
        self.channel_index = {name.upper(): idx for idx, name in enumerate(cfg.eeg_channels)}

    def read_window(self) -> Tuple[np.ndarray, Optional[np.ndarray], float, Optional[str]]:
        if self.index >= len(self.schedule):
            raise StopIteration
        label = self.schedule[self.index]
        now = self.start_time + self.index * self.cfg.step_seconds
        self.index += 1
        eeg, aux = simulate_epoch(label, self.cfg, self.cfg.window_samples, self.noise)
        return eeg, aux, now, label


def simulate_epoch(label: str, cfg: SignalConfig, samples: int, noise: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    grip_toggle = is_grip_toggle_event(label)
    label = REST_LABEL if grip_toggle else normalize_label(label)
    rng = np.random.default_rng()
    eeg = rng.normal(0.0, noise, size=(len(cfg.eeg_channels), samples)).astype(np.float32)
    aux = rng.normal(0.0, noise * 0.45, size=(3, samples)).astype(np.float32)
    t = np.arange(samples, dtype=np.float32) / float(cfg.fs)
    mu = np.sin(2 * np.pi * 10.0 * t)
    beta = np.sin(2 * np.pi * 20.0 * t)
    gamma = np.sin(2 * np.pi * 45.0 * t)
    channel_index = {name.upper(): idx for idx, name in enumerate(cfg.eeg_channels)}

    def add(ch_name: str, signal: np.ndarray, amp: float) -> None:
        idx = channel_index.get(ch_name.upper())
        if idx is not None:
            eeg[idx] += amp * signal

    def add_left_hand_mi() -> None:
        add("C4", beta, 1.05)
        add("FC6", beta, 0.55)
        add("CP2", mu, -0.35)

    def add_right_hand_mi() -> None:
        add("C3", beta, 1.05)
        add("FC5", beta, 0.55)
        add("CP1", mu, -0.35)

    def add_left_masseter() -> None:
        aux[0] += 1.10 * gamma + 0.35 * np.sin(2 * np.pi * 62.0 * t)

    def add_right_masseter() -> None:
        aux[1] += 1.10 * gamma + 0.35 * np.sin(2 * np.pi * 57.0 * t)

    def add_both_masseter() -> None:
        add_left_masseter()
        add_right_masseter()
        aux[2] += 0.40 * np.sin(2 * np.pi * 34.0 * t)

    if grip_toggle:
        eeg *= 0.50
        aux *= 0.25
        blink = np.zeros(samples, dtype=np.float32)
        blink_centers = np.linspace(0.20, 0.80, 3)
        for center in blink_centers:
            blink += np.exp(-0.5 * ((t - center) / 0.035) ** 2).astype(np.float32)
        aux[2] += 2.20 * blink
    elif label == "rest":
        eeg *= 0.55
        aux *= 0.35
    elif label == "up":
        add_both_masseter()
        add_left_hand_mi()
    elif label == "down":
        add_both_masseter()
        add_right_hand_mi()
    elif label == "left":
        add_left_masseter()
        add_left_hand_mi()
    elif label == "right":
        add_left_masseter()
        add_right_hand_mi()
    elif label == "forward":
        add_right_masseter()
        add_left_hand_mi()
    elif label == "backward":
        add_right_masseter()
        add_right_hand_mi()
    return eeg, aux


def build_demo_calibration(cfg: SignalConfig, trials_per_class: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = int(round(cfg.calibration_trial_seconds * cfg.fs))
    eeg_trials: List[np.ndarray] = []
    aux_trials: List[np.ndarray] = []
    labels: List[str] = []
    for label in MODEL_LABELS:
        for _ in range(trials_per_class):
            eeg, aux = simulate_epoch(label, cfg, samples, noise=0.9)
            eeg_trials.append(eeg)
            aux_trials.append(aux)
            labels.append(label)
    return np.stack(eeg_trials), np.asarray(labels, dtype=object), np.stack(aux_trials)


def load_calibration_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[int]]:
    with np.load(path, allow_pickle=True) as data:
        if "eeg" not in data or "y" not in data:
            raise ValueError("Calibration npz must contain arrays: eeg and y")
        eeg = np.asarray(data["eeg"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=object)
        aux = np.asarray(data["aux"], dtype=np.float32) if "aux" in data else None
        fs = int(np.asarray(data["fs"]).reshape(-1)[0]) if "fs" in data else None
    return eeg, y, aux, fs


def timestamped_recording_path(base_path: str = RECORDINGS_DIR, timestamp: Optional[float] = None) -> str:
    value = os.path.abspath(base_path or RECORDINGS_DIR)
    root = os.path.dirname(value) if os.path.splitext(value)[1].lower() == ".npz" else value
    stamp = time.localtime(time.time() if timestamp is None else float(timestamp))
    day_dir = os.path.join(root, time.strftime("%Y-%m-%d", stamp))
    os.makedirs(day_dir, exist_ok=True)
    stem = time.strftime("%H-%M-%S", stamp) + "_session"
    candidate = os.path.join(day_dir, stem + ".npz")
    suffix = 2
    while os.path.exists(candidate):
        candidate = os.path.join(day_dir, f"{stem}_{suffix:02d}.npz")
        suffix += 1
    return candidate


def recording_npz_files(root: str = RECORDINGS_DIR) -> List[str]:
    if not os.path.isdir(root):
        return []
    paths: List[str] = []
    for directory, _subdirs, filenames in os.walk(root):
        paths.extend(os.path.join(directory, name) for name in filenames if name.lower().endswith(".npz"))
    return sorted(paths, key=lambda path: os.path.getmtime(path), reverse=True)


def inspect_recording_npz(path: str) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        if "eeg" not in data or "y" not in data:
            raise ValueError("缺少 eeg 或 y")
        eeg_shape = tuple(int(value) for value in np.asarray(data["eeg"]).shape)
        aux_shape = tuple(int(value) for value in np.asarray(data["aux"]).shape) if "aux" in data else ()
        labels = np.asarray(data["y"], dtype=object).reshape(-1)
        fs = int(np.asarray(data["fs"]).reshape(-1)[0]) if "fs" in data else 0
        mode = str(np.asarray(data["mode"]).reshape(-1)[0]) if "mode" in data else "unknown"
        completed = bool(np.asarray(data["completed"]).reshape(-1)[0]) if "completed" in data else True
    return {
        "path": os.path.abspath(path),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
        "trials": int(labels.size),
        "eeg_channels": int(eeg_shape[1]) if len(eeg_shape) >= 2 else 0,
        "aux_channels": int(aux_shape[1]) if len(aux_shape) >= 2 else 0,
        "samples": int(eeg_shape[-1]) if eeg_shape else 0,
        "fs": fs,
        "mode": mode,
        "completed": completed,
    }


def load_calibration_npz_many(
    paths: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[int], List[str]]:
    selected = [os.path.abspath(path) for path in paths if str(path).strip()]
    if not selected:
        raise ValueError("没有选择采集记录")
    eeg_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    aux_parts: List[np.ndarray] = []
    reference_fs: Optional[int] = None
    reference_names: Optional[List[str]] = None
    has_aux: Optional[bool] = None
    for path in selected:
        eeg, y, aux, fs = load_calibration_npz(path)
        eeg = ensure_3d_epochs(eeg, "eeg")
        names = load_npz_channel_names(path, eeg.shape[1])
        if reference_fs is None:
            reference_fs = fs
            reference_names = names
            has_aux = aux is not None
        if reference_fs and fs and int(fs) != int(reference_fs):
            raise ValueError(f"采样率不一致：{path} 为 {fs} Hz，期望 {reference_fs} Hz")
        if reference_names is not None and names != reference_names:
            raise ValueError(f"EEG 通道名称/顺序不一致：{path}")
        if (aux is not None) != bool(has_aux):
            raise ValueError(f"AUX/肌电字段不一致：{path}")
        if len(y) != len(eeg):
            raise ValueError(f"Trial 与标签数量不一致：{path}")
        eeg_parts.append(eeg)
        y_parts.append(np.asarray(y, dtype=object).reshape(-1))
        if aux is not None:
            aux_parts.append(ensure_3d_epochs(aux, "aux"))
    eeg_all = np.concatenate(eeg_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    aux_all = np.concatenate(aux_parts, axis=0) if aux_parts else None
    return eeg_all, y_all, aux_all, reference_fs, list(reference_names or [])


def load_npz_channel_names(path: str, n_channels: int) -> List[str]:
    try:
        data = np.load(path, allow_pickle=True)
        if "channel_names" in data:
            names = [str(item) for item in data["channel_names"].tolist()]
            if len(names) == n_channels:
                return names
    except Exception:
        pass
    return [f"Ch{i + 1}" for i in range(n_channels)]


