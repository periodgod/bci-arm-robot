# -*- coding: utf-8 -*-
"""
EEG8 robotic-arm BCI core utilities.

Current online workflow:
  1. Window 1 jointly collects 2 s EEG+AUX, maps the masseter state to one
     C1-C8 candidate pair, and down-weights probabilities outside that pair.
  2. AUX1/2 eye channels are used only for explicit three-blink state switching.
  3. Window 2 counts independent jaw executions and only confirms the frozen
     window-1 class; it never remaps or reclassifies that result.
  4. C1-C6 move continuously while fixed window-3 segments count blinks for
     stop_motion; C7-C8 execute once and return to window 1.

Expected calibration .npz format:
  eeg: (trials, eeg_channels, samples)
  y:   (trials,), EEG8 labels C1-C8 for the active paradigm, or legacy labels
  aux: optional (trials, aux_channels, samples), EOG/left masseter/right masseter
  fs:  optional scalar sampling rate
"""

from __future__ import annotations

import json
import os
import random
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral, Real
from queue import Empty, Full, Queue
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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

import numpy as np

try:
    from scipy.signal import butter, iirnotch, sosfilt, sosfiltfilt, filtfilt

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import mne
    from mne.decoding import CSP

    mne.set_log_level("WARNING")
    HAS_MNE_CSP = True
except Exception:
    mne = None
    CSP = None
    HAS_MNE_CSP = False

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(APP_DIR, "data", "recordings")
ACTION_COMMANDS = {
    "up": "move_up",
    "down": "move_down",
    "left": "move_left",
    "right": "move_right",
    "forward": "move_forward",
    "backward": "move_backward",
    "grab": "close_gripper",
    "release": "open_gripper",
    # 姿态微调仅供手动控制，不占用 EEG C1-C8 分类。
    "roll_increase": "rotate_roll_positive",
    "roll_decrease": "rotate_roll_negative",
    "pitch_increase": "rotate_pitch_positive",
    "pitch_decrease": "rotate_pitch_negative",
    "yaw_increase": "rotate_yaw_positive",
    "yaw_decrease": "rotate_yaw_negative",
}

COMMAND_INTENTS = {command: intent for intent, command in ACTION_COMMANDS.items()}

def resolve_robot_intent(command: str = "", intent: str = "") -> str:
    """Normalize GUI command names to one canonical mechanical-arm intent."""
    value = str(intent or "").strip().lower()
    if value in ACTION_COMMANDS or value in {"home", "stop_motion"}:
        return value
    return COMMAND_INTENTS.get(str(command or "").strip().lower(), str(command or "").strip().lower())

CONTINUOUS_MOVE_VECTORS = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}
EEG8_RUNTIME_STRATEGY_MASSETER = "eeg8_two_phase_masseter"
EEG8_RUNTIME_STRATEGY_EEG_ONLY = "eeg8_two_phase_eeg_only"
DEFAULT_EEG8_RUNTIME_STRATEGY = EEG8_RUNTIME_STRATEGY_EEG_ONLY
EEG8_RUNTIME_WINDOW_SECONDS = 2.0
EEG8_CONFIRMATION_MASSETER_EXECUTIONS = 1
# Minimum validation scores for real-arm output.
ROBOT_MIN_EEG_ACCURACY = 0.60
ROBOT_MIN_BALANCED_ACCURACY = 0.60
ROBOT_MIN_MACRO_F1 = 0.55
ROBOT_MIN_KAPPA = 0.45
ROBOT_MIN_CLASS_RECALL = 0.40
# Reserve one 50 ms decision tick after an NCC template is collected.
EEG8_ONLINE_TIMER_INTERVAL_SECONDS = 0.05
EEG8_NCC_CONFIRMATION_SAFETY_MARGIN_SECONDS = EEG8_ONLINE_TIMER_INTERVAL_SECONDS
EEG8_NCC_CONFIRMATION_STATES = frozenset(("01", "10", "11"))
LEGACY_EEG8_RUNTIME_STRATEGY_MAP = {
    # Map retired UI strategies without changing model SignalConfig.
    "eeg8_5s_masseter_gate": EEG8_RUNTIME_STRATEGY_MASSETER,
    "eeg8_5s_eeg_only": EEG8_RUNTIME_STRATEGY_EEG_ONLY,
}
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


EEG8_EEG_CHANNEL_COUNT = 32


def normalize_eeg_channel_name(name: object) -> str:
    """Normalize case and separators for EEG channel comparison."""
    if not isinstance(name, str):
        return ""
    return "".join(char for char in name.strip().casefold() if char.isalnum())


def _is_placeholder_eeg_channel_name(normalized_name: str) -> bool:
    """Return whether a normalized label carries no physical electrode ID."""
    if not normalized_name:
        return True
    # Generic CH/EEG labels do not identify physical electrodes.
    for prefix in ("ch", "channel", "eeg"):
        suffix = normalized_name[len(prefix) :]
        if normalized_name.startswith(prefix) and suffix.isdecimal():
            return True
    return False


def _is_stable_brainvision_channel_name(name: object) -> bool:
    """Accept BrainVision Connector's bank/channel IDs (for example EEG 1_1)."""
    if not isinstance(name, str):
        return False
    return re.fullmatch(r"eeg\s+\d+\s*[_-]\s*\d+", name.strip(), re.IGNORECASE) is not None


def validate_eeg8_online_channel_contract(
    reported_channel_names: Sequence[object],
    trained_eeg_channels: Sequence[object],
) -> Tuple[str, ...]:
    """Verify that LSL and model channel names have the same physical order."""
    try:
        reported = tuple(reported_channel_names)
        trained = tuple(trained_eeg_channels)
    except TypeError as exc:
        raise ValueError(
            "无法验证 EEG 物理通道：LSL 标签和模型 SignalConfig 必须是可迭代的 32 路名称；"
            "请配置 LSL 电极标签或使用匹配标签重新训练模型。"
        ) from exc

    if len(trained) != EEG8_EEG_CHANNEL_COUNT:
        raise ValueError(
            "模型 SignalConfig 未保存完整的 32 路实际 EEG 通道名，无法安全在线解码；"
            "请使用带真实 LSL 电极标签的记录重新训练模型。"
        )
    if len(reported) < EEG8_EEG_CHANNEL_COUNT:
        raise ValueError(
            f"LSL 仅报告 {len(reported)} 路标签，无法验证 EEG8 所需的 32 路物理通道顺序；"
            "请补全 LSL 电极标签或使用匹配标签重新训练模型。"
        )

    trained_keys = tuple(normalize_eeg_channel_name(name) for name in trained)
    reported_keys = tuple(
        normalize_eeg_channel_name(name) for name in reported[:EEG8_EEG_CHANNEL_COUNT]
    )

    for source, names, keys in (
        ("模型 SignalConfig", trained, trained_keys),
        ("LSL", reported[:EEG8_EEG_CHANNEL_COUNT], reported_keys),
    ):
        invalid = [
            index + 1
            for index, (name, key) in enumerate(zip(names, keys))
            if _is_placeholder_eeg_channel_name(key)
            and not _is_stable_brainvision_channel_name(name)
        ]
        if invalid:
            positions = ", ".join(map(str, invalid[:8]))
            suffix = "…" if len(invalid) > 8 else ""
            raise ValueError(
                f"{source} 的 EEG 通道标签缺失或使用 CH/EEG 占位符（第 {positions}{suffix} 路）；"
                "系统不会默认套用蒙太奇。请配置真实 LSL 电极标签，或使用匹配标签重新训练模型。"
            )
        if len(set(keys)) != EEG8_EEG_CHANNEL_COUNT:
            raise ValueError(
                f"{source} 的 32 路 EEG 物理通道标签存在重复，无法验证通道顺序；"
                "请修正 LSL 标签，或使用匹配标签重新训练模型。"
            )

    if reported_keys != trained_keys:
        first_mismatch = next(
            index
            for index, (reported_key, trained_key) in enumerate(zip(reported_keys, trained_keys))
            if reported_key != trained_key
        )
        raise ValueError(
            "LSL 前 32 路 EEG 物理通道名/顺序与模型训练 SignalConfig 不一致"
            f"（第 {first_mismatch + 1} 路：LSL={reported[first_mismatch]!r}，"
            f"模型={trained[first_mismatch]!r}）；系统已安全拒绝在线解码。"
            "请配置与训练记录一致的 LSL 电极标签，或用当前标签重新训练模型。"
        )
    return reported_keys


def brainvision_lsl_aux_indices(channel_count: int, eeg_channels: int = 32, aux_channels: int = 4) -> List[int]:
    total = int(channel_count)
    if total <= int(eeg_channels):
        return []
    available_aux = min(int(aux_channels), total - int(eeg_channels))
    return list(range(int(eeg_channels), int(eeg_channels) + available_aux))


@dataclass
class SignalConfig:
    fs: int = 250
    eeg_channels: List[str] = field(default_factory=lambda: list(DEFAULT_EEG_CHANNELS))
    # Training and online decoding use the persisted window values.
    window_seconds: float = 2.0
    step_seconds: float = 2.0
    aux_channel_names: List[str] = field(default_factory=list)
    masseter_gate_capable: bool = False
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
    command_confidence: float = 0.0
    min_margin: float = 0.0
    # Serialized for backward compatibility; online classification always uses
    # exactly one physical, non-overlapping EEG+AUX acquisition window.
    stable_windows: int = 1
    history_windows: int = 1
    refractory_seconds: float = 0.25
    command_duration_seconds: float = 0.55
    artifact_z_threshold: float = 9.0
    masseter_gate_enabled: bool = True
    masseter_left_count: int = 1
    masseter_right_count: int = 1
    masseter_bite_ratio: float = 3.0
    masseter_side_ratio: float = 1.35
    masseter_detector: str = "ncc"
    masseter_template_path: str = ""
    masseter_ncc_min_score: float = 0.45
    masseter_ncc_min_margin: float = 0.08
    confirmation_window_seconds: float = 1.0
    confirmation_masseter_count: int = 1
    masseter_outside_candidate_weight: float = 0.3
    candidate_random_margin: float = 0.05
    continuous_move_speed: float = 0.5
    continuous_blink_window_seconds: float = 2.0
    continuous_blink_count: int = 3
    # Older joblib models may not contain this field.
    masseter_state_detectors: Dict[str, str] = field(default_factory=dict)

    def validated_eeg_weight(self) -> float:
        return min(0.85, max(0.35, float(self.eeg_weight)))


@dataclass(frozen=True)
class FusionPolicy:
    """Safety settings for an EEG8 online strategy."""

    runtime_strategy: str
    requires_aux: bool
    requires_confirmation: bool
    allows_continuous_motion: bool
    description: str


def resolve_eeg8_fusion_policy(runtime_strategy: object) -> FusionPolicy:
    """Return the only supported EEG8 runtime policy or reject safely."""
    name = str(runtime_strategy or "").strip().lower()
    if name == EEG8_RUNTIME_STRATEGY_MASSETER:
        return FusionPolicy(
            runtime_strategy=EEG8_RUNTIME_STRATEGY_MASSETER,
            requires_aux=True,
            requires_confirmation=True,
            allows_continuous_motion=True,
            description=(
                "窗口1 EEG+EMG 映射分类 -> 窗口2 咬肌计数确认 -> "
                "C1-C6 窗口3分段眨眼停止 / C7-C8 抓取与释放单次执行"
            ),
        )
    if name == EEG8_RUNTIME_STRATEGY_EEG_ONLY:
        return FusionPolicy(
            runtime_strategy=EEG8_RUNTIME_STRATEGY_EEG_ONLY,
            requires_aux=False,
            requires_confirmation=False,
            allows_continuous_motion=False,
            description="EEG 安全分类 -> 一次性有时长限制的命令",
        )
    raise ValueError(
        "未知 EEG8 在线策略："
        f"{runtime_strategy!r}。系统已安全拒绝启动；可选策略为 "
        f"{EEG8_RUNTIME_STRATEGY_MASSETER} 或 {EEG8_RUNTIME_STRATEGY_EEG_ONLY}。"
    )


def migrate_legacy_eeg8_realtime_config(
    raw_config: Optional[Mapping[str, object]],
) -> Tuple[Dict[str, object], Tuple[str, ...]]:
    """Migrate retired 5 s GUI settings without changing the model."""
    config = dict(raw_config or {})
    strategy = str(config.get("runtime_strategy", "") or "").strip().lower()
    migrated_strategy = LEGACY_EEG8_RUNTIME_STRATEGY_MAP.get(strategy)
    if migrated_strategy is None:
        return config, tuple()

    config["runtime_strategy"] = migrated_strategy
    config["decode_window_seconds"] = EEG8_RUNTIME_WINDOW_SECONDS
    config["decode_step_seconds"] = EEG8_RUNTIME_WINDOW_SECONDS
    source = strategy or "未声明策略"
    return config, (
        f"已将旧实时策略 {source} 迁移为 {migrated_strategy}，"
        "GUI 解码窗口固定为 2s/2s。模型窗口未被修改；旧 5s 模型必须重新训练。",
    )


@dataclass(frozen=True)
class FrozenDecoderConfig:
    """Immutable decoder settings for one online session."""

    eeg_weight: float = 0.55
    aux_weight_cap: float = 0.45
    command_confidence: float = 0.0
    min_margin: float = 0.0
    stable_windows: int = 1
    history_windows: int = 1
    refractory_seconds: float = 0.25
    command_duration_seconds: float = 0.55
    artifact_z_threshold: float = 9.0
    masseter_gate_enabled: bool = True
    masseter_left_count: int = 1
    masseter_right_count: int = 1
    masseter_bite_ratio: float = 3.0
    masseter_side_ratio: float = 1.35
    masseter_detector: str = "ncc"
    masseter_template_path: str = ""
    masseter_ncc_min_score: float = 0.45
    masseter_ncc_min_margin: float = 0.08
    confirmation_window_seconds: float = 1.0
    confirmation_masseter_count: int = 1
    masseter_outside_candidate_weight: float = 0.3
    candidate_random_margin: float = 0.05
    continuous_move_speed: float = 0.5
    continuous_blink_window_seconds: float = 2.0
    continuous_blink_count: int = 3
    masseter_state_detectors: Tuple[Tuple[str, str], ...] = tuple()

    def validated_eeg_weight(self) -> float:
        return min(0.85, max(0.35, float(self.eeg_weight)))

    @property
    def state_detector_mapping(self) -> Mapping[str, str]:
        """Return a copy of the immutable detector-routing tuple."""
        return dict(self.masseter_state_detectors)


def freeze_decoder_config(config: object) -> FrozenDecoderConfig:
    """Validate and freeze a DecoderConfig or JSON mapping."""
    defaults = DecoderConfig()
    allowed = set(DecoderConfig.__dataclass_fields__)
    if isinstance(config, Mapping):
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"DecoderConfig 含未知字段：{', '.join(sorted(map(str, unknown)))}")
        raw_values = {name: config.get(name, getattr(defaults, name)) for name in allowed}
    elif isinstance(config, (DecoderConfig, FrozenDecoderConfig)):
        try:
            raw_values = {name: getattr(config, name) for name in allowed}
        except AttributeError as exc:
            raise ValueError("DecoderConfig 缺少当前运行时所需字段") from exc
    else:
        raise ValueError("DecoderConfig 必须是 DecoderConfig、FrozenDecoderConfig 或 Mapping")

    def number(name: str, minimum: float, maximum: Optional[float] = None) -> float:
        value = raw_values[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"DecoderConfig.{name} 必须是有限数值")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < minimum or (maximum is not None and numeric > maximum):
            bounds = f"[{minimum:g}, {maximum:g}]" if maximum is not None else f">= {minimum:g}"
            raise ValueError(f"DecoderConfig.{name} 必须在 {bounds} 范围内")
        return numeric

    def count(name: str, minimum: int = 1) -> int:
        value = raw_values[name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"DecoderConfig.{name} 必须是整数")
        numeric = int(value)
        if numeric < minimum:
            raise ValueError(f"DecoderConfig.{name} 必须 >= {minimum}")
        return numeric

    gate_enabled = raw_values["masseter_gate_enabled"]
    if type(gate_enabled) is not bool:
        raise ValueError("DecoderConfig.masseter_gate_enabled 必须是布尔值")
    template_path = raw_values["masseter_template_path"]
    if not isinstance(template_path, str):
        raise ValueError("DecoderConfig.masseter_template_path 必须是字符串")
    detector = raw_values["masseter_detector"]
    if not isinstance(detector, str) or detector.strip().lower() not in {"threshold", "ncc", "precise"}:
        raise ValueError("DecoderConfig.masseter_detector 仅支持 threshold、ncc 或 precise")
    states_raw = raw_values["masseter_state_detectors"]
    if isinstance(states_raw, tuple):
        try:
            states_raw = dict(states_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("DecoderConfig.masseter_state_detectors 格式无效") from exc
    if not isinstance(states_raw, Mapping):
        raise ValueError("DecoderConfig.masseter_state_detectors 必须是 Mapping")
    allowed_states = {"00", "01", "10", "11"}
    normalized_states = {}
    for key, value in states_raw.items():
        if not isinstance(key, str) or key not in allowed_states:
            raise ValueError("masseter_state_detectors 仅支持状态 00、01、10、11")
        if not isinstance(value, str) or value.strip().lower() not in {"threshold", "ncc"}:
            raise ValueError("masseter_state_detectors 的检测器仅支持 threshold 或 ncc")
        normalized_states[key] = value.strip().lower()

    return FrozenDecoderConfig(
        eeg_weight=number("eeg_weight", 0.0, 1.0),
        aux_weight_cap=number("aux_weight_cap", 0.0, 1.0),
        command_confidence=number("command_confidence", 0.0, 1.0),
        min_margin=number("min_margin", 0.0, 1.0),
        stable_windows=count("stable_windows"),
        history_windows=count("history_windows"),
        refractory_seconds=number("refractory_seconds", 0.0),
        command_duration_seconds=number("command_duration_seconds", 0.01),
        artifact_z_threshold=number("artifact_z_threshold", 0.01),
        masseter_gate_enabled=gate_enabled,
        masseter_left_count=count("masseter_left_count"),
        masseter_right_count=count("masseter_right_count"),
        masseter_bite_ratio=number("masseter_bite_ratio", 1.0),
        masseter_side_ratio=number("masseter_side_ratio", 1.0),
        masseter_detector=detector.strip().lower(),
        masseter_template_path=template_path.strip(),
        masseter_ncc_min_score=number("masseter_ncc_min_score", 0.0, 1.0),
        masseter_ncc_min_margin=number("masseter_ncc_min_margin", 0.0, 1.0),
        confirmation_window_seconds=number("confirmation_window_seconds", 0.2),
        confirmation_masseter_count=count("confirmation_masseter_count", 1),
        masseter_outside_candidate_weight=number("masseter_outside_candidate_weight", 0.0, 1.0),
        candidate_random_margin=number("candidate_random_margin", 0.0, 1.0),
        continuous_move_speed=number("continuous_move_speed", 0.1),
        continuous_blink_window_seconds=number("continuous_blink_window_seconds", 0.5),
        continuous_blink_count=count("continuous_blink_count"),
        masseter_state_detectors=tuple(sorted(normalized_states.items())),
    )


@dataclass(frozen=True)
class FrozenSignalConfig:
    """The full SignalConfig subset consumed by EEG8 inference and LSL input."""

    fs: float
    eeg_channels: Tuple[str, ...]
    window_seconds: float
    step_seconds: float
    notch_hz: float
    use_csp: bool
    csp_components: int
    filter_bands: Tuple[Tuple[float, float], ...]


def freeze_signal_config(config: object) -> FrozenSignalConfig:
    """Validate and freeze every SignalConfig value that affects inference."""
    if not isinstance(config, SignalConfig):
        raise ValueError("模型缺少有效 SignalConfig")
    if isinstance(config.fs, bool) or not isinstance(config.fs, Real) or not np.isfinite(float(config.fs)) or float(config.fs) <= 0.0:
        raise ValueError("SignalConfig.fs 必须为正的有限数值")
    if type(config.use_csp) is not bool:
        raise ValueError("SignalConfig.use_csp 必须是布尔值")
    if isinstance(config.csp_components, bool) or not isinstance(config.csp_components, Integral) or int(config.csp_components) < 1:
        raise ValueError("SignalConfig.csp_components 必须为正整数")
    channels = tuple(config.eeg_channels)
    if len(channels) != 32 or any(not isinstance(name, str) or not name.strip() for name in channels):
        raise ValueError("SignalConfig.eeg_channels 必须是完整的 32 路 EEG 通道名")
    values = {
        "window_seconds": config.window_seconds,
        "step_seconds": config.step_seconds,
        "notch_hz": config.notch_hz,
    }
    normalized = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"SignalConfig.{name} 必须为正的有限数值")
        normalized[name] = float(value)
    bands = []
    try:
        iterable_bands = tuple(config.filter_bands)
    except TypeError as exc:
        raise ValueError("SignalConfig.filter_bands 格式无效") from exc
    if not iterable_bands:
        raise ValueError("SignalConfig.filter_bands 不能为空")
    for band in iterable_bands:
        if not isinstance(band, (tuple, list)) or len(band) != 2:
            raise ValueError("SignalConfig.filter_bands 必须由 (low, high) 二元组组成")
        low, high = band
        if any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)) for value in (low, high)):
            raise ValueError("SignalConfig.filter_bands 必须为有限数值")
        low_f, high_f = float(low), float(high)
        if not 0.0 < low_f < high_f < float(config.fs) / 2.0:
            raise ValueError("SignalConfig.filter_bands 必须满足 0 < low < high < fs/2")
        bands.append((low_f, high_f))
    return FrozenSignalConfig(
        fs=float(config.fs),
        eeg_channels=channels,
        window_seconds=normalized["window_seconds"],
        step_seconds=normalized["step_seconds"],
        notch_hz=normalized["notch_hz"],
        use_csp=config.use_csp,
        csp_components=int(config.csp_components),
        filter_bands=tuple(bands),
    )


@dataclass(frozen=True)
class RuntimeSessionConfig:
    """Immutable online-control settings for one session."""

    runtime_strategy: str
    fusion_policy: FusionPolicy
    model_path: str
    model_version: str
    signal_fs: float
    signal_window_seconds: float
    signal_step_seconds: float
    signal_config: FrozenSignalConfig
    decoder_config: FrozenDecoderConfig
    decoder_config_source: str
    aux_blink_ratio: float
    aux_bite_ratio: float
    aux_side_ratio: float
    aux_blink_count: int
    aux_left_count: int
    aux_right_count: int
    target: str
    robot_enabled: bool
    robot_host: str
    robot_port: int
    robot_speed: float
    continuous_move_speed: float
    confirmation_window_seconds: float
    continuous_blink_window_seconds: float

    @property
    def requires_aux(self) -> bool:
        return bool(self.fusion_policy.requires_aux)

    @property
    def requires_confirmation(self) -> bool:
        return bool(self.fusion_policy.requires_confirmation)

    @property
    def allows_continuous_motion(self) -> bool:
        return bool(self.fusion_policy.allows_continuous_motion)

    def audit_summary(self) -> str:
        return (
            f"策略={self.runtime_strategy}；Decoder 来源={self.decoder_config_source}；"
            f"EEG={self.signal_window_seconds:g}s/{self.signal_step_seconds:g}s@{self.signal_fs:g}Hz；"
            f"目标={self.target}；机械臂={'启用' if self.robot_enabled else '关闭'}"
        )


def build_runtime_session_config(
    raw_config: Mapping[str, object],
    model: object,
    *,
    use_config_thresholds: Optional[bool] = None,
) -> RuntimeSessionConfig:
    """Validate and freeze model and GUI settings for an online session."""
    if model is None:
        raise ValueError("无法建立运行快照：模型为空")
    if not isinstance(raw_config, Mapping):
        raise ValueError("运行配置必须是 Mapping")
    config, _migration_notes = migrate_legacy_eeg8_realtime_config(raw_config)
    policy = resolve_eeg8_fusion_policy(config.get("runtime_strategy"))
    signal = getattr(model, "signal_config", None)
    signal_contract = freeze_signal_config(signal)
    requested_window = config.get("decode_window_seconds", getattr(signal, "window_seconds", None))
    requested_step = config.get("decode_step_seconds", getattr(signal, "step_seconds", None))
    try:
        requested_window_number = float(requested_window)
        requested_step_number = float(requested_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("窗口1 EEG+EMG 联合采集时长无效") from exc
    if not np.isfinite(requested_window_number) or requested_window_number <= 0.0:
        raise ValueError("窗口1 EEG+EMG 联合采集时长必须为正数")
    if not np.isclose(requested_step_number, requested_window_number, rtol=0.0, atol=1e-6):
        raise ValueError("窗口1 必须使用不重叠采集：解码间隔必须等于联合采集时长")
    validate_eeg8_runtime_window_contract(signal, requested_window, requested_step)

    if use_config_thresholds is None:
        use_config_thresholds = config.get("use_config_thresholds", False)
    if type(use_config_thresholds) is not bool:
        raise ValueError("运行配置 use_config_thresholds 必须是布尔值")
    if bool(use_config_thresholds):
        effective_decoder = config.get("decoder")
        decoder_source = "config"
    else:
        effective_decoder = getattr(model, "decoder_config", None)
        decoder_source = "model"
    decoder = freeze_decoder_config(effective_decoder)
    # Legacy models/configs may contain multi-window stability values.  They
    # must not alter the current single-window classification paradigm.
    decoder = replace(decoder, stable_windows=1, history_windows=1)
    if not policy.requires_aux:
        decoder = replace(decoder, masseter_gate_enabled=False)

    def strict_number(name: str, default: float, minimum: float, maximum: Optional[float] = None) -> float:
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"运行配置 {name} 必须是有限数值")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < minimum or (maximum is not None and numeric > maximum):
            raise ValueError(f"运行配置 {name} 不在允许范围内")
        return numeric

    def strict_count(name: str, default: int) -> int:
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"运行配置 {name} 必须是正整数")
        return int(value)

    robot_enabled = config.get("robot_enabled", False)
    if type(robot_enabled) is not bool:
        raise ValueError("运行配置 robot_enabled 必须是布尔值")
    robot_host = config.get("robot_host", "192.168.1.100")
    target = config.get("target", "target_1")
    if not isinstance(robot_host, str) or not robot_host.strip():
        raise ValueError("运行配置 robot_host 不能为空")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("运行配置 target 不能为空")
    robot_port = config.get("robot_port", 9527)
    if isinstance(robot_port, bool) or not isinstance(robot_port, Integral) or not 1 <= int(robot_port) <= 65535:
        raise ValueError("运行配置 robot_port 必须是 1-65535 的整数")

    model_version = str(
        getattr(model, "model_version", "")
        or getattr(model, "version", "")
        or getattr(model, "created_at", "")
        or "未声明"
    )
    return RuntimeSessionConfig(
        runtime_strategy=policy.runtime_strategy,
        fusion_policy=policy,
        model_path=str(config.get("model_path", "") or ""),
        model_version=model_version,
        signal_fs=signal_contract.fs,
        signal_window_seconds=signal_contract.window_seconds,
        signal_step_seconds=signal_contract.step_seconds,
        signal_config=signal_contract,
        decoder_config=decoder,
        decoder_config_source=decoder_source,
        aux_blink_ratio=strict_number("aux_blink_ratio", 4.0, 1.0),
        aux_bite_ratio=strict_number("aux_bite_ratio", decoder.masseter_bite_ratio, 1.0),
        aux_side_ratio=strict_number("aux_side_ratio", decoder.masseter_side_ratio, 1.0),
        aux_blink_count=strict_count("aux_blink_count", 3),
        aux_left_count=strict_count("aux_left_count", decoder.masseter_left_count),
        aux_right_count=strict_count("aux_right_count", decoder.masseter_right_count),
        target=target.strip(),
        robot_enabled=robot_enabled,
        robot_host=robot_host.strip(),
        robot_port=int(robot_port),
        robot_speed=strict_number("robot_speed", 50.0, np.finfo(float).tiny),
        continuous_move_speed=max(0.1, float(decoder.continuous_move_speed)),
        confirmation_window_seconds=max(0.2, float(decoder.confirmation_window_seconds)),
        continuous_blink_window_seconds=max(0.5, float(decoder.continuous_blink_window_seconds)),
    )


def runtime_signal_contract_matches(session: RuntimeSessionConfig, signal_config: object) -> bool:
    """Return whether a mutable model SignalConfig still matches its session."""
    if not isinstance(session, RuntimeSessionConfig):
        return False
    try:
        return freeze_signal_config(signal_config) == session.signal_config
    except ValueError:
        return False


def runtime_session_robot_output(
    session: RuntimeSessionConfig,
) -> Tuple[bool, str, int, float, str]:
    """Return the robot endpoint and target stored in the session."""
    if not isinstance(session, RuntimeSessionConfig):
        raise ValueError("机械臂输出需要 RuntimeSessionConfig")
    return (
        bool(session.robot_enabled),
        str(session.robot_host),
        int(session.robot_port),
        float(session.robot_speed),
        str(session.target),
    )


def validate_eeg8_runtime_window_contract(
    signal_config: SignalConfig,
    decode_window_seconds: float,
    decode_step_seconds: float,
) -> None:
    """Reject different training and online window settings."""
    if signal_config is None:
        raise ValueError("模型缺少 SignalConfig；无法确认窗口1训练时长，请按当前窗口1配置重新训练")
    try:
        trained_window = float(signal_config.window_seconds)
        trained_step = float(signal_config.step_seconds)
        runtime_window = float(decode_window_seconds)
        runtime_step = float(decode_step_seconds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("模型或实时配置的窗口参数无效；请按当前配置重新训练") from exc
    values = (trained_window, trained_step, runtime_window, runtime_step)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("模型或实时配置的窗口参数必须为正数；请按当前配置重新训练")
    if not np.isclose(trained_window, runtime_window, rtol=0.0, atol=1e-6) or not np.isclose(
        trained_step, runtime_step, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            "模型窗口不兼容：模型训练为 "
            f"{trained_window:g}s 窗口 / {trained_step:g}s 间隔，"
            f"实时配置为 {runtime_window:g}s 窗口 / {runtime_step:g}s 间隔。"
            "请按当前实时窗口重新训练模型，系统不会裁剪或补零后直接运行。"
        )


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
class MasseterGateResult:
    side: str
    left_count: int
    right_count: int
    both_count: int
    left_ratio: float
    right_ratio: float
    candidates: Tuple[int, ...]
    reason: str

    @property
    def left_peak(self) -> float:
        """Peak left-channel ratio retained as a compatibility alias."""
        return self.left_ratio

    @property
    def right_peak(self) -> float:
        """Peak right-channel ratio retained as a compatibility alias."""
        return self.right_ratio


@dataclass(frozen=True)
class EmgEventSnapshot:
    """AUX event counts for one runtime phase."""

    phase: str
    blink: int = 0
    left_jaw: int = 0
    right_jaw: int = 0
    both_jaw: int = 0
    jaw_executions: int = 0
    left_peak_ratio: float = 0.0
    right_peak_ratio: float = 0.0
    reason: str = "ok"

    def as_counts(self) -> Dict[str, int]:
        return {
            "blink": int(self.blink),
            "left_jaw": int(self.left_jaw),
            "right_jaw": int(self.right_jaw),
        }


class PhaseEmgEventEngine:
    """Incremental, phase-local AUX event detector.

    Inputs contain all AUX samples collected since the current phase began.
    Detection uses baseline-normalized RMS, rising edges and per-event debounce.
    """

    def __init__(
        self,
        fs: float = 250.0,
        baseline_rms: Optional[Sequence[float]] = None,
        blink_ratio: float = 4.0,
        bite_ratio: float = 3.0,
        side_ratio: float = 1.35,
        rms_window_seconds: float = 0.18,
        rms_hop_seconds: float = 0.09,
        blink_min_gap_seconds: float = 0.18,
        masseter_min_gap_seconds: float = 0.25,
        event_window_seconds: Optional[Mapping[str, float]] = None,
    ):
        self.fs = max(1.0, float(fs or 250.0))
        self.blink_ratio = max(1.0, float(blink_ratio))
        self.bite_ratio = max(1.0, float(bite_ratio))
        self.side_ratio = max(1.0, float(side_ratio))
        self.rms_window_samples = max(8, int(round(max(0.02, float(rms_window_seconds)) * self.fs)))
        self.rms_hop_samples = max(1, int(round(max(0.01, float(rms_hop_seconds)) * self.fs)))
        self.blink_min_gap_seconds = max(0.0, float(blink_min_gap_seconds))
        self.masseter_min_gap_seconds = max(0.0, float(masseter_min_gap_seconds))
        self.event_window_seconds = {
            str(name): max(0.01, float(seconds))
            for name, seconds in dict(event_window_seconds or {}).items()
            if str(name) in {"blink", "left_jaw", "right_jaw", "both_jaw", "jaw"}
            and np.isfinite(float(seconds))
            and float(seconds) > 0.0
        }
        self.baseline_rms: Optional[np.ndarray] = None
        self.set_baseline(baseline_rms)
        self.reset("idle", 0.0)

    def set_baseline(self, baseline_rms: Optional[Sequence[float]]) -> None:
        """Set the calibrated four-channel RMS baseline without resetting events."""
        if baseline_rms is None:
            self.baseline_rms = None
            return
        try:
            baseline = np.asarray(baseline_rms, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            self.baseline_rms = None
            return
        if baseline.size < 4 or not np.all(np.isfinite(baseline[:4])):
            self.baseline_rms = None
            return
        self.baseline_rms = np.maximum(baseline[:4], 1e-6)

    def reset(self, phase: str, started_at: float) -> EmgEventSnapshot:
        """Start a new phase and discard every active counter and edge state."""
        self.phase = str(phase or "idle")
        self.started_at = float(started_at)
        self._seen_samples = 0
        self._next_frame_start = 0
        self._active = {
            "blink": False,
            "left_jaw": False,
            "right_jaw": False,
            "both_jaw": False,
            "jaw": False,
        }
        self._event_times = {
            "blink": [],
            "left_jaw": [],
            "right_jaw": [],
            "both_jaw": [],
            "jaw": [],
        }
        self._last_event_time = {key: float("-inf") for key in self._event_times}
        self._left_peak_ratio = 0.0
        self._right_peak_ratio = 0.0
        self._reason = "ok"
        return self.snapshot()

    def roll_buffer(self, started_at: float) -> EmgEventSnapshot:
        """Rebase the sample cursor while retaining rolling event counts."""
        self.started_at = float(started_at)
        self._seen_samples = 0
        self._next_frame_start = 0
        self._left_peak_ratio = 0.0
        self._right_peak_ratio = 0.0
        self._reason = "ok"
        return self.snapshot()

    def clear_events(self, *names: str) -> EmgEventSnapshot:
        """Clear selected counters after a completed command without touching phase time."""
        selected = tuple(names) or tuple(self._event_times)
        for name in selected:
            if name in self._event_times:
                self._event_times[name] = []
                self._last_event_time[name] = float("-inf")
                self._active[name] = False
        return self.snapshot()

    def record_external_event(self, name: str, timestamp: float) -> EmgEventSnapshot:
        """Record a labelled replay event through the normal debounce path."""
        key = str(name)
        if key not in self._event_times:
            return self.snapshot()
        self._register_edge(key, True, float(timestamp), self._minimum_gap(key))
        self._active[key] = False
        return self.snapshot()

    def snapshot(self) -> EmgEventSnapshot:
        return EmgEventSnapshot(
            phase=str(self.phase),
            blink=len(self._event_times["blink"]),
            left_jaw=len(self._event_times["left_jaw"]),
            right_jaw=len(self._event_times["right_jaw"]),
            both_jaw=len(self._event_times["both_jaw"]),
            jaw_executions=len(self._event_times["jaw"]),
            left_peak_ratio=float(self._left_peak_ratio),
            right_peak_ratio=float(self._right_peak_ratio),
            reason=str(self._reason),
        )

    def update(self, phase_local_aux: Optional[np.ndarray], window_end_timestamp: float) -> EmgEventSnapshot:
        """Consume new samples from a phase-local AUX buffer."""
        self._prune_events(float(window_end_timestamp))
        if phase_local_aux is None or np.size(phase_local_aux) == 0:
            self._reason = "no_aux"
            return self.snapshot()
        try:
            arr = np.asarray(phase_local_aux, dtype=np.float64)
        except (TypeError, ValueError):
            self._reason = "invalid_aux"
            return self.snapshot()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] == 0:
            self._reason = "missing_aux_channels"
            return self.snapshot()
        if not np.all(np.isfinite(arr)):
            self._reason = "invalid_aux"
            return self.snapshot()
        if self.baseline_rms is None:
            self._reason = "missing_aux_baseline"
            return self.snapshot()

        samples = int(arr.shape[1])
        if samples < self._seen_samples:
            # Restart sample cursors if the upstream buffer was truncated.
            self.reset(self.phase, float(window_end_timestamp))
        if samples == self._seen_samples:
            return self.snapshot()

        frame = self.rms_window_samples
        while self._next_frame_start + frame <= samples:
            start = self._next_frame_start
            stop = start + frame
            rms = _window_channel_rms(arr[:, start:stop])
            ratios = rms[:4] / self.baseline_rms[:4]
            if not np.all(np.isfinite(ratios)):
                self._next_frame_start += self.rms_hop_samples
                continue
            left_ratio = float(ratios[2])
            right_ratio = float(ratios[3])
            self._left_peak_ratio = max(self._left_peak_ratio, left_ratio)
            self._right_peak_ratio = max(self._right_peak_ratio, right_ratio)
            event_time = float(window_end_timestamp) - float(samples - (start + frame / 2.0)) / self.fs

            blink_on = bool(min(float(ratios[0]), float(ratios[1])) >= self.blink_ratio)
            left_on = bool(left_ratio >= self.bite_ratio)
            right_on = bool(right_ratio >= self.bite_ratio)
            both_on = bool(left_on and right_on)
            left_event = bool(both_on or (left_on and left_ratio >= right_ratio * self.side_ratio))
            right_event = bool(both_on or (right_on and right_ratio >= left_ratio * self.side_ratio))

            self._register_edge("blink", blink_on, event_time, self.blink_min_gap_seconds)
            self._register_edge("left_jaw", left_event, event_time, self.masseter_min_gap_seconds)
            self._register_edge("right_jaw", right_event, event_time, self.masseter_min_gap_seconds)
            self._register_edge("both_jaw", both_on, event_time, self.masseter_min_gap_seconds)
            # Window 2 counts physical jaw executions, not active channels. A
            # bilateral bite is therefore one confirmation event rather than two.
            self._register_edge("jaw", left_on or right_on, event_time, self.masseter_min_gap_seconds)
            self._next_frame_start += self.rms_hop_samples

        self._seen_samples = samples
        self._prune_events(float(window_end_timestamp))
        self._reason = "ok"
        return self.snapshot()

    def _prune_events(self, timestamp: float) -> None:
        if not np.isfinite(float(timestamp)):
            return
        for name, seconds in self.event_window_seconds.items():
            cutoff = float(timestamp) - float(seconds)
            self._event_times[name] = [
                event_time
                for event_time in self._event_times[name]
                if float(event_time) >= cutoff
            ]

    def _minimum_gap(self, name: str) -> float:
        return self.blink_min_gap_seconds if name == "blink" else self.masseter_min_gap_seconds

    def _register_edge(self, name: str, active: bool, timestamp: float, minimum_gap: float) -> None:
        was_active = bool(self._active[name])
        if active and not was_active:
            if float(timestamp) - float(self._last_event_time[name]) >= float(minimum_gap):
                self._event_times[name].append(float(timestamp))
                self._last_event_time[name] = float(timestamp)
            self._active[name] = True
        elif not active:
            self._active[name] = False


@dataclass
class MasseterTemplateBank:
    fs: float
    template_seconds: float
    states: Dict[str, np.ndarray]


def eeg8_runtime_uses_ncc_confirmation(
    runtime_strategy: object,
    decoder_config: object,
) -> bool:
    """Return whether any window-1 mapping state uses NCC.

    The historical function name is retained for API compatibility.
    """
    policy = resolve_eeg8_fusion_policy(runtime_strategy)
    if not policy.requires_aux or not bool(
        getattr(decoder_config, "masseter_gate_enabled", True)
    ):
        return False
    mode = str(getattr(decoder_config, "masseter_detector", "threshold") or "threshold").strip().lower()
    if mode == "ncc":
        return True
    if mode != "precise":
        return False
    try:
        state_detectors = {
            str(key): str(value or "threshold").strip().lower()
            for key, value in dict(
                getattr(decoder_config, "masseter_state_detectors", {}) or {}
            ).items()
        }
    except (TypeError, ValueError):
        return False
    return any(
        state_detectors.get(state, "threshold") == "ncc"
        for state in EEG8_NCC_CONFIRMATION_STATES
    )


def validate_eeg8_ncc_confirmation_timing(
    runtime_strategy: object,
    decoder_config: object,
    template_bank: Optional[MasseterTemplateBank],
    *,
    classification_window_seconds: Optional[float] = None,
    safety_margin_seconds: float = EEG8_NCC_CONFIRMATION_SAFETY_MARGIN_SECONDS,
) -> None:
    """Require a window-1 NCC mapping template to fit the joint window.

    The public name is retained for saved configurations and older callers. NCC
    is no longer evaluated in window 2; that window only counts jaw executions.
    """
    if template_bank is None or not eeg8_runtime_uses_ncc_confirmation(
        runtime_strategy, decoder_config
    ):
        return
    try:
        template_seconds = float(template_bank.template_seconds)
        classification_seconds = float(
            EEG8_RUNTIME_WINDOW_SECONDS
            if classification_window_seconds is None
            else classification_window_seconds
        )
        margin_seconds = float(safety_margin_seconds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "NCC 模板或窗口1时长无效；请缩短或重新采集 NCC 模板。"
        ) from exc
    values = (template_seconds, classification_seconds, margin_seconds)
    if not all(np.isfinite(value) for value in values) or template_seconds <= 0.0 or classification_seconds <= 0.0 or margin_seconds <= 0.0:
        raise ValueError(
            "NCC 模板或窗口1时长无效；请缩短或重新采集 NCC 模板。"
        )
    if template_seconds + margin_seconds >= classification_seconds:
        raise ValueError(
            "NCC 模板时长与窗口1不兼容："
            f"模板 {template_seconds:g}s + 安全裕量 {margin_seconds:g}s 必须严格小于 "
            f"窗口1 {classification_seconds:g}s；"
            "请缩短或重新采集 NCC 模板。"
        )


@dataclass
class RobotCommand:
    name: str
    intent: str
    target: str
    confidence: float
    sequence_id: int
    timestamp: float
    duration_seconds: float = 0.55
    continuous: bool = False
    move_speed: float = 0.0
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Optional speed in the Ubuntu receiver's protocol units. A positive value
    # overrides the client's default translation speed for this command only.
    protocol_speed: float = 0.0


def robot_command_payload(command: RobotCommand) -> Dict[str, object]:
    """Build the single GUI/VR/TCP payload for one accepted robot command."""
    if not isinstance(command, RobotCommand):
        raise TypeError("command 必须是 RobotCommand")
    payload = asdict(command)
    payload["cmd"] = payload.pop("name")
    return payload


class CommandPayloadFanout:
    """Send one command payload to each registered output."""

    def __init__(self, *consumers: Callable[[Dict[str, object]], None]):
        if not consumers:
            raise ValueError("命令分发至少需要一个输出端")
        if any(not callable(consumer) for consumer in consumers):
            raise TypeError("命令分发输出端必须可调用")
        self._consumers = tuple(consumers)

    def dispatch(self, payload: Dict[str, object]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("命令 payload 必须是 dict")
        for consumer in self._consumers:
            consumer(payload)


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


def make_classifier(kind: str = "logreg"):
    kind = str(kind or "logreg").strip().lower()
    if kind in {"best_mi_fusion", "mi_fusion", "eegnet_shallow_fusion"}:
        from .domain.mi_fusion import BestMIFusionClassifier

        return BestMIFusionClassifier()
    if kind in {"lda", "fbcsp-lda", "fbcsp_lda"}:
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if kind in {"svm", "rbf_svm", "svc"}:
        return make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced"),
        )
    if kind in {"extra_trees", "extratrees", "et"}:
        return ExtraTreesClassifier(
            n_estimators=220,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
        )
    if kind in {"random_forest", "rf"}:
        return RandomForestClassifier(
            n_estimators=220,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
        )
    if kind in {"gaussian_nb", "gnb", "nb"}:
        return make_pipeline(StandardScaler(), GaussianNB())
    if kind in {"vote", "vote-ml-best", "voting"}:
        return VotingClassifier(
            estimators=[
                ("lda", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
                (
                    "logreg",
                    make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=1200, class_weight="balanced", solver="lbfgs"),
                    ),
                ),
                (
                    "rbf_svm",
                    make_pipeline(
                        StandardScaler(),
                        SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced"),
                    ),
                ),
                (
                    "extra_trees",
                    ExtraTreesClassifier(
                        n_estimators=160,
                        random_state=42,
                        class_weight="balanced",
                        max_features="sqrt",
                        min_samples_leaf=1,
                    ),
                ),
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=160,
                        random_state=42,
                        class_weight="balanced",
                        max_features="sqrt",
                        min_samples_leaf=1,
                    ),
                ),
                ("gaussian_nb", make_pipeline(StandardScaler(), GaussianNB())),
            ],
            voting="soft",
            weights=[2, 1, 1, 2, 1, 1],
            n_jobs=1,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )


def _window_channel_rms(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    centered = arr - np.median(arr, axis=1, keepdims=True)
    return np.sqrt(np.mean(np.square(centered), axis=1))


def _masseter_envelope(aux_window: np.ndarray, fs: float, left_index: int = 2, right_index: int = 3) -> np.ndarray:
    arr = np.asarray(aux_window, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.float64)
    if left_index >= arr.shape[0] or right_index >= arr.shape[0]:
        return np.zeros((2, 0), dtype=np.float64)
    fs = max(1.0, float(fs or 250.0))
    pair = arr[[left_index, right_index], :]
    pair = pair - np.median(pair, axis=1, keepdims=True)
    rectified = np.abs(pair)
    smooth = max(3, int(round(fs * 0.04)))
    if smooth % 2 == 0:
        smooth += 1
    kernel = np.ones(smooth, dtype=np.float64) / float(smooth)
    env = np.vstack([np.convolve(ch, kernel, mode="same") for ch in rectified])
    # Keep the amplitude ratio between the two masseter channels.
    scale = float(np.percentile(env, 95))
    return env / max(scale, 1e-9)


def _eye_envelope(aux_window: np.ndarray, fs: float, left_index: int = 0, right_index: int = 1) -> np.ndarray:
    arr = np.asarray(aux_window, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.float64)
    if left_index >= arr.shape[0] or right_index >= arr.shape[0]:
        return np.zeros((2, 0), dtype=np.float64)
    fs = max(1.0, float(fs or 250.0))
    pair = arr[[left_index, right_index], :]
    pair = pair - np.median(pair, axis=1, keepdims=True)
    smooth = max(3, int(round(fs * 0.03)))
    if smooth % 2 == 0:
        smooth += 1
    kernel = np.ones(smooth, dtype=np.float64) / float(smooth)
    env = np.vstack([np.convolve(ch, kernel, mode="same") for ch in pair])
    scale = np.percentile(np.abs(env), 95, axis=1, keepdims=True)
    return env / np.maximum(scale, 1e-9)


def normalized_cross_correlation(signal: np.ndarray, template: np.ndarray) -> float:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    y = np.asarray(template, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n < 4:
        return 0.0
    x = x[:n] - float(np.mean(x[:n]))
    y = y[:n] - float(np.mean(y[:n]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _resample_template(template: np.ndarray, target_samples: int) -> np.ndarray:
    arr = np.asarray(template, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] == target_samples:
        return arr
    if arr.shape[1] <= 1 or target_samples <= 1:
        return np.resize(arr, (arr.shape[0], max(1, target_samples)))
    old_x = np.linspace(0.0, 1.0, arr.shape[1])
    new_x = np.linspace(0.0, 1.0, target_samples)
    return np.vstack([np.interp(new_x, old_x, row) for row in arr])


def load_masseter_template_bank(path: str) -> Optional[MasseterTemplateBank]:
    path = os.path.abspath(str(path or "").strip())
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            fs = float(np.asarray(data.get("fs", 250.0)).reshape(-1)[0])
            template_seconds = float(np.asarray(data.get("template_seconds", 1.0)).reshape(-1)[0])
            states = {
                key[9:]: np.asarray(data[key], dtype=np.float64)
                for key in data.files
                if key.startswith("template_")
            }
        return MasseterTemplateBank(fs=fs, template_seconds=template_seconds, states=states)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    fs = float(raw.get("fs", 250.0))
    template_seconds = float(raw.get("template_seconds", 1.0))
    states = {
        str(key): np.asarray(value, dtype=np.float64)
        for key, value in dict(raw.get("states", {}) or {}).items()
    }
    return MasseterTemplateBank(fs=fs, template_seconds=template_seconds, states=states)


def save_masseter_template_bank(path: str, bank: MasseterTemplateBank) -> str:
    path = os.path.abspath(str(path or "").strip())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    states = {str(key): np.asarray(value, dtype=np.float64) for key, value in bank.states.items()}
    if ext == ".npz":
        payload = {
            "fs": np.asarray([float(bank.fs)], dtype=np.float64),
            "template_seconds": np.asarray([float(bank.template_seconds)], dtype=np.float64),
        }
        payload.update({f"template_{key}": value for key, value in states.items()})
        np.savez_compressed(path, **payload)
        return path
    payload = {
        "fs": float(bank.fs),
        "template_seconds": float(bank.template_seconds),
        "states": {key: value.tolist() for key, value in states.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def make_masseter_template(
    aux_windows: Sequence[np.ndarray],
    fs: float,
    template_seconds: float = 1.0,
    left_index: int = 2,
    right_index: int = 3,
) -> np.ndarray:
    fs = max(1.0, float(fs or 250.0))
    target_samples = max(8, int(round(float(template_seconds or 1.0) * fs)))
    envelopes = []
    for aux in aux_windows:
        env = _masseter_envelope(aux, fs=fs, left_index=left_index, right_index=right_index)
        if env.shape[1] == 0:
            continue
        if env.shape[1] <= target_samples:
            envelopes.append(_resample_template(env, target_samples))
            continue
        event_strength = np.max(env, axis=0)
        peak = int(np.argmax(event_strength))
        pre_samples = int(round(target_samples * 0.375))
        start = max(0, min(env.shape[1] - target_samples, peak - pre_samples))
        envelopes.append(env[:, start : start + target_samples])
    if not envelopes:
        return np.zeros((2, target_samples), dtype=np.float64)
    return np.median(np.stack(envelopes, axis=0), axis=0)


def make_eye_template(
    aux_windows: Sequence[np.ndarray],
    fs: float,
    template_seconds: float = 1.0,
    left_index: int = 0,
    right_index: int = 1,
) -> np.ndarray:
    fs = max(1.0, float(fs or 250.0))
    target_samples = max(8, int(round(float(template_seconds or 1.0) * fs)))
    envelopes = []
    for aux in aux_windows:
        env = _eye_envelope(aux, fs=fs, left_index=left_index, right_index=right_index)
        if env.shape[1] == 0:
            continue
        if env.shape[1] <= target_samples:
            envelopes.append(_resample_template(env, target_samples))
            continue
        event_strength = np.max(env, axis=0)
        peak = int(np.argmax(event_strength))
        pre_samples = int(round(target_samples * 0.375))
        start = max(0, min(env.shape[1] - target_samples, peak - pre_samples))
        envelopes.append(env[:, start : start + target_samples])
    if not envelopes:
        return np.zeros((2, target_samples), dtype=np.float64)
    return np.median(np.stack(envelopes, axis=0), axis=0)


def match_masseter_ncc(
    aux_window: Optional[np.ndarray],
    fs: float,
    template_bank: Optional[MasseterTemplateBank],
    min_score: float = 0.45,
    min_margin: float = 0.08,
    left_index: int = 2,
    right_index: int = 3,
) -> MasseterGateResult:
    """Match a phase-local masseter window against calibrated NCC templates.

    The input window and every candidate template represent physical time.  A
    template may be resampled from its *declared calibration sampling rate* to
    the online sampling rate, but a short online window is never stretched to
    make it look like a complete template.  That would make the NCC score
    depend on an artificial time scale and is unsafe for command gating.
    """

    def reject(reason: str) -> MasseterGateResult:
        return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), reason)

    if aux_window is None or np.size(aux_window) == 0:
        return reject("no_aux")
    if template_bank is None or not template_bank.states:
        return reject("ncc_no_templates")
    try:
        input_fs = float(fs)
    except (TypeError, ValueError):
        return reject("ncc_invalid_input_fs")
    if not np.isfinite(input_fs) or input_fs <= 0.0:
        return reject("ncc_invalid_input_fs")
    try:
        template_fs = float(template_bank.fs)
    except (TypeError, ValueError):
        return reject("ncc_invalid_template_bank_fs")
    if not np.isfinite(template_fs) or template_fs <= 0.0:
        return reject("ncc_invalid_template_bank_fs")
    try:
        template_seconds = float(template_bank.template_seconds)
    except (TypeError, ValueError):
        return reject("ncc_invalid_template_duration")
    if not np.isfinite(template_seconds) or template_seconds <= 0.0:
        return reject("ncc_invalid_template_duration")

    template_samples_at_bank_fs = int(round(template_seconds * template_fs))
    template_samples_at_input_fs = int(round(template_seconds * input_fs))
    if template_samples_at_bank_fs < 8 or template_samples_at_input_fs < 8:
        return reject("ncc_invalid_template_duration")

    try:
        aux_arr = np.asarray(aux_window, dtype=np.float64)
    except (TypeError, ValueError):
        return reject("ncc_invalid_aux")
    if aux_arr.ndim == 1:
        aux_arr = aux_arr.reshape(1, -1)
    if aux_arr.ndim != 2 or aux_arr.shape[1] == 0 or not np.all(np.isfinite(aux_arr)):
        return reject("ncc_invalid_aux")
    if left_index < 0 or right_index < 0:
        return reject("ncc_invalid_masseter_indices")
    if left_index >= aux_arr.shape[0] or right_index >= aux_arr.shape[0]:
        return reject("missing_masseter_channels")
    if aux_arr.shape[1] < template_samples_at_input_fs:
        return reject(
            "ncc_input_shorter_than_template_"
            f"{aux_arr.shape[1]}_of_{template_samples_at_input_fs}_samples"
        )

    # Stored sample count must match the declared duration and rate.
    templates: Dict[str, np.ndarray] = {}
    for state, template in template_bank.states.items():
        key = str(state)
        if key not in MASSETER_BINARY_CLASS_MAP or key == "00":
            continue
        try:
            stored = np.asarray(template, dtype=np.float64)
        except (TypeError, ValueError):
            return reject(f"ncc_incompatible_template_{key}_values")
        if stored.ndim != 2 or stored.shape[0] < 2:
            return reject(f"ncc_incompatible_template_{key}_channels")
        if stored.shape[1] != template_samples_at_bank_fs:
            return reject(
                f"ncc_incompatible_template_{key}_samples_"
                f"{stored.shape[1]}_expected_{template_samples_at_bank_fs}"
            )
        if not np.all(np.isfinite(stored)):
            return reject(f"ncc_incompatible_template_{key}_values")
        templates[key] = stored[:2, :]
    if not templates:
        return reject("ncc_no_valid_templates")

    env = _masseter_envelope(aux_arr, fs=input_fs, left_index=left_index, right_index=right_index)
    if env.shape[1] == 0:
        return reject("missing_masseter_channels")
    scores: Dict[str, float] = {}
    best_offsets: Dict[str, int] = {}
    hop = max(1, int(round(input_fs * 0.02)))
    starts = list(range(0, env.shape[1] - template_samples_at_input_fs + 1, hop))
    last_start = env.shape[1] - template_samples_at_input_fs
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    candidate_windows = [
        (start, env[:, start : start + template_samples_at_input_fs]) for start in starts
    ]
    for key, template in templates.items():
        # Resample the calibration template, not the online signal.
        tpl = _resample_template(template, template_samples_at_input_fs)
        window_scores = [
            normalized_cross_correlation(window.reshape(-1), tpl.reshape(-1))
            for _start, window in candidate_windows
        ]
        best_index = int(np.argmax(window_scores))
        scores[key] = float(window_scores[best_index])
        best_offsets[key] = int(candidate_windows[best_index][0])
    if not scores:
        return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), "ncc_no_valid_templates")
    if len(scores) < 2:
        available = ",".join(sorted(scores))
        return MasseterGateResult(
            "00",
            0,
            0,
            0,
            float(scores.get("10", 0.0)),
            float(scores.get("01", 0.0)),
            masseter_candidate_indices("00"),
            f"ncc_incomplete_templates_{available or 'none'}",
        )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_state, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = float(best_score - second_score)
    if best_score < float(min_score) or margin < float(min_margin):
        side = "00"
        reason = f"ncc_uncertain_best_{best_state}_{best_score:.3f}_margin_{margin:.3f}"
    else:
        side = best_state
        offset_seconds = float(best_offsets.get(best_state, 0)) / input_fs
        rate_note = ""
        if not np.isclose(template_fs, input_fs, rtol=0.0, atol=1e-9):
            rate_note = f"_template_fs_{template_fs:g}_to_input_fs_{input_fs:g}"
        reason = (
            f"ncc_masseter_binary_{side}_score_{best_score:.3f}_"
            f"margin_{margin:.3f}_offset_{offset_seconds:.3f}s{rate_note}"
        )
    left_count = 1 if side in {"10", "11"} else 0
    right_count = 1 if side in {"01", "11"} else 0
    both_count = 1 if side == "11" else 0
    return MasseterGateResult(
        side,
        left_count,
        right_count,
        both_count,
        float(scores.get("10", 0.0)),
        float(scores.get("01", 0.0)),
        masseter_candidate_indices(side),
        reason,
    )


MASSETER_BINARY_CLASS_MAP: Dict[str, Tuple[int, ...]] = {
    "00": (2, 3),  # C3 双手想象, C4 双脚想象
    "01": (0, 1),  # C1 左手想象, C2 右手想象
    "10": (4, 5),  # C5 左手执行, C6 右手执行
    "11": (6, 7),  # C7 抓取, C8 释放
}


def masseter_binary_state(
    left_count: int,
    right_count: int,
    left_target: int = 1,
    right_target: int = 1,
    both_count: int = 0,
) -> str:
    if int(both_count) > 0:
        return "11"
    left_bit = "1" if int(left_count) >= max(1, int(left_target)) else "0"
    right_bit = "1" if int(right_count) >= max(1, int(right_target)) else "0"
    return left_bit + right_bit


def masseter_candidate_indices(side: str) -> Tuple[int, ...]:
    key = str(side or "00").strip().lower()
    if key in MASSETER_BINARY_CLASS_MAP:
        return MASSETER_BINARY_CLASS_MAP[key]
    if key == "none":
        # Missing AUX is invalid, not a neutral 00 state.
        return tuple()
    if key == "right":
        return MASSETER_BINARY_CLASS_MAP["01"]
    if key == "left":
        return MASSETER_BINARY_CLASS_MAP["10"]
    if key == "both":
        return MASSETER_BINARY_CLASS_MAP["11"]
    return tuple()


def is_eeg_only_runtime_strategy(runtime_strategy: object) -> bool:
    """Return whether the runtime operates without AUX input."""
    return str(runtime_strategy or "").strip().lower() == EEG8_RUNTIME_STRATEGY_EEG_ONLY


def eeg8_command_is_continuous(intent: str, runtime_strategy: object) -> bool:
    """Allow continuous movement only when AUX stop confirmation is available."""
    try:
        policy = resolve_eeg8_fusion_policy(runtime_strategy)
    except ValueError:
        return False
    return str(intent) in CONTINUOUS_MOVE_VECTORS and policy.allows_continuous_motion


def resolve_eeg8_confirmation_action(
    runtime_strategy: object,
    has_pending_confirmation: bool,
    confirmation_started_monotonic: Optional[float],
    now_monotonic: float,
    confirmation_window_seconds: float,
    gate_side: object,
    gate_reason: object,
    gate_observed_before_deadline: bool = False,
    masseter_execution_count: Optional[int] = None,
    required_masseter_executions: int = EEG8_CONFIRMATION_MASSETER_EXECUTIONS,
) -> Tuple[str, str]:
    """Resolve window-2 confirmation without changing window-1's class."""
    if not bool(has_pending_confirmation):
        return "none", "no_pending_confirmation"
    try:
        policy = resolve_eeg8_fusion_policy(runtime_strategy)
    except ValueError:
        return "cancel", "unknown_runtime_strategy"
    if not policy.requires_confirmation:
        return "execute", "eeg_only_direct_execute"

    try:
        started_at = float(confirmation_started_monotonic)
        timestamp = float(now_monotonic)
        window_seconds = max(0.2, float(confirmation_window_seconds))
    except (TypeError, ValueError):
        return "cancel", "confirmation_invalid_timing"
    if not np.isfinite(started_at) or not np.isfinite(timestamp) or started_at <= 0.0:
        return "cancel", "confirmation_invalid_timing"

    side = str(gate_side or "none").strip().lower()
    reason = str(gate_reason or "unknown_gate").strip() or "unknown_gate"
    if masseter_execution_count is None:
        # Window 2 has a strict count contract. A side label alone must never
        # confirm the frozen window-1 class.
        execution_count = 0
        required_count = max(1, int(required_masseter_executions))
        valid_confirmation = False
        confirmation_suffix = f"count_{execution_count}_of_{required_count}"
    else:
        try:
            execution_count = max(0, int(masseter_execution_count))
            required_count = max(1, int(required_masseter_executions))
        except (TypeError, ValueError):
            return "cancel", "confirmation_invalid_masseter_count"
        valid_confirmation = execution_count >= required_count
        confirmation_suffix = f"count_{execution_count}_of_{required_count}"
    elapsed = max(0.0, timestamp - started_at)
    if elapsed >= window_seconds:
        # A delayed tick may use a source slice clipped to the deadline.
        if bool(gate_observed_before_deadline) and valid_confirmation:
            return "execute", f"confirmed_by_masseter_{confirmation_suffix}_at_deadline"
        return "cancel", f"confirmation_timeout_no_valid_masseter_{reason}"
    if valid_confirmation:
        return "execute", f"confirmed_by_masseter_{confirmation_suffix}"
    return "wait", f"awaiting_confirmation_{confirmation_suffix}_{reason}"


def aux_stream_watchdog_expired(
    last_received_monotonic: object,
    now_monotonic: object,
    timeout_seconds: float = 0.5,
) -> bool:
    """Return whether a live AUX stream is too stale for continuous motion."""
    try:
        last_received = float(last_received_monotonic)
        now = float(now_monotonic)
        timeout = max(0.05, float(timeout_seconds))
    except (TypeError, ValueError):
        return True
    if not np.isfinite(last_received) or not np.isfinite(now) or last_received <= 0.0:
        return True
    return max(0.0, now - last_received) > timeout


def resolve_continuous_stop_window(
    blink_count: object,
    required_blinks: object,
    elapsed_seconds: object,
    window_seconds: object,
) -> Tuple[str, str]:
    """Resolve one fixed, non-overlapping window-3 blink segment."""
    try:
        count = max(0, int(blink_count))
        required = max(1, int(required_blinks))
        elapsed = max(0.0, float(elapsed_seconds))
        window = max(0.1, float(window_seconds))
    except (TypeError, ValueError):
        return "stop", "invalid_window3_state"
    if not np.isfinite(elapsed) or not np.isfinite(window):
        return "stop", "invalid_window3_timing"
    if elapsed >= window:
        if count >= required:
            return "stop", f"window3_segment_complete_with_blink_target_{count}_of_{required}"
        return "next_window", f"window3_segment_complete_{count}_of_{required}"
    if count >= required:
        return "wait", f"window3_blink_target_pending_boundary_{count}_of_{required}"
    return "wait", f"window3_collecting_{count}_of_{required}"


def select_binary_candidate(
    probabilities: np.ndarray,
    candidates: Sequence[int],
    random_margin: float = 0.05,
    chooser=None,
) -> Tuple[Optional[int], int, bool, float]:
    """Randomly select inside the mapped pair when its probability gap is small."""
    proba = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    valid = tuple(int(idx) for idx in candidates if 0 <= int(idx) < proba.size)
    if not valid:
        raise ValueError("No valid candidate classes")
    ranked = tuple(sorted(valid, key=lambda idx: float(proba[idx]), reverse=True))
    if len(ranked) == 1:
        return ranked[0], ranked[0], False, 0.0
    gap = abs(float(proba[ranked[0]]) - float(proba[ranked[1]]))
    if len(valid) == 2 and gap <= max(0.0, float(random_margin)):
        pick = chooser or random.choice
        selected = int(pick(valid))
        second = int(valid[1] if selected == valid[0] else valid[0])
        return selected, second, True, gap
    return ranked[0], ranked[1], False, gap


def eeg8_command_eligibility(
    history: Sequence[Tuple[str, float, float, float]],
    intent: str,
    config: object,
) -> Tuple[bool, str]:
    """Validate the class selected from exactly one EEG+AUX acquisition window.

    Confidence and class margin are measurements only.  They must be finite, but
    do not gate entry into the confirmation phase.
    """
    recent = list(history)[-1:]
    if not recent:
        return False, "waiting_for_single_eeg_window"
    record = recent[0]
    try:
        recorded_intent = str(record[0])
        confidence = float(record[1])
        margin = float(record[2])
    except (IndexError, TypeError, ValueError):
        return False, "invalid_eeg8_history_record"
    if recorded_intent != str(intent):
        return False, "eeg8_intent_mismatch"
    if not np.isfinite(confidence) or not np.isfinite(margin):
        return False, "invalid_eeg8_history_record"
    return True, "eeg8_single_window_eligible"


def integrate_velocity_position(
    position: Sequence[float],
    direction: Sequence[float],
    speed: float,
    dt: float,
) -> Tuple[float, float, float]:
    """Integrate one continuous-motion frame without grid rounding."""
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    axis = np.asarray(direction, dtype=np.float64).reshape(-1)
    if pos.size != 3 or axis.size != 3:
        raise ValueError("Continuous motion requires three-dimensional vectors")
    result = pos + axis * max(0.0, float(speed)) * max(0.0, float(dt))
    return tuple(float(value) for value in result)


def apply_masseter_candidate_weight(
    probabilities: np.ndarray,
    candidates: Sequence[int],
    outside_weight: float = 0.3,
) -> np.ndarray:
    proba = np.asarray(probabilities, dtype=np.float64).reshape(-1).copy()
    if proba.size == 0:
        return proba
    weight = float(outside_weight)
    if not np.isfinite(weight):
        weight = 0.3
    weight = min(1.0, max(0.0, weight))
    candidate_set = {int(i) for i in candidates if 0 <= int(i) < proba.size}
    if not candidate_set:
        return proba
    for idx in range(proba.size):
        if idx not in candidate_set:
            proba[idx] *= weight
    return proba


def count_masseter_events(
    aux_window: Optional[np.ndarray],
    fs: float,
    baseline_rms: Optional[Sequence[float]] = None,
    bite_ratio: float = 3.0,
    side_ratio: float = 1.35,
    left_target: int = 1,
    right_target: int = 1,
    left_index: int = 2,
    right_index: int = 3,
) -> MasseterGateResult:
    """Count a phase-local AUX window with the runtime event engine."""
    if aux_window is None or np.size(aux_window) == 0:
        return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), "no_aux")

    arr = np.asarray(aux_window, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), "aux_window_too_short")
    if left_index >= arr.shape[0] or right_index >= arr.shape[0]:
        return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), "missing_masseter_channels")

    fs = max(1.0, float(fs or 250.0))
    left_target = max(1, int(left_target))
    right_target = max(1, int(right_target))

    base = None
    if baseline_rms is not None:
        base = np.asarray(baseline_rms, dtype=np.float64).reshape(-1)
    if base is None or base.size <= max(left_index, right_index) or not np.all(np.isfinite(base)):
        base = _window_channel_rms(arr)
    if base.size <= max(left_index, right_index):
        return MasseterGateResult("none", 0, 0, 0, 0.0, 0.0, tuple(), "missing_masseter_baseline")
    engine = PhaseEmgEventEngine(
        fs=fs,
        baseline_rms=base,
        bite_ratio=bite_ratio,
        side_ratio=side_ratio,
        # Match the online 250 ms jaw debounce.
        masseter_min_gap_seconds=0.25,
    )
    engine.reset("offline_window", 0.0)
    snapshot = engine.update(arr, float(arr.shape[1]) / fs)
    left_count = int(snapshot.left_jaw)
    right_count = int(snapshot.right_jaw)
    both_reached = int(left_count >= left_target and right_count >= right_target)
    side = masseter_binary_state(left_count, right_count, left_target, right_target, both_reached)
    reason = f"masseter_binary_{side}"
    return MasseterGateResult(
        side,
        int(left_count),
        int(right_count),
        int(both_reached),
        float(snapshot.left_peak_ratio),
        float(snapshot.right_peak_ratio),
        masseter_candidate_indices(side),
        reason,
    )


def robust_center_mad(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)) * 1.4826)
    return center, max(mad, 1e-6)


def stabilize_gaussian_nb_estimators(estimator) -> None:
    if estimator is None:
        return
    if isinstance(estimator, GaussianNB) and hasattr(estimator, "var_"):
        var = np.asarray(estimator.var_, dtype=np.float64)
        floor = max(float(getattr(estimator, "epsilon_", 0.0) or 0.0), 1e-9)
        estimator.var_ = np.where(np.isfinite(var) & (var > floor), var, floor)
        estimator.epsilon_ = floor
        return
    for _, step in getattr(estimator, "steps", []):
        stabilize_gaussian_nb_estimators(step)
    for child in getattr(estimator, "estimators_", []) or []:
        stabilize_gaussian_nb_estimators(child)
    for child in getattr(estimator, "named_estimators_", {}).values():
        stabilize_gaussian_nb_estimators(child)


def align_proba(proba: np.ndarray, classes: Sequence[int], n_classes: Optional[int] = None) -> np.ndarray:
    if len(classes) == 0:
        raise ValueError("分类器没有输出类别")
    n_classes = max(int(n_classes or 0), int(max(classes)) + 1)
    out = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
    for src_idx, cls in enumerate(classes):
        if 0 <= int(cls) < n_classes:
            out[:, int(cls)] = proba[:, src_idx]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0.0] = 0.0
    row_sum = np.sum(out, axis=1, keepdims=True)
    empty_rows = (~np.isfinite(row_sum[:, 0])) | (row_sum[:, 0] <= 0.0)
    if np.any(empty_rows):
        out[empty_rows, :] = 1.0 / float(n_classes)
        row_sum[empty_rows, :] = 1.0
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


MANIPULATOR_DIRECTION_MAP = {
    "move_forward": (0, 1),
    "move_backward": (0, -1),
    "move_left": (1, 1),
    "move_right": (1, -1),
    "move_up": (2, 1),
    "move_down": (2, -1),
    # 与 Manipulator-communication/src/arm_sender.py 完全一致：
    # axis 3=Roll、4=Pitch、5=Yaw。
    "rotate_roll_positive": (3, 1),
    "rotate_roll_negative": (3, -1),
    "rotate_pitch_positive": (4, 1),
    "rotate_pitch_negative": (4, -1),
    "rotate_yaw_positive": (5, 1),
    "rotate_yaw_negative": (5, -1),
}

def pack_manipulator_message(message: Dict) -> bytes:
    """Encode the length-prefixed UTF-8 JSON protocol used by receiver_arm.py."""
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(body)) + body

def manipulator_message_for_command(command: RobotCommand, speed: float = 50.0) -> Dict:
    """Translate one GUI command into the real-arm receiver protocol."""
    if command.name in MANIPULATOR_DIRECTION_MAP:
        axis, sign = MANIPULATOR_DIRECTION_MAP[command.name]
        return {"type": "cmd", "key": "move_start", "axis": axis, "sign": sign,
                "speed": float(speed), "timestamp": float(command.timestamp)}
    key = {"stop_motion": "move_stop", "close_gripper": "c",
           "open_gripper": "v", "home": "r"}.get(command.name)
    if key is None:
        raise ValueError(f"Unsupported robot command: {command.name}")
    return {"type": "cmd", "key": key, "timestamp": float(command.timestamp)}

class ManipulatorProtocolRobotClient(RobotArmClient):
    """Persistent, non-blocking TCP sender for the Ubuntu arm receiver."""

    def __init__(self, host: str = "192.168.1.100", port: int = 9527,
                 timeout: float = 1.0, speed: float = 50.0, heartbeat_interval: float = 5.0):
        self.host, self.port = str(host), int(port)
        self.timeout, self.speed = float(timeout), float(speed)
        self.heartbeat_interval = float(heartbeat_interval)
        self._queue: Queue = Queue(maxsize=128)
        self._running, self._connected = threading.Event(), threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop_timer: Optional[threading.Timer] = None
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self.last_error = ""

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True, name="manipulator-tcp")
        self._thread.start()

    def execute(self, command: RobotCommand) -> None:
        self.start()
        if command.name == "stop_motion":
            self._cancel_stop_timer()
        command_speed = (
            float(command.protocol_speed)
            if float(command.protocol_speed) > 0.0
            else self.speed
        )
        self._enqueue(manipulator_message_for_command(command, command_speed),
                      ttl=30.0 if command.name == "stop_motion" else 2.0)
        if command.name in MANIPULATOR_DIRECTION_MAP and not command.continuous:
            self._schedule_stop(max(0.05, float(command.duration_seconds)))

    def _enqueue(self, message: Dict, ttl: float = 2.0) -> None:
        item = (message, time.monotonic() + max(0.1, float(ttl)))
        try:
            self._queue.put_nowait(item)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(item)

    def _schedule_stop(self, delay: float) -> None:
        self._cancel_stop_timer()
        timer = threading.Timer(delay, lambda: self._enqueue(
            {"type": "cmd", "key": "move_stop", "timestamp": time.time()}, ttl=30.0))
        timer.daemon = True
        self._stop_timer = timer
        timer.start()

    def _cancel_stop_timer(self) -> None:
        timer, self._stop_timer = self._stop_timer, None
        if timer is not None:
            timer.cancel()

    def _connect(self) -> bool:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(self.timeout)
            with self._lock:
                self._socket = sock
            self._connected.set()
            self.last_error = ""
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self._connected.clear()
            return False

    def _disconnect(self) -> None:
        with self._lock:
            sock, self._socket = self._socket, None
        self._connected.clear()
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _send(self, message: Dict) -> bool:
        with self._lock:
            sock = self._socket
        if sock is None:
            return False
        try:
            sock.sendall(pack_manipulator_message(message))
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self._disconnect()
            return False

    def _run(self) -> None:
        reconnect_at, last_heartbeat = 0.0, 0.0
        while self._running.is_set():
            if not self.is_connected:
                now = time.monotonic()
                if now < reconnect_at:
                    time.sleep(min(0.1, reconnect_at - now))
                    continue
                if not self._connect():
                    reconnect_at = time.monotonic() + 1.0
                    continue
                last_heartbeat = time.monotonic()
            try:
                message, deadline = self._queue.get(timeout=0.1)
            except Empty:
                message, deadline = None, 0.0
            if message is not None and time.monotonic() <= deadline:
                if not self._send(message):
                    reconnect_at = time.monotonic() + 0.5
                    continue
            if time.monotonic() - last_heartbeat >= self.heartbeat_interval:
                if not self._send({"type": "ping", "timestamp": time.time()}):
                    reconnect_at = time.monotonic() + 0.5
                last_heartbeat = time.monotonic()
        self._disconnect()

    def close(self) -> None:
        self._cancel_stop_timer()
        if self.is_connected:
            self._send({"type": "cmd", "key": "move_stop", "timestamp": time.time()})
        self._running.clear()
        self._disconnect()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

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
        raw_shape = tuple(int(value) for value in np.asarray(data["raw_stream"]).shape) if "raw_stream" in data else ()
        labels = np.asarray(data["y"], dtype=object).reshape(-1)
        records = []
        if "trial_records_json" in data:
            try:
                records = json.loads(str(np.asarray(data["trial_records_json"]).item()))
            except Exception:
                records = []
        fs = int(np.asarray(data["fs"]).reshape(-1)[0]) if "fs" in data else 0
        mode = str(np.asarray(data["mode"]).reshape(-1)[0]) if "mode" in data else "unknown"
        completed = bool(np.asarray(data["completed"]).reshape(-1)[0]) if "completed" in data else True
        if len(eeg_shape) >= 3:
            eeg_channels = int(eeg_shape[1])
            samples = int(eeg_shape[-1])
        elif len(eeg_shape) == 2:
            eeg_channels = int(eeg_shape[0])
            samples = int(eeg_shape[1])
        else:
            eeg_channels = 0
            samples = 0
        if len(aux_shape) >= 3:
            aux_channels = int(aux_shape[1])
        elif len(aux_shape) == 2:
            aux_channels = int(aux_shape[0])
        elif len(raw_shape) == 2 and int(raw_shape[1]) > 32:
            aux_channels = min(4, int(raw_shape[1]) - 32)
        else:
            aux_channels = 0
        valid_records = [r for r in records if isinstance(r, dict) and not r.get("bad_trial") and not r.get("is_practice")]
        trial_count = len(valid_records) * 2 if valid_records else int(labels.size)
    return {
        "path": os.path.abspath(path),
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
        "trials": int(trial_count),
        "eeg_channels": eeg_channels,
        "aux_channels": aux_channels,
        "samples": samples,
        "fs": fs,
        "mode": mode,
        "completed": completed,
    }
