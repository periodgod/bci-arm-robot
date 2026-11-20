# -*- coding: utf-8 -*-
"""EEG8 — 八分类主动脑电采集系统  (独立模块，供 gui.py 调用)"""

from __future__ import annotations

import json, os
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# 八分类定义

CLASS_NAMES = {
    1: "左手运动想象", 2: "右手运动想象",
    3: "双手运动想象", 4: "双脚运动想象",
    5: "左手握拳",      6: "右手握拳",
    7: "双手握拳",      8: "双脚执行",
}

CLASS_TO_ARM = {
    1: "left", 2: "right", 3: "forward", 4: "backward",
    5: "up", 6: "down", 7: "grab", 8: "release",
}

CLASS_ARM_LABELS = {
    1: "左移", 2: "右移", 3: "前移", 4: "后移",
    5: "上移", 6: "下移", 7: "抓取", 8: "释放",
}

# 八分类任务详细定义
EEG8_TASKS = [
    {"class_id": 1, "class_name": "左手运动想象", "class_name_en": "Left-hand MI",
     "cue_title": "左手想象", "cue_subtitle": "反复握拳松开",
     "instruction": "想象左手反复握拳、松开，感受手指屈伸和手掌用力的运动感觉。",
     "forbidden": "不要真的动手、不要动肩膀、不要看手。"},
    {"class_id": 2, "class_name": "右手运动想象", "class_name_en": "Right-hand MI",
     "cue_title": "右手想象", "cue_subtitle": "反复握拳松开",
     "instruction": "想象右手反复握拳、松开，感受手指、手掌和前臂的运动感觉。",
     "forbidden": "不要真的动手、不要动肩膀、不要用力。"},
    {"class_id": 3, "class_name": "双手运动想象", "class_name_en": "Both-hands MI",
     "cue_title": "双手想象", "cue_subtitle": "双手反复握拳松开",
     "instruction": "想象双手同时反复握拳、松开，感受双手手指屈伸和手掌用力的运动感觉。",
     "forbidden": "不要真的动手、不要动肩膀、不要咬牙。"},
    {"class_id": 4, "class_name": "双脚运动想象", "class_name_en": "Feet MI",
     "cue_title": "双脚想象", "cue_subtitle": "脚背上抬",
     "instruction": "想象双脚脚背同时上抬再放下，感受脚踝和脚背的运动感觉。",
     "forbidden": "不要真动脚、不要绷腿、不要踩地、不要晃动身体。"},
    {"class_id": 5, "class_name": "左手握拳", "class_name_en": "Left fist",
     "cue_title": "左手握拳", "cue_subtitle": "持续用力握紧",
     "instruction": "左手持续用力握紧拳头，保持张力，感受左手肌肉的持续收缩感。",
     "forbidden": "不要松开、不要换手、不要动手臂。"},
    {"class_id": 6, "class_name": "右手握拳", "class_name_en": "Right fist",
     "cue_title": "右手握拳", "cue_subtitle": "持续用力握紧",
     "instruction": "右手持续用力握紧拳头，保持张力，感受右手肌肉的持续收缩感。",
     "forbidden": "不要松开、不要换手、不要动手臂。"},
    {"class_id": 7, "class_name": "双手握拳", "class_name_en": "Both fists",
     "cue_title": "双手握拳", "cue_subtitle": "双手同时用力",
     "instruction": "双手同时持续用力握紧拳头，保持张力。",
     "forbidden": "不要松开、不要动手臂和肩膀。"},
    {"class_id": 8, "class_name": "双脚执行", "class_name_en": "Feet Execute",
     "cue_title": "双脚执行", "cue_subtitle": "实际脚背上抬",
     "instruction": "双脚脚背同时实际上抬再放下，动作幅度小且稳定。",
     "forbidden": "不要大幅晃动身体、不要移动椅子、不要绷紧上肢。"},
]

START_NUMBERS = [487, 653, 729, 816, 592, 764, 935, 678, 841, 526]

def eeg8_task_map() -> Dict[int, Dict]:
    return {int(t["class_id"]): t for t in EEG8_TASKS}

def eeg8_format_task(task, rng, slot_index=0):
    item = dict(task)
    if int(item["class_id"]) == 8:
        nums = list(START_NUMBERS); rng.shuffle(nums)
        n = nums[slot_index % len(nums)]
        item["cue_subtitle"] = f"{n} - 7"
    return item

# 数据加载器（支持八分类和旧格式）

EEG8_EEG_CHANNEL_COUNT = 32
EEG8_REQUIRED_AUX_CHANNEL_COUNT = 4
EEG8_INFERRED_CHANNEL_NAMES_TRAINING_ERROR = (
    "该历史 EEG8 记录缺少真实 32 路 EEG 通道名，仅允许离线只读导入，"
    "不能用于训练或演示训练。请使用保存真实 LSL 电极标签的记录重新采集或迁移后再训练。"
)


def _channel_key(name: object) -> str:
    return "".join(char for char in str(name).upper() if char.isalnum())


def _read_npz_names(data, key: str) -> List[str]:
    if key not in data:
        return []
    return [str(value).strip() for value in np.asarray(data[key], dtype=object).reshape(-1).tolist()]


def _read_contract_int(data, key: str, path: str, *, required: bool = False) -> Optional[int]:
    """Read one integer contract field without silently truncating its value."""
    if key not in data:
        if required:
            raise ValueError(f"v2 记录缺少必填元数据 {key}：{path}")
        return None
    values = np.asarray(data[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{key} 必须是单个整数：{path}")
    value = values[0]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} 必须是整数，不能为布尔值：{path}")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是整数：{path}") from None
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{key} 必须是有限整数：{path}")
    return int(numeric)


def _read_contract_indices(data, key: str, path: str, *, required: bool = False) -> Optional[List[int]]:
    """Read a vector of integer physical-channel indices without coercion."""
    if key not in data:
        if required:
            raise ValueError(f"v2 记录缺少必填元数据 {key}：{path}")
        return None
    values = np.asarray(data[key]).reshape(-1)
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{key} 必须是整数通道索引：{path}") from None
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{key} 必须是有限整数通道索引：{path}")
    return [int(value) for value in numeric.tolist()]


def _read_contract_bool(data, key: str, path: str, *, required: bool = False) -> Optional[bool]:
    """Read a scalar boolean field; reject truthy strings such as ``\"false\"``."""
    if key not in data:
        if required:
            raise ValueError(f"v2 记录缺少必填元数据 {key}：{path}")
        return None
    values = np.asarray(data[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{key} 必须是单个布尔值：{path}")
    value = values[0]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ValueError(f"{key} 必须是布尔值或 0/1：{path}")


def _read_eeg8_recording_contract(data, path: str) -> Tuple[np.ndarray, Optional[np.ndarray], int, List[str], List[str], Dict[str, Any]]:
    """Read one active-recording NPZ without guessing its channel layout."""
    fs = int(np.asarray(data["fs"]).reshape(-1)[0]) if "fs" in data else 250
    if fs <= 0:
        raise ValueError(f"采样率无效：{path}（fs={fs}）")
    raw = np.asarray(data["raw_stream"], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"raw_stream 必须为 (samples, channels)：{path}，实际 {raw.shape}")
    if raw.shape[1] < EEG8_EEG_CHANNEL_COUNT:
        raise ValueError(
            f"EEG8 训练需要前 {EEG8_EEG_CHANNEL_COUNT} 路 EEG：{path} 只有 {raw.shape[1]} 路，拒绝错位加载。"
        )

    declared_contract_version = _read_contract_int(data, "recording_contract_version", path)
    contract_version = 1 if declared_contract_version is None else declared_contract_version
    if contract_version < 1:
        raise ValueError(f"recording_contract_version 必须不小于 1：{path}")
    is_v2_contract = contract_version >= 2

    # v1 supports read-only import; new recordings use the v2 channel contract.
    eeg_indices = _read_contract_indices(data, "eeg_indices", path, required=is_v2_contract)
    if eeg_indices is not None and eeg_indices != list(range(EEG8_EEG_CHANNEL_COUNT)):
        raise ValueError(f"eeg_indices 必须明确为 raw_stream 的前 32 路（0..31）：{path}")
    declared_eeg_count = _read_contract_int(data, "eeg_channel_count", path, required=is_v2_contract)
    if declared_eeg_count is not None and declared_eeg_count != EEG8_EEG_CHANNEL_COUNT:
        raise ValueError(
            f"eeg_channel_count 必须为 {EEG8_EEG_CHANNEL_COUNT}，实际为 {declared_eeg_count}：{path}"
        )
    declared_raw_count = _read_contract_int(data, "raw_channel_count", path, required=is_v2_contract)
    if declared_raw_count is not None and declared_raw_count != raw.shape[1]:
        raise ValueError(
            f"raw_channel_count 与 raw_stream 通道数不一致：{path}，"
            f"声明 {declared_raw_count}，实际 {raw.shape[1]}"
        )

    raw_names = _read_npz_names(data, "raw_channel_names")
    if is_v2_contract and not raw_names:
        raise ValueError(f"v2 记录缺少 raw_channel_names，无法验证 EEG 物理通道顺序：{path}")
    if raw_names and len(raw_names) != raw.shape[1]:
        raise ValueError(f"raw_channel_names 数量与 raw_stream 不一致：{path}")
    stored_eeg_names = _read_npz_names(data, "channel_names")
    if is_v2_contract and not stored_eeg_names:
        raise ValueError(f"v2 记录缺少 channel_names，无法验证 EEG 通道名称：{path}")
    if stored_eeg_names:
        if len(stored_eeg_names) not in {EEG8_EEG_CHANNEL_COUNT, raw.shape[1]}:
            raise ValueError(
                f"channel_names 必须是 {EEG8_EEG_CHANNEL_COUNT} 个 EEG 名称或完整原始通道名称：{path}"
            )
        eeg_names = stored_eeg_names[:EEG8_EEG_CHANNEL_COUNT]
    elif raw_names:
        eeg_names = raw_names[:EEG8_EEG_CHANNEL_COUNT]
    else:
        # Preserve channel order when old recordings have no names.
        eeg_names = [f"CH{index + 1}" for index in range(EEG8_EEG_CHANNEL_COUNT)]
    if len(eeg_names) != EEG8_EEG_CHANNEL_COUNT or any(not name for name in eeg_names):
        raise ValueError(f"EEG 通道名称不完整：{path}")
    eeg_name_keys = [_channel_key(name) for name in eeg_names]
    if any(not key for key in eeg_name_keys) or len(set(eeg_name_keys)) != EEG8_EEG_CHANNEL_COUNT:
        raise ValueError(f"EEG 通道名称必须为 32 个非空且不重复的物理通道：{path}")
    if raw_names and stored_eeg_names:
        raw_eeg_keys = [_channel_key(name) for name in raw_names[:EEG8_EEG_CHANNEL_COUNT]]
        if raw_eeg_keys != eeg_name_keys:
            raise ValueError(f"channel_names 与 raw_channel_names 的前 32 路顺序不一致：{path}")

    explicit_indices = "aux_indices" in data
    if is_v2_contract and not explicit_indices:
        raise ValueError(f"v2 记录缺少 aux_indices，无法验证 AUX 物理通道：{path}")
    if explicit_indices:
        aux_indices = _read_contract_indices(data, "aux_indices", path) or []
        if len(aux_indices) > EEG8_REQUIRED_AUX_CHANNEL_COUNT or len(set(aux_indices)) != len(aux_indices):
            raise ValueError(f"aux_indices 必须是不重复的 0-4 个通道：{path}")
        if any(index < EEG8_EEG_CHANNEL_COUNT or index >= raw.shape[1] for index in aux_indices):
            raise ValueError(f"aux_indices 不在 raw_stream 的 AUX 区间：{path}")
        if aux_indices != sorted(aux_indices):
            raise ValueError(f"aux_indices 必须按物理通道升序保存：{path}")
        # Import trailing AUX columns from affected v1 recordings.
        if not aux_indices and raw.shape[1] > EEG8_EEG_CHANNEL_COUNT and contract_version < 2:
            aux_indices = list(range(EEG8_EEG_CHANNEL_COUNT, min(EEG8_EEG_CHANNEL_COUNT + 4, raw.shape[1])))
    else:
        aux_indices = list(range(EEG8_EEG_CHANNEL_COUNT, min(EEG8_EEG_CHANNEL_COUNT + 4, raw.shape[1])))
    declared_aux_count = _read_contract_int(data, "aux_channel_count", path, required=is_v2_contract)
    if declared_aux_count is not None and declared_aux_count != len(aux_indices):
        raise ValueError(f"aux_channel_count 与 aux_indices 不一致：{path}")

    aux_raw: Optional[np.ndarray] = None
    stored_aux = np.asarray(data["aux"], dtype=np.float32) if "aux" in data else None
    if stored_aux is not None and stored_aux.size:
        if stored_aux.ndim != 2:
            raise ValueError(f"aux 必须为二维数组：{path}，实际 {stored_aux.shape}")
        if stored_aux.shape[1] == raw.shape[0] and stored_aux.shape[0] <= EEG8_REQUIRED_AUX_CHANNEL_COUNT:
            aux_raw = stored_aux.T
        elif stored_aux.shape[0] == raw.shape[0] and stored_aux.shape[1] <= EEG8_REQUIRED_AUX_CHANNEL_COUNT:
            aux_raw = stored_aux
        else:
            raise ValueError(f"aux 形状无法与 raw_stream 时间轴对齐：{path}，aux={stored_aux.shape} raw={raw.shape}")
        if aux_raw.shape[1] != len(aux_indices):
            raise ValueError(
                f"aux 通道数与 aux_indices 不一致：{path}，aux={aux_raw.shape[1]}，indices={len(aux_indices)}"
            )
    elif aux_indices:
        aux_raw = raw[:, aux_indices]

    stored_aux_names = _read_npz_names(data, "aux_channel_names")
    if stored_aux_names:
        if len(stored_aux_names) != len(aux_indices):
            raise ValueError(f"aux_channel_names 数量与 aux_indices 不一致：{path}")
        aux_names = stored_aux_names
    elif raw_names and aux_indices:
        aux_names = [raw_names[index] for index in aux_indices]
    else:
        aux_names = [f"Aux{index - EEG8_EEG_CHANNEL_COUNT + 1}" for index in aux_indices]
    if aux_raw is not None and aux_raw.shape[1] != len(aux_names):
        raise ValueError(f"AUX 数据与名称数量不一致：{path}")

    expected_aux_indices = list(range(EEG8_EEG_CHANNEL_COUNT, EEG8_EEG_CHANNEL_COUNT + EEG8_REQUIRED_AUX_CHANNEL_COUNT))
    gate_capable = aux_indices == expected_aux_indices and aux_raw is not None and aux_raw.shape[1] == EEG8_REQUIRED_AUX_CHANNEL_COUNT
    declared_gate_capable = _read_contract_bool(
        data, "masseter_gate_capable", path, required=is_v2_contract
    )
    if declared_gate_capable is not None and declared_gate_capable != bool(gate_capable):
        raise ValueError(f"masseter_gate_capable 与实际 AUX 契约不一致：{path}")
    missing_gate_channels = [f"AUX{index - EEG8_EEG_CHANNEL_COUNT + 1}" for index in expected_aux_indices if index not in aux_indices]
    gate_status = "完整 AUX1-AUX4，可用于咬肌门控" if gate_capable else (
        "缺少 " + "、".join(missing_gate_channels or ["有效 AUX3/AUX4"]) + "，不具备咬肌门控条件"
    )
    metadata = {
        "recording_contract_version": contract_version,
        "eeg_channel_count": EEG8_EEG_CHANNEL_COUNT,
        "eeg_indices": list(range(EEG8_EEG_CHANNEL_COUNT)),
        "raw_channel_count": raw.shape[1],
        "eeg_channel_names_inferred": not bool(stored_eeg_names or raw_names),
        "aux_indices": aux_indices,
        "aux_channel_names": aux_names,
        "aux_channel_count": len(aux_indices),
        "masseter_gate_capable": bool(gate_capable),
        "masseter_gate_status": gate_status,
    }
    return raw[:, :EEG8_EEG_CHANNEL_COUNT], aux_raw, fs, eeg_names, aux_names, metadata


def load_eeg8_recordings(paths, log=None, return_metadata: bool = False):
    """Load C1-C8 task epochs and their recorded channel contract."""
    if not paths: raise ValueError("没有选择数据文件")
    first = os.path.abspath(str(paths[0]))
    is_eeg8 = False
    try:
        with np.load(first, allow_pickle=True) as d:
            is_eeg8 = "trial_records_json" in d and "raw_stream" in d
    except Exception: pass
    if not is_eeg8:
        raise ValueError("仅支持包含 raw_stream 和 trial_records_json 的 EEG8 采集文件")
    all_epochs, all_labels, all_aux_epochs = [], [], []
    ref_fs, ref_names, ref_aux_names, ref_contract = None, None, None, None
    ref_class_names = None
    has_aux: Optional[bool] = None
    for p in paths:
        p = os.path.abspath(str(p))
        with np.load(p, allow_pickle=True) as d:
            eeg_raw, aux_raw, fs, ch_names, aux_names, contract = _read_eeg8_recording_contract(d, p)
            records = json.loads(str(np.asarray(d.get("trial_records_json", "[]")).item())) if "trial_records_json" in d else []
            task_defs = json.loads(str(np.asarray(d.get("task_definitions_json", "[]")).item())) if "task_definitions_json" in d else []
            class_names = [str(item.get("class_name", "")).strip() for item in task_defs]
            if not class_names or any(not name for name in class_names):
                class_names = [CLASS_NAMES.get(index, f"类别 {index}") for index in range(1, 9)]
            if ref_class_names is None:
                ref_class_names = class_names
            elif class_names != ref_class_names:
                raise ValueError(f"采集记录的类别数量或名称不一致：{p}")
            if ref_fs is None:
                ref_fs, ref_names, ref_aux_names, ref_contract = fs, ch_names, aux_names, contract
                has_aux = aux_raw is not None and aux_raw.shape[1] > 0
            if fs != ref_fs: raise ValueError(f"采样率不一致：{p}（{fs}Hz，期望 {ref_fs}Hz）")
            if [_channel_key(name) for name in ch_names] != [_channel_key(name) for name in ref_names or []]:
                raise ValueError(f"EEG 通道名称或顺序不一致：{p}")
            if (aux_raw is not None and aux_raw.shape[1] > 0) != bool(has_aux):
                raise ValueError(f"AUX/肌电字段不一致：{p}")
            if list(contract["aux_indices"]) != list((ref_contract or {}).get("aux_indices", [])):
                raise ValueError(f"AUX 物理通道索引不一致：{p}")
            if [_channel_key(name) for name in aux_names] != [_channel_key(name) for name in ref_aux_names or []]:
                raise ValueError(f"AUX 通道名称或顺序不一致：{p}")
            vr = [r for r in records if not r.get("bad_trial") and not r.get("is_practice")]
            if vr:
                mx = max(float(r.get("trial_end_ms", r.get("task_end_ms", 0)) or 0) for r in vr)
                offset = min(0, eeg_raw.shape[0] - int(round(mx / 1000.0 * fs)))
            else: offset = 0
            for rec in vr:
                class_id = int(rec.get("class_id", 0) or 0)
                if class_id not in range(1, len(class_names) + 1):
                    raise ValueError(f"EEG 记录包含无效类别 {class_id}：{p}")
                task_start = float(rec.get("task_start_ms", 0) or 0)
                task_end = float(rec.get("task_end_ms", rec.get("trial_end_ms", 0)) or 0)
                if task_end <= task_start:
                    raise ValueError(f"EEG8 记录缺少有效任务时间段：{p}")
                s = max(0, min(int(round(task_start / 1000.0 * fs)) + offset, eeg_raw.shape[0]))
                e = max(s + 1, min(int(round(task_end / 1000.0 * fs)) + offset, eeg_raw.shape[0]))
                all_epochs.append(eeg_raw[s:e, :].T.astype(np.float32))
                if aux_raw is not None and aux_raw.shape[1] > 0:
                    aux_e = max(s + 1, min(e, aux_raw.shape[0]))
                    all_aux_epochs.append(aux_raw[s:aux_e, :].T.astype(np.float32))
                all_labels.append(class_id - 1)
    if not all_epochs: raise RuntimeError("没有提取到有效 epoch")
    ml = min(e.shape[1] for e in all_epochs)
    epochs_3d = np.stack([e[:, :ml] for e in all_epochs], axis=0).astype(np.float32)
    aux_3d = None
    if all_aux_epochs:
        aux_ml = min(e.shape[1] for e in all_aux_epochs)
        keep = min(ml, aux_ml)
        epochs_3d = epochs_3d[:, :, :keep]
        aux_3d = np.stack([e[:, :keep] for e in all_aux_epochs], axis=0).astype(np.float32)
    contract_meta = dict(ref_contract or {})
    contract_meta["class_count"] = len(ref_class_names or [])
    contract_meta["class_names"] = list(ref_class_names or [])
    contract_meta["class_labels"] = [f"C{index + 1} {name}" for index, name in enumerate(ref_class_names or [])]
    contract_meta["files"] = [os.path.abspath(str(path)) for path in paths]
    if log:
        log(
            f"EEG8 C1-C8 加载：{len(all_epochs)} 个任务 epoch，EEG={epochs_3d.shape}, "
            f"AUX={None if aux_3d is None else aux_3d.shape}, fs={ref_fs}Hz；{contract_meta.get('masseter_gate_status', '')}"
        )
        if contract_meta.get("eeg_channel_names_inferred"):
            log("警告：记录未保存 EEG 通道名，已按固定 CH1-CH32 顺序加载；请在下一次采集中保存完整通道元数据。")
    result = (epochs_3d, np.asarray(all_labels, dtype=object), aux_3d, ref_fs, list(ref_names or []))
    return (*result, contract_meta) if return_metadata else result


def validate_eeg8_training_data_contract(data_contract: Mapping[str, Any]) -> None:
    """Require recorded physical channel labels before training."""
    if not isinstance(data_contract, Mapping):
        raise TypeError("EEG8 训练数据契约必须为映射对象")
    if bool(data_contract.get("eeg_channel_names_inferred", False)):
        raise RuntimeError(EEG8_INFERRED_CHANNEL_NAMES_TRAINING_ERROR)

# 固定 C1-C8 训练/评估标签契约

EEG8_ACTION_LABEL_CONTRACT: Dict[str, Any] = {
    "name": "EEG8 C1-C8 动作分类",
    "key": "eeg8_c1_c8",
    "runtime_compatible": True,
    "description": "训练、评估与在线控制固定使用 C1-C8；不提供静息或派生标签路径。",
}


def eeg8_action_label_info(class_names: Optional[Sequence[str]] = None, runtime_compatible: bool = True) -> Dict[str, Any]:
    """Return the C1-C8 label contract."""
    return {
        **EEG8_ACTION_LABEL_CONTRACT,
        "name": "EEG 类别分类" if class_names else EEG8_ACTION_LABEL_CONTRACT["name"],
        "key": "custom_eeg_categories" if class_names else EEG8_ACTION_LABEL_CONTRACT["key"],
        "runtime_compatible": bool(runtime_compatible),
        "class_labels": ([f"C{i + 1} {name}" for i, name in enumerate(class_names)] if class_names else [
            f"C{i + 1} {CLASS_ARM_LABELS[i + 1]}" for i in range(8)
        ]),
    }


def prepare_eeg8_action_labels(y_raw: Sequence, n_classes: int = 8, class_names=None, runtime_compatible=True):
    """Validate zero-based class labels without remapping."""
    y = np.asarray(y_raw, dtype=np.int64).reshape(-1)
    if y.size == 0:
        raise RuntimeError("EEG8 C1-C8 训练没有可用样本")
    if np.any((y < 0) | (y >= int(n_classes))):
        invalid = sorted({int(value) for value in y[(y < 0) | (y >= int(n_classes))].tolist()})
        raise ValueError(f"EEG 训练标签必须位于 0-{int(n_classes) - 1}，发现无效标签：{invalid}")
    info = eeg8_action_label_info(class_names if not runtime_compatible else None, runtime_compatible)
    info.update({"kept_epochs": int(y.size), "dropped_epochs": 0})
    return np.ones(y.shape, dtype=bool), y, info

def _align_eeg8_proba(classifier, raw_proba, n_classes=None):
    """Align sklearn predict_proba output to fixed C1-C8 zero-based columns."""
    proba = np.asarray(raw_proba, dtype=np.float64)
    classes = np.asarray(getattr(classifier, "classes_", np.arange(proba.shape[1])), dtype=np.int64)
    if n_classes is None:
        n_classes = int(np.max(classes)) + 1 if classes.size else proba.shape[1]
    out = np.zeros((proba.shape[0], int(n_classes)), dtype=np.float64)
    for src_idx, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls < out.shape[1] and src_idx < proba.shape[1]:
            out[:, cls] = proba[:, src_idx]
    row_sum = np.sum(out, axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return out / row_sum


def eeg8_binary_confusion(y_true, eeg_proba, pair):
    """Build a 2x2 confusion matrix for any selected pair of EEG8 classes.

    `pair` uses zero-based class indices. Only windows whose true label belongs
    to the pair are counted; prediction is the higher probability within the
    selected pair.
    """
    a, b = [int(x) for x in pair]
    if a == b:
        raise ValueError("二分类混淆矩阵需要选择两个不同类别")
    if not (0 <= a < 8 and 0 <= b < 8):
        raise ValueError("类别范围必须是 C1-C8")
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    proba = np.asarray(eeg_proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[0] != y.shape[0] or proba.shape[1] < 8:
        raise ValueError("二分类混淆矩阵需要窗口级真实标签和 8 类概率")
    mask = np.isin(y, [a, b])
    matrix = np.zeros((2, 2), dtype=np.int64)
    if np.any(mask):
        y_pair = y[mask]
        pair_proba = proba[mask][:, [a, b]]
        pred_pair = np.asarray([a, b], dtype=np.int64)[np.argmax(pair_proba, axis=1)]
        for true_label, pred_label in zip(y_pair, pred_pair):
            ti = 0 if int(true_label) == a else 1
            pi = 0 if int(pred_label) == a else 1
            matrix[ti, pi] += 1
    support = int(matrix.sum())
    accuracy = float(np.trace(matrix) / support) if support else None
    return {
        "pair": [a + 1, b + 1],
        "labels": [CLASS_NAMES.get(a + 1, f"C{a + 1}"), CLASS_NAMES.get(b + 1, f"C{b + 1}")],
        "actions": [CLASS_ARM_LABELS.get(a + 1, ""), CLASS_ARM_LABELS.get(b + 1, "")],
        "support": support,
        "accuracy": accuracy,
        "matrix": matrix.tolist(),
        "note": "仅统计真实标签属于所选两类的窗口；预测在所选两类概率之间二选一。",
    }

def train_eeg8_model(
    eeg,
    y_raw,
    aux,
    signal_cfg,
    decoder_cfg,
    classifier_kind,
    data_contract: Mapping[str, Any],
    paradigm_key: str = "eeg8",
    run_cv: bool = True,
    progress_callback=None,
):
    """Train the fixed C1-C8 model with trial-level cross validation."""
    from arm_bci.core import ensure_3d_epochs, make_windows_from_trials
    from arm_bci.domain import EEG8Model
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    def emit_progress(percent: int, stage: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(int(percent), stage)
        except Exception:
            pass

    validate_eeg8_training_data_contract(data_contract)

    emit_progress(12, "整理标签和 EEG/AUX 数据")
    paradigm_key = str(paradigm_key or "eeg8").strip().lower()
    class_names = list(data_contract.get("class_names", []) or [])
    is_custom = paradigm_key == "custom_mi"
    n_configured_classes = int(data_contract.get("class_count", len(class_names) or 8))
    if is_custom and not (2 <= n_configured_classes <= 16):
        raise ValueError("自定义 MI 类别数必须为 2-16")
    keep, y_idx, label_contract = prepare_eeg8_action_labels(
        y_raw, n_configured_classes, class_names, not is_custom
    )
    eeg = ensure_3d_epochs(eeg, "eeg")
    eeg = eeg[keep]
    aux = ensure_3d_epochs(aux, "aux") if aux is not None and aux.size else None
    if aux is not None:
        aux = aux[keep]

    expected_classes = [0, 1] if paradigm_key == "pure_mi_binary" else list(range(n_configured_classes))
    classes = sorted(int(v) for v in np.unique(y_idx))
    if classes != expected_classes:
        missing = [class_id + 1 for class_id in expected_classes if class_id not in classes]
        if paradigm_key == "pure_mi_binary":
            raise RuntimeError(f"纯 MI 训练必须同时包含 C1-C2（左/右手 MI），缺少：{missing}")
        raise RuntimeError(f"训练必须包含所有配置类别，缺少：{missing}")
    n_classes = len(classes)
    label_count = Counter(y_idx.tolist())
    n_splits = min(5, min(label_count.values())) if label_count else 0
    metrics = {
        "validation": "5-fold cross validation" if run_cv else "final fit only",
        "requested_folds": 5,
        "n_classes": n_classes,
        "label_contract": label_contract,
        "runtime_compatible": not is_custom,
        "paradigm_key": paradigm_key,
        "labels": [label_contract["class_labels"][i] for i in classes],
        "class_labels": [label_contract["class_labels"][i] for i in classes],
    }

    if run_cv and n_splits >= 2:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_accuracy: List[float] = []
        fold_kappa: List[float] = []
        all_true: List[int] = []
        all_pred: List[int] = []
        all_proba: List[List[float]] = []
        for fold_index, (tr_idx, va_idx) in enumerate(splitter.split(np.arange(len(y_idx)), y_idx), start=1):
            emit_progress(20 + int(45 * (fold_index - 1) / max(1, n_splits)), f"交叉验证 Fold {fold_index}/{n_splits}")
            train_eeg, train_y, train_aux = make_windows_from_trials(
                eeg[tr_idx],
                y_idx[tr_idx],
                signal_cfg,
                aux[tr_idx] if aux is not None else None,
            )
            val_eeg, val_y, _val_aux = make_windows_from_trials(
                eeg[va_idx],
                y_idx[va_idx],
                signal_cfg,
                aux[va_idx] if aux is not None else None,
            )
            fold_model = EEG8Model(signal_cfg, decoder_cfg)
            fold_model.classifier_kind = classifier_kind
            fold_model._fit_core(train_eeg, train_y, train_aux)
            proba = fold_model.predict_eeg_proba(val_eeg)
            pred = np.argmax(proba, axis=1)
            fold_accuracy.append(float(accuracy_score(val_y, pred)))
            fold_kappa.append(float(cohen_kappa_score(val_y, pred, labels=classes)))
            all_true.extend(np.asarray(val_y, dtype=np.int64).tolist())
            all_pred.extend(np.asarray(pred, dtype=np.int64).tolist())
            all_proba.extend(proba.astype(float).tolist())
            emit_progress(20 + int(45 * fold_index / max(1, n_splits)), f"Fold {fold_index}/{n_splits} 完成")
        y_true_arr = np.asarray(all_true, dtype=np.int64)
        y_pred_arr = np.asarray(all_pred, dtype=np.int64)
        matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
        support = matrix.sum(axis=1).astype(float)
        correct = np.diag(matrix).astype(float)
        class_accuracy = np.divide(correct, np.maximum(1.0, support))
        metrics.update({
            "fold_labels": [f"Fold {i}" for i in range(1, n_splits + 1)],
            "fold_accuracy": fold_accuracy,
            "fold_kappa": fold_kappa,
            "eeg_accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "eeg_kappa": float(cohen_kappa_score(y_true_arr, y_pred_arr, labels=classes)),
            "overall_accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "kappa": float(cohen_kappa_score(y_true_arr, y_pred_arr, labels=classes)),
            "class_labels": [label_contract["class_labels"][i] for i in classes],
            "class_accuracy": class_accuracy.astype(float).tolist(),
            "confusion_matrix": matrix.tolist(),
            "window_count": int(len(y_true_arr)),
            "_y_true_windows": y_true_arr.tolist(),
            "_eeg_proba_windows": all_proba,
        })
    elif run_cv:
        emit_progress(35, "样本不足，跳过交叉验证并生成窗口")
        all_eeg, all_y, _all_aux = make_windows_from_trials(eeg, y_idx, signal_cfg, aux)
        metrics["window_count"] = int(len(all_y))
        metrics["warning"] = "每类 trial 数不足，无法执行五折交叉验证。"
    else:
        emit_progress(35, "生成最终训练窗口")
        all_eeg_preview, all_y_preview, _all_aux_preview = make_windows_from_trials(eeg, y_idx, signal_cfg, aux)
        metrics["window_count"] = int(len(all_y_preview))
        metrics["note"] = "仅训练最终模型，未请求交叉验证。"

    emit_progress(72, "训练最终模型")
    all_eeg, all_y, all_aux = make_windows_from_trials(eeg, y_idx, signal_cfg, aux)
    model = EEG8Model(signal_cfg, decoder_cfg)
    model.classifier_kind = classifier_kind
    model._fit_core(all_eeg, all_y, all_aux)
    emit_progress(88, "最终模型训练完成")
    model.paradigm_key = paradigm_key
    model.metrics_ = metrics
    model.training_strategy_ = {
        "label_contract": label_contract,
        "class_labels": list(label_contract["class_labels"]),
        "class_names": list(class_names) if is_custom else [],
        "class_count": int(n_configured_classes),
        "classifier": classifier_kind,
        "validation": metrics.get("validation", ""),
        "runtime_compatible": not is_custom,
    }
    return model, metrics

def eval_eeg8_model(model, eeg, y_raw, signal_cfg):
    """Evaluate the fixed C1-C8 model."""
    from arm_bci.core import ensure_3d_epochs, make_windows_from_trials
    from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    strategy = dict(getattr(model, "training_strategy_", {}) or {})
    model_contract = dict(strategy.get("label_contract", {}) or {})
    class_labels = list(model_contract.get("class_labels", []) or [])
    class_names = [str(label).split(" ", 1)[-1] for label in class_labels] if model_contract.get("key") == "custom_eeg_categories" else None
    n_classes = int(strategy.get("class_count", len(class_labels) or 8))
    keep, y_idx, label_contract = prepare_eeg8_action_labels(
        y_raw, n_classes, class_names, model_contract.get("key") != "custom_eeg_categories"
    )
    eeg = ensure_3d_epochs(eeg, "eeg")
    eeg = eeg[keep]
    nc = len(np.unique(y_idx))
    classes = sorted(int(v) for v in np.unique(y_idx))
    if model.eeg_features is None or model.eeg_classifier is None:
        raise RuntimeError("模型未训练")
    eeg_w, y_w, _ = make_windows_from_trials(eeg, y_idx, signal_cfg, None)
    proba = model.predict_eeg_proba(eeg_w)
    pred = np.argmax(proba, axis=1)
    return {
        "window_count": len(y_w), "n_classes": nc,
        "label_contract": label_contract,
        "runtime_compatible": model_contract.get("runtime_compatible", True),
        "eeg_accuracy": float(accuracy_score(y_w, pred)),
        "eeg_kappa": float(cohen_kappa_score(y_w, pred)),
        "confusion_matrix": confusion_matrix(y_w, pred, labels=classes).tolist(),
        "labels": [label_contract["class_labels"][i] for i in classes],
        "class_labels": [label_contract["class_labels"][i] for i in classes],
        "_y_true_windows": np.asarray(y_w, dtype=np.int64).tolist(),
        "_eeg_proba_windows": proba.astype(float).tolist(),
    }

# Trial 计划生成

def generate_balanced_block(class_ids, trials_per_class, seed, allow_repeat=False, task_definitions=None):
    """生成一个 block 内均衡的 trial 顺序"""
    rng = np.random.default_rng(seed)
    pool = []
    task_map = {int(item["class_id"]): dict(item) for item in (task_definitions or EEG8_TASKS)}
    for cid in class_ids:
        for rep in range(trials_per_class):
            task = dict(task_map[int(cid)])
            if task_definitions is None:
                task = eeg8_format_task(task, rng, rep)
            else:
                task.setdefault("cue_title", task["class_name"])
                task.setdefault("cue_subtitle", "请根据提示进行运动想象")
                task.setdefault("instruction", f"想象与“{task['class_name']}”对应的动作，不要实际运动。")
                task.setdefault("forbidden", "不要实际运动或改变身体姿势。")
            pool.append({**task, "slot": rep})
    # 随机打乱，避免相邻重复
    for _ in range(200):
        rng.shuffle(pool)
        if allow_repeat: break
        ok = True
        for i in range(1, len(pool)):
            if pool[i]["class_id"] == pool[i-1]["class_id"]:
                ok = False; break
        if ok: break
    return pool
