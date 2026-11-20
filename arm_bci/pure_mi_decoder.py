# -*- coding: utf-8 -*-
"""纯运动想象（MI）可变长二进制指令解码。

本模块只负责把左/右手 MI 分类结果组成指令，不负责发送机械臂命令，
也不改变既有的确认窗口、持续运动或安全输出逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


PURE_MI_CODE_TO_INTENT = {
    "1": "left",
    "0": "right",
    "10": "forward",
    "01": "backward",
    "11": "up",
    "00": "down",
}

PURE_MI_INTENT_LABELS = {
    "left": "左移",
    "right": "右移",
    "forward": "前进",
    "backward": "后退",
    "up": "上升",
    "down": "下降",
    "grab": "抓取",
    "release": "放置",
    "stop_motion": "迅速停止",
}


@dataclass(frozen=True)
class PureMIDecodeResult:
    """一次解码事件。

    ``intent`` 使用系统既有的意图名称，因此上层可继续使用原输出逻辑。
    抓取/放置通过 ``requires_confirmation`` 告知上层沿用既有确认窗口。
    """

    action: str
    intent: Optional[str]
    code: str
    label: str
    requires_confirmation: bool = False
    continuous: bool = False
    reason: str = ""


def decode_pure_mi_code(code: object) -> PureMIDecodeResult:
    """将 1～3 位纯 MI 码解释为既有机械臂意图。

    一位和两位码按固定表解释；三位码的前两位是任意 ``xx``，
    最后一位 ``1`` 为抓取、``0`` 为放置。
    """

    bits = str(code or "").strip()
    if not bits or any(bit not in "01" for bit in bits):
        return PureMIDecodeResult(
            "invalid", None, bits, "无效编码", reason="code_must_contain_only_0_or_1"
        )
    if len(bits) in (1, 2):
        intent = PURE_MI_CODE_TO_INTENT[bits]
        return PureMIDecodeResult(
            "command",
            intent,
            bits,
            PURE_MI_INTENT_LABELS[intent],
            continuous=True,
            reason=f"decoded_{bits}",
        )
    if len(bits) == 3:
        intent = "grab" if bits[-1] == "1" else "release"
        return PureMIDecodeResult(
            "command",
            intent,
            bits,
            PURE_MI_INTENT_LABELS[intent],
            requires_confirmation=True,
            continuous=False,
            reason=f"decoded_xx{bits[-1]}",
        )
    return PureMIDecodeResult(
        "invalid", None, bits, "编码过长", reason="code_length_must_be_1_to_3"
    )


class PureMISequenceDecoder:
    """维护纯 MI 位缓存，并处理左右咬肌的序列控制语义。

    - 左手 MI 分类写入 ``1``，右手 MI 分类写入 ``0``。
    - 左咬肌提交已完成窗口，不包含当前尚未完成的窗口。
    - 右咬肌撤销最近一个已完成窗口，回退一位。
    - 双侧咬肌优先产生 ``stop_motion``，并清空当前序列。
    """

    def __init__(self) -> None:
        self._completed_bits = []
        self._active_bit: Optional[str] = None

    @property
    def completed_code(self) -> str:
        return "".join(self._completed_bits)

    @property
    def active_bit(self) -> Optional[str]:
        return self._active_bit

    def begin_window(self, prediction: object) -> str:
        """记录当前窗口的暂存结果；该位尚不能被左咬肌提交。"""

        self._active_bit = self._normalize_prediction(prediction)
        return self._active_bit

    def complete_window(self, prediction: object = None) -> str:
        """完成当前窗口并把其结果加入可提交序列。"""

        if prediction is not None:
            self._active_bit = self._normalize_prediction(prediction)
        if self._active_bit is None:
            raise ValueError("当前没有可完成的 MI 窗口")
        if len(self._completed_bits) >= 3:
            # 三位编码只能在下一窗口通过咬肌提交。若下一窗口没有咬肌，
            # 默认取消并清空当前全部计数；本窗口不进入下一组。
            self._completed_bits.clear()
            self._active_bit = None
            return ""
        self._completed_bits.append(self._active_bit)
        self._active_bit = None
        return self.completed_code

    def feed_completed_window(self, prediction: object) -> str:
        """便捷地加入一个已经结束的分类窗口。"""

        self.begin_window(prediction)
        return self.complete_window()

    def handle_masseter(
        self, left: bool = False, right: bool = False
    ) -> PureMIDecodeResult:
        """解释咬肌事件；双侧事件始终优先作为迅速停止。"""

        if bool(left) and bool(right):
            self.clear()
            return PureMIDecodeResult(
                "stop",
                "stop_motion",
                "",
                PURE_MI_INTENT_LABELS["stop_motion"],
                reason="both_masseters_emergency_stop",
            )
        if bool(right):
            removed_bit = self._completed_bits.pop() if self._completed_bits else ""
            # 当前窗口尚未完成，不属于保留的历史编码。
            self._active_bit = None
            return PureMIDecodeResult(
                "rollback",
                None,
                self.completed_code,
                "回退一个窗口",
                reason=(
                    f"right_masseter_rollback_removed_{removed_bit}"
                    if removed_bit
                    else "right_masseter_rollback_no_completed_window"
                ),
            )
        if bool(left):
            # active_bit 有意不参与提交，实现“不包含当前窗口”的截断。
            code = self.completed_code
            result = decode_pure_mi_code(code)
            self.clear()
            if result.action == "invalid" and not code:
                return PureMIDecodeResult(
                    "wait",
                    None,
                    "",
                    "等待已完成窗口",
                    reason="left_masseter_no_completed_window",
                )
            return result
        return PureMIDecodeResult(
            "wait",
            None,
            self.completed_code,
            "等待咬肌事件",
            reason="no_masseter_event",
        )

    def clear(self) -> None:
        self._completed_bits.clear()
        self._active_bit = None

    @staticmethod
    def _normalize_prediction(prediction: object) -> str:
        """接受位值或左右手名称，输出统一的 ``0``/``1``。"""

        if prediction in (1, True):
            return "1"
        if prediction in (0, False):
            return "0"
        value = str(prediction or "").strip().lower().replace("-", "_")
        if value in {"1", "left", "left_hand", "左", "左手", "左手运动想象"}:
            return "1"
        if value in {"0", "right", "right_hand", "右", "右手", "右手运动想象"}:
            return "0"
        raise ValueError(f"未知纯 MI 分类结果：{prediction!r}")


def pure_mi_codebook() -> Tuple[Tuple[str, str], ...]:
    """返回稳定顺序的编码说明，供界面或日志展示。"""

    return (
        ("1", "左移"),
        ("0", "右移"),
        ("10", "前进"),
        ("01", "后退"),
        ("11", "上升"),
        ("00", "下降"),
        ("xx1", "抓取（沿用确认窗口）"),
        ("xx0", "放置（沿用确认窗口）"),
    )
