# -*- coding: utf-8 -*-
"""Navigation labels and presentation text for the BCI console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


CONSOLE_WINDOW_TITLE = "机械臂 BCI 控制台 — EEG8 多模态在线系统"
CONSOLE_BRAND = "ARM BCI"
CONSOLE_SUBTITLE = "EEG8 采集 / 训练 / 测试 / 在线控制"
CONSOLE_READY_STATUS = "CONSOLE READY"


@dataclass(frozen=True)
class ConsolePageSpec:
    code: str
    label: str
    purpose: str


CONSOLE_PAGES: Tuple[ConsolePageSpec, ...] = (
    ConsolePageSpec("01", "准备配置", "设置采集、训练、测试和在线控制参数"),
    ConsolePageSpec("02", "信号监测", "检查 EEG/AUX 输入质量和基线"),
    ConsolePageSpec("03", "训练采集", "按范式采集可训练 EEG8 trial"),
    ConsolePageSpec("04", "模型训练", "选择训练集与独立测试集并生成模型"),
    ConsolePageSpec("05", "在线控制", "真实 LSL 在线解码并门控机械臂输出"),
)

CONSOLE_PAGE_NAMES = [page.label for page in CONSOLE_PAGES]
CONSOLE_NAV_CODES = [page.code for page in CONSOLE_PAGES]

CONSOLE_WORKFLOW_TEXT = (
    "控制台工作流\n\n"
    "1  准备配置：确认设备、数据目录和模型路径\n"
    "2  信号监测：检查 EEG/AUX 通道质量\n"
    "3  训练采集：保存可训练 trial 到 recordings\n"
    "4  模型训练：训练集和独立测试集分开勾选\n"
    "5  在线控制：显式启用后才输出机械臂命令"
)

CONSOLE_HINT_TEXT = (
    "BrainVision LiveAmp\n"
    "32 导 EEG + 可选 AUX\n"
    "训练、测试、在线控制分阶段执行"
)

CONFIG_PAGE_TITLE = "准备配置"
CONFIG_PAGE_DESC = "集中设置训练采集、模型训练和在线控制参数。"
