# 机械臂 BCI 项目结构说明

## 目标

本次整理把原来集中在单个大文件里的系统拆成可维护的包结构，并把数据、模型、实验报告和运行入口分开存放。

`arm_bci_system.py` 仍保留为兼容入口，原有命令不需要改成新的模块路径。

## 代码结构

```text
arm_bci_robot/
├─ arm_bci_system.py          # 兼容启动器，导出旧脚本依赖的符号
├─ arm_bci/
│  ├─ core.py                 # 配置、标签、信号处理、模型、数据源、RDA/LSL、机械臂控制
│  ├─ commands.py             # train/evaluate/make-demo-data/simulate/online
│  ├─ gui.py                  # PyQt GUI
│  ├─ cli.py                  # argparse 命令行
│  └─ __init__.py             # 包导出
├─ tools/                     # 数据准备、PhysioNet 实验、模型构建脚本
├─ tests/                     # 单元测试和实验验证测试
├─ docs/                      # 报告和项目说明
└─ examples/                  # 范式示例
```

## 数据结构

```text
data/
├─ recordings/                # 现场采集 session，GUI 和训练页默认扫描这里
├─ raw/                       # 原始下载缓存，例如 EDF、未完成下载
├─ processed/                 # 标准化 NPZ、demo calibration
└─ experiments/               # 实验报告、清单、注册表 JSON

models/
├─ group_csp_left_right_mi.joblib
├─ group_csp_left_right_mi.json
├─ pure32_left_right_mi.joblib
└─ mi_experiments/            # 实验脚本保存的对照模型
```

## 路径约定

- 现场采集不变：`data/recordings/YYYY-MM-DD/HH-MM-SS_session.npz`。
- PhysioNet 派生数据默认读取：`data/processed/physionet_eegmmidb_sXXX_left_right_mi.npz`。
- 原始 EDF 下载缓存默认写入：`data/raw/MNE-eegbci-data/...`。
- 实验 JSON 默认写入：`data/experiments/`。
- 实验模型默认写入：`models/mi_experiments/`。

`tools/mi_paths.py` 是实验脚本的共享路径库；以后如果目录要改，优先改这里，不要在每个脚本里重复写路径。

## 修改建议

- 改信号处理、模型、数据源时优先改 `arm_bci/core.py`。
- 改命令行行为时改 `arm_bci/commands.py` 或 `arm_bci/cli.py`。
- 改 GUI 时只改 `arm_bci/gui.py`，避免把界面代码混回核心算法。
- 新增实验脚本时复用 `tools/mi_paths.py` 的路径常量。
