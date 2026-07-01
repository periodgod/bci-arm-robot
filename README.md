# EEG + EMG 混合机械臂控制系统

基于 SSVEP + 运动想象 + EMG 的多模态 BCI 机械臂控制系统。

## 系统概述

使用三类咬肌事件选择动作组，再由反射群 CSP + shrinkage LDA 判别左/右手运动想象，形成六个移动方向。三次眨眼独立切换抓取/松开状态。

## 目录结构

```
arm_bci_robot/
├── arm_bci/
│   ├── core.py          # 核心控制逻辑
│   └── gui.py           # 图形用户界面 (PyQt5)
├── examples/            # 不同实验范式示例
├── data/                # 数据采集目录 (本地, 未上传)
├── docs/                # 文档和论文
├── notebooks/           # 分析 notebook
├── tools/               # 辅助工具 (MI 优化, 迁移学习等)
└── requirements.txt     # 依赖包
```

## 快速开始

```bash
cd arm_bci_robot
pip install -r requirements.txt
python arm_bci/gui.py
```

## 硬件要求

| 组件 | 说明 |
|------|------|
| EEG 放大器 | NeuroDance / BrainVision LSL |
| EMG 传感器 | 咬肌 + 眼部肌电 |
| 机械臂 | 多自由度机械臂 (串口/TCP 控制) |
| 计算机 | Windows 10/11, Python 3.10+ |

## 环境

```bash
conda create -n bci_arm python=3.10 -y && conda activate bci_arm
pip install numpy scipy pyqt5 pyserial pylsl scikit-learn
```

---

**仓库**: https://github.com/wyubci/arm_bci_robot
