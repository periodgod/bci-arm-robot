# 贡献与迭代说明

本项目面向机械臂 BCI 应用交付。协作者 clone 后，先保证 GUI、采集、训练、实时控制这条主链路能跑，再做算法或报告迭代。

## 开发环境

```powershell
cd E:\世界机器人大赛\projects\arm_bci_robot
conda env create -f environment.yml
conda activate bciwork_1

cd E:\世界机器人大赛\wyu-bci-bench
pip install -e ".[signal]"
```

如果只更新项目 GUI/在线控制，至少要能运行：

```powershell
cd E:\世界机器人大赛\projects\arm_bci_robot
python arm_bci_system.py --help
python -m py_compile arm_bci\core.py arm_bci\domain\eeg8_model.py arm_bci\gui.py arm_bci\eeg8_system.py
python -m unittest tests.test_eeg8_masseter_runtime tests.test_manipulator_protocol -v
```

## 代码边界

- `arm_bci_robot`：采集、GUI、在线控制、机械臂动作映射、项目报告和现场配置。
- `wyu-bci-bench`：通用 EEG 数据读取、连续 raw_stream 处理、坏道/QC、epoch 切分、特征、模型、交叉验证和可复用可视化。
- 新模型、新特征、新验证方式优先进入 `wyu-bci-bench/src/wyu_bci_bench/`，本项目只保留调用入口和项目专属配置。
- 一次性分析脚本可以先放 `tools/analysis/`，但确认会复用后应迁移到 bench。

## 实时策略

当前真实系统策略固定为：

1. EEG/MI 在 2s 窗口内输出 C1-C8 概率。
2. AUX1/2 眼电只用于三次眨眼切换阶段。
3. AUX3/4 左右咬肌在随后 1s 确认窗口内完成 NCC/阈值门控。
4. 咬肌不直接控制机械臂，最终动作仍由 EEG 八分类概率决定。

修改实时链路时必须同步更新：

- `arm_bci/domain/eeg8_model.py`、`arm_bci/core.py` 的模型、配置或门控接口。
- `arm_bci/gui.py` 的实时控制页面。
- `tests/test_eeg8_masseter_runtime.py` 的策略测试。
- `README.md` 和 `docs/BENCH_INTEGRATION.md` 中对应说明。

## 数据和产物

不要提交：

- `data/`
- `models/`
- `logs/`
- `outputs/`
- `config/gui_defaults.json`
- `*.npz`, `*.joblib`, `*.pt`, `*.pth`, `*.onnx`

可以提交：

- 文档、结构图、代码、测试。
- 小型示例配置，例如 `config/gui_defaults.example.json`。
- 复现实验命令和 JSON 摘要，但不要提交原始采集数据或大模型权重。

## 提交前检查

```powershell
$env:PYTHONIOENCODING='utf-8'
[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new()

python -m py_compile arm_bci\core.py arm_bci\domain\eeg8_model.py arm_bci\gui.py arm_bci\eeg8_system.py
python -m unittest tests.test_eeg8_masseter_runtime tests.test_manipulator_protocol -v
python arm_bci_system.py --help
```

如果改了 bench：

```powershell
cd E:\世界机器人大赛\wyu-bci-bench
python -m unittest discover tests -v
wyu-bci-bench --help
```
