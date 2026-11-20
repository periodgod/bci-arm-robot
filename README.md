# EEG8 机械臂 BCI

这是世界机器人大赛机械臂 BCI 项目的当前交付链路：**32 路 EEG 的 C1–C8 解码**，可选的 **AUX 咬肌确认**，以及由同一命令 payload 同步驱动的本地 VR 与真实机械臂。项目不再提供旧的静息类/九分类、HybridBCIModel、六方向模型或第二条在线运行路径。

在既有八分类链路之外，`arm_bci.pure_mi_decoder` 提供一条独立的纯 MI
二进制解码结果，不替换或改写原输出链路。左/右手 MI 分别形成 `1/0`：
`1=左移`、`0=右移`、`10=前进`、`01=后退`、`11=上升`、`00=下降`，
`xx1=抓取`、`xx0=放置`。左咬肌提交已经完成的窗口（不包含当前窗口），
右咬肌撤销最近一个已完成窗口并回退一位，双侧咬肌产生 `stop_motion`
迅速停止。六方向
仍标记为持续指令；抓取/放置只标记需要沿用既有确认窗口。
累计三位后，若下一个分类窗口仍未检测到咬肌，系统默认取消并清空目前
全部编码计数；该窗口不计入下一组，不再抛出长度错误。
在线纯 MI 运行时，左右咬肌按 50 ms 轮询实时处理，不等待 EEG 窗口结束。
检测到咬肌后立即截断当前窗口且不保存其分类结果：左咬肌提交此前编码，
右咬肌回退一个已完成窗口，双侧咬肌立即停止；随后从新窗口继续。

在 GUI 的“准备配置”页，将“采集/在线范式”选择为“纯 MI 二分类
（左手=1 / 右手=0）”。这个统一选择会让采集页只生成 C1/C2 任务、
训练页保存纯 MI 模型标识，并让在线页按纯 MI 序列解码；选择“原八分类
范式”时继续使用原有流程。在线启动会拒绝范式与模型类别不匹配的组合。

## 当前功能边界

- 唯一模型任务为 EEG8：C1–C8；AUX 不参与 EEG 分类器的概率训练或加权融合。
- 默认在线策略为 `eeg8_two_phase_masseter`：窗口1按模型时长做 EEG+EMG 联合决策，窗口2做短时咬肌次数确认，C1–C4 再进入窗口3分段眨眼停止监测。
- 可选策略 `eeg8_two_phase_eeg_only` 不需要 AUX；通过 EEG 安全门限后只发送有时长限制的一次性命令，绝不进入持续移动。
- 真实机械臂默认关闭。只有“参数配置 → 真实机械臂”显式启用后，才会向 TCP 接收端连接和发送；VR 始终接收同一条命令。

当前目录中的历史模型 `models/arm_bci_gui.joblib` 不满足当前上线契约：其历史评估为 5 s 窗口、EEG 5 折准确率约 17.5%，不能用于真实机械臂。请用当前 2 s/2 s 契约重新训练、完成独立验证后才考虑启用真实机械臂。

## C1–C8 与机械臂映射

任务标签与动作映射的唯一代码源是 `arm_bci/eeg8_system.py` 中的 `CLASS_TO_ARM`。采集标签描述运动想象/执行任务；在线输出则固定映射为机械臂动作：

| EEG 类别 | 采集任务 | 机械臂动作 | TCP 命令 |
| --- | --- | --- | --- |
| C1 | 左手运动想象 | 左移 | `move_left` |
| C2 | 右手运动想象 | 右移 | `move_right` |
| C3 | 双手运动想象 | 前移 | `move_forward` |
| C4 | 双脚运动想象 | 后移 | `move_backward` |
| C5 | 左手握拳 | 上移 | `move_up` |
| C6 | 右手握拳 | 下移 | `move_down` |
| C7 | 双手握拳 | 抓取 | `close_gripper` |
| C8 | 双脚执行 | 释放 | `open_gripper` |

不要在 GUI、VR 或 TCP 接收端另写一份 C1–C8 映射。

## 在线安全流程

默认咬肌确认策略需要 LSL 提供按顺序排列的 **32 EEG + 4 AUX**：AUX1/2 是双眼眨眼，AUX3/4 是左/右咬肌。启动会拒绝采样率、前 32 路真实电极名或电极顺序与模型 `SignalConfig` 不一致的流；`CH1`/`EEG1` 这类占位标签不会被自动替换为默认蒙太奇。

```text
启动校验：EEG8 模型 v1 + 2s/2s SignalConfig + LSL 物理通道契约
    |
    +-- 默认策略：eeg8_two_phase_masseter
    |      |
    |      +-- 静息锁定时：AUX1/2 连续眨眼 3 次才解锁控制
    |      +-- 窗口1（时长匹配模型）：EEG+EMG 计数、候选映射、映射外概率乘倍率、单窗门限
    |      +-- 窗口2（均可配置）：独立咬肌执行次数达标 -> 执行窗口1结果；不足则取消
    |      +-- C1-C6：前后左右上下持续移动并进入窗口3，每个配置分段内眨眼次数达标则停止
    |      +-- C7-C8：抓取/释放单次执行后直接回窗口1
    |
    +-- EEG-only：EEG 安全检查 -> 有时长限制的一次性命令
```

窗口1的咬肌二进制状态只用于候选映射，不会单独发机械臂命令：

| 咬肌状态 | 窗口1 EEG 候选组 |
| --- | --- |
| `00` | C3/C4 |
| `01` | C1/C2 |
| `10` | C5/C6 |
| `11` | C7/C8 |

默认配置中 EEG 安全条件为 `command_confidence=0.62`、`min_margin=0.08`；窗口1只使用当前单个物理窗口，一个低门限结果不会被此前窗口“平均补救”。窗口2只统计合并后的独立咬合次数，不用左右方向重映射窗口1类别；一次双侧咬合只算一次。连续 5 次确认取消后，系统重新锁定。

窗口1的咬肌映射检测器由 `masseter_detector` 决定：`ncc` 模式下模板缺失、不兼容或未匹配都会安全拒绝，**不会**回退成阈值；`threshold` 使用 RMS/事件计数。NCC 模板时长加 50 ms 调度裕量必须严格小于当前窗口1时长。窗口2不使用 NCC，只统计咬肌执行事件。

完整时序、状态机与双端输出见 [在线控制流程图](docs/ONLINE_FLOW_DIAGRAMS.md)。

## 数据与训练契约

- 新采集记录必须保存 32 路 EEG 的真实通道名/索引，以及 0–4 路 AUX 元数据；默认咬肌策略上线需要完整 AUX1–AUX4。
- 窗口1时长可配置，但训练和在线必须与模型 `SignalConfig` 完全一致；间隔必须等于窗口时长，保证窗口不重叠。改变窗口1时长后必须按相同设置重新训练模型。
- 采集界面中的 `epoch_seconds` 是一次提示/任务 trial 的采集时长，**不是模型输入窗口**。训练按配置的窗口1时长/同等间隔从 trial 内生成 EEG8 窗口。
- 旧 5 s 模型、`HybridBCIModel` 或没有 `model_format_version=1` 的 joblib 会被安全拒绝；系统不会裁剪、补零、转换或回退到旧运行链路。

模型训练页必须把训练记录与独立测试记录分开。同一条记录不能同时用于拟合和测试；连续记录的评测还必须按 trial/session 分组，避免重叠窗口跨折泄漏。共享离线 bench 的边界与命令见 [Bench 集成说明](docs/BENCH_INTEGRATION.md)。

## 启动与检查

```powershell
cd E:\世界机器人大赛\projects\arm_bci_robot
& F:\anaconda\envs\bciwork_1\python.exe .\arm_bci_system.py gui
```

首次使用可建立环境并安装共享 bench：

```powershell
conda env create -f environment.yml
conda activate bciwork_1
cd E:\世界机器人大赛\wyu-bci-bench
pip install -e ".[signal]"
```

修改后至少运行：

```powershell
cd E:\世界机器人大赛\projects\arm_bci_robot
$py = 'F:\anaconda\envs\bciwork_1\python.exe'
& $py -m py_compile arm_bci\core.py arm_bci\domain\eeg8_model.py arm_bci\eeg8_system.py arm_bci\gui.py
& $py -m unittest tests.test_confirmation_state_machine tests.test_eeg8_data_contract tests.test_eeg8_model_compatibility tests.test_phase_emg_event_engine tests.test_runtime_session_config tests.test_eeg8_masseter_runtime tests.test_manipulator_protocol -v
```

## 配置与输出

个人配置写入 `config/gui_defaults.json`，该文件不入库；可从 [配置示例](config/gui_defaults.example.json) 开始。启动时会构建不可变 `RuntimeSessionConfig`：模型窗口、阈值来源、AUX 计数、目标、真实机械臂开关、主机、端口与速度均冻结到本次会话，运行中修改 GUI 不会改变已启动的输出。

`use_config_thresholds=true` 时，`realtime.decoder` 中的门限和确认参数作为本次会话值；设为 `false` 时使用模型中保存的解码器配置。无论哪一种，窗口1实时设置都必须与模型 `SignalConfig` 一致且不重叠。

每条已接受的命令只构造一次 `RobotCommand`，由 `robot_command_payload()` 转为一个 dict，再经 `CommandPayloadFanout` 同步分发给：

- `KinematicOpenGLArmWidget.execute_payload`（本地 VR）；
- `NetworkRobotForwarder.execute_payload`（仅真实机械臂已启用时实际建立 TCP 连接）。

TCP 使用持久连接、4 字节大端长度前缀和 UTF-8 JSON；限时位移会补发 `move_stop`。接收端协议与动作含义见 [模块指南](docs/ARM_BCI_MODULE_GUIDE.md)。

## 目录职责

```text
arm_bci/       当前 GUI、EEG8 训练/在线控制、VR 与 TCP 协议
config/        入库的示例配置；本地 gui_defaults.json 被忽略
data/          原始采集、派生数据与实验输出（不提交）
models/        本地模型、报告与模板（不提交）
docs/          当前操作说明、模块契约与在线流程
tests/         数据契约、状态机、协议与运行期安全回归测试
```

通用离线读取、预处理、模型比较、交叉验证和报告沉淀到 `E:\世界机器人大赛\wyu-bci-bench`；本项目只保留与机械臂应用直接相关的入口、当前操作文档与结果。
