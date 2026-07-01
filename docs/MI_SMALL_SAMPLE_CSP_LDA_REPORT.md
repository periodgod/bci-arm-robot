# 小样本左右手运动想象分类：可解释 CSP-LDA 优化报告

> 历史报告：本文件基于 S001-S010。当前 20 人低样本严格确认结果请以 `MI_LOW_SHOT_INNOVATION_REPORT.md` 为准。

## 1. 结论

当前推荐候选为 **跨受试者协方差先验 CSP-LDA**：

- 输入：21 个运动区 EEG 通道，8-30 Hz，3 秒 trial。
- 空间特征：6 个 CSP 分量。
- 分类器：带自动收缩的 LDA。
- 先验强度：`alpha=0.25`，该值仅由开发受试者 S001 选择。
- S002-S010 固定确认准确率：**67.65% +/- 18.73%**。
- 原 MNE Ledoit-Wolf CSP-LDA：**64.94% +/- 20.57%**。
- 平均提升：**2.72 个百分点**。
- 预滤波单 trial 推理：中位数 **0.265 ms**，P95 **0.519 ms**，1000 次重复。

该提升尚未达到统计显著：配对 Wilcoxon `p=0.609`，受试者 bootstrap 95% 区间为 `[-2.72, 9.38]` 个百分点。因此它是当前最合理的候选，但不能宣称已经稳定优于基线，更不能直接按 67.65% 进入机械臂控制。

## 2. 数据与严格验证协议

### 2.1 数据

- PhysioNet EEG Motor Movement/Imagery Dataset。
- S001-S010，共 10 名受试者。
- 每人使用 run 4、8、12 的左右手运动想象，共 45 个 trial。
- 每个 trial：64 通道，160 Hz，3 秒。
- 分类标签：左手 MI / 右手 MI。

### 2.2 防止数据泄漏

1. 交叉验证按完整 trial 划分，不把同一 trial 的重叠窗口放入训练集和测试集。
2. S001 只用于候选模型和 `alpha` 选择。
3. S002-S010 只用于固定配置确认。
4. 目标受试者的外层测试 trial 不参与 CSP、LDA、`alpha` 选择或对齐矩阵估计。
5. 跨受试者先验只使用非目标受试者的数据。

## 3. 实验结果

### 3.1 S001 候选筛选

| 模型 | 准确率 | 5 折总时间 |
|---|---:|---:|
| 8-14 Hz CSP(4)+LDA | 88.89% | 约 1.0 s |
| 8-30 Hz CSP(6)+LDA 基线 | 86.67% | 约 1.1 s |
| CSP 分量集成 | 82.22% | 约 2.5 s |
| 11 通道部署 CSP | 75.56% | 约 0.6 s |
| EEGNet，11 通道 | 75.56% | 见深度模型报告 |
| ShallowConvNet，11 通道 | 75.56% | 见深度模型报告 |
| Riemann 加权距离 LDA | 73.33% | 约 3.3 s |
| 生理 Laplacian ERD-LDA | 62.22% | 约 0.3 s |

S001 的 88.89% 不能代表跨受试者性能。8-14 Hz 配置固定到 S002-S010 后均值只有 64.69%，说明单受试者上的频带最优点不稳定。

### 3.2 S002-S010 固定确认

| 方法 | 平均准确率 | 标准差 | 平均 Kappa | 结论 |
|---|---:|---:|---:|---|
| 原 MNE CSP-LDA | 64.94% | 20.57% | 0.294 | 基线 |
| 目标受试者自估计 CSP，`alpha=0` | 64.20% | 21.53% | - | 自实现负控 |
| 协方差先验 CSP，`alpha=0.25` | **67.65%** | 18.73% | **0.347** | 当前候选 |
| 内层 CV 动态选择 `alpha` | 67.16% | 19.33% | - | 更慢，未超过固定值 |
| Euclidean Alignment 扩展 | 64.44% | - | - | S001 选中但确认失败，拒绝 |

`alpha=0.25` 对 9 名确认受试者的准确率依次为：

`93.33%, 48.89%, 75.56%, 62.22%, 44.44%, 95.56%, 66.67%, 48.89%, 73.33%`

个体差异非常大。S003、S006、S009 接近或低于随机水平，必须在真实用户校准阶段加入“不适合控制/需要重采”的质量门槛。

## 4. 数学推导

设第 `i` 个 trial 的带通、CAR 后 EEG 为：

`X_i in R^(C x T)`，其中 `C=21`，`T=480`。

### 4.1 trial 归一化协方差

先对每通道减去时间均值，再计算：

`C_i = X_i X_i^T / tr(X_i X_i^T)`。

迹归一化消除不同 trial 总能量尺度的影响，使 CSP 主要比较空间能量分布。

目标受试者类别 `c` 的样本协方差为：

`T_c = (1/N_c) sum_(i:y_i=c) C_i`。

### 4.2 等受试者权重的协方差先验

对每个非目标受试者 `s`，先独立求类别均值 `S_(s,c)`，再按受试者等权平均：

`P_c = (1/M) sum_(s=1)^M S_(s,c)`。

目标协方差与群体先验收缩：

`Sigma_c(alpha) = (1-alpha) T_c + alpha P_c`，`0 <= alpha <= 1`。

- `alpha=0`：退化为普通目标受试者 CSP。
- `alpha=1`：完全使用群体空间模式，忽略目标个体差异。
- `alpha=0.25`：保留 75% 个体协方差，同时用 25% 群体先验降低小样本方差。

从均方误差看，若 `T_c = Sigma_c^* + epsilon_t` 且先验偏差为 `b_c = P_c-Sigma_c^*`，则：

`E||Sigma_c(alpha)-Sigma_c^*||_F^2 = (1-alpha)^2 V_t + alpha^2 ||b_c||_F^2`。

这里 `V_t=E||epsilon_t||_F^2`。其理论最优值为：

`alpha^* = V_t / (V_t + ||b_c||_F^2)`。

小样本使 `V_t` 较大，因此适度使用群体先验有合理性；受试者差异使 `b_c` 不为零，因此不能把 `alpha` 设得过大。

### 4.3 CSP 广义特征值问题

CSP 寻找空间滤波器 `w`，最大化两类方差比：

`J(w) = (w^T Sigma_1 w) / (w^T Sigma_0 w)`。

等价的广义特征值问题为：

`Sigma_1 w = lambda (Sigma_0 + Sigma_1) w`。

取最大和最小特征值对应的各 3 个特征向量，得到 `W in R^(6 x 21)`。最大特征值方向偏向类别 1，最小特征值方向偏向类别 0，因此每个 CSP 分量都可通过头皮拓扑和类间方差解释。

### 4.4 对数方差特征

对投影信号 `Z_i = W X_i`，第 `k` 个特征为：

`f_(i,k) = log(var(Z_(i,k)) / sum_j var(Z_(i,j)))`。

归一化后，特征表示该空间模式在 trial 总 CSP 能量中的比例。

### 4.5 收缩 LDA

LDA 使用共享类内协方差：

`Sigma_f(lambda) = (1-lambda) S_f + lambda mu I`，

其中 `mu=tr(S_f)/K`，`lambda` 由训练数据自动估计。类别 `c` 的判别函数为：

`g_c(f) = f^T Sigma_f^-1 m_c - 0.5 m_c^T Sigma_f^-1 m_c + log(pi_c)`。

输出 `argmax_c g_c(f)`。收缩可避免 45 个 trial 下特征协方差接近奇异。

## 5. 速度与复杂度

### 5.1 训练

- trial 协方差：`O(N C^2 T)`。
- 21 维广义特征分解：`O(C^3)`。
- 6 维 LDA：开销很小。

最终固定模型在预滤波数据上的一次完整训练约 **4.9 ms**。复杂候选搜索只用于离线实验，不应放入在线系统。

### 5.2 在线推理

- CSP 投影：`O(K C T)`。
- 方差和归一化：`O(K T)`。
- LDA：`O(K)`。

在当前 Windows/Conda 环境中，预滤波单 trial 推理中位数为 **0.265 ms**，P95 为 **0.519 ms**。该数字不包含 8-30 Hz 流式带通；正式部署应使用有状态 SOS/IIR 滤波器，避免每次对整个窗口重复 `filtfilt`。

## 6. 生理可解释性

左右手 MI 的主要判别依据应是对侧感觉运动区的 mu/beta ERD/ERS：

- 左手 MI 更关注右半球 C4 周围空间模式。
- 右手 MI 更关注左半球 C3 周围空间模式。
- 8-30 Hz 覆盖 mu 和 beta 范围。
- CSP pattern 必须检查是否集中在 C3/C4/CP/FC 运动区；若主要集中在额极、颞侧或单个坏导，应判为伪迹模型。

建议在线训练完成后显示 6 个 CSP pattern、两类 log-variance 分布和混淆矩阵。准确率高但拓扑不符合运动区生理结构时，不允许进入控制阶段。

## 7. 与面部肌电同时使用的边界

公开 PhysioNet 数据没有“咬肌动作 + 左右手 MI 同时发生”的条件，因此当前准确率不能证明模型在强面部 EMG 污染下仍有效。面部 EMG 可经容积传导污染额区和颞区，并通过参考电极扩散到全部 EEG。

推荐控制时序：

1. 先用 0.3-0.5 秒 EMG 窗口选择控制轴。
2. EMG 回静后等待约 0.3 秒。
3. 再用 2-3 秒 EEG MI 窗口判断二分类方向。

这比“持续咬肌同时做 MI”更容易获得可靠 EEG。如果比赛必须同时执行，必须采集本人的同步 EMG+MI 数据，并增加以下对照：

- 只咬肌、不做 MI。
- 只做 MI、不咬肌。
- 咬肌与 MI 同时执行。
- 用 AUX EMG 线性回归/伪迹门控前后的准确率对比。

## 8. 下一阶段验收标准

建议每位真实用户至少采集 80-120 个完整 trial，按 session 分组验证，而不是随机切窗口。进入机械臂控制前应同时满足：

- 左右手平衡准确率至少 75%。
- Kappa 至少 0.5。
- 3 次不同时间校准的标准差不超过 8 个百分点。
- CSP pattern 以 C3/C4 周围为主。
- 同步 EMG 条件相对纯 MI 条件下降不超过 5 个百分点。
- 低置信度 trial 必须输出“未识别”，不强制二选一。

当前 67.65% 尚未达到该门槛。

## 9. 复现实验

```powershell
& F:\anaconda\Scripts\conda.exe run -n bciwork_1 python tools\mi_physionet_benchmark.py
& F:\anaconda\Scripts\conda.exe run -n bciwork_1 python tools\mi_physiology_optimization.py
& F:\anaconda\Scripts\conda.exe run -n bciwork_1 python tools\mi_multisubject_validation.py --folds 5 --seed 42
& F:\anaconda\Scripts\conda.exe run -n bciwork_1 python tools\mi_transfer_regularized_csp.py --folds 5 --inner-folds 3 --seed 42
```

主要结果文件：

- `data/experiments/physionet_eegmmidb_s001_left_right_mi_benchmark.json`
- `data/experiments/physionet_eegmmidb_s001_mi_optimization.json`
- `data/experiments/physionet_mi_multisubject_validation.json`
- `data/experiments/physionet_mi_transfer_regularized_csp.json`

## 10. 参考资料

- PhysioNet EEGMMIDB: https://physionet.org/content/eegmmidb/1.0.0/
- BCI Competition IV review: https://doi.org/10.3389/fnins.2012.00055
- FBCSP: https://doi.org/10.3389/fnins.2012.00039
- 近期 Riemannian filter-bank 方向: https://doi.org/10.1186/s40708-026-00295-0
- Riemannian geometry networks: https://doi.org/10.1109/JBHI.2024.3496757
