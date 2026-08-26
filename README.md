# SmolVLA Piper Towel Folding（叠毛巾）

中文 | English

**SmolVLA（SmolVLM2-500M）驱动**的双 Piper 双臂真实机器人叠毛巾项目：SmolVLA 在三路 RealSense 观测下，通过**连续 authority 塑形机制**主导手臂轨迹，在安全边界内完成 HQ60 毛巾折叠。**多次完整 750 步真机运行全部 `status=0`**，轨迹丝滑、无安全停机。

A real-robot towel-folding project driven by **SmolVLA (SmolVLM2-500M)** on dual Piper arms. SmolVLA leads the arm trajectory under a **continuous authority-shaping mechanism** bounded by safety checks, completing HQ60 towel folding under three RealSense cameras. **Multiple full 750-step runs finished with `status=0`** — smooth trajectory, zero safety stops.

## Caution

真机运行时必须保持急停可用。机械臂在无物理支撑时不得自动失能，否则可能因重力下坠。
Keep the emergency stop ready during real-robot operation. Never automatically disable unsupported arms, because they may fall under gravity.

---

## 目录 / Table of Contents

1. [项目状态](#1-项目状态)
2. [仓库结构](#2-仓库结构)
3. [SmolVLA 模型与架构](#3-smolvla-模型与架构)
4. [训练](#4-训练)
5. [RTC 实时推理管线](#5-rtc-实时推理管线)
6. [连续 authority 塑形设计（核心）](#6-连续-authority-塑形设计核心)
7. [硬件与环境](#7-硬件与环境)
8. [输入输出](#8-输入输出)
9. [部署与运行](#9-部署与运行)
10. [安全模型](#10-安全模型)
11. [真机结果](#11-真机结果)
12. [已知问题与教训](#12-已知问题与教训)
13. [数据与模型管理](#13-数据与模型管理)
14. [上游与许可](#14-上游与许可)

---

## 1. 项目状态

### 已完成 ✅

- **SmolVLA 模型训练**：SmolVLM2-500M 从已有 50k checkpoint 热启动训练 5k 步（`smolvla_hq60_newonly_from50k_b8_5k_v2`）；
- **SmolVLA RTC 真机推理**：异步实时通道（Real-Time Channel）引导式推理，端到端部署到双 Piper 机械臂；
- **连续 authority 塑形**：`target = anchor + authority × (Smol_EMA − anchor)`，SmolVLA 在安全边界内主导手臂轨迹；
- **7 层防抖栈**：消除切换抖动，真机轨迹丝滑（`joint_step` 大多 0.002–0.03 rad）；
- **验证**：离线 4 场景回归 + 真机 dry-run + **多次完整 750 步真机运行全部 `status=0`**。

### 核心亮点

| 指标 | 数值 |
|---|---|
| 任务 | HQ60 叠毛巾（双 Piper 双臂） |
| 主导策略 | SmolVLA（SmolVLM2-500M，VLM 驱动的 VLA） |
| 观测 | 3× RealSense D435i RGB + 28 维双臂状态 |
| 动作 | 14 维绝对关节位置目标 |
| 单次时长 | 750 步 @ 30 Hz ≈ 25 s |
| 真机结果 | 2 次完整运行 `status=0`，SmolVLA 参与 83–84% 混合步 |

本仓库是**完整可运行工程**：SmolVLA 推理框架（`src/lerobot`）+ 叠毛巾部署脚本（`scripts/`）。**不包含数据集与模型 checkpoint**。

---

## 2. 仓库结构

本仓库是 LeRobot 的 SmolVLA 分支（fork），在其基础上加入叠毛巾任务的训练、推理与部署代码：

```
├── src/lerobot/               # LeRobot 框架（含 SmolVLA 策略）
│   ├── policies/smolvla/      # SmolVLA 策略实现（SmolVLM2-500M + 动作头）
│   └── async_inference/       # RTC 异步实时推理框架（policy server + robot client）
├── scripts/                   # 叠毛巾部署脚本（混合塑形控制器 + RTC 推理）
├── tests/                     # 框架回归测试
├── docs/ examples/            # 框架文档与示例
├── docker/ media/             # 容器与文档素材
├── pyproject.toml setup.py    # 工程配置
└── README.md                  # 本文档
```

关键子目录说明：

| 路径 | 说明 |
|---|---|
| `src/lerobot/policies/smolvla/` | **SmolVLA 策略本体**：`modeling_smolvla.py`（模型）、`configuration_smolvla.py`（配置）、`processor_smolvla.py`（图像/文本处理）、`smolvlm_with_expert.py`（VLM 专家头） |
| `src/lerobot/async_inference/` | **RTC 推理框架**：`policy_server.py`（推理服务）、`robot_client.py`（机器人端客户端）、`helpers.py`、`configs.py` |
| `scripts/` | **叠毛巾部署**：`act_smolvla_hybrid_towel_blend.py`（混合塑形控制器）、`blend_core.py`（纯数学核心）、`test_blend_offline.py`（离线测试）、`run_hybrid_towel_blend750*.sh`（一键运行）、`smolvla_piper_rtc_*.py`（RTC 推理入口） |

---

## 3. SmolVLA 模型与架构

SmolVLA 是视觉-语言-动作（VLA）策略，核心是 **SmolVLM2-500M** 视觉语言模型：

| 组件 | 说明 |
|---|---|
| 视觉骨干 | SigLIP 图像编码器，融合多视角图像 |
| 语言骨干 | SmolVLM2-500M（~500M 参数） |
| 动作头 | 从 VLM 特征解码出连续关节动作（expert head） |
| 观测输入 | 3 路图像（left / middle / right）+ 28 维双臂状态 |
| 动作输出 | 14 维绝对关节位置目标 |

本项目在 LeRobot 框架内完成 SmolVLA 策略的配置、训练与推理接入，重点解决**把 VLM 驱动的策略安全、稳定地部署到真实双臂机械臂**这一问题。

---

## 4. 训练

### 4.1 数据集

| 数据集 | 用途 | 规模 |
|---|---|---|
| `newonly` 高质量子集 | SmolVLA 训练 | 高质量毛巾折叠示范 |
| `towel_fold_dataset` | 基线训练 | 60 episodes / 42373 frames |
| `towel_fold_dataset_aug_v1` | 基线增强 | 120 episodes / 85187 frames |

### 4.2 训练策略

- **热启动**：SmolVLA 从已有 50k 步 checkpoint 继续训练（而非从零），显著缩短收敛时间；
- **训练量**：5k 步，batch size 8；
- **checkpoint**：`smolvla_hq60_newonly_from50k_b8_5k_v2/checkpoints/005000`；
- 训练基于高质量示范子集，避免低质量轨迹污染策略。

> 数据集与 checkpoint 均不进入本仓库，按第 13 节管理。

---

## 5. RTC 实时推理管线

RTC（Real-Time Channel）是面向真机部署的异步推理框架：

```
robot_client ──(观测)──▶ policy_server ──(推理队列)──▶ 动作
      ▲                                                     │
      └──────────────────(异步返回)─────────────────────────┘
```

| 配置项 | 值 | 说明 |
|---|---|---|
| 推理模式 | guided（引导式） | 机器人端引导推理节奏 |
| 执行时域 `execution_horizon` | 10 | 模型预测未来 10 步动作 |
| 队列阈值 `queue_threshold` | 30 | 动作队列低于该值触发新推理 |

关键点：

- **推理与执行解耦**：`policy_server` 负责模型推理，`robot_client` 负责与机械臂通信，互不阻塞；
- **动作队列缓冲**：推理结果入队，执行端持续消费，保证 30 Hz 输出不中断；
- **队列为空回退**：队列暂时为空时安全回退到已有动作，避免输出空洞；
- 推理脚本：`scripts/smolvla_piper_rtc_*.py`。

---

## 6. 连续 authority 塑形设计（核心）

### 6.1 塑形公式

手臂 12 个关节的命令：

```
target_arm = anchor_arm + authority × (smol_ema_arm − anchor_arm)
```

- `authority ∈ [0, max_authority]`，连续权重，**不存在二选一切换**；
- 锚点由**已验证基线策略**提供，负责保持基础轨迹与夹爪控制；
- SmolVLA 候选动作先过 EMA 平滑（`smol_ema`），再以 `authority` 权重叠加到锚点上；
- SmolVLA 修正量有**幅度上限**与**每步 slew 上限**，只能在安全边界内"小修"——正是这一持续修正让 **SmolVLA 主导塑形**。

### 6.2 authority 动力学

| 事件 | authority 更新 |
|---|---|
| SmolVLA 达标（accepted） | `a += 0.15 × (max − a)` 指数上升 |
| 软拒绝 / 队列为空 | `a ×= 0.85` |
| 硬拒绝（越界 / 原始分歧过大） | `a ×= 0.5` |

信任变化永远平滑 → 命令自然平滑。当 SmolVLA 与锚点高度一致时 authority 持续逼近上限，SmolVLA 权重占主导；一旦分歧超限，authority 平滑回落，安全收敛。

### 6.3 7 层防抖栈

| 层 | 机制 | 作用 |
|---|---|---|
| 1 | 连续 authority 塑形 | 消除二进制切换（旧版 165 次切换是抖动根因） |
| 2 | 候选 EMA（`smol-lowpass-alpha` 0.25） | SmolVLA 候选动作先过一阶低通 |
| 3 | 修正每步 slew 限幅（0.02 rad/step） | 修正每步最多变化 0.02 rad |
| 4 | 修正幅度上限（0.15 rad） | 修正 ≤ 0.15 rad |
| 5 | 全局低通（`global-lowpass-alpha` 0.85） | 最终命令再平滑 |
| 6 | 夹爪恒为锚点策略所有 | 夹爪不参与塑形 |
| 7 | 信任门 | 软门（分歧 0.14）+ 硬门（绝对限位、步长/分歧上限） |

### 6.4 参数表（基线 v1）

| 参数 | 值 | 作用 |
|---|---|---|
| `max-authority` | 0.9 | SmolVLA 权重上限（留 10% 锚点） |
| `authority-up-rate` | 0.15 | 每接受步 authority 指数逼近 max |
| `authority-decay` | 0.85 | 软拒绝 / 空队列乘性衰减 |
| `authority-hard-decay` | 0.5 | 硬拒绝（越界）乘性衰减 |
| `smol-lowpass-alpha` | 0.25 | 候选 EMA 权重 |
| `smol-raw-step-limit` | 0.60 | 原始候选单步上限（硬） |
| `smol-raw-policy-disagreement` | 0.50 | 原始候选与锚点分歧上限（硬） |
| `policy-disagreement` | 0.14 | 软信任门：过滤后候选与锚点最大分歧 |
| `joint-step-limit` | 0.10 | 软信任门：过滤后候选单步上限 |
| `smol-correction-limit` | 0.15 | 修正幅度上限（rad） |
| `correction-alpha` | 0.5 | 修正 EMA 权重 |
| `correction-step-limit` | 0.02 | 修正每步 slew 上限（rad/step） |
| `global-lowpass-alpha` | 0.85 | 最终命令低通 |
| `act-joint-step-limit` | 1.0 | 锚点单步上限 |
| `act-tracking-limit` | 3.0 | 锚点跟踪误差上限 |
| `handoff-step` | 300 | 第 300 步切入混合模式 |
| `max-actions` | 750 | 总步数（30 Hz ≈ 25 s） |

**变体：**

| 变体 | 相对基线的改动 |
|---|---|
| v2 | `max-authority` 0.95、`smol-correction-limit` 0.18、`correction-step-limit` 0.025 |
| v3 | `policy-disagreement` 0.16、`global-lowpass-alpha` 0.90 |

---

## 7. 硬件与环境

### 7.1 Piper 双臂

| 机械臂 | SocketCAN 接口 |
|---|---|
| 左从臂 | can1 |
| 右从臂 | can0 |

启动 CAN：

```bash
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 txqueuelen 1000
sudo ip link set can0 up
# 同法配置 can1
```

### 7.2 RealSense 相机

三路 D435i（left / middle / right），使用稳定 udev 别名：

```bash
readlink -e /dev/camera_left
readlink -e /dev/camera_middle
readlink -e /dev/camera_right
```

不要长期写死 `/dev/videoN`。三路 640×480、30 FPS 同时运行时，建议保持 USB 3.x（5000M）枚举。

### 7.3 软件环境

- Ubuntu 22.04，Python 3.12；
- Conda 环境：`lerobot_v30`（基线）、`smolvla_piper`（SmolVLA + 塑形控制器）；
- 运行时根目录：`/home/databall_02/VLA/smolvla_piper_runtime/lerobot_official`。

---

## 8. 输入输出

观测：

```
obs = {
    "observation.images.left":   image_tensor,   # 640×480 RGB
    "observation.images.middle": image_tensor,
    "observation.images.right":  image_tensor,
    "observation.state":         state_tensor,   # 28 维：双臂位置/effort 交错
}
```

动作：14 维绝对关节位置目标（左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪）。注意这是**绝对位置目标**，不要解释为速度/增量控制。

---

## 9. 部署与运行

### 9.1 部署

将控制器与核心拷贝到运行脚本同目录：

```bash
# 在 $SMOL_ROOT/scripts/ 下
cp act_smolvla_hybrid_towel_blend.py blend_core.py run_hybrid_towel_blend750*.sh ./
```

### 9.2 一键运行（需真机在场）

```bash
bash run_hybrid_towel_blend750.sh
```

脚本分两阶段：

1. 用 `lerobot_v30` 环境 `reset_piper_pose.py --arm both --left-can can1 --right-can can0 --execute` 安全归位；
2. 用 `smolvla_piper` 环境运行 750 步混合执行（`handoff-step 300`，第 300 步切入混合）。

### 9.3 运行产物

结果写入 `$EXPERIMENT_ROOT/blend_one_click_runs/$RUN_ID/`：

| 文件 | 内容 |
|---|---|
| `reset.log` | 归位日志 |
| `hybrid_full750.log` | 750 步执行日志（authority / 拒绝 / 修正 / tracking） |
| `summary.txt` | 运行汇总 |

---

## 10. 安全模型

| 环节 | 校验 |
|---|---|
| 锚点 tick | 沿用已验证基线语义（joint step ≤ 1.0、tracking ≤ 3.0） |
| 混合 tick | 额外执行绝对关节限位校验 + 修正 slew 校验 |
| 夹爪 | 恒为锚点策略所有，SmolVLA 不参与 |
| 修正 | 幅度 ≤ 0.15 rad、每步 slew ≤ 0.02 rad，双重限位 |
| 启动姿态 | 偏离起始位 > 0.12 rad 拒绝启动 |
| 退出/异常 | 发送当前测量姿态作为 hold command，**不自动 disable** |

安全原则：SmolVLA 候选永远只是"小修"，只能塑形、不能主导失控；一切异常走安全回退。

---

## 11. 真机结果

**2 次完整 750 步运行，全部 `status=0`，无安全停机：**

| 运行 | 时间 | status | SmolVLA 混合步 | 拒绝 | 回退 | authority | 修正 (rad) |
|---|---|---|---|---|---|---|---|
| 20260826_111826 | 08-26 | 0 | 379 / 450 (84%) | 55 | 71 | 0.76–0.90 稳定 | 0.08–0.12 |
| 20260826_113844 | 08-26 | 0 | 374 / 450 (83%) | 62 | 76（含 14 空队列） | 尾段钉 0.90 | 0.06–0.11 |

**关键指标：**

- 命令步长 `joint_step` 大多 **0.002–0.03 rad**（旧二进制切换版抖动段 p90 0.048 → 丝滑）；
- 跟踪误差 `tracking` 大多 **< 0.15 rad**；
- 修正量持续 **0.06–0.12 rad** → **SmolVLA 真实主导塑形而非摆设**；
- SmolVLA 参与 83–84% 的混合步，拒绝多为软拒绝（分歧恰过门限），被收回再采纳；
- 对比旧二进制切换版（094458 运行：SmolVLA 仅 36 步 / 165 次切换）——**切换抖动彻底消除**。

---

## 12. 已知问题与教训

1. **float 边界 bug（已修）**：correction-step 校验严格 `>` 无容差，float32 把 0.02 边界测成 0.020000 误停（566 步处）。已改为 `> limit + 1e-5`，与离线测试一致。
2. **安全收权重对准**：若 SmolVLA 与锚点分歧骤增（如 0.30–0.65 rad），authority 会正确塌陷，机械臂物理上重新对准目标位姿（tracking 峰值 ~0.58 rad，单步峰值 0.25）。这是安全系统在正确工作，**不应被"抹平"**。
3. **handoff 后越界暖机**：SmolVLA 切入后头几帧偶发 joint_8 越界（>2.10），被硬拒绝吸收，authority 短暂为 0 后自行恢复。

---

## 13. 数据与模型管理

以下内容不存入本仓库：

```
outputs/  experiments/  logs/
downloaded_models/  datasets/  backups/
*.safetensors  *.pt / *.pth
数据集视频  评估 rollout
*_one_click_runs/  latest
```

正式模型应单独记录：数据集名称与 episode 数、checkpoint 路径、训练参数、SHA256、真机测试结果。

---

## 14. 上游与许可

- https://github.com/huggingface/lerobot（SmolVLA、RTC 推理框架）
- https://github.com/GrahamZen/lerobot_piper（Piper 部署参考）

请遵循上游 LICENSE（本仓库沿用 LeRobot 的 Apache License 2.0）。

---

# English

## 1. Project status

### Done ✅

- **SmolVLA training**: SmolVLM2-500M warm-started from a 50k checkpoint, trained 5k steps (`smolvla_hq60_newonly_from50k_b8_5k_v2`);
- **SmolVLA RTC real-robot inference**: async Real-Time Channel guided inference deployed end-to-end on dual Piper arms;
- **Continuous authority shaping**: `target = anchor + authority × (Smol_EMA − anchor)` — SmolVLA leads the trajectory within safety bounds;
- **7-layer jitter-reduction stack**: eliminated switch jitter; `joint_step` mostly 0.002–0.03 rad;
- **Verification**: offline 4-scenario regression + real-robot dry-runs + **multiple full 750-step runs, all `status=0`**.

### Highlights

| Metric | Value |
|---|---|
| Task | HQ60 towel folding (dual Piper arms) |
| Leading policy | SmolVLA (SmolVLM2-500M, VLM-driven VLA) |
| Observation | 3× RealSense D435i RGB + 28-dim dual-arm state |
| Action | 14-dim absolute joint-position targets |
| One run | 750 steps @ 30 Hz ≈ 25 s |
| Real-robot | 2 full runs `status=0`, SmolVLA active in 83–84% of blend steps |

This is a complete, runnable project: the SmolVLA inference framework (`src/lerobot`) plus the towel-folding deployment scripts (`scripts/`). Datasets and checkpoints are excluded.

## 2. Repository structure

This repository is a LeRobot fork with SmolVLA, plus the towel-folding training / inference / deployment code:

```
├── src/lerobot/               # LeRobot framework (incl. SmolVLA policy)
│   ├── policies/smolvla/      # SmolVLA policy implementation (SmolVLM2-500M + action head)
│   └── async_inference/       # RTC async inference framework (policy server + robot client)
├── scripts/                   # Towel-folding deployment (blend controller + RTC inference)
├── tests/                     # Framework regression tests
├── docs/ examples/            # Framework docs and examples
├── docker/ media/             # Containers and doc assets
├── pyproject.toml setup.py    # Project config
└── README.md                  # This file
```

Key directories:

| Path | Description |
|---|---|
| `src/lerobot/policies/smolvla/` | **SmolVLA policy**: `modeling_smolvla.py`, `configuration_smolvla.py`, `processor_smolvla.py`, `smolvlm_with_expert.py` |
| `src/lerobot/async_inference/` | **RTC framework**: `policy_server.py`, `robot_client.py`, `helpers.py`, `configs.py` |
| `scripts/` | **Deployment**: `act_smolvla_hybrid_towel_blend.py` (blend controller), `blend_core.py` (pure-math core), `test_blend_offline.py`, `run_hybrid_towel_blend750*.sh`, `smolvla_piper_rtc_*.py` |

## 3. SmolVLA model and architecture

SmolVLA is a vision-language-action (VLA) policy built on **SmolVLM2-500M**:

| Component | Description |
|---|---|
| Vision backbone | SigLIP image encoder, fuses multi-view images |
| Language backbone | SmolVLM2-500M (~500M params) |
| Action head | Decodes continuous joint actions from VLM features |
| Observation | 3 images (left / middle / right) + 28-dim dual-arm state |
| Action | 14-dim absolute joint-position targets |

This project integrates SmolVLA into the LeRobot framework for config, training, and inference — with the focus on **safely deploying a VLM-driven policy on real dual-arm robots**.

## 4. Training

### 4.1 Datasets

| Dataset | Purpose | Size |
|---|---|---|
| `newonly` high-quality subset | SmolVLA training | high-quality towel-folding demos |
| `towel_fold_dataset` | Baseline training | 60 episodes / 42373 frames |
| `towel_fold_dataset_aug_v1` | Baseline augmentation | 120 episodes / 85187 frames |

### 4.2 Strategy

- **Warm start**: resume from a 50k-step checkpoint instead of from scratch — much faster convergence;
- **Training**: 5k steps, batch size 8;
- **Checkpoint**: `smolvla_hq60_newonly_from50k_b8_5k_v2/checkpoints/005000`;
- Trained on the high-quality subset to avoid low-quality trajectory contamination.

## 5. RTC real-time inference pipeline

RTC (Real-Time Channel) is an async inference framework for real-robot deployment:

```
robot_client ──(observation)──▶ policy_server ──(inference queue)──▶ action
      ▲                                                                │
      └────────────────────────(async return)──────────────────────────┘
```

| Config | Value | Description |
|---|---|---|
| Mode | guided | robot-driven inference cadence |
| `execution_horizon` | 10 | predicts 10 future steps |
| `queue_threshold` | 30 | triggers new inference when queue drops below |

Key points: inference and execution are decoupled (policy server vs robot client); the action-queue buffer keeps 30 Hz output uninterrupted; on an empty queue the controller safely falls back to the existing action. Inference entry points: `scripts/smolvla_piper_rtc_*.py`.

## 6. Continuous authority shaping (core)

### 6.1 Formula

For the 12 arm joints:

```
target_arm = anchor_arm + authority × (smol_ema_arm − anchor_arm)
```

- `authority ∈ [0, max_authority]`, fully continuous — no binary switch;
- The anchor comes from a validated baseline policy, holding the base trajectory and the gripper;
- SmolVLA candidates pass an EMA (`smol_ema`) and are blended onto the anchor by `authority`;
- The correction is bounded in magnitude and per-step slew, so it only fine-shapes within safety limits — and it is precisely this sustained correction that lets **SmolVLA lead**.

### 6.2 Authority dynamics

Accepted step: `a += 0.15 × (max − a)`. Soft reject / empty queue: `a ×= 0.85`. Hard reject (out-of-bounds / raw disagreement): `a ×= 0.5`.

### 6.3 The 7-layer jitter-reduction stack

| # | Mechanism | Effect |
|---|---|---|
| 1 | Continuous authority shaping | removes binary switching (165 switches was the old jitter root cause) |
| 2 | Candidate EMA (0.25) | smooths SmolVLA candidates |
| 3 | Correction slew limit (0.02 rad/step) | per-step bound |
| 4 | Correction magnitude cap (0.15 rad) | bound on the correction |
| 5 | Global lowpass (0.85) | smooths the final command |
| 6 | Gripper owned by anchor | gripper not shaped |
| 7 | Trust gates | soft (disagreement 0.14) + hard (absolute/step bounds) |

### 6.4 Parameters (baseline v1)

`max-authority 0.9`, `authority-up-rate 0.15`, `authority-decay 0.85`, `authority-hard-decay 0.5`, `smol-lowpass-alpha 0.25`, `smol-raw-step-limit 0.60`, `smol-raw-policy-disagreement 0.50`, `policy-disagreement 0.14`, `joint-step-limit 0.10`, `smol-correction-limit 0.15`, `correction-alpha 0.5`, `correction-step-limit 0.02`, `global-lowpass-alpha 0.85`, `act-joint-step-limit 1.0`, `act-tracking-limit 3.0`, `handoff-step 300`, `max-actions 750`.

Variants: **v2** → `max-authority 0.95`, `smol-correction-limit 0.18`, `correction-step-limit 0.025`. **v3** → `policy-disagreement 0.16`, `global-lowpass-alpha 0.90`.

## 7. Hardware and environment

Dual Piper arms — left `can1`, right `can0` (SocketCAN, 1 Mbps). Three RealSense D435i cameras with udev aliases `/dev/camera_left`, `/dev/camera_middle`, `/dev/camera_right`. Ubuntu 22.04, Python 3.12; conda envs `lerobot_v30` (baseline) and `smolvla_piper` (SmolVLA + blend). Runtime root: `/home/databall_02/VLA/smolvla_piper_runtime/lerobot_official`.

## 8. I/O

Observation: 3 RGB images + 28-dim dual-arm state (positions/efforts interleaved). Action: 14-dim absolute joint-position targets (left 6 joints + left gripper + right 6 joints + right gripper). Do not reinterpret the action space as velocity/delta control.

## 9. Deployment and running

```bash
bash run_hybrid_towel_blend750.sh
```

Phase 1: safe reset with `reset_piper_pose.py` (`lerobot_v30`). Phase 2: 750-step blend run (`smolvla_piper`), handoff at step 300. Results land in `$EXPERIMENT_ROOT/blend_one_click_runs/$RUN_ID/` (`reset.log`, `hybrid_full750.log`, `summary.txt`).

## 10. Safety model

Anchor ticks keep the proven baseline limits (joint step 1.0, tracking 3.0). Blended ticks add absolute joint-limit validation and correction-slew validation. Gripper is owned by the anchor; the correction is double-bounded, so SmolVLA can only fine-shape. Start-pose check rejects runs > 0.12 rad from the proven start. On exit/error the measured pose is held; arms are never auto-disabled.

## 11. Real-robot results

**2 full 750-step runs, all `status=0`, no safety stop:**

| Run | status | SmolVLA blended steps | rejected | fallback | authority | corr (rad) |
|---|---|---|---|---|---|---|
| 20260826_111826 | 0 | 379 / 450 (84%) | 55 | 71 | 0.76–0.90 | 0.08–0.12 |
| 20260826_113844 | 0 | 374 / 450 (83%) | 62 | 76 | pinned 0.90 tail | 0.06–0.11 |

`joint_step` mostly 0.002–0.03 rad; `tracking` mostly < 0.15 rad; sustained correction 0.06–0.12 rad proves **SmolVLA genuinely leads the shaping**; SmolVLA is active in 83–84% of blend steps. Compared with the old binary-switch version (SmolVLA 36 steps / 165 switches) — switch jitter is eliminated.

## 12. Known issues

1. **Float boundary bug (fixed)**: the correction-slew check used a strict `>` with no epsilon; float32 measured 0.02 as 0.020000 and stopped the run at step 566. Fixed with `> limit + 1e-5`, matching the offline test.
2. **Safety re-aim**: on a hard divergence (0.30–0.65 rad), authority correctly collapses and the arm physically re-aims (tracking peak ~0.58 rad). This is the safety system working — do not smooth it away.
3. **Handoff warm-up**: SmolVLA occasionally emits out-of-bounds `joint_8` (> 2.10) right after handoff; the hard-reject path absorbs it and authority recovers on its own.

## 13. Artifact management

No datasets, checkpoints, or rollout videos in this repo (`outputs/`, `experiments/`, `logs/`, `downloaded_models/`, `datasets/`, `backups/`, `*.safetensors`, `*.pt`, `*.pth`, `*_one_click_runs/`, `latest`). Record dataset id, checkpoint path, training config, SHA256, and evaluation results for released models.

## 14. Upstream and license

- https://github.com/huggingface/lerobot (SmolVLA, RTC inference)
- https://github.com/GrahamZen/lerobot_piper (Piper deployment reference)

See upstream LICENSE (this repo follows LeRobot's Apache License 2.0).
