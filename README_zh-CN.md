<div align="center">

# 🦾 双臂 SmolVLA 叠毛巾

**一条视觉语言动作模型(VLA)在真实机械臂上叠毛巾——SmolVLA 在已验证的 ACT 安全基线上做"塑形",连续混合,不跳变、不会失控。**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Policy](https://img.shields.io/badge/policy-SmolVLA%20%E2%80%A2%20SmolVLM2--500M-orange.svg)
![Robot](https://img.shields.io/badge/robot-dual%20AgileX%20Piper-brightgreen.svg)
![Control](https://img.shields.io/badge/control-30%20Hz%20real--time-9cf.svg)

[**English**](README.md) · 中文版

</div>

---

> ## 🎬 真机演示
>
> 一次连续拍摄:双臂看着三路相机画面,**自己**完成抓取 → 折叠 → 释放;随着运行推进,
> SmolVLA 对动作的塑形程度越来越高。

<div align="center">

<img src="assets/towel/demo_hero.webp" width="320" alt="SmolVLA 真机叠毛巾演示">

</div>

---

## 亮点

- **真 VLA、真机械臂** — [SmolVLA](https://huggingface.co/HuggingFaceTB/SmolVLA-500M-Instruct)
  (`SmolVLM2-500M` 视觉语言骨干 + 动作专家头)把三路相机画面 + 关节状态直接映射成关节动作。
- **两条策略,一次运行** — 已验证的 **ACT** 控制器作为安全基线并**始终持有夹爪**;
  **SmolVLA** 持续"塑形"基线轨迹。
- **混合,而不是切换** — `authority ∈ [0, 0.9]` 连续升降,不存在二进制的切换点,
  所以在第 300 步**不会有指令跳变**。
- **结构上就安全** — SmolVLA 修正量有幅度上限(≤ 0.15 rad)与每步 slew 上限(≤ 0.02 rad/步);
  夹爪永远不会交给 SmolVLA。
- **一条命令部署** — `bash run_hybrid_towel_blend750.sh` 自动归位双臂并跑完真机 750 步叠毛巾。

本仓库是纯 ACT 基线的 **SmolVLA 演进版**——见 [相关项目](#相关项目)。
它依赖的 ACT checkpoint 正是那个项目发布的同一个 checkpoint。

---

## 目录

- [核心思路:两条策略,一个混合](#核心思路两条策略一个混合)
- [混合与安全数学](#混合与安全数学)
- [系统架构](#系统架构)
- [真机一条命令](#真机一条命令)
- [真机运行结果](#真机运行结果)
- [硬件与机器人接口](#硬件与机器人接口)
- [仓库布局](#仓库布局)
- [开始 / 复现](#开始--复现)
- [配套图文课程](#配套图文课程)
- [相关项目](#相关项目)
- [下载](#下载)
- [状态与边界](#状态与边界)
- [致谢与许可](#致谢与许可)

---

## 核心思路:两条策略,一个混合

叠一块可形变的毛巾,既要**稳**(一个已验证好用的控制器),又要**灵活**(一个能看到画面就反应的
策略)。本项目的做法是让两条策略**同时跑**,而不是二选一:

| | ACT 安全基线 | SmolVLA 塑形器 |
|---|---|---|
| 角色 | 提供已验证的基础轨迹;**始终持有夹爪** | 看着相机,提出一个*修正量* |
| 动作来源 | 保守、经验证 | 学习而来、视觉驱动、表达力更强 |
| 对机械臂的掌控 | 1.0 → 随 SmolVLA 被信任而让渡 | 0 → 一路升到 0.9(只要保持一致) |

从第 300 步起,两条输出每 tick(30 Hz)融合一次:

```text
target_arm = safe_base_arm + authority × (smol_ema_arm − safe_base_arm)
```

SmolVLA 与基线一致,就赢得影响力;一旦分歧过大或越界,它的影响力就衰减,指令平滑滑回安全基线——
是柔软的连续接管,不是切换的悬崖。

## 混合与安全数学

| 概念 | 含义 |
|------|------|
| 安全基线 | 已验证的 ACT 控制器:提供基础轨迹,**始终持有夹爪** |
| SmolVLA 平滑 | SmolVLA 候选动作先过 EMA 低通,再参与混合 |
| `authority` | 连续权重 ∈ [0, 0.9] —— 不是二进制的开关 |

**`authority` 怎么变:**

| 事件 | `authority` 更新 |
|------|------|
| SmolVLA 被接受(两策略一致) | 指数上升逼近上限 |
| 软拒绝 / 队列暂时为空 | × 0.85 |
| 硬拒绝(越界 / 分歧过大) | × 0.5 |

**为什么不可能失控** —— 四道独立限制:

1. SmolVLA 修正量幅度封顶(≤ 0.15 rad)。
2. 每步变化受 slew 限制(≤ 0.02 rad/步)。
3. 夹爪归 ACT 基线所有,SmolVLA 从不参与。
4. 最终指令再过一层全局低通。

这段数学被抽成纯函数模块 `smolvla_piper_towel/scripts/blend_core.py`,可离线单测,
与实时引擎解耦。

## 系统架构

<div align="center">

<img src="assets/towel/smolvla_flow.png" width="640" alt="SmolVLA 推理流程">

</div>

```
三路相机画面 ──┐
任务文本 ──────┼─► SmolVLM2-500M 骨干 ─► 动作专家头 ─► 50 步动作块
关节状态(28维) ─┘                                        │
                                                         ▼
                     安全 ACT 基线(持夹爪) ◄── 混合 ── EMA 平滑后的 SmolVLA 手臂目标
                                                         │
                                      30 Hz 最终指令 → 双 Piper(CAN)
```

训练遵循论文 *SmolVLA: Smol Models for Vision-Language-Action* 的两阶段配方:
**Stage A** 掩码预训练对齐视觉与动作;**Stage B** "one-more-step" 微调适配叠毛巾任务。
策略代码在 `smolvla_piper_towel/src/lerobot/policies/smolvla/`。

## 真机一条命令

```bash
cd smolvla_piper_towel/scripts
bash run_hybrid_towel_blend750.sh
```

终端会看到:

```
============================================================
双Piper毛巾折叠：SmolVLA 一键完整750步
...
===== 第一阶段：双臂归位 =====
...
===== 第二阶段：完整750步组合执行 =====
相机：left=18 middle=12 right=4
Loading SmolVLA candidate: .../smolvla_hq60_newonly_from50k_b8_5k_v2/checkpoints/005000/pretrained_model
Mode=EXECUTE max_actions=750 handoff_step=300 max_authority=0.90
step=0001 source=SAFE    joint_step=0.01818 tracking=0.15758 queue=0   authority=0.00
...
HANDOFF_REQUEST step=300: SmolVLA blend mode enabled
step=0317 source=SmolVLA joint_step=0.00756 tracking=0.01042 queue=16  authority=0.14
...
step=00508 source=SmolVLA joint_step=0.01568 tracking=0.05126 queue=30  authority=0.86
step=00509 source=SmolVLA joint_step=0.01418 tracking=0.04306 queue=29  authority=0.86
...
run_id=20260828_...
status=0
safe_guard_ticks=NNN
smolvla_blend_actions=MMM
```

| 阶段 | 步数 | 说明 |
|------|------|------|
| 阶段一 | — | 双臂安全归位到起始位 |
| 阶段二(保底) | 1–300 | 已验证的 ACT 基线主控(`source=SAFE`,`authority=0.00`) |
| 切入 | 300 | `HANDOFF_REQUEST step=300: SmolVLA blend mode enabled` |
| 阶段二(混合) | 301–750 | SmolVLA 塑形基线,`authority` 爬升并稳定在 0.7–0.9 |

完整跑完退出码为 **`status=0`**。每次运行的日志、分项计数与模型路径会写入实验目录下
按 `run_id` 命名的目录(`summary.txt`、`hybrid_full750.log`、`reset.log`)。

## 真机运行结果

| 项 | 值 |
|----|----|
| 任务 | 完整 750 步叠毛巾(抓取 → 折叠 → 释放) |
| 结果 | 真机完整跑通,**`status=0`** |
| 混合接管 | 第 300 步 `HANDOFF_REQUEST`,此后 `source=SmolVLA` |
| 塑形强度 | SmolVLA 主导阶段 `authority` 爬升到 ~0.7–0.9 |
| 控制频率 | 30 Hz 实时,每一步都受界 |

数字直接读自真实运行日志,不是编的——与 ACT 基线仓库对自己的 10/10 说法采用同一标准。

## 硬件与机器人接口

| 项 | 内容 |
|----|------|
| 机械臂 | 双 AgileX Piper——左臂 `can1`,右臂 `can0` |
| 相机 | 3 路 RGB(左 / 中 / 右),640×480——真机索引 `left=video18 middle=video12 right=video4` |
| 观测 | 3 张图像 + 28 维双臂状态(位置 / effort 交错) |
| 动作 | 14 维绝对关节位置目标(左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪) |

动作是**绝对关节位置目标**,不是速度 / 增量控制。

## 仓库布局

```text
piper-dual-arm-smolvla-towel-folding/
├── README.md               本文件(英文)
├── README_zh-CN.md         中文版
├── doc.md                  配套图文课程(中文,带截图)
├── assets/towel/           Hero 演示 + 课程图片
├── smolvla_piper_towel/    基于 LeRobot 的 SmolVLA 项目
│   ├── README.md           课程入口文档
│   ├── scripts/            一键运行脚本(blend750.sh、r750.sh、blend_core.py)
│   └── src/lerobot/        SmolVLA 版 LeRobot fork(policies/smolvla、async inference 等)
└── pyshim/                 运行时 authority 混合引擎(hy.py、env.sh、blend_core.py)
```

一键脚本依赖这个**精确的平级布局**:它会 `source /workspace/pyshim/env.sh` 并运行
`/workspace/pyshim/hy.py`。要复现,把仓库根放到工作区根,让 `smolvla_piper_towel/` 与
`pyshim/` 平级即可(例如 `/workspace/smolvla_piper_towel` + `/workspace/pyshim`)。

## 开始 / 复现

### 在同一台真机 / 平台卡片上

1. 挂载仓库,使 `/workspace/smolvla_piper_towel` 与 `/workspace/pyshim` 存在。
2. 把 ACT 与 SmolVLA 权重放到 `r750.sh` 里的路径(或改这两个模型变量)。
3. 执行一键脚本(见 [真机一条命令](#真机一条命令))。

### 换到别的机器

脚本硬编码了真机的主机路径。先改 `run_hybrid_towel_blend750.sh` / `r750.sh` 顶部的变量:

| 变量 | 真机值 |
|------|--------|
| `CONDA_SH` | `/opt/miniconda3_databall01/etc/profile.d/conda.sh` |
| `ACT_ROOT` | ACT LeRobot 部署路径(提供 `reset_piper_pose.py`) |
| `SMOL_ROOT` | 本仓库 `smolvla_piper_towel`(部署后的路径) |
| `SMOL_MODEL` / `ACT_MODEL` | 两个 checkpoint 目录 |

然后检查你机器上的 `/dev/video*` 相机索引。

## 配套图文课程

从任务设计、SmolVLA 原理、相机与机械臂接口,到真机运行、读日志、调 `authority`、排障——
完整的图文教程(中文,带截图)在 **[`doc.md`](doc.md)**。

## 相关项目

同台真机、同一任务的**纯 ACT 基线**(10/10 连续成功,不含 SmolVLA)是另一个独立仓库:

> **[🦾 Dual-Arm ACT Towel Folding](https://github.com/dk2472780158-ctrl/piper-dual-arm-act-towel-folding)**
>
> ACT 模仿学习:遥操作数据采集 → 训练 → 真机部署;单次连续拍摄评估 **10/10 consecutive trials**。

本仓库是它的 SmolVLA 演进:保留 ACT 基线及其已发布的 checkpoint,在它上面叠加连续 SmolVLA
authority 混合。

## 下载

数据与权重**从不入库**。请到 Hugging Face 获取:

- **ACT 打底权重**(`towel_fold_act_v4_040000`,v4 / 040000 = last):
  <https://huggingface.co/1goldexperience1/towel_fold_act_v4_040000>
- **SmolVLA 塑形策略**(`towel_fold_smolvla_shaping_005000`,HQ60 微调,5k 步):
  <https://huggingface.co/1goldexperience1/towel_fold_smolvla_shaping_005000>
- **数据集**(120 条真实双臂 demo,85,187 帧):
  <https://huggingface.co/datasets/1goldexperience1/towel_fold_dataset_aug_v1>

## 状态与边界

- [x] 真机 750 步完整跑通,`status=0`
- [x] 第 300 步起连续 authority 混合生效(非二进制切换)
- [x] 每一步都执行安全限制(幅度 + slew + 夹爪归属)
- [x] SmolVLA 塑形权重发布到 Hugging Face(`towel_fold_smolvla_shaping_005000`)
- [ ] 其他毛巾 / 物体位姿下的混合基准测试

**安全操作:** 运行前清空工作台、给双臂支撑、人站到急停旁。先确认三路相机画面正常再开始
(那是模型的"眼睛");脚本提示相机占用时先关闭画面面板。中途停下时机械臂保持姿态、不会失能下坠。

## 致谢与许可

- **SmolVLA** — *SmolVLA: Smol Models for Vision-Language-Action*(SmolVLM2 骨干 + 动作专家)。
- **LeRobot** / ACTPolicy — Apache 2.0,基于 Tony Z. Zhao 的 ALOHA 工作。
- **AgileX Piper SDK** — 按其自身许可。
- 本仓库以 Apache 2.0 发布([`LICENSE`](LICENSE))。
