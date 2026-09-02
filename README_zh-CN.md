<div align="center">

# 双臂 SmolVLA 叠毛巾

**在真实双臂 AgileX Piper 上、用视觉语言动作模型(SmolVLA)在已验证的 ACT 安全基线上做连续 authority 塑形,完成叠毛巾。**

SmolVLA(SmolVLM2-500M)· LeRobot fork · 30 Hz 实时控制 · 连续 authority 混合 · 安全优先

**English**：[README.md](README.md)

</div>

---

## 这个项目做了什么

**一条命令**驱动两台真实 AgileX Piper 机械臂,完整跑完一次叠毛巾(抓取 → 折叠 → 归位):

1. **安全归位** — 双臂自动回到起始位(无需手动输入 MOVE)。
2. **750 步执行** — 每一步都由模型预测 + 安全机制兜底。
3. **连续 SmolVLA authority 混合** — 从第 300 步起,SmolVLA 在 ACT 基线上"塑形",**不是二选一切换**,切换瞬间不会有指令跳变。

- **VLA 真机落地** — SmolVLA(SmolVLM2-500M 视觉语言骨干 + 动作专家头)把三路相机画面 + 关节状态直接映射成关节动作。
- **双臂协同 + 柔性物体** — 两台 6 自由度机械臂 + 夹爪叠一块可变形毛巾,单个 14 维动作向量驱动。
- **稳,因为"混合"而不是"切换"** — `authority ∈ [0, 0.9]` 连续升降,天然平滑。
- **夹爪恒由安全基线持有** — 修正量有幅度与每步 slew 上限,SmolVLA 只能塑形、不可能把机械臂推越界。

> 本仓库是纯 ACT 基线的 **SmolVLA 演进版** — 见 [相关项目](#相关项目)。这里的 ACT 打底 checkpoint 就是那个项目发布的同一个 checkpoint。

---

## 一键运行(750 步)

按 [仓库布局](#仓库布局) 放到正确位置后,在真机端执行:

```bash
cd smolvla_piper_towel/scripts
bash run_hybrid_towel_blend750.sh
```

脚本打印横幅(`双Piper毛巾折叠：SmolVLA 一键完整750步`),双臂归位后开始执行:

| 阶段 | 步数 | 说明 |
|------|------|------|
| 阶段一 | — | 双臂安全归位到起始位 |
| 阶段二(保底) | 1–300 | 已验证的 ACT 安全基线主控(`source=SAFE`,`authority=0.00`) |
| 切入 | 300 | `HANDOFF_REQUEST step=300: SmolVLA blend mode enabled` |
| 阶段二(混合) | 301–750 | SmolVLA 塑形基线,`authority` 爬升并稳定在 0.7–0.9 |

完整跑完退出码为 **`status=0`**。每次运行的日志、分项计数与模型路径会写入实验目录下按 `run_id` 命名的目录(`summary.txt`、`hybrid_full750.log`、`reset.log`)。

---

## 混合与安全数学

```text
目标动作 = 安全基线动作 + authority × (SmolVLA 平滑 − 安全基线动作)
```

| 概念 | 含义 |
|------|------|
| 安全基线 | 已验证的 ACT 控制器:提供基础轨迹,并**始终持有夹爪** |
| SmolVLA 平滑 | SmolVLA 候选动作先过 EMA 低通,再叠加到基线 |
| authority | 连续权重 ∈ [0, 0.9],不是二选一切换 |

**authority 怎么变** — 两模型一致时指数上升逼近上限;软拒绝 / 队列暂时为空时 ×0.85;硬拒绝(越界 / 分歧过大)时 ×0.5。

**为什么不会失控:**

1. SmolVLA 修正量有幅度上限(≤ 0.15 rad)。
2. 每步变化有 slew 上限(≤ 0.02 rad/步)。
3. 夹爪恒为安全基线所有,SmolVLA 不参与。
4. 最终命令再过一层全局低通。

所以 SmolVLA 永远只是"小修"——**能塑形,不能主导失控**。

这段数学的纯函数实现见 `smolvla_piper_towel/scripts/blend_core.py`(可离线单测)。

---

## 模型与权重

| 角色 | 模型 | 位置 |
|------|------|------|
| 安全基线(持夹爪) | **ACT** `towel_fold_act_v4_scratch60k` / checkpoint `040000` | 已发布 Hugging Face(见 [下载](#下载)) |
| 塑形策略(默认) | **SmolVLA** `smolvla_hq60_newonly_from50k_b8_5k_v2` / checkpoint `005000`(ACT 初始化,batch 8,5k 步) | 真机实验目录;路径在 `run_hybrid_towel_blend750.sh` / `r750.sh` |

两模型一致(分歧在容差内)时 authority 才会上升;`--max-authority 0.9` 上限加 slew 限制让混合天然保守。

这里的 SmolVLA 采用论文 *SmolVLA: Smol Models for Vision-Language-Action* 的两阶段配方:Stage A 掩码预训练对齐视觉-动作,Stage B "one-more-step" 微调适配叠毛巾任务。模型结构在 `smolvla_piper_towel/src/lerobot/policies/smolvla/`。

---

## 硬件与机器人接口

| 项 | 内容 |
|----|------|
| 机械臂 | 双 AgileX Piper——左臂 `can1`,右臂 `can0` |
| 相机 | 3 路 RGB(左 / 中 / 右),640×480——真机索引 `left=video18 middle=video12 right=video4` |
| 观测 | 3 张图像 + 28 维双臂状态(位置 / effort 交错) |
| 动作 | 14 维绝对关节位置目标(左 6 关节 + 左夹爪 + 右 6 关节 + 右夹爪) |

动作是**绝对关节位置目标**,不是速度 / 增量控制。

---

## 仓库布局

```text
piper-dual-arm-smolvla-towel-folding/
├── README.md              本文件
├── README_zh-CN.md        中文版
├── smolvla_piper_towel/   基于 LeRobot 的 SmolVLA 项目(代码、策略、脚本)
│   ├── README.md          完整课程 / 卡片文档
│   ├── scripts/           一键运行脚本
│   └── src/lerobot/       SmolVLA 版 LeRobot fork(policies/smolvla、async inference 等)
└── pyshim/                运行时 authority 混合核心(hy.py、blend_core.py、env.sh)
```

一键脚本依赖这个**精确的平级布局**:它会 `source /workspace/pyshim/env.sh` 并运行
`/workspace/pyshim/hy.py`。要复现,把仓库根放到工作区根,让 `smolvla_piper_towel/` 与
`pyshim/` 平级即可(例如 `/workspace/smolvla_piper_towel` + `/workspace/pyshim`)。

---

## 复现步骤

### 在同一台真机 / 平台卡片上

1. 挂载仓库,使 `/workspace/smolvla_piper_towel` 与 `/workspace/pyshim` 存在。
2. 把 ACT 与 SmolVLA 权重放到 `r750.sh` 里的路径(或改这两个模型变量)。
3. 在带运行环境的终端里执行:

```bash
cd /workspace/smolvla_piper_towel/scripts
bash run_hybrid_towel_blend750.sh
```

### 换到别的机器

脚本硬编码了真机的主机路径。先改 `run_hybrid_towel_blend750.sh` / `r750.sh` 顶部的变量:

| 变量 | 真机值 |
|------|--------|
| `CONDA_SH` | `/opt/miniconda3_databall01/etc/profile.d/conda.sh` |
| `ACT_ROOT` | ACT LeRobot 部署路径(提供 `reset_piper_pose.py`) |
| `SMOL_ROOT` | 本仓库 `smolvla_piper_towel`(部署后的路径) |
| `SMOL_MODEL` / `ACT_MODEL` | 两个 checkpoint 目录 |

然后检查你机器上的 `/dev/video*` 相机索引。

---

## 相关项目

纯 ACT 基线(同一台真机、同一个任务,10/10 连续成功,不含 SmolVLA)是另一个独立仓库:

**[Dual-Arm ACT Towel Folding](https://github.com/dk2472780158-ctrl/piper-dual-arm-act-towel-folding)** —
ACT 模仿学习:遥操作数据采集 → 训练 → 真机部署。

本仓库是它的 SmolVLA 演进:保留 ACT 基线及其已发布的 checkpoint,在它上面叠加连续 SmolVLA
authority 混合。

---

## 下载(数据与权重 —— 不入 git)

- **ACT 打底权重**(`towel_fold_act_v4_040000`,v4 / 040000 = last):
  <https://huggingface.co/1goldexperience1/towel_fold_act_v4_040000>
- **数据集**(120 条真实双臂 demo,85,187 帧——用于训练 ACT 基线):
  <https://huggingface.co/datasets/1goldexperience1/towel_fold_dataset_aug_v1>
- SmolVLA 塑形 checkpoint 目前仍在真机本地,其实验目录在 `r750.sh` 中引用。

---

## 安全与边界

1. 运行前:清空工作台、给双臂支撑、人站到急停旁。
2. 先确认三路相机画面正常再开始(那是模型的"眼睛")。
3. 若脚本提示相机占用,关闭画面面板释放 `/dev/video*`。
4. 中途停下时机械臂保持姿态、不会失能下坠——确认急停可用后再重新运行。
5. 中途安全停机(日志里 `joint_step` / `tracking` 越界,或 `authority` 被压到接近 0)说明安全机制在工作,不是故障。

---

## 致谢与许可

- **SmolVLA** — *SmolVLA: Smol Models for Vision-Language-Action*(SmolVLM2 骨干 + 动作专家)。
- **LeRobot** / ACTPolicy — Apache 2.0,基于 Tony Z. Zhao 的 ALOHA 工作。
- **AgileX Piper SDK** — 按其自身许可。
- 本仓库以 Apache 2.0 发布(`LICENSE`)。
