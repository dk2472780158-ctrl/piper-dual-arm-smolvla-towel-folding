# 智能抓取与叠毛巾(SmolVLA 视觉语言动作模型)

## 卡片简介

本卡片基于 **SmolVLA**,一个开源轻量的**视觉-语言-动作(VLA)模型**,让机械臂通过"看 + 听 + 动"完成抓取与叠毛巾任务。卡内预置训练好的 SmolVLA 权重与 LeRobot 运行环境,带你从模型原理走到实际运行。

## 核心概念:VLA 模型在做什么

传统机械臂编程需要人写好每步动作。VLA 模型改变了这一点:

1. **视觉理解**:读入相机图像,看懂场景里有什么(毛巾、边缘、目标位置)
2. **语言指令**:读懂自然语言任务(如"把毛巾对折")或任务编码
3. **动作生成**:直接输出机械臂关节动作序列,不再依赖手工编程

一句话:**VLA = 视觉语言模型(VLM)+ 动作专家(Action Expert)**。

## 为什么用 SmolVLA

- **轻量开源**:500M 参数,约 6GB 显存即可推理,单张消费级 GPU 可跑
- **训练成本低**:相比几十亿参数的大 VLA,数据量和算力需求小很多
- **可微调**:在本项目数据上做两阶段微调即可适配叠毛巾任务

## 本卡片两大模块

| 模块 | 作用 |
|---|---|
| 视觉语言骨干 | 基于 SmolVLM2-500M,处理图像 + 指令 |
| 动作专家头 | 从骨干解码出连续关节动作 |

**任务目标(参考)**:机械臂在毛巾台上完成抓取、折叠、归位等连续动作,全程由模型预测动作、安全栈兜底。

---

# SmolVLA 模型原理

## 1. 整体架构

模型由两大部分组成:

1. **视觉语言骨干(backbone)**:SmolVLM2 负责把图像和文本编码成特征
2. **动作专家头(action expert)**:在骨干的隐状态上解码出动作

**关键机制 —— `ACTION_INPUT` 特殊 token**:

- 输入文本里插入一个占位 token `ACTION_INPUT`
- 骨干在处理到这个 token 的位置时,输出一组隐状态
- 动作专家头把这组隐状态映射成一段**动作序列**

## 2. 动作分块(Action Chunk)

模型不是逐帧预测动作,而是**一次预测未来 50 步的动作块**:

- 优点:减少逐帧预测的累积误差,动作更平滑
- 执行:推理时生成一块,机械臂执行;再生成下一块,循环推进

## 3. 两阶段训练流程

| 阶段 | 训练方式 | 对应权重 |
|---|---|---|
| **Stage A(掩码预训练)** | 遮盖部分动作块,让模型从"视觉+文本"猜被遮的动作,学习视觉-动作对齐 | `smolvla_base` → `stageA` → `stageA_480` → `stageA_add` |
| **Stage B(动作微调)** | "one more step"策略:基于上一段动作上下文预测下一步,适配具体任务数据 | `stageB_30000_v1` → `stageB_v1_unfreeze_local` → `stageB_v2_mixed_unfreeze` |

**归一化方式**:动作归一化分为 `local` 和 `global` 两种,chunk 采用绝对位置编码。

## 4. 本卡片使用的模型权重

平台已在 `/workspace/models/` 预置全部训练阶段权重:

```
smolvla_base
smolvla_stageA / stageA_480 / stageA_add
smolvla_stageB_30000_v1 / stageB_v1_unfreeze_local / stageB_v2_mixed_unfreeze
SmolVLM2-500M-Video-Instruct(视觉骨干)
```

**推荐使用**:`smolvla_stageB_v2_mixed_unfreeze`(最终微调版本,任务适配最完整)。

## 5. 进阶:混合塑形控制器

实际部署中,为兼顾稳定性与灵活性,本项目使用 **ACT 保底 + SmolVLA 塑形** 的混合策略:

```
目标动作 = ACT动作 + authority × (SmolVLA平滑修正 − ACT动作)
```

- `authority ∈ [0, 0.9]`:两模型判断一致时上升,分歧大时自动降低
- 结果:连续平滑的动作修正,消除两模型二进制切换带来的抖动

---

# 环境准备与启动

## 1. 卡片内置环境

打开「终端」后,终端会自动进入 **`lerobot_v30`** conda 环境(看到 `(lerobot_v30)` 前缀即正常)。

**之后每打开一个新终端,都建议先执行:**

```
conda activate lerobot_v30
export PYTHONNOUSERSITE=1
cd /workspace/smolvla_piper_towel
```

## 2. 已导入的源码

代码位于 `/workspace/smolvla_piper_towel/`,是完整的 LeRobot fork:

| 目录/文件 | 内容 |
|---|---|
| `src/lerobot/` | LeRobot 核心库(SmolVLA 训练/推理) |
| `scripts/` | 叠毛巾控制器与运行脚本 |
| `README.md` | 项目说明 |

## 3. 预置模型权重

模型权重已通过符号链接预置:

```
/workspace/models/  →  /home/devuser/VLA/downloaded_models/
```

包含全部 Stage A / Stage B 的 SmolVLA 权重与 SmolVLM2 骨干,**无需自行下载**。

## 4. 启动步骤

1. **安装源码包**(可编辑模式,改代码即时生效):

```
cd /workspace/smolvla_piper_towel
pip install -e . --no-build-isolation
```

2. **验证模型文件存在**:

```
ls /workspace/models/
```

3. **运行推理/演示脚本**(具体脚本与命令见后续章节)。

## 注意事项

- 不要 `pip install --user` 到系统 Python,一切装进 `lerobot_v30`
- 模型已预置,不要重复下载,避免访问外网
- 长命令建议用反斜杠 `\` 续行,避免在终端内折行出错
