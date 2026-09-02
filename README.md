<div align="center">

# 🦾 Dual-Arm SmolVLA Towel Folding

**A vision-language-action robot that folds a real towel — SmolVLA *shaping* on top of a proven ACT safety base, blended continuously so it never jumps and can't lose control.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Policy](https://img.shields.io/badge/policy-SmolVLA%20%E2%80%A2%20SmolVLM2--500M-orange.svg)
![Robot](https://img.shields.io/badge/robot-dual%20AgileX%20Piper-brightgreen.svg)
![Control](https://img.shields.io/badge/control-30%20Hz%20real--time-9cf.svg)

**English** · [**中文版**](README_zh-CN.md)

</div>

---

> ## 🎬 Real-robot demo
>
> One continuous take on the rig: the arms watch three camera views and fold the towel
> **by themselves** — grasp → fold → release — with SmolVLA increasingly shaping the motion
> as the run goes on.

<div align="center">

<img src="assets/towel/demo_hero.webp" width="320" alt="SmolVLA towel-folding real-robot demo">

</div>

---

## Highlights

- **A real VLA on real arms** — [SmolVLA](https://huggingface.co/HuggingFaceTB/SmolVLA-500M-Instruct)
  (`SmolVLM2-500M` vision-language backbone + action-expert head) turns three camera views and
  joint state directly into joint actions.
- **Two policies, one run** — a proven **ACT** controller is the safety base and always owns the
  grippers; **SmolVLA** continuously *shapes* the base trajectory.
- **Blended, never switched** — `authority ∈ [0, 0.9]` rises and falls continuously. There is no
  binary handoff, so no command jump at step 300.
- **Safe by construction** — SmolVLA corrections are bounded in magnitude (≤ 0.15 rad) and slew
  (≤ 0.02 rad/step); the gripper is never SmolVLA's to command.
- **One-command deployment** — `bash run_hybrid_towel_blend750.sh` resets both arms and runs the
  full 750-step folding skill on the real robot.

This repository is the **SmolVLA follow-up** to a pure-ACT baseline — see
[Related project](#related-project). The ACT checkpoint it relies on is the one released from that
project.

---

## Table of contents

- [How it works: two policies, one blend](#how-it-works-two-policies-one-blend)
- [The blend & the safety math](#the-blend--the-safety-math)
- [Architecture](#architecture)
- [One command on the real robot](#one-command-on-the-real-robot)
- [Results from the real run](#results-from-the-real-run)
- [Hardware & robot interface](#hardware--robot-interface)
- [Repository layout](#repository-layout)
- [Get started / reproduce](#get-started--reproduce)
- [Full illustrated course](#full-illustrated-course)
- [Related project](#related-project)
- [Downloads](#downloads)
- [Status & boundaries](#status--boundaries)
- [Acknowledgements & license](#acknowledgements--license)

---

## How it works: two policies, one blend

Folding a deformable towel needs both **stability** (a controller that is known to work) and
**flexibility** (a policy that reacts to what it *sees*). This project gets both by running the two
policies at the same time instead of picking one:

| | ACT safety base | SmolVLA shaper |
|---|---|---|
| Role | proven base trajectory; **always owns the gripper** | looks at the cameras and proposes a *correction* |
| Source of motion | conservative, verified | learned, vision-driven, more expressive |
| Authority over the arm | 1.0 → fades as SmolVLA is trusted | 0 → up to 0.9 as it stays coherent |

From step 300 onward the two outputs are fused every tick (30 Hz):

```text
target_arm = safe_base_arm + authority × (smol_ema_arm − safe_base_arm)
```

When SmolVLA agrees with the base it earns influence; the moment it disagrees too much or steps out
of bounds, its influence decays and the command glides back to the safe base — a soft, continuous
takeover, never a hand-off cliff.

## The blend & the safety math

| Concept | Meaning |
|---|---|
| Safe base | The proven ACT controller; provides the base trajectory and always owns the gripper |
| SmolVLA smooth | SmolVLA's candidate action, EMA-low-pass filtered before it is blended |
| `authority` | continuous weight ∈ [0, 0.9] — not a binary switch |

**How `authority` moves:**

| Event | `authority` update |
|---|---|
| SmolVLA accepted (both policies agree) | rises exponentially toward the cap |
| soft reject / queue temporarily empty | × 0.85 |
| hard reject (out of bounds / too much disagreement) | × 0.5 |

**Why it cannot lose control** — four independent limits:

1. SmolVLA's correction magnitude is capped (≤ 0.15 rad).
2. Its per-step change is slew-limited (≤ 0.02 rad/step).
3. The gripper is owned by the ACT base; SmolVLA never commands it.
4. The final command passes one more global low-pass.

The math is isolated in `smolvla_piper_towel/scripts/blend_core.py` — a pure, importable module
that is unit-testable offline, separate from the real-time engines.

## Architecture

<div align="center">

<img src="assets/towel/smolvla_flow.png" width="640" alt="SmolVLA inference flow">

</div>

```
three camera views ─┐
task text (fold) ───┼─► SmolVLM2-500M backbone ─► action-expert head ─► 50-step action chunk
joint state (28D) ──┘                                                  │
                                                                       ▼
                          safe ACT base (owns gripper) ◄── blend ── EMA-smoothed SmolVLA arm target
                                                                       │
                                                   final command @ 30 Hz → dual Piper (CAN)
```

Two-stage training follows the paper *SmolVLA: Smol Models for Vision-Language-Action*:
**Stage A** masked pre-training aligns vision with action; **Stage B** "one-more-step" fine-tuning
adapts the policy to the towel task. The policy code lives in
`smolvla_piper_towel/src/lerobot/policies/smolvla/`.

## One command on the real robot

```bash
cd smolvla_piper_towel/scripts
bash run_hybrid_towel_blend750.sh
```

What the terminal shows:

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

| Phase | Steps | What happens |
|---|---|---|
| Stage 1 | — | both arms safely reset to the start pose |
| Stage 2 (base) | 1–300 | proven ACT base controls (`source=SAFE`, `authority=0.00`) |
| Handoff | 300 | `HANDOFF_REQUEST step=300: SmolVLA blend mode enabled` |
| Stage 2 (blend) | 301–750 | SmolVLA shapes the base; `authority` climbs and settles around 0.7–0.9 |

A complete run exits with **`status=0`**. Per-run logs and per-step counts are written to a
`run_id` folder under the experiment dir (`summary.txt`, `hybrid_full750.log`, `reset.log`).

## Results from the real run

| Item | Value |
|---|---|
| Task | full 750-step towel folding (grasp → fold → release) |
| Result | complete run on the real rig, **`status=0`** |
| Blend take-over | `HANDOFF_REQUEST` at step 300, `source=SmolVLA` thereafter |
| Shaping strength | `authority` climbs to ~0.7–0.9 during SmolVLA-dominated segments |
| Control rate | 30 Hz real-time, bounded corrections at every step |

Numbers are read from real run logs, not guessed — the same standard the ACT baseline repo applies
to its own 10/10 claim.

## Hardware & robot interface

| Item | Value |
|---|---|
| Arms | dual AgileX Piper — left on `can1`, right on `can0` |
| Cameras | 3× RGB views (left / middle / right), 640×480 — rig indexes `left=video18 middle=video12 right=video4` |
| Observation | 3 images + 28-dim dual-arm state (position / effort interleaved) |
| Action | 14-dim absolute joint-position targets (left 6 joints + left gripper + right 6 joints + right gripper) |

Actions are **absolute joint-position targets**, not velocity / incremental commands.

## Repository layout

```text
piper-dual-arm-smolvla-towel-folding/
├── README.md               ← you are here
├── README_zh-CN.md         中文版
├── doc.md                  full illustrated course (中文, with screenshots)
├── assets/towel/           hero demo + course images
├── smolvla_piper_towel/    the LeRobot-based SmolVLA project
│   ├── README.md           course entry doc
│   ├── scripts/            one-command run scripts (blend750.sh, r750.sh, blend_core.py)
│   └── src/lerobot/        SmolVLA fork of LeRobot (policies/smolvla, async inference, …)
└── pyshim/                 runtime authority-blend engine (hy.py, env.sh, blend_core.py)
```

The run scripts assume this **exact sibling layout**: they `source /workspace/pyshim/env.sh` and
run `/workspace/pyshim/hy.py`. To reproduce, place the repo root at the workspace root so that
`smolvla_piper_towel/` and `pyshim/` sit side by side (e.g. `/workspace/smolvla_piper_towel` +
`/workspace/pyshim`).

## Get started / reproduce

### On the same rig / platform card

1. Mount the repo so `/workspace/smolvla_piper_towel` and `/workspace/pyshim` exist.
2. Place the ACT + SmolVLA checkpoints at the paths in `r750.sh` (or edit the two model variables).
3. Run the one-command script (see [above](#one-command-on-the-real-robot)).

### On different hardware

The scripts hard-code the rig's host paths. Adapt these variables at the top of
`run_hybrid_towel_blend750.sh` / `r750.sh` before running:

| Variable | Rig value |
|---|---|
| `CONDA_SH` | `/opt/miniconda3_databall01/etc/profile.d/conda.sh` |
| `ACT_ROOT` | path to the ACT LeRobot deployment (provides `reset_piper_pose.py`) |
| `SMOL_ROOT` | path to this repo's `smolvla_piper_towel` (deployed) |
| `SMOL_MODEL` / `ACT_MODEL` | the two checkpoint directories |

Then verify the camera `video*` indexes on your machine (`ls /dev/video*`).

## Full illustrated course

The complete step-by-step walkthrough — task design, SmolVLA theory, camera setup, running the real
robot, reading the logs, tuning `authority`, and troubleshooting — is in
**[`doc.md`](doc.md)** (中文, with screenshots).

## Related project

The **pure-ACT baseline** on the same rig and task — 10/10 consecutive trials, no SmolVLA — is a
separate repository:

> **[🦾 Dual-Arm ACT Towel Folding](https://github.com/dk2472780158-ctrl/piper-dual-arm-act-towel-folding)**
>
> ACT imitation learning: teleop data collection → training → real-robot deployment, evaluated as
> **10/10 consecutive trials** from one continuous take.

This repo is its SmolVLA evolution: it keeps the ACT base and its released checkpoint, and layers
continuous SmolVLA authority blending on top.

## Downloads

Data and weights are **never committed to git**. Get them from Hugging Face:

- **ACT base checkpoint** (`towel_fold_act_v4_040000`, v4 / 040000 = last) —
  <https://huggingface.co/1goldexperience1/towel_fold_act_v4_040000>
- **SmolVLA shaping policy** (`towel_fold_smolvla_shaping_005000`, HQ60 fine-tune, 5k steps) —
  <https://huggingface.co/1goldexperience1/towel_fold_smolvla_shaping_005000>
- **Dataset** (120 real dual-arm demos, 85,187 frames) —
  <https://huggingface.co/datasets/1goldexperience1/towel_fold_dataset_aug_v1>

## Status & boundaries

- [x] 750-step real-robot run reaches `status=0`
- [x] Continuous authority blend active from step 300 (no binary handoff)
- [x] Safety limits (magnitude + slew + gripper ownership) enforced at every step
- [x] SmolVLA shaping weights published to Hugging Face (`towel_fold_smolvla_shaping_005000`)
- [ ] Blend benchmarks on other towels / object poses

**Operate safely:** clear the worktable, support both arms, and stand by the e-stop before any run.
Confirm the three camera views are live first (they are the model's "eyes"); close the camera panel
if the script reports busy cameras. If the arms stop mid-run they hold pose (no de-energized drop).

## Acknowledgements & license

- **SmolVLA** — *SmolVLA: Smol Models for Vision-Language-Action* (SmolVLM2 backbone + action expert).
- **LeRobot** / ACTPolicy — Apache 2.0, based on Tony Z. Zhao's ALOHA work.
- **AgileX Piper SDK** — per its own license.
- This repository is released under Apache 2.0 ([`LICENSE`](LICENSE)).
