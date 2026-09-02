<div align="center">

# Dual-Arm SmolVLA Towel Folding

**Vision-language-action towel folding on real dual-AgileX-Piper arms — SmolVLA shapes a proven ACT safety base through a continuous authority blend.**

SmolVLA (SmolVLM2-500M) · LeRobot fork · 30 Hz real-time control · continuous authority blending · safety-first

**中文版**：[README_zh-CN.md](README_zh-CN.md)

</div>

---

## What this project shows

A **single command** drives two physical AgileX Piper arms through a complete, real-world
towel-folding run (grasp → fold → release):

1. **Safe reset** — both arms return to the proven start pose (no manual `MOVE` needed).
2. **750-step execution** — every step is predicted by models and guarded by a safety layer.
3. **Continuous SmolVLA authority blend** — from step 300 onward, SmolVLA *shapes* the
   trajectory on top of the ACT base. There is **no binary switch**, so there is no command
   jump at the handoff.

- **VLA on real hardware** — SmolVLA (`SmolVLM2-500M` vision-language backbone + action-expert
  head) turns three camera views + joint state directly into joint actions.
- **Bi-manual, soft-object** — folding a deformable towel with two 6-DoF arms + grippers,
  driven from a single 14-dim action vector.
- **Stable because it is blended, not switched** — `authority ∈ [0, 0.9]` rises and falls
  continuously.
- **The safety layer owns the gripper** — corrections are bounded in both magnitude and
  slew, so SmolVLA can shape but never push the arms out of range.

> This repository is the **SmolVLA follow-up** to the pure-ACT baseline — see
> [Related project](#related-project). The ACT base checkpoint used here is the exact
> checkpoint released from that project.

---

## One-command run (750 steps)

From the rig, inside the repo layout described in [Repository layout](#repository-layout):

```bash
cd smolvla_piper_towel/scripts
bash run_hybrid_towel_blend750.sh
```

The script prints a banner (`双Piper毛巾折叠：SmolVLA 一键完整750步`), resets both arms, then runs:

| Phase | Steps | What happens |
|-------|-------|--------------|
| Stage 1 | — | Both arms safely reset to the start pose |
| Stage 2 (base) | 1 – 300 | Proven ACT safety base controls (`source=SAFE`, `authority=0.00`) |
| Handoff | 300 | `HANDOFF_REQUEST step=300: SmolVLA blend mode enabled` |
| Stage 2 (blend) | 301 – 750 | SmolVLA shapes the base; `authority` climbs and settles around 0.7 – 0.9 |

A successful full run exits with **`status=0`**. Per-run logs, per-step counts and the model
paths land in a `run_id` directory under the experiment folder (`summary.txt`,
`hybrid_full750.log`, `reset.log`).

---

## The blend & the safety math

```text
target_arm = safe_base_arm + authority × (smol_ema_arm − safe_base_arm)
```

| Concept | Meaning |
|---------|---------|
| Safe base | The proven ACT controller; provides the base trajectory and **always owns the gripper** |
| SmolVLA smooth | SmolVLA's candidate action, EMA-low-pass filtered before blending |
| authority | Continuous weight ∈ [0, 0.9] — not a binary switch |

**How authority moves** — when the two policies agree it rises (approaching the cap); on a
soft reject / temporarily empty queue it decays (×0.85); on a hard reject (out-of-bounds /
too much disagreement) it decays hard (×0.5).

**Why it cannot lose control:**

1. SmolVLA's correction magnitude is capped (≤ 0.15 rad).
2. Its per-step slew is capped (≤ 0.02 rad/step).
3. The gripper is always owned by the ACT base.
4. The final command passes one more global low-pass.

So SmolVLA can *shape* the trajectory but can never drive it out of range.

See `smolvla_piper_towel/scripts/blend_core.py` — a pure, importable, offline-testable
implementation of this math.

---

## Models & checkpoints

| Role | Model | Where |
|------|-------|-------|
| Safety base (owns gripper) | **ACT** `towel_fold_act_v4_scratch60k` / checkpoint `040000` | Published on Hugging Face (see [Downloads](#downloads)) |
| Shaping policy (default) | **SmolVLA** `smolvla_hq60_newonly_from50k_b8_5k_v2` / checkpoint `005000` (ACT-init, batch 8, 5k steps) | Rig experiment dir; path set in `run_hybrid_towel_blend750.sh` / `r750.sh` |

The two policies agree within a tolerance before authority is allowed to rise; a `--max-authority
0.9` cap and the slew limits above keep the blend conservative by construction.

SmolVLA here is the two-stage recipe from the paper *SmolVLA: Smol Models for
Vision-Language-Action* — Stage A masked pre-training for vision–action alignment, then Stage B
“one-more-step” fine-tuning for the towel task. Its architecture lives under
`smolvla_piper_towel/src/lerobot/policies/smolvla/`.

---

## Hardware & robot interface

| Item | Value |
|------|-------|
| Arms | Dual AgileX Piper — left on `can1`, right on `can0` |
| Cameras | 3× RGB views (left / middle / right), 640×480 — rig indexes `left=video18 middle=video12 right=video4` |
| Observation | 3 images + 28-dim dual-arm state (position / effort interleaved) |
| Action | 14-dim absolute joint-position targets (left 6 joints + left gripper + right 6 joints + right gripper) |

Actions are **absolute joint-position targets**, not velocity / incremental commands.

---

## Repository layout

```text
piper-dual-arm-smolvla-towel-folding/
├── README.md               this file
├── README_zh-CN.md        中文版
├── smolvla_piper_towel/    the LeRobot-based SmolVLA project (code, policies, scripts)
│   ├── README.md           full course / card doc (中文)
│   ├── scripts/            one-click run scripts
│   └── src/lerobot/        SmolVLA fork of LeRobot (policies/smolvla, async inference, …)
└── pyshim/                 runtime authority-blend core (hy.py, blend_core.py, env.sh)
```

The one-click scripts assume this **exact sibling layout**: they `source /workspace/pyshim/env.sh`
and run `/workspace/pyshim/hy.py`. To reproduce, place the repo root at the workspace root so that
`smolvla_piper_towel/` and `pyshim/` sit side by side (e.g. `/workspace/smolvla_piper_towel` +
`/workspace/pyshim`).

---

## Reproduction

### On the same rig / platform card

1. Mount the repo so `/workspace/smolvla_piper_towel` and `/workspace/pyshim` exist.
2. Pre-place the ACT + SmolVLA checkpoints at the paths in `r750.sh` (or edit the two model
   variables).
3. From a terminal with the runtime conda env active:

```bash
cd /workspace/smolvla_piper_towel/scripts
bash run_hybrid_towel_blend750.sh
```

### On different hardware

The scripts hard-code the rig's host paths — adapt these variables at the top of
`run_hybrid_towel_blend750.sh` / `r750.sh` before running:

| Variable | Rig value |
|----------|-----------|
| `CONDA_SH` | `/opt/miniconda3_databall01/etc/profile.d/conda.sh` |
| `ACT_ROOT` | path to the ACT LeRobot deployment (provides `reset_piper_pose.py`) |
| `SMOL_ROOT` | path to this repo's `smolvla_piper_towel` (deployed) |
| `SMOL_MODEL` / `ACT_MODEL` | checkpoint directories |

Then check the camera `video*` indexes and re-verify `/dev/video*` on your machine.

---

## Related project

The pure-ACT baseline (same rig, same task, 10/10 consecutive trials, no SmolVLA) is a separate
repository:

**[Dual-Arm ACT Towel Folding](https://github.com/dk2472780158-ctrl/piper-dual-arm-act-towel-folding)** —
ACT imitation learning: teleop data collection → training → real-robot deployment.

This repo is its SmolVLA evolution: it keeps the ACT base and its released checkpoint, and adds
continuous SmolVLA authority blending on top.

---

## Downloads (data & weights — never in git)

- **ACT base checkpoint** (`towel_fold_act_v4_040000`, v4 / 040000 = last):
  <https://huggingface.co/1goldexperience1/towel_fold_act_v4_040000>
- **Dataset** (120 real dual-arm demos, 85,187 frames — used to train the ACT base):
  <https://huggingface.co/datasets/1goldexperience1/towel_fold_dataset_aug_v1>
- The SmolVLA shaping checkpoint is rig-local for now; its experiment dir is referenced in
  `r750.sh`.

---

## Safety & boundaries

1. Before any run: clear the workspace, support both arms, stand by the e-stop.
2. Confirm the three camera views are live before starting (they are the model's “eyes”).
3. Close the camera-view panel if the script reports cameras busy — it frees `/dev/video*`.
4. If the arm stops mid-run it holds its pose (no de-energized drop) — re-run only after
   confirming the e-stop is within reach.
5. A mid-run safety stop (log shows out-of-range `joint_step` / `tracking`, or `authority`
   collapsed toward 0) means the safety layer did its job, not a fault.

---

## Acknowledgements & license

- **SmolVLA** — *SmolVLA: Smol Models for Vision-Language-Action* (SmolVLM2 backbone + action expert).
- **LeRobot** / ACTPolicy — Apache 2.0, based on Tony Z. Zhao's ALOHA work.
- **AgileX Piper SDK** — per its own license.
- This repository is released under Apache 2.0 (`LICENSE`).
