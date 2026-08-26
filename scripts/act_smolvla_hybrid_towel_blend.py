#!/usr/bin/env python3
"""Dual-Piper towel folding: trusted ACT base + continuously blended SmolVLA.

This file is independent of the existing ACT deployment.  It loads the proven
ACT checkpoint as the trusted base controller and lets SmolVLA shape the arm
trajectory through a *continuous authority blend*:

    target_arm = act_arm + authority * (smol_ema_arm - act_arm)

authority in [0, max_authority] rises while SmolVLA stays coherent with ACT and
decays whenever it is rejected, out of bounds, or the RTC queue is empty.
There is no ACT<->SMOL binary switch, so no per-tick command jump at handoff.

Gripper is always owned by the proven ACT controller.  The final blended
command is absolutely limited and the SmolVLA correction is bounded in both
magnitude and per-step slew, so it can never push the arm out of range or
cause a discontinuity.

The blend math lives in ``blend_core`` (pure, importable, unit-tested offline);
this module only wires it to the real ACT + SmolVLA inference engines.

Safety model (kept strict, no relaxation of the proven ACT semantics):
  - pure ACT ticks  -> trusted ACT limits (joint step 1.0 / tracking 3.0), as before
  - blended ticks   -> validate_absolute(target) + ACT limits + bounded/slewed correction
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.robots.piper_dual.config_piper_dual import PIPERDualConfig
from lerobot.robots.piper_dual.piper_dual import PIPERDual
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

from blend_core import (
    ACTION_NAMES,
    ARM,
    GRIPPER,
    LOWER,
    UPPER,
    apply_global_smoothing,
    blend_step,
    max_parts,
    positions,
    validate_absolute,
)

SOFT_LIMIT_PROJECTION_MARGIN = 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--act-model", type=Path, required=True)
    p.add_argument("--smol-model", type=Path, required=True)
    p.add_argument("--left-index", type=int, required=True)
    p.add_argument("--middle-index", type=int, required=True)
    p.add_argument("--right-index", type=int, required=True)
    p.add_argument("--left-can", default="can1")
    p.add_argument("--right-can", default="can0")
    p.add_argument("--task", default="Fold the towel with both Piper arms.")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-actions", type=int, default=750)
    p.add_argument("--handoff-step", type=int, default=300)
    p.add_argument("--handoff-file", type=Path, default=Path("/tmp/piper_hybrid_handoff"))
    p.add_argument("--guidance-weight", type=float, default=2.5)
    p.add_argument("--execution-horizon", type=int, default=10)
    p.add_argument("--queue-threshold", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)

    # SmolVLA shaping / trust gates.
    p.add_argument(
        "--smol-lowpass-alpha",
        type=float,
        default=0.25,
        help="EMA weight of the raw SmolVLA action toward the current command.",
    )
    p.add_argument("--smol-raw-step-limit", type=float, default=0.60)
    p.add_argument("--smol-raw-policy-disagreement", type=float, default=0.50)
    p.add_argument(
        "--policy-disagreement",
        type=float,
        default=0.14,
        help="Soft trust gate: SmolVLA must stay this close to ACT to grow authority.",
    )
    p.add_argument(
        "--joint-step-limit",
        type=float,
        default=0.10,
        help="Soft trust gate: filtered SmolVLA per-step joint increment bound.",
    )

    # Authority dynamics.
    p.add_argument(
        "--max-authority",
        type=float,
        default=0.9,
        help="Upper bound of the continuous SmolVLA authority weight.",
    )
    p.add_argument(
        "--authority-up-rate",
        type=float,
        default=0.15,
        help="Exponential approach rate of authority toward max per accepted step.",
    )
    p.add_argument(
        "--authority-decay",
        type=float,
        default=0.85,
        help="Multiplicative authority decay per soft-reject / empty-queue step.",
    )
    p.add_argument(
        "--authority-hard-decay",
        type=float,
        default=0.5,
        help="Multiplicative authority decay per hard-reject (out-of-bounds) step.",
    )

    # SmolVLA correction (the actual thing added to ACT).
    p.add_argument(
        "--smol-correction-limit",
        type=float,
        default=0.15,
        help="Per-joint maximum |authority*(smol_ema - act)| correction (rad).",
    )
    p.add_argument(
        "--correction-alpha",
        type=float,
        default=0.5,
        help="EMA weight of the correction toward its desired value per step.",
    )
    p.add_argument(
        "--correction-step-limit",
        type=float,
        default=0.02,
        help="Per-joint maximum correction change per control step (rad).",
    )

    # Final smoothing / safety.
    p.add_argument(
        "--global-lowpass-alpha",
        type=float,
        default=0.85,
        help="Final command EMA weight applied to both ACT and the blend.",
    )
    p.add_argument("--act-joint-step-limit", type=float, default=1.0)
    p.add_argument("--act-tracking-limit", type=float, default=3.0)
    p.add_argument("--start-joint-tolerance", type=float, default=0.12)
    p.add_argument("--start-gripper-tolerance", type=float, default=0.012)
    p.add_argument("--max-consecutive-rejects", type=int, default=500)
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument(
        "--resume-current",
        action="store_true",
        help="Continue from a held in-trajectory pose instead of requiring ACT_START.",
    )
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.fps <= 0 or args.max_actions < 1:
        p.error("fps and max-actions must be positive")
    if not 0 <= args.handoff_step < args.max_actions:
        p.error("handoff-step must be in [0, max-actions)")
    if not 1 <= args.execution_horizon < 50:
        p.error("execution-horizon must be in [1, 49]")
    if not 1 <= args.queue_threshold < 50:
        p.error("queue-threshold must be in [1, 49]")
    if not 0.0 < args.smol_lowpass_alpha <= 1.0:
        p.error("smol-lowpass-alpha must be in (0, 1]")
    if not 0.0 < args.global_lowpass_alpha <= 1.0:
        p.error("global-lowpass-alpha must be in (0, 1]")
    if not 0.0 < args.max_authority <= 1.0:
        p.error("max-authority must be in (0, 1]")
    if not 0.0 < args.authority_up_rate <= 1.0:
        p.error("authority-up-rate must be in (0, 1]")
    if not 0.0 < args.authority_decay <= 1.0:
        p.error("authority-decay must be in (0, 1]")
    if not 0.0 < args.authority_hard_decay <= 1.0:
        p.error("authority-hard-decay must be in (0, 1]")
    if args.smol_correction_limit <= 0 or args.correction_step_limit <= 0:
        p.error("smol-correction-limit and correction-step-limit must be positive")
    if not 0.0 < args.correction_alpha <= 1.0:
        p.error("correction-alpha must be in (0, 1]")
    return args


def action_dict(action: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(ACTION_NAMES, action, strict=True)}


def as_action(value: torch.Tensor, *, project_soft_limits: bool) -> np.ndarray:
    result = value.detach().float().cpu().numpy().reshape(-1)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid action: shape={result.shape}, finite={np.isfinite(result).all()}")
    result = result.astype(np.float32, copy=False)
    # Piper's gripper driver uses abs(); make the intended physical boundary explicit.
    result[GRIPPER] = np.clip(result[GRIPPER], 0.0, 0.08)
    if project_soft_limits:
        violation = np.maximum(LOWER - result, result - UPPER)
        joint_violation = float(np.max(violation[ARM]))
        if 0.0 < joint_violation <= SOFT_LIMIT_PROJECTION_MARGIN:
            result[ARM] = np.clip(result[ARM], LOWER[ARM], UPPER[ARM])
            print(
                f"soft_limit_projection max_correction={joint_violation:.6f} rad",
                flush=True,
            )
    return result


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    device = torch.device("cuda")
    period = 1.0 / args.fps
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
        print("Stop requested; current measured pose will be held.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if args.handoff_file.exists():
        args.handoff_file.unlink()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading ACT reference: {args.act_model}", flush=True)
    act = ACTPolicy.from_pretrained(str(args.act_model))
    act.to(device).eval()
    act_pre, act_post = make_pre_post_processors(
        act.config, str(args.act_model),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"Loading SmolVLA candidate: {args.smol_model}", flush=True)
    smol = SmolVLAPolicy.from_pretrained(str(args.smol_model))
    smol.to(device).eval()
    original_predict = smol.predict_action_chunk

    def deterministic_predict(batch, noise=None, **kwargs):
        if noise is None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        return original_predict(batch, noise=noise, **kwargs)

    smol.predict_action_chunk = deterministic_predict
    rtc_cfg = RTCConfig(
        enabled=True,
        mode="guided",
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=args.guidance_weight,
        execution_horizon=args.execution_horizon,
        debug=False,
    )
    smol.config.rtc_config = rtc_cfg
    smol.init_rtc_processor()
    smol_pre, smol_post = make_pre_post_processors(
        smol.config, str(args.smol_model),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    cameras = {
        "left": OpenCVCameraConfig(index_or_path=args.left_index, width=640, height=480, fps=30),
        "middle": OpenCVCameraConfig(index_or_path=args.middle_index, width=640, height=480, fps=30),
        "right": OpenCVCameraConfig(index_or_path=args.right_index, width=640, height=480, fps=30),
    }
    robot = PIPERDual(PIPERDualConfig(
        left_port=args.left_can,
        right_port=args.right_can,
        cameras=cameras,
        read_only=not args.execute,
    ))
    wrapper = ThreadSafeRobot(robot)
    act_features = {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.observation_features, "observation", use_video=False),
    }
    ordered_actions = list(robot.action_features)
    act_engine = SyncInferenceEngine(
        policy=act,
        preprocessor=act_pre,
        postprocessor=act_post,
        dataset_features=act_features,
        ordered_action_keys=ordered_actions,
        task=args.task,
        device=str(device),
        robot_type=robot.robot_type,
    )
    smol_engine = RTCInferenceEngine(
        policy=smol,
        preprocessor=smol_pre,
        postprocessor=smol_post,
        robot_wrapper=wrapper,
        rtc_config=rtc_cfg,
        hw_features=act_features,
        task=args.task,
        fps=args.fps,
        device=str(device),
        use_torch_compile=False,
        rtc_queue_threshold=args.queue_threshold,
    )

    connected = False
    smol_started = False
    smol_active = False
    handoff_cancelled = False
    previous: np.ndarray | None = None
    correction = np.zeros(14, dtype=np.float32)
    authority = 0.0
    accepted = rejected = fallback = 0
    consecutive_rejects = 0

    print(
        f"Mode={'EXECUTE' if args.execute else 'DRY-RUN'} max_actions={args.max_actions} "
        f"handoff_step={args.handoff_step} max_authority={args.max_authority:.2f}",
        flush=True,
    )
    try:
        robot.connect()
        connected = True
        obs = wrapper.get_observation()
        measured = positions(obs)
        previous = measured.copy()
        from blend_core import ACT_START

        start_joint, start_grip = max_parts(np.abs(measured - ACT_START))
        print(
            f"Initial pose error: joint={start_joint:.6f} gripper={start_grip:.6f}",
            flush=True,
        )
        if args.execute and not args.resume_current and (
            start_joint > args.start_joint_tolerance
            or start_grip > args.start_gripper_tolerance
        ):
            raise RuntimeError(
                "Start pose is not the proven ACT start. Run reset_piper_pose.py first. "
                f"joint={start_joint:.6f}, gripper={start_grip:.6f}"
            )
        if args.execute and args.resume_current:
            print(
                "RESUME_CURRENT enabled: treating the measured held pose as an "
                "in-trajectory ACT observation.",
                flush=True,
            )

        act_engine.start()
        smol_engine.start()
        smol_started = True
        # RTC remains paused until the handoff point.

        if args.execute:
            for n in range(args.countdown, 0, -1):
                print(f"Hybrid execution starts in {n}...", flush=True)
                time.sleep(1)

        for step in range(args.max_actions):
            if stop:
                break
            tick = time.perf_counter()
            obs = wrapper.get_observation()
            measured = positions(obs)

            act_frame = build_dataset_frame(act_features, obs, prefix="observation")
            act_tensor = act_engine.get_action(act_frame)
            if act_tensor is None:
                raise RuntimeError("ACT returned no action")
            act_action = as_action(act_tensor, project_soft_limits=False)

            trigger = (
                not handoff_cancelled
                and (step >= args.handoff_step or args.handoff_file.exists())
            )
            if trigger and not smol_active:
                if args.handoff_file.exists():
                    args.handoff_file.unlink()
                smol_engine.reset()
                smol_engine.notify_observation(obs)
                smol_engine.resume()
                smol_active = True
                consecutive_rejects = 0
                print(f"HANDOFF_REQUEST step={step}: SmolVLA blend mode enabled", flush=True)

            source = "ACT"
            detail = "primary"
            if smol_active:
                smol_engine.notify_observation(obs)
                if smol_engine.failed:
                    raise RuntimeError(f"SmolVLA RTC failed:\n{smol_engine.failure_traceback}")
                candidate_tensor = smol_engine.get_action(None)
                if candidate_tensor is None:
                    raw_candidate = None
                else:
                    raw_candidate = as_action(
                        candidate_tensor,
                        project_soft_limits=True,
                    )
                previous_correction = correction.copy()
                res = blend_step(
                    act_action,
                    raw_candidate,
                    previous,
                    correction,
                    authority,
                    args,
                )
                source = res.source
                authority = res.authority
                correction = res.correction
                detail = res.detail
                accepted += res.accepted
                rejected += res.rejected
                fallback += res.fallback
                if res.rejected:
                    consecutive_rejects += 1
                else:
                    consecutive_rejects = 0
                if res.source != "SMOL_BLEND":
                    if consecutive_rejects >= args.max_consecutive_rejects:
                        smol_engine.pause()
                        smol_engine.reset()
                        smol_active = False
                        handoff_cancelled = True
                        detail += "; takeover cancelled, ACT-only"

            # Blend onto the trusted ACT base; gripper stays ACT-owned.
            raw_target = act_action.copy()
            raw_target[ARM] = act_action[ARM] + correction[ARM]
            raw_target[GRIPPER] = act_action[GRIPPER]

            # Final global smoothing; re-assert ACT gripper (no lag on gripper).
            target = apply_global_smoothing(
                raw_target, previous, act_action[GRIPPER], args.global_lowpass_alpha
            )

            # The command carries a SmolVLA correction (even a decaying one), so
            # validate it in the strict blended mode; pure-ACT ticks keep the
            # proven trusted-ACT semantics exactly.
            blended = float(np.max(np.abs(correction[ARM]))) > 1e-4
            joint_step, grip_step, tracking, grip_tracking = validate_execution(
                target,
                measured,
                previous,
                args,
                trusted_act=not blended,
                correction=correction if blended else None,
                previous_correction=previous_correction if blended else None,
            )
            if args.execute:
                wrapper.send_action(action_dict(target))
            previous = target
            queue = smol_engine.action_queue.qsize() if smol_engine.action_queue is not None else 0
            print(
                f"step={step:04d} source={source} joint_step={joint_step:.5f} "
                f"tracking={tracking:.5f} queue={queue} "
                f"authority={authority:.2f} corr={float(np.max(np.abs(correction[ARM]))):.4f} "
                f"{detail}",
                flush=True,
            )

            elapsed = time.perf_counter() - tick
            if elapsed < period:
                time.sleep(period - elapsed)

        if connected and args.execute:
            hold = positions(wrapper.get_observation())
            wrapper.send_action(action_dict(hold))
            print("Current measured pose is held.", flush=True)
        print(
            f"HYBRID_COMPLETE accepted_smol={accepted} rejected_smol={rejected} "
            f"act_fallback={fallback}",
            flush=True,
        )
        return 0
    except Exception:
        if connected and args.execute:
            try:
                hold = positions(wrapper.get_observation())
                wrapper.send_action(action_dict(hold))
                print("Safety exit: current measured pose is held.", flush=True)
            except Exception as exc:
                print(f"WARNING: hold failed: {exc}", flush=True)
        raise
    finally:
        if smol_started:
            smol_engine.stop()
        act_engine.stop()
        if connected:
            robot.disconnect()
        print("Piper disconnected; no disable command was sent.", flush=True)
        print("Existing ACT code/model and SmolVLA model were not modified.", flush=True)


def validate_execution(
    target: np.ndarray,
    measured: np.ndarray,
    previous: np.ndarray,
    args: argparse.Namespace,
    *,
    trusted_act: bool,
    correction: np.ndarray | None = None,
    previous_correction: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    joint_step, grip_step = max_parts(np.abs(target - previous))
    tracking, grip_tracking = max_parts(np.abs(measured - previous))
    if trusted_act:
        # Pure ACT tick: preserve the proven ACT deployment semantics exactly.
        if joint_step > args.act_joint_step_limit:
            raise RuntimeError(
                f"Safety stop: trusted ACT joint step {joint_step:.6f} > "
                f"{args.act_joint_step_limit:.6f}"
            )
        if tracking > args.act_tracking_limit:
            raise RuntimeError(
                f"Safety stop: trusted ACT tracking {tracking:.6f} > "
                f"{args.act_tracking_limit:.6f}"
            )
        return joint_step, grip_step, tracking, grip_tracking

    # Blended tick: ACT is still the base (so ACT limits on step/tracking apply),
    # but SmolVLA influence is additionally absolutely limited and slew-bounded.
    validate_absolute(target)
    if joint_step > args.act_joint_step_limit:
        raise RuntimeError(
            f"Safety stop: blended joint step {joint_step:.6f} > "
            f"{args.act_joint_step_limit:.6f}"
        )
    if tracking > args.act_tracking_limit:
        raise RuntimeError(
            f"Safety stop: blended tracking {tracking:.6f} > "
            f"{args.act_tracking_limit:.6f}"
        )
    if correction is not None and previous_correction is not None:
        corr_step = float(np.max(np.abs(correction[ARM] - previous_correction[ARM])))
        # _apply_desired clips the slew to exactly correction_step_limit; float32
        # rounding can make the measured change land a hair over the bound, so
        # allow the same 1e-5 rad epsilon the offline test asserts (matches S3).
        if corr_step > args.correction_step_limit + 1e-5:
            raise RuntimeError(
                f"Safety stop: correction step {corr_step:.6f} > "
                f"{args.correction_step_limit + 1e-5:.6f}"
            )
    return joint_step, grip_step, tracking, grip_tracking


if __name__ == "__main__":
    raise SystemExit(main())
