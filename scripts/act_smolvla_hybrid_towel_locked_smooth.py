#!/usr/bin/env python3
"""Smooth experimental dual-Piper ACT + guarded SmolVLA controller.

This file is independent of the existing ACT deployment.  It loads the proven
ACT checkpoint as the reference controller and lets SmolVLA take over only
when its proposed action is continuous, inside Piper limits, and sufficiently
close to ACT.  A source-independent velocity/acceleration limiter makes both
policy commands and ACT/SmolVLA transitions continuous.
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


ACTION_NAMES = [
    "left_joint_1.pos", "left_joint_2.pos", "left_joint_3.pos",
    "left_joint_4.pos", "left_joint_5.pos", "left_joint_6.pos",
    "left_gripper.pos", "right_joint_1.pos", "right_joint_2.pos",
    "right_joint_3.pos", "right_joint_4.pos", "right_joint_5.pos",
    "right_joint_6.pos", "right_gripper.pos",
]
ARM = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER = np.asarray([6, 13])

# The start used by the proven ACT deployment/reset_piper_pose.py.
ACT_START = np.asarray([
    0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
    0.2780382633, 0.0, -0.0443961099, 0.0, 0.0,
    0.0536503866, 0.3397218585, -0.0396250561, 0.0,
], dtype=np.float32)

SINGLE_LOWER = np.asarray(
    [-1.61, -0.05, -1.93, -1.58, -1.40, -1.58, 0.0],
    dtype=np.float32,
)
SINGLE_UPPER = np.asarray(
    [1.61, 2.10, 0.06, 1.58, 1.40, 1.58, 0.08],
    dtype=np.float32,
)
LOWER = np.concatenate([SINGLE_LOWER, SINGLE_LOWER])
UPPER = np.concatenate([SINGLE_UPPER, SINGLE_UPPER])
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
    p.add_argument("--handoff-step", type=int, default=450)
    p.add_argument("--handoff-file", type=Path, default=Path("/tmp/piper_hybrid_handoff"))
    p.add_argument("--guidance-weight", type=float, default=2.5)
    p.add_argument("--execution-horizon", type=int, default=10)
    p.add_argument("--queue-threshold", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--joint-step-limit", type=float, default=0.10)
    p.add_argument("--gripper-step-limit", type=float, default=0.012)
    p.add_argument("--tracking-limit", type=float, default=0.20)
    p.add_argument("--gripper-tracking-limit", type=float, default=0.03)
    p.add_argument("--policy-disagreement", type=float, default=0.12)
    p.add_argument("--gripper-disagreement", type=float, default=0.012)
    p.add_argument(
        "--smol-lowpass-alpha",
        type=float,
        default=1.0,
        help="SmolVLA EMA weight; 1.0 disables filtering, 0.25 is smoother.",
    )
    p.add_argument("--smol-raw-step-limit", type=float, default=0.60)
    p.add_argument("--smol-raw-policy-disagreement", type=float, default=0.50)
    p.add_argument(
        "--global-lowpass-alpha",
        type=float,
        default=1.0,
        help="Final command EMA weight applied to both ACT and SmolVLA.",
    )
    p.add_argument(
        "--smooth-joint-step-limit",
        type=float,
        default=0.040,
        help="Maximum final joint command increment per control cycle (rad).",
    )
    p.add_argument(
        "--smooth-joint-accel-limit",
        type=float,
        default=0.004,
        help="Maximum change of joint command increment per cycle (rad/cycle^2).",
    )
    p.add_argument(
        "--smooth-gripper-step-limit",
        type=float,
        default=0.003,
        help="Maximum final gripper command increment per cycle (m).",
    )
    p.add_argument(
        "--smooth-gripper-accel-limit",
        type=float,
        default=0.0004,
        help="Maximum change of gripper command increment per cycle (m/cycle^2).",
    )
    p.add_argument(
        "--smooth-start-step",
        type=int,
        default=0,
        help="Keep the original policy command unchanged before this control step.",
    )
    p.add_argument("--start-joint-tolerance", type=float, default=0.12)
    p.add_argument("--start-gripper-tolerance", type=float, default=0.012)
    p.add_argument("--max-consecutive-rejects", type=int, default=15)
    p.add_argument("--smol-accept-streak", type=int, default=3)
    p.add_argument("--smol-reject-streak", type=int, default=3)
    p.add_argument("--source-min-dwell", type=int, default=8)
    p.add_argument("--act-joint-step-limit", type=float, default=1.0)
    p.add_argument("--act-tracking-limit", type=float, default=3.0)
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
    if min(
        args.smooth_joint_step_limit,
        args.smooth_joint_accel_limit,
        args.smooth_gripper_step_limit,
        args.smooth_gripper_accel_limit,
    ) <= 0:
        p.error("all smooth step/acceleration limits must be positive")
    if not 0 <= args.smooth_start_step < args.max_actions:
        p.error("smooth-start-step must be in [0, max-actions)")
    if min(args.smol_accept_streak, args.smol_reject_streak, args.source_min_dwell) < 1:
        p.error("source streak and dwell values must be positive")
    return args


def positions(obs: dict) -> np.ndarray:
    return np.asarray([float(obs[k]) for k in ACTION_NAMES], dtype=np.float32)


def action_dict(action: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(ACTION_NAMES, action, strict=True)}


def as_action(value: torch.Tensor, *, project_soft_limits: bool) -> np.ndarray:
    result = value.detach().float().cpu().numpy().reshape(-1)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid action: shape={result.shape}, finite={np.isfinite(result).all()}")
    result = result.astype(np.float32, copy=False)
    # Piper's gripper driver uses abs(); make the intended physical boundary explicit.
    result[GRIPPER] = np.clip(result[GRIPPER], 0.0, 0.08)
    # Learned controllers can overshoot a conservative software boundary by a
    # few milliradians.  Project only small violations; large violations remain
    # untouched so validate_absolute() stops execution instead of hiding them.
    if project_soft_limits:
        violation = np.maximum(LOWER - result, result - UPPER)
        joint_violation = float(np.max(violation[ARM]))
        if 0.0 < joint_violation <= SOFT_LIMIT_PROJECTION_MARGIN:
            before = result.copy()
            result[ARM] = np.clip(result[ARM], LOWER[ARM], UPPER[ARM])
            correction = float(np.max(np.abs(result[ARM] - before[ARM])))
            print(
                f"soft_limit_projection max_correction={correction:.6f} rad",
                flush=True,
            )
    return result


def max_parts(delta: np.ndarray) -> tuple[float, float]:
    return float(np.max(delta[ARM])), float(np.max(delta[GRIPPER]))


def smooth_rate_limit(
    filtered_target: np.ndarray,
    previous: np.ndarray,
    previous_step: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Bound command velocity and acceleration without changing policy state."""
    desired_step = filtered_target - previous
    velocity_limited = desired_step.copy()
    velocity_limited[ARM] = np.clip(
        velocity_limited[ARM],
        -args.smooth_joint_step_limit,
        args.smooth_joint_step_limit,
    )
    velocity_limited[GRIPPER] = np.clip(
        velocity_limited[GRIPPER],
        -args.smooth_gripper_step_limit,
        args.smooth_gripper_step_limit,
    )

    step_change = velocity_limited - previous_step
    step_change[ARM] = np.clip(
        step_change[ARM],
        -args.smooth_joint_accel_limit,
        args.smooth_joint_accel_limit,
    )
    step_change[GRIPPER] = np.clip(
        step_change[GRIPPER],
        -args.smooth_gripper_accel_limit,
        args.smooth_gripper_accel_limit,
    )
    limited_step = previous_step + step_change
    limited_step[ARM] = np.clip(
        limited_step[ARM],
        -args.smooth_joint_step_limit,
        args.smooth_joint_step_limit,
    )
    limited_step[GRIPPER] = np.clip(
        limited_step[GRIPPER],
        -args.smooth_gripper_step_limit,
        args.smooth_gripper_step_limit,
    )

    target = (previous + limited_step).astype(np.float32, copy=False)
    target[GRIPPER] = np.clip(target[GRIPPER], 0.0, 0.08)
    limited_step = target - previous
    velocity_correction = max_parts(np.abs(desired_step - velocity_limited))[0]
    accel_correction = max_parts(np.abs(velocity_limited - limited_step))[0]
    return target, limited_step.astype(np.float32, copy=False), velocity_correction, accel_correction


def validate_absolute(action: np.ndarray) -> None:
    bad = np.flatnonzero((action < LOWER - 1e-6) | (action > UPPER + 1e-6))
    if bad.size:
        i = int(bad[0])
        raise RuntimeError(
            f"Safety stop: action[{i}]={action[i]:.6f} outside "
            f"[{LOWER[i]:.6f}, {UPPER[i]:.6f}]"
        )


def validate_execution(
    target: np.ndarray,
    measured: np.ndarray,
    previous: np.ndarray,
    args: argparse.Namespace,
    *,
    trusted_act: bool,
    trusted_gripper: bool = False,
) -> tuple[float, float, float, float]:
    joint_step, grip_step = max_parts(np.abs(target - previous))
    tracking, grip_tracking = max_parts(np.abs(measured - previous))
    if trusted_act:
        # Preserve the proven ACT deployment semantics.  Its original runner
        # allows large action-chunk targets and lets the Piper position loop
        # track them smoothly; SmolVLA never receives these relaxed limits.
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

    validate_absolute(target)
    if joint_step > args.joint_step_limit:
        raise RuntimeError(
            f"Safety stop: command joint step {joint_step:.6f} > {args.joint_step_limit:.6f}"
        )
    if not trusted_gripper and grip_step > args.gripper_step_limit:
        raise RuntimeError(
            f"Safety stop: command gripper step {grip_step:.6f} > {args.gripper_step_limit:.6f}"
        )
    if tracking > args.tracking_limit:
        raise RuntimeError(
            f"Safety stop: joint tracking {tracking:.6f} > {args.tracking_limit:.6f}"
        )
    if not trusted_gripper and grip_tracking > args.gripper_tracking_limit:
        raise RuntimeError(
            f"Safety stop: gripper tracking {grip_tracking:.6f} > {args.gripper_tracking_limit:.6f}"
        )
    return joint_step, grip_step, tracking, grip_tracking


def smol_is_acceptable(
    raw_candidate: np.ndarray,
    candidate: np.ndarray,
    act_reference: np.ndarray,
    previous: np.ndarray,
    args: argparse.Namespace,
) -> tuple[bool, str, float, float, float]:
    try:
        validate_absolute(raw_candidate)
        validate_absolute(candidate)
    except RuntimeError as exc:
        return False, str(exc), float("inf"), float("inf"), float("inf")
    raw_step_joint, raw_step_grip = max_parts(np.abs(raw_candidate - previous))
    raw_disagree_joint, raw_disagree_grip = max_parts(
        np.abs(raw_candidate - act_reference)
    )
    if raw_step_joint > args.smol_raw_step_limit:
        return (
            False,
            f"raw_step={raw_step_joint:.6f}",
            raw_disagree_joint,
            raw_disagree_grip,
            raw_step_joint,
        )
    if raw_disagree_joint > args.smol_raw_policy_disagreement:
        return (
            False,
            f"raw_ACT_disagreement={raw_disagree_joint:.6f}",
            raw_disagree_joint,
            raw_disagree_grip,
            raw_step_joint,
        )
    step_joint, _step_grip = max_parts(np.abs(candidate - previous))
    disagree_joint, disagree_grip = max_parts(np.abs(candidate - act_reference))
    if step_joint > args.joint_step_limit:
        return False, f"filtered_step={step_joint:.6f}", disagree_joint, disagree_grip, raw_step_joint
    if disagree_joint > args.policy_disagreement:
        return False, f"filtered_ACT_disagreement={disagree_joint:.6f}", disagree_joint, disagree_grip, raw_step_joint
    return True, "accepted", disagree_joint, disagree_grip, raw_step_joint


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
    smol_control_started = False
    control_mode = "ACT"
    accept_streak = 0
    reject_streak = 0
    source_dwell = 0
    handoff_cancelled = False
    previous: np.ndarray | None = None
    previous_command_step = np.zeros(14, dtype=np.float32)
    accepted = rejected = fallback = 0
    consecutive_rejects = 0

    print(
        f"Mode={'EXECUTE' if args.execute else 'DRY-RUN'} max_actions={args.max_actions} "
        f"handoff_step={args.handoff_step} guidance={args.guidance_weight:.1f}",
        flush=True,
    )
    try:
        robot.connect()
        connected = True
        obs = wrapper.get_observation()
        measured = positions(obs)
        previous = measured.copy()
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
                print(f"HANDOFF_REQUEST step={step}: SmolVLA guarded mode enabled", flush=True)

            source = "ACT"
            target = act_action
            detail = "primary"
            if source_dwell > 0:
                source_dwell -= 1
            if smol_active:
                smol_engine.notify_observation(obs)
                if smol_engine.failed:
                    raise RuntimeError(f"SmolVLA RTC failed:\n{smol_engine.failure_traceback}")
                candidate_tensor = smol_engine.get_action(None)
                candidate_ok = False
                candidate = None
                reason = "Smol queue empty"
                dj = dg = raw_step = float("nan")
                if candidate_tensor is not None:
                    raw_candidate = as_action(
                        candidate_tensor,
                        project_soft_limits=True,
                    )
                    candidate = (
                        previous
                        + args.smol_lowpass_alpha
                        * (raw_candidate - previous)
                    ).astype(np.float32, copy=False)
                    candidate_ok, reason, dj, dg, raw_step = smol_is_acceptable(
                        raw_candidate,
                        candidate,
                        act_action,
                        previous,
                        args,
                    )
                    if candidate_ok:
                        live_tracking, _live_grip_tracking = max_parts(
                            np.abs(measured - previous)
                        )
                        if live_tracking > args.tracking_limit:
                            candidate_ok = False
                            reason = (
                                "tracking_gate="
                                f"{live_tracking:.6f}"
                            )

                if control_mode == "ACT":
                    reject_streak = 0
                    if candidate_ok:
                        accept_streak += 1
                        detail = (
                            f"candidate_warmup={accept_streak}/"
                            f"{args.smol_accept_streak}"
                        )
                        if (
                            source_dwell == 0
                            and accept_streak >= args.smol_accept_streak
                        ):
                            control_mode = "SMOL"
                            source_dwell = args.source_min_dwell
                            accept_streak = 0
                            target = candidate
                            source = "SMOL"
                            smol_control_started = True
                            accepted += 1
                            detail = (
                                "MODE_SWITCH ACT->SMOL "
                                f"raw_step={raw_step:.4f} "
                                f"ACT_delta={dj:.4f}/{dg:.4f}"
                            )
                        else:
                            fallback += 1
                    else:
                        accept_streak = 0
                        fallback += 1
                        if candidate_tensor is not None:
                            rejected += 1
                        detail = f"ACT_LOCK candidate_rejected:{reason}"
                else:
                    accept_streak = 0
                    if candidate_ok:
                        reject_streak = 0
                        target = candidate
                        source = "SMOL"
                        accepted += 1
                        detail = (
                            f"SMOL_LOCK raw_step={raw_step:.4f} "
                            f"ACT_delta={dj:.4f}/{dg:.4f}"
                        )
                    else:
                        rejected += int(candidate_tensor is not None)
                        fallback += 1
                        reject_streak += 1
                        if (
                            source_dwell == 0
                            and reject_streak >= args.smol_reject_streak
                        ):
                            control_mode = "ACT"
                            source_dwell = args.source_min_dwell
                            reject_streak = 0
                            target = act_action
                            source = "ACT"
                            detail = f"MODE_SWITCH SMOL->ACT reason={reason}"
                        else:
                            # Briefly brake instead of jumping to ACT for one bad frame.
                            target = previous.copy()
                            source = "HOLD"
                            detail = (
                                f"SMOL_LOCK_HOLD reject={reject_streak}/"
                                f"{args.smol_reject_streak} reason={reason}"
                            )

            raw_selected_target = target
            if step < args.smooth_start_step:
                # Preserve the proven ACT approach/grasp trajectory exactly.
                target = raw_selected_target.astype(np.float32, copy=False)
                previous_command_step = (target - previous).astype(
                    np.float32,
                    copy=False,
                )
                velocity_correction = 0.0
                accel_correction = 0.0
                detail += "; smoothing=off-proven-ACT"
            else:
                target = (
                    previous
                    + args.global_lowpass_alpha
                    * (raw_selected_target - previous)
                ).astype(np.float32, copy=False)
                target, previous_command_step, velocity_correction, accel_correction = (
                    smooth_rate_limit(
                        target,
                        previous,
                        previous_command_step,
                        args,
                    )
                )

            # SmolVLA may help with the arm trajectory, but it must never reopen
            # a gripper that the proven ACT controller has already closed.
            target[GRIPPER] = act_action[GRIPPER]
            previous_command_step[GRIPPER] = target[GRIPPER] - previous[GRIPPER]
            if source == "SMOL":
                detail += "; gripper_source=ACT"

            joint_step, grip_step, tracking, grip_tracking = validate_execution(
                target,
                measured,
                previous,
                args,
                trusted_act=(source in ("ACT", "HOLD")),
                trusted_gripper=True,
            )
            if args.execute:
                wrapper.send_action(action_dict(target))
            previous = target
            queue = smol_engine.action_queue.qsize() if smol_engine.action_queue is not None else 0
            print(
                f"step={step:04d} source={source} joint_step={joint_step:.5f} "
                f"tracking={tracking:.5f} queue={queue} "
                f"global_alpha={args.global_lowpass_alpha:.2f} "
                f"vel_correction={velocity_correction:.5f} "
                f"accel_correction={accel_correction:.5f} {detail}",
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


if __name__ == "__main__":
    raise SystemExit(main())
