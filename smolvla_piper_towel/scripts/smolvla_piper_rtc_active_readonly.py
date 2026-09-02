#!/usr/bin/env python
"""Official RTC short-run safety harness for dual Piper + SmolVLA.

Independent from the working ACT deployment. Real commands require --execute.
The learned actions are never clipped or filtered: an unsafe action stops the run.
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
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.robots.piper_dual.config_piper_dual import PIPERDualConfig
from lerobot.robots.piper_dual.piper_dual import PIPERDual
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.utils.feature_utils import hw_to_dataset_features


ACTION_NAMES = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_gripper.pos",
]

ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--left-index", type=int, required=True)
    parser.add_argument("--middle-index", type=int, required=True)
    parser.add_argument("--right-index", type=int, required=True)
    parser.add_argument("--left-can", default="can1")
    parser.add_argument("--right-can", default="can0")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--queue-threshold", type=int, default=44)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--guidance-weight", type=float, default=5.0)
    parser.add_argument("--first-joint-step-limit", type=float, default=0.10)
    parser.add_argument("--joint-step-limit", type=float, default=0.06)
    parser.add_argument("--gripper-step-limit", type=float, default=0.01)
    parser.add_argument("--joint-tracking-limit", type=float, default=0.20)
    parser.add_argument("--gripper-tracking-limit", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument(
        "--discard-first-actions",
        type=int,
        default=0,
        help="Discard this many initial RTC actions without sending them.",
    )
    parser.add_argument(
        "--task",
        default="Fold the towel with both Piper arms.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send real commands. Omit for a read-only RTC chain test.",
    )
    parser.add_argument(
        "--active-readonly",
        action="store_true",
        help=(
            "Enable Piper to obtain real effort feedback, but never call "
            "send_action(). Mutually exclusive with --execute."
        ),
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 1 <= args.max_actions <= 60:
        parser.error("--max-actions must be in [1, 60]")
    if not 0 <= args.discard_first_actions <= 49:
        parser.error("--discard-first-actions must be in [0, 49]")
    if not 1 <= args.execution_horizon < 50:
        parser.error("--execution-horizon must be in [1, 49]")
    if not 1 <= args.queue_threshold < 50:
        parser.error("--queue-threshold must be in [1, 49]")
    if not 0 < args.joint_step_limit <= 0.10:
        parser.error("--joint-step-limit must be in (0, 0.10]")
    if not args.joint_step_limit <= args.first_joint_step_limit <= 0.15:
        parser.error("--first-joint-step-limit is invalid")
    if args.execute and args.active_readonly:
        parser.error("--execute and --active-readonly are mutually exclusive")
    return args


def positions_from_observation(observation: dict) -> np.ndarray:
    return np.asarray(
        [float(observation[name]) for name in ACTION_NAMES],
        dtype=np.float32,
    )


def action_dict(action: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(ACTION_NAMES, action, strict=True)
    }


def tensor_action(value: torch.Tensor) -> np.ndarray:
    result = value.detach().float().cpu().numpy().reshape(-1)
    if result.shape != (14,):
        raise RuntimeError(f"Unexpected RTC action shape: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("RTC action contains NaN or Inf")
    return result.astype(np.float32, copy=False)


def validate_tracking(
    measured: np.ndarray,
    previous_command: np.ndarray,
    joint_limit: float,
    gripper_limit: float,
) -> tuple[float, float]:
    error = np.abs(measured - previous_command)
    joint_error = float(error[ARM_INDICES].max())
    gripper_error = float(error[GRIPPER_INDICES].max())
    if joint_error > joint_limit:
        raise RuntimeError(
            "Safety stop: joint tracking error "
            f"{joint_error:.6f} > {joint_limit:.6f}"
        )
    if gripper_error > gripper_limit:
        raise RuntimeError(
            "Safety stop: gripper tracking error "
            f"{gripper_error:.6f} > {gripper_limit:.6f}"
        )
    return joint_error, gripper_error


def validate_step(
    target: np.ndarray,
    previous_command: np.ndarray,
    joint_limit: float,
    gripper_limit: float,
) -> tuple[float, float]:
    delta = np.abs(target - previous_command)
    joint_step = float(delta[ARM_INDICES].max())
    gripper_step = float(delta[GRIPPER_INDICES].max())
    if joint_step > joint_limit:
        index = int(ARM_INDICES[np.argmax(delta[ARM_INDICES])])
        raise RuntimeError(
            "Safety stop: consecutive joint command change is too large: "
            f"index={index}, value={joint_step:.6f}, limit={joint_limit:.6f}"
        )
    if gripper_step > gripper_limit:
        index = int(GRIPPER_INDICES[np.argmax(delta[GRIPPER_INDICES])])
        raise RuntimeError(
            "Safety stop: consecutive gripper command change is too large: "
            f"index={index}, value={gripper_step:.6f}, limit={gripper_limit:.6f}"
        )
    return joint_step, gripper_step


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device("cuda")
    period = 1.0 / args.fps
    stop_requested = False

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; holding current pose.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"Loading SmolVLA checkpoint: {args.model_dir}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(args.model_dir))
    policy.to(device)
    policy.eval()
    rtc_config = RTCConfig(
        enabled=True,
        mode="guided",
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=args.guidance_weight,
        execution_horizon=args.execution_horizon,
        debug=False,
    )
    policy.config.rtc_config = rtc_config
    policy.init_rtc_processor()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        str(args.model_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    cameras = {
        "left": OpenCVCameraConfig(
            index_or_path=args.left_index, width=640, height=480, fps=30
        ),
        "middle": OpenCVCameraConfig(
            index_or_path=args.middle_index, width=640, height=480, fps=30
        ),
        "right": OpenCVCameraConfig(
            index_or_path=args.right_index, width=640, height=480, fps=30
        ),
    }
    robot = PIPERDual(
        PIPERDualConfig(
            left_port=args.left_can,
            right_port=args.right_can,
            cameras=cameras,
            read_only=not (args.execute or args.active_readonly),
        )
    )
    wrapper = ThreadSafeRobot(robot)
    hw_features = {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.observation_features, "observation"),
    }
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=wrapper,
        rtc_config=rtc_config,
        hw_features=hw_features,
        task=args.task,
        fps=args.fps,
        device=str(device),
        use_torch_compile=False,
        rtc_queue_threshold=args.queue_threshold,
    )

    connected = False
    engine_started = False
    last_command: np.ndarray | None = None
    action_count = 0
    discarded_actions = 0
    empty_ticks = 0
    mode = (
        "EXECUTE"
        if args.execute
        else "ACTIVE-READONLY"
        if args.active_readonly
        else "DRY-RUN"
    )
    print(
        f"Mode={mode} max_actions={args.max_actions} fps={args.fps:.1f} "
        f"guidance={args.guidance_weight:.2f} horizon={args.execution_horizon} "
        f"queue_threshold={args.queue_threshold}",
        flush=True,
    )

    try:
        robot.connect()
        connected = True
        initial_observation = wrapper.get_observation()
        last_command = positions_from_observation(initial_observation)
        print("Initial measured positions:", last_command, flush=True)

        engine.start()
        engine_started = True
        engine.resume()

        if args.execute:
            for remaining in range(args.countdown, 0, -1):
                print(f"Real RTC execution starts in {remaining}...", flush=True)
                time.sleep(1)

        while action_count < args.max_actions and not stop_requested:
            loop_start = time.perf_counter()
            observation = wrapper.get_observation()
            engine.notify_observation(observation)
            action_tensor = engine.get_action(None)

            if engine.failed:
                raise RuntimeError(
                    "Official RTC inference thread failed:\n"
                    f"{engine.failure_traceback}"
                )

            if action_tensor is None:
                empty_ticks += 1
                if empty_ticks % 30 == 0:
                    print(f"Waiting for RTC queue, ticks={empty_ticks}", flush=True)
            else:
                empty_ticks = 0
                target = tensor_action(action_tensor)
                if discarded_actions < args.discard_first_actions:
                    measured = positions_from_observation(observation)
                    joint_delta = float(
                        np.max(np.abs(target[ARM_INDICES] - measured[ARM_INDICES]))
                    )
                    print(
                        f"discarded_initial_action={discarded_actions:04d} "
                        f"joint_delta_from_measured={joint_delta:.6f}",
                        flush=True,
                    )
                    discarded_actions += 1
                    elapsed = time.perf_counter() - loop_start
                    if elapsed < period:
                        time.sleep(period - elapsed)
                    continue
                measured = positions_from_observation(observation)
                if args.execute:
                    tracking, gripper_tracking = validate_tracking(
                        measured,
                        last_command,
                        args.joint_tracking_limit,
                        args.gripper_tracking_limit,
                    )
                else:
                    tracking = 0.0
                    gripper_tracking = 0.0
                step_limit = (
                    args.first_joint_step_limit
                    if action_count == 0
                    else args.joint_step_limit
                )
                joint_step, gripper_step = validate_step(
                    target,
                    last_command,
                    step_limit,
                    args.gripper_step_limit,
                )
                if args.execute:
                    wrapper.send_action(action_dict(target))
                last_command = target
                queue_size = (
                    engine.action_queue.qsize()
                    if engine.action_queue is not None
                    else -1
                )
                print(
                    f"action={action_count:04d} joint_step={joint_step:.6f} "
                    f"gripper_step={gripper_step:.6f} tracking={tracking:.6f} "
                    f"gripper_tracking={gripper_tracking:.6f} queue={queue_size}",
                    flush=True,
                )
                action_count += 1

            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

        if connected and args.execute:
            final_observation = wrapper.get_observation()
            final_position = positions_from_observation(final_observation)
            wrapper.send_action(action_dict(final_position))
            print("Current measured pose is held.", flush=True)
        print(f"Reached max_actions={action_count}", flush=True)
        print("SMOLVLA_OFFICIAL_RTC_SAFE_RUN_SUCCESS", flush=True)
        return 0
    except Exception:
        if connected and args.execute:
            try:
                hold_observation = wrapper.get_observation()
                hold_position = positions_from_observation(hold_observation)
                wrapper.send_action(action_dict(hold_position))
                print("Safety exit: current measured pose is held.", flush=True)
            except Exception as hold_error:
                print(f"WARNING: failed to hold pose: {hold_error}", flush=True)
        raise
    finally:
        if engine_started:
            engine.stop()
        if connected:
            robot.disconnect()
        print("Piper disconnected; no disable command was sent.", flush=True)
        if args.active_readonly:
            print("ACTIVE-READONLY: send_action() was never called.", flush=True)
        print("ACT was not modified.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
