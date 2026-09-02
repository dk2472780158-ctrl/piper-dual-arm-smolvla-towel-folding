#!/usr/bin/env python
"""Safe standalone SmolVLA runner for a dual Piper setup.

This file is intentionally independent from the working ACT deployment.
It defaults to dry-run. Passing --execute is required to send commands.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame
from lerobot.robots.piper_dual.config_piper_dual import PIPERDualConfig
from lerobot.robots.piper_dual.piper_dual import PIPERDual
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
    parser.add_argument("--execution-horizon", type=int, default=5)
    parser.add_argument("--joint-step-limit", type=float, default=0.015)
    parser.add_argument("--gripper-step-limit", type=float, default=0.001)
    parser.add_argument("--joint-tracking-limit", type=float, default=0.20)
    parser.add_argument("--gripper-tracking-limit", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument(
        "--task",
        default="Fold the towel with both Piper arms.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send bounded position commands. Without this flag the runner is read-only.",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_actions <= 0:
        parser.error("--max-actions must be positive")
    if not 1 <= args.execution_horizon <= 5:
        parser.error("--execution-horizon must be between 1 and 5")
    if args.joint_step_limit <= 0 or args.joint_step_limit > 0.10:
        parser.error("--joint-step-limit must be in (0, 0.10]")
    if args.gripper_step_limit <= 0 or args.gripper_step_limit > 0.02:
        parser.error("--gripper-step-limit must be in (0, 0.02]")
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


def limit_command(
    target: np.ndarray,
    previous: np.ndarray,
    joint_limit: float,
    gripper_limit: float,
) -> tuple[np.ndarray, float]:
    limits = np.full(14, joint_limit, dtype=np.float32)
    limits[GRIPPER_INDICES] = gripper_limit
    raw_delta = target - previous
    bounded_delta = np.clip(raw_delta, -limits, limits)
    correction = float(np.max(np.abs(raw_delta - bounded_delta)))
    return previous + bounded_delta, correction


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    period = 1.0 / args.fps
    stop_requested = False

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True
        print("收到停止请求；将在本控制周期结束后保持当前位置。", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"Loading SmolVLA checkpoint: {args.model_dir}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(args.model_dir))
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(args.model_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    camera_config = {
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
            cameras=camera_config,
            read_only=not args.execute,
        )
    )

    connected = False
    last_command: np.ndarray | None = None
    action_count = 0
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(
        f"Mode={mode} max_actions={args.max_actions} "
        f"horizon={args.execution_horizon} fps={args.fps:.1f} "
        f"joint_step={args.joint_step_limit:.4f}",
        flush=True,
    )

    try:
        robot.connect()
        connected = True
        features = {
            **hw_to_dataset_features(robot.action_features, "action"),
            **hw_to_dataset_features(robot.observation_features, "observation"),
        }
        observation = robot.get_observation()
        last_command = positions_from_observation(observation)
        print("Initial measured positions:", last_command, flush=True)

        if args.execute:
            for remaining in range(args.countdown, 0, -1):
                print(f"Real execution starts in {remaining}...", flush=True)
                time.sleep(1)

        block_index = 0
        while action_count < args.max_actions and not stop_requested:
            observation = robot.get_observation()
            frame = build_inference_frame(
                observation=observation,
                ds_features=features,
                device=device,
                task=args.task,
                robot_type="piper_dual",
            )
            processed = preprocess(frame)
            policy.reset()
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.synchronize()
            planning_start = time.perf_counter()
            with torch.inference_mode():
                chunk_tensor = policy.predict_action_chunk(processed)
            torch.cuda.synchronize()
            planning_ms = (time.perf_counter() - planning_start) * 1000.0
            chunk_tensor = postprocess(chunk_tensor)
            chunk = chunk_tensor.detach().float().cpu().numpy()
            if chunk.ndim == 3:
                chunk = chunk[0]
            if chunk.shape != (50, 14):
                raise RuntimeError(f"Unexpected action chunk shape: {chunk.shape}")
            if not np.isfinite(chunk).all():
                raise RuntimeError("Action chunk contains NaN or Inf")
            print(
                f"plan block={block_index:03d} planning_ms={planning_ms:.1f}",
                flush=True,
            )

            for local_index in range(args.execution_horizon):
                if action_count >= args.max_actions or stop_requested:
                    break
                loop_start = time.perf_counter()
                current_observation = robot.get_observation()
                measured = positions_from_observation(current_observation)
                tracking_error = np.abs(measured - last_command)
                max_joint_tracking = float(tracking_error[ARM_INDICES].max())
                max_gripper_tracking = float(
                    tracking_error[GRIPPER_INDICES].max()
                )
                if max_joint_tracking > args.joint_tracking_limit:
                    raise RuntimeError(
                        "Safety stop: joint tracking error "
                        f"{max_joint_tracking:.6f} > {args.joint_tracking_limit:.6f}"
                    )
                if max_gripper_tracking > args.gripper_tracking_limit:
                    raise RuntimeError(
                        "Safety stop: gripper tracking error "
                        f"{max_gripper_tracking:.6f} > "
                        f"{args.gripper_tracking_limit:.6f}"
                    )
                command, correction = limit_command(
                    chunk[local_index],
                    last_command,
                    args.joint_step_limit,
                    args.gripper_step_limit,
                )
                if args.execute:
                    robot.send_action(action_dict(command))
                print(
                    f"action={action_count:04d} block={block_index:03d} "
                    f"index={local_index} tracking={max_joint_tracking:.6f} "
                    f"slew_correction={correction:.6f}",
                    flush=True,
                )
                last_command = command if args.execute else measured
                action_count += 1
                remaining = period - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
            block_index += 1

        if connected and args.execute:
            final_observation = robot.get_observation()
            final_position = positions_from_observation(final_observation)
            robot.send_action(action_dict(final_position))
            print("Current measured pose is held.", flush=True)
        print(f"Reached max_actions={action_count}", flush=True)
        print("SMOLVLA_SAFE_RUN_SUCCESS", flush=True)
        return 0
    except Exception:
        if connected and args.execute:
            try:
                hold_observation = robot.get_observation()
                hold_position = positions_from_observation(hold_observation)
                robot.send_action(action_dict(hold_position))
                print("Safety exit: current measured pose is held.", flush=True)
            except Exception as hold_error:
                print(f"WARNING: failed to hold current pose: {hold_error}", flush=True)
        raise
    finally:
        if connected:
            robot.disconnect()
        print("Piper disconnected; no disable command was sent.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
