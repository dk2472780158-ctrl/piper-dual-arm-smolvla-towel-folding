#!/usr/bin/env python
"""Offline A/B evaluation of plain replanning versus SmolVLA RTC."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla import SmolVLAPolicy


ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/towel_fold_dataset_aug_v1")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--executed-actions", type=int, default=6)
    parser.add_argument("--inference-delay", type=int, default=5)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--prefix-attention-schedule",
        choices=["EXP", "ONES", "LINEAR", "ZEROS"],
        default="EXP",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def to_numpy(value: torch.Tensor) -> np.ndarray:
    result = value.detach().float().cpu().numpy()
    if result.ndim == 3:
        result = result[0]
    return result


def scalar_episode(sample: dict) -> int:
    value = sample["episode_index"]
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def main() -> int:
    args = parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    if args.executed_actions <= 0:
        raise ValueError("--executed-actions must be positive")
    device = torch.device("cuda")

    print("Loading dataset:", args.dataset_root, flush=True)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    print("Dataset frames:", len(dataset), flush=True)
    print("Dataset episodes:", dataset.num_episodes, flush=True)

    print("Loading policy:", args.model_dir, flush=True)
    policy = SmolVLAPolicy.from_pretrained(str(args.model_dir))
    policy.to(device)
    policy.eval()
    if not hasattr(policy, "init_rtc_processor"):
        raise RuntimeError("This SmolVLA build has no init_rtc_processor()")
    policy.config.rtc_config = RTCConfig(
        enabled=True,
        mode="guided",
        prefix_attention_schedule=RTCAttentionSchedule[
            args.prefix_attention_schedule
        ],
        max_guidance_weight=args.max_guidance_weight,
        execution_horizon=args.execution_horizon,
        debug=False,
    )
    policy.init_rtc_processor()
    print("RTC processor:", type(policy.rtc_processor).__name__, flush=True)

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(args.model_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Choose pairs spread through the dataset while preventing episode crossings.
    candidates: list[tuple[int, dict, dict]] = []
    desired_scan = max(args.pairs * 4, 16)
    stride = max(1, len(dataset) // desired_scan)
    index = max(60, stride // 2)
    while index + args.executed_actions < len(dataset) and len(candidates) < args.pairs:
        first = dataset[index]
        second = dataset[index + args.executed_actions]
        if scalar_episode(first) == scalar_episode(second):
            candidates.append((index, first, second))
        index += stride
    if len(candidates) < args.pairs:
        raise RuntimeError(
            f"Only found {len(candidates)} valid same-episode pairs, requested {args.pairs}"
        )

    rows: list[dict[str, float | int]] = []
    for pair_index, (frame_index, first, second) in enumerate(candidates):
        first = copy.deepcopy(first)
        second = copy.deepcopy(second)
        first.setdefault("task", "Fold the towel with both Piper arms.")
        second.setdefault("task", "Fold the towel with both Piper arms.")
        expert = second["action"].detach().float().cpu().numpy().reshape(-1)
        processed_first = preprocess(first)
        processed_second = preprocess(second)

        torch.manual_seed(args.seed + pair_index * 2)
        torch.cuda.manual_seed_all(args.seed + pair_index * 2)
        prev_noise = torch.randn(
            1,
            policy.config.chunk_size,
            policy.config.max_action_dim,
            device=device,
            dtype=torch.float32,
        )
        torch.manual_seed(args.seed + pair_index * 2 + 1)
        torch.cuda.manual_seed_all(args.seed + pair_index * 2 + 1)
        next_noise = torch.randn(
            1,
            policy.config.chunk_size,
            policy.config.max_action_dim,
            device=device,
            dtype=torch.float32,
        )

        policy.config.rtc_config.enabled = False
        policy.reset()
        with torch.inference_mode():
            previous_raw = policy.predict_action_chunk(
                processed_first, noise=prev_noise.clone()
            )
        policy.reset()
        with torch.inference_mode():
            plain_raw = policy.predict_action_chunk(
                processed_second, noise=next_noise.clone()
            )

        leftover_raw = previous_raw[:, args.executed_actions :, :]
        policy.config.rtc_config.enabled = True
        policy.reset()
        torch.cuda.synchronize()
        rtc_start = time.perf_counter()
        # RTC internally re-enables autograd to compute its guidance
        # correction. torch.inference_mode() would make that impossible;
        # torch.no_grad() is intentionally used because it can be locally
        # overridden by RTCProcessor.denoise_step().
        with torch.no_grad():
            rtc_raw = policy.predict_action_chunk(
                processed_second,
                noise=next_noise.clone(),
                prev_chunk_left_over=leftover_raw,
                inference_delay=args.inference_delay,
                execution_horizon=args.execution_horizon,
            )
        torch.cuda.synchronize()
        rtc_ms = (time.perf_counter() - rtc_start) * 1000.0

        previous = to_numpy(postprocess(previous_raw.clone()))
        plain = to_numpy(postprocess(plain_raw.clone()))
        rtc = to_numpy(postprocess(rtc_raw.clone()))
        old_boundary = previous[args.executed_actions]
        plain_boundary = np.abs(plain[0] - old_boundary)
        rtc_boundary = np.abs(rtc[0] - old_boundary)
        plain_steps = np.abs(np.diff(plain[: args.executed_actions], axis=0))
        rtc_steps = np.abs(np.diff(rtc[: args.executed_actions], axis=0))
        plain_mae = float(np.mean(np.abs(plain[0] - expert)))
        rtc_mae = float(np.mean(np.abs(rtc[0] - expert)))
        row = {
            "pair": pair_index,
            "frame": frame_index,
            "episode": scalar_episode(first),
            "plain_boundary_max": float(plain_boundary[ARM_INDICES].max()),
            "rtc_boundary_max": float(rtc_boundary[ARM_INDICES].max()),
            "plain_prefix_step_max": float(plain_steps[:, ARM_INDICES].max()),
            "rtc_prefix_step_max": float(rtc_steps[:, ARM_INDICES].max()),
            "plain_expert_mae": plain_mae,
            "rtc_expert_mae": rtc_mae,
            "rtc_ms": rtc_ms,
        }
        rows.append(row)
        print(
            f"pair={pair_index:02d} episode={row['episode']:03d} frame={frame_index:06d} "
            f"boundary plain={row['plain_boundary_max']:.6f} "
            f"rtc={row['rtc_boundary_max']:.6f} "
            f"prefix plain={row['plain_prefix_step_max']:.6f} "
            f"rtc={row['rtc_prefix_step_max']:.6f} "
            f"expert_mae plain={plain_mae:.6f} rtc={rtc_mae:.6f} "
            f"rtc_ms={rtc_ms:.1f}",
            flush=True,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows])

    plain_boundary = values("plain_boundary_max")
    rtc_boundary = values("rtc_boundary_max")
    plain_prefix = values("plain_prefix_step_max")
    rtc_prefix = values("rtc_prefix_step_max")
    plain_mae = values("plain_expert_mae")
    rtc_mae = values("rtc_expert_mae")
    rtc_times = values("rtc_ms")

    print("\n===== RTC A/B SUMMARY =====")
    print("pairs:", len(rows))
    print("plain boundary mean:", float(plain_boundary.mean()))
    print("rtc boundary mean:", float(rtc_boundary.mean()))
    print("plain boundary p95:", float(np.percentile(plain_boundary, 95)))
    print("rtc boundary p95:", float(np.percentile(rtc_boundary, 95)))
    print("plain boundary max:", float(plain_boundary.max()))
    print("rtc boundary max:", float(rtc_boundary.max()))
    print("plain prefix p95:", float(np.percentile(plain_prefix, 95)))
    print("rtc prefix p95:", float(np.percentile(rtc_prefix, 95)))
    print("plain expert MAE:", float(plain_mae.mean()))
    print("rtc expert MAE:", float(rtc_mae.mean()))
    print("rtc latency p95 ms:", float(np.percentile(rtc_times, 95)))
    boundary_improvement = 1.0 - rtc_boundary.mean() / plain_boundary.mean()
    print("boundary improvement ratio:", float(boundary_improvement))
    pass_boundary = (
        rtc_boundary.mean() < plain_boundary.mean()
        and np.percentile(rtc_boundary, 95) <= 0.10
    )
    pass_prefix = np.percentile(rtc_prefix, 95) <= 0.10
    pass_mae = rtc_mae.mean() <= plain_mae.mean() * 1.50
    print("RTC_BOUNDARY_PASS" if pass_boundary else "RTC_BOUNDARY_FAIL")
    print("RTC_PREFIX_PASS" if pass_prefix else "RTC_PREFIX_FAIL")
    print("RTC_MAE_PASS" if pass_mae else "RTC_MAE_FAIL")
    print("CSV:", args.output_csv)
    print("SMOLVLA_RTC_DATASET_AB_SUCCESS")
    print("No CAN, camera, or robot connection was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
