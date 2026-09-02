"""Pure, importable core of the authority-blend hybrid controller.

Contains no torch/lerobot/hardware imports so it can be unit-tested offline on
any machine.  ``act_smolvla_hybrid_towel_blend.py`` and
``test_blend_offline.py`` share exactly this code path.

Design: target_arm = act_arm + authority * (smol_ema_arm - act_arm),
authority in [0, max_authority].  The SmolVLA correction is bounded in
magnitude and per-step slew; gripper is always owned by ACT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


def positions(obs: dict) -> np.ndarray:
    return np.asarray([float(obs[k]) for k in ACTION_NAMES], dtype=np.float32)


def action_dict(action: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(ACTION_NAMES, action, strict=True)}


def max_parts(delta: np.ndarray) -> tuple[float, float]:
    return float(np.max(delta[ARM])), float(np.max(delta[GRIPPER]))


def validate_absolute(action: np.ndarray) -> None:
    bad = np.flatnonzero((action < LOWER - 1e-6) | (action > UPPER + 1e-6))
    if bad.size:
        i = int(bad[0])
        raise RuntimeError(
            f"Safety stop: action[{i}]={action[i]:.6f} outside "
            f"[{LOWER[i]:.6f}, {UPPER[i]:.6f}]"
        )


def smol_trust(
    raw_candidate: np.ndarray,
    candidate: np.ndarray,
    act_reference: np.ndarray,
    previous: np.ndarray,
    a,
) -> tuple[bool, str, float]:
    """Return (trust_grows, reason, filtered_disagreement).

    Hard sanity first (absolute limits, gross step / ACT divergence) - a failure
    here decays authority fast.  Otherwise a soft trust gate decides whether
    SmolVLA is coherent enough to grow authority.
    """
    try:
        validate_absolute(raw_candidate)
        validate_absolute(candidate)
    except RuntimeError as exc:
        return False, f"hard:{exc}", float("inf")
    raw_step_joint, _ = max_parts(np.abs(raw_candidate - previous))
    raw_disagree_joint, _ = max_parts(np.abs(raw_candidate - act_reference))
    if raw_step_joint > a.smol_raw_step_limit:
        return False, f"hard:raw_step={raw_step_joint:.6f}", float("inf")
    if raw_disagree_joint > a.smol_raw_policy_disagreement:
        return False, f"hard:raw_guard_disagreement={raw_disagree_joint:.6f}", float("inf")
    disagree_joint = float(np.max(np.abs(candidate[ARM] - act_reference[ARM])))
    step_joint = float(np.max(np.abs(candidate[ARM] - previous[ARM])))
    if disagree_joint > a.policy_disagreement:
        return False, f"soft:guard_disagreement={disagree_joint:.6f}", disagree_joint
    if step_joint > a.joint_step_limit:
        return False, f"soft:step={step_joint:.6f}", disagree_joint
    return True, "accepted", disagree_joint


@dataclass
class BlendResult:
    source: str
    trusted_act: bool
    authority: float
    correction: np.ndarray
    detail: str
    accepted: int
    rejected: int
    fallback: int


def blend_step(
    act_action: np.ndarray,
    raw_candidate: np.ndarray | None,
    previous: np.ndarray,
    correction: np.ndarray,
    authority: float,
    a,
) -> BlendResult:
    """Advance one SmolVLA-blend tick.

    raw_candidate is None when the RTC queue is empty.  Returns the new
    source/trusted_act/authority/correction and the counters that changed.
    ``correction`` is mutated in place (the caller's copy).
    """
    source = "ACT"
    trusted_act = True
    detail = "primary"
    accepted = rejected = fallback = 0

    if raw_candidate is None:
        fallback = 1
        authority *= a.authority_decay
        _decay_correction(correction, a)
        detail = f"queue empty; authority={authority:.2f}"
    else:
        candidate = (
            previous
            + a.smol_lowpass_alpha * (raw_candidate - previous)
        ).astype(np.float32, copy=False)
        trust, reason, disagree = smol_trust(
            raw_candidate, candidate, act_action, previous, a
        )
        if trust:
            authority += a.authority_up_rate * (a.max_authority - authority)
            desired = np.clip(
                authority * (candidate[ARM] - act_action[ARM]),
                -a.smol_correction_limit,
                a.smol_correction_limit,
            )
            _apply_desired(correction, desired, a)
            accepted = 1
            source = "SMOL_BLEND"
            trusted_act = False
            detail = f"authority={authority:.2f} disagree={disagree:.4f}"
        else:
            rejected = 1
            fallback = 1
            if reason.startswith("hard:"):
                authority *= a.authority_hard_decay
            else:
                authority *= a.authority_decay
            _decay_correction(correction, a)
            detail = f"rejected:{reason}; authority={authority:.2f}"

    return BlendResult(
        source=source,
        trusted_act=trusted_act,
        authority=float(authority),
        correction=correction,
        detail=detail,
        accepted=accepted,
        rejected=rejected,
        fallback=fallback,
    )


def _decay_correction(correction: np.ndarray, a) -> None:
    """Slew the correction back to zero (no jump) when SmolVLA loses trust."""
    _apply_desired(correction, np.zeros(12, dtype=np.float32), a)


def _apply_desired(
    correction: np.ndarray,
    desired: np.ndarray,
    a,
) -> None:
    """Slew-bounded EMA of the correction toward its desired value.

    Guarantees |correction change per step| <= correction_step_limit and
    |correction| <= smol_correction_limit, so the SmolVLA influence can never
    jump or push the arm far from the trusted ACT base.
    """
    desired = np.clip(desired, -a.smol_correction_limit, a.smol_correction_limit)
    target_correction = correction[ARM] + a.correction_alpha * (
        desired - correction[ARM]
    )
    delta = np.clip(
        target_correction - correction[ARM],
        -a.correction_step_limit,
        a.correction_step_limit,
    )
    correction[ARM] = correction[ARM] + delta
    correction[ARM] = np.clip(
        correction[ARM],
        -a.smol_correction_limit,
        a.smol_correction_limit,
    )


def apply_global_smoothing(
    raw_target: np.ndarray,
    previous: np.ndarray,
    act_gripper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Final command EMA across all dims, then re-assert the ACT gripper.

    ``act_gripper`` must be the 2-element gripper slice (act_action[GRIPPER]).
    """
    target = (
        previous + alpha * (raw_target - previous)
    ).astype(np.float32, copy=False)
    target[GRIPPER] = act_gripper
    return target
