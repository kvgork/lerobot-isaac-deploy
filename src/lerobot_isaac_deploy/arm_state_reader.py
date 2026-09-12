"""Real SO-101 joint-state reader. NO motor writes — read-only.

Wraps lerobot's SO101Follower so the deploy session can read joint_pos
at rate_hz without going through the robot-data-run subprocess. Used for
DreamerV3-actor dry-run paths where the policy needs real joint obs but
no motor writes occur.

Lerobot import path (verified from robot-data-runner/runner.py):
    lerobot.robots.so_follower.SO101Follower
    lerobot.robots.so_follower.SO101FollowerConfig

This module is intentionally tiny — it contains only the arm-state read
path. Camera reads, motor writes, and safety monitoring all live in
robot-data-runner.

For outcome verification after a real rollout (did the object land in the
bin?), see :mod:`lerobot_isaac_deploy.outcome_reader`.
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

# Canonical SO-101 joint order — matches the training SO101_JOINT_NAMES and
# the lerobot 0.5 default motor ordering for the so_follower.
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def open_arm(
    port: str,
    calibration_dir: str | None = None,  # noqa: ARG001
    max_relative_target: float = 5.0,
):
    """Open SO-101 follower on port in read-only mode.

    Returns a robot handle that exposes:
        .get_observation()  -> dict
        .disconnect()
        .send_action(dict)  (used by motor-write paths)

    The robot is connected with ``calibrate=False`` — never recalibrate
    from a deploy session; calibration belongs in a dedicated setup flow.

    Parameters
    ----------
    port:
        Serial port, e.g. ``/dev/ttyACM0``.
    calibration_dir:
        Ignored (accepted for API symmetry with future callers). The
        SO101FollowerConfig does not accept an external calibration_dir
        in lerobot 0.5 — calibration is embedded in the port config.
    max_relative_target:
        Server-side per-step position clamp in degrees (arm joints) or
        % (gripper). SO101Follower refuses any single Goal_Position that
        is more than this value away from the current position. This is a
        second-layer safety guard complementing the client-side clamp in
        arm_motor_writer.compute_targets(). Set to the same value as
        max_step_deg in the deploy loop so both layers agree.
        Default 5.0 is conservative for initial wiring validation.

    Raises
    ------
    ImportError
        When lerobot is not installed in the active environment.
    RuntimeError / serial.SerialException
        When the port cannot be opened (arm not plugged in, wrong path).
    """
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise ImportError(
            "lerobot >= 0.5 is required to open an SO-101 follower. "
            "Install with: pip install lerobot"
        ) from exc

    # No camera config — read-only joint-pos loop, no vision needed.
    follower_cfg = SO101FollowerConfig(
        port=port,
        id="so101",
        max_relative_target=max_relative_target,
    )
    robot = SO101Follower(follower_cfg)
    robot.connect(calibrate=False)
    return robot


def stream_joint_pos(
    robot,
    rate_hz: float = 30.0,
    duration_s: float = 30.0,
) -> Iterator[np.ndarray]:
    """Yield joint_pos arrays at rate_hz for duration_s. Read-only.

    Does NOT send motor targets. Timing is best-effort wall-clock: if a
    ``get_observation()`` call takes longer than ``1/rate_hz``, the next
    iteration starts immediately (no backpressure).

    Parameters
    ----------
    robot:
        Handle returned by :func:`open_arm`.
    rate_hz:
        Target poll frequency in Hz.
    duration_s:
        Total observation window duration in seconds.

    Yields
    ------
    np.ndarray
        Shape ``(6,)`` float32 joint positions in canonical SO-101 order:
        shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
        gripper.
    """
    dt = 1.0 / rate_hz if rate_hz > 0 else 0.0
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        t0 = time.monotonic()
        raw = robot.get_observation()
        jp = _extract_joint_pos(raw)
        yield jp
        elapsed = time.monotonic() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


def _extract_joint_pos(obs: dict) -> np.ndarray:
    """SO-101 obs dict → 6-dim joint_pos in canonical order.

    Handles two observation dict shapes produced by different lerobot
    versions / robot configurations:

    Shape A — flat state array (older lerobot)::

        {"observation.state": np.ndarray (6,)}

    Shape B — observation.state.<joint> keys::

        {"observation.state.shoulder_pan": float, ...}

    Shape C — <joint>.pos keys (lerobot 0.5+ SO101Follower.get_observation())::

        {"shoulder_pan.pos": float, "shoulder_lift.pos": float, ...}

    Falls back to zeros for missing keys so the caller always receives a
    valid array (the WM actor will emit garbage actions, but no crash).

    Parameters
    ----------
    obs:
        Raw dict from ``robot.get_observation()``.

    Returns
    -------
    np.ndarray
        Shape ``(6,)`` float32.
    """
    # Shape A: flat state vector
    if "observation.state" in obs:
        arr = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        # Guard against unexpected dimensions — take first 6 or pad.
        if arr.shape[0] >= 6:
            return arr[:6]
        padded = np.zeros(6, dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded

    # Shape C: lerobot 0.5+ format — <joint>.pos keys
    if any(f"{k}.pos" in obs for k in SO101_JOINT_NAMES):
        return np.array(
            [float(obs.get(f"{k}.pos", 0.0)) for k in SO101_JOINT_NAMES],
            dtype=np.float32,
        )

    # Shape B: per-joint keys with observation.state prefix
    return np.array(
        [
            float(obs.get(f"observation.state.{k}", 0.0))
            for k in SO101_JOINT_NAMES
        ],
        dtype=np.float32,
    )
