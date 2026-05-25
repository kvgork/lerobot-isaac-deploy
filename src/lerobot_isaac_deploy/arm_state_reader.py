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


def open_arm(port: str, calibration_dir: str | None = None):  # noqa: ARG001
    """Open SO-101 follower on port in read-only mode.

    Returns a robot handle that exposes:
        .get_observation()  -> dict
        .disconnect()

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

    Shape A — flat state array (lerobot 0.5.x default)::

        {"observation.state": np.ndarray (6,)}

    Shape B — per-joint keys::

        {"observation.state.shoulder_pan": float, ...}

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
    # Shape A: flat state vector (most common with lerobot 0.5)
    if "observation.state" in obs:
        arr = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        # Guard against unexpected dimensions — take first 6 or pad.
        if arr.shape[0] >= 6:
            return arr[:6]
        padded = np.zeros(6, dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded

    # Shape B: per-joint keys
    return np.array(
        [
            float(obs.get(f"observation.state.{k}", 0.0))
            for k in SO101_JOINT_NAMES
        ],
        dtype=np.float32,
    )
