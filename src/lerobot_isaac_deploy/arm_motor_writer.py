"""Real SO-101 motor-write helper for closed-loop policy deploy.

Mirror of arm_state_reader.py — same robot handle, but writes joint
position TARGETS with per-step safety clamps. Used by deploy session's
step_execute_* paths.

Action mapping (normalized actor output → motor target):
  - Actor output a[i] ∈ [-1, 1] for each of the 6 joints
  - For arm joints (0..4): target = current_pos + a[i] * max_step_deg
  - For gripper (5): target = current_pos + a[5] * max_step_gripper_pct
    (gripper has its own RANGE_0_100 norm mode)
  - Targets are clamped to per-joint safe ranges before write
  - SAFETY: actions are clipped to [-1, 1] BEFORE scaling so that a
    saturated/pathological actor head (e.g. raw logits, unconverged
    DreamerV3) cannot produce unbounded per-step motion regardless of
    max_step_deg.

Motor-write API (verified from robot-data-runner/runner.py line 140):
  robot.send_action({"<joint>.pos": float, ...})
  The dict keys use the "<joint>.pos" format, NOT bare joint names.
  This is the SO101Follower.send_action() high-level path — it applies
  max_relative_target clamping server-side (configured at robot init) AND
  accepts the per-motor goal positions in one call.

Joint-limit clamping strategy (two-layer):
  Layer 1 — action clip: action is clipped to [-1, 1] before scaling so
    per-step motion is bounded by max_step_deg regardless of upstream bugs.
  Layer 2 — cal-derived limits intersected with hardcoded floor: the
    calibration-derived limits are constrained to be a STRICT SUBSET of
    the hardcoded safety floor (_DEFAULT_JOINT_LIMITS_MIN/MAX). A symmetric
    calibration (ticks 0..4095 → ±180°) can NEVER widen the safe range
    beyond the hardcoded floor. The elbow_flex -10° table-avoidance floor
    is preserved even when calibration returns ±90°.

  Default hardcoded ranges (degrees / gripper %):
    shoulder_pan   : [-90,  90]
    shoulder_lift  : [-90,  90]
    elbow_flex     : [-10,  90]   (negative extent limited — avoids table)
    wrist_flex     : [-90,  90]
    wrist_roll     : [-90,  90]
    gripper        : [  0, 100]   (0% closed … 100% open)

  These are labeled FIRST-PASS SAFETY FLOOR — tighten with real
  calibration data once measured.

Ramped home:
  home-on-exit uses ramped_home() rather than the old single-shot
  home_targets() so the arm never receives an instant goto from an
  arbitrary pose — which risks high-velocity slam if the server-side
  max_relative_target is not independently clamped.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np

# Canonical SO-101 joint order — matches arm_state_reader.SO101_JOINT_NAMES
# and the training-time joint ordering.
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# FIRST-PASS SAFETY FLOOR — conservative per-joint bounds.
# Arm joints (0..4) in degrees. Gripper (5) in RANGE_0_100 (%).
# Tighten with real calibration data once available.
# Cal-derived limits from read_joint_limits() are always intersected
# with these values — they can only be TIGHTER, never wider.
_DEFAULT_JOINT_LIMITS_MIN = np.array(
    [-90.0, -90.0, -10.0, -90.0, -90.0, 0.0], dtype=np.float32
)
_DEFAULT_JOINT_LIMITS_MAX = np.array(
    [90.0, 90.0, 90.0, 90.0, 90.0, 100.0], dtype=np.float32
)


def compute_targets(
    current_jp: np.ndarray,
    action: np.ndarray,
    max_step_deg: float,
    max_step_gripper_pct: float = 5.0,
    joint_limits_min: np.ndarray | None = None,
    joint_limits_max: np.ndarray | None = None,
) -> np.ndarray:
    """Map a normalized [-1, 1] actor action to a clamped joint target.

    Computes:
        target[i] = current[i] + action[i] * max_step  (per-joint)

    Then clamps to [joint_min[i], joint_max[i]].

    SAFETY: action is clipped to [-1, 1] BEFORE scaling. If the actor
    head ever emits raw logits, NaN, or values outside [-1, 1], the clip
    ensures per-step motion is bounded by max_step_deg. NaN / inf values
    raise ValueError immediately (do not silently corrupt the target).

    Parameters
    ----------
    current_jp:
        Shape (6,) float32. Current joint positions in canonical order.
        Indices 0..4 in degrees; index 5 in gripper % (RANGE_0_100).
    action:
        Shape (6,) float32. Normalized actor output in [-1, 1].
    max_step_deg:
        Maximum step size in degrees for arm joints (0..4) per control
        cycle. E.g. 1.0 for tight execute, 3.0 for loose execute.
    max_step_gripper_pct:
        Maximum step size for gripper joint (index 5) in % per cycle.
        Default 5.0%.
    joint_limits_min:
        Shape (6,) float32. Lower bound per joint. If None, uses the
        hardcoded default safety floor.
    joint_limits_max:
        Shape (6,) float32. Upper bound per joint. If None, uses the
        hardcoded default safety floor.

    Returns
    -------
    np.ndarray
        Shape (6,) float32. Clamped joint position targets.

    Raises
    ------
    ValueError
        When action contains NaN or infinite values.
    """
    current = np.asarray(current_jp, dtype=np.float32).reshape(6)
    action = np.asarray(action, dtype=np.float32)
    # Pathology guard: if the actor head ever emits raw logits or NaN,
    # per-step motion is unbounded. Clip BEFORE scaling so the per-step
    # bound is mathematically guaranteed regardless of upstream bugs.
    if not np.isfinite(action).all():
        raise ValueError(f"action contains non-finite values: {action!r}")
    action = np.clip(action, -1.0, 1.0).reshape(6)

    targets = np.empty(6, dtype=np.float32)
    # Arm joints 0..4: scale by max_step_deg
    for i in range(5):
        targets[i] = current[i] + action[i] * float(max_step_deg)
    # Gripper joint 5: scale by max_step_gripper_pct
    targets[5] = current[5] + action[5] * float(max_step_gripper_pct)

    # Clamp to joint limits (fall back to hardcoded defaults)
    lo = (
        np.asarray(joint_limits_min, dtype=np.float32)
        if joint_limits_min is not None
        else _DEFAULT_JOINT_LIMITS_MIN
    )
    hi = (
        np.asarray(joint_limits_max, dtype=np.float32)
        if joint_limits_max is not None
        else _DEFAULT_JOINT_LIMITS_MAX
    )
    targets = np.clip(targets, lo, hi)
    return targets


def write_targets(
    robot: Any,
    targets_deg: np.ndarray,
) -> None:
    """Write joint position targets to the motor bus via robot.send_action().

    Uses the ``"<joint>.pos"`` key format that SO101Follower.send_action()
    expects (verified from robot-data-runner/runner.py line 140-144 and
    home-on-exit pattern at line 154).

    SO101Follower.send_action() applies max_relative_target clamping
    server-side (set when the robot was constructed in arm_state_reader.
    open_arm). The targets passed here are already clamped to joint limits
    by compute_targets() — two independent safety layers.

    Parameters
    ----------
    robot:
        SO101Follower handle (already connected + calibrated). Must expose
        ``send_action(dict[str, float])``.
    targets_deg:
        Shape (6,) float array. Indices 0..4 in degrees (arm joints).
        Index 5 in gripper % (RANGE_0_100). Must already be clamped to
        safe range (call compute_targets() before this function).
    """
    t = np.asarray(targets_deg, dtype=np.float32).reshape(6)
    action_dict = {
        f"{name}.pos": float(t[i])
        for i, name in enumerate(SO101_JOINT_NAMES)
    }
    robot.send_action(action_dict)


def home_targets(
    robot: Any,
) -> None:
    """Send a zero-position target to all joints (home pose).

    Mirrors the robot-data-runner home-on-exit pattern:
        ``robot.send_action({f"{m}.pos": 0.0 for m in motor_names})``

    Sends 0.0 for all 6 joints. For the arm (degrees) this is the
    upright neutral pose. For the gripper this is fully closed — safe
    resting position between runs.

    .. deprecated::
        Prefer :func:`ramped_home` for deploy sessions. ``home_targets``
        sends an instant goto which risks high-velocity slam from an
        arbitrary pose if the server-side max_relative_target is not
        independently clamped. Kept for backwards compat and testing.

    Parameters
    ----------
    robot:
        SO101Follower handle (already connected).
    """
    home_dict = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
    robot.send_action(home_dict)


def ramped_home(
    robot: Any,
    current_jp: np.ndarray,
    home_pose: np.ndarray | None = None,
    max_step_deg: float = 1.0,
    rate_hz: float = 30.0,
    settle_timeout_s: float = 5.0,
) -> None:
    """Ramp arm from current_jp to home_pose (default zeros) at max_step_deg per step.

    Each iteration computes a target one step closer to home along each
    joint, sleeps 1/rate_hz, then re-reads joint state to detect arrival
    or stall. Terminates when |jp - home| < tolerance for all joints OR
    settle_timeout_s elapses.

    This replaces the prior single-shot home_targets(0.0) which sent an
    instant goto target from arbitrary pose — risk of high-velocity slam
    if motors aren't independently clamped server-side.

    Parameters
    ----------
    robot:
        SO101Follower handle (already connected). Must expose
        ``get_observation() -> dict`` and ``send_action(dict)``.
    current_jp:
        Shape (6,) float32. Current joint positions at the start of the
        ramp. Used as the initial position estimate — subsequent steps
        re-read from the robot.
    home_pose:
        Shape (6,) float32. Target home pose in degrees (arm) / % (gripper).
        Defaults to zeros (all joints at neutral / gripper closed).
    max_step_deg:
        Maximum per-step delta in degrees for arm joints (0..4).
        Gripper steps are capped at 5.0%/step regardless of this value.
    rate_hz:
        Control loop frequency in Hz. Default 30 Hz.
    settle_timeout_s:
        Maximum wall-clock seconds to wait before giving up. Default 5 s.
    """
    from lerobot_isaac_deploy.arm_state_reader import _extract_joint_pos

    if home_pose is None:
        home_pose = np.zeros(6, dtype=np.float32)
    home_pose = np.asarray(home_pose, dtype=np.float32).reshape(6)

    dt = 1.0 / rate_hz
    t_end = _time.monotonic() + settle_timeout_s

    while _time.monotonic() < t_end:
        # Re-read current state.
        obs = robot.get_observation()
        jp = _extract_joint_pos(obs)

        delta = home_pose - jp
        # Settle check: tighter tol for arm joints, looser for gripper.
        if (np.abs(delta[:5]) < 2.0).all() and abs(delta[5]) < 5.0:
            return

        # Step toward home, capped at max_step_deg per joint (arm), 5%/step gripper.
        step = np.clip(delta[:5], -max_step_deg, max_step_deg)
        gripper_step = np.clip(delta[5], -5.0, 5.0)
        target = np.zeros(6, dtype=np.float32)
        target[:5] = jp[:5] + step
        target[5] = jp[5] + gripper_step

        # Write target (no exception propagation — best-effort).
        try:
            write_targets(robot, target)
        except Exception:  # noqa: BLE001
            pass

        _time.sleep(dt)


def read_joint_limits(robot: Any) -> tuple[np.ndarray, np.ndarray]:
    """Attempt to read joint-position calibration ranges from the robot.

    Returns (min_deg, max_deg) arrays shape (6,) float32.

    SO101Follower in lerobot 0.5 exposes ``robot.calibration`` as a
    dict[motor_name, MotorCalibration] where MotorCalibration has
    ``range_min`` and ``range_max`` in raw motor ticks. The tick-to-degree
    conversion is: degrees = (ticks / 4096) * 360 − 180.

    NOTE: this conversion is an approximation — the actual mapping may
    differ per motor depending on homing_offset. If calibration is
    unavailable or incomplete, falls back to the module-level hardcoded
    safety floor (_DEFAULT_JOINT_LIMITS_MIN / _MAX) and logs a warning
    to stdout. Callers MUST treat the returned limits as the minimum
    safety constraint and may tighten them further.

    SAFETY: the cal-derived limits are INTERSECTED with the hardcoded
    floor after conversion. A symmetric calibration (ticks 0..4095 →
    ±180°) MUST NOT widen the safe range beyond the floor. The
    elbow_flex -10° table-avoidance floor is always preserved.

    Parameters
    ----------
    robot:
        SO101Follower handle (already connected).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (min_arr, max_arr) shape (6,) float32. Always returns valid arrays —
        never raises.
    """
    cal = getattr(robot, "calibration", None)
    if not cal:
        # No calibration data — use hardcoded safety floor.
        return _DEFAULT_JOINT_LIMITS_MIN.copy(), _DEFAULT_JOINT_LIMITS_MAX.copy()

    min_arr = _DEFAULT_JOINT_LIMITS_MIN.copy()
    max_arr = _DEFAULT_JOINT_LIMITS_MAX.copy()
    for i, name in enumerate(SO101_JOINT_NAMES):
        c = cal.get(name)
        if c is None:
            # No entry for this joint — keep hardcoded default.
            continue
        range_min = getattr(c, "range_min", None)
        range_max = getattr(c, "range_max", None)
        if range_min is not None and range_max is not None:
            try:
                # Approximate tick→degree: 4096 ticks = 360°, centered at 0.
                min_arr[i] = float(range_min) / 4096.0 * 360.0 - 180.0
                max_arr[i] = float(range_max) / 4096.0 * 360.0 - 180.0
            except (TypeError, ValueError):
                # Conversion failed — keep hardcoded default for this joint.
                pass

    # Make cal-derived limits a STRICT subset of the hardcoded floor.
    # A symmetric cal (ticks 0..4095 → ±180°) MUST NOT widen the safe range.
    # Also preserves the elbow_flex -10° table-avoidance floor when cal
    # returns symmetric ranges.
    min_arr = np.maximum(min_arr, _DEFAULT_JOINT_LIMITS_MIN)
    max_arr = np.minimum(max_arr, _DEFAULT_JOINT_LIMITS_MAX)
    return min_arr, max_arr
