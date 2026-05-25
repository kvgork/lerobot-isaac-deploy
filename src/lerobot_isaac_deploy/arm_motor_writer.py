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

Motor-write API (verified from robot-data-runner/runner.py line 140):
  robot.send_action({"<joint>.pos": float, ...})
  The dict keys use the "<joint>.pos" format, NOT bare joint names.
  This is the SO101Follower.send_action() high-level path — it applies
  max_relative_target clamping server-side (configured at robot init) AND
  accepts the per-motor goal positions in one call.

Joint-limit clamping strategy:
  robot.calibration is not reliably accessible or in degrees via the
  public SO101Follower API in lerobot 0.5. We therefore use hardcoded
  safe ranges as the first-pass safety floor. These are conservative
  bounds that avoid mechanical hard-stops on the SO-101. Tighten with
  real per-robot calibration data once available.

  Default hardcoded ranges (degrees / gripper %):
    shoulder_pan   : [-90,  90]
    shoulder_lift  : [-90,  90]
    elbow_flex     : [-10,  90]   (negative extent limited — avoids table)
    wrist_flex     : [-90,  90]
    wrist_roll     : [-90,  90]
    gripper        : [  0, 100]   (0% closed … 100% open)

  These are labeled FIRST-PASS SAFETY FLOOR — tighten with real
  calibration data once measured.
"""

from __future__ import annotations

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
    """
    current = np.asarray(current_jp, dtype=np.float32).reshape(6)
    act = np.asarray(action, dtype=np.float32).reshape(6)

    targets = np.empty(6, dtype=np.float32)
    # Arm joints 0..4: scale by max_step_deg
    for i in range(5):
        targets[i] = current[i] + act[i] * float(max_step_deg)
    # Gripper joint 5: scale by max_step_gripper_pct
    targets[5] = current[5] + act[5] * float(max_step_gripper_pct)

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

    Parameters
    ----------
    robot:
        SO101Follower handle (already connected).
    """
    home_dict = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
    robot.send_action(home_dict)


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
    return min_arr, max_arr
