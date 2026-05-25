"""Tests for arm_motor_writer — SO-101 motor-write helper.

All tests use mocks — no hardware required. Covers:
  - compute_targets: action scaling, gripper separate scale, clamping
  - write_targets: correct robot.send_action() call shape
  - home_targets: correct zero-position dict sent to robot
  - read_joint_limits: calibration parsing + hardcoded fallback
  - safety: action clip, NaN rejection, cal-floor intersection, ramped_home
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from lerobot_isaac_deploy.arm_motor_writer import (
    SO101_JOINT_NAMES,
    _DEFAULT_JOINT_LIMITS_MAX,
    _DEFAULT_JOINT_LIMITS_MIN,
    compute_targets,
    home_targets,
    ramped_home,
    read_joint_limits,
    write_targets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def zero_jp() -> np.ndarray:
    """Current joint positions all at zero."""
    return np.zeros(6, dtype=np.float32)


@pytest.fixture()
def zero_action() -> np.ndarray:
    """Zero normalized action."""
    return np.zeros(6, dtype=np.float32)


@pytest.fixture()
def mock_robot() -> MagicMock:
    """Mock SO101Follower handle with send_action stub."""
    robot = MagicMock()
    robot.send_action = MagicMock()
    return robot


# ---------------------------------------------------------------------------
# compute_targets — happy path
# ---------------------------------------------------------------------------


class TestComputeTargets:
    """compute_targets: action → target mapping tests."""

    def test_zero_action_returns_current_positions(
        self, zero_jp: np.ndarray, zero_action: np.ndarray
    ) -> None:
        """Zero action leaves targets equal to current positions."""
        jp = np.array([10.0, -5.0, 20.0, -15.0, 30.0, 50.0], dtype=np.float32)
        result = compute_targets(jp, zero_action, max_step_deg=1.0)
        # Clamp is applied but the default limits bracket all values above.
        # jp[2]=20.0 is within [-10, 90]; jp[4]=30.0 within [-90, 90], etc.
        np.testing.assert_allclose(result, jp, atol=1e-5)

    def test_full_positive_action_adds_max_step_deg(
        self, zero_jp: np.ndarray
    ) -> None:
        """action=+1.0 on arm joints adds exactly max_step_deg."""
        action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
        max_step = 2.5
        result = compute_targets(zero_jp, action, max_step_deg=max_step)
        expected_arm = np.full(5, max_step, dtype=np.float32)
        np.testing.assert_allclose(result[:5], expected_arm, atol=1e-5)

    def test_full_negative_action_subtracts_max_step_deg(
        self, zero_jp: np.ndarray
    ) -> None:
        """action=-1.0 on arm joints subtracts exactly max_step_deg."""
        action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        max_step = 3.0
        result = compute_targets(zero_jp, action, max_step_deg=max_step)
        expected_arm = np.full(5, -max_step, dtype=np.float32)
        np.testing.assert_allclose(result[:5], expected_arm, atol=1e-5)

    def test_gripper_uses_separate_scale(self, zero_jp: np.ndarray) -> None:
        """Gripper (index 5) uses max_step_gripper_pct, not max_step_deg."""
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        # Start gripper at 50% to stay within [0, 100] after step
        jp = zero_jp.copy()
        jp[5] = 50.0
        result = compute_targets(
            jp, action, max_step_deg=90.0, max_step_gripper_pct=7.0
        )
        assert result[5] == pytest.approx(57.0, abs=1e-4)

    def test_gripper_negative_action_decreases(self, zero_jp: np.ndarray) -> None:
        """Gripper action=-1.0 decreases by max_step_gripper_pct."""
        jp = zero_jp.copy()
        jp[5] = 30.0
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        result = compute_targets(
            jp, action, max_step_deg=5.0, max_step_gripper_pct=5.0
        )
        assert result[5] == pytest.approx(25.0, abs=1e-4)

    def test_arm_and_gripper_simultaneously(self, zero_jp: np.ndarray) -> None:
        """Both arm and gripper joints update correctly in the same call."""
        jp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0], dtype=np.float32)
        action = np.array([1.0, -1.0, 0.5, -0.5, 0.0, 0.8], dtype=np.float32)
        max_step = 2.0
        max_grip = 10.0
        result = compute_targets(
            jp, action, max_step_deg=max_step, max_step_gripper_pct=max_grip
        )
        assert result[0] == pytest.approx(2.0, abs=1e-4)   # +1 * 2
        assert result[1] == pytest.approx(-2.0, abs=1e-4)  # -1 * 2
        assert result[2] == pytest.approx(1.0, abs=1e-4)   # +0.5 * 2
        assert result[3] == pytest.approx(-1.0, abs=1e-4)  # -0.5 * 2
        assert result[4] == pytest.approx(0.0, abs=1e-4)   # 0 * 2
        assert result[5] == pytest.approx(58.0, abs=1e-4)  # 50 + 0.8*10


# ---------------------------------------------------------------------------
# compute_targets — clamping
# ---------------------------------------------------------------------------


class TestComputeTargetsClamping:
    """compute_targets clamping to joint limits."""

    def test_default_limits_clamp_arm_over_90(self) -> None:
        """Saturated +1.0 action from 89° with max_step=5° is clamped to 90°."""
        jp = np.array([89.0, 0.0, 0.0, 0.0, 0.0, 50.0], dtype=np.float32)
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = compute_targets(jp, action, max_step_deg=5.0)
        # shoulder_pan default max = 90°; 89+5=94 → clamped to 90.
        assert result[0] == pytest.approx(90.0, abs=1e-4)

    def test_default_limits_clamp_arm_under_neg90(self) -> None:
        """Saturated -1.0 action from -89° clamped to -90°."""
        jp = np.array([-89.0, 0.0, 0.0, 0.0, 0.0, 50.0], dtype=np.float32)
        action = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = compute_targets(jp, action, max_step_deg=5.0)
        assert result[0] == pytest.approx(-90.0, abs=1e-4)

    def test_gripper_clamped_to_100_pct(self) -> None:
        """Gripper cannot exceed 100% even with large step."""
        jp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 98.0], dtype=np.float32)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        result = compute_targets(
            jp, action, max_step_deg=1.0, max_step_gripper_pct=10.0
        )
        assert result[5] == pytest.approx(100.0, abs=1e-4)

    def test_gripper_clamped_to_0_pct(self) -> None:
        """Gripper cannot go below 0%."""
        jp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        result = compute_targets(
            jp, action, max_step_deg=1.0, max_step_gripper_pct=10.0
        )
        assert result[5] == pytest.approx(0.0, abs=1e-4)

    def test_explicit_limits_override_defaults(self) -> None:
        """Custom joint_limits_min/max take precedence over defaults."""
        jp = np.zeros(6, dtype=np.float32)
        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        custom_min = np.array([-5.0, -90.0, -10.0, -90.0, -90.0, 0.0], dtype=np.float32)
        custom_max = np.array([5.0, 90.0, 90.0, 90.0, 90.0, 100.0], dtype=np.float32)
        result = compute_targets(
            jp, action, max_step_deg=10.0,
            joint_limits_min=custom_min,
            joint_limits_max=custom_max,
        )
        # 10° step but max is 5° → clamped to 5.
        assert result[0] == pytest.approx(5.0, abs=1e-4)

    def test_elbow_flex_hardcoded_min_is_neg10(self) -> None:
        """Default elbow_flex min is -10° (table-avoidance constraint)."""
        assert _DEFAULT_JOINT_LIMITS_MIN[2] == pytest.approx(-10.0, abs=1e-4)

    def test_output_dtype_is_float32(self, zero_jp: np.ndarray, zero_action: np.ndarray) -> None:
        """compute_targets always returns float32."""
        result = compute_targets(zero_jp, zero_action, max_step_deg=1.0)
        assert result.dtype == np.float32

    def test_output_shape_is_6(self, zero_jp: np.ndarray, zero_action: np.ndarray) -> None:
        """compute_targets returns shape (6,)."""
        result = compute_targets(zero_jp, zero_action, max_step_deg=1.0)
        assert result.shape == (6,)


# ---------------------------------------------------------------------------
# write_targets — robot API call shape
# ---------------------------------------------------------------------------


class TestWriteTargets:
    """write_targets: verify robot.send_action() receives correct dict."""

    def test_write_targets_calls_send_action(self, mock_robot: MagicMock) -> None:
        """write_targets calls robot.send_action() exactly once."""
        targets = np.array([10.0, -5.0, 20.0, -15.0, 30.0, 50.0], dtype=np.float32)
        write_targets(mock_robot, targets)
        mock_robot.send_action.assert_called_once()

    def test_write_targets_uses_dotpos_key_format(self, mock_robot: MagicMock) -> None:
        """send_action receives '<joint>.pos' keys, matching runner.py pattern."""
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        write_targets(mock_robot, targets)
        call_kwargs = mock_robot.send_action.call_args
        action_dict = call_kwargs[0][0]  # first positional arg
        for name in SO101_JOINT_NAMES:
            assert f"{name}.pos" in action_dict, f"missing key {name}.pos"

    def test_write_targets_values_match_input(self, mock_robot: MagicMock) -> None:
        """The values in the send_action dict match the input array exactly."""
        targets = np.array([11.1, -22.2, 33.3, -44.4, 55.5, 66.6], dtype=np.float32)
        write_targets(mock_robot, targets)
        action_dict = mock_robot.send_action.call_args[0][0]
        for i, name in enumerate(SO101_JOINT_NAMES):
            assert action_dict[f"{name}.pos"] == pytest.approx(
                float(targets[i]), abs=1e-4
            )

    def test_write_targets_all_six_joints_present(self, mock_robot: MagicMock) -> None:
        """send_action dict has exactly 6 keys, one per joint."""
        targets = np.zeros(6, dtype=np.float32)
        write_targets(mock_robot, targets)
        action_dict = mock_robot.send_action.call_args[0][0]
        assert len(action_dict) == 6

    def test_write_targets_accepts_plain_list(self, mock_robot: MagicMock) -> None:
        """write_targets coerces plain Python lists to ndarray before sending."""
        write_targets(mock_robot, [0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
        mock_robot.send_action.assert_called_once()


# ---------------------------------------------------------------------------
# home_targets — zero-position write
# ---------------------------------------------------------------------------


class TestHomeTargets:
    """home_targets: verify zero-position dict sent to robot."""

    def test_home_targets_calls_send_action_once(self, mock_robot: MagicMock) -> None:
        """home_targets calls robot.send_action() exactly once."""
        home_targets(mock_robot)
        mock_robot.send_action.assert_called_once()

    def test_home_targets_all_values_zero(self, mock_robot: MagicMock) -> None:
        """All values in the home send_action dict are 0.0."""
        home_targets(mock_robot)
        action_dict = mock_robot.send_action.call_args[0][0]
        for k, v in action_dict.items():
            assert v == pytest.approx(0.0, abs=1e-9), f"{k} should be 0.0"

    def test_home_targets_uses_dotpos_format(self, mock_robot: MagicMock) -> None:
        """home_targets uses '<joint>.pos' key format (same as write_targets)."""
        home_targets(mock_robot)
        action_dict = mock_robot.send_action.call_args[0][0]
        for name in SO101_JOINT_NAMES:
            assert f"{name}.pos" in action_dict


# ---------------------------------------------------------------------------
# read_joint_limits — calibration + fallback
# ---------------------------------------------------------------------------


class TestReadJointLimits:
    """read_joint_limits: calibration reading and fallback behaviour."""

    def _make_mock_cal_entry(self, range_min: float, range_max: float) -> MagicMock:
        """Return a mock MotorCalibration with ticks range_min / range_max."""
        c = MagicMock()
        c.range_min = range_min
        c.range_max = range_max
        return c

    def test_no_calibration_returns_hardcoded_defaults(self) -> None:
        """When robot.calibration is None/empty, defaults are returned."""
        robot = MagicMock()
        robot.calibration = None
        lo, hi = read_joint_limits(robot)
        np.testing.assert_allclose(lo, _DEFAULT_JOINT_LIMITS_MIN, atol=1e-4)
        np.testing.assert_allclose(hi, _DEFAULT_JOINT_LIMITS_MAX, atol=1e-4)

    def test_empty_calibration_dict_returns_defaults(self) -> None:
        """Empty calibration dict falls back to hardcoded defaults."""
        robot = MagicMock()
        robot.calibration = {}
        lo, hi = read_joint_limits(robot)
        np.testing.assert_allclose(lo, _DEFAULT_JOINT_LIMITS_MIN, atol=1e-4)
        np.testing.assert_allclose(hi, _DEFAULT_JOINT_LIMITS_MAX, atol=1e-4)

    def test_partial_calibration_uses_defaults_for_missing_joints(self) -> None:
        """Joints absent from calibration dict retain hardcoded defaults."""
        robot = MagicMock()
        # Only provide shoulder_pan; all others missing.
        # 4096 ticks → 360°; center at 0° = 2048 ticks.
        robot.calibration = {
            "shoulder_pan": self._make_mock_cal_entry(
                range_min=1024, range_max=3072  # ±90° from center
            )
        }
        lo, hi = read_joint_limits(robot)
        # shoulder_pan tick conversion: (1024/4096)*360-180 = -90
        assert lo[0] == pytest.approx(-90.0, abs=0.5)
        assert hi[0] == pytest.approx(90.0, abs=0.5)
        # Other joints untouched → hardcoded defaults
        assert lo[1] == pytest.approx(_DEFAULT_JOINT_LIMITS_MIN[1], abs=1e-4)
        assert hi[1] == pytest.approx(_DEFAULT_JOINT_LIMITS_MAX[1], abs=1e-4)

    def test_full_calibration_all_joints_converted(self) -> None:
        """All 6 joints converted from ticks to degrees."""
        robot = MagicMock()
        # Symmetric ±90°: 1024..3072 around centre 2048
        robot.calibration = {
            name: self._make_mock_cal_entry(1024, 3072)
            for name in SO101_JOINT_NAMES
        }
        lo, hi = read_joint_limits(robot)
        # Cal gives ±90°, but limits are intersected with hardcoded floor.
        # Most joints: lo=-90°, hi=90°. Exceptions:
        #   elbow_flex (idx 2): lo=-10° (table-avoidance floor).
        #   gripper (idx 5): lo=0% (floor), hi=90% (cal max < floor max 100%).
        expected_lo = [-90.0, -90.0, -10.0, -90.0, -90.0, 0.0]
        expected_hi = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0]
        for i in range(6):
            assert lo[i] == pytest.approx(expected_lo[i], abs=0.5), f"lo[{i}] wrong"
            assert hi[i] == pytest.approx(expected_hi[i], abs=0.5), f"hi[{i}] wrong"

    def test_returns_float32_arrays(self) -> None:
        """read_joint_limits always returns float32 arrays."""
        robot = MagicMock()
        robot.calibration = None
        lo, hi = read_joint_limits(robot)
        assert lo.dtype == np.float32
        assert hi.dtype == np.float32

    def test_returns_shape_6_arrays(self) -> None:
        """Both returned arrays have shape (6,)."""
        robot = MagicMock()
        robot.calibration = None
        lo, hi = read_joint_limits(robot)
        assert lo.shape == (6,)
        assert hi.shape == (6,)

    def test_no_range_min_max_attrs_falls_back(self) -> None:
        """Calibration entry without range_min/max attributes keeps defaults."""
        robot = MagicMock()
        bad_entry = MagicMock()
        bad_entry.range_min = None
        bad_entry.range_max = None
        robot.calibration = {"shoulder_pan": bad_entry}
        lo, hi = read_joint_limits(robot)
        # Should keep hardcoded default for shoulder_pan
        assert lo[0] == pytest.approx(_DEFAULT_JOINT_LIMITS_MIN[0], abs=1e-4)
        assert hi[0] == pytest.approx(_DEFAULT_JOINT_LIMITS_MAX[0], abs=1e-4)


# ---------------------------------------------------------------------------
# SO101_JOINT_NAMES contract
# ---------------------------------------------------------------------------


def test_so101_joint_names_has_six_entries() -> None:
    """The canonical joint list must have exactly 6 entries."""
    assert len(SO101_JOINT_NAMES) == 6


def test_so101_joint_names_canonical_order() -> None:
    """Joint names match the training-time SO-101 canonical order."""
    expected = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert SO101_JOINT_NAMES == expected


# ---------------------------------------------------------------------------
# Session integration — NotImplementedError no longer raised for dreamerv3
# ---------------------------------------------------------------------------


def test_step_execute_tight_no_longer_raises_not_implemented(tmp_path: Any) -> None:
    """step_execute_tight should NOT raise NotImplementedError for dreamerv3.

    We patch the inner loop to avoid hardware calls and verify that the
    old NotImplementedError guard is gone — i.e. the dreamerv3 branch
    is entered and dispatches to _execute_dreamerv3_loop.
    """
    import json
    from unittest.mock import patch

    from lerobot_isaac_deploy.session import DeploySession, SessionConfig

    # Build a minimal synthetic ckpt directory (dreamerv3 kind).
    ckpt = tmp_path / "dreamer-ckpt"
    ckpt.mkdir()
    (ckpt / "synthetic_marker.json").write_text(
        json.dumps({"kind": "dreamerv3", "action_dim": 6}), encoding="utf-8"
    )
    (ckpt / ".hydra").mkdir()
    (ckpt / ".hydra" / "config.yaml").write_text("placeholder: 1\n", encoding="utf-8")
    (ckpt / "ckpt_0.ckpt").write_text("fake", encoding="utf-8")

    dataset = tmp_path / "dataset"
    dataset.mkdir()

    cfg = SessionConfig(
        policy_path=ckpt,
        dataset_root=dataset,
        do_execute=True,
        clamp_tight_deg=1.0,
        duration_tight_s=0.1,
        home_on_exit=False,
        assume_yes=False,
    )
    session = DeploySession(cfg)
    session._ckpt_kind = "dreamerv3"

    # Patch _execute_dreamerv3_loop to avoid real hardware calls.
    with patch.object(session, "_execute_dreamerv3_loop") as mock_loop, \
         patch.object(session, "_confirm"):
        session.step_execute_tight()
        # Must be called exactly once with the tight parameters.
        mock_loop.assert_called_once_with(
            duration_s=cfg.duration_tight_s,
            max_step_deg=cfg.clamp_tight_deg,
            step_label="tight",
        )


def test_step_execute_loose_no_longer_raises_not_implemented(tmp_path: Any) -> None:
    """step_execute_loose should NOT raise NotImplementedError for dreamerv3."""
    import json
    from unittest.mock import patch

    from lerobot_isaac_deploy.session import DeploySession, SessionConfig

    ckpt = tmp_path / "dreamer-ckpt"
    ckpt.mkdir()
    (ckpt / "synthetic_marker.json").write_text(
        json.dumps({"kind": "dreamerv3", "action_dim": 6}), encoding="utf-8"
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    cfg = SessionConfig(
        policy_path=ckpt,
        dataset_root=dataset,
        do_execute=True,
        clamp_loose_deg=3.0,
        duration_loose_s=0.1,
        home_on_exit=False,
        assume_yes=False,
    )
    session = DeploySession(cfg)
    session._ckpt_kind = "dreamerv3"

    with patch.object(session, "_execute_dreamerv3_loop") as mock_loop, \
         patch.object(session, "_confirm"):
        session.step_execute_loose()
        mock_loop.assert_called_once_with(
            duration_s=cfg.duration_loose_s,
            max_step_deg=cfg.clamp_loose_deg,
            step_label="loose",
        )


# ---------------------------------------------------------------------------
# Safety fixes — 4 new tests (code review b4d3d0e)
# ---------------------------------------------------------------------------


def test_compute_targets_clamps_actor_above_one() -> None:
    """Saturated/pathological action >1.0 must be clipped before scaling."""
    cur = np.zeros(6, dtype=np.float32)
    bad_action = np.array([10.0, -10.0, 5.0, -3.0, 1.5, -1.5], dtype=np.float32)
    t = compute_targets(cur, bad_action, max_step_deg=3.0, max_step_gripper_pct=5.0)
    # All arm targets capped at ±3° (max_step_deg * clipped action)
    assert np.allclose(t[:5], [3.0, -3.0, 3.0, -3.0, 3.0])
    # Gripper: action -1.5 clips to -1.0, step = -1.0*5.0 = -5.0 from 0.0 = -5.0,
    # then clamped to gripper floor 0.0%. Correct result is 0.0.
    assert np.isclose(t[5], 0.0)


def test_compute_targets_rejects_nan_action() -> None:
    """NaN/inf in action must raise, not silently corrupt target."""
    cur = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        compute_targets(cur, np.array([np.nan] * 6, dtype=np.float32), max_step_deg=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        compute_targets(cur, np.array([np.inf] * 6, dtype=np.float32), max_step_deg=1.0)


def test_read_joint_limits_never_wider_than_hardcoded_floor() -> None:
    """Symmetric cal returning ±180° must not weaken the ±90° safety floor."""
    class _MockCal:
        def __init__(self, mn: int, mx: int) -> None:
            self.range_min = mn
            self.range_max = mx

    class _MockRobot:
        def __init__(self) -> None:
            # Cal that would imply ±180° if used naively (ticks 0..4095)
            self.calibration = {
                n: _MockCal(0, 4095) for n in SO101_JOINT_NAMES
            }

    lo, hi = read_joint_limits(_MockRobot())
    # Cal-derived MUST be INSIDE the hardcoded floor (subset).
    assert (lo >= _DEFAULT_JOINT_LIMITS_MIN).all(), (
        f"lo {lo} weakens floor {_DEFAULT_JOINT_LIMITS_MIN}"
    )
    assert (hi <= _DEFAULT_JOINT_LIMITS_MAX).all(), (
        f"hi {hi} weakens floor {_DEFAULT_JOINT_LIMITS_MAX}"
    )
    # elbow_flex (idx 2) must keep the -10° table-avoidance floor
    assert lo[2] >= -10.0


def test_ramped_home_terminates_when_close() -> None:
    """ramped_home should return when |jp - home| < tolerance."""
    import time as _time

    class _MockBus:
        def __init__(self) -> None:
            self.writes: list = []

    class _MockRobot:
        def __init__(self) -> None:
            self.calibration: dict = {}
            self.bus = _MockBus()
            self._calls = 0

        def get_observation(self) -> dict:
            self._calls += 1
            # Already at home pose
            return {f"{n}.pos": 0.0 for n in SO101_JOINT_NAMES}

        def send_action(self, action_dict: dict) -> None:
            self.bus.writes.append(action_dict)

    robot = _MockRobot()
    t0 = _time.monotonic()
    ramped_home(
        robot,
        np.zeros(6, dtype=np.float32),
        max_step_deg=1.0,
        rate_hz=60.0,
        settle_timeout_s=2.0,
    )
    elapsed = _time.monotonic() - t0
    # Should return immediately because already at home (no writes needed)
    assert elapsed < 0.5
    assert len(robot.bus.writes) == 0  # no step needed
