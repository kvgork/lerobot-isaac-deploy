"""Tests for arm_state_reader — real SO-101 joint-state read helpers.

All tests use mocks — no hardware required.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from lerobot_isaac_deploy.arm_state_reader import (
    SO101_JOINT_NAMES,
    _extract_joint_pos,
    open_arm,
    stream_joint_pos,
)


# ---------------------------------------------------------------------------
# _extract_joint_pos — unit tests for both obs dict shapes
# ---------------------------------------------------------------------------


class TestExtractJointPos:
    """Tests for _extract_joint_pos — the obs dict → 6-array converter."""

    def test_shape_a_flat_state_array(self) -> None:
        """Shape A: {'observation.state': ndarray(6,)} returns correct array."""
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        obs = {"observation.state": expected.copy()}
        result = _extract_joint_pos(obs)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float32
        assert result.shape == (6,)

    def test_shape_a_larger_state_truncates_to_6(self) -> None:
        """Shape A with >6 elements returns only the first 6."""
        big = np.arange(10, dtype=np.float32)
        obs = {"observation.state": big}
        result = _extract_joint_pos(obs)
        assert result.shape == (6,)
        np.testing.assert_array_equal(result, big[:6])

    def test_shape_a_smaller_state_pads_with_zeros(self) -> None:
        """Shape A with <6 elements pads the remainder with zeros."""
        small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        obs = {"observation.state": small}
        result = _extract_joint_pos(obs)
        assert result.shape == (6,)
        np.testing.assert_array_equal(result[:3], small)
        np.testing.assert_array_equal(result[3:], np.zeros(3))

    def test_shape_b_per_joint_keys(self) -> None:
        """Shape B: per-joint keys produce correct 6-array in canonical order."""
        obs = {
            "observation.state.shoulder_pan": 0.1,
            "observation.state.shoulder_lift": 0.2,
            "observation.state.elbow_flex": 0.3,
            "observation.state.wrist_flex": 0.4,
            "observation.state.wrist_roll": 0.5,
            "observation.state.gripper": 0.6,
        }
        result = _extract_joint_pos(obs)
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result.dtype == np.float32
        assert result.shape == (6,)

    def test_shape_b_missing_keys_default_to_zero(self) -> None:
        """Shape B with partial keys fills missing joints with 0.0."""
        obs = {
            "observation.state.shoulder_pan": 1.0,
            # all others missing
        }
        result = _extract_joint_pos(obs)
        assert result.shape == (6,)
        assert result[0] == pytest.approx(1.0)
        assert result[1:].tolist() == [0.0] * 5

    def test_shape_b_canonical_joint_order(self) -> None:
        """Shape B respects SO101_JOINT_NAMES ordering."""
        obs = {f"observation.state.{k}": float(i) for i, k in enumerate(SO101_JOINT_NAMES)}
        result = _extract_joint_pos(obs)
        expected = np.arange(len(SO101_JOINT_NAMES), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_shape_a_takes_priority_over_shape_b(self) -> None:
        """When both shape A and B keys are present, shape A wins."""
        obs = {
            "observation.state": np.array([9.0, 9.0, 9.0, 9.0, 9.0, 9.0], dtype=np.float32),
            "observation.state.shoulder_pan": 1.0,
        }
        result = _extract_joint_pos(obs)
        np.testing.assert_array_equal(result, np.full(6, 9.0, dtype=np.float32))

    def test_empty_obs_returns_zeros(self) -> None:
        """Empty obs dict (no recognized keys) returns all-zeros array."""
        result = _extract_joint_pos({})
        assert result.shape == (6,)
        np.testing.assert_array_equal(result, np.zeros(6))


# ---------------------------------------------------------------------------
# stream_joint_pos — timing + iteration tests using a fake robot
# ---------------------------------------------------------------------------


def _make_fake_robot(obs_sequence: list[dict]) -> Any:
    """Create a mock robot handle with a predetermined sequence of obs dicts."""
    robot = MagicMock()
    robot.get_observation.side_effect = obs_sequence
    return robot


class TestStreamJointPos:
    """Tests for stream_joint_pos — generator over robot observations."""

    def test_yields_correct_number_of_steps(self) -> None:
        """stream_joint_pos yields at most rate_hz * duration_s steps."""
        rate_hz = 10.0
        duration_s = 0.5
        # Enough obs to saturate the loop
        obs_list = [{"observation.state": np.zeros(6, dtype=np.float32)}] * 100
        robot = _make_fake_robot(obs_list)

        results = list(
            stream_joint_pos(robot, rate_hz=rate_hz, duration_s=duration_s)
        )
        # Should have roughly rate_hz * duration_s = 5 steps (allow 1 extra
        # due to timing jitter in the test runner).
        assert len(results) >= 4
        assert len(results) <= 7

    def test_yields_ndarray_shape_6(self) -> None:
        """Each yielded value is a (6,) float32 ndarray."""
        obs = {"observation.state": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)}
        robot = _make_fake_robot([obs] * 50)

        step = next(iter(stream_joint_pos(robot, rate_hz=100.0, duration_s=0.05)))
        assert isinstance(step, np.ndarray)
        assert step.shape == (6,)
        assert step.dtype == np.float32

    def test_yields_correct_joint_values(self) -> None:
        """Values from the fake robot are passed through unchanged."""
        vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        obs = {"observation.state": vals.copy()}
        robot = _make_fake_robot([obs] * 20)

        first = next(iter(stream_joint_pos(robot, rate_hz=100.0, duration_s=0.05)))
        np.testing.assert_allclose(first, vals, atol=1e-6)

    def test_zero_duration_yields_nothing(self) -> None:
        """duration_s=0 yields no observations."""
        robot = _make_fake_robot([{"observation.state": np.zeros(6)}] * 10)
        results = list(stream_joint_pos(robot, rate_hz=10.0, duration_s=0.0))
        assert len(results) == 0

    def test_respects_rate_hz_wall_time(self) -> None:
        """Wall-clock elapsed should be approximately duration_s."""
        rate_hz = 20.0
        duration_s = 0.2
        obs = {"observation.state": np.zeros(6, dtype=np.float32)}
        robot = _make_fake_robot([obs] * 200)

        t0 = time.monotonic()
        list(stream_joint_pos(robot, rate_hz=rate_hz, duration_s=duration_s))
        elapsed = time.monotonic() - t0

        # Allow 50 ms tolerance for test-runner overhead.
        assert elapsed >= duration_s * 0.9
        assert elapsed <= duration_s + 0.05 + duration_s * 0.5

    def test_per_joint_obs_shape_also_works(self) -> None:
        """Shape B obs dicts are correctly forwarded through the generator."""
        obs = {f"observation.state.{k}": float(i) for i, k in enumerate(SO101_JOINT_NAMES)}
        robot = _make_fake_robot([obs] * 20)

        first = next(iter(stream_joint_pos(robot, rate_hz=100.0, duration_s=0.05)))
        expected = np.arange(6, dtype=np.float32)
        np.testing.assert_allclose(first, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# open_arm — import-error path (no lerobot installed)
# ---------------------------------------------------------------------------


class TestOpenArm:
    """Tests for open_arm error paths — no real hardware needed."""

    def test_open_arm_raises_import_error_when_lerobot_missing(self) -> None:
        """open_arm raises ImportError with an actionable message when lerobot
        is not importable (simulated via sys.modules manipulation)."""
        import sys

        # Temporarily hide lerobot from the import system.
        saved = sys.modules.get("lerobot.robots.so_follower")
        sys.modules["lerobot.robots.so_follower"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="lerobot >= 0.5 is required"):
                open_arm("/dev/null")
        finally:
            if saved is None:
                sys.modules.pop("lerobot.robots.so_follower", None)
            else:
                sys.modules["lerobot.robots.so_follower"] = saved

    def test_open_arm_propagates_connection_error(self) -> None:
        """open_arm propagates serial / runtime errors from robot.connect()."""
        mock_robot = MagicMock()
        mock_robot.connect.side_effect = RuntimeError("cannot open port /dev/null")

        mock_cfg_cls = MagicMock(return_value=MagicMock())
        mock_robot_cls = MagicMock(return_value=mock_robot)

        import sys
        mock_module = MagicMock()
        mock_module.SO101Follower = mock_robot_cls
        mock_module.SO101FollowerConfig = mock_cfg_cls

        saved = sys.modules.get("lerobot.robots.so_follower")
        sys.modules["lerobot.robots.so_follower"] = mock_module
        try:
            with pytest.raises(RuntimeError, match="cannot open port"):
                open_arm("/dev/null")
        finally:
            if saved is None:
                sys.modules.pop("lerobot.robots.so_follower", None)
            else:
                sys.modules["lerobot.robots.so_follower"] = saved


# ---------------------------------------------------------------------------
# SO101_JOINT_NAMES contract
# ---------------------------------------------------------------------------


def test_so101_joint_names_has_six_entries() -> None:
    """The canonical joint list must always have exactly 6 entries."""
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


def test_extract_joint_pos_shape_c_joint_pos_keys():
    """lerobot 0.5+ SO101Follower.get_observation() returns
    {'<joint>.pos': float, ...} — the wm dry-run must read these."""
    from lerobot_isaac_deploy.arm_state_reader import _extract_joint_pos, SO101_JOINT_NAMES

    # Realistic observation as returned by SO101Follower.get_observation().
    obs = {f"{name}.pos": float(i) for i, name in enumerate(SO101_JOINT_NAMES)}
    # Plus a camera key that should be ignored.
    obs["wrist.image"] = "dummy-image-tensor"
    arr = _extract_joint_pos(obs)
    assert arr.shape == (6,)
    assert arr.dtype.name == "float32"
    assert arr.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
