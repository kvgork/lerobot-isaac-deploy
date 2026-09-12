"""Tests for lerobot_isaac_deploy.outcome_reader.

All tests are pure pytest — no hardware, no vision stack, no torch.
Injected fake predicates are used throughout so the tests pass even when
``lerobot-isaac-env`` is not installed in the test environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_isaac_deploy.outcome_reader import (
    manual_confirm,
    read_object_in_bin,
    read_task_success,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TARGET_POS = [0.22, -0.13, 0.01]
SUCCESS_RADIUS = 0.06


def _fake_predicate(object_pos, target_pos, success_radius):
    """Pure-numpy XY-distance predicate matching object_in_bin semantics."""
    obj = np.asarray(object_pos, dtype=float)
    tgt = np.asarray(target_pos, dtype=float)
    xy_dist = float(np.sqrt(((obj[:2] - tgt[:2]) ** 2).sum()))
    return xy_dist < success_radius


def _in_bin_pos():
    """Object position that is within SUCCESS_RADIUS of TARGET_POS."""
    return [TARGET_POS[0] + 0.01, TARGET_POS[1] + 0.01, 0.008]


def _out_of_bin_pos():
    """Object position clearly outside SUCCESS_RADIUS."""
    return [TARGET_POS[0] + 0.20, TARGET_POS[1], 0.008]


# ---------------------------------------------------------------------------
# 1. Injected predicate, pose source returns a valid pose
# ---------------------------------------------------------------------------


def test_in_bin_returns_true_with_injected_predicate():
    result = read_object_in_bin(
        object_pose_source=_in_bin_pos,
        target_pos=TARGET_POS,
        success_radius=SUCCESS_RADIUS,
        predicate=_fake_predicate,
    )
    assert result is True


def test_out_of_bin_returns_false_with_injected_predicate():
    result = read_object_in_bin(
        object_pose_source=_out_of_bin_pos,
        target_pos=TARGET_POS,
        success_radius=SUCCESS_RADIUS,
        predicate=_fake_predicate,
    )
    assert result is False


def test_result_is_plain_bool_not_numpy():
    """read_object_in_bin always returns a plain Python bool."""
    result = read_object_in_bin(
        object_pose_source=_in_bin_pos,
        target_pos=TARGET_POS,
        success_radius=SUCCESS_RADIUS,
        predicate=_fake_predicate,
    )
    assert type(result) is bool  # noqa: E721


# ---------------------------------------------------------------------------
# 2. Pose source returns None + manual_fallback=True
# ---------------------------------------------------------------------------


def test_none_pose_manual_fallback_yes():
    result = read_object_in_bin(
        object_pose_source=lambda: None,
        target_pos=TARGET_POS,
        predicate=_fake_predicate,
        manual_fallback=True,
        input_fn=lambda _: "yes",
    )
    assert result is True


def test_none_pose_manual_fallback_y_short():
    result = read_object_in_bin(
        object_pose_source=lambda: None,
        target_pos=TARGET_POS,
        predicate=_fake_predicate,
        manual_fallback=True,
        input_fn=lambda _: "Y",
    )
    assert result is True


def test_none_pose_manual_fallback_no():
    result = read_object_in_bin(
        object_pose_source=lambda: None,
        target_pos=TARGET_POS,
        predicate=_fake_predicate,
        manual_fallback=True,
        input_fn=lambda _: "no",
    )
    assert result is False


def test_none_pose_manual_fallback_arbitrary_string():
    result = read_object_in_bin(
        object_pose_source=lambda: None,
        target_pos=TARGET_POS,
        predicate=_fake_predicate,
        manual_fallback=True,
        input_fn=lambda _: "maybe",
    )
    assert result is False


# ---------------------------------------------------------------------------
# 3. Pose source returns None + manual_fallback=False (no stdin)
# ---------------------------------------------------------------------------


def test_none_pose_no_fallback_returns_false():
    """When pose is None and manual_fallback=False, return False without calling input."""
    called = []

    def _bad_input(_):
        called.append(True)
        return "yes"

    result = read_object_in_bin(
        object_pose_source=lambda: None,
        target_pos=TARGET_POS,
        predicate=_fake_predicate,
        manual_fallback=False,
        input_fn=_bad_input,
    )
    assert result is False
    assert called == [], "input_fn must not be called when manual_fallback=False"


# ---------------------------------------------------------------------------
# 4. Canonical predicate path (predicate=None)
# ---------------------------------------------------------------------------


def test_canonical_predicate_or_import_error():
    """
    When predicate=None:
    - If lerobot_isaac_env is importable, assert equivalence with direct call.
    - If not, assert ImportError with a helpful message.
    """
    try:
        from lerobot_isaac_env.outcome_verifier import object_in_bin as canonical
    except ImportError:
        canonical = None

    if canonical is not None:
        # Env package is available — result must match a direct call.
        pos = _in_bin_pos()
        expected = bool(canonical(pos, TARGET_POS, SUCCESS_RADIUS))
        result = read_object_in_bin(
            object_pose_source=lambda: pos,
            target_pos=TARGET_POS,
            success_radius=SUCCESS_RADIUS,
            predicate=None,  # let it load canonical
        )
        assert result == expected
    else:
        # Env package is NOT installed — must raise ImportError with a message
        # that names both remediation options.
        with pytest.raises(ImportError) as exc_info:
            read_object_in_bin(
                object_pose_source=lambda: _in_bin_pos(),
                target_pos=TARGET_POS,
                predicate=None,
            )
        msg = str(exc_info.value)
        assert "lerobot-isaac-env" in msg
        assert "predicate=" in msg


# ---------------------------------------------------------------------------
# 5. read_task_success returns 0 / 1 ints
# ---------------------------------------------------------------------------


def test_read_task_success_returns_one_on_success():
    result = read_task_success(
        object_pose_source=_in_bin_pos,
        target_pos=TARGET_POS,
        success_radius=SUCCESS_RADIUS,
        predicate=_fake_predicate,
    )
    assert result == 1
    assert type(result) is int


def test_read_task_success_returns_zero_on_failure():
    result = read_task_success(
        object_pose_source=_out_of_bin_pos,
        target_pos=TARGET_POS,
        success_radius=SUCCESS_RADIUS,
        predicate=_fake_predicate,
    )
    assert result == 0
    assert type(result) is int


def test_read_task_success_accumulates_correctly():
    results = [
        read_task_success(
            object_pose_source=src,
            target_pos=TARGET_POS,
            success_radius=SUCCESS_RADIUS,
            predicate=_fake_predicate,
        )
        for src in [_in_bin_pos, _out_of_bin_pos, _in_bin_pos]
    ]
    assert sum(results) == 2


# ---------------------------------------------------------------------------
# 6. manual_confirm standalone tests
# ---------------------------------------------------------------------------


def test_manual_confirm_yes_variations():
    for answer in ("yes", "YES", "Yes", "y", "Y"):
        assert manual_confirm(input_fn=lambda _: answer) is True


def test_manual_confirm_no_variations():
    for answer in ("no", "NO", "n", "nope", ""):
        assert manual_confirm(input_fn=lambda _: answer) is False


def test_manual_confirm_strips_whitespace():
    assert manual_confirm(input_fn=lambda _: "  yes  ") is True
    assert manual_confirm(input_fn=lambda _: "  no  ") is False


# ---------------------------------------------------------------------------
# 7. Module-level import smoke
# ---------------------------------------------------------------------------


def test_outcome_reader_importable_without_vision_or_torch():
    """Module import must succeed in a bare numpy-only environment."""
    import lerobot_isaac_deploy.outcome_reader as m

    assert callable(m.read_object_in_bin)
    assert callable(m.read_task_success)
    assert callable(m.manual_confirm)


def test_outcome_reader_all_exports():
    import lerobot_isaac_deploy.outcome_reader as m

    for name in ("manual_confirm", "read_object_in_bin", "read_task_success"):
        assert name in m.__all__
