"""Hardware-side outcome reader for SO-101 real-robot rollout verification.

Emits ``task_success`` (bool / 0-1 int) using the **same canonical predicate**
that the sim uses, so a real SO-101 rollout is evaluated with an identical RLVR
criterion.

Design contract
---------------
* **Dependency injection** — the outcome predicate and the object-pose source are
  both injected by the caller, so this module never imports a vision stack, a
  fiducial detector, or ``lerobot-isaac-env`` at the module level.
* When ``predicate`` is ``None``, the function lazily attempts
  ``from lerobot_isaac_env.outcome_verifier import object_in_bin``.  If that
  import also fails, an :class:`ImportError` is raised with a clear message
  telling the caller to either install ``lerobot-isaac-env`` or pass
  ``predicate=`` explicitly.  This keeps the deploy package decoupled by default
  (predicate injected on a laptop without the env package), while using the ONE
  canonical definition when the env package is present — preventing any drift.
* **No torch / isaaclab / vision imports** at module top-level.  This module is
  importable on a bare laptop with only ``numpy`` available.

Camera reads, motor writes, and safety monitoring all live in robot-data-runner.
See :mod:`lerobot_isaac_deploy.arm_state_reader` for joint-state reading.
For outcome verification after a rollout, see :func:`read_object_in_bin` here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ArrayLike = list | np.ndarray
PoseSource = Callable[[], ArrayLike | None]
Predicate = Callable[..., bool]


# ---------------------------------------------------------------------------
# Internal: lazy canonical predicate loader
# ---------------------------------------------------------------------------


def _load_canonical_predicate() -> Predicate:
    """Attempt to load the canonical ``object_in_bin`` predicate from the env package.

    Returns
    -------
    Callable
        The ``object_in_bin`` function from
        ``lerobot_isaac_env.outcome_verifier``.

    Raises
    ------
    ImportError
        When ``lerobot-isaac-env`` is not installed, with a message explaining
        the two remediation options.
    """
    try:
        from lerobot_isaac_env.outcome_verifier import object_in_bin  # type: ignore[import]

        return object_in_bin
    except ImportError as exc:
        raise ImportError(
            "Cannot resolve the canonical outcome predicate: "
            "'lerobot-isaac-env' is not installed in this environment. "
            "Remediation options:\n"
            "  1. Install the env package:  pip install lerobot-isaac-env\n"
            "  2. Pass a predicate explicitly:  predicate=lambda pos, tgt, r: ...\n"
            "     (see lerobot_isaac_env.outcome_verifier.object_in_bin for the signature)"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manual_confirm(
    prompt: str = "Did the object land in the target bin? [yes/no]: ",
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Scripted physical-check fallback when vision or fiducial is unavailable.

    Reads one line via *input_fn* (injected so tests never block on stdin) and
    returns ``True`` iff the answer is an affirmative (``yes`` / ``y``,
    case-insensitive).

    Parameters
    ----------
    prompt:
        Text shown to the operator.
    input_fn:
        Callable that receives the prompt string and returns the operator's
        answer as a string.  Defaults to the built-in :func:`input`.

    Returns
    -------
    bool
        ``True`` for ``"yes"`` / ``"y"`` (case-insensitive), ``False`` otherwise.
    """
    answer = input_fn(prompt).strip().lower()
    return answer in {"yes", "y"}


def read_object_in_bin(
    object_pose_source: PoseSource,
    target_pos: ArrayLike,
    success_radius: float = 0.06,
    predicate: Predicate | None = None,
    manual_fallback: bool = True,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Read whether the object is in the target bin on real hardware.

    This function is the hardware-side mirror of
    :func:`lerobot_isaac_env.outcome_verifier.object_in_bin`.  It uses the same
    canonical predicate (injected or lazily loaded) so the RLVR criterion is
    identical in sim and on real hardware.

    Parameters
    ----------
    object_pose_source:
        A **callable** ``() -> object`` that returns the current object
        position as a ``(3,)`` array-like (metres), or ``None`` when the vision
        or fiducial system is unavailable.  The D435 detector or fiducial reader
        is injected by the caller; this module never imports a vision stack.
    target_pos:
        Position of the target bin centre.  Shape ``(3,)`` or list of 3 floats.
        Units: metres.
    success_radius:
        XY Euclidean distance threshold in metres.  Default: 6 cm — matches the
        canonical sim predicate default.
    predicate:
        The outcome predicate callable.  Signature:
        ``(object_pos, target_pos, success_radius) -> bool``.
        When ``None``, the function lazily attempts
        ``from lerobot_isaac_env.outcome_verifier import object_in_bin``.
        If that import also fails, an :class:`ImportError` is raised with a
        remediation message.
    manual_fallback:
        When ``True`` (default) and *object_pose_source* returns ``None``,
        fall back to :func:`manual_confirm` to ask the operator.
        When ``False`` and the source returns ``None``, return ``False``
        immediately without calling :func:`input`.
    input_fn:
        Passed through to :func:`manual_confirm`.  Injected so tests don't
        block on stdin.

    Returns
    -------
    bool
        ``True`` iff the outcome predicate (or the operator) confirms success.

    Raises
    ------
    ImportError
        When *predicate* is ``None`` and ``lerobot-isaac-env`` is not installed.
    """
    # Resolve predicate lazily — only load env package if caller didn't inject one.
    if predicate is None:
        predicate = _load_canonical_predicate()

    object_pos = object_pose_source()

    if object_pos is None:
        if manual_fallback:
            return manual_confirm(input_fn=input_fn)
        return False

    return bool(predicate(object_pos, target_pos, success_radius))


def read_task_success(
    object_pose_source: PoseSource,
    target_pos: ArrayLike,
    **kw: object,
) -> int:
    """Thin wrapper around :func:`read_object_in_bin` that returns 0 or 1.

    Callers that accumulate a success count can use this instead of casting the
    bool result themselves.

    Parameters
    ----------
    object_pose_source:
        Forwarded to :func:`read_object_in_bin`.
    target_pos:
        Forwarded to :func:`read_object_in_bin`.
    **kw:
        Additional keyword arguments forwarded to :func:`read_object_in_bin`
        (e.g. ``success_radius``, ``predicate``, ``manual_fallback``,
        ``input_fn``).

    Returns
    -------
    int
        ``1`` on success, ``0`` on failure.
    """
    return int(read_object_in_bin(object_pose_source, target_pos, **kw))


__all__ = [
    "manual_confirm",
    "read_object_in_bin",
    "read_task_success",
]
