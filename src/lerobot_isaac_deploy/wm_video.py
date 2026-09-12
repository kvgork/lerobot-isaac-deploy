"""Stub loaders for video world models (V-JEPA / Cosmos / GAIA).

These are placeholders. None of these architectures has a standard
real-robot-control deploy path today — they're either data engines
(Cosmos) or encoders-without-actors (V-JEPA). Real loaders land in a
future research phase. See plans/2026-05-22-wm-deploy-on-so101.md.

Each loader raises ``WMDeployNotSupported`` with a clear message so the
session ladder fails fast and points the operator at the correct path
(offline rollout via wm_rollout, OR train a LeRobot policy on the task).
"""

from __future__ import annotations

from pathlib import Path

from lerobot_isaac_deploy.wm_loader import WMDeployNotSupported


_VIDEO_WM_HINT = (
    "Video world models do not have a real-robot-control deploy path "
    "in this release. Options:\n"
    "  • train a LeRobot policy (smolvla/act/diffusion) on the same task "
    "and deploy that on the SO-101 arm.\n"
    "  • use the wm-rollout subcommand once a research-phase loader "
    "lands (tracked in plans/2026-05-22-wm-deploy-on-so101.md)."
)


def load_vjepa(checkpoint_path: Path | str):  # noqa: D401
    """Refuse: V-JEPA has no actor; encoder-only architecture."""
    raise WMDeployNotSupported(
        f"V-JEPA checkpoint at {checkpoint_path} is encoder-only and has "
        f"no actor head.\n{_VIDEO_WM_HINT}"
    )


def load_cosmos(checkpoint_path: Path | str):  # noqa: D401
    """Refuse: NVIDIA Cosmos is a data engine, not a policy."""
    raise WMDeployNotSupported(
        f"NVIDIA Cosmos checkpoint at {checkpoint_path} is a generative "
        f"data engine, not a policy.\n{_VIDEO_WM_HINT}"
    )


def load_gaia(checkpoint_path: Path | str):  # noqa: D401
    """Refuse: GAIA-style generative video WMs have no actor head."""
    raise WMDeployNotSupported(
        f"GAIA-style video WM checkpoint at {checkpoint_path} has no "
        f"actor head.\n{_VIDEO_WM_HINT}"
    )
