"""
test_scene_default.py
=====================
IsaacSceneSession resolves the bundled scene USD from lerobot-isaac-configs
when ``usd_path=None`` (no reliance on the gitignored workspace assets/ dir).

These tests construct the dataclass only — no Isaac Sim, no robot. They require
``lerobot-isaac-configs`` to be importable (it is, as an editable sibling in the
workspace env).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lerobot_isaac_configs")


def test_default_usd_path_resolves_from_configs(tmp_path):
    from lerobot_isaac_configs import get_scene_path
    from lerobot_isaac_deploy.sim.isaac_scene_session import IsaacSceneSession

    sess = IsaacSceneSession(
        usd_path=None,
        policy_path=tmp_path / "ckpt",
        dataset_root=tmp_path / "ds",
    )
    assert sess.usd_path == get_scene_path("so101_workspace")
    assert sess.usd_path.is_file()


def test_explicit_usd_path_is_respected(tmp_path):
    from lerobot_isaac_deploy.sim.isaac_scene_session import IsaacSceneSession

    explicit = tmp_path / "custom_scene.usd"
    explicit.write_text("#usda 1.0\n")
    sess = IsaacSceneSession(
        usd_path=explicit,
        policy_path=tmp_path / "ckpt",
        dataset_root=tmp_path / "ds",
    )
    assert sess.usd_path == explicit


def test_default_render_cameras_is_d435():
    """DR100 Phase 1 migration: single d435_rgb camera, not overhead+wrist."""
    from lerobot_isaac_deploy.sim.isaac_scene_session import IsaacSceneSession

    sess = IsaacSceneSession(
        usd_path=None,
        policy_path=Path("ckpt"),
        dataset_root=Path("ds"),
    )
    assert sess.render_cameras == ("d435_rgb",)
