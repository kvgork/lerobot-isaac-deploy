"""Tests for the checkpoint-kind detector. No torch / lerobot / sheeprl needed."""

from __future__ import annotations

import json
from pathlib import Path

from lerobot_isaac_deploy.policy_kind import detect_policy_kind, explain


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", encoding="utf-8")


def test_unknown_for_nonexistent(tmp_path) -> None:
    assert detect_policy_kind(tmp_path / "nope") == "unknown"


def test_lerobot_minimal(tmp_path) -> None:
    p = tmp_path / "ckpt"
    p.mkdir()
    _touch(p / "model.safetensors")
    _touch(p / "config.json")
    assert detect_policy_kind(p) == "lerobot"


def test_lerobot_inside_subdir(tmp_path) -> None:
    """Caller may point at a parent dir; we walk one level."""
    p = tmp_path / "run"
    inner = p / "pretrained_model"
    inner.mkdir(parents=True)
    _touch(inner / "model.safetensors")
    _touch(inner / "train_config.json")
    assert detect_policy_kind(p) == "lerobot"


def test_dreamerv3_with_ckpt_file(tmp_path) -> None:
    p = tmp_path / "wm-run"
    p.mkdir()
    _touch(p / ".hydra" / "config.yaml")
    _touch(p / "ckpt_5000.ckpt")
    assert detect_policy_kind(p) == "dreamerv3"


def test_dreamerv3_hydra_only(tmp_path) -> None:
    """Even without ckpt files yet, hydra dir signals dreamerv3."""
    p = tmp_path / "wm-run"
    p.mkdir()
    _touch(p / ".hydra" / "config.yaml")
    assert detect_policy_kind(p) == "dreamerv3"


def test_lewm_marker_file(tmp_path) -> None:
    p = tmp_path / "lewm-run"
    p.mkdir()
    _touch(p / "leworldmodel_config.json")
    assert detect_policy_kind(p) == "lewm"


def test_lewm_via_policy_json(tmp_path) -> None:
    p = tmp_path / "lewm-run"
    p.mkdir()
    (p / "policy.json").write_text(json.dumps({"type": "le_world_model"}))
    assert detect_policy_kind(p) == "lewm"


def test_explain_all_kinds() -> None:
    for kind in ("lerobot", "dreamerv3", "lewm", "unknown"):
        msg = explain(kind)
        assert isinstance(msg, str) and len(msg) > 5
