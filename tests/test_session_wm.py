"""Tests for session-level WM dispatch + real-arm gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_isaac_deploy.session import DeploySession, SessionConfig


def _make_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "dataset"
    ds.mkdir()
    return ds


def _write_synthetic_dreamer(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "synthetic_marker.json").write_text(
        json.dumps({"kind": "dreamerv3", "action_dim": 6}), encoding="utf-8"
    )
    # Also write a hydra-style sidecar so detect_policy_kind classifies as dreamerv3.
    (d / ".hydra").mkdir()
    (d / ".hydra" / "config.yaml").write_text("placeholder: 1\n", encoding="utf-8")
    (d / "ckpt_0.ckpt").write_text("not a real torch file", encoding="utf-8")
    return d


def test_validate_inputs_accepts_dreamerv3(tmp_path: Path) -> None:
    cfg = SessionConfig(
        policy_path=_write_synthetic_dreamer(tmp_path),
        dataset_root=_make_dataset(tmp_path),
    )
    session = DeploySession(cfg)
    # Should NOT raise — DreamerV3 is now first-class.
    session._validate_inputs()
    assert getattr(session, "_ckpt_kind", None) == "dreamerv3"


def test_require_real_ckpt_refuses_synthetic_execute(tmp_path: Path) -> None:
    cfg = SessionConfig(
        policy_path=_write_synthetic_dreamer(tmp_path),
        dataset_root=_make_dataset(tmp_path),
        require_real_ckpt=True,
        assume_yes=True,
    )
    session = DeploySession(cfg)
    session._validate_inputs()
    with pytest.raises(RuntimeError, match="synthetic test fixture"):
        session.step_execute_tight()


def test_validate_inputs_refuses_vjepa(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "vjepa_config.json").write_text("{}", encoding="utf-8")
    cfg = SessionConfig(policy_path=ckpt, dataset_root=_make_dataset(tmp_path))
    session = DeploySession(cfg)
    with pytest.raises(RuntimeError, match="video"):
        session._validate_inputs()
