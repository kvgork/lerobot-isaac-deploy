"""Tests for wm_rollout bodies (synthetic short-circuit + summary JSON shape)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from lerobot_isaac_deploy.wm_rollout import rollout


def _synth_ckpt(tmp_path: Path, kind: str, **extra) -> Path:
    d = tmp_path / f"{kind}-fixture"
    d.mkdir()
    payload = {"kind": kind, "action_dim": 6, "image_shape": [3, 64, 64], **extra}
    (d / "synthetic_marker.json").write_text(json.dumps(payload), encoding="utf-8")
    # Also write a marker the kind-detector can latch onto.
    if kind == "dreamerv3":
        (d / ".hydra").mkdir()
        (d / ".hydra" / "config.yaml").write_text("placeholder: 1\n", encoding="utf-8")
        (d / "ckpt_0.ckpt").write_text("synthetic", encoding="utf-8")
    elif kind == "lewm":
        (d / "leworldmodel_config.json").write_text("{}", encoding="utf-8")
    return d


def _ds(tmp_path: Path) -> Path:
    d = tmp_path / "ds"; d.mkdir(); return d


def test_rollout_dreamerv3_synthetic(tmp_path: Path) -> None:
    cp = _synth_ckpt(tmp_path, "dreamerv3")
    out = tmp_path / "out"
    summary_path = rollout(cp, dataset_root=_ds(tmp_path), output_dir=out, horizon_steps=12)
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["kind"] == "dreamerv3"
    assert summary["horizon"] == 12
    assert summary["synthetic"] is True
    assert "mean_recon_loss" in summary

    npz = np.load(out / "next_state_pred.npz")
    assert npz["pred"].shape == (12, 3, 64, 64)


def test_rollout_lewm_synthetic(tmp_path: Path) -> None:
    cp = _synth_ckpt(tmp_path, "lewm", latent_dim=192)
    out = tmp_path / "out"
    summary_path = rollout(cp, dataset_root=_ds(tmp_path), output_dir=out, horizon_steps=5)
    summary = json.loads(summary_path.read_text())
    assert summary["kind"] == "lewm"
    assert summary["synthetic"] is True
    assert summary["horizon"] == 5
    assert "mean_pred_loss" in summary

    npz = np.load(out / "next_state_pred.npz")
    assert npz["pred"].shape == (5, 192)


def test_rollout_unknown_kind(tmp_path: Path) -> None:
    """No detectable kind -> RuntimeError."""
    cp = tmp_path / "garbage"; cp.mkdir()
    with pytest.raises(RuntimeError, match="unknown checkpoint kind"):
        rollout(cp, dataset_root=_ds(tmp_path), output_dir=tmp_path / "out")
