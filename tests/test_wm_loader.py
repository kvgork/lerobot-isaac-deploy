"""Tests for wm_loader (synthetic-marker short-circuit, refusal paths)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_isaac_deploy.wm_loader import (
    WMDeployNotSupported,
    load_dreamerv3,
    load_lewm,
)


def _write_synthetic_marker(d: Path, **fields) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "dreamerv3", "action_dim": 6, "image_shape": [3, 64, 64], **fields}
    (d / "synthetic_marker.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def test_load_dreamerv3_synthetic_short_circuit(tmp_path: Path) -> None:
    """Synthetic marker → no torch/sheeprl needed, returns a stub actor."""
    ckpt = _write_synthetic_marker(tmp_path / "dreamer-fixture")
    actor = load_dreamerv3(ckpt)
    assert hasattr(actor, "select_action")
    assert hasattr(actor, "reset")
    actor.reset()
    out = actor.select_action({"state": None})
    import numpy as np

    arr = np.asarray(out)
    assert arr.shape == (6,)
    assert arr.dtype.kind == "f"


def test_load_dreamerv3_synthetic_custom_action_dim(tmp_path: Path) -> None:
    ckpt = _write_synthetic_marker(tmp_path / "dreamer-7dof", action_dim=7)
    actor = load_dreamerv3(ckpt)
    import numpy as np

    arr = np.asarray(actor.select_action({}))
    assert arr.shape == (7,)


def test_load_lewm_refuses(tmp_path: Path) -> None:
    with pytest.raises(WMDeployNotSupported):
        load_lewm(tmp_path)
