"""Tests for WM loader + rollout stubs + session gate. No heavy deps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot_isaac_deploy import (
    DeploySession,
    SessionConfig,
    WMDeployNotSupported,
    load_lewm,
)


def _make_lerobot_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "lerobot-ckpt"
    p.mkdir()
    (p / "model.safetensors").write_bytes(b"")
    (p / "config.json").write_text("{}")
    return p


def _make_dreamer_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "wm-dreamer"
    (p / ".hydra").mkdir(parents=True)
    (p / ".hydra" / "config.yaml").write_text("")
    (p / "ckpt_1000.ckpt").write_bytes(b"")
    return p


def _make_lewm_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "wm-lewm"
    p.mkdir()
    (p / "policy.json").write_text(json.dumps({"type": "le_world_model"}))
    return p


# --------------------------------------------------------------------------- #
# wm_loader.load_lewm — always refuses
# --------------------------------------------------------------------------- #


def test_load_lewm_always_raises(tmp_path) -> None:
    p = _make_lewm_ckpt(tmp_path)
    with pytest.raises(WMDeployNotSupported, match=r"no actor"):
        load_lewm(p)


# --------------------------------------------------------------------------- #
# Session gate — refuses non-lerobot kinds with actionable messages
# --------------------------------------------------------------------------- #


def test_session_dreamerv3_fails_preflight_against_lerobot_runner(tmp_path) -> None:
    """DreamerV3 ckpts are accepted by _validate_inputs (Phase 2) but
    fail in step_preflight when robot-data-run-check tries to load them
    as a lerobot policy. The session exits rc=1. The mock-hardware path
    (--mock-hardware) is the supported smoke route for DreamerV3 ckpts
    without a real arm — see docs/world-model-deploy.md.
    """
    ckpt = _make_dreamer_ckpt(tmp_path)
    ds = tmp_path / "dataset"
    ds.mkdir()
    sess = DeploySession(SessionConfig(policy_path=ckpt, dataset_root=ds))
    rc = sess.run()
    assert rc == 1


def test_session_refuses_lewm_with_hint(tmp_path) -> None:
    ckpt = _make_lewm_ckpt(tmp_path)
    ds = tmp_path / "dataset"
    ds.mkdir()
    sess = DeploySession(SessionConfig(policy_path=ckpt, dataset_root=ds))
    rc = sess.run()
    assert rc == 1


def test_session_refuses_unknown(tmp_path) -> None:
    p = tmp_path / "noise"
    p.mkdir()
    (p / "random.txt").write_text("not a model")
    ds = tmp_path / "dataset"
    ds.mkdir()
    sess = DeploySession(SessionConfig(policy_path=p, dataset_root=ds))
    rc = sess.run()
    assert rc == 1


# --------------------------------------------------------------------------- #
# wm-rollout CLI parser — flag shape is locked
# --------------------------------------------------------------------------- #


def test_wm_rollout_parser_required_flags() -> None:
    from lerobot_isaac_deploy.wm_rollout import build_rollout_parser

    parser = build_rollout_parser()
    ns = parser.parse_args([
        "--checkpoint", "/tmp/c",
        "--dataset", "/tmp/d",
        "--output-dir", "/tmp/o",
    ])
    assert ns.horizon_steps == 50
    assert ns.n_seed_episodes == 1


def test_wm_rollout_stub_returns_2_for_unimpl(tmp_path) -> None:
    """The body is unimplemented; CLI must exit cleanly with rc=2."""
    from lerobot_isaac_deploy.wm_rollout import main

    ckpt = _make_dreamer_ckpt(tmp_path)
    ds = tmp_path / "dataset"
    ds.mkdir()
    rc = main([
        "--checkpoint", str(ckpt),
        "--dataset", str(ds),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 2  # NotImplementedError caught → rc=2


def test_wm_rollout_refuses_unknown_kind(tmp_path) -> None:
    """rollout() raises RuntimeError on unrecognized kind."""
    from lerobot_isaac_deploy.wm_rollout import main

    p = tmp_path / "noise"
    p.mkdir()
    (p / "random.txt").write_text("noise")
    ds = tmp_path / "dataset"
    ds.mkdir()
    rc = main([
        "--checkpoint", str(p),
        "--dataset", str(ds),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 1  # generic error path


# --------------------------------------------------------------------------- #
# CLI dispatch lists the new subcommands
# --------------------------------------------------------------------------- #


def test_cli_lists_wm_subcommands(capsys) -> None:
    from lerobot_isaac_deploy.cli import main as cli_main

    rc = cli_main(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    for sub in ("wm-rollout", "kind"):
        assert sub in out


def test_cli_kind_detects_lerobot(tmp_path, capsys) -> None:
    from lerobot_isaac_deploy.cli import main as cli_main

    p = _make_lerobot_ckpt(tmp_path)
    rc = cli_main(["kind", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("lerobot")
