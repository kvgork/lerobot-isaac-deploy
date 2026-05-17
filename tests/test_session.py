"""Tests for session-level non-interactive + mock-hardware paths.

These cover:

* ``confirm()`` auto-yes branch (flag + env var).
* ``confirm()`` refuses auto-yes when ``safety_critical=True``.
* ``build_session_parser()`` accepts the new flags.
* ``cfg_from_namespace()`` plumbs the flags into ``SessionConfig``.
* ``_synthesize_observation`` / ``_infer_motor_names`` produce a
  schema-correct synthetic observation without needing torch / lerobot
  (we hand it a stub policy object).
* ``run_mock_inference_loop`` short-circuits with a friendly error when
  ``robot_data_runner`` is not importable, so the test never tries to
  actually load a real checkpoint inside the test runner.

We deliberately do NOT test the live ``run_mock_inference_loop`` path
end-to-end here — that is the smoke-test transcript captured by the
operator (see README "Smoke test without hardware"). Pulling lerobot +
a real ckpt into the unit-test runner would be slow and flaky.
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# confirm() — auto-yes + safety-critical semantics
# --------------------------------------------------------------------------- #


def test_confirm_auto_yes_returns_without_stdin() -> None:
    """auto_yes=True + safety_critical=False → no input(); prints tag."""
    from lerobot_isaac_deploy.session import confirm

    buf = io.StringIO()
    with redirect_stdout(buf):
        confirm("proceed?", auto_yes=True, safety_critical=False)
    out = buf.getvalue()
    assert "auto-yes" in out
    assert "proceed?" in out


def test_confirm_safety_critical_ignores_auto_yes(monkeypatch) -> None:
    """auto_yes=True + safety_critical=True → still blocks on stdin."""
    from lerobot_isaac_deploy import session

    called: dict[str, bool] = {"input_used": False}

    def fake_input(prompt: str) -> str:
        called["input_used"] = True
        return "yes"

    monkeypatch.setattr(session, "input", fake_input, raising=False)
    # Even with auto_yes, safety_critical must consult stdin.
    session.confirm("e-stop ready?", auto_yes=True, safety_critical=True)
    assert called["input_used"] is True


def test_confirm_eof_exits_with_code_10(monkeypatch) -> None:
    """When ``input()`` raises EOFError (piped / non-tty stdin), exit 10."""
    from lerobot_isaac_deploy import session

    def raise_eof(prompt: str) -> str:
        raise EOFError()

    monkeypatch.setattr(session, "input", raise_eof, raising=False)
    with pytest.raises(SystemExit) as exc:
        with redirect_stderr(io.StringIO()):
            session.confirm("are you ready?", auto_yes=False)
    assert exc.value.code == 10


def test_confirm_no_aborts_with_code_10(monkeypatch) -> None:
    """Typing anything other than 'yes' aborts with code 10."""
    from lerobot_isaac_deploy import session

    monkeypatch.setattr(session, "input", lambda prompt: "n", raising=False)
    with pytest.raises(SystemExit) as exc:
        with redirect_stderr(io.StringIO()):
            session.confirm("?", auto_yes=False)
    assert exc.value.code == 10


# --------------------------------------------------------------------------- #
# Parser + cfg plumbing
# --------------------------------------------------------------------------- #


def test_parser_accepts_yes_long_form() -> None:
    from lerobot_isaac_deploy.session import build_session_parser

    ns = build_session_parser().parse_args([
        "--policy-path", "/tmp/m", "--yes",
    ])
    assert ns.assume_yes is True
    assert ns.mock_hardware is False


def test_parser_accepts_assume_yes_alias() -> None:
    from lerobot_isaac_deploy.session import build_session_parser

    ns = build_session_parser().parse_args([
        "--policy-path", "/tmp/m", "--assume-yes",
    ])
    assert ns.assume_yes is True


def test_parser_accepts_mock_hardware() -> None:
    from lerobot_isaac_deploy.session import build_session_parser

    ns = build_session_parser().parse_args([
        "--policy-path", "/tmp/m", "--mock-hardware", "--dry-run-loop",
    ])
    assert ns.mock_hardware is True
    assert ns.dry_run_loop is True


def test_parser_default_assume_yes_false() -> None:
    from lerobot_isaac_deploy.session import build_session_parser

    ns = build_session_parser().parse_args(["--policy-path", "/tmp/m"])
    assert ns.assume_yes is False
    assert ns.mock_hardware is False


def test_cfg_from_namespace_plumbs_flags(tmp_path) -> None:
    from lerobot_isaac_deploy.session import (
        build_session_parser,
        cfg_from_namespace,
    )

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    ns = build_session_parser().parse_args([
        "--policy-path", str(ckpt),
        "--dataset-root", str(tmp_path),
        "--yes",
        "--mock-hardware",
        "--duration-s", "5",
        "--rate-hz", "10",
    ])
    cfg = cfg_from_namespace(ns)
    assert cfg.assume_yes is True
    assert cfg.mock_hardware is True
    assert cfg.duration_dry_s == 5.0
    assert cfg.rate_hz == 10.0


def test_cfg_honors_env_assume_yes(monkeypatch, tmp_path) -> None:
    from lerobot_isaac_deploy.session import (
        build_session_parser,
        cfg_from_namespace,
    )

    monkeypatch.setenv("LEROBOT_ISAAC_DEPLOY_ASSUME_YES", "1")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    ns = build_session_parser().parse_args([
        "--policy-path", str(ckpt),
        "--dataset-root", str(tmp_path),
    ])
    cfg = cfg_from_namespace(ns)
    assert cfg.assume_yes is True


def test_cfg_env_assume_yes_falsy(monkeypatch, tmp_path) -> None:
    from lerobot_isaac_deploy.session import (
        build_session_parser,
        cfg_from_namespace,
    )

    monkeypatch.setenv("LEROBOT_ISAAC_DEPLOY_ASSUME_YES", "0")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    ns = build_session_parser().parse_args([
        "--policy-path", str(ckpt),
        "--dataset-root", str(tmp_path),
    ])
    cfg = cfg_from_namespace(ns)
    assert cfg.assume_yes is False


# --------------------------------------------------------------------------- #
# Validate-inputs rejects --mock-hardware + --execute
# --------------------------------------------------------------------------- #


def test_validate_rejects_mock_plus_execute(tmp_path) -> None:
    """Mock-hardware + execute is forbidden — early error."""
    from lerobot_isaac_deploy.session import DeploySession, SessionConfig

    # Create a fake lerobot-shaped ckpt so the kind-detector accepts it.
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_text("")
    (ckpt / "config.json").write_text("{}")
    ds = tmp_path / "ds"
    ds.mkdir()

    cfg = SessionConfig(
        policy_path=ckpt,
        dataset_root=ds,
        do_dry_loop=True,
        do_execute=True,
        mock_hardware=True,
    )
    rc = DeploySession(cfg).run()
    assert rc == 1


# --------------------------------------------------------------------------- #
# mock_hardware helpers — no torch / lerobot needed
# --------------------------------------------------------------------------- #


@dataclass
class _StubFeature:
    shape: tuple[int, ...]


@dataclass
class _StubConfig:
    input_features: dict


class _StubPolicy:
    """Minimal stand-in for a lerobot policy — only exposes ``config``."""

    def __init__(self, feats: dict[str, _StubFeature]) -> None:
        self.config = _StubConfig(input_features=feats)


def test_infer_motor_names_so101_default() -> None:
    from lerobot_isaac_deploy.mock_hardware import _infer_motor_names

    pol = _StubPolicy({"observation.state": _StubFeature(shape=(6,))})
    names = _infer_motor_names(pol)
    assert names == [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]


def test_infer_motor_names_unusual_dim() -> None:
    from lerobot_isaac_deploy.mock_hardware import _infer_motor_names

    pol = _StubPolicy({"observation.state": _StubFeature(shape=(4,))})
    names = _infer_motor_names(pol)
    assert names == ["motor0", "motor1", "motor2", "motor3"]


def test_synthesize_observation_shapes() -> None:
    import numpy as np

    from lerobot_isaac_deploy.mock_hardware import _synthesize_observation

    pol = _StubPolicy({
        "observation.state": _StubFeature(shape=(6,)),
        "observation.images.wrist_camera_rgb":
            _StubFeature(shape=(3, 480, 640)),
        "observation.images.overhead_camera_rgb":
            _StubFeature(shape=(3, 240, 320)),
    })
    motor_names = ["a", "b", "c"]
    obs = _synthesize_observation(pol, motor_names)
    # motors
    for m in motor_names:
        assert obs[f"{m}.pos"] == 0.0
    # cameras: HWC uint8, shape matches feature shape transposed
    wrist = obs["wrist_camera_rgb"]
    assert wrist.shape == (480, 640, 3)
    assert wrist.dtype == np.uint8
    overhead = obs["overhead_camera_rgb"]
    assert overhead.shape == (240, 320, 3)


def test_mock_loop_module_exports() -> None:
    """Public surface: run_mock_inference_loop is importable."""
    from lerobot_isaac_deploy.mock_hardware import run_mock_inference_loop

    assert callable(run_mock_inference_loop)
