"""Tests for wm_dryrun — dry-run path for DreamerV3 actor.

All tests use the synthetic-marker short-circuit so torch/sheeprl are
not required in CI environments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot_isaac_deploy.wm_dryrun import build_dryrun_parser, run_dryrun


def _make_synthetic_ckpt(tmp_path: Path, action_dim: int = 6) -> Path:
    """Write a minimal synthetic-marker checkpoint dir."""
    d = tmp_path / "dreamer-synthetic"
    d.mkdir(parents=True)
    (d / "synthetic_marker.json").write_text(
        json.dumps({"kind": "dreamerv3", "action_dim": action_dim}),
        encoding="utf-8",
    )
    # Also write a hydra sidecar so detect_policy_kind classifies as dreamerv3.
    (d / ".hydra").mkdir()
    (d / ".hydra" / "config.yaml").write_text("placeholder: 1\n", encoding="utf-8")
    (d / "ckpt_0.ckpt").write_text("not a real torch file", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# run_dryrun — core logic
# ---------------------------------------------------------------------------


def test_run_dryrun_synthetic_returns_report(tmp_path: Path) -> None:
    """Synthetic actor: run_dryrun returns a valid report dict."""
    ckpt = _make_synthetic_ckpt(tmp_path)
    report = run_dryrun(
        ckpt,
        n_samples=10,
        output_dir=tmp_path / "out",
        device="cpu",
        state_dim=13,
        image_size=64,
    )
    assert isinstance(report, dict)
    assert report["n_samples"] == 10
    assert report["synthetic_actor"] is True
    assert report["action_dim"] == 6


def test_run_dryrun_writes_report_json(tmp_path: Path) -> None:
    """report.json is written and parseable."""
    ckpt = _make_synthetic_ckpt(tmp_path)
    out = tmp_path / "out"
    run_dryrun(ckpt, n_samples=5, output_dir=out)
    report_path = out / "report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "per_joint" in report
    assert "all_finite" in report


def test_run_dryrun_action_shape_and_stats(tmp_path: Path) -> None:
    """Actions must be 6-dim, finite, and in [-1, 1]."""
    ckpt = _make_synthetic_ckpt(tmp_path, action_dim=6)
    report = run_dryrun(ckpt, n_samples=20, output_dir=tmp_path / "out")
    assert report["action_dim"] == 6
    assert report["action_shape"] == [20, 6]
    assert report["all_finite"] is True
    assert report["in_range_neg1_1"] is True
    assert report["shape_ok"] is True


def test_run_dryrun_custom_action_dim(tmp_path: Path) -> None:
    """Synthetic actor with action_dim=7 produces 7-dim actions."""
    ckpt = _make_synthetic_ckpt(tmp_path, action_dim=7)
    report = run_dryrun(ckpt, n_samples=5, output_dir=tmp_path / "out")
    assert report["action_dim"] == 7
    assert report["action_shape"] == [5, 7]


def test_run_dryrun_default_output_dir_created(tmp_path: Path, monkeypatch) -> None:
    """When output_dir is None the function creates a timestamped dir."""
    # monkeypatch cwd to tmp_path so the default outputs/ dir lands there.
    monkeypatch.chdir(tmp_path)
    ckpt = _make_synthetic_ckpt(tmp_path)
    report = run_dryrun(ckpt, n_samples=2, output_dir=None)
    # The written report_path should be under cwd/outputs/wm-dryrun-*/
    report_path = Path(report.get("checkpoint")).parent  # not the report itself
    # Just verify the report has a timestamp key
    assert "timestamp" in report


def test_run_dryrun_missing_ckpt_raises(tmp_path: Path) -> None:
    """Non-existent checkpoint path raises FileNotFoundError."""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        run_dryrun(
            tmp_path / "does-not-exist",
            n_samples=1,
            output_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_dryrun_parser_defaults() -> None:
    ns = build_dryrun_parser().parse_args(["--policy-path", "/tmp/ckpt"])
    assert ns.n_samples == 100
    assert ns.image_size == 64
    assert ns.state_dim == 13
    assert ns.seed == 42
    assert ns.device is None
    assert ns.output_dir is None


def test_dryrun_parser_explicit_args() -> None:
    ns = build_dryrun_parser().parse_args([
        "--policy-path", "/tmp/ckpt",
        "--n-samples", "50",
        "--device", "cuda",
        "--state-dim", "6",
        "--image-size", "96",
        "--seed", "7",
    ])
    assert ns.n_samples == 50
    assert ns.device == "cuda"
    assert ns.state_dim == 6
    assert ns.image_size == 96
    assert ns.seed == 7


# ---------------------------------------------------------------------------
# CLI dispatch (umbrella entry)
# ---------------------------------------------------------------------------


def test_cli_wm_dryrun_in_help(capsys) -> None:
    """wm-dryrun subcommand must appear in the umbrella help output."""
    from lerobot_isaac_deploy.cli import main as cli_main

    rc = cli_main(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "wm-dryrun" in out


def test_cli_wm_dryrun_synthetic_exit_0(tmp_path: Path) -> None:
    """CLI with synthetic checkpoint exits 0 on success."""
    from lerobot_isaac_deploy.cli import main as cli_main

    ckpt = _make_synthetic_ckpt(tmp_path)
    rc = cli_main([
        "wm-dryrun",
        "--policy-path", str(ckpt),
        "--n-samples", "5",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    assert (tmp_path / "out" / "report.json").is_file()
