"""Smoke: package + sub-modules import without lerobot/robot-data-runner."""

from __future__ import annotations

import json
from pathlib import Path


def test_top_level_import() -> None:
    import lerobot_isaac_deploy as m

    assert hasattr(m, "DeploySession")
    assert hasattr(m, "SessionConfig")
    assert isinstance(m.__version__, str)


def test_session_config_defaults() -> None:
    from lerobot_isaac_deploy import SessionConfig

    cfg = SessionConfig(policy_path="/tmp/p", dataset_root="/tmp/d")
    assert cfg.task == "pick and place cube"
    assert cfg.port == "/dev/ttyACM0"
    assert cfg.do_execute is False
    assert cfg.do_dry_loop is False
    # Path coercion
    assert isinstance(cfg.policy_path, Path)
    assert isinstance(cfg.dataset_root, Path)


def test_resolve_winner_policy(tmp_path) -> None:
    from lerobot_isaac_deploy.session import resolve_winner_policy

    winner = tmp_path / "winner.json"
    expected = "/abs/path/to/ckpt/pretrained_model"
    winner.write_text(
        json.dumps({"winner_policy_path": expected, "ranking": []})
    )
    assert resolve_winner_policy(winner) == Path(expected)


def test_session_parser_accepts_winner() -> None:
    from lerobot_isaac_deploy.session import build_session_parser

    ns = build_session_parser().parse_args(["--winner", "/tmp/w.json"])
    assert ns.winner == "/tmp/w.json"
    assert ns.policy_path is None
    assert ns.execute is False


def test_session_parser_accepts_policy_path() -> None:
    from lerobot_isaac_deploy.session import build_session_parser

    ns = build_session_parser().parse_args([
        "--policy-path", "/tmp/m", "--execute",
        "--task", "pick and place cube",
    ])
    assert ns.policy_path == "/tmp/m"
    assert ns.execute is True
    assert ns.task == "pick and place cube"


def test_session_parser_rejects_both_inputs() -> None:
    import pytest

    from lerobot_isaac_deploy.session import build_session_parser

    with pytest.raises(SystemExit):
        build_session_parser().parse_args([
            "--policy-path", "/tmp/m", "--winner", "/tmp/w.json",
        ])


def test_sync_ckpt_parser() -> None:
    from lerobot_isaac_deploy.sync import build_sync_ckpt_parser

    ns = build_sync_ckpt_parser().parse_args(["--run-dir", "/tmp/run"])
    assert ns.run_dir == "/tmp/run"
    assert ns.host == "laptop"
    assert ns.dry_run is False


def test_sync_eval_parser() -> None:
    from lerobot_isaac_deploy.sync import build_sync_eval_parser

    ns = build_sync_eval_parser().parse_args([])
    assert ns.host == "laptop"


def test_cli_dispatch_lists_subcommands(capsys) -> None:
    from lerobot_isaac_deploy.cli import main

    rc = main(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    for sub in ("session", "sync-ckpt", "sync-eval", "bootstrap"):
        assert sub in out


def test_cli_dispatch_rejects_unknown(capsys) -> None:
    from lerobot_isaac_deploy.cli import main

    rc = main(["frobnicate"])
    assert rc == 2
