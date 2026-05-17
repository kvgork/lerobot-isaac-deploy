"""Tests for sync.py + session.py dataset-shipping path.

Covers:

* ``sync_ckpt_to_laptop`` rsyncs a dataset tree when ``dataset_root`` is set,
  using a second rsync call with the right source/dest.
* The rewritten ``winner.json`` carries ``dataset_root`` pointing at the
  laptop-local path when both ``winner_json`` and ``dataset_root`` are passed.
* ``_resolve_dataset_root`` honors the precedence ladder:
  flag > winner.json > env > hardcoded fallback.
* ``cfg_from_namespace`` plumbs the resolved path through.
* ``build_sync_ckpt_parser`` accepts the new ``--dataset-root`` flag.

We monkeypatch ``subprocess.run`` to capture rsync / ssh invocations
without touching the network or the filesystem outside of pytest's
``tmp_path``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeCompleted:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def _make_fake_run(captured: list[list[str]]):
    """Return a stub ``subprocess.run`` that records every argv."""

    def fake_run(args, check=False, **kwargs):
        # Argv is the first positional. We record a copy so the caller's
        # list mutations after dispatch don't bleed into our log.
        captured.append(list(args))
        return _FakeCompleted(returncode=0)

    return fake_run


def _make_fake_run_dir(tmp_path: Path) -> Path:
    """Build a minimal autoresearch-shaped run dir.

    Layout::

        <tmp>/trial_X/
          checkpoints/
            045000/
              pretrained_model/
                config.json
    """
    run_dir = tmp_path / "trial_X"
    pm = run_dir / "checkpoints" / "045000" / "pretrained_model"
    pm.mkdir(parents=True)
    (pm / "config.json").write_text("{}")
    return run_dir


def _make_fake_dataset(tmp_path: Path, name: str = "so101-pickplace1") -> Path:
    ds = tmp_path / name
    (ds / "data").mkdir(parents=True)
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")
    return ds


# --------------------------------------------------------------------------- #
# Parser accepts --dataset-root
# --------------------------------------------------------------------------- #


def test_sync_ckpt_parser_accepts_dataset_root() -> None:
    from lerobot_isaac_deploy.sync import build_sync_ckpt_parser

    ns = build_sync_ckpt_parser().parse_args([
        "--run-dir", "/tmp/run",
        "--dataset-root", "/tmp/ds",
    ])
    assert ns.dataset_root == "/tmp/ds"


def test_sync_ckpt_parser_dataset_root_optional() -> None:
    """Flag is optional — falls back to None when omitted."""
    from lerobot_isaac_deploy.sync import build_sync_ckpt_parser

    ns = build_sync_ckpt_parser().parse_args(["--run-dir", "/tmp/run"])
    assert ns.dataset_root is None


# --------------------------------------------------------------------------- #
# sync_ckpt_to_laptop dispatches a second rsync for the dataset
# --------------------------------------------------------------------------- #


def test_dataset_root_triggers_second_rsync(tmp_path, monkeypatch) -> None:
    """When dataset_root is set, sync_ckpt_to_laptop rsyncs the dataset tree."""
    from lerobot_isaac_deploy import sync as sync_mod

    run_dir = _make_fake_run_dir(tmp_path)
    ds = _make_fake_dataset(tmp_path)

    captured: list[list[str]] = []
    monkeypatch.setattr(sync_mod.subprocess, "run", _make_fake_run(captured))

    rc = sync_mod.sync_ckpt_to_laptop(
        run_dir,
        host="laptop",
        laptop_base="~/workspaces/lerobot-isaac-deploy",
        dataset_root=ds,
    )
    assert rc == 0

    # Find the rsync calls that ship dataset content.
    rsync_calls = [c for c in captured if c and c[0] == "rsync"]
    dataset_calls = [
        c for c in rsync_calls
        if any(str(ds) in str(arg) for arg in c)
    ]
    assert len(dataset_calls) == 1, (
        f"expected one dataset rsync; got {len(dataset_calls)}: {dataset_calls}"
    )
    dataset_call = dataset_calls[0]
    # Source is the dataset dir with trailing slash.
    assert dataset_call[-2] == f"{ds}/"
    # Destination is <laptop_base>/datasets/<basename>/ on the remote host.
    expected_dst = (
        "laptop:~/workspaces/lerobot-isaac-deploy/datasets/"
        f"{ds.name}/"
    )
    assert dataset_call[-1] == expected_dst


def test_no_dataset_root_means_no_dataset_rsync(tmp_path, monkeypatch) -> None:
    """Without dataset_root, no dataset rsync is issued."""
    from lerobot_isaac_deploy import sync as sync_mod

    run_dir = _make_fake_run_dir(tmp_path)

    captured: list[list[str]] = []
    monkeypatch.setattr(sync_mod.subprocess, "run", _make_fake_run(captured))

    rc = sync_mod.sync_ckpt_to_laptop(run_dir, host="laptop")
    assert rc == 0

    rsync_calls = [c for c in captured if c and c[0] == "rsync"]
    # The only rsync is the ckpt push (one call).
    assert len(rsync_calls) == 1
    assert "pretrained_model" in rsync_calls[0][-1]


def test_missing_dataset_root_logs_warning_does_not_fail(
    tmp_path, monkeypatch, capsys
) -> None:
    """If dataset_root points at a non-existent dir, skip with a warning."""
    from lerobot_isaac_deploy import sync as sync_mod

    run_dir = _make_fake_run_dir(tmp_path)

    captured: list[list[str]] = []
    monkeypatch.setattr(sync_mod.subprocess, "run", _make_fake_run(captured))

    rc = sync_mod.sync_ckpt_to_laptop(
        run_dir,
        host="laptop",
        dataset_root=tmp_path / "no-such-ds",
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARN" in out and "dataset-root" in out

    # No rsync for the dataset.
    rsync_calls = [c for c in captured if c and c[0] == "rsync"]
    assert len(rsync_calls) == 1  # ckpt only


# --------------------------------------------------------------------------- #
# Rewritten winner.json gains a dataset_root field
# --------------------------------------------------------------------------- #


def test_rewritten_winner_json_has_dataset_root(tmp_path, monkeypatch) -> None:
    from lerobot_isaac_deploy import sync as sync_mod

    run_dir = _make_fake_run_dir(tmp_path)
    ds = _make_fake_dataset(tmp_path)

    desktop_winner = tmp_path / "winner.json"
    desktop_winner.write_text(json.dumps({
        "winner_run_id": "trial_X",
        "winner_policy_path": str(
            run_dir / "checkpoints" / "045000" / "pretrained_model"
        ),
        "winner_pc_success": 0.5,
    }))

    # Capture the tmp winner.json the function writes before rsync ships it.
    real_write_text = Path.write_text
    seen_tmp_paths: list[Path] = []

    def spy_write_text(self, data, *args, **kwargs):
        # Record any /tmp/winner-*.json writes so we can inspect them.
        if str(self).startswith("/tmp/winner-") and str(self).endswith(".json"):
            seen_tmp_paths.append(Path(str(self)))
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", spy_write_text)

    captured: list[list[str]] = []

    def fake_run(args, check=False, **kwargs):
        # The function unlinks the /tmp/winner-*.json file after the rsync
        # of it succeeds. We've already saved a copy of its CONTENT (in
        # seen_tmp_paths via spy_write_text) so we can read it from the
        # path the spy recorded — but we need to read it BEFORE unlink
        # happens. Strategy: snapshot the file content at rsync time.
        captured.append(list(args))
        if args and args[0] == "rsync":
            # The source path is args[-2]; if it's a /tmp/winner-*.json,
            # snapshot the content now.
            src = args[-2]
            if str(src).startswith("/tmp/winner-") and str(src).endswith(".json"):
                content = Path(src).read_text()
                # Stash on the captured entry for later inspection.
                captured[-1] = list(args) + [f"__CONTENT__={content}"]
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(sync_mod.subprocess, "run", fake_run)

    rc = sync_mod.sync_ckpt_to_laptop(
        run_dir,
        host="laptop",
        laptop_base="~/workspaces/lerobot-isaac-deploy",
        winner_json=desktop_winner,
        dataset_root=ds,
    )
    assert rc == 0

    # Find the rsync call carrying the rewritten winner.json content.
    content_call = next(
        c for c in captured
        if any(str(a).startswith("__CONTENT__=") for a in c)
    )
    content_arg = next(
        a for a in content_call if str(a).startswith("__CONTENT__=")
    )
    rewritten = json.loads(content_arg[len("__CONTENT__="):])
    assert "dataset_root" in rewritten
    expected_path = (
        f"~/workspaces/lerobot-isaac-deploy/datasets/{ds.name}"
    )
    assert rewritten["dataset_root"] == expected_path
    # The winner_policy_path was also rewritten to the laptop layout.
    assert rewritten["winner_policy_path"].endswith(
        f"models/{run_dir.name}/045000/pretrained_model"
    )


def test_rewritten_winner_no_dataset_root_field_when_no_ds(
    tmp_path, monkeypatch
) -> None:
    """Without --dataset-root, the rewritten JSON does NOT add a dataset_root."""
    from lerobot_isaac_deploy import sync as sync_mod

    run_dir = _make_fake_run_dir(tmp_path)
    desktop_winner = tmp_path / "winner.json"
    desktop_winner.write_text(json.dumps({
        "winner_policy_path": str(
            run_dir / "checkpoints" / "045000" / "pretrained_model"
        ),
    }))

    captured_content: list[str] = []

    def fake_run(args, check=False, **kwargs):
        if args and args[0] == "rsync":
            src = args[-2]
            if str(src).startswith("/tmp/winner-") and str(src).endswith(".json"):
                captured_content.append(Path(src).read_text())
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(sync_mod.subprocess, "run", fake_run)

    rc = sync_mod.sync_ckpt_to_laptop(
        run_dir,
        host="laptop",
        winner_json=desktop_winner,
    )
    assert rc == 0
    assert len(captured_content) == 1
    rewritten = json.loads(captured_content[0])
    assert "dataset_root" not in rewritten


# --------------------------------------------------------------------------- #
# session.py dataset precedence
# --------------------------------------------------------------------------- #


def test_resolve_dataset_root_explicit_flag_wins(tmp_path, monkeypatch) -> None:
    """Explicit --dataset-root beats winner.json AND env."""
    from lerobot_isaac_deploy.session import _resolve_dataset_root

    monkeypatch.setenv(
        "LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", str(tmp_path / "env-ds")
    )
    winner = tmp_path / "winner.json"
    winner.write_text(json.dumps({
        "winner_policy_path": "/tmp/x",
        "dataset_root": str(tmp_path / "winner-ds"),
    }))

    explicit = tmp_path / "flag-ds"
    out = _resolve_dataset_root(str(explicit), winner)
    assert out == explicit


def test_resolve_dataset_root_winner_beats_env(tmp_path, monkeypatch) -> None:
    """winner.json dataset_root beats env when --dataset-root not passed."""
    from lerobot_isaac_deploy.session import _resolve_dataset_root

    monkeypatch.setenv(
        "LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", str(tmp_path / "env-ds")
    )
    winner = tmp_path / "winner.json"
    winner.write_text(json.dumps({
        "winner_policy_path": "/tmp/x",
        "dataset_root": str(tmp_path / "winner-ds"),
    }))

    out = _resolve_dataset_root(None, winner)
    assert out == tmp_path / "winner-ds"


def test_resolve_dataset_root_env_beats_fallback(tmp_path, monkeypatch) -> None:
    """env var beats hardcoded fallback when winner.json lacks the field."""
    from lerobot_isaac_deploy.session import _resolve_dataset_root

    monkeypatch.setenv(
        "LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", str(tmp_path / "env-ds")
    )
    winner = tmp_path / "winner.json"
    winner.write_text(json.dumps({"winner_policy_path": "/tmp/x"}))

    out = _resolve_dataset_root(None, winner)
    assert out == tmp_path / "env-ds"


def test_resolve_dataset_root_no_winner_no_env_falls_back(
    tmp_path, monkeypatch
) -> None:
    """With no winner and no env, returns the hardcoded fallback."""
    from lerobot_isaac_deploy.session import (
        _hardcoded_dataset_fallback,
        _resolve_dataset_root,
    )

    monkeypatch.delenv("LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", raising=False)
    out = _resolve_dataset_root(None, None)
    assert out == _hardcoded_dataset_fallback()


def test_resolve_dataset_root_winner_missing_file(tmp_path, monkeypatch) -> None:
    """A non-existent winner.json path is tolerated (falls through to env)."""
    from lerobot_isaac_deploy.session import _resolve_dataset_root

    monkeypatch.setenv(
        "LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", str(tmp_path / "env-ds")
    )
    bogus = tmp_path / "no-such-winner.json"
    out = _resolve_dataset_root(None, bogus)
    assert out == tmp_path / "env-ds"


# --------------------------------------------------------------------------- #
# cfg_from_namespace integration
# --------------------------------------------------------------------------- #


def test_cfg_from_namespace_picks_dataset_from_winner(
    tmp_path, monkeypatch
) -> None:
    """When --winner is used and winner.json has dataset_root, cfg uses it."""
    from lerobot_isaac_deploy.session import (
        build_session_parser,
        cfg_from_namespace,
    )

    # Build a fake winner.json + its winner_policy_path (must exist).
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    ds = tmp_path / "synced-ds"
    ds.mkdir()
    winner = tmp_path / "winner.json"
    winner.write_text(json.dumps({
        "winner_policy_path": str(ckpt),
        "dataset_root": str(ds),
    }))

    # Make sure env var doesn't shadow.
    monkeypatch.delenv("LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", raising=False)

    ns = build_session_parser().parse_args(["--winner", str(winner)])
    cfg = cfg_from_namespace(ns)
    assert cfg.dataset_root == ds


def test_cfg_explicit_dataset_root_overrides_winner(
    tmp_path, monkeypatch
) -> None:
    """Explicit --dataset-root overrides winner.json's field."""
    from lerobot_isaac_deploy.session import (
        build_session_parser,
        cfg_from_namespace,
    )

    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    ds_winner = tmp_path / "winner-ds"
    ds_winner.mkdir()
    ds_flag = tmp_path / "flag-ds"
    ds_flag.mkdir()
    winner = tmp_path / "winner.json"
    winner.write_text(json.dumps({
        "winner_policy_path": str(ckpt),
        "dataset_root": str(ds_winner),
    }))

    monkeypatch.delenv("LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", raising=False)

    ns = build_session_parser().parse_args([
        "--winner", str(winner),
        "--dataset-root", str(ds_flag),
    ])
    cfg = cfg_from_namespace(ns)
    assert cfg.dataset_root == ds_flag
