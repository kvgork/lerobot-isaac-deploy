"""Sync helpers — desktop↔laptop rsync wrappers.

Two directions:

* ``sync_ckpt_to_laptop`` — runs on DESKTOP, copies a checkpoint dir
  (and optional dashboard manifest) to the laptop's deploy workspace.
* ``sync_eval_from_laptop`` — runs on DESKTOP, pulls closed-loop eval
  JSONs back so the dashboard's Evaluation tab picks them up.

Both are thin argparse wrappers around `rsync` over SSH. The functions
are importable so the CLI can also re-use them programmatically.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_LAPTOP_HOST = "laptop"
DEFAULT_LAPTOP_BASE = "~/workspaces/lerobot-isaac-deploy"


def _run_rsync(src: str, dst: str, *, dry_run: bool = False) -> int:
    # --mkpath (rsync ≥ 3.2.3) creates missing destination directories. The
    # explicit SSH mkdir below is the belt-and-suspenders fallback for older
    # rsync versions on the receiver.
    args = ["rsync", "-avhP", "--mkpath"]
    if dry_run:
        args.append("--dry-run")
    args += [src, dst]
    print(f"[sync] $ {' '.join(args)}", flush=True)
    return subprocess.run(args, check=False).returncode


def _ensure_remote_dir(host: str, remote_path: str, *, dry_run: bool = False) -> int:
    """SSH-mkdir -p the remote directory before rsync. Idempotent.

    rsync's default behavior only creates ONE level of missing parent dirs.
    Multi-level destinations (e.g. `<base>/checkpoints/<run>/<ckpt>/`) fail
    with `mkdir … No such file or directory (2)` on first sync. This
    pre-flight closes that gap.
    """
    cmd = ["ssh", host, "mkdir", "-p", remote_path]
    print(f"[sync] $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


# --------------------------------------------------------------------------- #
# desktop → laptop: checkpoint shipping
# --------------------------------------------------------------------------- #


def sync_ckpt_to_laptop(
    run_dir: Path,
    *,
    host: str = DEFAULT_LAPTOP_HOST,
    laptop_base: str = DEFAULT_LAPTOP_BASE,
    remote_dir: str | None = None,
    dry_run: bool = False,
) -> int:
    """Copy the latest pretrained_model + dashboard manifest to the laptop.

    Parameters
    ----------
    run_dir:
        Workspace-side training run directory containing
        ``policy-*/checkpoints/<NNNN>/pretrained_model/`` AND optional
        ``dashboard/manifest.json``.
    host:
        SSH alias for the laptop.
    laptop_base:
        Base path on the laptop. The default destination is
        ``<laptop_base>/checkpoints/<run_name>/``. Ignored if
        ``remote_dir`` is set.
    remote_dir:
        If set, becomes the EXACT destination directory on the remote
        host (overrides the computed ``<laptop_base>/checkpoints/<run_name>``
        path). Use this to send to an external drive or a custom layout.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    run_name = run_dir.name

    # Two supported layouts:
    #   1. nightly tonight-script runs   → <run-dir>/policy-<arch>/checkpoints/
    #   2. autoresearch trial dirs       → <run-dir>/checkpoints/
    policy_dirs = sorted(p for p in run_dir.glob("policy-*") if p.is_dir())
    if policy_dirs:
        ckpts_dir = policy_dirs[-1] / "checkpoints"
        layout_hint = f"under {policy_dirs[-1]}"
    elif (run_dir / "checkpoints").is_dir():
        ckpts_dir = run_dir / "checkpoints"
        layout_hint = f"under {run_dir}"
    else:
        raise FileNotFoundError(
            f"no checkpoints found under {run_dir}: expected either "
            f"policy-*/checkpoints/ (nightly layout) or checkpoints/ "
            f"(autoresearch trial layout)"
        )
    if not ckpts_dir.is_dir():
        raise FileNotFoundError(f"no checkpoints/ {layout_hint}")

    last = ckpts_dir / "last"
    if not (last / "pretrained_model").is_dir():
        numbered = sorted(
            d for d in ckpts_dir.iterdir() if (d / "pretrained_model").is_dir()
        )
        if not numbered:
            raise FileNotFoundError(f"no pretrained_model dirs under {ckpts_dir}")
        last = numbered[-1]

    if remote_dir is not None:
        remote_base_path = remote_dir.rstrip("/")
    else:
        remote_base_path = f"{laptop_base.rstrip('/')}/models/{run_name}"
    dst_base = f"{host}:{remote_base_path}"

    # Pre-flight: ensure the destination dir exists on the remote. Closes
    # the gap left by rsync only auto-creating ONE level of parents.
    rc = _ensure_remote_dir(
        host,
        f"{remote_base_path}/{last.name}/pretrained_model",
        dry_run=dry_run,
    )
    if rc != 0:
        return rc

    rc = _run_rsync(
        f"{last}/pretrained_model/",
        f"{dst_base}/{last.name}/pretrained_model/",
        dry_run=dry_run,
    )
    if rc != 0:
        return rc

    manifest = run_dir / "dashboard" / "manifest.json"
    if manifest.exists():
        _run_rsync(str(manifest), f"{dst_base}/manifest.json", dry_run=dry_run)

    return rc


# --------------------------------------------------------------------------- #
# laptop → desktop: eval JSON pull
# --------------------------------------------------------------------------- #


def sync_eval_from_laptop(
    desktop_eval_dir: Path,
    *,
    host: str = DEFAULT_LAPTOP_HOST,
    laptop_eval_dir: str = "~/outputs/eval/",
    dry_run: bool = False,
) -> int:
    """Pull all closed-loop eval JSONs back to the desktop's outputs/eval/."""
    desktop_eval_dir = Path(desktop_eval_dir).resolve()
    desktop_eval_dir.mkdir(parents=True, exist_ok=True)
    return _run_rsync(
        f"{host}:{laptop_eval_dir}",
        f"{desktop_eval_dir}/",
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #


def build_sync_ckpt_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="li-deploy-sync-ckpt",
                                description="desktop → laptop ckpt sync")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--host", default=DEFAULT_LAPTOP_HOST)
    p.add_argument("--laptop-base", default=DEFAULT_LAPTOP_BASE,
                   help="base path on the remote; ckpts land in "
                        "<laptop-base>/models/<run_name>/ (default: "
                        f"{DEFAULT_LAPTOP_BASE})")
    p.add_argument("--remote-dir",
                   help="EXACT remote destination dir (overrides --laptop-base "
                        "+ the auto-generated 'models/<run_name>' suffix). Use "
                        "to send to an external drive, e.g. /mnt/nvme/models/foo")
    p.add_argument("--dry-run", action="store_true")
    return p


def build_sync_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="li-deploy-sync-eval",
                                description="laptop → desktop eval JSON pull")
    p.add_argument("--desktop-eval-dir", default="outputs/eval/")
    p.add_argument("--host", default=DEFAULT_LAPTOP_HOST)
    p.add_argument("--laptop-eval-dir", default="~/outputs/eval/")
    p.add_argument("--dry-run", action="store_true")
    return p
