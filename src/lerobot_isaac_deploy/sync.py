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
    args = ["rsync", "-avhP"]
    if dry_run:
        args.append("--dry-run")
    args += [src, dst]
    print(f"[sync] $ {' '.join(args)}", flush=True)
    return subprocess.run(args, check=False).returncode


# --------------------------------------------------------------------------- #
# desktop → laptop: checkpoint shipping
# --------------------------------------------------------------------------- #


def sync_ckpt_to_laptop(
    run_dir: Path,
    *,
    host: str = DEFAULT_LAPTOP_HOST,
    laptop_base: str = DEFAULT_LAPTOP_BASE,
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
        Base path on the laptop where ckpts land. Will create
        ``<base>/checkpoints/<run_name>/`` on first sync.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    run_name = run_dir.name

    # Find the policy dir (policy-smolvla, policy-diffusion, etc.)
    policy_dirs = sorted(p for p in run_dir.glob("policy-*") if p.is_dir())
    if not policy_dirs:
        raise FileNotFoundError(
            f"no policy-* directory under {run_dir} (expected policy-smolvla/, "
            f"policy-diffusion/, …)"
        )
    policy_dir = policy_dirs[-1]
    ckpts_dir = policy_dir / "checkpoints"
    if not ckpts_dir.is_dir():
        raise FileNotFoundError(f"no checkpoints/ under {policy_dir}")

    last = ckpts_dir / "last"
    if not (last / "pretrained_model").is_dir():
        numbered = sorted(
            d for d in ckpts_dir.iterdir() if (d / "pretrained_model").is_dir()
        )
        if not numbered:
            raise FileNotFoundError(f"no pretrained_model dirs under {ckpts_dir}")
        last = numbered[-1]

    dst_base = f"{host}:{laptop_base}/checkpoints/{run_name}"
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
    p.add_argument("--laptop-base", default=DEFAULT_LAPTOP_BASE)
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
