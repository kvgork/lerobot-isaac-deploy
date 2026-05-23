"""Sync helpers — desktop↔laptop rsync wrappers.

Two directions:

* ``sync_ckpt_to_laptop`` — runs on DESKTOP, copies a checkpoint dir
  (and optional dashboard manifest + LeRobotDataset) to the laptop's
  deploy workspace.
* ``sync_eval_from_laptop`` — runs on DESKTOP, pulls closed-loop eval
  JSONs back so the dashboard's Evaluation tab picks them up.

Both are thin argparse wrappers around `rsync` over SSH. The functions
are importable so the CLI can also re-use them programmatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
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


def _resolve_run_dir_from_winner(winner_json: Path) -> Path:
    """Read winner.json, return the training-run dir hosting the winner ckpt.

    Walks the ``winner_policy_path`` up until it leaves the
    ``pretrained_model/<ckpt-id>/checkpoints[/policy-<arch>]`` suffix.
    Supports both layouts handled by ``sync_ckpt_to_laptop``.
    """
    data = json.loads(Path(winner_json).read_text())
    ckpt = Path(data["winner_policy_path"]).resolve()
    if ckpt.name == "pretrained_model":
        ckpt = ckpt.parent                              # → <ckpt-id>/
    if ckpt.parent.name == "checkpoints":
        run_dir = ckpt.parent.parent                    # → <run> OR policy-<arch>/
        if run_dir.name.startswith("policy-"):
            run_dir = run_dir.parent                    # nightly layout: → <run>/
        return run_dir
    raise ValueError(
        f"unrecognized winner_policy_path layout: {data['winner_policy_path']!r}. "
        f"Expected …/checkpoints/<ckpt-id>/pretrained_model/."
    )


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
    winner_json: Path | None = None,
    dataset_root: Path | None = None,
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
        Dataset destination is independent of ``remote_dir`` — datasets
        always land under ``<laptop_base>/datasets/<basename>/`` since
        multiple ckpts often share a dataset.
    winner_json:
        Optional path to the desktop winner.json. When set, a rewritten
        copy is shipped alongside the ckpt with ``winner_policy_path``
        pointing at the laptop-local layout. When ``dataset_root`` is
        also set, the rewritten JSON gains a ``dataset_root`` field
        pointing at the laptop-local dataset path.
    dataset_root:
        Optional path to the LeRobotDataset tree the ckpt was trained
        on. When set, rsync the tree to
        ``<laptop_base>/datasets/<basename>/`` after the ckpt rsync.
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

    # Dataset rsync — independent of remote_dir, lands at
    # <laptop_base>/datasets/<basename>/. Performed BEFORE the winner.json
    # rewrite so the rewritten JSON can record the laptop-local path.
    laptop_dataset_path: str | None = None
    if dataset_root is not None:
        ds_path = Path(dataset_root).resolve()
        if not ds_path.is_dir():
            print(
                f"[sync] WARN: --dataset-root not a directory, skipping: {ds_path}",
                flush=True,
            )
        else:
            ds_basename = ds_path.name
            laptop_dataset_path = (
                f"{laptop_base.rstrip('/')}/datasets/{ds_basename}"
            )
            rc = _ensure_remote_dir(host, laptop_dataset_path, dry_run=dry_run)
            if rc != 0:
                return rc
            rc = _run_rsync(
                f"{ds_path}/",
                f"{host}:{laptop_dataset_path}/",
                dry_run=dry_run,
            )
            if rc != 0:
                return rc
            print(
                f"[sync] dataset → {host}:{laptop_dataset_path}/",
                flush=True,
            )

    # If a winner.json was provided, write a laptop-local copy with the
    # `winner_policy_path` rewritten to the destination layout. The
    # source winner.json has desktop-absolute paths under
    # `outputs/autoresearch-*/trial_N/checkpoints/<ckpt>/pretrained_model`
    # which do NOT exist on the laptop. The rewritten copy lets the
    # laptop run `pixi run session --winner models/<run>/winner.json`
    # directly without manual path edits.
    if winner_json is not None:
        wj_path = Path(winner_json).resolve()
        if not wj_path.is_file():
            print(f"[sync] WARN: --winner-json not found: {wj_path}", flush=True)
        else:
            laptop_ckpt = f"{remote_base_path}/{last.name}/pretrained_model"
            data = json.loads(wj_path.read_text())
            data["winner_policy_path"] = laptop_ckpt
            if laptop_dataset_path is not None:
                data["dataset_root"] = laptop_dataset_path
            data["_source_winner_json"] = str(wj_path)
            data["_synced_at"] = datetime.utcnow().isoformat() + "Z"
            tmp = Path("/tmp") / f"winner-{run_name}-{os.getpid()}.json"
            tmp.write_text(json.dumps(data, indent=2))
            rc = _run_rsync(str(tmp), f"{dst_base}/winner.json", dry_run=dry_run)
            if not dry_run:
                tmp.unlink(missing_ok=True)
            if rc != 0:
                return rc
            extras = (
                f" + dataset_root={laptop_dataset_path}"
                if laptop_dataset_path is not None
                else ""
            )
            print(
                f"[sync] wrote laptop winner.json → {dst_base}/winner.json "
                f"(policy_path rewritten to {laptop_ckpt}{extras})",
                flush=True,
            )

    return rc


# --------------------------------------------------------------------------- #
# desktop → laptop: world-model checkpoint shipping
# --------------------------------------------------------------------------- #


def _stage_wm_ckpt(
    run_dir: Path, hydra_cfg_dir: Path, stage_dir: Path
) -> tuple[Path, Path]:
    """Stage a sheeprl run dir into the deploy-format layout the laptop
    expects: ``<stage>/.hydra/config.yaml`` + ``<stage>/checkpoint/ckpt_*.ckpt``.

    sheeprl writes ckpts under
    ``logs/runs/<algo>/<env>/<run_name>/version_*/checkpoint/ckpt_*.ckpt``
    while ``detect_policy_kind`` expects ``.hydra/config.yaml`` co-located
    with the ckpt. This staging is idempotent — caller may rerun safely.

    Returns
    -------
    (staged_ckpt_path, staged_config_yaml)
        Absolute paths under ``stage_dir``.
    """
    ckpts = sorted(run_dir.glob("version_*/checkpoint/ckpt_*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"no ckpt_*.ckpt under {run_dir} — sheeprl must save at least "
            f"one checkpoint before sync. Set checkpoint.every to a value "
            f"≤ algo.total_steps in the training cmd."
        )
    src_ckpt = ckpts[-1]  # highest step

    cfg_src = hydra_cfg_dir / "config.yaml"
    if not cfg_src.is_file():
        raise FileNotFoundError(
            f"hydra config.yaml missing at {cfg_src}. Pass --hydra-cfg-dir "
            f"pointing at the Hydra run dir for this trial."
        )

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / ".hydra").mkdir(exist_ok=True)
    (stage_dir / "checkpoint").mkdir(exist_ok=True)

    import shutil
    shutil.copy2(cfg_src, stage_dir / ".hydra" / "config.yaml")
    dst_ckpt = stage_dir / "checkpoint" / src_ckpt.name
    if not dst_ckpt.exists() or src_ckpt.stat().st_mtime > dst_ckpt.stat().st_mtime:
        shutil.copy2(src_ckpt, dst_ckpt)
    return dst_ckpt, stage_dir / ".hydra" / "config.yaml"


def sync_wm_ckpt_to_laptop(
    sheeprl_run_dir: Path,
    *,
    hydra_cfg_dir: Path,
    host: str = DEFAULT_LAPTOP_HOST,
    laptop_base: str = DEFAULT_LAPTOP_BASE,
    remote_dir: str | None = None,
    label: str | None = None,
    metadata_files: list[Path] | None = None,
    dry_run: bool = False,
    stage_dir: Path | None = None,
) -> int:
    """Stage + ship a sheeprl world-model checkpoint to the laptop.

    Layout on the laptop after sync (matches ``detect_policy_kind``'s
    DreamerV3 contract):

        <laptop_base>/checkpoints/wm/<label>/
            .hydra/config.yaml
            checkpoint/ckpt_<step>_<rank>.ckpt
            best.json                                (if provided)
            eval_rollout.json                        (if provided)

    Parameters
    ----------
    sheeprl_run_dir
        ``logs/runs/<algo>/<env>/<run_name>/`` — sheeprl-managed dir
        containing ``version_*/checkpoint/ckpt_*.ckpt``.
    hydra_cfg_dir
        Dir holding the Hydra ``config.yaml`` (training-time
        ``hydra.run.dir``). Required because sheeprl writes ckpts and
        Hydra config to DIFFERENT roots.
    host
        SSH alias for the laptop.
    laptop_base
        Base path on the laptop. Ignored when ``remote_dir`` is set.
    remote_dir
        EXACT destination dir; overrides the auto-generated
        ``<laptop_base>/checkpoints/wm/<label>/`` path.
    label
        Subdir name on the laptop. Defaults to ``sheeprl_run_dir.name``.
    metadata_files
        Extra JSON files (best.json, eval_rollout.json, …) to ship into
        the staged dir.
    dry_run
        Pass ``--dry-run`` to rsync and skip the SSH mkdir.
    stage_dir
        Where to stage the deploy-format layout BEFORE rsync. Default:
        ``outputs/wm_for_laptop/<label>/``.

    Returns
    -------
    rsync exit code (0 = success).
    """
    sheeprl_run_dir = Path(sheeprl_run_dir).resolve()
    hydra_cfg_dir = Path(hydra_cfg_dir).resolve()
    if not sheeprl_run_dir.is_dir():
        raise FileNotFoundError(f"sheeprl run dir not found: {sheeprl_run_dir}")
    if not hydra_cfg_dir.is_dir():
        raise FileNotFoundError(f"hydra cfg dir not found: {hydra_cfg_dir}")

    label = label or sheeprl_run_dir.name
    if stage_dir is None:
        stage_dir = Path("outputs/wm_for_laptop") / label
    stage_dir = Path(stage_dir).resolve()

    _stage_wm_ckpt(sheeprl_run_dir, hydra_cfg_dir, stage_dir)

    if metadata_files:
        import shutil
        for meta_src in metadata_files:
            meta_src = Path(meta_src)
            if meta_src.is_file():
                shutil.copy2(meta_src, stage_dir / meta_src.name)

    if remote_dir is not None:
        remote_base_path = remote_dir.rstrip("/")
    else:
        remote_base_path = f"{laptop_base.rstrip('/')}/checkpoints/wm/{label}"
    dst = f"{host}:{remote_base_path}/"

    rc = _ensure_remote_dir(host, remote_base_path, dry_run=dry_run)
    if rc != 0:
        return rc
    return _run_rsync(f"{stage_dir}/", dst, dry_run=dry_run)


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
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir",
                    help="training run dir (auto-discovers latest ckpt inside)")
    src.add_argument("--winner",
                    help="winner.json from an autoresearch sweep "
                         "(auto-resolves --run-dir from winner_policy_path)")
    p.add_argument("--host", default=DEFAULT_LAPTOP_HOST)
    p.add_argument("--laptop-base", default=DEFAULT_LAPTOP_BASE,
                   help="base path on the remote; ckpts land in "
                        "<laptop-base>/models/<run_name>/ (default: "
                        f"{DEFAULT_LAPTOP_BASE})")
    p.add_argument("--remote-dir",
                   help="EXACT remote destination dir (overrides --laptop-base "
                        "+ the auto-generated 'models/<run_name>' suffix). Use "
                        "to send to an external drive, e.g. /mnt/nvme/models/foo")
    p.add_argument("--winner-json",
                   help="path to the desktop winner.json to also ship alongside "
                        "the ckpt with `winner_policy_path` rewritten for the "
                        "laptop layout. When --winner is the source flag, this "
                        "is set automatically by the CLI.")
    p.add_argument("--dataset-root",
                   help="path to the LeRobotDataset tree the ckpt was trained "
                        "on. When set, rsync it to "
                        "<laptop-base>/datasets/<basename>/ alongside the ckpt. "
                        "The rewritten winner.json (if --winner / --winner-json "
                        "is also set) gains a `dataset_root` field that "
                        "session.py picks up automatically.")
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


def build_sync_wm_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="li-deploy-sync-wm",
        description="desktop → laptop world-model checkpoint sync "
                    "(stages sheeprl run dir into deploy-format layout)",
    )
    p.add_argument(
        "--sheeprl-run-dir",
        required=True,
        help="logs/runs/<algo>/<env>/<run_name>/ dir containing "
             "version_*/checkpoint/ckpt_*.ckpt",
    )
    p.add_argument(
        "--hydra-cfg-dir",
        required=True,
        help="Hydra working dir holding config.yaml. For an autoresearch "
             "trial this is `outputs/autoresearch-<slug>/trial_<i>/.hydra/`.",
    )
    p.add_argument("--host", default=DEFAULT_LAPTOP_HOST)
    p.add_argument(
        "--laptop-base",
        default=DEFAULT_LAPTOP_BASE,
        help=f"base path on the remote (default: {DEFAULT_LAPTOP_BASE})",
    )
    p.add_argument(
        "--remote-dir",
        help="EXACT destination dir (overrides --laptop-base + the auto "
             "`/checkpoints/wm/<label>/` suffix). For external drives.",
    )
    p.add_argument(
        "--label",
        help="subdir name on the laptop (default: sheeprl run dir basename)",
    )
    p.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="extra JSON file to ship next to the ckpt (repeatable). "
             "Typical: best.json + eval_rollout.json from the autoresearch "
             "state dir.",
    )
    p.add_argument(
        "--stage-dir",
        help="where to stage the deploy-format layout before rsync "
             "(default: outputs/wm_for_laptop/<label>/)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p
