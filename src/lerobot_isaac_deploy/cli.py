"""Console-script entries.

One umbrella entry (``lerobot-isaac-deploy``) plus three direct
shortcuts that mirror the bash scripts they replace.
"""

from __future__ import annotations

import logging
import os
import sys


def _quiet_hf_hub() -> None:
    """Force HF offline + drop request logs BEFORE any lerobot/hf_hub import.

    The SmolVLM2 backbone is already cached for deploy, but huggingface_hub
    HEAD/GETs the hub on every policy load (log flood + latency). Setting these
    env vars here — before DeploySession imports lerobot, and before the
    robot-data-run subprocess inherits os.environ — stops the requests. The
    subprocess reads HF_HUB_OFFLINE at its own startup. Override: HF_HUB_OFFLINE=0.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for name in ("httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def session_main(argv: list[str] | None = None) -> int:
    _quiet_hf_hub()
    from lerobot_isaac_deploy.session import (
        DeploySession,
        build_session_parser,
        cfg_from_namespace,
    )

    parser = build_session_parser()
    ns = parser.parse_args(argv)

    if ns.safety_ack_only:
        from lerobot_isaac_deploy.session import SessionConfig
        from pathlib import Path

        # Build a minimal cfg just to access the ack path helper.
        DeploySession(
            SessionConfig(policy_path=Path("/tmp/x"), dataset_root=Path("/tmp/x"))
        ).write_safety_ack()
        return 0

    cfg = cfg_from_namespace(ns)
    return DeploySession(cfg).run()


def sync_ckpt_main(argv: list[str] | None = None) -> int:
    from lerobot_isaac_deploy.sync import (
        _resolve_run_dir_from_winner,
        build_sync_ckpt_parser,
        sync_ckpt_to_laptop,
    )
    from pathlib import Path

    ns = build_sync_ckpt_parser().parse_args(argv)
    winner_json = None
    if ns.winner:
        run_dir = _resolve_run_dir_from_winner(Path(ns.winner))
        winner_json = Path(ns.winner)
        print(f"[sync] winner.json → run_dir: {run_dir}", flush=True)
    else:
        run_dir = Path(ns.run_dir)
    # Explicit --winner-json overrides the auto-derived one from --winner.
    if getattr(ns, "winner_json", None):
        winner_json = Path(ns.winner_json)
    dataset_root = Path(ns.dataset_root) if getattr(ns, "dataset_root", None) else None
    return sync_ckpt_to_laptop(
        run_dir,
        host=ns.host,
        laptop_base=ns.laptop_base,
        remote_dir=ns.remote_dir,
        winner_json=winner_json,
        dataset_root=dataset_root,
        dry_run=ns.dry_run,
    )


def sync_eval_main(argv: list[str] | None = None) -> int:
    from lerobot_isaac_deploy.sync import build_sync_eval_parser, sync_eval_from_laptop
    from pathlib import Path

    ns = build_sync_eval_parser().parse_args(argv)
    return sync_eval_from_laptop(
        Path(ns.desktop_eval_dir),
        host=ns.host,
        laptop_eval_dir=ns.laptop_eval_dir,
        dry_run=ns.dry_run,
    )


def bootstrap_main(argv: list[str] | None = None) -> int:
    from lerobot_isaac_deploy.bootstrap import main as _bootstrap

    return _bootstrap(argv)


def sync_wm_main(argv: list[str] | None = None) -> int:
    """`li-deploy-sync-wm` — push a sheeprl WM checkpoint to the laptop.

    Stages the sheeprl run dir into the deploy-format layout that
    ``detect_policy_kind`` recognises as ``dreamerv3``
    (``<root>/.hydra/config.yaml`` + ``<root>/checkpoint/ckpt_*.ckpt``)
    and rsyncs the staged tree to ``<laptop_base>/checkpoints/wm/<label>/``.
    """
    from pathlib import Path
    from lerobot_isaac_deploy.sync import build_sync_wm_parser, sync_wm_ckpt_to_laptop

    ns = build_sync_wm_parser().parse_args(argv)
    return sync_wm_ckpt_to_laptop(
        Path(ns.sheeprl_run_dir),
        hydra_cfg_dir=Path(ns.hydra_cfg_dir),
        host=ns.host,
        laptop_base=ns.laptop_base,
        remote_dir=ns.remote_dir,
        label=ns.label,
        metadata_files=[Path(p) for p in ns.metadata] if ns.metadata else None,
        stage_dir=Path(ns.stage_dir) if ns.stage_dir else None,
        dry_run=ns.dry_run,
    )


def wm_rollout_main(argv: list[str] | None = None) -> int:
    from lerobot_isaac_deploy.wm_rollout import main as _rollout

    return _rollout(argv)


def wm_dryrun_main(argv: list[str] | None = None) -> int:
    """`lerobot-isaac-deploy wm-dryrun` — dry-run DreamerV3 actor on synthetic obs.

    Loads a sheeprl checkpoint, feeds N random observations through the actor,
    prints per-joint action statistics, and writes a report.json.
    No hardware, no camera, no serial port required.
    """
    from lerobot_isaac_deploy.wm_dryrun import main as _dryrun

    return _dryrun(argv)


def kind_main(argv: list[str] | None = None) -> int:
    """`lerobot-isaac-deploy kind <PATH>` — print detected ckpt kind."""
    import sys
    from lerobot_isaac_deploy.policy_kind import detect_policy_kind, explain

    if not argv:
        print("usage: lerobot-isaac-deploy kind <CHECKPOINT_DIR>", file=sys.stderr)
        return 2
    kind = detect_policy_kind(argv[0])
    print(f"{kind}\t{explain(kind)}")
    return 0


# --------------------------------------------------------------------------- #
# Umbrella entry
# --------------------------------------------------------------------------- #

_SUBCOMMANDS = {
    "session":    (session_main,    "Run the confirm-gated deploy ladder (LeRobot or DreamerV3-actor)"),
    "wm-rollout": (wm_rollout_main, "Offline state-prediction rollout (DreamerV3 / LeWM); no robot"),
    "wm-dryrun":  (wm_dryrun_main,  "DreamerV3 actor dry-run: load ckpt + run N synthetic obs; no robot"),
    "kind":       (kind_main,       "Detect what kind of checkpoint a directory holds"),
    "sync-ckpt":  (sync_ckpt_main,  "Desktop → laptop ckpt sync (run on desktop)"),
    "sync-wm":    (sync_wm_main,    "Desktop → laptop world-model ckpt sync (run on desktop)"),
    "sync-eval":  (sync_eval_main,  "Laptop → desktop eval JSON pull (run on desktop)"),
    "bootstrap":  (bootstrap_main,  "One-shot laptop env setup"),
}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: lerobot-isaac-deploy <subcommand> [args]\n\nSubcommands:")
        for name, (_, doc) in _SUBCOMMANDS.items():
            print(f"  {name:12s}  {doc}")
        return 0
    sub = argv[0]
    if sub not in _SUBCOMMANDS:
        print(f"unknown subcommand: {sub!r}", file=sys.stderr)
        return 2
    return _SUBCOMMANDS[sub][0](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
