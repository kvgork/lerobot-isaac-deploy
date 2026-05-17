"""Console-script entries.

One umbrella entry (``lerobot-isaac-deploy``) plus three direct
shortcuts that mirror the bash scripts they replace.
"""

from __future__ import annotations

import argparse
import sys


def session_main(argv: list[str] | None = None) -> int:
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
    return sync_ckpt_to_laptop(
        run_dir,
        host=ns.host,
        laptop_base=ns.laptop_base,
        remote_dir=ns.remote_dir,
        winner_json=winner_json,
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


def wm_rollout_main(argv: list[str] | None = None) -> int:
    from lerobot_isaac_deploy.wm_rollout import main as _rollout

    return _rollout(argv)


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
    "kind":       (kind_main,       "Detect what kind of checkpoint a directory holds"),
    "sync-ckpt":  (sync_ckpt_main,  "Desktop → laptop ckpt sync (run on desktop)"),
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
