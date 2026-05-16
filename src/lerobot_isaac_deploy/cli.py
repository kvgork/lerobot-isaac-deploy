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
    from lerobot_isaac_deploy.sync import build_sync_ckpt_parser, sync_ckpt_to_laptop
    from pathlib import Path

    ns = build_sync_ckpt_parser().parse_args(argv)
    return sync_ckpt_to_laptop(
        Path(ns.run_dir),
        host=ns.host,
        laptop_base=ns.laptop_base,
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


# --------------------------------------------------------------------------- #
# Umbrella entry
# --------------------------------------------------------------------------- #

_SUBCOMMANDS = {
    "session":    (session_main,   "Run the confirm-gated deploy ladder"),
    "sync-ckpt":  (sync_ckpt_main, "Desktop → laptop ckpt sync (run on desktop)"),
    "sync-eval":  (sync_eval_main, "Laptop → desktop eval JSON pull (run on desktop)"),
    "bootstrap":  (bootstrap_main, "One-shot laptop env setup"),
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
