"""wm-rollout — offline state-prediction rollout for a world model.

Loads a WM checkpoint, feeds it observation sequences from a
LeRobotDataset, runs the world model forward step-by-step, and writes:

* ``next_state_pred.npz``   — predicted next-state arrays per timestep
* ``rollout_summary.json``  — mean reconstruction loss, length, etc.

No motors involved. CPU or GPU both work.

Supported checkpoints
---------------------
* DreamerV3 (sheeprl) — uses ``world_model.encoder`` + ``world_model.rssm``
  + ``world_model.decoder`` to roll forward in latent space and decode
  back to image-space predictions.
* LeWM (HF) — uses ``LeWorldModel.predict`` (per-step or chunked, depending
  on the model class).

Implementation status
---------------------
Stub. The function shape + CLI flags are locked; the per-architecture
rollout body is a small TODO once a real checkpoint exists. Each
TODO raises ``NotImplementedError`` with a clear message so the user
knows what's missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rollout(
    checkpoint_path: Path | str,
    *,
    dataset_root: Path | str,
    output_dir: Path | str,
    horizon_steps: int = 50,
    n_seed_episodes: int = 1,
    device: str | None = None,
) -> Path:
    """Run an offline state-prediction rollout. Returns the summary JSON path.

    Parameters
    ----------
    checkpoint_path:
        WM checkpoint dir (auto-detects dreamerv3 vs lewm).
    dataset_root:
        LeRobotDataset to seed the rollout (first frame of each episode).
    output_dir:
        Where ``next_state_pred.npz`` + ``rollout_summary.json`` land.
    horizon_steps:
        How many steps to roll forward from the seed frame. Default 50.
    n_seed_episodes:
        How many episodes to seed from. Default 1.
    """
    from lerobot_isaac_deploy.policy_kind import detect_policy_kind

    cp = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kind = detect_policy_kind(cp)
    if kind == "dreamerv3":
        return _rollout_dreamerv3(cp, Path(dataset_root), output_dir,
                                  horizon_steps, n_seed_episodes, device)
    if kind == "lewm":
        return _rollout_lewm(cp, Path(dataset_root), output_dir,
                             horizon_steps, n_seed_episodes, device)
    raise RuntimeError(
        f"wm-rollout: unknown checkpoint kind at {cp} (detected={kind})"
    )


def _rollout_dreamerv3(
    cp: Path, ds_root: Path, out_dir: Path,
    horizon: int, n_seeds: int, device: str | None,
) -> Path:
    """Stub: DreamerV3 dream-rollout body."""
    raise NotImplementedError(
        "DreamerV3 dream-rollout body — implement after the first real "
        "sheeprl ckpt lands. Steps:\n"
        "  1. lerobot_isaac_deploy.wm_loader.load_dreamerv3(cp)\n"
        "  2. for each seed episode: encode first frame → init RSSM state\n"
        "  3. for t in range(horizon): step world_model.rssm with sampled action,\n"
        "     decode the predicted latent → reconstruct image / state\n"
        "  4. save the (T, 3, H, W) sequence to next_state_pred.npz"
    )


def _rollout_lewm(
    cp: Path, ds_root: Path, out_dir: Path,
    horizon: int, n_seeds: int, device: str | None,
) -> Path:
    """Stub: LeWM rollout body."""
    raise NotImplementedError(
        "LeWM rollout body — implement after the first real LeWorldModel "
        "ckpt lands. Steps:\n"
        "  1. Load the model class declared in policy.json\n"
        "  2. Feed N seed frames (window_size from training config)\n"
        "  3. Call model.predict(...) for `horizon` steps\n"
        "  4. Save outputs to next_state_pred.npz"
    )


# --------------------------------------------------------------------------- #
# argparse (used by cli.py wm-rollout subcommand)
# --------------------------------------------------------------------------- #


def build_rollout_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="li-deploy-wm-rollout",
        description="Offline state-prediction rollout for a world model.",
    )
    p.add_argument("--checkpoint", required=True, help="WM checkpoint dir")
    p.add_argument("--dataset", required=True, help="LeRobotDataset root")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--horizon-steps", type=int, default=50)
    p.add_argument("--n-seed-episodes", type=int, default=1)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_rollout_parser().parse_args(argv)
    try:
        out = rollout(
            ns.checkpoint,
            dataset_root=ns.dataset,
            output_dir=ns.output_dir,
            horizon_steps=ns.horizon_steps,
            n_seed_episodes=ns.n_seed_episodes,
            device=ns.device,
        )
        print(f"rollout wrote {out}")
        return 0
    except NotImplementedError as exc:
        print(f"wm-rollout: not yet implemented — {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"wm-rollout: error — {exc}")
        return 1
