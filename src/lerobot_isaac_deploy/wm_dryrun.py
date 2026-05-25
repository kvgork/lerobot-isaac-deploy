"""wm-dryrun — standalone dry-run path for a DreamerV3 actor.

No hardware. No camera. No serial port.

Steps
-----
1. Load the WM checkpoint via :func:`~lerobot_isaac_deploy.wm_loader.load_dreamerv3`.
2. Construct N synthetic observations:
   - ``state``: random float32 values drawn from the training-time joint-limit
     range (approximately [-π, π] per joint for SO-101).
   - ``rgb``: random uint8 (3, image_size, image_size) — the WM was trained on
     zero-RGB (cameras were off), so random-RGB will also be OOD. That's
     expected for this diagnostic step.
3. Call ``actor.select_action(obs)`` for each.
4. Print per-joint action statistics: shape, min/max/mean/std.
5. Write a JSON report to ``outputs/wm-dryrun-<timestamp>/report.json``.

Exit codes
----------
0  — all N actions collected; report.json written.
1  — ckpt load failed or inference failed.
2  — missing dependency (torch / sheeprl).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Core logic (importable for tests)
# ---------------------------------------------------------------------------


def run_dryrun(
    checkpoint_path: Path | str,
    *,
    config_yaml: Path | str | None = None,
    n_samples: int = 100,
    output_dir: Path | str | None = None,
    device: str | None = None,
    image_size: int = 64,
    state_dim: int = 13,
    seed: int = 42,
) -> dict:
    """Run the dry-run and return the report dict.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``ckpt_*.ckpt`` file or a run directory.
    config_yaml:
        Optional explicit path to the Hydra/Lightning config YAML.
    n_samples:
        Number of synthetic observations to feed through the actor.
    output_dir:
        Where to write ``report.json``.  Default:
        ``outputs/wm-dryrun-<timestamp>/``.
    device:
        ``"cpu"`` / ``"cuda"`` / ``None`` (auto).
    image_size:
        Spatial size of the synthetic RGB image (pixels, square).
    state_dim:
        Joint-state vector length. Default 13 matches SO-101 Isaac setup
        (6 joint_pos + 7 object_pose). Override when deploying a different
        observation.
    seed:
        RNG seed for reproducible synthetic obs.

    Returns
    -------
    dict
        Report dict also written to ``output_dir/report.json``.
    """
    from lerobot_isaac_deploy.wm_loader import load_dreamerv3, _SyntheticActor

    cp = Path(checkpoint_path)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    if output_dir is None:
        output_dir = Path("outputs") / f"wm-dryrun-{ts}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load actor -----------------------------------------------------------
    print(f"[wm-dryrun] loading checkpoint: {cp}", flush=True)
    actor = load_dreamerv3(cp, config_yaml=config_yaml, device=device)
    is_synthetic = isinstance(actor, _SyntheticActor)
    resolved_device = actor.device if not is_synthetic else "cpu"
    print(
        f"[wm-dryrun] loaded actor kind={actor.kind!r} "
        f"device={resolved_device!r} synthetic={is_synthetic}",
        flush=True,
    )

    # Determine actual action_dim from actor
    action_dim = getattr(actor, "action_dim", 6)

    # Build synthetic observations -----------------------------------------
    rng = np.random.default_rng(seed)
    actions_collected: list[np.ndarray] = []
    actor.reset()

    for i in range(n_samples):
        # State: random values in [-pi, pi] (plausible joint-angle range)
        state = rng.uniform(-np.pi, np.pi, size=(state_dim,)).astype(np.float32)
        # RGB: random uint8 image
        rgb = rng.integers(0, 256, size=(3, image_size, image_size), dtype=np.uint8)

        obs = {"state": state, "rgb": rgb}
        action = actor.select_action(obs)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        actions_collected.append(action)

        if i < 3 or i == n_samples - 1:
            print(
                f"[wm-dryrun] step {i:4d}: action={[round(float(a), 4) for a in action]}",
                flush=True,
            )

    # Compute statistics ---------------------------------------------------
    actions_arr = np.stack(actions_collected, axis=0)  # (N, action_dim)
    per_joint_stats = []
    for j in range(actions_arr.shape[1]):
        col = actions_arr[:, j]
        per_joint_stats.append(
            {
                "joint": j,
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std()),
            }
        )

    print("\n[wm-dryrun] action statistics:", flush=True)
    for s in per_joint_stats:
        print(
            f"  joint {s['joint']}: "
            f"mean={s['mean']:+.4f}  std={s['std']:.4f}  "
            f"[{s['min']:+.4f}, {s['max']:+.4f}]",
            flush=True,
        )

    # Acceptance checks ----------------------------------------------------
    all_finite = bool(np.all(np.isfinite(actions_arr)))
    in_range = bool(np.all(actions_arr >= -1.0) and np.all(actions_arr <= 1.0))
    shape_ok = actions_arr.shape[1] == action_dim

    print(
        f"\n[wm-dryrun] checks: finite={all_finite}  "
        f"in[-1,1]={in_range}  shape_ok={shape_ok}",
        flush=True,
    )

    # Write report ---------------------------------------------------------
    report = {
        "checkpoint": str(cp),
        "config_yaml": str(config_yaml) if config_yaml else None,
        "n_samples": n_samples,
        "action_dim": int(action_dim),
        "action_shape": list(actions_arr.shape),
        "state_dim_used": int(state_dim),
        "image_size_used": int(image_size),
        "synthetic_actor": is_synthetic,
        "device": resolved_device,
        "all_finite": all_finite,
        "in_range_neg1_1": in_range,
        "shape_ok": shape_ok,
        "per_joint": per_joint_stats,
        "global_mean": float(actions_arr.mean()),
        "global_std": float(actions_arr.std()),
        "global_min": float(actions_arr.min()),
        "global_max": float(actions_arr.max()),
        "timestamp": ts,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[wm-dryrun] report written: {report_path}", flush=True)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_dryrun_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="li-deploy-wm-dryrun",
        description=(
            "DreamerV3 actor dry-run: load ckpt, feed N synthetic obs, "
            "print action stats, write report.json. No hardware required."
        ),
    )
    p.add_argument(
        "--policy-path",
        required=True,
        help="ckpt_*.ckpt file or run directory containing one",
    )
    p.add_argument(
        "--config-yaml",
        default=None,
        help="Explicit path to Hydra/Lightning config.yaml (auto-discovered if omitted)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of synthetic observations to run through the actor (default 100)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write report.json (default: outputs/wm-dryrun-<timestamp>/)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cpu / cuda (default: auto)",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Spatial size (pixels) of synthetic RGB obs (default 64)",
    )
    p.add_argument(
        "--state-dim",
        type=int,
        default=13,
        help=(
            "State vector dim for synthetic obs. Default 13 = SO-101 "
            "(6 joint_pos + 7 object_pose). Use 6 for joints-only configs."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_dryrun_parser().parse_args(argv)
    try:
        report = run_dryrun(
            ns.policy_path,
            config_yaml=ns.config_yaml,
            n_samples=ns.n_samples,
            output_dir=ns.output_dir,
            device=ns.device,
            image_size=ns.image_size,
            state_dim=ns.state_dim,
            seed=ns.seed,
        )
    except ImportError as exc:
        print(
            f"[wm-dryrun] missing dependency — {exc}\n"
            "Install with: pip install 'sheeprl[dreamer]>=0.5' torch gymnasium",
            file=sys.stderr,
            flush=True,
        )
        return 2
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[wm-dryrun] error — {exc}", file=sys.stderr, flush=True)
        return 1

    passed = report["all_finite"] and report["in_range_neg1_1"] and report["shape_ok"]
    if passed:
        print("[wm-dryrun] PASS — actor emits valid actions end-to-end.", flush=True)
        return 0
    else:
        print(
            "[wm-dryrun] FAIL — one or more acceptance checks failed. "
            "See report.json for details.",
            file=sys.stderr,
            flush=True,
        )
        return 1
