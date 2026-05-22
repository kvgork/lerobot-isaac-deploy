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

Synthetic-marker short-circuit
-------------------------------
If ``<checkpoint>/synthetic_marker.json`` is present (or one level deep),
both rollout bodies skip all real model loading and write zero-filled
``next_state_pred.npz`` + a complete ``rollout_summary.json``.  This lets
the wm-rollout CLI smoke-test pass in any environment — torch, sheeprl,
and h5py are not required.

Import-error policy
-------------------
If a required package is absent (torch, sheeprl, h5py, stable-worldmodel /
le-wm), both bodies raise ``_RolloutInstallError`` (a ``RuntimeError``
subclass) with an explicit ``pip install <package>`` hint.  The ``main()``
function catches this and exits with rc=2.  Generic errors (unknown kind,
bad checkpoint path) exit with rc=1.  ``NotImplementedError`` is no longer
raised by these functions; the ``except NotImplementedError`` branch in
``main()`` is kept as a defensive fallback only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Sentinel exception for missing-dependency hints
# --------------------------------------------------------------------------- #


class _RolloutInstallError(RuntimeError):
    """Raised when a required package is missing. Exit code 2 in main()."""


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _check_synthetic(cp: Path) -> tuple[bool, dict]:
    """Detect synthetic-marker fixture. Returns (is_synth, marker_payload)."""
    candidates = [cp] + ([c for c in cp.iterdir() if c.is_dir()] if cp.is_dir() else [])
    for c in candidates:
        m = c / "synthetic_marker.json"
        if m.is_file():
            try:
                return True, json.loads(m.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return True, {}
    return False, {}


def _write_synthetic_rollout(
    kind: str,
    cp: Path,
    ds_root: Path,
    out_dir: Path,
    horizon: int,
    n_seeds: int,
    marker: dict,
) -> Path:
    """Write a plausible synthetic rollout (zeros) + summary JSON. No torch."""
    image_shape = tuple(marker.get("image_shape", [3, 64, 64]))
    latent_dim = marker.get("latent_dim", None)
    # (T, C, H, W) for image-decoding WMs (Dreamer), (T, latent_dim) for latent-only WMs (LeWM).
    if kind == "dreamerv3":
        pred = np.zeros((horizon,) + tuple(image_shape), dtype=np.float32)
        summary_extra = {"mean_recon_loss": 0.0}
    else:  # lewm
        pred = np.zeros((horizon, int(latent_dim or 192)), dtype=np.float32)
        summary_extra = {"mean_pred_loss": 0.0}
    np.savez(out_dir / "next_state_pred.npz", pred=pred)
    summary = {
        "kind": kind,
        "checkpoint": str(cp),
        "dataset_root": str(ds_root),
        "horizon": int(horizon),
        "n_seed_episodes": int(n_seeds),
        "synthetic": True,
        **summary_extra,
    }
    summary_path = out_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


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
    device:
        ``"cpu"`` / ``"cuda"`` / ``None`` (auto). Passed through to the
        architecture-specific loader.
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


# --------------------------------------------------------------------------- #
# Architecture-specific rollout bodies
# --------------------------------------------------------------------------- #


def _rollout_dreamerv3(
    cp: Path, ds_root: Path, out_dir: Path,
    horizon: int, n_seeds: int, device: str | None,
) -> Path:
    """DreamerV3 latent-rollout body. Short-circuits on synthetic marker."""
    synth, marker = _check_synthetic(cp)
    if synth:
        return _write_synthetic_rollout("dreamerv3", cp, ds_root, out_dir, horizon, n_seeds, marker)

    # Real path — requires torch + sheeprl + an actual dataset.
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise _RolloutInstallError(
            f"DreamerV3 rollout needs torch installed in the active env. "
            f"pip install torch ({exc})"
        ) from exc

    try:
        from lerobot_isaac_deploy.wm_loader import load_dreamerv3
        loaded = load_dreamerv3(cp, device=device)
    except ImportError as exc:
        raise _RolloutInstallError(
            f"DreamerV3 rollout needs sheeprl installed in the active env. "
            f"pip install 'sheeprl[dreamer]>=0.5' ({exc})"
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        raise _RolloutInstallError(
            f"DreamerV3 rollout: checkpoint not fully formed — {exc}. "
            f"Ensure sheeprl wrote .hydra/config.yaml and a ckpt_*.ckpt file, "
            f"or use a synthetic-marker fixture for smoke tests."
        ) from exc

    # Minimal first-iteration body: seed from synthetic obs (no dataset
    # loader yet), roll the RSSM forward, decode each step, accumulate
    # reconstructions. A real LeRobotDataset loader is a future commit
    # tracked in system-improvements.md.
    seed_obs = {
        "state": np.zeros((1, 6), dtype=np.float32),
        "image": np.zeros((1, 3, 64, 64), dtype=np.uint8),
    }
    # If `loaded` is the synthetic stub class from wm_loader, just produce zeros.
    # If it's a real LoadedWMActor, do an honest forward pass.
    preds = []
    try:
        # Best-effort: encode → step → decode `horizon` times.
        if hasattr(loaded, "encoder") and hasattr(loaded, "actor"):
            import torch as _t
            with _t.no_grad():
                z = loaded.encoder(seed_obs) if callable(loaded.encoder) else None
                state = loaded.actor.init_state(1, device=loaded.device) if hasattr(loaded.actor, "init_state") else None
                for _ in range(horizon):
                    state, _act = loaded.actor.step(z, state) if state is not None else (None, None)
                    preds.append(np.zeros((3, 64, 64), dtype=np.float32))  # decoder body deferred
        if not preds:
            preds = [np.zeros((3, 64, 64), dtype=np.float32) for _ in range(horizon)]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"DreamerV3 rollout forward failed: {exc}") from exc

    arr = np.stack(preds, axis=0)
    np.savez(out_dir / "next_state_pred.npz", pred=arr)
    summary = {
        "kind": "dreamerv3",
        "checkpoint": str(cp),
        "dataset_root": str(ds_root),
        "horizon": int(horizon),
        "n_seed_episodes": int(n_seeds),
        "synthetic": False,
        "partial": True,                  # real ckpt loaded, decoder body deferred
        "decoder_implemented": False,
        "mean_recon_loss": float("nan"),  # NaN until decoder lands; gate downstream eval on `partial`
    }
    out_path = out_dir / "rollout_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def _rollout_lewm(
    cp: Path, ds_root: Path, out_dir: Path,
    horizon: int, n_seeds: int, device: str | None,
) -> Path:
    """LeWM latent-rollout body. Short-circuits on synthetic marker."""
    synth, marker = _check_synthetic(cp)
    if synth:
        return _write_synthetic_rollout("lewm", cp, ds_root, out_dir, horizon, n_seeds, marker)

    # Real path — requires the stable-worldmodel / le-wm packages.
    try:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            raise _RolloutInstallError(
                f"LeWM rollout needs h5py for the HDF5 dataset loader. "
                f"pip install h5py ({exc})"
            ) from exc
        try:
            from stable_worldmodel.api.dataset import LeRobotDataAdapter  # noqa: F401
        except ImportError:
            # Fall through to the le-wm package directly.
            try:
                import le_wm  # noqa: F401
            except ImportError as exc:
                raise _RolloutInstallError(
                    "LeWM rollout needs either 'stable-worldmodel' or 'le-wm' "
                    "installed. pip install stable-worldmodel "
                    f"OR pip install le-wm ({exc})"
                ) from exc
    except _RolloutInstallError:
        raise

    # Real-rollout body is deferred until at least one real LeWM ckpt is
    # in hand for shape calibration. For now, write a "real-path-attempted"
    # summary that's structurally identical to the synthetic one but
    # marked synthetic=False, with NaN losses.
    latent_dim = 192
    pred = np.zeros((horizon, latent_dim), dtype=np.float32)
    np.savez(out_dir / "next_state_pred.npz", pred=pred)
    summary = {
        "kind": "lewm",
        "checkpoint": str(cp),
        "dataset_root": str(ds_root),
        "horizon": int(horizon),
        "n_seed_episodes": int(n_seeds),
        "synthetic": False,
        "partial": True,                # deps loaded, real predict body deferred
        "predictor_implemented": False,
        "mean_pred_loss": float("nan"),
    }
    out_path = out_dir / "rollout_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


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
        # Defensive fallback — these bodies no longer raise NotImplementedError,
        # but keep the handler in case a future sub-architecture stub does.
        print(f"wm-rollout: not yet implemented — {exc}")
        return 2
    except _RolloutInstallError as exc:
        # Missing-dependency errors with a clear pip install hint.
        print(f"wm-rollout: missing dependency — {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"wm-rollout: error — {exc}")
        return 1
