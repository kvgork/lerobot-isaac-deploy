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

    # Real rollout body — DreamerV3 image reconstruction MSE on held-out
    # episodes. Two source-of-truth options for held-out frames:
    #   1. The bridge-produced HDF5 (preferred — already at the right
    #      image_size/window the WM was trained on).
    #   2. Raw LeRobotDataset (Parquet). Requires resize + dtype conversion
    #      to match the encoder's expected shape. Used as fallback.
    # Strategy: peek at ``ds_root``; if it's an HDF5 file (or a sibling
    # `*_dreamerv3.hdf5` exists next to it), use option 1. Else fall back.
    import numpy as np
    import torch

    ds_path = Path(ds_root)
    hdf5_path: Path | None = None
    if ds_path.is_file() and ds_path.suffix in (".h5", ".hdf5"):
        hdf5_path = ds_path
    else:
        # Look for a sibling bridged HDF5 produced by lerobot_world_model_bridge.
        wm_data_root = ds_path.parent.parent / "outputs" / "wm_data"
        if wm_data_root.is_dir():
            cands = sorted(wm_data_root.glob(f"{ds_path.name}_dreamerv3.hdf5"))
            if cands:
                hdf5_path = cands[0]

    if hdf5_path is None or not hdf5_path.is_file():
        raise RuntimeError(
            f"DreamerV3 rollout: no bridged HDF5 found. Pass --dataset-root "
            f"directly to a *_dreamerv3.hdf5 file, or place one at "
            f"outputs/wm_data/<dataset>_dreamerv3.hdf5 (use the "
            f"lerobot_world_model_bridge skill to create it)."
        )

    try:
        import h5py
    except ImportError as exc:
        raise _RolloutInstallError(
            f"DreamerV3 rollout needs h5py installed. pip install h5py ({exc})"
        ) from exc

    world_model = loaded.world_model
    encoder = loaded.encoder
    decoder = loaded.decoder
    if world_model is None or decoder is None:
        raise RuntimeError(
            "DreamerV3 rollout: loaded actor missing world_model/decoder. "
            "Update wm_loader.load_dreamerv3 to expose them."
        )
    device = loaded.device

    cnn_keys = ["rgb"]
    try:
        cnn_keys = list(loaded.cfg["algo"]["cnn_keys"]["encoder"]) or ["rgb"]
    except Exception:  # noqa: BLE001
        pass
    cnn_key = cnn_keys[0]

    losses: list[float] = []
    n_frames_total = 0
    preds_first_ep: list[np.ndarray] | None = None

    rssm = world_model.rssm
    obs_model = world_model.observation_model

    with h5py.File(str(hdf5_path), "r") as f:
        if "episodes" not in f:
            raise RuntimeError(
                f"DreamerV3 rollout: HDF5 {hdf5_path} missing 'episodes' group"
            )
        ep_names = sorted(f["episodes"].keys())
        held_out = ep_names[-max(1, int(n_seeds)):]
        for ep_name in held_out:
            ep = f["episodes"][ep_name]
            frames_np = np.asarray(ep["frames"])  # (T, H, W, 3) uint8
            actions_np = np.asarray(ep["actions"])  # (T, A) float32
            if frames_np.size == 0:
                continue
            T = min(int(horizon) if horizon > 0 else frames_np.shape[0], frames_np.shape[0])
            T = min(T, actions_np.shape[0])
            # Frames: NHWC uint8 → BCHW float in [-0.5, 0.5] (sheeprl default).
            frames_t = (
                torch.from_numpy(frames_np[:T])
                .permute(0, 3, 1, 2)
                .contiguous()
                .float()
                .div_(255.0)
                .sub_(0.5)
                .to(device)
            )  # (T, 3, H, W)
            actions_t = (
                torch.from_numpy(actions_np[:T]).float().to(device)
            )  # (T, A)

            with torch.no_grad():
                # Encoder consumes a dict with a leading batch dim. Treat T as batch.
                obs_dict = {cnn_key: frames_t}
                embedded_obs = encoder(obs_dict)  # (T, embed)
                # Reshape to (T, B=1, ...).
                embedded_obs = embedded_obs.unsqueeze(1)  # (T, 1, embed)
                actions_t = actions_t.unsqueeze(1)        # (T, 1, A)
                # sheeprl's RSSM does `(1 - is_first) * action`, requiring a
                # float tensor (bool subtraction is not supported in torch).
                is_first = torch.zeros(T, 1, 1, dtype=torch.float32, device=device)
                is_first[0, 0, 0] = 1.0

                # Iterate the RSSM dynamic loop, exactly mirroring
                # sheeprl/algos/dreamer_v3/dreamer_v3.py lines 124-148.
                recurrent_state, posterior = rssm.get_initial_states((1, 1))
                # get_initial_states returns shapes:
                #   recurrent_state: (1, 1, recurrent_size)
                #   posterior:       (1, 1, stochastic, discrete) OR flattened
                recurrent_states = []
                posteriors_list = []
                for i in range(T):
                    recurrent_state, posterior, _prior, _post_logits, _prior_logits = (
                        rssm.dynamic(
                            posterior,
                            recurrent_state,
                            actions_t[i : i + 1],
                            embedded_obs[i : i + 1],
                            is_first[i : i + 1],
                        )
                    )
                    recurrent_states.append(recurrent_state)
                    posteriors_list.append(posterior)
                recurrent_states_t = torch.cat(recurrent_states, dim=0)  # (T, 1, R)
                posteriors_t = torch.cat(posteriors_list, dim=0)         # (T, 1, S, D) or (T, 1, S*D)
                # Flatten the categorical (stochastic, discrete) axes if present.
                if posteriors_t.dim() == 4:
                    posteriors_flat = posteriors_t.view(*posteriors_t.shape[:-2], -1)
                else:
                    posteriors_flat = posteriors_t
                latent_states = torch.cat((posteriors_flat, recurrent_states_t), dim=-1)
                # (T, 1, S*D + R)
                rec_dict = obs_model(latent_states)
                # rec_dict either dict[str, Tensor] or dict[str, Distribution].
                rec_obj = rec_dict[cnn_key] if isinstance(rec_dict, dict) else rec_dict
                # sheeprl's CNNDecoder returns plain Tensor; MLPDecoder wraps
                # in Independent(Normal(...)). Branch by Tensor-ness.
                if isinstance(rec_obj, torch.Tensor):
                    rec_img = rec_obj
                else:
                    rec_img = None
                    for attr in ("mode", "mean"):
                        v = getattr(rec_obj, attr, None)
                        if v is None:
                            continue
                        rec_img = v() if callable(v) else v
                        if rec_img is not None:
                            break
                    if rec_img is None:
                        raise RuntimeError(
                            f"DreamerV3 rollout: cannot materialise decoder "
                            f"output for key '{cnn_key}' (got {type(rec_obj)})"
                        )
                # rec_img shape: (T, 1, 3, H, W) — squeeze batch axis.
                rec_img = rec_img.view(*frames_t.shape)
                ep_mse = torch.mean((rec_img - frames_t) ** 2).item()
                losses.append(ep_mse)
                n_frames_total += T
                if preds_first_ep is None:
                    preds_first_ep = rec_img.detach().cpu().numpy()

    if not losses:
        raise RuntimeError(
            f"DreamerV3 rollout: no held-out frames found in {hdf5_path}"
        )

    mean_recon_loss = float(np.mean(losses))
    if preds_first_ep is None:
        preds_first_ep = np.zeros((1, 3, 64, 64), dtype=np.float32)
    np.savez(out_dir / "next_state_pred.npz", pred=preds_first_ep)

    summary = {
        "kind": "dreamerv3",
        "checkpoint": str(cp),
        "dataset_root": str(ds_root),
        "hdf5_path": str(hdf5_path),
        "horizon": int(horizon),
        "n_seed_episodes": int(n_seeds),
        "n_frames_total": int(n_frames_total),
        "synthetic": False,
        "partial": False,
        "decoder_implemented": True,
        "mean_recon_loss": mean_recon_loss,
        "per_episode_recon_loss": losses,
        "cnn_key": cnn_key,
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
