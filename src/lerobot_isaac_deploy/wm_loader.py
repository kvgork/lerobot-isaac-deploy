"""World-model loaders + inference adapters.

Two paths:

* :func:`load_dreamerv3` — open a sheeprl DreamerV3 checkpoint, pull out
  the actor + world-model encoder.  Wraps them in a callable that takes
  the same ``obs`` dict as a LeRobot policy and returns an action tensor.
  Stateful — maintains the recurrent hidden state across calls.

  If the checkpoint directory contains a ``synthetic_marker.json`` file,
  ``load_dreamerv3`` short-circuits immediately and returns a
  :class:`_SyntheticActor` (no torch / sheeprl import required).  This
  lets tests and mock-hardware smoke runs work in any environment.

* :func:`load_lewm` — refuses with a clear message that LeWM has no
  actor and points at :mod:`lerobot_isaac_deploy.wm_rollout` for offline
  simulation.

Callers should duck-type on ``.select_action(obs)`` + ``.reset()`` only.
Both :class:`LoadedWMActor` and :class:`_SyntheticActor` expose that
minimal interface; ``select_action`` always returns a ``numpy.ndarray``
of shape ``(action_dim,)`` and dtype ``float32``.

The dreamer loader is intentionally lazy about its imports — it can be
called from any environment as long as ``torch`` + ``sheeprl`` are
available at load time. The deploy package itself does NOT depend on
sheeprl; the user adds it to the pixi env when they actually need
dreamer deploy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class WMDeployNotSupported(RuntimeError):
    """Raised when a checkpoint kind cannot be deployed on hardware."""


# ---------------------------------------------------------------------------
# Synthetic-marker helpers
# ---------------------------------------------------------------------------


def _is_synthetic_marker(cp: Path) -> tuple[bool, dict | None]:
    """Return (True, marker_dict) if cp (or one level up) has synthetic_marker.json."""
    import json

    p = Path(cp)
    candidates = [p if p.is_dir() else p.parent, p.parent if p.is_file() else None]
    for c in candidates:
        if c and c.is_dir():
            m = c / "synthetic_marker.json"
            if m.is_file():
                try:
                    return True, json.loads(m.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    return True, {}
    return False, None


class _SyntheticActor:
    """No-op actor returned when a synthetic_marker.json is present.

    Returns zero-valued actions of the declared ``action_dim``.
    No torch / sheeprl dependency.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = int(action_dim)

    def select_action(self, obs: Any) -> Any:  # noqa: ARG002
        import numpy as np

        return np.zeros((self.action_dim,), dtype=np.float32)

    def reset(self) -> None:
        pass

    @property
    def kind(self) -> str:
        return "dreamerv3"

    @property
    def device(self) -> str:
        return "cpu"


def _make_synthetic_actor(action_dim: int, kind: str) -> _SyntheticActor:  # noqa: ARG001
    """Return a :class:`_SyntheticActor` for the given action dimensionality."""
    return _SyntheticActor(action_dim=action_dim)


# ---------------------------------------------------------------------------
# LoadedWMActor — real sheeprl actor wrapper
# ---------------------------------------------------------------------------


@dataclass
class LoadedWMActor:
    """DreamerV3 actor + recurrent-state container.

    Mirrors :class:`robot_data_runner.policy_loader.LoadedPolicy` so the
    session can hold either type and call ``.select_action(obs_dict)``
    uniformly.

    ``select_action`` always returns a ``numpy.ndarray`` of shape
    ``(action_dim,)`` dtype ``float32`` — callers should not rely on a
    torch tensor being returned.
    """

    actor: Any
    encoder: Any
    device: str
    kind: str = "dreamerv3"
    _state: Any = None  # recurrent hidden state (h, z)
    # Optional handles set by load_dreamerv3 for downstream rollout / eval use.
    # `world_model` is sheeprl's full WorldModel (encoder + rssm + decoder +
    # reward_model + continue_model). `decoder` is the observation decoder.
    # Both are None for the synthetic stub.
    world_model: Any = None
    decoder: Any = None
    cfg: Any = None  # the resolved cfg dict (dotdict)

    def reset(self) -> None:
        """Clear the recurrent state. Call on each new episode."""
        self._state = None

    def select_action(self, obs: dict) -> Any:
        """Run encoder → recurrent update → actor; return action as numpy array."""
        import numpy as np
        import torch

        # Encode current observation
        with torch.no_grad():
            z = self.encoder(obs)
            # On first call, init state with the encoder output shape.
            if self._state is None:
                bs = next(iter(obs.values())).shape[0] if obs else 1
                self._state = self.actor.init_state(bs, device=self.device)
            # Update recurrent state + sample action.
            self._state, action = self.actor.step(z, self._state)

        # Normalise to ndarray so callers don't need to import torch.
        if hasattr(action, "detach"):
            return action.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return np.asarray(action, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_dreamerv3(
    checkpoint_path: Path | str,
    *,
    config_yaml: Path | str | None = None,
    device: str | None = None,
) -> LoadedWMActor | _SyntheticActor:
    """Load a sheeprl DreamerV3 checkpoint into an inference-ready actor.

    Parameters
    ----------
    checkpoint_path:
        Either a ``ckpt_*.ckpt`` file or a directory containing one.
    config_yaml:
        Optional explicit path to the hydra config. Default: discovered
        from ``<run-dir>/.hydra/config.yaml``.
    device:
        Default: ``cuda`` if available else ``cpu``.

    Notes
    -----
    sheeprl checkpoints are a `torch.save` of a dict with keys
    ``world_model``, ``actor``, ``critic``, ``actor_target``, etc.
    We restore only ``world_model.encoder`` + ``actor`` since deploy
    doesn't need critic/replay buffer.

    Synthetic-marker short-circuit
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    If ``checkpoint_path`` (or its parent directory) contains a
    ``synthetic_marker.json`` file, this function returns a
    :class:`_SyntheticActor` immediately — torch and sheeprl are never
    imported. The marker JSON may include ``action_dim`` (default 6).
    This allows tests and mock-hardware smoke runs in any environment.
    """
    cp = Path(checkpoint_path)
    synth, marker = _is_synthetic_marker(cp)
    if synth:
        action_dim = int((marker or {}).get("action_dim", 6))
        return _make_synthetic_actor(action_dim=action_dim, kind="dreamerv3")

    # --- real sheeprl path below ---
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to load a DreamerV3 checkpoint. "
            "pip install torch"
        ) from exc

    try:
        import yaml  # ships with sheeprl
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required to read sheeprl hydra configs. "
            "pip install pyyaml"
        ) from exc

    if cp.is_dir():
        candidates = sorted(cp.glob("**/ckpt_*.ckpt"))
        if not candidates:
            raise FileNotFoundError(
                f"no ckpt_*.ckpt file under {cp} — DreamerV3 must save "
                f"at least one checkpoint before deploy."
            )
        cp = candidates[-1]  # most recent step

    cfg_path = Path(config_yaml) if config_yaml else (cp.parent.parent / ".hydra" / "config.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"hydra config not found at {cfg_path} — sheeprl needs it "
            f"to reconstruct the policy class."
        )
    # Match sheeprl's own loader: OmegaConf.load → resolve interpolations →
    # to_container → wrap in dotdict. This converts ListConfig/DictConfig to
    # native Python list/dict so `isinstance(..., list)` checks inside
    # sheeprl's `make_env` succeed.
    try:
        from omegaconf import OmegaConf
        from sheeprl.utils.utils import dotdict
        # sheeprl/Hydra configs reference `${now:%Y-%m-%d_%H-%M-%S}` etc.
        # That resolver lives in `hydra.core.utils.setup_globals()` and is
        # only installed when Hydra boots. Register it manually here so
        # `OmegaConf.to_container(resolve=True)` doesn't trip on
        # `UnsupportedInterpolationType`.
        try:
            from hydra.core.utils import setup_globals as _hydra_setup_globals
            _hydra_setup_globals()
        except Exception:  # noqa: BLE001
            from datetime import datetime
            if not OmegaConf.has_resolver("now"):
                OmegaConf.register_new_resolver(
                    "now",
                    lambda pattern: datetime.now().strftime(pattern),
                    replace=True,
                )
        raw_cfg = OmegaConf.load(str(cfg_path))
        cfg = dotdict(OmegaConf.to_container(raw_cfg, resolve=True, throw_on_missing=False))
    except ImportError:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(cp, map_location=device, weights_only=False)

    # sheeprl 0.5+ DreamerV3 build_agent signature:
    #   build_agent(fabric, actions_dim, is_continuous, cfg, obs_space,
    #               world_model_state=None, actor_state=None,
    #               critic_state=None, target_critic_state=None)
    # We need a Fabric instance + concrete obs/action spaces from a real
    # env. Build a single-env vector to extract spaces, then tear it down.
    try:
        from sheeprl.algos.dreamer_v3.agent import build_agent
        import gymnasium as gym
        from lightning.fabric import Fabric
    except ImportError as exc:
        raise ImportError(
            "sheeprl + lightning + gymnasium are required to deploy "
            "DreamerV3 checkpoints. pip install 'sheeprl[dreamer]>=0.5' "
            "'lightning>=2' gymnasium"
        ) from exc

    accelerator = "cuda" if device.startswith("cuda") else "cpu"
    fabric = Fabric(accelerator=accelerator, devices=1)
    if not fabric._launched:
        fabric.launch()

    # Build observation_space + action_space directly from cfg, bypassing
    # sheeprl's make_env path. make_env tries to instantiate the training-
    # time env via Hydra (`env._target_` resolves to e.g.
    # `lerobot_isaac_adapters.sheeprl_plugin.hdf5_env.get_hdf5_env`) which
    # would force the deploy laptop to install the training-only adapter
    # package. We only need the spaces, not a step-able env — and the
    # spaces are fully captured by cfg.algo.{cnn,mlp}_keys + cfg.env.
    cnn_keys = list(cfg.get("algo", {}).get("cnn_keys", {}).get("encoder", []) or [])
    mlp_keys = list(cfg.get("algo", {}).get("mlp_keys", {}).get("encoder", []) or [])
    env_cfg = cfg.get("env", {})
    image_size = int(
        env_cfg.get("image_size")
        or cfg.get("algo", {}).get("world_model", {}).get("image_size")
        or 64
    )
    # SO-101 has 6 joints — default. Override via cfg.env.action_dim when present.
    action_dim = int(env_cfg.get("action_dim") or env_cfg.get("num_actions") or 6)

    space_dict: dict[str, Any] = {}
    for k in cnn_keys:
        space_dict[k] = gym.spaces.Box(
            low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8
        )
    for k in mlp_keys:
        # Most MLP keys in sheeprl configs are 1-D feature vectors; use
        # action_dim as a sensible default fallback when the cfg doesn't
        # pin per-key dims. Real cfgs add explicit shapes in env.* — read
        # them when available.
        per_key = env_cfg.get(f"{k}_dim") or action_dim
        space_dict[k] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(per_key),), dtype=np.float32
        )
    observation_space = gym.spaces.Dict(space_dict) if space_dict else gym.spaces.Dict()
    action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
    )

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape
        if is_continuous
        else (
            action_space.nvec.tolist()
            if is_multidiscrete
            else [action_space.n]
        )
    )

    world_model, actor, critic, target_critic, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state.get("world_model"),
        state.get("actor"),
        state.get("critic"),
        state.get("target_critic"),
    )
    world_model.eval()
    actor.eval()

    return LoadedWMActor(
        actor=actor,
        encoder=world_model.encoder,
        device=device,
        world_model=world_model,
        decoder=getattr(world_model, "observation_model", None) or getattr(world_model, "decoder", None),
        cfg=cfg,
    )


def load_lewm(checkpoint_path: Path | str) -> LoadedWMActor:
    """Refuse: LeWM has no actor head.

    Always raises :class:`WMDeployNotSupported`. Use
    :mod:`lerobot_isaac_deploy.wm_rollout` instead for offline simulation.
    """
    raise WMDeployNotSupported(
        f"LeWorldModel checkpoints have no actor head and cannot drive "
        f"motors directly. Use the wm-rollout subcommand for offline "
        f"state-prediction rollouts:\n"
        f"  lerobot-isaac-deploy wm-rollout "
        f"--checkpoint {checkpoint_path} --dataset <PATH>\n"
        f"For real-robot control on this task, deploy a LeRobot policy "
        f"(smolvla/act/diffusion) trained on the same dataset."
    )
