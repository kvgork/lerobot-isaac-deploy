"""World-model loaders + inference adapters.

Two paths:

* :func:`load_dreamerv3` — open a sheeprl DreamerV3 checkpoint, pull out
  the actor + world-model encoder.  Wraps them in a callable that takes
  the same ``obs`` dict as a LeRobot policy and returns an action tensor.
  Stateful — maintains the recurrent hidden state across calls via
  ``PlayerDV3``.

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

Config-path discovery (sheeprl Lightning vs Hydra layout)
----------------------------------------------------------
Sheeprl 0.5+ saves run configs in two possible places:

* Hydra layout (standalone CLI):  ``<run-dir>/.hydra/config.yaml``
* Lightning/CSV logger layout:    ``<run-dir>/version_<N>/config.yaml``

``load_dreamerv3`` probes both.  Pass ``config_yaml=`` explicitly to
override discovery.

State-dim extraction from checkpoint weights
--------------------------------------------
The config file only contains ``env.id`` / ``env.image_size``; it does
NOT record per-key observation dimensionality in a reliable, consistent
field.  The actual shapes live in the saved weights:

* ``world_model["encoder.mlp_encoder.model._model.0.weight"]`` has shape
  ``(hidden, state_dim)`` — first-layer input dim equals the MLP-key
  observation dimension.
* ``actor["mlp_heads.0.weight"]`` has shape ``(2 * action_dim, hidden)``
  for continuous (TanhNormal distribution outputs mean + log-std) —
  divide by 2 to get the true action_dim.

This approach is robust to any SO-101 obs variant (6-DOF joints only,
13-D joint+object-pose, etc.) without requiring an explicit cfg field.

Tensor shape convention (sheeprl ``prepare_obs``)
-------------------------------------------------
``PlayerDV3.get_actions`` expects observations in the same layout that
sheeprl's ``prepare_obs`` produces:

* CNN keys:  float32 tensor shape ``(T=1, num_envs=1, C, H, W)``
  normalized to ``[-0.5, 0.5]``  (i.e. ``uint8 / 255 - 0.5``).
* MLP keys:  float32 tensor shape ``(T=1, num_envs=1, D)``

``LoadedWMActor.select_action`` applies this conversion automatically.

The dreamer loader is intentionally lazy about its imports — it can be
called from any environment as long as ``torch`` + ``sheeprl`` are
available at load time. The deploy package itself does NOT depend on
sheeprl; the user adds it to the pixi env when they actually need
dreamer deploy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
# Config-path discovery helpers
# ---------------------------------------------------------------------------


def _discover_config_yaml(ckpt_file: Path) -> Path | None:
    """Probe standard sheeprl config-file locations for a given ckpt file.

    Returns the first existing path from:
    1. ``<ckpt_parent>/../.hydra/config.yaml``  (hydra standalone CLI layout)
    2. ``<ckpt_parent>/../config.yaml``          (Lightning CSV-logger layout, e.g. version_0/config.yaml)
    3. ``<ckpt_parent>/../../.hydra/config.yaml`` (one extra level up)
    4. ``<ckpt_parent>/../../config.yaml``
    """
    p = ckpt_file.parent  # typically: <run>/version_0/checkpoint/
    candidates = [
        p.parent / ".hydra" / "config.yaml",        # <run>/version_0/.hydra/config.yaml
        p.parent / "config.yaml",                    # <run>/version_0/config.yaml  ← Lightning layout
        p.parent.parent / ".hydra" / "config.yaml",  # <run>/.hydra/config.yaml
        p.parent.parent / "config.yaml",             # <run>/config.yaml
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


# ---------------------------------------------------------------------------
# Checkpoint-weight dimension extractors
# ---------------------------------------------------------------------------


def _state_dim_from_weights(world_model_state: dict) -> int | None:
    """Read the MLP-key observation dim from the first encoder layer weight.

    Returns None when the key is absent (CNN-only configs).
    """
    key = "encoder.mlp_encoder.model._model.0.weight"
    w = world_model_state.get(key)
    if w is not None and hasattr(w, "shape") and len(w.shape) >= 2:
        return int(w.shape[1])
    return None


def _action_dim_from_weights(actor_state: dict) -> int | None:
    """Read the continuous action dim from the actor MLP head.

    sheeprl continuous actors output (mean, log_std) concatenated, so the
    head weight shape is (2*action_dim, hidden). Divide by 2.
    """
    key = "mlp_heads.0.weight"
    w = actor_state.get(key)
    if w is not None and hasattr(w, "shape") and len(w.shape) >= 2:
        raw = int(w.shape[0])
        # raw == 2*action_dim for continuous (TanhNormal).
        # Guard against discrete configs (raw wouldn't be even, or would
        # be small) — if odd, return raw as-is.
        if raw % 2 == 0 and raw >= 2:
            return raw // 2
        return raw
    return None


# ---------------------------------------------------------------------------
# LoadedWMActor — real sheeprl actor wrapper
# ---------------------------------------------------------------------------


@dataclass
class LoadedWMActor:
    """DreamerV3 actor + recurrent-state container.

    Wraps sheeprl's ``PlayerDV3`` so the session can hold either
    type and call ``.select_action(obs_dict)`` uniformly.

    ``select_action`` always returns a ``numpy.ndarray`` of shape
    ``(action_dim,)`` dtype ``float32`` — callers should not rely on a
    torch tensor being returned.

    Observation dict accepted by ``select_action``
    ----------------------------------------------
    Keys must match the training-time encoder keys exactly.  Typical SO-101
    DreamerV3 config uses ``cnn_keys.encoder = ["rgb"]`` and
    ``mlp_keys.encoder = ["state"]``.  Values must be:

    * CNN keys: ``np.ndarray`` uint8 shape ``(C, H, W)`` OR ``(H, W, C)``
      — transposed to ``(C, H, W)`` automatically, then normalized to
      ``[-0.5, 0.5]`` as sheeprl's ``prepare_obs`` does.
    * MLP keys: ``np.ndarray`` float32 shape ``(D,)``.

    Both are reshaped to ``(T=1, num_envs=1, ...)`` internally to match
    ``PlayerDV3.get_actions``'s expected tensor layout.
    """

    player: Any  # PlayerDV3 instance
    device: str
    action_dim: int
    cnn_keys: list = field(default_factory=list)  # list of CNN key names for normalization
    kind: str = "dreamerv3"
    # Optional handles for rollout / eval code.
    world_model: Any = None
    decoder: Any = None
    cfg: Any = None  # resolved cfg dict (dotdict)
    # Internal fields (repr=False to avoid noise in prints)
    _unused: Any = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Deprecated attribute access shims (backwards compat)
    # ------------------------------------------------------------------

    @property
    def actor(self) -> Any:
        """Backwards compat: return the raw sheeprl actor from the player."""
        return getattr(self.player, "actor", None)

    @property
    def encoder(self) -> Any:
        """Backwards compat: return the encoder from the player."""
        return getattr(self.player, "encoder", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the recurrent state. Call at the start of each new episode."""
        self.player.init_states()

    def select_action(self, obs: dict) -> Any:
        """Encode obs → update RSSM → sample action via PlayerDV3.get_actions.

        Mirrors sheeprl's ``prepare_obs`` convention:
        * CNN keys: (C, H, W) uint8 → (1, 1, C, H, W) float32 in [-0.5, 0.5]
        * MLP keys: (D,) float32 → (1, 1, D) float32

        Parameters
        ----------
        obs:
            Dict with keys matching the training-time encoder keys.
            CNN keys: uint8 ndarray (C, H, W) or (H, W, C) — auto-transposed.
            MLP keys: float32 ndarray (D,).

        Returns
        -------
        np.ndarray shape (action_dim,) float32, values in [-1, 1].
        """
        import torch

        cnn_key_set = set(self.cnn_keys)
        tensor_obs: dict[str, Any] = {}

        for k, v in obs.items():
            arr = np.asarray(v)
            if k in cnn_key_set:
                # CNN key: ensure (C, H, W) layout
                if arr.ndim == 3 and arr.shape[-1] in (1, 3) and arr.shape[0] not in (1, 3):
                    # (H, W, C) → (C, H, W)
                    arr = arr.transpose(2, 0, 1)
                t = torch.from_numpy(arr.copy()).to(self.device)
                # Normalize: uint8 → float32 in [-0.5, 0.5] (sheeprl prepare_obs convention)
                if t.dtype == torch.uint8:
                    t = t.float().div_(255.0).sub_(0.5)
                else:
                    t = t.float().sub_(0.5)
                # Reshape to (T=1, num_envs=1, C, H, W)
                while t.ndim < 5:
                    t = t.unsqueeze(0)
            else:
                # MLP key: flatten to 1-D then reshape to (1, 1, D)
                t = torch.from_numpy(arr.reshape(-1).copy()).to(self.device).float()
                # Reshape to (T=1, num_envs=1, D)
                t = t.view(1, 1, -1)

            tensor_obs[k] = t

        with torch.no_grad():
            actions = self.player.get_actions(tensor_obs, greedy=False)

        # actions is a sequence of tensors (one per action head).
        # Continuous SO-101: single head of shape (1, 1, action_dim).
        action = torch.cat(actions, dim=-1)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


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
        Optional explicit path to the Hydra/Lightning config.  Default:
        auto-discovered by :func:`_discover_config_yaml`.
    device:
        Default: ``cuda`` if available else ``cpu``.

    Notes
    -----
    sheeprl checkpoints are a `torch.save` of a dict with keys
    ``world_model``, ``actor``, ``critic``, ``target_critic``, etc.
    We restore only ``world_model`` + ``actor`` via ``build_agent``
    (no critic/replay-buffer needed for deploy).

    Dimension resolution
    ~~~~~~~~~~~~~~~~~~~~
    * ``state_dim`` is read from ``world_model["encoder.mlp_encoder...weight"]``
      shape — NOT from cfg.  This is robust to any SO-101 obs variant
      (6-DOF joints, 13-D joint+object-pose, etc.).
    * ``action_dim`` is read from ``actor["mlp_heads.0.weight"]`` shape
      divided by 2 (sheeprl continuous actors output mean + log-std
      concatenated).

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
        import yaml  # noqa: F401  ships with sheeprl
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

    # Config-path resolution: explicit > auto-discovery.
    if config_yaml is not None:
        cfg_path = Path(config_yaml)
    else:
        cfg_path = _discover_config_yaml(cp)
    if cfg_path is None or not cfg_path.is_file():
        raise FileNotFoundError(
            f"hydra/lightning config not found for checkpoint {cp}. "
            f"Searched: .hydra/config.yaml and config.yaml in parent dirs. "
            f"Pass config_yaml= explicitly, or check the run directory layout."
        )

    # Load config (OmegaConf when available, yaml.safe_load fallback).
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
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Robust load: weights_only=False needed for nested dict structure
    # (world_model/actor/critic/...), BUT un-pickling fails on numpy RNG
    # state across numpy 1.x ↔ 2.x boundary with
    # `<class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator`.
    # Workaround: temporarily patch numpy's BitGenerator ctor to fall
    # back to a default PCG64 on lookup failure. We don't consume the
    # saved RNG state for inference, so any valid bit generator is fine.
    try:
        ckpt_state = torch.load(cp, map_location=device, weights_only=False)
    except (ValueError, ImportError) as exc:
        msg = str(exc)
        if "BitGenerator" not in msg and "PCG64" not in msg:
            raise
        import numpy as _np
        import numpy.random._pickle as _np_pickle
        _orig_ctor = _np_pickle.__bit_generator_ctor
        def _tolerant_ctor(bit_generator_name="PCG64", *args, **kwargs):
            try:
                return _orig_ctor(bit_generator_name, *args, **kwargs)
            except ValueError:
                return _np.random.PCG64()
        _np_pickle.__bit_generator_ctor = _tolerant_ctor
        try:
            ckpt_state = torch.load(cp, map_location=device, weights_only=False)
        finally:
            _np_pickle.__bit_generator_ctor = _orig_ctor

    # sheeprl 0.5+ DreamerV3 build_agent signature:
    #   build_agent(fabric, actions_dim, is_continuous, cfg, obs_space,
    #               world_model_state=None, actor_state=None,
    #               critic_state=None, target_critic_state=None)
    # Returns: (world_model, actor, critic, target_critic, player)
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

    # --- Derive observation + action spaces from checkpoint weights --------
    # The cfg DOES contain cnn_keys/mlp_keys but does NOT reliably encode
    # per-key dims for all training variants.  Read dims from saved weights
    # to be robust to any obs variant (e.g. 6-DOF vs 13-D joint+object-pose).
    cnn_keys = list(cfg.get("algo", {}).get("cnn_keys", {}).get("encoder", []) or [])
    mlp_keys = list(cfg.get("algo", {}).get("mlp_keys", {}).get("encoder", []) or [])
    env_cfg = cfg.get("env", {}) or {}
    image_size = int(
        env_cfg.get("image_size")
        or cfg.get("algo", {}).get("world_model", {}).get("image_size")
        or 64
    )

    # Read MLP state dim from first-layer weight if available.
    wm_state = ckpt_state.get("world_model") or {}
    actor_state_raw = ckpt_state.get("actor") or {}
    state_dim_from_weights = _state_dim_from_weights(wm_state)
    action_dim_from_weights = _action_dim_from_weights(actor_state_raw)

    # Fallback to cfg-declared dims when weight introspection isn't possible
    # (e.g. CNN-only config with no MLP encoder).
    env_action_dim = int(
        env_cfg.get("action_dim") or env_cfg.get("num_actions") or 6
    )
    action_dim = action_dim_from_weights if action_dim_from_weights is not None else env_action_dim

    space_dict: dict[str, Any] = {}
    for k in cnn_keys:
        space_dict[k] = gym.spaces.Box(
            low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8
        )
    for k in mlp_keys:
        # Use weight-derived dim when available, else cfg fallback.
        if state_dim_from_weights is not None:
            per_key_dim = state_dim_from_weights
        else:
            per_key_dim = int(env_cfg.get(f"{k}_dim") or action_dim)
        space_dict[k] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(per_key_dim),), dtype=np.float32
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

    world_model, _actor_module, critic, target_critic, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        wm_state or None,
        actor_state_raw or None,
        ckpt_state.get("critic"),
        ckpt_state.get("target_critic"),
    )
    world_model.eval()
    _actor_module.eval()
    player.eval()

    # Initialise PlayerDV3 recurrent + stochastic states (num_envs=1).
    player.init_states()

    return LoadedWMActor(
        player=player,
        device=device,
        action_dim=int(action_dim),
        cnn_keys=list(cnn_keys),
        world_model=world_model,
        decoder=(
            getattr(world_model, "observation_model", None)
            or getattr(world_model, "decoder", None)
        ),
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
