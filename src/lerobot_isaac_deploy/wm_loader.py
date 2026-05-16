"""World-model loaders + inference adapters.

Two paths:

* :func:`load_dreamerv3` — open a sheeprl DreamerV3 checkpoint, pull out
  the actor + world-model encoder.  Wraps them in a callable that takes
  the same ``obs`` dict as a LeRobot policy and returns an action tensor.
  Stateful — maintains the recurrent hidden state across calls.

* :func:`load_lewm` — refuses with a clear message that LeWM has no
  actor and points at :mod:`lerobot_isaac_deploy.wm_rollout` for offline
  simulation.

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


class WMDeployNotSupported(RuntimeError):
    """Raised when a checkpoint kind cannot be deployed on hardware."""


@dataclass
class LoadedWMActor:
    """DreamerV3 actor + recurrent-state container.

    Mirrors :class:`robot_data_runner.policy_loader.LoadedPolicy` so the
    session can hold either type and call ``.select_action(obs_dict)``
    uniformly.
    """

    actor: Any
    encoder: Any
    device: str
    kind: str = "dreamerv3"
    _state: Any = None  # recurrent hidden state (h, z)

    def reset(self) -> None:
        """Clear the recurrent state. Call on each new episode."""
        self._state = None

    def select_action(self, obs: dict) -> Any:
        """Run encoder → recurrent update → actor; return action tensor."""
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
        return action


def load_dreamerv3(
    checkpoint_path: Path | str,
    *,
    config_yaml: Path | str | None = None,
    device: str | None = None,
) -> LoadedWMActor:
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
    """
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

    cp = Path(checkpoint_path)
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
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(cp, map_location=device, weights_only=False)

    # sheeprl's DreamerV3 keys (subject to sheeprl API drift)
    try:
        from sheeprl.algos.dreamer_v3.agent import build_agent
    except ImportError as exc:
        raise ImportError(
            "sheeprl is required to deploy DreamerV3 checkpoints. "
            "pip install 'sheeprl[dreamer]>=0.5'"
        ) from exc

    agent = build_agent(
        cfg,
        obs_space=state.get("obs_space"),
        action_space=state.get("action_space"),
    )
    agent.load_state_dict(state["agent"], strict=False)
    agent.to(device)
    agent.eval()

    return LoadedWMActor(
        actor=agent.actor,
        encoder=agent.world_model.encoder,
        device=device,
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
