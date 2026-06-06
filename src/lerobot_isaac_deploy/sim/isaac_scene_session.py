"""Closed-loop sim deploy backend on an Isaac Sim USD scene.

Pairs the deploy session pattern with the USD scene generator at
``~/workspaces/isaac-auto-scene`` so policies can be evaluated WITHOUT
hardware.

Status: SCAFFOLD (Phase 2 of plans/2026-05-23-sim-deploy-pipeline.md).
``run()`` either drives the synthetic-marker stub OR raises
``NotImplementedError`` until the real Isaac Sim wiring lands.

Why a scaffold:
  - Lets the autoresearcher's ``EVAL_MODE=sim`` knob be wired now
    against a stable Python API (no signature churn later).
  - Keeps the deploy package importable in any env (Isaac Sim is a
    soft import — only loaded inside ``run()``).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _default_scene_path(name: str = "so101_workspace") -> Path:
    """Resolve the bundled scene USD from lerobot-isaac-configs.

    Soft-imported so this module stays importable on a laptop that has the
    deploy package but not the configs leaf installed. Raises a clear error
    pointing at the explicit ``usd_path`` arg if configs is unavailable.
    """
    try:
        from lerobot_isaac_configs import get_scene_path
    except ImportError as exc:  # pragma: no cover - exercised only sans configs
        raise ImportError(
            "usd_path was None and lerobot-isaac-configs is not installed, so "
            "the default scene cannot be resolved. Either `pip install "
            "lerobot-isaac-configs` or pass an explicit usd_path."
        ) from exc
    return get_scene_path(name)


@dataclass
class SimEpisodeResult:
    """One rollout's outcome."""

    episode_idx: int
    length: int
    success: bool
    fail_reason: str = ""
    final_object_pose: list[float] = field(default_factory=list)


@dataclass
class SimRolloutSummary:
    """Aggregate rollout output. Serialised to JSON as
    ``rollout_summary.json`` (schema mirrors the open-loop eval JSON so the
    dashboard's autoresearch loader handles both uniformly)."""

    pc_success: float
    mean_ep_len: float
    n_episodes: int
    per_episode: list[SimEpisodeResult]
    policy_path: str
    usd_path: str
    wall_clock_s: float
    backend: str = "isaac_auto_scene"
    task: str = "so101-pickplace1-sim-closed-loop"


def default_pickplace_basket_success(obs: dict, info: dict) -> bool:
    """Default success criterion: object Z above basket floor by ≥ 5 cm.

    obs keys expected: ``object.pose`` (xyz quaternion) and
    ``basket.bounds`` (xmin, xmax, ymin, ymax, z).
    """
    try:
        obj_xyz = obs["object.pose"][:3]
        basket = obs.get("basket.bounds")
        if basket is None:
            return False
        xmin, xmax, ymin, ymax, zfloor = basket
        return (
            xmin <= obj_xyz[0] <= xmax
            and ymin <= obj_xyz[1] <= ymax
            and obj_xyz[2] >= zfloor + 0.05
        )
    except Exception:  # noqa: BLE001
        return False


SUCCESS_CRITERIA: dict[str, Callable[[dict, dict], bool]] = {
    "pickplace_basket": default_pickplace_basket_success,
    "any_motion": lambda obs, info: bool(info.get("any_motion", False)),
}


@dataclass
class IsaacSceneSession:
    """Drive a trained policy through an Isaac Sim USD scene.

    Parameters
    ----------
    usd_path
        USD file emitted by ``isaac-auto-scene render``.
    policy_path
        ``pretrained_model/`` dir (any kind detect_policy_kind recognises).
    dataset_root
        LeRobotDataset whose obs schema the policy was trained on.
        Used to infer feature shapes when policy ckpt lacks them.
    n_episodes
        How many rollouts to run.
    max_steps
        Per-episode horizon. 600 steps @ 30 Hz = 20 s.
    rate_hz
        Sim control frequency.
    render_cameras
        Camera names that match observation.images.* keys in the dataset.
    device
        ``cuda`` or ``cpu``. Inference device for the policy.
    success_criterion
        Callable or registry name resolving via SUCCESS_CRITERIA.
    output_dir
        Where to write ``rollout_summary.json``. Defaults to a timestamped
        directory under ``outputs/sim_deploy/``.
    dr_config
        Optional DR config file (see Phase 4 of the plan).
    """

    # Pass ``None`` to resolve the bundled scene from lerobot-isaac-configs
    # (``get_scene_path("so101_workspace")``) — keeps callers off the
    # gitignored workspace ``assets/`` path. Field stays positionally required
    # (no default) so the two required fields below need no defaults.
    usd_path: Path | None
    policy_path: Path
    dataset_root: Path
    n_episodes: int = 10
    max_steps: int = 600
    rate_hz: float = 30.0
    # DR100 Phase 1: single D435 wrist cam (was overhead+wrist). Matches the
    # env ``d435_rgb`` obs term and the real SO-101 dataset column.
    render_cameras: tuple[str, ...] = ("d435_rgb",)
    device: str = "cuda"
    success_criterion: str | Callable[[dict, dict], bool] = "pickplace_basket"
    output_dir: Path | None = None
    dr_config: Path | None = None

    def __post_init__(self) -> None:
        if self.usd_path is None:
            self.usd_path = _default_scene_path()
        self.usd_path = Path(self.usd_path)
        self.policy_path = Path(self.policy_path)
        self.dataset_root = Path(self.dataset_root)
        if self.output_dir is None:
            ts = time.strftime("%Y%m%dT%H%M%S")
            self.output_dir = Path(f"outputs/sim_deploy/{ts}")
        else:
            self.output_dir = Path(self.output_dir)
        if isinstance(self.success_criterion, str):
            if self.success_criterion not in SUCCESS_CRITERIA:
                raise ValueError(
                    f"unknown success_criterion '{self.success_criterion}'. "
                    f"available: {sorted(SUCCESS_CRITERIA)}"
                )
            self.success_criterion = SUCCESS_CRITERIA[self.success_criterion]
        # Load sibling meta.json if present. The USD itself is loaded lazily
        # inside the real Isaac Sim runtime; here we only validate the file
        # exists + remember the meta for the rollout summary.
        if not self.usd_path.is_file():
            # Allowed during synthetic-marker runs (no real USD needed).
            self._meta: dict[str, Any] = {}
        else:
            meta_path = self.usd_path.with_suffix(".meta.json")
            if meta_path.is_file():
                try:
                    self._meta = json.loads(meta_path.read_text())
                except Exception:  # noqa: BLE001
                    self._meta = {}
            else:
                self._meta = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> Path:
        """Execute the rollout. Returns path to rollout_summary.json."""
        from .. import policy_kind as pk

        self.output_dir.mkdir(parents=True, exist_ok=True)
        kind = pk.detect_policy_kind(self.policy_path)

        # Synthetic-marker short-circuit: produce a zero-success rollout
        # JSON without touching Isaac Sim. Lets CI smoke-test the wiring.
        synth_marker = self.policy_path / "synthetic_marker.json"
        if synth_marker.is_file() or kind == "unknown":
            return self._run_synthetic(reason=f"synthetic_marker or kind={kind}")

        # Real Isaac Sim path — soft-import to avoid forcing isaaclab into
        # any caller that doesn't need a real rollout.
        try:
            from ._isaac_runtime import IsaacSimRuntime  # type: ignore
        except ImportError as exc:
            raise NotImplementedError(
                "IsaacSceneSession real backend is Phase 2 future work. "
                "Pass a synthetic-marker ckpt or run via the open-loop "
                "rollout backend until the Isaac Sim wiring lands. "
                f"({exc})"
            ) from exc

        rt = IsaacSimRuntime(
            usd_path=self.usd_path,
            render_cameras=self.render_cameras,
            rate_hz=self.rate_hz,
        )
        from ..policy_loader import load_policy  # type: ignore
        policy = load_policy(self.policy_path, dataset_root=self.dataset_root, device=self.device)

        per_ep: list[SimEpisodeResult] = []
        t0 = time.time()
        for ep_idx in range(self.n_episodes):
            rt.reset_episode(seed=ep_idx)
            policy.reset()
            success = False
            step_count = 0
            fail_reason = "max_steps_reached"
            for step in range(self.max_steps):
                obs = rt.get_obs()
                action = policy.select_action(obs)
                rt.apply_action(action)
                rt.step()
                step_count = step + 1
                info = rt.get_info()
                if self.success_criterion(obs, info):
                    success = True
                    fail_reason = ""
                    break
                if info.get("contact_terminal", False):
                    fail_reason = "contact_terminal"
                    break
            obs_final = rt.get_obs()
            per_ep.append(
                SimEpisodeResult(
                    episode_idx=ep_idx,
                    length=step_count,
                    success=success,
                    fail_reason=fail_reason,
                    final_object_pose=list(obs_final.get("object.pose", []))[:7],
                )
            )
        wall = time.time() - t0

        summary = SimRolloutSummary(
            pc_success=sum(r.success for r in per_ep) / max(1, len(per_ep)),
            mean_ep_len=sum(r.length for r in per_ep) / max(1, len(per_ep)),
            n_episodes=len(per_ep),
            per_episode=per_ep,
            policy_path=str(self.policy_path),
            usd_path=str(self.usd_path),
            wall_clock_s=wall,
        )
        return self._write_summary(summary)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_synthetic(self, reason: str) -> Path:
        """Emit a stub rollout (no sim, no policy load). Useful for CI."""
        per_ep = [
            SimEpisodeResult(
                episode_idx=i, length=0, success=False, fail_reason=f"synthetic ({reason})"
            )
            for i in range(self.n_episodes)
        ]
        summary = SimRolloutSummary(
            pc_success=0.0,
            mean_ep_len=0.0,
            n_episodes=self.n_episodes,
            per_episode=per_ep,
            policy_path=str(self.policy_path),
            usd_path=str(self.usd_path),
            wall_clock_s=0.0,
            backend="synthetic_stub",
        )
        return self._write_summary(summary)

    def _write_summary(self, summary: SimRolloutSummary) -> Path:
        out_path = self.output_dir / "rollout_summary.json"
        out_path.write_text(
            json.dumps(
                {
                    "pc_success": summary.pc_success,
                    "mean_ep_len": summary.mean_ep_len,
                    "n_episodes": summary.n_episodes,
                    "per_episode": [r.__dict__ for r in summary.per_episode],
                    "policy_path": summary.policy_path,
                    "usd_path": summary.usd_path,
                    "wall_clock_s": summary.wall_clock_s,
                    "backend": summary.backend,
                    "task": summary.task,
                },
                indent=2,
            )
        )
        return out_path
