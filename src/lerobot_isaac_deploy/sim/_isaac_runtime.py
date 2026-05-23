"""Isaac Sim runtime for IsaacSceneSession.

Phase 2 of plans/2026-05-23-sim-deploy-pipeline.md.

Status: SKELETON. Boots Isaac Sim, loads a USD scene, attaches the SO-101
ArticulationCfg from lerobot-isaac-env, registers cameras, and implements
get_obs / apply_action / step / reset_episode against Isaac Lab's API.

Soft imports throughout so that:
  - The deploy package stays importable in any env (sheeprl-only,
    autoresearch-only, dashboard-only). Isaac Sim is hauled in lazily
    inside `IsaacSimRuntime.__init__`.
  - Tests that only need the synthetic-marker path avoid spinning up
    SimulationApp.

Open work tracked inline as TODO(phase2.<n>).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# 30 warm-up sim steps are MANDATORY on Isaac Sim 6.0 + Isaac Lab v2.3.2
# before camera obs are valid. Documented in ~/workspaces/isaac-auto-scene/
# CLAUDE.md §Isaac Sim integration. Skipping this returns black frames.
WARM_UP_FRAMES = 30


@dataclass
class IsaacSimRuntime:
    """Boot + manage one Isaac Sim instance for closed-loop sim deploy.

    Lifecycle:
        rt = IsaacSimRuntime(usd_path, render_cameras, rate_hz)
        for ep in range(N):
            rt.reset_episode(seed=ep)
            for step in range(MAX_STEPS):
                obs = rt.get_obs()
                action = policy.select_action(obs)
                rt.apply_action(action)
                rt.step()
                if rt.get_info().get("episode_done"): break
        rt.close()
    """

    usd_path: Path
    render_cameras: tuple[str, ...] = ("overhead_camera_rgb", "wrist_camera_rgb")
    rate_hz: float = 30.0
    headless: bool = True
    enable_cameras: bool = True
    device: str = "cuda"

    # Runtime handles — populated lazily by _boot().
    _app: Any = field(default=None, init=False, repr=False)
    _sim: Any = field(default=None, init=False, repr=False)
    _scene: Any = field(default=None, init=False, repr=False)
    _articulation: Any = field(default=None, init=False, repr=False)
    _cameras: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _booted: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _boot(self) -> None:
        """Spin up SimulationApp + Isaac Lab context. Idempotent."""
        if self._booted:
            return
        try:
            from isaaclab.app import AppLauncher  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Isaac Lab is required for IsaacSimRuntime. "
                "Install via `pixi install -e sim && pixi run install-isaac-lab`. "
                f"({exc})"
            ) from exc

        launcher = AppLauncher(
            headless=self.headless,
            enable_cameras=self.enable_cameras,
        )
        self._app = launcher.app

        # Wait until app is fully ready — emits known "ready" message.
        # Some Isaac Sim builds require an explicit warm-up update loop.
        for _ in range(2):
            self._app.update()

        # NB: AppLauncher initialises a global state; we must defer Isaac Lab
        # imports until AFTER `AppLauncher(...)` to avoid `RuntimeError:
        # SimulationApp not initialised` from the lab assets module.
        from isaaclab.sim import SimulationContext, SimulationCfg  # type: ignore

        sim_cfg = SimulationCfg(
            dt=1.0 / self.rate_hz,
            device=self.device,
        )
        self._sim = SimulationContext(sim_cfg)
        self._load_usd()
        self._attach_articulation()
        self._attach_cameras()
        self._sim.reset()
        # Camera.set_world_poses_from_view + similar must come AFTER reset
        # (see auto-scene CLAUDE.md pitfalls). Camera config wired below.
        self._post_reset_camera_init()
        # Mandatory 30-frame warm-up to settle the renderer.
        for _ in range(WARM_UP_FRAMES):
            self._sim.step(render=True)
        self._booted = True
        logger.info("IsaacSimRuntime booted on %s @ %.1f Hz", self.device, self.rate_hz)

    def close(self) -> None:
        """Tear down Isaac Sim. SimulationApp.close() deadlocks on Sim 6.0
        per auto-scene CLAUDE.md — caller may prefer `os._exit(0)` at the
        end of a script."""
        if not self._booted:
            return
        try:
            self._sim.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._app.close()
        except Exception:  # noqa: BLE001
            logger.warning(
                "SimulationApp.close() raised — known Isaac Sim 6.0 issue. "
                "Caller should consider os._exit(0) for clean process exit."
            )
        self._booted = False

    # ------------------------------------------------------------------ #
    # Scene setup
    # ------------------------------------------------------------------ #

    def _load_usd(self) -> None:
        """Stage the USD scene file under /World."""
        if not self.usd_path.is_file():
            raise FileNotFoundError(
                f"USD scene not found: {self.usd_path}. "
                f"Run isaac-auto-scene to generate it, then place at "
                f"`assets/sim_scenes/<name>.usd`."
            )
        # TODO(phase2.1): use isaacsim.core.utils.stage.add_reference_to_stage
        # to mount the USD under /World/Scene. The scene already contains
        # SO101 + object + basket + camera prims at the paths
        # check_sim_scene.sh validates.
        raise NotImplementedError(
            "TODO(phase2.1): mount USD via add_reference_to_stage. "
            "See plans/2026-05-23-sim-deploy-pipeline.md Phase 2 §2."
        )

    def _attach_articulation(self) -> None:
        """Wrap the SO-101 prim from the USD in an Isaac Lab Articulation."""
        # Reuse lerobot-isaac-env's already-built ArticulationCfg.
        try:
            from lerobot_isaac_env.so101_articulation import build_articulation_cfg  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "lerobot_isaac_env not importable — verify the sibling is "
                f"installed editable. ({exc})"
            ) from exc

        cfg = build_articulation_cfg(usd_path=self.usd_path)
        if cfg is None:
            raise RuntimeError(
                "build_articulation_cfg returned None — Isaac Lab probably "
                "missing or USD path unresolved."
            )
        # TODO(phase2.2): instantiate isaaclab.assets.Articulation(cfg) and
        # store on self._articulation. Cfg already targets `/World/SO101`
        # which matches the auto-scene USD layout (verified by
        # check_sim_scene.sh).
        raise NotImplementedError(
            "TODO(phase2.2): instantiate Articulation(cfg). "
            "See lerobot-isaac-env/.../so101_env_cfg.py for the working pattern."
        )

    def _attach_cameras(self) -> None:
        """Register CameraCfg for every name in self.render_cameras.

        Camera prims live under /World/cameras/* per the auto-scene USD
        contract; we attach Camera sensors that read from those prims.
        """
        try:
            from isaaclab.sensors import CameraCfg  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"isaaclab.sensors.CameraCfg not importable ({exc})"
            ) from exc

        # TODO(phase2.3): for each name in self.render_cameras, build a
        # CameraCfg pointing at `/World/cameras/<name>`. Use 64×64 RGB to
        # match the dataset schema the policy expects (verified via
        # `_open_loop_eval.py`'s normalization path).
        # Sample shape:
        #     CameraCfg(
        #         prim_path=f"/World/cameras/{name}",
        #         width=64, height=64,
        #         data_types=["rgb"],
        #     )
        raise NotImplementedError(
            "TODO(phase2.3): build CameraCfgs + register sensors. "
            "Closes CLAUDE.md §Build Status Checklist 'Camera observation wiring'."
        )

    def _post_reset_camera_init(self) -> None:
        """After sim.reset(), do anything that requires _ALL_INDICES set."""
        # TODO(phase2.4): if any camera needs set_world_poses_from_view,
        # call it here. Currently the USD already encodes camera world
        # poses, so this is likely a no-op — kept as a hook.
        return

    # ------------------------------------------------------------------ #
    # Episode-step API
    # ------------------------------------------------------------------ #

    def reset_episode(self, seed: int) -> None:
        """Reset arm to home, randomise object pose, zero camera buffers."""
        if not self._booted:
            self._boot()
        # TODO(phase2.5): articulation.reset() + write home joint positions
        # + sample object xy within basket bounds (read from .meta.json's
        # `basket_bounds` field). Use `seed` to deterministically vary.
        raise NotImplementedError("TODO(phase2.5): reset_episode body")

    def get_obs(self) -> dict[str, Any]:
        """Return an obs dict matching the LeRobotDataset schema."""
        # TODO(phase2.6): read RGB tensors from each registered camera,
        # joint state from articulation, object pose from the USD prim.
        # Format:
        #     {
        #         "observation.images.overhead_camera_rgb": (3, 64, 64) float32 in [-0.5, 0.5],
        #         "observation.images.wrist_camera_rgb":    (3, 64, 64) float32,
        #         "observation.state":                      (6,)        float32 (joint positions),
        #         "object.pose":                            (7,)        float32 (xyz + xyzw quat),
        #     }
        raise NotImplementedError("TODO(phase2.6): get_obs body")

    def apply_action(self, action: np.ndarray) -> None:
        """Write joint position targets. Caller is responsible for clamping
        (mirror robot-data-runner's `--max-relative-target` for parity)."""
        # TODO(phase2.7): articulation.set_joint_position_target(action) +
        # articulation.write_data_to_sim()
        raise NotImplementedError("TODO(phase2.7): apply_action body")

    def step(self) -> None:
        """Advance physics one tick. Camera obs refreshes on next get_obs()."""
        # TODO(phase2.8): self._sim.step(render=True)
        raise NotImplementedError("TODO(phase2.8): step body")

    def get_info(self) -> dict[str, Any]:
        """Per-step info: contacts, terminal flags, debug counters."""
        # TODO(phase2.9): read articulation/object collisions; flag
        # `contact_terminal` on arm-base hits.
        return {"episode_done": False, "contact_terminal": False}
