"""Isaac Sim runtime for IsaacSceneSession.

Phase 2 of plans/2026-05-23-sim-deploy-pipeline.md.

Status: IMPLEMENTED (phase2.1–2.9). Boots Isaac Sim, loads a USD scene,
attaches the SO-101 ArticulationCfg from lerobot-isaac-env, registers
cameras, and implements get_obs / apply_action / step / reset_episode
against Isaac Lab's API.

Soft imports throughout so that:
  - The deploy package stays importable in any env (sheeprl-only,
    autoresearch-only, dashboard-only). Isaac Sim is hauled in lazily
    inside `IsaacSimRuntime.__init__`.
  - Tests that only need the synthetic-marker path avoid spinning up
    SimulationApp.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# 30 warm-up sim steps are MANDATORY on Isaac Sim 6.0 + Isaac Lab v2.3.2
# before camera obs are valid. Documented in ~/workspaces/isaac-auto-scene/
# CLAUDE.md §Isaac Sim integration. Skipping this returns black frames.
WARM_UP_FRAMES = 30


# Map a dataset/obs camera key -> the camera prim NAME baked into the
# isaac-auto-scene USD (under the scene root). The generator emits a single
# wrist camera as `def Camera "D435"`. Add entries here if future scenes bake
# additional named cameras.
_SCENE_CAMERA_PRIMS: dict[str, str] = {
    "d435_rgb": "D435",
    "d435": "D435",
}


def _scene_camera_prim(obs_camera_name: str) -> str:
    """Resolve an obs camera key to its USD prim name in the auto-scene USD.

    Falls back to the name itself (so an exact-match prim still works).
    """
    return _SCENE_CAMERA_PRIMS.get(obs_camera_name, obs_camera_name)


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
    # Default matches DR100 sim env + real LeRobot dataset (single D435 wrist cam).
    # Callers training against a multi-cam scene can override per-deploy.
    render_cameras: tuple[str, ...] = ("d435_rgb",)
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
    # Sibling meta.json (basket_bounds etc.), loaded once from <usd>.meta.json
    _meta: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Optional per-joint relative-motion clamp (radians).  None = unclamped.
    _max_relative_target: float | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def _boot(self) -> None:
        """Spin up SimulationApp + Isaac Lab context. Idempotent."""
        if self._booted:
            return

        # Load sibling meta.json (basket_bounds, camera world poses, etc.)
        meta_path = Path(self.usd_path).with_suffix(".meta.json")
        if meta_path.is_file():
            try:
                self._meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                self._meta = {}

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
        """Stage the USD scene file under /World/Scene.

        phase2.1: mount USD via isaacsim.core.utils.stage.add_reference_to_stage.
        check_sim_scene.sh verifies that /World/SO101, /World/object,
        /World/basket, /World/cameras/{overhead,wrist} exist in the USD.
        After referencing, those prims live at /World/Scene/SO101 etc.
        """
        if not self.usd_path.is_file():
            raise FileNotFoundError(
                f"USD scene not found: {self.usd_path}. "
                f"Generate it with `isaac-auto-scene generate` into the configs "
                f"leaf (lerobot_isaac_configs/scenes/<name>.usd), then resolve "
                f"via `lerobot_isaac_configs.get_scene_path('<name>')`."
            )
        # phase2.1 — soft-import: this function is only called after AppLauncher
        # has initialised the Isaac Sim stage.
        from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore

        add_reference_to_stage(usd_path=str(self.usd_path), prim_path="/World/Scene")
        logger.debug("USD scene mounted at /World/Scene from %s", self.usd_path)

    def _attach_articulation(self) -> None:
        """Wrap the SO-101 prim from the USD in an Isaac Lab Articulation.

        phase2.2: instantiate isaaclab.assets.Articulation(cfg) with the
        prim_path overridden to /World/Scene/SO101 (the USD is nested under
        /World/Scene after _load_usd).
        """
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

        # phase2.2 — override prim_path because the USD is mounted under
        # /World/Scene/SO101 (not the default /World/envs/env_0/SO101).
        from isaaclab.assets import Articulation  # type: ignore

        cfg = cfg.replace(prim_path="/World/Scene/SO101")
        self._articulation = Articulation(cfg)
        # NOTE: do NOT access num_joints / any articulation *data* here — the
        # underlying PhysX view is only created by `self._sim.reset()` (called
        # later in _boot). Touching `num_joints` now raises
        # `'Articulation' object has no attribute '_root_physx_view'`.
        logger.debug("SO-101 Articulation attached at /World/Scene/SO101 (pre-reset)")

    def _attach_cameras(self) -> None:
        """Register CameraCfg for every name in self.render_cameras.

        phase2.3: for each name in self.render_cameras, build a CameraCfg
        pointing at /World/Scene/cameras/<name>. 64×64 RGB matches the
        bridged HDF5 schema that the SmolVLA / DreamerV3 pipelines expect.
        Camera prims already encode world poses baked by isaac-auto-scene.

        Closes CLAUDE.md §Build Status Checklist — Camera observation wiring.
        """
        try:
            from isaaclab.sensors import Camera, CameraCfg  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"isaaclab.sensors.CameraCfg not importable ({exc})"
            ) from exc

        # The isaac-auto-scene USD bakes each camera pose as a single
        # `matrix4d xformOp:transform`, but isaaclab's Camera sensor requires a
        # standard [translate, orient, scale] xform-op stack. Standardise the
        # camera prim(s) in place first (preserves the world pose), else
        # CameraCfg raises "not a xformable prim with standard transform ops".
        import omni.usd  # type: ignore
        from isaaclab.sim.utils import (  # type: ignore
            standardize_xform_ops,
            validate_standard_xform_ops,
        )

        stage = omni.usd.get_context().get_stage()

        # phase2.3 — register one Camera sensor per requested camera name.
        # The isaac-auto-scene USD bakes the camera as `def Camera "D435"` under
        # the scene root, so under /World/Scene it lives at /World/Scene/D435
        # (NOT /World/Scene/cameras/<name>). We attach to the existing prim with
        # spawn=None — CameraCfg requires spawn to be set explicitly (even None),
        # otherwise it raises "Missing values detected ... spawn".
        for name in self.render_cameras:
            cam_prim_path = f"/World/Scene/{_scene_camera_prim(name)}"
            prim = stage.GetPrimAtPath(cam_prim_path)
            # Canonical [translate, orient, scale] is required by isaaclab. Scenes
            # from current isaac-auto-scene already emit it; only standardise a
            # legacy matrix4d prim (runtime AddXformOp(scale) can fail on Camera
            # prims, so guard it and let CameraCfg surface any real problem).
            if prim and prim.IsValid() and not validate_standard_xform_ops(prim):
                try:
                    standardize_xform_ops(prim)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "standardize_xform_ops failed for %s (%s); regenerate the "
                        "scene with current isaac-auto-scene.", cam_prim_path, exc
                    )
            cam_cfg = CameraCfg(
                prim_path=cam_prim_path,
                spawn=None,  # attach to the existing USD camera prim
                width=64,
                height=64,
                data_types=["rgb"],
                update_period=1.0 / self.rate_hz,
            )
            cam = Camera(cam_cfg)
            self._cameras[name] = cam
            logger.debug(
                "Camera registered: /World/Scene/%s -> obs '%s' (64x64 RGB)",
                _scene_camera_prim(name),
                name,
            )

    def _post_reset_camera_init(self) -> None:
        """After sim.reset(), do anything that requires _ALL_INDICES set.

        phase2.4: The auto-scene USD already encodes camera world poses
        (verified by check_sim_scene.sh). This hook is a documented no-op.
        If a future camera needs runtime repositioning, call:
            self._cameras[name].set_world_poses_from_view(eyes=..., targets=...)
        here — NOT before sim.reset() (_ALL_INDICES unset before reset).
        """
        return  # no-op: USD bakes camera poses; no runtime re-placement needed.

    # ------------------------------------------------------------------ #
    # Episode-step API
    # ------------------------------------------------------------------ #

    def reset_episode(self, seed: int) -> None:
        """Reset arm to home, randomise object pose within basket bounds, zero camera buffers.

        phase2.5: articulation → home joint positions, object XY sampled
        deterministically from `basket_bounds` in <usd>.meta.json, then
        WARM_UP_FRAMES physics steps to settle the renderer.

        Parameters
        ----------
        seed:
            Passed straight to numpy default_rng for reproducibility —
            same seed ↔ identical object placement.
        """
        if not self._booted:
            self._boot()

        import torch  # type: ignore  # soft: only reachable in sim env

        # 1. Reset articulation to home (all-zeros joint positions).
        #    build_articulation_cfg.init_state.joint_pos is the canonical
        #    source when available, but zero is a safe default for SO-101.
        num_j = self._articulation.num_joints
        home_q = torch.zeros(num_j, device=self.device)
        self._articulation.write_joint_state_to_sim(
            position=home_q.unsqueeze(0),
            velocity=torch.zeros_like(home_q).unsqueeze(0),
        )
        self._articulation.reset()

        # 2. Randomise object XY within basket_bounds from meta.json.
        basket = self._meta.get("basket_bounds")
        if basket is not None:
            rng = np.random.default_rng(seed)
            new_x = float(rng.uniform(basket["xmin"], basket["xmax"]))
            new_y = float(rng.uniform(basket["ymin"], basket["ymax"]))
            try:
                from pxr import Usd, UsdGeom  # type: ignore  # Isaac Sim ships pxr

                stage = Usd.Stage.Open(str(self.usd_path))
                obj_prim = stage.GetPrimAtPath("/World/Scene/object")
                if obj_prim.IsValid():
                    xformable = UsdGeom.Xformable(obj_prim)
                    # Preserve existing Z while randomising XY.
                    world_xform = xformable.ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    current_z = float(world_xform.ExtractTranslation()[2])
                    translate_ops = [
                        op
                        for op in xformable.GetOrderedXformOps()
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    ]
                    if translate_ops:
                        translate_ops[0].Set(
                            (new_x, new_y, current_z), Usd.TimeCode.Default()
                        )
                    else:
                        xformable.AddTranslateOp().Set(
                            (new_x, new_y, current_z), Usd.TimeCode.Default()
                        )
                    stage.Save()
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Object randomization failed (pxr not available or prim "
                    "invalid) — object stays at previous pose."
                )

        # 3. Re-warm camera buffers: WARM_UP_FRAMES is mandatory after any
        #    reset to avoid stale / black frames (Isaac Sim 6.0 constraint).
        for _ in range(WARM_UP_FRAMES):
            self._sim.step(render=True)

    def get_obs(self) -> dict[str, Any]:
        """Return an obs dict matching the LeRobotDataset schema.

        phase2.6: reads RGB from each registered camera, joint positions
        from the articulation, and object pose from the USD prim via pxr.

        Returns
        -------
        dict with keys:
            "observation.images.<name>"  — (3, 64, 64) float32 in [-0.5, 0.5]
            "observation.state"          — (6,) float32 joint positions
            "object.pose"                — (7,) float32 [xyz + xyzw quaternion]
            "basket.bounds"              — dict or None (for success criterion)
        """

        # 1. Joint positions — shape (num_joints,).
        joint_pos = self._articulation.data.joint_pos[0].cpu().numpy()  # (6,)

        # 2. Camera RGB → float32 CHW in [-0.5, 0.5].
        imgs: dict[str, Any] = {}
        for name, cam in self._cameras.items():
            rgb = cam.data.output["rgb"][0]  # (H, W, 3) uint8 tensor
            # permute HWC → CHW, .contiguous() to materialize C-order, then
            # cast to float32, normalize [0,255] → [-0.5, 0.5].
            rgb_t = rgb.permute(2, 0, 1).contiguous().float().div_(255.0).sub_(0.5)
            imgs[f"observation.images.{name}"] = rgb_t.cpu().numpy()

        # 3. Object pose (xyz + xyzw quaternion) from USD prim via pxr.
        obj_pose = np.zeros(7, dtype=np.float32)
        try:
            from pxr import Gf, Usd, UsdGeom  # type: ignore

            stage = Usd.Stage.Open(str(self.usd_path))
            obj_prim = stage.GetPrimAtPath("/World/Scene/object")
            if obj_prim.IsValid():
                xform = UsdGeom.Xformable(obj_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                t = xform.ExtractTranslation()
                q = Gf.Rotation(xform.ExtractRotationMatrix()).GetQuat()
                # TODO(verify-on-isaac-sim-6.0): Gf.Quatd stores (real, imaginary).
                # GetImaginary() returns Gf.Vec3d of xyz components; GetReal() is w.
                # Final layout written here is xyzw to match LeRobot schema. Confirm
                # imaginary-component ordering on first live deploy — if mirrored,
                # swap indices [3,4,5]. See review note on commit 6b7484c.
                im = q.GetImaginary()
                obj_pose[:3] = [float(t[0]), float(t[1]), float(t[2])]
                obj_pose[3:6] = [float(im[0]), float(im[1]), float(im[2])]  # xyz imag
                obj_pose[6] = float(q.GetReal())  # w
        except Exception:  # noqa: BLE001
            logger.debug("Could not read object pose from USD — returning zeros.")

        return {
            "observation.state": joint_pos.astype(np.float32),
            **imgs,
            "object.pose": obj_pose,
            "basket.bounds": self._meta.get("basket_bounds"),
        }

    def apply_action(self, action: np.ndarray) -> None:
        """Write joint position targets to the simulated SO-101.

        phase2.7: converts the (6,) float32 action array to a torch tensor,
        applies an optional per-joint relative-motion safety clamp (mirrors
        robot-data-runner's --max-relative-target), then calls
        set_joint_position_target + write_data_to_sim.

        Parameters
        ----------
        action:
            (6,) float32 joint-position targets in radians.

        Raises
        ------
        ValueError:
            If action.shape[0] != articulation.num_joints.
        """
        import torch  # type: ignore  # soft

        num_j = self._articulation.num_joints
        if action.shape[0] != num_j:
            raise ValueError(
                f"apply_action: action dim {action.shape[0]} != "
                f"articulation.num_joints {num_j}. "
                "Ensure the policy was trained on the same SO-101 config."
            )

        action_t = torch.from_numpy(action).to(self.device).unsqueeze(0)  # (1, 6)

        # Optional safety clamp — mirror robot-data-runner's --max-relative-target.
        if self._max_relative_target is not None:
            current = self._articulation.data.joint_pos[0]  # (6,)
            delta = action_t.squeeze(0) - current
            delta_clamped = torch.clamp(
                delta, -self._max_relative_target, self._max_relative_target
            )
            action_t = (current + delta_clamped).unsqueeze(0)

        self._articulation.set_joint_position_target(action_t)
        self._articulation.write_data_to_sim()

    def step(self) -> None:
        """Advance physics one tick. Camera obs refreshes on next get_obs().

        phase2.8: render=True is mandatory — without it cameras don't refresh.
        Wall-clock per call ≈ 1/rate_hz (≈33 ms @ 30 Hz on RTX 3080).
        """
        self._sim.step(render=True)

    def get_info(self) -> dict[str, Any]:
        """Return per-step termination flags and debug counters.

        phase2.9: cheap path — episode_done is always False here (success
        criterion is evaluated by the caller via get_obs()). contact_terminal
        is False until a contact sensor is registered; the commented block
        below shows the wiring when a table/arm-base sensor is added.

        Runtime overhead target: < 1 ms.
        """
        info: dict[str, Any] = {"episode_done": False, "contact_terminal": False}

        # When a contact sensor is registered (future Phase 3 safety work):
        #     contact_data = self._sensors["table_contact"].data.net_forces_w
        #     info["contact_terminal"] = bool(
        #         (contact_data.norm(dim=-1) > 50.0).any()
        #     )

        return info
