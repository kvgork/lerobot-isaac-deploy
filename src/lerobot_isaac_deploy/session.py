"""DeploySession — confirm-gated SO-101 deploy ladder.

Subprocess-driven wrapper around the ``robot-data-runner`` CLI family.
Steps:

    1. preflight                robot-data-run-check   (no robot)
    2. dry-run loop             robot-data-run          (robot, no motors)
       (or in-process mock loop when ``--mock-hardware`` is set)
       (or in-process real-arm read + dreamerv3 actor for dreamerv3 ckpts)
    3. execute @ 1°/step        robot-data-run --execute
    4. execute @ 3°/step        robot-data-run --execute
    5. closed-loop N-ep eval    robot-data-run-eval

Each step prompts stdin for ``yes`` before advancing. Operator aborts
at any step → graceful exit code 10.

Non-interactive mode
--------------------
Pass ``--yes`` / ``--assume-yes`` (or set
``LEROBOT_ISAAC_DEPLOY_ASSUME_YES=1``) to auto-answer "yes" to confirm
prompts. As a defense-in-depth measure, ``--yes`` alone NEVER
auto-confirms a step that sends motor commands (the two ``--execute``
steps). Operators must pass BOTH ``--yes`` AND ``--execute`` to
auto-advance through motor-write gates — this preserves a typed gate
between the smoke-test path and any real-hardware path.

Mock-hardware mode
------------------
Pass ``--mock-hardware`` to skip the ``robot-data-run`` subprocess in
step 2 and instead run an in-process inference loop with synthetic
observations. The policy is loaded via ``robot-data-runner``'s
``load_policy`` (proven by preflight), then fed zero-valued state and
image observations matching ``policy.config.input_features``. Useful
for verifying end-to-end inference on a desktop with no SO-101 plugged
in. Mock-hardware is incompatible with ``--execute`` and the
closed-loop eval — those paths require real hardware.

Real-ckpt gate
--------------
Pass ``--require-real-ckpt`` (or set
``LI_DEPLOY_REQUIRE_REAL_CKPT=1``) to refuse motor-write steps when the
checkpoint directory contains a ``synthetic_marker.json`` test fixture.
This prevents accidental real-arm execution with a mock checkpoint.

DreamerV3 dry-run (Path B)
--------------------------
When ``_ckpt_kind == "dreamerv3"`` and ``--mock-hardware`` is NOT set,
``step_dry_loop`` uses an in-process loop that:

    1. Loads the DreamerV3 actor via
       :func:`lerobot_isaac_deploy.wm_loader.load_dreamerv3`.
    2. Opens the real SO-101 at ``cfg.port`` via
       :func:`lerobot_isaac_deploy.arm_state_reader.open_arm` (read-only,
       no motor writes).
    3. Streams joint positions at ``cfg.rate_hz`` for
       ``cfg.duration_dry_s`` seconds via
       :func:`lerobot_isaac_deploy.arm_state_reader.stream_joint_pos`.
    4. Constructs a state vector ``[joint_pos(6) | obj_pose_dummy(7)]``
       and a zeroed RGB image (the WM actor does not require real vision
       in this path).
    5. Calls ``actor.select_action(obs)`` and logs the action vector.
       NEVER writes to motors.

DreamerV3 execute paths (Path C — tight + loose)
-------------------------------------------------
When ``_ckpt_kind == "dreamerv3"`` and ``--execute`` is set,
``step_execute_tight`` and ``step_execute_loose`` use an in-process
motor-write loop implemented in
:func:`lerobot_isaac_deploy.arm_motor_writer`:

    1. Loads the DreamerV3 actor via
       :func:`lerobot_isaac_deploy.wm_loader.load_dreamerv3`.
    2. Opens the real SO-101 at ``cfg.port`` with
       ``max_relative_target=max_step_deg`` (server-side clamp).
    3. Reads joint-position calibration limits (falls back to hardcoded
       conservative defaults — see arm_motor_writer module docstring).
       Cal-derived limits are always a STRICT SUBSET of the hardcoded
       safety floor.
    4. Runs a rate-limited loop at ``cfg.rate_hz`` for ``duration_s``:
       a. Read obs via ``robot.get_observation()``.
       b. Validate joint positions — skip step on non-finite or
          implausible values rather than writing bad motor targets.
       c. Build state = [joint_pos(6) | obj_pose_dummy(7)].
       d. Call ``actor.select_action(obs)``.
       e. Compute targets: current + action * max_step_deg (per-joint),
          clipped to [-1, 1] BEFORE scaling, then clamped to calibration
          limits (intersected with hardcoded floor).
       f. Write targets via ``robot.send_action({<joint>.pos: ...})``.
       g. Log: jp, action, targets — operator can verify clamp effect.
    5. home-on-exit uses RAMPED return via arm_motor_writer.ramped_home()
       rather than an instant single-write to avoid high-velocity slam.
    6. KeyboardInterrupt triggers immediate ramped home + disconnect.

WARNING: the deployed ckpt is likely unconverged. Saturated actions
(±1.0) will cause the arm to jitter at clamp-max-deg per step. This is
intentional for wiring validation. Use clamp_tight_deg=1.0 for the
first run and keep a hand near the power switch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SessionConfig:
    """All operator-facing knobs for a single laptop deploy session."""

    policy_path: Path
    dataset_root: Path
    port: str = "/dev/ttyACM0"
    camera: str = "d435_rgb=/dev/video0,640,480"
    task: str = "pick and place cube"
    rate_hz: float = 30.0
    duration_dry_s: float = 30.0
    duration_tight_s: float = 30.0
    duration_loose_s: float = 60.0
    clamp_tight_deg: float = 1.0
    clamp_loose_deg: float = 3.0
    n_eval_episodes: int = 10
    eval_duration_per_episode_s: float = 15.0
    home_on_exit: bool = True
    do_dry_loop: bool = False
    do_execute: bool = False
    skip_closed_loop: bool = False
    assume_yes: bool = False
    mock_hardware: bool = False
    require_real_ckpt: bool = False
    eval_output_dir: Path = field(
        default_factory=lambda: Path.home() / "outputs" / "eval"
    )

    def __post_init__(self) -> None:
        self.policy_path = Path(self.policy_path)
        self.dataset_root = Path(self.dataset_root)
        self.eval_output_dir = Path(self.eval_output_dir)


# ---------------------------------------------------------------------------
# Console / TTY helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[0;32m"
_RED = "\033[0;31m"
_CYAN = "\033[0;36m"
_YELLOW = "\033[1;33m"
_NC = "\033[0m"

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _env_assume_yes() -> bool:
    """Honor ``LEROBOT_ISAAC_DEPLOY_ASSUME_YES`` (case-insensitive truthy)."""
    val = os.environ.get("LEROBOT_ISAAC_DEPLOY_ASSUME_YES", "").strip().lower()
    return val in _TRUTHY


def _env_require_real_ckpt() -> bool:
    """Honor ``LI_DEPLOY_REQUIRE_REAL_CKPT`` (case-insensitive truthy)."""
    val = os.environ.get("LI_DEPLOY_REQUIRE_REAL_CKPT", "").strip().lower()
    return val in _TRUTHY


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def info(msg: str) -> None:
    print(f"{_CYAN}[{_stamp()} INFO]{_NC} {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"{_GREEN}[{_stamp()}  OK ]{_NC} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{_YELLOW}[{_stamp()} WARN]{_NC} {msg}", flush=True)


def err(msg: str) -> None:
    print(f"{_RED}[{_stamp()} ERR ]{_NC} {msg}", file=sys.stderr, flush=True)


def confirm(
    msg: str,
    *,
    auto_yes: bool = False,
    safety_critical: bool = False,
) -> None:
    """Read 'yes' from stdin or abort the session with exit 10.

    When ``auto_yes`` is True AND ``safety_critical`` is False, the prompt
    is auto-answered (the question is still printed for the operator log,
    tagged ``[auto-yes]``).

    When ``safety_critical=True`` the auto-yes path is refused — the prompt
    always blocks on stdin. This is the defense-in-depth gate guarding
    motor-write steps: even with ``--yes`` set, the operator must
    explicitly type ``yes`` (or pass the dedicated motor-write flag,
    which the caller decides) before motors move.
    """
    if auto_yes and not safety_critical:
        print(
            f"{_YELLOW}{msg} [auto-yes]{_NC}",
            flush=True,
        )
        return
    try:
        ans = input(
            f"{_YELLOW}{msg} [type 'yes' to continue]:{_NC} "
        ).strip().lower()
    except EOFError:
        err(
            f"stdin closed at: {msg}. "
            "Pass --yes / --assume-yes for non-interactive runs."
        )
        sys.exit(10)
    if ans != "yes":
        err(f"operator aborted at: {msg}")
        sys.exit(10)


# ---------------------------------------------------------------------------
# DeploySession
# ---------------------------------------------------------------------------


class DeploySession:
    """Stateful runner of the deploy ladder."""

    def __init__(self, cfg: SessionConfig):
        self.cfg = cfg
        self.safety_ack = (
            Path.home() / ".config" / "robot-data-runner" / "safety_ack"
        )
        # Set by _validate_inputs so downstream steps can dispatch on kind.
        self._ckpt_kind: str = "unknown"

    # ----- discovery ---------------------------------------------------- #

    def _find_runner_bin(self, name: str) -> str:
        from shutil import which

        path = which(name)
        if path is None:
            raise FileNotFoundError(
                f"{name!r} not on PATH. Install robot-data-runner first: "
                f"pip install robot-data-runner (or run laptop_bootstrap.sh)."
            )
        return path

    # ----- preflight ---------------------------------------------------- #

    def _validate_inputs(self) -> None:
        if not self.cfg.policy_path.is_dir():
            raise FileNotFoundError(
                f"policy_path not a directory: {self.cfg.policy_path}"
            )
        if not self.cfg.dataset_root.is_dir():
            raise FileNotFoundError(
                f"dataset_root not a directory: {self.cfg.dataset_root}"
            )

        # Mock-hardware is incompatible with motor-write paths. We reject
        # the combination early so the operator gets a clear error rather
        # than a surprise later in the ladder.
        if self.cfg.mock_hardware and self.cfg.do_execute:
            raise RuntimeError(
                "--mock-hardware is a no-motor smoke path and cannot be "
                "combined with --execute. Drop one of the two flags."
            )

        # Checkpoint-kind gate. The session's confirm-gated ladder routes
        # through robot-data-runner's CLI which loads via the lerobot
        # policy factory only. WM checkpoints need separate paths.
        from lerobot_isaac_deploy.policy_kind import detect_policy_kind, explain
        kind = detect_policy_kind(self.cfg.policy_path)
        info(f"detected policy kind: {kind} — {explain(kind)}")

        if kind == "lerobot":
            self._ckpt_kind = "lerobot"
            self._check_real_ckpt_gate()
            return

        if kind == "dreamerv3":
            # DreamerV3 is first-class: actor head used for closed-loop deploy.
            self._ckpt_kind = "dreamerv3"
            self._check_real_ckpt_gate()
            return

        if kind == "lewm":
            raise RuntimeError(
                "LeWorldModel checkpoints have no actor head. Use "
                "`lerobot-isaac-deploy wm-rollout` for offline rollouts. "
                "For real-robot control on this task, deploy a LeRobot "
                "policy trained on the same dataset."
            )

        if kind in ("vjepa", "cosmos", "gaia"):
            raise RuntimeError(
                f"{kind!r} is a video world model with no robot-control actor. "
                f"See lerobot_isaac_deploy.wm_video.load_{kind} for the stub "
                f"entry-point — these models are deferred research "
                f"(plans/2026-05-22-wm-deploy-on-so101.md)."
            )

        raise RuntimeError(
            f"could not detect checkpoint kind at {self.cfg.policy_path}; "
            f"expected lerobot / dreamerv3 / lewm shape"
        )

    def _check_real_ckpt_gate(self) -> None:
        """Raise if require_real_ckpt is set and the ckpt is synthetic."""
        if self.cfg.require_real_ckpt:
            from lerobot_isaac_deploy.policy_kind import is_synthetic

            if is_synthetic(self.cfg.policy_path):
                raise RuntimeError(
                    f"--require-real-ckpt: refusing motor write — checkpoint "
                    f"at {self.cfg.policy_path} is a synthetic test fixture "
                    f"(has synthetic_marker.json). Provide a real ckpt or "
                    f"drop --require-real-ckpt."
                )

    def write_safety_ack(self) -> None:
        """Create the one-time safety-ack marker so the eval CLI doesn't block."""
        self.safety_ack.parent.mkdir(parents=True, exist_ok=True)
        self.safety_ack.write_text("acked", encoding="utf-8")
        ok(f"safety-ack written → {self.safety_ack}")

    # ----- ladder steps ------------------------------------------------- #

    def _confirm(self, msg: str, *, safety_critical: bool = False) -> None:
        """Session-scoped confirm honoring ``cfg.assume_yes``."""
        confirm(
            msg,
            auto_yes=self.cfg.assume_yes,
            safety_critical=safety_critical,
        )

    def step_preflight(self) -> None:
        info("STEP 1: preflight (load policy + I/O schema, no motors)")
        # robot-data-run-check is the LeRobot policy-factory smoke check;
        # WM checkpoints (DreamerV3 et al.) load via lerobot_isaac_deploy.wm_loader
        # and have no compatible config.json. Skip the subprocess for non-lerobot
        # kinds — the kind detection already happened in _validate_inputs and
        # downstream steps dispatch on self._ckpt_kind.
        if getattr(self, "_ckpt_kind", "lerobot") != "lerobot":
            ok(f"preflight skipped — {self._ckpt_kind} ckpt loads via wm_loader, not LeRobot runner")
            return
        check = self._find_runner_bin("robot-data-run-check")
        rc = subprocess.run(
            [
                check,
                "--policy-path", str(self.cfg.policy_path),
                "--dataset-root", str(self.cfg.dataset_root),
            ],
            check=False,
        ).returncode
        if rc != 0:
            raise RuntimeError(f"preflight failed rc={rc}")
        ok("policy loads cleanly")

    def step_dry_loop(self) -> None:
        if self.cfg.mock_hardware:
            self._confirm(
                "Run MOCK-HARDWARE inference loop "
                "(no serial port, no camera, synthetic obs)?"
            )
            info(
                f"STEP 2 (mock): in-process synthetic-obs inference "
                f"({self.cfg.duration_dry_s:.0f}s @ {self.cfg.rate_hz:.0f} Hz)"
            )

            kind = self._ckpt_kind
            if kind == "lerobot":
                from lerobot_isaac_deploy.mock_hardware import run_mock_inference_loop
                rc = run_mock_inference_loop(self.cfg)
            elif kind == "dreamerv3":
                from lerobot_isaac_deploy.mock_hardware import run_mock_inference_loop_wm
                rc = run_mock_inference_loop_wm(self.cfg)
            else:
                raise RuntimeError(
                    f"mock-hardware loop not supported for checkpoint kind "
                    f"{kind!r}. Supported kinds: lerobot, dreamerv3."
                )

            if rc != 0:
                raise RuntimeError(f"mock-hardware loop failed rc={rc}")
            ok("mock-hardware loop complete — policy emits actions end-to-end")
            return

        self._confirm(
            f"SO-101 plugged in at {self.cfg.port}? Workspace clear?"
        )
        info(f"STEP 2: dry-run loop ({self.cfg.duration_dry_s:.0f}s, NO motor writes)")

        if self._ckpt_kind == "dreamerv3":
            # Path B: in-process real-arm read + DreamerV3 actor + log-only output.
            # Real joint positions are read from the SO-101 at cfg.port.
            # NEVER writes motors — this is the dry-run path regardless of --execute.
            import numpy as np

            from lerobot_isaac_deploy.arm_state_reader import open_arm, stream_joint_pos
            from lerobot_isaac_deploy.wm_loader import load_dreamerv3

            actor = load_dreamerv3(self.cfg.policy_path)
            robot = open_arm(self.cfg.port)
            try:
                # Dummy object pose: position zeros + identity quaternion.
                # Real vision is deferred — the WM actor tolerates zeroed image obs.
                obj_pose_dummy = np.zeros(7, dtype=np.float32)
                obj_pose_dummy[3] = 1.0  # w component of identity quaternion

                cnn_keys = list(getattr(actor, "cnn_keys", ["rgb"]) or ["rgb"])
                image_size = 64
                env_cfg = {}
                if hasattr(actor, "cfg") and actor.cfg is not None:
                    env_cfg = (
                        actor.cfg.get("env")
                        if hasattr(actor.cfg, "get")
                        else {}
                    ) or {}
                image_size = int(env_cfg.get("image_size") or 64)

                step = 0
                for jp in stream_joint_pos(
                    robot,
                    rate_hz=self.cfg.rate_hz,
                    duration_s=self.cfg.duration_dry_s,
                ):
                    state = np.concatenate([jp, obj_pose_dummy]).astype(np.float32)
                    obs: dict = {"state": state}
                    for k in cnn_keys:
                        obs[k] = np.zeros((3, image_size, image_size), dtype=np.uint8)
                    action = actor.select_action(obs)
                    action = np.asarray(action, dtype=np.float32).reshape(-1)
                    info(
                        f"real-wm step {step} "
                        f"jp={jp.round(3).tolist()} "
                        f"action={action.round(3).tolist()}"
                    )
                    step += 1

                ok(
                    f"real-hw dry-run loop complete — {step} steps, NO motor writes"
                )
            finally:
                try:
                    robot.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            return

        # Existing lerobot path — subprocess to robot-data-run.
        run = self._find_runner_bin("robot-data-run")
        cmd = self._base_runner_cmd(run, duration_s=self.cfg.duration_dry_s)
        cmd += ["-v"]
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"dry-run loop failed rc={rc}")
        ok("dry-run complete — verify action lines made sense")

    # ----- DreamerV3 motor-write loop (shared between tight + loose) ---- #

    def _execute_dreamerv3_loop(
        self,
        duration_s: float,
        max_step_deg: float,
        step_label: str,
    ) -> None:
        """In-process DreamerV3 motor-write loop.

        Reads joint positions from the real SO-101, runs the DreamerV3
        actor, applies per-step safety clamping, and writes motor targets
        via ``robot.send_action()``.

        This method is shared by ``step_execute_tight`` (max_step_deg =
        clamp_tight_deg) and ``step_execute_loose`` (max_step_deg =
        clamp_loose_deg) to avoid code duplication.

        Safety layers (in order of application):
          1. Per-step joint-pos validation — skip step on non-finite or
             implausible values (Fix #6).
          2. Action clip to [-1, 1] in compute_targets() before scaling
             (Fix #1).
          3. Cal-derived joint limits intersected with hardcoded floor
             (Fix #2/#3).
          4. Server-side max_relative_target == max_step_deg passed to
             open_arm() (Fix #5).
          5. Ramped home on exit — no instant goto from arbitrary pose
             (Fix #4).

        Parameters
        ----------
        duration_s:
            Total wall-clock duration of the motor-write loop.
        max_step_deg:
            Maximum per-step delta in degrees for arm joints (0..4).
            Gripper delta is always capped at 5.0% regardless of this
            value — gripper needs a separate slower scale.
            Also used as the server-side max_relative_target clamp.
        step_label:
            Short label for log messages, e.g. "tight" or "loose".
        """
        import numpy as np

        from lerobot_isaac_deploy.arm_state_reader import (
            _extract_joint_pos,
            open_arm,
        )
        from lerobot_isaac_deploy.arm_motor_writer import (
            compute_targets,
            ramped_home,
            read_joint_limits,
            write_targets,
        )
        from lerobot_isaac_deploy.wm_loader import load_dreamerv3

        info(
            f"STEP execute ({step_label}): {duration_s:.0f}s "
            f"@ {max_step_deg}°/step arm clamp, 5%/step gripper clamp"
        )

        actor = load_dreamerv3(self.cfg.policy_path)

        # Determine CNN key list + image size from actor config.
        cnn_keys = list(getattr(actor, "cnn_keys", ["rgb"]) or ["rgb"])
        image_size = 64
        env_cfg = {}
        if hasattr(actor, "cfg") and actor.cfg is not None:
            env_cfg = (
                actor.cfg.get("env")
                if hasattr(actor.cfg, "get")
                else {}
            ) or {}
        image_size = int(env_cfg.get("image_size") or 64)

        # Fix #5: pass max_step_deg as server-side clamp so the follower
        # refuses any single Goal_Position more than max_step_deg away
        # from current — double-locking with compute_targets()'s clamp.
        robot = open_arm(self.cfg.port, max_relative_target=max_step_deg)
        step = 0
        rate_hz = float(getattr(self.cfg, "rate_hz", 30.0))
        dt = 1.0 / rate_hz

        try:
            # Read joint limits from calibration; cal-derived limits are
            # intersected with the hardcoded safety floor in read_joint_limits().
            jmin, jmax = read_joint_limits(robot)

            # Object pose is unknown on real arm — zero position + identity quat.
            # Matches training-time default (zero-camera + ones-quat default).
            obj_pose_dummy = np.zeros(7, dtype=np.float32)
            obj_pose_dummy[3] = 1.0  # w component of identity quaternion

            t_end = time.monotonic() + duration_s

            try:
                while time.monotonic() < t_end:
                    t0 = time.monotonic()

                    # Read current joint positions.
                    obs_dict = robot.get_observation()
                    jp = _extract_joint_pos(obs_dict)

                    # Fix #6: validate current joint positions before use.
                    # Skip the step (no motor write) on bad sensor data.
                    if not np.isfinite(jp).all():
                        info(f"WARN: non-finite joint pos {jp.tolist()}; skipping step {step}")
                        step += 1
                        elapsed = time.monotonic() - t0
                        if elapsed < dt:
                            time.sleep(dt - elapsed)
                        continue
                    if (np.abs(jp[:5]) > 180.0).any():
                        info(f"WARN: implausible joint pos {jp.tolist()}; skipping step {step}")
                        step += 1
                        elapsed = time.monotonic() - t0
                        if elapsed < dt:
                            time.sleep(dt - elapsed)
                        continue
                    if jp[5] < -10.0 or jp[5] > 110.0:
                        info(f"WARN: gripper out of range {jp[5]}; skipping step {step}")
                        step += 1
                        elapsed = time.monotonic() - t0
                        if elapsed < dt:
                            time.sleep(dt - elapsed)
                        continue

                    # Build actor observation.
                    state = np.concatenate([jp, obj_pose_dummy]).astype(np.float32)
                    obs: dict = {"state": state}
                    for k in cnn_keys:
                        obs[k] = np.zeros(
                            (3, image_size, image_size), dtype=np.uint8
                        )

                    # Run actor forward pass.
                    action = actor.select_action(obs)
                    action = np.asarray(action, dtype=np.float32).reshape(-1)

                    # Compute clamped targets (Fix #1: action clipped to [-1,1]
                    # inside compute_targets before scaling).
                    targets = compute_targets(
                        jp,
                        action,
                        max_step_deg=max_step_deg,
                        max_step_gripper_pct=5.0,
                        joint_limits_min=jmin,
                        joint_limits_max=jmax,
                    )

                    # Write motor targets.
                    write_targets(robot, targets)

                    info(
                        f"wm-exec [{step_label}] step {step} "
                        f"jp={jp.round(2).tolist()} "
                        f"action={action.round(2).tolist()} "
                        f"targets={targets.round(2).tolist()}"
                    )
                    step += 1

                    # Rate-limit: sleep remainder of dt.
                    elapsed = time.monotonic() - t0
                    if elapsed < dt:
                        time.sleep(dt - elapsed)

            except KeyboardInterrupt:
                info("KeyboardInterrupt — homing arm before disconnect")

        finally:
            # Fix #4: ramped home on exit — replace instant single-write with
            # a gradual ramp to avoid high-velocity slam from arbitrary pose.
            if self.cfg.home_on_exit:
                try:
                    obs = robot.get_observation()
                    cur = _extract_joint_pos(obs)
                    ramped_home(robot, cur, max_step_deg=max_step_deg, rate_hz=rate_hz)
                    info("ramped home complete")
                except Exception as exc:  # noqa: BLE001
                    warn(f"ramped home failed: {exc}")
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass

        ok(f"execute ({step_label}) complete — {step} steps")

    # ----- execute steps ------------------------------------------------ #

    def step_execute_tight(self) -> None:
        if self._ckpt_kind == "dreamerv3":
            self._check_real_ckpt_gate()
            self._confirm(
                f"READY for TIGHT execute? Hand on e-stop. "
                f"{self.cfg.clamp_tight_deg}°/step, {self.cfg.duration_tight_s:.0f}s. "
                f"DreamerV3 actor — expect jitter at clamp max (wiring validation).",
                safety_critical=True,
            )
            self._execute_dreamerv3_loop(
                duration_s=self.cfg.duration_tight_s,
                max_step_deg=self.cfg.clamp_tight_deg,
                step_label="tight",
            )
            return

        self._check_real_ckpt_gate()
        self._confirm(
            f"READY for tight execute? Hand on e-stop. "
            f"{self.cfg.clamp_tight_deg}°/step, {self.cfg.duration_tight_s:.0f}s.",
            safety_critical=True,
        )
        info(
            f"STEP 3: execute @ {self.cfg.clamp_tight_deg}° clamp, "
            f"{self.cfg.duration_tight_s:.0f}s"
        )
        run = self._find_runner_bin("robot-data-run")
        cmd = self._base_runner_cmd(
            run,
            duration_s=self.cfg.duration_tight_s,
            max_relative_target=self.cfg.clamp_tight_deg,
        )
        cmd += ["--execute"]
        if self.cfg.home_on_exit:
            cmd += ["--home-on-exit"]
        cmd += ["-v"]
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"tight execute failed rc={rc}")
        ok("tight execute complete — abort here if motion looked wrong")

    def step_execute_loose(self) -> None:
        if self._ckpt_kind == "dreamerv3":
            self._check_real_ckpt_gate()
            self._confirm(
                f"Step tight OK. Proceed to LOOSE execute? "
                f"{self.cfg.clamp_loose_deg}°/step, {self.cfg.duration_loose_s:.0f}s. "
                f"DreamerV3 actor — expect faster jitter at clamp max.",
                safety_critical=True,
            )
            self._execute_dreamerv3_loop(
                duration_s=self.cfg.duration_loose_s,
                max_step_deg=self.cfg.clamp_loose_deg,
                step_label="loose",
            )
            return

        self._check_real_ckpt_gate()
        self._confirm(
            f"Step 3 OK. Proceed to {self.cfg.clamp_loose_deg}°/step, "
            f"{self.cfg.duration_loose_s:.0f}s?",
            safety_critical=True,
        )
        info(
            f"STEP 4: execute @ {self.cfg.clamp_loose_deg}° clamp, "
            f"{self.cfg.duration_loose_s:.0f}s"
        )
        run = self._find_runner_bin("robot-data-run")
        cmd = self._base_runner_cmd(
            run,
            duration_s=self.cfg.duration_loose_s,
            max_relative_target=self.cfg.clamp_loose_deg,
        )
        cmd += ["--execute"]
        if self.cfg.home_on_exit:
            cmd += ["--home-on-exit"]
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"loose execute failed rc={rc}")
        ok("loose execute complete")

    def step_closed_loop(self) -> Path:
        self._check_real_ckpt_gate()
        self._confirm(
            f"Proceed to {self.cfg.n_eval_episodes}-episode closed-loop eval?",
            safety_critical=True,
        )
        info("STEP 5: closed-loop eval (prompt_user_observer scoring)")
        self.cfg.eval_output_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"laptop-{datetime.now().strftime('%Y-%m-%dT%H%M%S')}"
        out = self.cfg.eval_output_dir / f"{run_id}-closed-loop.json"

        # Write safety-ack if missing — operator already confirmed verbally above.
        if not self.safety_ack.exists():
            self.write_safety_ack()

        eval_bin = self._find_runner_bin("robot-data-run-eval")
        cmd = [
            eval_bin,
            "--policy-path", str(self.cfg.policy_path),
            "--dataset-root", str(self.cfg.dataset_root),
            "--port", self.cfg.port,
            "--camera", self.cfg.camera,
            "--rate-hz", str(self.cfg.rate_hz),
            "--max-relative-target", str(self.cfg.clamp_loose_deg),
            "--task", self.cfg.task,
            "--task-spec", "prompt_user_observer",
            "--n-episodes", str(self.cfg.n_eval_episodes),
            "--duration-per-episode-s", str(self.cfg.eval_duration_per_episode_s),
            "--output-json", str(out),
            "--i-have-read-the-safety-runbook",
        ]
        if self.cfg.home_on_exit:
            cmd.append("--home-on-exit")
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"closed-loop eval failed rc={rc}")
        ok(f"closed-loop eval written → {out}")
        return out

    # ----- helpers ------------------------------------------------------ #

    def _base_runner_cmd(
        self,
        runner_bin: str,
        *,
        duration_s: float,
        max_relative_target: float | None = None,
    ) -> list[str]:
        cmd = [
            runner_bin,
            "--policy-path", str(self.cfg.policy_path),
            "--dataset-root", str(self.cfg.dataset_root),
            "--port", self.cfg.port,
            "--camera", self.cfg.camera,
            "--rate-hz", str(self.cfg.rate_hz),
            "--duration-s", str(duration_s),
            "--task", self.cfg.task,
        ]
        if max_relative_target is not None:
            cmd += ["--max-relative-target", str(max_relative_target)]
        return cmd

    # ----- top-level ---------------------------------------------------- #

    def run(self) -> int:
        """Run the configured ladder. Returns the final exit code."""
        try:
            self._validate_inputs()
        except FileNotFoundError as exc:
            err(str(exc))
            return 2
        except RuntimeError as exc:
            # Checkpoint-kind gate refused with an actionable hint.
            err(str(exc))
            return 1

        info("laptop deploy session")
        info(f"  policy-path : {self.cfg.policy_path}")
        info(f"  dataset     : {self.cfg.dataset_root}")
        info(f"  port        : {self.cfg.port}")
        info(f"  camera      : {self.cfg.camera}")
        info(f"  task        : {self.cfg.task!r}")
        info(f"  dry-loop    : {self.cfg.do_dry_loop}")
        info(f"  execute     : {self.cfg.do_execute}")
        info(
            "  closed-loop : "
            + ("skip" if self.cfg.skip_closed_loop else f"{self.cfg.n_eval_episodes} eps")
        )
        info(f"  assume-yes  : {self.cfg.assume_yes}")
        info(f"  mock-hw     : {self.cfg.mock_hardware}")
        info(f"  require-real-ckpt: {self.cfg.require_real_ckpt}")

        try:
            self.step_preflight()
            if not (self.cfg.do_dry_loop or self.cfg.do_execute):
                ok("preflight only — done. Pass --dry-run-loop or --execute.")
                return 0
            self.step_dry_loop()
            if not self.cfg.do_execute:
                ok("dry-run only — done. Pass --execute to send motor commands.")
                return 0
            self.step_execute_tight()
            self.step_execute_loose()
            if self.cfg.skip_closed_loop:
                ok("skipped closed-loop eval — done.")
                return 0
            self.step_closed_loop()
        except RuntimeError as exc:
            err(str(exc))
            return 1

        ok("session complete")
        return 0


# ---------------------------------------------------------------------------
# Winner-JSON helper (consumes desktop output)
# ---------------------------------------------------------------------------


def resolve_winner_policy(winner_json: Path) -> Path:
    """Read a winner.json and return its ``winner_policy_path`` as a Path.

    The winner.json schema is produced by `_run_tonight_smolvla_12h.sh`
    STAGE 3 on the desktop. We don't import its schema validation here —
    just pluck the field and fail loud if missing.
    """
    data = json.loads(Path(winner_json).read_text(encoding="utf-8"))
    p = data.get("winner_policy_path")
    if not p:
        raise KeyError(f"winner JSON has no winner_policy_path: {winner_json}")
    return Path(p)


def resolve_winner_dataset_root(winner_json: Path) -> Path | None:
    """Read a (rewritten) winner.json, return ``dataset_root`` if present.

    Returns ``None`` when the JSON lacks the field — caller should fall
    back to the env var / hardcoded default.
    """
    data = json.loads(Path(winner_json).read_text(encoding="utf-8"))
    p = data.get("dataset_root")
    if not p:
        return None
    return Path(p)


def _hardcoded_dataset_fallback() -> Path:
    """The deploy-pkg-local default for ``dataset_root``."""
    return (
        Path.home() / "workspaces" / "lerobot-isaac-deploy"
        / "datasets" / "so101-pickplace1"
    )


def _resolve_dataset_root(
    cli_value: str | None,
    winner_json: Path | None,
) -> Path:
    """Apply the dataset-root precedence ladder.

    Order:
        1. Explicit ``--dataset-root`` (CLI flag).
        2. ``dataset_root`` field from ``--winner`` JSON.
        3. ``LEROBOT_ISAAC_DEPLOY_DATASET_ROOT`` env var.
        4. Hardcoded fallback under the deploy pkg's ``datasets/``.
    """
    if cli_value:
        return Path(cli_value)
    if winner_json is not None:
        try:
            from_winner = resolve_winner_dataset_root(winner_json)
        except (OSError, json.JSONDecodeError):
            from_winner = None
        if from_winner is not None:
            return from_winner
    env_val = os.environ.get("LEROBOT_ISAAC_DEPLOY_DATASET_ROOT", "").strip()
    if env_val:
        return Path(env_val)
    return _hardcoded_dataset_fallback()


# ---------------------------------------------------------------------------
# argparse wiring (used by cli.py)
# ---------------------------------------------------------------------------


def build_session_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="li-deploy-session",
        description=(
            "Confirm-gated SO-101 deploy ladder. Wraps the "
            "robot-data-runner CLI family."
        ),
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--policy-path", default=None,
                     help="pretrained_model/ directory")
    grp.add_argument("--winner", default=None,
                     help="winner.json from the desktop sweep (resolves --policy-path)")
    # Default is None so cfg_from_namespace() can detect explicit-vs-default
    # and apply the precedence ladder (winner.json → env → hardcoded).
    p.add_argument("--dataset-root", default=None,
                   help=("LeRobotDataset root used to recover preprocessor stats. "
                         "Precedence: explicit flag > winner.json[dataset_root] > "
                         "LEROBOT_ISAAC_DEPLOY_DATASET_ROOT env > "
                         "<deploy>/datasets/so101-pickplace1."))
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--camera", default="d435_rgb=/dev/video0,640,480")
    p.add_argument("--task", default="pick and place cube")
    p.add_argument("--rate-hz", type=float, default=30.0)
    p.add_argument("--duration-s", dest="duration_dry_s", type=float,
                   default=30.0,
                   help="duration of the dry / mock loop in seconds")
    p.add_argument("--clamp-tight", type=float, default=1.0)
    p.add_argument("--clamp-loose", type=float, default=3.0)
    p.add_argument("--dry-run-loop", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--skip-closed-loop", action="store_true")
    p.add_argument("--n-eval-episodes", type=int, default=10)
    p.add_argument("--no-home-on-exit", dest="home_on_exit",
                   action="store_false", default=True)
    p.add_argument("--safety-ack-only", action="store_true",
                   help="write the safety-ack file and exit")
    p.add_argument("--yes", "--assume-yes", dest="assume_yes",
                   action="store_true", default=False,
                   help=("auto-answer 'yes' on confirm prompts (env: "
                         "LEROBOT_ISAAC_DEPLOY_ASSUME_YES=1). NEVER "
                         "auto-yes the --execute / closed-loop gates — "
                         "the operator must still pass --execute "
                         "explicitly for motor writes."))
    p.add_argument("--mock-hardware", dest="mock_hardware",
                   action="store_true", default=False,
                   help=("skip the robot-data-run subprocess in the "
                         "dry-loop step and run an in-process synthetic-"
                         "obs inference loop. Incompatible with "
                         "--execute. Use for smoke tests without "
                         "serial port / camera."))
    p.add_argument("--require-real-ckpt", dest="require_real_ckpt",
                   action="store_true", default=False,
                   help=("refuse motor-write steps when the checkpoint "
                         "contains a synthetic_marker.json test fixture. "
                         "env: LI_DEPLOY_REQUIRE_REAL_CKPT=1."))
    return p


def cfg_from_namespace(ns: argparse.Namespace) -> SessionConfig:
    if ns.winner:
        policy_path = resolve_winner_policy(Path(ns.winner))
    elif ns.policy_path:
        policy_path = Path(ns.policy_path)
    else:
        raise SystemExit("--policy-path or --winner required")

    assume_yes = bool(ns.assume_yes) or _env_assume_yes()
    require_real_ckpt = bool(ns.require_real_ckpt) or _env_require_real_ckpt()
    winner_path = Path(ns.winner) if ns.winner else None
    dataset_root = _resolve_dataset_root(ns.dataset_root, winner_path)

    return SessionConfig(
        policy_path=policy_path,
        dataset_root=dataset_root,
        port=ns.port,
        camera=ns.camera,
        task=ns.task,
        rate_hz=ns.rate_hz,
        duration_dry_s=ns.duration_dry_s,
        clamp_tight_deg=ns.clamp_tight,
        clamp_loose_deg=ns.clamp_loose,
        do_dry_loop=bool(ns.dry_run_loop),
        do_execute=bool(ns.execute),
        skip_closed_loop=bool(ns.skip_closed_loop),
        n_eval_episodes=ns.n_eval_episodes,
        home_on_exit=bool(ns.home_on_exit),
        assume_yes=assume_yes,
        mock_hardware=bool(ns.mock_hardware),
        require_real_ckpt=require_real_ckpt,
    )
