"""DeploySession — confirm-gated SO-101 deploy ladder.

Subprocess-driven wrapper around the ``robot-data-runner`` CLI family.
Steps:

    1. preflight                robot-data-run-check   (no robot)
    2. dry-run loop             robot-data-run          (robot, no motors)
       (or in-process mock loop when ``--mock-hardware`` is set)
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
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
            return

        if kind == "dreamerv3":
            # DreamerV3 is first-class: actor head used for closed-loop deploy.
            self._ckpt_kind = "dreamerv3"
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
        run = self._find_runner_bin("robot-data-run")
        cmd = self._base_runner_cmd(run, duration_s=self.cfg.duration_dry_s)
        cmd += ["-v"]
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            raise RuntimeError(f"dry-run loop failed rc={rc}")
        ok("dry-run complete — verify action lines made sense")

    def step_execute_tight(self) -> None:
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
