"""mock_hardware — in-process inference loop with synthetic observations.

Purpose
-------
Smoke-test "can this machine actually run the model end-to-end?" without
needing a serial port (``/dev/ttyACM0``), a camera (``/dev/video0``), or
operator-in-the-loop confirms. Used by ``DeploySession.step_dry_loop``
when ``--mock-hardware`` is passed.

What it does
------------
1. Reuses ``robot_data_runner.policy_loader.load_policy`` — the exact
   loader the real preflight uses, proven to load SmolVLA/ACT/Diffusion
   checkpoints on this machine.
2. Inspects ``policy.config.input_features`` to learn the state
   dimensionality and per-camera image shapes the model expects.
3. Synthesises a single "observation" of zeros matching that schema and
   hands it to ``obs_to_policy_input`` (the same mapper the real runner
   uses) so the preprocessor pipeline runs unchanged.
4. Calls ``policy.select_action`` in a loop bounded by
   ``cfg.duration_dry_s × cfg.rate_hz`` steps, printing the emitted
   action vector each step.

What it deliberately does NOT do
--------------------------------
* No motor writes. There is no ``robot.send_action`` call anywhere in
  this module.
* No camera I/O. Image observations are zero arrays.
* No subprocess. The loop runs in the same Python process as the
  session driver, so no extra environment plumbing is needed.

Failure modes
-------------
* If lerobot / torch is not installed in the active env, ``load_policy``
  raises ``ImportError`` and we propagate exit code 2.
* If the policy crashes mid-step (e.g. shape mismatch on a malformed
  checkpoint), we log the error and return exit code 6 — matching the
  real ``runner.py`` exit code for "policy inference failed".
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot_isaac_deploy.session import SessionConfig

logger = logging.getLogger(__name__)


def _synthesize_observation(
    policy: Any,
    motor_names: list[str],
) -> dict:
    """Build a flat ``SO101Follower``-style observation dict from zeros.

    The shape of each field is derived from the policy's declared
    ``input_features``. We mimic the exact dict shape that the real
    ``robot.get_observation()`` returns so ``obs_to_policy_input`` can
    consume it unchanged.

    Observation schema produced
    ---------------------------
    * ``<motor>.pos`` → float (one per motor in ``motor_names``).
    * ``<camera_name>`` → ``np.ndarray`` of shape ``(H, W, 3)``,
      ``dtype=uint8``, all zeros.

    The camera name is taken from the lerobot feature key
    ``observation.images.<name>`` — stripping the prefix so
    ``obs_to_policy_input`` recreates the right key after its
    transpose.
    """
    import numpy as np

    obs: dict[str, Any] = {f"{m}.pos": 0.0 for m in motor_names}

    cfg = getattr(policy, "config", None)
    input_features = getattr(cfg, "input_features", None) or {}

    for feat_name, feat in input_features.items():
        if not feat_name.startswith("observation.images."):
            continue
        cam_name = feat_name[len("observation.images."):]
        shape = tuple(getattr(feat, "shape", ()) or ())
        # lerobot 0.5 declares image features in (C, H, W) order.
        # SO101Follower emits (H, W, C) uint8. obs_to_policy_input
        # handles the transpose, so we hand it (H, W, 3).
        if len(shape) == 3:
            c, h, w = shape
        elif len(shape) == 2:
            h, w = shape
            c = 3
        else:
            h, w, c = 480, 640, 3
        if c not in (1, 3):
            c = 3
        obs[cam_name] = np.zeros((h, w, c), dtype=np.uint8)
    return obs


def _infer_motor_names(policy: Any) -> list[str]:
    """Best-effort: derive motor names from the policy's state feature.

    SO-101 has six motors with conventional names. lerobot 0.5
    checkpoints embed the state-vector dimension in
    ``input_features['observation.state'].shape``; we use that to size
    the motor list.

    Falls back to the standard SO-101 motor names if the state feature
    is not introspectable.
    """
    so101_motors = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    cfg = getattr(policy, "config", None)
    feats = getattr(cfg, "input_features", None) or {}
    state = feats.get("observation.state")
    shape = tuple(getattr(state, "shape", ()) or ()) if state is not None else ()
    if shape and shape[-1] != len(so101_motors):
        # Non-standard state size — fall back to anonymous motor names.
        return [f"motor{i}" for i in range(shape[-1])]
    return so101_motors


def run_mock_inference_loop(cfg: "SessionConfig") -> int:
    """Run the synthetic-obs inference loop. Returns 0 on clean exit.

    Parameters
    ----------
    cfg:
        Live :class:`SessionConfig` — uses ``policy_path``,
        ``dataset_root``, ``task``, ``rate_hz``, ``duration_dry_s``.

    Exit codes mirror ``robot_data_runner.runner``:

    * 0 — clean
    * 2 — policy load failed (lerobot / torch missing or bad ckpt)
    * 6 — policy inference failed mid-loop
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Load the policy via the same path the real preflight uses.
    try:
        from robot_data_runner.policy_loader import load_policy
        from robot_data_runner.mappers import (
            action_to_robot_dict,
            obs_to_policy_input,
        )
    except ImportError as exc:
        logger.error(
            "mock-hardware needs robot-data-runner installed in the active "
            "env (same dep used by preflight). Got: %s",
            exc,
        )
        return 2

    try:
        loaded = load_policy(
            Path(cfg.policy_path),
            Path(cfg.dataset_root),
            seed=42,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("policy load failed: %s", exc)
        return 2

    policy = loaded.policy
    motor_names = _infer_motor_names(policy)
    logger.info(
        "mock-hardware: loaded %s on %s | motors=%s",
        type(policy).__name__,
        loaded.device,
        motor_names,
    )

    synth_obs = _synthesize_observation(policy, motor_names)
    cam_keys = [k for k in synth_obs if not k.endswith(".pos")]
    logger.info(
        "mock-hardware: synthetic obs keys: state(D=%d) + images=%s",
        sum(1 for k in synth_obs if k.endswith(".pos")),
        cam_keys,
    )

    n_steps = max(1, int(cfg.duration_dry_s * cfg.rate_hz))
    dt = 1.0 / cfg.rate_hz if cfg.rate_hz > 0 else 0.0
    logger.info(
        "mock-hardware: running %d steps @ %.1f Hz (dt=%.4fs) task=%r",
        n_steps,
        cfg.rate_hz,
        dt,
        cfg.task,
    )

    try:
        import torch  # local; loader already imported it.
    except ImportError as exc:
        logger.error("torch not importable: %s", exc)
        return 2

    rc = 0
    deadline = time.monotonic() + cfg.duration_dry_s
    for step in range(n_steps):
        if time.monotonic() >= deadline:
            break
        step_start = time.monotonic()
        try:
            with torch.no_grad():
                pol_in = obs_to_policy_input(
                    synth_obs, loaded.device, task=cfg.task
                )
                if loaded.preprocessor is not None:
                    pol_in = loaded.preprocessor(pol_in)
                action = policy.select_action(pol_in)
                if loaded.postprocessor is not None:
                    action = loaded.postprocessor(action)
            action_dict = action_to_robot_dict(action, motor_names)
        except Exception as exc:  # noqa: BLE001
            logger.error("mock-hardware: policy inference failed at step %d: %s",
                         step, exc)
            rc = 6
            break

        logger.info(
            "mock step %d action=%s",
            step,
            {k: round(v, 3) for k, v in action_dict.items()},
        )

        slack = dt - (time.monotonic() - step_start)
        if slack > 0:
            time.sleep(slack)

    logger.info("mock-hardware: %d step(s) complete (rc=%d)", step + 1, rc)
    return rc


def run_mock_inference_loop_wm(cfg: "SessionConfig") -> int:
    """Mock-hardware loop for a DreamerV3 actor head (or synthetic stub).

    Mirrors :func:`run_mock_inference_loop` but skips the lerobot
    policy-factory path. Loads via :func:`lerobot_isaac_deploy.wm_loader.load_dreamerv3`,
    synthesises 6-DOF SO-101 state + a single 64x64x3 image observation,
    iterates ``cfg.duration_dry_s * cfg.rate_hz`` steps, prints the
    action vector each step.

    Exit codes mirror :func:`run_mock_inference_loop`: 0 clean, 2 load failure, 6 inference failure.
    """
    import logging
    import time
    from pathlib import Path

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    _logger = logging.getLogger(__name__)

    try:
        from lerobot_isaac_deploy.wm_loader import load_dreamerv3
    except ImportError as exc:
        _logger.error("wm_loader import failed: %s", exc)
        return 2

    try:
        actor = load_dreamerv3(Path(cfg.policy_path))
    except Exception as exc:  # noqa: BLE001
        _logger.error("DreamerV3 actor load failed: %s", exc)
        return 2

    so101_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    obs = {
        "state": np.zeros((6,), dtype=np.float32),
        "image": np.zeros((3, 64, 64), dtype=np.uint8),
    }

    n_steps = max(1, int(cfg.duration_dry_s * cfg.rate_hz))
    dt = 1.0 / cfg.rate_hz if cfg.rate_hz > 0 else 0.0
    _logger.info("mock-hardware (wm): %d steps @ %.1f Hz task=%r", n_steps, cfg.rate_hz, cfg.task)

    rc = 0
    deadline = time.monotonic() + cfg.duration_dry_s
    step = 0
    for step in range(n_steps):
        if time.monotonic() >= deadline:
            break
        step_start = time.monotonic()
        try:
            action = actor.select_action(obs)
            # Normalise to ndarray
            if hasattr(action, "detach"):
                action = action.detach().cpu().numpy()
            action = np.asarray(action).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            _logger.error("wm-mock-hardware: inference failed at step %d: %s", step, exc)
            rc = 6
            break

        action_dict = {m: float(action[i]) if i < action.size else 0.0 for i, m in enumerate(so101_motors)}
        _logger.info("mock-wm step %d action=%s", step, {k: round(v, 3) for k, v in action_dict.items()})

        slack = dt - (time.monotonic() - step_start)
        if slack > 0:
            time.sleep(slack)

    _logger.info("mock-hardware (wm): %d step(s) complete (rc=%d)", step + 1, rc)
    return rc
