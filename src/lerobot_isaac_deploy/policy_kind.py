"""policy_kind — detect what kind of checkpoint a directory contains.

Returns one of:

* ``lerobot``    LeRobot 0.5+ checkpoint (lerobot-isaac-adapters / smolvla /
                 act / diffusion).  Identified by ``config.json`` +
                 ``model.safetensors`` + ``policy_preprocessor.json``.
* ``dreamerv3``  sheeprl DreamerV3 checkpoint.  Identified by
                 ``.hydra/config.yaml`` + ``ckpt_*.ckpt`` (.pt).
* ``lewm``       HF LeWorldModel checkpoint.  Identified by a top-level
                 ``leworldmodel_config.json`` or a ``policy.json`` whose
                 ``type`` is ``le_world_model``.
* ``vjepa``      V-JEPA video world model — encoder-only, no actor head.
                 Identified by ``vjepa_config.json``.
* ``cosmos``     NVIDIA Cosmos video WM — data engine, no actor head.
                 Identified by ``cosmos_config.json``.
* ``gaia``       GAIA-style generative video WM — no actor head.
                 Identified by ``gaia_config.json``.
* ``unknown``    Nothing matched; the caller should refuse to load.

Pure detection — does not import lerobot / sheeprl / torch. Safe to call
from any environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

PolicyKind = Literal["lerobot", "dreamerv3", "lewm", "vjepa", "cosmos", "gaia", "unknown"]


def detect_policy_kind(path: Path | str) -> PolicyKind:
    """Detect a checkpoint's kind by directory shape and tiny config peeks.

    Parameters
    ----------
    path:
        Directory containing the checkpoint (or a parent directory that
        contains one).  Common shapes:

        * lerobot:   ``<path>/pretrained_model/{config.json,model.safetensors,policy_preprocessor.json}``
                     OR ``<path>/{config.json,...}`` directly
        * dreamerv3: ``<path>/{.hydra/config.yaml,ckpt_*.ckpt}``
                     OR ``<path>/<run-name>/{.hydra/...,ckpt_*.ckpt}``
        * lewm:      ``<path>/leworldmodel_config.json`` OR
                     ``<path>/policy.json`` with type ``le_world_model``
        * vjepa:     ``<path>/vjepa_config.json``
        * cosmos:    ``<path>/cosmos_config.json``
        * gaia:      ``<path>/gaia_config.json``

    Returns
    -------
    PolicyKind
        ``"unknown"`` when nothing recognisable is found.
    """
    p = Path(path)
    if not p.exists():
        return "unknown"

    # Walk one level deep if the user pointed at a parent.
    candidates: list[Path] = [p]
    if p.is_dir():
        for sub in p.iterdir():
            if sub.is_dir():
                candidates.append(sub)

    for c in candidates:
        if not c.is_dir():
            continue

        # LeRobot — must have model.safetensors + (config.json or train_config.json)
        if (c / "model.safetensors").is_file() and (
            (c / "config.json").is_file() or (c / "train_config.json").is_file()
        ):
            return "lerobot"

        # DreamerV3 sheeprl — has .hydra/config.yaml + a ckpt_*.ckpt file
        if (c / ".hydra" / "config.yaml").is_file():
            if any(c.glob("ckpt_*.ckpt")) or any(c.glob("**/ckpt_*.ckpt")):
                return "dreamerv3"
            # No ckpt file yet — still claim dreamerv3 by the hydra signature.
            return "dreamerv3"

        # LeWM — explicit marker file
        if (c / "leworldmodel_config.json").is_file():
            return "lewm"

        # LeWM via policy.json type field
        pj = c / "policy.json"
        if pj.is_file():
            try:
                data = json.loads(pj.read_text(encoding="utf-8"))
                if data.get("type") == "le_world_model":
                    return "lewm"
            except Exception:  # noqa: BLE001
                pass

        # V-JEPA video world model — encoder-only, no actor
        if (c / "vjepa_config.json").is_file():
            return "vjepa"

        # NVIDIA Cosmos — generative data engine, no actor
        if (c / "cosmos_config.json").is_file():
            return "cosmos"

        # GAIA-style generative video WM — no actor
        if (c / "gaia_config.json").is_file():
            return "gaia"

    return "unknown"


def explain(kind: PolicyKind) -> str:
    """Human-readable one-line description for logging."""
    return {
        "lerobot":   "LeRobot policy (smolvla / act / diffusion) — direct deploy",
        "dreamerv3": "DreamerV3 (sheeprl) — actor head used for deploy",
        "lewm":      "HF LeWorldModel — no actor; use wm-rollout for offline sim",
        "vjepa":     "V-JEPA video world model — no robot-control path (deferred research)",
        "cosmos":    "NVIDIA Cosmos video WM — data engine, no actor (deferred research)",
        "gaia":      "GAIA-style generative video WM — no actor (deferred research)",
        "unknown":   "unknown — cannot route to any loader",
    }[kind]


def is_synthetic(path: Path | str) -> bool:
    """True iff <path>/synthetic_marker.json exists.

    Used by motor-write gates to refuse real-arm execution against
    a fixture/test checkpoint. See plans/2026-05-22-wm-deploy-on-so101.md.
    """
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False
    # Check the path itself and one level deep — same shape as detect_policy_kind.
    if (p / "synthetic_marker.json").is_file():
        return True
    for sub in p.iterdir():
        if sub.is_dir() and (sub / "synthetic_marker.json").is_file():
            return True
    return False
