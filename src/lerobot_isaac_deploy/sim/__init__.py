"""Sim-deploy backends for closed-loop policy eval.

Currently shipped:
  * IsaacSceneSession — Isaac Sim + isaac-auto-scene USD scene.

See `plans/2026-05-23-sim-deploy-pipeline.md` in the training workspace
for the build plan.
"""
from __future__ import annotations

__all__ = ["IsaacSceneSession"]


def __getattr__(name: str):
    # Lazy-import to avoid pulling Isaac Sim into module-load when callers
    # only want the package's other (lightweight) APIs.
    if name == "IsaacSceneSession":
        from .isaac_scene_session import IsaacSceneSession
        return IsaacSceneSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
