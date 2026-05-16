"""lerobot-isaac-deploy — laptop-side deploy session orchestrator.

Wraps the ``robot-data-runner`` CLI family (``robot-data-run``,
``robot-data-run-check``, ``robot-data-run-eval``) with a confirm-gated
ladder: preflight → dry-run loop → 1° execute → 3° execute →
closed-loop N-episode eval.

The session module is the operator-facing entry; sync + bootstrap are
helpers for the desktop↔laptop hybrid workflow.
"""

from lerobot_isaac_deploy.session import DeploySession, SessionConfig

__version__ = "0.1.0"
__all__ = ["DeploySession", "SessionConfig", "__version__"]
