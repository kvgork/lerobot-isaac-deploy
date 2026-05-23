"""bootstrap — one-shot laptop env setup helpers.

Replaces the procedural `scripts/laptop_bootstrap.sh` from the training
workspace with a python orchestrator. Idempotent.

What this does on a fresh laptop:

1. Create ``$HOME/workspaces/lerobot-isaac-deploy/`` and subdirs.
2. Install ``robot-data-runner`` + ``lerobot==<pinned>`` into a venv.
3. Prefetch SmolVLM2-500M-Video-Instruct weights (~6.7 GB).
4. Drop a sentinel marker so re-runs short-circuit.

The lerobot version is read from this package's own metadata at
install time so the laptop matches the desktop's tested version.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_LEROBOT_VERSION = "0.5.1"
DEFAULT_BASE = Path.home() / "workspaces" / "lerobot-isaac-deploy"
HF_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def ensure_workspace(base: Path = DEFAULT_BASE) -> Path:
    """Create the standard laptop deploy workspace layout."""
    base = Path(base)
    for sub in ("checkpoints", "datasets", "outputs", "outputs/eval", "logs"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def pip_install_runtime(
    *,
    lerobot_version: str = DEFAULT_LEROBOT_VERSION,
    extra_index: str | None = None,
) -> int:
    """Install lerobot + robot-data-runner via pip into the current Python."""
    args = [sys.executable, "-m", "pip", "install"]
    if extra_index:
        args += ["--extra-index-url", extra_index]
    args += [
        f"lerobot[smolvla]=={lerobot_version}",
        "robot-data-runner>=0.1.0",
        "lerobot-isaac-deploy>=0.1.0",
    ]
    return subprocess.run(args, check=False).returncode


def pip_install_dreamerv3() -> int:
    """Install sheeprl from git master (PyPI wheels pin python<3.12).

    Required only when a DreamerV3 ckpt is being deployed. Skipped by
    default — call explicitly via `--with-dreamerv3` on `li-deploy-bootstrap`.
    """
    args = [
        sys.executable, "-m", "pip", "install",
        "--ignore-requires-python",
        "sheeprl @ git+https://github.com/Eclectic-Sheep/sheeprl.git",
    ]
    print(f"[bootstrap] {' '.join(args)}")
    return subprocess.run(args, check=False).returncode


def prefetch_smolvlm2() -> int:
    """Download the SmolVLM2-500M backbone into ~/.cache/huggingface/hub/."""
    cache_marker = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{HF_REPO.replace('/', '--')}"
    )
    if cache_marker.is_dir():
        print(f"[bootstrap] {HF_REPO} already cached at {cache_marker}")
        return 0
    print(f"[bootstrap] downloading {HF_REPO} (~6.7 GB)…")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[bootstrap] huggingface_hub not installed; install lerobot first",
              file=sys.stderr)
        return 127
    p = snapshot_download(repo_id=HF_REPO)
    print(f"[bootstrap] cached at {p}")
    return 0


def write_sentinel(base: Path) -> Path:
    """Write a small marker file recording the bootstrap version + date."""
    from datetime import UTC, datetime

    sentinel = base / ".bootstrap.json"
    sentinel.write_text(
        '{"bootstrapped": "%s", "lerobot": "%s"}\n'
        % (datetime.now(UTC).isoformat(), DEFAULT_LEROBOT_VERSION),
        encoding="utf-8",
    )
    return sentinel


def main(argv: list[str] | None = None) -> int:
    """Combined idempotent bootstrap. Returns 0 on success."""
    import argparse

    p = argparse.ArgumentParser(prog="li-deploy-bootstrap")
    p.add_argument("--base", default=str(DEFAULT_BASE))
    p.add_argument("--skip-pip", action="store_true",
                   help="skip pip install (use when env is managed by pixi/conda)")
    p.add_argument("--skip-prefetch", action="store_true",
                   help="skip SmolVLM2 weight prefetch")
    p.add_argument("--with-dreamerv3", action="store_true",
                   help="install sheeprl (from git master, "
                        "--ignore-requires-python on Py3.12) for "
                        "DreamerV3 ckpt deploy")
    p.add_argument("--lerobot-version", default=DEFAULT_LEROBOT_VERSION)
    ns = p.parse_args(argv)

    base = ensure_workspace(Path(ns.base))
    print(f"[bootstrap] workspace ready at {base}")

    if not ns.skip_pip:
        rc = pip_install_runtime(lerobot_version=ns.lerobot_version)
        if rc != 0:
            return rc
    if ns.with_dreamerv3:
        rc = pip_install_dreamerv3()
        if rc != 0:
            return rc
    if not ns.skip_prefetch:
        rc = prefetch_smolvlm2()
        if rc != 0:
            return rc

    sentinel = write_sentinel(base)
    print(f"[bootstrap] done — sentinel {sentinel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
