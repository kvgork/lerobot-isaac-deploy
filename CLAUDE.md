# lerobot-isaac-deploy — Package Orientation

**Role:** Laptop-side orchestrator that walks an operator through the
SO-101 deploy ladder. Wraps `robot-data-runner`'s three CLIs
(`robot-data-run`, `robot-data-run-check`, `robot-data-run-eval`) with a
confirm-gated state machine.

**Working tree:** `~/workspaces/spinouts/lerobot_isaac_deploy/` (NOT a
bare repo — matches `robot_data_runner` and `robot_data_recorder`).

**Install:**

```bash
pip install lerobot-isaac-deploy[runner]
# or on the laptop, one-shot:
li-deploy-bootstrap
```

---

## Console scripts

```
lerobot-isaac-deploy <subcommand>   umbrella
li-deploy-session                   ladder runner (laptop)
li-deploy-sync-ckpt                 desktop → laptop ckpt rsync
li-deploy-sync-eval                 laptop → desktop eval JSON rsync
li-deploy-bootstrap                 one-shot laptop env setup
```

## File map

| File | Role |
|------|------|
| `src/lerobot_isaac_deploy/__init__.py` | exports `DeploySession`, `SessionConfig` |
| `src/lerobot_isaac_deploy/session.py` | confirm-gated ladder (preflight → dry → 1° → 3° → closed-loop) |
| `src/lerobot_isaac_deploy/sync.py` | rsync wrappers for ckpt + eval |
| `src/lerobot_isaac_deploy/bootstrap.py` | laptop env + weight prefetch |
| `src/lerobot_isaac_deploy/cli.py` | argparse dispatch + console entries |
| `tests/test_imports.py` | smoke tests (10 cases) — no robot, no lerobot needed |

## Coupling

| Dep | Hard / Soft | Notes |
|-----|-------------|-------|
| `numpy` | hard | tiny — used for shape sanity |
| `rsync` | hard | system binary; assumed on PATH |
| `robot-data-runner` | soft (extras `[runner]`) | session looks up its CLIs via `shutil.which` at run time |
| `huggingface_hub` | lazy | only imported by `bootstrap.prefetch_smolvlm2` |

## Why a separate package

Same reasoning as `robot_data_runner` (parent rationale):

- **Hardware-bound code lives outside the training meta repo** so
  workspaces without an SO-101 don't pull the dep.
- The package can be released to PyPI on its own cadence — bug fixes
  ship per-repo, not gated by a monorepo bump.
- The laptop bootstrap can `pip install lerobot-isaac-deploy` straight
  from PyPI without needing the full training workspace cloned locally.

## Spinout status

Phase 0 — package skeleton + tests + console scripts. Logic ported from
the `scripts/_run_laptop_deploy_session.sh` + `scripts/laptop_bootstrap.sh`
+ `scripts/sync_*.sh` triplet in the training workspace (those bash
scripts remain as backwards-compatible wrappers for users who haven't
upgraded yet).

## Related

- Sibling on the same laptop: `robot_data_runner` (provides the CLIs
  this package orchestrates).
- Counterpart on the desktop: `lerobot-isaac-training/scripts/_run_tonight_smolvla_12h.sh`
  (overnight trainer + re-rank that produces `winner.json` consumed here).
- Deprecated console name: `lerobot-isaac-deploy` previously pointed at
  `lerobot_isaac_adapters.deploy:main` (the in-process open-loop deploy
  shim). That entry has been replaced by this package's CLI; the adapter
  module is kept only for backwards import compatibility.
