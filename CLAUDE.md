# lerobot-isaac-deploy — Package Orientation

**Role:** Laptop-side orchestrator that walks an operator through the
SO-101 deploy ladder. Wraps `robot-data-runner`'s three CLIs
(`robot-data-run`, `robot-data-run-check`, `robot-data-run-eval`) with a
confirm-gated state machine.

**Working tree:** `~/workspaces/spinouts/lerobot_isaac_deploy/` (NOT a
bare repo — matches `robot_data_runner` and `robot_data_recorder`).

**Install (pixi, recommended for laptop):**

```bash
git clone https://github.com/kvgork/lerobot-isaac-deploy.git
cd lerobot-isaac-deploy
pixi install          # python + lerobot==0.5.1[smolvla] + robot-data-runner + this pkg
pixi run bootstrap    # workspace dirs + SmolVLM2-500M weight prefetch
```

**Install (pip alternative):**

```bash
pip install "lerobot-isaac-deploy[runner]"
li-deploy-bootstrap
```

**Why pixi:** matches the `lerobot-isaac-*` family pattern (the 6 bare-repo
siblings all carry an active `pixi.toml`). Pinned reproducible env on
the laptop, conda-managed system deps (`rsync`), pip-managed python deps
(lerobot pinned to 0.5.1 to match the desktop training env).

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
| `src/lerobot_isaac_deploy/policy_kind.py` | ckpt-kind detector (lerobot / dreamerv3 / lewm / vjepa / cosmos / gaia / unknown) + `is_synthetic` helper |
| `src/lerobot_isaac_deploy/wm_loader.py` | DreamerV3 actor loader + synthetic-marker stub; LeWM refuser |
| `src/lerobot_isaac_deploy/wm_rollout.py` | offline state-prediction rollout (Dreamer / LeWM) — emits `next_state_pred.npz` + `rollout_summary.json` |
| `src/lerobot_isaac_deploy/wm_video.py` | refusing stubs for V-JEPA / Cosmos / GAIA (deferred research) |
| `src/lerobot_isaac_deploy/mock_hardware.py` | `--mock-hardware` in-process smoke loops (LeRobot policies AND DreamerV3 actors) |
| `tests/test_imports.py` | smoke tests — no robot, no lerobot needed |
| `docs/world-model-deploy.md` | operator guide for deploying world models on SO-101 |

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

## WM deploy invariants

- `DeploySession.step_preflight` short-circuits when `_ckpt_kind != "lerobot"` — WM ckpts have no LeRobot `config.json`, so calling `robot-data-run-check` against them always fails. New ckpt kinds must add an entry in `_validate_inputs` AND ensure `step_preflight` skips them.
- `--require-real-ckpt` (or `LI_DEPLOY_REQUIRE_REAL_CKPT=1`) refuses motor writes against any ckpt with a `synthetic_marker.json`. The gate is enforced in `_validate_inputs` (early) AND inside each `step_execute_*`/`step_closed_loop` (defense-in-depth).
- Synthetic-marker fixtures (`<ckpt>/synthetic_marker.json`) let `pytest tests/` pass without `torch`/`sheeprl`/`h5py`/`stable-worldmodel` installed. See `docs/world-model-deploy.md` for the schema. They are test-only — `--require-real-ckpt` refuses them at runtime.

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
