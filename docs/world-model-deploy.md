# World-Model Deploy Reference

**TL;DR.** This package can deploy DreamerV3 actor-driven policies on the SO-101 arm,
run offline rollouts for DreamerV3 + LeWM, and detect (but refuse) V-JEPA / Cosmos / GAIA
checkpoints. Real-arm motor writes are gated behind `--require-real-ckpt` to prevent
accidental execution with synthetic test fixtures.

---

## Supported checkpoint kinds

| Kind | Detected by | Actor head? | Offline rollout? | Closed-loop? | Install hint |
|------|-------------|-------------|-----------------|--------------|--------------|
| `lerobot` | `model.safetensors` + `config.json` | yes (policy) | no | yes | default; no extras needed |
| `dreamerv3` | `.hydra/config.yaml` + `ckpt_*.ckpt` | yes (actor head) | yes | yes | `pip install sheeprl` (or synthetic fixture) |
| `lewm` | `leworldmodel_config.json` or `policy.json` with `"type":"le_world_model"` | no | yes | no | `pip install stable-worldmodel` or `le-wm` + `h5py` |
| `vjepa` | `vjepa_config.json` marker file | no | no | no | research phase only — see below |
| `cosmos` | `cosmos_config.json` marker file | no | no | no | research phase only — see below |
| `gaia` | `gaia_config.json` marker file | no | no | no | research phase only — see below |

---

## DreamerV3 closed-loop deploy

### Expected checkpoint directory layout

```
<run-dir>/
  .hydra/
    config.yaml       ← Hydra sweep config (agent + env)
  ckpt_<step>.ckpt    ← sheeprl checkpoint (actor + world model weights)
  synthetic_marker.json   ← OPTIONAL: present only in synthetic test fixtures
```

`detect_policy_kind` recognises a directory as `dreamerv3` when it contains both
`.hydra/config.yaml` and at least one `ckpt_*.ckpt` file.

### Step-by-step

**1. Detect the checkpoint kind.**

```bash
lerobot-isaac-deploy kind <run-dir>
# expected output: dreamerv3
```

**2. Smoke-test without hardware (recommended first step).**

```bash
li-deploy-session \
    --policy-path <run-dir> \
    --dataset-root <ds-root> \
    --dry-run-loop \
    --mock-hardware \
    --yes \
    --duration-s 5 \
    --rate-hz 5
```

The `--mock-hardware` flag skips the `robot-data-run` subprocess and calls
`run_mock_inference_loop_wm` instead. The DreamerV3 actor is loaded via
`wm_loader.load_dreamerv3`, synthetic zero-valued observations are generated
(6-DOF SO-101 schema), and `action={...}` lines are printed for
`duration_s * rate_hz` steps. No serial port, no camera, no motor writes.

If the checkpoint has `synthetic_marker.json`, `load_dreamerv3` uses a no-op
stub actor (returns zero actions). This lets `pytest tests/` pass without
torch or sheeprl installed.

**3. Full preflight only (loads and checks the policy, no motors).**

```bash
li-deploy-session \
    --policy-path <run-dir> \
    --dataset-root <ds-root>
# exits after step_preflight when no --dry-run-loop or --execute is given
```

Note: `step_preflight` calls `robot-data-run-check` which uses the lerobot policy
loader. For DreamerV3 checkpoints this will fail (rc=1) — the supported smoke
path is `--mock-hardware`. See `tests/test_wm.py::test_session_dreamerv3_fails_preflight_against_lerobot_runner`.

**4. Deploy to the real SO-101.**

```bash
li-deploy-session \
    --policy-path <run-dir> \
    --dataset-root <ds-root> \
    --dry-run-loop \
    --execute \
    --require-real-ckpt
```

`--execute` gates the 1° and 3°/step motor-write steps. `--require-real-ckpt`
refuses if the checkpoint contains `synthetic_marker.json`. Drop `--require-real-ckpt`
only if you deliberately want to run a synthetic fixture on real hardware (not recommended).

---

## DreamerV3 offline rollout

Runs the RSSM forward pass over episodes from a dataset, without any motor writes.
Useful for evaluating world-model prediction quality before real-arm deploy.

```bash
lerobot-isaac-deploy wm-rollout \
    --checkpoint <run-dir> \
    --dataset <ds-root> \
    --output-dir <out-dir> \
    --horizon-steps 50 \
    --n-seed-episodes 5
```

**Outputs written to `<out-dir>/`:**

| File | Shape | Description |
|------|-------|-------------|
| `next_state_pred.npz` | `(T, C, H, W)` | predicted next-frame image reconstructions, or `(T, latent_dim)` for latent-only mode |
| `rollout_summary.json` | — | scalar metrics + metadata |

**`rollout_summary.json` keys:**

```json
{
  "kind": "dreamerv3",
  "checkpoint": "/abs/path/to/run-dir",
  "dataset_root": "/abs/path/to/ds",
  "horizon": 50,
  "n_seed_episodes": 5,
  "mean_recon_loss": 0.0423,
  "synthetic": false
}
```

**On missing dependencies:** if `torch` or `sheeprl` are not installed, the CLI
exits with rc=2 and prints:

```
ImportError: torch/sheeprl not installed — run:
    pip install sheeprl  (or pixi install with the [wm] extra)
to use DreamerV3 rollouts.
```

**Synthetic-marker shortcut:** if the checkpoint contains `synthetic_marker.json`,
`_rollout_dreamerv3` generates plausible zero/noise outputs and writes the summary
without importing torch or sheeprl. The summary will carry `"synthetic": true`.

---

## LeWM offline rollout

LeWorldModel (LeWM) has no actor head and cannot drive the arm directly.
Offline rollout only.

```bash
lerobot-isaac-deploy wm-rollout \
    --checkpoint <lewm-run-dir> \
    --dataset <ds-root> \
    --output-dir <out-dir>
```

**`next_state_pred.npz` shape:** `(T, latent_dim)` — latent predictions, not image pixels.

**Required packages:**

```bash
pip install h5py
pip install stable-worldmodel   # or: pip install le-wm
```

**`rollout_summary.json` keys** follow the same schema as DreamerV3 but use
`mean_pred_loss` instead of `mean_recon_loss`:

```json
{
  "kind": "lewm",
  "checkpoint": "...",
  "dataset_root": "...",
  "horizon": 50,
  "n_seed_episodes": 1,
  "mean_pred_loss": 0.0871,
  "synthetic": false
}
```

**Closed-loop / real-arm deploy is NOT supported for LeWM.** The session ladder
refuses LeWM checkpoints with an actionable message:

```
LeWorldModel checkpoints have no actor head. Use
`lerobot-isaac-deploy wm-rollout` for offline rollouts.
```

The research path for adding an MPC/planning shim over LeWM latents is tracked in
vault `[[Model-Based-Planning-(Robot-Arms)]]` and `system-improvements.md`.

---

## Video world models (V-JEPA / Cosmos / GAIA) — deferred

These model families are **detected but refused** in this release.

| Model | Detection | Reason for refusal |
|-------|-----------|-------------------|
| V-JEPA | `vjepa_config.json` in ckpt dir | encoder-only; no actor head |
| NVIDIA Cosmos | `cosmos_config.json` in ckpt dir | generative data engine; no policy |
| GAIA-style | `gaia_config.json` in ckpt dir | generative video WM; no actor head |

Both the session ladder (`_validate_inputs`) and the stub loaders in
`wm_video.py` (`load_vjepa`, `load_cosmos`, `load_gaia`) raise
`WMDeployNotSupported` with a hint pointing at the LeRobot policy path
or this document.

Real loaders + planning shims are tracked in `system-improvements.md`.

---

## Real-arm gate — `--require-real-ckpt`

Pass `--require-real-ckpt` (or set `LI_DEPLOY_REQUIRE_REAL_CKPT=1`) when running
`--execute` or closed-loop eval to refuse motor writes against any checkpoint that
contains a `synthetic_marker.json` test fixture.

**Behavior:**

- When `require_real_ckpt=True` and `is_synthetic(policy_path)` returns True,
  the session exits rc=1 with:
  ```
  --require-real-ckpt: refusing motor write — checkpoint at <path> is a
  synthetic test fixture (has synthetic_marker.json). Provide a real ckpt
  or drop --require-real-ckpt.
  ```
- Default is `False` so mock-hardware smoke runs are unaffected.
- The env var `LI_DEPLOY_REQUIRE_REAL_CKPT=1` has the same effect as the flag.
- The gate fires in `_check_real_ckpt_gate()`, called from `step_execute_tight`,
  `step_execute_loose`, and `step_closed_loop`.

**Typical usage:**

```bash
# Smoke test — synthetic fixture OK, gate OFF (default)
li-deploy-session --policy-path tests/fixtures/dreamer-synthetic \
    --dataset-root tests/fixtures/ds \
    --dry-run-loop --mock-hardware --yes

# Real arm deploy — gate ON, synthetic fixtures refused
li-deploy-session --policy-path models/run1/ckpt_50000 \
    --dataset-root datasets/so101-pickplace1 \
    --dry-run-loop --execute --require-real-ckpt
```

---

## Synthetic-marker test fixtures

Any checkpoint directory may contain `synthetic_marker.json`. This file marks
the directory as a no-hardware test fixture and enables short-circuit paths
in `wm_loader` and `wm_rollout` that do not require torch, sheeprl, h5py, or a
real checkpoint file.

**Schema:**

```json
{
  "kind": "dreamerv3",
  "action_dim": 6,
  "image_shape": [3, 64, 64],
  "latent_dim": 192
}
```

| Field | Type | Description |
|-------|------|-------------|
| `kind` | string | `"dreamerv3"` or `"lewm"` |
| `action_dim` | int | number of motor joints (6 for SO-101) |
| `image_shape` | [C, H, W] | expected observation image shape |
| `latent_dim` | int or null | RSSM latent size; null for lerobot kind |

**Effect in `wm_loader.load_dreamerv3`:** returns a no-op stub actor that
yields zero-filled action tensors of shape `(action_dim,)` without importing
sheeprl or torch.

**Effect in `wm_rollout._rollout_dreamerv3` / `_rollout_lewm`:** generates
zero/noise arrays of the declared shape and writes them to
`next_state_pred.npz`. Writes `rollout_summary.json` with `"synthetic": true`.

**Effect in `is_synthetic(path)`:** returns `True` if and only if
`<path>/synthetic_marker.json` exists. Used by `_check_real_ckpt_gate()`.

This design lets `pytest tests/` run fully on a machine without any ML
framework installed.

---

## Cross-references

### Vault notes

- `[[DreamerV3]]` — architecture overview, sheeprl integration, actor head shape.
- `[[LeWorldModel]]` — offline-rollout adapter, stable-worldmodel API contract.
- `[[World-Model-Backbone-Catalogue]]` — comparison table of all WM families; detection heuristics.
- `[[Model-Based-Planning-(Robot-Arms)]]` — future research path for MPC over LeWM latents.

### In-repo

- `src/lerobot_isaac_deploy/policy_kind.py` — `detect_policy_kind`, `is_synthetic`, `PolicyKind`.
- `src/lerobot_isaac_deploy/wm_loader.py` — `load_dreamerv3`, `load_lewm`, `WMDeployNotSupported`.
- `src/lerobot_isaac_deploy/wm_video.py` — `load_vjepa`, `load_cosmos`, `load_gaia` (stubs).
- `src/lerobot_isaac_deploy/wm_rollout.py` — `rollout`, `build_rollout_parser`.
- `src/lerobot_isaac_deploy/mock_hardware.py` — `run_mock_inference_loop_wm`.
- `src/lerobot_isaac_deploy/session.py` — `SessionConfig.require_real_ckpt`, `_check_real_ckpt_gate`.
- `plans/2026-05-22-wm-deploy-on-so101.md` — full phasing plan + API contracts.
- `system-improvements.md` — deferred items: video WM loaders, LeWM-MPC planner.

### Video WM deploy

Video world model closed-loop deploy (V-JEPA action mapping, LeWM-MPPI planner,
Cosmos-based data augmentation for policy training) is a deferred research phase.
Tracked in `system-improvements.md`. These items will land in a follow-on branch after
the offline-rollout path is validated on real SO-101 data.
