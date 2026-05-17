# lerobot-isaac-deploy

Laptop-side deploy session orchestrator for SO-101 trained policies.

Pairs with [`robot-data-runner`](https://github.com/kvgork/robot_data_runner)
(the low-level deploy CLI) and the
[`lerobot-isaac-training`](https://github.com/kvgork/lerobot-isaac-training)
workspace (the desktop trainer).

## What this package is

A **confirm-gated ladder** that walks an operator through the SO-101
deploy lifecycle on the laptop:

```
preflight  →  dry-run loop  →  execute 1°  →  execute 3°  →  closed-loop eval
```

Each step prompts `yes` to advance. The whole sequence is one console
script:

```bash
lerobot-isaac-deploy session \
    --winner ~/path/to/winner.json \
    --execute
```

## Install

**Recommended (pixi, fresh laptop):**

```bash
git clone https://github.com/kvgork/lerobot-isaac-deploy.git
cd lerobot-isaac-deploy
pixi install                  # installs python + lerobot + robot-data-runner + this pkg
pixi run bootstrap            # prefetches SmolVLM2-500M (~6.7 GB) + workspace dirs
```

**Alt (pip into an existing venv):**

```bash
pip install "lerobot-isaac-deploy[runner]"
li-deploy-bootstrap
```

**Pixi task shortcuts:**

```bash
pixi run session -- --winner /path/to/winner.json --execute
pixi run sync-ckpt -- --run-dir outputs/overnight-smolvla-<ts>-anchor
pixi run sync-eval
pixi run test
```

## Smoke test without hardware

Verify the checkpoint actually loads and emits actions on this machine
*before* you plug the SO-101 in. Combines two flags:

* `--yes` (or `--assume-yes`, or env `LEROBOT_ISAAC_DEPLOY_ASSUME_YES=1`)
  — auto-answers the confirm prompts so the run is non-interactive.
* `--mock-hardware` — skips the `robot-data-run` subprocess in the
  dry-loop step and instead runs an in-process inference loop with
  synthetic (zero-filled) observations sized to the policy's declared
  `input_features`. No `/dev/ttyACM0`, no `/dev/video0`, no motor writes.

```bash
pixi run session -- \
    --winner ~/path/to/winner.json \
    --dataset-root ~/path/to/datasets/<user>/<dataset> \
    --dry-run-loop --mock-hardware --yes \
    --duration-s 5 --rate-hz 5     # short for smoke
```

Expected: preflight loads the policy, the mock loop prints
`mock step N action={...}` for `duration_s × rate_hz` steps, then exits 0.

### Safety: `--yes` does NOT auto-confirm motor writes

The two `--execute` gates (1° and 3° clamps) and the closed-loop eval
prompt are flagged `safety_critical` internally. `--yes` is **ignored**
for those — the operator must still type `yes` on stdin, AND the
`--execute` flag itself must be passed explicitly. This is intentional
defense-in-depth: a CI run or a stuck shell pipe cannot accidentally
drive the robot.

`--mock-hardware` is incompatible with `--execute`. The session refuses
the combination early (exit 1).

## Subcommands

| Command             | Runs on   | What it does |
|---------------------|-----------|--------------|
| `li-deploy-session` | laptop    | Confirm-gated dry → 1° → 3° → eval ladder |
| `li-deploy-sync-ckpt` | desktop | rsync ckpt → laptop |
| `li-deploy-sync-eval` | desktop | rsync eval JSONs ← laptop |
| `li-deploy-bootstrap` | laptop  | One-shot env + weight prefetch |

## Hybrid workflow (desktop trains, laptop deploys)

```bash
# Desktop, overnight:
bash scripts/_run_tonight_smolvla_12h.sh    # in lerobot-isaac-training/

# Desktop, morning:
li-deploy-sync-ckpt --run-dir outputs/overnight-smolvla-<ts>-anchor

# Laptop:
li-deploy-session --winner $HOME/workspaces/lerobot-isaac-deploy/checkpoints/.../winner.json --execute

# Desktop, after laptop session:
li-deploy-sync-eval
```

## Safety contract

* `--execute` is required to send motor commands. Default = dry-run.
* Each motor-write step prompts `yes` on stdin first.
* `--yes` / `--assume-yes` never auto-confirms a motor-write step.
* `robot-data-runner` enforces server-side `--max-relative-target` clamp.
* A physical e-stop is mandatory hardware; this script cannot replace it.

## License

MIT.
