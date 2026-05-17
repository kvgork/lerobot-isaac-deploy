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
* `robot-data-runner` enforces server-side `--max-relative-target` clamp.
* A physical e-stop is mandatory hardware; this script cannot replace it.

## License

MIT.
