# datasets/

Canonical drop zone for the LeRobotDataset trees the synced checkpoints
were trained against on the laptop.

The preprocessor that ships inside `pretrained_model/` needs the dataset
it was trained on to recover normalizer stats + feature shapes — without
the matching dataset on disk, the policy fails to initialize. This folder
is the laptop-side mirror of the desktop's training datasets.

## Layout

```
datasets/
├── README.md          ← this file
├── .gitignore         ← keeps the dir tracked, ignores everything inside
├── <dataset_name>/    ← one dir per dataset, e.g. so101-pickplace1/
│   ├── data/          ← Parquet chunks (LeRobotDataset format)
│   ├── images/        ← optional image side-car (if --image-mode=jpeg)
│   └── meta/          ← episode_index.parquet, info.json, stats.json
└── ...
```

## How it gets populated

`pixi run sync-winner` (run on the **desktop**) ships both the
checkpoint AND its matching dataset in a single command. The dataset
basename is preserved on the laptop:

```
desktop:$LEROBOT_ISAAC_TRAIN_WS/datasets/kvgork/so101-pickplace1/
    → laptop:~/workspaces/lerobot-isaac-deploy/datasets/so101-pickplace1/
```

The destination is auto-created via SSH `mkdir -p` before rsync runs.

`pixi run sync-winner` reads the dataset path from the autoresearch
`program.json` next to the winner.json (training workspace's
`.agent-state/<session>-ar/autoresearch/<slug>/program.json`). If it
cannot find a program.json it logs a warning and skips the dataset
rsync — the operator may have already shipped it manually.

## How it gets used

The rewritten `winner.json` that lands at `models/<run>/winner.json`
carries a `dataset_root` field pointing at this folder. `pixi run
deploy-winner` reads it and passes it through to `li-deploy-session`
automatically — no `--dataset-root` flag needed.

The resolution precedence inside `session.py` is:

1. Explicit `--dataset-root` flag (highest).
2. `dataset_root` field in `--winner` JSON.
3. `LEROBOT_ISAAC_DEPLOY_DATASET_ROOT` env var.
4. Hardcoded fallback `<deploy>/datasets/so101-pickplace1`.

## Manual override

Send a dataset to an external drive:

```bash
# desktop
pixi run sync-ckpt -- \
    --run-dir <desktop-run-dir> \
    --dataset-root /home/koen/workspaces/lerobot-isaac-training/datasets/kvgork/so101-pickplace1 \
    --remote-dir /mnt/nvme/models/my-experiment
```

(The dataset still lands under `<laptop_base>/datasets/<basename>/`
regardless of `--remote-dir` — datasets and ckpts are independent on
the laptop, since multiple ckpts may share a single dataset.)

## Why is this committed?

Only `README.md` and `.gitignore` are tracked. The `datasets/`
directory itself is a contract: `pixi run sync-winner` defaults to
writing here, and `session.py`'s fallback default reads from here.
The `.gitignore` keeps every actual Parquet chunk (datasets are large)
out of git while ensuring the directory always exists after a fresh
clone.
