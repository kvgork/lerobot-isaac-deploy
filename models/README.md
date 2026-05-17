# models/

Canonical drop zone for trained policy checkpoints on the laptop.

## Layout

```
models/
├── README.md                  ← this file
├── .gitignore                 ← keeps the dir tracked, ignores everything inside
├── <run_name>/                ← one dir per training run
│   ├── <ckpt_id>/             ← e.g. 045000/, or `last/`
│   │   └── pretrained_model/  ← rsync-ed by `pixi run sync-ckpt`
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── policy_preprocessor.json
│   │       ├── policy_preprocessor_step_*_normalizer_processor.safetensors
│   │       ├── policy_postprocessor.json
│   │       ├── policy_postprocessor_step_*_unnormalizer_processor.safetensors
│   │       └── train_config.json
│   └── manifest.json          ← optional dashboard sidecar
└── ...
```

## How it gets populated

`pixi run sync-ckpt` (run on the **desktop**) rsyncs the latest
checkpoint of a given training-run dir to:

```
laptop:~/workspaces/lerobot-isaac-deploy/models/<run_name>/<ckpt_id>/pretrained_model/
```

The destination is auto-created via SSH `mkdir -p` before rsync runs.

## How it gets used

`pixi run session` (run on the **laptop**) takes a `--policy-path`
relative to this directory:

```bash
cd ~/workspaces/lerobot-isaac-deploy
pixi run session -- \
    --policy-path models/trial_7/045000/pretrained_model \
    --execute
```

## Override the destination

Send a checkpoint to an external drive or a custom layout:

```bash
pixi run sync-ckpt -- \
    --run-dir <desktop-run-dir> \
    --remote-dir /mnt/nvme/models/my-experiment
```

## Why is this committed?

Only `README.md` and `.gitignore` are tracked. The `models/` directory
itself is a contract: `pixi run sync-ckpt` defaults to writing here,
and `pixi run session` defaults to reading from here. The .gitignore
keeps every actual checkpoint (large `.safetensors` blobs) out of git
while ensuring the directory always exists after a fresh clone.
