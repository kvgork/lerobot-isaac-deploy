# System Improvements — lerobot-isaac-deploy

Pending work surfaced during orchestration runs. Each entry is a deferred capability or systemic gap; tracked here so it doesn't get lost.

## 2026-05-22 — WM deploy ladder (post-`feature/world-model`)

### Deferred capabilities (intentional — out of scope this PR)

- **LeWM planning shim (MPC / CEM / MPPI over latent rollouts).** Required for LeWM to drive motors. Current state: `wm_loader.load_lewm` refuses with `WMDeployNotSupported`. References: vault `[[Model-Based-Planning-(Robot-Arms)]]`, `[[LeWorldModel]]` (rollout horizon 10, K=64 elites of N=512 samples for CEM).
- **V-JEPA / Cosmos / GAIA real loaders.** `wm_video.py` stubs refuse with `WMDeployNotSupported`. Real loaders + (where applicable) policy distillation or LeRobot-policy bridging deferred until at least one real ckpt exists. References: vault `[[V-JEPA-2-AC]]`, `[[Cosmos-Pipeline-for-Robotics]]`.
- **`wm_rollout._rollout_dreamerv3` real LeRobotDataset loader.** Currently synthesises zero seed observations. Should read the first frame of `n_seed_episodes` from the supplied `dataset_root` via `lerobot.LeRobotDataset`. Also: hook the actual `world_model.decoder` to produce non-stub reconstructions (currently produces zero-array predictions even when `loaded.encoder`/`loaded.actor` are real). Marked `partial: True` + `decoder_implemented: False` in the summary JSON.
- **`wm_rollout._rollout_lewm` real predict body.** Currently writes a zero-filled summary marked `partial: True` + `predictor_implemented: False`. Needs `stable_worldmodel` / `le_wm` integration once the deps land and a real ckpt is available. Same shape: read seed frames from `dataset_root`, call `model.predict(...)` for `horizon` steps.

### Test honesty

- `tests/test_wm.py::test_wm_rollout_stub_returns_2_for_unimpl` passes for the wrong reason now — it relies on `_RolloutInstallError` raising (torch absent in pytest env) rather than `NotImplementedError`. Should either mock `import torch` to force the path, OR drop the test (the synthetic-marker tests already cover the happy path).
- `tests/test_wm.py::test_session_dreamerv3_preflight_skipped_for_wm_kind` was tightened in 2026-05-22 to assert `rc=0` after the preflight short-circuit. Still doesn't exercise an actual subprocess — fine for unit but a future smoke-on-fixture-env test would be useful.

### Pipeline lessons (no agent file change needed)

- Code-review-orchestrator OK'd the real-arm gate placement that the verification-loop later flagged as unreachable for WM ckpts. The two steps catch different defect classes (semantic vs reachability). Keep both gates active — don't let `RIGOR=vibe` ever skip 7.5b on this codebase.

### Future research phase (not blocked on this PR)

- LeWM-MPC closed-loop on real arm. Requires planning shim above + a trained LeWM ckpt + the safety gate to be exercised against a real ckpt artifact.
- Imagined-planning policy (MPC over WM, then execute first action) — same prereqs.
- V-JEPA-2-AC as an action-conditioned policy on SO-101 — needs vault `[[V-JEPA-2-AC]]` patterns + action head training pipeline.

---

*Older entries are appended at the bottom in reverse-chronological order.*
