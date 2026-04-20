# TASK.md — Autonomous Experiment Runner

Run a queue of short (<15 min) experiments while user is away. Commit to a
dedicated branch, log to a dedicated W&B group, notify on phone only when
needed.

## Experiment Types
- **Training** (needs GPU retrain) → runs on the **vast.ai instance** via
  `train.py`.
- **Analysis** (reuses existing checkpoints in `wavefront_learning/results/`,
  which already contain 3 seeds per model) → runs **locally** on this machine.

## Model Sets
When the user says "all models": **FNO, DeepONet, WaveNOBare, WaveNOBiasOnly,
WaveNO**.
- **Baselines:** FNO, DeepONet.
- **Ablations:** WaveNOBare, WaveNOBiasOnly, WaveNO.

"All baselines" / "all ablations" refer to the corresponding subset. Training
on "all models" means one `train.py` run per model (× seeds if specified).

When the user describes the queue, classify each entry. **Always keep one
training job on vast and one analysis job local in flight concurrently** if
both types exist in the queue — launch the remote job (non-blocking) before
starting the local one, then poll.

## Constraints
- **One instance**, created once, never stopped/destroyed.
- **15 min cap** per experiment: wrap in `timeout 900`.
- **3 retries** per crash (diagnose → fix → commit → retry). Then log to
  `FAILURES.md` and move on.
- **No local prompts.** Unforeseen blockers → `PushNotification` + wait.
- **Grid-plot axes convention.** Whenever plotting a 2D field over the
  space–time grid (GT, prediction, error, residual), the plot MUST use
  **space on the x-axis and time on the y-axis**, `origin="lower"` so
  time increases upward. Use the label `x (space)` on the horizontal axis
  and `t (time)` on the vertical axis. Do not transpose to swap these.
- **Saturate the GPU.** Always push parallelism as high as the hardware
  supports — idle VRAM and idle SMs are wasted $/hr. Monitor with
  `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader`
  and `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  (per-process VRAM). Sizing rule: pick `PARALLEL` so peak concurrent
  VRAM ≲ 85 % of total; the per-process footprint is readable from the
  per-process query after the first experiment starts. Raise `PARALLEL`
  as soon as you see significant spare capacity (e.g. <50 % VRAM or
  <70 % GPU util sustained). Drop only on observed OOM. A starting guess
  of 3 is just that — a guess; escalate immediately once the real
  per-process memory is known. Each concurrent training must have a
  **unique `--save_path`** and a unique `--run_name`; failure to do so
  will clobber checkpoints and W&B runs. Same rule for local analysis
  jobs that don't fight for the same resource (CPU, disk).

## Instance Lifecycle
Kickoff (once):
```
uv run python /Users/mchami/ETH/Thesis/scripts/vast-instance.py --create-new
```
Save SSH info to `.vast_instance.json` (gitignored) at repo root:
```json
{"id": <id>, "ssh_host": "...", "ssh_port": <port>, "user": "root",
 "workspace": "/workspace/saint-venant/wavefront_learning"}
```
All SSH uses `-o StrictHostKeyChecking=accept-new`.

## Remote Setup (once, after create)
```
ssh ... "cd /workspace/saint-venant && uv sync"
ssh ... "cd /workspace/saint-venant/wavefront_learning && uv run python -c 'import torch;assert torch.cuda.is_available()'"
```
Verify `WANDB_API_KEY` on remote (copied via `vast.yaml` env_files).

## Branch & W&B
- Branch: `exp/<session-name>` from `main`, pushed immediately.
- Commits: **only code fixes** applied during the run (retries, bug fixes).
  Results/checkpoints/logs are gitignored, never committed. Message format:
  `fix(<exp-name>): <1-line>`.
- W&B experiment/group (`--exp`): short **keywords derived from the user's
  plain-language description** of that queue item (not the session name). E.g.
  a queue item "train all models on LWR varying n_samples" → `--exp per-n-samples`;
  "histogram of MSE per timestep bucket" → `--exp timestep-histogram`. Runs
  sharing the same semantic experiment share the same `--exp` tag.
- W&B run name (`--run_name`): the unique `<exp-name>` from the queue table
  (e.g. `001-fno-lwr-n50-s42`).
- Actual CLI flags are `--exp` and `--run_name` (Task.md previously referred
  to `--wandb_group` / `--wandb_name`, which don't exist).

## Session Folder
At kickoff, generate a random session slug (e.g. `petunia-fox`,
`amber-lattice` — two words, lowercase-hyphenated) and create
`wavefront_learning/results/session/<session-name>/`. Save the slug in
`.vast_instance.json` for reuse.

Layout:
```
results/session/<session-name>/
  queue.md                         # confirmed expanded queue
  <exp-name>/                      # one folder per experiment
    ckpts/                         # training checkpoints
    analysis/                      # analysis outputs (plots, csvs, md)
    <exp-id>.log                   # run log
```
`<exp-name>` = exp-id + short keywords of the experiment, e.g.
`001-fno-baseline-s0`, `007-waveno-rh-s2`. Keep under 40 chars, lowercase,
hyphenated.

Auto-inject `--save_path results/session/<session-name>/<exp-name>/ckpts/model.pth`
for training (the script's `--save_path` is a **file path**, not a directory —
`torch.save` fails with "invalid file name" if you pass a trailing `/`).
Analysis outputs go in the matching `analysis/` folder. Analysis is run as
the user requests (plain-language); do not auto-schedule.

## Remote → Local Mirror
The vast instance writes to
`/workspace/saint-venant/wavefront_learning/results/session/<session-name>/<exp-name>/`
(same relative path). After each remote experiment completes successfully,
`rsync` that experiment's folder back to the identical local path:
```
rsync -az -e "ssh -p <port> -o StrictHostKeyChecking=accept-new" \
  root@<host>:/workspace/saint-venant/wavefront_learning/results/session/<session-name>/<exp-name>/ \
  /Users/mchami/ETH/Thesis/saint-venant/wavefront_learning/results/session/<session-name>/<exp-name>/
```
**Do not commit** the synced artifacts — results and logs are gitignored
(see §Artifacts). Only code fixes are committed to the branch.

## Per-Experiment Loop
**Training (remote):** `ssh ... "cd <workspace> && git pull && timeout 900 <cmd>"`
in background (`run_in_background=true`), tee to `.vast_logs/<exp-id>.log`.
Monitor while local jobs run.

**Analysis (local):** run directly from `wavefront_learning/`, tee to
`.local_logs/<exp-id>.log`. Use `timeout 900` same as remote.

**Scheduling:** after each completion, immediately launch the next job of the
same kind (keep one remote + one local active whenever both kinds remain).

On failure (either kind): fetch last 200 lines, diagnose, apply one targeted
fix, commit to branch, retry. Max 3 attempts.

On 3x failure, append to `FAILURES.md`:
   ```
   ## <exp-id> — <name> — <UTC>
   Attempts: 3 | Last error: <summary>
   Fixes tried: <bullets>
   <details>Log excerpts…</details>
   ```
   Move on.

Never infinite-loop. Never "fix" by commenting out failing code.

## Phone Setup (user does once, before kickoff)
1. Launch this session with `claude --remote-control`.
2. On phone: install Claude mobile app (or open claude.ai/code), sign in
   with same Anthropic account, scan the QR / open the session URL.
3. Enable push notifications.

From phone, user can send new prompts, add experiments, cancel, or approve
blockers. Notify on: instance ready / 3x-failure / queue done / unforeseen
block.

## Experiment Queue
**Not in this file.** At session start, ask the user what experiments to run —
accept plain-language descriptions (e.g. "WaveNO baseline vs with RH loss, 3
seeds each" or "MSE-by-resolution plot over existing checkpoints"). For each:
- **Classify** as Training or Analysis (see §Experiment Types).
- **Translate** to a concrete command by reading the relevant script's args
  (`train.py` for training; the scripts in `results/` for analysis). Pick
  sensible defaults. Auto-inject W&B/save/seed flags for training:
  `--exp <keywords-from-user-prompt>` (e.g. `per-n-samples`,
  `timestep-histogram`), `--run_name <exp-name>` (the unique queue-table
  name like `001-fno-lwr-n50-s42`),
  `--save_path results/session/<session-name>/<exp-name>/ckpts/model.pth`
  (file path, NOT a directory), `--seed`.
  Runs sharing the same semantic experiment share the same `--exp`; each
  gets a unique `--run_name`.
- Assign exp-ids `001`, `002`, ... in order.

Show the user the expanded queue (with Type column) for one-shot confirmation
before kicking off.

**Adding experiments mid-session:** the user can send a new prompt at any time
(from this terminal or from phone via Remote Control) describing more
experiments. Between runs, check for any new user messages; if they contain
experiment requests, expand them the same way, assign next exp-ids, append to
`queue.md`, and include in the next available slot. Confirm the additions in
one line before running them.

## Kickoff (on session start)
1. Ask user for the experiment queue (see above). Wait for reply.
2. Verify `vastai` auth (`scripts/.env`).
3. Create instance (§Instance Lifecycle).
4. Save `.vast_instance.json`, run §Remote Setup.
5. Generate random `<session-name>` slug; save in `.vast_instance.json`.
6. Create+push branch `exp/<session-name>`.
7. Create session folder `results/session/<session-name>/` and write
   confirmed queue to `queue.md` inside it.
8. `PushNotification`: "vast ready, N experiments queued (<session-name>)".
9. Iterate queue per §Per-Experiment Loop. After each successful remote
   training, rsync artifacts back per §Remote → Local Mirror.
10. `PushNotification`: "queue done: X ok / Y failed (<session-name>)".

## Artifacts (all gitignored)
- `.vast_instance.json` — live SSH info + session-name.
- `results/session/` — all session outputs: ckpts, analysis, logs, queue.md,
  FAILURES.md. Nothing under this path is ever committed.

Add `results/session/` and `.vast_instance.json` to `.gitignore` at kickoff.

