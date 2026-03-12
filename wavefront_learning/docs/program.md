# autoresearch

This is an experiment to have the LLM autonomously research and improve a neural operator for 1D conservation laws (LWR traffic flow / shallow water equations). The model learns to map piecewise-constant initial conditions to full space-time solution grids.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git fetch origin main && git checkout main && git pull origin main && git checkout -b autoresearch/<tag>`.
3. **Read the in-scope files**: Read these files for full context:
   - `wavefront_learning/CLAUDE.md` — module-level conventions, available models, losses, and code style rules.
   - `wavefront_learning/train.py` — training orchestration (model creation, data loading, training, evaluation).
   - `wavefront_learning/model.py` — model registry and factory.
   - `wavefront_learning/configs/training.yaml` — default hyperparameters and CLI defaults.
   - `wavefront_learning/models/waveno.py` — the WaveNO model family (the model you are optimizing).
   - `wavefront_learning/testing/test_results.py` — evaluation pipeline (standard, high-res, step-count tests).
   - `wavefront_learning/testing/test_running.py` — profiling and sanity checks.
   - `wavefront_learning/losses/` — browse available loss functions.
   - `wavefront_learning/models/base/` — shared building blocks used by WaveNO.
4. **Initialize results.tsv**: Create `wavefront_learning/results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script is launched from inside `wavefront_learning/`:

```bash
cd wavefront_learning
uv run python train.py --model WaveNO --n_samples 500 --exp autoresearch --run_name "<description of this change>" > run.log 2>&1
```

The `--exp autoresearch` flag groups all runs under the same W&B experiment. The `--run_name` must be a short, descriptive name of the specific change being tested (e.g. "increase LR to 0.003", "add skip connections in decoder").

**What you CAN modify:**
- `wavefront_learning/train.py` — training orchestration, optimizer, scheduler, hyperparameters.
- `wavefront_learning/models/` — model architecture, building blocks, new components.
- `wavefront_learning/losses/` — loss functions, loss presets.
- `wavefront_learning/configs/` — configuration files, presets, default values.
- `wavefront_learning/training_loop.py` — the inner training loop.
- `wavefront_learning/loss.py` — loss factory and preset registration.
- `wavefront_learning/model.py` — model registry.
- You may install new packages if needed (`uv add <package>`).

**What you CANNOT modify:**
- `wavefront_learning/data/` — data pipeline is fixed.
- `wavefront_learning/testing/` — evaluation harness is fixed.
- `wavefront_learning/metrics.py` — metric computation is fixed.
- `wavefront_learning/plotting/` — visualization is fixed.
- `wavefront_learning/logger.py` — logging utilities are fixed.

**Hard constraint — no autoregressive operations**: This project learns a neural operator that maps initial conditions to full space-time solutions in a single forward pass. There must NEVER be any autoregressive, time-stepping, or recurrent operations in the model. The model takes the IC and produces the entire `(nt, nx)` grid at once.

**The goal: minimize the composite MSE score.** The model is evaluated on 3 scenarios, each weighted equally at 1/3:

1. **Standard MSE** (`test/metrics/mse`): MSE on the standard test set (same resolution and step count as training).
2. **Step-count generalization MSE** (`test_steps/*/mse`): MSE averaged equally across all sub-tests from `max_steps=2` to `max_steps=max_test_steps`. Each sub-test has equal weight.
3. **High-resolution generalization MSE** (`test_high_res/*/mse`): MSE averaged equally across all sub-tests from `2x` to `max_high_res`x resolution. Each sub-test has equal weight.

The **composite score** is:

```
score = (1/3) * standard_mse + (1/3) * mean(step_mses) + (1/3) * mean(highres_mses)
```

Lower is better. Use the default `--max_test_steps` (10) and `--max_high_res` (5) values and `--max_steps` (4).

**Time budget**: Each experiment has a wall-clock budget of **10 minutes**. If a run exceeds 20 minutes, kill it and treat it as a failure.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful MSE gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script with the default WaveNO configuration and no modifications.

## Output format

Once the script finishes it prints test results like this:

```
Test Results:
----------------------------------------
  mse: 0.001234
  mae: 0.023456
  rel_l2: 0.045678
  max_error: 0.234567
----------------------------------------

High-Res Test (2x):
----------------------------------------
  mse: 0.001500
  ...

Step-Count Test (max_steps=5):
----------------------------------------
  mse: 0.001800
  ...
```

Extract the key metrics from the log file using section headers as anchors (a bare `grep "mse:"` also picks up training-epoch loss lines — avoid that):

```bash
# Standard test MSE
grep -A2 "^Test Results:" run.log | grep "mse:"

# High-res MSEs (one per resolution multiplier)
grep -A2 "^High-Res Test" run.log | grep "mse:"

# Step-count MSEs (one per step count)
grep -A2 "^Step-Count Test" run.log | grep "mse:"
```

To compute the composite score:
- `base_mse` = the single MSE from `Test Results`
- `highres_mse` = mean of all high-res sub-test MSEs (equal weight each)
- `steps_mse` = mean of all step-count sub-test MSEs (equal weight each)
- `composite = (1/3) * base_mse + (1/3) * highres_mse + (1/3) * steps_mse`

## Logging results

When an experiment is done, log it to `wavefront_learning/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	base_mse	steps_mse	highres_mse	composite	status	description
```

1. git commit hash (short, 7 chars)
2. standard test MSE (e.g. 0.001234) — use 0.000000 for crashes
3. mean step-count MSE (e.g. 0.001800) — use 0.000000 for crashes
4. mean high-res MSE (e.g. 0.001500) — use 0.000000 for crashes
5. composite score (e.g. 0.001511) — use 0.000000 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	base_mse	steps_mse	highres_mse	composite	status	description
a1b2c3d	0.001234	0.001800	0.001500	0.001511	keep	baseline
b2c3d4e	0.001100	0.001600	0.001400	0.001367	keep	increase LR to 0.003
c3d4e5f	0.001300	0.002000	0.001800	0.001700	discard	switch to GeLU activation
d4e5f6g	0.000000	0.000000	0.000000	0.000000	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar11`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Modify files with an experimental idea.
3. git commit the changes.
4. Run the experiment:
   ```bash
   cd wavefront_learning
   uv run python train.py --model WaveNO --n_samples 500 --epochs 100 --plot grid_residual --exp autoresearch --run_name "<description>" > run.log 2>&1
   ```
   Redirect everything — do NOT use tee or let output flood your context.
5. Read out the results:
   ```bash
   grep "mse:" run.log
   ```
6. If the grep output is empty or doesn't contain test results, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
7. Compute the composite score from the 3 MSE scenarios and record the results in `wavefront_learning/results.tsv`. (NOTE: do not commit results.tsv, leave it untracked by git.)
8. If the composite score improved (lower), you "advance" the branch, keeping the git commit.
9. If the composite score is equal or worse, you `git reset --hard` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~10 minutes total. If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**Research-first, not hyperparameter-first**: Prioritize correct architecture for the problem, appropriate loss functions, and deep analysis of where the model fails. Do NOT spend most of your time tuning hyperparameters (LR, dropout, scheduler settings, etc.). Instead: (1) analyze error patterns to understand *why* the model fails, (2) search online for what the research community does for this class of problems, (3) implement architectural or loss changes that address the root cause. Hyperparameter sweeps are low-value — a well-chosen architecture with default hyperparameters beats a poorly-chosen architecture with perfect hyperparameters.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the model code for new angles, re-read the CLAUDE.md for architectural patterns, try combining previous near-misses, try more radical architectural changes, explore different loss functions or training schedules. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~20 minutes then you can run ~3/hour, for a total of about 24 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
