# GPU Runbook (Vast.ai A100)

How to run the full-scale experiments for `PRD.md` on a rented A100 instead of
Colab. Local CPU was only used for smoke tests — the runs below are the ones
that produce the paper data.

This is the Vast.ai counterpart to [COLAB.md](COLAB.md). The science, configs,
gates, and entry points are **identical**; only the platform plumbing changes.
The big tradeoff: Vast.ai is pay-per-hour and your data **persists on disk**
between sessions (no Drive-mounting ritual, no 12 h idle disconnect nuking an
overnight E2 run), but you pay for every hour the instance exists and you are
responsible for stopping/destroying it.

## Colab vs. Vast.ai — when to pick which

| | Colab Pro | Vast.ai |
|---|-----------|---------|
| Cost model | Flat ~$10–20/mo subscription | Pay per GPU-hour (~$0.7–1.9/hr A100) |
| Best for | Many short sessions over a month | A few long unattended runs |
| Disconnect risk | Idle timeout reaps the VM (~12 h, sometimes less) | No idle timeout, but **interruptible** offers can be preempted by the host |
| Data persistence | Wiped on disconnect — must rsync to Drive | `/workspace` survives **stop**; only **destroy** wipes it |
| Setup | One notebook, magics + Drive | CLI + SSH + Docker image |

Rule of thumb: if your real cost is "one ~15 h E2 run plus a few hours of E0/E1,"
Vast.ai is cheaper than a Colab Pro month and won't drop your overnight run.
If you'll iterate daily for weeks, Colab Pro's flat rate wins.

## What you need

- A [vast.ai](https://vast.ai/) account with **credits loaded** (card or
  prepaid). Check the live A100 price at
  [vast.ai/pricing](https://vast.ai/pricing) before renting — listed A100
  SXM4 80GB is ~$0.77/hr but real on-demand asks run $0.7–1.9/hr.
- The `vastai` CLI on your laptop.
- This repo on GitHub — already pushed to
  `https://github.com/jverene/neuromodulation.git` (confirmed via
  `git remote -v`), so cloning inside the instance is one line.
- A budget. The full sequence is ~8–18 GPU-hours of compute; at ~$1/hr that's
  roughly **$8–20 of GPU time** plus storage. Set this aside and set a spend
  alert in the Vast.ai dashboard.

## The one Vast.ai concept that matters most

Every instance has a persistent disk mounted at **`/workspace`**.

- **Files in `/workspace` survive `stop`.** You pay a much lower storage-only
  rate while stopped. Stopping is how you pause billing without losing the
  trained params / checkpoints mid-sprint.
- **`destroy` wipes everything**, including `/workspace`. Copy results out
  (Cell 9 / `vastai copy`) before destroying.
- Everything **outside** `/workspace` (pip installs into the container FS,
  your git clone if you put it in `/root`) is tied to that specific instance.
  It survives `stop`/`start` of the *same* instance, but is lost on `destroy`.
  ⇒ **Clone the repo and install deps inside `/workspace`** so a recreated
  instance only needs a `pip install` to be runnable again.

## Order of operations (with sanity gates)

| # | Stage | Command | A100 estimate | Gate before continuing |
|---|-------|---------|---------------|------------------------|
| 1 | Rent + provision | (CLI, below) | setup | `nvidia-smi` + `jax.devices()` show A100 |
| 2 | E0 baseline | `python -m src.train --config configs/e0_baseline.yaml` | ~0.5–1 h | **M1**: `growth.gif` grows the lizard; `recovery.gif` recovers from the 20×20 lesion |
| 3 | E0' parent | `python -m src.train --config configs/e0_channel_aware.yaml` | ~0.5–1 h | same acceptance |
| 4 | E1 sweep | `python -m src.rollout --config configs/e1_lesion_sweep.yaml` | ~0.5–1 h | **M2**: fig1 curves look clean (ablated collapses only past perception radius) |
| 5 | E2 evolution | `python -m src.evolve --config configs/e2_closedloop.yaml` | ~6–15 h | check `evolve_metrics.csv` fitness drops in the first ~20 gens |
| 6 | Figures | `python -m src.figures --e1 ... --e2 ...` | 1 min | — |

Estimates are honest guesses from the architecture size, not measurements —
time the first 50 steps/gens and extrapolate before leaving a run unattended.

## Step by step

### Step 1 — install the CLI and authenticate (on your laptop)

```bash
pip install vastai
vastai --help                       # sanity check
vastai set api-key PASTE_YOUR_KEY   # from https://cloud.vast.ai/account/cli/
```

### Step 2 — find an A100 offer

```bash
# min 40GB VRAM (A100 40 or 80), decent reliability, sort by perf/$
vastai search offers 'gpu_ram >= 40000 reliability > 0.95 rentable = true' \
  -o 'dlperf_usd-'
```

Pick an offer ID from the output. Notes:
- `reliability > 0.95` avoids flaky hosts. For the long E2 run, prefer hosts
  marked **verified** and avoid the cheapest **interruptible** offers — a host
  preemption mid-evolution is the main way to lose hours here.
- `gpu_ram >= 40000` admits both A100 40GB and 80GB; the 512-way batched E2
  rollouts are comfortable on either. Use `>= 76000` to force 80GB.
- A100 SXM4 80GB lists ~$0.77/hr; expect $0.7–1.9/hr on-demand in practice.

### Step 3 — rent it (a CUDA 12 image you can `pip install jax` into)

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime \
  --disk 80 \
  --ssh --direct \
  --onstart-cmd "nvidia-smi"
```

We use a PyTorch CUDA image only because it ships a reliable Python + CUDA 12
runtime — **the project does not use torch**; it's just a known-good base to
`pip install jax[cuda12]` into. `--disk 80` reserves 80 GB on `/workspace`
(plenty for params, checkpoints, and rollout GIFs).

### Step 4 — connect and verify the GPU

```bash
vastai ssh-url INSTANCE_ID          # prints: ssh root@HOST -p PORT
ssh root@HOST -p PORT               # use the printed host/port
# --- now inside the instance ---
nvidia-smi                          # must show an A100
```

### Step 5 — clone the repo into /workspace and install deps

Do this inside `/workspace` so a recreated instance is easy to rebuild:

```bash
cd /workspace
git clone https://github.com/jverene/neuromodulation.git nca-mod
cd nca-mod
pip install -U "jax[cuda12]" -r requirements.txt
python -c "import jax; print(jax.devices())"   # MUST print a CudaDevice
python -m pytest tests/ -q                     # optional: 34 tests, ~2 min
```

If `jax.devices()` does not show a `CudaDevice`, the JAX/CUDA versions
mismatched — match the `[cuda12]` wheel to the image's CUDA (12.4 here) and
reinstall. Do not proceed until a CUDA device is visible; everything downstream
assumes GPU.

### Step 6 — E0 baseline (then STOP and check the gate)

```bash
python -m src.train --config configs/e0_baseline.yaml
```

~0.5–1 h. Then **open the GIFs before continuing** — pull them to your laptop:

```bash
# from your laptop:
vastai copy INSTANCE_ID:/workspace/nca-mod/results/ local:./results/
```

Open `results/<ts>_e0_baseline/growth.gif` and `recovery.gif`. **Gate (M1):**
the lizard grows from a seed and regrows after the centered 20×20 lesion. If
not, see "If E0 misbehaves" below before spending more hours. Note the run dir
path — Step 8 needs it. Params land at `results/<ts>_e0_baseline/params.pkl`.

### Step 7 — E0' channel-aware parent

```bash
python -m src.train --config configs/e0_channel_aware.yaml
```

Same ~0.5–1 h, same acceptance check. Note this run dir too.

### Step 8 — point configs at the trained parents

The E1/E2 YAMLs have `parent_params: null` / `baseline_params: null`
placeholders that must point at the params you just trained. Edit the two
files on the instance (e.g. `nano configs/e1_lesion_sweep.yaml`):

- `configs/e1_lesion_sweep.yaml` → `model.parent_params` = the **E0′
  channel-aware** run dir's `params.pkl`
- `configs/e2_closedloop.yaml` → `model.parent_params` = the **E0′
  channel-aware** dir's `params.pkl`, and `model.baseline_params` = the **E0
  baseline** dir's `params.pkl`

Editing by hand is less error-prone than the `sed` one-liner in COLAB.md.

### Step 9 — E1 sweep (then STOP and check the gate)

```bash
python -m src.rollout --config configs/e1_lesion_sweep.yaml
python -m src.figures --e1 results/<ts>_e1_lesion_sweep --out figures
```

~0.5–1 h. **Gate (M2):** `figures/fig1_recovery_vs_radius.png` shows clean
recovery-vs-radius curves. PRD §10: if these aren't clean by the checkpoint
date, pivot rather than burning E2 hours. Note: with `controller_params: null`
the "intact" arm uses a neutral controller — after E2 you can re-run E1 with
`controller_params: results/<ts>_e2_closedloop/controller_params.pkl` for the
evolved-controller version.

### Step 10 — E2 evolution (the long one)

Before launching the full run, do the cheap rehearsal from "Cost control"
below (`evolve.num_generations: 30`) once to confirm the loop and time a gen.

```bash
python -m src.evolve --config configs/e2_closedloop.yaml
```

- Hard budget: 300 generations, pop 64, T=2000 (PRD §7).
- Progress: `results/<ts>_e2_closedloop/evolve_metrics.csv` (best fitness per
  gen) and `evolve_checkpoint.pkl` every 10 gens.
- After ~20 gens: fitness should trend **down** (fitness = mean Hamming,
    minimized). If flat, check `rollout_closed_loop.gif` for fitness hacking
    (alpha suppression; if seen, set `evolve.alive_floor: 0.3` and restart —
    PRD §12).
- **Leaving it unattended:** use `tmux` so an SSH drop doesn't kill the run:
  ```bash
  tmux new -s e2
  python -m src.evolve --config configs/e2_closedloop.yaml
  # Ctrl-B then D to detach; `tmux attach -t e2` to return
  ```
- **After an interruption:** the run resumes manually — rerun the cell with
  `--eval-only results/<ts>_e2_closedloop/controller_params.pkl` to skip
  finished evolution and just do the condition evaluation. Checkpoints are
  every 10 gens, so you lose ≤10 gens of progress.

### Step 11 — figures + pull results

```bash
python -m src.figures --e1 results/<ts>_e1_lesion_sweep --e2 results/<ts>_e2_closedloop --out figures
```

Deliverables: `figures/fig1_recovery_vs_radius.png`,
`figures/fig2_hamming_vs_time.png`, `figures/table1_survival_halflife.{csv,md}`,
plus per-condition rollout GIFs in the E2 run dir. Copy them to your laptop:

```bash
# from your laptop, before destroying the instance:
vastai copy INSTANCE_ID:/workspace/nca-mod/figures local:./figures/
vastai copy INSTANCE_ID:/workspace/nca-mod/results/  local:./results/
```

### Step 12 — stop or destroy the instance (control your spend)

```bash
vastai stop instance INSTANCE_ID      # pauses billing to storage-only; data kept
vastai destroy instance INSTANCE_ID   # PERMANENT — /workspace wiped; copy first
```

- **Stop** if you might run more experiments later this week — you keep the
  trained params and the pip-installed environment, paying only storage.
- **Destroy** when you're done and have copied everything out. This is the
  only way to fully stop billing.

## If E0 misbehaves (PRD §12 gotchas)

- Alpha blowups / pool instability in the GIFs → lower
  `train.learning_rate` to `1.0e-3` (grad clipping is already on).
- Nothing grows after ~2k steps → check `metrics.csv` loss trend; reseed
  (`seed: 1`) before changing anything else.
- Checkpoints every 500 steps (`params_step*.pkl`) mean you can resume a
  crashed run by loading the latest into a fresh model — or just restart;
  E0 is the cheap stage.

## Cost control

- Do the gates. Do not start E2 before E0/E0' GIFs and E1 curves look right.
- For a cheap E2 rehearsal, run with `evolve.num_generations: 30` first, time
  a generation, then extrapolate to 300 before committing.
- Prefer **non-interruptible / verified** offers for the E2 run; a preemption
  loses ≤10 gens (checkpoint cadence) but is still annoying at 3am.
- Run everything under **`tmux`** so an SSH disconnect doesn't terminate the
  process.
- **Always `vastai copy` results out before `destroy`.** Stopping is safe;
  destroying is not.
- Keep an eye on the dashboard spend counter; set a budget alert.
