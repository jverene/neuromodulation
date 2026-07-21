# GPU Runbook (Colab Pro A100)

How to run the full-scale experiments for `PRD.md`. Local CPU was only used
for smoke tests — the runs below are the ones that produce the paper data.

## What you need

- A Colab Pro account with an **A100 runtime** (Runtime → Change runtime
  type → A100 GPU, high-RAM).
- This repo on GitHub (recommended) or as an upload.
  - GitHub: from the repo root locally —
    `git remote add origin <your-repo-url> && git push -u origin main`
  - Or zip it (without `.venv/` and `results/`) and upload in Colab.
- Roughly **one day of A100 time** for the full sequence below (estimates
  per stage; E2 is the long one). Spread across sessions is fine —
  everything checkpoints.

## Order of operations (with sanity gates)

| # | Stage | Command | A100 estimate | Gate before continuing |
|---|-------|---------|---------------|------------------------|
| 1 | E0 baseline | `src.train --config configs/e0_baseline.yaml` | ~0.5–1 h | **M1**: `growth.gif` grows the lizard; `recovery.gif` recovers from the 20×20 lesion |
| 2 | E0' parent | `src.train --config configs/e0_channel_aware.yaml` | ~0.5–1 h | same acceptance |
| 3 | E1 sweep | `src.rollout --config configs/e1_lesion_sweep.yaml` | ~0.5–1 h | **M2**: fig1 curves look clean (ablated collapses only past perception radius) |
| 4 | E2 evolution | `src.evolve --config configs/e2_closedloop.yaml` | ~6–15 h | check `evolve_metrics.csv` fitness drops in the first ~20 gens |
| 5 | Figures | `src.figures --e1 ... --e2 ...` | 1 min | — |

Estimates are honest guesses from the architecture size, not measurements —
time the first 50 steps/gens and extrapolate before leaving a run
unattended.

## Cell-by-cell

**Cell 1 — runtime check + deps**

```python
!nvidia-smi
!pip install -U "jax[cuda12]" cax==0.3.3 evosax==0.2.0 numpy matplotlib imageio pyyaml
import jax; print(jax.devices())   # must show a cuda device
```

If Colab already had an older JAX loaded, restart the runtime once after the
install (Runtime → Restart), then re-run only the `import jax` line.

**Cell 2 — get the repo**

```python
!git clone <your-repo-url> nca-mod
%cd nca-mod
!python -m pytest tests/ -q   # optional: 34 tests, ~2 min on GPU
```

**Cell 3 — mount Drive and mirror results (Colab disconnects; PRD §12)**

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/nca-mod-results
# run this cell again any time to sync:
!rsync -a results/ /content/drive/MyDrive/nca-mod-results/
```

**Cell 4 — E0 (then STOP and check the gate)**

```python
!python -m src.train --config configs/e0_baseline.yaml
```

Open `results/<ts>_e0_baseline/growth.gif` and `recovery.gif`.
Gate: the lizard grows from a seed and regrows after the centered 20×20
lesion. If not, see "If E0 misbehaves" below before spending more hours.
Params land at `results/<ts>_e0_baseline/params.pkl`.

**Cell 5 — E0' channel-aware parent**

```python
!python -m src.train --config configs/e0_channel_aware.yaml
```

**Cell 6 — point configs at the trained parents (edit the YAMLs in place)**

```python
# Fill in the paths printed by cells 4-5:
# configs/e1_lesion_sweep.yaml:  model.parent_params -> results/<ts>_e0_channel_aware/params.pkl
# configs/e2_closedloop.yaml:    model.parent_params -> results/<ts>_e0_channel_aware/params.pkl
#                                model.baseline_params -> results/<ts>_e0_baseline/params.pkl
!sed -i 's|parent_params: null|parent_params: results/PASTE_E0P_DIR/params.pkl|' configs/e1_lesion_sweep.yaml configs/e2_closedloop.yaml
!sed -i 's|baseline_params: null|baseline_params: results/PASTE_E0_DIR/params.pkl|' configs/e2_closedloop.yaml
```

**Cell 7 — E1 (then STOP and check the gate)**

```python
!python -m src.rollout --config configs/e1_lesion_sweep.yaml
!python -m src.figures --e1 results/<ts>_e1_lesion_sweep --out figures
```

Gate (M2): `figures/fig1_recovery_vs_radius.png` shows clean recovery-vs-
radius curves. PRD §10: if these aren't clean by the checkpoint date, pivot
rather than burning E2 hours. Note: with `controller_params: null` the
"intact" arm uses a neutral controller — after E2 you can re-run E1 with
`controller_params: results/<ts>_e2_closedloop/controller_params.pkl` for
the evolved-controller version.

**Cell 8 — E2 evolution (the long one)**

```python
!python -m src.evolve --config configs/e2_closedloop.yaml
```

- Hard budget: 300 generations, pop 64, T=2000 (PRD §7).
- Progress: `results/<ts>_e2_closedloop/evolve_metrics.csv` (best fitness
  per gen) and `evolve_checkpoint.pkl` every 10 gens.
- After ~20 gens: fitness should trend down. If flat, check
  `rollout_closed_loop.gif` for fitness hacking (alpha suppression; if seen,
  set `evolve.alive_floor: 0.3` and restart — PRD §12).
- Disconnects: the run resumes manually — rerun the cell with
  `--eval-only results/<ts>_e2_closedloop/controller_params.pkl` to skip
  finished evolution and just do the condition evaluation.

**Cell 9 — figures + archive**

```python
!python -m src.figures --e1 results/<ts>_e1_lesion_sweep --e2 results/<ts>_e2_closedloop --out figures
!rsync -a results/ figures /content/drive/MyDrive/nca-mod-results/
```

Deliverables: `figures/fig1_recovery_vs_radius.png`,
`figures/fig2_hamming_vs_time.png`, `figures/table1_survival_halflife.{csv,md}`,
plus per-condition rollout GIFs in the E2 run dir.

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
- For a cheap E2 rehearsal, run with `evolve.num_generations: 30` first.
- Always sync to Drive before stopping the runtime.
