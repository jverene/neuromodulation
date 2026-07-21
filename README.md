# Closed-Loop Neuromodulation in NCAs

JAX pipeline for the 6-week sprint in `PRD.md` (E0–E2): train a Growing NCA,
add global modulator channels with a learned controller, evolve the
controller's release policy with CMA-ES against a long-horizon
recurring-damage objective, and produce Figure 1 (lesion-size ×
channel-ablation) and Figure 2 (closed-loop vs. baselines).

**Status: fully implemented and smoke-verified on CPU. Full-scale runs
(E0/E0' training, E2 evolution) are GPU-scale — see [COLAB.md](COLAB.md) for
the step-by-step A100 runbook.**

## Stack

Python 3.11+, JAX, CAX 0.3.3, Flax (nnx), Optax, Evosax 0.2.0, NumPy,
Matplotlib, imageio, PyYAML. Logging: CSV + Matplotlib only (no wandb).

## Layout

```
configs/   e0_baseline.yaml, e0_channel_aware.yaml, e1_lesion_sweep.yaml, e2_closedloop.yaml
src/
  nca.py         # NCA model (CAX ConvPerceive + NCAUpdate), params IO
  targets.py     # emoji loading (cached in assets/), premultiply, padding
  train.py       # E0 pool training + mid-episode damage, checkpoints, GIFs
  damage.py      # disc/multi_disc/edge_disc lesions, recurring schedules, seed split
  channels.py    # modulator state: tonic (EMA a=0.95) + phasic (tau=20)
  controller.py  # controller MLP 4->32->K (functional params), target-free stats
  rollout.py     # closed-loop rollout engine (5 modes), E1 lesion sweep CLI
  metrics.py     # hamming, repair half-life, survival, AUC
  evolve.py      # Evosax CMA-ES loop + E2 5-condition evaluation
  figures.py     # fig1/fig2/table1 from results/
tests/     34 tests: nca, damage, channels, leakage, rollout, metrics
results/   runs/<ts>/{config.yaml, metrics.csv, params.pkl, *.gif}  (gitignored)
```

## Local setup & tests

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/          # 34 tests
```

## Pipeline entry points

```bash
python -m src.train   --config configs/e0_baseline.yaml        # E0 (K=0)
python -m src.train   --config configs/e0_channel_aware.yaml   # E0' parent (K=3, m=0)
python -m src.rollout --config configs/e1_lesion_sweep.yaml    # E1 sweep
python -m src.evolve  --config configs/e2_closedloop.yaml      # E2 evolution + conditions
python -m src.figures --e1 results/<ts>_e1_lesion_sweep --e2 results/<ts>_e2_closedloop
```

`--smoke` on E1/E2 runs a tiny CPU-feasible config. `--steps N` overrides
E0 training length. After E0/E0', fill in `model.parent_params` /
`model.baseline_params` / `controller_params` in the E1/E2 configs with the
printed run-dir paths.

## Verification status

Verified locally (2026-07-20, Apple M3 CPU):

- 34/34 unit + smoke tests pass.
- E2 smoke end-to-end: CMA-ES (3 gens), constant-tonic grid search, all 5
  conditions, trajectories/metrics/GIFs written.
- E1 smoke end-to-end; fig1/fig2/table1 render correctly.
- GIF path (growth + recovery) verified.
- Learning signal: 300-step small-canvas run, MSE 2.35e-2 -> 7.09e-3 (-70%).

Not verified locally (needs the A100 run): full E0 acceptance (8000-step
growth + centered 20x20 lesion recovery) and real fitness improvement over
300 generations. The local box was too saturated to even reach step 50 in
4.6 h of wall time — this is a hardware limitation, not a code issue.

## Design decisions / PRD notes

- **Perception = 3 kernels (48 dims).** PRD §3 says "identity + Sobel-x +
  Sobel-y + Laplacian (48 dims)" — 4 kernels would be 64 dims; 48 is what the
  rest of the PRD uses (MLP 48->128->16, injection 48+K=51), and CAX
  `grad_kernel` has no Laplacian (`grad2_kernel` exists if you want it).
- **CAX conventions**: alpha is the LAST channel; RGBA = last 4 channels.
- **Channel-aware parent (PRD §5)** is trained with the controller frozen at
  neutral (m=0), so its K modulator-input weights get zero gradient. If the
  evolved controller shows no effect, retrain the parent with
  `train.mod_noise: true` in `configs/e0_channel_aware.yaml` (escape hatch)
  so the channel weights are exercised during training.
- **Evosax 0.2.0**: fitness is MINIMIZED (fitness = mean Hamming, no sign
  flip); `std_init` lives in `default_params`, not the constructor; the PyPI
  README snippet is stale. Solutions are plain pytrees (controller params as
  nested dicts), vmapped directly.
- **PRNG keys are passed as arrays** through the rollout engine (no traced
  ints through jit); dropout randomness flows via `nnx.split_rngs` +
  `nnx.StateAxes`, as in the CAX examples.
- Controller input stats are structurally target-free
  (`tests/test_leakage.py` enforces it); the Hamming-proxy reference mask is
  the rollout's own t=0 pattern, never the target.
- Evolution fitness = mean Hamming over T=2000 with recurring damage on the
  train damage-seed set; the held-out test damage-seed set is always
  reported at the end (PRD §12). `evolve.alive_floor` (default 0) adds an
  alive-fraction floor if fitness hacking is observed in rollout GIFs.

## Milestones (PRD §10)

M1 E0 acceptance · M2 E1 data + rollout speed ≥ 50 pop-parallel (batch
rollouts are fully vmapped; 512-way in the E2 default config) · M3 E2 results
· M4 figures + draft · M5 submission. Kill criteria and non-goals per PRD
§10-11 — nothing in the sprint non-goals list is implemented here.
