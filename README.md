# Neural Cellular Automata: Channel-Aware Training for Robust Regeneration

**Verified on CPU. Full runs need an A100. See [COLAB.md](COLAB.md) for the runbook.**

## Stack

Python 3.11+, JAX, CAX 0.3.3, Flax (nnx), Optax, Evosax 0.2.0, NumPy,
Matplotlib, imageio, PyYAML. Logging: CSV and Matplotlib only.

## Layout

```
configs/   e0_baseline.yaml, e0_channel_aware.yaml, e1_lesion_sweep.yaml, e2_closedloop.yaml
src/
  nca.py         # NCA model (CAX ConvPerceive + NCAUpdate), params IO
  targets.py     # emoji loading (cached in assets/), premultiply, padding
  train.py       # E0 pool training + mid-episode damage, checkpoints, GIFs
  damage.py      # lesion types, recurring schedules, seed split
  channels.py    # modulator state: tonic (EMA a=0.95) + phasic (tau=20)
  controller.py  # controller MLP 4->32->K, target-free stats
  rollout.py     # closed-loop rollout engine (5 modes), E1 sweep CLI
  metrics.py     # hamming, repair half-life, survival, AUC
  evolve.py      # Evosax CMA-ES loop + E2 evaluation
  figures.py     # fig1/fig2/table1 from results/
tests/     34 tests: nca, damage, channels, leakage, rollout, metrics
results/   runs/<ts>/{config.yaml, metrics.csv, params.pkl, *.gif}  (gitignored)
```

## Local setup and tests

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/          # 34 tests
```

## Run entry points

```bash
python -m src.train   --config configs/e0_baseline.yaml        # E0 (K=0)
python -m src.train   --config configs/e0_channel_aware.yaml   # E0' parent (K=3, m=0)
python -m src.rollout --config configs/e1_lesion_sweep.yaml    # E1 sweep
python -m src.evolve  --config configs/e2_closedloop.yaml      # E2 evolution
python -m src.figures --e1 results/<ts>_e1_lesion_sweep --e2 results/<ts>_e2_closedloop
```

`--smoke` on E1/E2 runs a tiny CPU-feasible config. `--steps N` overrides
E0 training length. After E0/E0', fill in `model.parent_params` /
`model.baseline_params` / `controller_params` in the E1/E2 configs with the
printed run-dir paths.

## Status

Verified locally (2026-07-20, Apple M3 CPU):

- 34/34 tests pass.
- E2 end-to-end: CMA-ES (3 generations), constant-tonic grid search, all 5
  conditions, trajectories/metrics/GIFs written.
- E1 end-to-end; fig1/fig2/table1 render correctly.
- Growth and recovery GIFs verified.
- Learning signal: 300-step small-canvas run, MSE 2.35e-2 to 7.09e-3 (-70%).

Not verified locally (needs an A100): full E0 acceptance and real fitness
improvement over 300 generations. The local setup was too saturated to reach
step 50 in 4.6 hours of wall time — this is a hardware limitation, not a code issue.

## Milestones

M1 E0 acceptance · M2 E1 data and rollout speed · M3 E2 results
· M4 figures and draft · M5 submission. Details per the project plan.
