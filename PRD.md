Build PRD — Closed-Loop Neuromodulation in NCAs
v1.0 — 2026-07-18 · Covers E0–E2 of the Research Protocol (6-week sprint)
Companion doc: Closed-Loop Neuromodulation NCA — Research Protocol.md. This PRD is the implementation spec; the protocol is the science spec. If they conflict, the protocol wins on claims, this wins on code.

1. Objective
   Build a JAX pipeline that (a) trains a standard Growing NCA to regenerate a target pattern, (b) adds global modulator channels with a learned controller, (c) evolves the controller's release policy with CMA-ES against a long-horizon recurring-damage objective, and (d) produces Figure 1 (lesion-size × channel-ablation) and Figure 2 (closed-loop vs. baselines) for a 4-page NeurIPS-workshop short paper due 2026-08-29.
2. Stack & Environment
   Python 3.11+, JAX + jaxlib (CUDA build), CAX, Flax, Optax, Evosax (CMA-ES ask/tell, JIT-compatible), NumPy, Matplotlib, imageio (GIFs), PyYAML for configs.
   Logging: CSV + Matplotlib only. No wandb — setup drag is a non-goal at this scale.
   Hardware: 1 GPU (Colab Pro A100 or local). All rollouts vectorized with vmap; time-budgeted via lax.scan.
   Repo layout:
   plain
   nca-mod/
   configs/ # e0_baseline.yaml, e1_lesion_sweep.yaml, e2_closedloop.yaml
   src/
   nca.py # NCA model definition (perception + update)
   targets.py # emoji loading, padding, target tensor
   train.py # E0 growth training loop (pool + damage)
   damage.py # lesion generators
   channels.py # modulator state (tonic/phasic) + injection
   controller.py # controller MLP (grid stats -> modulator levels)
   evolve.py # Evosax CMA-ES loop over controller params
   rollout.py # vmapped rollouts, recurring-damage schedules
   metrics.py # hamming, half-life, survival, AUC
   figures.py # fig1/fig2/table1 generation from results/
   notebooks/ # scratch only; nothing required for pipeline
   results/ # runs/<timestamp>/{config.yaml,metrics.csv,params.pkl,\*.gif}
   tests/ # test_channels.py, test_damage.py, test_leakage.py
3. E0 — Baseline GNCA (Week 1)
   Target: 40×40 RGBA emoji (lizard 🦎), zero-padded to 96×96 canvas, alpha premultiplied per Mordvintsev et al. 2020.
   Model: 16 channels; perception = identity + Sobel-x + Sobel-y + Laplacian (48 dims); update MLP: 48→128 (ReLU) →16, zero-init final layer; stochastic update mask p=0.5; alive-cell masking via 3×3 max-pool on alpha>0.1.
   Training: pool size 1024, batch 8, sample-with-replacement, worst-in-batch replaced by seed; damage = random square erasure mid-episode; loss = MSE(grid[:,:,:4], target); Adam lr 2e-3, 8000 steps.
   Acceptance: grows target from seed; recovers from a centered 20×20 lesion; GIF saved to results/.
4. Damage module (damage.py)
   sample_lesion(rng, kind, radius, center) -> mask — kinds: disc, multi_disc(n), edge_disc. Radii r ∈ {2, 4, 8, 16} px, chosen relative to effective perception radius (~1–2 cells).
   Recurring-damage schedule: lesions at fixed intervals (every 250 steps, T=2000) with parameters drawn from a fixed evaluation seed set (train/test split of damage seeds — holdout for generalization check).
5. Modulation layer (channels.py)
   ModulatorState: K=3 scalars = tonic (EMA, α=0.95) + phasic (spike × exp-decay τ=20 steps).
   Injection: broadcast K scalars, concatenate to every cell's perception vector → perception dim 48+K=51.
   Channel-aware baseline: retrain E0 with channels present and controller frozen at neutral (m=0). This is the ablated condition's parent model — never ablate by removing channels from a model trained without them.
6. Controller (controller.py)
   Input (grid summary stats, no target access — enforced by test_leakage.py): alive-cell fraction; fraction killed in last 10 steps; spatial entropy of alpha; normalized Hamming-proxy (alpha-mask mismatch rate vs. pool reference, computed target-free).
   MLP 4→32 (tanh) →K, output scaled to [-1, 1]. ~200 params.
   Decision period τ=10 NCA steps.
7. Evolution (evolve.py)
   Evosax CMA-ES, pop 64, σ₀=0.3, ≤300 generations (hard eval budget cap).
   Fitness: −mean Hamming(binα, target) over T=2000 with recurring damage (train damage-seed set), fully vmapped across population.
   Baseline conditions (all same channel-aware parent): closed-loop evolved · static conditioning (goal embedding fixed at t=0, Stovold-style) · constant tonic (hand grid-searched) · random schedule (seeded uniform) · no-modulation GNCA.
8. Experiments & Harness
   Every run: results/<ts>/ with config.yaml, metrics.csv, params.pkl, rollout GIFs. 5 seeds minimum per condition.
   E1: lesion radius {2,4,8,16} × {single, multi} × {channel intact, ablated} → recovery vs. radius curves.
   E2: 5 conditions × T=2000 recurring damage → survival fraction, repair half-life, Hamming AUC.
9. Metrics & Figures (metrics.py, figures.py)
   hamming_to_target(t) on binarized alpha; repair_half_life (time to 50% recovery); survival@2000; AUC.
   Fig 1: recovery rate vs. lesion radius, intact vs. ablated, ±SEM bands. Prediction: ablated collapses only past perception radius.
   Fig 2: Hamming-vs-time, 5 conditions. Table 1: survival/half-life means ± SEM.
10. Milestones & Kill Criteria
    M1 (Jul 26): E0 acceptance passed.
    M2 (Aug 2): E1 data collected; rollout speed ≥ 50 pop-parallel.
    M3 (Aug 12): E2 results in hand. Checkpoint: if E1 curves aren't clean by Aug 12 → skip Aug 29, pivot to ALIFE 2027 runway (per protocol §7).
    M4 (Aug 22): figures + 4-page draft. M5 (Aug 29): submitted.
11. Explicit Non-Goals (sprint)
    Metamorphosis (E4) · transfer-entropy analysis (H1 formal test) · multi-target · isotropy variants · wandb/infra polish · multi-GPU. Park all of these for the full paper.
12. Gotchas
    Pool-training instability → lower lr, clip updates; watch alpha blowups in GIFs.
    Fitness hacking (controller games Hamming by suppressing alpha) → watch rollout videos every run; add alive-fraction floor to fitness if seen.
    Colab disconnects → checkpoint params every 500 steps.
    Damage overfit → always report the held-out damage-seed set.
