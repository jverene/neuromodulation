# REGIME.md - Regime structure, not kingdom: does temporal structure break tonic degeneracy?

**Status:** preregisterable design, written 2026-10-14, before any data. Grew out of the Sena correspondence (logged in /Users/hjiang/Developer/coldemails/CORRESPONDENCE.md): his objection was that tonic constants are what an optimizer finds in a stationary, memoryless damage regime, while biological habituation is change produced by exposure itself. This experiment makes that distinction testable in the existing NCA stack.

**Question:** Is the tonic degeneracy of evolved controllers a property of the damage regime's stationarity, or of the controllers/organisms generally? Concretely: when the damage process carries temporal structure, do evolved controllers stop emitting constants and start emitting policies, and does control then beat a tuned constant dose?

## Design

**Manipulation: the damage process only.** Everything else inherits the defense-study pipeline (parents, channels, CMA-ES controllers, held-out seeds). Four regimes, implemented as damage-process modules:

- **R0, stationary memoryless:** the current regime (i.i.d. lesions every 150 steps, n=4 blocks). Baseline; should replicate EvoRobust.
- **R1, bursty / autocorrelated:** lesion rate driven by a two-state Markov process (calm vs storm) with dwell times much longer than the lesion interval. Memoryful, unpredictable in the short run, statistically stable in the long run.
- **R2, periodic:** lesion clusters on a fixed period. Predictable structure.
- **R3, reactive:** lesion probability increases with recent repair success. Exposure-dependent, i.e., the environment responds to the organism (the closest in-silico analogue of "changes produced by the exposure itself").

Structure-strength axis for the phase-diagram version: vary the R1 dwell time from near-i.i.d. to strongly persistent.

**Conditions per regime:** parent trained under that regime (K=3, channels present), then on held-out damage seeds: (a) modulation pinned to zero, (b) own evolved controller, (c) tuned constant, a grid-searched fixed dose calibrated under that same regime. Condition (c) is the addition the defense study lacked: "closed loop beats recipe" requires the recipe baseline, and it makes this experiment exactly isomorphic to the mycelium design (m=0 / G1 recipe / G2 closed loop, Blocks C and D).

**Metrics:**
1. Tonic index: within-rollout per-channel std of m_t, plus response-lockedness (correlation of controller output with current grid state beyond its own baseline).
2. Controller benefit: delta of own controller over m=0.
3. Policy benefit: delta of own controller over the tuned constant. This is the number the whole experiment turns on.
4. Transfer check: controllers evolved under R1 evaluated under R0.

**Scale:** 5 parent seeds x 2 controller evolutions x 4 regimes, reusing E0'/E2. Roughly defense-study scale times four, on the order of $100 of rented GPU.

## Preregistered predictions (fixed 2026-10-14, before any data)

- **P1 (high confidence):** Under R0, replication: flat tonics, controller benefit ~zero over m=0, and policy benefit ~zero or negative versus the tuned constant.
- **P2 (moderate confidence, the Sena hypothesis):** Under R1 through R3, tonic index rises with structure strength, and policy benefit over the tuned constant becomes positive in the strongest-structure regimes. The controller starts earning its loop exactly when the regime has something to track.
- **P3 (control):** Controllers evolved under R1 and evaluated under R0 fall back toward tonic behavior: the policy-ness is structure-seeking, not intrinsic.
- **P4 (phase diagram, optional):** Policy benefit as a monotone-increasing function of R1 dwell time, locating the boundary between the tonic regime and the tracking regime.

Either outcome is a finding: P2 confirmed gives the regime-structure claim computational grounding and a direct prediction for the fungal drift block; P2 refuted means tonic degeneracy survives even exploitable structure, which strengthens and generalizes the training-dominates story and makes the mycelium Block D result more interesting, not less.

## Positioning

Either Study 1 of the mycelium paper (in-silico regime experiment, then organismal replication) or a standalone short follow-up to the EvoRobust workshop paper. Decide after results. The report-back email to Sena should carry the result either way: his objection, tested, with numbers.

## Sequencing

After the EvoRobust camera-ready (deadline Oct 29). The pipeline work (damage-process modules, tuned-constant condition, tonic-index metric) can be written and unit-tested before the deadline without touching the camera-ready text, then launched as one GPU block immediately after submission.
