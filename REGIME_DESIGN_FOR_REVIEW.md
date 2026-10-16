# Regime-structure design: does temporal structure break tonic degeneracy?

Design note prepared for review (October 2026). This is a preregisterable design; no data has been collected. It grew out of a correspondence with Dr. Giovanni Sena, whose distinction between exposure-produced change (biological habituation) and optimizer artifacts in stationary regimes motivated the experiment.

**Background.** In "Channel-Aware Training, Not Closed-Loop Control, Drives Robust Regeneration in Neural Cellular Automata" (EvoRobust Workshop, NeurIPS 2026), every controller evolved for regenerative NCAs converged to a flat, organism-specific constant, and the benefit of control was ~zero across preregistered comparisons. The paper's own interpretation is that the damage process was stationary and memoryless, leaving no temporal structure for an adaptive policy to exploit. The question this design asks is whether that degeneracy is a property of the controllers or of the regime: when the damage process carries temporal structure, do evolved controllers stop emitting constants and start emitting policies, and does control then beat a tuned constant dose?

**Design.** Everything inherits the defense-study pipeline (five parent seeds, channel-aware training, CMA-ES controllers, held-out damage seeds). Only the damage process varies, in four regimes:

- R0, stationary memoryless: i.i.d. lesions at fixed intervals (the published regime). Baseline; should replicate.
- R1, bursty: lesion rate driven by a two-state Markov process (calm/storm) with dwell times far exceeding the lesion interval. Memoryful, statistically stable.
- R2, periodic: lesion clusters on a fixed period. Predictable.
- R3, reactive: lesion probability increases with recent repair success. The environment responds to the organism, the in-silico analogue of exposure-produced change.

Per regime, three conditions on held-out seeds: modulation pinned to zero; the organism's own evolved controller; and a tuned constant (a grid-searched fixed dose calibrated under the same regime). The tuned-constant condition is the addition the published study lacked, and it makes this design directly comparable to a planned organismal experiment on electrically stimulated mycelium (fixed recipe versus closed loop).

**Metrics.** (1) Tonic index: within-rollout variability of controller output, plus its response-lockedness to current state. (2) Controller benefit over zero modulation. (3) Policy benefit over the tuned constant. (4) Transfer: controllers evolved under R1 and evaluated under R0.

**Registered predictions.** P1: under R0, replication (flat tonics; no benefit over the tuned constant). P2 (the hypothesis under test): under R1 through R3, the tonic index rises with structure strength and policy benefit over the tuned constant becomes positive in the strongest-structure regimes. P3: controllers evolved under R1 fall back toward tonic behavior under R0, i.e., any policy-ness is structure-seeking rather than intrinsic. P4 (optional): policy benefit as a function of R1 dwell time locates the boundary between the tonic regime and the tracking regime.

Either outcome is informative: confirmation grounds a prediction for the organismal drift-block experiment; refutation means tonic degeneracy survives exploitable structure, which strengthens and generalizes the training-dominates claim.

**Scale.** Five parents, two evolutions, four regimes, reusing existing code; roughly four times the published study's compute.
