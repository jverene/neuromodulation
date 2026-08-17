# Preregistration: per-parent evolution study (2026-08-17)

Written and committed BEFORE any per-parent evolution results exist, per
reviewer instruction. Interpretation criteria fixed in advance.

## Central question

Which component produces regeneration robustness — channel-aware parent
training, evolved modulation, or their interaction — and does any of it
transfer across independently trained organisms?

## Design

Five parent seeds s in {0,1,2,3,4}. For each seed, independently:
- K=0 parent and K=3 channel-aware parent trained at seed s (seeds 0-2
  exist from 2026-08-16; seeds 3-4 trained fresh tonight).
- Evolve a controller ON parent s (CMA-ES, 300 gens, sigma_0=0.01,
  evolution seed = parent seed = s), same hard multi-block regime
  (block_side 16, n=4, interval 150, T=2000), same train/test damage-seed
  split as all prior work.

Conditions per seed (all on the SAME 8 held-out damage seeds x 5 condition
seeds, shared-state0 protocol):
1. no_modulation:  K=0 parent, seed s
2. zero_output:    K=3 parent seed s, modulation pinned m=0
3. own_controller: K=3 parent seed s, controller evolved on parent s
4. transferred:    K=3 parent seed s, the July-30 controller (evolved on the
                   July-30 parent) — already measured for s=0,1,2 on 2026-08-16

Primary comparison (per seed):
    Delta_s = H(zero_output, s) - H(own_controller, s)
Positive Delta_s = the evolved controller helps its own parent.

Primary metrics: final Hamming and AUC. Survival reported but coarse at
n=5 parent seeds. Controller-output (m_t) time series recorded per evolved
controller to distinguish "parent-specific tonic calibration" (near-zero
variance in m_t) from "release policy" (time-varying m_t).

## Predefined interpretation

- CONTROLLER EFFECT SUPPORTED if own_controller beats zero_output in
  >= 4/5 parent seeds (or 3/3 if only 3 seeds complete), with the per-seed
  difference exceeding ordinary damage-seed noise (non-overlapping SDs or a
  sign-consistent paired difference).
- CHANNEL-TRAINING EFFECT SUPPORTED if zero_output beats no_modulation in
  >= 4/5 seeds (3/5 = suggestive, report as mixed).
- TRANSFER FAILURE SUPPORTED if transferred < zero_output substantially in
  >= 4/5 seeds (already observed 2/3 on 2026-08-16).
- NO UNIVERSAL CLAIM: if effects vary by parent seed, report parent-
  dependence explicitly (median + range across seeds); do NOT average into
  a single multiplier.

## Paper outcomes (fixed in advance)

- Outcome A (controller effect): "Channel-aware training improves parent
  robustness, and evolution further discovers a useful parent-specific
  modulation policy. However, that policy is parent-locked and lethal under
  cross-parent transfer."
- Outcome B (no controller effect): "The major robustness gain comes from
  channel-aware parent training, not active closed-loop modulation.
  Evolution produces parent-specific calibrations that fail to transfer."
- Outcome C (mixed): "Controller efficacy is parent-dependent; evolution
  can exploit parent-specific dynamics, but neither the benefit nor the
  policy transfers reliably."

The 2.2x (paper Table 1) and 2.9x (seed-study mean) multipliers are RETIRED
from claims regardless of outcome; all comparisons become per-seed-resolved.

## Artifacts saved per run

Parent params, controller params, CMA-ES checkpoint + metrics, run config,
code git SHA, full damage trajectories, per-damage-seed final metrics,
controller-output m_t series. Everything tripled: instance /workspace,
local results_local/, and experiment_results/ on GitHub.

## Budget and bound

~16h GPU at $0.60/hr (~$9.60). Instance self-stops on completion marker or
an 18h hard deadline, whichever comes first (watchdog validated 2026-08-16).
