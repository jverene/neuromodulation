# Research Notes (live experiment log)

Running log of findings, decisions, and interpretive observations from the
GPU runs. Append-only; dated. Distinct from README (status/verification) and
the runbooks (how-to). This is the *why* and *what it means*.

---

## 2026-07-24 — E0 baseline (LR 1e-3): M1 passed, with the paper's motivation on screen

**Run:** `results/20260724_065141_e0_baseline` (A100 SXM4, 8000 steps, 27 min)
**Final loss:** 8.59e-5 (start 1.71e-2, ~200× reduction). Post-lesion MSE 6.85e-3.

### M1 acceptance: passed
Main body regenerates cleanly in all four recovery samples after the centered
20×20 lesion. The growth GIF shows the lizard forming from a seed.

### The important observation — recovery is imperfect *on purpose*, and that's the story
Two visible artifacts in `recovery.gif`, both normal GNCA repair signatures:

1. **Detached green fragments (severed pieces that survived).** When the lesion
   cuts the pattern, chunks larger than a couple of cells stay "alive" —
   alive-masking only kills isolated single cells, not clusters. These fragments
   float with no global context: they can't reattach, can't die, and can't decide
   what to grow into. A piece of tail doesn't know it's not supposed to grow a
   second head.

2. **Grayish patches = half-alpha cells (dying/undecided states)** = scar tissue
   at the wound site.

**This strip is the paper's motivation in one image.** The failure mode a global
chemical/modulator channel is supposed to address is *precisely* "severed
fragments can't tell what they are because cells have no global information."
This is candidate panel (a) of the Figure 1 intro. **Save these frames.**

### What NOT to do
Do not retrain chasing prettier recovery. Baseline imperfection is the
phenomenon under study, not a bug. If the unmodulated lizard healed perfectly,
the closed-loop story would have no problem to solve. Document it, quantify it
(E1's Hamming curves do this), and let the modulated version try to beat it.

### Three follow-ups noted (not blocking)
1. **Run recovery rollout 2–3× longer** (5 min) to see which debris self-cleans
   vs. is permanent. Matters for E2: permanent debris inflates Hamming distance
   forever — equally for all conditions (comparisons stay fair), but it sets the
   noise floor of our own metric.
2. **Damage-shape mismatch.** Training used random square erasure; eval uses disc
   lesions. Recovery is wonkiest on shapes the model never trained on. Decide for
   E1/E2: train-eval consistency, OR declare it deliberate (held-out damage shape
   = generalization test, which actually supports the H2 story if framed that way).
3. **Save the recovery frames** for the Figure 1 motivation panel.

### Training incident (forensic, kept for reference)
First attempt at LR 2.0e-3 diverged to NaN at step 5532 — textbook late-stage
NCA instability: single ×400 loss spike → partial recovery → exponential cascade
→ inf → nan in ~30 steps. Grad clip (global-norm 1.0) held until it didn't.
Runbook fix (LR → 1e-3) cleared the exact same step cleanly: the same kind of
spike at step 5531 was absorbed in a single step instead of cascading. Failed
run dir `results/20260724_061149_e0_baseline` kept for forensics. Lesson: the
per-step CSV visualization paid for itself immediately by making the cascade
forensically legible.

---

## 2026-07-25 — Sensitivity probe (pre-E2): signal confirmed, alive_floor=0.3 REJECTED by data

**Probe:** `src/e2_probe.py` — constant-tonic grid `[-1,-0.5,0,+0.5,+1]` over the 8
train damage seeds at T=2000, reusing `evolve.py`'s exact eval path. ~7s on A100.

**Results:**
| tonic c | mean Hamming | alive |
|---------|-------------|-------|
| -1.0 | 0.650 | 0.986 |
| -0.5 | 0.886 | 0.984 |
| 0.0 | **0.0026** | 0.057 |
| +0.5 | 0.909 | 0.986 |
| +1.0 | 0.928 | 0.986 |

**Signal:** unambiguous. Controller input swings fitness 0.0026 → 0.93. The PRD §5
concern (channel weights dead because parent trained at m=0) does NOT apply —
the channels carry strong, exploitable signal. Green light for E2.

**The trap I almost fell into (and the data corrected):** the c=0 row at
alive=0.057 + Hamming 0.0026 looked like a "death-hack" — a dead grid gaming a
sparse-target Hamming metric. The runbook suggests `alive_floor: 0.3` for
exactly this scenario. **Wrong here.** Checking the target sparsity settles it:
the lizard mask is only **4.6% alive**, so a correctly-grown lizard lives at
~5–6% alive. c=0's 0.057 *is* the healthy steady state, not death. E1's
trajectories.csv confirms it — intact AND ablated conditions both hold
alive ≈ 0.057 across every radius/seed/step (median 0.057, p10 0.057).

The Hamming math confirms the metric is NOT death-hackable:
- perfect lizard → Hamming 0.0000
- all-dead grid → Hamming 0.0461 (target cells missing) — **18× worse than healthy**
- overgrown grid → Hamming 0.9539 — this is what large |c| does (alive → 99%)

So death is already a losing strategy; large tonic levels lose via **overgrowth**,
not via cheap death.

**Decision: `alive_floor` stays 0.0.** The runbook's 0.3 would penalize every
healthy candidate (penalty `10×(0.3−0.057)=+2.43` on the correct state) and push
CMA-ES toward overgrowth — actively backwards. The escape hatch is there for a
real hacking scenario; the data shows this isn't one. E2 launches with the
canonical config unchanged. **Lesson: calibrate hyperparameters from the
experiment's own data, not from runbook defaults — the default existed for a
failure mode that doesn't occur with this target sparsity.**

**Paper relevance:** worth a sentence in Methods/Discussion — the fitness
function rewards *size-maintenance under recurring damage*, not mere survival,
because the Hamming reference is the target alpha mask. A controller that lets
the lizard overgrow or die both lose; only accurate maintenance wins. Goodhart
was a concern (alive_floor exists precisely because of it) but did not manifest
here once the target sparsity was accounted for.

---

## 2026-07-25 — E2 stall diagnosis: task too easy, baseline solves it trivially

**E2 launched** (canonical config, alive_floor=0.0 verified by probe). Early gate
*passed* on paper: best 0.173→0.144 by gen 11, mean 0.71→0.35. But best-so-far
then **froze at 0.1435 for 23 generations** (gen 11→34) with best_in_gen actually
*rising* — classic CMA-ES stall. Killed at gen 34. Cost: ~$5, ~1h. Cheap tuition.

**Three diagnostics (`src/e2_diag.py`), all decisive:**

1. **Apples-to-apples (1b):** worried the probe's 0.0026 (mode=constant, m=0
   pinned) wasn't comparable to E2 fitness (mode=closed_loop, m flows through
   decision()+step_decay()). Measured both: constant 0.0024 vs closed_loop
   neutral 0.0023 — **negligible difference, comparison valid.** No bookkeeping
   error.

2. **Fitness decomposition (2):** alive_floor=0 → penalty 0.0000 for all
   controllers. The floor isn't taxing activity. Crossed off.

3. **The real finding:** the **evolved best controller is 62× WORSE than doing
   nothing.** neutral closed_loop = Hamming 0.0023, alive 0.057 (correct lizard).
   Evolved best (gen 29) = Hamming 0.1431, alive **0.901** — it's overgrowing,
   filling 90% of the canvas. CMA-ES isn't searching for repair; it's searching
   for overgrowth, and "do nothing" already beats every overgrown solution.

**Root cause (confirmed independently by E1):** the baseline solves the E2
recurring-damage schedule (lesion every 250 steps, T=2000) **trivially** — the
unmodulated lizard shrugs it off and stays at near-perfect Hamming. There is no
gap for modulation to close, so CMA-ES can only make things worse by perturbing
the controller away from the optimal zero point. **The task is too easy.**

E1 data corroborates: even the hardest E1 condition (radius-16 multi-disc,
single-shot) recovers to final Hamming ~0.011 — barely a failure. And
intact≈ablated at every radius (because with a neutral controller, closed_loop
== ablated; the channel weights only differ when the controller is non-zero,
which none exists yet). So the channel architecture has signal (probe showed
it) but no *useful* role under the current damage distribution.

**The fix is task design, not search tuning.** Rescope the damage schedule into
a regime where the baseline genuinely struggles (large/multi-site lesions,
tighter recurrence, possibly larger T) so a controller has a gap to close.
Candidate: recurring radius-16 multi-disc every 100–150 steps, T=2000 — push
damage past the radius where E1 showed recovery degradation (radius ≥ 8–16).

**Lesson (paper Methods/Discussion):** the benchmark needed to be adversarial,
not routine. If the baseline heals everything, there's nothing for modulation to
prove — E2 was pointing at a task with no gap to close. Caught for $5 and one
hour via the stall + three 30-second diagnostics. The sensitivity probe fork-
decision literally paid for itself a second time: it surfaced the "signal
exists" green light AND, via the stall, exposed that signal-existence ≠ signal-
usefulness when the task lacks a failure regime.

---

## 2026-07-25 — E2-hard gate v1 failed: std_init=0.3, not a razor-thin basin

**Setup:** hard regime calibrated (`configs/e2_hard.yaml`, branch
`e2-hard-regime`): multi_block lesions side=16 n=4 every 150 steps, T=2000 →
neutral raw H 0.020, alive 0.051 (10× harder than the old disc task). Fitness
switched to event-weighted (tau_w=interval/3=50; repair half-life 20.1 → ratio
0.40, kernel stays meaningful — amendment 2 passed). Death-hack re-probe on the
new fitness+damage: death 0.0461 vs struggling-neutral 0.0205, death loses 2.2×
→ alive_floor=0.0 holds (amendment 1 passed).

**Gate v1 (std_init=0.3, 20 gens): FAILED.** best_so_far froze at 0.150 from
gen 2 onward — 7× worse than neutral, identical stall signature to the first
E2. Mean dropped 0.70→0.30 (sigma shrinking) but no sample ever came near
neutral.

**Sigma probe (`/workspace/sigma_probe.py`, 25 Gaussian perturbations of the
neutral controller per sigma, hard-regime eval):**

| sigma | mean_H | min_H | mean_alive | verdict |
|-------|--------|-------|------------|---------|
| 0 (neutral) | 0.0201 | — | 0.051 | reference |
| 0.001 | 0.0184 | 0.0169 | 0.052 | **some BEAT neutral** |
| 0.01  | 0.0197 | 0.0182 | 0.053 | at/above neutral |
| 0.05  | 0.2029 | 0.0212 | 0.356 | bimodal |
| 0.1   | 0.4453 | 0.0215 | 0.703 | bimodal |
| 0.3   | 0.7318 | 0.1595 | 0.960 | ALL overgrow |

**Corrected diagnosis:** the neutral basin is NOT razor-thin — perturbations at
sigma ≤ 0.01 stay healthy and some (0.0169) are already better than doing
nothing. The failure is the *sampling distribution*: at std_init=0.3, 100% of
the population lands in saturated overgrowth (alive→0.96) where fitness
differences reflect overgrowth degree, not repair quality. CMA-ES ranks noise
in the wrong regime, and its mean drifts along the overgrowth gradient instead
of toward zero. The task redesign was necessary but not sufficient; the search
had to start inside the informative regime.

**Fix:** std_init 0.3 → 0.01. At sigma=0.01 samples bracket neutral (mean
0.0197 ≈ neutral 0.0201, min 0.0182 below it), so gen-0 ranking already sees
repair-quality signal. Gate v2 (20 gens, std_init=0.01) running.

**Lesson:** when every sample in gen 0 sits in a saturated regime, "the search
can't find the solution" and "the init distribution never samples the
informative regime" are indistinguishable from the metrics alone. A 30-second
sigma-sweep probe separates them. This is the second time a cheap probe
pre-empted a wrong structural conclusion (first: alive_floor).

---

## 2026-07-30 — E2 COMPLETE: modulation helps (2.4× faster repair), closed_loop ≈ static

**Full run:** 300 gens, std_init=0.01, hard regime (multi_block side=16 n=4 every
150 steps), event-weighted fitness. Evolution finished in ~2.5h; 5-condition eval
+ figures ran after. Gate re-verified on fresh hardware (parents retrained: E0
1196s MSE 3.33e-3, E0′ 1278s MSE 7.19e-3; both cleared the step-5532 danger zone
cleanly at LR 1e-3, reproducing the earlier M1 results).

**Evolution:** best fitness 0.0135 vs neutral baseline 0.0205 — **34% better**.
Converged smoothly, no stall, no overgrowth (the std_init fix held for all 300 gens).

**Table 1 — 5 conditions (mean ± SD over 5 condition seeds × 8 held-out test
damage seeds), hard-regime recurring damage:**

| condition | survival | repair half-life (steps) | final Hamming | Hamming AUC |
|-----------|----------|--------------------------|---------------|-------------|
| **closed_loop** (evolved controller) | 1.00 | **6.6 ± 0.7** | 0.028 ± 0.003 | 0.016 ± 0.002 |
| static (controller fixed at t=0) | 1.00 | 7.2 ± 0.9 | 0.030 ± 0.003 | 0.017 ± 0.002 |
| constant (best tonic level) | 1.00 | 6.4 ± 0.5 | 0.030 ± 0.003 | 0.017 ± 0.002 |
| **no_modulation** (K=0 baseline) | 1.00 | **2.7 ± 0.5** | 0.063 ± 0.003 | 0.064 ± 0.002 |
| random (noise modulation) | 0.00 | 0.0 ± 0.0 | 0.825 ± 0.016 | 0.819 ± 0.003 |

**Two clean wins for modulation** (the headline):
- **Repair speed: 6.6 vs 2.7 steps** → modulation makes the lizard recover
  **2.4× faster**. This is exactly the "speed is where closed-loop should win"
  thesis from the event-weighted redesign, validated.
- **Final recovery: 0.028 vs 0.063** → modulation recovers **2.2× more
  completely** (lower residual damage).

**Honest caveat — writeup-critical:** closed_loop ≈ static ≈ constant on every
metric (final H 0.028–0.030, half-life 6.4–7.2, all within ±1 SD). So the
defensible claim is "the chemical/modulator layer helps substantially over no
modulation," NOT "dynamic closed-loop decision-making beats fixed tonic
modulation." The dynamic controller's value isn't separated from simpler
modulation baselines in this task. Options to address before claiming a
closed-loop-specific win: (a) reframe the contribution as "modulation helps"
and demote the closed-loop-vs-static comparison; (b) find a task regime where
*temporal* decisions matter (e.g. damage that changes character over time, so a
fixed tonic is structurally inadequate); (c) harder/different damage where
static's t=0 snapshot is wrong. Don't overclaim — reviewers will see the
closed_loop/static gap is noise.

**Artifacts (results_local/e2_hard_20260730/):** controller_params.pkl (4→32→3
MLP, verified loadable), evolve_metrics.csv (300 gens), metrics.csv +
trajectories.csv (per-condition), fig2_hamming_vs_time.png,
table1_survival_halflife.{csv,md}, 5 rollout GIFs.

**Cost:** ~$3 today (setup + E2 + eval at $0.68/hr). Project total ~$13.

---
