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

## 2026-08-16 — Late-session sweep: undocumented findings, framings, and one false conviction

Collected at the end of the writing phase. Things noticed during the runs that
never made it into an entry at the time.

### 1. The step-5532 spike is a fixed feature of the seed, not an event

The LR 2e-3 run died at step 5532. Checking the LR 1e-3 run's per-step CSV:
**the same perturbation appears at steps 5531–5532 there too** — it's absorbed
in a single step instead of cascading. Same seed → same pool/damage sampling →
the unlucky batch draw is *deterministic*. So the divergence was never "a bad
thing happened"; the bad thing happens in every run at the same step. The
divergence was the learning rate amplifying a survivable perturbation. The
enemy isn't noise — it's noise amplified. (Framing not in the original entry.)

### 2. The Hamming metric is asymmetric ~20:1 against overgrowth — and that's target-dependent

Beyond the ~0.046 saturation cap: on a sparse target, **death can cost at most
4.6% error, overgrowth up to 95%**. The metric punishes overgrowth roughly 20×
harder than death. This is why random modulation scores 0.82 and why the
death-hack never materialized — the guard was never needed *for this target*.

⚠ **Warning for anyone porting the benchmark:** on a dense target (e.g. a
pattern filling 60% of canvas), the asymmetry collapses toward symmetry and
the death-hack becomes genuinely viable. The `alive_floor` question we
resolved empirically (0.0 for a 4.6%-alive target) is only resolved *for this
target*. Denser morphology ⇒ re-probe before evolving.

### 3. The static condition is quietly doing set-point discovery

We report static ≈ closed-loop as "temporal control buys nothing." But look at
what static *computes*: the evolved controller reads the freshly grown,
undamaged grid at t=0 and outputs the tonic level that is then frozen. The
system **derives its operating point from its own healthy self-model**. That's
homeostatic set-point discovery — read "what healthy looks like," set the gain
accordingly. We frame it as a null result; a biology audience might read it as
the most interesting result in the paper. (Consider for the extended version.)

### 4. Fragment-level selection in the unmodulated run

Observed in the no_modulation GIF (user's eyeball find): the second duplicated
lizard eventually dies and leaves debris. Framing: under no modulation the
system resolves "what am I supposed to be?" by **letting one candidate die —
consensus by execution, the cheapest possible quorum mechanism**, selection
operating at fragment level rather than cell level. With modulation, both
fragments survive longer. Too cute for the paper; exactly right for notes.

### 5. Vast.ai CLI: stale-session trap and the `--api-key` bypass (OPERATIONAL)

The CLI silently uses a stale session file — *every* command fails with
"Session expired," including `tfa send-email` and even after
`vastai set api-key` with a fresh key. Chicken-and-egg: 2FA needs a session,
the session needs 2FA. **Fix: pass `--api-key <key>` explicitly on the
command** — that path bypasses the stale session cache and lets
`tfa send-email` fire. Cost us ~20 min and nearly blocked an overnight run.
(Mirrored in VAST.md troubleshooting.)

### 6. Two independent agents, identical diagnosis

The sigma probe (σ=0.3 catastrophic, σ=0.01 brackets neutral) was derived
independently twice — once by the session agent, once by a second agent after
a context loss — converging on the same numbers and fix from the same data.
The probe is robust to *who* runs it. Cheap, decisive, and reproducible-by-
construction: the profile of a diagnostic worth keeping.

### 7. The cost center was deliberation, not compute

Sigma probe: ~30 s of GPU. Damage calibration: ~85 s. The single largest line
item in the whole project was the instance **idling at $0.61/hr while we
discussed options**. Compute was never the bottleneck; deliberation was.
Budget accordingly: it's cheaper to run three probes than to argue about one.

### 8. The murder that wasn't (full story of the false death-hack conviction)

The afternoon of the sensitivity probe, the constant-tonic grid showed:
tonic 0.0 → Hamming 0.0026, alive **0.057**; tonic ±0.5 → 0.9. Pattern-match:
"great score + low alive" = classic RL fitness-hacking signature (agent
discovers an empty board scores well against a sparse target). The PRD even
ships a defense (`alive_floor`). Diagnosis announced with real confidence:
*neutral wins by killing the lizard; enable the floor.*

User pushed back: calibrate the floor from the healthy-state distribution,
not a runbook menu. Pulled E1 trajectories: the healthy lizard's alive
fraction, across every radius/seed/step, in **both** conditions, is 0.057.

**The corpse was the healthiest lizard we had.**

The arithmetic that flipped it: target mask is 4.6% alive → a correct lizard
occupies ~5% of canvas (exactly what the "dead" grid showed) → an actually-
dead grid scores 0.046, 18× *worse* than 0.0026. Death was never cheap; it was
expensive. Neutral wasn't cheating — it was the only contestant maintaining
target size.

**The stakes:** had the floor gone to the runbook's 0.3 that afternoon, every
healthy candidate would have been fined +2.43 fitness per generation for the
crime of being the right size, shoving CMA-ES toward overgrowth. Third dead
E2 run, poisoned by its own guardrail — a defense attacking the organism it
protects. We'd have spent an evening debugging a self-inflicted sabotage.

**Lesson (twice-earned, see also the sigma probe):** the prior "low alive =
dead" was imported from dense-target intuition, and the import was wrong.
Thirty seconds of arithmetic — *what fraction of this canvas should be
alive?* — is the entire cost of not shipping a self-inflicted failure. When
every sample looks broken, check whether the measuring stick — or the prior —
is what's broken.

---

## 2026-08-17 — Seed study verdict: channels-help-parent-training is robust; the evolved controller is parent-locked

**The 3-seed study completed autonomously overnight.** Both automation layers
worked: the remote self-stop watchdog fired at study completion
(`{"success": true}`), data survived the stop/restart cycle on /workspace,
and the local watcher's final pull raced the shutdown — recovery was a
2-minute restart-and-pull. Total study cost: ~$2.20 (instance stopped itself
~1.5h in, ahead of the 3.5h deadline).

**Results (final H, per parent seed):**

| seed | zero_output (K=3, m=0) | closed_loop (evolved) | no_modulation (K=0) |
|------|------------------------|----------------------|---------------------|
| 0    | 0.029                  | 0.036                | 0.034               |
| 1    | 0.034                  | **0.249 (surv 0.00)** | 0.177 (surv 0.03)  |
| 2    | 0.020                  | **0.434 (surv 0.00)** | 0.028               |

**Three findings:**

1. **The robust effect is training-with-channels (3/3 seeds).** The K=3
   parent with modulation pinned to neutral beats the K=0 parent in every
   seed. Whatever the channels do, they do it during parent training — the
   run-time modulation is not needed. "Train with channels, modulate nothing"
   is the most robust configuration anywhere in this project (survival 1.00
   in all seeds).

2. **The evolved controller is PARENT-LOCKED.** Transferred to K=3 parents
   trained identically but from different seeds, the one evolved controller
   is lethally miscalibrated (survival 0.00 on 2/3). It found an operating
   point calibrated to its own parent's channel weights — not a policy. This
   is the most interesting negative result of the project: evolution
   *appears* to learn modulation, but what it actually learns is a
   parent-specific constant.

3. **The paper's 2.2x gap was partly parent-luck.** The K=0 baseline itself
   swings 0.028–0.177 across parent seeds. The gap is real on average
   (fragile K=0 parents drive it) but the magnitude is parent-seed-dependent
   and the original single-parent table overstated its reliability.

**Paper implications (pending user decision):** add the zero_output row +
seed study to Table 1; reframe the contribution from "broadcast modulation
repairs damage" to (a) channel-aware training yields robust parents, (b)
evolved policies are parent-locked and transfer lethally, (c) per-parent
comparisons only. This *strengthens* the stress-test framing — the benchmark
caught a generalization failure the single-parent eval hid entirely.

**Ops notes for the record:** remote self-stop (instance API key, validated
with a no-op PUT first) is now the standard pattern for unattended runs —
Mac-independent, session-independent, bounded worst-case billing. The
stop-preserved-disk claim is empirically confirmed twice now (Jul 25→30 and
today's stop/restart cycle). Artifacts: results_local/seed_study_20260816/ +
experiment_results/20260816_seed_study/ (git).

---
