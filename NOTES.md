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
