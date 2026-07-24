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
