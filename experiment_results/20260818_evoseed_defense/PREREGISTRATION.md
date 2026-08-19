# Preregistration: evolution-seed defense study (2026-08-18)

Committed BEFORE launch. This is a robustness extension of the completed
per-parent study (20260817_per_parent); the existing five-controller
result remains the primary preregistered finding and is NOT replaced,
re-analyzed, or re-classified by this extension.

## Motivation

The main study evolved one controller per parent. The strongest remaining
reviewer attack: "You evolved only one controller per parent — maybe
another CMA-ES seed would find a useful policy." The five flat m_t series
weaken that objection; this study closes it.

## Design

Two ADDITIONAL independent CMA-ES evolutions per parent seed s in
{0,1,2,3,4}, evolution RNG seeds e in {1,2} (the main study used
evolution seed = parent seed = s). Everything else identical to the main
study: same parents (the exact trained weights, no retraining), same
train/test damage seeds, same event-weighted fitness, population 64,
sigma_0 = 0.01, 300 generations, same evaluation protocol and metrics.

Per run, record: final Hamming (own-controller vs zero-output), AUC,
survival, and the m_t series (mean + std per channel).

Primary statistic per (s, e):
    D_{s,e} = H(zero_output, s) - H(own_{s,e}, s)
plus the m_t flatness/level diagnostic.

## Pre-committed interpretation (fixed now)

- CONFIRMS the main finding if most reruns (>= 8 of 10) also show
  noise-level |D_{s,e}| (order of magnitude below per-run SD) AND flat
  non-lesion-locked m_t. Then the negative controller result stands
  against the search-stochasticity objection, and the paper may adopt the
  stronger title: "Channel-Aware Training, Not Closed-Loop Control, Drives
  Robust Regeneration in Neural Cellular Automata."
- SOFTENS if any rerun produces a materially positive effect
  (noise-exceeding, sign-consistent across its condition seeds): report
  the full distribution of D across parent and evolution seeds and soften
  the claim to "controller efficacy varies across parent and evolution
  seeds." The paper then keeps the safer title.

## Search-integrity rule (fixed now)

Reruns are reported individually, ALL of them. No rerun is selected,
cherry-picked, or presented as representative. If a rerun finds a large
positive effect, it is reported as prominently as the negatives.

## Budget and bound

10 evolutions. Target host selected by performance (dlperf), not price —
the previous host ran 2.2x slower than budgeted. Expected ~2.5h per
evolution on a fast host = ~25h; hard-stop watchdog at 30h. Estimated
cost $15-20.
