# P1/P1b preregistration — channel-exercise retrains and init-luck null

Registered 2026-10-12, before any run. Motivated by the camera-ready
review of `mod_noise: false` (channels silent during all parent training;
App. D) and the post-hoc controls of 20261011 (5/5 finding dissolved to
s1-only under own-grown states + translation-aligned scoring).

## Conditions

Training recipe = the study parents' recipe (App. D): 8000 steps, pool
1024, batch 8, Adam **1e-3** (NOT the 2e-3 currently in
configs/e0_baseline.yaml, which diverges), emoji target, same model dims.
Everything else per configs/e0_baseline.yaml / e0_channel_aware.yaml.

- **P1 — channel-exercise pairs** (5 fresh pairs, seeds s = 0..4):
  per seed, train BOTH a fresh K=0 parent AND a fresh K=3 parent with
  `mod_noise: true` (random modulator inputs each pre/post-damage phase,
  the flag's intended use). Fresh-on-fresh pairing avoids comparing a
  new draw against an old one.
- **P1b — init-luck null pairs** (4 fresh pairs, seeds t = 10..13):
  per seed, fresh K=0 and fresh K=3 with `mod_noise: false` (silent
  channels, i.e. the original condition, new draws).

Evaluation = the Appendix I battery unchanged (`src/posthoc_metrics.py`):
held-out damage seeds 10000–10007, 5 condition seeds, T=2000, plain and
translation-aligned (±10 px) final Hamming, shared (K=3-grown) and
own-grown initial states, survival at 0.1. Key streams: config seed
200+s (P1), 300+t (P1b).

## Decision rules (fixed before runs)

**P1 primary endpoint:** per-seed gap g_s = H_aligned_own(K=0) −
H_aligned_own(K=3-noise, m pinned 0). *Channel-experience effect
supported* if g_s > 0 in ≥4/5 seeds AND median g_s > 0.005 (mirrors the
original ≥4/5 bar plus a triviality guard). Report the original-protocol
version (plain, shared-state) alongside for comparability with the
original 5/5.

**P1 secondary:** survival_plain per condition; note any lethality.

**P1b endpoint:** the distribution of per-pair gaps (both protocols)
across the 4 fresh silent pairs, read against the original study's 5/5.
Interpretation rule: mixed signs at magnitudes comparable to the
original healthy-parent gaps ⇒ the original 5/5 is consistent with init
luck + protocol provenance; near-zero corrected-protocol gaps across
all fresh pairs ⇒ the original healthy-parent differences were
drift/provenance artifacts.

## Budget and stop rules

18 trainings × ~20–50 min + 18 evals (~minutes on GPU). Watchdog caps
billing at 16 h; instance self-stops on completion or timeout. Artifacts
(params per parent, per-pair CSVs) land in this directory via the local
watcher and are committed incrementally.
