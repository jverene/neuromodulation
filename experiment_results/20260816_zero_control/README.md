# Zero-output control (2026-08-16)

Question: does the broadcast-modulation improvement come from *active
modulation* or from the channels being present during parent training?

Conditions on fresh parents, held-out damage seeds, hard multi-block regime:
- zero_output:   K=3 parent, modulation PINNED to m=0 (mode "ablated")
- closed_loop:   the evolved controller (from the 2026-07-30 E2 run) on K=3
- no_modulation: K=0 parent

## Results (shared-state0 protocol, matches paper)

| condition      | survival | half-life | final H      | AUC          |
|----------------|----------|-----------|--------------|--------------|
| zero_output    | 1.00     | 10.0±9.7  | 0.0291±0.0136| 0.0165±0.0086|
| closed_loop    | 1.00     | 8.4±7.6   | 0.0361±0.0097| 0.0203±0.0067|
| no_modulation  | 1.00     | 8.1±9.8   | 0.0343±0.0117| 0.0219±0.0079|

## Findings

1. zero_output lands WITH the modulated band (and nominally best): the
   channel-aware parent's advantage does not require active release.
2. The paper's headline 2.2x gap (0.028 vs 0.063) did NOT reproduce on fresh
   parents: today closed_loop 0.036 vs no_modulation 0.034 - no gap.
   Parent-training luck is a live hypothesis for the original gap.
3. The evolved controller scores slightly WORSE than m=0 on a fresh parent
   (0.0361 vs 0.0291): the evolved tonic is calibrated to its own parent.

-> Triggered the 3-seed parent study (seeds 0/1/2) to determine whether the
   gap is parent-seed-specific or absent entirely. Results: ../20260816_seed_study/
