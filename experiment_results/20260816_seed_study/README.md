# 3-seed parent study (2026-08-16)

Question: does the paper's headline modulated-vs-unmodulated gap (final H
0.028 vs 0.063) survive parent retraining, or was it single-parent-seed luck?

Design: parents retrained at seeds {0,1,2} for both K=0 and K=3 (same configs,
LR 1e-3). Per seed: zero_output (K=3, m pinned 0), closed_loop (the ONE
evolved controller from the 2026-07-30 run), no_modulation (K=0). All on the
same 8 held-out damage seeds x 5 condition seeds, hard multi-block regime,
shared-state0 protocol matching the paper.

## Results (final Hamming, mean +/- SD)

| parent seed | zero_output (K=3, m=0) | closed_loop (evolved ctrl) | no_modulation (K=0) |
|-------------|------------------------|----------------------------|---------------------|
| 0           | 0.0291 +/- 0.014       | 0.0361 +/- 0.010           | 0.0343 +/- 0.012    |
| 1           | 0.0336 +/- 0.021       | 0.2492 +/- 0.038 (surv 0.00) | 0.1769 +/- 0.030 (surv 0.03) |
| 2           | 0.0197 +/- 0.016       | 0.4344 +/- 0.014 (surv 0.00) | 0.0281 +/- 0.018    |
| **mean**    | **0.0275**             | 0.2399                     | 0.0798              |

## Findings

1. **Training-with-channels is the robust effect (3/3 seeds).** zero_output
   (K=3 parent, modulation pinned to neutral) beats no_modulation (K=0) in
   every parent seed: 0.029<0.034, 0.034<0.177, 0.020<0.028. Mean ~2.9x.
   The channel-aware PARENT is consistently more damage-robust than the K=0
   parent, with zero modulation at run time.

2. **The evolved controller is PARENT-LOCKED.** The one evolved controller
   (trained against the July seed-0 K=3 parent) transfers lethally to
   other K=3 parents: survival 0.00 on seeds 1 and 2 (final H 0.25 / 0.43).
   Its tonic is calibrated to its own parent's channel weights; on
   differently-trained parents the same output is catastrophically
   miscalibrated. Evolution found an operating point, not a policy.

3. **The paper's 2.2x gap was partly parent-luck.** The original table's
   0.028-vs-0.063 compares the July parents; across 3 fresh parent seeds the
   K=0 baseline itself swings 0.028-0.177. The gap is real on average
   (driven by fragile K=0 parents) but its magnitude is parent-seed-dependent.

4. **zero_output is the most robust configuration anywhere in the project**
   (survival 1.00 in all seeds; best mean final H). The most reliable way to
   run this system is: train with channels, modulate nothing.

## Implications for the paper

- Table 1 needs the zero_output row + this study.
- The claim shifts from "broadcast modulation repairs damage" to:
  (a) channel-aware training yields robust parents (robust across seeds),
  (b) evolved release policies are parent-locked and transfer lethally,
  (c) closed-loop vs tonic comparisons are only meaningful per-parent.
- This STRENGTHENS the stress-test framing: the benchmark caught a
  generalization failure (controller parent-locking) that the single-parent
  evaluation hid entirely.

## Ops

Self-stop watchdog fired cleanly at study completion ({"success": true});
data survived the stop/restart cycle on /workspace; results tripled to
Mac + GitHub. Instance 47872539 left in storage mode.
