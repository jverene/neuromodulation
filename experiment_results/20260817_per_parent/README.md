# Per-parent evolution study — COMPLETE (2026-08-18)

Five parent seeds x four conditions, preregistered (criteria and analysis
script committed before data: a00bb89, 511370b). Official output:
ANALYSIS_OFFICIAL.txt (produced by the committed script, unmodified).

## Results

| seed | K=0 | K=3,m=0 | K=3,own | K=3,July | E_train | Delta_s | E_transfer |
|------|-----|---------|---------|----------|---------|---------|------------|
| 0 | 0.0343 | 0.0291 | 0.0277 | 0.0361 | +0.0053 | +0.0013 | +0.0083 (mild) |
| 1 | 0.1790 | 0.0390 | 0.0306 | 0.2495 | +0.1400 | +0.0084 | +0.2189 (lethal, surv 0.00) |
| 2 | 0.0288 | 0.0210 | 0.0211 | 0.4340 | +0.0077 | -0.0001 | +0.4129 (lethal, surv 0.00) |
| 3 | 0.0467 | 0.0328 | 0.0339 | 0.1004 | +0.0139 | -0.0011 | +0.0665 (severe, surv 0.45) |
| 4 | 0.0439 | 0.0365 | 0.0323 | 0.0411 | +0.0074 | +0.0042 | +0.0088 (mild) |
| median | | | | | +0.0077 | +0.0013 | +0.0665 |

## Preregistered outcomes (mechanical rule)

- CHANNEL-TRAINING EFFECT: SUPPORTED (5/5 positive; bar was >=4/5).
- CONTROLLER EFFECT: OUTCOME C — mixed (3/5 positive; bar was >=4/5).
  Median Delta_s +0.0013, all magnitudes far below per-run SDs (~0.02):
  no noise-exceeding controller benefit in any seed.
- TRANSFER FAILURE: SUPPORTED (5/5 positive; bar was >=4/5). Severity
  parent-dependent: mild/mild on seeds 0/4, severe on 3, lethal on 1/2.
- m_t WORDING: 5/5 TONIC CALIBRATION (std(m_t) ~0.002-0.004, no
  event-locked response, lesion-step |dm| indistinguishable from drift).

## One-paragraph verdict

Training with global channels present is the robust cause of regeneration
robustness (5/5 seeds, up to 4.6x on a fragile K=0 parent). Evolving a
controller on top adds nothing that exceeds damage-seed noise (Delta_s
median +0.001), and the evolved artifact is a parent-specific tonic
constant (flat m_t) that transfers as a penalty everywhere (5/5) and
lethally to two of five siblings. Single-parent evaluation hid all three
facts.

## Ops

Host was 2.2x slower than July's (66s/gen vs 30s); one ~6-min network
outage mid-study (survived; container never restarted); final self-stop
fired cleanly at 10:29 UTC; queued-restart recovery pulled the last two
files and re-stopped within seconds. Total study cost ~$20 (33h).
