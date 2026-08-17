"""Per-parent effect decomposition + m_t diagnostics (preregistered analysis).

Committed BEFORE results existed. Computes, per parent seed s:
  E_train,s     = H(K=0) - H(K=3, m=0)          channel-aware training effect
  E_ctrl,s      = H(K=3, m=0) - H(K=3, own)     preregistered primary Delta_s
  E_transfer,s  = H(K=3, July) - H(K=3, own)    cross-parent transfer probe penalty

Positive = the named component helps. Reads the per_parent_s{s} CSVs produced
by the study; applies the preregistered outcome rule; produces m_t wording
diagnostics per controller.

Usage: python paper_drafts/analyze_per_parent.py [results_root]
  results_root defaults to experiment_results/20260817_per_parent
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "experiment_results/20260817_per_parent")
SEEDS = [0, 1, 2, 3, 4]
LESION_TIMES = list(range(150, 2000, 150))


def read_conditions(csv_path: Path) -> dict:
    """zero_control.csv -> {condition: (final_hamming_mean, sd, auc_mean, sd)}"""
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            out[r["condition"]] = (
                float(r["final_hamming"]), float(r["final_hamming_sd"]),
                float(r["auc"]), float(r["auc_sd"]),
            )
    return out


def m_diagnostics(csv_path: Path) -> dict:
    """m_series.csv -> std, post-lesion correlation, lesion-step change."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append([float(r[f"m_k{k}"]) for k in range(3)])
    M = np.array(rows)  # (T, K)
    T = M.shape[0]
    # post-lesion indicator: 1 within 50 steps after a lesion
    post = np.zeros(T)
    for tl in LESION_TIMES:
        post[tl:min(tl + 50, T)] = 1.0
    # correlation: average over channels of corr(m_k, post)
    cors = []
    for k in range(M.shape[1]):
        if M[:, k].std() > 1e-6 and post.std() > 1e-6:
            cors.append(np.corrcoef(M[:, k], post)[0, 1])
    corr = float(np.mean(cors)) if cors else 0.0
    # mean |delta m| in the 10 steps after each lesion vs mean |delta m| elsewhere
    jumps, base = [], []
    m_abs = np.abs(np.diff(M, axis=0))  # (T-1, K)
    for t in range(T - 1):
        if any(tl <= t + 1 < tl + 10 for tl in LESION_TIMES):
            jumps.append(m_abs[t].mean())
        else:
            base.append(m_abs[t].mean())
    return {
        "std": float(M.std(axis=0).mean()),
        "corr_post_lesion": corr,
        "lesion_jump": float(np.mean(jumps)) if jumps else 0.0,
        "baseline_drift": float(np.mean(base)) if base else 0.0,
    }


def main() -> None:
    print("=" * 92)
    print("PER-PARENT EFFECT DECOMPOSITION (preregistered analysis — script committed before results)")
    print("=" * 92)
    per_seed = {}
    for s in SEEDS:
        own_p = ROOT / f"s{s}" / "own" / "zero_control.csv"
        xfer_p = ROOT / f"s{s}" / "transfer" / "zero_control.csv"
        if not own_p.exists():
            print(f"seed {s}: own eval missing — skipped")
            continue
        own = read_conditions(own_p)
        xfer = read_conditions(xfer_p) if xfer_p.exists() else {}
        try:
            h_k0 = own["no_modulation"][0]
            h_m0 = own["zero_output"][0]
            h_own = own["closed_loop"][0]
        except KeyError as e:
            print(f"seed {s}: incomplete own CSV ({e}) — skipped")
            continue
        h_july = xfer.get("closed_loop", (float("nan"),) * 4)[0]
        e_train = h_k0 - h_m0
        e_ctrl = h_m0 - h_own
        e_transfer = h_july - h_own if not np.isnan(h_july) else float("nan")
        per_seed[s] = dict(h_k0=h_k0, h_m0=h_m0, h_own=h_own, h_july=h_july,
                           e_train=e_train, e_ctrl=e_ctrl, e_transfer=e_transfer)
        print(f"\nseed {s}:")
        print(f"  H: K0={h_k0:.4f}  m=0={h_m0:.4f}  own={h_own:.4f}  july={h_july:.4f}")
        print(f"  E_train={e_train:+.4f} ({'channels help' if e_train>0 else 'channels hurt'})   "
              f"E_ctrl(Delta_s)={e_ctrl:+.4f} ({'controller helps' if e_ctrl>0 else 'controller does not help'})   "
              f"E_transfer={e_transfer:+.4f}" + (" (transfer penalty)" if e_transfer > 0 else ""))

    if not per_seed:
        print("no data yet"); return
    n = len(per_seed)
    e_ctrl_pos = sum(1 for v in per_seed.values() if v["e_ctrl"] > 0)
    e_train_pos = sum(1 for v in per_seed.values() if v["e_train"] > 0)
    e_xfer_pos = sum(1 for v in per_seed.values()
                     if not np.isnan(v["e_transfer"]) and v["e_transfer"] > 0)
    print("\n" + "-" * 92)
    print(f"sign consistency: E_ctrl>0 in {e_ctrl_pos}/{n}   E_train>0 in {e_train_pos}/{n}   "
          f"E_transfer>0 in {e_xfer_pos}/{n}")
    for name, key in [("E_train", "e_train"), ("E_ctrl", "e_ctrl"), ("E_transfer", "e_transfer")]:
        vals = [v[key] for v in per_seed.values() if not np.isnan(v[key])]
        print(f"  {name}: median {np.median(vals):+.4f}  mean {np.mean(vals):+.4f}  "
              f"range [{min(vals):+.4f}, {max(vals):+.4f}]")

    print("\nPREREGISTERED OUTCOME:")
    thresh = 4 if n >= 4 else n  # >=4/5 (or all if fewer seeds completed)
    if e_ctrl_pos >= thresh:
        print("  OUTCOME A: controller effect supported — own-controller beats zero-output "
              f"in {e_ctrl_pos}/{n} seeds")
    elif e_ctrl_pos <= n - thresh if thresh else True:
        pass
    if e_ctrl_pos < thresh:
        if e_ctrl_pos in (0, 1) or (n >= 4 and e_ctrl_pos <= 1):
            print("  OUTCOME B: no consistent own-controller benefit "
                  f"({e_ctrl_pos}/{n}) — channel-training effect is the story")
        else:
            print("  OUTCOME C: mixed parent effects "
                  f"({e_ctrl_pos}/{n}) — controller efficacy is parent-dependent")

    print("\nM_t WORDING DIAGNOSTICS (per own-controller):")
    for s in SEEDS:
        mp = ROOT / f"s{s}" / "m_series.csv"
        if not mp.exists():
            continue
        d = m_diagnostics(mp)
        low_var = d["std"] < 0.05
        weak_event = abs(d["corr_post_lesion"]) < 0.3 and d["lesion_jump"] < 3 * max(d["baseline_drift"], 1e-9)
        if low_var and weak_event:
            wording = "parent-specific TONIC CALIBRATION"
        elif not weak_event:
            wording = ("adaptive closed-loop RELEASE POLICY (if E_ctrl>0) / "
                       "state-dependent policy, scheduling unnecessary (if E_ctrl<=0)")
        else:
            wording = "intermediate"
        print(f"  seed {s}: std(m)={d['std']:.4f}  corr(m,post-lesion)={d['corr_post_lesion']:+.2f}  "
              f"|Δm|@lesions={d['lesion_jump']:.4f} vs drift={d['baseline_drift']:.4f}  -> {wording}")


if __name__ == "__main__":
    main()
