"""Mechanical P1/P1b analysis per experiment_results/20261012_p1_p1b/PREREGISTRATION.md.

Reads the per-pair CSVs written by the in-VM driver (each row: label,
condition, n, plain/aligned hamming mean/sd, survival_plain/aligned) and
applies the fixed decision rules, writing ANALYSIS.md next to the data.
Run from the repo root: python scripts/analyze_p1.py
"""
import csv
import statistics as st
from pathlib import Path

OUT = Path("experiment_results/20261012_p1_p1b")


def conds_for(fname):
    out = {}
    for row in csv.DictReader(open(OUT / fname)):
        out[row["condition"]] = {
            k: float(v) for k, v in row.items() if k not in ("label", "condition", "n")
        }
        out[row["condition"]]["n"] = int(row["n"])
    return out


rows, gaps_p1, gaps_p1_orig, gaps_p1b, gaps_p1b_orig = [], [], [], [], []
for s in range(5):
    c = conds_for(f"p1_pair_s{s}.csv")
    if len(c) < 3:
        continue
    ga = c["k0_own"]["aligned_hamming_mean"] - c["k3_m0_shared"]["aligned_hamming_mean"]
    gp = c["k0_shared"]["plain_hamming_mean"] - c["k3_m0_shared"]["plain_hamming_mean"]
    gaps_p1.append(ga)
    gaps_p1_orig.append(gp)
    rows.append((f"p1_s{s}", c, ga, gp))
for t in range(10, 14):
    c = conds_for(f"p1b_pair_t{t}.csv")
    if len(c) < 3:
        continue
    ga = c["k0_own"]["aligned_hamming_mean"] - c["k3_m0_shared"]["aligned_hamming_mean"]
    gp = c["k0_shared"]["plain_hamming_mean"] - c["k3_m0_shared"]["plain_hamming_mean"]
    gaps_p1b.append(ga)
    gaps_p1b_orig.append(gp)
    rows.append((f"p1b_t{t}", c, ga, gp))

lines = [
    "# P1/P1b analysis (mechanical, preregistered rules)",
    "",
    "g_aligned_own = H_aligned(K=0, own-grown) − H_aligned(K=3 m=0, own-grown); positive =",
    "the K=3 parent of the pair is more robust under the corrected protocol.",
    "g_plain_shared = the same gap under the ORIGINAL protocol (plain, shared state).",
    "",
    "| pair | K=3 m=0 own (plain/aligned, surv) | K=0 shared (plain/aligned, surv) | K=0 own (plain/aligned, surv) | g_aligned_own | g_plain_shared |",
    "|---|---|---|---|---|---|",
]
for lab, c, ga, gp in rows:
    a, b, d = c["k3_m0_shared"], c["k0_shared"], c["k0_own"]
    lines.append(
        f"| {lab} | {a['plain_hamming_mean']:.3f}/{a['aligned_hamming_mean']:.3f} ({a['survival_plain']:.2f}) "
        f"| {b['plain_hamming_mean']:.3f}/{b['aligned_hamming_mean']:.3f} ({b['survival_plain']:.2f}) "
        f"| {d['plain_hamming_mean']:.3f}/{d['aligned_hamming_mean']:.3f} ({d['survival_plain']:.2f}) "
        f"| {ga:+.4f} | {gp:+.4f} |"
    )
lines += ["", "## Decision rules (fixed before runs)"]
if gaps_p1:
    pos = sum(g > 0 for g in gaps_p1)
    med = st.median(gaps_p1)
    verdict = "SUPPORTED" if (pos >= 4 and med > 0.005) else "NOT SUPPORTED"
    lines += [
        f"- **P1** (channel-exercise pairs, n={len(gaps_p1)}): g_aligned_own > 0 in "
        f"{pos}/{len(gaps_p1)}; median {med:+.4f} → channel-experience effect **{verdict}**.",
        f"  - g_aligned_own per seed: {[round(g, 4) for g in gaps_p1]}",
        f"  - original-protocol gaps (plain, shared): {[round(g, 4) for g in gaps_p1_orig]} "
        f"({sum(g > 0 for g in gaps_p1_orig)}/{len(gaps_p1_orig)} positive)",
    ]
if gaps_p1b:
    lines += [
        f"- **P1b** (init-luck null pairs, n={len(gaps_p1b)}): g_aligned_own per pair "
        f"{[round(g, 4) for g in gaps_p1b]} ({sum(g > 0 for g in gaps_p1b)}/{len(gaps_p1b)} positive)",
        f"  - original-protocol gaps (plain, shared): {[round(g, 4) for g in gaps_p1b_orig]} "
        f"({sum(g > 0 for g in gaps_p1b_orig)}/{len(gaps_p1b_orig)} positive)",
    ]

text = "\n".join(lines) + "\n"
(OUT / "ANALYSIS.md").write_text(text)
print(text)
