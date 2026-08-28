"""Generate paper figures (fig1-fig4) into paper_drafts/figures/ from repo data.

Journal-grade plotting: colorblind-safe palette (Wong 2011), 300 DPI for print,
clean insets with borders, readable fonts, no legend/axis overlap.
"""
import csv
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper_drafts/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Journal-quality global settings.
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Wong (2011, Nature Methods) colorblind-safe palette.
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_RED    = "#D55E00"
C_PURPLE = "#CC79A7"
C_GREY   = "#555555"

# ---------------------------------------------------------------- fig1: E1 sweep
traj = defaultdict(list)  # (radius, kind, condition) -> [final hamming per seed]
with open("results_local/e1_lesion_sweep/trajectories.csv") as f:
    rows = list(csv.DictReader(f))
finals = defaultdict(dict)
for r in rows:
    key = (int(r["radius"]), r["kind"], r["condition"], int(r["seed"]))
    finals[key][int(r["step"])] = float(r["hamming"])
for (radius, kind, cond, seed), series in finals.items():
    traj[(radius, kind, cond)].append(series[max(series)])

radii = [2, 4, 8, 16]
# Single wider/taller figure; inset in upper-right where the curves are low.
fig, ax = plt.subplots(figsize=(5.6, 3.4))
styles = {("single", "intact"):  ("o-",  C_BLUE,   "single-site, intact"),
          ("single", "ablated"): ("o--", C_BLUE,   "single-site, ablated"),
          ("multi",  "intact"):  ("s-",  C_ORANGE, "multi-site, intact"),
          ("multi",  "ablated"): ("s--", C_ORANGE, "multi-site, ablated")}
for (kind, cond), (ls, c, lab) in styles.items():
    means = [np.mean(traj[(r, kind, cond)]) for r in radii]
    sds = [np.std(traj[(r, kind, cond)]) for r in radii]
    ax.errorbar(radii, means, yerr=sds, fmt=ls, color=c, label=lab,
                capsize=2.5, ms=4.5, elinewidth=0.9)
ax.set_xscale("log", base=2)
ax.set_xticks(radii); ax.set_xticklabels(radii)
ax.set_xlabel("Lesion radius (cells)")
ax.set_ylabel("Final Hamming distance to target")
ax.set_ylim(0, None)
# Top-left: low-radius points sit near zero here, curves rise to the right.
ax.legend(frameon=False, ncol=2, fontsize=7, loc="upper left",
          handlelength=2.0, columnspacing=1.0, borderpad=0.4)
ax.grid(True, which="both", axis="y", alpha=0.25, lw=0.5)

# Inset: debris/scar frame. Moved left + slightly up (legend now occupies upper-left,
# so the inset sits in the mid-left band where small-radius curves are low/near zero).
rec = imageio.mimread("results_local/e0_lr1e3/recovery.gif")
frame = rec[-1][:, 0:96]  # last frame, first recovery sample
axin = ax.inset_axes([0.03, 0.20, 0.30, 0.36])
axin.imshow(frame, interpolation="nearest")
axin.set_xticks([]); axin.set_yticks([])
for spine in axin.spines.values():
    spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color("0.3")
axin.patch.set_facecolor("white"); axin.patch.set_alpha(1.0)
axin.set_title("post-regrowth debris\n+ scar tissue", fontsize=6.5, pad=2)
fig.savefig(OUT / "fig1_recovery_vs_radius.png")
plt.close(fig)
print("fig1 done")

# ---------------------------------------------------------------- fig2: hamming vs time
runs = defaultdict(dict)  # cond -> (cseed,dseed) -> {step: h}
with open("results_local/e2_hard_20260730/trajectories.csv") as f:
    for r in csv.DictReader(f):
        runs[r["condition"]].setdefault(
            (int(r["condition_seed"]), int(r["damage_seed"])), {})[int(r["step"])] = float(r["hamming"])

# Colorblind-safe ordering; solid for modulated, distinct for baselines.
order = [("closed_loop",  C_BLUE,   "-",  "closed-loop (evolved)"),
         ("static",       C_ORANGE, "-",  "static"),
         ("constant",     C_GREEN,  "-",  "constant tonic"),
         ("no_modulation",C_RED,    "-",  "no modulation"),
         ("random",       C_PURPLE, "-",  "random")]

fig, ax = plt.subplots(figsize=(5.8, 3.6))
for cond, color, ls, lab in order:
    steps = sorted(next(iter(runs[cond].values())).keys())
    M = np.array([[series[s] for s in steps] for series in runs[cond].values()])
    m, sd = M.mean(axis=0), M.std(axis=0)
    ax.plot(steps, m, color=color, lw=1.3, ls=ls, label=lab)
    ax.fill_between(steps, m - sd, m + sd, color=color, alpha=0.14, lw=0)
for t in range(150, 2000, 150):
    ax.axvline(t, color="0.88", lw=0.5, zorder=0)
ax.set_xlabel("NCA step")
ax.set_ylabel("Hamming distance to target")
ax.set_xlim(0, 2000)
# Legend pushed down ~1.2 of its own heights from upper-left, toward graph center.
# bbox_to_anchor in axes-fraction: x=0.0 (left edge), y=0.62 (~1.2 legend-heights down).
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1,
          handlelength=1.8, borderpad=0.3, bbox_to_anchor=(0.05, 0.50))
ax.grid(True, axis="y", alpha=0.25, lw=0.5)

# Inset: zoom on the low-Hamming range, right side. Moved up ~0.3 of its own height.
axin = ax.inset_axes([0.55, 0.20, 0.42, 0.38])
for cond, color, ls, lab in order[:4]:  # exclude random
    steps = sorted(next(iter(runs[cond].values())).keys())
    M = np.array([[series[s] for s in steps] for series in runs[cond].values()])
    m = M.mean(axis=0)
    axin.plot(steps, m, color=color, lw=1.0, ls=ls)
for t in range(150, 2000, 150):
    axin.axvline(t, color="0.88", lw=0.4, zorder=0)
axin.set_ylim(0, 0.10)
axin.set_xlabel("step", fontsize=6.5)
axin.set_ylabel("Hamming", fontsize=6.5)
axin.tick_params(axis="both", which="major", labelsize=6.5)
axin.set_title("modulated conditions (zoom)", fontsize=7, pad=2)
for spine in axin.spines.values():
    spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color("0.3")
# ensure inset has an opaque patch (background) so the border reads against the plot
axin.patch.set_facecolor("white")
axin.patch.set_alpha(1.0)
fig.savefig(OUT / "fig2_hamming_vs_time.png")
plt.close(fig)
print("fig2 done")

# ---------------------------------------------------------------- fig3: bisection + asymmetric recovery (3 panels)
# Step-1050 lesion genuinely bisects the morphology into two substantial fragments.
# They survive ~60 steps, then the larger monopolizes regrowth and the smaller is
# absorbed (they do NOT re-merge — verified by connected-component tracking).
frames = imageio.mimread("results_local/e2_hard_20260730/rollout_closed_loop.gif")
panels = [(105, "a", "Bisection", 1050),        # lesion frame: two fragments first appear
          (108, "b", "Both persist", 1080),      # 30 steps later: both still substantial
          (115, "c", "Asym. recovery", 1150)]    # 100 steps later: larger dominates, smaller absorbed
fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0))
for ax, (fi, lab, title, step) in zip(axes, panels):
    ax.imshow(frames[fi][16:80, 16:80], interpolation="nearest")
    ax.axis("off")
    # Two-line title: label+name on line 1, step on line 2 — prevents horizontal overflow/collision.
    ax.set_title(f"({lab}) {title}\nstep {step}", fontsize=7.5, pad=3)
fig.savefig(OUT / "fig3_fission_sequence.png")
plt.close(fig)
print("fig3 done")

# ---------------------------------------------------------------- fig4: evolution
gen, big, bsf, mean = [], [], [], []
with open("results_local/e2_hard_20260730/evolve_metrics.csv") as f:
    for r in csv.DictReader(f):
        gen.append(int(r["gen"])); big.append(float(r["best_in_gen"]))
        bsf.append(float(r["best_so_far"])); mean.append(float(r["mean_fitness"]))
fig, ax = plt.subplots(figsize=(5.6, 3.0))
ax.plot(gen, mean, color=C_GREY, lw=0.8, label="population mean")
ax.plot(gen, big, color=C_ORANGE, lw=0.8, alpha=0.55, label="best in generation")
ax.plot(gen, bsf, color=C_BLUE, lw=1.7, label="best so far")
ax.axhline(0.0205, color=C_RED, ls="--", lw=1.0, label="neutral controller (0.0205)")
# Point the arrow at the ACTUAL lowest point (true min of best-so-far), not the endpoint.
min_idx = int(np.argmin(bsf))
min_gen, min_val = gen[min_idx], bsf[min_idx]
ax.annotate(f"best = {min_val:.4f}", xy=(min_gen, min_val),
            xytext=(min_gen - 130, min_val - 0.0008),
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"), fontsize=8)
ax.set_xlabel("Generation")
ax.set_ylabel("Event-weighted fitness (Hamming)")
ax.set_ylim(0.012, 0.023)
# Small legend, tucked tight into the top-right corner (curves descend, so empty there).
ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.5,
          borderpad=0.2, labelspacing=0.3)
ax.grid(True, axis="y", alpha=0.25, lw=0.5)
fig.savefig(OUT / "fig4_evolution_trajectory.png")
plt.close(fig)
print("fig4 done")

# ------------------------------------------------------- fig5: transfer matrix
# Redesigned punchline figure: two facing panels (controller vs tonic transplant).
# Cell color = performance penalty Delta = H(donor) - H(recipient's own), binned:
# green (<=0, matches/beats own), amber (0<Delta<=0.02, noise), crimson (>0.02, harm).
# Bottom-right triangle: survival break - green if survival >= 0.90, amber + skull if < 0.90.
# Rows/cols ordered by greedy cosine seriation of the parents' tonic vectors:
# aligned pair (s4,s1; cos 0.99) adjacent to the diagonal, anti-aligned far apart.
TM = Path("experiment_results/20260822_transfer_matrix")
seeds = [0, 1, 2, 3, 4]

def load_matrix(kind):
    """H[donor, recip] mean over replicas x cond seeds; S worst-case survival."""
    H = np.full((5, 5), np.nan); S = np.full((5, 5), np.nan)
    for rep in ["e1", "e2"]:
        for ri, recip in enumerate(seeds):
            rows = list(csv.DictReader(open(TM / f"cs3_{rep}" / f"recipient_s{recip}" / "transfer.csv")))
            for donor_label in ["own", "s0", "s1", "s2", "s3", "s4"]:
                if donor_label == f"s{recip}":
                    continue
                sub = [x for x in rows if x["kind"] == kind and x["donor"] == donor_label]
                if not sub:
                    continue
                d = recip if donor_label == "own" else int(donor_label[1:])
                h = np.mean([float(x["final_hamming"]) for x in sub])
                s = min(float(x["survival"]) for x in sub)
                H[d, ri] = np.nanmean([H[d, ri], h]) if not np.isnan(H[d, ri]) else h
                S[d, ri] = np.nanmin([S[d, ri], s]) if not np.isnan(S[d, ri]) else s
    return H, S

def delta_from_self(H):
    D = np.copy(H)
    for r in range(5):
        D[:, r] -= H[r, r]     # recipient's own diagonal = zero baseline
    return D

# tonic-vector seriation order (greedy chain over pairwise cosines, most-negative start)
DF = Path("experiment_results/20260818_evoseed_defense")
tonics = {}
for s in seeds:
    vs = []
    for rep in ["e1", "e2"]:
        rows = list(csv.DictReader(open(DF / f"defense_s{s}_{rep}" / "m_series.csv")))
        M = np.array([[float(r[f"m_k{k}"]) for k in range(3)] for r in rows])
        vs.append(M.mean(axis=0))
    tonics[s] = np.mean(vs, axis=0)
COS = np.zeros((5, 5))
for i, a in enumerate(seeds):
    for j, b in enumerate(seeds):
        COS[i, j] = np.dot(tonics[a], tonics[b]) / (np.linalg.norm(tonics[a]) * np.linalg.norm(tonics[b]))
import itertools
best = min(itertools.combinations(range(5), 2), key=lambda p: COS[p[0], p[1]])
order = [best[0]]
remaining = [x for x in range(5) if x != best[0]]
while remaining:
    nxt = max(remaining, key=lambda x: COS[order[-1], x])
    order.append(nxt); remaining.remove(nxt)

SURV_GREEN = "#009E73"
SURV_AMBER = "#E69F00"
def bin_survival(sval):
    if np.isnan(sval):
        return NEUTRAL
    return SURV_GREEN if sval >= 0.90 else SURV_AMBER

GREEN, AMBER, CRIMSON, NEUTRAL = "#009E73", "#E69F00", "#D55E00", "#F0F0F0"
def bin_color(d):
    if np.isnan(d):
        return NEUTRAL
    return GREEN if d <= 0 else (AMBER if d <= 0.02 else CRIMSON)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55), constrained_layout=True)
for ax, kind, title in zip(axes, ["ctrl", "tonic"],
                            ["A  controller transfer", "B  tonic transplant (constant injection)"]):
    H, S = load_matrix(kind)
    D = delta_from_self(H)
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(4.5, -0.5)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"s{seeds[c]}" for c in order])
    ax.set_yticklabels([f"s{seeds[r]}" for r in order])
    ax.set_xlabel("recipient parent", fontsize=8)
    ax.set_ylabel("donor", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_aspect("equal")
    for gi, d_idx in enumerate(order):        # grid position gi = row, gj = col
        for gj, r_idx in enumerate(order):
            dval, sval = D[d_idx, r_idx], S[d_idx, r_idx]
            x, y = gj, d_idx                   # cell center in data coords
            # top-left triangle: Delta color (diagonal = self, neutral)
            col = NEUTRAL if d_idx == r_idx else bin_color(dval)
            ax.add_patch(plt.Polygon([(x-0.5, y-0.5), (x+0.5, y-0.5), (x-0.5, y+0.5)],
                                     closed=True, fc=col, ec="0.3", lw=0.4))
            # bottom-right triangle: survival break (colorblind-friendly)
            sc = bin_survival(sval)
            ax.add_patch(plt.Polygon([(x+0.5, y-0.5), (x+0.5, y+0.5), (x-0.5, y+0.5)],
                                     closed=True, fc=sc, ec="0.3", lw=0.4))
            # annotations: Delta in top-left; skull in bottom-right when lethal-ish
            if d_idx == r_idx:
                ax.text(gj, d_idx, "self", ha="center", va="center", fontsize=5.5, color="0.35", style="italic")
            else:
                ax.text(gj-0.24, d_idx-0.24, f"{dval:+.3f}", ha="center", va="center",
                        fontsize=5.4, color="white" if col == CRIMSON else "black")
            if (not np.isnan(sval)) and sval < 0.90:
                ax.text(gj+0.22, d_idx+0.24, "\u2620", ha="center", va="center",
                        fontsize=8, color="black", fontfamily="DejaVu Sans")
# shared legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(fc=GREEN, label="$\\Delta\\leq 0$ (matches own)"),
           mpatches.Patch(fc=AMBER, label="$0<\\Delta\\leq0.02$ (noise)"),
           mpatches.Patch(fc=CRIMSON, label="$\\Delta>0.02$ (harm)"),
           mpatches.Patch(fc=SURV_GREEN, label="survival $\\geq0.9$"),
           mpatches.Patch(fc=SURV_AMBER, label="survival $<0.9$  (skull)")]
fig.legend(handles=handles, frameon=False, fontsize=6.4, ncol=5,
           loc="lower center", bbox_to_anchor=(0.5, -0.045), handlelength=1.2, columnspacing=0.9)
fig.savefig(OUT / "fig5_transfer_matrix.png", bbox_inches="tight")
plt.close(fig)
print("fig5 done (redesigned: delta-from-self + survival split + cosine order)")

# ------------------------------------------------------- fig6: tonic m_t traces
# Per parent seed: m_t per channel over the rollout, both independent evolutions
# overlaid. Flat lines at parent-distinct offsets = tonic calibration. Each panel
# carries a micro-zoom inset around one lesion event (rotating across panels):
# y-axis stretched to +/-0.01, lesion marked by a red dotted line — showing that
# not even a phasic flicker occurs at the moment of damage.
DF = Path("experiment_results/20260818_evoseed_defense")
ch_cols = [C_BLUE, C_ORANGE, C_GREEN]
fig, axes = plt.subplots(1, 5, figsize=(8.4, 2.1), sharey=True, constrained_layout=True)
zoom_lesion = [300, 450, 600, 750, 900]   # one distinct lesion per panel
for ax, s, les in zip(axes, seeds, zoom_lesion):
    for rep, ls in [("e1", "-"), ("e2", "--")]:
        rows = list(csv.DictReader(open(DF / f"defense_s{s}_{rep}" / "m_series.csv")))
        t = [int(r["step"]) for r in rows]
        for k in range(3):
            ax.plot(t, [float(r[f"m_k{k}"]) for r in rows],
                    color=ch_cols[k], lw=0.7, ls=ls,
                    alpha=0.9 if rep == "e1" else 0.6)
    ax.set_title(f"parent s{s}", fontsize=8.5)
    ax.set_xlabel("step", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    # micro-zoom inset around this panel's lesion
    axi = ax.inset_axes([0.52, 0.60, 0.44, 0.36])
    for rep, ls in [("e1", "-"), ("e2", "--")]:
        rows = list(csv.DictReader(open(DF / f"defense_s{s}_{rep}" / "m_series.csv")))
        for k in range(3):
            axi.plot([int(r["step"]) for r in rows], [float(r[f"m_k{k}"]) for r in rows],
                     color=ch_cols[k], lw=0.7, ls=ls, alpha=0.9 if rep == "e1" else 0.6)
    axi.axvline(les, color="#333333", ls=":", lw=1.0)
    axi.set_xlim(les - 30, les + 30)
    axi.set_ylim(-0.01, 0.01)
    axi.set_xticks([les]); axi.tick_params(labelsize=4.5)
    axi.set_yticks([-0.01, 0.01])
    for sp in axi.spines.values():
        sp.set_linewidth(0.6); sp.set_color("0.3")
    axi.set_facecolor("white")
    ax.indicate_inset_zoom(axi, edgecolor="0.3", lw=0.6, alpha=0.8)
axes[0].set_ylabel("$m_t$", fontsize=8)
axes[0].set_ylim(-0.06, 0.06)
# Shared legend: panel s2's traces are all negative, so its upper area is empty.
axes[2].plot([], [], color="0.4", lw=0.8, ls="-", label="evolution 1")
axes[2].plot([], [], color="0.4", lw=0.8, ls="--", label="evolution 2")
for k, c in enumerate(ch_cols):
    axes[2].plot([], [], color=c, lw=1.2, label=f"channel {k}")
axes[2].legend(frameon=False, fontsize=6, loc="upper left", ncol=1,
               handlelength=1.4, borderpad=0.2, labelspacing=0.25)
fig.savefig(OUT / "fig6_tonic_traces.png")
plt.close(fig)
print("fig6 done (with lesion micro-zoom insets)")
