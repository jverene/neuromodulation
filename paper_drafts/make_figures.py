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
panels = [(105, "a", "Bisection (1050)"),       # lesion frame: two fragments first appear
          (108, "b", "Both persist (1080)"),     # 30 steps later: both still substantial
          (115, "c", "Asymmetric recovery (1150)")]  # 100 steps later: larger dominates, smaller absorbed
fig, axes = plt.subplots(1, 3, figsize=(5.0, 1.9))
for ax, (fi, lab, title) in zip(axes, panels):
    ax.imshow(frames[fi][16:80, 16:80], interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"({lab}) {title}", fontsize=7.5, pad=2)
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
