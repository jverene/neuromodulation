"""Generate paper figures (fig1-fig4) into paper_drafts/figures/ from repo data."""
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
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

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
fig, ax = plt.subplots(figsize=(6.4, 2.6))
styles = {("single", "intact"): ("o-", "C0", "single, intact"),
          ("single", "ablated"): ("o--", "C0", "single, ablated"),
          ("multi", "intact"): ("s-", "C1", "multi, intact"),
          ("multi", "ablated"): ("s--", "C1", "multi, ablated")}
for (kind, cond), (ls, c, lab) in styles.items():
    means = [np.mean(traj[(r, kind, cond)]) for r in radii]
    sds = [np.std(traj[(r, kind, cond)]) for r in radii]
    ax.errorbar(radii, means, yerr=sds, fmt=ls, color=c, label=lab, capsize=3, ms=4)
ax.set_xscale("log", base=2); ax.set_xticks(radii); ax.set_xticklabels(radii)
ax.set_xlabel("lesion radius (cells)"); ax.set_ylabel("final Hamming")
ax.legend(frameon=False, ncol=2, fontsize=8)

# inset: debris/scar frame from E0 recovery (4-sample strip; crop sample 0)
rec = imageio.mimread("results_local/e0_lr1e3/recovery.gif")
frame = rec[-1][:, 0:96]  # last frame, first recovery sample
axin = ax.inset_axes([0.40, 0.45, 0.28, 0.5])
axin.imshow(frame, interpolation="nearest")
axin.set_xticks([]); axin.set_yticks([])
axin.set_title("debris + scar tissue", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig1_recovery_vs_radius.png", dpi=200)
plt.close(fig)
print("fig1 done")

# ---------------------------------------------------------------- fig2: hamming vs time
runs = defaultdict(dict)  # cond -> (cseed,dseed) -> {step: h}
with open("results_local/e2_hard_20260730/trajectories.csv") as f:
    for r in csv.DictReader(f):
        runs[r["condition"]].setdefault(
            (int(r["condition_seed"]), int(r["damage_seed"])), {})[int(r["step"])] = float(r["hamming"])

order = [("closed_loop", "C0", "closed-loop (evolved)"),
         ("static", "C1", "static"),
         ("constant", "C2", "constant tonic"),
         ("no_modulation", "C3", "no modulation"),
         ("random", "C4", "random")]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for cond, color, lab in order:
    steps = sorted(next(iter(runs[cond].values())).keys())
    M = np.array([[series[s] for s in steps] for series in runs[cond].values()])
    m, sd = M.mean(axis=0), M.std(axis=0)
    ax.plot(steps, m, color=color, lw=1.2, label=lab)
    ax.fill_between(steps, m - sd, m + sd, color=color, alpha=0.15, lw=0)
for t in range(150, 2000, 150):
    ax.axvline(t, color="0.85", lw=0.5, zorder=0)
ax.set_xlabel("NCA step"); ax.set_ylabel("Hamming distance to target")
ax.legend(frameon=False, fontsize=8, loc="center left")
axin = ax.inset_axes([0.45, 0.22, 0.5, 0.40])
for cond, color, lab in order[:4]:
    steps = sorted(next(iter(runs[cond].values())).keys())
    M = np.array([[series[s] for s in steps] for series in runs[cond].values()])
    axin.plot(steps, M.mean(axis=0), color=color, lw=1.0)
axin.set_ylim(0, 0.1); axin.set_title("zoom (random omitted)", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig2_hamming_vs_time.png", dpi=200)
plt.close(fig)
print("fig2 done")

# ---------------------------------------------------------------- fig3: fission panels
frames = imageio.mimread("results_local/e2_hard_20260730/rollout_closed_loop.gif")
panels = [(104, "a", "pre-lesion (step 1040)"), (106, "b", "midline lesion (1060)"),
          (108, "c", "split (1080)"), (110, "d", "two growth fronts (1100)")]
fig, axes = plt.subplots(1, 4, figsize=(6.4, 1.9))
for ax, (fi, lab, title) in zip(axes, panels):
    ax.imshow(frames[fi][16:80, 16:80], interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"({lab}) {title}", fontsize=7.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_fission_sequence.png", dpi=200)
plt.close(fig)
print("fig3 done")

# ---------------------------------------------------------------- fig4: evolution
gen, big, bsf, mean = [], [], [], []
with open("results_local/e2_hard_20260730/evolve_metrics.csv") as f:
    for r in csv.DictReader(f):
        gen.append(int(r["gen"])); big.append(float(r["best_in_gen"]))
        bsf.append(float(r["best_so_far"])); mean.append(float(r["mean_fitness"]))
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.plot(gen, mean, color="0.6", lw=0.8, label="population mean")
ax.plot(gen, big, color="C1", lw=0.8, alpha=0.6, label="best in generation")
ax.plot(gen, bsf, color="C0", lw=1.6, label="best so far")
ax.axhline(0.0205, color="C3", ls="--", lw=1.0, label="neutral controller (0.0205)")
ax.annotate("0.0135", xy=(299, 0.0135), xytext=(235, 0.0128),
            arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
ax.set_xlabel("generation"); ax.set_ylabel("event-weighted fitness")
ax.set_ylim(0.012, 0.021)
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig4_evolution_trajectory.png", dpi=200)
plt.close(fig)
print("fig4 done")
