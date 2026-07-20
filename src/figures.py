"""Figure 1 / Figure 2 / Table 1 generation from results/ (PRD §9).

Usage:
  python -m src.figures --e1 results/<ts>_e1_lesion_sweep --e2 results/<ts>_e2_closedloop [--out figures/]

Fig 1: recovery rate vs. lesion radius, intact vs. ablated, +/- SEM bands.
Fig 2: Hamming-vs-time for the 5 E2 conditions, mean +/- SEM.
Table 1: survival / half-life / final-Hamming means +/- SEM per condition.
CSV + Matplotlib only (PRD §2).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: str | Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def mean_sem(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def fig1(e1_dir: Path, out_dir: Path) -> None:
    """Recovery fraction at final step vs. lesion radius, per kind x condition."""
    rows = read_rows(e1_dir / "trajectories.csv")
    # group trajectories: (radius, kind, condition, seed) -> {step: hamming}
    trajs: dict = defaultdict(dict)
    for r in rows:
        key = (int(r["radius"]), r["kind"], r["condition"], int(r["seed"]))
        trajs[key][int(r["step"])] = float(r["hamming"])

    # recovery fraction = (h_peak - h_final) / h_peak (post-lesion peak); 0 if no damage
    rec: dict = defaultdict(list)  # (radius, kind, condition) -> [fraction per seed]
    for (radius, kind, condition, _seed), traj in trajs.items():
        steps = sorted(traj)
        h = np.array([traj[s] for s in steps])
        h_peak, h_final = h.max(), h[-1]
        frac = (h_peak - h_final) / h_peak if h_peak > 1e-6 else 1.0
        rec[(radius, kind, condition)].append(float(frac))

    kinds = sorted({k for (_, k, _) in rec})
    conditions = sorted({c for (_, _, c) in rec})
    radii = sorted({r for (r, _, _) in rec})
    colors = {"intact": "tab:green", "ablated": "tab:red"}

    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), sharey=True)
    if len(kinds) == 1:
        axes = [axes]
    for ax, kind in zip(axes, kinds):
        for condition in conditions:
            means, sems = [], []
            for radius in radii:
                m, s = mean_sem(rec.get((radius, kind, condition), []))
                means.append(m)
                sems.append(s)
            means, sems = np.array(means), np.array(sems)
            ax.plot(radii, means, marker="o", color=colors.get(condition), label=condition)
            ax.fill_between(radii, means - sems, means + sems, alpha=0.2, color=colors.get(condition))
        ax.set_xscale("log", base=2)
        ax.set_xticks(radii, [str(r) for r in radii])
        ax.set_xlabel("lesion radius (px)")
        ax.set_title(f"kind = {kind}")
        ax.legend()
    axes[0].set_ylabel("recovery fraction at T")
    fig.suptitle("Fig 1 — recovery vs. lesion radius, intact vs. ablated")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_recovery_vs_radius.png", dpi=150)
    plt.close(fig)


def fig2(e2_dir: Path, out_dir: Path) -> None:
    """Hamming-vs-time per condition, mean +/- SEM across seeds and damage seeds."""
    rows = read_rows(e2_dir / "trajectories.csv")
    series: dict = defaultdict(lambda: defaultdict(list))  # condition -> step -> [hamming]
    for r in rows:
        series[r["condition"]][int(r["step"])].append(float(r["hamming"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for condition in sorted(series):
        steps = sorted(series[condition])
        means = np.array([mean_sem(series[condition][s])[0] for s in steps])
        sems = np.array([mean_sem(series[condition][s])[1] for s in steps])
        ax.plot(steps, means, label=condition)
        ax.fill_between(steps, means - sems, means + sems, alpha=0.15)
    ax.set_xlabel("NCA step")
    ax.set_ylabel("Hamming to target")
    ax.set_title("Fig 2 — closed-loop vs. baselines under recurring damage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_hamming_vs_time.png", dpi=150)
    plt.close(fig)


def table1(e2_dir: Path, out_dir: Path) -> None:
    """Survival rate, repair half-life, final Hamming per condition (mean +/- SEM)."""
    rows = read_rows(e2_dir / "metrics.csv")
    data: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        data[c]["survival"].append(float(r["survived"]))
        data[c]["half_life"].append(float(r["half_life"]))
        data[c]["final_hamming"].append(float(r["final_hamming"]))
        data[c]["auc"].append(float(r["auc"]))

    out_csv = out_dir / "table1_survival_halflife.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "survival", "half_life_mean", "half_life_sem",
                         "final_hamming_mean", "final_hamming_sem", "auc_mean", "auc_sem"])
        md_lines = ["| condition | survival | half-life | final Hamming | Hamming AUC |",
                    "|---|---|---|---|---|"]
        for condition in sorted(data):
            surv = float(np.mean(data[condition]["survival"]))
            hl_m, hl_s = mean_sem(data[condition]["half_life"])
            fh_m, fh_s = mean_sem(data[condition]["final_hamming"])
            auc_m, auc_s = mean_sem(data[condition]["auc"])
            writer.writerow([condition, f"{surv:.3f}", f"{hl_m:.1f}", f"{hl_s:.1f}",
                             f"{fh_m:.4f}", f"{fh_s:.4f}", f"{auc_m:.4f}", f"{auc_s:.4f}"])
            md_lines.append(f"| {condition} | {surv:.2f} | {hl_m:.1f} ± {hl_s:.1f} | "
                            f"{fh_m:.3f} ± {fh_s:.3f} | {auc_m:.3f} ± {auc_s:.3f} |")
    (out_dir / "table1_survival_halflife.md").write_text("\n".join(md_lines) + "\n")
    print(f"table 1 -> {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", default=None, help="E1 run dir with trajectories.csv")
    parser.add_argument("--e2", default=None, help="E2 run dir with trajectories.csv + metrics.csv")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.e1:
        fig1(Path(args.e1), out_dir)
        print("fig 1 -> fig1_recovery_vs_radius.png")
    if args.e2:
        fig2(Path(args.e2), out_dir)
        table1(Path(args.e2), out_dir)
        print("fig 2 -> fig2_hamming_vs_time.png")


if __name__ == "__main__":
    main()
