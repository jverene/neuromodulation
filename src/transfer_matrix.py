"""Cross-parent transfer matrix among the five defense-study parents.

Question: does the July-controller transfer result generalize? Evaluate a
donor controller (evolved on parent d) on a recipient parent r for all
d != r, using the controllers and parents already produced by the defense
study. Evaluation only — no training, no evolution.

Two donor kinds:
  --donor  name=path/to/controller.pkl   full controller, closed_loop mode
  --tonic  name=path/to/m_series.csv     tonic transplant: donor's realized
           mean m_t injected as a constant (mode "constant"). Since
           m_series records level(mod) — the exact quantity injected into
           the NCA — this isolates the learned tonic vector from the
           (tiny) dynamic residual.

Protocol matches src.zero_control (same state0 grown from the recipient
K=3 parent, same held-out damage seeds 10000-10007, T from config).
--cond-seeds N evaluates N condition seeds (rkeys =
split(key(cfg["seed"] + 1000*cond_seed), S), matching zero_control);
classifications can then be checked for stability across stochastic
update streams. The recipient's own controller is included as a donor row
labelled "own" for an internal consistency check; "own_tonic" does the
same for the transplant mode.

Usage:
  python -m src.transfer_matrix --config /workspace/def_s0_e1.yaml \
      --cond-seeds 3 \
      --donor own=results/defense_s0_e1/own_controller.pkl \
      --donor s1=results/defense_s1_e1/own_controller.pkl ... \
      --out results/transfer_matrix/recipient_s0
"""

import argparse
import csv
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.damage import damage_seed_sets, make_recurring_schedule_block
from src.evolve import build_model, make_eval_fn
from src.metrics import repair_half_life, target_alpha_mask
from src.rollout import dense_lesion_masks, grow_from_seed
from src.targets import load_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="recipient-parent config (model.parent_params -> recipient K=3 params)")
    parser.add_argument("--donor", action="append", default=[],
                        help="name=path/to/controller.pkl (repeatable)")
    parser.add_argument("--tonic", action="append", default=[],
                        help="name=path/to/m_series.csv (repeatable); mean m_t injected as constant")
    parser.add_argument("--cond-seeds", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg["rollout"]
    dmg = cfg["damage"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    K = rc["K"]
    T = rc["T"]

    cs = build_model(cfg, K, cfg["model"].get("parent_params"))
    state0 = grow_from_seed(cs, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    target_mask = target_alpha_mask(load_target(**cfg["target"]))

    _, test_seeds = damage_seed_sets(cfg["evolve"]["train_seeds"], cfg["evolve"]["test_seeds"])
    scheds = []
    for s in test_seeds:
        times, masks = make_recurring_schedule_block(
            s, T=T, interval=rc["lesion_interval"], shape=shape,
            block_side=dmg["block_side"], n_blocks=dmg["n_blocks"])
        scheds.append(dense_lesion_masks(times, masks, T, shape))
    schedules = jnp.stack(scheds)
    lesion_times = np.arange(rc["lesion_interval"], T, rc["lesion_interval"])
    S = schedules.shape[0]

    eval_ctrl = make_eval_fn(cfg, target_mask, "closed_loop", K)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"transfer eval: recipient config={args.config} | {S} held-out damage seeds x "
          f"{args.cond_seeds} condition seed(s), T={T}")
    print("-" * 78)
    rows = []

    def run_eval(name, kind, const_level, params):
        eval_fn = eval_ctrl
        if kind == "tonic":
            eval_fn = make_eval_fn(cfg, target_mask, "constant", K, const_level=const_level)
        for cond_seed in range(args.cond_seeds):
            t0 = time.time()
            states = jnp.concatenate([state0[None]] * S)
            pb = None
            if params is not None:
                pb = jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * S), params)
            rkeys = jax.random.split(jax.random.key(cfg["seed"] + 1000 * cond_seed), S)
            _, (hamming, alive) = eval_fn(cs, states, pb, schedules, rkeys)
            hamming = np.asarray(hamming)
            finals = hamming[:, -1]
            aucs = hamming.mean(axis=1)
            hls = [repair_half_life(hamming[i], lesion_times,
                                    window=cfg["eval"]["half_life_window"]) for i in range(S)]
            surv = float(np.mean(finals < cfg["eval"]["survival_eps"]))
            dt = time.time() - t0
            rows.append((name, kind, cond_seed, surv,
                         float(np.nanmean(hls)), float(np.nanstd(hls)),
                         float(finals.mean()), float(finals.std()),
                         float(aucs.mean()), float(aucs.std()), dt))
            print(f"  {kind:<6} {name:<10} cs{cond_seed} survival {surv:.2f}  "
                  f"final H {rows[-1][6]:.4f}+-{rows[-1][7]:.4f}  "
                  f"AUC {rows[-1][8]:.4f}+-{rows[-1][9]:.4f}  [{dt:.0f}s]")

    for d in args.donor:
        name, path = d.split("=", 1)
        with open(path, "rb") as f:
            params = pickle.load(f)
        run_eval(name, "ctrl", None, params)
    for d in args.tonic:
        name, path = d.split("=", 1)
        series = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3))
        run_eval(name, "tonic", jnp.asarray(series.mean(axis=0)), None)

    print("-" * 78)
    with open(out / "transfer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["donor", "kind", "cond_seed", "survival", "half_life", "half_life_sd",
                    "final_hamming", "final_hamming_sd", "auc", "auc_sd", "elapsed_s"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.3f}", f"{r[4]:.2f}", f"{r[5]:.2f}",
                        f"{r[6]:.6f}", f"{r[7]:.6f}", f"{r[8]:.6f}", f"{r[9]:.6f}",
                        f"{r[10]:.1f}"])
    print(f"written -> {out / 'transfer.csv'}")


if __name__ == "__main__":
    main()
