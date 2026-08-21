"""Cross-parent transfer matrix among the five defense-study parents.

Question: does the July-controller transfer result generalize? Evaluate a
donor controller (evolved on parent d) on a recipient parent r for all
d != r, using the controllers and parents already produced by the defense
study. Evaluation only — no training, no evolution.

Protocol matches src.zero_control exactly (same state0 grown from the
recipient K=3 parent, same held-out damage seeds 10000-10007, same
closed_loop eval path, T from config), but with ONE condition seed
(cond_seed 0 -> rkeys = split(key(cfg["seed"]), S)) instead of five, so
each cell is 8 rollouts and the matrix is 5 donors x 4 recipients x 8.
The recipient's own controller is included as a "donor" row labelled
"own" for an internal consistency check against the defense numbers.

Usage:
  python -m src.transfer_matrix --config /workspace/def_s0_e1.yaml \
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
    parser.add_argument("--donor", action="append", required=True,
                        help="name=path/to/controller.pkl (repeatable)")
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

    eval_fn = make_eval_fn(cfg, target_mask, "closed_loop", K)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"transfer eval: recipient config={args.config} | {S} held-out damage seeds, "
          f"1 condition seed, T={T}")
    print("-" * 78)
    rows = []
    for d in args.donor:
        name, path = d.split("=", 1)
        with open(path, "rb") as f:
            params = pickle.load(f)
        t0 = time.time()
        pb = jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * S), params)
        states = jnp.concatenate([state0[None]] * S)
        rkeys = jax.random.split(jax.random.key(cfg["seed"]), S)  # cond_seed 0
        _, (hamming, alive) = eval_fn(cs, states, pb, schedules, rkeys)
        hamming = np.asarray(hamming)
        finals = hamming[:, -1]
        aucs = hamming.mean(axis=1)
        hls = [repair_half_life(hamming[i], lesion_times,
                                window=cfg["eval"]["half_life_window"]) for i in range(S)]
        surv = float(np.mean(finals < cfg["eval"]["survival_eps"]))
        dt = time.time() - t0
        rows.append((name, surv, float(np.nanmean(hls)), float(np.nanstd(hls)),
                     float(finals.mean()), float(finals.std()),
                     float(aucs.mean()), float(aucs.std()), dt))
        print(f"  donor {name:<4} survival {surv:.2f}  half-life {rows[-1][2]:.1f}  "
              f"final H {rows[-1][4]:.4f}+-{rows[-1][5]:.4f}  "
              f"AUC {rows[-1][6]:.4f}+-{rows[-1][7]:.4f}  [{dt:.0f}s]")

    print("-" * 78)
    with open(out / "transfer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["donor", "survival", "half_life", "half_life_sd",
                    "final_hamming", "final_hamming_sd", "auc", "auc_sd", "elapsed_s"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.2f}", f"{r[3]:.2f}",
                        f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}",
                        f"{r[8]:.1f}"])
    print(f"written -> {out / 'transfer.csv'}")


if __name__ == "__main__":
    main()
