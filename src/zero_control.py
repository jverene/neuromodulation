"""Zero-output control (the missing condition from the paper's Limitations).

Question: does the broadcast-modulation improvement come from *active
modulation* or merely from the channels being present during parent training?

Conditions (held-out damage seeds, hard multi-block regime, same protocol as
the 5-condition eval):
  - zero_output:  K=3 channel-aware parent, modulation PINNED to m=0 for the
                  whole rollout (mode "ablated" — controller ignored).
  - closed_loop:  the evolved controller (evolved_controller.pkl) on the same
                  K=3 parent — internal-consistency reference on fresh parents.
  - no_modulation: K=0 baseline parent.

Verdict logic:
  zero_output ≈ 0.028-0.030  -> improvement is a parent-training effect; the
                                 paper's claim needs weakening.
  zero_output ≈ 0.063         -> active modulation genuinely drives the gain.

Usage: python -m src.zero_control --config <wired e2_hard.yaml>
       (model.parent_params -> fresh K=3 run dir, model.baseline_params -> fresh K=0 run dir)
"""

import argparse
import csv
import pickle
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


def build_block_schedules(seeds, T, interval, shape, block_side, n_blocks):
    scheds = []
    for s in seeds:
        times, masks = make_recurring_schedule_block(
            s, T=T, interval=interval, shape=shape, block_side=block_side, n_blocks=n_blocks)
        scheds.append(dense_lesion_masks(times, masks, T, shape))
    return jnp.stack(scheds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller", default="evolved_controller.pkl")
    parser.add_argument("--out", default="results/zero_control")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg["rollout"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    K = rc["K"]
    T = rc["T"]
    dmg = cfg["damage"]

    cs_k3 = build_model(cfg, K, cfg["model"].get("parent_params"))
    cs_k0 = build_model(cfg, 0, cfg["model"].get("baseline_params"))
    state0_k3 = grow_from_seed(cs_k3, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    state0_k0 = grow_from_seed(cs_k0, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    target_mask = target_alpha_mask(load_target(**cfg["target"]))

    _, test_seeds = damage_seed_sets(cfg["evolve"]["train_seeds"], cfg["evolve"]["test_seeds"])
    schedules = build_block_schedules(test_seeds, T, rc["lesion_interval"], shape,
                                      dmg["block_side"], dmg["n_blocks"])
    lesion_times = np.arange(rc["lesion_interval"], T, rc["lesion_interval"])
    S = schedules.shape[0]
    n_cond = cfg["eval"]["num_condition_seeds"]

    with open(args.controller, "rb") as f:
        evolved = pickle.load(f)

    conditions = [
        ("zero_output",   cs_k3, 3, "ablated",    None),
        ("closed_loop",   cs_k3, 3, "closed_loop", evolved),
        ("no_modulation", cs_k0, 0, "ablated",    None),
    ]
    # Protocol note: evolve.py's evaluate_conditions uses ONE shared state0,
    # grown from the K=3 parent, for ALL conditions (no_modulation included —
    # it begins as a K=3-grown lizard living under K=0 dynamics). Match it
    # exactly so this control is apples-to-apples with the paper's table.
    state0 = state0_k3

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"zero-output control: {S} held-out damage seeds x {n_cond} condition seeds, T={T}")
    print("-" * 78)
    for name, cs_c, K_c, mode, params in conditions:
        eval_fn = make_eval_fn(cfg, target_mask, mode, K_c)
        finals, aucs, hls = [], [], []
        for cond_seed in range(n_cond):
            states = jnp.concatenate([state0[None]] * S)
            if params is not None:
                pb = jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * S), params)
            else:
                pb = None
            rkeys = jax.random.split(jax.random.key(cfg["seed"] + 1000 * cond_seed), S)
            _, (hamming, alive) = eval_fn(cs_c, states, pb, schedules, rkeys)
            hamming, alive = np.asarray(hamming), np.asarray(alive)
            for d in range(S):
                finals.append(float(hamming[d, -1]))
                aucs.append(float(hamming[d].mean()))
                hls.append(repair_half_life(hamming[d], lesion_times, window=cfg["eval"]["half_life_window"]))
        f_mean, f_sd = float(np.mean(finals)), float(np.std(finals))
        a_mean, a_sd = float(np.mean(aucs)), float(np.std(aucs))
        h_mean, h_sd = float(np.nanmean(hls)), float(np.nanstd(hls))
        surv = float(np.mean(np.array(finals) < cfg["eval"]["survival_eps"]))
        rows.append((name, surv, h_mean, h_sd, f_mean, f_sd, a_mean, a_sd))
        print(f"  {name:<14} survival {surv:.2f}  half-life {h_mean:.1f}±{h_sd:.1f}  "
              f"final H {f_mean:.4f}±{f_sd:.4f}  AUC {a_mean:.4f}±{a_sd:.4f}")

    print("-" * 78)
    zo = rows[0]; cl = rows[1]; nm = rows[2]
    print(f"VERDICT: zero_output final H = {zo[4]:.4f}")
    print(f"  vs modulated band (closed_loop here: {cl[4]:.4f}; paper's 0.028-0.030)")
    print(f"  vs no_modulation (here: {nm[4]:.4f}; paper's 0.063)")
    mid = (cl[4] + nm[4]) / 2
    if zo[4] < mid:
        print("  -> zero_output lands with the MODULATED band: improvement is largely")
        print("     a parent-training effect (channels present during training).")
    else:
        print("  -> zero_output lands with NO_MODULATION: active modulation drives")
        print("     the improvement; the claim is strengthened.")

    with open(out / "zero_control.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "survival", "half_life", "half_life_sd",
                    "final_hamming", "final_hamming_sd", "auc", "auc_sd"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.2f}", f"{r[3]:.2f}",
                        f"{r[4]:.6f}", f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.6f}"])
    print(f"written -> {out / 'zero_control.csv'}")


if __name__ == "__main__":
    main()
