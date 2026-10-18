"""Post-hoc reviewer-requested re-evaluation (camera-ready revision, 2026-10).

Two clarifications the submitted table cannot answer:
  1. Reviewer (b): the paper's protocol evaluates the K=0 parent from a
     t=0 state grown by the K=3 parent ("shared state0"), so the K=0
     condition starts from another network's latent code. This script adds
     k0_own: the same K=0 parent evaluated from a state grown by itself.
  2. Metric review: final Hamming conflates translation drift with shape
     error (App. H). This script also reports translation-aligned final
     Hamming: min over integer shifts |dy|,|dx| <= max_shift of the
     binarized-alpha mismatch rate (np.roll wrap-around is safe: the
     40px target sits inside a 96px canvas, 28px margins >> shifts).

Protocol otherwise matches src.zero_control exactly (same held-out damage
seeds 10000-10007, same condition-seed key streams, same T/interval/blocks),
so plain-Hamming columns land next to the preregistered numbers.

Conditions:
  k3_m0_shared: K=3 parent, m pinned 0, shared K=3-grown state0 (paper)
  k0_shared:    K=0 parent from the SAME K=3-grown state0 (paper's K=0 col)
  k0_own:       K=0 parent from its own-grown state0 (new control)

Usage (one parent at a time, config wired like zero_control):
  python -m src.posthoc_metrics --config <yaml> --label s0_e1 --out out.csv
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.damage import damage_seed_sets
from src.evolve import build_model, make_eval_fn
from src.metrics import target_alpha_mask
from src.rollout import grow_from_seed
from src.targets import load_target
from src.zero_control import build_block_schedules


def aligned_hamming(alpha_bin: np.ndarray, target: np.ndarray, max_shift: int) -> float:
    """Min mismatch rate over integer translations within +-max_shift pixels."""
    best = np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            rolled = np.roll(np.roll(alpha_bin, dy, axis=0), dx, axis=1)
            best = min(best, float(np.mean(rolled != target)))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", required=True, help="e.g. s0_e1 (parent seed, replica)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-shift", type=int, default=10)
    parser.add_argument("--controller", default=None,
                        help="evolved controller pkl; adds a k3_ctrl closed-loop condition")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg["rollout"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    dmg = cfg["damage"]

    cs_k3 = build_model(cfg, rc["K"], cfg["model"].get("parent_params"))
    cs_k0 = build_model(cfg, 0, cfg["model"].get("baseline_params"))
    state0_k3 = grow_from_seed(cs_k3, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    state0_k0 = grow_from_seed(cs_k0, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    target_mask = target_alpha_mask(load_target(**cfg["target"]))

    _, test_seeds = damage_seed_sets(cfg["evolve"]["train_seeds"], cfg["evolve"]["test_seeds"])
    schedules = build_block_schedules(test_seeds, rc["T"], rc["lesion_interval"], shape,
                                      dmg["block_side"], dmg["n_blocks"])
    S = schedules.shape[0]
    n_cond = cfg["eval"]["num_condition_seeds"]
    eps = cfg["eval"]["survival_eps"]
    T = rc["T"]

    # (name, model, K, state0, mode, params) — ablated rows pin m=0.
    conditions = [
        ("k3_m0_shared", cs_k3, rc["K"], state0_k3, "ablated", None),
        ("k0_shared", cs_k0, 0, state0_k3, "ablated", None),
        ("k0_own", cs_k0, 0, state0_k0, "ablated", None),
    ]
    if args.controller:
        with open(args.controller, "rb") as f:
            import pickle
            ctrl_params = pickle.load(f)
        conditions = [
            ("k3_m0_shared", cs_k3, rc["K"], state0_k3, "ablated", None),
            ("k3_ctrl", cs_k3, rc["K"], state0_k3, "closed_loop", ctrl_params),
        ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    rows = []
    print(f"post-hoc metrics: {args.label}, {S} held-out damage seeds x {n_cond} condition seeds, "
          f"T={T}, max_shift={args.max_shift}")
    print("-" * 78)
    for name, cs_c, K_c, state0, mode, params in conditions:
        eval_fn = make_eval_fn(cfg, target_mask, mode, K_c)
        plain, aligned = [], []
        for cond_seed in range(n_cond):
            states = jnp.concatenate([state0[None]] * S)
            rkeys = jax.random.split(jax.random.key(cfg["seed"] + 1000 * cond_seed), S)
            pb = None
            if params is not None:
                pb = jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * S), params)
            states_f, (hamming, _) = eval_fn(cs_c, states, pb, schedules, rkeys)
            alpha_bin = np.asarray(states_f[..., -1] > 0.5)
            for d in range(S):
                plain.append(float(hamming[d, -1]))
                aligned.append(aligned_hamming(alpha_bin[d], np.asarray(target_mask), args.max_shift))
        p_m, p_sd = float(np.mean(plain)), float(np.std(plain))
        a_m, a_sd = float(np.mean(aligned)), float(np.std(aligned))
        sp = float(np.mean(np.asarray(plain) < eps))
        sa = float(np.mean(np.asarray(aligned) < eps))
        rows.append((args.label, name, len(plain), p_m, p_sd, a_m, a_sd, sp, sa))
        print(f"{name:12s}  plain {p_m:.4f}±{p_sd:.4f}  aligned {a_m:.4f}±{a_sd:.4f}  "
              f"surv plain {sp:.3f}  aligned {sa:.3f}")

    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["label", "condition", "n", "plain_hamming_mean", "plain_hamming_sd",
                        "aligned_hamming_mean", "aligned_hamming_sd",
                        "survival_plain", "survival_aligned"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
