"""E2 stall diagnostics (run after killing the stalled evolution).

Diagnostic 1b (apples-to-apples): measure the NEUTRAL closed-loop controller
(zero params) on the exact E2 fitness function (make_fitness, mode=closed_loop).
The probe's 0.0026 used mode=constant with m pinned to 0 every step; E2 uses
mode=closed_loop where m flows through decision()+step_decay(). These are NOT
the same. This tells us the true gen-0 baseline CMA-ES started from.

Diagnostic 2 (decompose fitness): for both neutral and the best evolved genome
(from evolve_checkpoint.pkl), report raw mean Hamming vs alive_floor penalty
separately. Confirms whether the floor (0.0) is moot and whether evolved
solutions are being taxed for activity.

Usage: python -m src.e2_diag --config /workspace/e2_wired.yaml [--checkpoint results/.../evolve_checkpoint.pkl]
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.controller import zero_params
from src.damage import damage_seed_sets
from src.evolve import build_model, build_schedules, make_eval_fn, make_fitness
from src.metrics import target_alpha_mask
from src.rollout import grow_from_seed
from src.targets import load_target


def raw_hamming_alive(cs, state0, schedules, cfg, params, mode, const_level=None, label=""):
    """Evaluate a controller the way E2 fitness does, returning (mean_hamming, mean_alive, penalty)."""
    rc = cfg["rollout"]
    K = rc["K"]
    target_mask = target_alpha_mask(load_target(**cfg["target"]))
    alive_floor = cfg["evolve"].get("alive_floor", 0.0)
    S = schedules.shape[0]
    states = jnp.concatenate([state0[None]] * S)
    rkeys = jax.random.split(jax.random.key(cfg["seed"]), S)
    # tile params across S seeds
    if params is not None:
        params_b = jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * S), params)
    else:
        params_b = None
    eval_fn = make_eval_fn(cfg, target_mask, mode, K, const_level=const_level)
    _, (hamming, alive) = eval_fn(cs, states, params_b, schedules, rkeys)
    mean_h = float(hamming.mean())
    mean_a = float(alive.mean())
    penalty = 10.0 * float(jnp.mean(jnp.relu(alive_floor - alive))) if alive_floor > 0 else 0.0
    print(f"  {label:35s} hamming {mean_h:.4f}  alive {mean_a:.3f}  penalty {penalty:.4f}  total {mean_h+penalty:.4f}")
    return mean_h, mean_a, penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="evolve_checkpoint.pkl with the best evolved genome")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg["rollout"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    K = rc["K"]

    cs = build_model(cfg, K, cfg["model"].get("parent_params"))
    state0 = grow_from_seed(cs, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    train_seeds, _ = damage_seed_sets(cfg["evolve"]["train_seeds"], cfg["evolve"]["test_seeds"])
    schedules = build_schedules(train_seeds, rc["T"], rc["lesion_interval"], shape)

    print(f"=" * 70)
    print(f"DIAGNOSTIC 1b: apples-to-apples neutral baseline on E2 fitness")
    print(f"  {len(schedules)} train damage seeds, T={rc['T']}, interval={rc['lesion_interval']}, K={K}")
    print(f"  alive_floor = {cfg['evolve'].get('alive_floor', 0.0)}")
    print(f"=" * 70)
    print()

    # The probe measured this (mode=constant, m pinned to 0):
    print("[probe-equivalent: constant mode, m=0 pinned every step]")
    raw_hamming_alive(cs, state0, schedules, cfg, None, "constant", const_level=jnp.zeros((K,)),
                      label="constant m=0 (probe's 0.0026)")
    print()

    # The TRUE E2 gen-0 baseline: mode=closed_loop with zero controller params
    print("[E2 gen-0 baseline: closed_loop mode, neutral zero controller]")
    neutral = zero_params(K=K)
    raw_hamming_alive(cs, state0, schedules, cfg, neutral, "closed_loop",
                      label="neutral closed_loop (true gen-0)")
    print()

    # The best evolved genome
    if args.checkpoint:
        print("[best evolved genome from checkpoint]")
        with open(args.checkpoint, "rb") as f:
            ck = pickle.load(f)
        best_params = ck["best"]["params"]
        print(f"  checkpoint gen: {ck['gen']}, recorded fitness: {ck['best']['fitness']:.4f}")
        raw_hamming_alive(cs, state0, schedules, cfg, best_params, "closed_loop",
                          label=f"evolved best (gen {ck['gen']})")
        print()
        print("=" * 70)
        print("DIAGNOSTIC 2: is the evolved best better or worse than neutral, and why?")
        print("=" * 70)
    else:
        print("(pass --checkpoint to also evaluate the best evolved genome)")


if __name__ == "__main__":
    main()
