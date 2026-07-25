"""Damage calibration for the hard-regime E2 (post-redesign).

Three jobs in one run:
  1. Sweep (block_side, n_blocks, interval) for the multi_block schedule and
     report the NEUTRAL controller's raw endpoint Hamming + alive. Goal: land
     baseline in the struggle zone (raw H ~0.02-0.046, alive > 0.03) — visibly
     failing, not dead. Picks the cheapest schedule in that zone.
  2. For the chosen schedule, also report the EVENT-WEIGHTED fitness (the new
     scalar E2 minimizes) so we know what CMA-ES starts from.
  3. Measure the baseline's repair HALF-LIFE in the chosen regime, to sanity-
     check tau_w = interval/3 (amendment 2): if half-life >> tau_w, the kernel
     decays before Hamming recovers -> bump tau_w toward interval/2.

Uses the SAME closed_loop neutral evaluation + make_fitness as E2 (apples-to-
apples). Minutes on A100.

Usage: python -m src.calibrate_damage --config /workspace/e2_wired.yaml
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.controller import zero_params
from src.damage import damage_seed_sets, make_recurring_schedule_block
from src.evolve import build_model, make_eval_fn, make_fitness
from src.metrics import repair_half_life, target_alpha_mask
from src.rollout import dense_lesion_masks, grow_from_seed
from src.targets import load_target


def build_block_schedules(seeds, T, interval, shape, block_side, n_blocks):
    scheds = []
    for s in seeds:
        times, masks = make_recurring_schedule_block(
            s, T=T, interval=interval, shape=shape, block_side=block_side, n_blocks=n_blocks)
        scheds.append(dense_lesion_masks(times, masks, T, shape))
    return jnp.stack(scheds), times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg["rollout"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    K = rc["K"]
    T = rc["T"]

    cs = build_model(cfg, K, cfg["model"].get("parent_params"))
    state0 = grow_from_seed(cs, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    target_mask = target_alpha_mask(load_target(**cfg["target"]))
    neutral = zero_params(K=K)

    train_seeds, _ = damage_seed_sets(cfg["evolve"]["train_seeds"], cfg["evolve"]["test_seeds"])
    n_seeds = len(train_seeds)
    states = jnp.concatenate([state0[None]] * n_seeds)
    rkeys = jax.random.split(jax.random.key(cfg["seed"]), n_seeds)
    params_b = jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * n_seeds), neutral)

    eval_fn_const = make_eval_fn(cfg, target_mask, "closed_loop", K)  # neutral closed_loop

    print(f"damage calibration: neutral closed_loop, {n_seeds} seeds, T={T}, K={K}")
    print(f"struggle zone target: raw endpoint Hamming ~0.02-0.046, alive > 0.03")
    print("-" * 95)
    print(f"{'side':>5} {'n_blk':>5} {'intvl':>5} {'raw_H_mean':>10} {'raw_H_max':>9} {'alive':>6} "
          f"{'event_w_fit':>11} {'half_life':>9}  verdict")
    print("-" * 95)

    grid = [(side, nb, iv)
            for side in [8, 12, 16, 20]
            for nb in [2, 3, 4]
            for iv in [150, 100]]

    t0 = time.time()
    results = []
    for side, nb, iv in grid:
        schedules, times = build_block_schedules(train_seeds, T, iv, shape, side, nb)
        _, (hamming, alive) = eval_fn_const(cs, states, params_b, schedules, rkeys)
        hamming = np.asarray(hamming); alive = np.asarray(alive)
        raw_mean = float(hamming.mean()); raw_max = float(hamming.max())
        mean_alive = float(alive.mean())
        # event-weighted fitness (the new scalar E2 uses) — needs make_fitness with this interval
        cfg_iv = {**cfg, "rollout": {**rc, "lesion_interval": iv},
                  "evolve": {"alive_floor": 0.0, "fitness": "event_weighted", "tau_w": iv / 3.0}}
        fb = make_fitness(cfg_iv, target_mask, np.asarray(times))
        fit_pop = fb(cs, states, params_b, schedules, rkeys)
        ew_fit = float(fit_pop.mean())
        # repair half-life (mean across seeds), for tau_w sanity (amendment 2)
        hls = [repair_half_life(hamming[d], times, window=iv) for d in range(n_seeds)]
        hl = float(np.nanmean(hls)) if not np.all(np.isnan(hls)) else float("nan")
        if raw_mean >= 0.02 and raw_mean <= 0.046 and mean_alive > 0.03:
            verdict = "*** STRUGGLE ZONE ***"
        elif raw_mean > 0.046:
            verdict = "too hard (near cap)"
        elif mean_alive <= 0.03:
            verdict = "too dead"
        else:
            verdict = "too easy (<0.02)"
        results.append((side, nb, iv, raw_mean, raw_max, mean_alive, ew_fit, hl, verdict))
        print(f"{side:>5} {nb:>5} {iv:>5} {raw_mean:>10.4f} {raw_max:>9.4f} {mean_alive:>6.3f} "
              f"{ew_fit:>11.4f} {hl:>9.1f}  {verdict}")

    print("-" * 95)
    print(f"calibration took {time.time()-t0:.0f}s for {len(grid)} schedules")
    print()
    zone = [r for r in results if "STRUGGLE ZONE" in r[8]]
    if zone:
        # cheapest = largest interval, fewest blocks, smallest side (least damage) in the zone
        pick = min(zone, key=lambda r: (r[0]*r[1] - r[2]/10.0))  # prefer less damage, more interval
        print(f"RECOMMENDED (struggle zone, least-damaging):")
        print(f"  block_side={pick[0]}, n_blocks={pick[1]}, interval={pick[2]}")
        print(f"  raw H mean={pick[3]:.4f}, alive={pick[5]:.3f}, event-weighted fit={pick[6]:.4f}")
        print(f"  repair half-life={pick[7]:.1f} steps (vs tau_w=interval/3={pick[2]/3.0:.1f})")
        if not np.isnan(pick[7]):
            ratio = pick[7] / (pick[2] / 3.0)
            print(f"  tau_w check: half-life/tau_w = {ratio:.2f}", end="")
            if ratio > 2.0:
                print(f"  -> half-life >> tau_w, BUMP tau_w to ~interval/2={pick[2]/2.0:.1f} (amendment 2)")
            else:
                print(f"  -> tau_w ~ interval/3 is fine")
    else:
        print("No schedule landed in the struggle zone. Closest:")
        candidates = [r for r in results if r[5] > 0.03]
        if candidates:
            best = min(candidates, key=lambda r: abs(r[3] - 0.033))
            print(f"  side={best[0]}, n={best[1]}, iv={best[2]}: raw H {best[3]:.4f}, alive {best[5]:.3f} [{best[8]}]")


if __name__ == "__main__":
    main()
