"""Damage-regime probe (post-E2-stall rescoping).

Question: what recurring-damage schedule actually BREAKS the neutral baseline?
E2's current schedule (radius ~8 single-site every 250 steps) leaves neutral at
Hamming 0.0023 — trivially solved, no gap for modulation. We need a regime where
the unmodulated lizard genuinely fails (Hamming climbs well above ~0.01).

Probes recurring damage at a grid of (radius, kind, interval) and reports mean
Hamming + alive for the NEUTRAL controller. Looks for the knee: the cheapest
schedule that pushes neutral's Hamming into a clearly-failing range (say >0.1)
while keeping the lizard alive (so the task is still recoverable, not hopeless).

Uses the SAME closed_loop neutral evaluation as E2 fitness (apples-to-apples).

Usage: python -m src.damage_probe --config /workspace/e2_wired.yaml
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.damage import damage_seed_sets, make_recurring_schedule
from src.evolve import build_model, make_eval_fn
from src.metrics import target_alpha_mask
from src.rollout import dense_lesion_masks, grow_from_seed
from src.targets import load_target
from src.controller import zero_params


def build_schedules_custom(seeds, T, interval, shape, radius, kind):
    """Recurring multi/single-disc damage at the given radius/kind."""
    scheds = []
    for s in seeds:
        times, masks = make_recurring_schedule(s, T=T, interval=interval, shape=shape)
        # make_recurring_schedule produces disc lesions by default at some radius;
        # we override masks to the requested kind/radius via sample_lesion
        from src.damage import sample_lesion
        new_masks = []
        rs = jax.random.key(s)
        for t in times:
            rs, sub = jax.random.split(rs)
            m = sample_lesion(sub, kind, radius, shape=shape)
            new_masks.append(np.asarray(m))
        if new_masks:
            masks_arr = np.stack(new_masks)
        else:
            masks_arr = np.zeros((0, *shape), dtype=bool)
        dense = dense_lesion_masks(times, masks_arr, T, shape)
        scheds.append(dense)
    return jnp.stack(scheds)


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
    states = jnp.concatenate([state0[None]] * len(train_seeds))
    rkeys = jax.random.split(jax.random.key(cfg["seed"]), len(train_seeds))
    params_b = jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * len(train_seeds)), neutral)

    print(f"damage-regime probe: neutral closed_loop, {len(train_seeds)} seeds, T={T}")
    print(f"goal: find schedule where neutral Hamming >> 0.01 (currently 0.0023)")
    print("-" * 75)
    print(f"{'radius':>6} {'kind':>10} {'interval':>8} {'mean_H':>8} {'max_H':>8} {'mean_alive':>10}  verdict")
    print("-" * 75)

    # grid: radius x kind x interval
    grid = []
    for radius in [8, 12, 16, 20]:
        for kind in ["single", "multi"]:
            for interval in [250, 150, 100]:
                grid.append((radius, kind, interval))

    t0 = time.time()
    results = []
    for radius, kind, interval in grid:
        schedules = build_schedules_custom(train_seeds, T, interval, shape, radius, kind)
        eval_fn = make_eval_fn(cfg, target_mask, "closed_loop", K)
        _, (hamming, alive) = eval_fn(cs, states, params_b, schedules, rkeys)
        mean_h = float(hamming.mean())
        max_h = float(hamming.max())
        mean_a = float(alive.mean())
        if mean_h > 0.1 and mean_a > 0.03:
            verdict = "★★ HARD+ALIVE"
        elif mean_h > 0.1:
            verdict = "★ hard (low alive?)"
        elif mean_h > 0.02:
            verdict = "~ marginal"
        else:
            verdict = "  too easy"
        results.append((radius, kind, interval, mean_h, max_h, mean_a, verdict))
        print(f"{radius:>6} {kind:>10} {interval:>8} {mean_h:>8.4f} {max_h:>8.4f} {mean_a:>10.3f}  {verdict}")

    print("-" * 75)
    print(f"probe took {time.time()-t0:.0f}s for {len(grid)} schedules")
    print()
    hard = [r for r in results if "HARD+ALIVE" in r[6]]
    if hard:
        # cheapest = largest interval (least frequent damage) among the hard ones
        cheapest = max(hard, key=lambda r: r[2])
        print(f"RECOMMENDED regime (hardest that keeps lizard alive, least frequent):")
        print(f"  radius={cheapest[0]}, kind={cheapest[1]}, interval={cheapest[2]}")
        print(f"  -> neutral Hamming {cheapest[3]:.4f}, alive {cheapest[5]:.3f}")
    else:
        print("No regime pushed neutral past 0.1 while keeping alive>0.03.")
        print("Best available (highest Hamming with alive>0.03):")
        candidates = [r for r in results if r[5] > 0.03]
        if candidates:
            best = max(candidates, key=lambda r: r[3])
            print(f"  radius={best[0]}, kind={best[1]}, interval={best[2]} -> H {best[3]:.4f}, alive {best[5]:.3f}")


if __name__ == "__main__":
    main()
