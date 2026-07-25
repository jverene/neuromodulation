"""Sensitivity probe (pre-E2 fork decision, PRD §10 M2-adjacent).

Question: does the channel-aware parent's controller input actually carry a
learnable signal? The parent was trained with the controller frozen at neutral
m=0 (PRD §5), so its K modulator-input weights got ZERO gradient. If they're
dead, CMA-ES in E2 will spin 300 generations finding nothing.

Probe: run the constant-tonic grid [-1, -0.5, 0, +0.5, +1] over the TRAIN damage
seeds at full T=2000 and report mean Hamming per level. If the curve is flat
across levels -> signal is dead -> enable `train.mod_noise` and retrain the
parent before E2. If it varies -> green light for E2.

Reuses evolve.py's make_eval_fn + build_schedules + the wired E2 config, so it
exercises the exact same code path E2 will. ~10 min on A100.

Usage: python -m src.e2_probe --config /workspace/e2_wired.yaml
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.damage import damage_seed_sets, make_recurring_schedule
from src.evolve import build_model, build_schedules, make_eval_fn
from src.metrics import target_alpha_mask
from src.nca import model_from_config
from src.rollout import dense_lesion_masks, grow_from_seed
from src.targets import load_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--levels", default="-1.0,-0.5,0.0,0.5,1.0",
                        help="comma-separated tonic levels to probe")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg["rollout"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    K = rc["K"]

    cs = build_model(cfg, K, cfg["model"].get("parent_params"))
    state0 = grow_from_seed(cs, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    target_mask = target_alpha_mask(load_target(**cfg["target"]))

    train_seeds, _ = damage_seed_sets(cfg["evolve"]["train_seeds"], cfg["evolve"]["test_seeds"])
    schedules = build_schedules(train_seeds, rc["T"], rc["lesion_interval"], shape)
    states = jnp.concatenate([state0[None]] * schedules.shape[0])
    rkeys = jax.random.split(jax.random.key(cfg["seed"]), schedules.shape[0])

    levels = [float(x) for x in args.levels.split(",")]
    print(f"sensitivity probe: {len(schedules)} train damage seeds, T={rc['T']}, K={K}")
    print(f"tonic levels: {levels}")
    print("-" * 50)
    results = []
    t0 = time.time()
    for c in levels:
        eval_fn = make_eval_fn(cfg, target_mask, "constant", K, const_level=jnp.full((K,), c))
        _, (hamming, alive) = eval_fn(cs, states, None, schedules, rkeys)
        mean_h = float(hamming.mean())
        mean_alive = float(alive.mean())
        results.append((c, mean_h, mean_alive))
        print(f"  tonic c={c:+.2f} -> mean hamming {mean_h:.4f}  alive {mean_alive:.3f}")
    print("-" * 50)

    hams = np.array([r[1] for r in results])
    spread = float(hams.max() - hams.min())
    rel_spread = spread / float(hams.mean()) if hams.mean() > 0 else 0.0
    best_c = levels[int(np.argmin(hams))]
    neutral = results[[i for i, r in enumerate(results) if abs(r[0]) < 1e-9][0]][1] if any(abs(r[0]) < 1e-9 for r in results) else None

    print(f"hamming range across levels: {hams.min():.4f} -> {hams.max():.4f}  (spread {spread:.4f}, {rel_spread*100:.1f}% of mean)")
    print(f"best tonic level: c={best_c:+.2f} (hamming {hams.min():.4f})")
    print(f"probe took {time.time()-t0:.0f}s")
    print()
    if rel_spread < 0.02:
        print("VERDICT: ⚠ FLAT (<2% spread) — channel signal looks dead.")
        print("  -> Enable train.mod_noise and retrain the parent before E2 (PRD §5 escape hatch).")
    elif rel_spread < 0.05:
        print("VERDICT: ~ WEAK signal (2-5% spread). E2 will work but may be slow; proceed with caution.")
    else:
        print("VERDICT: ✅ CLEAR signal (>5% spread). Green light for E2 evolution.")


if __name__ == "__main__":
    main()
