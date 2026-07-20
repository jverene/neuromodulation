"""Evosax CMA-ES over controller params + E2 condition evaluation (PRD §7-§8).

Usage:
  python -m src.evolve --config configs/e2_closedloop.yaml            # evolve + evaluate
  python -m src.evolve --config configs/e2_closedloop.yaml --smoke    # tiny CPU check

Fitness = mean Hamming(bin alpha, target) over T=2000 with recurring damage
on the train damage-seed set (Evosax MINIMIZES, so no sign flip). The held-out
damage-seed set is always reported at the end (PRD §12). All conditions share
the same channel-aware parent model (PRD §5); the no-modulation condition runs
the K=0 E0 baseline model.
"""

import argparse
import csv
import pickle
import time
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import nnx

from evosax.algorithms import CMA_ES

from src.controller import num_params as controller_num_params
from src.controller import zero_params
from src.damage import damage_seed_sets, make_recurring_schedule
from src.metrics import auc, repair_half_life, survival, target_alpha_mask
from src.nca import GrowingNCA, load_params, model_from_config
from src.rollout import batch_rollout, dense_lesion_masks, grow_from_seed, run_rollout
from src.targets import load_target
from src.train import make_run_dir


def build_model(cfg: dict, num_mod_channels: int, params_path: str | None) -> GrowingNCA:
    return model_from_config(cfg, num_mod_channels, params_path)


def build_schedules(seeds: list[int], T: int, interval: int, shape: tuple[int, int]) -> jnp.ndarray:
    """(S, T, H, W) dense lesion masks, one schedule per damage seed."""
    return jnp.stack(
        [
            dense_lesion_masks(*make_recurring_schedule(s, T=T, interval=interval, shape=shape), T, shape)
            for s in seeds
        ]
    )


def make_fitness(cfg: dict, target_mask: jnp.ndarray):
    """Vmapped fitness: mean Hamming over the rollout (lower is better).

    Optional alive-fraction floor guards against the controller gaming
    Hamming by suppressing alpha (PRD §12); 0 disables.
    """
    rc = cfg["rollout"]
    alive_floor = cfg["evolve"].get("alive_floor", 0.0)
    rollout_kwargs = dict(
        T=rc["T"], target_mask=target_mask, K=rc["K"], tau_decision=rc["tau_decision"],
    )

    @nnx.jit
    def fitness_batch(cs, states, params_pop, masks_pop, rkeys):
        _, (hamming, alive) = batch_rollout(
            cs, states, params_pop, masks_pop, rkeys, mode="closed_loop", rollout_kwargs=rollout_kwargs
        )
        fit = hamming.mean(axis=1)
        if alive_floor > 0:
            fit = fit + 10.0 * jnp.mean(jnp.relu(alive_floor - alive), axis=1)
        return fit

    return fitness_batch


def evolve(cs, state0, schedules: jnp.ndarray, cfg: dict, run_dir: Path) -> dict:
    """CMA-ES ask/tell loop (PRD §7). Returns {'params': pytree, 'fitness': float}."""
    ec = cfg["evolve"]
    P, S = ec["population_size"], schedules.shape[0]
    K = cfg["rollout"]["K"]
    print(f"controller params: {controller_num_params(K=K)} | pop {P} x {S} damage seeds")

    solution = zero_params(K=K)  # neutral controller as init mean
    es = CMA_ES(population_size=P, solution=solution)
    es_params = es.default_params.replace(std_init=ec["std_init"])
    key = jax.random.key(cfg["seed"])
    key, k_init = jax.random.split(key)
    es_state = es.init(k_init, solution, es_params)

    fitness_batch = make_fitness(cfg, target_alpha_mask(load_target(**cfg["target"])))
    states_tiled = jnp.concatenate([state0[None]] * (P * S))
    masks_tiled = jnp.tile(schedules, (P, 1, 1, 1))  # block k = all S schedules
    rkeys_tiled = jax.random.split(jax.random.key(cfg["seed"]), P * S)

    best = {"params": None, "fitness": np.inf}
    t0 = time.time()
    with open(run_dir / "evolve_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", "best_in_gen", "best_so_far", "mean_fitness", "elapsed_sec"])
        for gen in range(ec["num_generations"]):
            key, k_ask, k_tell = jax.random.split(key, 3)
            pop, es_state = es.ask(k_ask, es_state, es_params)
            pop_tiled = jax.tree.map(lambda x: jnp.repeat(x, S, axis=0), pop)  # member-major
            fit = fitness_batch(cs, states_tiled, pop_tiled, masks_tiled, rkeys_tiled)
            fit = fit.reshape(P, S).mean(axis=1)
            es_state, _ = es.tell(k_tell, pop, fit, es_state, es_params)

            i = int(jnp.argmin(fit))
            if float(fit[i]) < best["fitness"]:
                best = {"params": jax.tree.map(lambda x: np.asarray(x[i]), pop), "fitness": float(fit[i])}
            writer.writerow([gen, f"{float(fit[i]):.6f}", f"{best['fitness']:.6f}",
                             f"{float(jnp.mean(fit)):.6f}", f"{time.time() - t0:.1f}"])
            f.flush()
            if gen % ec["log_interval"] == 0:
                print(f"gen {gen:4d}  best_in_gen {float(fit[i]):.4f}  best {best['fitness']:.4f}  "
                      f"elapsed {time.time() - t0:.0f}s", flush=True)
            if (gen + 1) % ec["checkpoint_interval"] == 0:
                with open(run_dir / "evolve_checkpoint.pkl", "wb") as ck:
                    pickle.dump({"gen": gen, "best": best}, ck)

    with open(run_dir / "controller_params.pkl", "wb") as f:
        pickle.dump(best["params"], f)
    print(f"evolution done in {time.time() - t0:.0f}s; best fitness {best['fitness']:.4f}")
    return best


def make_eval_fn(cfg: dict, target_mask: jnp.ndarray, mode: str, K: int, const_level=None):
    rc = cfg["rollout"]
    rollout_kwargs = dict(
        T=rc["T"], target_mask=target_mask, K=K, tau_decision=rc["tau_decision"],
        const_level=const_level,
    )

    @nnx.jit
    def eval_batch(cs, states, params, masks, rkeys):
        return batch_rollout(cs, states, params, masks, rkeys, mode=mode, rollout_kwargs=rollout_kwargs)

    return eval_batch


def save_gif(cs, state0, masks, *, mode, cfg, target_mask, K, params=None, const_level=None, path, seed=0):
    rc = cfg["rollout"]
    _, ys = run_rollout(
        cs, state0, mode=mode, T=rc["T"], lesion_masks=masks, target_mask=target_mask,
        controller_params=params, K=K, tau_decision=rc["tau_decision"],
        const_level=const_level, rand_key=jax.random.key(seed), emit_frames=True,
    )
    frames = np.asarray(ys[2])[:: rc.get("gif_frame_every", 10)]
    imageio.mimsave(path, list(frames), fps=cfg["eval"]["gif_fps"])


def evaluate_conditions(cs, cs_baseline, best_params, schedules_train, schedules_test,
                        lesion_times, cfg: dict, run_dir: Path, state0) -> None:
    """E2: 5 conditions x recurring damage -> trajectories + summary metrics (PRD §8)."""
    ev = cfg["eval"]
    rc = cfg["rollout"]
    K = rc["K"]
    target_mask = target_alpha_mask(load_target(**cfg["target"]))
    n_seeds = ev["num_condition_seeds"]
    S_test = schedules_test.shape[0]

    # --- constant tonic: hand grid search on the train damage seeds (PRD §7)
    best_c, best_fit = None, np.inf
    for c in ev["const_grid"]:
        const_level = jnp.full((K,), c)
        eval_fn = make_eval_fn(cfg, target_mask, "constant", K, const_level=const_level)
        states = jnp.concatenate([state0[None]] * schedules_train.shape[0])
        rkeys = jax.random.split(jax.random.key(cfg["seed"]), schedules_train.shape[0])
        _, (hamming, _) = eval_fn(cs, states, None, schedules_train, rkeys)
        fit = float(hamming.mean())
        print(f"  constant tonic c={c:+.2f} -> mean hamming {fit:.4f}")
        if fit < best_fit:
            best_c, best_fit = c, fit
    print(f"constant tonic grid search: best c = {best_c:+.2f}")

    conditions = [
        ("closed_loop", cs, K, best_params, None),
        ("static", cs, K, best_params, None),
        ("constant", cs, K, None, jnp.full((K,), best_c)),
        ("random", cs, K, None, None),
        ("no_modulation", cs_baseline, 0, None, None),
    ]

    traj_path = run_dir / "trajectories.csv"
    with open(traj_path, "w", newline="") as ft, open(run_dir / "metrics.csv", "w", newline="") as fs:
        wtraj = csv.writer(ft)
        wtraj.writerow(["condition", "condition_seed", "damage_seed", "step", "hamming", "alive"])
        wsum = csv.writer(fs)
        wsum.writerow(["condition", "condition_seed", "damage_seed", "final_hamming",
                       "auc", "half_life", "survived"])
        for name, cs_c, K_c, params, const_level in conditions:
            mode = "ablated" if name == "no_modulation" else name
            eval_fn = make_eval_fn(cfg, target_mask, mode, K_c, const_level=const_level)
            for cond_seed in range(n_seeds):
                states = jnp.concatenate([state0[None]] * S_test)
                params_b = (
                    jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * S_test), best_params)
                    if params is not None else None
                )
                rkeys = jax.random.split(jax.random.key(cfg["seed"] + 1000 * cond_seed), S_test)
                _, (hamming, alive) = eval_fn(cs_c, states, params_b, schedules_test, rkeys)
                hamming, alive = np.asarray(hamming), np.asarray(alive)
                for d in range(S_test):
                    hl = repair_half_life(hamming[d], lesion_times, window=ev["half_life_window"])
                    wsum.writerow([name, cond_seed, d, f"{hamming[d, -1]:.6f}",
                                   f"{auc(hamming[d]):.6f}", f"{hl:.1f}",
                                   int(hamming[d, -1] < ev["survival_eps"])])
                    stride = rc.get("traj_stride", 10)
                    for t in range(0, hamming.shape[1], stride):
                        wtraj.writerow([name, cond_seed, d, t, f"{hamming[d, t]:.6f}", f"{alive[d, t]:.6f}"])
                print(f"  {name} seed {cond_seed}: final hamming {hamming[:, -1].mean():.4f}")
            # one GIF per condition (condition seed 0, damage seed 0)
            save_gif(cs_c, state0, schedules_test[0], mode=mode, cfg=cfg, target_mask=target_mask,
                     K=K_c, params=best_params if params is not None else None,
                     const_level=const_level, path=run_dir / f"rollout_{name}.gif",
                     seed=cfg["seed"])
    print(f"condition evaluation written -> {traj_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny CPU end-to-end check")
    parser.add_argument("--eval-only", default=None, help="path to controller_params.pkl; skip evolution")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        for section, overrides in cfg["smoke"].items():
            cfg.setdefault(section, {}).update(overrides)

    run_dir = make_run_dir(cfg, args.config)
    rc = cfg["rollout"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])

    # channel-aware parent (K=3) and K=0 baseline for the no-modulation condition
    cs = build_model(cfg, rc["K"], cfg["model"].get("parent_params"))
    cs_baseline = build_model(cfg, 0, cfg["model"].get("baseline_params"))

    state0 = grow_from_seed(cs, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])

    n_train, n_test = cfg["evolve"]["train_seeds"], cfg["evolve"]["test_seeds"]
    train_seeds, test_seeds = damage_seed_sets(n_train, n_test)
    schedules_train = build_schedules(train_seeds, rc["T"], rc["lesion_interval"], shape)
    schedules_test = build_schedules(test_seeds, rc["T"], rc["lesion_interval"], shape)
    lesion_times = np.arange(rc["lesion_interval"], rc["T"], rc["lesion_interval"])

    if args.eval_only:
        with open(args.eval_only, "rb") as f:
            best = {"params": pickle.load(f), "fitness": float("nan")}
    else:
        best = evolve(cs, state0, schedules_train, cfg, run_dir)

    evaluate_conditions(cs, cs_baseline, best["params"], schedules_train, schedules_test,
                        lesion_times, cfg, run_dir, state0)
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
