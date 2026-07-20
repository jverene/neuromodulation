"""Vmapped rollouts with recurring-damage schedules and closed-loop control (PRD §4-§7).

One JITted nnx.scan over T NCA steps. Carry: (grid state, ModulatorState,
alive-fraction at last decision, held static level, step index, controller
params). Per step: apply scheduled lesion -> controller decision every
`tau_decision` steps (mode-dependent) -> NCA step with the current modulator
level injected -> phasic decay. Per-step outputs: Hamming-to-target and
alive fraction (optionally also uint8 RGB frames for GIFs).

Modes: closed_loop (evolved controller) · static (goal embedding fixed at
t=0, Stovold-style) · constant (hand-set tonic level) · random (seeded
uniform schedule) · ablated (m clamped to 0). The no-modulation GNCA
condition is the same machinery on a K=0 model.

Also hosts the E1 lesion-sweep runner (PRD §8):
  python -m src.rollout --config configs/e1_lesion_sweep.yaml
"""

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax import nnx

from src.channels import ModulatorState, decision, level, step_decay
from src.controller import alive_fraction, apply as controller_apply, grid_stats
from src.damage import apply_lesion, sample_lesion
from src.metrics import hamming_to_target, target_alpha_mask

MODES = ("closed_loop", "static", "constant", "random", "ablated")


def dense_lesion_masks(times: np.ndarray, masks: np.ndarray, T: int, shape: tuple[int, int]) -> jnp.ndarray:
    """Scatter an event schedule into a dense (T, H, W) bool array (all-False off-event)."""
    dense = np.zeros((T, *shape), dtype=bool)
    for t, m in zip(times, masks):
        if 0 <= int(t) < T:
            dense[int(t)] |= m
    return jnp.array(dense)


def _make_step_fn(mode: str, *, target_mask, reference_mask, K: int, tau_decision: int,
                  const_level, emit_frames: bool):
    def step_fn(cs, carry, x):
        state, mod, alive_prev, m_static, t, cparams = carry
        lesion_mask, u_rand = x

        # 1. scheduled damage (all-False mask on non-event steps)
        state = apply_lesion(state, lesion_mask)

        # 2. modulator level for this step
        is_decision = (t % tau_decision) == 0
        if mode == "closed_loop":
            stats = grid_stats(state, alive_prev, reference_mask)
            u = controller_apply(cparams, stats)
            mod = jax.lax.cond(is_decision, lambda m: decision(m, u), lambda m: m, mod)
            m = level(mod)
            alive_prev = jnp.where(is_decision, alive_fraction(state), alive_prev)
        elif mode == "random":
            mod = jax.lax.cond(is_decision, lambda m: decision(m, u_rand), lambda m: m, mod)
            m = level(mod)
            alive_prev = jnp.where(is_decision, alive_fraction(state), alive_prev)
        elif mode == "static":
            # goal embedding computed once at t=0 from the initial stats, then held
            stats = grid_stats(state, alive_prev, reference_mask)
            u0 = controller_apply(cparams, stats)
            m_static = jnp.where(t == 0, u0, m_static)
            m = m_static
        elif mode == "constant":
            m = const_level
        elif mode == "ablated":
            m = jnp.zeros((K,))
        else:
            raise ValueError(f"unknown mode: {mode}")

        # 3. NCA step with modulator injection (K=0 -> no channels at all)
        state = cs._step(state, m if K > 0 else None)

        # 4. phasic decay
        mod = step_decay(mod)

        h = hamming_to_target(state, target_mask)
        a = alive_fraction(state)
        y = (h, a, cs.render(state)) if emit_frames else (h, a)
        return (state, mod, alive_prev, m_static, t + 1, cparams), y

    return step_fn


def run_rollout(
    cs,
    state0: jnp.ndarray,
    *,
    mode: str,
    T: int,
    lesion_masks: jnp.ndarray,
    target_mask: jnp.ndarray,
    reference_mask: jnp.ndarray | None = None,
    controller_params: dict | None = None,
    K: int = 3,
    tau_decision: int = 10,
    const_level: jnp.ndarray | None = None,
    rng_seed: int = 0,
    emit_frames: bool = False,
):
    """Run one closed-loop rollout. Returns (final_state, ys) with ys the
    per-step (hamming, alive) [+ frames] stacked outputs."""
    if reference_mask is None:
        reference_mask = state0[..., -1] > 0.5
    if const_level is None:
        const_level = jnp.zeros((K,))
    if controller_params is None:
        from src.controller import zero_params

        controller_params = zero_params(K=K)
    u_rand = jax.random.uniform(jax.random.key(rng_seed), (T, K), minval=-1.0, maxval=1.0)

    step_fn = _make_step_fn(
        mode,
        target_mask=target_mask,
        reference_mask=reference_mask,
        K=K,
        tau_decision=tau_decision,
        const_level=const_level,
        emit_frames=emit_frames,
    )
    carry0 = (
        state0,
        ModulatorState.zeros(K),
        alive_fraction(state0),
        jnp.zeros((K,)),
        jnp.array(0, jnp.int32),
        controller_params,
    )
    state_axes = nnx.StateAxes({nnx.Intermediate: 0, ...: nnx.Carry})
    (state_f, *_), ys = nnx.scan(
        step_fn, in_axes=(state_axes, nnx.Carry, 0), out_axes=(nnx.Carry, 0), length=T
    )(cs, carry0, (lesion_masks, u_rand))
    return state_f, ys


def grow_from_seed(cs, spatial_dims: tuple[int, int], channel_size: int, num_steps: int, seed: int = 0) -> jnp.ndarray:
    """Grow a pattern from a single seed cell (no damage, neutral input)."""
    from src.nca import seed_state

    state0 = seed_state(spatial_dims, channel_size)
    return cs(state0, num_steps=num_steps)


def _vmapped_rollout(mode: str, emit_frames: bool, rollout_kwargs: dict):
    """run_rollout vmapped over a leading batch of (state0, controller_params, lesion_masks).

    Any of state0/controller_params/lesion_masks may be shared (pass in_axes=None entries)."""

    def one(cs, state0, cparams, masks):
        return run_rollout(
            cs, state0, mode=mode, controller_params=cparams, lesion_masks=masks,
            emit_frames=emit_frames, **rollout_kwargs,
        )

    return one


def batch_rollout(
    cs,
    states0: jnp.ndarray,
    params_batch,
    masks_batch: jnp.ndarray,
    *,
    mode: str,
    rollout_kwargs: dict,
    emit_frames: bool = False,
):
    """Vmap run_rollout over batch dims. states0 (B,...), params_batch pytree with
    leading B (or None for shared), masks_batch (B, T, H, W)."""
    state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
    one = _vmapped_rollout(mode, emit_frames, rollout_kwargs)
    b = states0.shape[0]
    fn = nnx.split_rngs(splits=b)(
        nnx.vmap(one, in_axes=(state_axes, 0, 0 if params_batch is not None else None, 0))
    )
    return fn(cs, states0, params_batch, masks_batch)


def run_lesion_sweep(cs, *, state0, target_mask, controller_params, cfg: dict, run_dir: Path) -> None:
    """E1 (PRD §8): lesion radius x {single, multi} x {intact, ablated}.

    Single lesion applied to the grown pattern at t=0; recovery is tracked via
    the Hamming trajectory over sweep.T steps. Writes trajectories.csv (long
    format) for figures.py. "intact" runs closed_loop with the given
    controller params (neutral zeros until an evolved controller exists);
    "ablated" clamps m=0 on the same channel-aware parent (PRD §5).
    """
    sw = cfg["sweep"]
    rc = cfg["rollout"]
    K = rc["K"]
    T = sw["T"]
    shape = state0.shape[:2]

    rows_meta, states, masks, params_map = [], [], [], []
    for radius in sw["radii"]:
        for kind in sw["kinds"]:
            for seed in range(sw["n_seeds"]):
                dmg_kind = "disc" if kind == "single" else "multi_disc"
                lesion = sample_lesion(
                    jax.random.key(cfg["seed"] + 7919 * seed + radius), dmg_kind, radius, shape=shape
                )
                dense = dense_lesion_masks(np.array([0]), np.asarray(lesion)[None], T, shape)
                for condition in sw["conditions"]:
                    rows_meta.append((radius, kind, condition, seed))
                    states.append(state0)
                    masks.append(dense)
                    params_map.append(controller_params if condition == "intact" else None)

    states = jnp.stack(states)
    masks = jnp.stack(masks)
    rollout_kwargs = dict(T=T, target_mask=target_mask, K=K, tau_decision=rc["tau_decision"],
                          rng_seed=cfg["seed"])

    # one vmapped call per condition (a batch mixes only same-mode rollouts)
    trajectories = {}
    for condition in sw["conditions"]:
        idx = [i for i, m in enumerate(rows_meta) if m[2] == condition]
        params_b = (
            jax.tree.map(lambda x: jnp.stack([jnp.array(x)] * len(idx)), controller_params)
            if condition == "intact" else None
        )
        mode = "closed_loop" if condition == "intact" else "ablated"
        _, (hamming, alive) = batch_rollout(
            cs, states[idx], params_b, masks[idx], mode=mode, rollout_kwargs=rollout_kwargs
        )
        trajectories[condition] = (np.asarray(hamming), np.asarray(alive), idx)

    stride = rc.get("traj_stride", 1)
    with open(run_dir / "trajectories.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["radius", "kind", "condition", "seed", "step", "hamming", "alive"])
        for condition, (hamming, alive, idx) in trajectories.items():
            for row, i in enumerate(idx):
                radius, kind, _, seed = rows_meta[i]
                for t in range(0, hamming.shape[1], stride):
                    writer.writerow([radius, kind, condition, seed, t,
                                     f"{hamming[row, t]:.6f}", f"{alive[row, t]:.6f}"])
    print(f"lesion sweep written -> {run_dir / 'trajectories.csv'}")


def main() -> None:
    from src.controller import zero_params
    from src.nca import model_from_config
    from src.targets import load_target
    from src.train import make_run_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.smoke:
        for section, overrides in cfg["smoke"].items():
            cfg.setdefault(section, {}).update(overrides)

    run_dir = make_run_dir(cfg, args.config)
    K = cfg["rollout"]["K"]
    cs = model_from_config(cfg, K, cfg["model"].get("parent_params"))

    controller_params = zero_params(K=K)
    if cfg.get("controller_params"):
        import pickle

        with open(cfg["controller_params"], "rb") as f:
            controller_params = pickle.load(f)

    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    state0 = grow_from_seed(cs, shape, cfg["model"]["channel_size"], cfg["rollout"]["grow_steps"],
                            seed=cfg["seed"])
    target_mask = target_alpha_mask(load_target(**cfg["target"]))
    run_lesion_sweep(cs, state0=state0, target_mask=target_mask,
                     controller_params=controller_params, cfg=cfg, run_dir=run_dir)


if __name__ == "__main__":
    main()
