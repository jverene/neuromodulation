"""E0 growth training loop (pool + damage) — PRD §3.

Usage: python -m src.train --config configs/e0_baseline.yaml [--steps N]

Pool-based training following Mordvintsev et al. 2020 / the CAX growing-NCA
example, with PRD additions: mid-episode random-square damage, param
checkpoints every 500 steps, CSV logging, run dir results/<ts>/.
"""

import argparse
import csv
import shutil
import time
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import nnx

from cax.nn.pool import Pool

from src.nca import GrowingNCA, load_params, save_params, seed_state
from src.targets import load_target


def make_run_dir(config: dict, config_path: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / f"{ts}_{config.get('run_name', 'run')}"
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir


def random_square_erase(state: jax.Array, key: jax.Array, prob: float, min_side: int, max_side: int) -> jax.Array:
    """Zero a random axis-aligned square with probability `prob` (training damage, PRD §3)."""
    h, w, _ = state.shape
    k_do, k_side, k_y, k_x = jax.random.split(key, 4)
    do = jax.random.uniform(k_do, ()) < prob
    side = jax.random.randint(k_side, (), min_side, max_side + 1)
    y0 = jax.random.randint(k_y, (), 0, h - side + 1)
    x0 = jax.random.randint(k_x, (), 0, w - side + 1)
    ys = jnp.arange(h)[:, None]
    xs = jnp.arange(w)[None, :]
    square = (ys >= y0) & (ys < y0 + side) & (xs >= x0) & (xs < x0 + side)
    return jnp.where((square & do)[..., None], 0.0, state)


def make_train_step(target: jax.Array, cfg: dict):
    """Build the jitted train step for a given target/config (pool + damage, PRD §3)."""
    tc = cfg["train"]
    batch_size = tc["batch_size"]
    n_pre = tc["num_steps"] // 2  # mid-episode damage happens at this boundary
    n_post = tc["num_steps"] - n_pre
    channel_size = cfg["model"]["channel_size"]
    update_params = nnx.All(nnx.Param, nnx.PathContains("update"))
    state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})

    def mse(state):
        return jnp.mean(jnp.square(state[..., -4:] - target))

    num_mod = cfg["model"]["num_mod_channels"]
    mod_noise = tc.get("mod_noise", False)

    def rollout(cs, state, key):
        input_pre = input_post = None
        if mod_noise and num_mod > 0:
            # Escape hatch (PRD §5): train with random modulator inputs so the
            # channel weights are exercised. Default off (frozen neutral m=0).
            key, k1, k2 = jax.random.split(key, 3)
            input_pre = jax.random.uniform(k1, (num_mod,), minval=-1.0, maxval=1.0)
            input_post = jax.random.uniform(k2, (num_mod,), minval=-1.0, maxval=1.0)
        state = cs(state, input_pre, num_steps=n_pre)
        state = random_square_erase(state, key, tc["damage_prob"], tc["damage_min_side"], tc["damage_max_side"])
        return cs(state, input_post, num_steps=n_post, sow=True)

    @nnx.jit
    def loss_fn(cs, state, key):
        dmg_key, idx_key = jax.random.split(key)
        nnx.split_rngs(splits=batch_size)(
            nnx.vmap(rollout, in_axes=(state_axes, 0, 0))
        )(cs, state, jax.random.split(dmg_key, batch_size))
        states = nnx.pop(cs, nnx.Intermediate).state[0]  # (B, n_post, H, W, C)
        idx = jax.random.randint(idx_key, (batch_size,), n_post // 2, n_post)
        final = states[jnp.arange(batch_size), idx]
        loss = mse(final)
        return loss, final

    @nnx.jit
    def train_step(cs, optimizer, pool, key):
        sample_key, loss_key = jax.random.split(key)

        # Sample from pool (with replacement), worst in batch -> fresh seed.
        pool_idx, batch = pool.sample(sample_key, batch_size=batch_size)
        current = batch["state"]
        order = jnp.argsort(jax.vmap(mse)(current), descending=True)
        pool_idx = pool_idx[order]
        current = current[order]
        current = current.at[0].set(seed_state(current.shape[1:3], channel_size))

        (loss, current), grad = nnx.value_and_grad(
            loss_fn, has_aux=True, argnums=nnx.DiffState(0, update_params)
        )(cs, current, loss_key)
        optimizer.update(cs, grad)

        pool = pool.update(pool_idx, {"state": current})
        return loss, pool

    return train_step


def sow_rollout(cs, state_init: jax.Array, num_steps: int) -> jax.Array:
    """Run the NCA on a batch of initial states, returning stacked states (B, T, H, W, C)."""
    n = state_init.shape[0]
    state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
    nnx.split_rngs(splits=n)(
        nnx.vmap(lambda cs, s: cs(s, num_steps=num_steps, sow=True), in_axes=(state_axes, 0))
    )(cs, state_init)
    return nnx.pop(cs, nnx.Intermediate).state[0]


def render_states(cs, states: jax.Array) -> np.ndarray:
    """Render (B, T, H, W, C) states to uint8 RGB frames (B, T, H, W, 3)."""
    b, t = states.shape[:2]
    flat = states.reshape(b * t, *states.shape[2:])
    frames = nnx.vmap(lambda cs, s: cs.render(s), in_axes=(None, 0))(cs, flat)
    return np.asarray(frames).reshape(b, t, *frames.shape[1:])


def center_square_erase(state: jax.Array, side: int) -> jax.Array:
    h, w = state.shape[:2]
    y0, x0 = (h - side) // 2, (w - side) // 2
    mask = jnp.zeros((h, w), bool).at[y0 : y0 + side, x0 : x0 + side].set(True)
    return jnp.where(mask[..., None], 0.0, state)


def save_gifs(cs, cfg: dict, run_dir: Path) -> None:
    """Growth-from-seed GIF and centered-lesion recovery GIF (PRD §3 acceptance)."""
    canvas = cfg["target"]["canvas"]
    channel_size = cfg["model"]["channel_size"]
    n_eval = cfg["eval"]["num_steps"]
    fps = cfg["eval"]["gif_fps"]
    n_show = 4

    state_init = jax.vmap(lambda _: seed_state((canvas, canvas), channel_size))(jnp.zeros(n_show))
    states = sow_rollout(cs, state_init, n_eval)
    frames = render_states(cs, states)
    tiled = [np.concatenate([frames[b, t] for b in range(n_show)], axis=1) for t in range(n_eval)]
    imageio.mimsave(run_dir / "growth.gif", tiled, fps=fps)

    # Recovery: grown pattern -> centered lesion -> regrow.
    lesion_side = cfg["eval"]["lesion_side"]
    grown = states[:, -1]
    lesioned = jax.vmap(lambda s: center_square_erase(s, lesion_side))(grown)
    rec_states = sow_rollout(cs, lesioned, n_eval)
    rec_frames = render_states(cs, rec_states)
    tiled = [np.concatenate([rec_frames[b, t] for b in range(n_show)], axis=1) for t in range(n_eval)]
    imageio.mimsave(run_dir / "recovery.gif", tiled, fps=fps)

    final_mse = float(jnp.mean(jnp.square(rec_states[:, -1, ..., -4:] - load_target(**cfg["target"]))))
    print(f"post-lesion final MSE: {final_mse:.4e}")


def take_snapshot(cs, cfg: dict, step: int, snap_dir: Path) -> None:
    """Render one tiled PNG of the current growth pattern at this training step.

    Grows `snapshot_num_show` seeds for `eval.num_steps` and tiles their final
    frames side-by-side into `snapshots/step_<NNNNNN>.png`. Cheap relative to a
    train step; called every `train.snapshot_interval` (default 50) steps so the
    whole run can be replayed as a 'watch it learn' time-lapse.
    """
    snap_dir.mkdir(parents=True, exist_ok=True)
    canvas = cfg["target"]["canvas"]
    channel_size = cfg["model"]["channel_size"]
    n_eval = cfg["eval"]["num_steps"]
    n_show = cfg["train"].get("snapshot_num_show", 4)

    state_init = jax.vmap(lambda _: seed_state((canvas, canvas), channel_size))(jnp.zeros(n_show))
    states = sow_rollout(cs, state_init, n_eval)
    frames = render_states(cs, states)
    tiled = np.concatenate([frames[b, -1] for b in range(n_show)], axis=1)  # final frame, side-by-side
    imageio.imwrite(snap_dir / f"step_{step:06d}.png", tiled)


def assemble_progress_gif(snap_dir: Path, run_dir: Path, fps: int) -> None:
    """Concatenate all snapshots (sorted by step) into a progress.gif time-lapse."""
    pngs = sorted(snap_dir.glob("step_*.png"))
    if not pngs:
        return
    frames = [imageio.imread(p) for p in pngs]
    imageio.mimsave(run_dir / "progress.gif", frames, fps=fps)
    print(f"progress.gif: {len(frames)} snapshots")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=None, help="override train.num_train_steps (smoke runs)")
    parser.add_argument("--run-dir", default=None, help="reuse an existing run dir instead of creating one")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.steps is not None:
        cfg["train"]["num_train_steps"] = args.steps
    run_dir = Path(args.run_dir) if args.run_dir else make_run_dir(cfg, args.config)

    key = jax.random.key(cfg["seed"])
    target = load_target(**cfg["target"])
    cs = GrowingNCA(
        num_mod_channels=cfg["model"]["num_mod_channels"],
        channel_size=cfg["model"]["channel_size"],
        hidden_size=cfg["model"]["hidden_size"],
        cell_dropout_rate=cfg["model"]["cell_dropout_rate"],
        rngs=nnx.Rngs(cfg["seed"]),
    )
    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(cs, nnx.Param)))
    print(f"model params: {n_params}  |  run dir: {run_dir}")

    canvas = cfg["target"]["canvas"]
    state0 = jax.vmap(lambda _: seed_state((canvas, canvas), cfg["model"]["channel_size"]))(
        jnp.zeros(cfg["train"]["pool_size"])
    )
    pool = Pool.create({"state": state0})

    optimizer = nnx.Optimizer(
        cs,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg["train"]["learning_rate"])),
        wrt=nnx.All(nnx.Param, nnx.PathContains("update")),
    )
    train_step = make_train_step(target, cfg)

    tc = cfg["train"]
    snapshot_interval = tc.get("snapshot_interval", 50)
    snap_dir = run_dir / "snapshots"
    t0 = time.time()
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "elapsed_sec"])
        for step in range(tc["num_train_steps"]):
            key, subkey = jax.random.split(key)
            loss, pool = train_step(cs, optimizer, pool, subkey)
            # Per-step loss to CSV (continuous curve; ~8000 rows).
            writer.writerow([step, f"{float(loss):.6e}", f"{time.time() - t0:.1f}"])
            f.flush()
            if step % tc["print_interval"] == 0:
                print(f"step {step:6d}  loss {float(loss):.4e}  elapsed {time.time() - t0:.0f}s", flush=True)
            if (step + 1) % tc["checkpoint_interval"] == 0:
                save_params(cs, run_dir / f"params_step{step + 1:05d}.pkl")
            if step % snapshot_interval == 0 or step == tc["num_train_steps"] - 1:
                take_snapshot(cs, cfg, step, snap_dir)

    save_params(cs, run_dir / "params.pkl")
    save_gifs(cs, cfg, run_dir)
    assemble_progress_gif(snap_dir, run_dir, cfg["eval"]["gif_fps"])
    print(f"done in {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
