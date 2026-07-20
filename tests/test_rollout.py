"""Smoke tests for rollout.py (PRD §4-§7)."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.controller import init_params, zero_params
from src.damage import disc_mask
from src.rollout import MODES, batch_rollout, dense_lesion_masks, run_rollout
from src.nca import GrowingNCA, seed_state

CANVAS = (48, 48)
K = 3
T = 30


def make_model(k=K, seed=0, active=True):
    cs = GrowingNCA(num_mod_channels=k, rngs=nnx.Rngs(seed))
    if active:
        # un-trained model has a zero-init final layer (dynamics frozen);
        # scramble it so rollouts actually do something in tests
        kernel = cs.update.layers[-1].kernel
        kernel[...] = jax.random.normal(jax.random.key(1), kernel[...].shape) * 0.05
    return cs


def make_inputs(t=T):
    state0 = seed_state(CANVAS).at[20:28, 20:28, -1].set(1.0)  # some alive mass
    target_mask = jnp.zeros(CANVAS, bool).at[20:28, 20:28].set(True)
    dense = np.zeros((t, *CANVAS), dtype=bool)
    dense[10] = np.asarray(disc_mask(CANVAS, 4, (24, 24)))
    return state0, target_mask, jnp.array(dense)


def run(mode, cs, state0, target_mask, dense, params, **kw):
    return run_rollout(
        cs, state0, mode=mode, T=T, lesion_masks=dense, target_mask=target_mask,
        controller_params=params, K=K, tau_decision=10, rng_seed=0, **kw,
    )


def test_all_modes_run_and_shapes():
    cs = make_model()
    state0, target_mask, dense = make_inputs()
    params = init_params(jax.random.key(0))
    for mode in MODES:
        state_f, ys = run(mode, cs, state0, target_mask, dense, params)
        hamming, alive = ys
        assert state_f.shape == (*CANVAS, 16)
        assert hamming.shape == (T,) and alive.shape == (T,)
        assert jnp.all(jnp.isfinite(hamming)) and jnp.all((hamming >= 0) & (hamming <= 1))
        assert jnp.all((alive >= 0) & (alive <= 1))


def test_determinism():
    state0, target_mask, dense = make_inputs()
    params = init_params(jax.random.key(0))
    _, (h1, _) = run("closed_loop", make_model(), state0, target_mask, dense, params)
    _, (h2, _) = run("closed_loop", make_model(), state0, target_mask, dense, params)
    np.testing.assert_array_equal(np.asarray(h1), np.asarray(h2))


def test_ablated_equals_zero_controller():
    state0, target_mask, dense = make_inputs()
    _, (h_abl, _) = run("ablated", make_model(), state0, target_mask, dense, zero_params())
    _, (h_zero, _) = run("closed_loop", make_model(), state0, target_mask, dense, zero_params())
    np.testing.assert_allclose(np.asarray(h_abl), np.asarray(h_zero), atol=1e-6)


def test_lesion_has_effect():
    cs = make_model()
    state0, target_mask, dense = make_inputs()
    _, (h_dmg, _) = run("ablated", cs, state0, target_mask, dense, zero_params())
    _, (h_clean, _) = run(
        "ablated", make_model(), state0, target_mask, jnp.zeros_like(dense), zero_params()
    )
    assert float(h_dmg[11]) != float(h_clean[11])  # damage at t=10 changes trajectory


def test_batch_rollout_vmap():
    cs = make_model()
    state0, target_mask, dense = make_inputs()
    b = 3
    states = jnp.stack([state0] * b)
    masks = jnp.stack([dense] * b)
    params = jax.tree.map(lambda x: jnp.stack([x] * b), init_params(jax.random.key(0)))
    finals, ys = batch_rollout(
        cs, states, params, masks, mode="closed_loop",
        rollout_kwargs=dict(T=T, target_mask=target_mask, K=K, tau_decision=10, rng_seed=0),
    )
    hamming, alive = ys
    assert finals.shape == (b, *CANVAS, 16)
    assert hamming.shape == (b, T)
    assert jnp.all(jnp.isfinite(hamming))
