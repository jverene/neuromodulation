"""Tests for the NCA model (PRD §3/§5) and a short pool-training smoke test."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from cax.nn.pool import Pool

from src.nca import GrowingNCA, seed_state
from src.train import make_train_step

CANVAS = (48, 48)


def build(k=0, seed=0):
    return GrowingNCA(num_mod_channels=k, rngs=nnx.Rngs(seed))


def param_count(cs):
    return sum(x.size for x in jax.tree.leaves(nnx.state(cs, nnx.Param)))


def test_builds_and_param_counts():
    # perception kernel 3*3*1*48 = 432 (fixed); update MLP (48+K)->128->16 + biases
    assert param_count(build(0)) == 432 + (48 * 128 + 128) + (128 * 16 + 16)
    assert param_count(build(3)) == 432 + (51 * 128 + 128) + (128 * 16 + 16)


def test_final_layer_zero_init():
    cs = build(0)
    kernel = cs.update.layers[-1].kernel[...]
    np.testing.assert_array_equal(np.asarray(kernel), np.zeros_like(np.asarray(kernel)))
    # ... so the very first update from a fresh model is a no-op modulo dropout
    state = seed_state(CANVAS)
    out = cs(state, num_steps=1)
    assert out.shape == state.shape
    np.testing.assert_allclose(np.asarray(out), np.asarray(state), atol=1e-6)


def test_step_shape_and_finite():
    cs = build(3)
    out = cs(seed_state(CANVAS), num_steps=4)
    assert out.shape == (*CANVAS, 16)
    assert jnp.all(jnp.isfinite(out))


def test_mod_channels_are_trailing_input_dims_and_zero_neutral():
    """With m=0 injected, scrambling the K modulator input weights must not
    change dynamics (zero input kills their contribution); also pins that the
    mod channels are the trailing K perception dims."""
    k = 3
    cs_a, cs_b = build(k, seed=0), build(k, seed=0)
    state = seed_state(CANVAS)

    kernel_b = cs_b.update.layers[0].kernel
    noise = jax.random.normal(jax.random.key(1), kernel_b[...][..., 48:, :].shape)
    kernel_b[..., 48:, :] = noise

    out_a = cs_a(state, num_steps=3)
    out_b = cs_b(state, num_steps=3)
    np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b), atol=1e-5)


def tiny_cfg():
    return {
        "train": {
            "batch_size": 4,
            "num_steps": 8,
            "damage_prob": 0.5,
            "damage_min_side": 4,
            "damage_max_side": 10,
        },
        "model": {"channel_size": 16, "num_mod_channels": 0},
    }


def synthetic_target():
    """Small RGBA disc target (no emoji download in unit tests)."""
    h, w = CANVAS
    ys, xs = jnp.mgrid[0:h, 0:w]
    disc = ((ys - h // 2) ** 2 + (xs - w // 2) ** 2 < 8**2).astype(jnp.float32)
    return jnp.stack([0.5 * disc, 0.8 * disc, 0.2 * disc, disc], axis=-1)


def test_train_step_smoke():
    """A few pool-training steps on a tiny canvas: loss finite, params update."""
    cfg = tiny_cfg()
    target = synthetic_target()
    cs = build(0, seed=0)
    before = np.asarray(nnx.state(cs, nnx.Param).to_pure_dict()["update"]["layers"][0]["kernel"]).copy()

    optimizer = nnx.Optimizer(
        cs,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(2e-3)),
        wrt=nnx.All(nnx.Param, nnx.PathContains("update")),
    )
    train_step = make_train_step(target, cfg)
    pool = Pool.create({"state": jax.vmap(lambda _: seed_state(CANVAS))(jnp.zeros(8))})

    key = jax.random.key(0)
    for _ in range(3):
        key, subkey = jax.random.split(key)
        loss, pool = train_step(cs, optimizer, pool, subkey)
    assert jnp.isfinite(loss)

    after = np.asarray(nnx.state(cs, nnx.Param).to_pure_dict()["update"]["layers"][0]["kernel"])
    assert not np.allclose(before, after)
