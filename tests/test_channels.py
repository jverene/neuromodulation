"""Tests for channels.py modulator dynamics (PRD §5)."""

import jax.numpy as jnp
import numpy as np

from src.channels import (
    ModulatorState,
    broadcast_map,
    decision,
    level,
    step_decay,
)


def test_tonic_ema_converges_to_constant_output():
    state = ModulatorState.zeros(3)
    u = jnp.array([1.0, -1.0, 0.5])
    for _ in range(200):
        state = decision(state, u)
    np.testing.assert_allclose(np.asarray(state.tonic), np.asarray(u), atol=1e-3)


def test_tonic_ema_math():
    state = ModulatorState.zeros(3)
    u = jnp.ones(3)
    state = decision(state, u)
    np.testing.assert_allclose(np.asarray(state.tonic), np.full(3, 0.05), rtol=1e-6)
    state = decision(state, u)
    np.testing.assert_allclose(
        np.asarray(state.tonic), np.full(3, 0.95 * 0.05 + 0.05), rtol=1e-6
    )


def test_phasic_spike_and_exp_decay():
    state = ModulatorState.zeros(3)
    u = jnp.array([1.0, 0.5, -0.5])
    state = decision(state, u)
    np.testing.assert_array_equal(np.asarray(state.phasic), np.asarray(u))  # spike
    for s in range(1, 21):
        state = step_decay(state)
        np.testing.assert_allclose(
            np.asarray(state.phasic), np.asarray(u) * np.exp(-s / 20.0), rtol=1e-5
        )


def test_level_clipped_to_unit_range():
    state = ModulatorState(tonic=jnp.full(3, 0.9), phasic=jnp.full(3, 0.9))
    np.testing.assert_array_equal(np.asarray(level(state)), np.ones(3))
    state = ModulatorState(tonic=jnp.full(3, -0.9), phasic=jnp.full(3, -0.9))
    np.testing.assert_array_equal(np.asarray(level(state)), -np.ones(3))


def test_neutral_zero():
    state = ModulatorState.zeros(3)
    np.testing.assert_array_equal(np.asarray(level(state)), np.zeros(3))


def test_broadcast_map():
    m = jnp.array([0.1, -0.2, 0.3])
    bmap = broadcast_map(m, (96, 96))
    assert bmap.shape == (96, 96, 3)
    np.testing.assert_array_equal(np.asarray(bmap[0, 0]), np.asarray(m))
    np.testing.assert_array_equal(np.asarray(bmap[95, 95]), np.asarray(m))
