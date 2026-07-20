"""Unit tests for metrics.py (PRD §9)."""

import jax.numpy as jnp
import numpy as np

from src.metrics import auc, hamming_to_target, repair_half_life, survival


def test_hamming_identical_and_opposite():
    target = jnp.zeros((96, 96, 4)).at[40:56, 40:56, -1].set(1.0)
    mask = target[..., -1] > 0.5
    assert float(hamming_to_target(target, mask)) == 0.0
    inv = 1.0 - target
    assert float(hamming_to_target(inv, mask)) == 1.0


def test_repair_half_life_synthetic():
    traj = np.full(2000, 0.1)
    traj[250:260] = 0.5          # lesion at t=250: jump to 0.5
    traj[260:360] = np.linspace(0.5, 0.1, 100)  # linear recovery over 100 steps
    hl = repair_half_life(traj, np.array([250]), window=250)
    # delta = 0.4, half-level = 0.3, crossed ~halfway through recovery
    assert 40.0 <= hl <= 60.0


def test_repair_half_life_unrecovered_counts_window():
    traj = np.full(2000, 0.1)
    traj[250:] = 0.6
    assert repair_half_life(traj, np.array([250]), window=250) == 250.0


def test_repair_half_life_no_effect_lesion_is_zero():
    traj = np.full(2000, 0.1)
    assert repair_half_life(traj, np.array([250]), window=250) == 0.0


def test_survival():
    trajs = np.stack([
        np.linspace(0.5, 0.05, 2000),   # survives (final < eps)
        np.linspace(0.5, 0.30, 2000),   # dies
        np.linspace(0.5, 0.09, 2000),   # survives
    ])
    assert survival(trajs, eps=0.1) == 2 / 3


def test_auc_constant():
    assert auc(np.full(2000, 0.25)) == 0.25
