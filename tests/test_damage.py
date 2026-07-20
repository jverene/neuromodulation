"""Tests for damage.py (PRD §4)."""

import jax
import jax.numpy as jnp
import numpy as np

from src.damage import (
    apply_lesion,
    damage_seed_sets,
    disc_mask,
    make_recurring_schedule,
    sample_lesion,
)

SHAPE = (96, 96)


def test_disc_mask_radius():
    mask = disc_mask(SHAPE, radius=8, center=(48, 48))
    assert mask[48, 48]  # center alive
    assert mask[48, 48 + 8]  # on the boundary
    assert not mask[48, 48 + 9]
    assert not mask[0, 0]
    area = int(mask.sum())
    assert abs(area - np.pi * 8**2) / (np.pi * 8**2) < 0.15


def test_multi_disc_count_and_determinism():
    k = jax.random.key(0)
    m1 = sample_lesion(k, "multi_disc", radius=4, shape=SHAPE, n=3)
    m2 = sample_lesion(k, "multi_disc", radius=4, shape=SHAPE, n=3)
    np.testing.assert_array_equal(np.asarray(m1), np.asarray(m2))  # seeded determinism
    single_area = np.pi * 4**2
    total = int(m1.sum())
    assert 0.5 * single_area < total <= 3 * single_area * 1.01  # overlaps only shrink area


def test_edge_disc_touches_border():
    for seed in range(5):
        mask = sample_lesion(jax.random.key(seed), "edge_disc", radius=8, shape=SHAPE)
        assert (
            mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
        ), "edge lesion must touch the canvas border"


def test_disc_pinned_center():
    mask = sample_lesion(jax.random.key(0), "disc", radius=2, shape=SHAPE, center=(10, 20))
    assert mask[10, 20]
    assert int(mask.sum()) == int(disc_mask(SHAPE, 2, (10, 20)).sum())


def test_apply_lesion_zeroes_inside_preserves_outside():
    state = jnp.ones((*SHAPE, 16))
    mask = disc_mask(SHAPE, radius=8, center=(48, 48))
    out = apply_lesion(state, mask)
    assert float(out[mask].max()) == 0.0
    assert float(out[~mask].min()) == 1.0


def test_recurring_schedule_times_and_shape():
    times, masks = make_recurring_schedule(seed=3, T=2000, interval=250, shape=SHAPE)
    np.testing.assert_array_equal(times, np.arange(250, 2000, 250))
    assert masks.shape == (len(times), *SHAPE)
    assert masks.dtype == bool
    # determinism
    times2, masks2 = make_recurring_schedule(seed=3, T=2000, interval=250, shape=SHAPE)
    np.testing.assert_array_equal(masks, masks2)
    # different seed -> different schedule
    _, masks3 = make_recurring_schedule(seed=4, T=2000, interval=250, shape=SHAPE)
    assert not np.array_equal(masks, masks3)


def test_damage_seed_split_disjoint():
    train, test = damage_seed_sets(8, 8)
    assert len(train) == 8 and len(test) == 8
    assert not set(train) & set(test)
