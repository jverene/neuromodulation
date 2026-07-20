"""Lesion generators and recurring-damage schedules (PRD §4).

Lesion kinds: disc, multi_disc(n), edge_disc. Radii r in {2, 4, 8, 16} px.
A lesion mask is a boolean (H, W) array; applying it zeroes all channels of
the masked cells. Recurring schedules draw lesion parameters from fixed,
seeded evaluation sets with a train/test split (holdout for the
generalization check, PRD §4/§12).
"""

import jax
import jax.numpy as jnp
import numpy as np

RADII = (2, 4, 8, 16)
KINDS = ("disc", "multi_disc")


def disc_mask(shape: tuple[int, int], radius: int, center: tuple[int, int]) -> jnp.ndarray:
    """Boolean disc of given radius centered at (y, x)."""
    h, w = shape
    ys, xs = jnp.mgrid[0:h, 0:w]
    cy, cx = center
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2


def _sample_center(rng: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    h, w = shape
    ky, kx = jax.random.split(rng)
    return jnp.array(
        [jax.random.randint(ky, (), 0, h), jax.random.randint(kx, (), 0, w)]
    )


def _edge_center(rng: jax.Array, shape: tuple[int, int]) -> jnp.ndarray:
    """Center constrained to the canvas border (edge_disc lesions, PRD §4)."""
    h, w = shape
    k_side, k_y, k_x = jax.random.split(rng, 3)
    side = jax.random.randint(k_side, (), 0, 4)
    yi = jax.random.randint(k_y, (), 0, h)
    xi = jax.random.randint(k_x, (), 0, w)
    y = jnp.where(side == 0, 0, jnp.where(side == 1, h - 1, yi))
    x = jnp.where(side == 2, 0, jnp.where(side == 3, w - 1, xi))
    return jnp.array([y, x])


def sample_lesion(
    rng: jax.Array,
    kind: str,
    radius: int,
    shape: tuple[int, int] = (96, 96),
    center: tuple[int, int] | None = None,
    n: int = 3,
) -> jnp.ndarray:
    """Sample a boolean lesion mask (H, W).

    kind: "disc" (single disc), "multi_disc" (n discs), "edge_disc" (disc
    centered on the canvas edge). `center` pins the disc position (eval);
    otherwise it is sampled uniformly.
    """
    if kind == "disc":
        if center is None:
            center = _sample_center(rng, shape)
        return disc_mask(shape, radius, center)
    if kind == "multi_disc":
        centers = jax.random.randint(rng, (n, 2), 0, jnp.array(shape))
        return jax.vmap(lambda c: disc_mask(shape, radius, c))(centers).any(axis=0)
    if kind == "edge_disc":
        return disc_mask(shape, radius, _edge_center(rng, shape))
    raise ValueError(f"unknown lesion kind: {kind}")


def apply_lesion(state: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Zero all channels of lesioned cells. state (..., H, W, C); mask (H, W) bool."""
    return jnp.where(mask[..., None], 0.0, state)


def make_recurring_schedule(
    seed: int,
    T: int = 2000,
    interval: int = 250,
    shape: tuple[int, int] = (96, 96),
    radii: tuple[int, ...] = RADII,
    kinds: tuple[str, ...] = KINDS,
    n_multi: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed recurring-damage schedule from one seed (PRD §4).

    Returns (times, masks): times (E,) int32 with the lesion steps (every
    `interval` steps), masks (E, H, W) bool. One seed fully determines the
    sequence; train/test seed sets are built from disjoint seed ranges.
    """
    times = np.arange(interval, T, interval, dtype=np.int32)
    rng = jax.random.key(seed)
    masks = []
    for _ in times:
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        kind = kinds[int(jax.random.randint(k1, (), 0, len(kinds)))]
        radius = radii[int(jax.random.randint(k2, (), 0, len(radii)))]
        masks.append(sample_lesion(k3, kind, int(radius), shape=shape, n=n_multi))
    return times, np.asarray(jnp.stack(masks))


def damage_seed_sets(n_train: int = 8, n_test: int = 8) -> tuple[list[int], list[int]]:
    """Disjoint train/test damage-seed sets (test = holdout, PRD §4)."""
    return list(range(0, n_train)), list(range(10_000, 10_000 + n_test))
