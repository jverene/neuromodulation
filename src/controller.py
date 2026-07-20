"""Controller MLP: grid summary stats -> modulator levels (PRD §6).

MLP 4 -> 32 (tanh) -> K (tanh, so outputs are in [-1, 1]); ~259 params.
Implemented functionally (param pytree of plain dicts) so Evosax can treat
the parameters as the solution pytree and jax.vmap can batch a population.

Inputs are target-free grid summary statistics (enforced by
tests/test_leakage.py):
1. alive-cell fraction (alpha > 0.1)
2. fraction killed in the last decision window (alive fraction then vs. now)
3. spatial entropy of alpha (normalized Shannon entropy of the alpha mass
   distribution over cells)
4. normalized Hamming-proxy: mismatch rate between binarized alpha (> 0.5)
   and a reference mask (the rollout's own t=0 pattern — never the target).
"""

import jax
import jax.numpy as jnp

NUM_STATS = 4
HIDDEN = 32


def init_params(rng: jax.Array, num_stats: int = NUM_STATS, hidden: int = HIDDEN, K: int = 3) -> dict:
    """Glorot-init params for the MLP; solution template for CMA-ES."""
    k1, k2 = jax.random.split(rng)
    w1 = jax.random.normal(k1, (num_stats, hidden)) * jnp.sqrt(2.0 / (num_stats + hidden))
    w2 = jax.random.normal(k2, (hidden, K)) * jnp.sqrt(2.0 / (hidden + K))
    return {
        "w1": w1,
        "b1": jnp.zeros(hidden),
        "w2": w2,
        "b2": jnp.zeros(K),
    }


def zero_params(num_stats: int = NUM_STATS, hidden: int = HIDDEN, K: int = 3) -> dict:
    """All-zero params -> controller outputs m = 0 (neutral); CMA-ES init mean."""
    return {
        "w1": jnp.zeros((num_stats, hidden)),
        "b1": jnp.zeros(hidden),
        "w2": jnp.zeros((hidden, K)),
        "b2": jnp.zeros(K),
    }


def num_params(num_stats: int = NUM_STATS, hidden: int = HIDDEN, K: int = 3) -> int:
    return num_stats * hidden + hidden + hidden * K + K


def apply(params: dict, stats: jnp.ndarray) -> jnp.ndarray:
    """stats (..., 4) -> modulator output (..., K) in [-1, 1]."""
    h = jnp.tanh(stats @ params["w1"] + params["b1"])
    return jnp.tanh(h @ params["w2"] + params["b2"])


def alive_fraction(state: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(state[..., -1] > 0.1)


def killed_fraction(alive_now: jnp.ndarray, alive_prev: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip((alive_prev - alive_now) / jnp.maximum(alive_prev, 1e-6), 0.0, 1.0)


def spatial_entropy(state: jnp.ndarray) -> jnp.ndarray:
    alpha = jnp.clip(state[..., -1], 0.0, 1.0)
    p = alpha / jnp.maximum(alpha.sum(), 1e-8)
    ent = -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0))
    return ent / jnp.log(float(alpha.size))


def hamming_proxy(state: jnp.ndarray, reference_mask: jnp.ndarray) -> jnp.ndarray:
    """Mismatch rate between binarized alpha and the (target-free) reference mask."""
    binalpha = state[..., -1] > 0.5
    return jnp.mean(binalpha != reference_mask)


def grid_stats(state: jnp.ndarray, alive_prev: jnp.ndarray, reference_mask: jnp.ndarray) -> jnp.ndarray:
    """The 4 controller inputs. NOTE: no target access (see test_leakage.py)."""
    a = alive_fraction(state)
    return jnp.stack(
        [
            a,
            killed_fraction(a, alive_prev),
            spatial_entropy(state),
            hamming_proxy(state, reference_mask),
        ]
    )
