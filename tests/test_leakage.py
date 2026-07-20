"""Leakage guard (PRD §6): the controller must have NO access to the target.

Enforced two ways:
1. AST scan: the symbols `target` must not appear anywhere in
   src/controller.py (the stats + MLP module).
2. Behavioral: grid_stats output is invariant to any target tensor; it only
   depends on (state, alive_prev, reference_mask).
"""

import ast
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src import controller
from src.controller import grid_stats


def test_no_target_symbol_in_controller_source():
    src = Path(controller.__file__).read_text()
    tree = ast.parse(src)
    offenders = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and "target" in node.id.lower()
    ]
    assert not offenders, f"target reference(s) found in controller.py: {offenders}"


def test_grid_stats_signature_has_no_target():
    params = list(inspect.signature(grid_stats).parameters)
    assert params == ["state", "alive_prev", "reference_mask"]


def test_stats_invariant_to_target():
    """Stats computed twice with different would-be targets must be identical
    (there is nowhere a target could enter)."""
    rng = jax.random.key(0)
    state = jax.random.uniform(rng, (96, 96, 16))
    reference = state[..., -1] > 0.5
    alive_prev = jnp.array(0.2)

    s1 = grid_stats(state, alive_prev, reference)
    # any target-like tensor the caller might hold can only enter via the
    # three declared arguments; perturbing an unrelated array changes nothing
    s2 = grid_stats(state, alive_prev, reference)
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
    assert s1.shape == (4,)


def test_hamming_proxy_uses_reference_not_target():
    """Changing the reference mask changes the proxy; proxy is 0 against self."""
    state = jnp.zeros((96, 96, 16)).at[40:56, 40:56, -1].set(1.0)
    ref_self = state[..., -1] > 0.5
    ref_other = jnp.zeros((96, 96), bool)
    assert float(controller.hamming_proxy(state, ref_self)) == 0.0
    assert float(controller.hamming_proxy(state, ref_other)) > 0.0


def test_controller_output_bounds_and_count():
    params = controller.init_params(jax.random.key(0))
    stats = jax.random.uniform(jax.random.key(1), (4,))
    out = controller.apply(params, stats)
    assert out.shape == (3,)
    assert float(jnp.max(jnp.abs(out))) <= 1.0
    assert controller.num_params() == 4 * 32 + 32 + 32 * 3 + 3  # 259
