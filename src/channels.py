"""Modulator state (tonic/phasic) + injection helpers (PRD §5).

K = 3 global modulator scalars, each with:
- a tonic component: EMA of controller outputs (alpha = 0.95, updated at
  controller decision steps),
- a phasic component: a spike set to the controller output at decision time,
  decaying exponentially (tau = 20 NCA steps).

The injected level is m = clip(tonic + phasic, -1, 1); nca._step broadcasts
it to every cell and concatenates it to the perception vector (48 + K = 51).
"""

from typing import NamedTuple

import jax.numpy as jnp

NUM_MOD_CHANNELS = 3
TONIC_ALPHA = 0.95
PHASIC_TAU = 20.0


class ModulatorState(NamedTuple):
    tonic: jnp.ndarray  # (K,)
    phasic: jnp.ndarray  # (K,)

    @classmethod
    def zeros(cls, K: int = NUM_MOD_CHANNELS) -> "ModulatorState":
        return cls(jnp.zeros(K), jnp.zeros(K))


def decision(state: ModulatorState, u: jnp.ndarray, alpha: float = TONIC_ALPHA) -> ModulatorState:
    """Controller decision: EMA-update tonic toward u; spike phasic to u."""
    return ModulatorState(alpha * state.tonic + (1.0 - alpha) * u, u)


def step_decay(state: ModulatorState, tau: float = PHASIC_TAU) -> ModulatorState:
    """Per-NCA-step exponential decay of the phasic component."""
    return ModulatorState(state.tonic, state.phasic * jnp.exp(-1.0 / tau))


def level(state: ModulatorState) -> jnp.ndarray:
    """Injected modulator level, clipped to [-1, 1]."""
    return jnp.clip(state.tonic + state.phasic, -1.0, 1.0)


def broadcast_map(m: jnp.ndarray, spatial_dims: tuple[int, int]) -> jnp.ndarray:
    """Broadcast K scalars to a per-cell (H, W, K) map (PRD §5 injection)."""
    h, w = spatial_dims
    return jnp.broadcast_to(m[None, None, :], (h, w, m.shape[-1]))
