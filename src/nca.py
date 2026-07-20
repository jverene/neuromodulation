"""NCA model definition (perception + update) — PRD §3 and §5.

Built on CAX primitives:
- perception: ConvPerceive with fixed identity + Sobel-x + Sobel-y kernels
  (3 kernels x 16 channels = 48 perception dims; PRD's "48→128→16" MLP and
  "48+K=51" injection dims pin the kernel count at 3).
- update: NCAUpdate MLP (48+K) -> 128 (ReLU) -> 16, zero-init final layer,
  per-cell stochastic update mask p=0.5 (CAX cell dropout), alive-cell
  masking via 3x3 max-pool on alpha > 0.1 (handled inside NCAUpdate;
  alpha is the LAST channel).

Modulator injection (PRD §5): K global scalars are broadcast to a per-cell
map and concatenated to every cell's perception vector via NCAUpdate's
`input` argument, giving perception dim 48 + K.
"""

import pickle
from pathlib import Path

import jax.numpy as jnp
from flax import nnx
from jax import Array

from cax.core import ComplexSystem
from cax.core.perceive import ConvPerceive, grad_kernel, identity_kernel
from cax.core.update import NCAUpdate
from cax.utils import clip_and_uint8, rgba_to_rgb

NUM_KERNELS = 3  # identity + Sobel-x + Sobel-y


class GrowingNCA(ComplexSystem):
    """Growing NCA with K optional global modulator input channels (K=0 -> E0 baseline)."""

    def __init__(
        self,
        *,
        num_mod_channels: int = 0,
        channel_size: int = 16,
        hidden_size: int = 128,
        cell_dropout_rate: float = 0.5,
        rngs: nnx.Rngs,
    ):
        self.num_mod_channels = num_mod_channels
        self.channel_size = channel_size
        perception_size = NUM_KERNELS * channel_size
        self.perceive = ConvPerceive(
            channel_size=channel_size,
            perception_size=perception_size,
            feature_group_count=channel_size,
            rngs=rngs,
        )
        # NCAUpdate concatenates `input` to the perception before its first
        # layer, so its first layer must absorb the extra K channels.
        self.update = NCAUpdate(
            channel_size=channel_size,
            perception_size=perception_size + num_mod_channels,
            hidden_layer_sizes=(hidden_size,),
            cell_dropout_rate=cell_dropout_rate,
            zeros_init=True,
            rngs=rngs,
        )

        # Fixed (non-learned in practice: optimizer is restricted to `update`)
        # perception kernels: identity + Sobel-x + Sobel-y per channel.
        kernel = jnp.concatenate([identity_kernel(ndim=2), grad_kernel(ndim=2)], axis=-1)
        kernel = jnp.expand_dims(jnp.concatenate([kernel] * channel_size, axis=-1), axis=-2)
        self.perceive.conv.kernel[...] = kernel

    def _step(self, state: Array, input: Array | None = None, *, sow: bool = False) -> Array:
        perception = self.perceive(state)
        if self.num_mod_channels > 0:
            if input is None:
                input = jnp.zeros((self.num_mod_channels,), dtype=state.dtype)
            # Broadcast the global modulator vector (..., K) to a per-cell map.
            input = jnp.broadcast_to(
                input[..., None, None, :], (*state.shape[:-1], self.num_mod_channels)
            )
        else:
            input = None
        next_state = self.update(state, perception, input)

        if sow:
            self.sow(nnx.Intermediate, "state", next_state)

        return next_state

    @nnx.jit
    def render(self, state: Array) -> Array:
        """Render state to uint8 RGB (alpha composited over white)."""
        return clip_and_uint8(rgba_to_rgb(state[..., -4:]))

    @nnx.jit
    def render_rgba(self, state: Array) -> Array:
        """Render state to uint8 RGBA."""
        return clip_and_uint8(state[..., -4:])


def seed_state(spatial_dims: tuple[int, int], channel_size: int = 16) -> Array:
    """Single alive cell (alpha = 1) at the canvas center, all else zero."""
    state = jnp.zeros((*spatial_dims, channel_size))
    mid = tuple(s // 2 for s in spatial_dims)
    return state.at[mid[0], mid[1], -1].set(1.0)


def model_from_config(cfg: dict, num_mod_channels: int, params_path: str | Path | None) -> "GrowingNCA":
    """Build a GrowingNCA from a run config, optionally loading saved params."""
    cs = GrowingNCA(
        num_mod_channels=num_mod_channels,
        channel_size=cfg["model"]["channel_size"],
        hidden_size=cfg["model"]["hidden_size"],
        cell_dropout_rate=cfg["model"]["cell_dropout_rate"],
        rngs=nnx.Rngs(cfg["seed"]),
    )
    if params_path:
        load_params(cs, params_path)
    return cs


def save_params(cs: GrowingNCA, path: str | Path) -> None:
    """Pickle model params (pure-dict form) to disk."""
    pure = nnx.state(cs, nnx.Param).to_pure_dict()
    with open(path, "wb") as f:
        pickle.dump(pure, f)


def load_params(cs: GrowingNCA, path: str | Path) -> GrowingNCA:
    """Load params saved with save_params into an existing model (in place)."""
    with open(path, "rb") as f:
        pure = pickle.load(f)
    state = nnx.state(cs, nnx.Param)
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(cs, state)
    return cs
