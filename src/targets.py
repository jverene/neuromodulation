"""Emoji loading, padding, target tensor (PRD §3).

Target: 40x40 RGBA emoji, alpha-premultiplied (Mordvintsev et al. 2020),
zero-padded to a 96x96 canvas. RGBA is stored in the LAST 4 channels of the
NCA state (CAX convention; alpha = channel -1).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import PIL.Image

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def get_emoji_pil(emoji: str) -> PIL.Image.Image:
    """Load an emoji as RGBA PIL image, caching the PNG under assets/.

    cax.utils.get_emoji downloads from the Noto emoji repo on every call;
    the cache keeps repeated runs (and offline work) fast.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    cache = ASSETS_DIR / f"emoji_u{ord(emoji):x}.png"
    if cache.exists():
        return PIL.Image.open(cache).convert("RGBA")
    from cax.utils import get_emoji

    img = get_emoji(emoji).convert("RGBA")
    img.save(cache)
    return img


def load_target(emoji: str = "\U0001f98e", size: int = 40, canvas: int = 96) -> jnp.ndarray:
    """Build the target tensor: (canvas, canvas, 4) float32 in [0, 1], premultiplied alpha."""
    img = get_emoji_pil(emoji).resize((size, size), resample=PIL.Image.Resampling.LANCZOS)
    y = np.asarray(img, dtype=np.float32) / 255.0
    y[..., :3] *= y[..., 3:4]  # alpha premultiply
    pad = (canvas - size) // 2
    y = np.pad(y, ((pad, pad), (pad, pad), (0, 0)))
    return jnp.array(y)
