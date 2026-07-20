"""Metrics: hamming, half-life, survival, AUC (PRD §9).

Hamming is computed on binarized alpha (alpha > 0.5) against the target
alpha mask. Trajectory-level analyses (half-life, survival, AUC) operate on
recorded per-step Hamming trajectories, host-side in numpy.
"""

import jax.numpy as jnp
import numpy as np

ALPHA_BIN_THRESHOLD = 0.5


def binarize_alpha(state: jnp.ndarray, threshold: float = ALPHA_BIN_THRESHOLD) -> jnp.ndarray:
    return state[..., -1] > threshold


def target_alpha_mask(target: jnp.ndarray, threshold: float = ALPHA_BIN_THRESHOLD) -> jnp.ndarray:
    return target[..., -1] > threshold


def hamming_to_target(state: jnp.ndarray, target_mask: jnp.ndarray, threshold: float = ALPHA_BIN_THRESHOLD) -> jnp.ndarray:
    """Mismatch rate between binarized alpha and the target alpha mask."""
    return jnp.mean(binarize_alpha(state, threshold) != target_mask)


def repair_half_life(traj: np.ndarray, lesion_times: np.ndarray, window: int = 250) -> float:
    """Mean time (steps) to recover 50% of the post-lesion Hamming jump.

    traj: (T,) Hamming per step. Lesions that never recover within `window`
    count as `window`. Returns NaN when there are no usable lesions.
    """
    traj = np.asarray(traj, dtype=np.float64)
    half_lives = []
    for t_l in lesion_times:
        t_l = int(t_l)
        if t_l <= 0 or t_l >= len(traj):
            continue
        h_pre = traj[t_l - 1]
        seg = traj[t_l : t_l + window]
        if len(seg) == 0:
            continue
        delta = seg.max() - h_pre
        if delta <= 1e-8:  # lesion had no measurable effect
            half_lives.append(0.0)
            continue
        below = np.nonzero(seg <= h_pre + delta / 2.0)[0]
        half_lives.append(float(below[0]) if len(below) else float(window))
    return float(np.mean(half_lives)) if half_lives else float("nan")


def survival(trajs: np.ndarray, eps: float = 0.1) -> float:
    """Fraction of runs with final-step Hamming below eps (survival@T, PRD §9)."""
    trajs = np.asarray(trajs, dtype=np.float64)
    return float(np.mean(trajs[:, -1] < eps))


def auc(traj: np.ndarray) -> float:
    """Area under the Hamming-vs-time curve, normalized to mean Hamming."""
    return float(np.mean(np.asarray(traj, dtype=np.float64)))
