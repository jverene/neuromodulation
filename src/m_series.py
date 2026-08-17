"""Controller-output (m_t) time-series probe.

Records the modulator level the controller actually emits over one hard-regime
rollout, to distinguish 'parent-specific tonic calibration' (m_t nearly
constant) from 'release policy' (m_t varies with damage events). Standalone
by design: does not touch the vmapped pipeline, so it cannot break the main
study. Slow-ish (plain Python loop over T steps) but runs in ~1-2 min.

Usage: python -m src.m_series --config <wired e2_hard.yaml> --controller <pkl> --out <csv>
"""

import argparse
import csv
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.channels import ModulatorState, decision, level, step_decay
from src.controller import alive_fraction, apply as controller_apply, grid_stats
from src.damage import damage_seed_sets, make_recurring_schedule_block
from src.evolve import build_model
from src.metrics import target_alpha_mask
from src.rollout import dense_lesion_masks, grow_from_seed
from src.targets import load_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--damage-seed", type=int, default=10000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rc = cfg["rollout"]
    shape = (cfg["target"]["canvas"], cfg["target"]["canvas"])
    K = rc["K"]
    T = rc["T"]
    dmg = cfg["damage"]

    cs = build_model(cfg, K, cfg["model"].get("parent_params"))
    state0 = grow_from_seed(cs, shape, cfg["model"]["channel_size"], rc["grow_steps"], seed=cfg["seed"])
    target_mask = target_alpha_mask(load_target(**cfg["target"]))
    reference_mask = state0[..., -1] > 0.5

    with open(args.controller, "rb") as f:
        params = pickle.load(f)

    times, masks = make_recurring_schedule_block(
        args.damage_seed, T=T, interval=rc["lesion_interval"], shape=shape,
        block_side=dmg["block_side"], n_blocks=dmg["n_blocks"])
    dense = dense_lesion_masks(times, masks, T, shape)

    state = state0
    mod = ModulatorState.zeros(K)
    alive_prev = alive_fraction(state)
    m_series, u_series = [], []
    for t in range(T):
        # scheduled lesion
        state = jnp.where(dense[t][..., None], 0.0, state)
        # controller decision every tau_decision steps
        if t % rc["tau_decision"] == 0:
            stats = grid_stats(state, alive_prev, reference_mask)
            u = controller_apply(params, stats)
            mod = decision(mod, u)
            alive_prev = alive_fraction(state)
        m = level(mod)
        m_series.append(np.asarray(m))
        u_series.append(np.asarray(u) if t % rc["tau_decision"] == 0 else np.full(K, np.nan))
        # NCA step with injection, then phasic decay
        state = cs._step(state, m)
        mod = step_decay(mod)

    M = np.array(m_series)  # (T, K)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"m_k{k}" for k in range(K)])
        for t in range(T):
            w.writerow([t] + [f"{v:.6f}" for v in M[t]])

    # summary statistics that directly answer constant-vs-policy
    variances = M.std(axis=0)
    print(f"m_t series: shape {M.shape}")
    print(f"per-channel std of m_t: {variances}")
    print(f"per-channel mean of m_t: {M.mean(axis=0)}")
    tonic_like = all(v < 0.05 for v in variances)
    print(f"verdict: {'TONIC-LIKE (near-constant release -> parent-specific tonic calibration)' if tonic_like else 'POLICY-LIKE (time-varying release -> genuine release policy)'}")


if __name__ == "__main__":
    main()
