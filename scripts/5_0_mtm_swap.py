#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean and 2.5% / 97.5% quantiles of MtM (bps) over time for a 5y swap with quarterly payments
under a simple Black–Scholes-style model for the underlying, using a martingale-style
representation of the MtM process.

This script uses shared helpers from xva_core (banner, savefig_both, fair_strike_and_nominal).
"""

import os
import math
import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# Optional: environment info (GPU availability)
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

# ---- shared helpers (local package) ----
from xva_core import banner, savefig_both, fair_strike_and_nominal, seed_all

# -------------------------- Plot defaults --------------------------
plt.rcParams["figure.figsize"] = (8, 5)


# -------------------------- Core model helpers --------------------------
def simulate_underlying_paths(n_sims: int, N_euler: int, dt: float,
                              S0: float, kappa: float, sigma: float) -> np.ndarray:
    """
    Euler–Maruyama simulation for dS = κ S dt + σ S dW.
    Returns array of shape (n_sims, N_euler+1).
    """
    S = np.zeros((n_sims, N_euler + 1), dtype=float)
    S[:, 0] = S0
    sqdt = math.sqrt(dt)
    for n in range(N_euler):
        Z = np.random.normal(size=n_sims)
        S[:, n+1] = S[:, n] + kappa * S[:, n] * dt + sigma * S[:, n] * sqdt * Z
    return S

def compute_swap_MtM_paths(S_paths: np.ndarray, times_euler: np.ndarray,
                           r: float, kappa: float, barS: float, Nom: float,
                           payment_times: np.ndarray, h: float) -> np.ndarray:
    """
    Vectorized computation of MtM along each simulated path:

      MtM_t = Nom * [ β_t^{-1} β_{T_{l_t}} h (S_{T_{l_t-1}} - barS)
                      + β_t^{-1} Σ_{l=l_t+1}^d β_{T_l} h (E[S_{T_{l-1}}|F_t] - barS) ]

    where l_t is the first payment index such that T_l > t, β_t = e^{rt}, and
    E[S_{T_{l-1}} | F_t] = S_t · e^{κ (T_{l-1}-t)} under this simple drifted model.
    """
    n_sims, Np1 = S_paths.shape
    MtM = np.zeros((n_sims, Np1), dtype=float)

    # Precomputations
    dt = float(times_euler[1] - times_euler[0])
    disc_pay = np.exp(-r * payment_times)          # β_{T_l}^{-1}
    disc_eul = np.exp(-r * times_euler)            # β_{t}^{-1}
    beta_inv = disc_eul
    num_pay  = len(payment_times)

    # Index of the first payment strictly after each t_n
    l_t_arr = np.searchsorted(payment_times, times_euler, side="right")

    # Euler indices for S_{T_{l-1}} for all l (including l=0 -> T_0=0)
    idx_T_lm1 = np.round(np.maximum(payment_times - h, 0.0) / dt).astype(int)  # length d
    idx_T_lm1 = np.clip(idx_T_lm1, 0, Np1 - 1)

    for n, t_n in enumerate(times_euler):
        l_t = int(l_t_arr[n])
        if l_t >= num_pay:
            # past last payment -> MtM = 0
            continue

        # β_t^{-1} β_{T_l}
        beta_ratio_l_t   = beta_inv[n] * disc_pay[l_t]
        S_T_lm1          = S_paths[:, idx_T_lm1[l_t]]
        partial_payoff   = h * (S_T_lm1 - barS) * beta_ratio_l_t

        # Future payments l = l_t+1 .. d-1
        if l_t + 1 < num_pay:
            ll = np.arange(l_t + 1, num_pay)
            beta_ratio_ll = beta_inv[n] * disc_pay[ll]   # shape (L,)
            T_lm1_vals    = np.maximum(payment_times[ll] - h, 0.0)  # T_{l-1}
            # E[S_{T_{l-1}} | F_{t_n}] = S_n * exp(κ (T_{l-1} - t_n))
            S_cond = S_paths[:, n][:, None] * np.exp(kappa * (T_lm1_vals[None, :] - t_n))
            future_payoffs = np.sum(h * (S_cond - barS) * beta_ratio_ll[None, :], axis=1)
        else:
            future_payoffs = 0.0

        MtM[:, n] = Nom * (partial_payoff + future_payoffs)

    return MtM

# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser(description="XVA 5.0 - MtM distribution over time for a 5y swap (bps)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--n_sims", type=int, default=5000, help="number of Euler paths")
    ap.add_argument("--T", type=float, default=5.0, help="maturity in years")
    ap.add_argument("--h", type=float, default=0.25, help="payment spacing in years")
    ap.add_argument("--dt_euler", type=float, default=1/252, help="Euler time step (years)")
    ap.add_argument("--r", type=float, default=0.02, help="risk-free rate")
    ap.add_argument("--kappa", type=float, default=0.12, help="drift")
    ap.add_argument("--sigma", type=float, default=0.20, help="volatility")
    ap.add_argument("--S0", type=float, default=100.0, help="initial underlying level")
    ap.add_argument("--figdir", type=str, default="figs", help="output directory for figures")
    args = ap.parse_args()

    # Environment printout
    banner("Environment")
    print(f"NumPy version: {np.__version__}")
    if _HAVE_TORCH:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available:  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"CUDA device:     {torch.cuda.get_device_name(0)}")
            except Exception:
                pass
    else:
        print("PyTorch not installed; running entirely on CPU.")

    # Seed RNGs
    seed_all(args.seed)

    # Parameters & grids
    r, kappa, sigma, S0 = args.r, args.kappa, args.sigma, args.S0
    T, h, dt = args.T, args.h, args.dt_euler
    N_euler = int(round(T / dt))
    times_euler = np.linspace(0.0, T, N_euler + 1)
    num_payments = int(round(T / h))
    payment_times = np.array([(l + 1) * h for l in range(num_payments)], dtype=float)

    banner("Setup")
    print(f"r={r:.4f}, kappa={kappa:.4f}, sigma={sigma:.4f}, S0={S0:.2f}")
    print(f"T={T:.2f}y, h={h:.2f}y, dt={dt:.6f}y -> N_euler={N_euler}")
    print(f"Payments d={num_payments}: first={payment_times[0]:.2f}, last={payment_times[-1]:.2f}")

    barS, Nom = fair_strike_and_nominal(S0, r, kappa, h, payment_times)
    print(f"Fair strike  barS = {barS:.6f}")
    print(f"Nominal      Nom  = {Nom:.6f}  (float-leg PV at t=0 equals 1.0)")

    banner("Euler simulation for S")
    S_paths = simulate_underlying_paths(args.n_sims, N_euler, dt, S0, kappa, sigma)
    print(f"Simulated S paths: shape={S_paths.shape}")

    banner("Compute MtM paths via martingale-style representation")
    MtM_paths = compute_swap_MtM_paths(S_paths, times_euler, r, kappa, barS, Nom, payment_times, h)
    print(f"Computed MtM paths: shape={MtM_paths.shape}")

    # Convert MtM to basis points (×1e4)
    MtM_bps = 1e4 * MtM_paths
    mean_MtM    = np.mean(MtM_bps, axis=0)
    pct2p5_MtM  = np.percentile(MtM_bps, 2.5, axis=0)
    pct97p5_MtM = np.percentile(MtM_bps, 97.5, axis=0)

    # Plot
    banner("Plot: mean and 2.5%/97.5% quantiles of MtM (bps)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times_euler, pct97p5_MtM, label="97.5% quantile")
    ax.plot(times_euler, pct2p5_MtM,  label="2.5% quantile")
    ax.plot(times_euler, mean_MtM,    label="Average")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Mark-to-Market (basis points)")
    ax.set_title("Figure 1: Mean and 2.5% / 97.5% quantiles of MtM (bps) over time")
    ax.grid(True); ax.legend(loc="best")
    savefig_both(fig, args.figdir, "5_0_mtm_swap")

    print("\n[DONE] MtM (bps) over time script finished successfully.")

if __name__ == "__main__":
    main()
