#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.2 MVA for PIM - v8.0
EPS+PNG + Timing + per-product cost tables
Shared utilities imported from xva_core.
"""

import os, math, argparse
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

from xva_core import (
    Timer, banner, savefig_both, print_table, relerr, fmt_pm_bps, trapz_weights,
    fair_strike_and_nominal, precompute_weights, f_func_vec, f_func_scalar,
    Shat_exact_cpu, S_exact_cpu, call_bs_np, pinball, pinball_vec
)

plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
np.set_printoptions(precision=6, suppress=True)

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

@dataclass
class Params:
    r: float = 0.02
    kappa: float = 0.12
    sigma: float = 0.20
    S0: float = 100.0
    T: float = 5.0
    h: float = 0.25
    delta: float = 1.0 / 52.0
    gamma1: float = 0.01
    gamma_fund: float = 0.01

# Explicit α(t; •) for PIM
def B_conf_vec(t_arr: np.ndarray, a_conf: float, f_func_handle, sigma: float, delta: float) -> np.ndarray:
    z = norm.ppf(1.0 - a_conf)
    return f_func_handle(t_arr) * (1.0 - np.exp(sigma * np.sqrt(delta) * z - 0.5 * sigma**2 * delta))

def B_tail_vec(t_arr: np.ndarray, a_tail: float, f_func_handle, sigma: float, delta: float) -> np.ndarray:
    z = norm.ppf(a_tail)
    return f_func_handle(t_arr) * (1.0 - np.exp(sigma * np.sqrt(delta) * z - 0.5 * sigma**2 * delta))

def Y_call_samples(t: float, S_t: float, n: int, r: float, sigma: float, delta: float, T: float, K: float):
    Z = np.random.normal(size=n)
    S_td = S_t * np.exp((r - 0.5 * sigma**2) * delta + sigma * np.sqrt(delta) * Z)
    beta_t = np.exp(r * t); beta_td = np.exp(r * (t + delta))
    c_t = call_bs_np(t, S_t, T, K, r, sigma)
    c_td = call_bs_np(t + delta, S_td, T, K, r, sigma)
    return -beta_td * c_td + beta_t * c_t  # (loss over MPoR for long call)

# -------- Engineered features for call VaR --------
def call_features(S, t, p):
    """
    Engineered features for call VaR:
      - log-moneyness m = ln(S/S0)
      - tau = T - t
      - sqrt_tau = sqrt(tau)
      - s_rel = S/S0
    Returns an (N,4) array.
    """
    tau = np.maximum(p.T - np.asarray(t, dtype=float), 1e-12)
    m   = np.log(np.asarray(S, dtype=float) / p.S0)
    X   = np.column_stack([m, tau, np.sqrt(tau), np.asarray(S, dtype=float)/p.S0])
    return X

def main():
    parser = argparse.ArgumentParser(description="XVA 5.2 MVA for PIM - v8.0 (PNG+EPS) + Timing + cost tables")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--figdir", type=str, default="figs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--inner_scale", type=float, default=1.0)
    parser.add_argument("--outer_scale", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.99)

    # Q6 options
    parser.add_argument("--a_min", type=float, default=0.001, help="tail prob min for multi-α")
    parser.add_argument("--a_max", type=float, default=0.15,  help="tail prob max for multi-α")
    parser.add_argument("--q6_nodes", type=int, default=21)      # interpolation nodes
    parser.add_argument("--q6_steps", type=int, default=2500)    # training iters
    parser.add_argument("--q6_batch", type=int, default=40000)   # batch size
    parser.add_argument("--q6_warm_inner", type=int, default=8000, help="inner samples for Q6 warm-start")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Device (general): {device}")

    # Timer + (optional) CUDA warmup
    T = Timer(device)
    if hasattr(device, "type") and device.type == "cuda":
        _ = torch.randn(1, device=device)
        torch.cuda.synchronize()

    p = Params()
    payment_times = np.arange(p.h, p.T + 1e-12, p.h)
    num_payments = len(payment_times)
    print(f"Num payments: {num_payments}  |  First {payment_times[0]:.2f}  Last {payment_times[-1]:.2f}")

    barS, Nom = fair_strike_and_nominal(p.S0, p.r, p.kappa, p.h, payment_times)
    print(f"barS = {barS:.6f} | Nom = {Nom:.6f}")

    weights, weights_suffix = precompute_weights(p.r, p.kappa, p.h, payment_times)
    def f_func_handle(t_arr):
        return f_func_vec(t_arr, payment_times, weights_suffix, p.delta)

    # unit converters for final print
    def swap_to_bps(x: float) -> float:
        """bps on the SAME (calculated) swap nominal: 10,000 × MVA₀ (no /Nom rescaling)."""
        return 10000.0 * x
    def call_to_bps(x: float) -> float:
        """bps per S0 for ATM call."""
        return 10000.0 * (x / p.S0)

    # Q1
    banner("Q1 - Proofs (Lemma 3.1 and Proposition 3.1)")
    print("Re-derive VaR and MVA_0 used for validation (not shown here; see my article).")

    # Q2 - robust single-α learning α(t) ≈ Nom B(t;a)
    banner("Q2 - Robust single-α learner: α(t) ≈ Nom · B(t; a_conf) via pinball")
    a_conf = args.a
    t_grid = np.arange(0.0, p.T + 1e-12, p.delta)
    f_all = f_func_handle(t_grid)
    mask = f_all > 1e-14
    t_eff = t_grid[mask]; f_eff = f_all[mask]; n_eff = len(t_eff)

    n_inner_train = max(2, int(8000 * args.inner_scale))
    with T.timeit("Q2: generate tildeY_mat (antithetic)"):
        Z = np.random.normal(size=(n_eff, n_inner_train // 2))
        Z = np.concatenate([Z, -Z], axis=1)
        tildeY_mat = Nom * f_eff[:, None] * (1.0 - np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Z))

    EPOCHS_Q2 = 1000
    with T.timeit("Q2: init & robust pinball training (single-α)"):
        alpha0 = np.quantile(tildeY_mat, a_conf, axis=1)
        eps = 1e-12
        theta0 = np.log(np.clip(1.0 - alpha0 / np.maximum(eps, Nom * f_eff), eps, 1.0))

        u = torch.zeros(n_eff, dtype=torch.float32, device=device, requires_grad=True)
        u_init = np.log(np.expm1(-theta0))
        u.data = torch.tensor(u_init, dtype=torch.float32, device=device)

        idx_t = torch.repeat_interleave(torch.arange(n_eff, device=device), n_inner_train)
        tildeY_t = torch.tensor(tildeY_mat.ravel(), dtype=torch.float32, device=device)

        opt_u = torch.optim.Adam([u], lr=2e-3)
        for _ in range(EPOCHS_Q2):
            opt_u.zero_grad()
            theta = -torch.nn.functional.softplus(u)
            alpha_bins = Nom * torch.tensor(f_eff, device=device) * (1.0 - torch.exp(theta))
            yhat = alpha_bins[idx_t]
            loss = pinball(yhat, tildeY_t, a_conf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([u], 1.0)
            opt_u.step()

    with torch.no_grad():
        theta = -torch.nn.functional.softplus(u)
        alpha_hat_eff = (Nom * torch.tensor(f_eff, device=device) * (1.0 - torch.exp(theta))).cpu().numpy()

    alpha_hat_full = np.zeros_like(t_grid); alpha_hat_full[mask] = alpha_hat_eff
    alpha_true_full = Nom * B_conf_vec(t_grid, a_conf, f_func_handle, p.sigma, p.delta)

    safe = f_all > 1e-8
    rel_err = np.abs(alpha_hat_full[safe] - alpha_true_full[safe]) / np.maximum(1e-12, np.abs(alpha_true_full[safe]))
    med_rel = float(np.median(rel_err)); p95_rel = float(np.quantile(rel_err, 0.95))
    print(f"[CHECK Q2 robust] α(t) rel.error - median={med_rel:.3e}, p95={p95_rel:.3e}")
    print(f"[MODEL] Robust single-α learner: parameters={n_eff}, loss=pinball, optimizer=Adam lr=2e-3, epochs={EPOCHS_Q2}")

    fig = plt.figure()
    plt.plot(t_grid, alpha_true_full, label="α true")
    plt.plot(t_grid, alpha_hat_full, label="α learned (robust)")
    plt.yscale("log"); plt.grid(True, which="both"); plt.legend()
    plt.title("α(t): true vs learned (log scale)")
    savefig_both(fig, args.figdir, "5_2_q2_alpha_true_vs_learned")

    # Q3 - per-t validation
    banner("Q3 - Validation at each grid time t (Explicit vs Twin/Nested vs Learner)")
    np.random.seed(args.seed + 1)
    Shat_t = Shat_exact_cpu(t_grid, S0=p.S0, sigma=p.sigma)

    def sample_quantile(values, q):
        return np.quantile(values, q, method="linear")

    n_small = max(1, int(2000 * args.inner_scale))
    n_big = max(1, int(10000 * args.inner_scale))

    Y_explicit = alpha_true_full * Shat_t
    Y_pred = alpha_hat_full * Shat_t
    Y_twin = np.zeros_like(t_grid)
    Y_nested = np.zeros_like(t_grid)

    rng = np.random.default_rng(args.seed + 11)
    with T.timeit(f"Q3: Twin MC over grid (n_small={n_small})"):
        for k, t in enumerate(t_grid):
            sh = Shat_t[k]
            if f_all[k] <= 1e-14:
                Y_twin[k] = 0.0
                continue
            Z1 = rng.standard_normal(n_small); Z2 = rng.standard_normal(n_small)
            sh_d1 = sh * np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Z1)
            sh_d2 = sh * np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Z2)
            Y1 = Nom * f_all[k] * (sh - sh_d1); Y2 = Nom * f_all[k] * (sh - sh_d2)
            Y_twin[k] = 0.5 * (sample_quantile(Y1, a_conf) + sample_quantile(Y2, a_conf))

    with T.timeit(f"Q3: Nested MC over grid (n_big={n_big})"):
        for k, t in enumerate(t_grid):
            sh = Shat_t[k]
            if f_all[k] <= 1e-14:
                Y_nested[k] = 0.0
                continue
            Zb = rng.standard_normal(n_big)
            sh_db = sh * np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Zb)
            Yb = Nom * f_all[k] * (sh - sh_db)
            Y_nested[k] = sample_quantile(Yb, a_conf)

    fig = plt.figure()
    plt.plot(t_grid, Y_explicit, label="Explicit (Lemma)")
    plt.plot(t_grid, Y_twin, label="Twin MC", linestyle=":")
    plt.plot(t_grid, Y_nested, label="Nested MC", linestyle="--")
    plt.plot(t_grid, Y_pred, label="Linear NN Robust pinball")
    plt.xlabel("t (years)"); plt.ylabel("VaR_a gap"); plt.grid(True)
    plt.legend(); plt.title("Swap VaR - v2-streaming")
    savefig_both(fig, args.figdir, "5_2_q3_swap_var_streaming")

    scale = np.maximum(1e-12, np.median(np.abs(Y_explicit[f_all > 1e-14])))
    rmse_learner = rmse(Y_pred, Y_explicit) / scale
    rmse_twin = rmse(Y_twin, Y_explicit) / scale
    rmse_nested = rmse(Y_nested, Y_explicit) / scale
    print(f"[CHECK Q3] RMSE/scale - learner={rmse_learner:.3e}, twin={rmse_twin:.3e}, nested={rmse_nested:.3e}")

    # Q4 - Swap MVA₀ (quadrature + learned-model)
    banner("Q4 - Compute MVA_0 (Swap): quadrature vs learned-model (Q2)")
    label_swap_quad = "Q4: MVA0 Swap - Quadrature (explicit)"
    with T.timeit(label_swap_quad):
        B_vals = B_conf_vec(t_grid, a_conf, f_func_handle, p.sigma, p.delta)
        integrand = B_vals * p.gamma_fund * np.exp(-p.gamma1 * t_grid)
        MVA_swap_quad = Nom * p.S0 * np.trapezoid(integrand, t_grid)
    label_swap_model = "Q4: MVA0 Swap - Linear NN robust pinball"
    with T.timeit(label_swap_model):
        integrand_hat = alpha_hat_full * p.gamma_fund * np.exp(-p.gamma1 * t_grid)
        MVA_swap_model = p.S0 * np.trapezoid(integrand_hat, t_grid)
    relerr_mva_swap = relerr(MVA_swap_model, MVA_swap_quad)
    print(f"[Q4] MVA0 Swap (quadrature)  = {swap_to_bps(MVA_swap_quad):.4f} bps")
    print(f"[Q4] MVA0 Swap (Linear NN robust pinball)  = {swap_to_bps(MVA_swap_model):.4f} bps")
    print(f"[CHECK Q4] relative error    = {relerr_mva_swap:.3e}")

    # Q5 - ATM Call VaR learners + validation + coverage
    banner("Q5 - ATM Call VaR: Polynomial vs Neural Network via pinball; validation + coverage")
    np.random.seed(args.seed + 2)
    N_outer = 800
    t_outer = np.random.uniform(0, p.T, size=N_outer)
    S_outer = S_exact_cpu(t_outer, p.S0, p.r, p.sigma)
    n_inner_call = max(1, int(1500 * args.inner_scale))

    with T.timeit(f"Q5: Nested labels generation (N_outer={N_outer}, inner={n_inner_call})"):
        yq_call = np.zeros(N_outer)
        for i, (t, S) in enumerate(zip(t_outer, S_outer)):
            Ys = Y_call_samples(t, S, n_inner_call, p.r, p.sigma, p.delta, p.T, p.S0)
            yq_call[i] = np.quantile(Ys, a_conf, method="linear")

    # Engineered features + scaler + degree-3 polynomial
    X_call_raw = call_features(S_outer, t_outer, p)
    scaler = StandardScaler()
    X_call = scaler.fit_transform(X_call_raw)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_poly = poly.fit_transform(X_call)

    # Poly (deg-3)
    Xp = torch.tensor(X_poly, dtype=torch.float32, device=device)
    y_call_q = torch.tensor(yq_call, dtype=torch.float32, device=device).view(-1, 1)
    EPOCHS_POLY = 1000
    with T.timeit("Q5: Polynomial (deg-3) pinball training"):
        w_poly = torch.nn.Parameter(torch.zeros((Xp.shape[1], 1), dtype=torch.float32, device=device))
        opt_p = torch.optim.Adam([w_poly], lr=3e-3, weight_decay=1e-4)
        for _ in range(EPOCHS_POLY):
            opt_p.zero_grad()
            yhat = Xp @ w_poly
            loss = pinball_vec(yhat, y_call_q, a_conf)
            loss.backward()
            opt_p.step()
    with torch.no_grad():
        yhat_poly = (Xp @ w_poly).cpu().numpy().ravel()
    print(f"[Q5] Polynomial (deg-3) - MAE vs nested labels: {mean_absolute_error(yq_call, yhat_poly):.6e}")
    print(f"[MODEL] PolynomialFeatures(degree=3, include_bias=True) + StandardScaler; loss=pinball; optimizer=Adam lr=3e-3 wd=1e-4; epochs={EPOCHS_POLY}")

    # NN (2x64x1)
    EPOCHS_NN = 1000
    with T.timeit("Q5: Neural Network (2×64×1) pinball training"):
        X_nn = torch.tensor(X_call, dtype=torch.float32, device=device)
        yq_t = torch.tensor(yq_call, dtype=torch.float32, device=device).view(-1, 1)
        mlp = nn.Sequential(
            nn.Linear(X_call.shape[1], 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)
        opt_mlp = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(EPOCHS_NN):
            opt_mlp.zero_grad()
            yhat = mlp(X_nn)
            loss = pinball_vec(yhat, yq_t, a_conf)
            loss.backward()
            opt_mlp.step()
    with torch.no_grad():
        yhat_nn = mlp(torch.tensor(X_call, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    nn_params = sum(p.numel() for p in mlp.parameters())
    print(f"[Q5] Neural Network (2×64×1) - MAE vs nested labels: {mean_absolute_error(yq_call, yhat_nn):.6e}")
    print(f"[MODEL] Neural Network: {mlp}")
    print(f"[MODEL] Training: loss=pinball, optimizer=Adam lr=3e-3 wd=1e-4, epochs={EPOCHS_NN}, parameters={nn_params}")

    # Isotonic calibration maps (pred -> true)
    iso_poly = IsotonicRegression(out_of_bounds="clip").fit(yhat_poly, yq_call)
    iso_nn   = IsotonicRegression(out_of_bounds="clip").fit(yhat_nn,   yq_call)
    print("[MODEL] Calibration: IsotonicRegression(out_of_bounds='clip') fitted on training predictions → nested labels")

    # Validation
    np.random.seed(args.seed + 3)
    N_val = 120
    t_val = np.linspace(0, p.T, N_val)
    S_val = S_exact_cpu(t_val, p.S0, p.r, p.sigma)

    y_nested = np.zeros(N_val); y_twin = np.zeros(N_val)
    n_small_val = max(1, int(1200 * args.inner_scale)); n_big_val = max(1, int(5000 * args.inner_scale))
    with T.timeit(f"Q5: Validation twin MC (n_small={n_small_val})"):
        for i, (t, S) in enumerate(zip(t_val, S_val)):
            Ys1 = Y_call_samples(t, S, n_small_val, p.r, p.sigma, p.delta, p.T, p.S0)
            Ys2 = Y_call_samples(t, S, n_small_val, p.r, p.sigma, p.delta, p.T, p.S0)
            y_twin[i] = 0.5 * (np.quantile(Ys1, a_conf, method="linear") + np.quantile(Ys2, a_conf, method="linear"))
    with T.timeit(f"Q5: Validation nested MC (n_big={n_big_val})"):
        for i, (t, S) in enumerate(zip(t_val, S_val)):
            Yb = Y_call_samples(t, S, n_big_val, p.r, p.sigma, p.delta, p.T, p.S0)
            y_nested[i] = np.quantile(Yb, a_conf, method="linear")

    # Validation features + calibration
    X_val_raw = call_features(S_val, t_val, p)
    X_val = scaler.transform(X_val_raw)
    Xv_poly = poly.transform(X_val)
    with T.timeit("Q5: Predict polynomial on validation"):
        with torch.no_grad():
            y_poly_v_raw = (torch.tensor(Xv_poly, dtype=torch.float32, device=device) @ w_poly).cpu().numpy().ravel()
    y_poly_v = iso_poly.predict(y_poly_v_raw)
    with T.timeit("Q5: Predict neural network on validation"):
        with torch.no_grad():
            y_nn_v_raw = mlp(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    y_nn_v = iso_nn.predict(y_nn_v_raw)

    mae_poly_val = mean_absolute_error(y_nested, y_poly_v)
    mae_nn_val = mean_absolute_error(y_nested, y_nn_v)
    print(f"[Q5] Polynomial vs nested MAE: {mae_poly_val:.6e}")
    print(f"[Q5] Neural Network vs nested MAE: {mae_nn_val:.6e}")

    def coverage_err(model_pred, n_per=4000):
        errs = []
        for (t, S, qhat) in zip(t_val, S_val, model_pred):
            Ys = Y_call_samples(t, S, n_per, p.r, p.sigma, p.delta, p.T, p.S0)
            errs.append(abs(np.mean(Ys <= qhat) - a_conf))
        return float(np.mean(errs)), float(np.max(errs))

    with T.timeit("Q5: Coverage evaluation (poly & NN)"):
        mean_cov_poly, max_cov_poly = coverage_err(y_poly_v)
        mean_cov_nn, max_cov_nn = coverage_err(y_nn_v)
    print(f"[CHECK Q5] Coverage |P(Y<=q)-a| - Polynomial: mean={mean_cov_poly:.3e}, max={max_cov_poly:.3e}")
    print(f"[CHECK Q5] Coverage |P(Y<=q)-a| - NeuralNet : mean={mean_cov_nn:.3e}, max={max_cov_nn:.3e}")

    fig = plt.figure()
    plt.plot(y_nested, label="Nested quantile", linewidth=2.0)
    plt.plot(y_twin, label="Twin quantile", linestyle=":")
    plt.plot(y_poly_v, label="Polynomial")
    plt.plot(y_nn_v, label="Neural Network")
    plt.xlabel("Validation index"); plt.ylabel("VaR_a (call)")
    plt.title("ATM Call VaR - nested/twin vs predictors"); plt.grid(True); plt.legend()
    savefig_both(fig, args.figdir, "5_2_q5_call_var_streaming")

    # Q4b - MVA_0 (Swap) via MC
    banner("Q4b - MVA_0 (Swap) via MC: Nested & Twin (integrate VaR over t)")
    n_small_mva = max(1, int(1000 * args.inner_scale))
    n_big_mva = max(1, int(2000 * args.inner_scale))
    w_dt = trapz_weights(t_grid) * p.gamma_fund * np.exp(-p.gamma1 * t_grid)

    def one_rep_swap_arrays(seed_shift: int):
        rng = np.random.default_rng(args.seed + seed_shift)
        Sh = Shat_exact_cpu(t_grid, p.S0, p.sigma)
        Yn = np.zeros_like(t_grid); Ytw = np.zeros_like(t_grid)
        for k, t in enumerate(t_grid):
            if f_all[k] <= 1e-14:
                Yn[k]=0.0; Ytw[k]=0.0; continue
            sh = Sh[k]
            # nested
            Zb = rng.standard_normal(n_big_mva)
            sh_db = sh * np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Zb)
            Yb = Nom * f_all[k] * (sh - sh_db)
            Yn[k] = np.quantile(Yb, a_conf, method="linear")
            # twin
            Z1 = rng.standard_normal(n_small_mva); Z2 = rng.standard_normal(n_small_mva)
            sh_d1 = sh * np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Z1)
            sh_d2 = sh * np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Z2)
            Y1 = Nom * f_all[k] * (sh - sh_d1); Y2 = Nom * f_all[k] * (sh - sh_d2)
            Ytw[k] = 0.5*(np.quantile(Y1, a_conf, method="linear")+np.quantile(Y2, a_conf, method="linear"))
        return Yn, Ytw

    label_swap_nested_mc = f"Q4b: MVA0 Swap - Nested MC (inner={n_big_mva})"
    with T.timeit(label_swap_nested_mc):
        Yn1,_ = one_rep_swap_arrays(101)
        Yn2,_ = one_rep_swap_arrays(202)
        MVA_swap_nested_1 = float(np.sum(w_dt * Yn1))
        MVA_swap_nested_2 = float(np.sum(w_dt * Yn2))
        MVA_swap_nested = 0.5*(MVA_swap_nested_1 + MVA_swap_nested_2)
        se_swap_nested = abs(MVA_swap_nested_1 - MVA_swap_nested_2)/math.sqrt(2.0)

    label_swap_twin_mc = f"Q4b: MVA0 Swap - Twin MC (inner={n_small_mva}×2)"
    with T.timeit(label_swap_twin_mc):
        _,Yt1 = one_rep_swap_arrays(303)
        _,Yt2 = one_rep_swap_arrays(404)
        MVA_swap_twin_1 = float(np.sum(w_dt * Yt1))
        MVA_swap_twin_2 = float(np.sum(w_dt * Yt2))
        MVA_swap_twin = 0.5*(MVA_swap_twin_1 + MVA_swap_twin_2)
        se_swap_twin = abs(MVA_swap_twin_1 - MVA_swap_twin_2)/math.sqrt(2.0)

    # Q5b - MVA_0 (ATM Call): model-based & MC
    banner("Q5b - MVA_0 (ATM Call): model-based (Polynomial / Neural Network) & MC (Nested/Twin)")
    N_outer_S = max(50, int(250 * args.outer_scale))                   # for MC
    n_inner_call_nested = max(500, int(2000 * args.inner_scale))
    n_inner_call_twin   = max(250, int(1000 * args.inner_scale))
    w_dt_call = w_dt

    # MC reference - Nested
    label_call_nested_mc = f"Q5b: MVA0 Call - Nested MC (S_outer={N_outer_S}, inner={n_inner_call_nested})"
    with T.timeit(label_call_nested_mc):
        m_t = np.zeros_like(t_grid); se_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S)
            Z = np.random.normal(size=(N_outer_S, n_inner_call_nested))
            S_td = Ss[:, None] * np.exp((p.r - 0.5 * p.sigma**2) * p.delta + p.sigma * math.sqrt(p.delta) * Z)
            beta_t  = math.exp(p.r * t)
            beta_td = math.exp(p.r * (t + p.delta))
            c_t  = call_bs_np(t, Ss, p.T, p.S0, p.r, p.sigma)[:, None]
            c_td = call_bs_np(t + p.delta, S_td, p.T, p.S0, p.r, p.sigma)
            Y = -beta_td * c_td + beta_t * c_t
            q = np.quantile(Y, a_conf, axis=1, method="linear")
            m_t[idx]  = float(np.mean(q))
            se_t[idx] = float(np.std(q, ddof=1) / math.sqrt(len(q))) if len(q) > 1 else 0.0
        MVA_call_nested = float(np.sum(w_dt_call * m_t))
        se_call_nested  = float(np.sqrt(np.sum((w_dt_call * se_t)**2)))

    # MC - Twin
    label_call_twin_mc = f"Q5b: MVA0 Call - Twin MC (S_outer={N_outer_S}, inner={n_inner_call_twin}×2)"
    with T.timeit(label_call_twin_mc):
        m_t = np.zeros_like(t_grid); se_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S)
            def var_per_outer(n_inner):
                Z = np.random.normal(size=(N_outer_S, n_inner))
                S_td = Ss[:, None] * np.exp((p.r - 0.5 * p.sigma**2) * p.delta + p.sigma * math.sqrt(p.delta) * Z)
                beta_t  = math.exp(p.r * t)
                beta_td = math.exp(p.r * (t + p.delta))
                c_t  = call_bs_np(t, Ss, p.T, p.S0, p.r, p.sigma)[:, None]
                c_td = call_bs_np(t + p.delta, S_td, p.T, p.S0, p.r, p.sigma)
                Y = -beta_td * c_td + beta_t * c_t
                return np.quantile(Y, a_conf, axis=1, method="linear")
            q1 = var_per_outer(n_inner_call_twin)
            q2 = var_per_outer(n_inner_call_twin)
            q  = 0.5 * (q1 + q2)
            m_t[idx]  = float(np.mean(q))
            se_t[idx] = float(np.std(q, ddof=1) / math.sqrt(len(q))) if len(q) > 1 else 0.0
        MVA_call_twin = float(np.sum(w_dt_call * m_t))
        se_call_twin  = float(np.sqrt(np.sum((w_dt_call * se_t)**2)))

    # Model-based0 - Polynomial & NN
    N_outer_S_fast = max(200, int(1000 * args.outer_scale))
    label_call_poly = f"Q5b: MVA0 Call - Polynomial (deg-3, S_outer={N_outer_S_fast})"
    with T.timeit(label_call_poly):
        m_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S_fast)
            Xs_raw = call_features(Ss, np.full_like(Ss, t, dtype=float), p)
            Xs = scaler.transform(Xs_raw)
            Xs_poly = poly.transform(Xs)
            with torch.no_grad():
                preds_raw = (torch.tensor(Xs_poly, dtype=torch.float32, device=device) @ w_poly).cpu().numpy().ravel()
            preds = IsotonicRegression(out_of_bounds="clip").fit(yhat_poly, yq_call).predict(preds_raw)  # quick cal
            m_t[idx] = float(np.mean(preds))
        MVA_call_poly = float(np.sum(w_dt_call * m_t))

    label_call_nn = f"Q5b: MVA0 Call - Neural Network (2×64×1, epochs={EPOCHS_NN}, S_outer={N_outer_S_fast})"
    with T.timeit(label_call_nn):
        m_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S_fast)
            Xs_raw = call_features(Ss, np.full_like(Ss, t, dtype=float), p)
            Xs = scaler.transform(Xs_raw)
            with torch.no_grad():
                preds_raw = mlp(torch.tensor(Xs, dtype=torch.float32, device=device)).cpu().numpy().ravel()
            preds = IsotonicRegression(out_of_bounds="clip").fit(yhat_nn, yq_call).predict(preds_raw)  # quick cal
            m_t[idx] = float(np.mean(preds))
        MVA_call_nn = float(np.sum(w_dt_call * m_t))

    # ---------------- Summaries (bps) ----------------
    banner("SUMMARY - MVA₀ Results (bps)")
    print("Definitions: swap bps = 10,000 × MVA₀ on the SAME calculated nominal; call bps = 10,000 × MVA₀ / S0.")

    # SWAP summary (ref = quadrature)
    swap_ref = MVA_swap_quad
    swap_rows = [
        ["Quadrature (explicit)", f"{swap_to_bps(MVA_swap_quad):.4f} bps", "-", "-"],
        ["Linear NN robust pinball",  f"{swap_to_bps(MVA_swap_model):.4f} bps", "-", f"{relerr(MVA_swap_model, swap_ref):.2%}"],
        ["Nested MC",             fmt_pm_bps(swap_to_bps(MVA_swap_nested), swap_to_bps(se_swap_nested)), f"{swap_to_bps(se_swap_nested):.4f} bps", f"{relerr(MVA_swap_nested, swap_ref):.2%}"],
        ["Twin MC",               fmt_pm_bps(swap_to_bps(MVA_swap_twin),   swap_to_bps(se_swap_twin)),   f"{swap_to_bps(se_swap_twin):.4f} bps",   f"{relerr(MVA_swap_twin,   swap_ref):.2%}"],
    ]
    print("SWAP (linear exposure) - MVA₀ in bps on calculated nominal")
    print_table(swap_rows, header=["Method", "MVA₀ (bps)", "StdErr (bps)", "RelErr vs Ref"])

    # CALL summary (ref = Nested MC)
    call_ref = MVA_call_nested
    call_rows = [
        ["Nested MC (ref)",       fmt_pm_bps(call_to_bps(MVA_call_nested), call_to_bps(se_call_nested)), f"{call_to_bps(se_call_nested):.4f} bps", "-"],
        ["Twin MC",               fmt_pm_bps(call_to_bps(MVA_call_twin),   call_to_bps(se_call_twin)),   f"{call_to_bps(se_call_twin):.4f} bps",   f"{relerr(MVA_call_twin, call_ref):.2%}"],
        ["Polynomial (deg-3)", f"{call_to_bps(MVA_call_poly):.4f} bps", "-", f"{relerr(MVA_call_poly, call_ref):.2%}"],
        [f"Neural Network (2×64×1, epochs={EPOCHS_NN})",            f"{call_to_bps(MVA_call_nn):.4f} bps",   "-", f"{relerr(MVA_call_nn,   call_ref):.2%}"],
    ]
    print("\nATM CALL (non-linear exposure) - MVA₀ in bps of S0")
    print_table(call_rows, header=["Method", "MVA₀ (bps)", "StdErr (bps)", "RelErr vs Ref"])

    # PASS/Check flags (5% tolerance)
    tol_rel = 0.05
    print("\nPASS/Check:")
    print("Swap Learned  ", "PASS" if relerr(MVA_swap_model, swap_ref) < tol_rel else "CHECK")
    print("Swap Nested   ", "PASS" if relerr(MVA_swap_nested, swap_ref) < tol_rel else "CHECK")
    print("Swap Twin     ", "PASS" if relerr(MVA_swap_twin,   swap_ref) < tol_rel else "CHECK")
    print("Call Poly     ", "PASS" if relerr(MVA_call_poly,   call_ref) < tol_rel else "CHECK")
    print("Call NN       ", "PASS" if relerr(MVA_call_nn,     call_ref) < tol_rel else "CHECK")
    print("Call Twin     ", "PASS" if relerr(MVA_call_twin,   call_ref) < tol_rel else "CHECK")

    # ---------------- Per-product COST tables ----------------
    banner("COST - Swap MVA₀ methods")
    swap_labels = [label_swap_quad, label_swap_model, label_swap_nested_mc, label_swap_twin_mc]
    swap_items = [(lab, T.times.get(lab, 0.0)) for lab in swap_labels]
    swap_total = sum(t for _, t in swap_items) or 1.0
    swap_rows_cost = []
    for lab, sec in swap_items:
        pct = 100.0 * sec / swap_total
        swap_rows_cost.append([lab.replace("Q4b: ", ""), f"{sec:.4f} s", f"{pct:.1f}%"])
    print_table(swap_rows_cost, header=["Method (Swap MVA)", "Wall Time", "Share"])

    banner("COST - ATM Call MVA₀ methods")
    call_labels = [label_call_poly, label_call_nn, label_call_nested_mc, label_call_twin_mc]
    call_items = [(lab, T.times.get(lab, 0.0)) for lab in call_labels]
    call_total = sum(t for _, t in call_items) or 1.0
    call_rows_cost = []
    for lab, sec in call_items:
        pct = 100.0 * sec / call_total
        call_rows_cost.append([lab.replace("Q5b: ", ""), f"{sec:.4f} s", f"{pct:.1f}%"])
    print_table(call_rows_cost, header=["Method (Call MVA)", "Wall Time", "Share"])

    # ---------------- Global timing summary ----------------
    banner("TIMING - Breakdown (all measured blocks)")
    total_t = sum(T.times.values()) if T.times else 0.0
    items = sorted(T.times.items(), key=lambda kv: kv[1], reverse=True)
    rows = [(lab, f"{sec:.4f} s", f"{(sec/total_t*100 if total_t>0 else 0):.1f}%") for lab, sec in items]
    print_table(rows, header=["Block", "Wall Time", "Share"])
    print(f"\nTotal timed wall clock: {total_t:.4f} s")

    # ---------------- Q6 - Multi-α learning (unchanged logic) ----------------
    banner("Q6 - Multi-α learning by randomizing α_tail (with interpolation nodes)")
    a_min, a_max = args.a_min, args.a_max
    assert 0.0 < a_min < a_max < 0.5, "Use tail probabilities in (0, 0.5)."

    t_eff = t_grid[f_all > 1e-14]
    f_eff = f_all[f_all > 1e-14]
    n_eff = len(t_eff)
    Nom_t = Nom * torch.tensor(f_eff, dtype=torch.float32, device=device)  # (n_eff,)

    g_tail = torch.linspace(a_min, a_max, args.q6_nodes, dtype=torch.float32, device=device)
    U = torch.zeros((n_eff, args.q6_nodes), dtype=torch.float32, device=device, requires_grad=True)

    with T.timeit(f"Q6: Warm-start quantiles (n_warm={max(2000, int(args.q6_warm_inner))})"):
        n_warm = max(2000, int(args.q6_warm_inner))
        Z = np.random.normal(size=(n_eff, n_warm // 2))
        Z = np.concatenate([Z, -Z], axis=1)
        tildeY_np = (Nom * f_eff[:, None]) * (1.0 - np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Z))
        theta0_nodes = np.zeros((n_eff, args.q6_nodes), dtype=np.float64)
        for j in range(args.q6_nodes):
            a_tail_j = float(g_tail[j].detach().cpu().item())
            q_conf_j = 1.0 - a_tail_j
            alpha0_j = np.quantile(tildeY_np, q_conf_j, axis=1)
            denom = np.maximum(1e-18, Nom * f_eff)
            ratio = np.clip(1.0 - alpha0_j / denom, 1e-12, 1.0)
            theta0_nodes[:, j] = np.log(ratio)
        U_init = np.log(np.expm1(-theta0_nodes)).astype(np.float32)
        with torch.no_grad():
            U.data = torch.tensor(U_init, dtype=torch.float32, device=device)

    opt = torch.optim.Adam([U], lr=2e-3)

    def interp_theta(theta_nodes, a_tail_batch):
        idx = torch.bucketize(a_tail_batch, g_tail) - 1
        idx = torch.clamp(idx, 0, g_tail.numel()-2)
        gL = g_tail[idx]; gR = g_tail[idx+1]
        wR = (a_tail_batch - gL) / (gR - gL)
        wL = 1.0 - wR
        return idx, wL, wR

    with T.timeit(f"Q6: Training (steps={int(args.q6_steps)}, batch={int(args.q6_batch)})"):
        B = int(args.q6_batch)
        for _ in range(int(args.q6_steps)):
            t_idx = torch.randint(0, n_eff, (B,), device=device)
            Zb = torch.randn(B, device=device)
            u = torch.rand(B, device=device)
            log_a = math.log(a_min) + u * (math.log(a_max) - math.log(a_min))
            a_tail_batch = torch.exp(log_a)
            a_conf_batch = 1.0 - a_tail_batch

            Y_tilde = Nom_t[t_idx] * (1.0 - torch.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * math.sqrt(p.delta) * Zb))

            theta_nodes = -torch.nn.functional.softplus(U)
            idx, wL, wR = interp_theta(theta_nodes, a_tail_batch)
            thetaL = theta_nodes[t_idx, idx]
            thetaR = theta_nodes[t_idx, idx+1]
            theta = wL * thetaL + wR * thetaR

            yhat = Nom_t[t_idx] * (1.0 - torch.exp(theta))

            opt.zero_grad()
            loss = pinball_vec(yhat, Y_tilde, a_conf_batch)
            pen = 5e-4 * torch.relu(theta_nodes[:, :-1] - theta_nodes[:, 1:]).mean()
            (loss + pen).backward()
            torch.nn.utils.clip_grad_norm_([U], 1.0)
            opt.step()

    alphas_tail_plot = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
    theta_nodes = (-torch.nn.functional.softplus(U)).detach()
    alpha_hat = {float(a): np.zeros_like(t_grid) for a in alphas_tail_plot}

    with T.timeit("Q6: Inference over α_tail grid"):
        with torch.no_grad():
            for a_tail in alphas_tail_plot:
                a_tail_t = torch.tensor([a_tail], dtype=torch.float32, device=device)
                idx = torch.bucketize(a_tail_t, g_tail) - 1
                idx = torch.clamp(idx, 0, g_tail.numel()-2)
                gL, gR = g_tail[idx], g_tail[idx+1]
                wR = (a_tail_t - gL)/(gR - gL); wL = 1.0 - wR
                theta = wL*theta_nodes[:, idx] + wR*theta_nodes[:, idx+1]
                alpha_eff = (Nom_t * (1.0 - torch.exp(theta.view(-1)))).cpu().numpy()
                out = np.zeros_like(t_grid); out[f_all > 1e-14] = alpha_eff
                alpha_hat[float(a_tail)] = out

    # enforce non-crossing (non-increasing in α_tail)
    tails_sorted = sorted(alphas_tail_plot.tolist())
    A = np.vstack([alpha_hat[a] for a in tails_sorted])
    A_mon = np.minimum.accumulate(A, axis=0)
    for j, a in enumerate(tails_sorted):
        alpha_hat[a] = A_mon[j]

    ok = True
    for k in range(len(t_grid)):
        vals = [alpha_hat[a][k] for a in tails_sorted]
        if any(vals[i] < vals[i+1] - 1e-10 for i in range(len(vals)-1)):
            ok = False; break
    print(f"[CHECK Q6] Non-crossing (and non-increasing in α_tail): {'PASS' if ok else 'FAIL'}")

    fig = plt.figure(figsize=(12,6), dpi=120)
    colors = ['tab:blue','tab:orange','tab:green','tab:pink','tab:olive','tab:red']
    for a_tail, col in zip(tails_sorted, colors):
        plt.plot(t_grid, Nom*B_tail_vec(t_grid, a_tail, f_func_handle, p.sigma, p.delta),
                 label=f'α true, α_tail={a_tail:.3f}', linestyle='--', color=col)
        plt.plot(t_grid, alpha_hat[a_tail], label=f'α learned, α_tail={a_tail:.3f}', color=col )
    plt.yscale("log"); plt.grid(True, which="both"); plt.legend(ncol=2)
    plt.title("Q6 - α(t; α_tail): true vs learned (multi-α)")
    savefig_both(fig, args.figdir, "5_2_q6_alpha_true_vs_learned_multi_alpha")

    med_all = []; p95_all = []
    for a_tail in tails_sorted:
        alpha_true = Nom*B_tail_vec(t_grid, a_tail, f_func_handle, p.sigma, p.delta)
        safe = f_all > 1e-12
        rel = np.abs(alpha_hat[a_tail][safe] - alpha_true[safe]) / np.maximum(1e-16, np.abs(alpha_true[safe]))
        med = float(np.median(rel)); p95 = float(np.quantile(rel, 0.95))
        med_all.append(med); p95_all.append(p95)
        print(f"[Q6] α_tail={a_tail:.3f} | median rel.err={med:.3e}, p95={p95:.3e}")
    print(f"[Q6] Overall median rel.err={np.median(med_all):.3e}, p95={np.median(p95_all):.3e}")

    print("\n[DONE] 5.2 MVA for PIM script finished.")

if __name__ == "__main__":
    main()
