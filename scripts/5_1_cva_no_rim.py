#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.1 CVA w/o RIM (Swap) + ATM Call v6.1
(CUDA-enforced, EPS+PNG, GPU MC) + Timing + bps output + per-calc cost tables
"""

import os, math, argparse
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

# -------- shared core --------the
from xva_core import (
    Timer, banner, savefig_both, print_table, relerr, fmt_pm_bps, trapz_weights,
    fair_strike_and_nominal, precompute_weights, f_func_vec, f_func_scalar,
    Shat_exact_gpu, Shat_step_delta_gpu, Shat_exact_cpu, S_exact_cpu, gap_LHS_MC_gpu,
    call_bs_torch, Params, swap_to_bps, call_to_bps
)

plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ------------------------ Call positive gap (wrapper) ------------------------
def call_gap_pos_mc_gpu(t: float, S_t: float, T: float, K: float, r: float, sigma: float, delta: float,
                        n_inner: int, device: torch.device) -> float:
    """
    E[ (β_{t+δ} C_{t+δ} - β_t C_t)^+ | S_t ] via inner MC (δ-step on S, then BS price).
    Uses xva_core.black_scholes.call_bs_torch; antithetic sampling.
    """
    beta_t  = math.exp(r * t)
    beta_td = math.exp(r * (t + delta))
    c_t = call_bs_torch(t, torch.tensor([S_t], dtype=torch.float32, device=device), T, K, r, sigma, device)[0]
    n = max(1, n_inner // 2)
    z = torch.randn(n, device=device)
    Z = torch.cat([z, -z], dim=0)
    S_td = torch.tensor(S_t, dtype=torch.float32, device=device) * torch.exp(
        torch.tensor((r - 0.5 * sigma**2) * delta, device=device) +
        torch.tensor(sigma * math.sqrt(delta), device=device) * Z
    )
    c_td = call_bs_torch(t + delta, S_td, T, K, r, sigma, device)
    gap = torch.relu(beta_td * c_td - beta_t * c_t)
    return float(gap.mean().item())

# ------------------------ Call plotting helpers ------------------------
def call_validation_plots(y_ref, y_hat, t_val, S_val, S0, model_name, stem_prefix, T_horizon, figdir):
    """
    Produce diagnostics:
    1) parity plot,
    2) boxplots of relative error by t-bins.
    """
    rel = (y_hat - y_ref) / np.maximum(1e-12, np.abs(y_ref))

    # 1) Parity plot with 45° line
    y_min = float(min(np.min(y_ref), np.min(y_hat)))
    y_max = float(max(np.max(y_ref), np.max(y_hat)))
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(y_ref, y_hat, s=12)
    plt.plot([y_min, y_max], [y_min, y_max], lw=1, color="k")
    plt.xlabel("Nested MC target")
    plt.ylabel("Model prediction")
    plt.title(f"ATM Call: parity plot ({model_name})")
    plt.grid(True)
    savefig_both(fig, figdir, f"{stem_prefix}_parity")

    # 2) Boxplots by t-bins (EPS-safe; no fills)
    nb = 10
    bins = np.linspace(0.0, T_horizon, nb + 1)
    idx = np.digitize(t_val, bins) - 1
    groups = [rel[idx == i] for i in range(nb)]
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = (bins[1] - bins[0]) * 0.8
    fig = plt.figure(figsize=(12, 4))
    plt.boxplot(groups, positions=centers, widths=widths, manage_ticks=False, showfliers=False)
    plt.xlabel("t (years)")
    plt.ylabel("Relative error")
    plt.title(f"ATM Call: relative error by t-bin ({model_name})")
    plt.grid(True)
    plt.xlim(0, T_horizon)
    savefig_both(fig, figdir, f"{stem_prefix}_relerr_by_tbin")

# ------------------------ Main ------------------------
def main():
    parser = argparse.ArgumentParser(description="5.1 CVA w/o RIM (Swap) + ATM Call")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--figdir", type=str, default="figs")
    parser.add_argument("--inner_scale", type=float, default=1.0, help="scales inner MC samples")
    parser.add_argument("--outer_scale", type=float, default=1.0, help="scales number of default-time samples")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. No GPU detected.")
    device = torch.device("cuda")
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Timer and CUDA warmup
    T = Timer(device)
    _ = torch.randn(1, device=device)
    torch.cuda.synchronize(device)

    print(f"Device: {device}")

    p = Params()
    payment_times = np.arange(p.h, p.T + 1e-12, p.h)
    num_payments = len(payment_times)
    print(f"Num payments: {num_payments}  |  First {payment_times[0]:.2f}  Last {payment_times[-1]:.2f}")

    barS, Nom = fair_strike_and_nominal(p.S0, p.r, p.kappa, p.h, payment_times)
    print(f"barS = {barS:.6f} | Nom = {Nom:.6f}")

    weights, weights_suffix = precompute_weights(p.r, p.kappa, p.h, payment_times)

    def f_func_scalar_local(t: float) -> float:
        return f_func_scalar(t, payment_times, weights_suffix, p.delta)

    const_black = norm.cdf(p.sigma * math.sqrt(p.delta) / 2.0) - norm.cdf(-p.sigma * math.sqrt(p.delta) / 2.0)
    def A_func_vec(t_arr: np.ndarray) -> np.ndarray:
        # A(t) = f(t) * const_black
        return f_func_vec(t_arr, payment_times, weights_suffix, p.delta) * const_black

    # Convenience wrapper for +/- value ± stderr formatting
    def fmt_pm(val_bps: float, se_bps: float) -> str:
        return fmt_pm_bps(val_bps, se_bps)

    # ---------------- Q1 ----------------
    banner("Q1 - Learn conditional expectation slope (swap) [CUDA]")
    N_grid = int(252 * p.T)  # daily grid
    t_grid = np.linspace(0, p.T, N_grid + 1)

    with T.timeit("Q1: draw Shat_t (GPU)"):
        Shat_t_gpu = Shat_exact_gpu(t_grid, p.S0, p.sigma, device)  # (N+1,)
    Shat_t = Shat_t_gpu.cpu().numpy()  # small transfer

    Y_true = Nom * A_func_vec(t_grid) * Shat_t
    inner_q1 = max(2, int(250 * args.inner_scale))

    # GPU MC per time bin (swap)
    with T.timeit(f"Q1: Nested MC over grid (inner={inner_q1})"):
        Y_mc = np.zeros_like(Shat_t)
        for k, (t, sh) in enumerate(zip(t_grid, Shat_t)):
            Y_mc[k] = gap_LHS_MC_gpu(sh, Nom, f_func_scalar_local(t), p.sigma, p.delta, inner_q1, device)

    # design matrix (CPU OK)
    n_bins = len(t_grid)
    X_lin = np.eye(n_bins) * Shat_t[:, None]

    # OLS (CPU)
    with T.timeit("Q1: OLS fit (CPU)"):
        ols = LinearRegression(fit_intercept=False).fit(X_lin, Y_mc)
        alpha_hat = ols.coef_.ravel()
        Y_pred_lin = alpha_hat * Shat_t
    print("[MODEL] LinearRegression(fit_intercept=False) for α(t) · Ŝ_t")

    # Linear NN (GPU)
    EPOCHS_Q1_NN = 1000
    with T.timeit("Q1: Linear neural network train (GPU)"):
        X_tensor = torch.tensor(X_lin, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(Y_mc, dtype=torch.float32, device=device).view(-1, 1)
        model = nn.Linear(in_features=n_bins, out_features=1, bias=False).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(EPOCHS_Q1_NN):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(X_tensor), y_tensor)
            loss.backward()
            opt.step()
    with T.timeit("Q1: Linear neural network predict (GPU->CPU)"):
        with torch.no_grad():
            Y_pred_nn = model(X_tensor).squeeze(1).cpu().numpy()
    nn_params_q1 = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Neural Network (linear layer): nn.Linear({n_bins}→1, bias=False)")
    print(f"[MODEL] Training: loss=MSE, optimizer=Adam lr=1e-2, epochs={EPOCHS_Q1_NN}, parameters={nn_params_q1}")

    # quick diagnostics
    mae_lin = mean_absolute_error(Nom * A_func_vec(t_grid) * Shat_t, Y_pred_lin)
    mae_nn = mean_absolute_error(Nom * A_func_vec(t_grid) * Shat_t, Y_pred_nn)
    print(f"OLS vs explicit - MAE: {mae_lin:.6e}")
    print(f"NeuralNet (linear) vs explicit - MAE: {mae_nn:.6e}")

    fig = plt.figure()
    plt.plot(t_grid, Y_true, lw=2, label="Explicit")
    plt.plot(t_grid, Y_mc, ls="--", label="Nested MC")
    plt.plot(t_grid, Y_pred_lin, alpha=0.7, label="Linear regression")
    plt.plot(t_grid, Y_pred_nn, alpha=0.7, label="Neural Net (linear)")
    plt.xlabel("t (years)"); plt.ylabel("E[(Ŝ_{t+δ}-Ŝ_t)^+] × Nom f(t)")
    plt.title("Swap: conditional expectation")
    plt.grid(True); plt.legend()
    savefig_both(fig, args.figdir, "5_1_q1_swap_conditional")

    # ---------------- Q2 ----------------
    banner("Q2 - Validate explicit vs twin/nested vs models (swap)")
    inner_twin = max(2, int(1000 * args.inner_scale))
    inner_nested = max(2, int(2000 * args.inner_scale))

    def twin_gpu(t, sh):
        v1 = gap_LHS_MC_gpu(sh, Nom, f_func_scalar_local(t), p.sigma, p.delta, inner_twin, device)
        v2 = gap_LHS_MC_gpu(sh, Nom, f_func_scalar_local(t), p.sigma, p.delta, inner_twin, device)
        return 0.5 * (v1 + v2)

    with T.timeit(f"Q2: Twin MC over grid (inner={inner_twin})"):
        Y_twin = np.array([twin_gpu(t, s) for t, s in zip(t_grid, Shat_t)])

    with T.timeit(f"Q2: Nested MC over grid (inner={inner_nested})"):
        Y_nested = np.array([
            gap_LHS_MC_gpu(s, Nom, f_func_scalar_local(t), p.sigma, p.delta, inner_nested, device)
            for t, s in zip(t_grid, Shat_t)
        ])

    # --- Two-panel figure: top = curves; bottom = relative error vs t ---
    fig = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # Top: explicit vs MC vs models
    ax_top.plot(t_grid, Y_true, lw=2, label="Explicit (Lemma)")
    ax_top.plot(t_grid, Y_twin, ls=":", label="Twin MC")
    ax_top.plot(t_grid, Y_nested, ls="--", label="Nested MC")
    ax_top.plot(t_grid, Y_pred_lin, label="Linear regression")
    ax_top.plot(t_grid, Y_pred_nn, label="Neural Net (linear)")
    ax_top.set_ylabel("Conditional positive gap")
    ax_top.grid(True); ax_top.legend(loc="best")
    ax_top.set_title("Swap: explicit vs MC vs models")

    # Bottom: relative error vs t (each series against explicit)
    def relerr_curve(y, ref):
        return (y - ref) / np.maximum(1e-12, np.abs(ref))

    r_twin   = relerr_curve(Y_twin,   Y_true)
    r_nested = relerr_curve(Y_nested, Y_true)
    r_lin    = relerr_curve(Y_pred_lin, Y_true)
    r_nn     = relerr_curve(Y_pred_nn,  Y_true)

    ax_bot.plot(t_grid, r_twin,  ls=":", label="Twin MC")
    ax_bot.plot(t_grid, r_nested,ls="--", label="Nested MC")
    ax_bot.plot(t_grid, r_lin,   label="Linear regression")
    ax_bot.plot(t_grid, r_nn,    label="Neural Net (linear)")
    ax_bot.axhline(0.0, color="k", lw=0.8)
    ax_bot.set_xlabel("t (years)")
    ax_bot.set_ylabel("Rel. error")
    ax_bot.grid(True)
    m = np.nanmax(np.abs(np.concatenate([r_twin, r_nested, r_lin, r_nn])))
    ax_bot.set_ylim(-1.1 * m, 1.1 * m)

    savefig_both(fig, args.figdir, "5_1_q2_swap_validation")

    # ---------------- Q3 ----------------
    banner("Q3 - CVA₀ (swap): quadrature vs models (OLS / Neural Net) vs MC (nested/twin)")
    label_swap_quad = "Q3: CVA0 swap - Quadrature (explicit)"
    with T.timeit(label_swap_quad):
        A_vals = A_func_vec(t_grid)
        integrand = A_vals * np.exp(-p.gamma1 * t_grid) * p.gamma1 * p.S0 * Nom
        CVA_quad = float(np.trapezoid(integrand, t_grid))
    print(f"CVA0 quadrature (swap): {swap_to_bps(CVA_quad):.4f} bps")

    # model-based (UNCONDITIONAL mean over τ; zeros when τ>T)
    np.random.seed(args.seed + 2)
    N_default = int(60000 * args.outer_scale)
    tau = -np.log(np.random.rand(N_default)) / p.gamma1
    mask = tau <= p.T

    label_swap_ols = "Q3: CVA0 swap - Linear regression model (OLS)"
    with T.timeit(label_swap_ols):
        Shat_tau = np.zeros_like(tau)
        t_tau = tau[mask]
        if t_tau.size > 0:
            Shat_tau_gpu = Shat_exact_gpu(t_tau, p.S0, p.sigma, device)
            Shat_tau[mask] = Shat_tau_gpu.cpu().numpy()
        idx = np.searchsorted(t_grid, tau[mask], side="left").clip(0, len(t_grid) - 1)
        g_all = np.zeros_like(tau)
        if idx.size > 0:
            g_all[mask] = alpha_hat[idx] * Shat_tau[mask]
        CVA_lin = float(g_all.mean())

    label_swap_nn = f"Q3: CVA0 swap - Neural Network model (linear layer, epochs={EPOCHS_Q1_NN})"
    with T.timeit(label_swap_nn):
        if mask.any():
            phi = torch.zeros((int(mask.sum()), len(t_grid)), dtype=torch.float32, device=device)
            row_idx = torch.arange(int(mask.sum()), device=device)
            with torch.no_grad():
                phi[row_idx, torch.tensor(idx, device=device)] = torch.tensor(Shat_tau[mask], device=device, dtype=torch.float32)
                g_cond_mean = model(phi).mean().item()
            CVA_nn = float(g_cond_mean * (mask.sum() / N_default))
        else:
            CVA_nn = 0.0

    # CVA₀ via nested & twin MC (swap) - unconditional with τ>T -> 0
    N_tau_swap = int(60000 * args.outer_scale)
    inner_swap_nested = max(2, int(2000 * args.inner_scale))
    inner_swap_twin = max(2, int(1000 * args.inner_scale))  # two evals averaged

    np.random.seed(args.seed + 20)
    tau_s = -np.log(np.random.rand(N_tau_swap)) / p.gamma1
    mask_s = tau_s <= p.T

    # draw Ŝ_{τ} for defaulted samples
    Shat_tau_swap = np.zeros_like(tau_s)
    if mask_s.any():
        with T.timeit("Q3: draw Shat_tau for swap (GPU)"):
            Shat_tau_swap_gpu = Shat_exact_gpu(tau_s[mask_s], p.S0, p.sigma, device)
            Shat_tau_swap[mask_s] = Shat_tau_swap_gpu.cpu().numpy()

    label_swap_nested = f"Q3: CVA0 swap - Nested MC (outer={N_tau_swap}, inner={inner_swap_nested})"
    with T.timeit(label_swap_nested):
        vals_nested = np.zeros(N_tau_swap, dtype=float)
        for i, t in enumerate(tau_s):
            if not mask_s[i]:
                vals_nested[i] = 0.0
            else:
                vals_nested[i] = gap_LHS_MC_gpu(Shat_tau_swap[i], Nom, f_func_scalar_local(t), p.sigma, p.delta, inner_swap_nested, device)
        CVA_swap_nested = float(vals_nested.mean())
        se_swap_nested = float(vals_nested.std(ddof=1) / math.sqrt(N_tau_swap))

    label_swap_twin = f"Q3: CVA0 swap - Twin MC (outer={N_tau_swap}, inner={inner_swap_twin}×2)"
    with T.timeit(label_swap_twin):
        vals_twin = np.zeros(N_tau_swap, dtype=float)
        for i, t in enumerate(tau_s):
            if not mask_s[i]:
                vals_twin[i] = 0.0
            else:
                a = gap_LHS_MC_gpu(Shat_tau_swap[i], Nom, f_func_scalar_local(t), p.sigma, p.delta, inner_swap_twin, device)
                b = gap_LHS_MC_gpu(Shat_tau_swap[i], Nom, f_func_scalar_local(t), p.sigma, p.delta, inner_swap_twin, device)
                vals_twin[i] = 0.5 * (a + b)
        CVA_swap_twin = float(vals_twin.mean())
        se_swap_twin = float(vals_twin.std(ddof=1) / math.sqrt(N_tau_swap))

    # ---------------- Q4 ----------------
    banner("Q4 - ATM Call: learn g(t,S) with Polynomial model vs Neural Network; validate vs MC")
    np.random.seed(args.seed + 3)
    N_outer = 800
    t_outer = np.random.uniform(0, p.T, size=N_outer)
    S_outer = S_exact_cpu(t_outer, p.S0, p.r, p.sigma)

    inner_call = max(2, int(2000 * args.inner_scale))
    with T.timeit(f"Q4: Generate call labels y_call (inner={inner_call})"):
        y_call = np.array([
            call_gap_pos_mc_gpu(t, s, p.T, p.S0, p.r, p.sigma, p.delta, inner_call, device) for t, s in zip(t_outer, S_outer)
        ])
    X_call = np.column_stack([S_outer, t_outer])

    # Poly deg-2 (CPU sklearn)
    with T.timeit("Q4: Polynomial deg-2 fit (CPU)"):
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X_call)
        reg_poly = LinearRegression().fit(X_poly, y_call)
        y_poly = reg_poly.predict(X_poly)
    print(f"[MODEL] PolynomialFeatures(degree=2, include_bias=True) + LinearRegression")
    print(f"Polynomial fit MAE: {mean_absolute_error(y_call, y_poly):.6e}")

    # NN (GPU)
    EPOCHS_Q4_NN = 1000
    with T.timeit("Q4: Neural Network (2×32×1) train (GPU)"):
        X_torch = torch.tensor(X_call, dtype=torch.float32, device=device)
        y_torch = torch.tensor(y_call, dtype=torch.float32, device=device).view(-1, 1)
        mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
        opt = optim.Adam(mlp.parameters(), lr=5e-3)
        for _ in range(EPOCHS_Q4_NN):
            opt.zero_grad()
            loss = nn.functional.mse_loss(mlp(X_torch), y_torch)
            loss.backward()
            opt.step()
    with T.timeit("Q4: Neural Network predict (train-set)"):
        with torch.no_grad():
            y_nn = mlp(X_torch).squeeze(1).cpu().numpy()
    nn_params_q4 = sum(p.numel() for p in mlp.parameters())
    print(f"[MODEL] Neural Network: {mlp}")
    print(f"[MODEL] Training: loss=MSE, optimizer=Adam lr=5e-3, epochs={EPOCHS_Q4_NN}, parameters={nn_params_q4}")
    print(f"Neural Network fit MAE: {mean_absolute_error(y_call, y_nn):.6e}")

    # Validation
    np.random.seed(args.seed + 4)
    N_val = 120
    t_val = np.linspace(0, p.T, N_val)
    S_val = S_exact_cpu(t_val, p.S0, p.r, p.sigma)
    inner_twin_call = max(2, int(1000 * args.inner_scale))
    with T.timeit(f"Q4: Twin MC validation (inner={inner_twin_call})"):
        y_twin = np.array([
            0.5 * (
                call_gap_pos_mc_gpu(t, s, p.T, p.S0, p.r, p.sigma, p.delta, inner_twin_call, device) +
                call_gap_pos_mc_gpu(t, s, p.T, p.S0, p.r, p.sigma, p.delta, inner_twin_call, device)
            ) for t, s in zip(t_val, S_val)
        ])
    with T.timeit(f"Q4: Nested MC validation (inner={max(2, int(2000 * args.inner_scale))})"):
        y_nested = np.array([
            call_gap_pos_mc_gpu(t, s, p.T, p.S0, p.r, p.sigma, p.delta, max(2, int(2000 * args.inner_scale)), device) for t, s in zip(t_val, S_val)
        ])

    with T.timeit("Q4: Polynomial predict (val)"):
        Xv = np.column_stack([S_val, t_val])
        y_poly_v = reg_poly.predict(poly.transform(Xv))
    with T.timeit("Q4: Neural Network predict (val)"):
        with torch.no_grad():
            y_nn_v = mlp(torch.tensor(Xv, dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()

    mae_poly_nested = mean_absolute_error(y_nested, y_poly_v)
    mae_nn_nested = mean_absolute_error(y_nested, y_nn_v)
    print(f"Polynomial vs nested MAE: {mae_poly_nested:.6e}")
    print(f"Neural Network vs nested MAE: {mae_nn_nested:.6e}")

    # Old overlaid plot kept for continuity (optional)
    fig = plt.figure()
    plt.plot(y_nested, lw=2, label="Nested MC")
    plt.plot(y_twin, ls="--", label="Twin MC")
    plt.plot(y_poly_v, label="Polynomial prediction")
    plt.plot(y_nn_v, label="Neural Network prediction")
    plt.legend(); plt.grid(True); plt.title("ATM Call: twin/nested vs predictors")
    savefig_both(fig, args.figdir, "5_1_q4_call_validation")

    # New diagnostics
    call_validation_plots(y_ref=y_nested, y_hat=y_poly_v,
                          t_val=t_val, S_val=S_val, S0=p.S0, model_name="Poly deg-2",
                          stem_prefix="5_1_q4_call_poly", T_horizon=p.T, figdir=args.figdir)

    call_validation_plots(y_ref=y_nested, y_hat=y_nn_v,
                          t_val=t_val, S_val=S_val, S0=p.S0, model_name="Neural Network",
                          stem_prefix="5_1_q4_call_nn", T_horizon=p.T, figdir=args.figdir)

    # ---------------- Q5 ----------------
    banner("Q5 - CVA₀ (ATM Call): model-based (Polynomial / Neural Network) vs MC (nested/twin)")
    np.random.seed(args.seed + 5)
    N_default_call = int(60000 * args.outer_scale)
    tau_c = -np.log(np.random.rand(N_default_call)) / p.gamma1
    mask_c = tau_c <= p.T
    S_tau = np.zeros_like(tau_c)
    if mask_c.any():
        S_tau[mask_c] = S_exact_cpu(tau_c[mask_c], p.S0, p.r, p.sigma)
    X_tau = np.column_stack([S_tau[mask_c], tau_c[mask_c]])

    label_call_poly = "Q5: CVA0 call - Polynomial model (deg-2)"
    with T.timeit(label_call_poly):
        plug_vals = np.zeros(N_default_call, dtype=float)
        if mask_c.any():
            plug_vals[mask_c] = reg_poly.predict(poly.transform(X_tau))
        CVA_call_poly = float(plug_vals.mean())

    EPOCHS_Q4_NN = 1000  # already defined; keep name in summary
    label_call_nn = f"Q5: CVA0 call - Neural Network model (2×32×1, epochs={EPOCHS_Q4_NN})"
    with T.timeit(label_call_nn):
        if mask_c.any():
            with torch.no_grad():
                cond_mean = float(mlp(torch.tensor(X_tau, dtype=torch.float32, device=device)).mean().item())
            CVA_call_nn = cond_mean * float(mask_c.mean())
        else:
            CVA_call_nn = 0.0

    inner_call_nested = max(2, int(2000 * args.inner_scale))
    inner_call_twin = max(2, int(1000 * args.inner_scale))  # two evals averaged

    label_call_nested = f"Q5: CVA0 call - Nested MC (outer={N_default_call}, inner={inner_call_nested})"
    with T.timeit(label_call_nested):
        vals_call_nested = np.zeros(N_default_call, dtype=float)
        for i, t in enumerate(tau_c):
            if not mask_c[i]:
                vals_call_nested[i] = 0.0
            else:
                vals_call_nested[i] = call_gap_pos_mc_gpu(t, S_tau[i], p.T, p.S0, p.r, p.sigma, p.delta, inner_call_nested, device)
        CVA_call_nested = float(vals_call_nested.mean())
        se_call_nested = float(vals_call_nested.std(ddof=1) / math.sqrt(N_default_call))

    label_call_twin = f"Q5: CVA0 call - Twin MC (outer={N_default_call}, inner={inner_call_twin}×2)"
    with T.timeit(label_call_twin):
        vals_call_twin = np.zeros(N_default_call, dtype=float)
        for i, t in enumerate(tau_c):
            if not mask_c[i]:
                vals_call_twin[i] = 0.0
            else:
                a = call_gap_pos_mc_gpu(t, S_tau[i], p.T, p.S0, p.r, p.sigma, p.delta, inner_call_twin, device)
                b = call_gap_pos_mc_gpu(t, S_tau[i], p.T, p.S0, p.r, p.sigma, p.delta, inner_call_twin, device)
                vals_call_twin[i] = 0.5 * (a + b)
        CVA_call_twin = float(vals_call_twin.mean())
        se_call_twin = float(vals_call_twin.std(ddof=1) / math.sqrt(N_default_call))

    # ---------------- Summaries (in bps) ----------------
    banner("SUMMARY - CVA₀ Results (bps)")
    print("Definitions: swap bps = 10,000 × CVA₀ on the SAME calculated nominal (no rescaling); "
          "call bps = 10,000 × (CVA₀ / S0).")

    # SWAP summary table (reference = quadrature)
    swap_ref = CVA_quad
    swap_rows = [
        ["Quadrature (explicit)", f"{swap_to_bps(CVA_quad):.4f} bps", "-", "-"],
        ["Linear regression (OLS)", f"{swap_to_bps(CVA_lin):.4f} bps", "-", f"{relerr(CVA_lin, swap_ref):.2%}"],
        [f"Neural Network (linear layer, epochs={EPOCHS_Q1_NN})", f"{swap_to_bps(CVA_nn):.4f} bps",  "-", f"{relerr(CVA_nn,  swap_ref):.2%}"],
        ["Nested MC",              fmt_pm(swap_to_bps(CVA_swap_nested), swap_to_bps(se_swap_nested)), f"{swap_to_bps(se_swap_nested):.4f} bps", f"{relerr(CVA_swap_nested, swap_ref):.2%}"],
        ["Twin MC",                fmt_pm(swap_to_bps(CVA_swap_twin),   swap_to_bps(se_swap_twin)),   f"{swap_to_bps(se_swap_twin):.4f} bps",   f"{relerr(CVA_swap_twin,   swap_ref):.2%}"],
    ]
    print("SWAP (linear exposure) - CVA₀ in bps on calculated nominal")
    print_table(swap_rows, header=["Method", "CVA₀ (bps)", "StdErr (bps)", "RelErr vs Ref"])

    # CALL summary table (reference = Nested MC)
    call_ref = CVA_call_nested
    call_rows = [
        ["Nested MC (ref)",        fmt_pm(call_to_bps(CVA_call_nested * 1.0, p.S0), call_to_bps(se_call_nested * 1.0, p.S0)), f"{call_to_bps(se_call_nested, p.S0):.4f} bps", "-"],  # using same fmt on S0-normalized below
        ["Twin MC",                fmt_pm(call_to_bps(CVA_call_twin   * 1.0, p.S0), call_to_bps(se_call_twin   * 1.0, p.S0)), f"{call_to_bps(se_call_twin, p.S0):.4f} bps",   f"{relerr(CVA_call_twin, call_ref):.2%}"],
        ["Polynomial model (deg-2)", f"{call_to_bps(CVA_call_poly, p.S0):.4f} bps", "-", f"{relerr(CVA_call_poly, call_ref):.2%}"],
        [f"Neural Network (2×32×1, epochs={EPOCHS_Q4_NN})", f"{call_to_bps(CVA_call_nn, p.S0):.4f} bps",   "-", f"{relerr(CVA_call_nn,   call_ref):.2%}"],
    ]
    print("\nATM CALL (non-linear exposure) - CVA₀ in bps of S0")
    print_table(call_rows, header=["Method", "CVA₀ (bps)", "StdErr (bps)", "RelErr vs Ref"])


    # PASS/Check flags (5% tolerance against respective references)
    tol_rel = 0.05
    print("\nPASS/Check:")
    print("Swap OLS  ", "PASS" if relerr(CVA_lin, swap_ref) < tol_rel else "CHECK")
    print("Swap NN   ", "PASS" if relerr(CVA_nn,  swap_ref) < tol_rel else "CHECK")
    print("Swap Nest ", "PASS" if relerr(CVA_swap_nested, swap_ref) < tol_rel else "CHECK")
    print("Swap Twin ", "PASS" if relerr(CVA_swap_twin,   swap_ref) < tol_rel else "CHECK")
    print("Call Poly ", "PASS" if relerr(CVA_call_poly, call_ref) < tol_rel else "CHECK")
    print("Call NN   ", "PASS" if relerr(CVA_call_nn,   call_ref) < tol_rel else "CHECK")
    print("Call Twin ", "PASS" if relerr(CVA_call_twin, call_ref) < tol_rel else "CHECK")

    # ---------------- Per-calculation COST tables ----------------
    banner("COST - Swap CVA₀ methods")
    swap_labels = [label_swap_quad, label_swap_ols, label_swap_nn, label_swap_nested, label_swap_twin]
    swap_items = [(lab, T.times.get(lab, 0.0)) for lab in swap_labels]
    swap_total = sum(t for _, t in swap_items) or 1.0
    swap_rows_cost = []
    for lab, sec in swap_items:
        pct = 100.0 * sec / swap_total
        swap_rows_cost.append([lab.replace("Q3: ", ""), f"{sec:.4f} s", f"{pct:.1f}%"])
    print_table(swap_rows_cost, header=["Method (swap)", "Wall Time", "Share of swap-CVA"])

    banner("COST - ATM Call CVA₀ methods")
    call_labels = [label_call_poly, label_call_nn, label_call_nested, label_call_twin]
    call_items = [(lab, T.times.get(lab, 0.0)) for lab in call_labels]
    call_total = sum(t for _, t in call_items) or 1.0
    call_rows_cost = []
    for lab, sec in call_items:
        pct = 100.0 * sec / call_total
        call_rows_cost.append([lab.replace("Q5: ", ""), f"{sec:.4f} s", f"{pct:.1f}%"])
    print_table(call_rows_cost, header=["Method (ATM call)", "Wall Time", "Share of call-CVA"])

    # ---------------- Global timing summary ----------------
    banner("TIMING - Breakdown (all measured blocks)")
    total_t = sum(T.times.values()) if T.times else 0.0
    items = sorted(T.times.items(), key=lambda kv: kv[1], reverse=True)
    rows = [(lab, f"{sec:.4f} s", f"{(sec/total_t*100 if total_t>0 else 0):.1f}%") for lab, sec in items]
    print_table(rows, header=["Block", "Wall Time", "Share"])
    print(f"\nTotal timed wall clock: {total_t:.4f} s")

    print("\n[DONE] 5.1 CVA without RIM script finished.")

if __name__ == "__main__":
    main()
