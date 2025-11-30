#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3 CVA with RIM - v3.1 (modular)
- EPS+PNG figures
- Timing harness & per-product cost tables
- Uses shared helpers from xva_core/
"""

import os, math, argparse
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error

# ---- shared library (local package) ----
from xva_core import (
    Timer, banner, savefig_both, print_table, relerr, fmt_pm_bps, trapz_weights,
    fair_strike_and_nominal, precompute_weights, f_func_vec, f_func_scalar,
    Shat_exact_cpu, S_exact_cpu, select_device, seed_all, pinball,
    draw_U, call_bs_np, Params, swap_to_bps, call_to_bps
)

# ----------------------------- Helpers (script-specific) -----------------------------
def C_explicit_noNom_noS(t_arr, a_conf: float, sigma: float, delta: float, f_func_handle):
    """
    α_C/Nom/Ŝ_t = f(t) * E[(U - q_a)^+], where U = e^{σ√δ Z - ½σ²δ} - 1.
    Closed-form tail term for RIM (see article/derivation).
    """
    t_arr = np.atleast_1d(t_arr)
    z = norm.ppf(a_conf)
    term = np.exp(sigma*np.sqrt(delta)*norm.pdf(z)/(1-a_conf)) - np.exp(sigma*np.sqrt(delta)*z)
    return (1-a_conf) * f_func_handle(t_arr) * np.exp(-0.5*sigma**2*delta) * term

def X_call_samples(t: float, S_t: float, n: int, r: float, sigma: float, delta: float, T: float, K: float):
    Z = np.random.normal(size=n)
    S_td = S_t * np.exp((r - 0.5 * sigma**2) * delta + sigma * np.sqrt(delta) * Z)
    beta_t = np.exp(r * t); beta_td = np.exp(r * (t + delta))
    c_t = call_bs_np(t, S_t, T, K, r, sigma)
    c_td = call_bs_np(t + delta, S_td, T, K, r, sigma)
    return beta_td * c_td - beta_t * c_t


def _ppf_torch(u: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

def _cdf_torch(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(description="XVA 5.3 CVA with RIM - v3.1 (modular)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--figdir", type=str, default="figs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--inner_scale", type=float, default=1.0)
    parser.add_argument("--outer_scale", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.99)
    # multi-α (Q6)
    parser.add_argument("--a_min", type=float, default=0.001)
    parser.add_argument("--a_max", type=float, default=0.15)
    parser.add_argument("--q6_nodes", type=int, default=25)
    parser.add_argument("--q6_steps", type=int, default=1500)
    parser.add_argument("--q6_batch", type=int, default=4096)
    parser.add_argument("--q6_innerU", type=int, default=512)
    args = parser.parse_args()

    seed_all(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    # Timer (+ optional CUDA warmup)
    T = Timer(device)
    if getattr(device, "type", "") == "cuda":
        _ = torch.randn(1, device=device)
        torch.cuda.synchronize()

    # Setup
    p = Params()
    payment_times = np.arange(p.h, p.T + 1e-12, p.h)
    print(f"Num payments: {len(payment_times)}  |  First {payment_times[0]:.2f}  Last {payment_times[-1]:.2f}")

    barS, Nom = fair_strike_and_nominal(p.S0, p.r, p.kappa, p.h, payment_times)
    print(f"barS = {barS:.6f} | Nom = {Nom:.6f}")

    w, w_suffix = precompute_weights(p.r, p.kappa, p.h, payment_times)
    def f_func_handle(t_arr):
        return f_func_vec(t_arr, payment_times, w_suffix, p.delta)

    t_grid = np.arange(0.0, p.T + 1e-12, p.delta)
    Shat_t = Shat_exact_cpu(t_grid, p.S0, p.sigma)

    # ---------------- Q1 ----------------
    banner("Q1 - Proofs (Lemma 4.1 & Prop. 4.1)")
    print("Sketch: E[(X - RIM)^+] = Nom * α_C(t) * Ŝ_t; CVA⁽RIM⁾ via ∫ γ e^{-γ t} α_C(t) Ŝ_t dt. (See notes.)")

    # ---------------- Q2 ----------------
    banner("Q2 - Linear quantile model for q_a(U) + ES; build α_C(t)")
    a_conf = args.a
    f_all = f_func_handle(t_grid)
    x_full = Nom * f_all
    mask = f_all > 1e-14
    x_eff = x_full[mask]
    n_eff = len(x_eff)

    n_inner = max(2, int(100000 * args.inner_scale))
    with T.timeit(f"Q2: draw U samples (n_inner={n_inner})"):
        U_samps = draw_U(n_inner, p.sigma, p.delta)

    # Learn q_a(U) with 1-param pinball: q_a(X_t) = w * x_t
    EPOCHS_PINBALL = 1000
    wq = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))
    x_eff_t = torch.tensor(x_eff, dtype=torch.float32, device=device)
    U_t = torch.tensor(U_samps, dtype=torch.float32, device=device)

    with T.timeit("Q2: train pinball linear quantile (q_a(X)=w·x)"):
        opt = torch.optim.Adam([wq], lr=2e-2)
        for _ in range(EPOCHS_PINBALL):
            opt.zero_grad()
            X = x_eff_t.view(-1, 1) * U_t.view(1, -1)
            qhat = (wq * x_eff_t).view(-1, 1)
            loss = pinball(qhat, X, a_conf)
            loss.backward()
            opt.step()
    qU_pinball = float(wq.detach().cpu().item())

    with T.timeit("Q2: estimate q_U and tail mean from U samples"):
        qU_hat = float(np.quantile(U_samps, a_conf, method="linear"))
        tail_mean_U = float(np.mean(np.maximum(U_samps - qU_hat, 0.0)))

    with T.timeit("Q2: build α_C(t) and compare to explicit"):
        alphaC_hat  = x_full * tail_mean_U
        alphaC_true = Nom * C_explicit_noNom_noS(t_grid, a_conf, p.sigma, p.delta, f_func_handle)
        mae_alphaC  = float(np.mean(np.abs(alphaC_hat - alphaC_true)))
    print(f"[CHECK Q2] Mean |α_C_hat - α_C_true| = {mae_alphaC:.6e}  "
          f"(q_U^pinball={qU_pinball:.6e}, q_U^emp={qU_hat:.6e})")
    print(f"[MODEL] Linear quantile: params=1, Adam lr=2e-2, epochs={EPOCHS_PINBALL}")

    # ---------------- Q3 ----------------
    banner("Q3 - Validation over t: explicit vs twin/nested MC vs learner")
    def nested_E_rim(t, Sh, n=20000):
        U = draw_U(n, p.sigma, p.delta)
        f_t = f_func_scalar(t, payment_times, w_suffix, p.delta)
        X = Nom * f_t * Sh * U
        qU = np.quantile(U, a_conf, method="linear")
        qX = Nom * f_t * Sh * qU
        return np.maximum(X - qX, 0.0).mean()
    def twin_E_rim(t, Sh, n_small=6000):
        return 0.5 * (nested_E_rim(t, Sh, n_small) + nested_E_rim(t, Sh, n_small))

    Y_explicit = alphaC_true * Shat_t
    Y_learner  = alphaC_hat * Shat_t
    with T.timeit(f"Q3: Twin MC over grid"):
        Y_twin = np.array([twin_E_rim(t, s, n_small=max(1, int(6000 * args.inner_scale)))
                           for t, s in zip(t_grid, Shat_t)])
    with T.timeit(f"Q3: Nested MC over grid"):
        Y_nested = np.array([nested_E_rim(t, s, n=max(1, int(6000 * args.inner_scale)))
                             for t, s in zip(t_grid, Shat_t)])

    fig = plt.figure()
    plt.plot(t_grid, Y_explicit, lw=2, label="Explicit")
    plt.plot(t_grid, Y_twin, linestyle=":", label="Twin MC")
    plt.plot(t_grid, Y_nested, linestyle="--", label="Nested MC")
    plt.plot(t_grid, Y_learner, label="Linear quantile learner")
    plt.xlabel("t (years)"); plt.ylabel("E_t[(X - RIM)^+]"); plt.grid(True); plt.legend()
    plt.title("Swap with RIM - explicit vs MC vs learner")
    savefig_both(fig, args.figdir, "5_3_q3_rim_swap_validation")

    # ---------------- Q4 ----------------
    banner("Q4 - CVA⁽RIM⁾₀ (Swap): quadrature vs learner + MC (nested/twin)")
    label_swap_quad = "Q4: Swap CVA^RIM - Quadrature (explicit)"
    with T.timeit(label_swap_quad):
        C_vals = C_explicit_noNom_noS(t_grid, a_conf, p.sigma, p.delta, f_func_handle)
        integrand = C_vals * np.exp(-p.gamma1 * t_grid) * p.gamma1
        CVA_rim_swap_quad = Nom * p.S0 * np.trapezoid(integrand, t_grid)

    label_swap_learned = "Q4: Swap CVA^RIM - Linear quantile learner"
    with T.timeit(label_swap_learned):
        integrand_hat = alphaC_hat * (p.gamma1 * np.exp(-p.gamma1 * t_grid))
        CVA_rim_swap_learned = p.S0 * np.trapezoid(integrand_hat, t_grid)

    # MC (time integration by trapezoid weights)
    w_dt = trapz_weights(t_grid) * (p.gamma1 * np.exp(-p.gamma1 * t_grid))

    def one_rep_swap_arrays(seed_shift: int, n_small=4000, n_big=12000):
        rng = np.random.default_rng(args.seed + seed_shift)
        Sh = Shat_exact_cpu(t_grid, p.S0, p.sigma)
        Yn = np.zeros_like(t_grid); Yt = np.zeros_like(t_grid)
        for k, t in enumerate(t_grid):
            f_t = f_func_scalar(t, payment_times, w_suffix, p.delta)
            if f_t <= 1e-14:
                Yn[k]=0.0; Yt[k]=0.0; continue
            sh = Sh[k]
            # Nested
            Ub = rng.standard_normal(n_big)
            U_big = np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * Ub) - 1.0
            q_big = np.quantile(U_big, a_conf, method="linear")
            Xb = Nom * f_t * sh * U_big
            Yn[k] = np.maximum(Xb - Nom*f_t*sh*q_big, 0.0).mean()
            # Twin
            U1 = np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * rng.standard_normal(n_small)) - 1.0
            U2 = np.exp(-0.5 * p.sigma**2 * p.delta + p.sigma * np.sqrt(p.delta) * rng.standard_normal(n_small)) - 1.0
            q1 = np.quantile(U1, a_conf, method="linear")
            q2 = np.quantile(U2, a_conf, method="linear")
            Y1 = np.maximum(Nom*f_t*sh*U1 - Nom*f_t*sh*q1, 0.0).mean()
            Y2 = np.maximum(Nom*f_t*sh*U2 - Nom*f_t*sh*q2, 0.0).mean()
            Yt[k] = 0.5 * (Y1 + Y2)
        return Yn, Yt

    n_small_m = max(2, int(4000 * args.inner_scale))
    n_big_m   = max(2, int(12000 * args.inner_scale))
    label_swap_nested_mc = f"Q4: Swap CVA^RIM - Nested MC (inner={n_big_m})"
    with T.timeit(label_swap_nested_mc):
        Yn1,_ = one_rep_swap_arrays(101, n_small=n_small_m, n_big=n_big_m)
        Yn2,_ = one_rep_swap_arrays(202, n_small=n_small_m, n_big=n_big_m)
        CVA_rim_swap_nested_1 = float(np.sum(w_dt * Yn1))
        CVA_rim_swap_nested_2 = float(np.sum(w_dt * Yn2))
        CVA_rim_swap_nested = 0.5*(CVA_rim_swap_nested_1 + CVA_rim_swap_nested_2)
        se_swap_nested = abs(CVA_rim_swap_nested_1 - CVA_rim_swap_nested_2)/math.sqrt(2.0)

    label_swap_twin_mc = f"Q4: Swap CVA^RIM - Twin MC (inner={n_small_m}×2)"
    with T.timeit(label_swap_twin_mc):
        _,Yt1 = one_rep_swap_arrays(303, n_small=n_small_m, n_big=n_big_m)
        _,Yt2 = one_rep_swap_arrays(404, n_small=n_small_m, n_big=n_big_m)
        CVA_rim_swap_twin_1 = float(np.sum(w_dt * Yt1))
        CVA_rim_swap_twin_2 = float(np.sum(w_dt * Yt2))
        CVA_rim_swap_twin = 0.5*(CVA_rim_swap_twin_1 + CVA_rim_swap_twin_2)
        se_swap_twin = abs(CVA_rim_swap_twin_1 - CVA_rim_swap_twin_2)/math.sqrt(2.0)

    # ---------------- Q5 ----------------
    banner("Q5 - ATM call with RIM: scaled Polynomial / Neural Network + validation")

    np.random.seed(args.seed + 3)
    N_outer = 800
    t_outer = np.random.uniform(0, p.T, size=N_outer)
    S_outer = S_exact_cpu(t_outer, p.S0, p.r, p.sigma)
    n_inn = max(1, int(1500 * args.inner_scale))

    with T.timeit(f"Q5: Build nested labels (outer={N_outer}, inner={n_inn})"):
        q_lbl = np.zeros(N_outer); es_lbl = np.zeros(N_outer)
        for i, (t, Sv) in enumerate(zip(t_outer, S_outer)):
            Xs = X_call_samples(t, Sv, n_inn, p.r, p.sigma, p.delta, p.T, p.S0)
            q_lbl[i] = np.quantile(Xs, a_conf, method="linear")
            tail = Xs[Xs >= q_lbl[i]]
            es_lbl[i] = tail.mean() if tail.size > 0 else q_lbl[i]
    y_lbl = (1 - a_conf) * (es_lbl - q_lbl)  # = E[(X-q)^+]

    X_base_train = np.column_stack([S_outer / p.S0, t_outer / p.T])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_base_train)
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)

    # scale labels
    y_mu, y_sig = float(y_lbl.mean()), float(y_lbl.std() if y_lbl.std() > 1e-12 else 1.0)
    yy_t = torch.tensor((y_lbl - y_mu) / y_sig, dtype=torch.float32)
    Xp = torch.tensor(X_poly, dtype=torch.float32)

    EPOCHS_POLY = 1000
    with T.timeit("Q5: Train polynomial (deg-2, scaled) for E[(X-q)^+]"):
        w_poly = torch.nn.Parameter(torch.zeros((Xp.shape[1], 1), dtype=torch.float32))
        opt_p = torch.optim.Adam([w_poly], lr=3e-3)
        for _ in range(EPOCHS_POLY):
            opt_p.zero_grad()
            yhat_s = Xp @ w_poly
            loss = torch.mean((yhat_s.squeeze(1) - yy_t)**2)
            loss.backward()
            opt_p.step()
    with torch.no_grad():
        yhat_poly_s = (Xp @ w_poly).numpy().ravel()
    yhat_poly = yhat_poly_s * y_sig + y_mu
    print(f"[Q5] Polynomial (deg-2, scaled) MAE vs labels: {mean_absolute_error(y_lbl, yhat_poly):.6e}")

    # NN (2×32×1)
    EPOCHS_NN = 1000
    with T.timeit("Q5: Train Neural Network (2×32×1) for E[(X-q)^+]"):
        X_torch = torch.tensor(X_base_train, dtype=torch.float32)
        y_target_s = yy_t
        mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        opt_mlp = torch.optim.Adam(mlp.parameters(), lr=5e-3)
        for _ in range(EPOCHS_NN):
            opt_mlp.zero_grad()
            yhat_s = mlp(X_torch).squeeze(1)
            loss = torch.mean((yhat_s - y_target_s)**2)
            loss.backward()
            opt_mlp.step()
    with torch.no_grad():
        yhat_nn_s = mlp(X_torch).squeeze(1).numpy()
    yhat_nn = yhat_nn_s * y_sig + y_mu
    print(f"[Q5] Neural Network MAE vs labels: {mean_absolute_error(y_lbl, yhat_nn):.6e}")
    nn_params = sum(p.numel() for p in mlp.parameters())
    print(f"[MODEL] Neural Network: {mlp}")
    print(f"[MODEL] Training: loss=MSE, Adam lr=5e-3, epochs={EPOCHS_NN}, parameters={nn_params}")

    # Validation twin vs nested
    np.random.seed(args.seed + 4)
    N_val = 120
    t_val = np.linspace(0, p.T, N_val)
    S_val = S_exact_cpu(t_val, p.S0, p.r, p.sigma)

    def Y_call_nested(t, S, n_small=1200, n_big=5000):
        X1 = X_call_samples(t, S, n_small, p.r, p.sigma, p.delta, p.T, p.S0)
        X2 = X_call_samples(t, S, n_small, p.r, p.sigma, p.delta, p.T, p.S0)
        q1 = np.quantile(X1, a_conf, method="linear")
        q2 = np.quantile(X2, a_conf, method="linear")
        twin = 0.5 * (np.maximum(X1 - q1, 0.0).mean() + np.maximum(X2 - q2, 0.0).mean())
        Xb = X_call_samples(t, S, n_big, p.r, p.sigma, p.delta, p.T, p.S0)
        qb = np.quantile(Xb, a_conf, method="linear")
        nested = np.maximum(Xb - qb, 0.0).mean()
        return twin, nested

    with T.timeit("Q5: Validation twin/nested MC"):
        y_twin = np.zeros(N_val); y_nested = np.zeros(N_val)
        for i, (t, Sv) in enumerate(zip(t_val, S_val)):
            y_twin[i], y_nested[i] = Y_call_nested(t, Sv,
                                                   n_small=max(1, int(1200 * args.inner_scale)),
                                                   n_big=max(1, int(5000 * args.inner_scale)))
    # Predictors on validation
    Xv_base = np.column_stack([S_val / p.S0, t_val / p.T])
    Xv_scaled = scaler.transform(Xv_base)
    Xv_poly = poly.transform(Xv_scaled)
    with torch.no_grad():
        y_poly_v = (torch.tensor(Xv_poly, dtype=torch.float32) @ w_poly).numpy().ravel() * y_sig + y_mu
        y_nn_v   = mlp(torch.tensor(Xv_base, dtype=torch.float32)).squeeze(1).numpy() * y_sig + y_mu

    fig = plt.figure()
    plt.plot(y_nested, lw=2, label="Nested")
    plt.plot(y_twin, linestyle=":", label="Twin")
    plt.plot(y_poly_v, label="Polynomial (deg-2, scaled)")
    plt.plot(y_nn_v, label="Neural Network (2×32×1)")
    plt.legend(); plt.grid(True); plt.title("ATM call with RIM - nested/twin vs predictors")
    savefig_both(fig, args.figdir, "5_3_q5_call_rim_scaled")

    print("Poly vs nested MAE:", mean_absolute_error(y_nested, y_poly_v))
    print(" NN  vs nested MAE:", mean_absolute_error(y_nested, y_nn_v))

    # ---------------- Q5b ----------------
    banner("Q5b - CVA⁽RIM⁾₀ (ATM Call): ML models & MC (Nested/Twin)")
    N_outer_S = max(50, int(250 * args.outer_scale))
    n_inner_call_nested = max(500, int(2000 * args.inner_scale))
    n_inner_call_twin   = max(250, int(1000 * args.inner_scale))
    w_call = trapz_weights(t_grid) * (p.gamma1 * np.exp(-p.gamma1 * t_grid))

    # Nested
    label_call_nested_mc = f"Q5b: Call CVA^RIM - Nested MC (S_outer={N_outer_S}, inner={n_inner_call_nested})"
    with T.timeit(label_call_nested_mc):
        m_t = np.zeros_like(t_grid); se_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S)
            Z = np.random.normal(size=(N_outer_S, n_inner_call_nested))
            S_td = Ss[:, None] * np.exp((p.r - 0.5 * p.sigma**2) * p.delta + p.sigma * math.sqrt(p.delta) * Z)
            beta_t = math.exp(p.r * t); beta_td = math.exp(p.r * (t + p.delta))
            c_t = call_bs_np(t, Ss, p.T, p.S0, p.r, p.sigma)[:, None]
            c_td = call_bs_np(t + p.delta, S_td, p.T, p.S0, p.r, p.sigma)
            Xs = beta_td * c_td - beta_t * c_t
            q = np.quantile(Xs, a_conf, axis=1, method="linear")
            vals = np.maximum(Xs - q[:, None], 0.0).mean(axis=1)
            m_t[idx] = float(np.mean(vals))
            se_t[idx] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        CVA_rim_call_nested = float(np.sum(w_call * m_t))
        se_call_nested = float(np.sqrt(np.sum((w_call * se_t)**2)))

    # Twin
    label_call_twin_mc = f"Q5b: Call CVA^RIM - Twin MC (S_outer={N_outer_S}, inner={n_inner_call_twin}×2)"
    with T.timeit(label_call_twin_mc):
        m_t = np.zeros_like(t_grid); se_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S)
            def one_vals(n_inner):
                Z = np.random.normal(size=(N_outer_S, n_inner))
                S_td = Ss[:, None] * np.exp((p.r - 0.5 * p.sigma**2) * p.delta + p.sigma * math.sqrt(p.delta) * Z)
                beta_t = math.exp(p.r * t); beta_td = math.exp(p.r * (t + p.delta))
                c_t = call_bs_np(t, Ss, p.T, p.S0, p.r, p.sigma)[:, None]
                c_td = call_bs_np(t + p.delta, S_td, p.T, p.S0, p.r, p.sigma)
                Xs = beta_td * c_td - beta_t * c_t
                q = np.quantile(Xs, a_conf, axis=1, method="linear")
                return np.maximum(Xs - q[:, None], 0.0).mean(axis=1)
            v1 = one_vals(n_inner_call_twin); v2 = one_vals(n_inner_call_twin)
            v = 0.5*(v1 + v2)
            m_t[idx] = float(np.mean(v))
            se_t[idx] = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
        CVA_rim_call_twin = float(np.sum(w_call * m_t))
        se_call_twin = float(np.sqrt(np.sum((w_call * se_t)**2)))

    # Model-based (plug-in)
    N_outer_S_fast = max(200, int(1000 * args.outer_scale))
    label_call_poly = f"Q5b: Call CVA^RIM - Polynomial (deg-2, S_outer={N_outer_S_fast})"
    with T.timeit(label_call_poly):
        m_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S_fast)
            Xs_base = np.column_stack([Ss / p.S0, np.full_like(Ss, t / p.T)])
            Xs_scaled = scaler.transform(Xs_base)
            Xs_poly = poly.transform(Xs_scaled)
            with torch.no_grad():
                preds = (torch.tensor(Xs_poly, dtype=torch.float32) @ w_poly).numpy().ravel() * y_sig + y_mu
            m_t[idx] = float(np.mean(np.maximum(preds, 0.0)))
        CVA_rim_call_poly = float(np.sum(w_call * m_t))

    label_call_nn = f"Q5b: Call CVA^RIM - Neural Network (2×32×1, epochs={EPOCHS_NN}, S_outer={N_outer_S_fast})"
    with T.timeit(label_call_nn):
        m_t = np.zeros_like(t_grid)
        for idx, t in enumerate(t_grid):
            Ss = S_exact_cpu(t, p.S0, p.r, p.sigma, size=N_outer_S_fast)
            Xs_base = np.column_stack([Ss / p.S0, np.full_like(Ss, t / p.T)])
            with torch.no_grad():
                preds = mlp(torch.tensor(Xs_base, dtype=torch.float32)).squeeze(1).numpy() * y_sig + y_mu
            m_t[idx] = float(np.mean(np.maximum(preds, 0.0)))
        CVA_rim_call_nn = float(np.sum(w_call * m_t))

    # ---------------- Summary (bps) ----------------
    banner("SUMMARY - CVA⁽RIM⁾₀ Results (bps)")
    print("swap bps = 10,000 × CVA⁽RIM⁾₀ on the SAME calculated nominal; call bps = 10,000 × CVA⁽RIM⁾₀ / S0.\n")

    swap_ref = CVA_rim_swap_quad
    swap_rows = [
        ["Quadrature (explicit)", f"{swap_to_bps(CVA_rim_swap_quad):.4f} bps", "-", "-"],
        ["Linear quantile learner",  f"{swap_to_bps(CVA_rim_swap_learned):.4f} bps", "-", f"{relerr(CVA_rim_swap_learned, swap_ref):.2%}"],
        ["Nested MC", fmt_pm_bps(swap_to_bps(CVA_rim_swap_nested), swap_to_bps(se_swap_nested)), f"{swap_to_bps(se_swap_nested):.4f} bps", f"{relerr(CVA_rim_swap_nested, swap_ref):.2%}"],
        ["Twin MC",   fmt_pm_bps(swap_to_bps(CVA_rim_swap_twin),   swap_to_bps(se_swap_twin)),   f"{swap_to_bps(se_swap_twin):.4f} bps",   f"{relerr(CVA_rim_swap_twin,   swap_ref):.2%}"],
    ]
    print("SWAP (linear exposure) - CVA⁽RIM⁾₀ in bps on calculated nominal")
    print_table(swap_rows, header=["Method", "CVA⁽RIM⁾₀ (bps)", "StdErr (bps)", "RelErr vs Ref"])

    call_ref = CVA_rim_call_nested
    call_rows = [
        ["Nested MC (ref)", fmt_pm_bps(call_to_bps(CVA_rim_call_nested, p.S0), call_to_bps(se_call_nested, p.S0)), f"{call_to_bps(se_call_nested, p.S0):.4f} bps", "-"],
        ["Twin MC",         fmt_pm_bps(call_to_bps(CVA_rim_call_twin, p.S0),   call_to_bps(se_call_twin, p.S0)),   f"{call_to_bps(se_call_twin, p.S0):.4f} bps",   f"{relerr(CVA_rim_call_twin, call_ref):.2%}"],
        ["Polynomial (deg-2, scaled)", f"{call_to_bps(CVA_rim_call_poly, p.S0):.4f} bps", "-", f"{relerr(CVA_rim_call_poly, call_ref):.2%}"],
        [f"Neural Network (2×32×1, epochs={EPOCHS_NN})", f"{call_to_bps(CVA_rim_call_nn, p.S0):.4f} bps", "-", f"{relerr(CVA_rim_call_nn, call_ref):.2%}"],
    ]
    print("\nATM CALL (non-linear exposure) - CVA⁽RIM⁾₀ in bps of S0")
    print_table(call_rows, header=["Method", "CVA⁽RIM⁾₀ (bps)", "StdErr (bps)", "RelErr vs Ref"])

    tol_rel = 0.05
    print("\nPASS/Check:")
    print("Swap Learned", "PASS" if relerr(CVA_rim_swap_learned, swap_ref) < tol_rel else "CHECK")
    print("Swap Nested ", "PASS" if relerr(CVA_rim_swap_nested,  swap_ref) < tol_rel else "CHECK")
    print("Swap Twin   ", "PASS" if relerr(CVA_rim_swap_twin,    swap_ref) < tol_rel else "CHECK")
    print("Call Poly   ", "PASS" if relerr(CVA_rim_call_poly,    call_ref) < tol_rel else "CHECK")
    print("Call NN     ", "PASS" if relerr(CVA_rim_call_nn,      call_ref) < tol_rel else "CHECK")
    print("Call Twin   ", "PASS" if relerr(CVA_rim_call_twin,    call_ref) < tol_rel else "CHECK")

    # ---------------- Cost tables ----------------
    banner("COST - Swap CVA⁽RIM⁾₀ methods")
    swap_labels = [label_swap_quad, label_swap_learned, label_swap_nested_mc, label_swap_twin_mc]
    swap_items = [(lab, T.times.get(lab, 0.0)) for lab in swap_labels]
    swap_total = sum(t for _, t in swap_items) or 1.0
    print_table([[lab.replace("Q4: ", ""), f"{sec:.4f} s", f"{(100*sec/swap_total):.1f}%"] for lab, sec in swap_items],
                header=["Method (Swap CVA^RIM)", "Wall Time", "Share"])

    banner("COST - ATM Call CVA⁽RIM⁾₀ methods")
    call_labels = [label_call_poly, label_call_nn, label_call_nested_mc, label_call_twin_mc]
    call_items = [(lab, T.times.get(lab, 0.0)) for lab in call_labels]
    call_total = sum(t for _, t in call_items) or 1.0
    print_table([[lab.replace("Q5b: ", ""), f"{sec:.4f} s", f"{(100*sec/call_total):.1f}%"] for lab, sec in call_items],
                header=["Method (Call CVA^RIM)", "Wall Time", "Share"])

    # ---------------- Timing breakdown ----------------
    banner("TIMING - Breakdown (all measured blocks)")
    total_t = sum(T.times.values()) if T.times else 0.0
    items = sorted(T.times.items(), key=lambda kv: kv[1], reverse=True)
    rows = [(lab, f"{sec:.4f} s", f"{(sec/total_t*100 if total_t>0 else 0):.1f}%") for lab, sec in items]
    print_table(rows, header=["Block", "Wall Time", "Share"])
    print(f"\nTotal timed wall clock: {total_t:.4f} s")

    # ---------------- Q6 (multi-α) ----------------
    banner("Q6 - Multi-α_tail learning for α_C(t; α_tail) [probit nodes]")

    a_min, a_max = args.a_min, args.a_max
    assert 0.0 < a_min < a_max < 0.5
    eff_mask = f_all > 1e-14
    t_eff = t_grid[eff_mask]
    f_eff = f_all[eff_mask]
    n_eff = len(t_eff)
    Nom_t = Nom * torch.tensor(f_eff, dtype=torch.float32, device=device)

    z_min = float(_ppf_torch(torch.tensor(a_min)))
    z_max = float(_ppf_torch(torch.tensor(a_max)))
    g_z = torch.linspace(z_min, z_max, args.q6_nodes, dtype=torch.float32, device=device)
    a_tail_nodes = _cdf_torch(g_z)

    BETA = torch.zeros((n_eff, args.q6_nodes), dtype=torch.float32, device=device, requires_grad=True)

    with T.timeit("Q6: Warm-start from explicit"):
        with torch.no_grad():
            alpha_true_nodes = np.stack([
                Nom * C_explicit_noNom_noS(t_eff, float(1.0 - a_tail), p.sigma, p.delta, f_func_handle)
                for a_tail in a_tail_nodes.detach().cpu().numpy()
            ], axis=1)
            denom = Nom_t.detach().cpu().numpy().reshape(-1, 1)
            ratio = np.clip(alpha_true_nodes / np.maximum(1e-30, denom), 1e-12, None)
            beta0 = np.log(np.exp(ratio) - 1.0).astype(np.float32)  # softplus^{-1}
            BETA.data.copy_(torch.tensor(beta0, dtype=torch.float32, device=device))

    opt = torch.optim.Adam([BETA], lr=2e-3)

    def interp_beta_z(beta_nodes, a_tail_batch):
        z = _ppf_torch(a_tail_batch)
        idx = torch.bucketize(z, g_z) - 1
        idx = torch.clamp(idx, 0, g_z.numel()-2)
        zL = g_z[idx]; zR = g_z[idx+1]
        wR = (z - zL) / (zR - zL)
        wL = 1.0 - wR
        return idx, wL, wR

    with T.timeit(f"Q6: Training (steps={int(args.q6_steps)}, batch={int(args.q6_batch)}, innerU={int(args.q6_innerU)})"):
        B = int(args.q6_batch)
        mU = int(args.q6_innerU)
        U_vec = torch.tensor(draw_U(mU, p.sigma, p.delta), dtype=torch.float32, device=device)
        U_sorted, _ = torch.sort(U_vec)
        for _ in range(int(args.q6_steps)):
            t_idx = torch.randint(0, n_eff, (B,), device=device)
            a_tail_batch = (a_min + (a_max - a_min) * torch.rand(B, device=device)).clamp(1e-6, 0.499999)
            q_idx = torch.clamp((a_tail_batch * (mU - 1)).long(), 0, mU-1)  # q at α_tail
            q_vec = U_sorted[q_idx]  # (B,)
            tail_mean = torch.clamp(U_sorted.view(1, mU) - q_vec.view(B, 1), min=0.0).mean(dim=1)
            y_batch = Nom_t[t_idx] * tail_mean

            beta_nodes = BETA
            idx, wL, wR = interp_beta_z(beta_nodes, a_tail_batch)
            beta = wL * beta_nodes[t_idx, idx] + wR * beta_nodes[t_idx, idx+1]
            yhat = Nom_t[t_idx] * torch.nn.functional.softplus(beta)

            opt.zero_grad()
            loss = torch.mean((yhat - y_batch)**2)
            pen = 1e-4 * torch.relu(beta_nodes[:, :-1] - beta_nodes[:, 1:]).mean()
            (loss + pen).backward()
            torch.nn.utils.clip_grad_norm_([BETA], 1.0)
            opt.step()

    # Inference
    alphas_tail_plot = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
    beta_nodes = BETA.detach()
    alphaC_hat = {float(a): np.zeros_like(t_grid) for a in alphas_tail_plot}

    with T.timeit("Q6: Inference over α_tail grid"):
        with torch.no_grad():
            for a_tail in alphas_tail_plot:
                a_tail_t = torch.tensor([a_tail], dtype=torch.float32, device=device)
                z_t = _ppf_torch(a_tail_t)
                idx = torch.bucketize(z_t, g_z) - 1
                idx = torch.clamp(idx, 0, g_z.numel()-2)
                zL, zR = g_z[idx], g_z[idx+1]
                wR = (z_t - zL)/(zR - zL); wL = 1.0 - wR
                beta = wL*beta_nodes[:, idx] + wR*beta_nodes[:, idx+1]
                alpha_eff = (Nom_t * torch.nn.functional.softplus(beta.view(-1))).cpu().numpy()
                out = np.zeros_like(t_grid); out[eff_mask] = alpha_eff
                alphaC_hat[float(a_tail)] = out

    # Check monotonicity
    ok = True
    for k in range(len(t_grid)):
        vals = [alphaC_hat[a][k] for a in alphas_tail_plot]
        if any(vals[i] > vals[i+1] + 1e-10 for i in range(len(vals)-1)):
            ok = False; break
    print(f"[CHECK Q6] Non-crossing (non-decreasing in α_tail): {'PASS' if ok else 'FAIL'}")

    # Plot vs explicit (α_C without Ŝ_t factor)
    fig = plt.figure(figsize=(12,6), dpi=120)
    colors = ['tab:blue','tab:orange','tab:green','tab:pink','tab:olive','tab:red']
    for a_tail, col in zip(alphas_tail_plot, colors):
        a_conf_plot = 1.0 - a_tail
        plt.plot(t_grid, Nom * C_explicit_noNom_noS(t_grid, a_conf_plot, p.sigma, p.delta, f_func_handle),
                 label=f'α_C true, α_tail={a_tail:.3f}', linestyle='--', color=col)
        plt.plot(t_grid, alphaC_hat[a_tail], label=f'α_C learned, α_tail={a_tail:.3f}', color=col)
    plt.yscale("log"); plt.grid(True, which="both"); plt.legend(ncol=2)
    plt.title("Q6 - α_C(t; α_tail): true vs learned (probit nodes)")
    savefig_both(fig, args.figdir, "5_3_q6_alphaC_true_vs_learned_multi_alpha")

    print("\n[DONE] 5.3 CVA with RIM finished.")

if __name__ == "__main__":
    main()
