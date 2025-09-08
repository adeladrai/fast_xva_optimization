import math
import numpy as np
import torch
from .black_scholes import call_bs_torch

# ---------- Ŝ and S draws ----------
def Shat_exact_cpu(t, S0: float, sigma: float):
    t = np.atleast_1d(t).astype(float)
    Z = np.random.normal(size=t.shape)
    return S0 * np.exp(-0.5 * sigma**2 * t + sigma * np.sqrt(t) * Z)

def Shat_exact_gpu(t_grid_np, S0: float, sigma: float, device: torch.device) -> torch.Tensor:
    t = torch.tensor(t_grid_np, dtype=torch.float64, device=device)
    Z = torch.randn_like(t)
    return (S0 * torch.exp(-0.5 * sigma**2 * t + sigma * torch.sqrt(t) * Z)).to(torch.float32)

def Shat_step_delta_gpu(Shat_t: torch.Tensor, sigma: float, delta: float, n_inner: int) -> torch.Tensor:
    n = max(1, n_inner // 2)
    z = torch.randn(n, device=Shat_t.device)
    Z = torch.cat([z, -z], dim=0)
    return Shat_t * torch.exp(
        torch.tensor(-0.5 * sigma**2 * delta, device=Shat_t.device) +
        torch.tensor(sigma * math.sqrt(delta), device=Shat_t.device) * Z
    )

def S_exact_cpu(t, S0: float, r: float, sigma: float, size=None):
    if size is not None:
        t = float(t)
        Z = np.random.normal(size=size)
        return S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * Z)
    t = np.atleast_1d(t).astype(float)
    Z = np.random.normal(size=t.shape)
    return S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z)

# ---------- U and gaps ----------
def draw_U(n: int, sigma: float, delta: float) -> np.ndarray:
    Z = np.random.normal(size=n)
    return np.exp(-0.5 * sigma**2 * delta + sigma * math.sqrt(delta) * Z) - 1.0

def gap_LHS_MC_gpu(Shat_t_scalar: float, Nom: float, f_scalar: float, sigma: float, delta: float,
                   n_inner: int, device: torch.device) -> float:
    Sh_t = torch.tensor(Shat_t_scalar, dtype=torch.float32, device=device)
    Sh_td = Shat_step_delta_gpu(Sh_t, sigma, delta, n_inner)
    payoff = torch.relu(Sh_td - Sh_t)
    val = Nom * f_scalar * payoff.mean()
    return float(val.item())

# Gap over MPoR for calls:
#   gain:  β_{t+δ} C_{t+δ} - β_t C_t
#   loss: -gain
def call_gap_gain(t: float, S_t: float, T: float, K: float, r: float, sigma: float, delta: float,
                  n_inner: int, device: torch.device) -> float:
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
    gap = beta_td * c_td - beta_t * c_t
    return float(gap.mean().item())

def call_gap_loss(*args, **kwargs) -> float:
    return -call_gap_gain(*args, **kwargs)
