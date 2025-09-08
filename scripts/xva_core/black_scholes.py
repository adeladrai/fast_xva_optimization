import math
import numpy as np
import torch

def call_bs_np(t: float, S_t, T: float, K: float, r: float, sigma: float):
    tau = np.maximum(T - t, 1e-12)
    S_t = np.asarray(S_t, dtype=float); tau = np.asarray(tau, dtype=float)
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    from scipy.stats import norm
    return S_t * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

def call_bs_torch(t: float, S_t: torch.Tensor, T: float, K: float, r: float, sigma: float, device: torch.device):
    tau_t = torch.tensor(max(T - t, 1e-12), dtype=torch.float32, device=device)
    sq = torch.sqrt(tau_t)
    d1 = (torch.log(S_t / K) + (r + 0.5 * sigma**2) * tau_t) / (sigma * sq)
    d2 = d1 - sigma * sq
    Nd1 = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))
    Nd2 = 0.5 * (1.0 + torch.erf(d2 / math.sqrt(2.0)))
    return S_t * Nd1 - K * math.exp(-r * float(tau_t)) * Nd2
