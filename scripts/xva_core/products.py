from typing import Tuple
import numpy as np

def fair_strike_and_nominal(S0: float, r: float, kappa: float, h: float,
                            payment_times: np.ndarray) -> Tuple[float, float]:
    disc = np.exp(-r * payment_times)
    Tlm1 = np.arange(len(payment_times)) * h
    float_leg_0 = float(np.sum(disc * h * S0 * np.exp(kappa * Tlm1)))
    denom = float(np.sum(disc * h))
    barS = float_leg_0 / denom
    Nom = 1.0 / float_leg_0
    return barS, Nom

def precompute_weights(r, kappa, h, payment_times):
    weights = np.exp(-r * payment_times) * h * np.exp(kappa * (np.arange(len(payment_times)) * h))
    weights_suffix = np.flip(np.cumsum(np.flip(weights)))
    return weights, weights_suffix

def f_func_vec(t_arr: np.ndarray, payment_times: np.ndarray,
               weights_suffix: np.ndarray, delta: float) -> np.ndarray:
    t_arr = np.atleast_1d(t_arr)
    idx = np.searchsorted(payment_times, t_arr + delta, side="right")
    out = np.zeros_like(t_arr, dtype=float)
    m = idx < len(payment_times)
    out[m] = weights_suffix[idx[m]]
    return out

def f_func_scalar(t: float, payment_times: np.ndarray,
                  weights_suffix: np.ndarray, delta: float) -> float:
    return float(f_func_vec(np.array([t]), payment_times, weights_suffix, delta)[0])
