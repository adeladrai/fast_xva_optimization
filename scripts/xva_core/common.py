import os, math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import numpy as np
import matplotlib.pyplot as plt
import torch
from time import perf_counter
from contextlib import contextmanager

# Matplotlib defaults (EPS-safe)
plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


@dataclass
class Params:
    """Common parameters used across XVA scripts."""
    r: float = 0.02
    kappa: float = 0.12
    sigma: float = 0.20
    S0: float = 100.0
    T: float = 5.0
    h: float = 0.25
    delta: float = 1.0 / 52.0
    gamma1: float = 0.01  # default intensity
    gamma_fund: float = 0.01  # funding intensity (used in MVA)

class Timer:
    def __init__(self, device: torch.device | None = None, echo: bool = True):
        self.device = device
        self.echo = echo
        self.times: Dict[str, float] = {}

    @contextmanager
    def timeit(self, label: str):
        dev = self.device
        if dev is not None and hasattr(torch, "cuda") and getattr(dev, "type", "") == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()
        yield
        if dev is not None and hasattr(torch, "cuda") and getattr(dev, "type", "") == "cuda":
            torch.cuda.synchronize()
        dt = perf_counter() - t0
        self.times[label] = self.times.get(label, 0.0) + dt
        if self.echo:
            print(f"[TIMING] {label}: {dt:.6f} s")

def banner(title: str, ch: str = "="):
    line = ch * max(3, len(title))
    print(f"\n{line}\n{title}\n{line}")

def savefig_both(fig, outdir: str, stem: str):
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, stem + ".png")
    eps = os.path.join(outdir, stem + ".eps")
    fig.savefig(png, bbox_inches="tight")
    # remove alphas for EPS safety
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_alpha() is not None:
                line.set_alpha(1.0)
    fig.savefig(eps, bbox_inches="tight", format="eps")
    print(f"[Figure saved] {png}")
    print(f"[Figure saved] {eps}")
    plt.close(fig)

def print_table(rows: List[List[str]], header: List[str] | None = None):
    if not rows: return
    cols = max(len(r) for r in rows)
    if header: cols = max(cols, len(header))
    widths = []
    for c in range(cols):
        w = 0
        if header and c < len(header): w = len(str(header[c]))
        for r in rows:
            if c < len(r): w = max(w, len(str(r[c])))
        widths.append(w)
    if header:
        print(" | ".join(str(header[i]).ljust(widths[i]) for i in range(cols)))
        print("-+-".join("-"*w for w in widths))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(cols)))

def relerr(x: float, ref: float) -> float:
    return abs(x - ref) / max(1e-12, abs(ref))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def fmt_pm_bps(val_bps: float, se_bps: float) -> str:
    return f"{val_bps:.4f} bps ± {se_bps:.4f} bps"

def trapz_weights(t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t_grid)
    w = np.zeros_like(t, dtype=float)
    if len(t) < 2: return w
    dt = np.diff(t)
    w[0] = 0.5 * dt[0]; w[-1] = 0.5 * dt[-1]
    if len(t) > 2: w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return w

def seed_all(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_device(mode: str = "auto") -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def swap_to_bps(x: float) -> float:
    """Convert swap value to basis points (bps) on the SAME calculated nominal: 10,000 × value."""
    return 10000.0 * x


def call_to_bps(x: float, S0: float) -> float:
    """Convert call value to basis points (bps) per S0: 10,000 × (value / S0).
    
    Raises:
        ValueError: If S0 is zero.
    """
    if S0 == 0:
        raise ValueError("S0 cannot be zero")
    return 10000.0 * (x / S0)
