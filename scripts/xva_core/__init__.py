# Re-export the most used symbols for convenience
from .common import (
    Timer, banner, savefig_both, print_table, relerr, rmse, fmt_pm_bps,
    trapz_weights, seed_all, select_device
)
from .products import (
    fair_strike_and_nominal, precompute_weights, f_func_vec, f_func_scalar
)
from .stochastic import (
    Shat_exact_cpu, Shat_exact_gpu, Shat_step_delta_gpu,
    S_exact_cpu, draw_U, gap_LHS_MC_gpu, call_gap_gain, call_gap_loss
)
from .black_scholes import call_bs_np, call_bs_torch
from .losses import pinball
