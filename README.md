# Fast XVA Optimization via Machine Learning

> **Python requirement:** **Python 3.12 only**. Please ensure your virtual environment uses **3.12**.

---

## Repository layout

```
xva-project/
├─ README.md
├─ requirements.txt
├─ scripts/
│  ├─ 5_0_mtm_swap.py
│  ├─ 5_1_cva_no_rim.py
│  ├─ 5_2_mva_pim.py
│  └─ 5_3_cva_with_rim.py
└─ xva_core/                # shared code
   ├─ __init__.py
   ├─ common.py             # Timer, banner, savefig (PNG+EPS), tables, relerr, rmse, trapz weights, seeding, device utils
   ├─ products.py           # swap weights & f(t); fair strike / nominal
   ├─ stochastic.py         # GBM/Ŝ draws (CPU/GPU), δ-step, MC helpers, call gap samplers
   ├─ black_scholes.py      # Black–Scholes pricing (NumPy + Torch)
   └─ losses.py             # pinball losses (NumPy/Torch-friendly)
```

> Always run the scripts **from the repository root** so `xva_core` is importable.

---

## Quick start

### 1) Create & activate a Python **3.12** virtual environment

**macOS / Linux (bash/zsh):**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version  # should print Python 3.12.x
```

**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version  # should print Python 3.12.x
```

### 2) Install requirements

```bash
pip install -U pip
pip install -r requirements.txt
```

> **GPU (optional):** If you want CUDA acceleration, install a CUDA-enabled PyTorch **for Python 3.12** that matches your CUDA toolkit. See: https://pytorch.org/get-started/locally/

### 3) Run any script

```bash
# 5.0 - MtM (bps) over time for a 5Y swap
python scripts/5_0_mtm_swap.py --figdir figs

# 5.1 - CVA without RIM (GPU-heavy; requires CUDA)
python scripts/5_1_cva_no_rim.py --figdir figs

# 5.2 - MVA for PIM (CPU or GPU)
python scripts/5_2_mva_pim.py --device auto --figdir figs

# 5.3 - CVA with RIM (CPU or GPU)
python scripts/5_3_cva_with_rim.py --device auto --figdir figs
```

All figures are written to `--figdir` as `.png` **and** `.eps` (EPS-safe, no transparency).

---


## How the modularization works

- **`xva_core/common.py`** - `Timer`, `banner`, `savefig_both` (PNG+EPS), ASCII `print_table`, `relerr`, `rmse`, trapezoid weights, seeding/device helpers.  
- **`xva_core/products.py`** - swap payment weights and suffix sums; `fair_strike_and_nominal(barS, Nom)`; `f(t)` builders per schedule.  
- **`xva_core/stochastic.py`** - GBM/Ŝ draws (NumPy & Torch), δ-step helpers, antithetic sampling, MC utilities.  
- **`xva_core/black_scholes.py`** - Black–Scholes call pricing (NumPy + Torch versions).  
- **`xva_core/losses.py`** - Pinball/quantile losses and coverage helpers.

This avoids duplication across 5.0–5.3 and makes extensions easier.

---

## Outputs (quick guide)

> **Runtime:** All examples assume **Python 3.12**.  
> **Figures:** Every plot is saved as both PNG and EPS in `figs/`.  
> **Units:** Swap results are in **bps on the same calculated nominal**; call results are **bps per unit of S₀**.  
> **Timing:** `[TIMING] …: X.XXXXXX s` lines are wall-clock timings; the final **TIMING - Breakdown** shows where time goes.  
> **PASS/Check:** Each method is compared to a reference (quadrature or nested MC) with a 5% tolerance.

### What you’ll see in the console
- **Environment / Setup:** Library versions, CUDA availability, grid sizes, fair strike `barS`, and nominal `Nom`.
- **Harmless EPS note:** “PostScript backend does not support transparency … rendered opaque” (expected when writing `.eps`).

### Per-script at a glance
- **`5_0_mtm_swap.py`**  
  Simulates GBM paths, computes swap MtM via martingale representation, and saves:
  - `figs/5_0_mtm_swap.(png|eps)` - mean and 2.5%/97.5% MtM **bps** bands over time.

- **`5_1_cva_no_rim.py`**  
  Swap CVA₀ without RIM: explicit vs OLS/NN vs twin/nested MC. Also trains call exposure models. Outputs:
  - `figs/5_1_q1_swap_conditional.(png|eps)` - learned conditional slope vs explicit.
  - `figs/5_1_q2_swap_validation.(png|eps)` - explicit vs twin/nested vs models.
  - `figs/5_1_q4_call_*.(png|eps)` - call predictors (validation + parity/error).
  - **SUMMARY - CVA₀ Results (bps)** tables + **COST** and **TIMING** breakdowns.

- **`5_2_mva_pim.py`**  
  MVA with PIM: robust single-α learner, swap MVA₀ (quadrature vs model), call VaR models, and multi-α tails. Outputs:
  - `figs/5_2_q2_alpha_true_vs_learned.(png|eps)`
  - `figs/5_2_q3_swap_var_streaming.(png|eps)`
  - `figs/5_2_q5_call_var_streaming.(png|eps)`
  - `figs/5_2_q6_alpha_true_vs_learned_multi_alpha.(png|eps)`
  - **SUMMARY - MVA₀ Results (bps)** + **COST** + **TIMING**.

- **`5_3_cva_with_rim.py`**  
  CVA with RIM: pinball for \(q_a(U)\) and ES, swap CVA⁽RIM⁾₀, call E[(X−q)⁺] models, multi-α (probit nodes). Outputs:
  - `figs/5_3_q3_rim_swap_validation.(png|eps)`
  - `figs/5_3_q5_call_rim_scaled.(png|eps)`
  - `figs/5_3_q6_alphaC_true_vs_learned_multi_alpha.(png|eps)`
  - **SUMMARY - CVA⁽RIM⁾₀ (bps)** + **COST** + **TIMING**.

> The sample logs in this repository show typical values (e.g., swap CVA₀ ≈ 3.06 bps without RIM, swap MVA₀ ≈ 17.3 bps with PIM). Your numbers may differ slightly due to RNG/GPU/library versions, but the structure and pass/fail checks should match.

---


**CUDA/GPU not detected**  
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

**EPS export warnings**  
We force line alpha to 1.0 before saving EPS; update Matplotlib if warnings persist.

**Slow runs / memory pressure**  
Lower `--inner_scale` / `--outer_scale`, or reduce `N_outer` / `n_inner_*` where exposed.

---

## Reproducing a typical figure set

```bash
# 5.0 MtM bands
python scripts/5_0_mtm_swap.py --figdir figs

# 5.1 CVA (no RIM) - heavy GPU run
python scripts/5_1_cva_no_rim.py --figdir figs 

# 5.2 MVA for PIM
python scripts/5_2_mva_pim.py --figdir figs

# 5.3 CVA with RIM
python scripts/5_3_cva_with_rim.py --figdir figs
```
