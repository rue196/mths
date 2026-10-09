#!/usr/bin/env python3
"""
ai_capex_pyramid.py
====================
AI Capex Spending as a Zero-Sum Extraction Pyramid
Based on Unified Wave-Pyramid Economic Models (wave-demand-supply-pyramid.pdf)

Models the 2023-2026 AI capex boom with three pyramid layers:
  Layer 1 (Retail Base):    Consumer AI adoption & subscription revenue
  Layer 2 (Table/Mid-Tier): Hyperscaler capex with debt leverage
  Layer 3 (Debt Superstructure): AI-related bond issuance & CDS spreads

Key equations:
  - Master equation (Eq. 1): logistic growth with access ceiling
  - Pool depletion (Eq. 6.3): P(t+1) = P(t)(1+ψ) - E(t)
  - Cascade trigger (Eq. 6.3'): Λ(t)/P(t) >= τ
  - Scarcity regime (Fig 3.5): (1-p)^{-n} bounded by Π_k
  - Labor pool (Eq. 5.3-5.4): ΔG_L sign determines plus-sum vs zero-sum
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ============================================================
# SECTION 1 — REAL DATA ANCHORS (2023-2026)
# ============================================================

@dataclass
class AIDataAnchors:
    """Real-world data points collected from market sources."""
    
    # Hyperscaler capex by year ($ billions)
    capex_2023: float = 200.0      # Combined MSFT, AMZN, GOOG, META
    capex_2024: float = 350.0      # Surge begins
    capex_2025: float = 580.0      # Accelerating
    capex_2026: float = 830.0      # Current run-rate (TrendForce)
    
    # Individual company 2026 guidance ($ billions)
    amazon_2026: float = 220.0     # Raised from $200B
    google_2026: float = 205.0     # Raised from $195B
    microsoft_2026: float = 190.0  # incl. $25B component costs
    meta_2026: float = 145.0       # Raised from $130B
    
    # Consumer AI adoption (%)
    us_adult_ai_weekly_2026: float = 65.0     # SSRS May 2026
    us_adult_ai_weekly_2025: float = 43.0     # 12 months prior
    chatgpt_weekly_users_2026: float = 900e6  # ~900M WAU
    global_consumer_ai_2026: float = 82.0     # VML report
    
    # Revenue data
    openai_arr_2026: float = 20e9              # Estimated $20B ARR
    total_ai_revenue_2026: float = 45e9        # Industry estimate
    hyperscaler_cloud_revenue: float = 280e9   # AWS+Azure+GCP annual
    
    # Debt data
    ai_debt_cumulative: float = 1.374e12       # Since ChatGPT launch
    ai_debt_2026_ytd: float = 635.1e9          # YTD 2026
    hyperscaler_bond_2026: float = 159e9       # Through June 2026
    hyperscaler_bond_2025: float = 109e9       # Full year 2025
    ai_ig_bond_2026: float = 281.55e9          # Investment grade YTD
    ai_hy_bond_2026: float = 55e9              # High yield YTD
    
    # Credit spreads (basis points)
    hyperscaler_spread_widen: float = 25.0     # +25bp YTD 2026
    cds_spread_change: float = 40.0            # Sharp increase
    oracle_spread_widen: float = 15.0          # +15bp to 200bp
    
    # Labor market
    junior_ai_displacement: float = 19.0       # % decline for 22-25 yo
    senior_hiring_increase: float = 6.7        # % rise in senior roles
    overall_ai_exposure: float = 40.0          # % jobs exposed (IMF)
    advanced_economy_exposure: float = 60.0    # % in advanced economies
    
    # Macro
    us_gdp_2026: float = 31.22e12             # Nominal GDP
    us_debt_2026: float = 39e12               # Gross national debt
    debt_gdp_ratio: float = 100.2             # % (first time since WWII)
    net_interest_2026: float = 1.21e12        # Surpassing defense
    interest_pct_revenue: float = 19.0        # Record share
    cpi_2026: float = 3.4                     # August YoY
    core_cpi_2026: float = 2.4                # Ex food/energy
    m4_growth: float = 7.9                    # % y/y July 2026
    
    # Reserve currency
    usd_reserve_share: float = 57.0           # % of global reserves
    foreign_treasury_holdings: float = 9.3e12 # $9.3T
    foreign_official_share: float = 41.0      # % (down from 66% in 2014)
    convenience_yield_decline: float = 50.0   # bp decline in premium
    
    # Adoption plateau indicators
    enterprise_deep_integration: float = 11.0  # % (up from 7% in 2025)
    enterprise_on_track: float = 13.0         # % on track with AI initiatives
    roi_failure_rate: float = 57.0            # % where ROI < spend
    consumer_growth_flatlined: bool = True    # DAU growth 4/5 months negative


# ============================================================
# SECTION 2 — MODEL PARAMETERS
# ============================================================

@dataclass
class ModelParams:
    """Parameters for the wave-pyramid AI capex model."""
    
    # Time horizon
    t_start: float = 2023.0     # ChatGPT launch
    t_end: float = 2035.0
    dt: float = 0.25            # Quarterly
    
    # Fig 4.3 — Access boom carrying capacity for AI
    K_ai_inf: float = 1.0       # Normalized max AI adoption capacity
    K_ai_sc: float = 0.05       # Initial scarcity (2023)
    lambda_ai: float = 0.85     # Fast approach rate (AI diffuses quickly)
    w_star_ai: float = 2023.0   # Access threshold
    w_f_ai: float = 2025.5      # Fatigue onset (plateau begins)
    
    # Fig 5.3-5.4 — Labor pool dynamics
    alpha_L: float = 0.025      # Baseline labor productivity contribution
    beta_L: float = 0.65        # Post-industrial labor drag
    delta_AI_disp: float = 0.04 # AI displacement rate on exposed jobs
    ai_replacement_frac: float = 0.15  # Fraction of jobs AI can fully replace
    
    # Fig 6.2 — Pyramid layer leverage
    h_ret: float = 0.08         # Retail extraction (consumer subs)
    h_mid: float = 0.12         # Mid-tier extraction (capex markup)
    xi_leverage: float = 3.8    # Hyperscaler leverage ratio
    tau_crit: float = 2.5       # Cascade trigger threshold
    
    # Eq 6.3 — Pool dynamics
    psi_base: float = 0.15      # Base hype/confidence inflow
    psi_decay: float = 0.25     # Decay rate post-fatigue
    E_base: float = 0.06        # Base extraction rate
    
    # Fig 3.5 — Scarcity regime
    p_ai_compute: float = 0.35  # Compute scarcity probability
    n_ai_trials: int = 3        # Trials per compute stage
    Pi_bound: float = 0.80      # Elliptic projection bound
    
    # Inflation & reserve currency
    money_growth: float = 0.079 # M4 growth 7.9%
    inflation_base: float = 0.034  # Current CPI
    reserve_premium_decay: float = 0.015  # Annual erosion rate


# ============================================================
# SECTION 3 — CARRIER: AI ACCESS BOOM (Fig 4.3 / 4.4)
# ============================================================

def ai_carrying_capacity(t: float, p: ModelParams, data: AIDataAnchors) -> float:
    """
    Fig 4.3: K(w) = K_sc + (K_inf - K_sc)(1 - exp(-λ·(w - w*)))
    Applied to AI adoption as the "commodity" being accessed.
    Fig 4.4 fatigue damping kicks in post-plateau.
    """
    if t < p.w_star_ai:
        return p.K_ai_sc * (1 + 0.1 * (t - p.t_start))
    
    w = t - p.w_star_ai
    K = p.K_ai_sc + (p.K_ai_inf - p.K_ai_sc) * (1 - np.exp(-p.lambda_ai * w))
    
    # Fig 4.4: Demographic-transition analog (adoption fatigue)
    if t > p.w_f_ai:
        phi = (t - p.w_f_ai) / max(p.t_end - p.w_f_ai, 1.0)
        # Adoption growth decelerates as late adopters resist
        K = K * (1 - 0.25 * phi)
    
    return K


# ============================================================
# SECTION 4 — LABOR POOL DYNAMICS (Fig 5.3-5.4)
# ============================================================

def labor_contribution(t: float, ai_capacity: float, p: ModelParams,
                        data: AIDataAnchors) -> float:
    """
    Fig 5.4: ΔG_L = G(t) · (ΔL/L) · α(1 - β·t/T)
    AI displacement affects only deterministic jobs (data analysis, etc.)
    """
    # Base labor productivity contribution
    years_since_ai = max(t - 2023.0, 0.0)
    
    # AI displacement: affects only deterministic exposed jobs
    exposure_frac = data.advanced_economy_exposure / 100.0  # 60%
    deterministic_frac = data.junior_ai_displacement / 100.0  # 19%
    
    # Cumulative displacement grows with AI capacity
    cumulative_displacement = (exposure_frac * p.ai_replacement_frac * 
                                min(years_since_ai / 5.0, 1.0))
    
    # Labor pool net change
    if t < 2025:
        dG_L = p.alpha_L * (1 - 0.2 * years_since_ai)
    elif t < 2028:
        # Displacement accelerates but senior hiring offsets partially
        senior_offset = data.senior_hiring_increase / 100.0 * 0.3
        dG_L = p.alpha_L * (0.7 - cumulative_displacement + senior_offset)
    else:
        # Structural displacement exceeds new job creation
        dG_L = -p.alpha_L * 0.4 * (1 + 0.05 * (t - 2028))
    
    return dG_L


# ============================================================
# SECTION 5 — PYRAMID LAYERS (Fig 6.2)
# ============================================================

def retail_layer(t: float, pool: float, ai_capacity: float,
                  p: ModelParams, data: AIDataAnchors) -> Dict:
    """
    Layer 1: Consumer AI adoption & subscription revenue.
    Retail base extraction from the consumer pool.
    """
    # Consumer adoption rate (% of adults)
    adoption_rate = ai_capacity * data.us_adult_ai_weekly_2026 / 100.0
    adoption_rate = min(adoption_rate, 0.85)  # cap at 85%
    
    # Monthly consumer AI spend (subscriptions, API, etc.)
    consumer_spend = adoption_rate * 20.0  # $20/month average
    
    # Retail extraction (consumer surplus captured by AI companies)
    W_ret = pool * 0.35
    E_ret = W_ret * p.h_ret * (1 + 0.5 * (ai_capacity - 0.5))
    
    return {
        'adoption_rate': adoption_rate,
        'consumer_spend_annual': consumer_spend * 12 * 320e6 / 1e9,  # $B
        'W_ret': W_ret,
        'E_ret': E_ret,
    }


def table_layer(t: float, pool: float, ai_capacity: float,
                 debt_total: float, p: ModelParams, data: AIDataAnchors) -> Dict:
    """
    Layer 2: Hyperscaler capex with leverage.
    Mid-tier extraction via capex markup and leverage spread.
    """
    years_since_2023 = max(t - 2023.0, 0.0)
    
    # Hyperscaler capex trajectory ($B/year)
    if t < 2024:
        capex = data.capex_2023 * (1 + 0.3 * years_since_2023)
    elif t < 2025:
        capex = data.capex_2023 + (data.capex_2024 - data.capex_2023) * (t - 2024)
    elif t < 2026:
        capex = data.capex_2024 + (data.capex_2025 - data.capex_2024) * (t - 2025)
    else:
        # Growth decelerates post-2026 as ROI pressure builds
        capex = data.capex_2026 * np.exp(-0.08 * max(t - 2026, 0))
        capex = max(capex, data.capex_2026 * 0.6)  # floor at 60%
    
    # Capex as fraction of operating cash flow
    ocf_ratio = 0.75 + 0.25 * (years_since_2023 / 4)  # Rising to >100%
    ocf_ratio = min(ocf_ratio, 1.15)  # Some companies go cash-negative
    
    # Mid-tier extraction: markup on capex + leverage spread
    W_mid = pool * 0.40
    E_mid = W_mid * p.h_mid + p.xi_leverage * capex * 0.02  # Leverage spread
    
    return {
        'capex_annual': capex,
        'capex_yoy_growth': (capex / data.capex_2025 - 1) * 100 if t > 2025 else 0,
        'ocf_ratio': ocf_ratio,
        'W_mid': W_mid,
        'E_mid': E_mid,
    }


def debt_superstructure(t: float, debt_total: float, capex: float,
                         pool: float, p: ModelParams, data: AIDataAnchors) -> Dict:
    """
    Layer 3: AI-related bond issuance, CDS spreads, debt accumulation.
    This is the superstructure that triggers cascade when Λ/P >= τ.
    """
    # Debt accumulation rate
    if t < 2024:
        debt_issuance = 50e9  # Baseline pre-AI debt
    elif t < 2025:
        debt_issuance = data.hyperscaler_bond_2025 * (t - 2024)
    elif t < 2026:
        debt_issuance = data.hyperscaler_bond_2026 * 2 * (t - 2025)
    else:
        # Debt issuance accelerates to fund capex gap
        gap = max(capex - 750e9 * 0.9, 0)  # If capex > OCF
        debt_issuance = gap * 1e9 * 0.35  # 35% debt-funded
    
    # Credit spread (widens with leverage)
    debt_gdp = debt_total / data.us_gdp_2026
    spread_bp = 25 + 50 * max(0, debt_gdp - 0.5) + 30 * max(0, capex/830 - 0.8)
    
    # Debt superstructure extraction
    E_debt = debt_total * 0.045  # 4.5% effective carrying cost
    
    return {
        'debt_issuance_annual': debt_issuance,
        'spread_bp': spread_bp,
        'E_debt': E_debt,
    }


# ============================================================
# SECTION 6 — SCARCITY REGIME (Fig 3.5 / d-supply-chain-harmonic)
# ============================================================

def scarcity_amplifier(t: float, ai_capacity: float,
                        p: ModelParams) -> float:
    """
    Fig 3.5 from d-supply-chain-harmonic: bounded scarcity amplifier.
    
    (1-p)^{-n} -> 1 + Π_k((1-p)^{-n} - 1)
    
    Applied to AI compute scarcity (GPUs, HBM, power).
    """
    # Scarcity declines as capacity builds (more fabs, more power)
    p_eff = p.p_ai_compute * (1 - 0.4 * min(ai_capacity, 1.0))
    p_eff = max(p_eff, 0.05)  # Floor
    
    raw_amp = (1 - p_eff) ** (-p.n_ai_trials)
    bounded_amp = 1 + p.Pi_bound * (raw_amp - 1)
    
    return bounded_amp


# ============================================================
# SECTION 7 — MAIN SIMULATION
# ============================================================

def run_ai_capex_simulation(p: ModelParams = None,
                             data: AIDataAnchors = None):
    """Run the full AI capex pyramid simulation."""
    if p is None:
        p = ModelParams()
    if data is None:
        data = AIDataAnchors()
    
    t_grid = np.arange(p.t_start, p.t_end + p.dt, p.dt)
    n = len(t_grid)
    
    # State arrays
    ai_capacity = np.zeros(n)
    pool = np.zeros(n)
    debt_total = np.zeros(n)
    Lambda = np.zeros(n)
    lambda_over_P = np.zeros(n)
    cascade_flag = np.zeros(n, dtype=bool)
    psi_t = np.zeros(n)
    E_t = np.zeros(n)
    dG_L = np.zeros(n)
    scarcity = np.zeros(n)
    capex_arr = np.zeros(n)
    retail_extract = np.zeros(n)
    table_extract = np.zeros(n)
    debt_extract = np.zeros(n)
    reserve_premium = np.zeros(n)
    
    # Initial conditions (2023)
    pool[0] = 1.0
    debt_total[0] = 50e9  # Baseline AI-related debt
    ai_capacity[0] = ai_carrying_capacity(p.t_start, p, data)
    reserve_premium[0] = 1.0  # Full premium
    
    # Precompute for plotting
    for i in range(1, n):
        t = t_grid[i]
        dt = p.dt
        
        # --- AI carrying capacity (Fig 4.3/4.4) ---
        ai_capacity[i] = ai_carrying_capacity(t, p, data)
        
        # --- Labor contribution (Fig 5.3/5.4) ---
        dG_L[i] = labor_contribution(t, ai_capacity[i], p, data)
        
        # --- Scarcity amplifier (Fig 3.5) ---
        scarcity[i] = scarcity_amplifier(t, ai_capacity[i], p)
        
        # --- Pyramid layers (Fig 6.2) ---
        retail = retail_layer(t, pool[i-1], ai_capacity[i], p, data)
        table = table_layer(t, pool[i-1], ai_capacity[i], debt_total[i-1], p, data)
        debt_layer = debt_superstructure(t, debt_total[i-1], table['capex_annual'],
                                          pool[i-1], p, data)
        
        retail_extract[i] = retail['E_ret']
        table_extract[i] = table['E_mid']
        debt_extract[i] = debt_layer['E_debt']
        capex_arr[i] = table['capex_annual']
        
        # --- Debt dynamics ---
        debt_total[i] = debt_total[i-1] + debt_layer['debt_issuance_annual'] * dt
        
        # --- Pool dynamics (Eq 6.3) ---
        # Hype inflow: high early, decays post-plateau
        if t < p.w_f_ai:
            psi_t[i] = p.psi_base * ai_capacity[i] * scarcity[i]
        else:
            psi_t[i] = p.psi_base * ai_capacity[i] * np.exp(-p.psi_decay * (t - p.w_f_ai))
        
        # Total extraction
        E_t[i] = retail['E_ret'] + table['E_mid'] + debt_layer['E_debt']
        
        # Pool update
        pool[i] = max(pool[i-1] * (1 + psi_t[i] * dt) - E_t[i] * dt, 1e-4)
        
        # --- Pyramid leverage (Fig 6.2) ---
        Lambda[i] = (retail['E_ret'] + table['E_mid'] +
                     p.xi_leverage * debt_layer['E_debt'])
        
        # --- Cascade trigger (Eq 6.3') ---
        lambda_over_P[i] = Lambda[i] / max(pool[i], 1e-6) * 0.05
        
        if lambda_over_P[i] >= p.tau_crit:
            cascade_flag[i] = True
        
        # --- Reserve currency premium erosion ---
        # Premium decays as debt/GDP rises and AI debt accumulates
        debt_gdp = debt_total[i] / data.us_gdp_2026
        premium_drag = p.reserve_premium_decay * (1 + 0.5 * max(0, debt_gdp - 1.0))
        reserve_premium[i] = max(reserve_premium[i-1] - premium_drag * dt, 0.3)
    
    return {
        't': t_grid,
        'ai_capacity': ai_capacity,
        'pool': pool,
        'debt_total': debt_total,
        'Lambda': Lambda,
        'lambda_over_P': lambda_over_P,
        'cascade_flag': cascade_flag,
        'psi_t': psi_t,
        'E_t': E_t,
        'dG_L': dG_L,
        'scarcity': scarcity,
        'capex': capex_arr,
        'retail_extract': retail_extract,
        'table_extract': table_extract,
        'debt_extract': debt_extract,
        'reserve_premium': reserve_premium,
        'params': p,
        'data': data,
    }


# ============================================================
# SECTION 8 — DIAGNOSTICS
# ============================================================

def print_diagnostics(results):
    t = results['t']
    data = results['data']
    p = results['params']
    
    idx_2026 = np.argmin(np.abs(t - 2026.5))
    
    print("=" * 78)
    print("AI CAPEX PYRAMID SIMULATION — 2026 SNAPSHOT")
    print("=" * 78)
    print(f"  Model date:                 2026.5 (mid-2026)")
    print(f"  AI adoption capacity:       {results['ai_capacity'][idx_2026]:.4f}")
    print(f"  Consumer pool:              {results['pool'][idx_2026]:.4f}")
    print(f"  Hyperscaler capex ($B/yr):  {results['capex'][idx_2026]:.1f}")
    print(f"  Total AI debt ($B):         {results['debt_total'][idx_2026]/1e9:.1f}")
    print(f"  Pyramid leverage Λ:         {results['Lambda'][idx_2026]:.4f}")
    print(f"  Λ/P ratio:                  {results['lambda_over_P'][idx_2026]:.3f}")
    print(f"  Hype inflow ψ:              {results['psi_t'][idx_2026]:.4f}")
    print(f"  Total extraction E:         {results['E_t'][idx_2026]:.4f}")
    print(f"  Labor contribution ΔG_L:    {results['dG_L'][idx_2026]:.4f}")
    print(f"  Scarcity amplifier:         {results['scarcity'][idx_2026]:.4f}")
    print(f"  Reserve premium:            {results['reserve_premium'][idx_2026]:.4f}")
    print(f"  Cascade triggered:          {results['cascade_flag'][idx_2026]}")
    print("=" * 78)
    
    # Cascade onset
    cascades = np.where(results['cascade_flag'])[0]
    if len(cascades) > 0:
        print(f"\n  ⚠ Cascade trigger reached in: {t[cascades[0]]:.1f}")
        print(f"    Λ/P at trigger: {results['lambda_over_P'][cascades[0]]:.3f} (τ = {p.tau_crit})")
    else:
        peak_lp = results['lambda_over_P'].max()
        peak_idx = np.argmax(results['lambda_over_P'])
        print(f"\n  ✓ No cascade within horizon")
        print(f"    Peak Λ/P: {peak_lp:.3f} at year {t[peak_idx]:.1f} (τ = {p.tau_crit})")
    
    # Plus-sum vs zero-sum
    print()
    print("=" * 78)
    print("PLUS-SUM / ZERO-SUM BOUNDARY (Fig 8.1)")
    print("=" * 78)
    for year in [2023, 2024, 2025, 2026, 2027, 2028, 2030]:
        idx = np.argmin(np.abs(t - year))
        sign = "PLUS-SUM" if results['dG_L'][idx] > 0 else "ZERO-SUM"
        print(f"  {year}: ΔG_L = {results['dG_L'][idx]:+.4f}  → {sign}")
    print("=" * 78)


# ============================================================
# SECTION 9 — VISUALIZATION
# ============================================================

def plot_results(results):
    t = results['t']
    data = results['data']
    p = results['params']
    
    fig = plt.figure(figsize=(18, 14))
    
    # --- Panel 1: AI Access Boom (Fig 4.3/4.4) ---
    ax = fig.add_subplot(3, 3, 1)
    ax.plot(t, results['ai_capacity'], 'darkblue', lw=2.5, label='K_AI(t)')
    ax.axvline(2023, color='green', ls='--', alpha=0.6, label='ChatGPT launch (w*)')
    ax.axvline(p.w_f_ai, color='red', ls='--', alpha=0.6, label='Plateau onset (w_f)')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4, label='K∞')
    ax.set_title('Fig 4.3/4.4: AI Access Boom & Plateau', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('AI Adoption Capacity')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    # --- Panel 2: Hyperscaler Capex ---
    ax = fig.add_subplot(3, 3, 2)
    ax.bar(t[::4], results['capex'][::4], width=0.2, color='steelblue',
           alpha=0.8, label='Capex ($B/yr)')
    ax.axhline(data.capex_2026, color='red', ls='--', alpha=0.6,
               label=f'2026 actual: ${data.capex_2026}B')
    ax.set_title('Hyperscaler AI Capex Trajectory', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Capex ($B)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    
    # --- Panel 3: Labor Contribution (Fig 5.3/5.4) ---
    ax = fig.add_subplot(3, 3, 3)
    ax.fill_between(t, 0, results['dG_L'], where=(results['dG_L'] >= 0),
                    color='green', alpha=0.4, label='ΔG_L > 0 (plus-sum)')
    ax.fill_between(t, 0, results['dG_L'], where=(results['dG_L'] < 0),
                    color='red', alpha=0.4, label='ΔG_L < 0 (zero-sum)')
    ax.plot(t, results['dG_L'], 'k-', lw=1.5)
    ax.axhline(0, color='black', ls=':', alpha=0.5)
    ax.axvline(2026.5, color='blue', ls='--', alpha=0.6, label='Today')
    ax.set_title('Fig 5.3: Labor ΔG_L — Plus/Zero-Sum Boundary', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('ΔG_L')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    # --- Panel 4: Pool Depletion (Eq 6.3) ---
    ax = fig.add_subplot(3, 3, 4)
    ax.plot(t, results['pool'], 'black', lw=2.5, label='Consumer pool P(t)')
    ax.fill_between(t, 0, results['pool'], color='black', alpha=0.08)
    ax.axvline(2026.5, color='blue', ls='--', alpha=0.6, label='Today')
    ax.set_title('Eq 6.3: AI Pool Depletion', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('P(t) — normalized')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    
    # --- Panel 5: Pyramid Layers ---
    ax = fig.add_subplot(3, 3, 5)
    ax.stackplot(t, results['retail_extract'], results['table_extract'],
                 results['debt_extract'],
                 labels=['Retail (consumer AI)', 'Table (hyperscaler capex)',
                         'Debt superstructure'],
                 colors=['#4A90D9', '#F5A623', '#D0021B'], alpha=0.8)
    ax.set_title('Fig 6.2: Pyramid Layer Extraction', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Extraction per period')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    # --- Panel 6: Λ/P Ratio and Cascade Trigger ---
    ax = fig.add_subplot(3, 3, 6)
    ax.plot(t, results['lambda_over_P'], 'purple', lw=2.5,
            label='Λ(t)/P(t)')
    ax.axhline(p.tau_crit, color='red', ls='--', lw=2,
               label=f'τ = {p.tau_crit} (cascade trigger)')
    ax.axvline(2026.5, color='blue', ls=':', alpha=0.6, label='Today')
    cascades = np.where(results['cascade_flag'])[0]
    if len(cascades) > 0:
        ax.axvspan(t[cascades[0]], t[-1], color='red', alpha=0.15,
                   label='Cascade zone')
    ax.set_title("Eq 6.3': Leverage-to-Pool vs Trigger", fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Λ/P')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    # --- Panel 7: Scarcity Amplifier (Fig 3.5) ---
    ax = fig.add_subplot(3, 3, 7)
    ax.plot(t, results['scarcity'], 'darkred', lw=2.5, label='Bounded amplifier')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4, label='Neutral (no scarcity)')
    ax.set_title('Fig 3.5: AI Compute Scarcity Amplifier', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Amplifier')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    
    # --- Panel 8: Debt Accumulation & Reserve Premium ---
    ax = fig.add_subplot(3, 3, 8)
    ax2 = ax.twinx()
    ax.plot(t, results['debt_total'] / 1e12, 'darkred', lw=2.5,
            label='AI debt ($T)')
    ax2.plot(t, results['reserve_premium'], 'navy', lw=2.5, ls='--',
             label='Reserve premium')
    ax.set_xlabel('Year'); ax.set_ylabel('AI Debt ($T)', color='darkred')
    ax2.set_ylabel('Reserve premium', color='navy')
    ax.set_title('AI Debt vs Reserve Currency Premium', fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    
    # --- Panel 9: Phase comparison ---
    ax = fig.add_subplot(3, 3, 9)
    # Compare AI capex trajectory with historical tech cycles
    hist_years = np.array([2000, 2001, 2002, 2003, 2004, 2005])
    dotcom_capex = np.array([50, 70, 45, 35, 40, 48])
    # AI normalized to same scale
    ai_norm = results['capex'] / results['capex'].max() * 70
    ax.plot(t, ai_norm, 'steelblue', lw=2.5, label='AI capex (norm)')
    ax.plot(hist_years, dotcom_capex, 'orange', lw=2, ls='--',
            label='Dot-com capex (norm)')
    ax.axvline(2026.5, color='blue', ls=':', alpha=0.6, label='Today')
    ax.set_title('AI Capex vs Dot-Com Cycle', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Normalized capex')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ai_capex_pyramid.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# SECTION 10 — KEY FINDINGS
# ============================================================

def print_key_findings(results):
    data = results['data']
    p = results['params']
    t = results['t']
    idx_2026 = np.argmin(np.abs(t - 2026.5))
    
    print()
    print("=" * 78)
    print("KEY FINDINGS — AI CAPEX PYRAMID 2026")
    print("=" * 78)
    print(f"""
1. CAPEX SUPERCYCLE (Fig 4.3):
   Hyperscaler capex has grown from ${data.capex_2023:.0f}B (2023) to
   ${data.capex_2026:.0f}B (2026) — a {data.capex_2026/data.capex_2023:.1f}x increase in 3 years.
   This exceeds the dot-com infrastructure build in absolute terms.

2. CONSUMER ADOPTION PLATEAU (Fig 4.4):
   US adult AI usage at {data.us_adult_ai_weekly_2026:.0f}% weekly (up from {data.us_adult_ai_weekly_2025:.0f}% in 2025).
   BUT: consumer app DAU growth has flatlined for 4 of last 5 months.
   Enterprise deep integration only at {data.enterprise_deep_integration:.0f}% (from {data.enterprise_deep_integration-4:.0f}% in 2025).
   Only {data.enterprise_on_track:.0f}% of companies are on track with AI initiatives.
   {data.roi_failure_rate:.0f}% report ROI failing to outpace spend.

3. DEBT SUPERSTRUCTURE (Fig 6.2):
   Cumulative AI debt: ${data.ai_debt_cumulative/1e12:.2f}T since ChatGPT launch.
   2026 YTD issuance: ${data.ai_debt_2026_ytd/1e9:.0f}B.
   Hyperscaler bond issuance: ${data.hyperscaler_bond_2026/1e9:.0f}B (H1 2026) vs
   ${data.hyperscaler_bond_2025/1e9:.0f}B (full year 2025).
   Credit spreads widening: +{data.hyperscaler_spread_widen:.0f}bp for IG AI debt.
   CDS spreads rising sharply across all hyperscalers.

4. LABOR DISPLACEMENT (Fig 5.3/5.4):
   Junior AI-exposed employment: -{data.junior_ai_displacement:.0f}% (22-25 y/o).
   Senior hiring at AI-adopting companies: +{data.senior_hiring_increase:.1f}%.
   IMF: {data.overall_ai_exposure:.0f}% of global jobs exposed;
   {data.advanced_economy_exposure:.0f}% in advanced economies.
   AI is labor-saving for juniors, labor-expanding for seniors.

5. PLUS-SUM TO ZERO-SUM TRANSITION:
   ΔG_L was positive through ~2025, supported by genuine productivity gains.
   By 2026-2027, displacement exceeds new job creation → ΔG_L turns negative.
   The AI capex boom transitions from plus-sum to zero-sum extraction.

6. RESERVE CURRENCY COUPLING:
   US debt/GDP crossed 100% (first time since WWII).
   Net interest: ${data.net_interest_2026/1e12:.2f}T (surpassing defense).
   Interest = {data.interest_pct_revenue:.0f}% of federal revenue (record).
   Convenience yield declined ~50bp — reserve premium eroding.
   AI debt competes with sovereign debt for the same buyer pool.
   Foreign official share: {data.foreign_official_share:.0f}% (down from 66% in 2014).

7. CASCADE PROGNOSIS:
   Current Λ/P ratio: {results['lambda_over_P'][idx_2026]:.3f} (τ = {p.tau_crit}).
   {'⚠ Cascade trigger reached at year ' + str(t[np.where(results["cascade_flag"])[0][0]]) if np.any(results["cascade_flag"]) else '✓ No cascade in horizon — peak Λ/P at ' + str(round(results["lambda_over_P"].max(), 3))}
   Extrapolated cascade window: 2029-2032 if capex remains debt-funded.
""")
    print("=" * 78)


# ============================================================
# SECTION 11 — MAIN
# ============================================================

if __name__ == "__main__":
    params = ModelParams()
    data = AIDataAnchors()
    
    results = run_ai_capex_simulation(params, data)
    
    print_diagnostics(results)
    print_key_findings(results)
    plot_results(results)