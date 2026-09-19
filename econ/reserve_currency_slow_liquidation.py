#!/usr/bin/env python3
"""
reserve_currency_depletion.py
==============================
Simulates the plus-sum/zero-sum boundary for a reserve currency (USD),
based on Unified Wave-Pyramid Economic Models:

  - Fig 5.1-5.3: Plus-sum GDP (labour, resource, capital components)
  - Fig 4.3:     Access boom carrying capacity K(w)
  - Fig 6.2:     Pyramid layer structure (slot, table, debt superstructure)
  - Eq 6.3:      Pool depletion: P(t+1) = P(t)(1+ψ) - E(t)
  - Eq 6.3':     Trigger: Λ/P >= τ => cascade

Real data anchor: 2026-09-19 US Treasury market.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

# ============================================================
# SECTION 1 — REAL-WORLD ANCHOR DATA (2026-09-19)
# ============================================================
ANCHORS_2026 = {
    'year':                   2026.72,        # Sept 19
    'us_debt_total':          40.1e12,        # $40.1T
    'us_debt_public':         32.3e12,        # publicly held
    'us_gdp':                 31.8e12,        # ~$31.8T
    'net_interest_annual':    1.15e12,        # $1.15T
    'foreign_total_holdings': 9.3e12,         # $9.3T
    'foreign_official_share': 0.41,           # 41% of foreign
    'usd_reserve_share':      0.5713,         # 57.13%
    'primary_dealer_share':   0.14,           # auction share
    'hedge_fund_holdings':    2.6e12,
    'china_holdings':         0.618e12,
}

# Historical anchors for calibration
HISTORICAL = {
    2014: {'debt_gdp': 0.74, 'foreign_official_pct': 0.66, 'reserve_share': 0.66},
    2020: {'debt_gdp': 1.00, 'foreign_official_pct': 0.55, 'reserve_share': 0.61},
    2026: {'debt_gdp': 1.26, 'foreign_official_pct': 0.41, 'reserve_share': 0.5713},
}

# ============================================================
# SECTION 2 — MODEL PARAMETERS
# ============================================================
@dataclass
class ReserveModelParams:
    # Time horizon
    t_start:  float = 1944.0   # Bretton Woods
    t_end:    float = 2100.0
    dt:       float = 0.25     # quarterly

    # Fig 4.3 — access boom carrying capacity
    K_inf:    float = 1.0      # normalized global carrying capacity (reserve demand)
    K_sc:     float = 0.15     # scarcity-era capacity
    lambda_K: float = 0.045    # inverse-exp approach rate
    w_star:   float = 1971.0   # access threshold (Nixon shock)
    w_f:      float = 2008.0   # fatigue onset (GFC)

    # Fig 5.x — plus-sum GDP contributions (normalized to [0,1])
    eta_R:    float = 0.012    # resource access growth rate
    alpha_L:  float = 0.028    # labour productivity contribution
    beta_L:   float = 0.6      # post-industrial labour drag
    rho_K:    float = 0.020    # capital accumulation rate
    delta_A:  float = 0.008    # automation/debt drag
    T_horizon: float = 156.0   # years from 1944

    # Fig 6.2 — pyramid layer leverage
    h_ret:    float = 0.02     # retail base extraction
    h_mid:    float = 0.05     # mid-tier extraction
    xi:       float = 3.5      # leverage ratio
    tau_crit: float = 2.0      # collapse trigger (lower bound)

    # Eq 6.3 — pool dynamics
    psi_base: float = 0.035    # baseline hype/confidence inflow
    psi_decay: float = 0.018   # decay rate post-fatigue
    E_base:   float = 0.045    # baseline extraction rate (interest/carry)


# ============================================================
# SECTION 3 — FIG 4.3: ACCESS BOOM CARRYING CAPACITY
# ============================================================
def carrying_capacity(t, p: ReserveModelParams):
    """
    Fig 4.3: K(w) = K_sc + (K_inf - K_sc)(1 - exp(-λ·I·ρ·(w - w*)))
    Applied to reserve currency demand as the "commodity" being accessed.
    """
    if t < p.w_star:
        # Scarcity era
        return p.K_sc * (1 + 0.005 * (t - p.t_start))
    else:
        w = t - p.w_star
        K = p.K_sc + (p.K_inf - p.K_sc) * (1 - np.exp(-p.lambda_K * w))
        # Fatigue damping post-GFC (Fig 4.4 analog)
        if t > p.w_f:
            phi = (t - p.w_f) / max(p.t_end - p.w_f, 1.0)
            K = K * (1 - 0.35 * phi)
        return K


# ============================================================
# SECTION 4 — FIG 5.x: PLUS-SUM GDP CONTRIBUTIONS
# ============================================================
def plus_sum_components(t, p: ReserveModelParams):
    """
    Returns the four GDP growth contributions (Fig 5.4):
      ΔG_R: resource access
      ΔG_L: labour (plus-sum; sign flips when demographic/immigration drag > gain)
      ΔG_K: capital
      ΔG_A: automation/debt drag
    """
    T = max(t - p.t_start, 0.0)

    # Resource: rises with access, decays post-peak
    dG_R = p.eta_R * np.exp(-0.008 * max(T - 30, 0))

    # Labour: plus-sum during demographic dividend, then drag
    # Positive 1944-2008, declining 2008-2026, turning negative after
    if t < 2008:
        dG_L = p.alpha_L * (1 - 0.3 * T / p.T_horizon)
    elif t < 2026:
        dG_L = p.alpha_L * 0.55 * (1 - p.beta_L * (t - 2008) / 18)
    else:
        # Negative: immigration restrictions + aging
        dG_L = -p.alpha_L * 0.35 * (1 + 0.04 * (t - 2026))

    # Capital: accumulation
    dG_K = p.rho_K * np.exp(-0.005 * T)

    # Automation / debt drag (Fig 5.4 component)
    dG_A = -p.delta_A * min(1.0, T / p.T_horizon * 1.5)

    return dG_R, dG_L, dG_K, dG_A


# ============================================================
# SECTION 5 — FIG 6.2: PYRAMID LEVERAGE
# ============================================================
def pyramid_leverage(t, debt_total, pool_value, p: ReserveModelParams):
    """
    Computes Λ(t) — total leverage of the pyramid superstructure.

    Λ(t) = W_ret·h_ret + W_mid·h_mid + ξ·Σ_j L_j·r_j(t)
    """
    # Retail (slot) layer
    W_ret = pool_value * 0.35
    E_ret = W_ret * p.h_ret

    # Mid-tier (table) layer with leverage
    W_mid = pool_value * 0.40
    E_mid = W_mid * p.h_mid

    # Debt superstructure
    E_debt = debt_total * 0.028   # effective carrying cost

    Lambda = E_ret + E_mid + p.xi * E_debt
    return Lambda, {'E_ret': E_ret, 'E_mid': E_mid, 'E_debt': E_debt}


# ============================================================
# SECTION 6 — MAIN SIMULATION LOOP
# ============================================================
def run_simulation(p: ReserveModelParams = None):
    if p is None:
        p = ReserveModelParams()

    t_grid = np.arange(p.t_start, p.t_end + p.dt, p.dt)
    n = len(t_grid)

    # State variables
    debt_total       = np.zeros(n)
    debt_gdp         = np.zeros(n)
    gdp              = np.zeros(n)
    pool             = np.zeros(n)         # buyer pool for reserves
    K_res            = np.zeros(n)         # carrying capacity
    Lambda           = np.zeros(n)         # pyramid leverage
    lambda_over_P    = np.zeros(n)         # Λ/P ratio
    psi_t            = np.zeros(n)         # hype inflow
    E_t              = np.zeros(n)         # extraction
    official_share   = np.zeros(n)         # foreign official %
    reserve_share    = np.zeros(n)         # USD global reserve %
    cascade_flag     = np.zeros(n, dtype=bool)

    # Initial conditions (1944)
    gdp[0]          = 0.30        # normalized (US share of global GDP ~30% → 1.0)
    debt_total[0]   = 0.12        # ~$280B in 1944 USD → normalized
    pool[0]         = 1.0         # full buyer pool
    K_res[0]        = carrying_capacity(p.t_start, p)
    official_share[0] = 0.80      # high official share at Bretton Woods
    reserve_share[0]  = 0.70      # USD/GBP split initially

    # Historical calibration injection points
    calibration_years = list(HISTORICAL.keys())

    for i in range(1, n):
        t = t_grid[i]
        dt = t_grid[i] - t_grid[i-1]

        # --- Fig 4.3: carrying capacity ---
        K_res[i] = carrying_capacity(t, p)

        # --- Fig 5.4: plus-sum GDP growth ---
        dG_R, dG_L, dG_K, dG_A = plus_sum_components(t, p)
        gdp[i] = gdp[i-1] * (1 + dG_R + dG_L + dG_K + dG_A)

        # --- Debt dynamics: fiscal deficit accumulates ---
        # Deficit rate grows when ΔG_L is negative (labour drag)
        primary_deficit_rate = 0.028 + max(0, -dG_L) * 1.5
        interest_rate = 0.032 + 0.025 * (debt_total[i-1] / max(gdp[i-1], 0.1))
        debt_total[i] = debt_total[i-1] * (1 + interest_rate * dt) + \
                        gdp[i-1] * primary_deficit_rate * dt
        debt_gdp[i] = debt_total[i] / max(gdp[i], 0.01)

        # --- Eq 6.3: pool dynamics ---
        # Hype inflow declines post-fatigue
        if t < p.w_f:
            psi_t[i] = p.psi_base * K_res[i]
        else:
            psi_t[i] = p.psi_base * K_res[i] * np.exp(-p.psi_decay * (t - p.w_f))

        # Extraction rises with debt/GDP
        E_t[i] = p.E_base * (1 + 0.4 * max(0, debt_gdp[i] - 0.7))

        pool[i] = max(pool[i-1] * (1 + psi_t[i] * dt) - E_t[i] * dt, 1e-4)

        # --- Fig 6.2: pyramid leverage ---
        Lambda[i], _ = pyramid_leverage(t, debt_total[i], pool[i], p)

        # --- Eq 6.3': cascade trigger ---
        lambda_over_P[i] = Lambda[i] / max(pool[i], 1e-6) * 0.05  # scaled
        if lambda_over_P[i] >= p.tau_crit:
            cascade_flag[i] = True

        # --- Official share dynamics ---
        # Official buyers exit when debt/GDP rises AND pool shrinks
        if t > 2014:
            exit_rate = 0.018 + 0.012 * max(0, debt_gdp[i] - 1.0)
            official_share[i] = max(official_share[i-1] - exit_rate * dt, 0.05)
        else:
            official_share[i] = official_share[i-1] * (1 - 0.002 * dt)

        # --- Reserve share dynamics ---
        # USD share declines when cascade risk is high
        reserve_drag = 0.005 * max(0, lambda_over_P[i] - 0.8)
        reserve_share[i] = max(reserve_share[i-1] * (1 - reserve_drag * dt), 0.15)

        # --- Inject historical calibration anchors ---
        for cy in calibration_years:
            if abs(t - cy) < p.dt/2:
                target = HISTORICAL[cy]
                # Soft pull toward historical anchor
                debt_gdp[i]     = 0.7 * debt_gdp[i]     + 0.3 * target['debt_gdp']
                official_share[i] = 0.7 * official_share[i] + 0.3 * target['foreign_official_pct']
                reserve_share[i]  = 0.7 * reserve_share[i]  + 0.3 * target['reserve_share']

    return {
        't': t_grid, 'gdp': gdp, 'debt_total': debt_total, 'debt_gdp': debt_gdp,
        'pool': pool, 'K_res': K_res, 'Lambda': Lambda,
        'lambda_over_P': lambda_over_P, 'psi_t': psi_t, 'E_t': E_t,
        'official_share': official_share, 'reserve_share': reserve_share,
        'cascade_flag': cascade_flag, 'params': p,
    }


# ============================================================
# SECTION 7 — DIAGNOSTIC: 2026 SNAPSHOT
# ============================================================
def print_2026_snapshot(results):
    t = results['t']
    idx_2026 = np.argmin(np.abs(t - 2026.72))

    print("=" * 70)
    print("RESERVE CURRENCY DEPLETION MODEL — 2026 SNAPSHOT")
    print("=" * 70)
    print(f"  Model date:                  2026.72 (Sept 19, 2026)")
    print(f"  GDP (normalized):            {results['gdp'][idx_2026]:.4f}")
    print(f"  Debt (normalized):           {results['debt_total'][idx_2026]:.4f}")
    print(f"  Debt/GDP:                    {results['debt_gdp'][idx_2026]*100:.1f}%")
    print(f"  Buyer pool (normalized):     {results['pool'][idx_2026]:.4f}")
    print(f"  Carrying capacity K_res:     {results['K_res'][idx_2026]:.4f}")
    print(f"  Pyramid leverage Λ:          {results['Lambda'][idx_2026]:.4f}")
    print(f"  Λ/P ratio:                   {results['lambda_over_P'][idx_2026]:.3f}")
    print(f"  Hype inflow ψ:               {results['psi_t'][idx_2026]:.4f}")
    print(f"  Extraction E:                {results['E_t'][idx_2026]:.4f}")
    print(f"  Foreign official share:      {results['official_share'][idx_2026]*100:.1f}%")
    print(f"  USD reserve share:           {results['reserve_share'][idx_2026]*100:.1f}%")
    print(f"  Cascade triggered:           {results['cascade_flag'][idx_2026]}")
    print("=" * 70)

    # Find cascade onset year
    cascades = np.where(results['cascade_flag'])[0]
    if len(cascades) > 0:
        print(f"\n  ⚠ Cascade trigger reached in year: {t[cascades[0]]:.1f}")
        print(f"    Λ/P at trigger: {results['lambda_over_P'][cascades[0]]:.3f} "
              f"(τ = {results['params'].tau_crit})")
    else:
        print(f"\n  ✓ No cascade trigger within horizon")
        print(f"    Peak Λ/P: {results['lambda_over_P'].max():.3f} "
              f"(τ = {results['params'].tau_crit})")


# ============================================================
# SECTION 8 — VISUALIZATION
# ============================================================
def plot_results(results):
    t = results['t']
    p = results['params']

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Panel 1: Plus-sum vs zero-sum boundary
    ax = axes[0, 0]
    # Recompute components on the grid for plotting
    dG_L_series = np.array([plus_sum_components(ti, p)[1] for ti in t])
    ax.fill_between(t, 0, dG_L_series, where=(dG_L_series >= 0),
                    color='green', alpha=0.4, label='ΔG_L > 0 (plus-sum)')
    ax.fill_between(t, 0, dG_L_series, where=(dG_L_series < 0),
                    color='red', alpha=0.4, label='ΔG_L < 0 (zero-sum)')
    ax.plot(t, dG_L_series, 'k-', lw=1.5)
    ax.axhline(0, color='black', ls=':', alpha=0.5)
    ax.axvline(2026.72, color='blue', ls='--', alpha=0.6, label='Today (Sept 2026)')
    ax.set_title('Fig 5.3: Labour contribution ΔG_L — Plus-Sum/Zero-Sum Boundary',
                 fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('ΔG_L (growth contribution)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Carrying capacity (access boom)
    ax = axes[0, 1]
    ax.plot(t, results['K_res'], 'darkblue', lw=2, label='K_res(t)')
    ax.axvline(1971, color='orange', ls='--', alpha=0.6, label='Nixon shock (w*)')
    ax.axvline(2008, color='red', ls='--', alpha=0.6, label='GFC fatigue onset')
    ax.axvline(2026.72, color='blue', ls=':', alpha=0.6, label='Today')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4, label='K∞')
    ax.set_title('Fig 4.3: Reserve Currency Access Boom', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('K(w) — normalized')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: Buyer pool depletion
    ax = axes[1, 0]
    ax.plot(t, results['pool'], 'black', lw=2.2, label='Buyer pool P(t)')
    ax.fill_between(t, 0, results['pool'], color='black', alpha=0.08)
    ax.axvline(2026.72, color='blue', ls='--', alpha=0.6, label='Today')
    ax.set_title('Eq 6.3: Buyer Pool Depletion', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('P(t) — normalized')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 4: Pyramid leverage vs pool
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(t, results['Lambda'], 'darkred', lw=2, label='Pyramid leverage Λ')
    ax2.plot(t, results['pool'], 'navy', lw=2, ls='--', label='Buyer pool P')
    ax.set_xlabel('Year'); ax.set_ylabel('Λ (leverage)', color='darkred')
    ax2.set_ylabel('P (pool)', color='navy')
    ax.set_title('Fig 6.2: Debt Superstructure vs Buyer Pool', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # Panel 5: Λ/P ratio and cascade trigger
    ax = axes[2, 0]
    ax.plot(t, results['lambda_over_P'], 'purple', lw=2.2, label='Λ(t)/P(t)')
    ax.axhline(p.tau_crit, color='red', ls='--', lw=1.8,
               label=f'τ = {p.tau_crit} (cascade trigger)')
    ax.axvline(2026.72, color='blue', ls=':', alpha=0.6, label='Today')
    cascades = np.where(results['cascade_flag'])[0]
    if len(cascades) > 0:
        ax.axvspan(t[cascades[0]], t[-1], color='red', alpha=0.15,
                   label='Cascade zone')
    ax.set_title('Eq 6.3\': Leverage-to-Pool Ratio vs Trigger', fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Λ/P')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 6: Official share and reserve share
    ax = axes[2, 1]
    ax.plot(t, results['official_share']*100, 'darkorange', lw=2,
            label='Foreign official share (%)')
    ax.plot(t, results['reserve_share']*100, 'darkgreen', lw=2,
            label='USD global reserve share (%)')
    ax.axvline(2026.72, color='blue', ls=':', alpha=0.6, label='Today')
    ax.axhline(15, color='red', ls=':', alpha=0.4, label='Critical floor (15%)')
    ax.set_title('Buyer Composition: Official Exit and Reserve Share Decline',
                 fontweight='bold')
    ax.set_xlabel('Year'); ax.set_ylabel('Percent (%)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('reserve_currency_depletion.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# SECTION 9 — ANGLO-AMERICAN LOAN COMPARISON
# ============================================================
def print_anglo_american_comparison():
    print()
    print("=" * 70)
    print("HISTORICAL PRECEDENT: ANGLO-AMERICAN LOAN (1946)")
    print("=" * 70)
    print("  Loan size:              $3.75B (~15% of UK GDP)")
    print("  Terms:                  50 years, 2% APR")
    print("  Annual repayment:       $138.4M")
    print("  Outcome:                No default, but GBP lost reserve premium")
    print()
    print("  RESERVE CURRENCY LIQUIDATION IS A PREMIUM EVENT, NOT A DEFAULT:")
    print("    - Currency continues to exist")
    print("    - London remains financial centre")
    print("    - But GBP-debt no longer = 'risk-free asset'")
    print()
    print("  USD parallel (2026):")
    print("    - Debt/GDP 126%, interest 15% of budget")
    print("    - Foreign official share falling (41% → trend 20% by 2035)")
    print("    - Reserve share 57% → trend 45-50% by 2035")
    print("    - No single alternative — fragmentation not substitution")
    print("=" * 70)


# ============================================================
# SECTION 10 — MAIN
# ============================================================
if __name__ == "__main__":
    params = ReserveModelParams()
    results = run_simulation(params)

    print_2026_snapshot(results)
    print_anglo_american_comparison()

    # Cascade onset projection
    t = results['t']
    cascades = np.where(results['cascade_flag'])[0]
    print()
    print("=" * 70)
    print("CASCADE ONSET PROJECTION")
    print("=" * 70)
    if len(cascades) > 0:
        for c in cascades[:5]:
            print(f"  Year {t[c]:.2f}:  Λ/P = {results['lambda_over_P'][c]:.3f}, "
                  f"debt/GDP = {results['debt_gdp'][c]*100:.1f}%, "
                  f"pool = {results['pool'][c]:.4f}")
    else:
        # Find when Λ/P will cross τ under current trend
        idx_peak = np.argmax(results['lambda_over_P'])
        growth_rate = (results['lambda_over_P'][idx_peak] -
                       results['lambda_over_P'][idx_peak-40]) / 40
        years_to_tau = (params.tau_crit - results['lambda_over_P'][idx_peak]) / max(growth_rate, 1e-6)
        print(f"  Peak Λ/P in horizon: {results['lambda_over_P'].max():.3f} at year {t[idx_peak]:.1f}")
        print(f"  Extrapolated τ crossing: ~{t[idx_peak] + years_to_tau:.1f}")
    print("=" * 70)

    plot_results(results)