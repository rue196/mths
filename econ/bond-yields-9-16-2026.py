#!/usr/bin/env python3
"""
bond_yield_supply_chain_simulation.py
=======================================
Government bond yield simulation using:
  - Figure 3.2: stage-by-stage cost propagation
  - Figure 3.5: elliptic projection bounded scarcity (Table 1)
  - Eq. 6.1: single-stage disruption propagation
  - Eq. 6.2: correlated disruptions across sub-chains
  - Live market data: 2026-09-16

Output: 2Y, 10Y, 30Y yield predictions with shock decomposition
"""

import numpy as np
import math
from scipy.stats import norm

# ============================================================
# SECTION 1 — MARKET INPUTS (2026-09-16)
# ============================================================

# Energy prices
WTI       = 105.02       # $/bbl
BRENT     = 108.89       # $/bbl
DIESEL    = 6.30         # $/gallon (record)
WTI_REF   = 72.0         # pre-shock baseline
DIESEL_REF= 3.69

# Market risk
VIX       = 16.74        # VIX index
VIX_REF   = 12.0         # complacency baseline
VIX_CRISIS= 40.0         # crisis level

# Geopolitical
NFSI      = 48.96        # NFSI stability (below neutral)
GEO_REF   = 50.0         # neutral

# Fed policy
FED_FUNDS = 0.0363       # 3.63% effective
HIKE_PROB = 0.93         # market-implied hike probability

# Yields (current)
Y_2Y      = 0.04638      # 2Y Treasury
Y_10Y     = 0.04980      # 10Y Treasury
Y_30Y     = 0.05340      # 30Y Treasury

# Inflation
BREAKEVEN_2Y  = 0.0250   # 2Y breakeven
BREAKEVEN_5Y  = 0.0246   # 5Y breakeven
BREAKEVEN_10Y = 0.0239   # 10Y breakeven

# Interest rate per stage (Fed funds / 5 stages)
RHO_STAGE = FED_FUNDS / 5  # 0.726% per stage

# ============================================================
# SECTION 2 — DERIVED MULTIPLIERS
# ============================================================

# Energy multiplier
ENERGY_MULT = 0.4 * (WTI/WTI_REF) + 0.6 * (DIESEL/DIESEL_REF)
# = 0.4*1.459 + 0.6*1.707 = 0.584 + 1.024 = 1.608

# VIX entropy
VIX_NORM = np.clip((VIX - VIX_REF) / (VIX_CRISIS - VIX_REF), 0, 1)  # 0.169
GEO_ENTROPY = np.clip((GEO_REF - NFSI) / GEO_REF, 0, 1)  # 0.0208
H_ENTROPY = VIX_NORM + GEO_ENTROPY  # 0.190

print(f"Energy multiplier:  {ENERGY_MULT:.4f}")
print(f"VIX entropy:        {H_ENTROPY:.4f}")

# ============================================================
# SECTION 3 — STAGE PARAMETERS FOR BOND YIELD CHAIN
# ============================================================

# Each stage: (m, p_base, n, ell_w, eta_e, tau_d, mu, Pi_k, MAT_WEIGHT, ENERGY_WEIGHT)
#   m     = material conversion ratio (how much prior stage flows through)
#   p_base= binomial failure probability
#   n     = trials per stage
#   ell_w = labour cost component
#   eta_e = energy cost component (base, before multiplier)
#   tau_d = logistics / distance cost
#   mu    = stage markup
#   Pi_k  = elliptic projection bound (Figure 3.5)
#   MAT_WEIGHT   = how sensitive this stage is to raw material scarcity
#   ENERGY_WEIGHT= how sensitive this stage is to energy scarcity

bond_stages = [
    # Stage 1: Energy shock (crude oil → input costs)
    # p_base elevated: WTI +46%, diesel +71%
    (1.4, 0.30, 3, 0.02, 0.18, 0.01, 0.05, 0.50, 0.4, 1.0),
    # Stage 2: Raw material pass-through
    # p_base: commodity index rising with energy
    (1.3, 0.25, 2, 0.03, 0.10, 0.02, 0.08, 0.50, 1.0, 0.6),
    # Stage 3: Geopolitical disruption (Iran conflict, Hormuz risk)
    # p_base: high, NFSI below neutral
    (1.2, 0.35, 2, 0.02, 0.05, 0.03, 0.10, 0.80, 0.6, 0.4),
    # Stage 4: Fed policy response (hike probability 93%)
    # p_base: high, binary outcome
    (1.1, 0.20, 1, 0.05, 0.02, 0.01, 0.15, 0.80, 0.3, 0.2),
    # Stage 5: Term premium / long-end supply
    # p_base: moderate, 30Y auction weak
    (1.0, 0.15, 1, 0.03, 0.01, 0.00, 0.20, 0.95, 0.2, 0.1),
]

# Baseline yield (raw material cost C0)
C0_2Y  = Y_2Y
C0_10Y = Y_10Y
C0_30Y = Y_30Y

# ============================================================
# SECTION 4 — EQUATION 6.1: SINGLE-STAGE DISRUPTION
# ============================================================

def run_bond_chain(C0, stages, energy_mult, h_entropy, disruption_stage=None, delta_p=0.0):
    """
    Run the stage-by-stage recursion with:
      - Figure 3.2: C_k = m*C*(1-p)^{-n}*Pi + lw + eta*e + tau*d + mu*C
      - Figure 3.5: elliptic projection Pi_k bounds the amplifier
      - Eq. 6.1: disruption at disruption_stage increases p by delta_p
    """
    C = C0
    cost_stack = [C]
    details = []

    for k, (m, p_base, n, ell_w, eta_e, tau_d, mu, Pi_k,
            mat_w, eng_w) in enumerate(stages):

        # Eq. 6.1: disruption at this stage
        p_stage = p_base
        if disruption_stage is not None and k == disruption_stage:
            p_stage = np.clip(p_base + delta_p, 0.0, 0.95)

        # Entropy amplifies failure probability
        p_eff = np.clip(p_stage * (1 + h_entropy * 2.0), 0.0, 0.95)

        # Figure 3.5: elliptic projection bounded amplifier
        # (1-p)^{-n} -> 1 + Pi_k * ((1-p)^{-n} - 1)
        raw_amp = (1 - p_eff) ** (-n)
        bounded_amp = 1 + Pi_k * (raw_amp - 1)

        # Material cost with bounded amplification
        mat_cost = m * C * bounded_amp

        # Energy cost with multiplier
        energy_cost = eta_e * energy_mult * (1 + eng_w * (energy_mult - 1))

        # Labour cost
        labour_cost = ell_w * (1 + 0.02 * h_entropy)

        # Logistics cost
        logist_cost = tau_d * (1 + 0.5 * (energy_mult - 1))

        # Stage markup
        markup = mu * C

        # Sum before interest
        stage_sum = mat_cost + labour_cost + energy_cost + logist_cost + markup

        # Apply interest
        C = (1 + RHO_STAGE) * stage_sum

        cost_stack.append(C)
        details.append({
            'stage': k + 1,
            'p_base': p_base,
            'p_eff': p_eff,
            'raw_amp': raw_amp,
            'bounded_amp': bounded_amp,
            'Pi_k': Pi_k,
            'mat_cost': mat_cost,
            'energy_cost': energy_cost,
            'labour_cost': labour_cost,
            'logist_cost': logist_cost,
            'markup': markup,
            'C_k': C
        })

    return cost_stack, details

# ============================================================
# SECTION 5 — RUN FOR 2Y, 10Y, 30Y
# ============================================================

# Base case (no additional disruption)
cs_2y,  det_2y  = run_bond_chain(C0_2Y,  bond_stages, ENERGY_MULT, H_ENTROPY)
cs_10y, det_10y = run_bond_chain(C0_10Y, bond_stages, ENERGY_MULT, H_ENTROPY)
cs_30y, det_30y = run_bond_chain(C0_30Y, bond_stages, ENERGY_MULT, H_ENTROPY)

# Disruption scenarios (Eq. 6.1)
# Scenario A: Energy disruption at Stage 1 (delta_p = +0.15)
cs_2y_A,  _ = run_bond_chain(C0_2Y,  bond_stages, ENERGY_MULT, H_ENTROPY, disruption_stage=0, delta_p=0.15)
cs_10y_A, _ = run_bond_chain(C0_10Y, bond_stages, ENERGY_MULT, H_ENTROPY, disruption_stage=0, delta_p=0.15)
cs_30y_A, _ = run_bond_chain(C0_30Y, bond_stages, ENERGY_MULT, H_ENTROPY, disruption_stage=0, delta_p=0.15)

# Scenario B: Geopolitical disruption at Stage 3 (delta_p = +0.20)
cs_2y_B,  _ = run_bond_chain(C0_2Y,  bond_stages, ENERGY_MULT, H_ENTROPY, disruption_stage=2, delta_p=0.20)
cs_10y_B, _ = run_bond_chain(C0_10Y, bond_stages, ENERGY_MULT, H_ENTROPY, disruption_stage=2, delta_p=0.20)
cs_30y_B, _ = run_bond_chain(C0_30Y, bond_stages, ENERGY_MULT, H_ENTROPY, disruption_stage=2, delta_p=0.20)

# ============================================================
# SECTION 6 — EQUATION 6.2: CORRELATED DISRUPTIONS
# ============================================================

def compute_covariance(stages, details, n_samples=100):
    """
    Eq. 6.2: Cov(xi_i, xi_j) across correlated sub-chains.
    Shared inputs: energy, logistics, geopolitical.
    """
    K = len(stages)
    # Disruption indicators based on p_eff
    xi = np.array([d['p_eff'] for d in details])

    # Correlation matrix: energy and logistics are highly correlated
    # via shared fuel costs; geopolitical correlates with both
    corr = np.eye(K)
    for i in range(K):
        for j in range(i+1, K):
            # Energy-logistics correlation
            if i in [0, 1] and j in [4]:
                corr[i, j] = corr[j, i] = 0.75
            # Geopolitical correlates with energy
            elif i in [0, 1] and j == 2:
                corr[i, j] = corr[j, i] = 0.60
            # Fed policy correlates with geopolitical
            elif i == 2 and j == 3:
                corr[i, j] = corr[j, i] = 0.55
            # General positive correlation
            else:
                corr[i, j] = corr[j, i] = 0.30

    # Variance of disruption shocks
    var_xi = xi ** 2

    # Eq. 6.2: total variance
    total_var = np.sum(var_xi)
    for i in range(K):
        for j in range(i+1, K):
            total_var += 2 * corr[i, j] * math.sqrt(var_xi[i] * var_xi[j])

    return corr, total_var

corr_matrix, total_var = compute_covariance(bond_stages, det_10y)

# ============================================================
# SECTION 7 — OUTPUT
# ============================================================

print("=" * 72)
print("GOVERNMENT BOND YIELD — SUPPLY-CHAIN DEPTH SIMULATION")
print("Date: 2026-09-16 (Fed decision day)")
print("=" * 72)
print()
print("--- Market Inputs ---")
print(f"  10Y Treasury:           {Y_10Y*100:.3f}%")
print(f"  30Y Treasury:           {Y_30Y*100:.3f}%")
print(f"  WTI crude:              ${WTI:.2f}/bbl ({WTI/WTI_REF:.2f}x baseline)")
print(f"  Diesel (US retail):     ${DIESEL:.2f}/gallon ({DIESEL/DIESEL_REF:.2f}x)")
print(f"  VIX:                    {VIX:.2f}")
print(f"  NFSI (geopolitical):    {NFSI:.2f}")
print(f"  Fed hike probability:   {HIKE_PROB*100:.0f}%")
print()
print("--- Derived Multipliers ---")
print(f"  Energy multiplier:      {ENERGY_MULT:.4f}")
print(f"  Entropy (VIX + geo):    {H_ENTROPY:.4f}")
print()

# Stage-by-stage for 10Y
print("--- 10Y Stage-by-Stage Decomposition ---")
print(f"{'Stage':<6} {'p_base':>7} {'p_eff':>7} {'Pi_k':>6} {'Raw Amp':>9} {'Bnd Amp':>9} {'C_k':>9}")
print("-" * 60)
for d in det_10y:
    print(f"S{d['stage']:<4} {d['p_base']:>7.3f} {d['p_eff']:>7.3f} "
          f"{d['Pi_k']:>6.2f} {d['raw_amp']:>9.4f} {d['bounded_amp']:>9.4f} "
          f"{d['C_k']*100:>8.3f}%")
print()

# Yield predictions
Y_2Y_pred  = cs_2y[-1]
Y_10Y_pred = cs_10y[-1]
Y_30Y_pred = cs_30y[-1]

print("--- Base Case Yield Predictions ---")
print(f"  2Y:   {Y_2Y_pred*100:.3f}%  (current: {Y_2Y*100:.3f}%)")
print(f"  10Y:  {Y_10Y_pred*100:.3f}%  (current: {Y_10Y*100:.3f}%)")
print(f"  30Y:  {Y_30Y_pred*100:.3f}%  (current: {Y_30Y*100:.3f}%)")
print()

# Disruption scenarios
print("--- Eq. 6.1: Single-Stage Disruption Scenarios ---")
print(f"{'Scenario':<30} {'2Y':>8} {'10Y':>8} {'30Y':>8}")
print("-" * 56)
print(f"{'Base (no disruption)':<30} {Y_2Y_pred*100:>7.3f}% {Y_10Y_pred*100:>7.3f}% {Y_30Y_pred*100:>7.3f}%")
print(f"{'A: Energy shock (S1, +15%p)':<30} {cs_2y_A[-1]*100:>7.3f}% {cs_10y_A[-1]*100:>7.3f}% {cs_30y_A[-1]*100:>7.3f}%")
print(f"{'B: Geopolitical (S3, +20%p)':<30} {cs_2y_B[-1]*100:>7.3f}% {cs_10y_B[-1]*100:>7.3f}% {cs_30y_B[-1]*100:>7.3f}%")
print()

# Eq. 6.2 covariance
print("--- Eq. 6.2: Correlated Disruption Matrix ---")
print("Correlation matrix (upper triangle):")
print("     ", "  ".join(f"S{i+1}" for i in range(5)))
for i in range(5):
    row = f"S{i+1}: "
    for j in range(5):
        if j >= i:
            row += f"{corr_matrix[i,j]:.2f} "
        else:
            row += "     "
    print(row)
print(f"\nTotal disruption variance (Eq. 6.2): {total_var:.6f}")
print(f"  Std deviation: {math.sqrt(total_var):.4f}")
print()

# Yield curve shape
spread_10_2 = Y_10Y_pred - Y_2Y_pred
spread_30_10 = Y_30Y_pred - Y_10Y_pred
print("--- Yield Curve Shape ---")
print(f"  2s10s spread:  {spread_10_2*10000:.1f} bp")
print(f"  10s30s spread: {spread_30_10*10000:.1f} bp")
print(f"  Curve:         {'Steepening' if spread_10_2 > 0 else 'Inverting'}")
print()

# Term premium decomposition
print("--- Term Premium Decomposition (10Y) ---")
for d in det_10y:
    print(f"  Stage {d['stage']}: {d['C_k']*100:.3f}%")
print()

# Sensitivity table
print("--- Sensitivity: 10Y Yield vs Energy Multiplier ---")
print(f"{'Energy Mult':>12} {'10Y Yield':>12} {'Δ vs Base':>12}")
print("-" * 40)
for em in [1.0, 1.2, 1.4, 1.608, 1.8, 2.0]:
    cs, _ = run_bond_chain(C0_10Y, bond_stages, em, H_ENTROPY)
    delta = (cs[-1] - Y_10Y_pred) * 10000
    print(f"{em:>12.3f} {cs[-1]*100:>11.3f}% {delta:>+11.1f}bp")
print()

print("--- Sensitivity: 10Y Yield vs VIX Entropy ---")
print(f"{'VIX':>8} {'Entropy':>10} {'10Y Yield':>12} {'Δ vs Base':>12}")
print("-" * 46)
for vix in [12, 15, 16.74, 20, 25, 30]:
    vn = np.clip((vix - 12)/(40-12), 0, 1)
    h = vn + GEO_ENTROPY
    cs, _ = run_bond_chain(C0_10Y, bond_stages, ENERGY_MULT, h)
    delta = (cs[-1] - Y_10Y_pred) * 10000
    print(f"{vix:>8.1f} {h:>10.4f} {cs[-1]*100:>11.3f}% {delta:>+11.1f}bp")
print()
print("=" * 72)