#!/usr/bin/env python3
"""
fed_rate_binomial_prediction.py
=================================
Binomial prediction of Fed rate decision (Sept 16, 2026)
using:
  - Supply-chain depth model (Figure 3.2 stage recursion)
  - Harmonic equilibrium spectral derivative dzeta_dt
  - Energy prices (WTI, diesel) as multiplicative cost factors
  - VIX entropy as geopolitical volatility multiplier

Date: 2026-09-15
"""

import math
import numpy as np
from scipy.stats import norm

# ============================================================
# SECTION 1 — SPECTRAL ENGINE (equilibrium.py)
# ============================================================
ALPHA = 0.3628
K_MODES = 25
C_COEFFS = np.ones(2 * K_MODES + 1)

def dzeta_dt(t, c, alpha=ALPHA):
    """Analytical spectral derivative dζ/dt"""
    K = (len(c) - 1) // 2
    d = 0.0
    for i in range(1, K + 1):
        d += -2.0 * c[K + i] * (i / alpha) * np.sin(t * i / alpha)
    return d

def zeta(t, c, alpha=ALPHA):
    """Spectral sum ζ(t)"""
    K = (len(c) - 1) // 2
    s = c[K]
    for i in range(1, K + 1):
        s += c[K + i] * np.cos(t * i / alpha) * 2
    return s

# ============================================================
# SECTION 2 — SUPPLY-CHAIN DEPTH MODEL (Figure 3.2)
# ============================================================

# --- Macro inputs (Sept 15, 2026) ---
WTI        = 103.42          # $/bbl
BRENT      = 106.93          # $/bbl
DIESEL_US  = 6.23            # $/gallon (retail)
VIX        = 17.10           # CBOE volatility index
RHO_10Y    = 0.0497          # 10Y Treasury yield
NFSI       = 48.96           # geopolitical stability (below neutral)

# Energy reference (pre-shock baseline, Sept 2025)
WTI_REF    = 72.0
DIESEL_REF = 3.69

# --- Entropy calculation (VIX as market entropy proxy) ---
# Normalize VIX to [0,1] where 0 = complacent (VIX=12), 1 = crisis (VIX=40)
VIX_NORM = np.clip((VIX - 12.0) / (40.0 - 12.0), 0.0, 1.0)

# Geopolitical entropy from NFSI (lower NFSI = higher instability)
# NFSI neutral = 50; below 50 = elevated risk
GEO_ENTROPY = np.clip((50.0 - NFSI) / 50.0, 0.0, 1.0)  # 0.0208

# Combined entropy multiplier
H_ENTROPY = VIX_NORM + GEO_ENTROPY  # ~0.203

# --- Energy cost multipliers ---
# Energy enters the supply chain as e_k at each stage.
# The ratio to baseline gives the multiplier.
ENERGY_MULT_WTI    = WTI / WTI_REF        # 103.42 / 72.0 = 1.436
ENERGY_MULT_DIESEL = DIESEL_US / DIESEL_REF  # 6.23 / 3.69 = 1.688

# Composite energy multiplier (weighted: diesel more relevant for logistics/transport)
ENERGY_MULT = 0.4 * ENERGY_MULT_WTI + 0.6 * ENERGY_MULT_DIESEL
print(f"Energy multiplier (composite): {ENERGY_MULT:.4f}")
print(f"  WTI component:    {ENERGY_MULT_WTI:.4f}")
print(f"  Diesel component: {ENERGY_MULT_DIESEL:.4f}")

# --- Supply-chain stage parameters for the Fed's "policy cost stack" ---
# The Fed's decision is modelled as a supply chain:
#   Stage 1: Inflation data input (CPI)
#   Stage 2: Energy pass-through
#   Stage 3: Labour market tightness
#   Stage 4: Financial conditions
#   Stage 5: Policy rate decision
#
# Each stage: (m, p, n, ell_w, eta_e, mu)
#   m     = material conversion ratio (how much of prior stage flows through)
#   p     = binomial failure probability
#   n     = trials per stage
#   ell_w = labour/political cost (plus-sum)
#   eta_e = energy cost (base, before multiplier)
#   mu    = stage markup

fed_stages = [
    # Stage 1: Inflation data (CPI input)
    # p elevated: August CPI MoM +0.3% vs consensus 0.2%
    (1.3, 0.25, 2, 0.0, 0.0, 0.05),
    # Stage 2: Energy pass-through
    # Energy CPI +2.1% MoM, gasoline +3.9% -> high failure probability
    (1.5, 0.35, 3, 0.0, 0.12, 0.08),
    # Stage 3: Labour market tightness
    # Jobs data mixed; p moderate
    (1.1, 0.18, 2, 0.0, 0.03, 0.06),
    # Stage 4: Financial conditions
    # 10Y broke 5%, VIX spiked -> elevated p
    (1.2, 0.22, 2, 0.0, 0.05, 0.10),
    # Stage 5: Policy rate decision (final stage)
    # Binary outcome: hike or hold
    (1.0, 0.15, 1, 0.0, 0.02, 0.15),
]

# Baseline raw cost C0: the "neutral" policy input cost
C0_FED = 1.0  # normalized

# Interest rate per stage (Fed funds effective rate / stages)
RHO_STAGE = 0.0363 / 5  # 3.63% effective / 5 stages = 0.726% per stage

# --- Run the stage recursion with energy and entropy multipliers ---
def run_fed_chain(energy_mult, h_entropy, stages):
    """
    Stage-by-stage cost recursion (Figure 3.2) with:
      - energy multiplier applied to eta*e terms
      - entropy multiplier applied to the scarcity amplifier
    """
    C = C0_FED
    cost_stack = [C]
    stage_details = []

    for k, (m, p, n, ell_w, eta_e, mu) in enumerate(stages):
        # Entropy amplifies the failure probability
        # Higher market entropy -> higher probability of policy error
        p_eff = np.clip(p * (1 + h_entropy * 2.0), 0.0, 0.95)

        # Scarcity amplifier
        amp = (1 - p_eff) ** (-n)

        # Material cost with amplification
        mat_cost = m * C * amp

        # Energy cost with multiplier
        energy_cost = eta_e * energy_mult

        # Stage markup
        markup = mu * C

        # Sum before interest
        stage_sum = mat_cost + energy_cost + markup

        # Apply interest
        C = (1 + RHO_STAGE) * stage_sum

        cost_stack.append(C)
        stage_details.append({
            'stage': k + 1,
            'p_base': p,
            'p_eff': p_eff,
            'amplifier': amp,
            'mat_cost': mat_cost,
            'energy_cost': energy_cost,
            'markup': markup,
            'C_k': C
        })

    return cost_stack, stage_details

cost_stack, details = run_fed_chain(ENERGY_MULT, H_ENTROPY, fed_stages)

# --- Final "policy cost" ---
P_fed = cost_stack[-1]

# --- Decompose into hike/hold components ---
# The scarcity compounding factor Phi_raw
Phi_raw = np.prod([(fed_stages[k][0] * (1 + details[k]['p_eff'])**(-fed_stages[k][2])
                     + fed_stages[k][5]) for k in range(len(fed_stages))])

# Energy contribution to the cost stack
energy_total = sum(d['energy_cost'] for d in details)
mat_total = sum(d['mat_cost'] for d in details)
markup_total = sum(d['markup'] for d in details)

# ============================================================
# SECTION 3 — BINOMIAL PROBABILITY MAPPING
# ============================================================

# Map the policy cost P_fed to a binomial probability of hike
# Logistic mapping: P(hike) = 1 / (1 + exp(-beta * (P_fed - P_threshold)))
# P_threshold is calibrated so that P_fed = 1.0 -> P(hike) = 0.50
P_threshold = 1.0
beta_scale = 3.5  # steepness of the logistic curve

P_hike_model = 1.0 / (1.0 + np.exp(-beta_scale * (P_fed - P_threshold)))
P_hold_model = 1.0 - P_hike_model

# --- Market-implied probabilities (CME FedWatch) ---
P_hike_market = 0.924
P_hold_market = 0.076

# --- Blend model and market (Bayesian update) ---
# Prior: market pricing (strong prior)
# Likelihood: our supply-chain model
# Posterior: weighted combination
w_model = 0.35  # weight on our model
w_market = 0.65 # weight on market pricing

P_hike_blend = w_model * P_hike_model + w_market * P_hike_market
P_hold_blend = 1.0 - P_hike_blend

# ============================================================
# SECTION 4 — SUPRETRACE / ENTROPY / MASS (harmonic model)
# ============================================================

# Time discretization at harmonic points
T_HARM = 122.5
C_HARM = np.ones(51)

zeta_val = zeta(T_HARM, C_HARM)
dzeta_val = dzeta_dt(T_HARM, C_HARM)

# Supertrace of the cost stack (alternating sum)
S_super = sum((-1)**k * cost_stack[k] for k in range(len(cost_stack)))

# Entropy
H_ent = -ALPHA * abs(S_super) / len(cost_stack) * math.log(abs(S_super) / len(cost_stack)) if S_super != 0 else 0.0

# Invariant mass
m_inv = abs(S_super) * math.exp(-H_ent)

# ============================================================
# SECTION 5 — OUTPUT
# ============================================================
print("=" * 68)
print("FED RATE DECISION — BINOMIAL PREDICTION")
print("Date: 2026-09-15 (decision on 2026-09-16)")
print("=" * 68)
print()
print("--- Market Inputs ---")
print(f"  Federal funds rate:     3.50% – 3.75%")
print(f"  WTI crude:              ${WTI:.2f}/bbl")
print(f"  Diesel (US retail):     ${DIESEL_US:.2f}/gallon")
print(f"  VIX:                    {VIX:.2f}")
print(f"  10Y Treasury:           {RHO_10Y*100:.2f}%")
print(f"  NFSI (geopolitical):    {NFSI:.2f}")
print()
print("--- Derived Multipliers ---")
print(f"  Energy multiplier:      {ENERGY_MULT:.4f}")
print(f"  VIX normalized entropy: {VIX_NORM:.4f}")
print(f"  Geo entropy:            {GEO_ENTROPY:.4f}")
print(f"  Combined entropy H:     {H_ENTROPY:.4f}")
print()
print("--- Supply-Chain Cost Stack ---")
print(f"{'Stage':<8} {'p_base':>8} {'p_eff':>8} {'Amplifier':>10} {'Mat Cost':>10} {'Energy':>10} {'C_k':>10}")
print("-" * 68)
for d in details:
    print(f"S{d['stage']:<6} {d['p_base']:>8.3f} {d['p_eff']:>8.3f} "
          f"{d['amplifier']:>10.4f} {d['mat_cost']:>10.4f} {d['energy_cost']:>10.4f} {d['C_k']:>10.4f}")
print(f"{'Final':<8} {'':>8} {'':>8} {'':>10} {'':>10} {'':>10} {P_fed:>10.4f}")
print()
print("--- Cost Decomposition ---")
print(f"  Scarcity compounding Φ_raw:  {Phi_raw:.4f}")
print(f"  Material stack total:        {mat_total:.4f}")
print(f"  Energy stack total:          {energy_total:.4f}")
print(f"  Markup stack total:          {markup_total:.4f}")
print()
print("--- Harmonic Engine State ---")
print(f"  ζ(t) at t={T_HARM}:            {zeta_val:+.4f}")
print(f"  dζ/dt at t={T_HARM}:           {dzeta_val:+.4f}")
print(f"  Supertrace S:                {S_super:+.4f}")
print(f"  Entropy H:                   {H_ent:.4f}")
print(f"  Invariant mass m:            {m_inv:.4f}")
print()
print("=" * 68)
print("BINOMIAL PREDICTION")
print("=" * 68)
print(f"  Model P(hike):        {P_hike_model*100:.1f}%")
print(f"  Model P(hold):        {P_hold_model*100:.1f}%")
print(f"  Market P(hike):       {P_hike_market*100:.1f}%")
print(f"  Market P(hold):       {P_hold_market*100:.1f}%")
print(f"  ─────────────────────────────────────")
print(f"  BLENDED P(hike):      {P_hike_blend*100:.1f}%")
print(f"  BLENDED P(hold):      {P_hold_blend*100:.1f}%")
print()
print(f"  >>> PREDICTION: {'HIKE 25bp' if P_hike_blend > 0.5 else 'HOLD'}")
print("=" * 68)

# ============================================================
# SECTION 6 — SENSITIVITY TABLE
# ============================================================
print()
print("--- SENSITIVITY: P(hike) vs Energy Multiplier ---")
print(f"{'Energy Mult':>12} {'P(hike) Model':>15} {'P(hike) Blend':>15}")
print("-" * 44)
for em in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    cs, _ = run_fed_chain(em, H_ENTROPY, fed_stages)
    pf = cs[-1]
    ph = 1.0 / (1.0 + np.exp(-beta_scale * (pf - P_threshold)))
    phb = w_model * ph + w_market * P_hike_market
    print(f"{em:>12.2f} {ph*100:>14.1f}% {phb*100:>14.1f}%")

print()
print("--- SENSITIVITY: P(hike) vs VIX Entropy ---")
print(f"{'VIX':>8} {'VIX Norm':>10} {'P(hike) Model':>15} {'P(hike) Blend':>15}")
print("-" * 52)
for vix_val in [12, 15, 17.1, 20, 25, 30, 40]:
    vn = np.clip((vix_val - 12.0) / (40.0 - 12.0), 0.0, 1.0)
    h_ent = vn + GEO_ENTROPY
    cs, _ = run_fed_chain(ENERGY_MULT, h_ent, fed_stages)
    pf = cs[-1]
    ph = 1.0 / (1.0 + np.exp(-beta_scale * (pf - P_threshold)))
    phb = w_model * ph + w_market * P_hike_market
    print(f"{vix_val:>8.1f} {vn:>10.4f} {ph*100:>14.1f}% {phb*100:>14.1f}%")