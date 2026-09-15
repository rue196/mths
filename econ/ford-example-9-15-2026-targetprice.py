import numpy as np
import math

# ============================================================
# FORD MOTOR COMPANY — Supply-Chain Depth Model (Figure 3.2)
# Date: 2026-09-15
# ============================================================

# --- Macro parameters (live, Sept 15 2026) ---
RHO_10Y = 0.04997          # 10-yr Treasury yield (4.997%)
WTI = 102.65               # $/bbl
NG_HENRY_HUB = 2.90        # $/MMBtu
COPPER_LME = 14021         # $/tonne
ALUMINUM_SHFE = 24140      # ¥/tonne (~$3,320 at 7.27 CNY/USD)
LITHIUM_CARB = 129100      # ¥/tonne (~$17,760 at 7.27)

# Interest rate per stage (annualized, scaled to stage duration)
# Paper uses rho as effective rate per stage
rho_stage = RHO_10Y / 8   # ~0.625% per stage for d=8

# --- Stage parameters for Ford Class III (d=8) ---
# Each: (m, p, n, ell, w, eta, e, tau, d_km, mu)
# m  = material conversion ratio
# p  = binomial failure probability
# n  = trials per stage
# ell= labour hours per unit
# w  = wage rate ($/hr)
# eta= energy intensity (kWh/unit)
# e  = energy price ($/kWh)
# tau= transport cost ($/unit/km)
# d  = logistic distance (km)
# mu = stage markup

stages = [
    # Stage 1: Raw material extraction (aluminum, copper, lithium)
    # Aluminum shortage -> elevated p
    (1.8, 0.15, 3,  0.8, 28.0,  8.0,  0.045, 0.003, 8000, 0.06),
    # Stage 2: Smelting & refining
    # Energy-intensive; natural gas at $2.90/MMBtu
    (1.5, 0.08, 2,  0.4, 35.0, 18.0,  0.032, 0.002, 3000, 0.08),
    # Stage 3: Component manufacturing (stamping, casting)
    (1.3, 0.06, 2,  1.2, 42.0,  6.0,  0.038, 0.0015,1200, 0.10),
    # Stage 4: Battery cell production (EV line)
    # Lithium price volatility -> elevated p
    (1.4, 0.12, 3,  1.8, 38.0, 12.0,  0.040, 0.002, 2000, 0.12),
    # Stage 5: Powertrain assembly
    (1.2, 0.05, 2,  2.5, 45.0,  4.0,  0.038, 0.001, 500,  0.15),
    # Stage 6: Body & chassis assembly
    (1.1, 0.04, 1,  3.2, 44.0,  5.0,  0.036, 0.001, 300,  0.18),
    # Stage 7: Final assembly & testing
    (1.05,0.03, 1,  4.0, 42.0,  3.0,  0.035, 0.001, 100,  0.15),
    # Stage 8: Logistics & dealer prep
    (1.0, 0.02, 1,  0.5, 25.0,  0.5,  0.045, 0.005, 1500, 0.08),
]

# --- Raw material baseline cost C0 ---
# Weighted average of Ford's material inputs per vehicle
# Aluminum ~$3,320/t (0.5t), Copper ~$14,021/t (25kg), Steel ~$900/t (1t)
C0 = (0.5 * 3320 + 0.025 * 14021 + 1.0 * 900) / 1.0  # per unit
print(f"Raw material baseline C0 = ${C0:.2f}")

# --- Stage-by-stage cost propagation (Eq. 3 from paper) ---
def stage_cost(C_prev, m, p, n, ell, w, eta, e, tau, d, mu, rho):
    """Figure 3.2 recursive stage cost."""
    amp = (1 - p) ** (-n)          # scarcity multiplier
    mat = m * C_prev * amp          # input cost + scarcity
    lab = ell * w                   # plus-sum labour
    eng = eta * e                   # energy
    log = tau * d                   # logistics
    mkp = mu * C_prev               # stage margin
    C_k = (1 + rho) * (mat + lab + eng + log + mkp)
    return C_k

# --- Run the chain ---
cost_stack = [C0]
C = C0
labour_stack = []
energy_stack = []
logistics_stack = []

for k, params in enumerate(stages):
    m, p, n, ell, w, eta, e, tau, d, mu = params
    amp = (1 - p) ** (-n)
    lab = ell * w
    eng = eta * e
    log = tau * d
    labour_stack.append(lab)
    energy_stack.append(eng)
    logistics_stack.append(log)
    C = stage_cost(C, m, p, n, ell, w, eta, e, tau, d, mu, rho_stage)
    cost_stack.append(C)

# --- Retail margin ---
mu_retail = 0.10  # Ford dealer margin ~10%
P_fin = cost_stack[-1] * (1 + mu_retail)

print(f"\n=== FORD SUPPLY-CHAIN COST STACK (Sept 15, 2026) ===")
print(f"{'Stage':<8} {'Cost ($)':>12} {'Δ ($)':>10} {'Amplifier':>10}")
print("-" * 45)
for k, c in enumerate(cost_stack):
    label = f"C{k}" if k > 0 else "C0"
    delta = c - cost_stack[k-1] if k > 0 else 0
    amp_val = (1 - stages[k-1][1]) ** (-stages[k-1][2]) if k > 0 else 1.0
    print(f"{label:<8} {c:>12,.2f} {delta:>10,.2f} {amp_val:>10.4f}")

print(f"\nFinal consumer price (before retail): ${cost_stack[-1]:,.2f}")
print(f"Final consumer price (with {mu_retail*100:.0f}% retail): ${P_fin:,.2f}")

# --- Decomposition (Eq. 22) ---
raw_pass = C0
labour_total = sum(labour_stack)
energy_total = sum(energy_stack)
logistics_total = sum(logistics_stack)

# Scarcity compounding factor
Phi_raw = np.prod([(stages[k][0] * (1 - stages[k][1])**(-stages[k][2]) + stages[k][9]) for k in range(len(stages))])
raw_amplified = Phi_raw * C0

print(f"\n=== COST DECOMPOSITION (Eq. 22) ===")
print(f"Raw material (amplified):  ${raw_amplified:>10,.2f}")
print(f"Plus-sum labour:           ${labour_total:>10,.2f}")
print(f"Energy stack:              ${energy_total:>10,.2f}")
print(f"Logistics stack:           ${logistics_total:>10,.2f}")
print(f"Scarcity compounding Φ:    {Phi_raw:>10,.2f}")

# Shares
total_inputs = raw_amplified + labour_total + energy_total + logistics_total
print(f"\nRaw share φ_raw:     {raw_amplified / total_inputs * 100:.1f}%")
print(f"Labour share Λ:      {labour_total / total_inputs * 100:.1f}%")
print(f"Energy share:        {energy_total / total_inputs * 100:.1f}%")
print(f"Logistics share:     {logistics_total / total_inputs * 100:.1f}%")