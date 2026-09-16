#!/usr/bin/env python3
"""
derivatives_harmonic_pricing.py
================================
Derivatives market pricing using the harmonic equilibrium model.

Mapping:
  - Algebraic parts  = agreed discrete contract terms (strike, notional, expiry)
  - Transcendental   = futures price adjustments driven by dzeta/dt
  - Inverse scalar   = 1/det(M) = supply/demand imbalance, gives the derivative

Date: 2026-09-16 (Fed decision day)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1 — HARMONIC EQUILIBRIUM ENGINE (from file)
# ============================================================
PI = math.pi
E  = math.e
ALPHA = 1.0 / (PI - E)

def mobius_sieve(K):
    mu = [0]*(K+1); mu[1] = 1
    primes = []; is_comp = [False]*(K+1)
    for i in range(2, K+1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i*p > K: break
            is_comp[i*p] = True
            if i % p == 0: mu[i*p] = 0; break
            else: mu[i*p] = -mu[i]
    return mu

def harmonic_numbers(N):
    H = np.zeros(N+1); H[1] = 1.0
    for n in range(2, N+1):
        H[n] = H[n-1] + 1.0/n
    return H

def build_coeffs(K):
    mu = mobius_sieve(K)
    c = np.zeros(2*K+1, dtype=complex)
    for n in range(1, K+1):
        c[K+n] = mu[n]; c[K-n] = mu[n]
    c[K] = 0.0
    return c

def zeta_at(t, c, alpha):
    K = (len(c)-1)//2
    total = 0j
    for i in range(-K, K+1):
        total += c[i+K] * np.exp(1j * t * i / alpha)
    return total

def dzeta_dt_analytic(t, c, alpha):
    """Analytical derivative of ζ(t)"""
    K = (len(c)-1)//2
    d = 0.0
    for i in range(1, K+1):
        d += -2.0 * c[K+i].real * (i/alpha) * np.sin(t*i/alpha)
    return d

def scarcity_multiplier(p, n):
    return (1 - p) ** (-n)

def run_equilibrium(K_modes=25, N_harmonics=100):
    c = build_coeffs(K_modes)
    H = harmonic_numbers(N_harmonics)
    time = H[1:]

    zeta_vals = np.zeros(N_harmonics)
    dzeta_vals = np.zeros(N_harmonics)
    for n in range(1, N_harmonics+1):
        zeta_vals[n-1] = zeta_at(H[n], c, ALPHA).real
        dzeta_vals[n-1] = dzeta_dt_analytic(H[n], c, ALPHA)

    M = np.zeros((N_harmonics, 2, 2))
    M[0] = [[1.0, 0.8], [1.0, 1.2]]

    p0_x, p_amp_x, freq_x, n_x = 0.2, 0.15, 0.5, 2
    p0_y, p_amp_y, freq_y, n_y = 0.3, 0.10, 0.3, 3

    for idx in range(1, N_harmonics):
        t, t_prev = time[idx], time[idx-1]
        dt = t - t_prev
        d_zeta = zeta_vals[idx]

        p_x = np.clip(p0_x + p_amp_x*np.sin(2*np.pi*freq_x*t), 0, 0.99)
        p_y = np.clip(p0_y + p_amp_y*np.sin(2*np.pi*freq_y*t), 0, 0.99)
        Sx = scarcity_multiplier(p_x, n_x)
        Sy = scarcity_multiplier(p_y, n_y)

        x_inv = max(M[idx-1,0,0]*(1 + d_zeta*dt) + (Sx - M[idx-1,0,0])*0.01, 0.1)
        y_inv = max(M[idx-1,0,1]*(1 + d_zeta*dt) + (Sy - M[idx-1,0,1])*0.01, 0.1)
        x_i = np.clip(M[idx-1,1,0]*(1 + 0.02*d_zeta*dt) + 0.01*np.sin(2*np.pi*0.1*t), 0.1, 5.0)
        y_i = np.clip(M[idx-1,1,1]*(1 + 0.02*d_zeta*dt) + 0.01*np.cos(2*np.pi*0.08*t), 0.1, 5.0)

        M[idx] = [[x_inv, y_inv], [x_i, y_i]]

    det = M[:,0,0]*M[:,1,1] - M[:,0,1]*M[:,1,0]
    trace = M[:,0,0] + M[:,1,1]
    price_x = M[:,0,0] * M[:,1,0]
    price_y = M[:,0,1] * M[:,1,1]

    return time, zeta_vals, dzeta_vals, M, det, trace, price_x, price_y

# ============================================================
# SECTION 2 — ALGEBRAIC CONTRACT DEFINITIONS (discrete terms)
# ============================================================

# Live market data: 2026-09-16
DERIVATIVES = {
    'WTI_Front': {
        'name': 'WTI Crude (Oct 2026)',
        'algebraic_base': 105.02,         # discrete: agreed spot-futures basis
        'notional': 1000,                  # 1000 bbl per contract
        'tick_size': 0.01,
        'expiry_days': 30,
        'sector': 'energy',
    },
    'WTI_Back': {
        'name': 'WTI Crude (Dec 2027)',
        'algebraic_base': 94.50,
        'notional': 1000,
        'tick_size': 0.01,
        'expiry_days': 450,
        'sector': 'energy',
    },
    'Brent_Front': {
        'name': 'Brent Crude (Nov 2026)',
        'algebraic_base': 108.89,
        'notional': 1000,
        'tick_size': 0.01,
        'expiry_days': 30,
        'sector': 'energy',
    },
    'Diesel_ULSD': {
        'name': 'NY Harbor ULSD (Oct 2026)',
        'algebraic_base': 2.4120,          # per gallon
        'notional': 42000,                  # 42,000 gal per contract
        'tick_size': 0.0001,
        'expiry_days': 30,
        'sector': 'energy',
    },
    'ZT_2Y': {
        'name': '2Y Treasury Note Futures',
        'algebraic_base': 104.1875,        # discrete: clean price at par yield
        'notional': 200000,
        'tick_size': 1/32,
        'expiry_days': 90,
        'sector': 'rates',
    },
    'ZN_10Y': {
        'name': '10Y Treasury Note Futures',
        'algebraic_base': 110.8125,
        'notional': 100000,
        'tick_size': 1/64,
        'expiry_days': 90,
        'sector': 'rates',
    },
    'ZB_30Y': {
        'name': '30Y Treasury Bond Futures',
        'algebraic_base': 118.5000,
        'notional': 100000,
        'tick_size': 1/32,
        'expiry_days': 90,
        'sector': 'rates',
    },
    'SR3_SOFR': {
        'name': '3M SOFR Futures (Dec 2026)',
        'algebraic_base': 96.28,           # 100 - 3.72% implied rate
        'notional': 2500,
        'tick_size': 0.005,
        'expiry_days': 90,
        'sector': 'rates',
    },
    'VIX_Front': {
        'name': 'VIX Futures (Oct 2026)',
        'algebraic_base': 18.50,
        'notional': 1000,
        'tick_size': 0.05,
        'expiry_days': 30,
        'sector': 'volatility',
    },
    'ES_SPX': {
        'name': 'E-mini S&P 500 (Dec 2026)',
        'algebraic_base': 7620.00,
        'notional': 50,
        'tick_size': 0.25,
        'expiry_days': 90,
        'sector': 'equity',
    },
}

# Sector-specific beta/gamma couplings to dzeta/dt and inverse det
SECTOR_COUPLING = {
    'energy':     {'beta': -2.4, 'gamma': 0.85},   # energy: strong dzeta coupling
    'rates':      {'beta': -3.1, 'gamma': 1.20},   # rates: very strong dzeta coupling
    'volatility': {'beta': +5.8, 'gamma': -0.90},  # VIX: inverse (crisis beta)
    'equity':     {'beta': -1.8, 'gamma': 0.40},   # equities: moderate
}

# ============================================================
# SECTION 3 — TRANSCENDENTAL ADJUSTMENT FUNCTION
# ============================================================

def transcendental_adjustment(t, dzeta_vals, det_vals, zeta_vals, sector):
    """
    The transcendental part of the futures price:
      adj(t) = beta * dzeta/dt + gamma * (1/det(M))
    Uses the current harmonic index.
    """
    beta  = SECTOR_COUPLING[sector]['beta']
    gamma = SECTOR_COUPLING[sector]['gamma']

    # dzeta/dt at this time
    dz = dzeta_vals[t] if abs(dzeta_vals[t]) > 1e-6 else 0.0
    # Inverse of det(M), bounded away from zero
    det_val = det_vals[t]
    inv_det = 1.0 / det_val if abs(det_val) > 0.05 else np.sign(det_val) * 20.0

    # Transcendental adjustment (both terms scale like the harmonic kernel)
    adj = beta * dz + gamma * inv_det * 0.01
    return adj

def futures_price(contract, t, dzeta_vals, det_vals, zeta_vals):
    """
    F(t) = algebraic_base * (1 + transcendental_adjustment)
    """
    A = contract['algebraic_base']
    sector = contract['sector']
    adj = transcendental_adjustment(t, dzeta_vals, det_vals, zeta_vals, sector)
    F = A * (1 + adj)
    return F, adj

# ============================================================
# SECTION 4 — RUN VALUATION
# ============================================================

time, zeta_vals, dzeta_vals, M, det, trace, price_x, price_y = run_equilibrium(
    K_modes=25, N_harmonics=100)

# Current harmonic index = last point
t_now = len(time) - 1

print("=" * 78)
print("DERIVATIVES MARKET — HARMONIC EQUILIBRIUM PRICING")
print("Date: 2026-09-16 | Harmonic step:", f"H_{t_now+1} = {time[t_now]:.4f}")
print("=" * 78)
print()

# Current engine state
print("--- Harmonic Engine State (t = today) ---")
print(f"  ζ(t):        {zeta_vals[t_now]:+.4f}")
print(f"  dζ/dt:       {dzeta_vals[t_now]:+.4f}")
print(f"  x⁻¹:         {M[t_now,0,0]:.4f}    y⁻¹: {M[t_now,0,1]:.4f}")
print(f"  xⁱ:          {M[t_now,1,0]:.4f}    yⁱ:  {M[t_now,1,1]:.4f}")
print(f"  det(M):      {det[t_now]:+.4f}")
print(f"  1/det(M):    {1/det[t_now]:+.4f}")
print(f"  tr(M):       {trace[t_now]:+.4f}")
print(f"  price_x:     {price_x[t_now]:.4f}")
print(f"  price_y:     {price_y[t_now]:.4f}")
print()

# Price all derivatives
print("--- Derivative Contract Pricing ---")
print(f"{'Contract':<28} {'Algebraic':>11} {'Adj %':>9} {'Futures':>12} {'Tick':>8}")
print("-" * 72)

results = {}
for key, contract in DERIVATIVES.items():
    F, adj = futures_price(contract, t_now, dzeta_vals, det, zeta_vals)
    results[key] = {'F': F, 'adj': adj, 'contract': contract}
    print(f"{contract['name']:<28} {contract['algebraic_base']:>11.4f} "
          f"{adj*100:>+8.3f}% {F:>12.4f} {contract['tick_size']:>8.4f}")

print()

# ============================================================
# SECTION 5 — REPRICING UNDER DISRUPTION (Eq. 6.1 analog)
# ============================================================

# Repricing shock: shift dzeta/dt by ±Δ at the current step
shock_scenarios = {
    'Base (no shock)':      0.0,
    'Hawkish Fed (+25bp)':  -0.30,   # rate hike pushes dzeta more negative
    'Geopolitical spike':   -0.50,   # risk-off strengthens the negative trend
    'Dovish hold':          +0.40,   # rate hold reverses the derivative
    'Risk-on rally':        +0.80,
}

print("--- Repricing Under Disruption Scenarios ---")
print(f"{'Scenario':<24} {'WTI_F':>10} {'ZN_10Y':>10} {'ES_SPX':>10} {'VIX_F':>10}")
print("-" * 68)

for scen, shift in shock_scenarios.items():
    # Apply the shift to dzeta at t_now
    dzeta_shifted = dzeta_vals.copy()
    dzeta_shifted[t_now] += shift

    row = [scen]
    for key in ['WTI_Front', 'ZN_10Y', 'ES_SPX', 'VIX_Front']:
        F, _ = futures_price(DERIVATIVES[key], t_now, dzeta_shifted, det, zeta_vals)
        row.append(F)
    print(f"{row[0]:<24} {row[1]:>10.2f} {row[2]:>10.4f} {row[3]:>10.2f} {row[4]:>10.2f}")

print()

# ============================================================
# SECTION 6 — INVERSE SCALAR ANALYSIS
# ============================================================

print("--- Inverse Scalar 1/det(M) as Derivative Driver ---")
print(f"{'Time H_n':>10} {'det(M)':>10} {'1/det':>10} {'dζ/dt':>10} {'Interpretation':<28}")
print("-" * 72)

# Sample at multiple time points
sample_idx = [5, 20, 40, 60, 80, 99]
for idx in sample_idx:
    dm = det[idx]
    inv_dm = 1/dm if abs(dm) > 0.05 else np.sign(dm)*20
    dz = dzeta_vals[idx]
    if dm > 0.3 and dz < 0:
        interp = "Bearish stress"
    elif dm < -0.3 and dz > 0:
        interp = "Bullish reversion"
    elif abs(dm) < 0.1:
        interp = "Volatility cascade zone"
    else:
        interp = "Mild regime"
    print(f"{time[idx]:>10.4f} {dm:>+10.4f} {inv_dm:>+10.4f} {dz:>+10.4f} {interp:<28}")

print()

# ============================================================
# SECTION 7 — TICK-LEVEL DECOMPOSITION
# ============================================================

print("--- Tick-Level Algebraic vs Transcendental Split (WTI Front) ---")
print(f"{'Component':<32} {'Value':>14}")
print("-" * 48)

wti = DERIVATIVES['WTI_Front']
F_wti, adj_wti = futures_price(wti, t_now, dzeta_vals, det, zeta_vals)
A_wti = wti['algebraic_base']
T_wti = F_wti - A_wti

print(f"{'Algebraic base (agreed)':<32} ${A_wti:>13.4f}")
print(f"{'Transcendental adjustment':<32} ${T_wti:>13.4f}")
print(f"{'  β × dζ/dt':<32} ${SECTOR_COUPLING['energy']['beta']*dzeta_vals[t_now]*A_wti:>13.4f}")
print(f"{'  γ × 1/det × 0.01':<32} ${SECTOR_COUPLING['energy']['gamma']/det[t_now]*0.01*A_wti:>13.4f}")
print(f"{'Futures price F(t)':<32} ${F_wti:>13.4f}")
print(f"{'Ticks moved from base':<32} {(F_wti - A_wti)/wti['tick_size']:>13.0f}")
print()

# ============================================================
# SECTION 8 — VISUALIZATION
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Panel 1: ζ(t) and dζ/dt — the transcendental driver
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(time, zeta_vals, color='purple', label='ζ(t)')
ax2.plot(time, dzeta_vals, color='orange', ls='--', label='dζ/dt')
ax.set_xlabel('Harmonic time H_n')
ax.set_ylabel('ζ(t)', color='purple')
ax2.set_ylabel('dζ/dt', color='orange')
ax.set_title('Spectral Drivers')
ax.grid(alpha=0.3)
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

# Panel 2: det(M) and inverse scalar
ax = axes[0, 1]
inv_det = np.where(np.abs(det) > 0.05, 1/det, np.sign(det)*20)
ax.plot(time, det, color='black', label='det(M)')
ax.axhline(y=0, color='red', ls=':', alpha=0.5)
ax2 = ax.twinx()
ax2.plot(time, inv_det, color='blue', ls='--', label='1/det(M)')
ax.set_xlabel('Harmonic time H_n')
ax.set_ylabel('det(M)', color='black')
ax2.set_ylabel('1/det(M)', color='blue')
ax.set_title('Inverse Scalar (Supply/Demand Imbalance)')
ax.grid(alpha=0.3)
ax.legend(loc='upper left'); ax2.legend(loc='upper right')

# Panel 3: Scarcity and demand factors
ax = axes[1, 0]
ax.plot(time, M[:,0,0], color='red', label='x⁻¹ (algebraic scarcity)')
ax.plot(time, M[:,0,1], color='blue', label='y⁻¹')
ax.plot(time, M[:,1,0], color='orange', label='xⁱ (transcendental demand)')
ax.plot(time, M[:,1,1], color='green', label='yⁱ')
ax.set_xlabel('Harmonic time H_n')
ax.set_title('Algebraic Scarcity vs Transcendental Demand')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 4: Futures prices for key contracts
ax = axes[1, 1]
wti_series = np.array([futures_price(DERIVATIVES['WTI_Front'], i, dzeta_vals, det, zeta_vals)[0] for i in range(len(time))])
zn_series  = np.array([futures_price(DERIVATIVES['ZN_10Y'],   i, dzeta_vals, det, zeta_vals)[0] for i in range(len(time))])
vix_series = np.array([futures_price(DERIVATIVES['VIX_Front'], i, dzeta_vals, det, zeta_vals)[0] for i in range(len(time))])
es_series  = np.array([futures_price(DERIVATIVES['ES_SPX'],   i, dzeta_vals, det, zeta_vals)[0] for i in range(len(time))])

ax.plot(time, wti_series / wti_series[0] * 100, color='black', label='WTI (norm)')
ax.plot(time, zn_series  / zn_series[0]  * 100, color='navy',  label='ZN (norm)')
ax.plot(time, vix_series / vix_series[0] * 100, color='red',   label='VIX (norm)')
ax.plot(time, es_series  / es_series[0]  * 100, color='green', label='ES (norm)')
ax.axhline(y=100, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Harmonic time H_n')
ax.set_ylabel('Indexed price (t=0 → 100)')
ax.set_title('Normalized Futures Prices')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 5: Repricing scenarios bar chart
ax = axes[2, 0]
scen_names = list(shock_scenarios.keys())
wti_vals = []
for scen, shift in shock_scenarios.items():
    dz_shift = dzeta_vals.copy(); dz_shift[t_now] += shift
    F, _ = futures_price(DERIVATIVES['WTI_Front'], t_now, dz_shift, det, zeta_vals)
    wti_vals.append(F)
ax.bar(range(len(scen_names)), wti_vals, color=['gray','darkred','red','green','darkgreen'])
ax.set_xticks(range(len(scen_names)))
ax.set_xticklabels([s.split('(')[0].strip() for s in scen_names], rotation=15, fontsize=8)
ax.set_ylabel('WTI Futures ($)')
ax.set_title('WTI Repricing Under Disruption Scenarios')
ax.axhline(y=wti_vals[0], color='black', ls=':', alpha=0.5)
ax.grid(alpha=0.3, axis='y')

# Panel 6: Transcendental contribution heatmap
ax = axes[2, 1]
contracts_list = list(DERIVATIVES.keys())
adj_matrix = np.zeros((len(contracts_list), len(sample_idx)))
for i, key in enumerate(contracts_list):
    for j, idx in enumerate(sample_idx):
        _, adj = futures_price(DERIVATIVES[key], idx, dzeta_vals, det, zeta_vals)
        adj_matrix[i, j] = adj * 100
im = ax.imshow(adj_matrix, aspect='auto', cmap='RdBu_r', vmin=-15, vmax=15)
ax.set_yticks(range(len(contracts_list)))
ax.set_yticklabels([DERIVATIVES[k]['name'][:22] for k in contracts_list], fontsize=7)
ax.set_xticks(range(len(sample_idx)))
ax.set_xticklabels([f"H_{i+1}" for i in sample_idx], fontsize=8)
ax.set_title('Transcendental Adjustment (% of algebraic base)')
plt.colorbar(im, ax=ax, label='Adjustment (%)')

plt.tight_layout()
plt.show()