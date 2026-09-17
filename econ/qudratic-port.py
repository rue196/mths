#!/usr/bin/env python3
"""
======================
Long-term stock portfolio simulation using quadratic harmonics over the
Möbius sieve, replacing Monte Carlo path generation.

Key features:
  - Chirped phase θ_i(n) = a_i·n² + b_i·n + c_i per asset
  - Möbius-weighted shock s_i(n) = μ(n)·cos(θ_i(n)) / DENSITY
  - Deterministic, reproducible paths
  - Phase-based contrarian rebalancing
  - Comparison with traditional Monte Carlo
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1 — CONSTANTS & SIEVE
# ============================================================
PI = math.pi
E  = math.e
ALPHA = 1.0 / (PI - E)
DENSITY = 6.0 / (PI * PI)      # ≈ 0.6079271018

def mobius_sieve(K):
    """O(K) linear sieve for μ(n)."""
    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K: break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

# ============================================================
# SECTION 2 — ASSET DEFINITIONS
# ============================================================
# (a, b, c) = harmonic parameters
#   a = chirp rate (regime-change acceleration)
#   b = baseline frequency (steady-state volatility clock)
#   c = phase offset (starting valuation / cycle position)
# mu, sigma = annualized return & volatility targets

ASSETS = {
    'US Large Growth': {'a': 4.0e-5, 'b': 0.105, 'c': 0.00, 'mu': 0.100, 'sigma': 0.20, 'w': 0.25},
    'US Large Value':  {'a': 1.5e-5, 'b': 0.070, 'c': 0.90, 'mu': 0.080, 'sigma': 0.15, 'w': 0.15},
    'US Small Cap':    {'a': 3.5e-5, 'b': 0.115, 'c': 1.80, 'mu': 0.090, 'sigma': 0.23, 'w': 0.10},
    'International':   {'a': 2.0e-5, 'b': 0.080, 'c': 2.70, 'mu': 0.070, 'sigma': 0.18, 'w': 0.10},
    'US Bonds':        {'a': 3.0e-6, 'b': 0.030, 'c': 3.60, 'mu': 0.045, 'sigma': 0.06, 'w': 0.20},
    'Gold':            {'a': 5.0e-5, 'b': 0.130, 'c': 4.50, 'mu': 0.050, 'sigma': 0.16, 'w': 0.05},
    'REITs':           {'a': 2.5e-5, 'b': 0.090, 'c': 5.40, 'mu': 0.075, 'sigma': 0.19, 'w': 0.10},
    'Cash':            {'a': 0.0,    'b': 0.005, 'c': 6.30, 'mu': 0.035, 'sigma': 0.005,'w': 0.05},
}

# ============================================================
# SECTION 3 — QUADRATIC HARMONIC SHOCK GENERATOR
# ============================================================

def generate_shocks(mu_sieve, K, a, b, c):
    """
    Returns the per-step shock s(n) = μ(n)·cos(a·n² + b·n + c) / DENSITY.
    Normalized to unit variance using the square-free density.
    """
    shock = np.zeros(K)
    for n in range(1, K + 1):
        m = mu_sieve[n]
        if m != 0:
            theta = a * n * n + b * n + c
            shock[n - 1] = m * math.cos(theta) / DENSITY
    # Normalize to unit variance (theoretical std ≈ 0.907)
    std = np.std(shock)
    if std > 0:
        shock /= std
    return shock


def instantaneous_frequency(n, a, b):
    """dφ/dn = 2a·n + b — the chirp rate."""
    return 2.0 * a * n + b


# ============================================================
# SECTION 4 — QUADRATIC HARMONIC PORTFOLIO
# ============================================================

def simulate_quadratic_portfolio(assets, K=360, months_per_year=12):
    """
    Deterministic long-term portfolio simulation using quadratic harmonics.

    Returns:
        time_years: time axis in years
        prices:     (K+1, n_assets) price paths
        shocks:     (K, n_assets) normalized shocks
        portfolio:  (K+1,) rebalanced portfolio value
    """
    mu_sieve = mobius_sieve(K)
    names = list(assets.keys())
    n_assets = len(names)

    # Per-step shocks
    shocks = np.zeros((K, n_assets))
    for j, name in enumerate(names):
        p = assets[name]
        shocks[:, j] = generate_shocks(mu_sieve, K, p['a'], p['b'], p['c'])

    # Convert to log-return increments
    log_returns = np.zeros((K, n_assets))
    for j, name in enumerate(names):
        p = assets[name]
        r_drift = p['mu'] / months_per_year
        r_vol   = p['sigma'] / math.sqrt(months_per_year)
        log_returns[:, j] = r_drift + r_vol * shocks[:, j]

    # Cumulative log returns
    cum_log = np.vstack([np.zeros(n_assets), np.cumsum(log_returns, axis=0)])

    # Price paths (P0 = 1.0 for all assets)
    prices = np.exp(cum_log)

    # Portfolio value (buy-and-hold AND rebalanced)
    w = np.array([assets[name]['w'] for name in names])
    w = w / w.sum()

    # Rebalanced portfolio: monthly rebalance to fixed weights
    port_val = np.zeros(K + 1)
    port_val[0] = 1.0
    current_value = 1.0
    for t in range(K):
        # Portfolio return = w · log_returns[t]
        port_ret = float(w @ log_returns[t])
        # Approximate log → simple
        current_value *= (1.0 + port_ret)
        port_val[t + 1] = current_value

    return {
        'time_years': np.arange(K + 1) / months_per_year,
        'prices': prices,
        'shocks': shocks,
        'log_returns': log_returns,
        'portfolio': port_val,
        'names': names,
        'weights': w,
    }


# ============================================================
# SECTION 5 — MONTE CARLO COMPARISON
# ============================================================

def simulate_monte_carlo_portfolio(assets, K=360, n_paths=1000, seed=42):
    """Traditional Monte Carlo GBM for comparison."""
    rng = np.random.default_rng(seed)
    months_per_year = 12
    names = list(assets.keys())
    n_assets = len(names)

    # Parameter arrays
    mu_arr = np.array([assets[n]['mu'] for n in names])
    sigma_arr = np.array([assets[n]['sigma'] for n in names])
    w = np.array([assets[n]['w'] for n in names]); w = w / w.sum()

    # Drift and vol per month
    drift = mu_arr / months_per_year - 0.5 * (sigma_arr ** 2) / months_per_year
    vol   = sigma_arr / math.sqrt(months_per_year)

    # Generate all paths at once
    Z = rng.standard_normal((n_paths, K, n_assets))
    log_returns = drift + vol * Z
    cum_log = np.concatenate([np.zeros((n_paths, 1, n_assets)),
                              np.cumsum(log_returns, axis=1)], axis=1)
    prices = np.exp(cum_log)

    # Portfolio: rebalanced monthly
    port_paths = np.ones((n_paths, K + 1))
    for t in range(K):
        port_ret = np.einsum('j,ijk->ik', w, log_returns[:, t:t+1, :])[:, 0]
        port_paths[:, t + 1] = port_paths[:, t] * (1.0 + port_ret)

    return {
        'time_years': np.arange(K + 1) / months_per_year,
        'port_paths': port_paths,
        'mean_path': port_paths.mean(axis=0),
        'p5': np.percentile(port_paths, 5, axis=0),
        'p25': np.percentile(port_paths, 25, axis=0),
        'p50': np.percentile(port_paths, 50, axis=0),
        'p75': np.percentile(port_paths, 75, axis=0),
        'p95': np.percentile(port_paths, 95, axis=0),
    }


# ============================================================
# SECTION 6 — PHASE-BASED REBALANCING
# ============================================================

def phase_signal(n, a, b, c):
    """
    Returns a contrarian signal in [0, 2]:
      +1 when the phase is at a trough (accumulate)
      -1 when the phase is at a peak (trim)
       0 neutral
    Uses sin(θ) — the sign tells you where in the cycle you are.
    """
    theta = a * n * n + b * n + c
    return -math.sin(theta)   # negative of sin → contrarian


def simulate_phase_weighted_portfolio(assets, K=360):
    """
    Portfolio where asset weights are modulated by the phase signal.
    w_i(t) ∝ w_i · (1 + λ · phase_signal(n))
    with λ = 0.30 (moderate tilt).
    """
    mu_sieve = mobius_sieve(K)
    names = list(assets.keys())
    n_assets = len(names)
    lambda_tilt = 0.30

    # Precompute shocks and returns
    shocks = np.zeros((K, n_assets))
    for j, name in enumerate(names):
        p = assets[name]
        shocks[:, j] = generate_shocks(mu_sieve, K, p['a'], p['b'], p['c'])

    log_returns = np.zeros((K, n_assets))
    for j, name in enumerate(names):
        p = assets[name]
        log_returns[:, j] = p['mu'] / 12 + p['sigma'] / math.sqrt(12) * shocks[:, j]

    # Dynamic weights
    base_w = np.array([assets[n]['w'] for n in names]); base_w /= base_w.sum()
    port_val = np.zeros(K + 1); port_val[0] = 1.0

    for t in range(K):
        w_t = base_w.copy()
        for j, name in enumerate(names):
            p = assets[name]
            s = phase_signal(t + 1, p['a'], p['b'], p['c'])
            w_t[j] *= (1 + lambda_tilt * s)
        w_t = np.clip(w_t, 0.0, None)
        w_t /= w_t.sum()
        port_ret = float(w_t @ log_returns[t])
        port_val[t + 1] = port_val[t] * (1.0 + port_ret)

    return port_val


# ============================================================
# SECTION 7 — PORTFOLIO METRICS
# ============================================================

def compute_metrics(port_val, months_per_year=12):
    K = len(port_val) - 1
    total_ret = port_val[-1] / port_val[0] - 1
    years = K / months_per_year
    cagr = (port_val[-1] / port_val[0]) ** (1.0 / years) - 1

    # Monthly returns
    rets = np.diff(port_val) / port_val[:-1]
    vol = np.std(rets) * math.sqrt(months_per_year)
    sharpe = (cagr - 0.035) / vol if vol > 0 else 0.0

    # Drawdown
    peak = np.maximum.accumulate(port_val)
    dd = (port_val - peak) / peak
    max_dd = dd.min()

    # Sortino
    downside = rets[rets < 0]
    downside_vol = np.std(downside) * math.sqrt(months_per_year) if len(downside) > 0 else vol
    sortino = (cagr - 0.035) / downside_vol if downside_vol > 0 else 0.0

    return {
        'total_return': total_ret,
        'cagr': cagr,
        'vol': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'final_value': port_val[-1],
    }


# ============================================================
# SECTION 8 — MAIN
# ============================================================

def main():
    K = 360          # 30 years, monthly
    months = 12

    print("=" * 72)
    print("LONG-TERM PORTFOLIO — QUADRATIC HARMONIC vs MONTE CARLO")
    print(f"Horizon: {K} months ({K//months} years)")
    print("=" * 72)
    print()

    # --- Quadratic harmonic portfolio ---
    t0 = time.time()
    qh = simulate_quadratic_portfolio(ASSETS, K=K, months_per_year=months)
    t_qh = time.time() - t0
    qh_metrics = compute_metrics(qh['portfolio'], months)

    print(f"Quadratic harmonic simulation: {t_qh:.3f} s")
    print()

    # --- Phase-weighted portfolio ---
    t0 = time.time()
    phase_port = simulate_phase_weighted_portfolio(ASSETS, K=K)
    t_ph = time.time() - t0
    ph_metrics = compute_metrics(phase_port, months)
    print(f"Phase-weighted simulation:     {t_ph:.3f} s")
    print()

    # --- Monte Carlo comparison ---
    t0 = time.time()
    mc = simulate_monte_carlo_portfolio(ASSETS, K=K, n_paths=1000, seed=42)
    t_mc = time.time() - t0
    mc_metrics = compute_metrics(mc['mean_path'], months)
    print(f"Monte Carlo simulation (1000p): {t_mc:.3f} s")
    print()

    # --- Comparison table ---
    print("--- Performance Comparison (30 years) ---")
    print(f"{'Metric':<22} {'Quadratic':>12} {'Phase-Wtd':>12} {'MC Mean':>12}")
    print("-" * 60)
    for key in ['cagr', 'vol', 'sharpe', 'sortino', 'max_dd', 'total_return', 'final_value']:
        fmt = {
            'cagr': '{:.2%}', 'vol': '{:.2%}', 'sharpe': '{:.3f}',
            'sortino': '{:.3f}', 'max_dd': '{:.2%}',
            'total_return': '{:.2%}', 'final_value': '{:.4f}'
        }[key]
        print(f"{key:<22} {fmt.format(qh_metrics[key]):>12} "
              f"{fmt.format(ph_metrics[key]):>12} {fmt.format(mc_metrics[key]):>12}")
    print()

    # --- Phase analysis per asset ---
    print("--- Asset Phase Parameters ---")
    print(f"{'Asset':<20} {'a':>10} {'b':>8} {'c':>6} {'Chirp@360':>11}")
    print("-" * 60)
    for name, p in ASSETS.items():
        chirp = instantaneous_frequency(360, p['a'], p['b'])
        print(f"{name:<20} {p['a']:>10.2e} {p['b']:>8.3f} {p['c']:>6.2f} {chirp:>11.3f}")
    print()

    # ============================================================
    # VISUALIZATION
    # ============================================================
    fig = plt.figure(figsize=(16, 12))

    # Panel 1: Individual asset price paths
    ax = fig.add_subplot(3, 3, 1)
    for j, name in enumerate(qh['names']):
        ax.plot(qh['time_years'], qh['prices'][:, j], label=name, lw=1.2)
    ax.set_xlabel('Years'); ax.set_ylabel('Price (log)')
    ax.set_title('Individual Asset Paths (Quadratic)')
    ax.set_yscale('log')
    ax.grid(alpha=0.3); ax.legend(fontsize=6, loc='upper left')

    # Panel 2: Portfolio comparison
    ax = fig.add_subplot(3, 3, 2)
    ax.plot(qh['time_years'], qh['portfolio'], 'k-', lw=2.2, label='Quadratic harmonic')
    ax.plot(qh['time_years'], phase_port, 'b--', lw=1.8, label='Phase-weighted')
    ax.plot(mc['time_years'], mc['mean_path'], 'r:', lw=1.8, label='MC mean')
    ax.fill_between(mc['time_years'], mc['p5'], mc['p95'], color='red', alpha=0.10, label='MC 5-95%')
    ax.fill_between(mc['time_years'], mc['p25'], mc['p75'], color='red', alpha=0.20, label='MC 25-75%')
    ax.set_xlabel('Years'); ax.set_ylabel('Portfolio value')
    ax.set_title('Portfolio Comparison')
    ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Panel 3: Normalized comparison
    ax = fig.add_subplot(3, 3, 3)
    ax.plot(qh['time_years'], qh['portfolio'] / qh['portfolio'][0], 'k-', lw=2, label='Quadratic')
    ax.plot(mc['time_years'], mc['mean_path'] / mc['mean_path'][0], 'r--', lw=2, label='MC mean')
    ax.plot(mc['time_years'], mc['p50'] / mc['p50'][0], 'orange', ls=':', lw=1.5, label='MC median')
    ax.set_xlabel('Years'); ax.set_ylabel('Normalized value')
    ax.set_title('Normalized Overlay')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 4: Shock series for select assets
    ax = fig.add_subplot(3, 3, 4)
    for j, name in enumerate(['US Large Growth', 'US Bonds', 'Gold']):
        idx = qh['names'].index(name)
        ax.plot(qh['time_years'][1:], qh['shocks'][:, idx], lw=0.7, label=name, alpha=0.8)
    ax.set_xlabel('Years'); ax.set_ylabel('Shock s(n)')
    ax.set_title('Chirped Möbius Shock Series')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Panel 5: Instantaneous frequency (chirp)
    ax = fig.add_subplot(3, 3, 5)
    n_grid = np.arange(1, K + 1)
    for name, p in ASSETS.items():
        if p['a'] > 0 or p['b'] > 0.01:
            freq = [instantaneous_frequency(n, p['a'], p['b']) for n in n_grid]
            ax.plot(n_grid / 12, freq, lw=1.2, label=name)
    ax.set_xlabel('Years'); ax.set_ylabel('dφ/dn')
    ax.set_title('Chirp — Instantaneous Frequency')
    ax.grid(alpha=0.3); ax.legend(fontsize=6)

    # Panel 6: Rolling 12-month volatility
    ax = fig.add_subplot(3, 3, 6)
    rets_qh = np.diff(qh['portfolio']) / qh['portfolio'][:-1]
    rets_mc = np.diff(mc['mean_path']) / mc['mean_path'][:-1]
    win = 12
    roll_qh = np.array([np.std(rets_qh[max(0,i-win):i+1]) for i in range(len(rets_qh))]) * math.sqrt(12)
    roll_mc = np.array([np.std(rets_mc[max(0,i-win):i+1]) for i in range(len(rets_mc))]) * math.sqrt(12)
    ax.plot(qh['time_years'][1:], roll_qh, 'k-', lw=1.5, label='Quadratic')
    ax.plot(mc['time_years'][1:], roll_mc, 'r--', lw=1.5, label='MC')
    ax.set_xlabel('Years'); ax.set_ylabel('Rolling 12M volatility')
    ax.set_title('Volatility Regimes')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 7: Drawdowns
    ax = fig.add_subplot(3, 3, 7)
    dd_qh = (qh['portfolio'] - np.maximum.accumulate(qh['portfolio'])) / np.maximum.accumulate(qh['portfolio'])
    dd_mc = (mc['mean_path'] - np.maximum.accumulate(mc['mean_path'])) / np.maximum.accumulate(mc['mean_path'])
    ax.fill_between(qh['time_years'], dd_qh*100, 0, color='black', alpha=0.5, label='Quadratic')
    ax.plot(mc['time_years'], dd_mc*100, 'r--', lw=1.5, label='MC mean')
    ax.set_xlabel('Years'); ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Profile')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # Panel 8: Phase signals
    ax = fig.add_subplot(3, 3, 8)
    n_grid = np.arange(1, K + 1)
    for name in ['US Large Growth', 'US Bonds', 'Gold']:
        p = ASSETS[name]
        sig = [phase_signal(n, p['a'], p['b'], p['c']) for n in n_grid]
        ax.plot(n_grid / 12, sig, lw=1.2, label=name)
    ax.axhline(0, color='black', ls=':', alpha=0.5)
    ax.axhline(+1, color='green', ls=':', alpha=0.3)
    ax.axhline(-1, color='red', ls=':', alpha=0.3)
    ax.set_xlabel('Years'); ax.set_ylabel('Phase signal')
    ax.set_title('Contrarian Rebalancing Signal')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Panel 9: Return distribution
    ax = fig.add_subplot(3, 3, 9)
    ax.hist(rets_qh * 100, bins=40, alpha=0.6, color='black', label='Quadratic', density=True)
    ax.hist(rets_mc * 100, bins=40, alpha=0.5, color='red', label='MC', density=True)
    ax.axvline(0, color='black', ls=':', alpha=0.5)
    ax.set_xlabel('Monthly return (%)'); ax.set_ylabel('Density')
    ax.set_title('Return Distribution')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ============================================================
    # FINAL DIAGNOSTICS
    # ============================================================
    print("=" * 72)
    print("FINAL DIAGNOSTICS — 30-YEAR HORIZON")
    print("=" * 72)
    print(f"Quadratic portfolio final value:   ${qh['portfolio'][-1]:.4f}")
    print(f"Phase-weighted final value:        ${phase_port[-1]:.4f}")
    print(f"MC mean final value:               ${mc['mean_path'][-1]:.4f}")
    print(f"MC median final value:             ${mc['p50'][-1]:.4f}")
    print(f"MC 5th percentile:                 ${mc['p5'][-1]:.4f}")
    print(f"MC 95th percentile:                ${mc['p95'][-1]:.4f}")
    print()
    print(f"Quadratic CAGR:                    {qh_metrics['cagr']*100:.2f}%")
    print(f"MC CAGR:                           {mc_metrics['cagr']*100:.2f}%")
    print(f"Quadratic Sharpe:                  {qh_metrics['sharpe']:.3f}")
    print(f"MC Sharpe:                         {mc_metrics['sharpe']:.3f}")
    print(f"Quadratic Max DD:                  {qh_metrics['max_dd']*100:.2f}%")
    print(f"MC Max DD:                         {mc_metrics['max_dd']*100:.2f}%")
    print()
    print(f"Square-free density 6/π²:          {DENSITY:.8f}")
    print(f"σ(μ)/√DENSITY (theoretical):       {math.sqrt(0.5/DENSITY):.6f}")
    print(f"Realized σ of shocks (asset 0):    {np.std(qh['shocks'][:,0]):.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()