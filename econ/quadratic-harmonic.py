#!/usr/bin/env python3
"""
quadratic_harmonics.py

Oscillatory functions as quadratic harmonics over the Möbius sieve.

A quadratic harmonic is a function of the form

    φ_k(n) = exp( i · (a_k n² + b_k n + c_k) )

so the phase is a quadratic polynomial in n.  Combined with the Möbius
weight μ(n), the partial sums

    H_a(K) = Σ_{n≤K} μ(n) · sin(a n²)
    H_b(K) = Σ_{n≤K} μ(n) · cos(b n²)

behave as slowly oscillating envelope functions whose amplitudes scale
like the square‑free density 6/π².  Unlike linear harmonics (which
oscillate with a fixed period), quadratic harmonics chirp — their
instantaneous frequency grows linearly with n.

All sums are computed in O(K) time after an O(K) Möbius sieve.

Reference identities used:
    Σ μ(n)/n²  → 6/π²
    Σ μ(n)/n   → 0
    Σ μ(n)·e^{iθn}  → oscillatory envelope
"""

import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
DENSITY = 6.0 / (PI * PI)          # ≈ 0.6079271018


# ============================================================
#  1.  Möbius sieve (linear, O(K))
# ============================================================
def mobius_sieve(K):
    """Linear sieve for μ(n), n = 1..K.  O(K) time, O(K) memory."""
    if K < 1:
        return [0] * (K + 1)
    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


# ============================================================
#  2.  Quadratic harmonics
# ============================================================
def quadratic_sin(n, a=1.0, b=0.0, c=0.0):
    """sin(a n² + b n + c) — a quadratic (chirped) harmonic."""
    return math.sin(a * n * n + b * n + c)


def quadratic_cos(n, a=1.0, b=0.0, c=0.0):
    """cos(a n² + b n + c)."""
    return math.cos(a * n * n + b * n + c)


def quadratic_phase(n, a=1.0, b=0.0, c=0.0):
    """Complex quadratic harmonic exp(i(a n² + b n + c))."""
    theta = a * n * n + b * n + c
    return complex(math.cos(theta), math.sin(theta))


# ============================================================
#  3.  Möbius‑weighted quadratic harmonic sums (O(K))
# ============================================================
def mobius_quadratic_sum(K, mu, a=1.0, b=0.0, c=0.0,
                        kind='sin', record_every=1000):
    """
    Compute the partial sums

        H(K) = Σ_{n≤K} μ(n) · φ(n)

    where φ is a quadratic harmonic ('sin', 'cos', or 'exp').

    Returns (n_vals, H_real, H_imag, aux) where aux holds
    the square‑free density partial sums Q(K)/K.
    """
    n_vals, H_real, H_imag, density = [], [], [], []
    acc_re, acc_im = 0.0, 0.0
    Q = 0
    for n in range(1, K + 1):
        m = mu[n]
        if m != 0:
            Q += 1
        if m != 0:
            if kind == 'sin':
                val = quadratic_sin(n, a, b, c)
                acc_re += m * val
                acc_im += 0.0
            elif kind == 'cos':
                val = quadratic_cos(n, a, b, c)
                acc_re += m * val
                acc_im += 0.0
            elif kind == 'exp':
                z = quadratic_phase(n, a, b, c)
                acc_re += m * z.real
                acc_im += m * z.imag
        if n % record_every == 0:
            n_vals.append(n)
            H_real.append(acc_re)
            H_imag.append(acc_im)
            density.append(Q / n)
    return np.array(n_vals), np.array(H_real), np.array(H_imag), np.array(density)


# ============================================================
#  4.  Comparison with linear harmonics (for reference)
# ============================================================
def mobius_linear_sum(K, mu, omega=1.0, record_every=1000):
    """Σ μ(n) · e^{i ω n}  — the classical linear harmonic."""
    n_vals, H_real, H_imag = [], [], []
    acc_re, acc_im = 0.0, 0.0
    for n in range(1, K + 1):
        m = mu[n]
        if m != 0:
            acc_re += m * math.cos(omega * n)
            acc_im += m * math.sin(omega * n)
        if n % record_every == 0:
            n_vals.append(n)
            H_real.append(acc_re)
            H_imag.append(acc_im)
    return np.array(n_vals), np.array(H_real), np.array(H_imag)


# ============================================================
#  5.  Instantaneous frequency of a quadratic harmonic
# ============================================================
def instantaneous_frequency(n, a=1.0, b=0.0):
    """
    For φ(n) = a n² + b n + c, the derivative dφ/dn = 2a n + b.
    This is the instantaneous (angular) frequency.
    """
    return 2.0 * a * n + b


# ============================================================
#  6.  Demonstration
# ============================================================
def main():
    K = 200_000
    print(f"=== Quadratic Harmonics over the Möbius Sieve (K = {K:,}) ===\n")

    # --- sieve ---
    t0 = time.time()
    mu = mobius_sieve(K)
    print(f"Möbius sieve: {time.time() - t0:.3f} s\n")

    # --- quadratic harmonic sums ---
    params = [
        dict(a=1e-4, b=0.0, c=0.0, kind='sin', label='sin(1e-4·n²)'),
        dict(a=1e-4, b=0.0, c=0.0, kind='cos', label='cos(1e-4·n²)'),
        dict(a=1e-5, b=1e-2, c=0.0, kind='exp',
             label='exp(i(1e-5·n² + 1e-2·n))'),
    ]

    results = {}
    for p in params:
        t0 = time.time()
        n_vals, H_re, H_im, dens = mobius_quadratic_sum(
            K, mu, a=p['a'], b=p['b'], c=p['c'],
            kind=p['kind'], record_every=2000
        )
        dt = time.time() - t0
        label = p['label']
        results[label] = (n_vals, H_re, H_im)
        print(f"{label:38s}  time={dt:.3f} s  "
              f"|H(K)|={abs(H_re[-1] + 1j*H_im[-1]):.4f}  "
              f"Q/K={dens[-1]:.6f}")

    # --- linear harmonic for comparison ---
    n_lin, H_re_lin, H_im_lin = mobius_linear_sum(K, mu, omega=0.01,
                                                  record_every=2000)
    print(f"{'linear exp(i·0.01·n)':38s}  "
          f"|H(K)|={abs(H_re_lin[-1] + 1j*H_im_lin[-1]):.4f}")

    # --- instantaneous frequency ---
    n_grid = np.linspace(0, K, 500)
    freq = instantaneous_frequency(n_grid, a=1e-4, b=0.0)

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Real part of quadratic harmonics
    ax = axes[0, 0]
    for label, (n_vals, H_re, H_im) in results.items():
        ax.plot(n_vals, H_re, lw=1.2, label=f"Re {label}")
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('K')
    ax.set_ylabel('Re H(K)')
    ax.set_title('Real part of quadratic harmonic sums')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Magnitude |H(K)| of quadratic harmonics
    ax = axes[0, 1]
    for label, (n_vals, H_re, H_im) in results.items():
        ax.plot(n_vals, np.hypot(H_re, H_im), lw=1.2, label=label)
    ax.plot(n_lin, np.hypot(H_re_lin, H_im_lin),
            'k--', lw=1.0, label='linear exp(i·0.01·n)')
    # Expected envelope: square‑free density · sqrt(K)  (random‑walk scaling)
    ax.plot(n_lin, np.sqrt(n_lin) * DENSITY,
            'r:', lw=1.0, label='6/π² · √K  (random‑walk envelope)')
    ax.set_xlabel('K')
    ax.set_ylabel('|H(K)|')
    ax.set_title('Magnitude envelope of quadratic vs linear harmonics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Instantaneous frequency of the quadratic phase
    ax = axes[1, 0]
    for a_val, b_val, lbl in [(1e-4, 0.0, 'a=1e-4, b=0'),
                              (1e-5, 1e-2, 'a=1e-5, b=1e-2'),
                              (1e-6, 0.0, 'a=1e-6, b=0')]:
        ax.plot(n_grid, instantaneous_frequency(n_grid, a_val, b_val),
                lw=1.5, label=lbl)
    ax.set_xlabel('n')
    ax.set_ylabel('dφ/dn = 2a·n + b')
    ax.set_title('Instantaneous frequency of quadratic harmonics (chirp)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Square‑free density convergence
    _, _, _, dens = mobius_quadratic_sum(
        K, mu, a=1e-4, kind='sin', record_every=2000
    )
    ax = axes[1, 1]
    ax.plot(n_vals, dens, 'b-', lw=1.2, label='Q(K)/K')
    ax.axhline(DENSITY, color='r', ls='--', label='6/π² = 0.607927…')
    ax.set_xlabel('K')
    ax.set_ylabel('Q(K)/K')
    ax.set_title('Square‑free density drives the harmonic envelope')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---------- Numerical summary ----------
    print("\n--- Summary ---")
    print(f"Square‑free density 6/π²           : {DENSITY:.8f}")
    print(f"Σ μ(n)/n²  (S2)                    : "
          f"{sum(mu[n]/(n*n) for n in range(1, K+1)):.8f}")
    print(f"Σ μ(n)/n   (S1)                    : "
          f"{sum(mu[n]/n for n in range(1, K+1)):.6e}")
    for label, (n_vals, H_re, H_im) in results.items():
        print(f"|H(K)| for {label:38s}: "
              f"{abs(H_re[-1] + 1j*H_im[-1]):.4f}")
    print(f"|H(K)| for linear exp(i·0.01·n)       : "
          f"{abs(H_re_lin[-1] + 1j*H_im_lin[-1]):.4f}")
    print(f"\nRandom‑walk envelope 6/π²·√K       : "
          f"{DENSITY * math.sqrt(K):.4f}")


if __name__ == "__main__":
    main()