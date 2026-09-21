#!/usr/bin/env python3
"""
plot_spectral_sum_eta.py

Spectral sum  ζ(t) = Σ C_i exp(i t i / α)  with α = 1/(π − e^0.3628).

Additions
---------
• 1D TSP route (chip-g.py bucket sort by pseudo-angle) — deterministic.
• Dirichlet eta function  η(a) = Σ_{n≥1} (−1)^{n−1} / n^a.
• Asymmetric polynomial solver on the TSP path, using the eta
  weight η(a) ≈ 0.659538886352 as the target norm.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
EXP = math.exp(0.3628)          # e^0.3628 ≈ 1.4371
A = 1.0 / (PI - EXP)            # ≈ 0.5866
ALPHA = 1.0 / (PI - E)
ETA_TARGET = 0.659538886352     # η(0.3628)
ALPHA_ASYM = 0.3628


# ============================================================
#  1. Möbius sieve  ·  O(K)
# ============================================================
def mobius_sieve(K):
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
#  2. Spectral sum  ζ(t)   (unchanged)
# ============================================================
def zeta(t, coeffs, alpha=ALPHA):
    total = 0.0 + 0.0j
    K = len(coeffs) - 1
    for i in range(1, K + 1):
        if coeffs[i] != 0:
            total += coeffs[i] * np.exp(1j * t * i / alpha)
    return total


def integrate_abs_zeta(coeffs, N_points=1000, alpha=ALPHA):
    period = 2 * PI * alpha
    t_vals = np.linspace(0, period, N_points)
    sum_abs = 0.0
    for t in t_vals:
        sum_abs += abs(zeta(t, coeffs, alpha))
    dt = t_vals[1] - t_vals[0]
    return sum_abs * dt


# ============================================================
#  3. chip-g.py 1D TSP route  ·  O(K)
# ============================================================
def tsp_route_1d(K, offset=0.0):
    """
    Deterministic TSP routing order for K nodes.
    Bucket sort by pseudo-angle (chip-g.py `_tsp_route`).
    """
    angles = np.zeros(K, dtype=float)
    for i in range(K):
        x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
        y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
        angles[i] = (math.atan2(y, x) + math.pi + offset) % (2 * math.pi)
    buckets = [[] for _ in range(360)]
    for i, a in enumerate(angles):
        idx = int((a / (2 * math.pi)) * 360) % 360
        buckets[idx].append(i)
    order = []
    for b in buckets:
        order.extend(b)
    return np.array(order, dtype=int)


# ============================================================
#  4. Dirichlet eta function  η(a)
# ============================================================
def dirichlet_eta(a=ALPHA_ASYM, N=2000):
    """
    η(a) = Σ_{n≥1} (−1)^{n−1} / n^a
         = (1 − 2^{1−a}) ζ(a)
    """
    s = 0.0
    for n in range(1, N + 1):
        s += ((-1) ** (n - 1)) / (n ** a)
    return s


# ============================================================
#  5. Asymmetric polynomial  (on the 1D TSP path)
# ============================================================
def asymmetric_poly(coeffs, x):
    """P(x) = Σ c_k x^k."""
    return sum(c * (x ** k) for k, c in enumerate(coeffs))


def solve_asym_poly_on_tsp(coeffs,
                           x_grid,
                           K_filter=None,
                           mu=None,
                           eta_weight=None,
                           chirp_scale=ALPHA_ASYM):
    """
    Solve an asymmetric polynomial P(x) on the 1D TSP route.

    Steps
    -----
    1. Route the K core indices by chip-g TSP bucket sort.
    2. Reorder x_grid by the TSP route.
    3. Evaluate P(x) at each routed core.
    4. Apply the Dirichlet eta weight η(a) ≈ 0.659538886352.
    5. Apply the quadratic chirp  cos(α_asym · k²).
    6. Root‑bracket the signed stream.
    """
    K = len(x_grid)
    order = tsp_route_1d(K, offset=math.pi / 2)     # asymmetric offset
    x_r = np.asarray(x_grid, dtype=float)[order]
    y_r = np.array([asymmetric_poly(coeffs, float(xi)) for xi in x_r])

    if eta_weight is None:
        eta_weight = dirichlet_eta(ALPHA_ASYM)

    chirp = np.array([math.cos(chirp_scale * k * k) for k in range(K)])
    signed = eta_weight * chirp * y_r

    # root bracketing across the signed stream
    sign = np.sign(signed)
    sign[sign == 0] = 1.0
    crossings = np.where(np.diff(sign) != 0)[0]
    roots = []
    for c in crossings:
        x0, x1 = x_r[c], x_r[c + 1]
        y0, y1 = signed[c], signed[c + 1]
        if y1 - y0 != 0.0:
            roots.append(x0 - y0 * (x1 - x0) / (y1 - y0))
        else:
            roots.append(0.5 * (x0 + x1))

    S_asym = float(np.sum(np.where(np.arange(K) % 2 == 0, signed, -signed)))

    return dict(
        order=order, x_r=x_r, y_r=y_r, signed=signed,
        roots=np.array(roots), eta_weight=eta_weight,
        chirp=chirp, S_asym=S_asym,
    )


# ============================================================
#  MAIN — spectral sum + asymmetric TSP polynomial
# ============================================================
def main():
    # ---------- original spectral sum demo ----------
    K = 100
    mu = mobius_sieve(K)
    coeffs = [mu[i] if i > 0 else 0 for i in range(K + 1)]

    I_abs = integrate_abs_zeta(coeffs, N_points=2000)
    target_eta = ETA_TARGET
    scaling = target_eta / I_abs
    scaled_coeffs = [c * scaling for c in coeffs]

    period = 2 * PI * ALPHA
    N_plot = 500
    t_vals = np.linspace(0, period, N_plot)
    z_vals = np.array([zeta(t, scaled_coeffs, ALPHA) for t in t_vals])
    real_vals = np.real(z_vals)
    imag_vals = np.imag(z_vals)
    abs_vals = np.abs(z_vals)

    avg_abs = np.mean(abs_vals)
    print("=" * 72)
    print("Spectral sum with α = 1/(π − e^0.3628)")
    print("=" * 72)
    print(f"  A                        = {A:.8f}")
    print(f"  α (spectral)             = {ALPHA:.8f}")
    print(f"  η target                 = {target_eta:.12f}")
    print(f"  average |ζ(t)|           = {avg_abs:.8f}")
    print(f"  integral over period     = {avg_abs * period:.8f}")

    # ---------- Dirichlet eta ----------
    eta_num = dirichlet_eta(ALPHA_ASYM, N=20000)
    eta_closed = (1.0 - 2.0 ** (1.0 - ALPHA_ASYM)) * _zeta_scalar(ALPHA_ASYM)
    print()
    print("--- Dirichlet eta ---")
    print(f"  η(0.3628) numerical     = {eta_num:.12f}")
    print(f"  η(0.3628) closed form    = {eta_closed:.12f}")
    print(f"  difference               = {abs(eta_num - eta_closed):.3e}")

    # ---------- asymmetric polynomial on the 1D TSP path ----------
    print()
    print("--- Asymmetric polynomial on the 1D TSP path ---")
    rng = np.random.default_rng(2024)
    degree = 12
    poly_coeffs = rng.standard_normal(degree + 1) * (0.8 ** np.arange(degree + 1))
    poly_coeffs[0] = -0.4                        # ensure a root near x ≈ 0

    x_grid = np.linspace(-1.0, 1.0, 128)
    result = solve_asym_poly_on_tsp(
        poly_coeffs, x_grid,
        K_filter=2 ** 10 + 1, mu=mu, eta_weight=ETA_TARGET,
        chirp_scale=ALPHA_ASYM,
    )
    print(f"  polynomial degree         : {degree}")
    print(f"  grid size                 : {len(x_grid)}")
    print(f"  TSP route (first 12)      : {result['order'][:12].tolist()}")
    print(f"  roots found               : {len(result['roots'])}")
    for r in result['roots'][:6]:
        print(f"    x ≈ {r:+.6f}")
    print(f"  supertrace S_asym         : {result['S_asym']:+.6f}")

    # ---------- plot ----------
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))

    # (1) Re ζ(t)
    axes[0].plot(t_vals, real_vals, 'b-', label='Re ζ(t)')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Real')
    axes[0].legend()
    axes[0].grid(True)

    # (2) Im ζ(t)
    axes[1].plot(t_vals, imag_vals, 'r-', label='Im ζ(t)')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Imag')
    axes[1].legend()
    axes[1].grid(True)

    # (3) |ζ(t)|
    axes[2].plot(t_vals, abs_vals, 'g-', label='|ζ(t)|')
    axes[2].axhline(y=avg_abs, color='m', linestyle='--',
                    label=f'Average = {avg_abs:.6f}')
    axes[2].axhline(y=target_eta, color='orange', linestyle=':',
                    label=f'η(a) = {target_eta:.6f}')
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('|ζ|')
    axes[2].legend()
    axes[2].grid(True)

    # (4) asymmetric polynomial on TSP route
    axes[3].plot(result['x_r'], result['y_r'],
                 color="#34495e", lw=1.0, alpha=0.55,
                 label="P(x) along TSP route")
    axes[3].plot(result['x_r'], result['signed'],
                 color="#c0392b", lw=1.5,
                 label=f"η(a)·chirp(k)·P(x)   (η = {ETA_TARGET:.6f})")
    axes[3].axhline(0, color='k', linestyle='--', alpha=0.5)
    if len(result['roots']) > 0:
        axes[3].scatter(result['roots'],
                        [0.0] * len(result['roots']),
                        color="#f1c40f", edgecolor="k", s=90,
                        zorder=5,
                        label=f"roots ({len(result['roots'])})")
    axes[3].set_xlabel('x  (routed core position)')
    axes[3].set_ylabel('amplitude')
    axes[3].legend(fontsize=9)
    axes[3].grid(True)

    plt.suptitle(
        f'Spectral sum + asymmetric polynomial on 1D TSP\n'
        f'α = {ALPHA:.4f}   η(0.3628) = {ETA_TARGET:.6f}',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ============================================================
#  helper: scalar ζ(a) for the closed-form η check
# ============================================================
def _zeta_scalar(a, N=20000):
    """ζ(a) = Σ_{n≥1} n^{-a}  (real, a > 1)."""
    s = 0.0
    for n in range(1, N + 1):
        s += 1.0 / (n ** a)
    return s


if __name__ == "__main__":
    main()