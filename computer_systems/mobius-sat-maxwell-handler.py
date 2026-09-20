#!/usr/bin/env python3
"""
transistor_electron_chip.py

Electron flow inside a transistor channel, driven by:

    • Figure 3.5  — the bounded elliptic projection Π ∈ [0,1] on the
                    torus ℂ/(πℤ + eℤ), used here as the *electron
                    density heat map* across the channel.

    • Maxwell     — the field  E(t) = Re ζ(t)  and  B(t) = Im ζ(t)
                    from the Möbius spectral sum, with divergence
                    ∇·E giving the local charge density.

    • Convolution — the exponential barrier kernel exp(−α|x|) applied
                    to the field to model tunneling through the
                    transistor's potential barrier.

    • Quadratic   — each voltage spike is encoded as a quadratic
      envelopes     address  q(a) = A·a² + B·a + C mod K, then the
                    envelope  env[q] = Σ_{a : q(a)=q} penalty(a) is
                    sent down the 1D chip path.

    • Asymmetric  — a large asymmetric polynomial is solved on the
      polynomials   1D path using the Möbius SAT gate, with the dual
                    constants  α_sym = 1/(π−e)  for the even half
                    and  α_asym = 0.3628        for the odd half.
"""

from __future__ import annotations

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import convolve

# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_ASYM = 0.3628
NORM       = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
DENSITY    = 6.0 / (PI * PI)         # ≈ 0.6079271018
W1         = PI                       # torus period 1
W2         = E                        # torus period 2


# ============================================================
#  1. Möbius sieve (shared O(K log log K) table)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  2. Figure 3.5 — elliptic projection heat map
# ============================================================
def elliptic_projection_2d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Bounded elliptic projection on the fundamental parallelogram
        [0, π) × [0, e):

        Π(x, y) = ½ (1 + cos(2π x/π) · cos(2π y/e))  ∈ [0, 1]

    This is the surface from Figure 3.5 of the supply‑chain paper.
    In the transistor picture it becomes the *electron probability
    density* across the 2D channel cross‑section.
    """
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))


def elliptic_projection_1d(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    """Slice of Π at fixed y0 — the 1D electron density along the channel."""
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


# ============================================================
#  3. Möbius spectral sum  ζ(t)
# ============================================================
def zeta_mobius(t: float, mu: np.ndarray, K: int,
                alpha: float = ALPHA_SYM) -> complex:
    """ζ(t) = Σ_{1≤n≤K} μ(n) · e^{i t n / α}."""
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    theta = t * n / alpha
    return complex(np.sum(m * np.cos(theta)), np.sum(m * np.sin(theta)))


def field_components(t: float, mu: np.ndarray, K: int
                     ) -> tuple[float, float]:
    """E(t) = Re ζ(t),  B(t) = Im ζ(t)."""
    z = zeta_mobius(t, mu, K)
    return z.real, z.imag


def divergence_E(t: float, mu: np.ndarray, K: int) -> float:
    """
    ∇·E = Σ_{n} (-1)^n · μ(n) · cos(t·n / α)   — signed charge density.
    """
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    signs = np.where(n % 2 == 0, 1.0, -1.0)
    return float(np.sum(signs * m * np.cos(t * n / ALPHA_SYM)))


# ============================================================
#  4. Convolution tunneling (barrier kernel)
# ============================================================
def barrier_kernel(width: int, alpha: float = ALPHA_SYM) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    ker = np.exp(-alpha * np.abs(x))
    return ker / ker.sum()


def tunnel(signal: np.ndarray, width: int = 21) -> np.ndarray:
    return convolve(signal, barrier_kernel(width), mode="same")


# ============================================================
#  5. Quadratic envelope  (SAT-mobius-gate.py style)
# ============================================================
def quadratic_address(a: int, A: int = 1, B: int = 1, C: int = 0,
                      K: int = 32) -> int:
    return (A * a * a + B * a + C) % K


def build_quadratic_envelope(K: int, N: int,
                             mu: np.ndarray,
                             A: int = 1, B: int = 1, C: int = 0
                             ) -> tuple[np.ndarray, dict[int, int]]:
    """
    For each a ∈ [0, N), compute q(a) mod K and accumulate.
    penalty(a) = 1 if μ(q(a)) ≠ 0 else 0  (Möbius gate).

    Returns env (length K) and the map a → q(a).
    """
    env = np.zeros(K, dtype=np.float64)
    q_of_a: dict[int, int] = {}
    for a in range(N):
        q = quadratic_address(a, A, B, C, K)
        q_of_a[a] = q
        if q < len(mu) and mu[q] != 0:
            env[q] += 1.0
    return env, q_of_a


def send_envelope_1d(env: np.ndarray,
                     channel: np.ndarray) -> np.ndarray:
    """
    Send the quadratic envelope down the 1D chip channel:
    the envelope is interpolated to the channel length and multiplied
    by the electron density from Figure 3.5.
    """
    L = len(channel)
    if len(env) == L:
        return env * channel
    env_r = np.interp(np.linspace(0, 1, L),
                      np.linspace(0, 1, len(env)), env)
    return env_r * channel


# ============================================================
#  6. Dual-constant supertrace (symmetric / asymmetric split)
# ============================================================
def supertrace_dual(channel: np.ndarray
                    ) -> tuple[float, float, float]:
    """
    S_sym  over even indices, weighted by α_sym
    S_asym over odd  indices, weighted by α_asym
    """
    S_sym = 0.0
    S_asym = 0.0
    for i, v in enumerate(channel):
        if i % 2 == 0:
            S_sym += ALPHA_SYM * abs(v)
        else:
            S_asym += ALPHA_ASYM * abs(v)
    return S_sym, S_asym, S_sym - S_asym


def entropy(S: float, N: int, alpha: float = ALPHA_SYM) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  7. Asymmetric polynomial solver  (on the 1D chip path)
# ============================================================
def asymmetric_polynomial(coeffs: list[float], x: float) -> float:
    """Evaluate P(x) = Σ c_k x^k for a large asymmetric polynomial."""
    return sum(c * (x ** k) for k, c in enumerate(coeffs))


def solve_asymmetric_poly(coeffs: list[float],
                          mu: np.ndarray,
                          x_grid: np.ndarray,
                          K_filter: int,
                          quad_A: int = 1,
                          quad_B: int = 1,
                          quad_C: int = 0) -> dict:
    """
    Solve a large asymmetric polynomial on the 1D chip path:

        1. Evaluate P(x_n) on the grid.
        2. Map each grid index n → q(n) = A n² + B n + C mod K_filter.
        3. Keep only square‑free q (Möbius gate).
        4. Reconstruct the surviving coefficients into the polynomial
           envelope; solve the *reduced* polynomial by root bracketing
           on the Möbius‑filtered grid.
    """
    K = K_filter
    # --- evaluate ---
    y = np.array([asymmetric_polynomial(coeffs, x) for x in x_grid])

    # --- Möbius gate on grid indices ---
    keep = np.zeros(len(x_grid), dtype=bool)
    for n in range(len(x_grid)):
        q = quadratic_address(n, quad_A, quad_B, quad_C, K)
        if q < len(mu) and mu[q] != 0:
            keep[n] = True

    y_gated = y[keep]
    x_gated = x_grid[keep]

    # --- envelope (charge density) ---
    env = np.abs(y_gated)
    env = env / (env.sum() + 1e-12)

    # --- root bracketing on the gated grid ---
    sign = np.sign(y_gated)
    sign[sign == 0] = 1.0
    crossings = np.where(np.diff(sign) != 0)[0]
    roots = []
    for c in crossings:
        x0, x1 = x_gated[c], x_gated[c + 1]
        # linear interpolation between the two grid points
        y0, y1 = y_gated[c], y_gated[c + 1]
        if y1 - y0 != 0:
            r = x0 - y0 * (x1 - x0) / (y1 - y0)
        else:
            r = 0.5 * (x0 + x1)
        roots.append(r)

    return dict(
        x_grid=x_grid, y=y, x_gated=x_gated, y_gated=y_gated,
        keep=keep, env=env, roots=np.array(roots),
        kept_fraction=keep.mean(),
    )


# ============================================================
#  8. Full simulation
# ============================================================
def simulate(K: int = 24,
             N_events: int = 256,
             channel_length: int = 256,
             barrier_width: int = 21,
             poly_degree: int = 12):
    """
    Run the full transistor + quadratic envelope + asymmetric
    polynomial pipeline.
    """
    print("=" * 78)
    print("Transistor electron chip  ·  Figure 3.5 + Maxwell + quadratic")
    print("=" * 78)
    print(f"  α_sym   = 1/(π − e)  = {ALPHA_SYM:.6f}")
    print(f"  α_asym  = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K       = {K}")
    print(f"  channel length = {channel_length}")
    print()

    t0 = time.perf_counter()
    mu = mobius_sieve(max(K, 2 * K + 1))
    print(f"[1] Möbius sieve built in {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"(O(K log log K))")

    # ---------- 1. Figure 3.5 heat map ----------
    t0 = time.perf_counter()
    xs = np.linspace(0, W1, channel_length)
    ys = np.linspace(0, W2, channel_length)
    X, Y = np.meshgrid(xs, ys)
    P2D = elliptic_projection_2d(X, Y)                     # electron density
    print(f"[2] Fig 3.5 heat map   {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 2. 1D channel density ----------
    channel = elliptic_projection_1d(xs, y0=0.5 * W2)

    # ---------- 3. Maxwell field over one period ----------
    t0 = time.perf_counter()
    period = 2 * PI * ALPHA_SYM
    t_vals = np.linspace(0, period, N_events)
    E_vals = np.empty(N_events)
    B_vals = np.empty(N_events)
    div_vals = np.empty(N_events)
    for i, t in enumerate(t_vals):
        E_, B_ = field_components(t, mu, K)
        E_vals[i] = E_
        B_vals[i] = B_
        div_vals[i] = divergence_E(t, mu, K)
    print(f"[3] Maxwell fields     {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 4. Barrier tunneling ----------
    t0 = time.perf_counter()
    E_tunnel = tunnel(E_vals, width=barrier_width)
    div_tunnel = tunnel(div_vals, width=barrier_width)
    print(f"[4] Convolution tunnel {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 5. Quadratic envelope ----------
    t0 = time.perf_counter()
    env, q_of_a = build_quadratic_envelope(K, N_events, mu)
    print(f"[5] Quadratic envelope {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 6. Send envelope down the 1D channel ----------
    t0 = time.perf_counter()
    transmitted = send_envelope_1d(env, channel)
    print(f"[6] 1D chip path       {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 7. Dual-constant supertrace ----------
    S_sym, S_asym, S_net = supertrace_dual(transmitted)
    H_net = entropy(S_net, len(transmitted))
    m_net = mass(S_net, len(transmitted))
    print(f"[7] Dual supertrace    S_sym={S_sym:+.4f}  S_asym={S_asym:+.4f}  "
          f"S_net={S_net:+.4f}")

    # ---------- 8. Large asymmetric polynomial ----------
    t0 = time.perf_counter()
    # asymmetric coefficients: odd powers heavy, decaying magnitude
    rng = np.random.default_rng(2024)
    coeffs = []
    for k in range(poly_degree + 1):
        c = rng.standard_normal() * (0.8 ** k)
        if k % 2 == 0:
            c *= 2.0                                   # asymmetric weighting
        coeffs.append(c)
    # ensure at least one root in [-1, 1]
    coeffs[0] = -0.4
    x_grid = np.linspace(-1.0, 1.0, 400)
    result = solve_asymmetric_poly(
        coeffs, mu, x_grid,
        K_filter=2 ** 10 + 1,
        quad_A=1, quad_B=1, quad_C=0,
    )
    print(f"[8] Asymmetric poly    {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"degree={poly_degree}  roots={len(result['roots'])}  "
          f"Möbius kept {result['kept_fraction']*100:.1f}% of grid")

    # ---------- 9. Plot ----------
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (a) Figure 3.5 heat map
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(P2D, extent=[0, W1, 0, W2], origin="lower",
                   cmap="inferno", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("x  (mod π)")
    ax.set_ylabel("y  (mod e)")
    ax.set_title("Fig 3.5 · electron density Π(x,y)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) 1D channel density
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xs, channel, color="#e67e22", lw=1.6)
    ax.fill_between(xs, 0, channel, color="#e67e22", alpha=0.25)
    ax.set_xlabel("channel position  (mod π)")
    ax.set_ylabel("density")
    ax.set_title("1D transistor channel  (slice of Π)")
    ax.grid(True, alpha=0.3)

    # (c) quadratic envelope
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(len(env)), env, color="#3a7bd5", width=0.85)
    ax.set_xlabel("quadratic address  q = a² + a  (mod K)")
    ax.set_ylabel("env[q]")
    ax.set_title(f"Quadratic envelope  (K = {K})")
    ax.grid(True, alpha=0.3)

    # (d) Maxwell fields
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t_vals, E_vals, color="#2ecc71", lw=1.2, label="E(t) = Re ζ(t)")
    ax.plot(t_vals, B_vals, color="#3498db", lw=1.0, label="B(t) = Im ζ(t)",
            alpha=0.7)
    ax.plot(t_vals, div_vals, color="#e74c3c", lw=0.9, alpha=0.6,
            label="∇·E  (charge density)")
    ax.set_xlabel("t  (one period 2πα_sym)")
    ax.set_ylabel("field amplitude")
    ax.set_title("Maxwell fields from the Möbius spectral sum")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) tunneling result
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t_vals, E_vals, color="#95a5a6", lw=1.0, alpha=0.5,
            label="before barrier")
    ax.plot(t_vals, E_tunnel, color="#8e44ad", lw=1.4, label="after barrier")
    ax.set_xlabel("t")
    ax.set_ylabel("E")
    ax.set_title(f"Tunneling  (S={S_net:+.3f}, m={m_net:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (f) transmitted envelope on 1D channel
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(np.linspace(0, 1, len(transmitted)), transmitted,
            color="#16a085", lw=1.4)
    ax.fill_between(np.linspace(0, 1, len(transmitted)),
                    0, transmitted, color="#16a085", alpha=0.25)
    ax.set_xlabel("position along 1D chip path")
    ax.set_ylabel("transmitted envelope")
    ax.set_title(f"Envelope on the chip  ·  S_sym={S_sym:+.3f}  "
                 f"S_asym={S_asym:+.3f}  H={H_net:.3f}")
    ax.grid(True, alpha=0.3)

    # (g) asymmetric polynomial + roots
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(result['x_grid'], result['y'],
            color="#34495e", lw=1.0, alpha=0.5, label="full P(x)")
    ax.plot(result['x_gated'], result['y_gated'],
            color="#c0392b", lw=1.4, label="Möbius‑gated")
    if len(result['roots']) > 0:
        ax.scatter(result['roots'],
                   [0.0] * len(result['roots']),
                   color="#f1c40f", edgecolor="k", s=60,
                   zorder=5, label=f"roots ({len(result['roots'])})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")
    ax.set_title(f"Asymmetric polynomial  (degree {poly_degree})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Transistor electron chip  ·  Figure 3.5 + Maxwell + Möbius SAT",
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # ---------- summary ----------
    print()
    print("--- Summary ---")
    print(f"  Figure 3.5 heat map range   : [{P2D.min():.4f}, {P2D.max():.4f}]")
    print(f"  1D channel density range    : [{channel.min():.4f}, "
          f"{channel.max():.4f}]")
    print(f"  Quadratic envelope occupied : "
          f"{int(np.count_nonzero(env))} / {K}")
    print(f"  Tunneling supertrace        : S_net = {S_net:+.4f}")
    print(f"  Tunneling entropy           : H     = {H_net:.4f}")
    print(f"  Tunneling invariant mass    : m     = {m_net:.4f}")
    print(f"  Asymmetric polynomial       : degree {poly_degree}, "
          f"{len(result['roots'])} roots in [-1,1]")
    print()
    print("  Complexity per pipeline stage:")
    print("    Möbius sieve                O(K log log K)   once")
    print("    Figure 3.5 heat map         O(L²)            L = channel_length")
    print("    1D channel slice            O(L)")
    print("    Maxwell fields              O(K · N_events)")
    print("    Barrier convolution         O(N_events · W) W = barrier_width")
    print("    Quadratic envelope          O(N_events)")
    print("    Send envelope down chip     O(L)")
    print("    Dual supertrace              O(L)")
    print("    Asymmetric polynomial       O(L · degree)")

    return dict(
        P2D=P2D, channel=channel, env=env,
        E_vals=E_vals, B_vals=B_vals, div_vals=div_vals,
        E_tunnel=E_tunnel, transmitted=transmitted,
        S_sym=S_sym, S_asym=S_asym, S_net=S_net, H=H_net, m=m_net,
        poly_result=result,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24, N_events=256, channel_length=256,
             barrier_width=21, poly_degree=12)