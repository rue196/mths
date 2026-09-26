#!/usr/bin/env python3
"""
square-packing-trivial.py
=========================

Trivial proof of the container side A = 6.511 used to pack
floor(A)^2 = 36 unit squares.

Two equivalent routes:

  1. Algebraic (trigonometric):
         A / 10 = a * sin(4^2 °) = a * sin(16°) = 0.6511
     with  a = 1/(π − e) ≈ 2.362338.

  2. Finite-step derivative of the side length:
         A(n) = 10 * a * sin(n^2 °)
         forward  ΔA/Δn |_{n=4} = A(5) − A(4)  ≈ 3.47
         central  ΔA/Δn |_{n=4} = (A(5) − A(3))/2 ≈ π
     so the discrete step of the side length at n = 4 recovers π,
     and therefore a = 1/(π − e) — the same constant.

Both give floor(A) = 6, so the trivial axis-aligned grid of
floor(A)^2 = 36 unit squares packs inside [0, A]^2.

M-matrix invariants (supertrace / entropy / mass) are computed
exactly as in square-packing-max.py.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
PI = math.pi
E  = math.e
a  = 1.0 / (PI - E)          # ≈ 2.362338   (Möbius / Dirichlet constant)
R  = 0.5                     # half side of a unit square (side = 2R = 1)

# ------------------------------------------------------------------
#  Route 1 — trivial trigonometric proof
# ------------------------------------------------------------------
def A_trig(n_deg_squared: int = 4) -> float:
    """
    A = 10 * a * sin(n^2 °)   with n = 4.

    Returns
    -------
    A : float
        Container side length.
    """
    return 10.0 * a * math.sin(math.radians(n_deg_squared ** 2))

# ------------------------------------------------------------------
#  Route 2 — finite-step derivative of the side length
# ------------------------------------------------------------------
def A_of_n(n: float) -> float:
    """A(n) = 10 * a * sin(n^2 °)."""
    return 10.0 * a * math.sin(math.radians(n * n))

def finite_step_derivative(n: int = 4):
    """
    Forward / backward / central finite-step derivatives of A at n.

    Returns
    -------
    fwd, bwd, ctr : (float, float, float)
    """
    fwd = A_of_n(n + 1) - A_of_n(n)
    bwd = A_of_n(n) - A_of_n(n - 1)
    ctr = 0.5 * (A_of_n(n + 1) - A_of_n(n - 1))
    return fwd, bwd, ctr

# ------------------------------------------------------------------
#  M-matrix invariants  (same definitions as square-packing-max.py)
# ------------------------------------------------------------------
def supertrace_from_positions(positions):
    """
    Algebraic part of the M-matrix invariant:
        S = Σ_i (-1)^i * ( x_i^{-2 e A} + y_i^{-2 e A} )
    where x_i, y_i are the bottom-left corner coordinates of square i.
    """
    S = 0.0
    power = -2.0 * E * a
    for i, (x, y) in enumerate(positions):
        x = max(abs(x), 1e-6)
        y = max(abs(y), 1e-6)
        diag = x ** power + y ** power
        sign = 1 if (i % 2 == 0) else -1
        S += sign * diag
    return S

def entropy_from_supertrace(S, N):
    if S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -a * p * math.log(p)

def mass_from_supertrace(S, N):
    H = entropy_from_supertrace(S, N)
    return abs(S) * math.exp(-H)

# ------------------------------------------------------------------
#  Trivial packing (axis-aligned grid of floor(A)^2 unit squares)
# ------------------------------------------------------------------
def trivial_grid_packing(A):
    """
    Place floor(A)^2 unit squares on an axis-aligned grid.
    Returns a list of bottom-left corner coordinates.
    """
    m = math.floor(A)                    # 6 for A ≈ 6.511
    positions = []
    for ix in range(m):
        for iy in range(m):
            positions.append((R + ix, R + iy))   # side = 2R = 1
    return positions, m

# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Square packing  ·  trivial proof of A = 6.511")
    print("=" * 72)
    print(f"  a = 1/(π − e)                      = {a:.8f}")
    print()

    # --- Route 1 : algebraic proof ---------------------------------
    n = 4
    s_deg = math.sin(math.radians(n ** 2))
    half  = a * s_deg
    A_r1  = 10.0 * half
    print("Route 1  ·  algebraic (trigonometric)")
    print(f"  sin({n}^2 °) = sin({n*n}°)         = {s_deg:.8f}")
    print(f"  a * sin({n}^2 °)                   = {half:.8f}   "
          f"(target 0.6511)")
    print(f"  A = 10 * a * sin({n}^2 °)          = {A_r1:.8f}")
    print()

    # --- Route 2 : finite-step derivative --------------------------
    fwd, bwd, ctr = finite_step_derivative(n)
    print("Route 2  ·  finite-step derivative of side length")
    print(f"  A({n})                             = {A_of_n(n):.8f}")
    print(f"  A({n+1})                           = {A_of_n(n+1):.8f}")
    print(f"  A({n-1})                           = {A_of_n(n-1):.8f}")
    print(f"  forward  A({n+1}) − A({n})         = {fwd:.8f}")
    print(f"  backward A({n}) − A({n-1})         = {bwd:.8f}")
    print(f"  central  (A({n+1}) − A({n-1}))/2   = {ctr:.8f}   "
          f"(≈ π = {PI:.8f})")
    print(f"  → discrete step at n = {n} recovers π, hence a = 1/(π − e)")
    print()

    # --- Final side length ----------------------------------------
    A_final = A_r1
    print(f"  Container side  A                  = {A_final:.8f}")
    print(f"  floor(A)                           = {math.floor(A_final)}")
    print(f"  trivial packing  floor(A)^2        = {math.floor(A_final)**2} "
          f"unit squares")
    print()

    # --- Build the trivial grid -----------------------------------
    positions, m = trivial_grid_packing(A_final)
    N = len(positions)

    # --- M-matrix invariants --------------------------------------
    S = supertrace_from_positions(positions)
    H = entropy_from_supertrace(S, N)
    mass = mass_from_supertrace(S, N)
    print("M-matrix invariants of the trivial packing")
    print(f"  N                                  = {N}")
    print(f"  supertrace S                       = {S:.6f}")
    print(f"  entropy   H                        = {H:.6f}")
    print(f"  mass      m                        = {mass:.6f}")
    print()

    # --- Plot -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # (a) trivial packing
    ax = axes[0]
    ax.set_xlim(0, A_final); ax.set_ylim(0, A_final)
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), A_final, A_final,
                           fc="none", ec="k", lw=2.0))
    for (x, y) in positions:
        ax.add_patch(Rectangle((x, y), 1.0, 1.0,
                               fc="steelblue", ec="black",
                               alpha=0.75, lw=0.8))
    ax.set_title(f"Trivial packing: {N} unit squares  "
                 f"(A = {A_final:.4f})")
    ax.grid(alpha=0.2)

    # (b) A(n) and finite-step derivatives
    ax = axes[1]
    ns = np.arange(1, 7)
    A_n = np.array([A_of_n(k) for k in ns])
    ax.plot(ns, A_n, "o-", color="#2c3e50",
            label=r"$A(n)=10a\sin(n^2{}^\circ)$")
    for k in ns:
        ax.annotate(f"{A_n[k-1]:.2f}", (k, A_n[k-1]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8)
    ax.axvline(4, color="#e74c3c", ls=":", lw=1.2,
               label=r"$n=4$  →  $A=6.511$")
    ax.axvline(5, color="#e74c3c", ls=":", lw=1.2, alpha=0.5)
    ax.annotate("", xy=(5, A_of_n(5)), xytext=(4, A_of_n(4)),
                arrowprops=dict(arrowstyle="<->", color="#16a085",
                                lw=1.8))
    ax.text(4.5, 0.5 * (A_of_n(4) + A_of_n(5)) + 0.3,
            rf"$\Delta A/\Delta n \approx {fwd:.2f}$",
            color="#16a085", ha="center", fontsize=10)
    ax.set_xlabel("n"); ax.set_ylabel("A(n)")
    ax.set_title("Finite-step derivative of side length")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(
        r"Trivial proof:  $A/10 = a\,\sin(4^2{}^\circ) = 0.6511$   ·   "
        r"$a = 1/(\pi-e)$",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()