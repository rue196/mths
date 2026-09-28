#!/usr/bin/env python3
"""
mobius_m_curve.py
=================

Curve primitive based on the M-matrix

    M = [x^-1, y^-1] ⊗ [x^i, y^i]      (2×2 outer product)

For two anchor points  A = (xa, ya)  and  B = (xb, yb):

    algebraic part   ->  Möbius interpolation of inverses
        1/x(t) = (1-t)/xa + t/xb
        1/y(t) = (1-t)/ya + t/yb

    transcendental   ->  [x(t)^i, y(t)^i]

    M-matrix at t    ->  M(t) = [[1/x(t)], [1/y(t)]] ⊗ [[x(t)^i, y(t)^i]]

    curve point      ->  C(t) = Im( [x(t)^i, y(t)^i] ) · (x(t), y(t))
                       = ( x(t)·sin(ln x(t)),  y(t)·sin(ln y(t)) )

The curve is literally the complex part of [x^i, y^i] rescaled by
the algebraic part (the trace of M(t) along its diagonal). No control
points, no polynomial fit — just one inverse and one complex power.

Storage in |Ci| arrays
----------------------
Both algebraic and transcendental values live at symmetric indices
±k around K, exactly like ChipProcessor:
    c[K + k] += value          c[K - k] += value
so the curve family can be swept by reading |Ci| entries without ever
materialising a control polygon.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
#  M-matrix and the curve primitive
# ------------------------------------------------------------------
def m_matrix(x: float, y: float) -> np.ndarray:
    """M(x, y) = [1/x, 1/y]^T ⊗ [x^i, y^i]  →  2×2 complex."""
    alg = np.array([1.0 / x, 1.0 / y], dtype=complex)
    trans = np.array([complex(x) ** 1j, complex(y) ** 1j])
    return np.outer(alg, trans)


def mobius_m_curve(A, B, n: int = 300) -> np.ndarray:
    """
    Curve from A to B via the M-matrix.

    x(t) = 1 / ( (1-t)/xa + t/xb )
    y(t) = 1 / ( (1-t)/ya + t/yb )
    C(t) = ( x(t)·Im(x(t)^i),  y(t)·Im(y(t)^i) )
    """
    t = np.linspace(0.0, 1.0, n)
    xa, ya = A
    xb, yb = B
    x_t = 1.0 / ((1 - t) / xa + t / xb)
    y_t = 1.0 / ((1 - t) / ya + t / yb)
    xt = x_t.astype(complex) ** 1j
    yt = y_t.astype(complex) ** 1j
    return np.stack([x_t * xt.imag, y_t * yt.imag], axis=-1)


def m_matrix_path(A, B, n: int = 300):
    """Return the full M(t) matrices along the curve for inspection."""
    t = np.linspace(0.0, 1.0, n)
    xa, ya = A
    xb, yb = B
    x_t = 1.0 / ((1 - t) / xa + t / xb)
    y_t = 1.0 / ((1 - t) / ya + t / yb)
    xt = x_t.astype(complex) ** 1j
    yt = y_t.astype(complex) ** 1j
    M = np.zeros((n, 2, 2), dtype=complex)
    M[:, 0, 0] = (1.0 / x_t) * xt
    M[:, 0, 1] = (1.0 / x_t) * yt
    M[:, 1, 0] = (1.0 / y_t) * xt
    M[:, 1, 1] = (1.0 / y_t) * yt
    return M, x_t, y_t, xt, yt


# ------------------------------------------------------------------
#  |Ci|-array storage  (symmetric around K, as in ChipProcessor)
# ------------------------------------------------------------------
def store_in_Ci_array(x: float, y: float, K: int = 16) -> np.ndarray:
    """
    Algebraic and transcendental entries at ±1, ±2 around K.

        c[K+1] += 1/x      c[K-1] += 1/x      (algebraic x)
        c[K+2] += 1/y      c[K-2] += 1/y      (algebraic y)
        c[K+1] += x^i      c[K-1] += x^i      (transcendental x)
        c[K+2] += y^i      c[K-2] += y^i      (transcendental y)
    """
    c = np.zeros(2 * K + 1, dtype=complex)
    c[K + 1] += 1.0 / x + complex(x) ** 1j
    c[K - 1] += 1.0 / x + complex(x) ** 1j
    c[K + 2] += 1.0 / y + complex(y) ** 1j
    c[K - 2] += 1.0 / y + complex(y) ** 1j
    return c


# ------------------------------------------------------------------
#  Reference Bezier
# ------------------------------------------------------------------
def cubic_bezier(P0, P1, P2, P3, n: int = 300):
    t = np.linspace(0.0, 1.0, n)[:, None]
    P0, P1, P2, P3 = map(np.asarray, (P0, P1, P2, P3))
    return (((1 - t) ** 3) * P0
            + 3 * ((1 - t) ** 2) * t * P1
            + 3 * (1 - t) * (t ** 2) * P2
            + (t ** 3) * P3)


# ------------------------------------------------------------------
#  A(n) anchors  (used to make the demos meaningful)
# ------------------------------------------------------------------
PI = math.pi
E  = math.e
a  = 1.0 / (PI - E)


def A_of_n(n: float) -> float:
    return 10.0 * a * math.sin(math.radians(n * n))


# ------------------------------------------------------------------
#  Demo
# ------------------------------------------------------------------
def main():
    print("=" * 74)
    print("M-matrix curves  ·  M = [x^-1, y^-1] ⊗ [x^i, y^i]")
    print("=" * 74)
    print(f"  a = 1/(π − e)  = {a:.8f}")
    print()

    # ---------------- Figure -----------------------------------
    fig = plt.figure(figsize=(15, 11))

    # ---- (a) single M-curve vs Bezier ------------------------
    ax = fig.add_subplot(2, 2, 1)
    A = (2.0, 3.0)
    B = (5.0, 4.0)

    C = mobius_m_curve(A, B, n=300)
    ax.plot(C[:, 0], C[:, 1], color="#2c3e50", lw=2.4,
            label="M-curve  (2 anchors)")

    # Bezier through the same endpoints with moderate controls
    P0, P3 = np.array(A), np.array(B)
    P1 = P0 + np.array([1.2, 0.9])
    P2 = P3 + np.array([-1.2, 0.6])
    Bz = cubic_bezier(P0, P1, P2, P3, n=300)
    ax.plot(Bz[:, 0], Bz[:, 1], color="#e74c3c", lw=2.0, ls="--",
            label="cubic Bezier  (4 control points)")

    ax.scatter([P0[0], P3[0]], [P0[1], P3[1]], color="#2c3e50",
               s=60, zorder=5, edgecolors="k", label="anchors")
    ax.scatter([P1[0], P2[0]], [P1[1], P2[1]], color="#e74c3c",
               s=40, marker="^", zorder=5, edgecolors="k",
               label="Bezier controls")

    ax.set_aspect("equal")
    ax.set_title("(a)  M-curve vs Bezier  —  same two anchors")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (b) M-matrix entries along the curve ----------------
    ax = fig.add_subplot(2, 2, 2)
    M, x_t, y_t, xt, yt = m_matrix_path(A, B, n=300)
    t = np.linspace(0, 1, 300)
    ax.plot(t, M[:, 0, 0].real, color="#3498db", lw=1.5,
            label=r"Re $M_{00}$")
    ax.plot(t, M[:, 0, 0].imag, color="#e74c3c", lw=1.5,
            label=r"Im $M_{00} = \sin(\ln x)/x$")
    ax.plot(t, M[:, 1, 1].real, color="#16a085", lw=1.5, ls="--",
            label=r"Re $M_{11}$")
    ax.plot(t, M[:, 1, 1].imag, color="#e67e22", lw=1.5, ls="--",
            label=r"Im $M_{11} = \sin(\ln y)/y$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t"); ax.set_ylabel("M entry")
    ax.set_title("(b)  M-matrix along the curve  A = (2,3), B = (5,4)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- (c) M-curve fan  (all sharing anchor A) --------------
    ax = fig.add_subplot(2, 2, 3)
    anchors = [(A_of_n(n), A_of_n(n + 1)) for n in range(1, 8)]
    print("  A(n)-anchors:")
    for i, p in enumerate(anchors):
        print(f"    P{i}:  ({p[0]:8.4f}, {p[1]:8.4f})")
    print()

    cmap = plt.get_cmap("viridis")
    for i in range(len(anchors) - 1):
        c = mobius_m_curve(anchors[i], anchors[i + 1], n=300)
        ax.plot(c[:, 0], c[:, 1],
                color=cmap(i / max(1, len(anchors) - 2)),
                lw=2.0, label=f"P{i} → P{i+1}")
    for i, p in enumerate(anchors):
        ax.scatter(p[0], p[1], color="k", s=28, zorder=5)
        ax.annotate(f"P{i}", (p[0], p[1]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8)
    ax.set_aspect("equal")
    ax.set_title("(c)  M-curve chain through A(n) anchors")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ---- (d) M-curve flower -----------------------------------
    ax = fig.add_subplot(2, 2, 4)
    n_petals = 24
    R = 4.0
    for k in range(n_petals):
        theta = 2 * math.pi * k / n_petals
        inner = (1.0 + 0.2 * math.cos(theta), 1.0 + 0.2 * math.sin(theta))
        outer = (R * math.cos(theta) + 4.5, R * math.sin(theta) + 4.5)
        # avoid zero / negative coordinates for the log to be real
        outer = (abs(outer[0]) + 0.5, abs(outer[1]) + 0.5)
        c = mobius_m_curve(inner, outer, n=200)
        ax.plot(c[:, 0], c[:, 1],
                color=cmap(k / n_petals), lw=1.6, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title(f"(d)  M-curve flower  ·  {n_petals} petals")
    ax.grid(True, alpha=0.15)

    plt.suptitle(
        "M-matrix curves  ·  M = [x^-1, y^-1]^T ⊗ [x^i, y^i]  ·  "
        "C(t) = ( x(t)·sin(ln x(t)),  y(t)·sin(ln y(t)) )",
        fontsize=12)

    # ---------------- Ci-array check --------------------------
    print("--- |Ci|-array storage ---")
    c = store_in_Ci_array(x=2.0, y=3.0, K=16)
    print(f"  K = 16,  x = 2,  y = 3")
    print(f"  c[K+1] = {c[17]:.4f}")
    print(f"  c[K-1] = {c[15]:.4f}")
    print(f"  c[K+2] = {c[18]:.4f}")
    print(f"  c[K-2] = {c[14]:.4f}")
    print()
    print("--- Cost comparison ---")
    print("  cubic Bezier :  8 floats (4 control points)")
    print("  M-curve      :  4 floats (2 anchors)")
    print("  per sample   :  Bezier = cubic polynomial")
    print("                  M-curve = 1 log + 1 sin + 1 mul")
    print()


if __name__ == "__main__":
    main()