#!/usr/bin/env python3
"""
finite_step_objects.py
======================

Torus, donut, and mug built from the finite-step sequence

    A(n)  = 10 · a · sin(n²°) ,   a = 1/(π − e)
    ΔA_n  = A(n+1) − A(n)

Interpretation across the three objects
---------------------------------------
    Torus      ring radius     ∝ A(n)          (major radius)
               tube radius     ∝ |ΔA_n|        (minor radius)
    Donut      same, but the tube is oblate:
               r_z = 0.55 · r_xy               (flattened cross-section)
    Mug        body radius      ∝ A(n)         (subtle taper)
               slice heights    ∝ cumsum(ΔA)   (non-uniform z spacing)
               handle arc       ∝ A_mid
               handle tube      ∝ |ΔA_mid|

The sequence is used for n = 1 … N (default N = 8), which stays
monotone increasing, so the shapes are clean and recognisable.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------------------------------------------------
#  Constants and the finite-step sequence
# ------------------------------------------------------------------
PI = math.pi
E  = math.e
a  = 1.0 / (PI - E)                       # ≈ 2.362338


def A_of_n(n: float) -> float:
    """A(n) = 10 · a · sin(n²°)."""
    return 10.0 * a * math.sin(math.radians(n * n))


def finite_step(n: int) -> float:
    """ΔA_n = A(n+1) − A(n)."""
    return A_of_n(n + 1) - A_of_n(n)


# ------------------------------------------------------------------
#  Interpolator — A(t) and ΔA(t) for real t ∈ [1, N]
# ------------------------------------------------------------------
class FiniteStepInterp:
    """Linear interpolation of A and ΔA on real t ∈ [1, N]."""

    def __init__(self, N: int = 8):
        self.N = int(N)
        ns = np.arange(1, N + 1, dtype=float)
        self.ns  = ns
        self.As  = np.array([A_of_n(n)     for n in ns])
        self.dAs = np.array([finite_step(int(n)) for n in ns])
        self.A_max  = float(np.max(np.abs(self.As)))
        self.dA_max = float(np.max(np.abs(self.dAs)))
        self.A1  = float(self.As[0])
        self.AN  = float(self.As[-1])

    def A(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.ns, self.As)

    def dA(self, t: np.ndarray) -> np.ndarray:
        return np.interp(t, self.ns, self.dAs)


# ==================================================================
#  TORUS
# ==================================================================
def make_torus(N: int = 8,
               u_res: int = 180, v_res: int = 50,
               R_min: float = 1.00, R_max: float = 3.00,
               r_min: float = 0.18, r_max: float = 0.48):
    """
    Torus with a finite-step-scaled major and minor radius.

        u ∈ [0, 2π)  →  n(u) = 1 + u/(2π) · (N − 1)
        R_major(u)   = R_min + (R_max − R_min) · (A(n(u)) − A₁)/(A_N − A₁)
        r_minor(u)   = r_min + (r_max − r_min) · |ΔA(n(u))| / ΔA_max
    """
    fsi = FiniteStepInterp(N)

    u = np.linspace(0.0, 2 * PI, u_res)
    v = np.linspace(0.0, 2 * PI, v_res)
    U, V = np.meshgrid(u, v, indexing="ij")

    n_t = 1.0 + (U / (2 * PI)) * (N - 1)
    A_t  = fsi.A(n_t)
    dA_t = np.abs(fsi.dA(n_t))

    t     = (A_t - fsi.A1) / (fsi.AN - fsi.A1)
    R     = R_min + (R_max - R_min) * t
    r     = r_min + (r_max - r_min) * (dA_t / fsi.dA_max)

    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    return X, Y, Z


# ==================================================================
#  DONUT  (oblate torus)
# ==================================================================
def make_donut(N: int = 8,
               u_res: int = 180, v_res: int = 50,
               R_min: float = 1.00, R_max: float = 2.00,
               r_xy_min: float = 0.30, r_xy_max: float = 0.60,
               flat: float = 0.55):
    """
    Donut = torus with an elliptical (oblate) tube cross-section:

        r_xy(u) = r_xy_min + (r_xy_max − r_xy_min) · |ΔA(n(u))| / ΔA_max
        r_z(u)  = flat · r_xy(u)
    """
    fsi = FiniteStepInterp(N)

    u = np.linspace(0.0, 2 * PI, u_res)
    v = np.linspace(0.0, 2 * PI, v_res)
    U, V = np.meshgrid(u, v, indexing="ij")

    n_t = 1.0 + (U / (2 * PI)) * (N - 1)
    A_t  = fsi.A(n_t)
    dA_t = np.abs(fsi.dA(n_t))

    t     = (A_t - fsi.A1) / (fsi.AN - fsi.A1)
    R     = R_min + (R_max - R_min) * t
    r_xy  = r_xy_min + (r_xy_max - r_xy_min) * (dA_t / fsi.dA_max)
    r_z   = flat * r_xy

    X = (R + r_xy * np.cos(V)) * np.cos(U)
    Y = (R + r_xy * np.cos(V)) * np.sin(U)
    Z = r_z * np.sin(V)
    return X, Y, Z


# ==================================================================
#  MUG
# ==================================================================
def make_mug_body(N: int = 8,
                  theta_res: int = 90,
                  R_min: float = 0.85, R_max: float = 1.00,
                  H: float = 2.20):
    """
    Surface of revolution.

        slice heights  z_k = cumsum(ΔA)_k / cumsum(ΔA)_N · H
        slice radii    R_k = R_min + (R_max − R_min) ·
                              (A(k) − A₁) / (A_N − A₁)
    """
    fsi = FiniteStepInterp(N)

    dAs     = np.array([finite_step(n) for n in range(1, N + 1)])
    z_slice = np.concatenate([[0.0], np.cumsum(dAs)])
    z_slice = z_slice / z_slice[-1] * H

    As      = np.array([A_of_n(n) for n in range(1, N + 1)])
    t_norm  = (As - As[0]) / (As[-1] - As[0])
    radii   = R_min + (R_max - R_min) * t_norm
    r_all   = np.concatenate([radii, [radii[-1]]])   # extend to N+1

    theta = np.linspace(0.0, 2 * PI, theta_res)
    k_idx, th = np.meshgrid(np.arange(N + 1), theta, indexing="ij")

    R_grid = r_all[k_idx]
    Z_grid = z_slice[k_idx]

    X = R_grid * np.cos(th)
    Y = R_grid * np.sin(th)
    return X, Y, Z_grid


def make_mug_handle(N: int = 8,
                    phi_res: int = 70, psi_res: int = 22,
                    x_attach: float = 1.00,
                    z_center: float | None = None,
                    R_arc: float | None = None,
                    r_tube: float | None = None,
                    H: float = 2.20):
    """
    Toroidal handle arc on the +x side of the mug.

        φ ∈ [−π/2, +π/2]   arc angle (in the x−z plane)
        ψ ∈ [0, 2π]        tube angle around the arc

        R_arc   ← A(N//2)            (mid-curve major radius)
        r_tube  ← |ΔA(N//2)|         (mid-curve tube radius)
    """
    fsi = FiniteStepInterp(N)

    if R_arc is None:
        R_arc  = 0.28 * fsi.A(float(N // 2 + 1))
    if r_tube is None:
        r_tube = 0.55 * abs(finite_step(N // 2))
    if z_center is None:
        z_center = 0.5 * H

    phi = np.linspace(-PI / 2, PI / 2, phi_res)
    psi = np.linspace(0.0, 2 * PI, psi_res)
    PHI, PSI = np.meshgrid(phi, psi, indexing="ij")

    cos_p, sin_p = np.cos(PHI), np.sin(PHI)
    cos_s, sin_s = np.cos(PSI), np.sin(PSI)

    X = x_attach + R_arc * cos_p + r_tube * cos_s * cos_p
    Y =             r_tube * sin_s
    Z = z_center + R_arc * sin_p + r_tube * cos_s * sin_p
    return X, Y, Z


# ==================================================================
#  Plotting
# ==================================================================
def plot_scaling_curve(ax, N: int):
    ns = np.arange(1, N + 1)
    As  = np.array([A_of_n(k)       for k in ns])
    dAs = np.array([finite_step(k)  for k in ns])

    ax.plot(ns, As, "o-", color="#2c3e50", lw=1.8,
            label=r"$A(n)$  (scaling)")
    ax.bar(ns + 0.15, dAs, width=0.30, color="#e67e22",
           alpha=0.80, label=r"$\Delta A_n$  (finite-step)")
    for k, (A_, d_) in enumerate(zip(As, dAs), start=1):
        ax.annotate(f"{A_:.2f}", (k, A_),
                    textcoords="offset points", xytext=(0, 6),
                    fontsize=7, ha="center", color="#2c3e50")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$A(n)$  /  $\Delta A_n$")
    ax.set_title(f"Finite-step sequence  n = 1 … {N}")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)


def style_3d(ax, title: str,
             X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    span = max(float(np.ptp(X)), float(np.ptp(Y)), float(np.ptp(Z)))
    for setter, mid in (
        (ax.set_xlim, 0.5 * (X.min() + X.max())),
        (ax.set_ylim, 0.5 * (Y.min() + Y.max())),
        (ax.set_zlim, 0.5 * (Z.min() + Z.max())),
    ):
        setter(mid - 0.55 * span, mid + 0.55 * span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=-58)


# ==================================================================
#  Main
# ==================================================================
def main():
    N = 8
    fsi = FiniteStepInterp(N)

    print("=" * 72)
    print("Finite-step objects  ·  A(n) = 10 a sin(n²°),  a = 1/(π − e)")
    print("=" * 72)
    print(f"  a = 1/(π − e)  = {a:.8f}")
    print()
    print(f"  {'n':>3}  {'A(n)':>10}  {'ΔA_n':>10}  "
          f"{'cumsum(ΔA)':>12}")
    cum = 0.0
    for n in range(1, N + 2):
        A_n = A_of_n(n)
        if n <= N:
            dA = finite_step(n)
            cum += dA
            print(f"  {n:>3}  {A_n:>10.5f}  {dA:>10.5f}  {cum:>12.5f}")
        else:
            print(f"  {n:>3}  {A_n:>10.5f}  (reference)")
    print()

    # ---- build the objects ----
    Xt, Yt, Zt = make_torus(N,
                            R_min=1.00, R_max=1.00,
                            r_min=0.16, r_max=0.46)
    Xd, Yd, Zd = make_donut(N,
                            R_min=1.00, R_max=1.00,
                            r_xy_min=0.30, r_xy_max=0.58,
                            flat=0.55)
    Xb, Yb, Zb = make_mug_body(N,
                               R_min=0.82, R_max=1.00, H=2.20)
    Xh, Yh, Zh = make_mug_handle(N,
                                 x_attach=1.00,
                                 z_center=1.10,
                                 R_arc=0.58,
                                 r_tube=0.12,
                                 H=2.20)

    # ---- figure ----
    fig = plt.figure(figsize=(15, 11))

    ax0 = fig.add_subplot(2, 2, 1)
    plot_scaling_curve(ax0, N)

    ax1 = fig.add_subplot(2, 2, 2, projection="3d")
    ax1.plot_surface(Xt, Yt, Zt, cmap="viridis",
                     alpha=0.92, linewidth=0, antialiased=True,
                     rstride=4, cstride=2, edgecolor="k")
    style_3d(ax1, "Torus  ·  R ∝ A(n),  r ∝ |ΔAₙ|",
             Xt, Yt, Zt)

    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    ax2.plot_surface(Xd, Yd, Zd, cmap="plasma",
                     alpha=0.92, linewidth=0, antialiased=True,
                     rstride=4, cstride=2, edgecolor="k")
    style_3d(ax2, "Donut  ·  oblate tube  (r_z = 0.55 · r_xy)",
             Xd, Yd, Zd)

    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    ax3.plot_surface(Xb, Yb, Zb, cmap="YlOrBr",
                     alpha=0.92, linewidth=0, antialiased=True,
                     rstride=2, cstride=4, edgecolor="k")
    ax3.plot_surface(Xh, Yh, Zh, cmap="YlOrBr",
                     alpha=0.92, linewidth=0, antialiased=True,
                     rstride=2, cstride=2, edgecolor="k")
    style_3d(ax3,
             "Mug  ·  slices ∝ cumsum(ΔAₙ),  handle ∝ A, ΔA",
             np.concatenate([Xb.ravel(), Xh.ravel()]),
             np.concatenate([Yb.ravel(), Yh.ravel()]),
             np.concatenate([Zb.ravel(), Zh.ravel()]))

    plt.suptitle(
        r"Finite-step objects  ·  "
        r"$A(n)=10a\sin(n^{2\circ})$  ·  "
        r"$\Delta A_n = A(n{+}1)-A(n)$  ·  "
        r"one sequence, three shapes",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()