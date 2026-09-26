#!/usr/bin/env python3
"""
finite_step_3d_object.py
========================

A 3D object built from the finite-step derivative of

    A(n) = 10 · a · sin(n²°) ,   a = 1/(π − e)

Interpretation
--------------
    A(n)               →  scaling value of shell / layer n
    ΔA_n = A(n+1) − A(n) →  finite-step derivative
                        =  local scaling of the grid
                        =  shell thickness / layer spacing
                        =  amplitude of the organic deformation

Organic surface
---------------
Each shell is a warped 2D parameter grid (u, v) ∈ [0, 1]² mapped onto
a sphere via

    θ = 2π · u_w ,      φ = π · v_w
    r = A(n) · (1 + ε · sin(2π f u) · sin(2π f v))
    x = r sin φ cos θ ,  y = r sin φ sin θ ,  z = r cos φ

The warp (u_w, v_w) is a smooth sinusoidal distortion of the flat
2D grid — this is what gives the object its organic character.
"""

from __future__ import annotations
import math
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
PI = math.pi
E  = math.e
a  = 1.0 / (PI - E)                       # ≈ 2.362338


# ------------------------------------------------------------------
#  Scaling and its finite-step derivative
# ------------------------------------------------------------------
def A_of_n(n: float) -> float:
    """A(n) = 10 · a · sin(n²°)  with a = 1/(π − e)."""
    return 10.0 * a * math.sin(math.radians(n * n))


def finite_step(n: int, mode: str = "forward") -> float:
    """Forward / backward / central finite-step derivative of A at n."""
    if mode == "forward":
        return A_of_n(n + 1) - A_of_n(n)
    if mode == "backward":
        return A_of_n(n) - A_of_n(n - 1)
    if mode == "central":
        return 0.5 * (A_of_n(n + 1) - A_of_n(n - 1))
    raise ValueError(mode)


# ------------------------------------------------------------------
#  Organic warped 2D grid
# ------------------------------------------------------------------
def organic_uv(nu: int = 60, nv: int = 60,
               warp: float = 0.18):
    """
    Warped (u_w, v_w) grid in [0, 1]².
    `warp` is the amplitude of the sinusoidal distortion.
    """
    u = np.linspace(0.0, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    U, V = np.meshgrid(u, v, indexing="ij")

    Uw = U + warp * np.sin(2 * PI * V + 0.4) \
           + 0.06 * np.cos(4 * PI * U + 0.9)
    Vw = V + warp * np.sin(2 * PI * U + 1.2) \
           + 0.06 * np.cos(4 * PI * V + 0.2)
    return U, V, Uw, Vw


# ------------------------------------------------------------------
#  Radial modulation — the "organic" deformation of the sphere
# ------------------------------------------------------------------
def organic_radius(U: np.ndarray, V: np.ndarray,
                   amp: float = 0.20,
                   freq: float = 3.0,
                   phase: float = 0.0) -> np.ndarray:
    """
    rho(U, V) = 1 + amp · sin(2π f U + φ) · sin(2π f V + φ)

    A smooth, closed, non-axisymmetric "blob" surface.
    """
    return 1.0 + amp * np.sin(2 * PI * freq * U + phase) \
                     * np.sin(2 * PI * freq * V + phase)


# ------------------------------------------------------------------
#  Spherical mapping of the warped grid
# ------------------------------------------------------------------
def shell_xyz(Uw: np.ndarray, Vw: np.ndarray,
              R: float = 1.0,
              radius_mod=None):
    """Map the warped (u, v) grid onto a sphere of radius R (modulated)."""
    theta = 2 * PI * Uw
    phi   = PI * Vw
    if radius_mod is None:
        r = np.full_like(Uw, R, dtype=float)
    else:
        r = R * radius_mod(Uw, Vw)
    X = r * np.sin(phi) * np.cos(theta)
    Y = r * np.sin(phi) * np.sin(theta)
    Z = r * np.cos(phi)
    return X, Y, Z


# ------------------------------------------------------------------
#  The 3D object — nested organic shells
# ------------------------------------------------------------------
def build_nested_shells(n_layers: int = 6,
                        nu: int = 60, nv: int = 60,
                        warp: float = 0.18,
                        mod_amp: float = 0.22,
                        mod_freq: float = 3.0):
    """
    One shell per n = 1 … n_layers.

    radius = A(n)          (the scaling value)
    mod_amp scaled by ΔA_n (finite-step derivative drives deformation)
    """
    U, V, Uw, Vw = organic_uv(nu, nv, warp)
    ref_step = abs(finite_step(4))          # ≈ 3.47, our reference
    shells = []
    for n in range(1, n_layers + 1):
        R     = A_of_n(n)
        step  = finite_step(n, "forward")
        amp   = mod_amp * (step / ref_step)
        phase = 0.10 * n
        mod   = partial(organic_radius,
                        amp=amp, freq=mod_freq, phase=phase)
        X, Y, Z = shell_xyz(Uw, Vw, R=R, radius_mod=mod)
        shells.append(dict(
            n=n, scale=R, step=step, amp=amp,
            U=U, V=V, Uw=Uw, Vw=Vw,
            X=X, Y=Y, Z=Z,
        ))
    return shells


# ------------------------------------------------------------------
#  Scaling tower — stacked warped 2D patches
# ------------------------------------------------------------------
def build_scaling_tower(n_layers: int = 8,
                        nu: int = 22, nv: int = 22,
                        warp: float = 0.12):
    """
    Layer n is a warped 2D patch in the plane z = z_n.
    patch scale   = A(n)              (the scaling value)
    layer spacing = ΔA_n              (finite-step derivative)
    """
    layers = []
    z = 0.0
    for n in range(1, n_layers + 1):
        scale = A_of_n(n)
        dz    = finite_step(n, "forward")
        z    += dz
        U, V, Uw, Vw = organic_uv(nu, nv, warp)
        X = scale * (Uw - 0.5)
        Y = scale * (Vw - 0.5)
        Z = np.full_like(X, z)
        layers.append(dict(n=n, z=z, scale=scale, dz=dz,
                           X=X, Y=Y, Z=Z))
    return layers


# ------------------------------------------------------------------
#  Plotting helpers
# ------------------------------------------------------------------
def plot_scaling_curve(ax):
    ns = np.arange(1, 10)
    A  = np.array([A_of_n(k)         for k in ns])
    dA = np.array([finite_step(k)    for k in ns])

    ax.plot(ns, A, "o-", color="#2c3e50", lw=1.6,
            label=r"$A(n)$")
    ax.bar(ns + 0.15, dA, width=0.3, color="#e67e22",
           alpha=0.75, label=r"$\Delta A_n = A(n{+}1)-A(n)$")
    ax.axvline(4, color="#e74c3c", ls=":", lw=1.2,
               label=r"$n=4$  →  $A=6.511$,  $\Delta A\approx 3.47$")
    ax.set_xlabel("n"); ax.set_ylabel(r"$A(n)$  /  $\Delta A_n$")
    ax.set_title(r"Scaling $A(n)$ and finite-step derivative $\Delta A_n$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def plot_nested_shells(ax, shells, alpha=0.16):
    ax.set_title("Nested organic shells  ·  radius $A(n)$  ·  "
                 "surface = warped 2D grid")
    cmap = plt.get_cmap("viridis")
    for sh in shells:
        col = cmap((sh["n"] - 1) / max(1, len(shells) - 1))
        ax.plot_surface(sh["X"], sh["Y"], sh["Z"],
                        color=col, alpha=alpha,
                        linewidth=0, antialiased=True)
        ax.plot_wireframe(sh["X"], sh["Y"], sh["Z"],
                          color=col, linewidth=0.25, alpha=0.55,
                          rstride=5, cstride=5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))


def plot_single_shell(ax, sh, alpha=0.32):
    ax.set_title(f"Single organic shell   n = {sh['n']}   ·   "
                 f"A({sh['n']}) = {sh['scale']:.3f}   ·   "
                 f"ΔA = {sh['step']:.3f}")
    ax.plot_surface(sh["X"], sh["Y"], sh["Z"],
                    cmap="viridis", alpha=alpha,
                    linewidth=0, antialiased=True)
    ax.plot_wireframe(sh["X"], sh["Y"], sh["Z"],
                      color="k", linewidth=0.4, alpha=0.55,
                      rstride=3, cstride=3)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))


def plot_scaling_tower(ax, layers):
    ax.set_title("Scaling tower  ·  patch scale $A(n)$  ·  "
                 "layer spacing $\\Delta A_n$")
    cmap = plt.get_cmap("plasma")
    for L in layers:
        col = cmap((L["n"] - 1) / max(1, len(layers) - 1))
        ax.plot_surface(L["X"], L["Y"], L["Z"],
                        color=col, alpha=0.68,
                        linewidth=0, antialiased=True)
        ax.plot_wireframe(L["X"], L["Y"], L["Z"],
                          color="k", linewidth=0.35, alpha=0.55,
                          rstride=2, cstride=2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1.4))


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Finite-step 3D object  ·  A(n) = 10 a sin(n²°),  a = 1/(π − e)")
    print("=" * 72)
    print(f"  a = 1/(π − e)   = {a:.8f}")
    for n in range(1, 9):
        print(f"  A({n}) = {A_of_n(n):>9.5f}    "
              f"ΔA_{n} = A({n+1})−A({n}) = {finite_step(n):>9.5f}")
    print()

    # --- build the 3D object ------------------------------------
    shells = build_nested_shells(n_layers=6, nu=60, nv=60,
                                 warp=0.18, mod_amp=0.22,
                                 mod_freq=3.0)
    tower  = build_scaling_tower(n_layers=8, nu=22, nv=22,
                                 warp=0.12)

    # --- figure --------------------------------------------------
    fig = plt.figure(figsize=(16, 10))

    # (a) scaling curve and finite-step derivative
    ax0 = fig.add_subplot(2, 2, 1)
    plot_scaling_curve(ax0)

    # (b) nested organic shells
    ax1 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_nested_shells(ax1, shells, alpha=0.16)

    # (c) single shell  n = 4  →  A = 6.511,  ΔA ≈ 3.47
    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    plot_single_shell(ax2, shells[3])

    # (d) scaling tower — stacked warped 2D patches
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    plot_scaling_tower(ax3, tower)

    plt.suptitle(
        r"Finite-step 3D object  ·  "
        r"$A(n)=10a\sin(n^{2\circ})$  ·  "
        r"$\Delta A_n = A(n{+}1)-A(n)$  as grid scaling  ·  "
        r"organic 2D grid projected in 3D",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()