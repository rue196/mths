#!/usr/bin/env python3
"""
lightweight_hyperrealistic_engine.py
=====================================

A lightweight hyper-realistic game engine built on three primitives:

    · Gravity        |a^i + e| = π,   a = π-e,   1/a = α = 1/(π-e)
    · Rendering      2D algebraic grid  →  3D via  finite step  A_STEP
                     tr M(x,y) = x^(i-1) + y^(i-1)
                               = x^{-1}(cos ln x + i sin ln x)
                               + y^{-1}(cos ln y + i sin ln y)
                     z = Re(tr M)        symmetric axis
                     |tr M|              transcendental magnitude
                     Im(tr M)            transcendental length
    · Lighting       g_ij = w_ij · κ_ij
                     w_ij = |χ_i − χ_j| / π         (chiral bond weight)
                     κ_ij = √(L_i L_j) / L_ref      (travel length)
                     even-even mask: (i,j) both even → retain
                     Π_e(t,r) = ½(1 + cos(2t)cos(2πr/e)) ∈ [0,1]
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)          # ≈ 2.3623   inverse of a
A_GRAV  = PI - E                  # ≈ 0.4233   a = π − e  (gravity coupling)
A_STEP  = ALPHA / 0.3628          # ≈ 6.511    finite-step projection
DENSITY = 6.0 / (PI * PI)         # ≈ 0.6079   μ density


# ============================================================
#  M-matrix trace  ·  algebraic ↔ transcendental
# ============================================================
def m_trace(x: float, y: float) -> complex:
    """
    tr M(x,y) = x^(i−1) + y^(i−1)
              = x⁻¹(cos ln x + i sin ln x)
              + y⁻¹(cos ln y + i sin ln y)
    """
    x = max(abs(x), 1e-9)
    y = max(abs(y), 1e-9)
    lnx, lny = math.log(x), math.log(y)
    tr_re = math.cos(lnx) / x + math.cos(lny) / y
    tr_im = math.sin(lnx) / x + math.sin(lny) / y
    return complex(tr_re, tr_im)


# ============================================================
#  Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    if K >= 1:
        mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(K ** 0.5) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    for p in range(2, K + 1):
        if is_prime[p]:
            mu[p::p] = -mu[p::p]
    for p in range(2, int(K ** 0.5) + 1):
        if is_prime[p]:
            mu[p * p :: p * p] = 0
    return mu


# ============================================================
#  Gravity  ·  a = π − e
# ============================================================
def gravity_accel(pos_i, mass_i, pos_j, mass_j):
    """a_grav = (π−e)·m_j·r̂ / r²   —  the finite-step identity |a^i+e|=π."""
    r_vec = np.asarray(pos_j) - np.asarray(pos_i)
    r = float(np.linalg.norm(r_vec)) + 1e-9
    return A_GRAV * mass_j * r_vec / (r ** 3 + 1e-6)


# ============================================================
#  2D algebraic grid  →  3D projection
# ============================================================
def project_grid(N: int, scale: float = A_STEP,
                 center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 sphere: bool = False):
    """
    Algebraic coords:
        x_i = exp(scale · (i − N/2) / N)
        y_j = exp(scale · (j − N/2) / N)
    Symmetric axis:     z_ij = tanh(Re(tr M))     (bounded)
    Transcendental:     MAG = |tr M|,   LEN = Im(tr M)
    """
    cx, cy, cz = center
    X = np.zeros((N, N)); Y = np.zeros((N, N)); Z = np.zeros((N, N))
    MAG = np.zeros((N, N)); LEN = np.zeros((N, N))

    for i in range(N):
        x_i = math.exp(scale * (i - N / 2) / N)
        for j in range(N):
            y_j = math.exp(scale * (j - N / 2) / N)
            tr = m_trace(x_i, y_j)
            if sphere:
                theta = 2 * PI * (i / max(N - 1, 1))
                phi   = PI * (j / max(N - 1, 1))
                R = 1.0 + 0.4 * math.tanh(tr.real * 0.3)
                X[i, j] = R * math.sin(phi) * math.cos(theta) + cx
                Y[i, j] = R * math.sin(phi) * math.sin(theta) + cy
                Z[i, j] = R * math.cos(phi) + cz
            else:
                X[i, j] = (i / N - 0.5) * 2.0 + cx
                Y[i, j] = (j / N - 0.5) * 2.0 + cy
                Z[i, j] = 0.3 * math.tanh(tr.real * 0.3) + cz
            MAG[i, j] = abs(tr)
            LEN[i, j] = tr.imag
    return X, Y, Z, MAG, LEN


# ============================================================
#  Lighting primitives  ·  chiral angles + travel length
# ============================================================
def chiral_angles(t):
    lt = np.log(np.maximum(np.abs(t), 1e-12))
    return (lt + PI) % (2 * PI) - PI


def travel_lengths(t, r):
    dt = np.diff(t, prepend=t[0])
    dr = np.diff(r, prepend=r[0])
    ds = np.sqrt(dt * dt + dr * dr)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-6


# ============================================================
#  Even-even mask
# ============================================================
def even_even_mask(N: int) -> np.ndarray:
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2 == 0) & (j % 2 == 0)


# ============================================================
#  Elliptic projection  ·  light cloud intensity
# ============================================================
def elliptic_projection(t, r):
    """Π_e(t, r) = ½(1 + cos(2t)·cos(2πr/e))  ∈ [0,1]   (Figure 3)."""
    return 0.5 * (1.0 + np.cos(2.0 * t) * np.cos(2.0 * PI * r / E))


# ============================================================
#  Surface  ·  rigid body with mesh + mass + velocity
# ============================================================
@dataclass
class Surface:
    name: str
    N: int = 48
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = A_STEP
    sphere: bool = False
    mass: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        (self.X, self.Y, self.Z,
         self.MAG, self.LEN) = project_grid(
             self.N, self.scale, self.center, sphere=self.sphere)

    def translate(self, delta):
        self.X += delta[0]; self.Y += delta[1]; self.Z += delta[2]

    def apply_gravity(self, other: "Surface", dt: float):
        accel = gravity_accel(self.center, self.mass,
                              other.center, other.mass)
        self.velocity += accel * dt
        delta = self.velocity * dt
        self.center = tuple(np.asarray(self.center) + delta)
        self.translate(delta)


# ============================================================
#  Light  ·  orbiting point source with Π_e intensity
# ============================================================
@dataclass
class Light:
    pos: Tuple[float, float, float]
    radius: float = 1.5
    color: Tuple[float, float, float] = (1.0, 0.95, 0.8)
    t: float = 0.0
    travel: float = 1.0
    orbit_radius: float = 0.0
    orbit_period: float = 1.0
    chi: float = field(init=False)

    def __post_init__(self):
        self.chi = float(chiral_angles(np.array([self.pos[0]]))[0])

    def update(self, dt: float) -> None:
        self.t += dt
        if self.orbit_radius > 0 and self.orbit_period > 0:
            ang = 2 * PI * (self.t / self.orbit_period)
            self.pos = (self.orbit_radius * math.cos(ang),
                        self.orbit_radius * math.sin(ang),
                        self.pos[2])
        self.chi = float(chiral_angles(np.array([self.pos[0]]))[0])


# ============================================================
#  Renderer  ·  lighting via coupling + mask + elliptic projection
# ============================================================
def compute_lighting(surface: Surface, lights: List[Light],
                     use_mask: bool = True) -> np.ndarray:
    """
    I_ij = Σ_lights  g_ij · Π_e(t, r_ij) / (1 + r_ij²)
    g_ij = w_ij · κ_ij  (chiral + travel coupling)
    """
    N = surface.N
    X, Y, Z = surface.X, surface.Y, surface.Z

    # algebraic grid for chirality + travel
    ii = np.arange(N)[:, None] * np.ones((1, N))
    jj = np.ones((N, 1)) * np.arange(N)[None, :]
    xi = np.exp(A_STEP * (ii - N / 2) / N)
    yj = np.exp(A_STEP * (jj - N / 2) / N)
    chi_surf = chiral_angles(xi.ravel()).reshape(N, N)
    L_surf = travel_lengths(xi.ravel(), yj.ravel()).reshape(N, N)
    L_ref = float(np.median(L_surf[L_surf > 0])) if np.any(L_surf > 0) else 1.0

    mask = even_even_mask(N) if use_mask else np.ones((N, N), dtype=bool)

    total = np.zeros((N, N))
    for light in lights:
        d = np.sqrt((X - light.pos[0]) ** 2 +
                    (Y - light.pos[1]) ** 2 +
                    (Z - light.pos[2]) ** 2)

        dchi = np.abs(chi_surf - light.chi)
        dchi = (dchi + PI) % (2 * PI) - PI
        w = np.abs(dchi) / PI

        kappa = np.sqrt(L_surf * light.travel) / L_ref
        kappa = np.clip(kappa, 0.0, 1.0)

        g = w * kappa
        pe = elliptic_projection(light.t, d / max(light.radius, 1e-9))
        falloff = 1.0 / (1.0 + d * d)

        total += np.where(mask, g * pe * falloff, 0.0)

    vmax = total.max()
    return total / vmax if vmax > 0 else total


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Lightweight hyper-realistic engine  ·  M-matrix + coupling lighting")
    print("=" * 76)
    print(f"  α      = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  a      = π − e       = {A_GRAV:.6f}   (gravity coupling)")
    print(f"  A_STEP               = {A_STEP:.6f}")
    print(f"  6/π²                 = {DENSITY:.6f}   (μ density)")
    print()

    # ---------- gravity identity check ----------
    ln_a = math.log(A_GRAV)
    a_i = complex(math.cos(ln_a), math.sin(ln_a))
    print("--- gravity identity  |a^i + e| ---")
    print(f"  a = π−e         = {A_GRAV:.6f}")
    print(f"  a^i             = {a_i:.6f}")
    print(f"  |a^i + e|       = {abs(a_i + E):.6f}   (thematic ≈ π)")
    print()

    # ---------- μ sieve density ----------
    mu = mobius_sieve(2048)
    live = np.count_nonzero(mu[1:]) / 2048
    print(f"--- μ sieve density ---")
    print(f"  live / K        = {live:.4f}    (target 6/π² = {DENSITY:.4f})")
    print()

    # ---------- scene ----------
    ground = Surface("ground", N=48, center=(0, 0, -0.6), mass=10.0)
    sphere = Surface("sphere", N=48, center=(0, 0, 1.2),
                     sphere=True, mass=1.0)
    print("--- scene surfaces ---")
    print(f"  ground  N={ground.N}  mass={ground.mass}")
    print(f"  sphere  N={sphere.N}  mass={sphere.mass}")
    print()

    light_A = Light(pos=(1.5, 0.0, 2.0), radius=1.0,
                    color=(1.0, 0.7, 0.3))
    light_B = Light(pos=(0.0, 1.8, 2.5), radius=1.2,
                    color=(0.3, 0.7, 1.0),
                    orbit_radius=2.0, orbit_period=4.0)
    lights = [light_A, light_B]
    print("--- lights ---")
    for L in lights:
        print(f"  pos={L.pos}  radius={L.radius}  "
              f"orbit={L.orbit_radius}  period={L.orbit_period}")
    print()

    # ---------- simulation ----------
    print("--- simulation (gravity + orbiting light) ---")
    dt = 0.05
    for step in range(60):
        for L in lights:
            L.update(dt)
        sphere.apply_gravity(ground, dt)
        I = compute_lighting(sphere, lights)
        if step % 15 == 0 or step == 59:
            print(f"  step {step:>3d}  sphere z = {sphere.center[2]:+.4f}"
                  f"  ⟨I⟩ = {I.mean():.4f}  max I = {I.max():.4f}")
    print()

    # ---------- final lighting & stats ----------
    I = compute_lighting(sphere, lights)
    mask = even_even_mask(sphere.N)
    print("--- final state ---")
    print(f"  ⟨I⟩ over grid   = {I.mean():.6f}")
    print(f"  max I           = {I.max():.6f}")
    print(f"  fraction lit    = {np.mean(I > 0.1):.4f}")
    print(f"  even-even cells = {mask.sum()} / {mask.size} = {mask.mean():.4f}")
    print()

    # ---------- visualisation ----------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)

    # (a) 3D scene coloured by lighting
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ground_col = cm.viridis(np.zeros_like(ground.Z))
    ax.plot_surface(ground.X, ground.Y, ground.Z,
                    facecolors=ground_col, alpha=0.35,
                    rstride=1, cstride=1, linewidth=0)
    sphere_col = cm.hot(I)
    ax.plot_surface(sphere.X, sphere.Y, sphere.Z,
                    facecolors=sphere_col, alpha=0.95,
                    rstride=1, cstride=1, linewidth=0)
    for L in lights:
        ax.scatter(*L.pos, color=L.color, s=120, marker="*",
                   edgecolors="k", linewidths=0.6)
    ax.set_title("Scene  ·  ground + sphere  ·  light from coupling g_ij")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1.2))

    # (b) transcendental magnitude  |tr M|
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(sphere.MAG, origin="lower", cmap="inferno")
    ax.set_title(r"Transcendental magnitude  $|tr M|$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) transcendental length  Im(tr M)
    ax = fig.add_subplot(gs[1, 0])
    m = abs(sphere.LEN).max() or 1.0
    im = ax.imshow(sphere.LEN, origin="lower", cmap="RdBu_r",
                   vmin=-m, vmax=+m)
    ax.set_title(r"Transcendental length  $\mathrm{Im}(tr M)$")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) light intensity  ·  coupling + Π_e + even-even mask
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(I, origin="lower", cmap="hot")
    ax.set_title(r"Light intensity  ·  $g_{ij}\cdot\Pi_e(t,r)$"
                 r"  ·  even-even mask")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Lightweight hyper-realistic engine  ·  "
        f"α = {ALPHA:.4f}  ·  A_STEP = {A_STEP:.4f}  ·  "
        f"g_ij = w_ij κ_ij  ·  Π_e ∈ [0,1]",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---------- summary ----------
    print("=" * 76)
    print("Engine summary")
    print("=" * 76)
    print("  Gravity        |a^i + e| = π   with  a = π − e")
    print("                 a_grav = (π−e)·m_j·r̂ / r²")
    print()
    print("  Rendering      2D algebraic grid  →  3D")
    print("                 x_i = exp(A_STEP·(i − N/2) / N)")
    print("                 y_j = exp(A_STEP·(j − N/2) / N)")
    print("                 z_ij = tanh(Re(tr M))       symmetric axis")
    print("                 MAG_ij = |tr M|             transcendental magnitude")
    print("                 LEN_ij = Im(tr M)           transcendental length")
    print()
    print("  Lighting       g_ij = w_ij · κ_ij")
    print("                 w_ij = |χ_i − χ_j| / π       chiral bond weight")
    print("                 κ_ij = √(L_i L_j) / L_ref    travel length")
    print("                 even-even mask  →  (i, j) both even")
    print("                 I_ij = Σ_lights g_ij · Π_e(t, r_ij) / (1 + r²)")
    print("                 Π_e(t, r) = ½(1 + cos 2t · cos 2πr/e)  ∈ [0,1]")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()