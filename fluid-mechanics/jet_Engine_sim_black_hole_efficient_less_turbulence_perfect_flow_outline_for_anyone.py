#!/usr/bin/env python3
"""
vortex_delta_jet.py
===================

Vortex-Delta jet engine — design mockup.

Layout
------
              ╱╲         ← apex (LOW-P chamber)
             ╱  ╲
            ╱    ╲        concave walls
           ╱      ╲       r(z) = R_tip + (R0-R_tip)(1-(z/H)²)^0.6
          ╱        ╲
         ╱          ╲
        ╱            ╲    HIGH-P chamber (surrounds the base)
       ╱______________╲
       │ ▓ │ ▓ │ ▓ │ ▓ │   ← 8 pistons
       ╰───┴───┴───┴───╯

  • H : width = 3 : 1            (tall, protruding cone)
  • concave wall profile         (focuses the flow)
  • high-P chamber at base       (hurricane-like pressure well)
  • low-P chamber at apex        (receiving chamber)
  • pressure releases along the concave walls → thrust

Topology
--------
12 vertices: 6 at the base ring, 6 at the apex ring.
Each vertex carries a 6-vector.  The invariant scalar is

    Π(12, 6) = Π_6(Z_base) · Π_6(Z_apex)

with Π_6 the rank-6 Levi-Civita contraction (720 terms).

Flow
----
The M-matrix at radius z and time t is

    M(t, z) = r(z)^t · exp(i α t z)             α = 1/(π − e)

The supertrace S(z) = Σ_t (−1)^{t-1} Re M(t, z) gives the pressure
field along the axis.  Navier-Stokes viscosity enters as the
exponential diffusion kernel exp(−α|Δz|).

Pistons
-------
8 pistons at the base ring, phase 2πk/8.  Piston amplitude follows
the finite-step sequence  A(n) = 10 a sin(n²°).  The pistons
synchronize with the cone's natural harmonic, modulating the
pressure release into the low-P chamber.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from itertools import permutations
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
#  Constants
# ============================================================
PI     = math.pi
E      = math.e
ALPHA  = 1.0 / (PI - E)               # ≈ 2.362

# Cone geometry: H : width = 3 : 1
H_CONE = 3.0                          # height
W_BASE = 1.0                          # base width
R_BASE = W_BASE / 2.0                 # base radius = 0.5
R_TIP  = 0.08                         # small apex opening
CONCAVE_POWER = 0.6                   # wall profile exponent

# Pistons
N_PISTONS = 8
OMEGA     = 2 * PI                        # piston base frequency


# ============================================================
#  Finite-step sequence  ·  A(n) = 10 a sin(n²°)
# ============================================================
def A_of_n(n: int) -> float:
    return 10.0 * ALPHA * math.sin(math.radians(n * n))

def finite_step(n: int) -> float:
    return A_of_n(n + 1) - A_of_n(n)


# ============================================================
#  Concave cone wall profile
# ============================================================
def r_wall(z):
    """r(z) = R_tip + (R_base − R_tip) · (1 − (z/H)²)^p."""
    z = np.asarray(z, dtype=float)
    zn = np.clip(z / H_CONE, 0.0, 1.0)
    return R_TIP + (R_BASE - R_TIP) * (1.0 - zn**2) ** CONCAVE_POWER


# ============================================================
#  Rank-6 Levi-Civita  ·  720 terms
# ============================================================
_TERMS_6 = []
for _p in permutations(range(6)):
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _p[i] > _p[j])
    _TERMS_6.append((_p, (-1) ** _inv))


def pi_6(Z: np.ndarray) -> complex:
    total = 0.0 + 0.0j
    for perm, sign in _TERMS_6:
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


def pi_12_6(Z12: np.ndarray) -> complex:
    """Π(12, 6) = Π_6(Z_base) · Π_6(Z_apex)."""
    return pi_6(Z12[0:6, :]) * pi_6(Z12[6:12, :])


def make_12_vertex(t: float) -> np.ndarray:
    """12 × 6 complex configuration at time t."""
    Z = np.zeros((12, 6), dtype=complex)
    for k in range(12):
        apex = k >= 6
        theta = 2 * PI * (k % 6) / 6
        z = H_CONE * (0.95 if apex else 0.05)
        r = r_wall(z)
        idx = 1 + np.arange(6) * 0.3
        Z[k, :] = (r ** idx) * np.exp(1j * ALPHA * t * (z + 0.1 * idx)) \
                  * np.exp(1j * theta)
    return Z


# ============================================================
#  M-matrix field  ·  pressure + flow
# ============================================================
def m_matrix(z, t):
    """M(t, z) = r(z)^t · exp(i α t z)."""
    return (r_wall(z) ** t) * np.exp(1j * ALPHA * t * z)


def pressure_field(z_grid, t_max: int = 8) -> np.ndarray:
    """Supertrace pressure S(z) = Σ_t (−1)^(t−1) Re M(t, z)."""
    S = np.zeros_like(z_grid, dtype=float)
    for t in range(1, t_max + 1):
        S += (1.0 if t % 2 == 1 else -1.0) * np.real(m_matrix(z_grid, t))
    return S


def ns_kernel(width: int = 15) -> np.ndarray:
    """ diffusion kernel exp(−α|Δz|), normalised."""
    dz = np.arange(-width // 2, width // 2)
    k = np.exp(-ALPHA * np.abs(dz))
    return k / k.sum()


def smooth(S: np.ndarray) -> np.ndarray:
    return np.convolve(S, ns_kernel(), mode="same")


# ============================================================
#  Piston harmonics
# ============================================================
def piston_state(t: float):
    """Amplitude, phases of the N pistons at time t."""
    n = max(1, int(round(t)) + 1)
    amp = A_of_n(n)
    phi = OMEGA * t + 2 * PI * np.arange(N_PISTONS) / N_PISTONS
    return amp, phi


# ============================================================
#  2D cross-section
# ============================================================
def draw_2d(ax, t: float):
    ax.clear()
    z_vals = np.linspace(0, H_CONE, 240)
    r_vals = r_wall(z_vals)

    # high-P chamber (base side, drawn as vertical band above the base)
    ax.fill_between([-R_BASE, R_BASE], -0.35, 0.0,
                    color="#e74c3c", alpha=0.35,
                    label="high-P chamber")
    # low-P chamber (apex side)
    ax.fill_between([-R_BASE, R_BASE], H_CONE, H_CONE + 0.35,
                    color="#3498db", alpha=0.35,
                    label="low-P chamber")

    # concave cone outline
    ax.plot(r_vals, z_vals, "k-", lw=2.0)
    ax.plot(-r_vals, z_vals, "k-", lw=2.0)
    ax.plot([-R_BASE, R_BASE], [0, 0], "k-", lw=2.0)
    ax.plot([-r_vals[-1], r_vals[-1]],
            [H_CONE, H_CONE], "k-", lw=2.0)

    # pressure along the axis (shifted right for clarity)
    S = smooth(pressure_field(z_vals, t_max=8))
    S_n = S / (np.max(np.abs(S)) + 1e-12)
    ax.plot(0.35 + 0.30 * S_n, z_vals, "r-", lw=1.8,
            label="S(z)")

    # pistons
    amp, phases = piston_state(t)
    for phi in phases:
        x = 1.10 * math.cos(phi)
        y = -0.20 - 0.15 * math.sin(phi)
        ax.plot([x, x], [-0.35, y], "k-", lw=1.2)
        ax.add_patch(Rectangle((x - 0.07, y - 0.06), 0.14, 0.12,
                                color="#7f8c8d", ec="k"))
    ax.text(0, -0.55, f"pistons  ·  amp = {amp:.3f}",
            ha="center", fontsize=8, color="#555")

    # thrust arrow
    ax.annotate("", xy=(0, H_CONE + 0.55),
                xytext=(0, H_CONE + 0.30),
                arrowprops=dict(arrowstyle="-|>", color="#16a085",
                                lw=2.6))
    ax.text(0.06, H_CONE + 0.45, "thrust", fontsize=9,
            color="#16a085")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.75, H_CONE + 0.70)
    ax.set_aspect("equal")
    ax.set_xlabel("r"); ax.set_ylabel("z")
    ax.set_title(f"2D cross-section  ·  t = {t:.2f}")
    ax.legend(fontsize=7, loc="center left")
    ax.grid(alpha=0.3)


# ============================================================
#  3D cone
# ============================================================
def draw_3d(ax, t: float):
    ax.clear()
    th = np.linspace(0, 2 * PI, 60)
    zz = np.linspace(0, H_CONE, 40)
    TH, Z = np.meshgrid(th, zz)
    R = r_wall(Z)
    X = R * np.cos(TH)
    Y = R * np.sin(TH)

    # colour by pressure field
    S = smooth(pressure_field(Z.ravel(), t_max=8))
    S_n = (S - S.min()) / (np.ptp(S) + 1e-12)
    C = S_n.reshape(Z.shape)
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r(C),
                    alpha=0.85, linewidth=0, antialiased=True,
                    rstride=2, cstride=2)

    # 12 vertices
    Z12 = make_12_vertex(t)
    for k in range(12):
        apex = k >= 6
        theta = 2 * PI * (k % 6) / 6
        zk = H_CONE * (0.95 if apex else 0.05)
        rk = r_wall(zk)
        ax.scatter([rk * math.cos(theta)], [rk * math.sin(theta)],
                   [zk],
                   color="#f1c40f" if apex else "#16a085",
                   s=45, edgecolors="k")

    pi_val = abs(pi_12_6(Z12))
    ax.set_xlim(-R_BASE * 1.3, R_BASE * 1.3)
    ax.set_ylim(-R_BASE * 1.3, R_BASE * 1.3)
    ax.set_zlim(0, H_CONE * 1.1)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"3D concave cone  ·  |Π(12, 6)| = {pi_val:.4e}")
    ax.view_init(elev=22, azim=-55)


# ============================================================
#  Demo figure
# ============================================================
def main():
    print("=" * 78)
    print("Vortex-Delta jet engine  ·  design mockup")
    print("=" * 78)
    print(f"  α              = 1/(π − e)  = {ALPHA:.6f}")
    print(f"  cone           H : width   = {H_CONE:.1f} : {W_BASE:.1f}  "
          f"= {H_CONE / W_BASE:.1f} : 1")
    print(f"  concave wall   exponent    = {CONCAVE_POWER}")
    print(f"  apex opening   R_tip       = {R_TIP}")
    print(f"  pistons        N           = {N_PISTONS}")
    print(f"  invariant      Π(12, 6)    from the rank-6 Levi-Civita")
    print(f"  flow           M(t,z)      = r(z)^t · exp(i α t z)")
    print(f"  viscosity      Diff. = exp(−α|Δz|)")
    print()

    # ---- invariant scalar trace ----
    t_vals = np.linspace(0.0, 12.0, 80)
    pi_vals = np.array([abs(pi_12_6(make_12_vertex(t))) for t in t_vals])
    print(f"  |Π(12, 6)| range  = "
          f"[{pi_vals.min():.3e}, {pi_vals.max():.3e}]")

    # ---- pressure along the axis ----
    z_grid = np.linspace(0.01, H_CONE * 0.99, 300)
    S = smooth(pressure_field(z_grid, t_max=8))
    print(f"  S(z) range        = "
          f"[{S.min():.4e}, {S.max():.4e}]")
    print(f"  apex S(z=H)       = {S[-1]:.4e}")
    print(f"  base S(z=0)       = {S[0]:.4e}")

    # ---- pistons ----
    amp, phases = piston_state(t=2.0)
    print(f"  piston amp at t=2 = {amp:.4f}   "
          f"phase spread = {phases.max() - phases.min():.3f} rad")
    print()

    # ---- figure ----
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # (a) 2D cross-section
    ax1 = fig.add_subplot(gs[0, 0])
    draw_2d(ax1, t=2.0)

    # (b) 3D cone
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    draw_3d(ax2, t=2.0)

    # (c) pressure along the axis at several t
    ax3 = fig.add_subplot(gs[0, 2])
    for t in (1, 3, 5, 7):
        ax3.plot(z_grid, pressure_field(z_grid, t_max=t),
                 lw=1.4, label=f"t_max = {t}")
    ax3.axvline(H_CONE, color="k", ls=":", label="apex")
    ax3.set_xlabel("z"); ax3.set_ylabel("S(z)")
    ax3.set_title("Pressure field along the cone axis")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # (d) invariant scalar trace
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_vals, pi_vals, color="#8e44ad", lw=1.6)
    ax4.set_xlabel("t"); ax4.set_ylabel("|Π(12, 6)|")
    ax4.set_title("Invariant scalar of the 12-vertex topology")
    ax4.grid(alpha=0.3)

    # (e) piston harmonics
    ax5 = fig.add_subplot(gs[1, 1])
    ts = np.linspace(0, 12, 500)
    for k in range(N_PISTONS):
        amps = np.array([A_of_n(max(1, int(round(t)) + 1)) for t in ts])
        phis = OMEGA * ts + 2 * PI * k / N_PISTONS
        ax5.plot(ts, amps * np.sin(phis), lw=1.0, alpha=0.85,
                 label=f"piston {k}")
    ax5.set_xlabel("t"); ax5.set_ylabel("displacement")
    ax5.set_title(f"{N_PISTONS} pistons  ·  harmonic oscillation")
    ax5.legend(fontsize=7, ncol=4); ax5.grid(alpha=0.3)

    # (f) thrust proxy
    ax6 = fig.add_subplot(gs[1, 2])
    thrust = np.abs(S)
    ax6.plot(z_grid, thrust, color="#16a085", lw=1.6)
    ax6.fill_between(z_grid, 0, thrust, color="#16a085", alpha=0.20)
    ax6.set_xlabel("z"); ax6.set_ylabel("|S(z)|  thrust proxy")
    ax6.set_title("Thrust estimate along the cone axis")
    ax6.grid(alpha=0.3)

    plt.suptitle(
        "Vortex-Delta jet engine  ·  3:1 concave cone  ·  "
        "8 harmonic pistons  ·  Π(12, 6) topology  ·  "
        "M-matrix pressure  ·  Diffusion kernel",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================
#  Animation  ·  2D cross-section over one harmonic cycle
# ============================================================
def animate():
    fig, ax = plt.subplots(figsize=(7, 9))

    def update(frame):
        t = frame * 0.15
        draw_2d(ax, t)
        return ax,

    anim = FuncAnimation(fig, update, frames=80, interval=80,
                         blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == "__main__":
    main()
    # Uncomment to run the animation:
    # animate()