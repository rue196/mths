#!/usr/bin/env python3
"""
vortex_delta_jet_v3.py
======================

Vortex-Delta jet v3 — chirality-driven blade array + hurricane exhaust.

Replaces the UGC blade assignment with the rank-6 Levi-Civita
chirality projection from chiral.pdf:

    Π₆(Z) = Σ_σ ε(σ) Π_k Z[k, σ(k)]           (rank-6 contraction)
    χ(Z)  = arg Π₆(Z)                           (chirality angle)
    c(Z)  = 0 if cos χ ≥ 0 else 1               (quantized chirality)
    w(A,B) = |χ(A) − χ(B)| / π ∈ [0,1]          (chiral bond weight)
    S     = Σ_k (−1)^{k+1} |Π₆(Z^(k))|          (supertrace)
    m     = |S| · e^{−H},  H = −α p log p       (invariant mass)

Every blade carries its own 6×6 spinor Z_blade.  The chirality χ_blade
determines the blade pitch angle; the chiral bond weight w(A,B)
between neighbouring blades determines the flow coupling.  The
quantized chirality c ∈ {0,1} partitions the blade ring into two
populations — exactly the two-blade-type structure that a real
turbine needs (stator vs rotor phases).

Bug fixes
---------
- draw_3d spiral: math.cos/math.sin → np.cos/np.sin (arrays, not scalars)
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)                # ≈ 2.362
DENSITY    = 6.0 / (PI * PI)               # ≈ 0.6079
W1         = PI
W2         = E

# --- cone ---
H_CONE  = 3.0
W_BASE  = 1.0
R_BASE  = W_BASE / 2.0
R_TIP   = 0.08
CONCAVE_POWER = 0.6

# --- cylinder ---
N_BLADES  = 24
BLADE_H   = 0.45
BLADE_R0  = R_BASE + 0.02
BLADE_R1  = R_BASE + 0.20
CYL_Z0    = -0.55
CYL_Z1    = -0.05

# --- pistons ---
N_PISTONS = 8
OMEGA     = 2 * PI

# --- exhaust ---
EXHAUST_L = 1.6
EXHAUST_W = 0.35


# ============================================================
#  Finite-step sequence
# ============================================================
def A_of_n(n: int) -> float:
    return 10.0 * ALPHA_SYM * math.sin(math.radians(n * n))

def finite_step(n: int) -> float:
    return A_of_n(n + 1) - A_of_n(n)


# ============================================================
#  Rank-6 Levi-Civita  ·  720 terms  ·  from chiral.pdf
# ============================================================
_TERMS_6: List[Tuple[Tuple[int, ...], int]] = []
for _p in permutations(range(6)):
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _p[i] > _p[j])
    _TERMS_6.append((_p, (-1) ** _inv))


def pi_6(Z: np.ndarray) -> complex:
    """Π₆(Z) = Σ_σ ε(σ) Π_k Z[k, σ(k)]."""
    total = 0.0 + 0.0j
    for perm, sign in _TERMS_6:
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


def chirality_angle(Z: np.ndarray) -> float:
    """χ(Z) = arg Π₆(Z)."""
    Pi = pi_6(Z)
    return math.atan2(Pi.imag, Pi.real)


def quantized_chirality(Z: np.ndarray) -> int:
    """c(Z) = 0 if cos χ ≥ 0 else 1."""
    chi = chirality_angle(Z)
    return 0 if math.cos(chi) >= 0.0 else 1


def chiral_bond_weight(ZA: np.ndarray, ZB: np.ndarray) -> float:
    """w(A,B) = |χ_A − χ_B| / π, folded into [0, 1]."""
    d = abs(chirality_angle(ZA) - chirality_angle(ZB))
    if d > PI:
        d = 2 * PI - d
    return d / PI


# ============================================================
#  Supertrace / entropy / invariant mass  ·  from chiral.pdf
# ============================================================
def supertrace_from_spinors(spinors: List[np.ndarray]) -> float:
    """S = Σ_k (−1)^{k+1} |Π₆(Z^(k))|."""
    S = 0.0
    for k, Z in enumerate(spinors):
        sign = 1.0 if (k % 2 == 0) else -1.0
        S += sign * abs(pi_6(Z))
    return float(S)


def entropy(S: float, N: int, alpha: float = ALPHA_SYM) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def invariant_mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  Chiral spinor per blade
# ============================================================
def make_blade_spinor(idx: int, n_blades: int,
                      seed: int = 0) -> np.ndarray:
    """
    A 6×6 complex spinor for blade `idx`.

    Uses a deterministic hash of the index so the same blade always
    gets the same spinor.  The spinor is centred so Π₆ is well-defined
    (no rank deficiency from a mean offset).
    """
    rng = np.random.default_rng(seed * 1000 + idx)
    Z = (rng.standard_normal((6, 6))
         + 1j * rng.standard_normal((6, 6)))
    Z = Z / (np.max(np.abs(Z)) + 1e-12) * 0.5
    Z = Z - Z.mean()
    # rotation offset around the ring (deterministic)
    theta = 2 * PI * idx / n_blades
    return Z * np.exp(1j * theta)


# ============================================================
#  Concave cone wall profile
# ============================================================
def r_wall(z):
    z = np.asarray(z, dtype=float)
    zn = np.clip(z / H_CONE, 0.0, 1.0)
    return R_TIP + (R_BASE - R_TIP) * (1.0 - zn ** 2) ** CONCAVE_POWER


def make_12_vertex(t: float) -> np.ndarray:
    """12×6 complex configuration at time t."""
    Z = np.zeros((12, 6), dtype=complex)
    for k in range(12):
        apex = k >= 6
        theta = 2 * PI * (k % 6) / 6
        z = H_CONE * (0.95 if apex else 0.05)
        r = r_wall(z)
        idx = 1 + np.arange(6) * 0.3
        Z[k, :] = (r ** idx) * np.exp(1j * ALPHA_SYM * t * (z + 0.1 * idx)) \
                  * np.exp(1j * theta)
    return Z


def pi_12_6(Z12: np.ndarray) -> complex:
    return pi_6(Z12[0:6, :]) * pi_6(Z12[6:12, :])


# ============================================================
#  Chirality-driven blade array
# ============================================================
@dataclass
class BladeConfig:
    """One blade on the cylindrical turbine, chirality-driven."""
    angle: float
    pitch: float
    length: float
    chord: float
    thickness: float
    chi: float
    c: int
    Z: np.ndarray


class ChiralBladeArray:
    """
    Chirality-driven blade array.

    Each blade carries a 6×6 spinor Z_blade.  The rank-6 projection
    Π₆(Z_blade) gives its chirality angle χ, which maps linearly to
    the blade pitch:

        pitch = pitch_max · (1 − |χ| / π)   (aligned to the flow)
        c     = 0 if cos χ ≥ 0 else 1       (two populations)

    The chiral bond weight between adjacent blades determines the
    flow coupling at each interface.

    Properties exposed:
        blades            list of BladeConfig
        chiralities       (N,) χ_i
        quantized         (N,) c_i ∈ {0,1}
        bond_weights      (N,) w(i, i+1) ∈ [0,1]
        supertrace        S over the blade spinors
        invariant_mass    m = |S| · e^{−H}
        c0, c1            population sizes
    """
    def __init__(self, N_blades: int = N_BLADES,
                 pitch_max: float = PI / 6,
                 seed: int = 0):
        self.N = N_blades
        self.pitch_max = pitch_max
        self.seed = seed

        # --- blade angles around the cylinder ---
        self.angles = np.linspace(0, 2 * PI, N_blades, endpoint=False)

        # --- spinors per blade ---
        self.spinors: List[np.ndarray] = [
            make_blade_spinor(i, N_blades, seed) for i in range(N_blades)
        ]

        # --- chirality per blade ---
        self.chiralities = np.array(
            [chirality_angle(Z) for Z in self.spinors])
        self.quantized = np.array(
            [quantized_chirality(Z) for Z in self.spinors])

        # --- chiral bond weights between adjacent blades ---
        self.bond_weights = np.array([
            chiral_bond_weight(self.spinors[i],
                               self.spinors[(i + 1) % N_blades])
            for i in range(N_blades)
        ])

        # --- supertrace and mass over the blade spinors ---
        self.supertrace = supertrace_from_spinors(self.spinors)
        self.mass = invariant_mass(self.supertrace, N_blades)
        self.H = entropy(self.supertrace, N_blades)

        # --- blade configurations ---
        self.blades: List[BladeConfig] = []
        for i, (ang, chi, c, Z) in enumerate(
                zip(self.angles, self.chiralities,
                    self.quantized, self.spinors)):
            # pitch: aligned blades get higher pitch, anti-aligned lower
            pitch = pitch_max * (1.0 - abs(chi) / PI)
            # length modulated by the finite-step sequence
            n = (i % 8) + 1
            L = BLADE_R1 - BLADE_R0
            L *= 0.5 + 0.5 * abs(finite_step(n)) / max(
                abs(finite_step(4)), 1e-9)
            self.blades.append(BladeConfig(
                angle=float(ang), pitch=float(pitch),
                length=float(L), chord=0.06, thickness=0.008,
                chi=float(chi), c=int(c), Z=Z,
            ))

    def surface_area(self) -> float:
        return sum(2.0 * b.length * b.chord for b in self.blades)

    def population(self) -> Dict:
        c0 = int(np.sum(self.quantized == 0))
        c1 = int(np.sum(self.quantized == 1))
        return dict(
            c0=c0, c1=c1,
            n_blades=self.N,                 # ← was `N=self.N`
            imbalance=abs(c0 - c1),
            imbalance_bound=self.N,
        )
    
    def summary(self) -> Dict:
        return dict(
            N=self.N,
            supertrace=self.supertrace,
            entropy=self.H,
            invariant_mass=self.mass,
            bond_weight_mean=float(self.bond_weights.mean()),
            bond_weight_max=float(self.bond_weights.max()),
            chiral_bonds=int(np.sum(self.bond_weights > 0.5)),
            surface_area=self.surface_area(),
            **self.population(),
        )


# ============================================================
#  Hurricane exhaust  ·  M-matrix spiral
# ============================================================
def hurricane_exhaust_field(x_grid, y_grid, t: float = 0.0,
                            N_eye: float = 1.0) -> Dict:
    R = np.sqrt(x_grid ** 2 + y_grid ** 2)
    R_safe = np.maximum(R, 0.2)
    Theta = np.arctan2(y_grid, x_grid)

    p_alg = -N_eye / (1.0 + R_safe ** 2)
    omega = ALPHA_SYM * 0.6
    phase = ALPHA_SYM * R_safe - omega * t
    p_trans = (-N_eye * np.sin(phase) * np.exp(-R_safe * 0.3)
               / (1.0 + R_safe ** 2))

    R_jet   = 0.6
    R_decay = 0.4
    v_r = -N_eye * (1.0 - R_safe / R_jet) * np.exp(-R_safe / R_decay)

    R_rot = 0.5
    v_theta = ALPHA_SYM * N_eye / R_safe * np.exp(-R_safe / R_rot)

    u = v_r * np.cos(Theta) - v_theta * np.sin(Theta)
    v = v_r * np.sin(Theta) + v_theta * np.cos(Theta)

    return dict(p_total=p_alg + p_trans, u=u, v=v,
                R=R, Theta=Theta, v_r=v_r, v_theta=v_theta)


# ============================================================
#  Pistons
# ============================================================
def piston_state(t: float):
    n = max(1, int(round(t)) + 1)
    amp = A_of_n(n)
    phi = OMEGA * t + 2 * np.pi * np.arange(N_PISTONS) / N_PISTONS
    return amp, phi


# ============================================================
#  Rendering  ·  2D
# ============================================================
def draw_2d(ax, t: float, blades: ChiralBladeArray):
    ax.clear()

    # --- concave cone outline ---
    z_vals = np.linspace(0, H_CONE, 200)
    r_vals = r_wall(z_vals)
    ax.plot(r_vals, z_vals, "k-", lw=2.0)
    ax.plot(-r_vals, z_vals, "k-", lw=2.0)
    ax.plot([-R_BASE, R_BASE], [0, 0], "k-", lw=2.0)
    ax.plot([-r_vals[-1], r_vals[-1]], [H_CONE, H_CONE], "k-", lw=2.0)

    # --- low-P chamber at apex ---
    ax.fill_between([-R_BASE, R_BASE], H_CONE, H_CONE + 0.25,
                    color="#3498db", alpha=0.35)
    ax.text(0, H_CONE + 0.12, "low-P", ha="center",
            fontsize=8, color="#2980b9")

    # --- cylinder housing ---
    ax.add_patch(Rectangle((BLADE_R0 - 0.03, CYL_Z0),
                           0.06, CYL_Z1 - CYL_Z0,
                           color="#34495e"))
    ax.add_patch(Rectangle((-BLADE_R0 + 0.03 - 0.06, CYL_Z0),
                           0.06, CYL_Z1 - CYL_Z0,
                           color="#34495e"))

    # --- chiral blades: color by quantized chirality c ∈ {0,1} ---
    cmap_c = {0: "#16a085", 1: "#e74c3c"}
    blade_z = CYL_Z0 + (CYL_Z1 - CYL_Z0) * 0.5
    for b in blades.blades:
        x0 = math.cos(b.angle) * BLADE_R0
        x1 = math.cos(b.angle) * (BLADE_R0 + b.length)
        ax.plot([x0, x1], [blade_z, blade_z],
                color=cmap_c[b.c], lw=2.0)
        # pitch marker
        dx = -math.sin(b.angle) * 0.05 * math.sin(b.pitch)
        dz = 0.05 * math.cos(b.pitch)
        ax.plot([x1, x1 + dx], [blade_z, blade_z + dz],
                color="#27ae60", lw=1.0)

    # --- high-P chamber ---
    ax.fill_between([-R_BASE, R_BASE], CYL_Z0 - 0.15, CYL_Z0,
                    color="#e74c3c", alpha=0.35)
    ax.text(0, CYL_Z0 - 0.08, "high-P", ha="center",
            fontsize=8, color="#c0392b")

    # --- pistons ---
    amp, phases = piston_state(t)
    for phi in phases:
        x = 1.10 * math.cos(phi)
        y = CYL_Z1 + 0.15
        ax.plot([x, x], [CYL_Z1, y], "k-", lw=1.2)
        ax.add_patch(Rectangle((x - 0.05, y - 0.04), 0.10, 0.08,
                               color="#7f8c8d", ec="k"))

    # --- hurricane exhaust spiral ---
    z_ex = np.linspace(CYL_Z0 - 0.15, CYL_Z0 - 0.15 - EXHAUST_L, 60)
    for k in range(7):
        phase = 2 * PI * k / 7 + OMEGA * t
        spiral_x = EXHAUST_W * math.cos(phase) * (1 - np.linspace(0, 1, 60))
        ax.plot(spiral_x, z_ex, color="#8e44ad", lw=0.8, alpha=0.7)
    ax.annotate("", xy=(0, CYL_Z0 - 0.15 - EXHAUST_L - 0.10),
                xytext=(0, CYL_Z0 - 0.15 - EXHAUST_L + 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#8e44ad",
                                lw=2.4))
    ax.text(0.06, CYL_Z0 - 0.15 - EXHAUST_L - 0.10,
            "jet exhaust", fontsize=8, color="#8e44ad")

    # --- thrust arrow ---
    ax.annotate("", xy=(0, H_CONE + 0.55),
                xytext=(0, H_CONE + 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#16a085",
                                lw=2.4))

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(CYL_Z0 - 0.15 - EXHAUST_L - 0.35, H_CONE + 0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("r"); ax.set_ylabel("z")
    ax.set_title(f"2D cross-section  ·  t = {t:.2f}")
    ax.grid(alpha=0.3)


# ============================================================
#  Rendering  ·  3D   (bug-fixed: np.cos / np.sin for arrays)
# ============================================================
def draw_3d(ax, t: float, blades: ChiralBladeArray):
    ax.clear()

    # --- cone surface ---
    th = np.linspace(0, 2 * PI, 40)
    zz = np.linspace(0, H_CONE, 30)
    TH, Z = np.meshgrid(th, zz)
    R = r_wall(Z)
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    ax.plot_surface(X, Y, Z, color="#3498db", alpha=0.35,
                    linewidth=0, antialiased=True)

    # --- cylinder housing ---
    zc = np.linspace(CYL_Z0, CYL_Z1, 10)
    th_c = np.linspace(0, 2 * PI, 40)
    TH_c, Z_c = np.meshgrid(th_c, zc)
    R_cyl = BLADE_R0 + 0.03
    X_c = R_cyl * np.cos(TH_c)
    Y_c = R_cyl * np.sin(TH_c)
    ax.plot_surface(X_c, Y_c, Z_c, color="#34495e", alpha=0.35,
                    linewidth=0, antialiased=True)

    # --- chiral blades: colour by c ∈ {0,1} ---
    cmap_c = {0: "#16a085", 1: "#e74c3c"}
    for b in blades.blades:
        ang = b.angle
        z0 = CYL_Z0 + (CYL_Z1 - CYL_Z0) * 0.5
        r0 = BLADE_R0
        r1 = BLADE_R0 + b.length
        x0, y0 = r0 * math.cos(ang), r0 * math.sin(ang)
        x1, y1 = r1 * math.cos(ang), r1 * math.sin(ang)
        z1 = z0 + 0.10 * math.sin(b.pitch)
        for dz in (-b.thickness, b.thickness):
            ax.plot([x0, x1], [y0, y1],
                    [z0 + dz, z1 + dz],
                    color=cmap_c[b.c], lw=1.5)

    # --- hurricane exhaust spiral (BUG FIX: np.cos / np.sin) ---
    for k in range(6):
        phase = 2 * PI * k / 6 + OMEGA * t
        tt = np.linspace(0, 1, 40)
        r_spiral = EXHAUST_W * (1 - tt)
        z_spiral = CYL_Z0 - 0.15 - EXHAUST_L * tt
        xs = r_spiral * np.cos(phase + 2 * PI * tt)
        ys = r_spiral * np.sin(phase + 2 * PI * tt)
        ax.plot(xs, ys, z_spiral, color="#8e44ad", lw=1.0, alpha=0.8)

    # --- 12-vertex cone topology ---
    Z12 = make_12_vertex(t)
    for k in range(12):
        apex = k >= 6
        theta = 2 * PI * (k % 6) / 6
        zk = H_CONE * (0.95 if apex else 0.05)
        rk = r_wall(zk)
        ax.scatter([rk * math.cos(theta)], [rk * math.sin(theta)], [zk],
                   color="#f1c40f" if apex else "#16a085",
                   s=40, edgecolors="k", zorder=5)

    pi_val = abs(pi_12_6(Z12))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(CYL_Z0 - 0.15 - EXHAUST_L - 0.10, H_CONE * 1.05)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"3D assembly  ·  |Π(12,6)| = {pi_val:.3e}")
    ax.view_init(elev=15, azim=-55)


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 82)
    print("Vortex-Delta jet v3  ·  chirality-driven blades + hurricane exhaust")
    print("=" * 82)
    print(f"  α_sym = 1/(π − e)  = {ALPHA_SYM:.6f}")
    print(f"  cone H : width     = {H_CONE} : {W_BASE}  = "
          f"{H_CONE/W_BASE:.1f} : 1")
    print(f"  concave power      = {CONCAVE_POWER}")
    print(f"  N_blades (chiral)  = {N_BLADES}")
    print(f"  N_pistons          = {N_PISTONS}")
    print(f"  exhaust length     = {EXHAUST_L}")
    print()
    print("Chirality invariants (from chiral.pdf):")
    print(f"  Π₆(Z)               rank-6 Levi-Civita contraction")
    print(f"  χ(Z)                arg Π₆(Z)                   ∈ (−π, π]")
    print(f"  c(Z)                0 if cos χ ≥ 0 else 1      ∈ {{0,1}}")
    print(f"  w(A,B)              |χ_A − χ_B| / π            ∈ [0,1]")
    print(f"  S = Σ (−1)^(k+1)|Π₆|    supertrace over spinors")
    print(f"  m = |S| e^(−H)          invariant mass")
    print()

    # ---- chiral blade array ----
    t0 = time.perf_counter()
    blades = ChiralBladeArray(N_blades=N_BLADES, seed=0)
    t_blades = (time.perf_counter() - t0) * 1e3
    s = blades.summary()
    print(f"[1] chiral blade array     {t_blades:.2f} ms")
    print(f"      N_blades             = {s['N']}")
    print(f"      c = 0 population     = {s['c0']}  (green blades)")
    print(f"      c = 1 population     = {s['c1']}  (red blades)")
    print(f"      |c0 − c1|            = {s['imbalance']}   "
          f"(bound = {s['imbalance_bound']})")
    print(f"      supertrace S         = {s['supertrace']:+.6e}")
    print(f"      entropy H            = {s['entropy']:.6f}")
    print(f"      invariant mass m     = {s['invariant_mass']:.6e}")
    print(f"      mean bond weight     = {s['bond_weight_mean']:.4f}")
    print(f"      max  bond weight     = {s['bond_weight_max']:.4f}")
    print(f"      chiral bonds (w>0.5) = {s['chiral_bonds']} / {s['N']}")
    print(f"      surface area         = {s['surface_area']:.4f}")
    print()

    # ---- 12-vertex topology ----
    t_vals = np.linspace(0, 12, 80)
    pi_vals = np.array([abs(pi_12_6(make_12_vertex(t))) for t in t_vals])
    print(f"[2] Π(12, 6) range         = [{pi_vals.min():.3e}, "
          f"{pi_vals.max():.3e}]")
    print()

    # ---- hurricane exhaust snapshot ----
    xg = np.linspace(-EXHAUST_W * 1.5, EXHAUST_W * 1.5, 80)
    yg = np.linspace(-EXHAUST_W * 1.5, EXHAUST_W * 1.5, 80)
    Xg, Yg = np.meshgrid(xg, yg)
    exh = hurricane_exhaust_field(Xg, Yg, t=0.0, N_eye=1.0)
    print(f"[3] hurricane exhaust")
    print(f"      p_total range        = [{exh['p_total'].min():.4f}, "
          f"{exh['p_total'].max():.4f}]")
    print(f"      v_r range            = [{exh['v_r'].min():.4f}, "
          f"{exh['v_r'].max():.4f}]")
    print(f"      v_θ range            = [{exh['v_theta'].min():.4f}, "
          f"{exh['v_theta'].max():.4f}]")
    print()

    # ---- figure ----
    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # (a) 2D cross-section
    ax1 = fig.add_subplot(gs[0, 0])
    draw_2d(ax1, t=2.0, blades=blades)

    # (b) 3D assembly
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    draw_3d(ax2, t=2.0, blades=blades)

    # (c) blades polar view, coloured by chirality
    ax3 = fig.add_subplot(gs[0, 2], projection="polar")
    cmap_c = {0: "#16a085", 1: "#e74c3c"}
    for b in blades.blades:
        ax3.plot([b.angle, b.angle],
                 [BLADE_R0, BLADE_R0 + b.length],
                 color=cmap_c[b.c], lw=2.0)
        ax3.text(b.angle, BLADE_R0 + b.length + 0.03,
                 f"{b.c}",
                 ha="center", fontsize=8, color=cmap_c[b.c])
    ax3.set_title(f"Chiral blade array  ·  {N_BLADES} stations\n"
                  f"c₀ = {s['c0']}   c₁ = {s['c1']}   "
                  f"m = {s['invariant_mass']:.2e}",
                  fontsize=9)
    ax3.set_ylim(0, BLADE_R1 + 0.15)

    # (d) chirality angle distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(blades.chiralities, bins=24, range=(-PI, PI),
             color="#8e44ad", edgecolor="k", alpha=0.85)
    ax4.axvline(-PI / 2, color="k", ls=":", lw=0.8)
    ax4.axvline(PI / 2, color="k", ls=":", lw=0.8)
    ax4.set_xlabel("χ = arg Π₆(Z)")
    ax4.set_ylabel("count")
    ax4.set_title("Chirality angle distribution\n"
                  "(left of |χ|=π/2 → c=0, right → c=1)")
    ax4.grid(alpha=0.3)

    # (e) hurricane exhaust field
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(exh["p_total"], origin="lower",
                    extent=[xg.min(), xg.max(), yg.min(), yg.max()],
                    cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax5.streamplot(xg, yg, exh["u"], exh["v"],
                   density=1.2, color="k", linewidth=0.5,
                   arrowsize=0.7)
    ax5.set_xlabel("x"); ax5.set_ylabel("y")
    ax5.set_title("Hurricane exhaust flow  ·  p + streamlines")
    ax5.set_aspect("equal")
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    # (f) thrust proxy
    ax6 = fig.add_subplot(gs[1, 2])
    z_grid = np.linspace(0.01, H_CONE * 0.99, 200)
    S_thrust = np.zeros_like(z_grid)
    for t_idx in range(1, 9):
        S_thrust += ((1.0 if t_idx % 2 == 1 else -1.0)
                     * (r_wall(z_grid) ** t_idx)
                     * np.cos(ALPHA_SYM * t_idx * z_grid))
    thrust = np.abs(S_thrust)
    ax6.plot(z_grid, thrust, color="#16a085", lw=1.6)
    ax6.fill_between(z_grid, 0, thrust, color="#16a085", alpha=0.2)
    ax6.set_xlabel("z"); ax6.set_ylabel("|S(z)|")
    ax6.set_title("Thrust proxy along the cone axis")
    ax6.grid(alpha=0.3)

    plt.suptitle(
        "Vortex-Delta jet v3  ·  concave cone  ·  chiral blade array "
        "(rank-6 Levi-Civita)  ·  hurricane exhaust",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()