#!/usr/bin/env python3
"""
black_hole_boson_blob.py
========================

A black hole as a 2D blob of bosons, with the M-matrix as the
relativistic frame.

Layout
------
Boson blob      b(x, y)   2D Gaussian at the origin       → the BH
Fermion gas     f(x, y)   sparse, lives where bosons don't
M-matrix        block-diagonal in time parity:

    ┌              ┐
    │ M_BB    0    │    M_BB = boson self-energy (BH blob)
    │   0   M_FF   │    M_FF = fermion self-energy
    └              ┘    M_BF = M_FB = 0   ← no scattering

Relativity
----------
The transcendental row of M is

    exp(i · α · t · R(x, y))         α = 1/(π − e) ≈ 2.362

Interpreted as a local clock rate: at large R (far from the BH),
the phase advances fast; at small R (near the horizon / center),
the phase advances slowly.  This is the discrete shadow of
gravitational time dilation.

The algebraic row is  rho^t  with  rho = R / L ∈ (0, 1].

Supertrace
----------
S = Σ_t (−1)^t · M_t,t      (0-based; bosons at even t_idx → +1)
  = b_field · Σ_{even t} A_t − f_field · Σ_{odd t} A_t

where  A_t(x, y) = rho^t · exp(i α t R)  is the per-time kernel.

Mass
----
m = |S| · e^{−H},   H = −α · p · log p,  p = |S| / (Nx · Ny).
"""

from __future__ import annotations

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)                   # ≈ 2.362

Nx, Ny   = 128, 128
L        = 3.0
T        = 64
R_bh     = 0.70
CAP_EXP  = 6                             # cap on rho^t for stability
N_TOTAL  = Nx * Ny


# ============================================================
#  Spatial grid
# ============================================================
x = np.linspace(-L, L, Nx)
y = np.linspace(-L, L, Ny)
X, Y = np.meshgrid(x, y, indexing="ij")
R = np.sqrt(X**2 + Y**2)
R_safe = np.maximum(R, 1e-3)
rho = R_safe / L                          # ∈ (0, 1]


# ============================================================
#  Time kernel  A_t(x, y)
# ============================================================
def alg_row(t_val: int) -> np.ndarray:
    """Algebraic row:  rho^t  (capped)."""
    return rho ** min(t_val, CAP_EXP)


def trans_row(t_val: int) -> np.ndarray:
    """Transcendental row:  exp(i α t R)   relativistic phase."""
    return np.exp(1j * ALPHA * t_val * R_safe)


def kernel(t_val: int) -> np.ndarray:
    return alg_row(t_val) * trans_row(t_val)


# ---- per-time slices (T = 64) ---------------------------------
A_slices = np.zeros((T, Nx, Ny), dtype=np.complex128)
t0 = time.perf_counter()
for t_idx in range(T):
    A_slices[t_idx] = kernel(t_idx + 1)
build_ms = (time.perf_counter() - t0) * 1e3

# ---- decompose into even / odd time sums -----------------------
B_sum = np.zeros((Nx, Ny), dtype=np.complex128)   # Σ_{even t} A_t
F_sum = np.zeros((Nx, Ny), dtype=np.complex128)   # Σ_{odd  t} A_t
for t_idx in range(T):
    if t_idx % 2 == 0:
        B_sum += A_slices[t_idx]
    else:
        F_sum += A_slices[t_idx]


# ============================================================
#  Boson blob  (the black hole)
# ============================================================
def boson_field(R_blob: float) -> np.ndarray:
    """2D Gaussian boson density."""
    return np.exp(-R**2 / (2 * R_blob**2))


def fermion_field(b_field: np.ndarray) -> np.ndarray:
    """Fermion gas: suppressed where bosons are dense."""
    envelope = np.exp(-R**2 / (2 * (2 * L) ** 2))
    return 0.15 * (1.0 - b_field) * envelope


b_field = boson_field(R_bh)
f_field = fermion_field(b_field)

M_bh = float(np.sum(b_field) * (x[1] - x[0]) * (y[1] - y[0]))


# ============================================================
#  M-matrix (block diagonal) and supertrace
# ============================================================
def supertrace_field(R_blob: float
                     ) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (S_total, S_field, b_field, f_field).

        S(x, y) = b_field(x, y) · B_sum(x, y)
                − f_field(x, y) · F_sum(x, y)
    """
    b = boson_field(R_blob)
    f = fermion_field(b)
    S_field = b * B_sum - f * F_sum
    return complex(np.sum(S_field)), S_field, b, f


S_total, S_field, b_field, f_field = supertrace_field(R_bh)
S_abs = abs(S_total)


# ============================================================
#  Entropy and mass
# ============================================================
def mass_of(S_abs_val: float, N: int = N_TOTAL) -> tuple[float, float]:
    """m = |S| e^{-H},  H = -α p log p,  p = |S| / N."""
    p = S_abs_val / N
    if not (0.0 < p < 1.0):
        return 0.0, 0.0
    H = -ALPHA * p * math.log(p)
    m = S_abs_val * math.exp(-H) if H < 700 else 0.0
    return H, m


H_bh, m_bh = mass_of(S_abs)


# ============================================================
#  World lines  —  fermions passing through the BH (no scattering)
# ============================================================
n_lines = 24
theta_lines = np.linspace(0.0, 2 * PI, n_lines, endpoint=False)
line_dirs = np.column_stack([np.cos(theta_lines), np.sin(theta_lines)])


# ============================================================
#  Radial scan: supertrace and mass vs blob radius
# ============================================================
R_scan = np.linspace(0.15, 1.60, 60)
S_scan_abs = np.zeros_like(R_scan)
m_scan     = np.zeros_like(R_scan)
for i, Rb in enumerate(R_scan):
    S_tot, _, _, _ = supertrace_field(Rb)
    S_abs_i = abs(S_tot)
    S_scan_abs[i] = S_abs_i
    _, m_scan[i] = mass_of(S_abs_i)


# ============================================================
#  Visualization
# ============================================================
fig = plt.figure(figsize=(17, 12))
gs  = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

# (a) Boson blob
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(b_field.T, origin="lower",
               extent=[-L, L, -L, L], cmap="inferno")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"Boson blob  b(x, y)  ·  M_bh = {M_bh:.4f}")
plt.colorbar(im, ax=ax)

# (b) Fermion gas
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(f_field.T, origin="lower",
               extent=[-L, L, -L, L], cmap="Blues")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Fermion gas  f(x, y)")
plt.colorbar(im, ax=ax)

# (c) World lines  —  no scattering
ax = fig.add_subplot(gs[0, 2])
s_par = np.linspace(-L, L, 220)
for dir_vec in line_dirs:
    ax.plot(dir_vec[0] * s_par, dir_vec[1] * s_par,
            color="steelblue", lw=0.7, alpha=0.7)
ax.contour(X.T, Y.T, b_field.T,
           levels=[0.1, 0.5, 0.9],
           colors="red", linewidths=1.2, alpha=0.85)
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
ax.set_aspect("equal")
ax.set_title("Fermion world lines through BH\n(no scattering)")

# (d) Supertrace magnitude
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(np.log1p(np.abs(S_field)).T, origin="lower",
               extent=[-L, L, -L, L], cmap="magma")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("log(1 + |S|)   supertrace magnitude")
plt.colorbar(im, ax=ax)

# (e) Supertrace phase
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(np.angle(S_field).T, origin="lower",
               extent=[-L, L, -L, L], cmap="twilight",
               vmin=-PI, vmax=PI)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("arg S(x, y)   relativistic phase")
plt.colorbar(im, ax=ax, label="rad")

# (f) Signed diagonal at the BH center
ax = fig.add_subplot(gs[1, 2])
j0 = Ny // 2
M_diag = np.zeros(T, dtype=complex)
for t_idx in range(T):
    t_val = t_idx + 1
    kernel_val = kernel(t_val)[j0, j0]
    if t_idx % 2 == 0:
        M_diag[t_idx] = +1.0 * b_field[j0, j0] * kernel_val
    else:
        M_diag[t_idx] = -1.0 * f_field[j0, j0] * kernel_val
ax.stem(np.arange(1, T + 1), M_diag.real,
        linefmt="C0-", markerfmt="C0o", basefmt="k-")
ax.axhline(0, color="k", lw=0.4)
ax.set_xlabel("t_idx"); ax.set_ylabel("signed Re M_t")
ax.set_title("Signed diagonal at BH center")
ax.grid(alpha=0.3)

# (g) Radial density profile
ax = fig.add_subplot(gs[2, 0])
r_vals = np.linspace(0.0, L, 200)
b_prof = np.exp(-r_vals**2 / (2 * R_bh**2))
f_prof = 0.15 * (1.0 - b_prof) * np.exp(-r_vals**2 / (2 * (2 * L) ** 2))
ax.plot(r_vals, b_prof, color="#e74c3c", lw=1.8, label="boson")
ax.plot(r_vals, f_prof, color="#3498db", lw=1.8, label="fermion")
ax.axvline(R_bh, color="k", ls=":", lw=1.0,
           label=f"R_bh = {R_bh}")
ax.set_xlabel("R"); ax.set_ylabel("density")
ax.set_title("Radial density profile")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (h) Relativistic clock rate  dφ/dt = α · R
ax = fig.add_subplot(gs[2, 1])
r_clock = np.linspace(0.02, L, 200)
clock_rate = ALPHA * r_clock              # d(α t R)/dt = α R
ax.plot(r_clock, clock_rate, color="#16a085", lw=1.8)
ax.axvline(R_bh, color="k", ls=":", lw=1.0,
           label=f"R_bh = {R_bh}")
ax.fill_between(r_clock, 0, clock_rate,
                where=r_clock > R_bh, color="#16a085", alpha=0.15)
ax.fill_between(r_clock, 0, clock_rate,
                where=r_clock <= R_bh, color="#e67e22", alpha=0.2)
ax.set_xlabel("R"); ax.set_ylabel("dφ/dt")
ax.set_title("Relativistic clock rate  dφ/dt = α · R")
ax.text(0.6 * L, 0.55 * clock_rate.max(),
        "slow clock\n(near BH)", ha="center",
        color="#e67e22", fontsize=9)
ax.text(0.85 * L, 0.15 * clock_rate.max(),
        "fast clock\n(far away)", ha="center",
        color="#16a085", fontsize=9)
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (i) Mass vs blob radius
ax = fig.add_subplot(gs[2, 2])
ax.plot(R_scan, S_scan_abs, color="#8e44ad", lw=1.6, label="|S|")
ax.plot(R_scan, m_scan, color="#16a085", lw=1.6, ls="--",
        label="mass = |S| e^{−H}")
ax.axvline(R_bh, color="k", ls=":", lw=1.0, label=f"R_bh")
ax.set_xlabel("blob radius  R_bh")
ax.set_ylabel("value")
ax.set_title("Supertrace and mass vs blob radius")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle(
    f"Black hole as 2D boson blob  ·  "
    f"α = 1/(π − e) = {ALPHA:.4f}  ·  "
    f"|S| = {S_abs:.3e}  ·  mass = {m_bh:.3e}",
    fontsize=13, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ============================================================
#  Summary
# ============================================================
print("=" * 74)
print("Black hole as 2D boson blob  ·  M-matrix relativistic frame")
print("=" * 74)
print(f"  α                    = 1/(π − e) = {ALPHA:.6f}")
print(f"  Grid                 = {Nx} × {Ny}")
print(f"  Time slices T        = {T}")
print(f"  BH blob radius       = {R_bh}")
print(f"  Boson mass M_bh      = {M_bh:.6f}")
print(f"  Total supertrace |S| = {S_abs:.6e}")
print(f"  arg(S)               = {np.angle(S_total):+.6f} rad")
print(f"  Entropy H            = {H_bh:.6f}")
print(f"  Mass  |S| e^{{-H}}     = {m_bh:.6e}")
print(f"  Kernel build time    = {build_ms:.2f} ms")
print()
print("M-matrix structure:")
print("  M(t, x, y) = rho^t · exp(i α t R(x, y))")
print("      algebraic row          transcendental row")
print("      rho = R / L            relativistic phase")
print()
print("Block structure (no fermion-boson scattering):")
print("      ┌               ┐")
print("      │  M_BB    0    │   M_BB = boson self-energy (BH blob)")
print("      │    0   M_FF   │   M_FF = fermion self-energy")
print("      └               ┘   M_BF = M_FB = 0   ← no scattering")
print()
print("Physical interpretation:")
print("  • Bosons dominate the diagonal at even t_idx → positive S")
print("  • Fermions occupy odd t_idx, suppressed inside the BH")
print("  • World lines pass straight through: M_BF = 0")
print("  • Mass is protected by supertrace invariance")
print()
print("Relativistic content:")
print("  dφ/dt = α · R   —  clock rate grows linearly with R")
print(f"    at R = {R_bh}:  dφ/dt = {ALPHA * R_bh:.4f}")
print(f"    at R = {L}:   dφ/dt = {ALPHA * L:.4f}")