#!/usr/bin/env python3
"""
hurricane_m_matrix.py

2D weather forecasting using the M-matrix with the eye of the
hurricane as a **discrete algebraic object N**.

The M-matrix entry at spatial point (x_j, y_j) and time t is

    M_{t, j} = x_j^t + y_j^t                    (algebraic row)
               · exp(i · α · t · R_j)           (transcendental row)

where α = 1/(π − e) ≈ 2.362 and R_j is the radial distance from the
eye.  The algebraic row grows (or decays) with time depending on
whether the point is inside or outside the unit circle centered at
the eye; the transcendental row carries the flow and the time
evolution as a complex phase.

The eye sits at an **integer lattice point** (0, 0) with "mass" N.
It behaves like a black hole:
    R < R_jet   →  radial velocity points inward   (accretion)
    R > R_jet   →  radial velocity points outward  (jets)

The pressure generated inward gets continuously pushed back out,
and the tangential velocity (rotation) comes entirely from the
transcendental phase.

Everything is computed in O(K) time on the 2D grid.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA      = 1.0 / (PI - E)          # ≈ 2.362


# ============================================================
#  1. 2D weather grid
# ============================================================
nx, ny = 200, 200
L      = 6.0
x      = np.linspace(-L, L, nx)
y      = np.linspace(-L, L, ny)
X, Y   = np.meshgrid(x, y)


# ============================================================
#  2. Eye of the hurricane — discrete algebraic object N
# ============================================================
# The eye is at an integer lattice point (discrete).
eye_x, eye_y = 0.0, 0.0
N_eye        = 1.0                   # the algebraic scalar

# Radial distance and angle from the eye
R      = np.sqrt((X - eye_x)**2 + (Y - eye_y)**2)
R_safe = np.maximum(R, 0.2)          # avoid division by zero at the eye
Theta  = np.arctan2(Y - eye_y, X - eye_x)


# ============================================================
#  3. Pressure field  —  algebraic + transcendental
# ============================================================
# --- Algebraic part: pressure well around the discrete eye ---
# Smooth 1/(1 + R²) profile, with a discrete enhancement at integer
# lattice points (the eye's neighbours are the "discrete algebraic
# objects" that carry the eye's N).
p_alg = -N_eye / (1.0 + R**2)

# Discrete correction: integer lattice points get an extra boost
i_int = np.round(X).astype(int)
j_int = np.round(Y).astype(int)
at_lattice = (np.abs(X - i_int) < 0.1) & (np.abs(Y - j_int) < 0.1)
p_alg = np.where(at_lattice, p_alg * 1.5, p_alg)

# --- Transcendental part: spiral wave (flow and time) ---
# Outgoing spiral phase: α·R − ω·t
omega  = ALPHA * 0.6
t_now  = 0.0
phase  = ALPHA * R_safe - omega * t_now

# The transcendental pressure is an oscillating wave that carries
# the flow pattern and the time evolution
p_trans = -N_eye * np.sin(phase) * np.exp(-R_safe * 0.3) / (1.0 + R_safe**2)

# Total pressure
p_total = p_alg + p_trans


# ============================================================
#  4. Velocity field  —  black hole accretion + jets
# ============================================================
# Radial velocity: inward for R < R_jet, outward for R > R_jet
R_jet   = 2.5
R_decay = 1.2
v_r = -N_eye * (1.0 - R_safe / R_jet) * np.exp(-R_safe / R_decay)

# Tangential velocity: rotation, from the transcendental phase
R_rot   = 1.5
v_theta = ALPHA * N_eye / R_safe * np.exp(-R_safe / R_rot)

# Cartesian components
u = v_r * np.cos(Theta) - v_theta * np.sin(Theta)
v = v_r * np.sin(Theta) + v_theta * np.cos(Theta)


# ============================================================
#  5. M-matrix and supertrace
# ============================================================
# At each grid point (x_j, y_j) and each discrete time t:
#     M_{t, j} = x_j^t + y_j^t  ·  exp(i · α · t · R_j)
T = 24
M_tensor = np.zeros((T, ny, nx), dtype=complex)
for t_idx in range(T):
    t_val = t_idx + 1
    # algebraic row: real powers  →  pressure well
    M_alg = X**t_val + Y**t_val
    # transcendental row: imaginary-time phase  →  flow and time
    M_trans = np.exp(1j * ALPHA * t_val * R_safe)
    M_tensor[t_idx] = M_alg * M_trans

# Supertrace over discrete time at each grid point
S_field = np.zeros((ny, nx), dtype=complex)
for t_idx in range(T):
    sign = 1.0 if t_idx % 2 == 0 else -1.0
    S_field += sign * M_tensor[t_idx]

# Discrete eye contribution to the supertrace
# (the eye at (0,0) has M_{t, eye} = 0 for t > 0,
#  but it contributes a discrete N to the field)
eye_contribution = 0.0
for t_idx in range(T):
    sign = 1.0 if t_idx % 2 == 0 else -1.0
    eye_contribution += sign * N_eye * ALPHA


# ============================================================
#  6. Visualization
# ============================================================
fig = plt.figure(figsize=(17, 13))
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

# (a) Total pressure
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(p_total, origin='lower',
               extent=[-L, L, -L, L], cmap='RdBu_r',
               vmin=-1.5, vmax=1.5)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Total pressure  p(x, y)")
plt.colorbar(im, ax=ax)

# (b) Algebraic part
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(p_alg, origin='lower',
               extent=[-L, L, -L, L], cmap='Blues_r')
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Algebraic part  (discrete eye N)")
plt.colorbar(im, ax=ax)

# (c) Transcendental part
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(p_trans, origin='lower',
               extent=[-L, L, -L, L], cmap='Purples_r')
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Transcendental part  (flow + time)")
plt.colorbar(im, ax=ax)

# (d) Velocity magnitude
vel_mag = np.sqrt(u**2 + v**2)
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(vel_mag, origin='lower',
               extent=[-L, L, -L, L], cmap='hot')
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Velocity magnitude  |v|")
plt.colorbar(im, ax=ax)

# (e) Streamlines
ax = fig.add_subplot(gs[1, 1])
strm = ax.streamplot(x, y, u, v, density=1.5,
                     color=np.arctan2(v, u),
                     cmap='twilight', linewidth=0.8)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Flow streamlines")
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
plt.colorbar(strm.lines, ax=ax, label="direction")

# (f) Supertrace magnitude
ax = fig.add_subplot(gs[1, 2])
S_mag = np.abs(S_field)
im = ax.imshow(np.log1p(S_mag), origin='lower',
               extent=[-L, L, -L, L], cmap='magma')
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("log(1 + |S|)  supertrace magnitude")
plt.colorbar(im, ax=ax)

# (g) Radial pressure profile
ax = fig.add_subplot(gs[2, 0])
r_vals = np.linspace(0.05, L, 300)
p_alg_r   = -N_eye / (1.0 + r_vals**2)
p_trans_r = -N_eye * np.sin(ALPHA * r_vals) * np.exp(-r_vals * 0.3) / (1.0 + r_vals**2)
ax.plot(r_vals, p_alg_r, 'b-', label='algebraic')
ax.plot(r_vals, p_trans_r, 'r--', label='transcendental')
ax.plot(r_vals, p_alg_r + p_trans_r, 'k-', label='total')
ax.axvline(R_jet, color='g', ls=':', label=f'R_jet = {R_jet}')
ax.set_xlabel("R"); ax.set_ylabel("p(R)")
ax.set_title("Radial pressure profile")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (h) Velocity profiles
ax = fig.add_subplot(gs[2, 1])
v_r_prof     = -N_eye * (1.0 - r_vals / R_jet) * np.exp(-r_vals / R_decay)
v_theta_prof = ALPHA * N_eye / r_vals * np.exp(-r_vals / R_rot)
ax.plot(r_vals, v_r_prof,     'b-', label='v_r   (in→out)')
ax.plot(r_vals, v_theta_prof, 'r-', label='v_θ   (rotation)')
ax.axhline(0, color='k', lw=0.5)
ax.axvline(R_jet, color='g', ls=':', label=f'R_jet')
ax.set_xlabel("R"); ax.set_ylabel("velocity")
ax.set_title("Velocity profile — black-hole jets")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (i) M-matrix slice along y = 0
ax = fig.add_subplot(gs[2, 2])
row = ny // 2
M_slice = np.abs(M_tensor[:, row, :])
im = ax.imshow(M_slice, aspect='auto', origin='lower',
               extent=[-L, L, 1, T], cmap='viridis')
ax.set_xlabel("x"); ax.set_ylabel("t")
ax.set_title("|M_{t,x}|  along  y = 0")
plt.colorbar(im, ax=ax)

plt.suptitle(
    f"2D weather forecasting  ·  eye as discrete algebraic object N = {N_eye}",
    fontsize=14, y=0.995
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# ============================================================
#  7. Summary
# ============================================================
print("=" * 72)
print("2D weather forecasting  ·  M-matrix eye model")
print("=" * 72)
print(f"  α                    = 1/(π − e) = {ALPHA:.6f}")
print(f"  N (eye)              = {N_eye}")
print(f"  Eye position         = ({eye_x:.1f}, {eye_y:.1f})  [integer lattice point]")
print(f"  Jet radius R_jet     = {R_jet}")
print(f"  Pressure min / max   = {p_total.min():.4f}  /  {p_total.max():.4f}")
print(f"  Velocity max |v|     = {vel_mag.max():.4f}")
print(f"  Supertrace max       = {np.abs(S_field).max():.4e}")
print(f"  Eye contribution     = {eye_contribution.real:+.6f}")
print()
print("M-matrix structure:")
print("  M_{t, j} = (x_j^t + y_j^t) · exp(i · α · t · R_j)")
print("      algebraic row           transcendental row")
print("      → pressure well         → flow and time")
print()
print("Black hole behaviour:")
print(f"  radial velocity reverses at R = {R_jet}")
print(f"    R < {R_jet}  →  inward (accretion)")
print(f"    R > {R_jet}  →  outward (jets)")
print()
print("Complexity: O(Nx · Ny · T) — one pass over the grid per time step.")