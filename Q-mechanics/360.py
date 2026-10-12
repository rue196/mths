#!/usr/bin/env python3
"""
cell_720_contraction_projection.py
==================================
Numerical demonstration of the 720-cell duality:

    · 6! = 720 vertices of the S6 polytope
    · 3D Cartan projection: 720 Weyl chambers in h ≅ R^3
    · 6D fundamental projection: 720 spinor blocks P_σ Z
    · oscillation length: 2 × 360 = 720
    · parity split: A_6 (360) ⊔ odd (360)
"""

from __future__ import annotations
import math
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)


# ============================================================
#  The 720-cell: all 6! permutations
# ============================================================
def build_720_cell():
    perms = list(permutations(range(6)))
    signs = []
    for p in perms:
        inv = sum(1 for i in range(6) for j in range(i + 1, 6)
                  if p[i] > p[j])
        signs.append((-1) ** inv)
    return np.array(perms, dtype=np.int32), np.array(signs, dtype=np.int32)


PERMS, SIGNS = build_720_cell()
N_CELL = PERMS.shape[0]                  # 720


# ============================================================
#  3D Cartan projection
# ============================================================
def cartan_coordinates(perm):
    """
    Simple Cartan coordinates for so(6):
        h1 = sigma_1 - sigma_2
        h2 = sigma_2 - sigma_3
        h3 = sigma_3 - sigma_4
    These three numbers identify the Weyl chamber of sigma.
    """
    s = perm
    return np.array([s[0] - s[1], s[1] - s[2], s[2] - s[3]],
                    dtype=float)


CARTAN_3D = np.array([cartan_coordinates(p) for p in PERMS])


# ============================================================
#  6D fundamental projection
# ============================================================
def fundamental_block(perm, Z_ref):
    """P_σ Z_ref: permute rows of the reference 6×6 block."""
    P = np.eye(6, dtype=complex)[list(perm), :]
    return P @ Z_ref


def build_reference_Z(seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))


Z_REF = build_reference_Z()
SIX_D_BLOCKS = np.array([fundamental_block(p, Z_REF) for p in PERMS])


# ============================================================
#  Chirality
# ============================================================
def levi_civita(Z):
    return complex(np.linalg.det(Z.astype(complex)))


def chirality(Z):
    Pi = levi_civita(Z)
    return math.atan2(Pi.imag, Pi.real), abs(Pi)


CHIRAL_ANGLES = np.array([chirality(Z)[0] for Z in SIX_D_BLOCKS])
CHIRAL_MAGS   = np.array([chirality(Z)[1] for Z in SIX_D_BLOCKS])


# ============================================================
#  Parity split: A6 vs odd complement
# ============================================================
EVEN_MASK = SIGNS > 0
ODD_MASK  = ~EVEN_MASK
N_EVEN = int(EVEN_MASK.sum())
N_ODD  = int(ODD_MASK.sum())


# ============================================================
#  Oscillation length
# ============================================================
T_CHI = PI / 3                     # chirality period
LAMBDA_720 = N_CELL * T_CHI * (2 / 6)   # = 4π/3
LAMBDA_DEG = LAMBDA_720 * 180 / PI      # = 720°


# ============================================================
#  Report
# ============================================================
def report():
    print("=" * 78)
    print("The 720-cell configuration  ·  3D contraction + 6D projection")
    print("=" * 78)
    print(f"  α = 1/(π − e)          = {ALPHA:.6f}")
    print(f"  |S_6| = 6!             = {N_CELL}")
    print(f"  |A_6| (even)           = {N_EVEN}")
    print(f"  odd complement         = {N_ODD}")
    print()
    print("--- 3D Cartan projection ---")
    print(f"  shape                  = {CARTAN_3D.shape}")
    print(f"  coordinate range       = "
          f"[{CARTAN_3D.min():.2f}, {CARTAN_3D.max():.2f}]")
    print(f"  distinct points        = "
          f"{len(np.unique(CARTAN_3D, axis=0))}")
    print()
    print("--- 6D fundamental projection ---")
    print(f"  shape                  = {SIX_D_BLOCKS.shape}")
    print(f"  distinct blocks        = "
          f"{len(np.unique(SIX_D_BLOCKS.reshape(N_CELL, -1), axis=0))}")
    print()
    print("--- chirality ---")
    print(f"  χ range                = "
          f"[{CHIRAL_ANGLES.min():+.4f}, {CHIRAL_ANGLES.max():+.4f}]")
    print(f"  |Π₆| range             = "
          f"[{CHIRAL_MAGS.min():.4f}, {CHIRAL_MAGS.max():.4f}]")
    print(f"  |Π₆| mean              = {CHIRAL_MAGS.mean():.4f}")
    print()
    print("--- oscillation length ---")
    print(f"  T_χ = π/3              = {T_CHI:.6f} rad")
    print(f"  Λ_720 = 720 · T_χ · 2/6 = {LAMBDA_720:.6f} rad")
    print(f"                    = {LAMBDA_DEG:.2f}°")
    print(f"  identity: 720° = 2 × 360°  ✓")
    print()


# ============================================================
#  Visualization
# ============================================================
def visualize():
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # (a) 3D Cartan scatter
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    colors = np.where(EVEN_MASK, "#16a085", "#e74c3c")
    ax.scatter(CARTAN_3D[:, 0], CARTAN_3D[:, 1], CARTAN_3D[:, 2],
               c=colors, s=8, alpha=0.7)
    ax.set_xlabel("h1"); ax.set_ylabel("h2"); ax.set_zlabel("h3")
    ax.set_title(f"3D Cartan projection\n"
                 f"green = A₆ (even)   red = odd")
    ax.view_init(elev=20, azim=-55)

    # (b) 6D projection — 2D PCA
    ax = fig.add_subplot(gs[0, 1])
    flat = SIX_D_BLOCKS.reshape(N_CELL, -1).real
    flat -= flat.mean(axis=0)
    U, S, Vt = np.linalg.svd(flat, full_matrices=False)
    proj = U[:, :2] * S[:2]
    ax.scatter(proj[EVEN_MASK, 0], proj[EVEN_MASK, 1],
               color="#16a085", s=8, alpha=0.7, label="A₆")
    ax.scatter(proj[ODD_MASK, 0], proj[ODD_MASK, 1],
               color="#e74c3c", s=8, alpha=0.7, label="odd")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("6D projection (PCA to 2D)\n"
                 "720 blocks · 2 clusters by parity")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) chirality histogram split by parity
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(CHIRAL_ANGLES[EVEN_MASK], bins=40, range=(-PI, PI),
            color="#16a085", alpha=0.7, label=f"A₆ ({N_EVEN})")
    ax.hist(CHIRAL_ANGLES[ODD_MASK], bins=40, range=(-PI, PI),
            color="#e74c3c", alpha=0.7, label=f"odd ({N_ODD})")
    ax.set_xlabel("χ = arg Π₆")
    ax.set_ylabel("count")
    ax.set_title("Chirality split by parity")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) |Π₆| distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(CHIRAL_MAGS, bins=60, color="#8e44ad", edgecolor="k",
            alpha=0.85)
    ax.set_xlabel(r"$|\Pi_6|$")
    ax.set_ylabel("count")
    ax.set_title(f"Levi-Civita magnitude over 720 blocks\n"
                 f"mean = {CHIRAL_MAGS.mean():.4f}")
    ax.grid(alpha=0.3)

    # (e) oscillation length illustration
    ax = fig.add_subplot(gs[1, 1])
    theta = np.linspace(0, 2 * PI, 400)
    ax.plot(theta, np.cos(6 * theta), color="#3a7bd5", lw=1.4,
            label=r"$\cos(6\theta)$  one chirality period")
    ax.plot(theta, np.cos(12 * theta), color="#e74c3c", lw=1.4,
            alpha=0.8, label=r"$\cos(12\theta)$  two periods")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(PI/3, color="#16a085", ls=":", lw=1.0,
               label=r"$\theta = \pi/3$ (one period)")
    ax.axvline(2*PI/3, color="#e67e22", ls=":", lw=1.0,
               label=r"$\theta = 2\pi/3$ (two periods)")
    ax.set_xlabel(r"phase $\theta$")
    ax.set_ylabel("chirality oscillation")
    ax.set_title(r"720 oscillations = $2 \times 360°$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (f) parity bar chart
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(["A₆", "odd"], [N_EVEN, N_ODD],
           color=["#16a085", "#e74c3c"], edgecolor="k")
    for i, v in enumerate([N_EVEN, N_ODD]):
        ax.text(i, v, f"  {v}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title(f"Parity split of the 720-cell\n"
                 f"720 = 360 + 360")
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"720-cell  ·  6! = {N_CELL}  ·  "
        f"3D Cartan contraction + 6D fundamental projection  ·  "
        f"Λ = 2 × 360° = {LAMBDA_DEG:.0f}°",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    report()
    visualize()