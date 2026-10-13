#!/usr/bin/env python3
"""
spin_mri.py
===========
MRI-like imaging using the rank-6 Levi-Civita spin-coupling framework.

Pipeline
--------
1. Spin axis [0,1] discretized into N_SPIN = ⌈180·e⌉ = 490 points.
   Each point σ_k carries a spinor Z(σ_k) = Z_base · exp(i(π/3)σ_k).

2. 720-cell parity split (from 360.py)
       A_6 (even perms, 360)  →  bosonic  /  magnetic  (B)
       odd complement (360)   →  fermionic /  electric (E)
   The signal is transmitted in even parity (UGC labels) and
   accumulates coupling from the object it passes through.

3. Object: 2D phantom, projected to a 1D ray per angle via Radon.

4. Propagation uses the M-matrix from the fluid paper:
       M(t, z) = r(z)^t · exp(i α t z)

5. Spectral derivative modulates the local oscillation frequency:
       dζ/dt(t) = −Σ_{j≠0} (j/α) Re(C_j e^{ijt/α})

6. Coupling strength (from supertrace_6_kernel.pdf):
       g_ij = w_ij · κ_ij
       w_ij = |χ_i − χ_j| / π      chiral bond weight
       κ_ij = √(L_i L_j) / L_ref   travel coupling

7. Fermionic force (Theorem 13):
       F_i = −Σ_j |g_ij| cos(φ_ij) + i Σ_j |g_ij| sin(φ_ij)
       φ_ij = χ_i − χ_j
       ω_ij(t) = α(j − i) dζ/dt(t)

   Re F  →  electric field (fermionic, conservative)
   Im F  →  magnetic field (bosonic, oscillatory)

8. Filtered backprojection → 2D E and B images.
"""

from __future__ import annotations
import math
from itertools import permutations
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)
PHI   = (1.0 + math.sqrt(5.0)) / 2.0

N_SPIN_DEFAULT = int(math.ceil(180 * E))   # = 490


# ============================================================
#  720-cell parity (from 360.py)
# ============================================================
def build_720_cell():
    perms = list(permutations(range(6)))
    signs = np.zeros(len(perms), dtype=np.int32)
    for i, p in enumerate(perms):
        inv = sum(1 for a in range(6) for b in range(a + 1, 6)
                  if p[a] > p[b])
        signs[i] = (-1) ** inv
    return np.array(perms, dtype=np.int32), signs


PERMS, SIGNS = build_720_cell()
EVEN_MASK = SIGNS > 0                       # A_6, bosonic
ODD_MASK  = ~EVEN_MASK                      # fermionic
N_EVEN = int(EVEN_MASK.sum())               # 360
N_ODD  = int(ODD_MASK.sum())                # 360


# ============================================================
#  Spinor family  Z(σ) = Z_base · exp(i (π/3) σ)
# ============================================================
def make_base_Z(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((6, 6))
            + 1j * rng.standard_normal((6, 6)))


Z_BASE = make_base_Z()


def spin_to_Z(sigma: float) -> np.ndarray:
    return Z_BASE * np.exp(1j * (PI / 3.0) * sigma)


def build_spin_family(N: int) -> np.ndarray:
    """(N, 6, 6) complex spinor family along σ ∈ [0,1]."""
    sigmas = np.linspace(0.0, 1.0, N)
    return np.array([spin_to_Z(s) for s in sigmas])


# ============================================================
#  Chirality and Levi-Civita
# ============================================================
def levi_civita(Z: np.ndarray) -> complex:
    return complex(np.linalg.det(Z))


def chirality_of(Z: np.ndarray) -> Tuple[float, float]:
    Pi = levi_civita(Z)
    return math.atan2(Pi.imag, Pi.real), abs(Pi)


# ============================================================
#  M-matrix and spectral derivative
# ============================================================
def M_matrix(t: float, z: float,
             R_tip: float = 0.08, R_base: float = 0.5,
             H_cone: float = 3.0, p: float = 0.6) -> complex:
    zc = float(np.clip(z, 0.0, H_cone))
    zn = zc / H_cone
    r = R_tip + (R_base - R_tip) * (1.0 - zn * zn) ** p
    r = max(r, 1e-6)
    log_r_t = float(np.clip(t * math.log(r), -30.0, 30.0))
    amp = math.exp(log_r_t)
    phase = ALPHA * t * zc
    return complex(amp * math.cos(phase), amp * math.sin(phase))


def spectral_derivative(t: float, C: np.ndarray) -> float:
    """
    dζ/dt(t) = −Σ_{j≠0} (j/α) Re(C_j e^{ijt/α}).
    C is indexed by j ∈ [−K, K] via C[K + j].
    """
    K = (len(C) - 1) // 2
    j = np.arange(-K, K + 1)
    j_nz = j[j != 0]
    phases = j_nz * t / ALPHA
    Cj = C[K + j_nz]
    return float(-np.sum((j_nz / ALPHA)
                         * np.real(Cj * np.exp(1j * phases))))


# ============================================================
#  Coupling force  (Theorem 13)
# ============================================================
def coupling_fields(chis: np.ndarray, L: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    F_i = −Σ_j |g_ij| cos(φ_ij) + i Σ_j |g_ij| sin(φ_ij)

    Vectorized over the N spin points.
    Returns (F_real, F_imag) — both length N.
    """
    N = len(chis)
    phi = chis[:, None] - chis[None, :]           # φ_ij
    phi_w = (phi + PI) % (2.0 * PI) - PI          # wrap to (−π, π]
    w = np.abs(phi_w) / PI                        # ∈ [0, 1]
    Lp = np.maximum(L, 1e-9)
    Lsqrt = np.sqrt(np.outer(Lp, Lp))
    L_ref = float(np.median(Lp)) + 1e-12
    kappa = np.clip(Lsqrt / L_ref, 0.0, 1.0)      # ∈ [0, 1]
    g = w * kappa
    np.fill_diagonal(g, 0.0)
    F_real = -np.sum(g * np.cos(phi_w), axis=1)
    F_imag =  np.sum(g * np.sin(phi_w), axis=1)
    return F_real, F_imag


# ============================================================
#  Object and Radon projection
# ============================================================
def shepp_logan(N: int = 64) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x, x)
    img = np.zeros_like(X)
    ellipses = [
        (0.0,  0.0,    0.69,  0.92,   0.0,  1.0),
        (0.0, -0.0184, 0.6624, 0.874, 0.0, -0.8),
        (0.22,  0.0,   0.11,  0.31, -18.0, -0.2),
        (-0.22, 0.0,   0.16,  0.41,  18.0, -0.2),
        (0.0,  0.35,   0.21,  0.25,   0.0,  0.1),
        (0.0,  0.10,   0.046, 0.046,  0.0,  0.1),
        (0.0, -0.10,   0.046, 0.046,  0.0,  0.1),
        (-0.08, -0.605, 0.046, 0.023, 0.0,  0.1),
        (0.0, -0.606,  0.023, 0.023,  0.0,  0.1),
        (0.06, -0.605, 0.023, 0.046,  0.0,  0.1),
    ]
    for cx, cy, a, b, th_deg, val in ellipses:
        th = math.radians(th_deg)
        Xr =  (X - cx) * math.cos(th) + (Y - cy) * math.sin(th)
        Yr = -(X - cx) * math.sin(th) + (Y - cy) * math.cos(th)
        inside = (Xr / a) ** 2 + (Yr / b) ** 2 <= 1.0
        img[inside] += val
    img -= img.min()
    img /= (img.max() + 1e-12)
    return img


def radon_1d(img: np.ndarray, angle: float,
             N_proj: int) -> np.ndarray:
    """1D Radon projection of a 2D image at the given angle."""
    N = img.shape[0]
    c, s = math.cos(angle), math.sin(angle)
    t = np.linspace(-1.0, 1.0, N_proj)
    sj = np.linspace(-1.0, 1.0, N)
    ds = 2.0 / N
    T, S = np.meshgrid(t, sj, indexing="ij")
    X = T * c - S * s
    Y = T * s + S * c
    Xi = np.clip(((X + 1.0) * 0.5 * (N - 1)).astype(int), 0, N - 1)
    Yi = np.clip(((Y + 1.0) * 0.5 * (N - 1)).astype(int), 0, N - 1)
    p = img[Yi, Xi].sum(axis=1) * ds
    p = np.clip(p, 0.0, None)
    p /= (p.max() + 1e-12)
    return p


# ============================================================
#  Forward model
# ============================================================
def forward_project(Z_init, object_ray, H_cone=3.0,
                    phase_gain=0.4, amp_gain=0.15,
                    N_layers_override=None):
    N = Z_init.shape[0]
    N_layers = N_layers_override or len(object_ray)
    z_grid = np.linspace(0.0, H_cone, N_layers)
    # pad or truncate the ray to N_layers
    if N_layers != len(object_ray):
        idx = np.linspace(0, len(object_ray) - 1, N_layers)
        object_ray = np.interp(idx, np.arange(len(object_ray)),
                               object_ray)
    ...
    """
    Propagate the spin family through a 1D object profile.

    For each layer z_m of the object:
        dζ/dt  ← spectral derivative at this layer
        Z_k    ← M(t_k, z_m) · exp(i·gain·g_m·dζ/dt·t_k/N) ·
                 (1 + gain·g_m·tanh(t_k/N)) · Z_k
    """
    N = Z_init.shape[0]
    N_layers = len(object_ray)
    z_grid = np.linspace(0.0, H_cone, N_layers)

    # C_j coefficients from the initial determinant family
    dets = np.array([levi_civita(Z) for Z in Z_init])
    K = N // 2
    C_arr = np.zeros(2 * K + 1, dtype=complex)
    for j in range(-K, K + 1):
        C_arr[K + j] = dets[j % N]

    t_vals = np.arange(N) - K          # centered time indices

    Z_cur = Z_init.copy()

    # layer-by-layer propagation
    for layer_idx, (z_m, g_m) in enumerate(zip(z_grid, object_ray)):
        zeta_dot = spectral_derivative(layer_idx, C_arr)

        zc = float(np.clip(z_m, 0.0, H_cone))
        zn = zc / H_cone
        r = 0.08 + (0.5 - 0.08) * (1.0 - zn ** 2) ** 0.6
        r = max(r, 1e-6)
        log_r_t = np.clip(t_vals * math.log(r), -30.0, 30.0)

        M_vals = (np.exp(log_r_t)
                  * np.exp(1j * ALPHA * t_vals * zc))

        # coupling modulation — phase AND amplitude
        phase_mod = np.exp(1j * phase_gain * g_m
                           * zeta_dot * t_vals / N)
        amp_mod = 1.0 + amp_gain * g_m * np.tanh(t_vals / N)

        scalars = M_vals * phase_mod * amp_mod
        Z_cur *= scalars[:, None, None]

    # final chirality and magnitude
    chis = np.zeros(N)
    mags = np.zeros(N)
    for k in range(N):
        chis[k], mags[k] = chirality_of(Z_cur[k])

    L = np.abs(mags) + 1e-6

    # coupling force (Theorem 13)
    F_real, F_imag = coupling_fields(chis, L)

    # parity: 1 = even (bosonic), 0 = odd (fermionic)
    parity = (np.cos(chis) >= 0).astype(int)

    return dict(E=F_real, B=F_imag, parity=parity,
                chis=chis, mags=mags)


# ============================================================
#  Filtered backprojection
# ============================================================
def ramp_filter(sinogram: np.ndarray) -> np.ndarray:
    """Ram-Lak ramp filter along the projection axis."""
    N_proj = sinogram.shape[1]
    f = np.fft.fftfreq(N_proj)
    ramp = 2.0 * np.abs(f)
    out = np.zeros_like(sinogram)
    for i in range(sinogram.shape[0]):
        F = np.fft.fft(sinogram[i])
        out[i] = np.real(np.fft.ifft(F * ramp))
    return out


def backproject(sinogram: np.ndarray, N: int,
                angles: np.ndarray) -> np.ndarray:
    N_ang, N_proj = sinogram.shape
    x = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x, x)
    img = np.zeros((N, N))
    for a_idx, theta in enumerate(angles):
        s = X * math.cos(theta) + Y * math.sin(theta)
        s_idx = np.clip(
            ((s + 1.0) * 0.5 * (N_proj - 1)).astype(int),
            0, N_proj - 1)
        img += sinogram[a_idx][s_idx]
    img /= N_ang
    return img


# ============================================================
#  Full pipeline
# ============================================================
def spin_mri(image: np.ndarray,
             N_angles: int = 72,
             N_layers: int = 32,
             N_spin: int = N_SPIN_DEFAULT,
             verbose: bool = True) -> Dict:
    if verbose:
        print("=" * 78)
        print("Spin MRI  ·  coupling-force reconstruction")
        print("=" * 78)
        print(f"  N_SPIN          = {N_spin}   "
              f"(⌈180·e⌉ = {int(math.ceil(180 * E))})")
        print(f"  N_angles        = {N_angles}")
        print(f"  N_layers        = {N_layers}")
        print(f"  720-cell        : A_6 = {N_EVEN} (bosonic), "
              f"odd = {N_ODD} (fermionic)")
        print(f"  α               = 1/(π−e) = {ALPHA:.6f}")
        print()

    Z_init = build_spin_family(N_spin)
    angles = np.linspace(0.0, PI, N_angles, endpoint=False)

    sinogram_E = np.zeros((N_angles, N_spin))
    sinogram_B = np.zeros((N_angles, N_spin))
    parities   = np.zeros((N_angles, N_spin))
    chis_all   = np.zeros((N_angles, N_spin))

    for a_idx, theta in enumerate(angles):
        ray = radon_1d(image, theta, N_layers)
        out = forward_project(Z_init, ray)          # ← N_layers removed
        sinogram_E[a_idx] = out["E"]
        sinogram_B[a_idx] = out["B"]
        parities[a_idx]   = out["parity"]
        chis_all[a_idx]   = out["chis"]

        if verbose and (a_idx + 1) % 18 == 0:
            print(f"  angle {a_idx + 1:>3d}/{N_angles}   "
                  f"⟨parity⟩ = {out['parity'].mean():.3f}")

    # filtered backprojection
    image_E = backproject(ramp_filter(sinogram_E),
                          image.shape[0], angles)
    image_B = backproject(ramp_filter(sinogram_B),
                          image.shape[0], angles)

    if verbose:
        print()
        print(f"  E sinogram range = [{sinogram_E.min():+.3f}, "
              f"{sinogram_E.max():+.3f}]")
        print(f"  B sinogram range = [{sinogram_B.min():+.3f}, "
              f"{sinogram_B.max():+.3f}]")
        print(f"  parity mean      = {parities.mean():.4f}")
        print()

    return dict(
        sinogram_E=sinogram_E, sinogram_B=sinogram_B,
        image_E=image_E, image_B=image_B,
        parities=parities, chis=chis_all, angles=angles,
    )
    # filtered backprojection
    image_E = backproject(ramp_filter(sinogram_E),
                          image.shape[0], angles)
    image_B = backproject(ramp_filter(sinogram_B),
                          image.shape[0], angles)

    if verbose:
        print()
        print(f"  E sinogram range = [{sinogram_E.min():+.3f}, "
              f"{sinogram_E.max():+.3f}]")
        print(f"  B sinogram range = [{sinogram_B.min():+.3f}, "
              f"{sinogram_B.max():+.3f}]")
        print(f"  parity mean      = {parities.mean():.4f}")
        print()

    return dict(
        sinogram_E=sinogram_E, sinogram_B=sinogram_B,
        image_E=image_E, image_B=image_B,
        parities=parities, chis=chis_all, angles=angles,
    )


# ============================================================
#  Demo
# ============================================================
def demo():
    N_img = 64
    obj = shepp_logan(N_img)

    result = spin_mri(obj, N_angles=72, N_layers=32,
                      N_spin=N_SPIN_DEFAULT)

    # ============================================================
    #  Main figure: object, sinograms, parity map, reconstructions
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(obj, cmap="gray", origin="lower")
    ax.set_title("Object  ·  Shepp-Logan phantom")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(result["sinogram_E"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ (spin)"); ax.set_ylabel("angle")
    ax.set_title("E sinogram  ·  Re F  (fermionic)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(result["sinogram_B"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ (spin)"); ax.set_ylabel("angle")
    ax.set_title("B sinogram  ·  Im F  (bosonic)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(result["parities"], aspect="auto",
                   origin="lower", cmap="RdYlGn",
                   vmin=0, vmax=1)
    ax.set_xlabel("σ"); ax.set_ylabel("angle")
    ax.set_title(f"Parity map  ·  ⟨even⟩ = "
                 f"{result['parities'].mean():.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(result["image_E"], cmap="inferno", origin="lower")
    ax.set_title("E reconstruction  ·  fermionic (Re F)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(result["image_B"], cmap="inferno", origin="lower")
    ax.set_title("B reconstruction  ·  bosonic (Im F)")
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        f"Spin MRI  ·  N_SPIN = {N_SPIN_DEFAULT}  ·  "
        f"720-cell parity  ·  M-matrix + spectral derivative  ·  "
        f"coupling force (Theorem 13)  ·  α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Detail figure: chirality, parity, field profiles
    # ============================================================
    fig2 = plt.figure(figsize=(14, 8))
    gs2 = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.30)

    ax = fig2.add_subplot(gs2[0, 0])
    chis_avg = result["chis"].mean(axis=0)
    x_sigma = np.linspace(0.0, 1.0, len(chis_avg))
    ax.plot(x_sigma, chis_avg, color="#8e44ad", lw=1.2)
    ax.axhline(0, color="k", ls=":", lw=0.7)
    ax.axhline(PI / 2, color="#e74c3c", ls=":", lw=0.7,
               label="±π/2  (parity flip)")
    ax.axhline(-PI / 2, color="#e74c3c", ls=":", lw=0.7)
    ax.set_xlabel("σ"); ax.set_ylabel("χ(σ)")
    ax.set_title("Average chirality along the spin axis")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig2.add_subplot(gs2[0, 1])
    chis_flat = result["chis"].flatten()
    even_par = np.cos(chis_flat) >= 0
    ax.hist(chis_flat[even_par], bins=40, range=(-PI, PI),
            color="#16a085", alpha=0.7,
            label=f"even / bosonic ({int(even_par.sum())})")
    ax.hist(chis_flat[~even_par], bins=40, range=(-PI, PI),
            color="#e74c3c", alpha=0.7,
            label=f"odd / fermionic ({int((~even_par).sum())})")
    ax.set_xlabel("χ"); ax.set_ylabel("count")
    ax.set_title("Chirality distribution  ·  parity split "
                 "(from 720-cell A_6 vs odd)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig2.add_subplot(gs2[1, 0])
    ax.plot(result["sinogram_E"][0], color="#e74c3c",
            lw=1.0, label="angle 0")
    ax.plot(result["sinogram_E"][N_angles // 2], color="#3a7bd5",
            lw=1.0, label=f"angle π/2")
    ax.set_xlabel("σ"); ax.set_ylabel("E(σ)")
    ax.set_title("E-field profile  ·  2 angles")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig2.add_subplot(gs2[1, 1])
    ax.plot(result["sinogram_B"][0], color="#e74c3c",
            lw=1.0, label="angle 0")
    ax.plot(result["sinogram_B"][N_angles // 2], color="#3a7bd5",
            lw=1.0, label=f"angle π/2")
    ax.set_xlabel("σ"); ax.set_ylabel("B(σ)")
    ax.set_title("B-field profile  ·  2 angles")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(
        "Spin MRI  ·  chirality, parity, and field profiles",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    demo()