#!/usr/bin/env python3
"""
spin_mri_finite_step.py
=======================
MRI-like imaging using the finite-step 2D-to-3D object.

Replaces the Radon acquisition (N_angles) with the finite-step
2D-to-3D mapping from 3d_object_finite_step.py.

Pipeline
--------
1. Finite-step object: nested organic shells
       A(n)   = 10 · a · sin(n²°),  a = 1/(π − e)
       ΔA_n   = A(n+1) − A(n)              (finite-step derivative)
       shell n = warped sphere of radius A(n),
                 deformation amplitude ∝ ΔA_n

2. Spin axis σ ∈ [0,1] → N_SPIN = ⌈180·e⌉ = 490 spinors
       Z(σ) = Z_base · exp(i (π/3) σ)

3. 720-cell parity split (from 360.py)
       A_6 (even, 360) → bosonic / magnetic (B)
       odd complement  → fermionic / electric (E)

4. Propagation: layer-by-layer through the finite-step shells
       M(t, z)  = r(z)^t · exp(i α t z)
       dζ/dt    = −Σ_{j≠0} (j/α) Re(C_j e^{ijt/α})
       g_n      = |ΔA_n| / A_ref              (coupling strength)

5. Fermionic force  (Theorem 13)
       F_i = −Σ_j |g_ij| cos φ_ij + i Σ_j |g_ij| sin φ_ij
       ω_ij(t) = α(j − i) dζ/dt(t)
       Re F → E (fermionic)   Im F → B (bosonic)

6. Reconstruction: E and B fields mapped onto the warped shell
   surfaces. The (N_shells, N_spin) array replaces the sinogram.
"""

from __future__ import annotations
import math
from functools import partial
from itertools import permutations
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)
a_c   = ALPHA                              # = 1/(π − e)
H_CONE = 3.0                               # from the fluid paper

N_SPIN_DEFAULT = int(math.ceil(180 * E))   # 490


# ============================================================
#  720-cell parity (from 360.py)
# ============================================================
def build_720_cell():
    perms = list(permutations(range(6)))
    signs = np.zeros(len(perms), dtype=np.int32)
    for i, p in enumerate(perms):
        inv = sum(1 for x in range(6) for y in range(x + 1, 6)
                  if p[x] > p[y])
        signs[i] = (-1) ** inv
    return np.array(perms, dtype=np.int32), signs


PERMS, SIGNS = build_720_cell()
EVEN_MASK = SIGNS > 0
ODD_MASK  = ~EVEN_MASK
N_EVEN = int(EVEN_MASK.sum())
N_ODD  = int(ODD_MASK.sum())


# ============================================================
#  Finite-step scaling A(n) and its derivative
# ============================================================
def A_of_n(n: float) -> float:
    """A(n) = 10 · a · sin(n²°)  with a = 1/(π − e)."""
    return 10.0 * a_c * math.sin(math.radians(n * n))


def finite_step(n: int, mode: str = "forward") -> float:
    if mode == "forward":
        return A_of_n(n + 1) - A_of_n(n)
    if mode == "backward":
        return A_of_n(n) - A_of_n(n - 1)
    if mode == "central":
        return 0.5 * (A_of_n(n + 1) - A_of_n(n - 1))
    raise ValueError(mode)


# ============================================================
#  Organic warped 2D grid → warped sphere
# ============================================================
def organic_uv(nu: int = 40, nv: int = 40,
               warp: float = 0.18):
    u = np.linspace(0.0, 1.0, nu)
    v = np.linspace(0.0, 1.0, nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    Uw = U + warp * np.sin(2 * PI * V + 0.4) \
           + 0.06 * np.cos(4 * PI * U + 0.9)
    Vw = V + warp * np.sin(2 * PI * U + 1.2) \
           + 0.06 * np.cos(4 * PI * V + 0.2)
    return U, V, Uw, Vw


def organic_radius(U: np.ndarray, V: np.ndarray,
                   amp: float = 0.20,
                   freq: float = 3.0,
                   phase: float = 0.0) -> np.ndarray:
    return 1.0 + amp * np.sin(2 * PI * freq * U + phase) \
                     * np.sin(2 * PI * freq * V + phase)


def shell_xyz(Uw, Vw, R=1.0, radius_mod=None):
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


def build_nested_shells(n_layers: int = 6,
                        nu: int = 40, nv: int = 40,
                        warp: float = 0.18,
                        mod_amp: float = 0.22,
                        mod_freq: float = 3.0):
    """
    One shell per n = 1 … n_layers.

        radius  = A(n)                    (scaling value)
        mod_amp = mod_amp · ΔA_n / ΔA_ref (finite-step derivative)
    """
    U, V, Uw, Vw = organic_uv(nu, nv, warp)
    ref_step = abs(finite_step(4))
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


# ============================================================
#  Spinor family
# ============================================================
def make_base_Z(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((6, 6))
            + 1j * rng.standard_normal((6, 6)))


Z_BASE = make_base_Z()


def spin_to_Z(sigma: float) -> np.ndarray:
    return Z_BASE * np.exp(1j * (PI / 3.0) * sigma)


def build_spin_family(N: int) -> np.ndarray:
    sigmas = np.linspace(0.0, 1.0, N)
    return np.array([spin_to_Z(s) for s in sigmas])


def levi_civita(Z: np.ndarray) -> complex:
    return complex(np.linalg.det(Z))


def chirality_of(Z: np.ndarray) -> Tuple[float, float]:
    Pi = levi_civita(Z)
    return math.atan2(Pi.imag, Pi.real), abs(Pi)


# ============================================================
#  Spectral derivative
# ============================================================
def spectral_derivative(t: float, C: np.ndarray) -> float:
    K = (len(C) - 1) // 2
    j = np.arange(-K, K + 1)
    j_nz = j[j != 0]
    phases = j_nz * t / ALPHA
    Cj = C[K + j_nz]
    return float(-np.sum((j_nz / ALPHA)
                         * np.real(Cj * np.exp(1j * phases))))


# ============================================================
#  Coupling force (Theorem 13)
# ============================================================
def coupling_fields(chis: np.ndarray, L: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
    phi = chis[:, None] - chis[None, :]
    phi_w = (phi + PI) % (2.0 * PI) - PI
    w = np.abs(phi_w) / PI
    Lp = np.maximum(L, 1e-9)
    Lsqrt = np.sqrt(np.outer(Lp, Lp))
    L_ref = float(np.median(Lp)) + 1e-12
    kappa = np.clip(Lsqrt / L_ref, 0.0, 1.0)
    g = w * kappa
    np.fill_diagonal(g, 0.0)
    F_real = -np.sum(g * np.cos(phi_w), axis=1)
    F_imag =  np.sum(g * np.sin(phi_w), axis=1)
    return F_real, F_imag


# ============================================================
#  Layer-by-layer propagation through the shells
# ============================================================
def propagate_through_shells(Z_init: np.ndarray,
                             shells: list,
                             phase_gain: float = 0.4,
                             amp_gain: float = 0.15,
                             verbose: bool = True) -> Dict:
    """
    Propagate the spinor family through each shell in the object.

    For each shell n with coupling strength g_n = |ΔA_n| / A_ref:
        dζ/dt ← spectral_derivative at this layer
        Z_k   ← M(t_k, A_n) · exp(i·g_n·dζ/dt·t_k/N)
              · (1 + g_n·tanh(t_k/N)) · Z_k
    """
    N = Z_init.shape[0]
    n_shells = len(shells)

    dets = np.array([levi_civita(Z) for Z in Z_init])
    K = N // 2
    C_arr = np.zeros(2 * K + 1, dtype=complex)
    for j in range(-K, K + 1):
        C_arr[K + j] = dets[j % N]

    t_vals = np.arange(N) - K

    # reference finite-step
    A_ref = max(abs(shell["scale"]) for shell in shells) + 1e-9

    Z_cur = Z_init.copy()
    sinogram_E = np.zeros((n_shells, N))
    sinogram_B = np.zeros((n_shells, N))
    parities   = np.zeros((n_shells, N))
    chis_all   = np.zeros((n_shells, N))
    mags_all   = np.zeros((n_shells, N))
    g_ns       = np.zeros(n_shells)

    for idx, sh in enumerate(shells):
        n     = sh["n"]
        R_n   = sh["scale"]
        dA_n  = sh["step"]
        g_n   = abs(dA_n) / A_ref          # coupling ∈ [0, 1]

        if verbose:
            print(f"  shell n={n:>2d}   A(n)={R_n:>8.4f}   "
                  f"ΔA={dA_n:>+8.4f}   g_n={g_n:.4f}")

        zeta_dot = spectral_derivative(idx, C_arr)

        zc = float(np.clip(abs(R_n), 0.0, H_CONE))
        zn = zc / H_CONE
        r = 0.08 + (0.5 - 0.08) * (1.0 - zn ** 2) ** 0.6
        r = max(r, 1e-6)
        log_r_t = np.clip(t_vals * math.log(r), -30.0, 30.0)

        M_vals = (np.exp(log_r_t)
                  * np.exp(1j * ALPHA * t_vals * zc))

        phase_mod = np.exp(1j * phase_gain * g_n
                           * zeta_dot * t_vals / N)
        amp_mod = 1.0 + amp_gain * g_n * np.tanh(t_vals / N)

        scalars = M_vals * phase_mod * amp_mod
        Z_cur *= scalars[:, None, None]

        chis = np.zeros(N)
        mags = np.zeros(N)
        for k in range(N):
            chis[k], mags[k] = chirality_of(Z_cur[k])

        L = np.abs(mags) + 1e-6
        F_real, F_imag = coupling_fields(chis, L)
        parity = (np.cos(chis) >= 0).astype(int)

        sinogram_E[idx] = F_real
        sinogram_B[idx] = F_imag
        parities[idx]   = parity
        chis_all[idx]   = chis
        mags_all[idx]   = mags
        g_ns[idx]       = g_n

    return dict(
        sinogram_E=sinogram_E, sinogram_B=sinogram_B,
        parities=parities, chis=chis_all, mags=mags_all,
        g_ns=g_ns, Z_final=Z_cur,
    )


# ============================================================
#  Map 1D field on σ ∈ [0,1] to the 2D warped shell surface
# ============================================================
def field_to_surface(field_1d: np.ndarray,
                     nu: int, nv: int) -> np.ndarray:
    """
    Spiral map from σ ∈ [0,1] to the (nu, nv) surface grid.

        σ(i, j) = (i · nu + j) / (nu · nv)  mod  1
    """
    N = len(field_1d)
    idx = np.arange(nu * nv) % N
    return field_1d[idx].reshape(nu, nv)


# ============================================================
#  Main pipeline
# ============================================================
def spin_mri_finite_step(n_layers: int = 6,
                         nu: int = 40, nv: int = 40,
                         N_spin: int = N_SPIN_DEFAULT,
                         verbose: bool = True) -> Dict:
    if verbose:
        print("=" * 78)
        print("Spin MRI  ·  finite-step 2D-to-3D object")
        print("=" * 78)
        print(f"  N_SPIN          = {N_spin}   "
              f"(⌈180·e⌉ = {int(math.ceil(180 * E))})")
        print(f"  N_shells        = {n_layers}")
        print(f"  nu × nv         = {nu} × {nv}")
        print(f"  720-cell        : A_6 = {N_EVEN} (bosonic), "
              f"odd = {N_ODD} (fermionic)")
        print(f"  α               = 1/(π−e) = {ALPHA:.6f}")
        print()

    shells = build_nested_shells(
        n_layers=n_layers, nu=nu, nv=nv,
        warp=0.18, mod_amp=0.22, mod_freq=3.0)

    if verbose:
        print("--- shells ---")
        for sh in shells:
            print(f"  n={sh['n']:>2d}  A(n)={sh['scale']:>8.4f}  "
                  f"ΔA={sh['step']:>+8.4f}  amp={sh['amp']:>+8.4f}")

    Z_init = build_spin_family(N_spin)
    if verbose:
        print()
        print("--- propagation through shells ---")

    result = propagate_through_shells(
        Z_init, shells, verbose=verbose)

    # map 1D fields onto the 2D shell surface
    surf_E = []
    surf_B = []
    surf_par = []
    for idx, sh in enumerate(shells):
        surf_E.append(field_to_surface(result["sinogram_E"][idx], nu, nv))
        surf_B.append(field_to_surface(result["sinogram_B"][idx], nu, nv))
        surf_par.append(field_to_surface(
            result["parities"][idx].astype(float), nu, nv))

    result["shells"] = shells
    result["surf_E"] = surf_E
    result["surf_B"] = surf_B
    result["surf_par"] = surf_par
    result["nu"] = nu
    result["nv"] = nv

    if verbose:
        print()
        print(f"  E sinogram range = "
              f"[{result['sinogram_E'].min():+.3f}, "
              f"{result['sinogram_E'].max():+.3f}]")
        print(f"  B sinogram range = "
              f"[{result['sinogram_B'].min():+.3f}, "
              f"{result['sinogram_B'].max():+.3f}]")
        print(f"  parity mean      = "
              f"{result['parities'].mean():.4f}")
        print()

    return result


# ============================================================
#  Visualisation
# ============================================================
def plot_scaling_curve(ax):
    ns = np.arange(1, 10)
    A  = np.array([A_of_n(k)      for k in ns])
    dA = np.array([finite_step(k) for k in ns])
    ax.plot(ns, A, "o-", color="#2c3e50", lw=1.6, label=r"$A(n)$")
    ax.bar(ns + 0.15, dA, width=0.3, color="#e67e22", alpha=0.75,
           label=r"$\Delta A_n$")
    ax.axvline(4, color="#e74c3c", ls=":", lw=1.2,
               label=r"$n=4$: $A=6.511$, $\Delta A\approx 3.47$")
    ax.set_xlabel("n"); ax.set_ylabel(r"$A(n)$  /  $\Delta A_n$")
    ax.set_title(r"Scaling $A(n)$ and finite-step $\Delta A_n$")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def plot_shell_field(ax, shell, surf_E, title_prefix="E-field"):
    X, Y, Z = shell["X"], shell["Y"], shell["Z"]
    Emin, Emax = surf_E.min(), surf_E.max()
    norm = (surf_E - Emin) / (Emax - Emin + 1e-12)
    facecolors = plt.cm.inferno(norm)
    ax.plot_surface(X, Y, Z, facecolors=facecolors,
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(f"{title_prefix}  ·  shell n={shell['n']}  ·  "
                 f"A(n)={shell['scale']:.3f}\n"
                 f"ΔA={shell['step']:+.3f}  ·  "
                 f"range=[{Emin:+.2f}, {Emax:+.2f}]")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)


def plot_nested_shells(ax, shells, surf_E, alpha=0.35):
    ax.set_title("Nested organic shells  ·  E-field on surface")
    Emin = min(s.min() for s in surf_E)
    Emax = max(s.max() for s in surf_E)
    for sh, sE in zip(shells, surf_E):
        norm = (sE - Emin) / (Emax - Emin + 1e-12)
        facecolors = plt.cm.inferno(norm)
        ax.plot_surface(sh["X"], sh["Y"], sh["Z"],
                        facecolors=facecolors,
                        rstride=2, cstride=2,
                        linewidth=0, antialiased=True, alpha=alpha)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)


def demo():
    N_shells = 6
    nu = nv = 40

    result = spin_mri_finite_step(
        n_layers=N_shells, nu=nu, nv=nv,
        N_spin=N_SPIN_DEFAULT, verbose=True)

    shells  = result["shells"]
    surf_E  = result["surf_E"]
    surf_B  = result["surf_B"]

    # ============================================================
    #  Figure 1: scaling, parity, sinograms, 3D shells
    # ============================================================
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    # (a) scaling curve
    ax = fig.add_subplot(gs[0, 0])
    plot_scaling_curve(ax)

    # (b) E sinogram  (N_shells, N_spin)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(result["sinogram_E"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ"); ax.set_ylabel("shell n")
    ax.set_yticks(np.arange(N_shells))
    ax.set_yticklabels([sh["n"] for sh in shells])
    ax.set_title("E field  ·  (N_shells, N_spin)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) B sinogram
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(result["sinogram_B"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ"); ax.set_ylabel("shell n")
    ax.set_yticks(np.arange(N_shells))
    ax.set_yticklabels([sh["n"] for sh in shells])
    ax.set_title("B field  ·  (N_shells, N_spin)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) parity map
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(result["parities"], aspect="auto",
                   origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("σ"); ax.set_ylabel("shell n")
    ax.set_yticks(np.arange(N_shells))
    ax.set_yticklabels([sh["n"] for sh in shells])
    ax.set_title(f"Parity  ·  ⟨even⟩ = "
                 f"{result['parities'].mean():.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (e) coupling strengths per shell
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(np.arange(N_shells), result["g_ns"],
           color="#8e44ad", edgecolor="k")
    ax.set_xticks(np.arange(N_shells))
    ax.set_xticklabels([sh["n"] for sh in shells])
    ax.set_xlabel("shell n"); ax.set_ylabel("g_n")
    ax.set_title("Coupling strength per shell\n"
                 r"$g_n = |\Delta A_n| / A_{\rm ref}$")
    ax.grid(alpha=0.3, axis="y")

    # (f) chirality distribution
    ax = fig.add_subplot(gs[1, 2])
    chis_flat = result["chis"].flatten()
    even_par = np.cos(chis_flat) >= 0
    ax.hist(chis_flat[even_par], bins=40, range=(-PI, PI),
            color="#16a085", alpha=0.7,
            label=f"even / bosonic ({int(even_par.sum())})")
    ax.hist(chis_flat[~even_par], bins=40, range=(-PI, PI),
            color="#e74c3c", alpha=0.7,
            label=f"odd / fermionic ({int((~even_par).sum())})")
    ax.set_xlabel("χ"); ax.set_ylabel("count")
    ax.set_title("Chirality distribution  ·  parity split")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (g) single shell n=4 with E-field
    ax = fig.add_subplot(gs[2, 0], projection="3d")
    plot_shell_field(ax, shells[3], surf_E[3], "E-field")

    # (h) single shell n=4 with B-field
    ax = fig.add_subplot(gs[2, 1], projection="3d")
    plot_shell_field(ax, shells[3], surf_B[3], "B-field")

    # (i) nested shells
    ax = fig.add_subplot(gs[2, 2], projection="3d")
    plot_nested_shells(ax, shells, surf_E, alpha=0.4)

    plt.suptitle(
        f"Spin MRI  ·  finite-step 2D-to-3D  ·  "
        f"N_SPIN = {N_SPIN_DEFAULT}  ·  {N_shells} shells  ·  "
        f"720-cell parity  ·  M-matrix + Theorem 13  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ============================================================
    #  Figure 2: field profile on σ at select shells
    # ============================================================
    fig2 = plt.figure(figsize=(14, 6))
    gs2 = GridSpec(1, 2, figure=fig2, hspace=0.35, wspace=0.30)

    ax = fig2.add_subplot(gs2[0, 0])
    for idx, sh in enumerate(shells):
        c = plt.cm.viridis(idx / max(N_shells - 1, 1))
        ax.plot(result["sinogram_E"][idx], color=c, lw=0.9,
                label=f"n={sh['n']} (A={sh['scale']:.2f})")
    ax.set_xlabel("σ"); ax.set_ylabel("E(σ)")
    ax.set_title("E-field profile on σ  ·  per shell")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    ax = fig2.add_subplot(gs2[0, 1])
    for idx, sh in enumerate(shells):
        c = plt.cm.plasma(idx / max(N_shells - 1, 1))
        ax.plot(result["sinogram_B"][idx], color=c, lw=0.9,
                label=f"n={sh['n']} (A={sh['scale']:.2f})")
    ax.set_xlabel("σ"); ax.set_ylabel("B(σ)")
    ax.set_title("B-field profile on σ  ·  per shell")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    plt.suptitle(
        "Spin MRI  ·  field profiles across the finite-step shells",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    demo()