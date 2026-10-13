#!/usr/bin/env python3
"""
spin_mri_unified.py
===================
Unified spin MRI pipeline.

New components
--------------
· Forward model uses tr M(x, y) = x^(i−1) + y^(i−1)   (gpu_64_bit.py)
     sym = Re(tr) → E-field (fermionic, conservative)
     asym = Im(tr) → B-field (bosonic, oscillatory)

· Sinogram reconstruction uses square-free FFT   (photo_compress.py)
     compress    : keep only μ(n) ≠ 0 bins in the spin axis
     decompress  : fill gaps via M-matrix curve (Möbius on inverses)
     accuracy    : PSNR(recon_full, recon_compressed)

· Objects:
     phantom mode       : Shepp-Logan (2D→3D via Radon)
     finite-step mode   : nested organic shells from 3d_object_finite_step
"""

from __future__ import annotations
import math
from functools import partial
from itertools import permutations
from typing import Dict, List, Tuple

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
a_c   = ALPHA
H_CONE = 3.0
N_SPIN_DEFAULT = int(math.ceil(180 * E))       # 490


# ============================================================
#  Möbius sieve  ·  O(N log log N)
# ============================================================
def mobius_sieve(N: int) -> np.ndarray:
    if N < 1:
        return np.zeros(N + 1, dtype=np.int8)
    mu = np.ones(N + 1, dtype=np.int8)
    is_comp = np.zeros(N + 1, dtype=bool)
    for i in range(2, N + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= N:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


def square_free_indices(N: int) -> np.ndarray:
    mu = mobius_sieve(N)
    keep = [0]
    keep += [n for n in range(1, N) if mu[n] != 0]
    return np.array(keep, dtype=np.int32)


# ============================================================
#  M-matrix trace  (from gpu_64_bit.py)
# ============================================================
def m_trace_array(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """tr_k = x_k^(i−1) + y_k^(i−1)   ·   vectorized."""
    x_ = x_arr.astype(np.complex128) + 1e-9
    y_ = y_arr.astype(np.complex128) + 1e-9
    return x_ ** (1j - 1.0) + y_ ** (1j - 1.0)


# ============================================================
#  |Ci| 64-bit layout  (from gpu_64_bit.py)
# ============================================================
class MobiusLinAlg64:
    """
        data[K + k]   symmetric     Re(tr M_k)
        data[K − k]   asymmetric    Im(tr M_k)
        data[K]       DC term
    """
    __slots__ = ("K", "data")

    def __init__(self, K: int = 256):
        self.K = int(K)
        self.data = np.zeros(2 * self.K + 1, dtype=np.float64)

    def set_trace(self, k: int, tr: complex) -> None:
        if 1 <= k <= self.K:
            self.data[self.K + k] = tr.real
            self.data[self.K - k] = tr.imag

    def set_dc(self, v: float) -> None:
        self.data[self.K] = float(v)

    def supertrace(self) -> float:
        signs = np.where(np.arange(self.data.size) & 1, -1.0, 1.0)
        return float(np.dot(signs, self.data))

    def sym_energy(self) -> float:
        return float(np.sum(self.data[self.K + 1:2 * self.K + 1] ** 2))

    def asym_energy(self) -> float:
        return float(np.sum(self.data[:self.K] ** 2))


# ============================================================
#  720-cell parity  (from 360.py)
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
N_EVEN = int((SIGNS > 0).sum())     # 360
N_ODD  = int((SIGNS < 0).sum())     # 360


# ============================================================
#  Finite-step scaling
# ============================================================
def A_of_n(n: float) -> float:
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
#  Organic warped grid → warped sphere
# ============================================================
def organic_uv(nu=40, nv=40, warp=0.18):
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, V = np.meshgrid(u, v, indexing="ij")
    Uw = U + warp * np.sin(2 * PI * V + 0.4) \
           + 0.06 * np.cos(4 * PI * U + 0.9)
    Vw = V + warp * np.sin(2 * PI * U + 1.2) \
           + 0.06 * np.cos(4 * PI * V + 0.2)
    return U, V, Uw, Vw


def organic_radius(U, V, amp=0.20, freq=3.0, phase=0.0):
    return (1.0 + amp * np.sin(2 * PI * freq * U + phase)
                 * np.sin(2 * PI * freq * V + phase))


def shell_xyz(Uw, Vw, R=1.0, radius_mod=None):
    theta = 2 * PI * Uw
    phi   = PI * Vw
    r = (np.full_like(Uw, R, dtype=float)
         if radius_mod is None else R * radius_mod(Uw, Vw))
    return (r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi))


def build_nested_shells(n_layers=6, nu=40, nv=40,
                        warp=0.18, mod_amp=0.22, mod_freq=3.0):
    U, V, Uw, Vw = organic_uv(nu, nv, warp)
    ref_step = abs(finite_step(4))
    shells = []
    for n in range(1, n_layers + 1):
        R = A_of_n(n)
        step = finite_step(n, "forward")
        amp = mod_amp * (step / ref_step)
        mod = partial(organic_radius, amp=amp,
                      freq=mod_freq, phase=0.10 * n)
        X, Y, Z = shell_xyz(Uw, Vw, R=R, radius_mod=mod)
        shells.append(dict(n=n, scale=R, step=step, amp=amp,
                           U=U, V=V, Uw=Uw, Vw=Vw,
                           X=X, Y=Y, Z=Z))
    return shells


# ============================================================
#  Shepp-Logan phantom + Radon
# ============================================================
def shepp_logan(N: int = 64) -> np.ndarray:
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    img = np.zeros_like(X)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0.0, 1.0),
        (0.0, -0.0184, 0.6624, 0.874, 0.0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18.0, -0.2),
        (-0.22, 0.0, 0.16, 0.41, 18.0, -0.2),
        (0.0, 0.35, 0.21, 0.25, 0.0, 0.1),
        (0.0, 0.10, 0.046, 0.046, 0.0, 0.1),
        (0.0, -0.10, 0.046, 0.046, 0.0, 0.1),
        (-0.08, -0.605, 0.046, 0.023, 0.0, 0.1),
        (0.0, -0.606, 0.023, 0.023, 0.0, 0.1),
        (0.06, -0.605, 0.023, 0.046, 0.0, 0.1),
    ]
    for cx, cy, a, b, th_deg, val in ellipses:
        th = math.radians(th_deg)
        Xr =  (X - cx) * math.cos(th) + (Y - cy) * math.sin(th)
        Yr = -(X - cx) * math.sin(th) + (Y - cy) * math.cos(th)
        img[((Xr / a) ** 2 + (Yr / b) ** 2) <= 1.0] += val
    img -= img.min()
    return img / (img.max() + 1e-12)


def radon_1d(img, angle, N_proj):
    N = img.shape[0]
    c, s = math.cos(angle), math.sin(angle)
    t = np.linspace(-1, 1, N_proj)
    sj = np.linspace(-1, 1, N)
    ds = 2.0 / N
    T, S = np.meshgrid(t, sj, indexing="ij")
    X = T * c - S * s
    Y = T * s + S * c
    Xi = np.clip(((X + 1) * 0.5 * (N - 1)).astype(int), 0, N - 1)
    Yi = np.clip(((Y + 1) * 0.5 * (N - 1)).astype(int), 0, N - 1)
    p = img[Yi, Xi].sum(axis=1) * ds
    p = np.clip(p, 0, None)
    return p / (p.max() + 1e-12)


# ============================================================
#  Spinor family
# ============================================================
def make_base_Z(seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))


Z_BASE = make_base_Z()


def build_spin_family(N):
    sigmas = np.linspace(0.0, 1.0, N)
    return np.array([Z_BASE * np.exp(1j * (PI / 3.0) * s)
                     for s in sigmas])


def levi_civita(Z):
    return complex(np.linalg.det(Z))


def chirality_of(Z):
    Pi = levi_civita(Z)
    return math.atan2(Pi.imag, Pi.real), abs(Pi)


# ============================================================
#  Spectral derivative
# ============================================================
def spectral_derivative(t, C):
    K = (len(C) - 1) // 2
    j = np.arange(-K, K + 1)
    j_nz = j[j != 0]
    phases = j_nz * t / ALPHA
    Cj = C[K + j_nz]
    return float(-np.sum((j_nz / ALPHA) * np.real(Cj * np.exp(1j * phases))))


# ============================================================
#  Trace-based forward projection
# ============================================================
def trace_forward(Z_init, layer_couplings, layer_scales,
                  phase_gain=0.4, amp_gain=0.15) -> Dict:
    """
    Forward model using tr M(x, y) = x^(i−1) + y^(i−1).

    For each layer m with coupling g_m:
        1. propagate the spin family through the layer
        2. for every spin point k, compute Π(Z_k) = det(Z_k)
        3. build the trace from the complex scalar:
              x_k = |Re Π_k| + ε,  y_k = |Im Π_k| + ε
              tr_k = x_k^(i−1) + y_k^(i−1)
        4. gated by g_m:
              E[m, k] = g_m · Re(tr_k)      (fermionic)
              B[m, k] = g_m · Im(tr_k)      (bosonic)
    """
    N = Z_init.shape[0]
    n_layers = len(layer_couplings)

    dets = np.array([levi_civita(Z) for Z in Z_init])
    K = N // 2
    C_arr = np.zeros(2 * K + 1, dtype=complex)
    for j in range(-K, K + 1):
        C_arr[K + j] = dets[j % N]

    t_vals = np.arange(N) - K

    Z_cur = Z_init.copy()
    E_out = np.zeros((n_layers, N))
    B_out = np.zeros((n_layers, N))
    parities = np.zeros((n_layers, N))
    chis_all = np.zeros((n_layers, N))
    mags_all = np.zeros((n_layers, N))
    tr_sym_all = np.zeros((n_layers, N))
    tr_asym_all = np.zeros((n_layers, N))
    g_ns = np.zeros(n_layers)

    for idx in range(n_layers):
        g_n = float(layer_couplings[idx])
        R_n = float(layer_scales[idx])
        zeta_dot = spectral_derivative(idx, C_arr)

        zc = float(np.clip(abs(R_n), 0.0, H_CONE))
        zn = zc / H_CONE
        r = 0.08 + (0.5 - 0.08) * (1.0 - zn ** 2) ** 0.6
        r = max(r, 1e-6)
        log_r_t = np.clip(t_vals * math.log(r), -30.0, 30.0)

        M_vals = np.exp(log_r_t) * np.exp(1j * ALPHA * t_vals * zc)
        phase_mod = np.exp(1j * phase_gain * g_n
                           * zeta_dot * t_vals / N)
        amp_mod = 1.0 + amp_gain * g_n * np.tanh(t_vals / N)

        Z_cur = Z_cur * (M_vals * phase_mod * amp_mod)[:, None, None]

        # --- extract trace per spin point ---
        x_arr = np.zeros(N)
        y_arr = np.zeros(N)
        chis = np.zeros(N)
        mags = np.zeros(N)
        for k in range(N):
            Pi_k = levi_civita(Z_cur[k])
            chis[k] = math.atan2(Pi_k.imag, Pi_k.real)
            mags[k] = abs(Pi_k)
            x_arr[k] = abs(Pi_k.real) + 1e-6
            y_arr[k] = abs(Pi_k.imag) + 1e-6

        tr_arr = m_trace_array(x_arr, y_arr)

        E_out[idx] = g_n * tr_arr.real
        B_out[idx] = g_n * tr_arr.imag
        parities[idx] = (np.cos(chis) >= 0).astype(int)
        chis_all[idx] = chis
        mags_all[idx] = mags
        tr_sym_all[idx] = tr_arr.real
        tr_asym_all[idx] = tr_arr.imag
        g_ns[idx] = g_n

    return dict(E=E_out, B=B_out, parities=parities,
                chis=chis_all, mags=mags_all,
                tr_sym=tr_sym_all, tr_asym=tr_asym_all, g_ns=g_ns)


# ============================================================
#  Square-free FFT reconstruction  (from photo_compress.py)
# ============================================================
def _m_algebraic_interp(ma, mb, t):
    if ma <= 1e-12 or mb <= 1e-12:
        return (1 - t) * ma + t * mb
    return 1.0 / ((1 - t) / ma + t / mb)


def _fill_gap_mcurve(k, left, right, C_left, C_right):
    t = (k - left) / max(right - left, 1)
    ma = abs(C_left); pa = math.atan2(C_left.imag, C_left.real)
    mb = abs(C_right); pb = math.atan2(C_right.imag, C_right.real)
    dp = pb - pa
    while dp >  math.pi: dp -= 2 * math.pi
    while dp < -math.pi: dp += 2 * math.pi

    m_alg = _m_algebraic_interp(ma, mb, t)
    s_a = math.sin(math.log(max(ma, 1e-12)))
    s_b = math.sin(math.log(max(mb, 1e-12)))
    s_t = math.sin(math.log(max(m_alg, 1e-12)))
    shape = 1.0 + 0.1 * (s_t - (1 - t) * s_a - t * s_b)
    mag = m_alg * shape
    phase = pa + t * dp
    return mag * complex(math.cos(phase), math.sin(phase))


def compress_sinogram(sinogram: np.ndarray, N_spin: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Square-free FFT compression along the spin axis:
        C = FFT(sinogram, axis=1)
        keep C[:, μ(n) ≠ 0]
    """
    C = np.fft.fft(sinogram, axis=1)
    sqfree = square_free_indices(N_spin)
    return C[:, sqfree], sqfree


def decompress_sinogram(C_comp: np.ndarray,
                        sqfree: np.ndarray,
                        n_layers: int, N_spin: int) -> np.ndarray:
    """Fill non-square-free bins via the M-matrix curve."""
    C = np.zeros((n_layers, N_spin), dtype=complex)
    C[:, sqfree] = C_comp
    sqset = set(int(s) for s in sqfree)
    sq_sorted = np.sort(sqfree)
    for row in range(n_layers):
        for k in range(N_spin):
            if k in sqset:
                continue
            ir = np.searchsorted(sq_sorted, k)
            l = int(sq_sorted[ir - 1]) if ir > 0 else None
            r = int(sq_sorted[ir])     if ir < sq_sorted.size else None
            if l is None and r is None:
                continue
            if l is None:
                C[row, k] = C[row, r]
            elif r is None:
                C[row, k] = C[row, l]
            else:
                C[row, k] = _fill_gap_mcurve(k, l, r, C[row, l], C[row, r])
    return np.fft.ifft(C, axis=1).real


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    span = float(a.max() - a.min())
    return 20.0 * math.log10(max(span, 1e-9) / math.sqrt(mse))


# ============================================================
#  Filtered backprojection  (for phantom mode)
# ============================================================
def ramp_filter(sino: np.ndarray) -> np.ndarray:
    N_proj = sino.shape[1]
    f = np.fft.fftfreq(N_proj)
    ramp = 2.0 * np.abs(f)
    out = np.zeros_like(sino)
    for i in range(sino.shape[0]):
        F = np.fft.fft(sino[i])
        out[i] = np.real(np.fft.ifft(F * ramp))
    return out


def backproject(sino: np.ndarray, N: int, angles: np.ndarray) -> np.ndarray:
    N_ang, N_proj = sino.shape
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    img = np.zeros((N, N))
    for a_idx, theta in enumerate(angles):
        s = X * math.cos(theta) + Y * math.sin(theta)
        s_idx = np.clip(((s + 1) * 0.5 * (N_proj - 1)).astype(int),
                        0, N_proj - 1)
        img += sino[a_idx][s_idx]
    return img / N_ang


# ============================================================
#  Pipeline: phantom mode
# ============================================================
def pipeline_phantom(image: np.ndarray,
                     N_angles: int = 72,
                     N_layers: int = 32,
                     N_spin: int = N_SPIN_DEFAULT) -> Dict:
    Z_init = build_spin_family(N_spin)
    angles = np.linspace(0, PI, N_angles, endpoint=False)
    scales = np.linspace(0.1, H_CONE, N_layers)

    sino_E = np.zeros((N_angles, N_spin))
    sino_B = np.zeros((N_angles, N_spin))
    pari   = np.zeros((N_angles, N_spin))

    for a_idx, theta in enumerate(angles):
        ray = radon_1d(image, theta, N_layers)
        out = trace_forward(Z_init, ray, scales)
        sino_E[a_idx] = out["E"][-1]      # final layer output
        sino_B[a_idx] = out["B"][-1]
        pari[a_idx]   = out["parities"][-1]

    # full-resolution reconstruction
    img_E_full = backproject(ramp_filter(sino_E), image.shape[0], angles)
    img_B_full = backproject(ramp_filter(sino_B), image.shape[0], angles)

    # square-free compressed reconstruction
    C_E, sqfree = compress_sinogram(sino_E, N_spin)
    C_B, _      = compress_sinogram(sino_B, N_spin)
    sino_E_hat  = decompress_sinogram(C_E, sqfree, N_angles, N_spin)
    sino_B_hat  = decompress_sinogram(C_B, sqfree, N_angles, N_spin)

    img_E_comp = backproject(ramp_filter(sino_E_hat), image.shape[0], angles)
    img_B_comp = backproject(ramp_filter(sino_B_hat), image.shape[0], angles)

    return dict(
        image=image, angles=angles,
        sinogram_E=sino_E, sinogram_B=sino_B, parities=pari,
        image_E_full=img_E_full, image_B_full=img_B_full,
        image_E_comp=img_E_comp, image_B_comp=img_B_comp,
        sqfree=sqfree, N_spin=N_spin, N_layers=N_layers,
        psnr_E_full=psnr(image, img_E_full),
        psnr_B_full=psnr(image, img_B_full),
        psnr_E_comp=psnr(img_E_full, img_E_comp),
        psnr_B_comp=psnr(img_B_full, img_B_comp),
    )


# ============================================================
#  Pipeline: finite-step mode
# ============================================================
def pipeline_finite_step(n_layers=6, nu=40, nv=40,
                         N_spin=N_SPIN_DEFAULT) -> Dict:
    shells = build_nested_shells(n_layers=n_layers, nu=nu, nv=nv)

    # couplings and scales per shell
    A_ref = max(abs(sh["scale"]) for sh in shells) + 1e-9
    couplings = np.array([abs(sh["step"]) / A_ref for sh in shells])
    scales    = np.array([sh["scale"] for sh in shells])

    Z_init = build_spin_family(N_spin)
    out = trace_forward(Z_init, couplings, scales)

    # square-free compressed version
    C_E, sqfree = compress_sinogram(out["E"], N_spin)
    C_B, _      = compress_sinogram(out["B"], N_spin)
    E_hat = decompress_sinogram(C_E, sqfree, n_layers, N_spin)
    B_hat = decompress_sinogram(C_B, sqfree, n_layers, N_spin)

    # map fields onto the warped shell surfaces
    def field_to_surface(f_1d, nu, nv):
        idx = np.arange(nu * nv) % len(f_1d)
        return f_1d[idx].reshape(nu, nv)

    surf_E_full, surf_B_full = [], []
    surf_E_comp, surf_B_comp = [], []
    for i in range(n_layers):
        surf_E_full.append(field_to_surface(out["E"][i], nu, nv))
        surf_B_full.append(field_to_surface(out["B"][i], nu, nv))
        surf_E_comp.append(field_to_surface(E_hat[i], nu, nv))
        surf_B_comp.append(field_to_surface(B_hat[i], nu, nv))

    return dict(
        shells=shells, couplings=couplings, scales=scales,
        sinogram_E=out["E"], sinogram_B=out["B"],
        sinogram_E_comp=E_hat, sinogram_B_comp=B_hat,
        parities=out["parities"], chis=out["chis"],
        g_ns=out["g_ns"], sqfree=sqfree,
        surf_E_full=surf_E_full, surf_B_full=surf_B_full,
        surf_E_comp=surf_E_comp, surf_B_comp=surf_B_comp,
        psnr_E=psnr(out["E"], E_hat),
        psnr_B=psnr(out["B"], B_hat),
        nu=nu, nv=nv, N_spin=N_spin, n_layers=n_layers,
    )


# ============================================================
#  Visualization
# ============================================================
def plot_phantom(result):
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(result["image"], cmap="gray", origin="lower")
    ax.set_title("Phantom"); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(result["sinogram_E"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ (spin)"); ax.set_ylabel("angle")
    ax.set_title("E sinogram  ·  Re tr M")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(result["sinogram_B"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ"); ax.set_ylabel("angle")
    ax.set_title("B sinogram  ·  Im tr M")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(result["image_E_full"], cmap="inferno", origin="lower")
    ax.set_title(f"E reconstruction  ·  PSNR = {result['psnr_E_full']:.2f} dB")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(result["image_E_comp"], cmap="inferno", origin="lower")
    ax.set_title(f"E compressed  ·  PSNR = {result['psnr_E_comp']:.2f} dB")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(result["parities"], aspect="auto",
                   origin="lower", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("σ"); ax.set_ylabel("angle")
    ax.set_title(f"Parity  ·  ⟨even⟩ = "
                 f"{result['parities'].mean():.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Phantom  ·  trace forward (gpu_64_bit)  ·  "
        f"square-free FFT recon (photo_compress)  ·  "
        f"N_spin = {result['N_spin']}  ·  α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_finite_step(result):
    shells = result["shells"]
    n_layers = result["n_layers"]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ns = np.arange(1, 10)
    A  = np.array([A_of_n(k)      for k in ns])
    dA = np.array([finite_step(k) for k in ns])
    ax.plot(ns, A, "o-", color="#2c3e50", lw=1.6, label="A(n)")
    ax.bar(ns + 0.15, dA, width=0.3, color="#e67e22",
           alpha=0.75, label=r"$\Delta A_n$")
    ax.axvline(4, color="#e74c3c", ls=":", lw=1.2,
               label="n=4  (reference)")
    ax.set_xlabel("n"); ax.set_ylabel("A / ΔA")
    ax.set_title("Finite-step scaling")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(result["sinogram_E"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ"); ax.set_ylabel("shell n")
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([sh["n"] for sh in shells])
    ax.set_title("E  ·  Re tr M  (full)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(result["sinogram_E_comp"], aspect="auto",
                   origin="lower", cmap="RdBu_r")
    ax.set_xlabel("σ"); ax.set_ylabel("shell n")
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([sh["n"] for sh in shells])
    ax.set_title(f"E  ·  square-free FFT recon  ·  "
                 f"PSNR = {result['psnr_E']:.2f} dB")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, 0])
    ax.bar(np.arange(n_layers), result["g_ns"],
           color="#8e44ad", edgecolor="k")
    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels([sh["n"] for sh in shells])
    ax.set_xlabel("shell n"); ax.set_ylabel("g_n")
    ax.set_title(r"Coupling  $g_n = |\Delta A_n|/A_{\rm ref}$")
    ax.grid(alpha=0.3, axis="y")

    # single shell n=4 with E field (full)
    ax = fig.add_subplot(gs[1, 1], projection="3d")
    sh = shells[3]
    sE = result["surf_E_full"][3]
    norm = (sE - sE.min()) / (sE.max() - sE.min() + 1e-12)
    ax.plot_surface(sh["X"], sh["Y"], sh["Z"],
                    facecolors=plt.cm.inferno(norm),
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(f"E  ·  shell n=4  ·  A={sh['scale']:.2f}\n"
                 f"ΔA={sh['step']:+.2f}")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)

    # single shell n=4 with E field (compressed)
    ax = fig.add_subplot(gs[1, 2], projection="3d")
    sE = result["surf_E_comp"][3]
    norm = (sE - sE.min()) / (sE.max() - sE.min() + 1e-12)
    ax.plot_surface(sh["X"], sh["Y"], sh["Z"],
                    facecolors=plt.cm.inferno(norm),
                    rstride=1, cstride=1,
                    linewidth=0, antialiased=True, alpha=0.95)
    ax.set_title(f"E compressed  ·  shell n=4\n"
                 f"PSNR = {result['psnr_E']:.2f} dB")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)

    plt.suptitle(
        f"Finite-step object  ·  trace forward  ·  "
        f"square-free FFT reconstruction  ·  "
        f"N_spin = {result['N_spin']}  ·  α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Unified spin MRI  ·  gpu_64_bit trace + photo_compress reconstruction")
    print("=" * 78)
    print(f"  α = 1/(π − e)          = {ALPHA:.6f}")
    print(f"  N_SPIN = ⌈180·e⌉       = {N_SPIN_DEFAULT}")
    print(f"  720-cell               : A_6 = {N_EVEN} (bosonic), "
          f"odd = {N_ODD} (fermionic)")
    print()

    # ---------------- phantom mode ----------------
    print("=" * 78)
    print("Phantom mode  ·  trace forward + square-free FFT recon")
    print("=" * 78)
    phantom = shepp_logan(64)
    res_p = pipeline_phantom(phantom,
                             N_angles=72, N_layers=32,
                             N_spin=N_SPIN_DEFAULT)
    print(f"  N_angles               = 72")
    print(f"  N_layers               = 32")
    print(f"  kept square-free bins  = {res_p['sqfree'].size} / "
          f"{res_p['N_spin']}  "
          f"= {res_p['sqfree'].size / res_p['N_spin']:.4f}")
    print(f"  PSNR E full            = {res_p['psnr_E_full']:>6.2f} dB")
    print(f"  PSNR E compressed      = {res_p['psnr_E_comp']:>6.2f} dB")
    print(f"  PSNR B full            = {res_p['psnr_B_full']:>6.2f} dB")
    print(f"  PSNR B compressed      = {res_p['psnr_B_comp']:>6.2f} dB")
    print()

    plot_phantom(res_p)

    # ---------------- finite-step mode ----------------
    print("=" * 78)
    print("Finite-step mode  ·  nested organic shells")
    print("=" * 78)
    res_f = pipeline_finite_step(n_layers=6, nu=40, nv=40,
                                 N_spin=N_SPIN_DEFAULT)
    print(f"  shells                 = 6")
    print(f"  kept square-free bins  = {res_f['sqfree'].size} / "
          f"{res_f['N_spin']}")
    print(f"  PSNR E (full vs comp)  = {res_f['psnr_E']:>6.2f} dB")
    print(f"  PSNR B (full vs comp)  = {res_f['psnr_B']:>6.2f} dB")
    print()

    plot_finite_step(res_f)

    print("Done.")


if __name__ == "__main__":
    demo()