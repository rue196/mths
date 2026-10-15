#!/usr/bin/env python3
"""
dynamic_hertz_grid.py
=====================
Even-even dynamic Hertz grid with:
    · coupling strength matrix     g_ij = w_ij · κ_ij
    · dynamic Hertz frequency     H_i = Σ_j |g_ij · ω_ij|
    · even-even mask              σ(v) ≡ 0 (mod 2)
    · symmetric path              O(K log log K)  (linear sieve)
    · asymmetric refresh          O(K log K)      (Möbius convolution)
    · photo compression           square-free FFT (μ(n) ≠ 0)
    · even/odd parity             color scattering
    · full wavelength pipeline    fiber kernel exp(−α|z|)

Constants:
    α = 1/(π − e) ≈ 2.362338
    a_step = α / 0.3628 ≈ 6.511
    density = 6/π² ≈ 0.607927
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)                # ≈ 2.362338
A_STEP  = ALPHA / 0.3628                # ≈ 6.511
DENSITY = 6.0 / (PI * PI)               # ≈ 0.607927
PHI     = (1.0 + math.sqrt(5.0)) / 2.0
W1, W2  = PI, E


# ============================================================
#  Möbius sieve  ·  O(K log log K)  symmetric path
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  Harmonic numbers  ·  O(K)
# ============================================================
def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


# ============================================================
#  Möbius convolution  F = μ * H  ·  O(K log K)  asymmetric
# ============================================================
def mu_convolution_H(K: int):
    mu = mobius_sieve(K)
    H  = harmonic_numbers(K)
    F  = np.zeros(K + 1, dtype=float)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = int(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F, mu, H


# ============================================================
#  Square-free FFT compression  ·  from photo_compress.py
# ============================================================
def square_free_indices(N: int) -> np.ndarray:
    mu = mobius_sieve(N)
    keep = [0]
    keep += [n for n in range(1, N) if mu[n] != 0]
    return np.array(keep, dtype=np.int32)


def _m_algebraic_interp(ma: float, mb: float, t: float) -> float:
    if ma <= 1e-12 or mb <= 1e-12:
        return (1 - t) * ma + t * mb
    return 1.0 / ((1 - t) / ma + t / mb)


def _fill_gap_mcurve(k: int, left: int, right: int,
                     C_left: complex, C_right: complex) -> complex:
    t = (k - left) / max(right - left, 1)
    ma = abs(C_left); pa = math.atan2(C_left.imag, C_left.real)
    mb = abs(C_right); pb = math.atan2(C_right.imag, C_right.real)
    dp = pb - pa
    while dp >  PI: dp -= 2 * PI
    while dp < -PI: dp += 2 * PI

    m_alg = _m_algebraic_interp(ma, mb, t)
    s_a = math.sin(math.log(max(ma, 1e-12)))
    s_b = math.sin(math.log(max(mb, 1e-12)))
    s_t = math.sin(math.log(max(m_alg, 1e-12)))
    shape = 1.0 + 0.1 * (s_t - (1 - t) * s_a - t * s_b)

    mag = m_alg * shape
    phase = pa + t * dp
    return mag * complex(math.cos(phase), math.sin(phase))


def compress_signal(signal: np.ndarray) -> Dict:
    """Square-free FFT compression of a 1D signal."""
    N = signal.size
    C = np.fft.fft(signal)
    sqfree = square_free_indices(N)
    return {
        'real': C[sqfree].real.copy(),
        'imag': C[sqfree].imag.copy(),
        'sqfree': sqfree,
        'N': N,
    }


def decompress_signal(payload: Dict) -> np.ndarray:
    """M-curve gap-fill reconstruction."""
    N = payload['N']
    sqfree = payload['sqfree']
    C_hat = np.zeros(N, dtype=complex)
    C_hat[sqfree] = payload['real'] + 1j * payload['imag']

    sqset = set(int(s) for s in sqfree)
    sq_sorted = np.sort(sqfree)
    for k in range(N):
        if k in sqset:
            continue
        ir = np.searchsorted(sq_sorted, k)
        l = int(sq_sorted[ir - 1]) if ir > 0 else None
        r = int(sq_sorted[ir])     if ir < sq_sorted.size else None
        if l is None and r is None:
            continue
        if l is None:
            C_hat[k] = C_hat[r]
        elif r is None:
            C_hat[k] = C_hat[l]
        else:
            C_hat[k] = _fill_gap_mcurve(k, l, r, C_hat[l], C_hat[r])
    return np.fft.ifft(C_hat).real


# ============================================================
#  Coupling strength matrix  ·  g_ij = w_ij · κ_ij
# ============================================================
def chiral_angles(t: np.ndarray) -> np.ndarray:
    lt = np.log(np.maximum(np.abs(t), 1e-12))
    return (lt + PI) % (2 * PI) - PI


def travel_lengths(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    dr = np.diff(r, prepend=r[0])
    ds = np.sqrt(dt * dt + dr * dr)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-6


def coupling_matrix(t: np.ndarray, r: np.ndarray,
                    L_ref: Optional[float] = None) -> np.ndarray:
    """g_ij = w_ij · κ_ij."""
    chi = chiral_angles(t)
    dchi = np.abs(chi[:, None] - chi[None, :])
    dchi = (dchi + PI) % (2 * PI) - PI
    W = np.abs(dchi) / PI

    L = travel_lengths(t, r)
    if L_ref is None:
        L_ref = float(np.median(L[L > 0])) if np.any(L > 0) else 1.0
    K = np.sqrt(np.maximum(L[:, None] * L[None, :], 0.0)) / L_ref
    K = np.clip(K, 0.0, 1.0)

    G = W * K
    np.fill_diagonal(G, 0.0)
    return G


def even_even_mask(N: int) -> np.ndarray:
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2 == 0) & (j % 2 == 0)


# ============================================================
#  Dynamic Hertz frequency field
# ============================================================
def dynamic_hertz(G: np.ndarray,
                  dzeta_dt: float = 0.5,
                  L_ref: Optional[float] = None) -> np.ndarray:
    """
    H_i = Σ_j |g_ij · ω_ij|

        ω_ij(t) = α (j − i) dζ/dt(t)      (Theorem 13, supertrace_6_kernel.pdf)

    Each cell i's Hertz frequency is the sum of its couplings to all
    neighbours, weighted by the local fermionic oscillation rate.
    """
    N = G.shape[0]
    j_idx = np.arange(N)
    omega = ALPHA * (j_idx[None, :] - j_idx[:, None]) * dzeta_dt
    H = np.sum(np.abs(G * omega), axis=1)
    return H


# ============================================================
#  Fiber kernel  ·  exp(−α|z|)  wavelength pipeline
# ============================================================
def fiber_kernel(width: int, alpha: float = ALPHA) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    h = np.exp(-alpha * np.abs(x))
    return h / h.sum()


def fiber_propagate(signal: np.ndarray, width: int = 41) -> np.ndarray:
    """Convolve the signal with the fiber kernel via FFT."""
    h = fiber_kernel(width)
    L = len(signal)
    N = 1 << (2 * L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L))
    ker_pad = np.pad(h, (0, N - len(h)))
    y = np.fft.ifft(np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:L]
    return np.real(y)


# ============================================================
#  Even-odd parity  ·  color scattering
# ============================================================
def parity_channels(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the Hertz field into even/odd parity channels.

        even index → R channel (bosonic, warm scatter)
        odd  index → B channel (fermionic, cool scatter)
    """
    N = H.size
    H_even = np.zeros(N)
    H_odd  = np.zeros(N)
    H_even[0::2] = H[0::2]
    H_odd [1::2] = H[1::2]
    return H_even, H_odd


# ============================================================
#  Full pipeline
# ============================================================
def pipeline(N: int = 128,
             seed: int = 42,
             dzeta_dt: float = 0.5,
             verbose: bool = True) -> Dict:
    if verbose:
        print("=" * 78)
        print("Dynamic Hertz even-even grid  ·  square-free FFT  ·  fiber pipeline")
        print("=" * 78)
        print(f"  α       = 1/(π − e)      = {ALPHA:.6f}")
        print(f"  a_step  = α / 0.3628     = {A_STEP:.6f}")
        print(f"  6/π²                     = {DENSITY:.6f}")
        print(f"  N                        = {N}")
        print()

    # ---------- 1. events ----------
    rng = np.random.default_rng(seed)
    t = np.sort(rng.exponential(scale=3.0, size=N))
    t = np.maximum(t, 1e-3)
    r = np.arange(1, N + 1, dtype=float)

    # ---------- 2. coupling strength matrix ----------
    t0 = time.perf_counter()
    G = coupling_matrix(t, r)
    t_coupling = (time.perf_counter() - t0) * 1e3
    if verbose:
        off = G[G > 0]
        print(f"[1] coupling matrix  g_ij = w_ij · κ_ij")
        print(f"      ⟨g⟩                = {off.mean():.6f}")
        print(f"      max g              = {G.max():.6f}")
        print(f"      build time         = {t_coupling:.2f} ms")

    # ---------- 3. dynamic Hertz field ----------
    t0 = time.perf_counter()
    H = dynamic_hertz(G, dzeta_dt=dzeta_dt)
    t_hertz = (time.perf_counter() - t0) * 1e3
    if verbose:
        print(f"[2] dynamic Hertz field  H_i = Σ_j |g_ij · ω_ij|")
        print(f"      range              = [{H.min():.4f}, {H.max():.4f}]")
        print(f"      mean               = {H.mean():.6f}")
        print(f"      build time         = {t_hertz:.2f} ms")

    # ---------- 4. even-even mask ----------
    mask = even_even_mask(N)
    even_idx = np.where(np.arange(N) % 2 == 0)[0]
    H_even_grid = H[even_idx]
    if verbose:
        print(f"[3] even-even mask")
        print(f"      retained indices   = {len(even_idx)} / {N}")
        print(f"      density            = {len(even_idx) / N:.4f}")
        print(f"      H_even mean        = {H_even_grid.mean():.6f}")

    # ---------- 5. symmetric path  ·  O(K log log K) ----------
    t0 = time.perf_counter()
    mu_sym = mobius_sieve(N)
    t_sym = (time.perf_counter() - t0) * 1e3
    if verbose:
        nz = int(np.count_nonzero(mu_sym[1:N + 1]))
        print(f"[4] symmetric path  ·  linear sieve  O(K log log K)")
        print(f"      μ non-zero         = {nz} / {N}  "
              f"= {nz / N:.4f}  (target {DENSITY:.4f})")
        print(f"      sieve time         = {t_sym:.2f} ms")

    # ---------- 6. asymmetric refresh  ·  O(K log K) ----------
    t0 = time.perf_counter()
    F, mu_asym, Harm = mu_convolution_H(N)
    t_asym = (time.perf_counter() - t0) * 1e3
    if verbose:
        print(f"[5] asymmetric refresh  ·  Möbius convolution  O(K log K)")
        print(f"      F range            = [{F[1:].min():+.4f}, "
              f"{F[1:].max():+.4f}]")
        print(f"      conv time          = {t_asym:.2f} ms")

    # ---------- 7. photo compression  ·  square-free FFT ----------
    t0 = time.perf_counter()
    payload_H = compress_signal(H)
    H_hat = decompress_signal(payload_H)
    t_comp = (time.perf_counter() - t0) * 1e3
    if verbose:
        kept = payload_H['sqfree'].size
        mse = float(np.mean((H - H_hat) ** 2))
        psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float('inf')
        print(f"[6] photo compression  ·  square-free FFT")
        print(f"      square-free kept   = {kept} / {N}  "
              f"= {kept / N:.4f}")
        print(f"      PSNR               = {psnr:.2f} dB")
        print(f"      compress time      = {t_comp:.2f} ms")

    # ---------- 8. even/odd parity  ·  color scattering ----------
    H_even_ch, H_odd_ch = parity_channels(H)
    if verbose:
        print(f"[7] even/odd parity  ·  color scattering")
        print(f"      even (R) mass      = "
              f"{np.sum(np.abs(H_even_ch)):.4f}")
        print(f"      odd  (B) mass      = "
              f"{np.sum(np.abs(H_odd_ch)):.4f}")

    # ---------- 9. fiber propagation  ·  full wavelength pipeline ----------
    t0 = time.perf_counter()
    H_fiber_even = fiber_propagate(H_even_ch, width=41)
    H_fiber_odd  = fiber_propagate(H_odd_ch,  width=41)
    H_fiber_full = fiber_propagate(H,         width=41)
    t_fiber = (time.perf_counter() - t0) * 1e3
    if verbose:
        print(f"[8] fiber propagation  ·  H(z) = exp(−α|z|)")
        print(f"      ‖H‖ full           = {np.linalg.norm(H):.4f}")
        print(f"      ‖H_fiber‖          = {np.linalg.norm(H_fiber_full):.4f}")
        print(f"      propagate time     = {t_fiber:.2f} ms")
        print()

    # ---------- 9. visualization ----------
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # (a) coupling matrix
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(G, origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_title(f"Coupling matrix  g_ij = w_ij · κ_ij\n"
                 f"⟨g⟩ = {G[G>0].mean():.4f}   max = {G.max():.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) dynamic Hertz field
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(N), H, color="#8e44ad", lw=1.2, label="H_i")
    ax.scatter(even_idx, H_even_grid, c="#16a085", s=12,
               edgecolors="k", linewidth=0.3, zorder=5,
               label="even-even mask")
    ax.set_xlabel("cell i"); ax.set_ylabel("H_i")
    ax.set_title(f"Dynamic Hertz field\n"
                 f"range = [{H.min():.4f}, {H.max():.4f}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) square-free spectrum
    ax = fig.add_subplot(gs[0, 2])
    C = np.fft.fft(H)
    ax.plot(np.abs(C), color="#2c3e50", lw=0.8, label="|C(k)|")
    ax.scatter(payload_H['sqfree'],
               np.abs(C[payload_H['sqfree']]),
               s=8, c="#e74c3c", label=f"square-free anchors "
                                        f"({payload_H['sqfree'].size})")
    ax.set_xlabel("k"); ax.set_ylabel("|C(k)|")
    ax.set_yscale("log")
    ax.set_title("Spectrum  ·  square-free anchors")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) reconstruction
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.arange(N), H, color="#2c3e50", lw=1.2, label="original H")
    ax.plot(np.arange(N), H_hat, color="#e74c3c", lw=1.0, ls="--",
            label="square-free reconstruction")
    ax.set_xlabel("cell i"); ax.set_ylabel("H_i")
    mse = float(np.mean((H - H_hat) ** 2))
    psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    ax.set_title(f"M-curve reconstruction\nPSNR = {psnr:.2f} dB")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) even/odd parity channels
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.arange(N), H_even_ch, color="#16a085", lw=1.0,
            label="even → R (bosonic)")
    ax.plot(np.arange(N), H_odd_ch, color="#e74c3c", lw=1.0,
            label="odd  → B (fermionic)")
    ax.set_xlabel("cell i"); ax.set_ylabel("scatter")
    ax.set_title("Even/odd parity  ·  color scattering")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (f) fiber-propagated full wavelength
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.arange(N), H, color="#2c3e50", lw=0.8, alpha=0.5,
            label="input H")
    ax.plot(np.arange(N), H_fiber_full, color="#8e44ad", lw=1.4,
            label="fiber-propagated H")
    ax.plot(np.arange(N), H_fiber_even, color="#16a085", lw=0.8,
            alpha=0.7, label="fiber (even)")
    ax.plot(np.arange(N), H_fiber_odd, color="#e74c3c", lw=0.8,
            alpha=0.7, label="fiber (odd)")
    ax.set_xlabel("cell i"); ax.set_ylabel("amplitude")
    ax.set_title(f"Fiber kernel  exp(−α|z|)\n"
                 f"‖H‖ = {np.linalg.norm(H):.3f}   "
                 f"‖H_fiber‖ = {np.linalg.norm(H_fiber_full):.3f}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle(
        "Dynamic Hertz even-even grid  ·  coupling strength matrix  ·  "
        "square-free FFT  ·  even/odd parity  ·  fiber pipeline",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---------- 10. summary ----------
    if verbose:
        print("--- pipeline summary ---")
        print(f"  N                        = {N}")
        print(f"  coupling matrix          = {N}×{N}")
        print(f"  Hertz field range        = "
              f"[{H.min():.4f}, {H.max():.4f}]")
        print(f"  even-even grid size      = {len(even_idx)}")
        print(f"  square-free kept         = {payload_H['sqfree'].size}")
        print(f"  reconstruction PSNR      = {psnr:.2f} dB")
        print(f"  fiber mass               = "
              f"{np.linalg.norm(H_fiber_full):.4f}")
        print()

        print("--- complexity ---")
        print("  symmetric path (sieve)   O(K log log K)   once")
        print("  asymmetric refresh       O(K log K)       per cycle")
        print("  coupling matrix          O(N²)")
        print("  dynamic Hertz field      O(N²)")
        print("  square-free compression  O(N log N)")
        print("  M-curve reconstruction   O(N)")
        print("  fiber propagation        O(N log N)       per pass")
        print()

    return dict(
        G=G, H=H, H_hat=H_hat,
        H_even_ch=H_even_ch, H_odd_ch=H_odd_ch,
        H_fiber_even=H_fiber_even,
        H_fiber_odd=H_fiber_odd,
        H_fiber_full=H_fiber_full,
        payload_H=payload_H,
        even_idx=even_idx,
        mask=mask,
        mu_sym=mu_sym, mu_asym=mu_asym, F=F,
        N=N,
    )


if __name__ == "__main__":
    result = pipeline(N=128, seed=42, dzeta_dt=0.5, verbose=True)