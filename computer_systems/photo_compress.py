#!/usr/bin/env python3
"""
squarefree_fft_mcurve.py
========================
Square-free FFT image compression with M-matrix curve reconstruction.

Pipeline
--------
compress:
    img.flatten()          (natural row-major)
      → FFT
      → keep only square-free bins     μ(n) ≠ 0   (density 6/π²)
      → Re(C[n])  → negative-index grid      (algebraic,     O(n))
      → Im(C[n])  → positive-index grid      (transcendental, O(K log K))

decompress:
    anchors at square-free bins
      → fill gaps via M-matrix curve
         algebraic      : Möbius on inverse magnitudes
         transcendental : sin(ln m) shape factor
      → IFFT
      → reshape (H, W)
"""

from __future__ import annotations
import math
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)


# ============================================================
#  Möbius sieve  ·  O(N)
# ============================================================
def mobius_sieve(N: int) -> np.ndarray:
    mu = np.zeros(N + 1, dtype=np.int8)
    if N >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(N + 1, dtype=bool)
    for i in range(2, N + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Square-free index list  (DC = 0 always kept)
# ============================================================
def square_free_indices(N: int) -> np.ndarray:
    mu = mobius_sieve(N)
    keep = [0]                                 # DC always kept
    keep += [n for n in range(1, N) if mu[n] != 0]
    return np.array(keep, dtype=np.int32)


# ============================================================
#  M-matrix curve primitives  (from curves_m.py)
# ============================================================
def m_algebraic_interp(ma: float, mb: float, t: float) -> float:
    """
    Algebraic part of the M-matrix curve:
        x(t) = 1 / ( (1-t)/ma + t/mb )
    Möbius interpolation of the inverses.
    """
    if ma <= 1e-12 or mb <= 1e-12:
        return (1 - t) * ma + t * mb
    return 1.0 / ((1 - t) / ma + t / mb)


# ============================================================
#  Compression
# ============================================================
def compress_image(img: np.ndarray) -> Dict:
    """
    Square-free FFT compression (no TSP).

    Returns
    -------
    payload : dict
        real   : Re(C[n])  for square-free n   → negative-index grid
        imag   : Im(C[n])  for square-free n   → positive-index grid
        sqfree : square-free index array
        N      : signal length
        shape  : image shape
    """
    H, W = img.shape
    signal = img.flatten().astype(np.float64)
    N = signal.size

    C = np.fft.fft(signal)
    sqfree = square_free_indices(N)

    return {
        'real':   C[sqfree].real.copy(),
        'imag':   C[sqfree].imag.copy(),
        'sqfree': sqfree,
        'N':      N,
        'shape':  (H, W),
    }


# ============================================================
#  M-curve gap fill
# ============================================================
def _fill_gap_mcurve(k: int, left: int, right: int,
                     C_left: complex, C_right: complex) -> complex:
    """
    Fill a non-square-free bin k between anchors (left, right) using
    the M-matrix curve primitive.

        algebraic      : Möbius on inverse magnitudes     →  m(t)
        transcendental : sin(ln m) shape factor           →  shape(t)
        phase          : linear unwrapped interpolation

    shape(t) = 1 at t = 0 and t = 1, so anchors are preserved.
    """
    t = (k - left) / max(right - left, 1)

    ma = abs(C_left);  pa = math.atan2(C_left.imag,  C_left.real)
    mb = abs(C_right); pb = math.atan2(C_right.imag, C_right.real)

    dp = pb - pa
    while dp >  math.pi: dp -= 2 * math.pi
    while dp < -math.pi: dp += 2 * math.pi

    # --- algebraic: Möbius on inverses ---
    m_alg = m_algebraic_interp(ma, mb, t)

    # --- transcendental: sin(ln m) correction, vanishes at endpoints ---
    s_a = math.sin(math.log(max(ma, 1e-12)))
    s_b = math.sin(math.log(max(mb, 1e-12)))
    s_t = math.sin(math.log(max(m_alg, 1e-12)))
    s_lin = (1 - t) * s_a + t * s_b
    shape = 1.0 + 0.1 * (s_t - s_lin)

    mag   = m_alg * shape
    phase = pa + t * dp
    return mag * complex(math.cos(phase), math.sin(phase))


# ============================================================
#  Decompression
# ============================================================
def decompress_image(payload: Dict) -> np.ndarray:
    N      = payload['N']
    sqfree = payload['sqfree']
    H, W   = payload['shape']

    C_hat = np.zeros(N, dtype=complex)
    C_hat[sqfree] = payload['real'] + 1j * payload['imag']

    sqset     = set(int(s) for s in sqfree)
    sq_sorted = np.sort(sqfree)

    # --- fill every non-square-free bin via the M-curve ---
    for k in range(N):
        if k in sqset:
            continue
        ir = np.searchsorted(sq_sorted, k)
        left  = int(sq_sorted[ir - 1]) if ir > 0 else None
        right = int(sq_sorted[ir])     if ir < sq_sorted.size else None

        if left is None and right is None:
            continue
        if left is None:
            C_hat[k] = C_hat[right]
        elif right is None:
            C_hat[k] = C_hat[left]
        else:
            C_hat[k] = _fill_gap_mcurve(k, left, right,
                                        C_hat[left], C_hat[right])

    # --- IFFT ---
    signal = np.fft.ifft(C_hat).real
    return np.clip(signal.reshape(H, W), 0, 1)


# ============================================================
#  Demo image
# ============================================================
def make_demo_image(size=128, n=5, scale=2.0):
    x = np.linspace(-scale, scale, size)
    y = np.linspace(-scale, scale, size)
    X, Y = np.meshgrid(x, y)
    if n == 1:
        img = np.ones_like(X) * 2.0
    else:
        img = np.abs(X) ** (n - 1) + np.abs(Y) ** (n - 1)
    img = img - img.min()
    return img / img.max()


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float('inf')
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 78)
    print("Square-free FFT compression  ·  M-curve reconstruction  ·  no TSP")
    print("=" * 78)
    print(f"  α = 1/(π − e)        = {ALPHA:.6f}")
    print(f"  square-free density  = 6/π² ≈ {6 / PI**2:.4f}")
    print()

    N_img = 128
    images = [make_demo_image(N_img, n=n) for n in (1, 5, 15)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    for col, (img, n) in enumerate(zip(images, (1, 5, 15))):
        payload = compress_image(img)
        recon   = decompress_image(payload)

        N     = payload['N']
        kept  = payload['sqfree'].size
        ratio = kept / N
        p     = psnr(img, recon)

        print(f"--- n = {n:>2d} ---")
        print(f"  signal length N  : {N}")
        print(f"  square-free kept : {kept}   ({100 * ratio:.2f} %)")
        print(f"  real scalars     : {2 * kept}  "
              f"({2 * kept * 8} bytes float64)")
        print(f"  PSNR             : {p:.2f} dB")
        print()

        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(f"original  n = {n}")
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

        axes[1, col].imshow(recon, cmap='gray')
        axes[1, col].set_title(
            f"recon  ·  kept {100 * ratio:.1f} %\nPSNR = {p:.2f} dB")
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

    plt.suptitle(
        "Square-free FFT compression  ·  "
        "O(n) algebraic grid  +  O(K log K) transcendental grid  ·  "
        "M-matrix curve reconstruction",
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ----------------------------------------------------------
    #  Spectrum diagnostics
    # ----------------------------------------------------------
    print("=" * 78)
    print("Spectrum diagnostics  ·  n = 5,  128×128")
    print("=" * 78)
    img = images[1]
    payload = compress_image(img)
    recon = decompress_image(payload)

    N = payload['N']
    signal = img.flatten().astype(np.float64)
    C_orig = np.fft.fft(signal)

    # rebuild the reconstructed spectrum for diagnostics
    C_hat = np.zeros(N, dtype=complex)
    C_hat[payload['sqfree']] = payload['real'] + 1j * payload['imag']
    sqset = set(int(s) for s in payload['sqfree'])
    sq_sorted = np.sort(payload['sqfree'])
    for k in range(N):
        if k in sqset:
            continue
        ir = np.searchsorted(sq_sorted, k)
        l = int(sq_sorted[ir - 1]) if ir > 0 else None
        r = int(sq_sorted[ir]) if ir < sq_sorted.size else None
        if l is None:   C_hat[k] = C_hat[r]
        elif r is None: C_hat[k] = C_hat[l]
        else:           C_hat[k] = _fill_gap_mcurve(k, l, r,
                                                    C_hat[l], C_hat[r])

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.4))

    axes2[0].plot(np.abs(C_orig), color='#2c3e50', lw=0.8,
                  label=r'original $|C(k)|$')
    axes2[0].scatter(payload['sqfree'],
                     np.abs(C_orig[payload['sqfree']]),
                     s=6, color='#e74c3c',
                     label=f"square-free anchors "
                           f"({payload['sqfree'].size})")
    axes2[0].set_xlabel('k'); axes2[0].set_ylabel('|C(k)|')
    axes2[0].set_yscale('log')
    axes2[0].set_title('Spectrum  ·  original + anchors')
    axes2[0].legend(fontsize=8); axes2[0].grid(alpha=0.3)

    axes2[1].plot(np.abs(C_orig), color='#2c3e50', lw=0.8,
                  label='original')
    axes2[1].plot(np.abs(C_hat),  color='#16a085', lw=0.8, alpha=0.9,
                  label='M-curve reconstruction')
    axes2[1].set_xlabel('k'); axes2[1].set_ylabel('|C(k)|')
    axes2[1].set_yscale('log')
    axes2[1].set_title('Magnitude reconstruction')
    axes2[1].legend(fontsize=8); axes2[1].grid(alpha=0.3)

    err = np.abs(np.abs(C_orig) - np.abs(C_hat))
    axes2[2].semilogy(err + 1e-12, color='#e67e22', lw=0.7)
    axes2[2].set_xlabel('k'); axes2[2].set_ylabel(r'$|\Delta|C||$')
    axes2[2].set_title('Magnitude error')
    axes2[2].grid(alpha=0.3)

    plt.suptitle(
        "Square-free anchor placement and M-curve magnitude reconstruction",
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


if __name__ == "__main__":
    main()