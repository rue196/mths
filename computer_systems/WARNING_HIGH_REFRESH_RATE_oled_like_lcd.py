#!/usr/bin/env python3
"""
oled_screen.py
==============

OLED-like display on the even-even pixel grid, extending
refresh_rate_screen.py.

Physical pixel grid
-------------------
    Only (i, j) with i, j even exist    →  natural pixel step 2 in each axis
    Grid size = (N/2) × (N/2) pixels
    Fill fraction = 1/4 of the full grid  (OLED sub-pixel economy)

Colour depth  (from the light engine in lightweight_gaming_engine.py)
-------------
    g_ij = w_ij · κ_ij                  coupling strength matrix
        w_ij = |χ_i − χ_j| / π          chiral bond weight
        κ_ij = √(L_i L_j) / L_ref       travel coupling
    intensity_ij = g_ij / g_max  ∈ [0, 1]

Sub-pixels  (parity split  →  R/B channels)
----------
    R  = bosonic channel    (even parity mass)
    G  = Π_e(t, r) · g_ij   bounded [0, 1]      (OLED luminance)
    B  = fermionic channel  (odd parity mass)

Refresh
-------
    H_i  = Σ_j |g_ij · ω_ij|           dynamic Hertz rate
    scan = sort by H_i  (high → low)   row-sequential display

OLED-like qualities
-------------------
    · true black        g_ij < threshold   →  (0, 0, 0)
    · sub-pixel bloom   fiber kernel exp(−α|z|) per axis
    · square-free FFT   only μ(n) ≠ 0 anchors stored  (6/π²)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from refresh_rate_screen import (
    PI, E, ALPHA, A_STEP, DENSITY,
    mobius_sieve, coupling_matrix, dynamic_hertz,
    even_even_mask, parity_channels,
    fiber_propagate,
    compress_signal, decompress_signal,
)


# ============================================================
#  OLED screen
# ============================================================
@dataclass
class OLEDPixel:
    i: int                # full-grid row
    j: int                # full-grid col
    R: float              # bosonic sub-pixel   [0, 1]
    G: float              # luminance sub-pixel [0, 1]  (Π_e · g)
    B: float              # fermionic sub-pixel [0, 1]
    g: float              # coupling depth      [0, 1]
    H: float              # Hertz refresh rate
    is_black: bool        # true black if g < threshold


class OLEDScreen:
    """
    Display driver for the even-even pixel grid.

        pixel grid   =  even-even mask (step 2)
        colour depth =  g_ij  (coupling strength)
        luminance    =  Π_e   (bounded elliptic projection)
        sub-pixels   =  parity channels (bosonic / fermionic)
        refresh      =  dynamic Hertz H_i
    """

    def __init__(self,
                 N: int = 64,
                 dzeta_dt: float = 0.5,
                 seed: int = 42,
                 black_threshold: float = 0.02):
        rng = np.random.default_rng(seed)
        self.N = int(N)
        self.dzeta_dt = float(dzeta_dt)
        self.black_threshold = float(black_threshold)

        # ---- physics pipeline (from refresh_rate_screen.py) ----
        self.t = np.maximum(
            np.sort(rng.exponential(scale=3.0, size=N)), 1e-3)
        self.r = np.arange(1, N + 1, dtype=float)

        self.G = coupling_matrix(self.t, self.r)         # g_ij = w · κ
        self.G_max = float(self.G.max()) + 1e-9
        self.G_norm = self.G / self.G_max

        self.H = dynamic_hertz(self.G, dzeta_dt=dzeta_dt)
        self.H_max = float(self.H.max()) + 1e-9

        # ---- even-even pixel grid (natural step 2) ----
        self.pixel_mask = even_even_mask(N)
        self.even_idx = np.where(np.arange(N) % 2 == 0)[0]
        self.N_pix = int(len(self.even_idx))

        # ---- parity channels → R / B sub-pixels ----
        self.H_even, self.H_odd = parity_channels(self.H)
        self.H_even_max = float(self.H_even.max()) + 1e-9
        self.H_odd_max  = float(self.H_odd.max())  + 1e-9

        # ---- Möbius sieve for square-free anchors ----
        self.mu = mobius_sieve(N)

        # ---- scan order: Hertz-sorted (high rate → first) ----
        self.scan_order = self._build_scan_order()

    # --------------------------------------------------------
    def _build_scan_order(self) -> List[Tuple[int, int, int, int]]:
        """Row-major over even-even pixels, grouped by Hertz priority."""
        pairs = [(pi, pj, int(i), int(j))
                 for pi, i in enumerate(self.even_idx)
                 for pj, j in enumerate(self.even_idx)]
        pairs.sort(key=lambda p: (-self.H[p[2]], p[2], p[3]))
        return pairs

    # --------------------------------------------------------
    def pixel_rgb(self, i: int, j: int, t_now: float
                  ) -> Tuple[float, float, float]:
        """
        Return (R, G, B) for pixel (i, j) at time t_now.

            R  = bosonic coupling
            G  = Π_e(t, r) · g_ij       (bounded [0,1])
            B  = fermionic coupling
        """
        g = float(self.G_norm[i, j])
        if g < self.black_threshold:
            return 0.0, 0.0, 0.0

        # R: bosonic channel  (even row parity)
        r_ch = g * float(self.H_even[i] / self.H_even_max)

        # G: Π_e  bounded luminance
        r_eff = (float(self.H[i]) / self.H_max) * PI
        pe = 0.5 * (1.0 + math.cos(2.0 * t_now)
                          * math.cos(2.0 * PI * r_eff / E))
        g_ch = g * pe

        # B: fermionic channel  (odd row parity)
        b_ch = g * float(self.H_odd[i] / self.H_odd_max)

        return r_ch, g_ch, b_ch

    # --------------------------------------------------------
    def render(self, t_now: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render one frame → three (N_pix × N_pix) sub-pixel canvases."""
        R   = np.zeros((self.N_pix, self.N_pix))
        Gch = np.zeros((self.N_pix, self.N_pix))
        B   = np.zeros((self.N_pix, self.N_pix))
        for pi, i in enumerate(self.even_idx):
            for pj, j in enumerate(self.even_idx):
                r, g, b = self.pixel_rgb(i, j, t_now)
                R[pi, pj]   = r
                Gch[pi, pj] = g
                B[pi, pj]   = b
        return R, Gch, B

    # --------------------------------------------------------
    def composite(self, R: np.ndarray, G: np.ndarray, B: np.ndarray
                  ) -> np.ndarray:
        """Pack three sub-pixel canvases → RGB display array."""
        img = np.zeros((R.shape[0], R.shape[1], 3))
        img[:, :, 0] = np.clip(R, 0, 1)
        img[:, :, 1] = np.clip(G, 0, 1)
        img[:, :, 2] = np.clip(B, 0, 1)
        return img

    # --------------------------------------------------------
    def bloom(self, canvas: np.ndarray, width: int = 11) -> np.ndarray:
        """
        OLED sub-pixel bloom.
        Apply fiber kernel exp(−α|z|) separably along both axes.
        """
        w = min(width, canvas.shape[0], canvas.shape[1])
        H, W = canvas.shape
        temp = np.zeros_like(canvas)
        for i in range(H):
            temp[i, :] = fiber_propagate(canvas[i, :], width=w)
        result = np.zeros_like(temp)
        for j in range(W):
            result[:, j] = fiber_propagate(temp[:, j], width=w)
        return result

    # --------------------------------------------------------
    def refresh_rate(self) -> float:
        """Effective scan rate = mean(Hertz) × α."""
        return float(self.H.mean() * ALPHA)

    # --------------------------------------------------------
    def compress(self, canvas: np.ndarray) -> List[Dict]:
        """Square-free FFT compression per row."""
        return [compress_signal(canvas[i, :])
                for i in range(canvas.shape[0])]

    def decompress(self, payload: List[Dict], shape: Tuple[int, int]
                   ) -> np.ndarray:
        rows, cols = shape
        result = np.zeros(shape)
        for i, p in enumerate(payload):
            result[i, :] = decompress_signal(p)
        return result

    # --------------------------------------------------------
    def to_full_grid(self, canvas: np.ndarray,
                     fill: float = 0.0) -> np.ndarray:
        """Embed a (N_pix × N_pix) canvas into the full N × N grid."""
        full = np.full((self.N, self.N), fill, dtype=float)
        for pi, i in enumerate(self.even_idx):
            for pj, j in enumerate(self.even_idx):
                full[i, j] = canvas[pi, pj]
        return full

    # --------------------------------------------------------
    def stats(self) -> Dict:
        live = int(self.pixel_mask.sum())
        n_sq_cols = int(sum(1 for j in self.even_idx
                            if j < len(self.mu) and self.mu[j] != 0))
        return dict(
            N=self.N,
            N_pix=self.N_pix,
            total_pixels=self.N_pix ** 2,
            full_grid=self.N ** 2,
            fill_fraction=self.N_pix ** 2 / self.N ** 2,
            live_cells=live,
            G_max=self.G_max,
            H_max=self.H_max,
            refresh_hz=self.refresh_rate(),
            square_free_cols=n_sq_cols,
            square_free_density=n_sq_cols / max(self.N_pix, 1),
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("OLED screen  ·  even-even pixel grid  ·  coupling → colour depth")
    print("=" * 78)
    print(f"  α        = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP                 = {A_STEP:.6f}")
    print(f"  6/π²                   = {DENSITY:.6f}   (μ density)")
    print()

    screen = OLEDScreen(N=64, dzeta_dt=0.5, seed=42,
                        black_threshold=0.02)
    st = screen.stats()

    print("--- pixel grid  ·  natural step = 2 (even-even) ---")
    print(f"  full grid              : {st['N']} × {st['N']}")
    print(f"  physical pixel grid    : {st['N_pix']} × {st['N_pix']}")
    print(f"  total physical pixels  : {st['total_pixels']}")
    print(f"  fill fraction          : {st['fill_fraction']:.4f}  "
          f"(theory 0.25)")
    print(f"  live cells             : {st['live_cells']}")
    print()

    print("--- coupling → colour depth ---")
    print(f"  G_max                  : {st['G_max']:.6f}")
    print(f"  H_max (Hz refresh)     : {st['H_max']:.6f}")
    print(f"  refresh rate           : {st['refresh_hz']:.4f}  Hz")
    print(f"  square-free anchors    : {st['square_free_cols']} / "
          f"{st['N_pix']}  = {st['square_free_density']:.4f}  "
          f"(target 6/π² = {DENSITY:.4f})")
    print()

    # ---- two frames: Π_e flips at t = π/2 ----
    t0 = 0.0
    t1 = PI / 2

    R0, G0, B0 = screen.render(t0)
    R1, G1, B1 = screen.render(t1)
    img0 = screen.composite(R0, G0, B0)
    img1 = screen.composite(R1, G1, B1)

    # ---- bloom on the t = π/2 frame ----
    Rb = screen.bloom(R1, width=11)
    Gb = screen.bloom(G1, width=11)
    Bb = screen.bloom(B1, width=11)
    img1_bloom = screen.composite(Rb, Gb, Bb)

    # ---- square-free FFT compression of the luminance channel ----
    payload = screen.compress(G1)
    G_recon = screen.decompress(payload, G1.shape)
    mse = float(np.mean((G1 - G_recon) ** 2))
    psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")

    n_kept = sum(p['sqfree'].size for p in payload)
    n_total = sum(p['N'] for p in payload)
    anchor_density = n_kept / max(n_total, 1)

    # ---- true-black statistics ----
    n_black_0 = int(np.sum(G0 < screen.black_threshold))
    n_black_1 = int(np.sum(G1 < screen.black_threshold))
    print("--- sub-pixel statistics ---")
    print(f"  frame t={t0:.2f}  ⟨R,G,B⟩ = "
          f"{R0.mean():.4f}, {G0.mean():.4f}, {B0.mean():.4f}")
    print(f"  frame t={t1:.2f}  ⟨R,G,B⟩ = "
          f"{R1.mean():.4f}, {G1.mean():.4f}, {B1.mean():.4f}")
    print(f"  true-black pixels t0  : {n_black_0} / {G0.size}")
    print(f"  true-black pixels t1  : {n_black_1} / {G1.size}")
    print()

    print("--- square-free FFT compression  (G channel) ---")
    print(f"  frequency bins total   : {n_total}")
    print(f"  square-free anchors    : {n_kept}")
    print(f"  anchor density         : {anchor_density:.4f}  "
          f"(theory 6/π² = {DENSITY:.4f})")
    print(f"  MSE                    : {mse:.6e}")
    print(f"  PSNR                   : {psnr:.2f} dB")
    print()

    print("--- OLED bloom  (fiber kernel exp(−α|z|)) ---")
    print(f"  ‖G‖ input              : {np.linalg.norm(G1):.4f}")
    print(f"  ‖G_bloom‖              : {np.linalg.norm(Gb):.4f}")
    print(f"  peak / mean ratio in   : "
          f"{G1.max() / max(G1.mean(), 1e-9):.2f}")
    print(f"  peak / mean ratio bloom: "
          f"{Gb.max() / max(Gb.mean(), 1e-9):.2f}")
    print()

    # ---- full-grid view showing natural step 2 ----
    full_grid = screen.to_full_grid(G1, fill=np.nan)

    # ============================================================
    #  Visualisation
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    # (a) full grid — only even-even cells lit
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(full_grid, cmap="inferno",
              origin="lower", aspect="equal")
    ax.set_title("Full N×N grid\n"
                 "only even-even cells lit  ·  natural step = 2")
    ax.set_xlabel("full-grid j")
    ax.set_ylabel("full-grid i")

    # (b) composite display (no bloom)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img1, origin="lower", aspect="equal")
    ax.set_title(f"Composite RGB  ·  t = {t1:.2f}\n"
                 f"⟨R,G,B⟩ = "
                 f"{R1.mean():.3f}, {G1.mean():.3f}, {B1.mean():.3f}")
    ax.set_xlabel("pixel j / 2")
    ax.set_ylabel("pixel i / 2")

    # (c) composite display with bloom
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img1_bloom, origin="lower", aspect="equal")
    ax.set_title("OLED bloom  ·  fiber exp(−α|z|)\n"
                 f"‖G‖ = {np.linalg.norm(G1):.3f}  →  "
                 f"{np.linalg.norm(Gb):.3f}")
    ax.set_xlabel("pixel j / 2")
    ax.set_ylabel("pixel i / 2")

    # (d) R sub-pixel (bosonic)
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(R1, cmap="Reds", origin="lower", aspect="equal")
    ax.set_title(f"R sub-pixel  ·  bosonic\n⟨R⟩ = {R1.mean():.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (e) G sub-pixel (luminance) + compression check
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(G_recon, cmap="Greens",
                   origin="lower", aspect="equal")
    ax.set_title(f"G sub-pixel  ·  Π_e · g_ij\n"
                 f"reconstructed  ·  PSNR = {psnr:.1f} dB")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (f) B sub-pixel (fermionic)
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(B1, cmap="Blues", origin="lower", aspect="equal")
    ax.set_title(f"B sub-pixel  ·  fermionic\n⟨B⟩ = {B1.mean():.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"OLED screen  ·  even-even pixel grid  ·  "
        f"g_ij = w·κ  ·  Π_e ∈ [0,1]  ·  "
        f"refresh = {st['refresh_hz']:.3f} Hz  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 78)
    print("OLED screen summary")
    print("=" * 78)
    print("  Pixel grid  (natural step 2)")
    print("      physical pixel at (i, j) iff i, j even")
    print(f"      grid size         = {st['N_pix']} × {st['N_pix']}")
    print(f"      fill fraction     = {st['fill_fraction']:.4f}")
    print()
    print("  Colour depth  (from coupling)")
    print("      g_ij = w_ij · κ_ij")
    print(f"      intensity         = g_ij / g_max  ∈ [0, 1]")
    print(f"      true-black cutoff = {screen.black_threshold}")
    print()
    print("  Sub-pixels  (parity split)")
    print("      R = bosonic channel  (even row parity mass)")
    print("      G = Π_e(t, r) · g_ij  ∈ [0, 1]  (bounded luminance)")
    print("      B = fermionic channel (odd row parity mass)")
    print()
    print("  Refresh  (dynamic Hertz)")
    print(f"      H_i = Σ_j |g_ij · ω_ij|")
    print(f"      mean refresh      = {st['refresh_hz']:.4f} Hz")
    print("      scan order        = sort by H_i  (high → low)")
    print()
    print("  OLED qualities")
    print("      · true black           g_ij < 0.02  →  (0, 0, 0)")
    print("      · sub-pixel bloom      fiber exp(−α|z|) per axis")
    print("      · anchor compression   square-free FFT (μ ≠ 0)")
    print(f"        PSNR                = {psnr:.2f} dB")
    print("      · refresh              driven by H_i per row")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()