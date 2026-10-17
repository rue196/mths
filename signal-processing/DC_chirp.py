#!/usr/bin/env python3
"""
dc_chirp_entangled.py
=====================

DC channel with an alternating-like chirp, entangled through the
even-parity even-even channel.

    DC channel  (throughput-supertrace.py)
        ζ(t) = Σ c_i exp(i·t·i/α),  c_i = μ(|i|)   (smoothed across zeros)
        DC throughput  = ∫ |ζ|² dt / period
        supertrace  S  = Σ (−1)^i c_i
        mass        m  = |S| exp(−H)

    Chirp trajectory  (quantum_geo_location.py)
        DC path     x(s)  = exp(A_STEP · s / N)
        chirp       w(s)  = cos(A_STEP · s² / N)
        phase       φ(s)  = arg ζ(s/N · period)
        entangled   c(s)  = w(s) · cos φ(s)
        coupling    g_ij  = w_ij · κ_ij
        even-even mask   M_ij = [i even ∧ j even]
        propagate on the even subgraph:
            s_{t+1} = G_even · s_t / row_sum
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)          # ≈ 2.362
A_STEP   = ALPHA / 0.3628          # ≈ 6.511
DENSITY  = 6.0 / (PI * PI)         # ≈ 0.6079
NORM = 1.0 - math.exp(-ALPHA * (PI + E))   # ≈ 0.999999


# ============================================================
#  Möbius sieve
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  DC channel  ·  EllipticMobiusGate  (throughput-supertrace.py)
# ============================================================
class DCMobiusGate:
    """
    ζ(t) = Σ c_i exp(i·t·i/α),   c_i = μ(|i|)  for i ≠ 0
    """
    def __init__(self, K: int = 20, smooth: bool = True):
        self.K = K
        mu = mobius_sieve(K)
        c = np.zeros(2 * K + 1)
        for i in range(-K, K + 1):
            if i != 0:
                c[i + K] = float(mu[abs(i)])
        if smooth:
            for i in range(-K, K + 1):
                if i == 0 or c[i + K] != 0.0:
                    continue
                l, r = i - 1, i + 1
                while l >= -K and c[l + K] == 0.0:
                    l -= 1
                while r <= K and c[r + K] == 0.0:
                    r += 1
                if l < -K or r > K or r == l:
                    continue
                dist = r - l
                c[i + K] = (r - i) / dist * c[l + K] \
                         + (i - l) / dist * c[r + K]
        self.full    = c
        idx = [i for i in range(-K, K + 1) if abs(c[i + K]) > 1e-12]
        self.indices = np.array(idx, dtype=float)
        self.coeffs  = np.array([c[i + K] for i in idx])
        self.period  = 2 * PI * ALPHA

    # ---- ζ(t) and |ζ(t)|² ----
    def zeta(self, t: float) -> complex:
        if len(self.indices) == 0:
            return 0j
        return complex(np.sum(self.coeffs *
                              np.exp(1j * t * self.indices / ALPHA)))

    def power(self, t: float) -> float:
        return abs(self.zeta(t)) ** 2

    # ---- DC throughput  ·  closed form via orthogonality + NORM ----
    def dc_throughput(self) -> float:
        """
        Period-averaged power  ·  closed form  ·  exponential NORM filter.

        Orthogonality of  e^{i·t·(i−j)/α}  over  T = 2πα  gives

            (1/T) ∫_0^T |ζ(t)|² dt  =  Σ_i c_i²      (exact, no quadrature)

        Per-coefficient power  x = (Σ c_i²)/N, then the exponential
        filter 1 − exp(−α·x), normalized by

            NORM = 1 − exp(−α·(π + e))

        which reaches exactly 1 when x = π + e.
        """
        N   = max(len(self.coeffs), 1)
        raw = float(np.sum(self.coeffs ** 2))      # period-averaged power
        x   = raw / N                              # per-coefficient power
        return float((1.0 - math.exp(-ALPHA * x)) / NORM)
    
    # ---- supertrace / entropy / mass ----
    def supertrace(self):
        c = self.full
        S = 0.0
        for i, v in enumerate(c):
            S += v if (i % 2 == 0) else -v
        N = len(c)
        if abs(S) < 1e-12 or N == 0:
            return float(S), 0.0, 0.0
        p = min(abs(S) / N, 0.9999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
        return float(S), float(H), float(m)


# ============================================================
#  Even-even mask
# ============================================================
def even_even_mask(N: int) -> np.ndarray:
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2 == 0) & (j % 2 == 0)


# ============================================================
#  DC path coupling  (quantum_geo_location.py)
# ============================================================
def chiral_angles_1d(x: np.ndarray) -> np.ndarray:
    lt = np.log(np.maximum(np.abs(x), 1e-12))
    return (lt + PI) % (2 * PI) - PI


def travel_lengths_1d(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x, prepend=x[0])
    ds = np.abs(dx)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-6


def coupling_matrix_1d(x: np.ndarray) -> np.ndarray:
    N = len(x)
    chi  = chiral_angles_1d(x)
    dchi = np.abs(chi[:, None] - chi[None, :])
    dchi = (dchi + PI) % (2 * PI) - PI
    W    = np.abs(dchi) / PI

    L = travel_lengths_1d(x)
    L_ref = float(np.median(L[L > 0])) if np.any(L > 0) else 1.0
    K = np.sqrt(np.maximum(L[:, None] * L[None, :], 0.0)) / L_ref
    K = np.clip(K, 0.0, 1.0)

    G = W * K
    np.fill_diagonal(G, 0.0)
    return G


def propagate_signal(G_even: np.ndarray, s0: np.ndarray,
                     steps: int = 40, decay: float = 0.02) -> np.ndarray:
    N = G_even.shape[0]
    s = s0.copy().astype(float)
    hist = [s.copy()]
    row_sum = G_even.sum(axis=1) + 1e-12
    for _ in range(steps):
        s_new = (G_even @ s) / row_sum * (1.0 - decay)
        s_new[np.arange(N) % 2 == 1] = 0.0
        s = s_new
        hist.append(s.copy())
    return np.array(hist)


# ============================================================
#  Demo
# ============================================================
def demo():
    K_gate = 20
    N_path = 80

    # ---- DC channel ----
    gate   = DCMobiusGate(K_gate, smooth=True)
    dc_tp  = gate.dc_throughput()
    S, H, m = gate.supertrace()

    print("=" * 76)
    print("DC chirp entangled  ·  even-parity even-even channel")
    print("=" * 76)
    print(f"  α       = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print()
    print("--- DC channel  (throughput-supertrace.py) ---")
    print(f"  K                     = {gate.K}")
    print(f"  non-zero coeffs       = {len(gate.indices)}")
    print(f"  period                = {gate.period:.6f}")
    print(f"  DC throughput         = {dc_tp:.6f}")
    print(f"  supertrace S          = {S:+.6f}")
    print(f"  entropy    H          = {H:.6f}")
    print(f"  mass       m          = {m:.6e}")
    print()

    # ---- DC path with entangled chirp ----
    s = np.arange(N_path, dtype=float)
    x = np.exp(A_STEP * s / N_path)      # monotone DC path

    # universal chirp
    w_chirp = np.cos(A_STEP * s * s / N_path)

    # entangled phase from ζ
    phis = np.array([np.angle(gate.zeta(si / N_path * gate.period))
                     for si in s])
    chirp_ent = w_chirp * np.cos(phis)

    # coupling + even-even mask
    G = coupling_matrix_1d(x)
    mask = even_even_mask(N_path)
    G_even = np.where(mask, G, 0.0)
    np.fill_diagonal(G_even, 0.0)
    kept = int(np.count_nonzero(G_even))

    print("--- DC path  (quantum_geo_location.py) ---")
    print(f"  N path points         = {N_path}")
    print(f"  x range               = [{x.min():.4f}, {x.max():.4f}]")
    print(f"  coupling G kept       = {kept} / {N_path ** 2}")
    print(f"  mask fraction         = {mask.mean():.4f}")
    print(f"  ⟨chirp⟩               = {w_chirp.mean():+.6f}")
    print(f"  ⟨entangled chirp⟩     = {chirp_ent.mean():+.6f}")
    print()

    # ---- propagate both impulses ----
    s0_plain = np.zeros(N_path); s0_plain[0] = 1.0
    s0_chirp = np.zeros(N_path); s0_chirp[0] = chirp_ent[0]

    hist_plain = propagate_signal(G_even, s0_plain, steps=40, decay=0.02)
    hist_chirp = propagate_signal(G_even, s0_chirp, steps=40, decay=0.02)

    # ---- chirp trajectory on the even subgraph ----
    even_idx = np.where(np.arange(N_path) % 2 == 0)[0]
    traj_chirp = hist_chirp[:, even_idx]

    # ---- entangled intensity from Π_e ----
    time_axis = np.arange(hist_chirp.shape[0])
    r_pos     = even_idx.astype(float)
    T, R = np.meshgrid(time_axis, r_pos, indexing="ij")
    pe = 0.5 * (1.0 + np.cos(2.0 * T)
                     * np.cos(2.0 * PI * R / E))

    print("--- chirp trajectory  (even subgraph) ---")
    print(f"  even positions        = {len(even_idx)}")
    print(f"  trajectory shape      = {traj_chirp.shape}")
    print(f"  final mass (plain)    = "
          f"{np.sum(np.abs(hist_plain[-1])):.4e}")
    print(f"  final mass (chirped)  = "
          f"{np.sum(np.abs(hist_chirp[-1])):.4e}")
    print(f"  Π_e range             = [{pe.min():.4f}, {pe.max():.4f}]")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # (a) DC throughput
    ax = fig.add_subplot(gs[0, 0])
    tt = np.linspace(0, gate.period, 512)
    zz = np.array([gate.zeta(ti) for ti in tt])
    ax.plot(tt, np.abs(zz) ** 2, color="#3498db", lw=1.2,
            label=r"$|\zeta(t)|^2$")
    ax.axhline(dc_tp, color="#e74c3c", ls="--", lw=1.0,
               label=f"DC throughput = {dc_tp:.4f}")
    ax.set_xlabel("t  (one period)")
    ax.set_ylabel(r"$|\zeta(t)|^2$")
    ax.set_title("DC channel  ·  EllipticMobiusGate throughput")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) ζ trajectory in the complex plane
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(zz.real, zz.imag, color="#8e44ad", lw=1.0, alpha=0.8)
    ax.scatter(zz.real[0], zz.imag[0], c="green", s=50,
               zorder=3, label="start")
    ax.scatter(zz.real[-1], zz.imag[-1], c="red", s=50,
               zorder=3, label="end")
    ax.set_aspect("equal")
    ax.set_xlabel("Re ζ"); ax.set_ylabel("Im ζ")
    ax.set_title("DC ζ(t) trajectory  ·  one period")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) chirp along the DC path
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s, w_chirp, "b--", lw=1.0, alpha=0.55,
            label=r"$w(s) = \cos(A_{STEP}\, s^2/N)$")
    ax.plot(s, chirp_ent, "r-", lw=1.2,
            label=r"entangled chirp  $w(s)\cos\varphi(s)$")
    ax.scatter(s[::4], chirp_ent[::4], c="k", s=15, zorder=3)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("path position  s")
    ax.set_ylabel("chirp amplitude")
    ax.set_title("Chirp along the DC path  ·  phase entangled through ζ")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) entangled chirp trajectory on the even subgraph
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(np.abs(traj_chirp).T, origin="lower",
                   aspect="auto", cmap="inferno")
    ax.set_xlabel("time step")
    ax.set_ylabel("even path position")
    ax.set_title(f"$|s|$ on even subgraph  ·  "
                 f"entangled chirp trajectory  ·  "
                 f"Π_e ∈ [{pe.min():.3f}, {pe.max():.3f}]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "DC chirp  ·  entangled through the even-parity even-even channel  ·  "
        f"α = {ALPHA:.4f}  ·  A_STEP = {A_STEP:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 76)
    print("Summary")
    print("=" * 76)
    print("  DC channel  (throughput-supertrace.py)")
    print("      ζ(t)       = Σ c_i exp(i·t·i/α),   c_i = μ(|i|)")
    print("      throughput = (1/period) ∫ |ζ|² dt")
    print(f"                 = {dc_tp:.6f}")
    print("      supertrace = Σ (−1)^i c_i")
    print(f"      S = {S:+.6f},  H = {H:.6f},  m = {m:.4e}")
    print()
    print("  Chirp trajectory  (quantum_geo_location.py)")
    print("      DC path         x(s) = exp(A_STEP · s / N)")
    print("      chirp           w(s) = cos(A_STEP · s² / N)")
    print("      entangled φ(s)  = arg ζ(s/N · period)")
    print("      entangled chirp = w(s) · cos φ(s)")
    print("      coupling        g_ij = w_ij · κ_ij")
    print("      even-even mask  M_ij = [i even ∧ j even]")
    print("      propagation     s_{t+1} = G_even · s_t / row_sum")
    print()
    print(f"  Even subgraph retains {kept}/{N_path ** 2} entries")
    print(f"  chirp trajectory shape = {traj_chirp.shape}")
    print()


if __name__ == "__main__":
    demo()