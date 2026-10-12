#!/usr/bin/env python3
"""
elliptic_log_rank_contracted.py
===============================
Elliptic log-rank test with:

    · Figure 3.5  ·  electron cloud density  Π(t, r) ∈ [0, 1]
    · fermion/photon contraction  g_ij = w_ij · κ_ij
    · elliptic algebraic / transcendental split (time on the imaginary axis)

Physics mapping
---------------
Per event n = (t_n, r_n):
    χ_n   = arg( t_n^i )             chirality angle      (from p_n = t_n^i)
    w_ij  = |χ_i − χ_j| / π          chiral bond weight   ∈ [0, 1]
    L_n   = arclength along (t, r)   travel length        (neutrino channel)
    κ_ij  = √(L_i L_j) / L_ref       travel coupling      ∈ [0, 1]
    g_ij  = w_ij · κ_ij              binary coupling      ∈ [0, 1]
    Π_n   = Π(t_n, r_n)              electron cloud density ∈ [0, 1]

Contracted log-rank statistic:
    Z_ell   = Σ_n Π_n · Z_n                    (elliptic weight only)
    Z_coup  = Σ_n Π_n · ḡ_n · Z_n              (with coupling)
    ḡ_n     = Σ_j g_nj / ⟨Σ_j g_nj⟩            normalized total coupling

Supertrace / mass on the contracted residuals:
    S_coup  = Σ_n (−1)^n (Π_n ḡ_n Z_n)
    m_coup  = |S_coup| · e^{−H_coup}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)            # ≈ 2.362
W1    = PI
W2    = E


# ============================================================
#  Figure 3.5  ·  elliptic electron cloud density
# ============================================================
def electron_density(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Π(t, r) ∈ [0, 1] on the torus ℂ / (πℤ + eℤ)."""
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * math.pi * u) * np.cos(2 * math.pi * v))


# ============================================================
#  Fermion / photon contraction  ·  g_ij = w_ij · κ_ij
# ============================================================
def chiral_angles(t: np.ndarray) -> np.ndarray:
    """
    χ_n = arg( t_n^i ) = log(t_n) mod 2π.
    This is the transcendental time phase from the p_n = t_n^i
    coordinate (Definition 4 of chiral.pdf).
    """
    lt = np.log(np.maximum(np.abs(t), 1e-12))
    return (lt + PI) % (2 * PI) - PI


def travel_lengths(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    L_n = cumulative arclength along the (t, r) event curve.
    This is the neutrino gravity channel (Definition 5 of the PDF).
    """
    dt = np.diff(t, prepend=t[0])
    dr = np.diff(r, prepend=r[0])
    ds = np.sqrt(dt * dt + dr * dr)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-6


def coupling_matrix(t: np.ndarray, r: np.ndarray,
                    L_ref: float = None) -> np.ndarray:
    """
    Binary coupling  g_ij = w_ij · κ_ij  over all event pairs.

        w_ij  = |χ_i − χ_j| / π          ∈ [0, 1]
        κ_ij  = √(L_i L_j) / L_ref       ∈ [0, 1]

    Diagonal is set to zero (no self-coupling).
    """
    N = len(t)

    # chiral bond weight
    chi = chiral_angles(t)
    dchi = np.abs(chi[:, None] - chi[None, :])
    dchi = (dchi + PI) % (2 * PI) - PI
    W = np.abs(dchi) / PI

    # travel coupling
    L = travel_lengths(t, r)
    if L_ref is None:
        L_ref = float(np.median(L[L > 0])) if np.any(L > 0) else 1.0
    K = np.sqrt(np.maximum(L[:, None] * L[None, :], 0.0)) / L_ref
    K = np.clip(K, 0.0, 1.0)

    # binary coupling
    G = W * K
    np.fill_diagonal(G, 0.0)
    return G


def coupling_strength(G: np.ndarray) -> np.ndarray:
    """
    Per-event total coupling  ḡ_n = Σ_j g_nj  normalized to mean 1.
    This is the effective fermion-photon activity at event n.
    """
    g = G.sum(axis=1)
    m = g.mean()
    return g / m if m > 0 else g


# ============================================================
#  Supertrace / entropy / mass
# ============================================================
def supertrace(z: np.ndarray) -> float:
    signs = np.where(np.arange(len(z)) & 1, -1.0, 1.0)
    return float(np.dot(signs, z))


def entropy(S: float, N: int) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -ALPHA * p * math.log(p)


def invariant_mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  Algebraic and transcendental event coordinates
# ============================================================
def inverse_coords(r, t, eps=1e-12):
    u = np.sign(r) / np.maximum(np.abs(r), eps)
    v = np.sign(t) / np.maximum(np.abs(t), eps)
    return u, v


def algebraic_fit(u, v):
    A = np.vstack([u, np.ones_like(u)]).T
    a, b = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(a), float(b)


def transcendental_coords(t, r):
    lt = np.log(np.maximum(np.abs(t), 1e-12))
    lr = np.log(np.maximum(np.abs(r), 1e-12))
    p = np.exp(1j * lt)
    q = np.exp(1j * lr)
    return p, q


def log_rank_residual(p, q, alpha=ALPHA):
    """Δ_n = α · Im( q̄ p ) / (1 + |p|²)  ·  transcendental residual."""
    num = np.imag(np.conj(q) * p)
    den = 1.0 + np.abs(p) ** 2
    return alpha * num / den


# ============================================================
#  Result container
# ============================================================
@dataclass
class ContractedLogRank:
    t: np.ndarray
    r: np.ndarray
    a: float
    b: float
    O: np.ndarray
    E: np.ndarray
    Z: np.ndarray
    Pi: np.ndarray                 # electron cloud density
    delta: np.ndarray              # transcendental residual
    G: np.ndarray                  # coupling matrix g_ij
    gbar: np.ndarray               # per-event total coupling (normalized)
    Z_ell: float                   # Π-weighted log-rank statistic
    var_ell: float
    chi2_ell: float
    Z_coup: float                  # Π · ḡ weighted statistic
    var_coup: float
    chi2_coup: float
    S_ell: float
    H_ell: float
    m_ell: float
    S_coup: float
    H_coup: float
    m_coup: float


# ============================================================
#  Contracted log-rank test
# ============================================================
def contracted_log_rank(t: np.ndarray, r: np.ndarray
                        ) -> ContractedLogRank:
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    N = len(t)

    # ---------- algebraic fit ----------
    u, v = inverse_coords(r, t)
    a, b = algebraic_fit(u, v)

    # ---------- observed / expected / residual ----------
    O  = np.diff(r, prepend=r[0])
    dU = np.diff(u, prepend=u[0])
    dV = np.diff(v, prepend=v[0])
    E  = a * dU + b * dV
    Z  = O - E

    # ---------- transcendental ----------
    p, q = transcendental_coords(t, r)
    delta = log_rank_residual(p, q)

    # ---------- electron cloud ----------
    Pi = electron_density(t, r)

    # ---------- fermion / photon coupling ----------
    G = coupling_matrix(t, r)
    gbar = coupling_strength(G)

    # ---------- elliptic log-rank (Π only) ----------
    Z_ell   = float(np.sum(Pi * Z))
    var_ell = float(np.sum((Pi * delta) ** 2))
    chi2_ell = (Z_ell ** 2) / var_ell if var_ell > 0 else 0.0

    # ---------- contracted log-rank (Π · ḡ) ----------
    W = Pi * gbar                       # total local weight
    Z_coup   = float(np.sum(W * Z))
    var_coup = float(np.sum((W * delta) ** 2))
    chi2_coup = (Z_coup ** 2) / var_coup if var_coup > 0 else 0.0

    # ---------- supertraces ----------
    S_ell = supertrace(Pi * Z)
    H_ell = entropy(S_ell, N)
    m_ell = invariant_mass(S_ell, N)

    S_coup = supertrace(W * Z)
    H_coup = entropy(S_coup, N)
    m_coup = invariant_mass(S_coup, N)

    return ContractedLogRank(
        t=t, r=r, a=a, b=b, O=O, E=E, Z=Z,
        Pi=Pi, delta=delta, G=G, gbar=gbar,
        Z_ell=Z_ell, var_ell=var_ell, chi2_ell=chi2_ell,
        Z_coup=Z_coup, var_coup=var_coup, chi2_coup=chi2_coup,
        S_ell=S_ell, H_ell=H_ell, m_ell=m_ell,
        S_coup=S_coup, H_coup=H_coup, m_coup=m_coup,
    )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Contracted elliptic log-rank  ·  fermion-photon + electron cloud")
    print("=" * 78)
    print(f"  α       = 1/(π − e) = {ALPHA:.6f}")
    print(f"  periods ω₁ = π      = {W1:.6f}")
    print(f"  periods ω₂ = e      = {W2:.6f}")
    print()
    print("  coupling  g_ij = w_ij · κ_ij")
    print("    w_ij = |χ_i − χ_j| / π         chiral bond weight")
    print("    κ_ij = √(L_i L_j) / L_ref      travel coupling")
    print("  cloud     Π(t, r)                 Figure 3.5")
    print()

    rng = np.random.default_rng(2024)

    # ---------- synthetic groups ----------
    tA = np.sort(rng.exponential(scale=2.0, size=60))
    tB = np.sort(rng.exponential(scale=4.0, size=60))
    tA = np.maximum(tA, 1e-3)
    tB = np.maximum(tB, 1e-3)
    rA = np.arange(1, len(tA) + 1, dtype=float)
    rB = np.arange(1, len(tB) + 1, dtype=float)

    t_all = np.sort(np.concatenate([tA, tB]))
    r_all = np.arange(1, len(t_all) + 1, dtype=float)

    # ---------- run ----------
    resA = contracted_log_rank(tA, rA)
    resB = contracted_log_rank(tB, rB)
    resAll = contracted_log_rank(t_all, r_all)

    # ---------- report ----------
    def _report(name, res):
        gmean = res.G[res.G > 0].mean() if np.any(res.G > 0) else 0.0
        gmax  = res.G.max()
        print(f"--- {name} ---")
        print(f"  N events             : {len(res.t)}")
        print(f"  algebraic a          : {res.a:+.6f}")
        print(f"  algebraic b          : {res.b:+.6f}")
        print(f"  ⟨g_ij⟩               : {gmean:.6f}")
        print(f"  max g_ij             : {gmax:.6f}")
        print(f"  Π range              : [{res.Pi.min():.4f}, {res.Pi.max():.4f}]")
        print(f"  ḡ range              : [{res.gbar.min():.4f}, {res.gbar.max():.4f}]")
        print(f"  Δ range              : [{res.delta.min():+.4f}, {res.delta.max():+.4f}]")
        print(f"  Z_ell                : {res.Z_ell:+.6f}")
        print(f"  chi²_ell             : {res.chi2_ell:.6e}")
        print(f"  Z_coup               : {res.Z_coup:+.6f}")
        print(f"  chi²_coup            : {res.chi2_coup:.6e}")
        print(f"  S_ell / m_ell        : {res.S_ell:+.6f} / {res.m_ell:.6e}")
        print(f"  S_coup / m_coup      : {res.S_coup:+.6f} / {res.m_coup:.6e}")
        print()

    _report("Group A (exponential, scale 2.0)", resA)
    _report("Group B (exponential, scale 4.0)", resB)
    _report("Union", resAll)

    # ---------- plots ----------
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # (a) coupling matrix (union)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(resAll.G, origin="lower", cmap="inferno",
                   aspect="equal")
    ax.set_xlabel("event j"); ax.set_ylabel("event i")
    ax.set_title(f"Coupling matrix  g_ij = w_ij · κ_ij\n"
                 f"⟨g⟩ = {resAll.G[resAll.G>0].mean():.4f}   "
                 f"max = {resAll.G.max():.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) normalized per-event coupling
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(resA.t, resA.gbar, color="#3a7bd5", lw=1.2,
            label="Group A")
    ax.plot(resB.t, resB.gbar, color="#e74c3c", lw=1.2, alpha=0.8,
            label="Group B")
    ax.axhline(1.0, color="k", ls=":", lw=0.7)
    ax.set_xlabel("time t"); ax.set_ylabel("ḡ_n")
    ax.set_title("Per-event coupling strength (normalized)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) electron cloud density
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(resA.t, resA.Pi, color="#8e44ad", lw=1.2,
            label="Π(t, r)  Group A")
    ax.plot(resB.t, resB.Pi, color="#e67e22", lw=1.2, alpha=0.8,
            label="Π(t, r)  Group B")
    ax.set_xlabel("time t"); ax.set_ylabel("Π ∈ [0, 1]")
    ax.set_title("Electron cloud density  (Figure 3.5)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) residuals — raw vs contracted
    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(resA.t, resA.Z, color="#2c3e50", lw=1.0,
            alpha=0.55, label="raw Z_n")
    ax.plot(resA.t, resA.Pi * resA.gbar * resA.Z, color="#16a085",
            lw=1.4, label="Π · ḡ · Z_n")
    ax.set_xlabel("time t"); ax.set_ylabel("residual")
    ax.set_title("Group A  ·  raw vs contracted residual")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (e) supertrace evolution
    ax = fig.add_subplot(gs[1, 1])
    z_ell_cum  = np.cumsum(resAll.Pi * resAll.Z)
    z_coup_cum = np.cumsum(resAll.Pi * resAll.gbar * resAll.Z)
    ax.plot(resAll.t, z_ell_cum, color="#8e44ad", lw=1.4,
            label=f"S_ell cumulative = {resAll.S_ell:+.4f}")
    ax.plot(resAll.t, z_coup_cum, color="#16a085", lw=1.4,
            label=f"S_coup cumulative = {resAll.S_coup:+.4f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time t"); ax.set_ylabel("cumulative supertrace")
    ax.set_title("Supertrace accumulation")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (f) mass landscape
    ax = fig.add_subplot(gs[1, 2])
    labels = ["S_ell", "S_coup", "m_ell", "m_coup"]
    vals = [resAll.S_ell, resAll.S_coup, resAll.m_ell, resAll.m_coup]
    colors = ["#8e44ad", "#16a085", "#c0392b", "#e67e22"]
    ax.bar(labels, vals, color=colors, edgecolor="k")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Invariant summary  ·  union stream")
    ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"Contracted elliptic log-rank  ·  "
        f"g_ij = w_ij · κ_ij  ·  Π(t,r) from Figure 3.5  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---------- summary ----------
    print("=" * 78)
    print("What each layer contributes")
    print("=" * 78)
    print("  Classical log-rank  :  Z = Σ (O_n − E_n)")
    print("  Elliptic log-rank   :  Z_ell  = Σ Π_n · Z_n")
    print("  Contracted log-rank :  Z_coup = Σ Π_n · ḡ_n · Z_n")
    print()
    print("  Π_n    = electron cloud density at (t_n, r_n)")
    print("         → modulates how much each event couples to the EM field")
    print()
    print("  ḡ_n    = normalized total coupling of event n to all others")
    print("         → the fermion-photon activity at event n")
    print()
    print("  Product Π · ḡ · Z")
    print("         → the true EM emission/absorption of the log-rank residual")
    print()
    print("  The supertrace S_coup and mass m_coup summarize the")
    print("  contracted residual stream as a single invariant.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()