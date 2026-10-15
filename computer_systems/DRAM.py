#!/usr/bin/env python3
"""
fermion_dram.py
===============

A DRAM in which every bit is stored as an entangled fermion pair.

Design
------
Memory cell
    A 6×6 kissing-number spinor Z_n built from the 12 icosahedron
    vertices lifted to ℝ⁶  (coupling_Q_entanglement.py).
    Each cell exists only at a *square-free* address k  (μ(k) ≠ 0).

Address bus
    The Möbius sieve μ(k) ∈ {-1, 0, +1} from mobius-memory.py acts
    as a READ-ONLY decoder: only μ(k) ≠ 0 addresses are physical.

Bit encoding
    The stored integer (the "coupling strength") appears directly
    as an entry of the coupling matrix

        G_ij = κ_ij · v_i · v_j ,   κ_ij = √(L_i L_j)/L_ref

    No normalization is applied — G_ij *is* the memory.

Read path
    A point particle is launched in the transcendental plane (complex s)
    and follows the Möbius–Dirichlet flow

        ds/dt = −ζ_μ'(s) ,   ζ_μ(s) = Σ_k μ(k) k^{-s}

    Its position at each step is snapped to the nearest square-free
    address, which is then read out through the coupling matrix.

No norm appears anywhere — reconstruction is purely by coupling.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.3623
PHI   = (1.0 + math.sqrt(5.0)) / 2.0


# ============================================================
#  Möbius address bus   (from mobius-memory.py)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    """Linear sieve for μ(k), k = 0..K."""
    mu = np.zeros(K + 1, dtype=np.int8)
    if K < 1:
        return mu
    mu[1] = 1
    primes: List[int] = []
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K: break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Kissing-number spinors   (from coupling_Q_entanglement.py)
# ============================================================
def icosahedron_vertices() -> np.ndarray:
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.append((0.0,      s1,      s2 * PHI))
            verts.append((s1,       s2 * PHI, 0.0))
            verts.append((s2 * PHI, 0.0,     s1))
    V = np.array(verts, dtype=float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def lift_to_six_dim(V3: np.ndarray) -> np.ndarray:
    V6 = np.concatenate([V3, PHI * V3], axis=1)
    n = np.linalg.norm(V6, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return V6 / n


# -- Levi-Civita as a 6×6 determinant  (720 terms) --
_PERMS_6 = list(permutations(range(6)))
_SIGNS_6: List[int] = []
for _p in _PERMS_6:
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6) if _p[i] > _p[j])
    _SIGNS_6.append((-1) ** _inv)


def levi_civita(Z: np.ndarray) -> complex:
    total = 0.0 + 0.0j
    for perm, sign in zip(_PERMS_6, _SIGNS_6):
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


# ============================================================
#  Fermion cell
# ============================================================
@dataclass
class FermionCell:
    addr:   int
    value:  int                        # integer memory word (coupling)
    t:      float
    r:      float
    spinor: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.spinor is None:
            self.spinor = self._build_spinor()

    def _build_spinor(self) -> np.ndarray:
        V6 = lift_to_six_dim(icosahedron_vertices())
        idx = np.arange(6)
        phase = np.outer(self.t * idx, np.ones(6)) + \
                np.outer(np.ones(6), self.r * idx)
        return V6[:6, :] * np.exp(1j * phase)

    def pi(self) -> complex:
        """Π(Z) = det(Z)  —  the 720-term Levi-Civita contraction."""
        return levi_civita(self.spinor)


def travel_lengths(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0])
    dr = np.diff(r, prepend=r[0])
    ds = np.sqrt(dt * dt + dr * dr)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-6


# ============================================================
#  Fermion DRAM
# ============================================================
class FermionDRAM:
    """DRAM where every cell is an entangled fermion pair."""

    def __init__(self, K: int = 120, coupling_strength: int = 1):
        self.K = int(K)
        self.coupling_strength = int(coupling_strength)
        self.mu = mobius_sieve(self.K)
        # float μ-vector for the Dirichlet flow: index 0 ↔ k = 1
        self.mu_arr = np.array([float(self.mu[k])
                                for k in range(1, self.K + 1)])

        # ---- populate cells at square-free addresses only ----
        self.cells: Dict[int, FermionCell] = {}
        for k in range(1, self.K + 1):
            if self.mu[k] != 0:
                t_k = math.log(k) if k > 1 else 1.0
                r_k = math.sqrt(k) / math.sqrt(self.K)
                self.cells[k] = FermionCell(
                    addr=k,
                    value=self.coupling_strength,
                    t=t_k, r=r_k,
                )

    # --------------------------------------------------------
    #  Address helpers
    # --------------------------------------------------------
    def address_list(self) -> List[int]:
        return sorted(self.cells.keys())

    def is_live(self, addr: int) -> bool:
        return 1 <= addr <= self.K and self.mu[addr] != 0

    # --------------------------------------------------------
    #  Direct write / read  (integer memory; no norms)
    # --------------------------------------------------------
    def write(self, addr: int, value: int) -> None:
        if not self.is_live(addr):
            raise ValueError(f"address {addr} is not square-free — cell absent")
        self.cells[addr].value = int(value)

    def read(self, addr: int) -> int:
        """Möbius-gated read — coupling matrix not consulted here."""
        if not self.is_live(addr) or addr not in self.cells:
            return 0
        return self.cells[addr].value

    # --------------------------------------------------------
    #  Coupling matrix  (raw — no normalization)
    # --------------------------------------------------------
    def coupling_matrix(self) -> Tuple[np.ndarray, List[int]]:
        addrs = self.address_list()
        t = np.array([self.cells[a].t for a in addrs])
        r = np.array([self.cells[a].r for a in addrs])
        V = np.array([self.cells[a].value for a in addrs], dtype=float)

        L = travel_lengths(t, r)
        L_ref = float(np.median(L[L > 0])) if np.any(L > 0) else 1.0
        kappa = np.sqrt(np.maximum(np.outer(L, L), 0.0)) / L_ref
        kappa = np.clip(kappa, 0.0, 1.0)

        # G_ij = κ_ij · v_i · v_j      (memory × coupling)
        G = kappa * np.outer(V, V)
        np.fill_diagonal(G, 0.0)
        return G, addrs

    # --------------------------------------------------------
    #  Möbius buffer scan  (read-only iteration over live cells)
    # --------------------------------------------------------
    def mobius_scan(self) -> List[Tuple[int, int, int]]:
        """Iterate k=1..K; μ(k) is the gate.  (addr, μ, value)."""
        out = []
        for k in range(1, self.K + 1):
            m = int(self.mu[k])
            if m != 0 and k in self.cells:
                out.append((k, m, self.cells[k].value))
        return out

    # --------------------------------------------------------
    #  Transcendental-plane particle search
    # --------------------------------------------------------
    def dzeta_flow(self, s0: complex,
                   steps: int = 300, dt: float = 0.01) -> np.ndarray:
        """
        Integrate   ds/dt = −ζ_μ'(s)   with

            ζ_μ(s)  = Σ_k μ(k) k^{-s}
            ζ_μ'(s) = −Σ_k μ(k) ln(k) k^{-s}

        Returns the complex trajectory s(t).
        """
        K = self.K
        k = np.arange(1, K + 1, dtype=float)
        logk = np.log(k)
        mu = self.mu_arr

        s = complex(s0)
        path = [s]
        for _ in range(steps):
            # -ζ_μ'(s) = + Σ μ(k) ln(k) k^{-s}
            ds = complex(np.sum(mu * logk * np.exp(-s * logk))) * dt
            if not np.isfinite(ds):
                break
            s = s + ds
            path.append(s)
            if abs(s) > 1e6:
                break
        return np.array(path)

    def snap_to_live(self, s: complex) -> Optional[int]:
        """Nearest square-free address to  |s|."""
        n = int(round(abs(s)))
        n = max(1, min(self.K, n))
        for off in range(self.K):
            for cand in (n + off, n - off):
                if 1 <= cand <= self.K and self.mu[cand] != 0:
                    return cand
        return None

    def read_via_particle(self, s0: complex,
                          steps: int = 300, dt: float = 0.01
                          ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Launch particle at s0.  At every step snap to a live address
        and read its coupling value.  Returns (trajectory, readings).
        """
        path = self.dzeta_flow(s0, steps=steps, dt=dt)
        readings: List[Tuple[int, int]] = []
        for s in path:
            addr = self.snap_to_live(s)
            if addr is not None and addr in self.cells:
                readings.append((addr, self.cells[addr].value))
        return path, readings


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Fermion DRAM  ·  entangled-fermion bits  ·  Möbius address bus")
    print("=" * 78)
    K = 120
    dram = FermionDRAM(K=K, coupling_strength=7)
    print(f"  K                     = {K}")
    print(f"  coupling strength     = {dram.coupling_strength}")
    print(f"  live (square-free)    = {len(dram.cells)} / {K}")
    print(f"  density (6/π² target) = {len(dram.cells)/K:.4f}   "
          f"(6/π² = {6/PI**2:.4f})")
    print()

    # ---- write distinct integers to every live cell ----
    rng = np.random.default_rng(0)
    addrs = dram.address_list()
    for a in addrs:
        dram.write(a, int(rng.integers(1, 64)))
    print("  wrote random 1..63 to every live cell")
    print()

    # ---- read back via the Möbius buffer ----
    scan = dram.mobius_scan()
    print(f"  Möbius scan kept {len(scan)} cells")
    print(f"  first 8 (addr, μ, val): {scan[:8]}")
    ok = all(dram.read(a) == v for a, _, v in scan)
    print(f"  direct read consistent: {ok}")
    print()

    # ---- coupling matrix ----
    G, gaddrs = dram.coupling_matrix()
    print(f"  coupling matrix shape = {G.shape}")
    print(f"  G max                 = {G.max():.4f}")
    print(f"  G mean (off-diag)     = {G[G > 0].mean():.4f}")
    print()

    # ---- particle read through the transcendental plane ----
    s0 = 1.5 + 5.0j
    path, readings = dram.read_via_particle(s0, steps=200, dt=0.02)
    unique, seen = [], set()
    for a, v in readings:
        if a not in seen:
            unique.append((a, v))
            seen.add(a)
    print(f"  particle launched at s0 = {s0}")
    print(f"  trajectory length      = {len(path)}")
    print(f"  distinct addresses hit = {len(unique)}")
    print(f"  first 10 hits          = {unique[:10]}")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # (a) Möbius address bus
    ax = fig.add_subplot(gs[0, 0])
    ax.vlines(np.arange(1, K + 1), 0, np.abs(dram.mu[1:K+1]),
              color="#e74c3c", lw=1.0)
    ax.set_xlabel("address k"); ax.set_ylabel("|μ(k)|")
    ax.set_title(f"Möbius address bus\n{len(dram.cells)} live / {K}")
    ax.grid(alpha=0.3)

    # (b) coupling matrix
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(G, origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel("cell j"); ax.set_ylabel("cell i")
    ax.set_title("coupling matrix  G_ij = κ_ij · v_i · v_j")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) kissing-number spinor invariant
    ax = fig.add_subplot(gs[0, 2])
    pis = np.array([abs(dram.cells[a].pi()) for a in gaddrs])
    ax.plot(gaddrs, pis, "o-", color="#8e44ad", ms=4, lw=1.0)
    ax.set_xlabel("address k"); ax.set_ylabel(r"$|\Pi(Z_k)|$")
    ax.set_title("Kissing-number spinor invariant  |det Z|")
    ax.grid(alpha=0.3)

    # (d) particle trajectory in the transcendental plane
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(path.real, path.imag, "-", color="#2c3e50", lw=1.0)
    ax.plot(path[0].real, path[0].imag, "go", ms=8, label="start")
    ax.plot(path[-1].real, path[-1].imag, "rs", ms=8, label="end")
    for a, _ in unique:
        ax.axvline(a, color="orange", alpha=0.10)
    ax.set_xlabel("Re(s)"); ax.set_ylabel("Im(s)")
    ax.set_title(r"$d\zeta_\mu/dt$ particle trajectory")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) landing histogram
    ax = fig.add_subplot(gs[1, 1])
    addrs_hit = [a for a, _ in readings]
    ax.hist(addrs_hit, bins=40, color="#16a085", alpha=0.85)
    ax.set_xlabel("address k"); ax.set_ylabel("visits")
    ax.set_title("particle landing distribution")
    ax.grid(alpha=0.3)

    # (f) memory content
    ax = fig.add_subplot(gs[1, 2])
    vals = np.array([dram.cells[a].value for a in gaddrs])
    ax.plot(gaddrs, vals, "o-", color="#e74c3c", ms=4, lw=1.0)
    ax.set_xlabel("address k"); ax.set_ylabel("stored int")
    ax.set_title("Memory content  (integer coupling strength)")
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"Fermion DRAM  ·  K = {K}  ·  "
        f"μ-bus density = {len(dram.cells)/K:.4f}  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    demo()