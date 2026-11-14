#!/usr/bin/env python3
"""
mosb_dram_ssd.py
================

An SSD with DRAM-style random access.
Signal-in (ALU reg2sig + ALU arclength) → even-even coupling
(coupling_Q_entanglement.py) → AC-carrier waveform out E(t).

Signal path
-----------
    register x
        │  alu_finite_step.QuantumALU.reg2sig
        ▼
    s_x[k] = x · μ(k) · cos(A_STEP · k² / K_alu)      input signal
        │
        │  alu_finite_step.QuantumALU.travel
        ▼
    L_i  = Σ_{m ≤ i} |s_x[m] − s_x[m−1]|              arclength of signal
        │
        │  coupling_Q_entanglement.coupling_matrix structure:
        │      κ_ij = √(L_i L_j) / L_ref               (ALU arclength)
        │      w_ij = |χ_i − χ_j| / π                  (chiral bond)
        │      ρ_ij = √(ρ_i ρ_j)                       (odd-slot particles)
        │      g_ij = w_ij · κ_ij · (1−w_d + w_d·ρ_ij)
        │
        │  coupling_Q_entanglement.entangled_coupling:
        │      M_ij = [i ≡ j ≡ 0 (mod 2)]
        ▼
    G_even[i, j]                                       physical cell array
        │
        │  DRAM read (row activate → column select → sense → data bus)
        ▼
    readout(i, j, t) = G_even[i, j] · s_x[i] · E(t)
    E(t)             = (S_νμ + S_ντ) · cos(ω t)

DRAM semantics
--------------
    write(i, j, v)       O(1)      passive cell update
    read_amplitude(i, j) O(1)      scalar decoupling (no waveform)
    read(i, j)           O(T)      E(t) waveform on the data bus
    read_burst(i, k)     O(k·T)    row activate + k-column burst
    read_random([...])   O(n·T)    arbitrary order, no page penalty
    read_all()           O(N²·T)   full page read (page-mode)

Differences from DRAM
---------------------
    No refresh cycles     — cell value is passive coupling, not charge
    No ECC                 — even-even mask is structural, not a code
    Readout is a WAVEFORM  — the AC carrier is the DRAM data bus
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- input signal + arclength (alu_finite_step.py) ----
from alu_finite_step import (
    QuantumALU,
    ALPHA,
    A_STEP,
)

# ---- coupling / even-even / AC carrier (coupling_Q_entanglement.py) ----
from coupling_Q_entanglement import (
    chiral_angles,
    entangled_coupling,
    even_even_mask,
    point_particle_density,
    compute_S_mu_tau,
    ac_carrier_field,
    make_geo_events,
    PI,
)

# ---- DC gate for E(t) ----  
from DC_chirp import DCMobiusGate


# ============================================================
#  ALU arclength  (exactly QuantumALU.travel)
# ============================================================
def alu_travel(s: np.ndarray) -> np.ndarray:
    """L_i = cumulative arclength of signal s  (ALU.travel)."""
    ds = np.abs(np.diff(s, prepend=s[0]))
    return np.cumsum(ds) + 1e-9


# ============================================================
#  SSD-with-DRAM-semantics
# ============================================================
class MOSB_DRAM_SSD:
    """
    SSD with DRAM random access.  Signal-in, AC-carrier-out.

    Physical layout:
        rows a ∈ {0, 1, …, N/2 − 1}        (even indices 2a)
        cols b ∈ {0, 1, …, N/2 − 1}        (even indices 2b)
        cell (a, b)  →  stored coupling  G[a, b]  (compact even-even)

    The compact cell value G[a, b] = G_even[2a, 2b] where G_even comes
    from the coupling_Q_entanglement.py pipeline.  Input signal s_x is
    the ALU reg2sig signal; row a injects s_x[2a].
    """

    def __init__(self,
                 K_alu:          int   = 64,
                 register_x:     float = 7.0,
                 K_ci:           int   = 20,
                 density_weight: float = 0.5,
                 omega:          float = 1.0,
                 t_samples:      int   = 256,
                 seed:           int   = 2024):
        # ---------- 1. input signal (ALU) ----------
        self.K_alu      = int(K_alu)
        self.register_x = float(register_x)
        self.alu        = QuantumALU(self.K_alu)
        self.s_x_full   = self.alu.reg2sig(self.register_x)   # (N,)
        self.N          = self.K_alu
        self.N_half     = self.N // 2

        # ---------- 2. arclength of signal (ALU travel) ----------
        self.L_full     = alu_travel(self.s_x_full)           # (N,)
        L_ref           = float(np.median(
            self.L_full[self.L_full > 1e-9])) \
            if np.any(self.L_full > 1e-9) else 1.0
        self.L_ref      = L_ref

        # ---------- 3. event times for chiral bond ----------
        self.t_events, _ = make_geo_events(
            n_events=self.N, seed=seed)
        chi_full         = chiral_angles(self.t_events)       # (N,)

        # ---------- 4. full coupling matrix ----------
        #    κ_ij  from ALU arclength
        kappa_full = np.sqrt(np.outer(self.L_full, self.L_full)) / L_ref
        kappa_full = np.clip(kappa_full, 0.0, 1.0)

        #    w_ij  from event times
        dchi_full = np.abs(chi_full[:, None] - chi_full[None, :])
        dchi_full = (dchi_full + PI) % (2 * PI) - PI
        w_full    = np.abs(dchi_full) / PI

        #    ρ_ij  from odd-slot point particles
        rho_full     = point_particle_density(self.N, self.s_x_full)
        rho_max      = float(np.max(rho_full)) if rho_full.size else 1.0
        rho_norm     = (rho_full / (rho_max + 1e-12)
                        if rho_max > 1e-12 else rho_full)
        rho_ij_full  = np.sqrt(
            np.maximum(rho_norm[:, None] * rho_norm[None, :], 0.0))
        modulation   = ((1.0 - density_weight)
                        + density_weight * rho_ij_full)

        #    G = w · κ · modulation
        G_full = w_full * kappa_full * modulation
        np.fill_diagonal(G_full, 0.0)
        self.G_full = G_full

        # ---------- 5. even-even mask ----------
        self.G_even_full, self.mask = entangled_coupling(G_full)

        # ---------- 6. compact even-even array (physical cells) ----------
        self.G = self.G_even_full[::2, ::2].astype(float)     # (N/2, N/2)
        self.s_x = self.s_x_full[::2].astype(float)           # (N/2,)
        self.L   = self.L_full[::2].astype(float)             # (N/2,)

        # ---------- 7. AC carrier  E(t) = (S_νμ+S_ντ)cos(ωt) ----------
        self.gate   = DCMobiusGate(K=K_ci)
        self.t_axis = np.linspace(0.0, self.gate.period, t_samples)
        self.S_mu, self.S_tau = compute_S_mu_tau(
            self.t_axis, self.gate, N_events=41)
        self.E_ac   = ac_carrier_field(
            self.t_axis, self.S_mu, self.S_tau, omega=omega)
        self.omega     = float(omega)
        self.t_samples = int(t_samples)

        # ---------- 8. read log ----------
        self._read_log: List[Dict] = []

    # ==================================================
    #  Address helpers
    # ==================================================
    def even_addresses(self) -> List[Tuple[int, int]]:
        """All valid DRAM cells  (compact even-even address space)."""
        return [(a, b)
                for a in range(self.N_half)
                for b in range(self.N_half)]

    def is_physical(self, a: int, b: int) -> bool:
        return 0 <= a < self.N_half and 0 <= b < self.N_half

    # ==================================================
    #  Write
    # ==================================================
    def write(self, a: int, b: int, value: float) -> None:
        """Write scalar to cell (a, b).  O(1)."""
        if not self.is_physical(a, b):
            raise ValueError(
                f"cell ({a},{b}) out of range "
                f"0..{self.N_half - 1}")
        self.G[a, b] = float(value)

    # ==================================================
    #  Read  ·  returns the AC carrier waveform E(t)
    # ==================================================
    def read(self, a: int, b: int,
             t: Optional[np.ndarray] = None,
             record: bool = False) -> np.ndarray:
        """
        DRAM random-access read at cell (a, b).

            readout(t) = G[a, b] · s_x[a] · E(t)

        Returns the AC carrier E(t) scaled by the cell amplitude
        G[a, b] · s_x[a].  The temporal shape is always E(t).
        """
        if not self.is_physical(a, b):
            raise ValueError(
                f"cell ({a},{b}) out of range "
                f"0..{self.N_half - 1}")

        if t is None:
            E = self.E_ac
        else:
            E = np.interp(np.asarray(t, dtype=float),
                          self.t_axis, self.E_ac)

        amplitude = float(self.G[a, b] * self.s_x[a])
        waveform  = amplitude * E

        if record:
            self._read_log.append(dict(
                address=(int(a), int(b)),
                amplitude=amplitude,
                t0=float(time.time()),
            ))

        return waveform

    # --------------------------------------------------
    def read_amplitude(self, a: int, b: int) -> float:
        """Scalar read (no waveform reconstruction).  O(1)."""
        if not self.is_physical(a, b):
            raise ValueError(
                f"cell ({a},{b}) out of range "
                f"0..{self.N_half - 1}")
        return float(self.G[a, b] * self.s_x[a])

    # --------------------------------------------------
    def read_burst(self, a: int, burst_len: int = 4,
                   record: bool = False
                   ) -> Dict[int, np.ndarray]:
        """DRAM burst read from row a  ·  `burst_len` consecutive cols."""
        if not 0 <= a < self.N_half:
            raise ValueError(f"row {a} out of range 0..{self.N_half - 1}")
        cols = list(range(min(burst_len, self.N_half)))
        return {b: self.read(a, b, record=record) for b in cols}

    # --------------------------------------------------
    def read_random(self,
                    addresses: List[Tuple[int, int]],
                    record: bool = False
                    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Random access  ·  arbitrary cells, any order."""
        return {(a, b): self.read(a, b, record=record)
                for (a, b) in addresses}

    # --------------------------------------------------
    def read_all(self) -> np.ndarray:
        """
        Full-page read.
        Returns (N/2, N/2, T) tensor of E(t)-shaped readouts.
        """
        Nh, T = self.N_half, self.t_samples
        out = np.zeros((Nh, Nh, T))
        for a in range(Nh):
            for b in range(Nh):
                out[a, b] = self.read(a, b)
        return out

    # ==================================================
    #  Signal-through-SSD  ·  s_x  →  G  →  E(t)
    # ==================================================
    def send_signal(self,
                    s_in: Optional[np.ndarray] = None
                    ) -> Dict:
        """
        Send a signal through the SSD.

        coupled  = G @ s_in                 (column-summed coupling)
        output   = coupled[:, None] · E(t)  (temporal output)
        """
        s = self.s_x if s_in is None else np.asarray(s_in, dtype=float)
        if s.size != self.N_half:
            s = np.pad(s, (0, self.N_half - s.size))[:self.N_half]
        coupled = self.G @ s
        out_t   = np.outer(coupled, self.E_ac)
        return dict(signal_in=s, coupled=coupled, output=out_t)

    # ==================================================
    #  Diagnostics
    # ==================================================
    def stats(self) -> Dict:
        nz = self.G[self.G > 0]
        return dict(
            N=self.N,
            N_half=self.N_half,
            K_alu=self.K_alu,
            K_ci=self.gate.K,
            register_x=self.register_x,
            L_ref=self.L_ref,
            L_total=float(self.L_full[-1]),
            n_cells=int(np.count_nonzero(self.G)),
            n_total_cells=int(self.N_half ** 2),
            cell_min=float(nz.min()) if nz.size else 0.0,
            cell_max=float(nz.max()) if nz.size else 0.0,
            cell_mean=float(nz.mean()) if nz.size else 0.0,
            s_in_rms=float(np.sqrt(np.mean(self.s_x ** 2))),
            E_rms=float(np.sqrt(np.mean(self.E_ac ** 2))),
            E_min=float(self.E_ac.min()),
            E_max=float(self.E_ac.max()),
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("MOSB DRAM-like SSD  ·  signal-in, AC-carrier-out")
    print("=" * 78)
    print(f"  ALPHA     = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  A_STEP    = ALPHA/0.3628   = {A_STEP:.6f}")
    print()

    ssd = MOSB_DRAM_SSD(
        K_alu=64, register_x=7.0, K_ci=20,
        density_weight=0.5, omega=1.0,
        t_samples=256, seed=2024)

    # ================================================================
    print("=" * 78)
    print("1  ·  SSD configuration")
    print("=" * 78)
    st = ssd.stats()
    for k in ("N", "N_half", "K_alu", "K_ci", "register_x",
             "L_ref", "L_total", "n_cells", "n_total_cells",
             "cell_min", "cell_max", "cell_mean",
             "s_in_rms", "E_rms", "E_min", "E_max"):
        v = st[k]
        if isinstance(v, float):
            print(f"  {k:<16s} = {v:+.6e}")
        else:
            print(f"  {k:<16s} = {v}")
    print()

    # ================================================================
    print("=" * 78)
    print("2  ·  input signal s_x  (ALU reg2sig)  +  arclength (ALU travel)")
    print("=" * 78)
    print(f"  s_x[0:8]      = {[f'{v:+.4f}' for v in ssd.s_x[:8]]}")
    print(f"  ⟨s_x⟩         = {np.mean(ssd.s_x):+.6f}")
    print(f"  rms(s_x)      = {np.sqrt(np.mean(ssd.s_x**2)):.6f}")
    print(f"  L[0]..L[3]    = {[f'{v:.4f}' for v in ssd.L[:4]]}")
    print(f"  L_total       = {ssd.L_full[-1]:.6f}")
    print()

    # ================================================================
    print("=" * 78)
    print("3  ·  DRAM random access  ·  read arbitrary cells")
    print("=" * 78)
    addresses = [(0, 0), (1, 2), (5, 5), (12, 4), (16, 20), (30, 30)]
    print(f"  {'addr':>10s}  {'amplitude':>15s}  "
          f"{'waveform rms':>15s}  {'peak':>12s}")
    print("  " + "-" * 60)
    for (a, b) in addresses:
        amp  = ssd.read_amplitude(a, b)
        wave = ssd.read(a, b)
        rms  = float(np.sqrt(np.mean(wave ** 2)))
        peak = float(np.max(np.abs(wave)))
        print(f"  ({a:>3},{b:>3})  {amp:>+15.6e}  "
              f"{rms:>+15.6e}  {peak:>12.6e}")
    print()

    # ================================================================
    print("=" * 78)
    print("4  ·  burst read  ·  row activate + multi-column")
    print("=" * 78)
    burst = ssd.read_burst(a=5, burst_len=4)
    for b, wave in burst.items():
        print(f"  row=5  col={b}  "
              f"rms={np.sqrt(np.mean(wave ** 2)):.4e}")
    print()

    # ================================================================
    print("=" * 78)
    print("5  ·  random access timing")
    print("=" * 78)
    rng = np.random.default_rng(0)
    Nh  = ssd.N_half

    n_scalar = 256
    t0 = time.perf_counter()
    for _ in range(n_scalar):
        ssd.read_amplitude(int(rng.integers(Nh)), int(rng.integers(Nh)))
    t_scalar = (time.perf_counter() - t0) / n_scalar

    n_wave = 32
    t0 = time.perf_counter()
    for _ in range(n_wave):
        ssd.read(int(rng.integers(Nh)), int(rng.integers(Nh)))
    t_wave = (time.perf_counter() - t0) / n_wave

    t0 = time.perf_counter()
    ssd.read_all()
    t_page = (time.perf_counter() - t0) * 1e3

    print(f"  scalar amplitude read   = {t_scalar * 1e6:>8.2f} µs")
    print(f"  waveform read           = {t_wave  * 1e6:>8.2f} µs")
    print(f"  full page ({Nh}×{Nh})     = {t_page:>8.2f} ms")
    print(f"  per-cell (page)         = "
          f"{(t_page * 1e3) / (Nh * Nh):>8.2f} µs")
    print()

    # ================================================================
    print("=" * 78)
    print("6  ·  signal-through-SSD  ·  s_x  →  G  →  E(t)")
    print("=" * 78)
    flow = ssd.send_signal()
    print(f"  coupled shape  = {flow['coupled'].shape}")
    print(f"  ⟨coupled⟩      = {np.mean(flow['coupled']):+.4e}")
    print(f"  rms(coupled)   = "
          f"{np.sqrt(np.mean(flow['coupled']**2)):.4e}")
    print(f"  output shape   = {flow['output'].shape}")
    print(f"  rms(output)    = "
          f"{np.sqrt(np.mean(flow['output']**2)):.4e}")
    print()

    # ================================================================
    print("=" * 78)
    print("7  ·  write / read cycle")
    print("=" * 78)
    a, b = 5, 5
    v0 = ssd.read_amplitude(a, b)
    ssd.write(a, b, 0.5)
    v1 = ssd.read_amplitude(a, b)
    ssd.write(a, b, v0 / (ssd.s_x[a] + 1e-12))
    v2 = ssd.read_amplitude(a, b)
    print(f"  original  ({a},{b})  = {v0:+.6e}")
    print(f"  wrote 0.5 to ({a},{b})  → {v1:+.6e}")
    print(f"  restored  ({a},{b})  = {v2:+.6e}")
    print(f"  round-trip OK = {abs(v2 - v0) < 1e-9}")
    print()

    # ================================================================
    #  Visualization
    # ================================================================
    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.30)

    # (a) input signal
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ssd.s_x, "o-", ms=3, lw=1.0, color="#2c3e50",
            label="s_x (compact)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("row a")
    ax.set_ylabel("signal")
    ax.set_title(f"input signal s_x  ·  ALU reg2sig(x={ssd.register_x})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) arclength
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ssd.L_full, color="#e74c3c", lw=1.2, label="L_full (N)")
    ax.plot(np.arange(ssd.N_half) * 2, ssd.L, "o", ms=3,
            color="#8e44ad", label="L_even")
    ax.fill_between(np.arange(ssd.N), 0, ssd.L_full,
                    color="#e74c3c", alpha=0.15)
    ax.set_xlabel("index")
    ax.set_ylabel("L")
    ax.set_title(f"arclength (ALU travel)  ·  L_total = "
                 f"{ssd.L_full[-1]:.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) cell array (compact even-even)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(ssd.G, origin="lower",
                   cmap="inferno", aspect="equal")
    ax.set_xlabel("column b")
    ax.set_ylabel("row a")
    ax.set_title(f"cell array G  ·  "
                 f"{np.count_nonzero(ssd.G)} nonzero / "
                 f"{ssd.N_half**2}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) AC carrier E(t)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ssd.t_axis, ssd.E_ac, color="#16a085", lw=1.2,
            label="E(t) = (S_νμ+S_ντ)cos(ωt)")
    ax.plot(ssd.t_axis, ssd.S_mu, "--", lw=0.7, alpha=0.6,
            color="#e74c3c", label="S_νμ")
    ax.plot(ssd.t_axis, ssd.S_tau, ":", lw=0.7, alpha=0.6,
            color="#3498db", label="S_ντ")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("E(t)")
    ax.set_title("AC carrier  ·  DRAM data-bus waveform")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) random access readouts
    ax = fig.add_subplot(gs[1, 1])
    for (a, b) in [(0, 0), (5, 5), (16, 20), (30, 30)]:
        wave = ssd.read(a, b)
        ax.plot(ssd.t_axis, wave, lw=1.0, label=f"({a},{b})")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("readout")
    ax.set_title("random-access readouts")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # (f) burst read from row 5
    ax = fig.add_subplot(gs[1, 2])
    burst = ssd.read_burst(a=5, burst_len=4)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(burst)))
    for (b, wave), c in zip(burst.items(), colors):
        ax.plot(ssd.t_axis, wave, lw=1.0, color=c, label=f"col {b}")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("burst readout")
    ax.set_title("burst read  ·  row 5 activate, 4 columns")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

    # (g) full-page readout rms heatmap
    ax = fig.add_subplot(gs[2, 0])
    arr = ssd.read_all()
    rms_map = np.sqrt(np.mean(arr ** 2, axis=-1))
    im = ax.imshow(rms_map, origin="lower",
                   cmap="viridis", aspect="equal")
    ax.set_xlabel("column b")
    ax.set_ylabel("row a")
    ax.set_title("full-page readout rms")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (h) signal-through-SSD
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(flow["coupled"], "o-", ms=3, lw=1.2,
            color="#e67e22")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("row a")
    ax.set_ylabel("coupled column sum")
    ax.set_title("signal-through-SSD  ·  G · s_x")
    ax.grid(alpha=0.3)

    # (i) latency bar
    ax = fig.add_subplot(gs[2, 2])
    ax.bar(["scalar\n(O(1))", "waveform\n(O(T))",
            "page\n(O(N²T))"],
           [t_scalar * 1e6, t_wave * 1e6, t_page * 1e3],
           color=["#3a7bd5", "#e74c3c", "#8e44ad"],
           edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel("latency (µs)")
    ax.set_title("DRAM-style latency  ·  random access")
    ax.grid(alpha=0.3, axis="y", which="both")

    plt.suptitle(
        "MOSB DRAM-like SSD  ·  ALU s_x  +  ALU arclength  →  "
        "even-even coupling (coupling_Q_entanglement)  →  "
        f"E(t) = (S_νμ+S_ντ)cos(ωt)  ·  A_STEP = {A_STEP:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ================================================================
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print("  Input signal:")
    print("    s_x[k]  = x · μ(k) · cos(A_STEP · k² / K_alu)     (ALU)")
    print("    L_i    = Σ |Δ s_x|  ·  arclength                (ALU.travel)")
    print()
    print("  Coupling (coupling_Q_entanglement.py structure):")
    print("    κ_ij = √(L_i L_j) / L_ref                       (ALU L)")
    print("    w_ij = |χ_i − χ_j| / π                          (chiral bond)")
    print("    ρ_ij = √(ρ_i ρ_j)                               (odd-slot ρ)")
    print("    g_ij = w_ij · κ_ij · (1−w_d + w_d·ρ_ij)")
    print("    M_ij = [i ≡ j ≡ 0 (mod 2)]                      (even-even)")
    print()
    print("  DRAM readout:")
    print("    readout(a, b, t) = G[a, b] · s_x[a] · E(t)")
    print("    E(t)             = (S_νμ + S_ντ) · cos(ω t)")
    print()
    print("  DRAM parallelism preserved:")
    print("    write(a, b, v)      O(1)")
    print("    read_amplitude(a,b) O(1)")
    print("    read(a, b)          O(T)  ← waveform on the data bus")
    print("    read_burst(a, k)    O(k·T)")
    print("    read_random([...])  O(n·T)")
    print()
    print(f"  Compact cell array  : {ssd.N_half}×{ssd.N_half} = "
          f"{ssd.N_half**2} physical cells")
    print(f"  |E(t)| rms          : {st['E_rms']:.4f}")
    print(f"  ⟨|G|⟩               : {st['cell_mean']:.4e}")
    print()
    print("  No refresh, no ECC — the mask is structural.  Every")
    print("  random-access read returns a full AC-carrier waveform;")
    print("  the scalar readout is just its peak amplitude.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()