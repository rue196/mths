#!/usr/bin/env python3
"""
atomic_clock_muon_tau.py
========================

Atomic clock timekeeping from muon/tau convolution zeros.

Physics  (neutral.pdf § 2, § 4)
-------------------------------
    Muon channel:   S_νμ(t) = Σ_n (−1)^{n+1} σ_μ(J_C) |Π₆(Z_n(t))|
    Tau  channel:   S_ντ(t) = Σ_n (−1)^{n+1} σ_τ(J_C) |Π₆(Z_n(t))|
    σ_μ(J_C) = (−1)^{2J_C}          (δ_μ = 0)
    σ_τ(J_C) = (−1)^{2J_C+1}        (δ_τ = 1)     ⟹ σ_τ = −σ_μ
    ⟹  S_ντ(t) = −S_νμ(t)  exactly

    Muon–tau convolution:
        C(t) = (S_νμ * S_ντ)(t) = −(S_νμ * S_νμ)(t)   [neg. autocorr.]

    Each zero of C(t) is an "electron"  →  one clock tick.

Parity cross-check  (neutral.pdf § 4, Figure 4)
-----------------------------------------------
    Even-even mask:   M_ee[i,j] = 1 iff i ≡ j ≡ 0 (mod 2)
    Even-odd  mask:   M_eo[i,j] = 1 iff (i mod 2) ≠ (j mod 2)

    G_ee = M_ee ⊙ g_ij            (parity protected — Figure 4)
    G_eo = M_eo ⊙ g_ij            (parity breaking)

    Protected:   ‖G_ee‖ > ‖G_eo‖        (Figure 4 unbroken)
    Broken:      ‖G_ee‖ ≤ ‖G_eo‖        (parity broken)

Electron density  (Figure 3)
----------------------------
    Π_e(t, r) = ½(1 + cos 2t · cos 2πr/e)   ∈ [0,1]
    ρ_elec(t) = |ζ(t)|² · Π_e(t, r_eff)

Tick confirmation  (all three conditions)
-----------------------------------------
    (1)  C(t) sign change            → zero → electron
    (2)  ‖G_ee‖ > ‖G_eo‖             → parity protected  (Figure 4)
    (3)  Π_e(t, r_eff) > 0.5         → electron cloud coherent
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from DC_chirp import DCMobiusGate
from wire_analog import spin_grade, pi_e
from coupling_Q_entanglement import coupling_matrix


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)
A_STEP  = ALPHA / 0.3628
DENSITY = 6.0 / (PI * PI)


# ============================================================
#  AtomicClockMuonTau
# ============================================================
class AtomicClockMuonTau:
    """
    Atomic clock driven by muon–tau convolution zeros.

        S_νμ(t)       muon alternating sum
        S_ντ(t)       tau alternating sum  (= −S_νμ)
        C(t)          their convolution    (zeros = electrons)
        M_ee, M_eo    parity masks         (Figure 4 cross-check)
        ρ_elec(t)     electron cloud       (Figure 3 density)
    """

    def __init__(self,
                 K: int = 20,
                 J_C: float = 1.0,
                 N_events: int = 41,
                 N_samples: int = 2048):
        self.K = int(K)
        self.J_C = float(J_C)
        self.N_events = int(N_events)
        self.N_samples = int(N_samples)

        # ---- DC Möbius gate ----
        self.gate = DCMobiusGate(K=K, smooth=True)
        self.T = self.gate.period
        self.dt_event = self.T / self.N_events

        # ---- time grid ----
        self.t = np.linspace(0.0, self.T, self.N_samples)
        self.dt = float(self.t[1] - self.t[0])

        # ---- |ζ| precomputed on the event grid for speed ----
        self._build_zeta_table()

        # ---- muon / tau channels ----
        self.S_mu = self._compute_S_mu()
        self.S_tau = -self.S_mu

        # ---- convolution (neg. autocorrelation) ----
        self.conv = self._compute_convolution()

        # ---- electrons = zero-crossings of C(t) ----
        self.electron_idx = self._find_zeros(self.conv)
        self.n_electrons = int(self.electron_idx.size)

        # ---- parity cross-check (even-even vs even-odd) ----
        self._build_masks()

    # --------------------------------------------------------
    def _build_zeta_table(self):
        """Sample ζ on a fine grid covering [0, T + 2·dt_event]."""
        hi = self.T + 2.0 * self.dt_event
        self._zgrid = np.linspace(0.0, hi, self.N_samples * 2)
        self._zvals = np.abs(
            np.array([self.gate.zeta(zg) for zg in self._zgrid]))

    def _zeta_at(self, tau):
        return np.interp(tau, self._zgrid, self._zvals)

    # --------------------------------------------------------
    def _compute_S_mu(self):
        """
        S_νμ(t) = σ_μ(J_C) · Σ_{n=0}^{N−1} (−1)^{n+1} |ζ(t + n·dt)|

        The alternating sum over the sliding event window is what
        makes S_νμ oscillate with t.
        """
        sig_mu = spin_grade(self.J_C, 0.0)
        S = np.zeros_like(self.t)
        n_offsets = np.arange(self.N_events, dtype=float)
        alts = (-1.0) ** (n_offsets + 1)
        for i, t_now in enumerate(self.t):
            taus = t_now + n_offsets * self.dt_event
            mags = self._zeta_at(taus)
            S[i] = sig_mu * float(np.dot(alts, mags))
        return S

    # --------------------------------------------------------
    def _compute_convolution(self):
        conv = np.convolve(self.S_mu, self.S_tau, mode="same") * self.dt
        return conv

    # --------------------------------------------------------
    def _find_zeros(self, arr):
        """Indices where arr changes sign."""
        s = np.sign(arr)
        s[s == 0] = 1
        return np.where(np.diff(s) != 0)[0]

    # --------------------------------------------------------
    def _build_masks(self):
        """Even-even and even-odd masks from Figure 4."""
        N = self.N_events
        t_ev = np.arange(1, N + 1) * self.dt_event
        r_ev = np.arange(1, N + 1, dtype=float)
        self.G = coupling_matrix(t_ev, r_ev)

        i_idx = np.arange(N)[:, None]
        j_idx = np.arange(N)[None, :]
        self.M_ee = ((i_idx % 2 == 0) & (j_idx % 2 == 0))
        self.M_eo = ((i_idx % 2) != (j_idx % 2))

        self.G_ee = self.G * self.M_ee
        self.G_eo = self.G * self.M_eo

        self.norm_ee = float(np.linalg.norm(self.G_ee))
        self.norm_eo = float(np.linalg.norm(self.G_eo))
        self.parity_ratio = self.norm_ee / max(self.norm_eo, 1e-12)

    # --------------------------------------------------------
    def electron_density(self):
        """ρ_elec(t) = |ζ(t)|² · Π_e(t, r_eff)."""
        rho = np.zeros_like(self.t)
        for i, t_now in enumerate(self.t):
            r_eff = (t_now / self.T) * PI
            pe = pi_e(t_now, r_eff)
            z = self.gate.zeta(t_now)
            rho[i] = (abs(z) ** 2) * pe
        return rho

    # --------------------------------------------------------
    def confirmed_ticks(self):
        """
        Ticks that satisfy all three conditions:
            1. zero of convolution   (already filtered by electron_idx)
            2. ‖G_ee‖ > ‖G_eo‖       (parity protected, Figure 4)
            3. Π_e(t, r_eff) > 0.5   (coherent cloud)
        """
        conf = []
        parity_ok = self.parity_ratio > 1.0
        for idx in self.electron_idx:
            t_tick = float(self.t[idx])
            r_eff = (t_tick / self.T) * PI
            pe = float(pi_e(t_tick, r_eff))
            if parity_ok and pe > 0.5:
                conf.append(dict(
                    t=t_tick,
                    idx=int(idx),
                    pe=pe,
                    conv=float(self.conv[idx]),
                    S_mu=float(self.S_mu[idx]),
                ))
        return conf

    # --------------------------------------------------------
    def tick_rate(self):
        return self.n_electrons / self.T

    # --------------------------------------------------------
    def report(self):
        L = ["=" * 76,
             "Atomic clock  ·  muon/tau convolution zeros",
             "=" * 76]
        L.append(f"  α                = 1/(π − e) = {ALPHA:.6f}")
        L.append(f"  A_STEP                        = {A_STEP:.6f}")
        L.append(f"  K                             = {self.K}")
        L.append(f"  J_C                           = {self.J_C}")
        L.append(f"  σ_μ(J_C)                      = "
                 f"{spin_grade(self.J_C, 0.0):+.0f}")
        L.append(f"  σ_τ(J_C)                      = "
                 f"{spin_grade(self.J_C, 1.0):+.0f}")
        L.append(f"  period T                      = {self.T:.6f}")
        L.append(f"  N events                      = {self.N_events}")
        L.append(f"  N samples                     = {self.N_samples}")
        L.append("")
        L.append("--- muon / tau channels ---")
        L.append(f"  S_νμ range                    = "
                 f"[{self.S_mu.min():+.6f}, {self.S_mu.max():+.6f}]")
        L.append(f"  S_ντ = −S_νμ  (exact, σ_τ = −σ_μ)")
        L.append("")
        L.append("--- convolution C(t) = (S_νμ * S_ντ)(t) ---")
        L.append(f"  C range                       = "
                 f"[{self.conv.min():+.6e}, {self.conv.max():+.6e}]")
        L.append(f"  electrons (zeros of C)        = {self.n_electrons}")
        L.append(f"  tick rate                     = "
                 f"{self.tick_rate():.4f}  per unit-t")
        L.append("")
        L.append("--- parity cross-check  (Figure 4 of neutral.pdf) ---")
        L.append("  M_ee[i,j] = 1 iff i ≡ j ≡ 0 (mod 2)")
        L.append("  M_eo[i,j] = 1 iff (i mod 2) ≠ (j mod 2)")
        L.append(f"  ‖G_ee‖                        = {self.norm_ee:.6f}")
        L.append(f"  ‖G_eo‖                        = {self.norm_eo:.6f}")
        L.append(f"  parity ratio                  = "
                 f"{self.parity_ratio:.4f}")
        L.append(f"  parity protected              = "
                 f"{self.parity_ratio > 1.0}")
        L.append("")
        conf = self.confirmed_ticks()
        L.append(f"--- confirmed ticks (all conditions) ---")
        L.append(f"  count                         = {len(conf)}")
        for c in conf[:8]:
            L.append(f"    t = {c['t']:>8.4f}   "
                     f"Π_e = {c['pe']:.4f}   "
                     f"S_νμ = {c['S_mu']:+.6f}")
        return "\n".join(L)

    # --------------------------------------------------------
    def visualize(self):
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)

        # (a) S_νμ and S_ντ
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.t, self.S_mu, color="#e74c3c", lw=1.2,
                label=r"$S_{\nu\mu}(t)$")
        ax.plot(self.t, self.S_tau, color="#3498db", lw=1.2,
                ls="--", label=r"$S_{\nu\tau}(t) = -S_{\nu\mu}(t)$")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("signal")
        ax.set_title("Muon / tau alternating sums")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (b) Convolution + electrons
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(self.t, self.conv, color="#8e44ad", lw=1.2,
                label=r"$C(t) = (S_{\nu\mu} * S_{\nu\tau})(t)$")
        ax.axhline(0, color="k", lw=0.4)
        if self.n_electrons:
            tick_t = self.t[self.electron_idx]
            ax.scatter(tick_t, np.zeros_like(tick_t),
                       color="#16a085", s=30, zorder=5,
                       edgecolors="k", linewidths=0.4,
                       label=f"{self.n_electrons} electrons")
        ax.set_xlabel("t"); ax.set_ylabel("C(t)")
        ax.set_title("Muon–tau convolution  ·  zeros = electrons")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (c) electron density  |ζ|² · Π_e
        ax = fig.add_subplot(gs[0, 2])
        rho = self.electron_density()
        ax.fill_between(self.t, 0, rho, color="#8e44ad", alpha=0.25,
                        label=r"$\rho_{elec}(t)$ density")
        ax.plot(self.t, rho, color="#8e44ad", lw=1.0)
        if self.n_electrons:
            tick_t = self.t[self.electron_idx]
            tick_rho = np.interp(tick_t, self.t, rho)
            ax.scatter(tick_t, tick_rho, color="#16a085",
                       s=30, zorder=5, edgecolors="k",
                       linewidths=0.4, label="electrons")
        ax.set_xlabel("t"); ax.set_ylabel(r"$\rho_{elec}$")
        ax.set_title(r"Electron density  ·  $|\zeta|^2\cdot\Pi_e$")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (d) even-even mask
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(self.M_ee.astype(int), origin="lower",
                  cmap="Greens", aspect="equal")
        ax.set_xlabel("j"); ax.set_ylabel("i")
        ax.set_title(f"Even-even mask  $M_{{ee}}$  (Figure 4)\n"
                     f"$\\|G_{{ee}}\\| = {self.norm_ee:.4f}$")

        # (e) even-odd cross mask
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(self.M_eo.astype(int), origin="lower",
                  cmap="Reds", aspect="equal")
        ax.set_xlabel("j"); ax.set_ylabel("i")
        ax.set_title(f"Even-odd cross mask  $M_{{eo}}$\n"
                     f"$\\|G_{{eo}}\\| = {self.norm_eo:.4f}$")

        # (f) parity ratio bar
        ax = fig.add_subplot(gs[1, 2])
        ax.bar(["‖G_ee‖", "‖G_eo‖"],
               [self.norm_ee, self.norm_eo],
               color=["#16a085", "#e74c3c"], edgecolor="k")
        ax.text(0, self.norm_ee,
                f"{self.norm_ee:.3f}", ha="center", va="bottom")
        ax.text(1, self.norm_eo,
                f"{self.norm_eo:.3f}", ha="center", va="bottom")
        protected = "YES" if self.parity_ratio > 1.0 else "NO"
        ax.set_ylabel("norm")
        ax.set_title(f"Parity protection\n"
                     f"ratio = {self.parity_ratio:.3f}  "
                     f"(protected = {protected})")
        ax.grid(alpha=0.3, axis="y")

        plt.suptitle(
            f"Atomic clock  ·  muon/tau convolution zeros  ·  "
            f"{self.n_electrons} electrons  ·  "
            f"tick rate = {self.tick_rate():.3f}  ·  "
            f"α = {ALPHA:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Atomic clock  ·  muon/tau convolution zeros")
    print("=" * 76)
    print(f"  α      = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print(f"  6/π²                  = {DENSITY:.6f}")
    print()

    clock = AtomicClockMuonTau(
        K=20, J_C=1.0, N_events=41, N_samples=2048)
    print(clock.report())
    print()

    # ---- explicit electron list ----
    if clock.n_electrons:
        print("--- electron list  (first 12 zero-crossings) ---")
        print(f"  {'k':>3}  {'t_k':>10}  {'S_νμ(t_k)':>12}  "
              f"{'Π_e':>8}  {'confirmed':>10}")
        conf_set = {c["idx"] for c in clock.confirmed_ticks()}
        for k, idx in enumerate(clock.electron_idx[:12]):
            t_tick = clock.t[idx]
            r_eff = t_tick / clock.T * PI
            pe = pi_e(t_tick, r_eff)
            flag = "✓" if idx in conf_set else "·"
            print(f"  {k:>3d}  {t_tick:>10.4f}  "
                  f"{clock.S_mu[idx]:>+12.6f}  "
                  f"{pe:>8.4f}  {flag:>10s}")
        print()

    # ---- cross-check summary ----
    print("--- cross-check  (Figure 4 of neutral.pdf) ---")
    print(f"  even-even ‖G_ee‖        = {clock.norm_ee:.6f}")
    print(f"  even-odd ‖G_eo‖         = {clock.norm_eo:.6f}")
    print(f"  parity ratio            = {clock.parity_ratio:.4f}")
    print(f"  parity protected        = "
          f"{clock.parity_ratio > 1.0}")
    print()

    print("--- method ---")
    print("  S_νμ(t)      = σ_μ(J_C) · Σ_n (−1)^{n+1} |ζ(t + n·dt)|")
    print("  S_ντ(t)      = −S_νμ(t)          (σ_τ = −σ_μ)")
    print("  C(t)         = (S_νμ * S_ντ)(t)   zero-crossings = electrons")
    print("  M_ee, M_eo   even-even / even-odd masks  (Figure 4)")
    print("  Π_e(t,r)     bounded cloud intensity      (Figure 3)")
    print("  tick         C(t)=0  ∧  ‖G_ee‖>‖G_eo‖  ∧  Π_e>0.5")
    print()

    clock.visualize()


if __name__ == "__main__":
    demo()