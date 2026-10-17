#!/usr/bin/env python3
"""
quantum_hall_joystick.py
========================

Quantum Hall-like single-sensor game controller.

One sensor produces (x, y) ∈ [-1,1]² per frame.  Split into:

    LEFT  (algebraic / discrete)  ←  movement signal
        x_L = exp(α·x),   y_L = exp(α·y)
        M(x_L, y_L) = [x_L⁻¹; y_L⁻¹] ⊗ [x_Lⁱ, y_Lⁱ]
        tr M = x_L^(i−1) + y_L^(i−1)
        coupling reconstruction  g_L = w · κ
        signal = (x_L⁻¹ + y_L⁻¹) · g_L
        QHE quantization:  q_L = n_L / N_PLATEAUS

    RIGHT (transcendental / time)
        xⁱ, yⁱ with time phase advanced by dzeta flow
            ds/dt = −ζ_μ'(s),   ζ_μ(s) = Σ μ(k) k^{−s}
        signal = |xⁱ − yⁱ| · (1 + |ds/dt|)
        QHE quantization:  q_R = n_R / N_PLATEAUS

    HALL  cross-coupling
        H = Im(tr M) · q_R          (perpendicular response)

    EVEN-EVEN MASK
        left coupling updates only on even frames
        right fills in the odd frames
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)      # ≈ 2.362338
DENSITY   = 6.0 / (PI * PI)     # ≈ 0.607927
N_PLATEAUS = 8                  # QHE quantization levels


# ============================================================
#  Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    if K >= 1:
        mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(K ** 0.5) + 1):
        if is_prime[p]:
            is_prime[p * p::p] = False
    for p in range(2, K + 1):
        if is_prime[p]:
            mu[p::p] = -mu[p::p]
    for p in range(2, int(K ** 0.5) + 1):
        if is_prime[p]:
            mu[p * p::p * p] = 0
    return mu


# ============================================================
#  Single-sensor controller
# ============================================================
class SingleSensorController:
    def __init__(self, K: int = 64, n_plateaus: int = N_PLATEAUS,
                 use_mask: bool = True):
        self.K = K
        self.mu = mobius_sieve(K)
        self.n_pl = n_plateaus
        self.use_mask = use_mask

        self.t = 0.0
        self.frame = 0
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.s = complex(1.5, 1.0)      # dzeta state
        self.left_held = None

        self.H = {k: [] for k in
                  ['t', 'x', 'y',
                   'left_raw', 'left_q', 'left_n', 'left_held',
                   'right_raw', 'right_q', 'right_n',
                   'hall', 's_real', 's_imag', 'ds_mag']}

    # -------- dzeta flow --------
    def _dzeta_prime(self, s: complex) -> complex:
        """
        −ζ_μ'(s) = Σ μ(k) ln(k) k^{−s}.

        Re(s) clamped ≥ 1.1 to keep the series convergent.
        """
        s_safe = complex(max(s.real, 1.1), s.imag)
        k = np.arange(1, self.K + 1, dtype=float)
        logk = np.log(k)
        mu_ = self.mu[1:].astype(float)
        return -np.sum(mu_ * logk * np.exp(-s_safe * logk))

    # -------- M-matrix trace --------
    @staticmethod
    def _trace(x: float, y: float):
        """tr M = x^{i−1} + y^{i−1}."""
        xa, ya = max(abs(x), 1e-9), max(abs(y), 1e-9)
        lnx, lny = np.log(xa), np.log(ya)
        re = np.cos(lnx) / xa + np.cos(lny) / ya
        im = np.sin(lnx) / xa + np.sin(lny) / ya
        return re, im

    # -------- LEFT: algebraic / discrete --------
    def _left(self, x: float, y: float) -> dict:
        xa = math.exp(ALPHA * x)
        ya = math.exp(ALPHA * y)
        inv_x = 1.0 / xa
        inv_y = 1.0 / ya

        re, im = self._trace(xa, ya)
        chi = math.atan2(im, re)
        w = abs(chi) / PI                               # chiral bond weight

        L = math.sqrt((x - self.x_prev) ** 2
                      + (y - self.y_prev) ** 2) + 1e-6
        kappa = float(np.clip(L / 0.1, 0.0, 1.0))       # travel length

        g = w * kappa                                    # coupling reconstruction
        raw = (inv_x + inv_y) * g
        raw_n = math.tanh(raw / 5.0)                     # soft → [0, 1)

        n = int(round(raw_n * self.n_pl))
        return dict(g=g, raw=raw, raw_n=raw_n,
                    q=n / self.n_pl, n=n,
                    sym=re, asym=im)

    # -------- RIGHT: transcendental / time --------
    def _right(self, x: float, y: float, dt: float) -> dict:
        ds_dt = self._dzeta_prime(self.s)
        self.s = self.s - ds_dt * dt

        xa = math.exp(ALPHA * x)
        ya = math.exp(ALPHA * y)
        lnx = math.log(max(xa, 1e-9))
        lny = math.log(max(ya, 1e-9))
        t = self.t
        xi = complex(math.cos(t * lnx), math.sin(t * lnx))   # x^i
        yi = complex(math.cos(t * lny), math.sin(t * lny))   # y^i

        dz = abs(ds_dt)
        raw = abs(xi - yi) * (1.0 + dz)
        raw_n = math.tanh(raw / 5.0)

        n = int(round(raw_n * self.n_pl))
        return dict(raw=raw, raw_n=raw_n,
                    q=n / self.n_pl, n=n,
                    xi=xi, yi=yi, ds=ds_dt, s=self.s)

    # -------- full read (even-even mask + dzeta) --------
    def read(self, x: float, y: float, dt: float):
        self.t += dt

        # even-even mask: left updates only on even frames
        if (self.use_mask and self.frame % 2 == 1
                and self.left_held is not None):
            left = dict(self.left_held)
            left['held'] = True
        else:
            left = self._left(x, y)
            left['held'] = False
            if self.use_mask:
                self.left_held = left

        # right always advances (transcendental channel)
        right = self._right(x, y, dt)

        # Hall transverse cross-coupling
        hall = math.tanh(left['asym']) * right['q']

        # log
        H = self.H
        H['t'].append(self.t)
        H['x'].append(x); H['y'].append(y)
        H['left_raw'].append(left['raw'])
        H['left_q'].append(left['q'])
        H['left_n'].append(left['n'])
        H['left_held'].append(1 if left['held'] else 0)
        H['right_raw'].append(right['raw'])
        H['right_q'].append(right['q'])
        H['right_n'].append(right['n'])
        H['hall'].append(hall)
        H['s_real'].append(self.s.real)
        H['s_imag'].append(self.s.imag)
        H['ds_mag'].append(abs(right['ds']))

        self.x_prev = x
        self.y_prev = y
        self.frame += 1
        return left, right, hall


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Quantum Hall-like single-sensor controller")
    print("=" * 76)
    print(f"  α      = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  6/π²                 = {DENSITY:.6f}")
    print(f"  N_PLATEAUS           = {N_PLATEAUS}")
    print(f"  even-even mask       : left updates on even frames only")
    print()

    ctrl = SingleSensorController(K=64, n_plateaus=N_PLATEAUS, use_mask=True)

    # circular stick sweep
    T, dt = 240, 0.05
    theta = np.linspace(0, 6 * PI, T)
    r = 0.75
    xs = r * np.cos(theta)
    ys = r * np.sin(theta)

    for i in range(T):
        ctrl.read(xs[i], ys[i], dt)

    H = ctrl.H
    t = np.array(H['t'])
    x = np.array(H['x']); y = np.array(H['y'])
    left_raw = np.array(H['left_raw'])
    left_q = np.array(H['left_q'])
    left_n = np.array(H['left_n'])
    left_held = np.array(H['left_held'])
    right_raw = np.array(H['right_raw'])
    right_q = np.array(H['right_q'])
    right_n = np.array(H['right_n'])
    hall = np.array(H['hall'])
    s_re = np.array(H['s_real'])
    s_im = np.array(H['s_imag'])

    # ---------- quantization lookup table ----------
    print("--- quantization lookup table  (y = 0, single frame) ---")
    print(f"  {'x':>6}  {'g_L':>8}  {'raw_L':>10}  "
          f"{'n_L':>4}  {'q_L':>6}")
    for xv in [-0.8, -0.4, 0.0, 0.4, 0.8]:
        c = SingleSensorController(K=64, n_plateaus=N_PLATEAUS, use_mask=False)
        L, _, _ = c.read(xv, 0.0, 0.05)
        print(f"  {xv:>+6.2f}  {L['g']:>8.4f}  {L['raw']:>10.4f}  "
              f"{L['n']:>4d}  {L['q']:>6.3f}")
    print()

    # ---------- statistics ----------
    print("--- runtime statistics ---")
    print(f"  left  raw range     : [{left_raw.min():+.4f}, "
          f"{left_raw.max():+.4f}]")
    print(f"  left  plateaus used : {sorted(set(left_n.tolist()))}")
    print(f"  left  held frames   : {int(left_held.sum())} / {T}")
    print(f"  right raw range     : [{right_raw.min():+.4f}, "
          f"{right_raw.max():+.4f}]")
    print(f"  right plateaus used : {sorted(set(right_n.tolist()))}")
    print(f"  Hall range          : [{hall.min():+.4f}, {hall.max():+.4f}]")
    print(f"  dzeta s(t) final    : {s_re[-1]:+.4f} + {s_im[-1]:+.4f}i")
    print()

    # ---------- plots ----------
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # (a) stick trajectory
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, y, color="#2c3e50", lw=0.9, alpha=0.55)
    sc = ax.scatter(x, y, c=t, cmap="viridis", s=12, zorder=3)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Single sensor trajectory (circular sweep)")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="time")
    ax.grid(alpha=0.3)

    # (b) LEFT: algebraic → QHE plateaus
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, left_raw, "--", color="#3498db", lw=1.0,
            alpha=0.55, label="continuous  $(x_L^{-1}+y_L^{-1})\\,g_L$")
    ax.step(t, left_q, "-", color="#2980b9", lw=1.6,
            where="mid", label="QHE plateaus  $n_L/N$")
    # mark held frames
    held_t = t[left_held == 1]
    for ht in held_t:
        ax.axvline(ht, color="#95a5a6", lw=0.25, alpha=0.3)
    ax.set_xlabel("t"); ax.set_ylabel("left signal")
    ax.set_title("LEFT  ·  algebraic discrete parts  →  QHE plateaus\n"
                 "(grey lines = even-even mask holding)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) RIGHT: transcendental + dzeta → QHE plateaus
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, right_raw, "--", color="#e74c3c", lw=1.0,
            alpha=0.55, label="continuous  $|x^i-y^i|\\,(1+|\\dot s|)$")
    ax.step(t, right_q, "-", color="#c0392b", lw=1.6,
            where="mid", label="QHE plateaus  $n_R/N$")
    ax.set_xlabel("t"); ax.set_ylabel("right signal")
    ax.set_title("RIGHT  ·  transcendental + dzeta flow  →  QHE plateaus")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) Hall transverse + dzeta trajectory inset
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, hall, color="#8e44ad", lw=1.2,
            label=r"Hall  $H=\tanh(\mathrm{Im}\,tr M)\cdot q_R$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t"); ax.set_ylabel("transverse signal")
    ax.set_title("HALL  ·  cross-coupling perpendicular to stick")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    inset = ax.inset_axes([0.60, 0.55, 0.38, 0.40])
    inset.plot(s_re, s_im, color="#8e44ad", lw=1.0)
    inset.scatter(s_re[0], s_im[0], c="green", s=40, zorder=3)
    inset.scatter(s_re[-1], s_im[-1], c="red", s=40, zorder=3)
    inset.set_xlabel("Re s", fontsize=8)
    inset.set_ylabel("Im s", fontsize=8)
    inset.set_title(r"$s(t)$ under $\dot s = -\zeta_\mu'(s)$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.3)

    plt.suptitle(
        f"Quantum Hall-like single-sensor controller  ·  "
        f"α = {ALPHA:.4f}  ·  N_PLATEAUS = {N_PLATEAUS}  ·  "
        f"left = algebraic  ·  right = transcendental  ·  "
        f"even-even mask on",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---------- summary ----------
    print("=" * 76)
    print("Summary")
    print("=" * 76)
    print("  LEFT  (algebraic / discrete)")
    print("      x_L = exp(α·x),  y_L = exp(α·y)")
    print("      tr M = x_L^(i−1) + y_L^(i−1)")
    print("      g_L  = w · κ                    (coupling reconstruction)")
    print("      signal = (x_L⁻¹ + y_L⁻¹) · g_L")
    print("      QHE:  q_L = n_L / N_PLATEAUS")
    print("      even-even mask:  updates on even frames only")
    print()
    print("  RIGHT (transcendental / time)")
    print("      xⁱ, yⁱ  with  ds/dt = −ζ_μ'(s)")
    print("      signal = |xⁱ − yⁱ| · (1 + |ds/dt|)")
    print("      QHE:  q_R = n_R / N_PLATEAUS")
    print("      always advances (fills odd frames)")
    print()
    print("  HALL  cross-coupling")
    print("      H = tanh(Im tr M) · q_R")
    print("      transverse to stick direction")
    print("      flips sign at each plateau crossing")
    print()
    print("  Signature of the quantum Hall-like effect:")
    print("      · continuous stick motion → discrete plateaus")
    print("      · both channels quantize independently")
    print("      · Hall response flips sign at transitions")
    print()


if __name__ == "__main__":
    demo()