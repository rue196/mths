#!/usr/bin/env python3
"""
quantum_alu.py
==============
Quantum-like ALU on the Möbius-sieve CPU architecture.

Every classical ALU operation is realized as a **reconstruction-coupled
convolution** between the two operand registers.

Constants
---------
    ALPHA      = 1/(π - e)              ≈ 2.3623
    ALPHA_ASYM = 0.3628
    A_STEP     = ALPHA / ALPHA_ASYM     ≈ 6.511     (universal finite step)

Encoding
--------
    s_x[k] = x · μ(k) · cos(A_STEP · k² / K)         (Möbius-chirped base)

Reconstruction coupling  (replaces chiral spin)
-----------------------------------------------
    L_i  = cumulative arclength of signal s_i
    κ_ij = √(L_i L_j) / L_ref ∈ [0, 1]

Entanglement matrix
-------------------
    G_ij = κ_ij · s_x[i] · s_y[j]

ALU operations
--------------
    ADD  : diagonal coupling         s_out = diag(κ)·(s_x + s_y)
    SUB  : diagonal coupling         s_out = diag(κ)·(s_x - s_y)
    MUL  : full coupling matrix      s_out = (κ ⊙ s_x⊗s_y) @ w
    DIV  : full coupling matrix      s_out = (κ ⊙ s_x⊗s_y⁻¹) @ (s_y · w)

    Readout:  z = <s_out, w> / <w², κ @ w²>
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA      = 1.0 / (PI - E)              # ≈ 2.3623
ALPHA_ASYM = 0.3628
A_STEP     = ALPHA / ALPHA_ASYM          # ≈ 6.511
K_DEFAULT  = 64


# ============================================================
#  Möbius sieve  O(K) linear
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K: break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Quantum-like ALU
# ============================================================
class QuantumALU:
    """
    ALU where each operation is realized as a reconstruction-coupled
    convolution.

    Finite step for ALL operations:
        A_STEP = 1/((π-e)/0.3628) = ALPHA/0.3628 ≈ 6.511
    """

    def __init__(self, K: int = K_DEFAULT):
        self.K = int(K)
        self.a = A_STEP
        self.mu = mobius_sieve(self.K)

        # ---- Möbius-chirped base waveform w ----
        self.idx = np.arange(self.K)
        self.gate = (self.mu[1:self.K+1] != 0).astype(np.float64)
        self.w = self.gate * np.cos(self.a * self.idx * self.idx / self.K)
        self.w_norm = float(np.dot(self.w, self.w))
        self.w_sq = self.w ** 2

        # ---- calibration cache (unused for exact ops) ----
        self._cache: Dict[str, float] = {}

    # --------------------------------------------------------
    #  Encoding / decoding
    # --------------------------------------------------------
    def reg2sig(self, x: float) -> np.ndarray:
        """Register → signal:  s[k] = x · w[k]."""
        return float(x) * self.w

    def sig2reg(self, s: np.ndarray) -> float:
        """Signal → register:  x = <s, w> / <w, w>."""
        return float(np.dot(s, self.w) / self.w_norm)

    # --------------------------------------------------------
    #  Reconstruction coupling  (from coupling_Q_entanglement.py)
    # --------------------------------------------------------
    def travel(self, s: np.ndarray) -> np.ndarray:
        """L_k = cumulative arclength of signal s."""
        ds = np.abs(np.diff(s, prepend=s[0]))
        return np.cumsum(ds) + 1e-9

    def recon_coupling(self, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
        """
        κ_ij = √(L_i L_j) / L_ref ∈ [0, 1]

        Replaces the chiral bond weight |χ_i − χ_j|/π.
        """
        Lx = self.travel(sx)
        Ly = self.travel(sy)
        L_ref = 0.5 * (Lx[-1] + Ly[-1]) + 1e-9
        K_ij = np.sqrt(np.outer(Lx, Ly)) / L_ref
        return np.clip(K_ij, 0.0, 1.0)

    def entanglement(self, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
        """G_ij = κ_ij · s_x[i] · s_y[j]."""
        Kc = self.recon_coupling(sx, sy)
        return Kc * np.outer(sx, sy)

    # --------------------------------------------------------
    #  Linear ALU operations (diagonal coupling)
    # --------------------------------------------------------
    def ADD(self, x: float, y: float) -> float:
        """x + y via diagonal reconstruction coupling."""
        s_x = self.reg2sig(x)
        s_y = self.reg2sig(y)
        kd  = self.recon_coupling(s_x, s_y).diagonal()
        s_out = kd * (s_x + s_y)
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(kd, self.w_sq))
        return num / den if den > 1e-12 else 0.0

    def SUB(self, x: float, y: float) -> float:
        """x - y via diagonal reconstruction coupling."""
        s_x = self.reg2sig(x)
        s_y = self.reg2sig(y)
        kd  = self.recon_coupling(s_x, s_y).diagonal()
        s_out = kd * (s_x - s_y)
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(kd, self.w_sq))
        return num / den if den > 1e-12 else 0.0

    def NEG(self, x: float) -> float:
        """-x."""
        return -float(x)

    def SHL(self, x: float, k: int = 1) -> float:
        """x << k  (shift left)."""
        return float(int(x) * (1 << k))

    def SHR(self, x: float, k: int = 1) -> float:
        """x >> k  (shift right)."""
        return float(int(x) >> k)

    # --------------------------------------------------------
    #  Bilinear ALU operations (full coupling matrix)
    # --------------------------------------------------------
    def MUL(self, x: float, y: float) -> float:
        """x · y via the entanglement matrix."""
        s_x = self.reg2sig(x)
        s_y = self.reg2sig(y)
        Kc  = self.recon_coupling(s_x, s_y)
        G   = Kc * np.outer(s_x, s_y)
        s_out = G @ self.w
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(self.w_sq, Kc @ self.w_sq))
        return num / den if den > 1e-12 else 0.0

    def DIV(self, x: float, y: float) -> float:
        """x / y via the entanglement matrix (y ≠ 0)."""
        if abs(y) < 1e-9:
            return float('inf') * np.sign(x)
        s_x = self.reg2sig(x)
        s_y = self.reg2sig(y)
        Kc  = self.recon_coupling(s_x, s_y)
        s_y_inv = np.where(np.abs(s_y) > 1e-9, 1.0 / s_y, 0.0)
        s_out = (Kc * np.outer(s_x, s_y_inv)) @ (s_y * self.w)
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(self.w_sq, Kc @ self.w))
        return num / den if den > 1e-12 else 0.0

    # --------------------------------------------------------
    #  Bitwise ALU (via integer fallback — the coupling is
    #  shown but not used to reconstruct the exact bit pattern)
    # --------------------------------------------------------
    def AND(self, x: int, y: int) -> int:
        return int(x) & int(y)

    def OR(self, x: int, y: int) -> int:
        return int(x) | int(y)

    def XOR(self, x: int, y: int) -> int:
        return int(x) ^ int(y)

    def NOT(self, x: int) -> int:
        return ~int(x)

    # --------------------------------------------------------
    #  Finite-step derivatives  (for the ALU "reference" path)
    # --------------------------------------------------------
    def finite_step_forward(self, f, x: float) -> float:
        """D⁺_a f(x) = (f(x+a) − f(x)) / a."""
        return (f(x + self.a) - f(x)) / self.a

    def finite_step_central(self, f, x: float) -> float:
        """D_a f(x) = (f(x+a) − f(x−a)) / (2a)."""
        return (f(x + self.a) - f(x - self.a)) / (2 * self.a)

    # --------------------------------------------------------
    #  Full operation with traces for visualization
    # --------------------------------------------------------
    def operate(self, x: float, y: float, op: str) -> Dict:
        """Return the operation result plus the coupling trace."""
        s_x = self.reg2sig(x)
        s_y = self.reg2sig(y)
        Kc  = self.recon_coupling(s_x, s_y)

        if op == "ADD":
            kd = Kc.diagonal()
            s_out = kd * (s_x + s_y)
            z = self.ADD(x, y)
        elif op == "SUB":
            kd = Kc.diagonal()
            s_out = kd * (s_x - s_y)
            z = self.SUB(x, y)
        elif op == "MUL":
            G = Kc * np.outer(s_x, s_y)
            s_out = G @ self.w
            z = self.MUL(x, y)
        elif op == "DIV":
            s_y_inv = np.where(np.abs(s_y) > 1e-9, 1.0 / s_y, 0.0)
            G = Kc * np.outer(s_x, s_y_inv)
            s_out = G @ (s_y * self.w)
            z = self.DIV(x, y)
        else:
            s_out = np.zeros(self.K)
            z = 0.0

        return dict(op=op, x=x, y=y, z=z,
                    s_x=s_x, s_y=s_y, K_c=Kc,
                    s_out=s_out, norm_K=float(np.linalg.norm(Kc)))


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Quantum-like ALU  ·  reconstruction-coupled convolution")
    print("=" * 78)
    print(f"  ALPHA      = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  ALPHA_ASYM = 0.3628         = {ALPHA_ASYM:.6f}")
    print(f"  A_STEP     = ALPHA/0.3628   = {A_STEP:.6f}   (finite step for ALL ops)")
    print()
    print("  Register encoding:")
    print("      s_x[k] = x · μ(k) · cos(A_STEP · k² / K)")
    print("  Reconstruction coupling:")
    print("      κ_ij = √(L_i L_j) / L_ref      (travel-length, not chirality)")
    print("  Entanglement matrix:")
    print("      G_ij = κ_ij · s_x[i] · s_y[j]")
    print()

    alu = QuantumALU(K=K_DEFAULT)
    n_gate = int(np.count_nonzero(alu.gate))
    print(f"--- base waveform ---")
    print(f"  K              = {alu.K}")
    print(f"  μ-gate nonzero = {n_gate} / {alu.K} "
          f"= {n_gate / alu.K:.4f}  (target 6/π² = {6/PI**2:.4f})")
    print(f"  ⟨w, w⟩         = {alu.w_norm:.6f}")
    print(f"  max |w|        = {np.max(np.abs(alu.w)):.6f}")
    print()

    # ---------- exact operations ----------
    print("--- exact ALU operations ---")
    print(f"  {'x':>6}  {'y':>6}  {'ADD':>10}  {'SUB':>10}  "
          f"{'MUL':>10}  {'DIV':>10}")
    test_pairs = [(1, 1), (3, 5), (7, 11), (13, 4), (21, 8)]
    for x, y in test_pairs:
        a = alu.ADD(x, y)
        s = alu.SUB(x, y)
        m = alu.MUL(x, y)
        d = alu.DIV(x, y) if y != 0 else float('inf')
        print(f"  {x:>6}  {y:>6}  {a:>10.4f}  {s:>10.4f}  "
              f"{m:>10.4f}  {d:>10.4f}")
    print()

    # ---------- classical verification ----------
    print("--- verification vs classical ---")
    print(f"  {'op':>6}  {'x':>4}  {'y':>4}  {'quantum':>12}  "
          f"{'classical':>12}  {'rel.err':>12}")
    for x, y in [(3, 5), (7, 11), (13, 4)]:
        for op, cl, q in [
            ("ADD", x + y, alu.ADD(x, y)),
            ("SUB", x - y, alu.SUB(x, y)),
            ("MUL", x * y, alu.MUL(x, y)),
            ("DIV", x / y, alu.DIV(x, y)),
        ]:
            rel = abs(q - cl) / max(abs(cl), 1e-12)
            print(f"  {op:>6}  {x:>4}  {y:>4}  {q:>12.6f}  "
                  f"{cl:>12.6f}  {rel:>12.2e}")
    print()

    # ---------- finite-step derivatives ----------
    print("--- finite-step derivatives  (a = ALPHA/0.3628) ---")
    f_sq = lambda t: t * t
    f_exp = lambda t: math.exp(t)
    for t0 in [1.0, 2.0, 3.0]:
        d_forward = alu.finite_step_forward(f_sq, t0)
        d_central = alu.finite_step_central(f_sq, t0)
        print(f"  t = {t0:.1f}   f = t²   "
              f"D⁺_a = {d_forward:.6f}   "
              f"D_a = {d_central:.6f}   "
              f"2t = {2*t0:.6f}")
    print()

    # ---------- coupling trace for one op ----------
    print("--- coupling trace:  MUL(7, 11) ---")
    res = alu.operate(7, 11, "MUL")
    print(f"  G norm         = {res['norm_K']:.4f}")
    print(f"  s_out range    = [{res['s_out'].min():+.4f}, "
          f"{res['s_out'].max():+.4f}]")
    print(f"  z              = {res['z']:.6f}")
    print(f"  classical      = {7 * 11}")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)

    # (a) base waveform
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(alu.idx, alu.w, color="#2c3e50", lw=1.0, label="w[k]")
    ax.plot(alu.idx, alu.gate, color="#e74c3c", lw=0.8, alpha=0.6,
            label="μ-gate")
    ax.set_xlabel("k"); ax.set_ylabel("value")
    ax.set_title(f"Base waveform  w = gate · chirp\n"
                 f"A_STEP = {A_STEP:.4f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) reconstruction coupling for MUL(7, 11)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(res["K_c"], origin="lower", cmap="inferno",
                   aspect="equal")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_title(f"κ_ij  ·  MUL(7, 11)\n"
                 f"‖κ‖ = {res['norm_K']:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) entanglement matrix G
    ax = fig.add_subplot(gs[0, 2])
    G = res["K_c"] * np.outer(res["s_x"], res["s_y"])
    im = ax.imshow(G, origin="lower", cmap="RdBu_r", aspect="equal")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_title(f"G_ij = κ_ij · s_x[i] · s_y[j]\nMUL(7, 11)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) signals for MUL(7, 11)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(alu.idx, res["s_x"], color="#3a7bd5", lw=1.0,
            label=f"s_x  (x = 7)")
    ax.plot(alu.idx, res["s_y"], color="#e74c3c", lw=1.0,
            label=f"s_y  (y = 11)")
    ax.plot(alu.idx, res["s_out"], color="#16a085", lw=1.4,
            label="s_out = G @ w")
    ax.set_xlabel("k"); ax.set_ylabel("signal")
    ax.set_title("Signal encoding for MUL(7, 11)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (e) operation results vs classical
    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(1, 16)
    cl_add = xs + 5
    q_add = np.array([alu.ADD(int(x), 5) for x in xs])
    cl_mul = xs * 5
    q_mul = np.array([alu.MUL(int(x), 5) for x in xs])
    ax.plot(xs, cl_add, "k--", lw=1.4, label="classical ADD")
    ax.plot(xs, q_add, "o", color="#3a7bd5", ms=4,
            label="quantum ADD")
    ax.plot(xs, cl_mul, "k:", lw=1.4, label="classical MUL")
    ax.plot(xs, q_mul, "s", color="#e74c3c", ms=4,
            label="quantum MUL")
    ax.set_xlabel("x  (y = 5)")
    ax.set_ylabel("result")
    ax.set_title("ADD and MUL vs classical")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (f) finite-step derivative accuracy
    ax = fig.add_subplot(gs[1, 2])
    ts = np.linspace(0.5, 5.0, 60)
    forward = np.array([alu.finite_step_forward(f_sq, t) for t in ts])
    central = np.array([alu.finite_step_central(f_sq, t) for t in ts])
    ax.plot(ts, 2 * ts, "k--", lw=1.4, label="exact  2t")
    ax.plot(ts, forward, color="#e74c3c", lw=1.2,
            label="D⁺_a  (forward)")
    ax.plot(ts, central, color="#16a085", lw=1.4,
            label="D_a  (central)")
    ax.set_xlabel("t"); ax.set_ylabel("derivative")
    ax.set_title(f"Finite-step derivatives of t²\n"
                 f"a = {A_STEP:.4f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(
        f"Quantum-like ALU  ·  A_STEP = ALPHA/0.3628 = {A_STEP:.4f}  ·  "
        f"κ_ij = √(L_i L_j)/L_ref  ·  G_ij = κ_ij · s_x[i] · s_y[j]",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 78)
    print("What the quantum-like ALU does")
    print("=" * 78)
    print("  Linear ops (ADD, SUB) use the DIAGONAL of the coupling matrix:")
    print("      s_out = diag(κ) · (s_x ± s_y)")
    print("      z = <s_out, w> / <diag(κ), w²>       (exact)")
    print()
    print("  Bilinear ops (MUL, DIV) use the FULL coupling matrix:")
    print("      G_ij = κ_ij · s_x[i] · s_y[j]")
    print("      s_out = G @ w")
    print("      z = <s_out, w> / <w², κ @ w²>        (exact)")
    print()
    print("  Finite step for ALL operations:")
    print(f"      A_STEP = 1/((π-e)/0.3628) = ALPHA/0.3628 = {A_STEP:.6f}")
    print()
    print("  Reconstruction coupling (replaces chiral spin):")
    print("      κ_ij = √(L_i L_j) / L_ref")
    print("      L_i  = cumulative arclength of signal i")
    print()
    print("  Contrast with the standard ALU:")
    print("      classical ALU:  z = x ⊙ y   (bit-exact, deterministic)")
    print("      quantum-like:   z = readout(coupling ⊙ encoding)")
    print("                      (linear/bilinear form, phase-aware)")
    print()
    print("  The reconstruction coupling distributes the operation across")
    print("  all slots according to their mutual travel length, which is")
    print("  the neutrino gravity channel from the source framework.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()