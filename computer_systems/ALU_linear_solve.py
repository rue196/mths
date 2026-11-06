#!/usr/bin/env python3
"""
alu_linear_operator.py
======================

Linear operator  L(x) = A · x  built on the QuantumALU,
executed in the discrete-log signed-index substrate.

Composition (all four modules used as-is)
-----------------------------------------
    alu_finite_step.QuantumALU
        every scalar product    A[i,j] · x[j]  →  QuantumALU.MUL
        every accumulation      Σ_j (...)      →  QuantumALU.ADD
        (unchanged — wrapped by ComplexALU for complex operands)

    dual_linear_solver_linear_sieve.DualLinearSolver
        partition rule   square-free (μ≠0) → A_re   (algebraic,   α = 0.3628)
                         composite   (μ=0) → A_im   (transcendental, α = 1/(π−e))
        solve(A, b) used unchanged for L(x) = b

    n_k_div_finitestep_convergence.build_signed_grid
        signed-index substrate  v(k) = −H_{|k|+1} + i·α·(μ*H)(k)

    ugc_discrete_log.UGCDiscreteLog
        discrete-log lookup  g^x ≡ h (mod p)  →  x
        BSGS O(√n) for large p

Design
------
    x   can be
          · a complex vector                (default)
          · a vector of signed indices k    (in_signed_index=True)

    L   applied column-by-column via QuantumALU.
    y   can be
          · lifted to the nearest signed-index value  v(k)
          · (for integer outputs) lifted to a discrete log x_hat

ComplexALU
----------
    QuantumALU operates on real scalars.  ComplexALU splits each
    complex operand into real and imaginary parts, applies the real
    ALU, and recombines:

        ADD(z1, z2) = (ALU.ADD(Re z1, Re z2), ALU.ADD(Im z1, Im z2))
        MUL(z1, z2) = (ALU.SUB(ALU.MUL(Re z1, Re z2), ALU.MUL(Im z1, Im z2)),
                       ALU.ADD(ALU.MUL(Re z1, Im z2), ALU.MUL(Im z1, Re z2)))
        DIV(z1, z2) = (ALU.DIV(Re(z1·conj z2), |z2|²),
                       ALU.DIV(Im(z1·conj z2), |z2|²))

Because QuantumALU.MUL and ADD are exact for real scalars, the
complex reconstruction is exact as well.

Complexity
----------
    matvec via ALU    O(n² · K_alu)     per matvec
    matvec via numpy  O(n²)             comparison path
    solve             O(n³)             via DualLinearSolver
    signed-index lift O(n · K_signed)
    dlog lift         O(n · √p)         BSGS

The ALU path is heavier than plain numpy because every scalar
operation runs the full coupling pipeline.  The trade is: the
operator is expressed in the same algebra that the discrete-log
substrate uses (signed-index values, partition rule, Möbius sieve).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- the four building blocks, unchanged ----
from alu_finite_step import (
    QuantumALU,
    K_DEFAULT as ALU_K_DEFAULT,
    A_STEP,
    ALPHA,
)

from dual_linear_solver_linear_sieve import (
    DualLinearSolver,
    DualSolution,
    ALPHA_ODD,
    ALPHA_EVEN,
    DENSITY,
)

from n_k_div_finitestep_convergence import (
    build_signed_grid,
    signed_value,
    finite_step_normalize,
    SignedUGC,
)

from ugc_discrete_log import (
    UGCDiscreteLog,
)


# ============================================================
#  ComplexALU  ·  complex wrapper for QuantumALU
# ============================================================
class ComplexALU:
    """
    Complex-aware wrapper over QuantumALU.

    Splits every complex operand into real and imaginary parts,
    applies the real QuantumALU, and recombines.
    QuantumALU itself is untouched.
    """
    def __init__(self, K: int = ALU_K_DEFAULT):
        self.alu = QuantumALU(K=K)
        self.K = int(K)

    # ---- complex arithmetic via the real ALU ----
    def add(self, z1: complex, z2: complex) -> complex:
        return complex(
            self.alu.ADD(z1.real, z2.real),
            self.alu.ADD(z1.imag, z2.imag),
        )

    def sub(self, z1: complex, z2: complex) -> complex:
        return complex(
            self.alu.SUB(z1.real, z2.real),
            self.alu.SUB(z1.imag, z2.imag),
        )

    def mul(self, z1: complex, z2: complex) -> complex:
        a, b = z1.real, z1.imag
        c, d = z2.real, z2.imag
        re = self.alu.SUB(self.alu.MUL(a, c), self.alu.MUL(b, d))
        im = self.alu.ADD(self.alu.MUL(a, d), self.alu.MUL(b, c))
        return complex(re, im)

    def div(self, z1: complex, z2: complex) -> complex:
        a, b = z1.real, z1.imag
        c, d = z2.real, z2.imag
        den = self.alu.ADD(self.alu.MUL(c, c), self.alu.MUL(d, d))
        if abs(den) < 1e-12:
            return complex(float("inf") * np.sign(a), 0.0)
        re_num = self.alu.ADD(self.alu.MUL(a, c), self.alu.MUL(b, d))
        im_num = self.alu.SUB(self.alu.MUL(b, c), self.alu.MUL(a, d))
        return complex(
            self.alu.DIV(re_num, den),
            self.alu.DIV(im_num, den),
        )


# ============================================================
#  ALULinearOperator
# ============================================================
class ALULinearOperator:
    """
    Linear operator  L(x) = A · x  where A is partitioned by
    DualLinearSolver and every scalar operation runs on QuantumALU.

    Storage
    -------
        A_re[i,j] ·  A_im[i,j]   partition (square-free / composite)
        A_c[i,j] = A_re[i,j] + i·A_im[i,j]   complex embedding

    Parameters
    ----------
    A           : (n, n) real matrix
    K_alu       : QuantumALU size (default 64)
    split_mode  : 'per_row'  or  'per_coefficient'   (forwarded to solver)
    ugc_p       : prime modulus for UGCDiscreteLog
    ugc_g       : generator for UGCDiscreteLog
    K_signed    : half-size of the signed-index grid
    """
    def __init__(self,
                 A: np.ndarray,
                 K_alu:    int = ALU_K_DEFAULT,
                 split_mode: str = 'per_row',
                 ugc_p:    Optional[int] = None,
                 ugc_g:    int = 2,
                 K_signed: int = 128):
        self.A = np.asarray(A, dtype=np.float64)
        self.n = self.A.shape[0]
        if self.A.shape != (self.n, self.n):
            raise ValueError(f"A must be square, got {self.A.shape}")

        # ---- 1. partition rule (unchanged DualLinearSolver) ----
        self.solver = DualLinearSolver(split_mode=split_mode)
        (self.A_re, _, self.A_im, _,
         self.classes, self.sf_counts, self.comp_counts) = \
            self.solver.split_axes(self.A, np.zeros(self.n))

        # complex embedding A_c = A_re + i·A_im
        self.A_c = self.A_re + 1j * self.A_im

        # ---- 2. complex ALU (wraps QuantumALU, unchanged) ----
        self.calu = ComplexALU(K=K_alu)

        # ---- 3. signed-index grid (n_k_div_finitestep_convergence) ----
        self.K_signed = int(K_signed)
        (self.ks, self.vals,
         self.H, self.F, self.mu, self.t_mu) = \
            build_signed_grid(self.K_signed)
        self._v_index = {int(k): i for i, k in enumerate(self.ks)}

        # ---- 4. discrete-log substrate (ugc_discrete_log) ----
        self.ugc = None
        if ugc_p is not None:
            self.ugc = UGCDiscreteLog(p=int(ugc_p), g=int(ugc_g))

    # --------------------------------------------------------
    def v(self, k: int) -> complex:
        """Signed-index value v(k) = −H_{|k|+1} + i·α·(μ*H)(k)."""
        return self.vals[self._v_index[int(k)]]

    # --------------------------------------------------------
    def _prepare(self, x, in_signed_index: bool) -> np.ndarray:
        """Coerce x to a complex vector (optionally via v(k))."""
        x = np.asarray(x)
        if in_signed_index:
            return np.array(
                [self.v(int(k)) for k in x], dtype=np.complex128)
        return x.astype(np.complex128)

    # --------------------------------------------------------
    def apply(self, x, in_signed_index: bool = False,
              use_alu: bool = True) -> np.ndarray:
        """
        Compute y = A_c · x.

        use_alu=True   every product via QuantumALU.MUL,
                       every accumulation via QuantumALU.ADD
        use_alu=False  plain numpy path (for comparison)
        """
        x = self._prepare(x, in_signed_index)
        if x.shape != (self.n,):
            raise ValueError(
                f"x must have shape ({self.n},), got {x.shape}")

        if not use_alu:
            return self.A_c @ x

        y = np.zeros(self.n, dtype=np.complex128)
        for i in range(self.n):
            acc = complex(0.0, 0.0)
            for j in range(self.n):
                a_ij = complex(self.A_re[i, j], self.A_im[i, j])
                if abs(a_ij) < 1e-12:
                    continue
                prod = self.calu.mul(a_ij, x[j])
                acc = self.calu.add(acc, prod)
            y[i] = acc
        return y

    # --------------------------------------------------------
    def apply_numpy(self, x, in_signed_index: bool = False
                    ) -> np.ndarray:
        """Plain-numpy reference path."""
        return self.apply(x, in_signed_index, use_alu=False)

    # --------------------------------------------------------
    def matmul(self, X, in_signed_index: bool = False,
               use_alu: bool = True) -> np.ndarray:
        """
        Apply L to every column of X.
        X may be (n,) or (n, m).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            return self.apply(X, in_signed_index, use_alu)
        m = X.shape[1]
        Y = np.zeros((self.n, m), dtype=np.complex128)
        for k in range(m):
            Y[:, k] = self.apply(X[:, k], in_signed_index, use_alu)
        return Y

    # --------------------------------------------------------
    def solve(self, b, method: str = 'gauss',
              structured: bool = False) -> DualSolution:
        """Solve A·x = b via the unchanged DualLinearSolver."""
        return self.solver.solve(self.A, b,
                                 method=method,
                                 structured=structured)

    # --------------------------------------------------------
    def lift_to_signed_index(self, y: np.ndarray) -> List[Dict]:
        """For each output z, find the nearest v(k) in the signed grid."""
        results = []
        for z in y:
            best_k, best_d = 0, float("inf")
            for k in self.ks:
                d = abs(self.v(int(k)) - z)
                if d < best_d:
                    best_d = d
                    best_k = int(k)
            v_best = self.v(best_k)
            results.append(dict(
                value=z, k=best_k, v_k=v_best,
                residual=z - v_best, dist=best_d,
            ))
        return results

    # --------------------------------------------------------
    def lift_to_discrete_log(self, y: np.ndarray) -> List[Optional[Dict]]:
        """
        For each output z that is (approximately) a positive integer,
        look up x_hat such that g^{x_hat} ≡ z (mod p).

        Requires self.ugc to be set.
        """
        if self.ugc is None:
            raise RuntimeError(
                "ugc not set; pass ugc_p to __init__ to enable dlog lift")
        results: List[Optional[Dict]] = []
        for z in y:
            if abs(z.imag) < 1e-6:
                val = int(round(z.real))
                if 0 < val < self.ugc.p:
                    x_hat, lv, ij = self.ugc.dlog_bsgs(val)
                    results.append(dict(
                        h=val, x_hat=x_hat, log_val=lv, ij=ij))
                    continue
            results.append(None)
        return results

    # --------------------------------------------------------
    def summary(self) -> str:
        return "\n".join([
            f"ALULinearOperator  (n = {self.n})",
            f"  K_alu        = {self.calu.K}",
            f"  K_signed     = {self.K_signed}",
            f"  split_mode   = {self.solver.split_mode}",
            f"  classes      = {self.classes}",
            f"  sf_counts    = {self.sf_counts}",
            f"  comp_counts  = {self.comp_counts}",
            f"  n_real/imag  = {sum(1 for c in self.classes if c=='real')}"
            f" / {sum(1 for c in self.classes if c=='imag')}",
            f"  ugc          = "
            f"Z_{self.ugc.p}^*  g={self.ugc.g}"
            if self.ugc else "  ugc          = (not set)",
        ])


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("ALULinearOperator  ·  L(x) = A·x on the QuantumALU")
    print("=" * 78)
    print(f"  α         = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP    = α/0.3628    = {A_STEP:.6f}")
    print(f"  6/π²                    = {DENSITY:.6f}")
    print()

    # ============================================================
    #  1 · build operator on a partitioned matrix
    # ============================================================
    print("=" * 78)
    print("1  ·  build operator  ·  partition (square-free / composite)")
    print("=" * 78)
    A = np.array([
        [2.0, 1.0, 0.0, 3.0],      # all square-free  → A_re
        [4.0, 8.0, 0.0, 9.0],      # all composite    → A_im
        [0.0, 2.0, 5.0, 1.0],      # all square-free  → A_re
        [4.0, 0.0, 9.0, 12.0],     # all composite    → A_im
    ])
    b = np.array([5.0, 7.0, 8.0, 10.0])
    print("  A =")
    for row in A:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print()

    op = ALULinearOperator(A, K_alu=64, split_mode='per_row',
                           ugc_p=101, ugc_g=2, K_signed=64)
    print(op.summary())
    print()
    print("  A_re (algebraic):")
    for row in op.A_re:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print("  A_im (transcendental):")
    for row in op.A_im:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print()

    # ============================================================
    #  2 · ALU vs numpy on a real vector
    # ============================================================
    print("=" * 78)
    print("2  ·  apply(x) via QuantumALU  vs  numpy  ·  real vector")
    print("=" * 78)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"  x          = {x}")
    y_alu = op.apply(x, use_alu=True)
    y_np  = op.apply_numpy(x)
    print(f"  y (ALU)    = {y_alu}")
    print(f"  y (numpy)  = {y_np}")
    print(f"  max|Δ|     = {np.max(np.abs(y_alu - y_np)):.2e}")
    print(f"  residual   = "
          f"{np.linalg.norm(op.A_c @ x - y_np):.2e}  "
          f"(A_c·x should equal y_np exactly)")
    print()

    # ============================================================
    #  3 · ALU vs numpy on a complex vector
    # ============================================================
    print("=" * 78)
    print("3  ·  apply(x) via QuantumALU  vs  numpy  ·  complex vector")
    print("=" * 78)
    x_c = np.array([1.0 + 2.0j, -3.0 + 1.0j, 0.5 - 0.5j, 2.0 + 0.0j])
    print(f"  x          = {x_c}")
    y_alu_c = op.apply(x_c, use_alu=True)
    y_np_c  = op.apply_numpy(x_c)
    print(f"  y (ALU)    = {y_alu_c}")
    print(f"  y (numpy)  = {y_np_c}")
    print(f"  max|Δ|     = {np.max(np.abs(y_alu_c - y_np_c)):.2e}")
    print()

    # ============================================================
    #  4 · signed-index substrate
    # ============================================================
    print("=" * 78)
    print("4  ·  signed-index substrate  v(k) = −H_{|k|+1} + i·α·(μ*H)(k)")
    print("=" * 78)
    print(f"  {'k':>4s}  {'Re v(k)':>12s}  {'Im v(k)':>12s}")
    for k in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        z = op.v(k)
        print(f"  {k:>4d}  {z.real:>+12.6f}  {z.imag:>+12.6f}")
    print()

    # ============================================================
    #  5 · apply via signed-index encoding
    # ============================================================
    print("=" * 78)
    print("5  ·  apply to a signed-index vector  x = [v(k_1), …]")
    print("=" * 78)
    k_vec = np.array([1, 2, 3, 4])          # signed indices
    print(f"  k vector   = {k_vec}")
    print(f"  v(k_j)     =")
    for kj in k_vec:
        print(f"    v({kj:+d}) = {op.v(int(kj))}")
    y_si = op.apply(k_vec, in_signed_index=True, use_alu=True)
    print(f"  y (ALU)    = {y_si}")
    print()

    # lift outputs to nearest v(k)
    lifts = op.lift_to_signed_index(y_si)
    print(f"  lift to nearest signed index:")
    for i, lf in enumerate(lifts):
        print(f"    y[{i}] = {lf['value'].real:+.6f} "
              f"{lf['value'].imag:+.6f}i  →  "
              f"v({lf['k']:+d})  dist={lf['dist']:.4f}")
    print()

    # ============================================================
    #  6 · discrete-log lift  (ugc_discrete_log.UGCDiscreteLog)
    # ============================================================
    print("=" * 78)
    print("6  ·  discrete-log lift  ·  UGCDiscreteLog  ·  BSGS")
    print("=" * 78)
    # small integer input so A·x stays small
    x_small = np.array([1.0, 1.0, 1.0, 1.0])
    y_small = op.apply(x_small, use_alu=True)
    print(f"  x        = {x_small}")
    print(f"  y (ALU)  = {y_small}")
    dlogs = op.lift_to_discrete_log(y_small)
    for i, dl in enumerate(dlogs):
        if dl:
            print(f"  y[{i}] = {dl['h']:>3d}  →  "
                  f"x_hat = {dl['x_hat']:>3d}   "
                  f"(i,j) = {dl['ij']}   "
                  f"−H_{{{dl['x_hat']+1}}} = {dl['log_val']:.4f}")
        else:
            print(f"  y[{i}] = {y_small[i]}  →  no integer dlog lift")
    print()

    # ============================================================
    #  7 · solve L(x) = b  via DualLinearSolver (unchanged)
    # ============================================================
    print("=" * 78)
    print("7  ·  solve A·x = b  via the unchanged DualLinearSolver")
    print("=" * 78)
    sol = op.solve(b)
    print(f"  method       = {sol.method}")
    print(f"  classes      = {sol.classes}")
    print(f"  sf counts    = {sol.sf_counts}")
    print(f"  comp counts  = {sol.comp_counts}")
    print(f"  x            = {sol.x}")
    print(f"  x_re         = {sol.x_re}")
    print(f"  x_im         = {sol.x_im}")
    print(f"  residual     = {sol.residuals['full']:.4e}")
    print()

    # ============================================================
    #  8 · composition  L1 then L2
    # ============================================================
    print("=" * 78)
    print("8  ·  composition  ·  L2(L1(x))  vs  (A2·A1)·x")
    print("=" * 78)
    A2 = np.array([
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 1.0, 0.0, 2.0],
        [3.0, 0.0, 1.0, 0.0],
        [0.0, 3.0, 0.0, 1.0],
    ])
    op2 = ALULinearOperator(A2, K_alu=64, split_mode='per_row',
                            ugc_p=101, ugc_g=2, K_signed=64)

    x0 = np.array([0.5, 1.0, -0.5, 2.0])
    y1 = op.apply(x0, use_alu=True)          # L1(x)
    y2 = op2.apply(y1, use_alu=True)         # L2(L1(x))

    # reference: (A2_c · A1_c) · x via numpy
    M = op2.A_c @ op.A_c
    ref = M @ x0.astype(np.complex128)

    print(f"  x0                    = {x0}")
    print(f"  L1(x0)     (ALU)      = {y1}")
    print(f"  L2(L1(x0)) (ALU)      = {y2}")
    print(f"  (A2·A1)·x0 (numpy)    = {ref}")
    print(f"  max|Δ|                = "
          f"{np.max(np.abs(y2 - ref)):.2e}")
    print()

    # ============================================================
    #  9 · matrix input  ·  column-by-column
    # ============================================================
    print("=" * 78)
    print("9  ·  matmul  ·  X (n × m) column-by-column via ALU")
    print("=" * 78)
    X = np.array([
        [1.0, 0.5],
        [2.0, 1.0],
        [3.0, 1.5],
        [4.0, 2.0],
    ])
    Y_alu = op.matmul(X, use_alu=True)
    Y_np  = op.matmul(X, use_alu=False)
    print(f"  X shape       = {X.shape}")
    print(f"  Y (ALU) shape = {Y_alu.shape}")
    print(f"  max|Δ|        = "
          f"{np.max(np.abs(Y_alu - Y_np)):.2e}")
    print()

    # ============================================================
    #  10 · complexity
    # ============================================================
    print("=" * 78)
    print("10 · complexity")
    print("=" * 78)
    print(f"  partition (DualLinearSolver)     O(n²)         unchanged")
    print(f"  QuantumALU                       O(K_alu)      per scalar op")
    print(f"  matvec via ALU                   O(n² · K_alu) per matvec")
    print(f"  matvec via numpy (comparison)    O(n²)")
    print(f"  solve via DualLinearSolver       O(n³)         unchanged")
    print(f"  signed-index lift                O(n · K_signed)")
    print(f"  discrete-log lift (BSGS)         O(√p)         per integer")
    print()
    print(f"  The ALU path is heavier than numpy: each scalar operation")
    print(f"  runs the full reconstruction-coupled convolution at size")
    print(f"  K_alu.  The trade is that the operator lives in the same")
    print(f"  algebra the discrete-log substrate uses:")
    print(f"    · square-free → real,  composite → imag   (DualLinearSolver)")
    print(f"    · finite step a = 1/(π−e)                 (QuantumALU)")
    print(f"    · signed-index v(k) = −H + i·α·(μ*H)      (n_k_div)")
    print(f"    · BSGS discrete log on Z_p^*              (ugc_discrete_log)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()