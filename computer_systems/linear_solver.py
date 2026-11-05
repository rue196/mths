#!/usr/bin/env python3
"""
dual_linear_solver.py
=====================

64-bit CPU linear algebra with Möbius-axis classification.

    odd  number of square-free factors  →  real axis       α = 0.3628
    even number of square-free factors  →  imaginary axis  α = 1/(π−e) ≈ 2.362

System Ax = b
    Each row is classified by factor.py's rule.  Rows are
    partitioned into A_re (real) and A_im (imag).

    Complex embedding:
        A_c = A_re + i·A_im
        b_c = b_re + i·b_im

    Solve:
        A_c · x = b_c     (x ∈ ℂⁿ, double precision)

    Split:
        x_re = Re(x)      →  real-axis component
        x_im = Im(x)      →  imag-axis component

    For real-valued A, b and a fully-classified split, this gives
    x_re solving A_re x_re = b_re and x_im solving A_im x_im = b_im.

Complexity
----------
    classify rows           O(n · |row support|)
    split axes              O(n²)
    Gaussian elimination    O(n³)   complex, partial pivoting
    residuals               O(n²)
    |Ci| buffer             O(n)

Method
------
    Primary:  Gaussian elimination with partial pivoting (complex, float64)
    Fallback: np.linalg.lstsq  (for singular / ill-conditioned)
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- reuse factor.py's classification rule ----
from factor import (
    classify,
    mobius_sieve,
    quadratic_address,
    ALPHA_ODD,
    ALPHA_EVEN,
    DENSITY,
)

# ---- reuse CPU_Lin_Alg.py's |Ci| buffer ----
from CPU_Lin_Alg import (
    ReadOnlyCiBuffer,
    A_STEP,
    ALPHA,
    PI,
    E,
)


# ============================================================
#  Solution dataclass
# ============================================================
@dataclass
class DualSolution:
    x:           np.ndarray            # complex solution x_re + i·x_im
    x_re:        np.ndarray            # real part  (symmetric / odd-Ω)
    x_im:        np.ndarray            # imag part  (asymmetric / even-Ω)
    A_re:        np.ndarray
    A_im:        np.ndarray
    b_re:        np.ndarray
    b_im:        np.ndarray
    A_c:         np.ndarray            # A_re + i·A_im
    b_c:         np.ndarray            # b_re + i·b_im
    classes:     List[str]             # per-row 'real' | 'imag'
    n_real:      int
    n_imag:      int
    residuals:   Dict[str, float]
    method:      str                   # 'gauss' | 'lstsq'
    elapsed_ms:  float


# ============================================================
#  Solver
# ============================================================
class DualLinearSolver:
    """
    Dual Möbius linear solver.

    Classification
    --------------
    For each row i of A:
        count = |{ j : |A[i,j]| > 0  and  μ(round(|A[i,j]|)) ≠ 0 }|
        side  = classify(count, μ)

        odd count  → 'real'      (symmetric,  α = 0.3628)
        even count → 'imag'      (asymmetric, α = 1/(π−e))

    Set  `use_coeff_classify=False`  to classify by row index instead
    (i → classify(i+1, μ))  — matches factor.py's default behaviour.
    """
    def __init__(self, K: Optional[int] = None, regularize: float = 1e-12,
                 use_coeff_classify: bool = True):
        self.K = K
        self.regularize = regularize
        self.use_coeff_classify = use_coeff_classify
        self._mu: Optional[List[int]] = None

    # --------------------------------------------------------
    def _get_mu(self, n: int) -> List[int]:
        need = max(n + 2, 64)
        if self._mu is None or len(self._mu) < need:
            self._mu = mobius_sieve(need)
        return self._mu

    # --------------------------------------------------------
    def classify_row(self, i: int, A: np.ndarray,
                     mu: List[int]) -> str:
        if not self.use_coeff_classify:
            return classify(i + 1, mu)
        count = 0
        for val in A[i, :]:
            if abs(val) < 1e-12:
                continue
            n = max(1, int(round(abs(val))))
            if n < len(mu) and mu[n] != 0:
                count += 1
        return classify(max(1, count), mu)

    # --------------------------------------------------------
    def split_axes(self, A: np.ndarray, b: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray, List[str]]:
        n = A.shape[0]
        mu = self._get_mu(n)
        A_re = np.zeros_like(A)
        A_im = np.zeros_like(A)
        b_re = np.zeros_like(b)
        b_im = np.zeros_like(b)
        classes: List[str] = []
        for i in range(n):
            side = self.classify_row(i, A, mu)
            classes.append(side)
            if side == 'real':
                A_re[i, :] = A[i, :]
                b_re[i] = b[i]
            else:
                A_im[i, :] = A[i, :]
                b_im[i] = b[i]
        return A_re, b_re, A_im, b_im, classes

    # --------------------------------------------------------
    def gaussian_elimination(self, A_c: np.ndarray,
                             b_c: np.ndarray) -> np.ndarray:
        """
        Complex Gaussian elimination with partial pivoting.

        Solves A_c · x = b_c  in float64 complex precision.
        """
        n = A_c.shape[0]
        M = np.zeros((n, n + 1), dtype=np.complex128)
        M[:, :n] = A_c
        M[:, n] = b_c

        # ---- forward elimination ----
        for k in range(n):
            col = np.abs(M[k:, k])
            pivot_val = float(col.max())
            if pivot_val < self.regularize:
                raise np.linalg.LinAlgError(
                    f"singular at column {k}  (|pivot|={pivot_val:.2e})")
            p = k + int(np.argmax(col))
            if p != k:
                M[[k, p]] = M[[p, k]]
            for i in range(k + 1, n):
                factor = M[i, k] / M[k, k]
                M[i, k:] -= factor * M[k, k:]

        # ---- back substitution ----
        x = np.zeros(n, dtype=np.complex128)
        for i in range(n - 1, -1, -1):
            rhs = M[i, n] - M[i, i + 1:n] @ x[i + 1:]
            x[i] = rhs / M[i, i]
        return x

    # --------------------------------------------------------
    def solve(self, A, b, method: str = 'gauss') -> DualSolution:
        t0 = time.perf_counter()
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"A must be square, got {A.shape}")
        if b.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {b.shape}")

        A_re, b_re, A_im, b_im, classes = self.split_axes(A, b)
        A_c = A_re + 1j * A_im
        b_c = b_re + 1j * b_im

        actual_method = method
        try:
            if method == 'gauss':
                x = self.gaussian_elimination(A_c, b_c)
            elif method == 'numpy':
                x = np.linalg.solve(A_c, b_c)
            else:
                raise ValueError(f"unknown method {method!r}")
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A_c, b_c, rcond=None)[0]
            actual_method = 'lstsq'

        x_re = np.real(x).copy()
        x_im = np.imag(x).copy()

        residuals = dict(
            full=float(np.linalg.norm(A_c @ x - b_c)),
            real=float(np.linalg.norm(A_re @ x - b_re)),
            imag=float(np.linalg.norm(A_im @ x - b_im)),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        return DualSolution(
            x=x, x_re=x_re, x_im=x_im,
            A_re=A_re, A_im=A_im, b_re=b_re, b_im=b_im,
            A_c=A_c, b_c=b_c,
            classes=classes,
            n_real=sum(1 for c in classes if c == 'real'),
            n_imag=sum(1 for c in classes if c == 'imag'),
            residuals=residuals,
            method=actual_method,
            elapsed_ms=elapsed_ms,
        )

    # --------------------------------------------------------
    def build_ci_buffer(self, sol: DualSolution) -> ReadOnlyCiBuffer:
        """
        Embed the solution into a ReadOnlyCiBuffer.

            c[K + k] = α_odd  · x_re[k]    (real half, symmetric)
            c[K − k] = α_even · x_im[k]    (imag half, asymmetric)
            c[K]     = Σ α_odd · x_re[k]   (DC)
        """
        n = sol.x.size
        K = max(n, 64)
        buf = ReadOnlyCiBuffer(K)
        for k in range(1, n + 1):
            side = sol.classes[k - 1]
            if side == 'real':
                buf.write_sym(k, ALPHA_ODD * float(sol.x_re[k - 1]))
            else:
                buf.write_asym(k, ALPHA_EVEN * float(sol.x_im[k - 1]))
        dc = sum(ALPHA_ODD * float(sol.x_re[k]) for k in range(n)
                 if sol.classes[k] == 'real')
        buf.write_dc(dc)
        buf.seal()
        return buf


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("64-bit dual linear algebra  ·  factor.py axis classification")
    print("=" * 78)
    print(f"  α_odd   = 0.3628                 (odd Ω  → real / symmetric)")
    print(f"  α_even  = 1/(π−e) = {ALPHA_EVEN:.6f}  (even Ω → imag / asym)")
    print(f"  A_STEP  = α / 0.3628 = {A_STEP:.6f}")
    print(f"  6/π²    = {DENSITY:.6f}")
    print()

    solver = DualLinearSolver(use_coeff_classify=True)

    # ---------- 1 · structured 4×4 system ----------
    print("=" * 78)
    print("1  ·  4×4 integer system")
    print("=" * 78)
    A = np.array([
        [2.0, 1.0, 0.0, 3.0],
        [1.0, 4.0, 2.0, 0.0],
        [0.0, 2.0, 5.0, 1.0],
        [3.0, 0.0, 1.0, 6.0],
    ])
    b = np.array([5.0, 7.0, 8.0, 10.0])
    print("  A =")
    for row in A:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print(f"  b = {b}")
    print()

    sol = solver.solve(A, b, method='gauss')
    print(f"  classes       : {sol.classes}")
    print(f"  n_real, n_imag: {sol.n_real}, {sol.n_imag}")
    print(f"  method        : {sol.method}   "
          f"time = {sol.elapsed_ms:.3f} ms")
    print(f"  x             = {sol.x}")
    print(f"  x_re          = {sol.x_re}")
    print(f"  x_im          = {sol.x_im}")
    print()
    print("  residuals:")
    for k, v in sol.residuals.items():
        print(f"    {k:>5s}  = {v:.4e}")

    # verify against standard real solve
    x_std = np.linalg.solve(A, b)
    print()
    print(f"  standard solve  x_std = {x_std}")
    print(f"  ‖x_re − x_std‖        = "
          f"{np.linalg.norm(sol.x_re - x_std):.4e}")
    print()

    # ---------- 2 · identity / diagonal ----------
    print("=" * 78)
    print("2  ·  diagonal system  (mixed classification)")
    print("=" * 78)
    A_d = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b_d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sol_d = solver.solve(A_d, b_d, method='gauss')
    print(f"  classes = {sol_d.classes}")
    print(f"  x       = {sol_d.x}")
    print(f"  ‖x‖     = {np.linalg.norm(sol_d.x):.6f}")
    print()

    # ---------- 3 · |Ci| reconstruction ----------
    print("=" * 78)
    print("3  ·  |Ci| buffer reconstruction")
    print("=" * 78)
    buf = solver.build_ci_buffer(sol)
    print(f"  buffer K   = {buf.K}")
    print(f"  supertrace = {buf.supertrace():+.6f}")
    print(f"  DC         = {buf.dc():+.6f}")
    print()
    print(f"  per-index:")
    print(f"  {'k':>3}  {'class':>5}  {'sym part':>14}  {'asym part':>14}")
    for k in range(1, sol.x.size + 1):
        print(f"  {k:>3}  {sol.classes[k-1]:>5}  "
              f"{buf.sym(k):>14.6f}  {buf.asym(k):>14.6f}")
    print()

    # ---------- 4 · larger random ----------
    print("=" * 78)
    print("4  ·  random n = 16 system")
    print("=" * 78)
    rng = np.random.default_rng(42)
    n = 16
    A_r = np.round(rng.standard_normal((n, n)) * 10).astype(np.float64)
    b_r = np.round(rng.standard_normal(n) * 10).astype(np.float64)
    sol_r = solver.solve(A_r, b_r, method='gauss')
    print(f"  n_real={sol_r.n_real}   n_imag={sol_r.n_imag}")
    print(f"  method={sol_r.method}   time={sol_r.elapsed_ms:.3f} ms")
    print(f"  ‖residual full‖    = {sol_r.residuals['full']:.4e}")
    print(f"  ‖residual real‖    = {sol_r.residuals['real']:.4e}")
    print(f"  ‖residual imag‖    = {sol_r.residuals['imag']:.4e}")
    print()

    # ---------- 5 · singular fallback ----------
    print("=" * 78)
    print("5  ·  singular system → lstsq fallback")
    print("=" * 78)
    A_s = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],        # singular row 3 = 2·row 2 − row 1
    ])
    b_s = np.array([1.0, 2.0, 3.0])
    sol_s = solver.solve(A_s, b_s, method='gauss')
    print(f"  method = {sol_s.method}")
    print(f"  x      = {sol_s.x}")
    print(f"  ‖residual‖ = {sol_s.residuals['full']:.4e}")
    print()

    # ---------- 6 · complexity ----------
    print("=" * 78)
    print("6  ·  complexity")
    print("=" * 78)
    print(f"  classify rows       O(n · support)     ≤ O(n²)")
    print(f"  split_axes          O(n²)")
    print(f"  gaussian (complex)  O(n³)              partial pivoting")
    print(f"  residuals           O(n²)")
    print(f"  |Ci| buffer         O(n)")
    print(f"  ─────────────────────────────────────")
    print(f"  total               O(n³)              dominant term")
    print()
    print(f"  All arithmetic in  float64 / complex128  (64-bit precision)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()