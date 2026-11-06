#!/usr/bin/env python3
"""
dual_linear_solver.py
=====================

64-bit CPU linear algebra with quadratic linear family partition.

Partition rule
--------------
For a matrix row i of A:

    n_sf   = |{ j : |A[i,j]| > 0  and  μ(round(|A[i,j]|)) ≠ 0 }|
    n_comp = |{ j : |A[i,j]| > 0  and  μ(round(|A[i,j]|)) =  0 }|

    n_sf  > 0    →  real axis         α = 0.3628       (algebraic)
                    (even OR odd count of square-free factors)
    n_sf  = 0    →  imaginary axis    α = 1/(π−e)      (transcendental)
                    (all coefficients composite)

Per-coefficient mode (split_mode='per_coefficient'):
    each coefficient A[i,j] is classified individually
        μ(round(|A[i,j]|)) ≠ 0 → contributes to A_re[i,j]
        μ(round(|A[i,j]|)) = 0 → contributes to A_im[i,j]
    and b[i] goes to the dominant axis for row i.

Complex embedding
-----------------
    A_c = A_re + i·A_im
    b_c = b_re + i·b_im
    Solve A_c · x = b_c

Complexity
----------
    Möbius sieve          O(K log log K)     numpy vectorized (CPU_Lin_Alg)
    per-row classify      O(n²)              one pass over coefficients
    per-coef separate     O(n² + K log K)    K = nonzero coefficients
    Gaussian elimination  O(n³)              general matrices
    tridiagonal solve     O(n)               structured fast path

For quadratic-family structured matrices (tridiagonal/banded), the
tridiagonal fast path reduces the solve to O(n).
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- reuse CPU_Lin_Alg.py's |Ci| buffer + numpy Möbius sieve ----
from CPU_Lin_Alg import (
    ReadOnlyCiBuffer,
    A_STEP,
    ALPHA,
    PI,
    E,
    mobius_sieve,      # numpy-vectorized  O(K log log K)
    DENSITY,
)

ALPHA_ODD  = 0.3628
ALPHA_EVEN = ALPHA   # 1/(π−e) ≈ 2.362


# ============================================================
#  Solution dataclass
# ============================================================
@dataclass
class DualSolution:
    x:           np.ndarray            # complex solution x_re + i·x_im
    x_re:        np.ndarray
    x_im:        np.ndarray
    A_re:        np.ndarray
    A_im:        np.ndarray
    b_re:        np.ndarray
    b_im:        np.ndarray
    A_c:         np.ndarray
    b_c:         np.ndarray
    classes:     List[str]             # per-row 'real' | 'imag'
    sf_counts:   List[int]             # square-free coefficient count per row
    comp_counts: List[int]             # composite coefficient count per row
    n_real:      int
    n_imag:      int
    residuals:   Dict[str, float]
    method:      str                   # 'gauss' | 'numpy' | 'tridiagonal' | 'lstsq'
    structured:  bool                  # tridiagonal fast path used?
    split_mode:  str                   # 'per_row' | 'per_coefficient'
    elapsed_ms:  float


# ============================================================
#  Solver
# ============================================================
class DualLinearSolver:
    """
    Dual Möbius linear solver — quadratic linear family partition.

    Classification
    --------------
    Any square-free coefficient (μ≠0) in row i  →  real (algebraic)
        (regardless of even or odd count of square-free factors)
    All coefficients composite (μ=0) in row i   →  imag (transcendental)

    Parameters
    ----------
    K : optional int
        Minimum sieve size.  The sieve is auto-grown to cover the
        largest coefficient magnitude in A.
    regularize : float
        Pivot threshold for singularity detection.
    use_coeff_classify : bool
        If True, classify by scanning row coefficients.
        If False, classify by row index  i → μ(i+1).
    split_mode : str
        'per_row'          (default)  — whole row → one axis
        'per_coefficient'  — each coefficient → its own axis
    """

    def __init__(self,
                 K: Optional[int] = None,
                 regularize: float = 1e-12,
                 use_coeff_classify: bool = True,
                 split_mode: str = 'per_row'):
        self.K = K
        self.regularize = regularize
        self.use_coeff_classify = use_coeff_classify
        if split_mode not in ('per_row', 'per_coefficient'):
            raise ValueError(f"unknown split_mode {split_mode!r}")
        self.split_mode = split_mode
        self._mu: Optional[np.ndarray] = None

    # --------------------------------------------------------
    def _get_mu(self, A: np.ndarray) -> np.ndarray:
        """Grow the numpy Möbius sieve to cover all coefficient magnitudes."""
        max_val = 1
        for val in A.flat:
            if abs(val) >= 1e-12:
                max_val = max(max_val, int(round(abs(val))))
        need = max(max_val + 2, 64)
        if self.K is not None:
            need = max(need, self.K + 2)
        if self._mu is None or len(self._mu) < need:
            self._mu = mobius_sieve(need)   # numpy O(K log log K)
        return self._mu

    # --------------------------------------------------------
    def classify_row(self, i: int, A: np.ndarray, mu: np.ndarray
                     ) -> Tuple[str, int, int]:
        """
        Per-row classification (quadratic family rule).

        Returns (side, n_square_free, n_composite).
            side = 'real' if n_square_free > 0  (even OR odd count)
            side = 'imag' if all coefficients are composite
        """
        if not self.use_coeff_classify:
            idx = i + 1
            if idx < len(mu) and mu[idx] != 0:
                return 'real', 1, 0
            return 'imag', 0, 1

        row = A[i, :]
        nz_mask = np.abs(row) >= 1e-12
        if not np.any(nz_mask):
            return 'real', 0, 0

        nz_vals = np.abs(row[nz_mask])
        ints = np.round(nz_vals).astype(np.int64)
        ints = np.maximum(ints, 1)

        in_range = ints < len(mu)
        n_sf = 0
        n_comp = 0
        for k in range(len(ints)):
            if in_range[k]:
                if mu[ints[k]] != 0:
                    n_sf += 1
                else:
                    n_comp += 1
            else:
                n_sf += 1     # beyond sieve → treat as square-free

        return ('real' if n_sf > 0 else 'imag'), n_sf, n_comp

    # --------------------------------------------------------
    def split_axes(self, A, b):
        """Dispatch to per-row or per-coefficient split."""
        if self.split_mode == 'per_row':
            return self._split_per_row(A, b)
        return self._split_per_coefficient(A, b)

    def _split_per_row(self, A, b):
        n = A.shape[0]
        mu = self._get_mu(A)
        A_re = np.zeros_like(A)
        A_im = np.zeros_like(A)
        b_re = np.zeros_like(b)
        b_im = np.zeros_like(b)
        classes: List[str] = []
        sf_counts: List[int] = []
        comp_counts: List[int] = []

        for i in range(n):
            side, n_sf, n_comp = self.classify_row(i, A, mu)
            classes.append(side)
            sf_counts.append(n_sf)
            comp_counts.append(n_comp)
            if side == 'real':
                A_re[i, :] = A[i, :]
                b_re[i] = b[i]
            else:
                A_im[i, :] = A[i, :]
                b_im[i] = b[i]

        return A_re, b_re, A_im, b_im, classes, sf_counts, comp_counts

    def _split_per_coefficient(self, A, b):
        n, m = A.shape
        mu = self._get_mu(A)
        A_re = np.zeros_like(A)
        A_im = np.zeros_like(A)
        b_re = np.zeros_like(b)
        b_im = np.zeros_like(b)
        classes: List[str] = []
        sf_counts: List[int] = []
        comp_counts: List[int] = []

        for i in range(n):
            row = A[i, :]
            n_sf = 0
            n_comp = 0
            norm_re = 0.0
            norm_im = 0.0

            for j in range(m):
                val = row[j]
                if abs(val) < 1e-12:
                    continue
                q = max(1, int(round(abs(val))))
                is_sf = (q < len(mu)) and (mu[q] != 0)
                if is_sf:
                    A_re[i, j] = val
                    n_sf += 1
                    norm_re += abs(val)
                else:
                    A_im[i, j] = val
                    n_comp += 1
                    norm_im += abs(val)

            sf_counts.append(n_sf)
            comp_counts.append(n_comp)

            if norm_re >= norm_im:
                b_re[i] = b[i]
                classes.append('real')
            else:
                b_im[i] = b[i]
                classes.append('imag')

        return A_re, b_re, A_im, b_im, classes, sf_counts, comp_counts

    # --------------------------------------------------------
    def separate_square_free(self, A, mu=None):
        """
        Separate coefficients into square-free and composite groups,
        sorted by (row, col) in O(K log K) where K = #nonzero coefficients.

        Returns (sf_list, comp_list), each a list of (i, j, value).
        """
        if mu is None:
            mu = self._get_mu(A)
        sf = []
        comp = []
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                val = abs(A[i, j])
                if val < 1e-12:
                    continue
                q = max(1, int(round(val)))
                if q < len(mu) and mu[q] != 0:
                    sf.append((i, j, float(A[i, j])))
                else:
                    comp.append((i, j, float(A[i, j])))
        sf.sort(key=lambda t: (t[0], t[1]))       # O(K log K)
        comp.sort(key=lambda t: (t[0], t[1]))
        return sf, comp

    # --------------------------------------------------------
    def gaussian_elimination(self, A_c, b_c):
        """Complex Gaussian elimination with partial pivoting."""
        n = A_c.shape[0]
        M = np.zeros((n, n + 1), dtype=np.complex128)
        M[:, :n] = A_c
        M[:, n] = b_c

        for k in range(n):
            col = np.abs(M[k:, k])
            pv = float(col.max())
            if pv < self.regularize:
                raise np.linalg.LinAlgError(
                    f"singular at column {k} (|pivot|={pv:.2e})")
            p = k + int(np.argmax(col))
            if p != k:
                M[[k, p]] = M[[p, k]]
            for i in range(k + 1, n):
                f = M[i, k] / M[k, k]
                M[i, k:] -= f * M[k, k:]

        x = np.zeros(n, dtype=np.complex128)
        for i in range(n - 1, -1, -1):
            rhs = M[i, n] - M[i, i + 1:n] @ x[i + 1:]
            x[i] = rhs / M[i, i]
        return x

    def _is_tridiagonal(self, A):
        n = A.shape[0]
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(A[i, j]) > 1e-12:
                    return False
        return True

    def _solve_tridiagonal(self, A_c, b_c):
        """Thomas algorithm — O(n)."""
        n = A_c.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.complex128)
        if n == 1:
            return np.array([b_c[0] / A_c[0, 0]], dtype=np.complex128)

        c = np.zeros(n - 1, dtype=np.complex128)
        d = np.zeros(n, dtype=np.complex128)
        c[0] = A_c[0, 1] / A_c[0, 0]
        d[0] = b_c[0] / A_c[0, 0]
        for i in range(1, n - 1):
            denom = A_c[i, i] - A_c[i, i - 1] * c[i - 1]
            c[i] = A_c[i, i + 1] / denom
            d[i] = (b_c[i] - A_c[i, i - 1] * d[i - 1]) / denom
        d[n - 1] = ((b_c[n - 1] - A_c[n - 1, n - 2] * d[n - 2])
                    / (A_c[n - 1, n - 1] - A_c[n - 1, n - 2] * c[n - 2]))

        x = np.zeros(n, dtype=np.complex128)
        x[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

    # --------------------------------------------------------
    def solve(self, A, b, method: str = 'gauss',
              structured: bool = False) -> DualSolution:
        t0 = time.perf_counter()
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"A must be square, got {A.shape}")
        if b.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {b.shape}")

        A_re, b_re, A_im, b_im, classes, sf_counts, comp_counts = \
            self.split_axes(A, b)
        A_c = A_re + 1j * A_im
        b_c = b_re + 1j * b_im

        actual_method = method
        used_structured = False
        try:
            if structured and self._is_tridiagonal(A_c):
                x = self._solve_tridiagonal(A_c, b_c)
                actual_method = 'tridiagonal'
                used_structured = True
            elif method == 'gauss':
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
            sf_counts=sf_counts,
            comp_counts=comp_counts,
            n_real=sum(1 for c in classes if c == 'real'),
            n_imag=sum(1 for c in classes if c == 'imag'),
            residuals=residuals,
            method=actual_method,
            structured=used_structured,
            split_mode=self.split_mode,
            elapsed_ms=elapsed_ms,
        )

    # --------------------------------------------------------
    def build_ci_buffer(self, sol: DualSolution) -> ReadOnlyCiBuffer:
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
    print("Dual Möbius linear algebra  ·  quadratic family partition")
    print("=" * 78)
    print(f"  α_odd   = 0.3628                 (square-free → real)")
    print(f"  α_even  = 1/(π−e) = {ALPHA_EVEN:.6f}   (composite  → imag)")
    print(f"  A_STEP  = α / 0.3628 = {A_STEP:.6f}")
    print(f"  6/π²    = {DENSITY:.6f}   (μ density)")
    print()
    print("  Partition rule:")
    print("    any square-free (μ≠0, even or odd count) → real (algebraic)")
    print("    all composite (μ=0)                      → imag (transcendental)")
    print()

    # ---------- 1 · mixed matrix ----------
    print("=" * 78)
    print("1  ·  per-row split  ·  mixed square-free / composite rows")
    print("=" * 78)
    A = np.array([
        [2.0, 1.0, 0.0, 3.0],   # 2,1,3 all square-free   → real
        [4.0, 8.0, 0.0, 9.0],   # 4,8,9 all composite     → imag
        [0.0, 2.0, 5.0, 1.0],   # 2,5,1 all square-free   → real
        [4.0, 0.0, 9.0, 12.0],  # 4,9,12 all composite    → imag
    ])
    b = np.array([5.0, 7.0, 8.0, 10.0])
    print("  A =")
    for row in A:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print(f"  b = {b}")
    print()

    solver = DualLinearSolver(split_mode='per_row')
    sol = solver.solve(A, b)
    print(f"  classes      : {sol.classes}")
    print(f"  sf counts    : {sol.sf_counts}")
    print(f"  comp counts  : {sol.comp_counts}")
    print(f"  n_real/imag  : {sol.n_real}/{sol.n_imag}")
    print(f"  method       : {sol.method}")
    print(f"  elapsed      : {sol.elapsed_ms:.3f} ms")
    print(f"  ‖residual‖   : {sol.residuals['full']:.4e}")
    print()
    print(f"  x            = {sol.x}")
    print(f"  x_re         = {sol.x_re}")
    print(f"  x_im         = {sol.x_im}")
    print()

    # ---------- 2 · per-coefficient split ----------
    print("=" * 78)
    print("2  ·  per-coefficient split  ·  each coefficient independently")
    print("=" * 78)
    solver2 = DualLinearSolver(split_mode='per_coefficient')
    sol2 = solver2.solve(A, b)
    print(f"  sf counts    : {sol2.sf_counts}")
    print(f"  comp counts  : {sol2.comp_counts}")
    print(f"  n_real/imag  : {sol2.n_real}/{sol2.n_imag}")
    print(f"  ‖residual‖   : {sol2.residuals['full']:.4e}")
    print(f"  A_re:")
    for row in sol2.A_re:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print(f"  A_im:")
    for row in sol2.A_im:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print()

    # ---------- 3 · tridiagonal fast path ----------
    print("=" * 78)
    print("3  ·  tridiagonal structured fast path  ·  O(n)")
    print("=" * 78)
    n_tri = 64
    A_tri = np.zeros((n_tri, n_tri))
    np.fill_diagonal(A_tri, 4.0)
    for i in range(n_tri - 1):
        A_tri[i, i + 1] = -1.0
        A_tri[i + 1, i] = -1.0
    b_tri = np.ones(n_tri)

    sol_tri  = solver.solve(A_tri, b_tri, structured=True)
    sol_gauss = solver.solve(A_tri, b_tri, structured=False)

    print(f"  n              = {n_tri}")
    print(f"  tridiag method = {sol_tri.method}")
    print(f"  tridiag time   = {sol_tri.elapsed_ms:.3f} ms")
    print(f"  gauss   time   = {sol_gauss.elapsed_ms:.3f} ms")
    print(f"  speedup        = "
          f"{sol_gauss.elapsed_ms / max(sol_tri.elapsed_ms, 0.001):.2f}×")
    print(f"  ‖x_tri − x_gauss‖ = "
          f"{np.linalg.norm(sol_tri.x - sol_gauss.x):.2e}")
    print()

    # ---------- 4 · separation O(K log K) ----------
    print("=" * 78)
    print("4  ·  separate_square_free  ·  O(K log K) sort")
    print("=" * 78)
    sf, comp = solver.separate_square_free(A)
    print(f"  square-free coefficients : {len(sf)}")
    for i, j, v in sf[:6]:
        print(f"    ({i},{j}) = {v:+.2f}")
    print(f"  composite coefficients   : {len(comp)}")
    for i, j, v in comp[:6]:
        print(f"    ({i},{j}) = {v:+.2f}")
    print()

    # ---------- 5 · larger random ----------
    print("=" * 78)
    print("5  ·  random n = 16  ·  quadratic family partition")
    print("=" * 78)
    rng = np.random.default_rng(42)
    n = 16
    A_r = np.round(rng.standard_normal((n, n)) * 10).astype(np.float64)
    b_r = np.round(rng.standard_normal(n) * 10).astype(np.float64)
    sol_r = solver.solve(A_r, b_r)
    print(f"  classes      : {sol_r.classes}")
    print(f"  n_real/imag  : {sol_r.n_real}/{sol_r.n_imag}")
    print(f"  sf counts    : {sol_r.sf_counts}")
    print(f"  comp counts  : {sol_r.comp_counts}")
    print(f"  method       : {sol_r.method}")
    print(f"  ‖residual‖   : {sol_r.residuals['full']:.4e}")
    print()

    # ---------- 6 · complexity ----------
    print("=" * 78)
    print("6  ·  complexity")
    print("=" * 78)
    print("  Möbius sieve          O(K log log K)    numpy vectorized")
    print("  per-row classify      O(n²)             one pass over coefficients")
    print("  per-coef classify     O(n²)")
    print("  separate_square_free  O(n² + K log K)   sort of nonzero entries")
    print("  Gaussian elimination  O(n³)             general matrices")
    print("  tridiagonal solve     O(n)              structured fast path")
    print()
    print("  K = sieve size,  n = matrix size")
    print()
    print("  For quadratic-family structured matrices (tridiagonal/banded),")
    print("  the tridiagonal fast path reduces the solve to O(n).")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()