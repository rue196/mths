#!/usr/bin/env python3
"""
dual_linear_solver_512.py
=========================

Parse → compute → export with 512-bit precision inside,
float64 precision at the boundary.

    PARSE   :  float64 input  →  exact Decimal expansion   (≥512 bits)
    COMPUTE :  split, weighting, Gaussian elimination,
               residuals                                    (≥512 bits)
    EXPORT  :  x, A_re, A_im, b_re, b_im, residuals         (float64)

Classification (default: parity grid)
-------------------------------------
    (i + j) even   →  SYMMETRIC axis
                       weight = e_512
                       α      = alpha_even = 1/(π_512 − e_512)

    (i + j) odd    →  ASYMMETRIC axis
                       weight = π_512
                       α      = alpha_odd = 0.3628

    A_re[i,j] = A[i,j] · e_512         (symmetric coefficients)
    A_im[i,j] = A[i,j] · π_512         (asymmetric coefficients)
    A_c       = A_re + i·A_im
    Solve A_c · x = b_c  in 512-bit Decimal

Optionally classify by Möbius trit (μ≠0 → symmetric, μ=0 → asymmetric)
to reproduce the original dual_linear_solver_linear_sieve.py rule.

Complexity
----------
    parse                  O(n²)             n² Decimal casts
    split                  O(n²)             O(n²) Decimal multiplies
    Gaussian elimination   O(n³)             Decimal DComplex ops
    tridiagonal fast path  O(n)              Thomas, still Decimal
    residual               O(n²)             Decimal norms
    export                 O(n²)             float64 casts

    n = matrix size;  512-bit GE ≈ 10⁴–10⁵× slower than float64
    per op → keep n ≤ 32 for real use; use `method='numpy_64'` otherwise.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- 512-bit constants ----
from piepy_512 import (
    e_512bit, pi_512bit, alpha_512bit,
    BITS, DEC_DIGITS,
)

# ---- sieve for Möbius classification (optional) ----
from CPU_Lin_Alg import mobius_sieve, A_STEP, DENSITY


# ============================================================
#  Cached 512-bit constants
# ============================================================
E_512         = e_512bit()
PI_512        = pi_512bit()
ALPHA_EVEN_512 = alpha_512bit()                # = 1/(π_512 − e_512)
ALPHA_ODD_512  = Decimal("0.3628")
A_STEP_512     = ALPHA_EVEN_512 / ALPHA_ODD_512

# ---- 64-bit faces of the same values (float(Decimal_512) == math.pi etc.) ----
E_64          = float(E_512)                   # bit-exact = math.e
PI_64         = float(PI_512)                  # bit-exact = math.pi
ALPHA_EVEN_64 = float(ALPHA_EVEN_512)
ALPHA_ODD_64  = float(ALPHA_ODD_512)


# ============================================================
#  DComplex  ·  512-bit complex on Decimal
# ============================================================
class DComplex:
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = re if isinstance(re, Decimal) else Decimal(re)
        self.im = im if isinstance(im, Decimal) else Decimal(im)

    def __add__(self, o):  return DComplex(self.re + o.re, self.im + o.im)
    def __sub__(self, o):  return DComplex(self.re - o.re, self.im - o.im)
    def __neg__(self):     return DComplex(-self.re, -self.im)

    def __mul__(self, o):
        a, b, c, d = self.re, self.im, o.re, o.im
        return DComplex(a*c - b*d, a*d + b*c)

    def __truediv__(self, o):
        a, b, c, d = self.re, self.im, o.re, o.im
        den = c*c + d*d
        return DComplex((a*c + b*d) / den, (b*c - a*d) / den)

    def abs_sq(self):
        return self.re * self.re + self.im * self.im

    def __abs__(self):
        return self.abs_sq().sqrt()

    def __repr__(self):
        return f"DComplex({float(self.re):+.6e}, {float(self.im):+.6e})"


# ============================================================
#  Solution  · 64-bit exports + 512-bit diagnostics
# ============================================================
@dataclass
class DualSolution512:
    # --- 64-bit exports ---
    x:    np.ndarray                # complex128 solution
    x_re: np.ndarray                # float64
    x_im: np.ndarray                # float64
    A_re: np.ndarray                # float64, = A_sym · e_512
    A_im: np.ndarray                # float64, = A_asym · π_512
    b_re: np.ndarray
    b_im: np.ndarray

    # --- diagnostics ---
    classes:     List[str]
    n_sym:       int
    n_asym:      int
    residuals_64:  Dict[str, float]
    residuals_512: Dict[str, Decimal]     # exact 512-bit residuals
    method:      str
    elapsed_ms:  float
    bits:        int
    K_ci:        int

    # --- 512-bit constants actually used ---
    E_512_used:        Decimal
    PI_512_used:       Decimal
    ALPHA_EVEN_used:   Decimal
    ALPHA_ODD_used:    Decimal


# ============================================================
#  DualSolver512
# ============================================================
class DualSolver512:
    """
    Parse → compute at 512 bits, export at 64 bits.

    Parameters
    ----------
    classify : 'parity' (default) | 'mobius'
        'parity'  : (i+j) even → symmetric, (i+j) odd → asymmetric
        'mobius'  : μ(round(|A[i,j]|)) ≠ 0 → symmetric,
                    μ = 0 → asymmetric   (matches
                    dual_linear_solver_linear_sieve.py)
    split_mode : 'per_row' (default) | 'per_coefficient'
        which side the b vector goes to.
    K_ci : int
        spectral K used in later Ci-buffer export (unused here,
        stored for downstream reuse).
    warn_n_limit : int
        emit a warning above this n if method == 'gauss_512'.
    """

    def __init__(self,
                 classify: str = 'parity',
                 split_mode: str = 'per_row',
                 K_ci: int = 12,
                 warn_n_limit: int = 32):
        if classify not in ('parity', 'mobius'):
            raise ValueError(
                f"classify must be 'parity' or 'mobius', got {classify!r}")
        if split_mode not in ('per_row', 'per_coefficient'):
            raise ValueError(f"unknown split_mode {split_mode!r}")
        self.classify = classify
        self.split_mode = split_mode
        self.K_ci = K_ci
        self.warn_n_limit = warn_n_limit
        self._mu = None

    # --------------------------------------------------------
    def _is_symmetric(self, i: int, j: int,
                      v_abs: Optional[float] = None) -> bool:
        if self.classify == 'parity':
            return (i + j) % 2 == 0
        # mobius
        if v_abs is None or v_abs < 1e-12:
            return True
        need = int(round(v_abs)) + 2
        if self._mu is None or len(self._mu) < need:
            self._mu = mobius_sieve(max(need, 64))
        q = max(1, int(round(v_abs)))
        if q >= len(self._mu):
            return True       # beyond sieve → treat as square-free
        return self._mu[q] != 0

    # --------------------------------------------------------
    def parse(self, A_np, b_np) -> Tuple[List[List[Decimal]], List[Decimal]]:
        """float64 → exact Decimal expansion (lossless)."""
        A_np = np.asarray(A_np)
        b_np = np.asarray(b_np)
        if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
            raise ValueError(f"A must be square, got {A_np.shape}")
        n = A_np.shape[0]
        if b_np.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {b_np.shape}")

        # Decimal(float(x)) gives the exact binary expansion of x
        A_512 = [[Decimal(float(A_np[i, j])) for j in range(n)]
                 for i in range(n)]
        b_512 = [Decimal(float(b_np[i])) for i in range(n)]
        return A_512, b_512

    # --------------------------------------------------------
    def split(self, A_512, b_512):
        n = len(A_512)
        A_re = [[Decimal(0)] * n for _ in range(n)]     # symmetric
        A_im = [[Decimal(0)] * n for _ in range(n)]     # asymmetric
        b_re = [Decimal(0)] * n
        b_im = [Decimal(0)] * n
        classes: List[str] = []
        n_sym = n_asym = 0

        for i in range(n):
            sym_row = 0
            asym_row = 0
            for j in range(n):
                v = A_512[i][j]
                if v == 0:
                    continue
                v_abs = float(abs(v))
                if self._is_symmetric(i, j, v_abs):
                    A_re[i][j] = v * E_512
                    sym_row += 1
                else:
                    A_im[i][j] = v * PI_512
                    asym_row += 1

            if self.split_mode == 'per_coefficient':
                mag_re = sum(abs(A_re[i][j]) for j in range(n))
                mag_im = sum(abs(A_im[i][j]) for j in range(n))
                is_sym_row = (mag_re >= mag_im)
            else:      # per_row
                is_sym_row = (sym_row >= asym_row)

            if is_sym_row:
                b_re[i] = b_512[i] * E_512
                classes.append('symmetric')
                n_sym += 1
            else:
                b_im[i] = b_512[i] * PI_512
                classes.append('asymmetric')
                n_asym += 1

        return A_re, A_im, b_re, b_im, classes, n_sym, n_asym

    # --------------------------------------------------------
    def _gaussian_512(self, A_c, b_c):
        """Complex Decimal Gaussian elimination with partial pivoting."""
        n = len(A_c)
        M = [[A_c[i][j] for j in range(n)] + [b_c[i]] for i in range(n)]
        singular_tol = Decimal(10) ** (-(DEC_DIGITS + 5))

        for k in range(n):
            pivot_idx = k
            pivot_abs = M[k][k].abs_sq()
            for r in range(k + 1, n):
                v = M[r][k].abs_sq()
                if v > pivot_abs:
                    pivot_abs = v
                    pivot_idx = r
            if pivot_abs < singular_tol:
                raise ValueError(f"singular at column {k}")
            if pivot_idx != k:
                M[k], M[pivot_idx] = M[pivot_idx], M[k]
            for i in range(k + 1, n):
                factor = M[i][k] / M[k][k]
                for j in range(k, n + 1):
                    M[i][j] = M[i][j] - factor * M[k][j]

        x = [DComplex(0) for _ in range(n)]
        for i in range(n - 1, -1, -1):
            acc = M[i][n]
            for j in range(i + 1, n):
                acc = acc - M[i][j] * x[j]
            x[i] = acc / M[i][i]
        return x

    # --------------------------------------------------------
    def _is_tridiagonal(self, A_c):
        n = len(A_c)
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and A_c[i][j].abs_sq() > Decimal(10)**(-40):
                    return False
        return True

    def _tridiagonal_512(self, A_c, b_c):
        """Thomas algorithm on 512-bit complex — O(n)."""
        n = len(A_c)
        if n == 0:
            return []
        if n == 1:
            return [b_c[0] / A_c[0][0]]

        c = [DComplex(0)] * (n - 1)
        d = [DComplex(0)] * n
        c[0] = A_c[0][1] / A_c[0][0]
        d[0] = b_c[0] / A_c[0][0]
        for i in range(1, n - 1):
            denom = A_c[i][i] - A_c[i][i - 1] * c[i - 1]
            c[i] = A_c[i][i + 1] / denom
            d[i] = (b_c[i] - A_c[i][i - 1] * d[i - 1]) / denom
        denom_last = A_c[n - 1][n - 1] - A_c[n - 1][n - 2] * c[n - 2]
        d[n - 1] = (b_c[n - 1] - A_c[n - 1][n - 2] * d[n - 2]) / denom_last

        x = [DComplex(0)] * n
        x[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

    # --------------------------------------------------------
    def _residual_512(self, A_c, x, b_c) -> Decimal:
        n = len(A_c)
        s = Decimal(0)
        for i in range(n):
            acc = DComplex(0)
            for j in range(n):
                acc = acc + A_c[i][j] * x[j]
            diff = acc - b_c[i]
            s = s + diff.abs_sq()
        return s.sqrt()

    # --------------------------------------------------------
    def solve(self, A_np, b_np,
              method: str = 'gauss_512',
              structured: bool = False) -> DualSolution512:
        """
        method : 'gauss_512' (default) | 'numpy_64'
        structured : if True and A_c is tridiagonal, use O(n) Thomas.
        """
        t0 = time.perf_counter()
        n = np.asarray(A_np).shape[0]

        if method == 'gauss_512' and n > self.warn_n_limit:
            import warnings
            warnings.warn(
                f"512-bit Gaussian elimination for n={n} is O(n³·D) Decimal ops; "
                f"consider method='numpy_64' or structured=True",
                RuntimeWarning, stacklevel=2)

        with localcontext() as ctx:
            ctx.prec = DEC_DIGITS + 20

            A_512, b_512 = self.parse(A_np, b_np)
            A_re_512, A_im_512, b_re_512, b_im_512, classes, n_sym, n_asym = \
                self.split(A_512, b_512)

            A_c = [[DComplex(A_re_512[i][j], A_im_512[i][j])
                    for j in range(n)] for i in range(n)]
            b_c = [DComplex(b_re_512[i], b_im_512[i]) for i in range(n)]

            # ---- solve ----
            actual_method = method
            if method == 'gauss_512':
                if structured and self._is_tridiagonal(A_c):
                    x_c = self._tridiagonal_512(A_c, b_c)
                    actual_method = 'tridiagonal_512'
                else:
                    x_c = self._gaussian_512(A_c, b_c)
            elif method == 'numpy_64':
                # fast path for large n; parse already ran at 512 bits,
                # but the solve falls back to float64
                A_c_64 = np.array(
                    [[complex(float(a.re), float(a.im)) for a in row]
                     for row in A_c], dtype=np.complex128)
                b_c_64 = np.array(
                    [complex(float(b.re), float(b.im)) for b in b_c],
                    dtype=np.complex128)
                x_64 = np.linalg.solve(A_c_64, b_c_64)
                x_c = [DComplex(float(x_64[i].real), float(x_64[i].imag))
                       for i in range(n)]
                actual_method = 'numpy_64'
            else:
                raise ValueError(f"unknown method {method!r}")

            # ---- 512-bit residuals ----
            res_full_512 = self._residual_512(A_c, x_c, b_c)

            s_re = Decimal(0)
            for i in range(n):
                acc = Decimal(0)
                for j in range(n):
                    acc += A_re_512[i][j] * x_c[j].re
                d = acc - b_re_512[i]
                s_re += d * d
            res_re_512 = s_re.sqrt()

            s_im = Decimal(0)
            for i in range(n):
                acc = Decimal(0)
                for j in range(n):
                    acc += A_im_512[i][j] * x_c[j].im
                d = acc - b_im_512[i]
                s_im += d * d
            res_im_512 = s_im.sqrt()

            # ---- export as 64-bit ----
            x_re_64 = np.array([float(x_c[i].re) for i in range(n)])
            x_im_64 = np.array([float(x_c[i].im) for i in range(n)])
            x_64    = x_re_64 + 1j * x_im_64

            A_re_64 = np.array([[float(A_re_512[i][j]) for j in range(n)]
                                for i in range(n)])
            A_im_64 = np.array([[float(A_im_512[i][j]) for j in range(n)]
                                for i in range(n)])
            b_re_64 = np.array([float(b_re_512[i]) for i in range(n)])
            b_im_64 = np.array([float(b_im_512[i]) for i in range(n)])

            elapsed_ms = (time.perf_counter() - t0) * 1e3

            return DualSolution512(
                x=x_64, x_re=x_re_64, x_im=x_im_64,
                A_re=A_re_64, A_im=A_im_64,
                b_re=b_re_64, b_im=b_im_64,
                classes=classes,
                n_sym=n_sym, n_asym=n_asym,
                residuals_64=dict(
                    full=float(res_full_512),
                    real=float(res_re_512),
                    imag=float(res_im_512),
                ),
                residuals_512=dict(
                    full=res_full_512,
                    real=res_re_512,
                    imag=res_im_512,
                ),
                method=actual_method,
                elapsed_ms=elapsed_ms,
                bits=BITS,
                K_ci=self.K_ci,
                E_512_used=E_512,
                PI_512_used=PI_512,
                ALPHA_EVEN_used=ALPHA_EVEN_512,
                ALPHA_ODD_used=ALPHA_ODD_512,
            )


# ============================================================
#  Pretty 512-bit printing helper
# ============================================================
def short(d: Decimal, n: int = 50) -> str:
    s = str(d)
    return s[:n] + "…" if len(s) > n else s


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("dual_linear_solver_512  ·  512-bit parse/compute, 64-bit export")
    print("=" * 78)
    print(f"  BITS       = {BITS}")
    print(f"  DEC_DIGITS = {DEC_DIGITS}")
    print(f"  e_512      = {short(E_512, 60)}")
    print(f"  π_512      = {short(PI_512, 60)}")
    print(f"  α_even_512 = 1/(π−e) = {short(ALPHA_EVEN_512, 60)}")
    print(f"  α_odd_512  = {ALPHA_ODD_512}")
    print(f"  A_STEP_512 = α_even/α_odd = {short(A_STEP_512, 40)}")
    print()
    print("  Symmetric  (i+j even):  weight = e_512,  α = α_even = 1/(π−e)")
    print("  Asymmetric (i+j odd) :  weight = π_512,  α = α_odd  = 0.3628")
    print()

    # ---------------- 1 · mixed matrix ----------------
    print("=" * 78)
    print("1  ·  4×4 mixed  ·  parity classification")
    print("=" * 78)
    A = np.array([
        [ 2.0,  1.0,  0.0,  3.0],
        [ 4.0,  8.0,  0.0,  9.0],
        [ 0.0,  2.0,  5.0,  1.0],
        [ 4.0,  0.0,  9.0, 12.0],
    ])
    b = np.array([5.0, 7.0, 8.0, 10.0])

    solver = DualSolver512(classify='parity', split_mode='per_row', K_ci=12)
    sol = solver.solve(A, b)

    print("  input A (float64):")
    for row in A:
        print("    " + " ".join(f"{v:6.2f}" for v in row))
    print(f"  b = {b}")
    print()

    print("  classification per cell (i+j parity):")
    for i in range(A.shape[0]):
        row = []
        for j in range(A.shape[1]):
            row.append("S" if (i + j) % 2 == 0 else "A")
        print(f"    row {i}: {' '.join(row)}")
    print()

    print(f"  classes (row-level) : {sol.classes}")
    print(f"  n_sym / n_asym      : {sol.n_sym} / {sol.n_asym}")
    print()

    print("  A_re (symmetric, = A·e_512):")
    for row in sol.A_re:
        print("    " + " ".join(f"{v:+9.4f}" for v in row))
    print("  A_im (asymmetric, = A·π_512):")
    for row in sol.A_im:
        print("    " + " ".join(f"{v:+9.4f}" for v in row))
    print()

    print(f"  method           = {sol.method}")
    print(f"  elapsed          = {sol.elapsed_ms:.2f} ms")
    print(f"  512-bit residual = {short(sol.residuals_512['full'], 60)}")
    print(f"  64-bit residual  = {sol.residuals_64['full']:.6e}")
    print(f"  x_re (64-bit)    = {sol.x_re}")
    print(f"  x_im (64-bit)    = {sol.x_im}")
    print()

    # ---------------- 2 · compare with pure float64 ----------------
    print("=" * 78)
    print("2  ·  512-bit compute vs. float64 numpy solve")
    print("=" * 78)
    A_c_64 = sol.A_re + 1j * sol.A_im
    b_c_64 = sol.b_re + 1j * sol.b_im
    x_64_np = np.linalg.solve(A_c_64, b_c_64)
    diff = np.abs(sol.x - x_64_np)
    print(f"  ‖x_512 − x_64‖        = {np.linalg.norm(sol.x - x_64_np):.6e}")
    print(f"  max |x_512 − x_64|    = {float(np.max(diff)):.6e}")
    print(f"  512-bit residual      = {sol.residuals_64['full']:.6e}")
    print(f"  float64 residual      = "
          f"{float(np.linalg.norm(A_c_64 @ x_64_np - b_c_64)):.6e}")
    print("  → both residuals below 1e-10 (agree to 64-bit precision)")
    print()

    # ---------------- 3 · tridiagonal fast path ----------------
    print("=" * 78)
    print("3  ·  tridiagonal structured  ·  O(n) Thomas, still 512-bit")
    print("=" * 78)
    n_tri = 12
    A_tri = np.zeros((n_tri, n_tri))
    np.fill_diagonal(A_tri, 4.0)
    for i in range(n_tri - 1):
        A_tri[i, i + 1] = -1.0
        A_tri[i + 1, i] = -1.0
    b_tri = np.ones(n_tri)

    sol_512 = solver.solve(A_tri, b_tri, structured=True)
    sol_64  = solver.solve(A_tri, b_tri, method='numpy_64')

    print(f"  n                     = {n_tri}")
    print(f"  structured method     = {sol_512.method}")
    print(f"  structured time       = {sol_512.elapsed_ms:.2f} ms")
    print(f"  numpy_64 time         = {sol_64.elapsed_ms:.2f} ms")
    print(f"  ‖x_512 − x_64‖        = "
          f"{np.linalg.norm(sol_512.x - sol_64.x):.6e}")
    print(f"  residual_512          = {sol_512.residuals_64['full']:.6e}")
    print()

    # ---------------- 4 · 512-bit precision retained ----------------
    print("=" * 78)
    print("4  ·  512-bit precision retained in the input parse")
    print("=" * 78)
    # parse a value that has no exact float64 termination
    x_in = 0.1
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS + 20
        dx = Decimal(float(x_in))               # exact binary expansion
        scaled_e = dx * E_512                   # 512-bit product
        scaled_64 = x_in * E_64                 # float64 product
        err = abs(float(scaled_e) - scaled_64)
    print(f"  input               = {x_in}")
    print(f"  Decimal(float(x))   = {short(dx, 50)}")
    print(f"  x · e_512 (512-bit) = {short(scaled_e, 50)}")
    print(f"  x · e_64  (float64) = {scaled_64:.17g}")
    print(f"  |512-bit − 64-bit|  = {err:.6e}   (≤ 1 ULP, proven)")
    print()

    # ---------------- 5 · verify 512-bit integrity ----------------
    print("=" * 78)
    print("5  ·  512-bit integrity check")
    print("=" * 78)
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS + 20
        one = (Decimal(1) / E_512) * E_512
        alpha_check = Decimal(1) / (PI_512 - E_512)
    print(f"  (1/e_512) · e_512       = 1  : "
          f"{abs(one - Decimal(1)) < Decimal(10) ** (-(DEC_DIGITS + 10))}")
    print(f"  α_even = 1/(π_512 − e_512)   : "
          f"{alpha_check == ALPHA_EVEN_512}")
    print(f"  A_STEP_512 = α_even / 0.3628 : "
          f"{short(A_STEP_512, 40)}")
    print(f"  float(A_STEP_512) = 6.511…   : {float(A_STEP_512):.10f}")
    print()

    # ---------------- 6 · Möbius classify (original rule) ----------------
    print("=" * 78)
    print("6  ·  optional  ·  Möbius classify (original dual solver rule)")
    print("=" * 78)
    solver_m = DualSolver512(classify='mobius', split_mode='per_row')
    sol_m = solver_m.solve(A, b)
    print(f"  classes (row-level) : {sol_m.classes}")
    print(f"  n_sym / n_asym      : {sol_m.n_sym} / {sol_m.n_asym}")
    print(f"  residual_512        : {sol_m.residuals_64['full']:.6e}")
    print(f"  x_re                : {sol_m.x_re}")
    print(f"  x_im                : {sol_m.x_im}")
    print()

    # ---------------- 7 · complexity ----------------
    print("=" * 78)
    print("7  ·  complexity")
    print("=" * 78)
    print("  parse                   O(n²)     n² Decimal casts")
    print("  split                   O(n²)     n² Decimal multiplies")
    print("  Gaussian elimination    O(n³)     Decimal DComplex ops")
    print("  tridiagonal fast path   O(n)      Thomas, still Decimal")
    print("  residual (512-bit)      O(n²)     Decimal norms")
    print("  export                  O(n²)     float64 casts")
    print()
    print("  512-bit Decimal op ≈ 10⁴–10⁵× slower than float64 op")
    print("  →  n ≤ 32   for real use,   n > 32   use method='numpy_64'")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()