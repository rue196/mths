#!/usr/bin/env python3
"""
fft_dual_linear_solver.py
=========================

FFT extension of dual_linear_solver_linear_sieve.py.

Each matrix row is a |Ci| array of length 2·K_ci + 1.  The pipeline:

    1. split row → symmetric S + asymmetric A       split_symmetric_asymmetric
    2. FFT compress each part, keep top-M by |mag|   fft_compress_part
    3. inverse FFT → reconstructed row                fft_decompress_part
    4. extract center window of length n
    5. A_re[i,:] = window(S),  A_im[i,:] = window(A)
    6. A_c = A_re + i·A_im                            (parent complex embed)
    7. Gaussian elimination with |supertrace| pivot   _supertrace
       → the alternating sum  S = Σ (−1)^k v_k  is the pivot criterion
    8. fall back to magnitude pivot when supertrace is degenerate

Everything else (per-row Möbius classification, separate_square_free,
tridiagonal fast path, traditional numpy solve, lstsq fallback,
build_ci_buffer) is inherited from DualLinearSolver unchanged.

Imports (both modules unchanged)
--------------------------------
    from dual_linear_solver_linear_sieve import DualLinearSolver, ...
    load 'memory-okloglogk-mobius.py' by path (hyphenated filename)
"""

from __future__ import annotations
import importlib.util
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- parent solver  ·  unchanged ----
from dual_linear_solver_linear_sieve import (
    DualLinearSolver,
    DualSolution,
    ALPHA_ODD,
    ALPHA_EVEN,
    DENSITY,
)

from memory_okloglogk_mobius import(
    mobius_sieve, 
     _supertrace,
     split_symmetric_asymmetric,
     fft_compress_part, 
    fft_decompress_part,
    FFTPart,
    conv_exp_kernel,
    build_k_sum,
    ALPHA,
    A_STEP,
    DENSITY

                                    

)


# ============================================================
#  Row classification container
# ============================================================
@dataclass
class FFTClassifiedRow:
    """Result of FFT-classifying one |Ci| row."""
    i_row:     int
    c_sym:     np.ndarray       # raw symmetric part of the row
    c_asym:    np.ndarray       # raw asymmetric part of the row
    part_sym:  FFTPart      # compressed S (top-M kept)
    part_asym: FFTPart      # compressed A (top-M kept)
    rec_sym:   np.ndarray       # reconstructed S (inverse FFT)
    rec_asym:  np.ndarray       # reconstructed A (inverse FFT)
    win_sym:   np.ndarray       # center window of rec_sym
    win_asym:  np.ndarray       # center window of rec_asym
    n_sym:     int
    n_asym:    int
    S_sym:     float            # supertrace of the S window
    S_asym:    float            # supertrace of the A window


# ============================================================
#  Extended solution
# ============================================================
@dataclass
class FFTDualSolution(DualSolution):
    row_classified:  List[FFTClassifiedRow] = field(default_factory=list)
    pivot_history:   List[Dict] = field(default_factory=list)
    pivot_mode:      str = 'supertrace'
    K_ci:            int = 0


# ============================================================
#  FFTDualLinearSolver
# ============================================================
class FFTDualLinearSolver(DualLinearSolver):
    """
    FFT extension of DualLinearSolver.

    Rows are |Ci| arrays of length 2·K_ci + 1.  Each row is split
    into S/A, FFT-compressed with top-M magnitude selection, inverse
    FFT'd, and the center window becomes A_re / A_im of the complex
    embedding.

    Pivot for Gaussian elimination is chosen by
        |S_row| = |Σ_k (−1)^k row_k|
    i.e. the supertrace of the row (alternating sum).  Magnitude
    acts as tie-breaker and safety fallback.
    """

    def __init__(self,
                 K_ci: int,
                 M_sym: Optional[int] = None,
                 M_asym: Optional[int] = None,
                 regularize: float = 1e-12,
                 split_mode: str = 'per_row',
                 use_conv_in_reconstruction: bool = False,
                 pivot_mode: str = 'supertrace'):
        super().__init__(
            K=K_ci,
            regularize=regularize,
            use_coeff_classify=True,
            split_mode=split_mode,
        )
        self.K_ci = int(K_ci)
        self.M_sym = M_sym
        self.M_asym = M_asym
        self.use_conv_in_reconstruction = bool(use_conv_in_reconstruction)
        if pivot_mode not in ('supertrace', 'magnitude'):
            raise ValueError(
                f"pivot_mode must be 'supertrace' or 'magnitude'; "
                f"got {pivot_mode!r}")
        self.pivot_mode = pivot_mode

        # ---- list-based Möbius sieve from memory module ----
        self.mu_mem = mobius_sieve(self.K_ci)

    # --------------------------------------------------------
    #  Row → |Ci| array (pure-symmetric embedding)
    # --------------------------------------------------------
    def row_to_ci(self, row: np.ndarray) -> np.ndarray:
        """
        Embed a length-n row into a |Ci| array of length 2·K_ci+1.

        The row's coefficients are mirrored symmetrically about the
        center, so the |Ci| array is pure-symmetric (A part = 0).
        """
        n = len(row)
        K = self.K_ci
        if n > 2 * K + 1:
            raise ValueError(f"row length {n} exceeds 2K+1 = {2*K+1}")
        ci = np.zeros(2 * K + 1)
        start = K - n // 2
        end = start + n
        ci[start:end] = row
        # mirror (except the center itself)
        for j in range(n):
            pos = start + j
            neg = 2 * K - pos
            if pos != K:
                ci[neg] = row[j]
        return ci

    # --------------------------------------------------------
    #  Row → |Ci| array (mixed symmetric + asymmetric)
    # --------------------------------------------------------
    def row_pair_to_ci(self,
                       v_sym: np.ndarray,
                       v_asym: Optional[np.ndarray] = None,
                       ) -> np.ndarray:
        """
        Build a |Ci| array from explicit symmetric and asymmetric
        vector parts of length n.

            ci[K + (j − n//2)] = v_sym[j] + v_asym[j]
            ci[K − (j − n//2)] = v_sym[j] − v_asym[j]
            (center handled specially: ci[K] = v_sym[center])
        """
        n = len(v_sym)
        K = self.K_ci
        if n > 2 * K + 1:
            raise ValueError(f"vector length {n} exceeds 2K+1")
        if v_asym is None:
            v_asym = np.zeros_like(v_sym)
        if len(v_asym) != n:
            raise ValueError("v_sym and v_asym must be same length")
        ci = np.zeros(2 * K + 1)
        for j in range(n):
            offset = j - n // 2
            if offset == 0:
                ci[K] = v_sym[j]
            else:
                ci[K + offset] = v_sym[j] + v_asym[j]
                ci[K - offset] = v_sym[j] - v_asym[j]
        return ci

    # --------------------------------------------------------
    #  FFT classify one row
    # --------------------------------------------------------
    def classify_row_fft(self, i_row: int,
                         ci_row: np.ndarray,
                         n_target: int) -> FFTClassifiedRow:
        """
        Split one |Ci| row into S/A, FFT compress each (top-M), inverse
        FFT, and extract the center window of length n_target.
        """
        K = self.K_ci
        if len(ci_row) != 2 * K + 1:
            raise ValueError(
                f"ci_row length {len(ci_row)} != 2K+1 = {2*K+1}")

        # complex → real pipeline: process re/im separately
        if np.iscomplexobj(ci_row):
            re_part = self._classify_real(ci_row.real, n_target)
            im_part = self._classify_real(ci_row.imag, n_target)
            return self._merge_complex_classified(
                i_row, re_part, im_part)
        return self._classify_real(ci_row, n_target, i_row=i_row)

    def _classify_real(self, ci_row: np.ndarray, n_target: int,
                       i_row: int = 0) -> FFTClassifiedRow:
        K = self.K_ci
        c_sym, c_asym, n_sym, n_asym = split_symmetric_asymmetric(
            ci_row, K)

        part_sym  = fft_compress_part(c_sym,  self.M_sym)
        part_asym = fft_compress_part(c_asym, self.M_asym)

        rec_sym  = fft_decompress_part(part_sym)
        rec_asym = fft_decompress_part(part_asym)

        if self.use_conv_in_reconstruction:
            rec_sym = conv_exp_kernel(rec_sym)

        start = K - n_target // 2
        end   = start + n_target
        win_sym  = rec_sym[start:end].copy()
        win_asym = rec_asym[start:end].copy()

        S_sym  = _supertrace(win_sym)
        S_asym = _supertrace(win_asym)

        return FFTClassifiedRow(
            i_row=i_row,
            c_sym=c_sym, c_asym=c_asym,
            part_sym=part_sym, part_asym=part_asym,
            rec_sym=rec_sym, rec_asym=rec_asym,
            win_sym=win_sym, win_asym=win_asym,
            n_sym=n_sym, n_asym=n_asym,
            S_sym=float(S_sym), S_asym=float(S_asym),
        )

    def _merge_complex_classified(self, i_row: int,
                                  re_part: FFTClassifiedRow,
                                  im_part: FFTClassifiedRow
                                  ) -> FFTClassifiedRow:
        """Merge real and imaginary classifications into one row."""
        def cplx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return a + 1j * b

        win_sym  = cplx(re_part.win_sym,  im_part.win_sym)
        win_asym = cplx(re_part.win_asym, im_part.win_asym)
        rec_sym  = cplx(re_part.rec_sym,  im_part.rec_sym)
        rec_asym = cplx(re_part.rec_asym, im_part.rec_asym)

        return FFTClassifiedRow(
            i_row=i_row,
            c_sym=cplx(re_part.c_sym,  im_part.c_sym),
            c_asym=cplx(re_part.c_asym, im_part.c_asym),
            part_sym=re_part.part_sym,    # keep real-part FFT as canonical
            part_asym=re_part.part_asym,
            rec_sym=rec_sym, rec_asym=rec_asym,
            win_sym=win_sym, win_asym=win_asym,
            n_sym=re_part.n_sym + im_part.n_sym,
            n_asym=re_part.n_asym + im_part.n_asym,
            S_sym=float(re_part.S_sym + im_part.S_sym),
            S_asym=float(re_part.S_asym + im_part.S_asym),
        )

    # --------------------------------------------------------
    #  Supertrace-pivot Gaussian elimination
    # --------------------------------------------------------
    def gaussian_elimination_supertrace(self,
                                        A_c: np.ndarray,
                                        b_c: np.ndarray
                                        ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Gaussian elimination where the pivot at column k is the row
        (among k..n) with the largest |supertrace|, supertrace being
        the alternating sum  S = Σ_j (−1)^j v_j  over the row's
        length-n real and imaginary parts.

        Magnitude is used as tie-breaker and as a safety fallback when
        the supertrace-selected pivot is numerically degenerate.
        """
        n = A_c.shape[0]
        M = np.zeros((n, n + 1), dtype=np.complex128)
        M[:, :n] = A_c
        M[:, n] = b_c

        pivot_history: List[Dict] = []

        for k in range(n):
            # score each candidate row k..n-1 by |supertrace|
            scored = []
            for i in range(k, n):
                row = M[i, :n]
                S_re = _supertrace(row.real)
                S_im = _supertrace(row.imag)
                S_mag = math.hypot(S_re, S_im)
                col_mag = abs(M[i, k])
                scored.append((S_mag, col_mag, i))

            if self.pivot_mode == 'supertrace':
                # sort by supertrace magnitude desc, then by column magnitude
                scored.sort(key=lambda t: (-t[0], -t[1]))
            else:
                # pure magnitude pivot
                scored.sort(key=lambda t: (-t[1], -t[0]))

            st_pick, mag_at_pick, p = scored[0]

            # safety fallback: if pivot value itself is too small,
            # take the max-magnitude candidate in the column
            if abs(M[p, k]) < self.regularize:
                col = np.abs(M[k:, k])
                p = k + int(np.argmax(col))
                st_pick = 0.0
                mag_at_pick = float(abs(M[p, k]))

            if abs(M[p, k]) < self.regularize:
                raise np.linalg.LinAlgError(
                    f"singular at column {k} "
                    f"(|pivot|={abs(M[p, k]):.2e})")

            if p != k:
                M[[k, p]] = M[[p, k]]

            pivot_history.append(dict(
                k=k,
                pivot_row=p,
                supertrace=float(st_pick),
                pivot_mag=float(abs(M[k, k])),
            ))

            for i in range(k + 1, n):
                f = M[i, k] / M[k, k]
                M[i, k:] -= f * M[k, k:]

        # back substitution
        x = np.zeros(n, dtype=np.complex128)
        for i in range(n - 1, -1, -1):
            rhs = M[i, n] - M[i, i + 1:n] @ x[i + 1:]
            x[i] = rhs / M[i, i]

        return x, pivot_history

    # --------------------------------------------------------
    #  End-to-end solve from |Ci| rows
    # --------------------------------------------------------
    def solve_ci(self,
                 ci_rows: List[np.ndarray],
                 b_ci: np.ndarray,
                 method: str = 'fft_gauss',
                 method_fallback: str = 'gauss') -> FFTDualSolution:
        """
        Full pipeline:
            ci_rows (list of |Ci| arrays) → per-row FFT classify
                                          → A_re, A_im, b_re, b_im
                                          → complex embed
                                          → solve (fft_gauss / gauss / numpy)

        Parameters
        ----------
        ci_rows : list of length-n rows, each of length 2·K_ci+1
        b_ci    : rhs |Ci| array of length 2·K_ci+1
        method  : 'fft_gauss' (supertrace pivot)
                  'gauss'     (parent magnitude pivot)
                  'numpy'
                  'tridiagonal'
        """
        t0 = time.perf_counter()
        n = len(ci_rows)
        K = self.K_ci

        if n == 0:
            raise ValueError("no rows provided")
        if len(b_ci) != 2 * K + 1:
            raise ValueError(
                f"b_ci length {len(b_ci)} != 2K+1 = {2*K+1}")
        if n > 2 * K + 1:
            raise ValueError(
                f"n={n} exceeds |Ci| length 2K+1 = {2*K+1}")

        # ---- 1. per-row FFT classification ----
        row_classified = [
            self.classify_row_fft(i, ci_rows[i], n)
            for i in range(n)
        ]

        # ---- 2. build A_re, A_im from S / A windows ----
        A_re = np.zeros((n, n), dtype=np.float64)
        A_im = np.zeros((n, n), dtype=np.float64)
        for i, rc in enumerate(row_classified):
            A_re[i, :] = np.real(rc.win_sym)
            A_im[i, :] = np.real(rc.win_asym)

        # ---- 3. classify rhs ----
        rc_b = self.classify_row_fft(-1, b_ci, n)
        b_re = np.real(rc_b.win_sym).astype(np.float64).copy()
        b_im = np.real(rc_b.win_asym).astype(np.float64).copy()

        # ---- 4. complex embed ----
        A_c = A_re + 1j * A_im
        b_c = b_re + 1j * b_im

        # ---- 5. solve ----
        actual_method = method
        pivot_history: List[Dict] = []
        used_structured = False

        try:
            if method == 'fft_gauss':
                x, pivot_history = self.gaussian_elimination_supertrace(
                    A_c, b_c)
            elif method == 'gauss':
                x = self.gaussian_elimination(A_c, b_c)
            elif method == 'numpy':
                x = np.linalg.solve(A_c, b_c)
            elif method == 'tridiagonal':
                if self._is_tridiagonal(A_c):
                    x = self._solve_tridiagonal(A_c, b_c)
                    used_structured = True
                else:
                    x, pivot_history = self.gaussian_elimination_supertrace(
                        A_c, b_c)
                    actual_method = 'fft_gauss'
            else:
                raise ValueError(f"unknown method {method!r}")
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A_c, b_c, rcond=None)[0]
            actual_method = 'lstsq'

        # ---- 6. residuals + partition diagnostics ----
        classes = ['real'] * n          # placeholder; full classify done below
        # classify rows via parent Möbius rule on real part of A_re
        mu_parent = self._get_mu(A_re)
        classes = []
        sf_counts = []
        comp_counts = []
        for i in range(n):
            side, n_sf, n_cp = self.classify_row(i, A_re, mu_parent)
            classes.append(side)
            sf_counts.append(n_sf)
            comp_counts.append(n_cp)

        residuals = dict(
            full=float(np.linalg.norm(A_c @ x - b_c)),
            real=float(np.linalg.norm(A_re @ x - b_re)),
            imag=float(np.linalg.norm(A_im @ x - b_im)),
        )

        elapsed_ms = (time.perf_counter() - t0) * 1e3

        return FFTDualSolution(
            x=x,
            x_re=np.real(x).copy(),
            x_im=np.imag(x).copy(),
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
            split_mode=f"fft/{self.split_mode}",
            elapsed_ms=elapsed_ms,
            row_classified=row_classified,
            pivot_history=pivot_history,
            pivot_mode=self.pivot_mode,
            K_ci=K,
        )

    # --------------------------------------------------------
    #  Convenience: solve traditional (n, n) matrix with FFT pass
    # --------------------------------------------------------
    def solve_fft(self, A: np.ndarray, b: np.ndarray,
                  method: str = 'fft_gauss') -> FFTDualSolution:
        """
        Accept a traditional (n, n) matrix, lift each row to a
        pure-symmetric |Ci| array, run the FFT pipeline, and solve.
        """
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"A must be square, got {A.shape}")
        if b.shape != (n,):
            raise ValueError(f"b must have shape ({n},), got {b.shape}")

        ci_rows = [self.row_to_ci(A[i, :]) for i in range(n)]
        b_ci    = self.row_to_ci(b)
        return self.solve_ci(ci_rows, b_ci, method=method)

    # --------------------------------------------------------
    #  Override solve to dispatch to FFT path when requested
    # --------------------------------------------------------
    def solve(self, A, b, method: str = 'gauss',
              structured: bool = False) -> DualSolution:
        if method in ('fft_gauss', 'fft'):
            return self.solve_fft(A, b, method='fft_gauss')
        return super().solve(A, b, method=method, structured=structured)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("FFT extension  ·  |Ci| rows  ·  supertrace-pivot Gaussian")
    print("=" * 78)
    print(f"  α           = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP      = α/0.3628    = {A_STEP:.6f}")
    print(f"  6/π²                       = {DENSITY:.6f}")
    print()

    K_ci = 64                # |Ci| half-size  → length 129
    n    = 8                 # matrix size    → 8 |Ci| rows
    rng  = np.random.default_rng(42)

    solver = FFTDualLinearSolver(
        K_ci=K_ci,
        M_sym=None,           # auto top-M from supertrace
        M_asym=None,
        regularize=1e-12,
        split_mode='per_row',
        pivot_mode='supertrace',
    )
    print(f"--- solver ---")
    print(f"  K_ci                    = {K_ci}  (|Ci| length {2*K_ci+1})")
    print(f"  n (rows)                = {n}")
    print(f"  pivot_mode              = {solver.pivot_mode}")
    print(f"  μ_mem density           = "
          f"{sum(1 for k in range(1, K_ci+1) if solver.mu_mem[k] != 0) / K_ci:.4f}")
    print()

    # ============================================================
    #  1 · build |Ci| rows with known solution
    # ============================================================
    print("=" * 78)
    print("1  ·  build |Ci| rows from a known x_true")
    print("=" * 78)
    x_true = rng.standard_normal(n)
    # each row: symmetric part = A[i,:],  asymmetric part = small noise
    A_true = rng.standard_normal((n, n))
    asym_noise = 0.3 * rng.standard_normal((n, n))

    ci_rows = []
    for i in range(n):
        ci = solver.row_pair_to_ci(
            v_sym=A_true[i, :],
            v_asym=asym_noise[i, :],
        )
        ci_rows.append(ci)

    # rhs = A_true · x_true,  embedded as |Ci| with small asym
    b_vec = A_true @ x_true
    b_ci  = solver.row_pair_to_ci(
        v_sym=b_vec,
        v_asym=0.1 * rng.standard_normal(n),
    )

    print(f"  x_true                  = {x_true.round(4)}")
    print(f"  |Ci| row length         = {len(ci_rows[0])}")
    print(f"  b_ci length             = {len(b_ci)}")
    print()

    # ============================================================
    #  2 · per-row FFT classification
    # ============================================================
    print("=" * 78)
    print("2  ·  per-row FFT classification  ·  S / A split + top-M")
    print("=" * 78)
    rc_rows = [solver.classify_row_fft(i, ci_rows[i], n)
               for i in range(n)]

    print(f"  {'row':>3s}  {'n_sym':>6s}  {'n_asym':>7s}  "
          f"{'M_sym':>6s}  {'M_asym':>7s}  "
          f"{'S_sym':>10s}  {'S_asym':>10s}")
    print("  " + "-" * 68)
    for rc in rc_rows:
        print(f"  {rc.i_row:>3d}  {rc.n_sym:>6d}  {rc.n_asym:>7d}  "
              f"{rc.part_sym.n_kept:>6d}  {rc.part_asym.n_kept:>7d}  "
              f"{rc.S_sym:>+10.4f}  {rc.S_asym:>+10.4f}")
    print()
    print(f"  note: M_sym/M_asym auto-chosen from |S| when M=None")
    print()

    # ============================================================
    #  3 · solve with three methods
    # ============================================================
    print("=" * 78)
    print("3  ·  solve  ·  fft_gauss vs gauss vs numpy")
    print("=" * 78)
    sols = {}
    for method in ('fft_gauss', 'gauss', 'numpy'):
        t0 = time.perf_counter()
        sol = solver.solve_ci(ci_rows, b_ci, method=method)
        t1 = time.perf_counter()
        sols[method] = sol
        print(f"  --- {method} ---")
        print(f"    method          = {sol.method}")
        print(f"    x               = {sol.x.round(4)}")
        print(f"    ‖residual full‖ = {sol.residuals['full']:.4e}")
        print(f"    ‖residual real‖ = {sol.residuals['real']:.4e}")
        print(f"    ‖residual imag‖ = {sol.residuals['imag']:.4e}")
        print(f"    elapsed         = {sol.elapsed_ms:.3f} ms")
        print()

    # ---- compare x across methods ----
    print(f"  ‖x(fft_gauss) − x(gauss)‖  = "
          f"{np.linalg.norm(sols['fft_gauss'].x - sols['gauss'].x):.2e}")
    print(f"  ‖x(fft_gauss) − x(numpy)‖  = "
          f"{np.linalg.norm(sols['fft_gauss'].x - sols['numpy'].x):.2e}")
    print()

    # ============================================================
    #  4 · supertrace pivot history
    # ============================================================
    print("=" * 78)
    print("4  ·  supertrace pivot history  (fft_gauss)")
    print("=" * 78)
    hist = sols['fft_gauss'].pivot_history
    print(f"  {'k':>3s}  {'pivot_row':>10s}  "
          f"{'supertrace':>12s}  {'|M[k,k]|':>12s}")
    print("  " + "-" * 44)
    for rec in hist:
        print(f"  {rec['k']:>3d}  {rec['pivot_row']:>10d}  "
              f"{rec['supertrace']:>+12.4f}  "
              f"{rec['pivot_mag']:>12.4f}")
    print()

    # ============================================================
    #  5 · traditional (n, n) matrix path
    # ============================================================
    print("=" * 78)
    print("5  ·  solve_fft  ·  traditional (n, n) matrix")
    print("=" * 78)
    A_dense = rng.standard_normal((n, n))
    b_dense = rng.standard_normal(n)
    sol_fft_dense = solver.solve_fft(
        A_dense, b_dense, method='fft_gauss')
    sol_np = np.linalg.solve(A_dense, b_dense)
    print(f"  matrix shape        = {A_dense.shape}")
    print(f"  method              = {sol_fft_dense.method}")
    print(f"  ‖x_fft − x_numpy‖   = "
          f"{np.linalg.norm(sol_fft_dense.x - sol_np):.4e}")
    print(f"  ‖residual full‖     = "
          f"{sol_fft_dense.residuals['full']:.4e}")
    print()

    # ============================================================
    #  6 · inherited features still work
    # ============================================================
    print("=" * 78)
    print("6  ·  inherited features  ·  tridiagonal + build_ci_buffer")
    print("=" * 78)
    n_tri = 32
    A_tri = np.zeros((n_tri, n_tri))
    np.fill_diagonal(A_tri, 4.0)
    for i in range(n_tri - 1):
        A_tri[i, i+1] = -1.0
        A_tri[i+1, i] = -1.0
    b_tri = np.ones(n_tri)

    sol_tri = solver.solve(A_tri, b_tri, structured=True)
    sol_np_tri = np.linalg.solve(A_tri, b_tri)
    print(f"  tridiagonal path     = {sol_tri.method}")
    print(f"  ‖x_tri − x_numpy‖    = "
          f"{np.linalg.norm(sol_tri.x.real - sol_np_tri):.2e}")

    buf = solver.build_ci_buffer(sols['fft_gauss'])
    print(f"  |Ci| buffer K        = {buf.K}")
    print(f"  |Ci| supertrace      = {buf.supertrace():+.6f}")
    print(f"  |Ci| DC              = {buf.dc():+.6f}")
    print()

    # ============================================================
    #  7 · what's new vs parent
    # ============================================================
    print("=" * 78)
    print("7  ·  what this extends")
    print("=" * 78)
    print(f"  rows are |Ci| arrays       : yes (length 2K+1 = {2*K_ci+1})")
    print(f"  per-row FFT classify       : split S/A + top-M per row")
    print(f"  S/A separation             : split_symmetric_asymmetric")
    print(f"  top-M per row magnitude    : fft_compress_part  (auto from |S|)")
    print(f"  reconstruction             : fft_decompress_part")
    print(f"  complex embed              : A_c = A_re + i·A_im")
    print(f"                              (A_re ← S, A_im ← A)")
    print(f"  Gaussian pivot criterion   : |supertrace(row)| "
          f"(alternating sum)")
    print(f"  safety fallback            : magnitude pivot when |S| degenerate")
    print(f"  traditional solve          : inherited (method='gauss' | 'numpy')")
    print(f"  tridiagonal fast path      : inherited (structured=True)")
    print(f"  per-row Möbius classify    : inherited (classify_row)")
    print(f"  separate_square_free       : inherited (unchanged)")
    print(f"  build_ci_buffer            : inherited (unchanged)")
    print()

    # ============================================================
    #  8 · complexity
    # ============================================================
    print("=" * 78)
    print("8  ·  complexity")
    print("=" * 78)
    print(f"  row_to_ci / row_pair_to_ci    O(n)")
    print(f"  split_symmetric_asymmetric    O(K_ci)")
    print(f"  fft_compress_part             O(K_ci log K_ci)   per row")
    print(f"  fft_decompress_part           O(K_ci log K_ci)   per row")
    print(f"  per-row classify (n rows)     O(n · K_ci log K_ci)")
    print(f"  supertrace pivot (per column) O(n²)              row scan")
    print(f"  Gaussian elimination          O(n³)              general")
    print(f"  tridiagonal solve             O(n)               fast path")
    print()
    print(f"  Dominant: O(n · K_ci log K_ci) FFT  +  O(n³) Gaussian")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()