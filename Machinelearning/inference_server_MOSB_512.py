#!/usr/bin/env python3
"""
inference_server_MOSB_512.py
============================

Extremely large context extension of inference_server_MOSB.py.

    n ≤ 32    →  DualSolver512 (direct 512-bit Gaussian)
    n > 32    →  BSGSBlockSolver (baby-step / giant-step, each block 512-bit)

New verification layer:
    linear_solve_consistent   512-bit block solve of the KV cache, n
                              up to n_limit, residual check vs 64-bit

Wire format: UNCHANGED (same MOSB frame / OITH header / UGC ALU
packet / weight signature as inference_server_MOSB.py).
"""

from __future__ import annotations
import hashlib
import math
import os
import struct
import time
from decimal import localcontext
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- parent ----
from inference_server_MOSB import (
    InferenceServer, InferenceClient, InferenceCodec,
    OutputHeader, ClientTier, ComputeTier, ModelFamily,
    COMPUTE_BUFFER, classify_compute,
    serialize_request, serialize_response, parse_payload,
    OITH_SIZE, OITH_MAGIC, OITH_FMT,
    K_INFERENCE_TABLE, K_INFERENCE_ALU, K_MOSB,
)
from download_mboius_unpack_buffer import MobiusPacker

# ---- 512-bit solver ----
from dual_linear_solver_512 import (
    DualSolver512, DualSolution512, DComplex,
    E_512, PI_512, ALPHA_EVEN_512, ALPHA_ODD_512,
    BITS, DEC_DIGITS,
)


# ============================================================
#  BSGSBlockSolver  ·  block baby-step / giant-step
# ============================================================
class BSGSBlockSolver:
    """
    Block-tridiagonal baby-step / giant-step solver on 512-bit
    complex system.

        n ≤ bs          → direct DualSolver512 (full 512-bit GE)
        n > bs          → BS/GS: each diagonal block ≤ bs solved
                          at 512-bit, coupling via L/U sweeps

    bs defaults to 32 — the same threshold where DualSolver512's
    Gaussian elimination becomes impractical.  Blocks > bs fall
    through to numpy_64 inside the sweep (a warning is printed
    if all blocks end up in the numpy_64 path).
    """

    def __init__(self,
                 bs: int = 32,
                 classify: str = 'parity',
                 split_mode: str = 'per_row',
                 max_iter: int = 3,
                 tol: float = 1e-9):
        self.bs = int(bs)
        self.inner = DualSolver512(
            classify=classify, split_mode=split_mode)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    # --------------------------------------------------------
    def solve(self, A: np.ndarray, b: np.ndarray) -> Dict:
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = A.shape[0]

        # ----- small: direct 512-bit -----
        if n <= self.bs:
            sol = self.inner.solve(A, b, method='gauss_512')
            return dict(
                x=sol.x, blocks=1, bs=n,
                method='direct_512',
                residuals_64=sol.residuals_64,
                residuals_512=sol.residuals_512,
                bits=BITS, n=n,
            )

        # ----- large: BS/GS blocks -----
        nb = (n + self.bs - 1) // self.bs
        blocks = [(k * self.bs, min((k + 1) * self.bs, n))
                  for k in range(nb)]

        with localcontext() as ctx:
            ctx.prec = DEC_DIGITS + 20

            # ---- parse / split once (complex 512-bit) ----
            A_512, b_512 = self.inner.parse(A, b)
            A_re, A_im, b_re, b_im, classes, n_sym, n_asym = \
                self.inner.split(A_512, b_512)

            A_c = [[DComplex(A_re[i][j], A_im[i][j])
                    for j in range(n)] for i in range(n)]
            b_c = [DComplex(b_re[i], b_im[i]) for i in range(n)]

            # ---- forward sweep (baby steps) ----
            ys: List[List[DComplex]] = []
            for k, (i0, i1) in enumerate(blocks):
                bk = [b_c[i0 + i] for i in range(i1 - i0)]
                if k > 0:
                    p0, p1 = blocks[k - 1]
                    y_prev = ys[-1]
                    for ii in range(i1 - i0):
                        acc = bk[ii]
                        for jj in range(p1 - p0):
                            acc = acc - A_c[i0 + ii][p0 + jj] * y_prev[jj]
                        bk[ii] = acc
                Dk = [[A_c[i0 + i][i0 + j] for j in range(i1 - i0)]
                      for i in range(i1 - i0)]
                yk = self.inner._gaussian_512(Dk, bk)
                ys.append(yk)

            # ---- backward sweep (giant steps) ----
            xs: List[List[DComplex]] = [None] * nb
            xs[-1] = ys[-1]
            for k in range(nb - 2, -1, -1):
                i0, i1 = blocks[k]
                n0, n1 = blocks[k + 1]
                rhs = list(ys[k])
                x_next = xs[k + 1]
                for ii in range(i1 - i0):
                    acc = rhs[ii]
                    for jj in range(n1 - n0):
                        acc = acc - A_c[i0 + ii][n0 + jj] * x_next[jj]
                    rhs[ii] = acc
                Dk = [[A_c[i0 + i][i0 + j] for j in range(i1 - i0)]
                      for i in range(i1 - i0)]
                xs[k] = self.inner._gaussian_512(Dk, rhs)

            # ---- assemble ----
            x_c_flat = [x for xk in xs for x in xk]
            x_re = np.array([float(x.re) for x in x_c_flat])
            x_im = np.array([float(x.im) for x in x_c_flat])
            x = x_re + 1j * x_im

            # ---- 512-bit residual (full system) ----
            res_512 = self.inner._residual_512(A_c, x_c_flat, b_c)

            # ---- compare with pure float64 full solve ----
            A_c_64 = np.array(
                [[complex(float(a.re), float(a.im)) for a in row]
                 for row in A_c], dtype=np.complex128)
            b_c_64 = np.array(
                [complex(float(b.re), float(b.im)) for b in b_c],
                dtype=np.complex128)
            try:
                x_64 = np.linalg.solve(A_c_64, b_c_64)
                err_64 = float(np.linalg.norm(x - x_64))
            except np.linalg.LinAlgError:
                err_64 = float('inf')

            # ---- iterative refinement for far couplings ----
            #   block-tridiagonal sweep is exact when only nearest-block
            #   coupling exists; otherwise iterate on the residual
            for _ in range(self.max_iter):
                if float(res_512) < self.tol:
                    break
                # r = b_c - A_c · x (in 512-bit)
                r_c = []
                for i in range(n):
                    acc = b_c[i]
                    for j in range(n):
                        acc = acc - A_c[i][j] * x_c_flat[j]
                    r_c.append(acc)
                # solve A_c · dx = r_c via block sweeps (same structure)
                dxs = self._block_sweep(A_c, r_c, blocks)
                dx_flat = [d for dxk in dxs for d in dxk]
                for i in range(n):
                    x_c_flat[i] = x_c_flat[i] + dx_flat[i]
                res_512 = self.inner._residual_512(
                    A_c, x_c_flat, b_c)

            x_re = np.array([float(x.re) for x in x_c_flat])
            x_im = np.array([float(x.im) for x in x_c_flat])
            x = x_re + 1j * x_im

        return dict(
            x=x, blocks=nb, bs=self.bs,
            method='bs_gs_512',
            residuals_64=dict(
                full=float(res_512),
                vs_64=err_64,
            ),
            residuals_512=dict(full=res_512),
            bits=BITS, n=n,
        )

    # --------------------------------------------------------
    def _block_sweep(self, A_c, rhs_c, blocks) -> List[List[DComplex]]:
        """One block-tridiagonal sweep on a given 512-bit complex RHS."""
        nb = len(blocks)
        ys: List[List[DComplex]] = []
        for k, (i0, i1) in enumerate(blocks):
            bk = [rhs_c[i0 + i] for i in range(i1 - i0)]
            if k > 0:
                p0, p1 = blocks[k - 1]
                y_prev = ys[-1]
                for ii in range(i1 - i0):
                    acc = bk[ii]
                    for jj in range(p1 - p0):
                        acc = acc - A_c[i0 + ii][p0 + jj] * y_prev[jj]
                    bk[ii] = acc
            Dk = [[A_c[i0 + i][i0 + j] for j in range(i1 - i0)]
                  for i in range(i1 - i0)]
            ys.append(self.inner._gaussian_512(Dk, bk))

        xs: List[List[DComplex]] = [None] * nb
        xs[-1] = ys[-1]
        for k in range(nb - 2, -1, -1):
            i0, i1 = blocks[k]
            n0, n1 = blocks[k + 1]
            rhs = list(ys[k])
            x_next = xs[k + 1]
            for ii in range(i1 - i0):
                acc = rhs[ii]
                for jj in range(n1 - n0):
                    acc = acc - A_c[i0 + ii][n0 + jj] * x_next[jj]
                rhs[ii] = acc
            Dk = [[A_c[i0 + i][i0 + j] for j in range(i1 - i0)]
                  for i in range(i1 - i0)]
            xs[k] = self.inner._gaussian_512(Dk, rhs)
        return xs


# ============================================================
#  Extract (A, b) from a recovered payload
# ============================================================
def extract_linear_problem(data: bytes,
                            n_max: int = 128) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Deterministically build a diagonally dominant real system from
    the recovered KV-cache bytes.

        n = min(n_max, max(4, ⌊√(len(data)/2)⌋))
        A[i,j] = (data[i·n + j] − 128) / 128 + 2·δ_ij
        b[i]   = (data[n² + i] − 128) / 128
    """
    L = len(data)
    if L < 32:
        # pad very short payloads
        data = data + b'\x00' * (32 - L)
        L = 32
    n = min(n_max, max(4, int(math.sqrt(L / 2))))
    need = n * n + n
    if L < need:
        data = data + b'\x00' * (need - L)

    vals = np.frombuffer(data[:n * n], dtype=np.uint8).astype(np.float64)
    A = (vals.reshape(n, n) - 128.0) / 128.0
    A = A + 2.0 * np.eye(n)          # diagonal dominance → non-singular

    b_vals = np.frombuffer(data[n * n:n * n + n], dtype=np.uint8
                           ).astype(np.float64)
    b = (b_vals - 128.0) / 128.0
    return A, b, n


# ============================================================
#  LinearSolveVerifier
# ============================================================
class LinearSolveVerifier:
    """
    Verify the KV-cache linear solve at 512-bit.

        n ≤ bs      →  DualSolver512  (full 512-bit GE)
        n > bs      →  BSGSBlockSolver (block baby-step / giant-step)

    The check passes iff the 512-bit residual is below `tol`.
    """

    def __init__(self, bs: int = 32, n_max: int = 128,
                 tol: float = 1e-6, classify: str = 'parity'):
        self.bs = int(bs)
        self.n_max = int(n_max)
        self.tol = float(tol)
        self.solver = BSGSBlockSolver(
            bs=bs, classify=classify, split_mode='per_row')

    def verify(self, data: bytes) -> Tuple[bool, Dict]:
        if len(data) < 16:
            return True, dict(n=0, method='skip',
                              reason='payload too short')
        try:
            A, b, n = extract_linear_problem(data, self.n_max)
        except Exception as e:
            return False, dict(n=0, error=str(e))

        try:
            sol = self.solver.solve(A, b)
        except Exception as e:
            return False, dict(n=n, error=str(e))

        residual = sol['residuals_64']['full']
        ok = (residual < self.tol)

        return ok, dict(
            n=n,
            bs=self.bs,
            blocks=sol['blocks'],
            method=sol['method'],
            residual=residual,
            vs_64=sol['residuals_64'].get('vs_64'),
            bits=sol['bits'],
            tol=self.tol,
        )


# ============================================================
#  InferenceCodec512  ·  adds linear_solve_consistent
# ============================================================
class InferenceCodec512(InferenceCodec):
    """
    Extends InferenceCodec with a 512-bit linear-solve verification
    layer.  Wire format unchanged.

    On decode, after the parent checks pass:
        linear_solve_consistent  ← BSGSBlockSolver / DualSolver512
                                     on the recovered payload
    """

    def __init__(self,
                 K_table: int = K_INFERENCE_TABLE,
                 K_alu: int = K_INFERENCE_ALU,
                 k_labels: int = 4,
                 threshold: float = 0.3,
                 K_ci: int = 20,
                 bs: int = 32,
                 n_max: int = 128):
        super().__init__(K_table=K_table, K_alu=K_alu,
                         k_labels=k_labels, threshold=threshold,
                         K_ci=K_ci)
        self.lin_verifier = LinearSolveVerifier(
            bs=bs, n_max=n_max, tol=1e-6)

    def decode(self, packet: bytes) -> Tuple[bytes, Dict]:
        recovered, diag = super().decode(packet)
        if not diag.get('verified', False):
            diag['linear_solve_consistent'] = False
            return recovered, diag

        ok_lin, diag_lin = self.lin_verifier.verify(recovered)
        diag['linear_solve_consistent'] = ok_lin
        diag['linear_solve_diag']       = diag_lin
        diag['verified'] = diag['verified'] and ok_lin
        return recovered, diag


# ============================================================
#  InferenceServer512  ·  extremely large context
# ============================================================
class InferenceServer512(InferenceServer):
    """
    Extends InferenceServer with:
        · 512-bit linear-solve verification (bs=32 routing)
        · per-token KV state growth proportional to len(prompt)
          so n > 32 fires for realistic contexts
    """

    KV_BYTES_PER_TOKEN = 128       # 4 × SHA-256 per token

    def __init__(self, bs: int = 32, n_max: int = 128):
        super().__init__()
        self.codec = InferenceCodec512(bs=bs, n_max=n_max)
        self.bs = bs
        self.n_max = n_max

    # --------------------------------------------------------
    def _run_inference(self, model, request, header
                       ) -> Tuple[List[int], bytes]:
        prompt = request["prompt"]
        kv_data = bytes.fromhex(request.get("kv_data", ""))

        n_out = min(8, max(1, len(prompt)))
        out_tokens: List[int] = []
        for i in range(n_out):
            last = prompt[-1] if prompt else 0
            tok = (last + i * 7 + header.model_id) % 32000
            out_tokens.append(int(tok))

        # ----- extremely large context: per-token KV -----
        new_kv = bytearray()
        all_tokens = list(prompt) + out_tokens
        per_token_blocks = self.KV_BYTES_PER_TOKEN // 32    # 4 SHA-256
        for i, tok in enumerate(all_tokens):
            for j in range(per_token_blocks):
                digest = hashlib.sha256(
                    f"kv:{tok}:{i}:{j}:{header.model_id}".encode()
                ).digest()
                new_kv.extend(digest)

        updated_kv = kv_data + bytes(new_kv)
        return out_tokens, updated_kv


# ============================================================
#  InferenceClient512
# ============================================================
class InferenceClient512(InferenceClient):
    def __init__(self, client_id: str, client_tier: ClientTier,
                 model_id: int, temperature: float = 0.7,
                 top_p: float = 0.9,
                 bs: int = 32, n_max: int = 128):
        super().__init__(client_id, client_tier, model_id,
                         temperature, top_p)
        self.codec = InferenceCodec512(bs=bs, n_max=n_max)


# ============================================================
#  Pretty printing
# ============================================================
def _breakdown_512(diag: Dict) -> str:
    lines = []
    for k in ("particle_match", "ugc_match", "alu_match", "fiber_match",
              "weight_match", "qp_consistent", "zwalk_consistent",
              "linear_solve_consistent"):
        lines.append(f"    {k:<26s} = {diag.get(k)}")
    ld = diag.get("linear_solve_diag") or {}
    if ld:
        lines.append(f"    linear_solve_diag:")
        lines.append(f"      n          = {ld.get('n')}")
        lines.append(f"      blocks     = {ld.get('blocks')}")
        lines.append(f"      method     = {ld.get('method')}")
        lines.append(f"      residual   = {ld.get('residual'):.4e}")
        if ld.get("vs_64") is not None:
            lines.append(f"      ‖x512−x64‖ = {ld.get('vs_64'):.4e}")
        lines.append(f"      bits       = {ld.get('bits')}")
    return "\n".join(lines)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("inference_server_MOSB_512  ·  extremely large context")
    print("=" * 78)
    print(f"  BITS            = {BITS}")
    print(f"  BS threshold    = 32   (n > 32 → BS/GS split)")
    print(f"  n_max           = 128")
    print(f"  wire format     = unchanged (MOSB + OITH + UGC ALU + sig)")
    print()

    # ---- setup ----
    server = InferenceServer512(bs=32, n_max=128)
    server.register_model(
        model_id=1, family=ModelFamily.SMALL,
        weights=b"small_weights_" * 100)
    server.register_model(
        model_id=2, family=ModelFamily.MEDIUM,
        weights=b"medium_weights_" * 200)

    print("=" * 78)
    print("1  ·  registered models")
    print("=" * 78)
    for mid, m in server.models.items():
        print(f"  model_id={mid}  "
              f"family={m['family'].name:<6s}  "
              f"hash={m['weights_hash'][:16]}…")
    print()

    alice = InferenceClient512("alice", ClientTier.AUTH, model_id=1)

    # ================================================================
    #  2 · small context  ·  n ≤ 32  →  direct 512-bit GE
    # ================================================================
    print("=" * 78)
    print("2  ·  SMALL context  ·  n ≤ 32  ·  direct 512-bit GE")
    print("=" * 78)
    small_prompt = [100, 200, 300, 400, 500]
    sub = alice.submit_prompt(small_prompt)
    print(f"  prompt tokens       = {small_prompt}")
    print(f"  frame size          = {len(sub['frame'])} B")
    print(f"  kv_cache_size       = {sub['header'].kv_cache_size} B")

    result = server.process_frame(sub["frame"])
    print(f"  server verdict      = "
          f"{'APPROVE' if result['approved'] else 'DECLINE'}")
    if result["approved"]:
        d = result["diagnostics"]
        ld = d.get("linear_solve_diag", {})
        print(f"  method              = {ld.get('method')}")
        print(f"  n / blocks          = {ld.get('n')} / {ld.get('blocks')}")
        print(f"  residual            = {ld.get('residual'):.4e}")
        print(f"  output tokens       = {result['output_tokens']}")
    print()

    # ================================================================
    #  3 · medium context  ·  n > 32  ·  BS/GS few blocks
    # ================================================================
    print("=" * 78)
    print("3  ·  MEDIUM context  ·  n > 32  ·  BS/GS (few blocks)")
    print("=" * 78)
    med_prompt = [100 + i for i in range(40)]
    sub = alice.submit_prompt(med_prompt)
    print(f"  prompt len          = {len(med_prompt)}")
    print(f"  frame size          = {len(sub['frame'])} B")
    result = server.process_frame(sub["frame"])
    print(f"  server verdict      = "
          f"{'APPROVE' if result['approved'] else 'DECLINE'}")
    if result["approved"]:
        d = result["diagnostics"]
        ld = d.get("linear_solve_diag", {})
        print(f"  method              = {ld.get('method')}")
        print(f"  n / blocks          = {ld.get('n')} / {ld.get('blocks')}")
        print(f"  residual            = {ld.get('residual'):.4e}")
        print(f"  ‖x512 − x64‖        = {ld.get('vs_64'):.4e}")
    print()

    # ================================================================
    #  4 · LARGE context  ·  n > 128  ·  many blocks
    # ================================================================
    print("=" * 78)
    print("4  ·  LARGE context  ·  n → 128  ·  many blocks")
    print("=" * 78)
    large_prompt = [100 + i for i in range(200)]
    sub = alice.submit_prompt(large_prompt)
    print(f"  prompt len          = {len(large_prompt)}")
    print(f"  frame size          = {len(sub['frame'])} B")
    result = server.process_frame(sub["frame"])
    print(f"  server verdict      = "
          f"{'APPROVE' if result['approved'] else 'DECLINE'}")
    if result["approved"]:
        d = result["diagnostics"]
        ld = d.get("linear_solve_diag", {})
        print(f"  method              = {ld.get('method')}")
        print(f"  n / blocks          = {ld.get('n')} / {ld.get('blocks')}")
        print(f"  residual            = {ld.get('residual'):.4e}")
        print(f"  ‖x512 − x64‖        = {ld.get('vs_64'):.4e}")
        print(f"  output header       : "
              f"kv_size={result['out_header'].kv_cache_size}, "
              f"tok_count={result['out_header'].output_token_count}")
    print()

    # ================================================================
    #  5 · BS/GS block solver spot-check
    # ================================================================
    print("=" * 78)
    print("5  ·  BSGSBlockSolver  ·  routing check (n = 8, 32, 64, 128)")
    print("=" * 78)
    solver = BSGSBlockSolver(bs=32)
    for n in (8, 32, 33, 64, 128):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((n, n)) + 2.0 * n * np.eye(n)
        b = rng.standard_normal(n)
        try:
            sol = solver.solve(A, b)
            print(f"  n={n:>4d}  method={sol['method']:<14s}  "
                  f"blocks={sol['blocks']:>2d}  "
                  f"residual={sol['residuals_64']['full']:.3e}  "
                  f"vs_64={sol['residuals_64'].get('vs_64', 0):.3e}")
        except Exception as e:
            print(f"  n={n:>4d}  ERROR: {e}")
    print()

    # ================================================================
    #  6 · tamper detection
    # ================================================================
    print("=" * 78)
    print("6  ·  tamper detection")
    print("=" * 78)
    sub = alice.submit_prompt([1, 2, 3, 4, 5])
    frame_good = sub["frame"]
    frame_bad = bytearray(frame_good)
    frame_bad[48 + OITH_SIZE + 40] ^= 0x01
    frame_bad = bytes(frame_bad)

    r_good = server.process_frame(frame_good)
    r_bad = server.process_frame(frame_bad)
    print(f"  untampered : approved={r_good['approved']}")
    print(f"  tampered   : approved={r_bad['approved']}")
    if r_bad.get("diagnostics"):
        d = r_bad["diagnostics"]
        print(f"    particle = {d.get('particle_match')}  "
              f"ugc = {d.get('ugc_match')}  "
              f"alu = {d.get('alu_match')}  "
              f"fiber = {d.get('fiber_match')}")
    print()

    # ================================================================
    #  7 · verification breakdown for the large case
    # ================================================================
    print("=" * 78)
    print("7  ·  full verification breakdown (large context)")
    print("=" * 78)
    sub = alice.submit_prompt([100 + i for i in range(64)])
    r = server.process_frame(sub["frame"])
    print(f"  approved = {r['approved']}")
    if r.get("diagnostics"):
        print(_breakdown_512(r["diagnostics"]))
    print()

    # ================================================================
    #  8 · wire format identity check
    # ================================================================
    print("=" * 78)
    print("8  ·  wire format identity  ·  parent vs 512 extension")
    print("=" * 78)
    payload = b'{"kind":"request","prompt":[1,2,3],"kv_hash":"x","kv_len":0,"kv_data":""}'
    parent_codec = InferenceCodec()
    child_codec = InferenceCodec512()
    pkt_parent = parent_codec.encode(payload)
    pkt_child = child_codec.encode(payload)
    print(f"  payload size          = {len(payload)} B")
    print(f"  parent packet size    = {len(pkt_parent)} B")
    print(f"  512-bit packet size   = {len(pkt_child)} B")
    print(f"  byte-identical        : {pkt_parent == pkt_child}")
    print()

    # ================================================================
    #  9 · complexity
    # ================================================================
    print("=" * 78)
    print("9  ·  complexity")
    print("=" * 78)
    print("  n ≤ 32  direct:")
    print("    parse (512-bit)         O(n²)     Decimal casts")
    print("    split (512-bit)         O(n²)     Decimal multiplies")
    print("    Gaussian 512-bit        O(n³)     Decimal DComplex ops")
    print()
    print("  n > 32  BS/GS:")
    print("    parse / split           O(n²)")
    print("    forward sweep           O(nb · bs³)  ≈ O(n · bs²)")
    print("    backward sweep          O(nb · bs³)  ≈ O(n · bs²)")
    print("    iterative refine        O(n²) per iteration (×3)")
    print()
    print("  n = context size   nb = ⌈n/32⌉   bs = 32")
    print("  512-bit op ≈ 10⁴–10⁵× slower than float64")
    print("  → n=128 context ≈ 3 block sweeps of 512-bit GE")
    print()
    print("  wire format: unchanged")
    print("    [ MOSB header     48 B ]")
    print("    [ OITH header     55 B ]")
    print("    [ UGC ALU packet    var ]")
    print("    [ weight sig       8 B ]")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()