#!/usr/bin/env python3
"""
chirp_bridge_pipeline.py
========================

CPU-only extension of the 64-bit |Ci| pipeline.

    Matrix:      M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]
    Trace:       tr M   = x^(i-1) + y^(i-1)
                        = x^-1(cos ln x + i sin ln x)
                        + y^-1(cos ln y + i sin ln y)
    sym  = Re(tr M)    real plane   →  |Ci| positive half  c[K+k]
    asym = Im(tr M)    imag plane   →  |Ci| negative half  c[K−k]

    Discrete sparse plane    O(K log log K)   sieve → live addresses
    Read-only |Ci| buffer    O(K)             sym + asym planes
    Chirp bridge             O(K)             finite step  sym ↔ asym
    Transcendental solve     O(K log K)       Bluestein chirp-z

    ALU finite step     A_STEP = ALPHA / 0.3628 ≈ 6.511
    Universal chirp     w[k]   = gate · cos(A_STEP · k² / K)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA       = 1.0 / (PI - E)          # ≈ 2.362338
ALPHA_ASYM  = 0.3628
A_STEP      = ALPHA / ALPHA_ASYM      # ≈ 6.511
DENSITY     = 6.0 / (PI * PI)         # ≈ 0.607927


# ============================================================
#  1.  Discrete sparse plane  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    """Eratosthenes-based μ sieve — O(K log log K)."""
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    if K >= 1:
        mu[1] = 1

    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(K ** 0.5) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False

    # toggle each prime
    for p in range(2, K + 1):
        if is_prime[p]:
            mu[p::p] = -mu[p::p]
    # zero squares
    for p in range(2, int(K ** 0.5) + 1):
        if is_prime[p]:
            mu[p * p :: p * p] = 0
    return mu


def discrete_sparse_grid(K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (μ, live_addresses).  O(K log log K)."""
    mu = mobius_sieve(K)
    live = np.where(mu[1:] != 0)[0] + 1
    return mu, live


# ============================================================
#  2.  Read-only |Ci| buffer  ·  O(K)
# ============================================================
class ReadOnlyCiBuffer:
    """
    Length-(2K+1) float64 |Ci| array, write-once.

        c[K + k]   k ∈ [1, K]   symmetric  (real plane)  Re(tr M_k)
        c[K − k]   k ∈ [1, K]   asymmetric (imag plane)  Im(tr M_k)
        c[K]                    DC term                  Σ Re(tr M_k)
    """
    __slots__ = ("K", "data", "_sealed")

    def __init__(self, K: int):
        self.K = int(K)
        self.data = np.zeros(2 * self.K + 1, dtype=np.float64)
        self._sealed = False

    # ---- writes (fail once sealed) ----
    def write_sym(self, k: int, v: float) -> None:
        if self._sealed:
            raise RuntimeError("buffer is sealed (read-only)")
        if not 1 <= k <= self.K:
            raise ValueError(f"k={k} out of [1, {self.K}]")
        self.data[self.K + k] = float(v)

    def write_asym(self, k: int, v: float) -> None:
        if self._sealed:
            raise RuntimeError("buffer is sealed (read-only)")
        if not 1 <= k <= self.K:
            raise ValueError(f"k={k} out of [1, {self.K}]")
        self.data[self.K - k] = float(v)

    def write_dc(self, v: float) -> None:
        if self._sealed:
            raise RuntimeError("buffer is sealed (read-only)")
        self.data[self.K] = float(v)

    def seal(self) -> None:
        self._sealed = True

    # ---- O(1) reads ----
    def sym(self, k: int) -> float:
        return float(self.data[self.K + k])

    def asym(self, k: int) -> float:
        return float(self.data[self.K - k])

    def dc(self) -> float:
        return float(self.data[self.K])

    # ---- O(K) bulk reads ----
    def read_planes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sym_plane, asym_plane), each of length K.

            sym_plane[i]  = c[K + (i+1)]  =  Re(tr M_{i+1})
            asym_plane[i] = c[K − (i+1)]  =  Im(tr M_{i+1})
        """
        sym = self.data[self.K + 1 : 2 * self.K + 1].copy()
        asym = self.data[self.K - 1 :: -1].copy()
        return sym, asym

    # ---- O(K) supertrace ----
    def supertrace(self) -> float:
        S = 0.0
        for i, v in enumerate(self.data):
            S += v if (i % 2 == 0) else -v
        return float(S)


# ============================================================
#  3.  ALU chirp + finite-step bridge  ·  O(K)
# ============================================================
def base_chirp(K: int, mu: np.ndarray) -> np.ndarray:
    """w[k] = gate · cos(A_STEP · k² / K).   CPU match of alu_finite_step."""
    gate = (mu[1:K + 1] != 0).astype(np.float64)
    idx = np.arange(K, dtype=np.float64)
    return gate * np.cos(A_STEP * idx * idx / K)


def finite_step_forward(f: np.ndarray, a: float = A_STEP) -> np.ndarray:
    """D⁺_a f(k) = (f(k+1) − f(k)) / a."""
    out = np.zeros_like(f)
    if f.size < 2:
        return out
    out[:-1] = (f[1:] - f[:-1]) / a
    out[-1] = out[-2]
    return out


def finite_step_central(f: np.ndarray, a: float = A_STEP) -> np.ndarray:
    """D_a f(k) = (f(k+1) − f(k−1)) / (2a)."""
    out = np.zeros_like(f)
    if f.size < 3:
        return out
    out[1:-1] = (f[2:] - f[:-2]) / (2 * a)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def chirp_bridge(sym: np.ndarray, asym: np.ndarray, w: np.ndarray,
                 a: float = A_STEP) -> Dict:
    """
    Universal symmetric / asymmetric bridge, chirp-modulated.

        sym → asym :  finite_step_forward(sym) · w
        asym → sym :  finite_step_central(asym) · w
        mixed      :  sym_to_asym + asym_to_sym
    """
    s2a = finite_step_forward(sym, a) * w
    a2s = finite_step_central(asym, a) * w
    mixed = s2a + a2s
    return dict(
        sym_to_asym=s2a,
        asym_to_sym=a2s,
        mixed=mixed,
        mix_norm=float(np.linalg.norm(mixed)),
    )


# ============================================================
#  4.  Bluestein chirp-z transform  ·  O(K log K)
# ============================================================
def _next_pow2(n: int) -> int:
    return 1 << max(1, int(n - 1).bit_length())


def chirp_z(C: np.ndarray, m: int, W: complex) -> np.ndarray:
    """
    Bluestein's chirp-z, O(N log N).

        X[k] = Σ_{n=0}^{N−1} C[n] · W^{n k},   k = 0 … m−1

    Uses the chirp identity:
        n k = (n² + k² − (k − n)²) / 2

    so the sum becomes a convolution with the chirp sequence
        b[d] = W^{-d²/2}.

    Parameters
    ----------
    C   : length-N complex coefficients
    m   : number of output samples
    W   : per-step multiplier  (any complex scalar)
    """
    C = np.asarray(C, dtype=np.complex128)
    N = C.size
    if N == 0 or m <= 0:
        return np.zeros(m, dtype=np.complex128)

    L = _next_pow2(N + m - 1)

    # chirp on the input side:  a[n] = C[n] · W^{n²/2}
    n = np.arange(N, dtype=np.float64)
    chirp_n = np.power(W, 0.5 * n * n)
    a = np.zeros(L, dtype=np.complex128)
    a[:N] = C * chirp_n

    # chirp on the kernel side:  b[j] = W^{-d²/2} with d = j − (m−1)
    d = np.arange(-(m - 1), N, dtype=np.float64)
    b = np.zeros(L, dtype=np.complex128)
    b[:d.size] = np.power(W, -0.5 * d * d)

    # FFT convolution
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    conv = np.fft.ifft(A * B)

    # output side:  X[k] = W^{k²/2} · conv[k + (m−1)]
    k = np.arange(m, dtype=np.float64)
    chirp_k = np.power(W, 0.5 * k * k)
    return chirp_k * conv[m - 1 : m - 1 + m]


def transcendental_plane(C: np.ndarray, t_samples: np.ndarray,
                         scale: float = A_STEP) -> np.ndarray:
    """
    X[t] = Σ_{k=0}^{K−1} C[k] · exp(i · t · scale · k / K)

    Matches the asymmetric matrix entries:
        [x_k^i, y_k^i]  with  x_k = exp(scale · k / K)

    Fast path — integer t samples  →  chirp-z (O(K log K))
    Slow path — arbitrary t        →  direct (O(T·K))
    """
    K = C.size
    W = np.exp(1j * scale / K)       # per-step multiplier

    # fast path
    if t_samples.size and np.allclose(t_samples, np.arange(t_samples.size)):
        return chirp_z(C.astype(np.complex128), t_samples.size, W)

    # slow path (arbitrary samples)
    out = np.zeros(t_samples.size, dtype=np.complex128)
    k_arr = np.arange(K, dtype=np.float64)
    for j, t in enumerate(t_samples):
        out[j] = np.sum(C * np.exp(1j * t * scale * k_arr / K))
    return out


# ============================================================
#  5.  Full pipeline
# ============================================================
@dataclass
class SolveResult:
    sym_plane: np.ndarray
    asym_plane: np.ndarray
    chirp: np.ndarray
    bridge: Dict
    Z_trans: np.ndarray
    S: float
    H: float
    m: float
    t_build_ms: float
    t_sieve_ms: float
    t_fill_ms: float
    t_czt_ms: float
    n_live: int


class ChirpBridgePipeline:
    """
        build   →  sieve + |Ci| buffer + sym/asym fill     O(K log log K)
        solve   →  chirp bridge + transcendental chirp-z   O(K log K)
    """
    def __init__(self, K: int = 512, alpha: float = ALPHA,
                 scale: float = A_STEP):
        self.K = int(K)
        self.alpha = float(alpha)
        self.scale = float(scale)

    # --------------------------------------------------------
    def build(self, x_grid: Optional[np.ndarray] = None
              ) -> ReadOnlyCiBuffer:
        K = self.K
        t0 = time.perf_counter()

        # ---- 1. discrete sparse plane (O(K log log K)) ----
        mu, live = discrete_sparse_grid(K)
        self.mu = mu
        self.live = live
        t_sieve = (time.perf_counter() - t0) * 1e3

        # ---- 2. grid  (exponential so ln x_k = scale·k/K) ----
        if x_grid is None:
            # x_k = exp(scale · (k+1) / K)   for k = 0 … K−1
            kk = np.arange(1, K + 1, dtype=np.float64)
            x_grid = np.exp(self.scale * kk / K)
        else:
            x_grid = np.asarray(x_grid, dtype=np.float64)
            if x_grid.size != K:
                raise ValueError(f"x_grid must have length {K}")
        self.x_grid = x_grid

        # ---- 3. fill sym / asym planes (O(K)) ----
        t0 = time.perf_counter()
        buf = ReadOnlyCiBuffer(K)
        for i, x_k in enumerate(x_grid):
            k = i + 1
            lnx = math.log(max(x_k, 1e-12))
            inv = 1.0 / max(x_k, 1e-12)
            sym = inv * math.cos(lnx)     # Re(x^{i−1}) = x^{-1} cos ln x
            asym = inv * math.sin(lnx)    # Im(x^{i−1}) = x^{-1} sin ln x
            buf.write_sym(k, sym)
            buf.write_asym(k, asym)
        buf.write_dc(float(np.sum(buf.data[K + 1 : 2 * K + 1])))
        buf.seal()
        t_fill = (time.perf_counter() - t0) * 1e3

        self.t_sieve_ms = t_sieve
        self.t_fill_ms  = t_fill
        return buf

    # --------------------------------------------------------
    def solve(self, buf: ReadOnlyCiBuffer,
              t_samples: Optional[np.ndarray] = None) -> SolveResult:
        K = self.K
        t0 = time.perf_counter()

        # ---- 1. O(K) plane reads
        sym, asym = buf.read_planes()

        # ---- 2. O(K) chirp
        w = base_chirp(K, self.mu)

        # ---- 3. O(K) chirp bridge (finite step)
        bridge = chirp_bridge(sym, asym, w, A_STEP)

        # ---- 4. O(K log K) transcendental solve (chirp-z)
        C = 1.0 / np.maximum(np.abs(self.x_grid), 1e-12)
        if t_samples is None:
            t_samples = np.arange(K, dtype=np.float64)
        t0_czt = time.perf_counter()
        Z = transcendental_plane(C, t_samples, self.scale)
        t_czt = (time.perf_counter() - t0_czt) * 1e3

        # ---- 5. O(K) supertrace / entropy / mass on Re(Z)
        Z_re = np.real(Z)
        S = 0.0
        for i, v in enumerate(Z_re):
            S += v if (i % 2 == 0) else -v

        if S == 0.0 or Z_re.size <= 0:
            H = 0.0
            mass = 0.0
        else:
            p = min(abs(S) / Z_re.size, 0.9999)
            H = -ALPHA * p * math.log(p) if p > 0 else 0.0
            mass = abs(S) * math.exp(-H) if H < 700 else 0.0

        t_solve = (time.perf_counter() - t0) * 1e3
        return SolveResult(
            sym_plane=sym,
            asym_plane=asym,
            chirp=w,
            bridge=bridge,
            Z_trans=Z,
            S=float(S),
            H=float(H),
            m=float(mass),
            t_build_ms=getattr(self, "t_sieve_ms", 0.0) +
                       getattr(self, "t_fill_ms", 0.0),
            t_sieve_ms=getattr(self, "t_sieve_ms", 0.0),
            t_fill_ms=getattr(self, "t_fill_ms", 0.0),
            t_czt_ms=t_czt,
            n_live=int(self.live.size),
        )


# ============================================================
#  Validation  ·  chirp-z vs direct
# ============================================================
def _validate_chirp_z():
    rng = np.random.default_rng(0)
    N = 64
    C = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    W = np.exp(1j * A_STEP / N)

    Z_czt = chirp_z(C, N, W)
    Z_direct = np.array([
        np.sum(C * (W ** (t * np.arange(N))))
        for t in range(N)
    ])
    err = float(np.max(np.abs(Z_czt - Z_direct)))
    return err


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Chirp bridge  ·  discrete sparse → transcendental plane")
    print("=" * 78)
    print(f"  α              = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  α_asym         = 0.3628")
    print(f"  A_STEP         = α / α_asym     = {A_STEP:.6f}")
    print(f"  6/π²           = {DENSITY:.6f}  (μ density)")
    print()
    print("  M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]")
    print("  tr M    = x^(i−1) + y^(i−1)")
    print("          = x^{-1}(cos ln x + i sin ln x)")
    print("          + y^{-1}(cos ln y + i sin ln y)")
    print()

    # ---------- validation ----------
    err = _validate_chirp_z()
    print(f"--- chirp-z validation  (N=64) ---")
    print(f"  max |czt − direct| = {err:.2e}")
    print()

    # ---------- build ----------
    K = 512
    pipe = ChirpBridgePipeline(K=K)
    buf = pipe.build()
    print(f"--- build  ·  K = {K} ---")
    print(f"  μ sieve         = {pipe.t_sieve_ms:.3f} ms   (O(K log log K))")
    print(f"  sym/asym fill   = {pipe.t_fill_ms:.3f} ms   (O(K))")
    print(f"  live addresses  = {pipe.live.size} / {K}  "
          f"= {pipe.live.size / K:.4f}  "
          f"(target 6/π² = {DENSITY:.4f})")
    print(f"  DC              = {buf.dc():+.6f}")
    print(f"  supertrace S    = {buf.supertrace():+.6f}")
    print()

    # ---------- a few buffer slots ----------
    print("--- read-only buffer  ·  selected slots ---")
    print(f"  {'k':>5}  {'x_k':>12}  {'sym = Re(tr M)':>20}  "
          f"{'asym = Im(tr M)':>20}")
    for k in [1, 7, 64, 128, 256, 512]:
        x_k = pipe.x_grid[k - 1]
        print(f"  {k:>5}  {x_k:>12.6f}  {buf.sym(k):>20.8f}  "
              f"{buf.asym(k):>20.8f}")
    print()

    # ---------- chirp ----------
    w = base_chirp(K, pipe.mu)
    print("--- ALU chirp  w[k] = gate · cos(A_STEP · k² / K) ---")
    print(f"  |w| max        = {np.max(np.abs(w)):.6f}")
    print(f"  nonzero count  = {int(np.count_nonzero(np.abs(w) > 1e-12))} / {K}")
    print()

    # ---------- chirp bridge ----------
    sym, asym = buf.read_planes()
    bridge = chirp_bridge(sym, asym, w)
    print(f"--- chirp bridge  ·  A_STEP = {A_STEP:.4f} ---")
    print(f"  ‖sym → asym‖   = {np.linalg.norm(bridge['sym_to_asym']):.6f}")
    print(f"  ‖asym → sym‖   = {np.linalg.norm(bridge['asym_to_sym']):.6f}")
    print(f"  ‖mixed‖        = {bridge['mix_norm']:.6f}")
    print()

    # ---------- transcendental solve ----------
    res = pipe.solve(buf)
    print("--- transcendental solve  ·  chirp-z  ·  O(K log K) ---")
    print(f"  m samples      = {res.Z_trans.size}")
    print(f"  chirp-z time   = {res.t_czt_ms:.3f} ms")
    print(f"  ζ(0) real      = {res.Z_trans[0].real:+.6f}")
    print(f"  ζ(1) real      = {res.Z_trans[1].real:+.6f}")
    print(f"  ζ(2) real      = {res.Z_trans[2].real:+.6f}")
    print(f"  |ζ| max        = {np.max(np.abs(res.Z_trans)):.6f}")
    print()

    # ---------- invariants ----------
    print("--- invariants ---")
    print(f"  S (supertrace) = {res.S:+.6f}")
    print(f"  H (entropy)    = {res.H:.6f}")
    print(f"  m = |S|e^{{−H}} = {res.m:.6e}")
    print()

    # ---------- timing summary ----------
    print("--- timing ---")
    print(f"  build   = {res.t_build_ms:7.3f} ms   "
          f"(O(K log log K))")
    print(f"  solve   = {res.t_czt_ms:7.3f} ms   "
          f"(O(K log K), chirp-z)")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  discrete sparse plane    O(K log log K)   sieve → live addresses")
    print("  |Ci| buffer              O(K)             sym + asym fill")
    print("  chirp bridge             O(K)             finite step sym ↔ asym")
    print("  transcendental solve     O(K log K)       Bluestein chirp-z")
    print("  supertrace               O(K)")
    print()
    print("  The chirp w[k] = gate·cos(A_STEP·k²/K) and the chirp-z kernel")
    print("  b[d] = W^{-d²/2} are the SAME quadratic structure — that is")
    print("  the discrete ↔ transcendental parse.  The finite step at")
    print(f"  A_STEP = {A_STEP:.4f} is the universal sym ↔ asym bridge.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()