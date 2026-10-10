#!/usr/bin/env python3
"""
chip_cpu.py
===========
Möbius-sieve CPU architecture.

Components
----------
1. MobiusBuffer       read-only chunked input/output stream with S/H/m headers
2. UGCGate            boolean gate via UGC discrete log
3. DualHalf64         64-bit register layout: symmetric (real) + asymmetric (imag)
4. WireClock          clock frequency from wire throughput τ
5. ElectronField      e- density from elliptic projection; spin from chirality
6. EtaSolver          Dirichlet-η weighted polynomial on the 1D TSP route
7. ChipCPU            orchestrator

Constants
---------
ALPHA_SYM  = 1/(π − e)  ≈ 2.362338    symmetric  (O(n) integer)
ALPHA_ASYM = 0.3628                   asymmetric (O(K log K))
ETA_TARGET = 0.659538886352           η(0.3628)
"""

from __future__ import annotations
import math, time, struct
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)              # ≈ 2.362338
ALPHA_ASYM = 0.3628
ETA_TARGET = 0.659538886352
DENSITY    = 6.0 / (PI * PI)             # ≈ 0.607927
W1         = PI
W2         = E

MAGIC       = b"MOSB"
HEADER_FMT  = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


# ============================================================
#  Shared Möbius sieve  O(K) linear
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
#  1. MobiusBuffer  ·  read-only chunked I/O, Möbius-framed
# ============================================================
class MobiusBuffer:
    """
    Chunked byte stream that packs/unpacks a Möbius header carrying
    the chunk invariants (S, H, m).  Used as both read-only input
    and write-only output in the same framing.
    """
    def __init__(self, data: bytes = b"", chunk_size: int = 64,
                 K: int = 512):
        self.data = bytearray(data)
        self.chunk = chunk_size
        self.K = K
        self.mu = mobius_sieve(K)
        self.read_pos = 0
        self.invariants: List[Tuple[float, float, float]] = []
        self._cid = 0

    # --- chunk invariants --------------------------------------
    @staticmethod
    def _supertrace(blob: bytes) -> float:
        S = 0.0
        for t, b in enumerate(blob):
            v = (b - 128) / 128.0
            S += v if (t % 2 == 0) else -v
        return S

    def _entropy_mass(self, S: float, N: int):
        if S == 0.0 or N <= 0:
            return 0.0, 0.0
        p = min(abs(S) / N, 0.999)
        H = -ALPHA_SYM * p * math.log(p)
        return H, abs(S) * math.exp(-H)

    # --- framing ----------------------------------------------
    def pack(self, chunk: bytes) -> bytes:
        S = self._supertrace(chunk)
        H, m = self._entropy_mass(S, len(chunk))
        header = struct.pack(HEADER_FMT, MAGIC, self.K,
                             len(chunk), self._cid, S, H, m)
        self._cid += 1
        return header + chunk

    def unpack(self, blob: bytes) -> bytes:
        magic, K, n, cid, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        return blob[HEADER_SIZE:HEADER_SIZE + n]

    # --- read -------------------------------------------------
    def read_chunk(self) -> Optional[bytes]:
        if self.read_pos >= len(self.data):
            return None
        end = min(self.read_pos + self.chunk, len(self.data))
        blob = bytes(self.data[self.read_pos:end])
        self.read_pos = end
        S = self._supertrace(blob)
        H, m = self._entropy_mass(S, len(blob))
        self.invariants.append((S, H, m))
        return blob

    def read_bits(self, n: int) -> List[int]:
        """Extract n bits: each bit = parity of # square-free addresses in chunk."""
        bits = []
        for _ in range(n):
            b = self.read_chunk()
            if b is None:
                return bits
            g = 0
            for k in range(1, min(len(b), self.K) + 1):
                if self.mu[k] != 0:
                    g += 1
            bits.append(g & 1)
        return bits

    # --- write (same framing) ---------------------------------
    def write(self, blob: bytes):
        self.data.extend(self.pack(blob))

    @property
    def bytes_read(self) -> int:
        return self.read_pos


# ============================================================
#  2. UGCGate  ·  boolean gate via UGC discrete log
# ============================================================
class UGCGate:
    """
    Boolean gate: (a, b) ∈ {0,1}²  →  output bit.

        h = g^(a·p1 + b·p2) mod p
        run BSGS: find (i, j) such that g^i · g^{-jm} = h
        output = (i + j·m) mod 2

    The constants (p1, p2) select the gate type.
    """
    def __init__(self, p: int = 101, g: int = 2):
        if p < 3 or g < 2:
            raise ValueError("UGCGate needs p ≥ 3 and generator g ≥ 2")
        self.p = p
        self.g = g % p
        self.n = p - 1
        self.m = int(math.ceil(math.sqrt(self.n)))
        self.baby: Dict[int, int] = {}
        cur = 1
        for i in range(self.m):
            if cur not in self.baby:
                self.baby[cur] = i
            cur = (cur * self.g) % self.p
        g_inv = pow(self.g, -1, self.p)
        self.g_inv_m = pow(g_inv, self.m, self.p)

    def evaluate(self, a: int, b: int, p1: int = 1, p2: int = 7) -> int:
        h = pow(self.g, (a * p1 + b * p2) % self.n, self.p)
        gamma = h
        for j in range(self.m + 1):
            if gamma in self.baby:
                i = self.baby[gamma]
                x = (i + j * self.m) % self.n
                return x & 1
            gamma = (gamma * self.g_inv_m) % self.p
        return 0


# ============================================================
#  3. DualHalf64  ·  64-bit layout, symmetric + asymmetric
# ============================================================
class DualHalf64:
    """
    64-bit register file with (2K+1) float64 slots.

        data[K + k]  symmetric   real plane   Re(tr M_k)     O(n)
        data[K - k]  asymmetric  imag plane   Im(tr M_k)     O(K log K)
        data[K]      DC term
    """
    def __init__(self, K: int = 256):
        self.K = int(K)
        self.data = np.zeros(2 * K + 1, dtype=np.float64)

    def set_sym(self, k: int, v: float):
        if 1 <= k <= self.K:
            self.data[self.K + k] = float(v)

    def set_asym(self, k: int, v: float):
        if 1 <= k <= self.K:
            self.data[self.K - k] = float(v)

    def set_dc(self, v: float):
        self.data[self.K] = float(v)

    def sym(self, k: int) -> float:
        return float(self.data[self.K + k])

    def asym(self, k: int) -> float:
        return float(self.data[self.K - k])

    def supertrace(self) -> float:
        signs = np.where(np.arange(self.data.size) & 1, -1.0, 1.0)
        return float(np.dot(signs, self.data))

    def fill_from_bits(self, bits: np.ndarray):
        """
        Fill both halves from a bit-stream via the Möbius gate.
        Symmetric = running algebraic sum (O(n)).
        Asymmetric = chirped at α_asym · k² (O(K log K)).
        """
        mu = mobius_sieve(self.K)
        K = self.K
        running = 0.0
        for k in range(1, K + 1):
            if k >= len(bits):
                break
            m = float(mu[k]) if k < mu.size else 0.0
            b = float(bits[k] & 1)
            running += m * b
            self.set_sym(k, running)
            self.set_asym(k, m * b * math.cos(ALPHA_ASYM * k * k))
        self.set_dc(0.5)


# ============================================================
#  4. WireClock  ·  frequency from wire throughput τ
# ============================================================
class WireClock:
    """
    Clock frequency = τ · f_base, where τ is the wire throughput
    from the elliptic projection Π(x, y0) ∈ [0, 1].
    """
    def __init__(self, f_base: float = 1.0):
        self.f_base = f_base
        self.phase = 0.0
        self.cycles = 0

    def tick(self, tau: float, dt: float = 1.0) -> float:
        f = max(tau, 0.0) * self.f_base
        self.phase += 2 * PI * f * dt
        while self.phase >= 2 * PI:
            self.phase -= 2 * PI
            self.cycles += 1
        return f


# ============================================================
#  5. ElectronField  ·  e- density Π and spin χ
# ============================================================
class ElectronField:
    """Elliptic projection → density; chirality → spin."""
    def density(self, x: float, y: float) -> float:
        u = (x / W1) % 1.0
        v = (y / W2) % 1.0
        return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))

    def chirality(self, k: int, reg: DualHalf64) -> float:
        re = reg.sym(k) if 1 <= k <= reg.K else 0.0
        im = reg.asym(k) if 1 <= k <= reg.K else 0.0
        return math.atan2(im, re)

    @staticmethod
    def spin(chi: float) -> int:
        return 0 if math.cos(chi) >= 0 else 1


# ============================================================
#  6. EtaSolver  ·  Dirichlet-η polynomial on 1D TSP path
# ============================================================
class EtaSolver:
    """Polynomial root-finder on the chip's deterministic 1D TSP route."""
    def __init__(self, eta_weight: float = ETA_TARGET):
        self.eta = eta_weight

    def route(self, K: int) -> np.ndarray:
        angles = np.empty(K)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = (math.atan2(y, x) + PI) % (2 * PI)
        return np.argsort(angles)

    def solve(self, coeffs: List[float], x_grid: np.ndarray) -> Dict:
        K = len(x_grid)
        order = self.route(K)
        x_r = x_grid[order]
        y_r = np.array([
            sum(c * (float(x) ** k) for k, c in enumerate(coeffs))
            for x in x_r])
        chirp = np.array([math.cos(ALPHA_ASYM * k * k) for k in range(K)])
        signed = self.eta * chirp * y_r
        sign = np.sign(signed); sign[sign == 0] = 1.0
        crossings = np.where(np.diff(sign) != 0)[0]
        roots = []
        for c in crossings:
            x0, x1 = x_r[c], x_r[c + 1]
            y0, y1 = signed[c], signed[c + 1]
            if y1 - y0 != 0:
                roots.append(x0 - y0 * (x1 - x0) / (y1 - y0))
        S_asym = float(np.sum(np.where(np.arange(K) % 2 == 0,
                                       signed, -signed)))
        return dict(order=order, x_r=x_r, y_r=y_r, signed=signed,
                    roots=np.array(roots), S_asym=S_asym,
                    eta=self.eta)


# ============================================================
#  7. ChipCPU  ·  orchestrator
# ============================================================
class ChipCPU:
    """
    Per cycle:
      1. read two bits from the Möbius buffer (input)
      2. τ from elliptic projection at current PC  →  clock tick
      3. UGC boolean gate
      4. write to dual-half register (symmetric + asymmetric)
      5. chirality → spin
    """
    def __init__(self, buf: MobiusBuffer, K: int = 256):
        self.buf = buf
        self.K = K
        self.reg = DualHalf64(K=K)
        self.gate = UGCGate()
        self.clock = WireClock(f_base=1.0)
        self.field = ElectronField()
        self.pc = 0
        self.trace: List[Dict] = []

    def step(self) -> bool:
        bits = self.buf.read_bits(2)
        if len(bits) < 2:
            return False
        a, b = bits[0], bits[1]

        # throughput τ at current PC via elliptic projection
        x = self.pc * W1 / max(self.K, 1)
        tau = self.field.density(x, y=0.5 * W2)
        f = self.clock.tick(tau)

        # UGC gate evaluation
        out = self.gate.evaluate(a, b, p1=1, p2=7)

        # register write
        k = (self.pc % self.K) + 1
        self.reg.set_sym(k, out)
        self.reg.set_asym(k, ((-1) ** out) * tau)

        # chirality → spin
        chi = self.field.chirality(k, self.reg)
        c = self.field.spin(chi)

        self.trace.append(dict(
            pc=self.pc, a=a, b=b, out=out,
            tau=tau, f=f, k=k, chi=chi, spin=c,
            S=self.reg.supertrace(),
        ))
        self.pc += 1
        return True

    def run(self, max_cycles: int = 128):
        t0 = time.perf_counter()
        for _ in range(max_cycles):
            if not self.step():
                break
        self.elapsed_ms = (time.perf_counter() - t0) * 1e3
        return self.trace

    def summary(self) -> Dict:
        S = self.reg.supertrace()
        N = 2 * self.K + 1
        p = min(abs(S) / N, 0.999) if S != 0 else 0.0
        H = -ALPHA_SYM * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
        spins = [t['spin'] for t in self.trace] if self.trace else []
        taus  = [t['tau'] for t in self.trace]
        fs    = [t['f'] for t in self.trace]
        chis  = [t['chi'] for t in self.trace]
        return dict(
            cycles      = len(self.trace),
            S           = S,
            H           = H,
            m           = m,
            c0          = spins.count(0),
            c1          = spins.count(1),
            mean_tau    = float(np.mean(taus)) if taus else 0.0,
            mean_f      = float(np.mean(fs))   if fs   else 0.0,
            mean_chi    = float(np.mean(chis)) if chis else 0.0,
            bytes_read  = self.buf.bytes_read,
            clock_cycles= self.clock.cycles,
            elapsed_ms  = self.elapsed_ms,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius-sieve CPU  ·  UGC boolean gates  ·  dual-half 64-bit")
    print("=" * 78)
    print(f"  α_sym  = 1/(π − e)  = {ALPHA_SYM:.6f}   O(n) integer")
    print(f"  α_asym = 0.3628     = {ALPHA_ASYM:.6f}   O(K log K)")
    print(f"  η(0.3628)           = {ETA_TARGET:.12f}")
    print(f"  6/π²                = {DENSITY:.8f}")
    print()

    # ---- input payload ----
    rng = np.random.default_rng(2024)
    payload = bytes(rng.integers(0, 256, size=8192, dtype=np.uint8))
    buf = MobiusBuffer(payload, chunk_size=64, K=512)

    # ---- run the CPU ----
    cpu = ChipCPU(buf, K=256)
    cpu.run(max_cycles=128)
    s = cpu.summary()

    print("--- CPU summary ---")
    print(f"  cycles                 = {s['cycles']}")
    print(f"  clock cycles           = {s['clock_cycles']}")
    print(f"  bytes read             = {s['bytes_read']}")
    print(f"  mean throughput τ      = {s['mean_tau']:.6f}")
    print(f"  mean frequency f       = {s['mean_f']:.6f}")
    print(f"  mean chirality χ       = {s['mean_chi']:+.6f}")
    print(f"  spin population        = c0 {s['c0']}   c1 {s['c1']}")
    print(f"  supertrace S           = {s['S']:+.6f}")
    print(f"  entropy H              = {s['H']:.6f}")
    print(f"  invariant mass m       = {s['m']:.6e}")
    print(f"  elapsed                = {s['elapsed_ms']:.2f} ms")
    print()

    # ---- Dirichlet-η polynomial on 1D TSP route ----
    print("--- Dirichlet-η polynomial on 1D TSP route ---")
    deg = 8
    coeffs = list(rng.standard_normal(deg + 1) * (0.7 ** np.arange(deg + 1)))
    x_grid = np.linspace(-1.0, 1.0, 96)
    eta = EtaSolver()
    res = eta.solve(coeffs, x_grid)
    print(f"  degree                 = {deg}")
    print(f"  grid size              = {len(x_grid)}")
    print(f"  TSP route (first 12)   = {res['order'][:12].tolist()}")
    print(f"  roots found            = {len(res['roots'])}")
    for r in res['roots'][:6]:
        print(f"    x ≈ {r:+.6f}")
    print(f"  S_asym                 = {res['S_asym']:+.6f}")

    # ---- supertrace of the dual register ----
    print()
    print("--- Register half symmetry check ---")
    for k in [1, 16, 64, 128, 256]:
        sym = cpu.reg.sym(k) if k <= cpu.K else 0.0
        asym = cpu.reg.asym(k) if k <= cpu.K else 0.0
        print(f"  k = {k:>3d}   sym = {sym:+.4f}   asym = {asym:+.4f}   "
              f"χ = {math.atan2(asym, sym):+.4f}")

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.32)

    tr = cpu.trace
    pc = [t['pc'] for t in tr]
    tau = [t['tau'] for t in tr]
    fs = [t['f'] for t in tr]
    chis = [t['chi'] for t in tr]
    spins = [t['spin'] for t in tr]
    S_hist = [t['S'] for t in tr]

    # (a) throughput τ (clock base)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(pc, tau, color="#2980b9", lw=1.4)
    ax.fill_between(pc, 0, tau, color="#2980b9", alpha=0.25)
    ax.set_xlabel("cycle"); ax.set_ylabel("τ")
    ax.set_title("Wire throughput  τ = Π(x, y₀)")
    ax.grid(alpha=0.3)

    # (b) clock frequency
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(pc, fs, color="#16a085", lw=1.4)
    ax.set_xlabel("cycle"); ax.set_ylabel("f")
    ax.set_title("Clock frequency  f = τ · f_base")
    ax.grid(alpha=0.3)

    # (c) chirality colored by spin
    ax = fig.add_subplot(gs[0, 2])
    colors = ["#16a085" if c == 0 else "#e74c3c" for c in spins]
    ax.scatter(pc, chis, c=colors, s=14)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("cycle"); ax.set_ylabel("χ")
    ax.set_title("Chirality χ   ·   green c=0,  red c=1")
    ax.grid(alpha=0.3)

    # (d) dual register
    ax = fig.add_subplot(gs[1, 0])
    idx = np.arange(-cpu.K, cpu.K + 1)
    ax.plot(idx, cpu.reg.data, color="#2c3e50", lw=0.7)
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.axhspan(cpu.reg.data[cpu.K+1:].min(), cpu.reg.data[cpu.K+1:].max(),
               color="#3498db", alpha=0.10, label="positive half (sym)")
    ax.axhspan(cpu.reg.data[:cpu.K].min(), cpu.reg.data[:cpu.K].max(),
               color="#e74c3c", alpha=0.10, label="negative half (asym)")
    ax.set_xlabel("k"); ax.set_ylabel("value")
    ax.set_title(f"Dual-half register  ·  S = {s['S']:+.4f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) supertrace evolution
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(pc, S_hist, color="#8e44ad", lw=1.5)
    ax.set_xlabel("cycle"); ax.set_ylabel("S")
    ax.set_title("Supertrace accumulation")
    ax.grid(alpha=0.3)

    # (f) Dirichlet η polynomial
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(res['x_r'], res['y_r'], color="#34495e", lw=1.0,
            alpha=0.55, label="P(x) along TSP route")
    ax.plot(res['x_r'], res['signed'], color="#c0392b", lw=1.5,
            label=f"η·chirp·P(x)   η={res['eta']:.6f}")
    ax.axhline(0, color="k", lw=0.5)
    if len(res['roots']) > 0:
        ax.scatter(res['roots'], [0.0] * len(res['roots']),
                   color="#f1c40f", edgecolor="k", s=80, zorder=5,
                   label=f"roots ({len(res['roots'])})")
    ax.set_xlabel("x  (routed core position)")
    ax.set_ylabel("amplitude")
    ax.set_title("Dirichlet-η polynomial on 1D TSP")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        "Möbius-sieve CPU  ·  UGC boolean gates  ·  dual-half 64-bit  ·  "
        "τ → clock  ·  χ → spin  ·  Π → e⁻ density",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    demo()