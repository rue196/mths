#!/usr/bin/env python3
"""
mobius_boot_pipeline.py
=======================
End-to-end Möbius boot:

    Stage 1  chirp signal       elliptic projection, a = α/0.3628
    Stage 2  server buffer      MOSB unpack from ExtendedDNSServer
    Stage 3  cpu boot           ChipCPU on the two paths:
                                    symmetric  O(K log log K)
                                    asymmetric O(K log K)
    Stage 4  file envelopes     ReadEnvelope / RunEnvelope / WriteEnvelope

Constants:
    α       = 1/(π − e)      ≈ 2.362338
    a_step  = α / 0.3628     ≈ 6.511
    density = 6/π²           ≈ 0.607927
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)
ALPHA_ASYM = 0.3628
A_STEP     = ALPHA_SYM / ALPHA_ASYM
DENSITY    = 6.0 / (PI * PI)
W1, W2     = PI, E


# ============================================================
#  Standalone fallbacks  (in case the source files are missing)
# ============================================================
def _mobius_sieve(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
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


def _elliptic_projection(x: float, y: float) -> float:
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


def _supertrace(data: bytes) -> float:
    S = 0.0
    for t, b in enumerate(data):
        v = (b - 128) / 128.0
        S += v if (t % 2 == 0) else -v
    return S


def _entropy_mass(S: float, n: int) -> Tuple[float, float]:
    if S == 0.0 or n <= 0:
        return 0.0, 0.0
    p = min(abs(S) / n, 0.9999)
    H = -ALPHA_SYM * p * math.log(p)
    return H, abs(S) * math.exp(-H)


# ============================================================
#  Stage 1  ·  Boot chirp  ·  elliptic projection
# ============================================================
@dataclass
class BootChirp:
    """
    The signal that wakes the machine.  A single cycle of the
    elliptic projection over the torus ℂ/(πℤ + eℤ), sampled at K
    points and weighted by the finite step a = α/0.3628.
    """
    K: int
    samples: np.ndarray
    weights: np.ndarray
    S: float
    H: float
    m: float
    t_ms: float

    @classmethod
    def build(cls, K: int = 256) -> "BootChirp":
        t0 = time.perf_counter()
        x = np.linspace(0.0, W1, K)
        y = 0.5 * W2
        samples = np.array([_elliptic_projection(float(xi), y) for xi in x])
        # phase rate from the finite step
        weights = np.cos(A_STEP * x * x / K)
        signal  = samples * weights
        # supertrace from the chirp as bytes
        payload = bytes(int(min(255, max(0, 128 + 127 * float(v))))
                        for v in signal)
        S = _supertrace(payload)
        H, m = _entropy_mass(S, K)
        return cls(K=K, samples=samples, weights=weights,
                   S=S, H=H, m=m,
                   t_ms=(time.perf_counter() - t0) * 1e3)


# ============================================================
#  Stage 2  ·  Server buffer unpack  ·  MOSB frames
# ============================================================
MAGIC       = b"MOSB"
HEADER_FMT  = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)         # 48


class ServerBuffer:
    """
    Wraps a chunked MOSB buffer.  Each appended chunk carries its
    own S, H, m in the header; readers re-verify the supertrace
    without trusting the writer.
    """
    def __init__(self, K: int = 512):
        self.K = int(K)
        self.mu = _mobius_sieve(K)
        self.data = bytearray()
        self.chunks: List[Tuple[int, int, float, float, float]] = []
        self._cid = 0

    def append(self, payload: bytes) -> int:
        n = len(payload)
        S = _supertrace(payload)
        H, m = _entropy_mass(S, n)
        cid = self._cid
        self._cid += 1
        header = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        start = len(self.data)
        self.data.extend(header + payload)
        self.chunks.append((start, start + HEADER_SIZE + n, S, H, m))
        return cid

    def read(self, cid: int) -> Optional[Tuple[bytes, float, float, float]]:
        if cid < 0 or cid >= len(self.chunks):
            return None
        start, end, *_ = self.chunks[cid]
        blob = bytes(self.data[start:end])
        magic, K, n, cid_stored, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])
        if magic != MAGIC or K != self.K or cid_stored != cid:
            return None
        payload = blob[HEADER_SIZE:HEADER_SIZE + n]
        S_check = _supertrace(payload)
        if abs(S_check - S) > 1e-9:
            return None
        return payload, S, H, m

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def size(self) -> int:
        return len(self.data)


# ============================================================
#  Stage 3  ·  CPU boot  ·  symmetric vs asymmetric
# ============================================================
@dataclass
class CPUBootResult:
    K: int
    mode: str                    # 'symmetric' or 'asymmetric'
    bits: List[int]
    S: float
    H: float
    m: float
    cycles: int
    t_ms: float

    def summary(self) -> Dict:
        return dict(
            K=self.K, mode=self.mode, cycles=self.cycles,
            n_bits=len(self.bits),
            S=self.S, H=self.H, m=self.m, t_ms=self.t_ms,
        )


class SymmetricBoot:
    """
    Symmetric boot path: linear Möbius sieve + per-bit supertrace.
    Complexity: O(K log log K) for the sieve, O(K) per boot cycle.
    """
    def __init__(self, K: int = 256):
        self.K = K
        self.mu = _mobius_sieve(K)

    def boot(self, chirp: BootChirp, chunks: List[bytes]) -> CPUBootResult:
        t0 = time.perf_counter()
        bits: List[int] = []
        # for each chunk, count nonzero-μ slots → 1 bit
        for blob in chunks:
            n = min(len(blob), self.K)
            if n == 0:
                bits.append(0)
                continue
            g = 0
            for k in range(1, n + 1):
                # include the chirp weight on each slot
                gate = 1 if self.mu[k] != 0 else 0
                w = float(chirp.samples[(k - 1) % chirp.K])
                if gate and w > 0.5:
                    g ^= 1
            bits.append(g & 1)

        # supertrace over the bit sequence
        S = sum((1 if i % 2 == 0 else -1) * (1 if b else 0)
                for i, b in enumerate(bits))
        H, m = _entropy_mass(S, len(bits))
        return CPUBootResult(
            K=self.K, mode='symmetric', bits=bits,
            S=float(S), H=H, m=m,
            cycles=len(bits),
            t_ms=(time.perf_counter() - t0) * 1e3,
        )


class AsymmetricBoot:
    """
    Asymmetric boot path: segmented Möbius sieve + chirped bit stream.
    Complexity: O(K log K) for the segmented sieve, O(K) per boot cycle.
    """
    def __init__(self, K: int = 256):
        self.K = K
        self.mu = self._segmented_sieve(K)

    @staticmethod
    def _segmented_sieve(K: int) -> np.ndarray:
        mu = np.ones(K + 1, dtype=np.int8)
        mu[0] = 0
        if K >= 1:
            mu[1] = 1
        limit = int(math.isqrt(K)) + 1
        is_comp = np.zeros(limit + 1, dtype=bool)
        for i in range(2, limit + 1):
            if not is_comp[i]:
                mu[i::i] = -mu[i::i]
                p2 = i * i
                if p2 <= K:
                    mu[p2::p2] = 0
                for j in range(i * i, limit + 1, i):
                    is_comp[j] = True
        return mu

    def boot(self, chirp: BootChirp, chunks: List[bytes]) -> CPUBootResult:
        t0 = time.perf_counter()
        bits: List[int] = []
        for ci, blob in enumerate(chunks):
            n = min(len(blob), self.K)
            if n == 0:
                bits.append(0)
                continue
            g = 0
            phase = ALPHA_ASYM * ci * ci          # chirped address
            for k in range(1, n + 1):
                gate = 1 if self.mu[k] != 0 else 0
                q = (k * k + k + int(phase * 100)) % self.K
                gate &= 1 if self.mu[q] != 0 else 0
                w = float(chirp.weights[(k - 1) % chirp.K])
                if gate and w > 0.0:
                    g ^= 1
            bits.append(g & 1)

        S = sum((1 if i % 2 == 0 else -1) * (1 if b else 0)
                for i, b in enumerate(bits))
        H, m = _entropy_mass(S, len(bits))
        return CPUBootResult(
            K=self.K, mode='asymmetric', bits=bits,
            S=float(S), H=H, m=m,
            cycles=len(bits),
            t_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  Stage 4  ·  File envelopes  ·  read / run / write
# ============================================================
from enum import Enum


class AccessMode(Enum):
    READ  = "read"
    RUN   = "run"
    WRITE = "write"


@dataclass
class Envelope:
    path: str
    mode: AccessMode
    sig: bytes
    qs: List[int]
    weights: List[float]
    challenge: bytes = b""
    _key_hash: bytes = b""
    _burned: bool = False
    _cursor: int = 0
    _writes: int = 0

    def fingerprint(self) -> str:
        h = hashlib.shake_256(self.sig)
        for q, w in zip(self.qs, self.weights):
            h.update(struct.pack('>Id', q, w))
        return h.digest(32).hex()

    def authorize(self, key: bytes) -> None:
        self._key_hash = hashlib.shake_256(key + self.challenge).digest(32)

    def _verify(self, key: bytes) -> bool:
        if not self._key_hash:
            return False
        h = hashlib.shake_256(key + self.challenge).digest(32)
        return h == self._key_hash

    def read(self) -> bytes:
        if self.mode == AccessMode.RUN and self._burned:
            raise PermissionError("RunEnvelope: burned")
        with open(self.path, "rb") as f:
            return f.read()

    def run(self, executor=None) -> Dict:
        if self.mode != AccessMode.RUN:
            raise PermissionError("RunEnvelope required")
        if self._burned:
            raise PermissionError("RunEnvelope: already burned")
        # walk the 1D path
        steps = 0
        while self._cursor < len(self.qs):
            q, w = self.qs[self._cursor], self.weights[self._cursor]
            if executor: executor(steps, q, w)
            self._cursor += 1
            steps += 1
        # exec the file
        with open(self.path, "rb") as f:
            src = f.read()
        ns = {"__name__": "__run__", "__file__": self.path}
        code = compile(src, self.path, "exec")
        try:
            exec(code, ns)
            status = "ok"
        except Exception as e:
            status = f"error: {e}"
        finally:
            self._burned = True
        return dict(path=self.path, status=status,
                    steps=steps, burned=self._burned)

    def write(self, key: bytes, data: bytes) -> Dict:
        if self.mode != AccessMode.WRITE:
            raise PermissionError("WriteEnvelope required")
        if not self._verify(key):
            raise PermissionError("WriteEnvelope: invalid key")
        with open(self.path, "wb") as f:
            f.write(data)
        self._writes += 1
        return dict(path=self.path, bytes_written=len(data),
                    writes=self._writes)


class EnvelopeFactory:
    """Builds file envelopes from a booted system."""
    def __init__(self, K: int = 256):
        self.K = K
        self.mu = _mobius_sieve(K)

    def _extract(self, path: str) -> Tuple[bytes, List[int], List[float]]:
        with open(path, "rb") as f:
            content = f.read()
        sig = hashlib.shake_256(content).digest(32)
        raw = hashlib.shake_256(sig).digest(self.K * 16)
        qs, ws = [], []
        for i in range(self.K):
            xi = int.from_bytes(raw[16 * i:16 * i + 8], "big")
            yi = int.from_bytes(raw[16 * i + 8:16 * i + 16], "big")
            x = 0.5 + (xi / (2**64 - 1)) * 4.5
            y = 0.5 + (yi / (2**64 - 1)) * 4.5
            h = int.from_bytes(
                hashlib.shake_256(struct.pack('>dd', x, y)).digest(8),
                'big')
            q = (h * h + h) % self.K
            if self.mu[q] != 0:
                qs.append(q)
                ws.append(_elliptic_projection(x, y))
        return sig, qs, ws

    def read(self, path: str) -> Envelope:
        sig, qs, ws = self._extract(path)
        return Envelope(path, AccessMode.READ, sig, qs, ws)

    def run(self, path: str) -> Envelope:
        sig, qs, ws = self._extract(path)
        return Envelope(path, AccessMode.RUN, sig, qs, ws)

    def write(self, path: str, challenge: bytes) -> Envelope:
        sig, qs, ws = self._extract(path)
        return Envelope(path, AccessMode.WRITE, sig, qs, ws,
                        challenge=challenge)


class OSPermissionBroker:
    def __init__(self, master_secret: bytes):
        self._master = master_secret

    def challenge(self) -> bytes:
        return os.urandom(32)

    def derive_write_key(self, env: Envelope,
                         master_secret: Optional[bytes] = None) -> bytes:
        secret = master_secret if master_secret is not None else self._master
        return hashlib.shake_256(
            secret + env.challenge + bytes.fromhex(env.fingerprint())
        ).digest(32)


# ============================================================
#  Full boot pipeline
# ============================================================
@dataclass
class BootReport:
    chirp: BootChirp
    n_chunks: int
    buffer_bytes: int
    symmetric: CPUBootResult
    asymmetric: CPUBootResult
    envelopes: Dict[str, Envelope]
    files: Dict[str, int]
    total_ms: float


class MobiusBootPipeline:
    """
    Ties all four stages together.

        chirp   →  server buffer  →  CPU (sym + asym)  →  file envelopes
    """
    def __init__(self,
                 K_chirp: int = 256,
                 K_buffer: int = 512,
                 K_cpu_sym: int = 256,
                 K_cpu_asym: int = 256,
                 K_env: int = 256):
        self.K_chirp = K_chirp
        self.K_buffer = K_buffer
        self.K_cpu_sym = K_cpu_sym
        self.K_cpu_asym = K_cpu_asym
        self.K_env = K_env

        self.chirp: Optional[BootChirp] = None
        self.buffer: Optional[ServerBuffer] = None
        self.sym_boot: Optional[SymmetricBoot] = None
        self.asym_boot: Optional[AsymmetricBoot] = None
        self.env_factory: Optional[EnvelopeFactory] = None
        self.broker: Optional[OSPermissionBroker] = None

    # --------------------------------------------------------
    def stage1_chirp(self) -> BootChirp:
        self.chirp = BootChirp.build(K=self.K_chirp)
        return self.chirp

    # --------------------------------------------------------
    def stage2_buffer(self, seed: int = 2024) -> ServerBuffer:
        if self.chirp is None:
            raise RuntimeError("stage1 must run first")
        rng = np.random.default_rng(seed)
        buf = ServerBuffer(K=self.K_buffer)
        # use the chirp bytes as the seed payload
        chirp_bytes = bytes(int(min(255, max(0, 128 + 127 * float(v))))
                            for v in self.chirp.samples)
        # append: chirp itself + random blocks
        buf.append(chirp_bytes)
        for _ in range(4):
            block = rng.integers(0, 256, size=self.K_buffer,
                                 dtype=np.uint8).tobytes()
            buf.append(block)
        # also append the concatenation as one big chunk
        big = b"".join(rng.integers(0, 256, size=512,
                                    dtype=np.uint8).tobytes()
                       for _ in range(8))
        buf.append(big)
        self.buffer = buf
        return buf

    # --------------------------------------------------------
    def stage3_cpu_boot(self) -> Tuple[CPUBootResult, CPUBootResult]:
        if self.chirp is None or self.buffer is None:
            raise RuntimeError("stage1 & stage2 must run first")
        # read all chunks from the buffer
        chunks: List[bytes] = []
        for cid in range(self.buffer.n_chunks):
            r = self.buffer.read(cid)
            if r is not None:
                chunks.append(r[0])
        # symmetric path
        self.sym_boot = SymmetricBoot(K=self.K_cpu_sym)
        sym_res = self.sym_boot.boot(self.chirp, chunks)
        # asymmetric path
        self.asym_boot = AsymmetricBoot(K=self.K_cpu_asym)
        asym_res = self.asym_boot.boot(self.chirp, chunks)
        return sym_res, asym_res

    # --------------------------------------------------------
    def stage4_envelopes(self, files: Dict[str, bytes]
                         ) -> Dict[str, Envelope]:
        self.env_factory = EnvelopeFactory(K=self.K_env)
        self.broker = OSPermissionBroker(master_secret=b"os-boot-secret")
        envs: Dict[str, Envelope] = {}
        for name, content in files.items():
            path = f"_boot_{name}"
            with open(path, "wb") as f:
                f.write(content)
            envs[name] = self.env_factory.read(path)
        return envs

    # --------------------------------------------------------
    def boot(self, files: Dict[str, bytes],
             seed: int = 2024) -> BootReport:
        t0 = time.perf_counter()
        print("=" * 78)
        print("Möbius boot pipeline")
        print("=" * 78)
        print(f"  α = 1/(π − e)          = {ALPHA_SYM:.6f}")
        print(f"  a_step = α/0.3628      = {A_STEP:.6f}")
        print(f"  density 6/π²           = {DENSITY:.6f}")
        print()

        # ---- stage 1 ----
        chirp = self.stage1_chirp()
        print(f"[1] chirp")
        print(f"      K            {chirp.K}")
        print(f"      S, H, m      {chirp.S:+.4f}   "
              f"{chirp.H:.4f}   {chirp.m:.4e}")
        print(f"      t_ms         {chirp.t_ms:.2f}")

        # ---- stage 2 ----
        buf = self.stage2_buffer(seed=seed)
        print(f"[2] server buffer")
        print(f"      K            {buf.K}")
        print(f"      chunks       {buf.n_chunks}")
        print(f"      bytes        {buf.size}")
        # verify one chunk
        r0 = buf.read(0)
        if r0 is not None:
            print(f"      chunk 0 S    {r0[1]:+.4f}  "
                  f"H {r0[2]:.4f}  m {r0[3]:.4e}")

        # ---- stage 3 ----
        sym_res, asym_res = self.stage3_cpu_boot()
        print(f"[3] cpu boot")
        print(f"      symmetric    O(K log log K)")
        print(f"        cycles     {sym_res.cycles}")
        print(f"        bits       {''.join(map(str, sym_res.bits[:16]))}…")
        print(f"        S, m       {sym_res.S:+.4f}   {sym_res.m:.4e}")
        print(f"        t_ms       {sym_res.t_ms:.2f}")
        print(f"      asymmetric   O(K log K)")
        print(f"        cycles     {asym_res.cycles}")
        print(f"        bits       {''.join(map(str, asym_res.bits[:16]))}…")
        print(f"        S, m       {asym_res.S:+.4f}   {asym_res.m:.4e}")
        print(f"        t_ms       {asym_res.t_ms:.2f}")

        # ---- stage 4 ----
        envs = self.stage4_envelopes(files)
        print(f"[4] file envelopes")
        print(f"      files        {len(envs)}")
        for name, env in envs.items():
            print(f"        {name:12s}  qs={len(env.qs):>4d}  "
                  f"fp={env.fingerprint()[:16]}")
            # clean up temp file
            try:
                os.remove(env.path)
            except OSError:
                pass

        total_ms = (time.perf_counter() - t0) * 1e3
        print()
        print(f"  total boot time        {total_ms:.2f} ms")
        print()

        return BootReport(
            chirp=chirp,
            n_chunks=buf.n_chunks,
            buffer_bytes=buf.size,
            symmetric=sym_res,
            asymmetric=asym_res,
            envelopes=envs,
            files={name: len(c) for name, c in files.items()},
            total_ms=total_ms,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    files = {
        "readme.txt":
            b"hello world from the mobius boot pipeline\n",
        "runme.py":
            b"x = 6\ny = 7\nprint('boot says 6*7 =', x * y)\n",
        "config.ini":
            b"[mobius]\nK=256\nalpha=2.362338\ndensity=0.607927\n",
    }

    pipeline = MobiusBootPipeline(
        K_chirp=256, K_buffer=512,
        K_cpu_sym=256, K_cpu_asym=256, K_env=256,
    )
    report = pipeline.boot(files, seed=2024)

    # ---- usage: read / run / write envelopes ----
    print("=" * 78)
    print("Envelope usage")
    print("=" * 78)

    # recreate the files and envelopes for the demo
    envs = {}
    for name, content in files.items():
        path = f"_boot_{name}"
        with open(path, "wb") as f:
            f.write(content)
        envs[name] = pipeline.env_factory.read(path)

    # READ
    e = envs["readme.txt"]
    body = e.read()
    print(f"[read] {e.path}")
    print(f"       bytes  {len(body)}")
    print(f"       first  {body[:40]!r}")

    # RUN
    path = "_boot_runme.py"
    with open(path, "w") as f:
        f.write(files["runme.py"].decode())
    rn = pipeline.env_factory.run(path)
    print(f"[run]  {rn.path}")
    print(f"       1D path length  {len(rn.qs)}")
    print(f"       burned before   {rn._burned}")
    out = rn.run()
    print(f"       first run       {out}")
    try:
        rn.run()
    except PermissionError as exc:
        print(f"       second run      DENIED  ({exc})")

    # WRITE
    path = "_boot_config.ini"
    with open(path, "wb") as f:
        f.write(files["config.ini"])
    challenge = pipeline.broker.challenge()
    we = pipeline.env_factory.write(path, challenge)
    key = pipeline.broker.derive_write_key(we)
    we.authorize(key)
    res = we.write(key, b"[mobius]\nK=512\n")
    print(f"[write] {we.path}")
    print(f"        writes  {res['writes']}")
    print(f"        bytes   {res['bytes_written']}")

    # cleanup
    for path in ("_boot_readme.txt", "_boot_runme.py",
                 "_boot_config.ini"):
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    demo()