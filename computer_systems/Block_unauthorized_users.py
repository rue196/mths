#!/usr/bin/env python3
"""
file_envelope_geo_sat.py
========================
Geo-fenced file envelopes.

    READ   →  ReadEnvelope    symmetric   permanent
    RUN    →  RunEnvelope     one-shot    1D path
    WRITE  →  WriteEnvelope   asymmetric  key-bound

    +  location gate:

        Local access is granted ONLY if the requester's location
        matches the CPU's location.  The match is verified by an
        O(K) Möbius-gated cross-coupling:

            sig  →  v(n) = μ(n) · cos(A_STEP · n² / K) · s(n mod L)

            cross = |⟨v_cpu, v_claim⟩| / (‖v_cpu‖ · ‖v_claim‖)

        A matching fingerprint gives cross = 1.0.
        Any other signature gives cross ≈ 1/√K (well below the
        threshold).  No N×N matrix is ever built.

        Mismatch  →  PermissionError, envelope never built.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA_SYM   = 1.0 / (PI - E)              # ≈ 2.362338
ALPHA_ASYM  = 0.3628
A_STEP      = ALPHA_SYM / ALPHA_ASYM      # ≈ 6.511
NORM        = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
DENSITY     = 6.0 / (PI * PI)             # ≈ 0.607927
W1          = PI
W2          = E

K_DEFAULT   = 256
K_FILTER    = 2048 + 1
MAX_FILE_B  = 16 * 1024 * 1024
UINT32_MAX  = 2 ** 32 - 1
UINT64_MAX  = 2 ** 64 - 1

# location-gate parameters
LOC_TOL      = 0.35                   # min cross-coupling for accept
SAT_MATCH    = 0.55                   # min SAT envelope overlap
CLAIM_TTL    = 30.0                   # seconds


# ============================================================
#  Access mode
# ============================================================
class AccessMode(Enum):
    READ  = "read"
    RUN   = "run"
    WRITE = "write"


# ============================================================
#  Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    if K < 1:
        return [0] * (K + 1)
    mu = [1] * (K + 1)
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = [-x for x in mu[i::i]]
            for j in range(i, K + 1, i):
                is_comp[j] = True
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    mu[0] = 0
    return mu


# ============================================================
#  Elliptic projection  ·  Figure 3.5
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2.0 * PI * u) * math.cos(2.0 * PI * v))


# ============================================================
#  SAT envelope  ·  q(a) = a² + a mod K,  μ(q) ≠ 0
# ============================================================
def sat_envelope(data: bytes, K: int, mu: List[int]) -> np.ndarray:
    if not data:
        return np.zeros(K, dtype=float)
    amps = np.array([b / 255.0 for b in data])
    if len(amps) < K:
        amps = np.tile(amps, int(math.ceil(K / len(amps))))
    amps = amps[:K]
    env = np.zeros(K, dtype=float)
    for a, v in enumerate(amps):
        q = (a * a + a) % K
        if 0 <= q < len(mu) and mu[q] != 0:
            env[q] += abs(v)
    return env


def sat_overlap(env_a: np.ndarray, env_b: np.ndarray) -> float:
    na = float(np.linalg.norm(env_a))
    nb = float(np.linalg.norm(env_b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(env_a, env_b) / (na * nb))


# ============================================================
#  μ-gated location vector  ·  O(K)
# ============================================================
def mu_gated_vector(sig: bytes, K: int,
                    mu: List[int]) -> np.ndarray:
    """
    Length-K vector:

        v(n) = μ(n) · cos(A_STEP · n² / K) · s(n mod L)

    where s = sig bytes normalised to [0, 1].

    · square-free slots survive (μ ≠ 0)
    · chirp modulates the phase across the slots
    · signature bytes tile across the length

    All computation is O(K) once μ is known.
    """
    if not sig or K <= 0:
        return np.zeros(K, dtype=float)

    s = np.array([b / 255.0 for b in sig], dtype=float)
    L = len(s)
    n = np.arange(K)
    mu_arr = np.array([float(mu[k]) if k < len(mu) else 0.0
                       for k in range(K)])
    chirp = np.cos(A_STEP * n * n / K)
    samp = s[n % L]
    return mu_arr * chirp * samp


def cross_coupling(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """
    Normalised cross-coupling of two μ-gated vectors, ∈ [0, 1].

        cross = |⟨v_a, v_b⟩| / (‖v_a‖ · ‖v_b‖)

    Identical vectors → 1.0
    Independent vectors → ≈ 1/√K
    """
    na = float(np.linalg.norm(v_a))
    nb = float(np.linalg.norm(v_b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.dot(v_a, v_b)) / (na * nb))


# ============================================================
#  CPU location  ·  hardware-derived, O(K) per build
# ============================================================
@dataclass
class CPULocation:
    hardware_id: bytes
    deployment_salt: bytes = b""
    K: int = K_DEFAULT

    sig: bytes = field(init=False)
    mu: List[int] = field(init=False)
    mu_vec: np.ndarray = field(init=False)
    sat_env: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mu = mobius_sieve(self.K)
        self.sig = hashlib.shake_256(
            self.hardware_id + b"|" + self.deployment_salt
        ).digest(32)
        self.mu_vec = mu_gated_vector(self.sig, self.K, self.mu)
        self.sat_env = sat_envelope(self.sig, self.K, self.mu)

    def fingerprint(self) -> str:
        return self.sig.hex()

    def summary(self) -> Dict:
        nz = int(np.count_nonzero(self.mu_vec))
        return dict(
            fingerprint=self.fingerprint()[:32],
            K=self.K,
            mu_nonzero=nz,
            mu_density=nz / self.K,
            sat_nonzero=int(np.count_nonzero(self.sat_env)),
        )


# ============================================================
#  Location claim  ·  proves knowledge of the CPU fingerprint
# ============================================================
@dataclass
class LocationClaim:
    """
    A claim is a proof of knowledge of the CPU's fingerprint.

        claimed_fingerprint : 64-hex string the requester says
                              the CPU has
        nonce               : fresh random bytes (replay protection)
        timestamp           : when the claim was made (freshness)
    """
    claimed_fingerprint: str
    nonce: bytes = field(default_factory=lambda: os.urandom(8))
    timestamp: float = field(default_factory=time.time)
    K: int = K_DEFAULT

    sig: bytes = field(init=False)
    mu: List[int] = field(init=False)
    mu_vec: np.ndarray = field(init=False)
    sat_env: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mu = mobius_sieve(self.K)
        try:
            self.sig = bytes.fromhex(self.claimed_fingerprint)
        except ValueError:
            self.sig = b""
        self.mu_vec = mu_gated_vector(self.sig, self.K, self.mu)
        self.sat_env = sat_envelope(self.sig, self.K, self.mu)

    def summary(self) -> Dict:
        return dict(
            fingerprint=self.claimed_fingerprint[:32],
            timestamp=self.timestamp,
            age_s=time.time() - self.timestamp,
            nonce=self.nonce.hex()[:16],
            sat_nonzero=int(np.count_nonzero(self.sat_env)),
        )


# ============================================================
#  Location gate  ·  DNS-style verification, O(K)
# ============================================================
@dataclass
class GateVerdict:
    allowed: bool
    cross_coupling: float
    sat_overlap: float
    reason: str
    elapsed_ms: float


class LocationGate:
    """
    Verify a claim against the CPU location.

    Three O(K) checks:

        1. fingerprint equality   (exact 64-hex match)
        2. freshness              age ≤ CLAIM_TTL
        3. μ-gated cross-coupling ≥ LOC_TOL
        4. SAT envelope overlap   ≥ SAT_MATCH

    Any failure blocks outright.  No N×N matrix is built.
    """
    def __init__(self,
                 loc_tol: float = LOC_TOL,
                 sat_match: float = SAT_MATCH,
                 ttl: float = CLAIM_TTL):
        self.loc_tol = loc_tol
        self.sat_match = sat_match
        self.ttl = ttl

    def verify(self, cpu: CPULocation,
               claim: LocationClaim) -> GateVerdict:
        t0 = time.perf_counter()

        # ---- 0. basic validity ----
        if len(claim.sig) == 0:
            return GateVerdict(
                allowed=False, cross_coupling=0.0, sat_overlap=0.0,
                reason="malformed claim fingerprint",
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
            )

        # ---- 1. fingerprint equality ----
        if claim.claimed_fingerprint != cpu.fingerprint():
            return GateVerdict(
                allowed=False, cross_coupling=0.0, sat_overlap=0.0,
                reason="fingerprint mismatch",
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
            )

        # ---- 2. freshness ----
        age = time.time() - claim.timestamp
        if age > self.ttl:
            return GateVerdict(
                allowed=False, cross_coupling=0.0, sat_overlap=0.0,
                reason=f"stale claim ({age:.1f}s > {self.ttl}s)",
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
            )

        # ---- 3. μ-gated cross-coupling  (O(K)) ----
        cc = cross_coupling(cpu.mu_vec, claim.mu_vec)

        # ---- 4. SAT envelope overlap  (O(K)) ----
        ov = sat_overlap(cpu.sat_env, claim.sat_env)

        if cc < self.loc_tol:
            return GateVerdict(
                allowed=False, cross_coupling=cc, sat_overlap=ov,
                reason=f"cross-coupling {cc:.3f} < {self.loc_tol}",
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
            )
        if ov < self.sat_match:
            return GateVerdict(
                allowed=False, cross_coupling=cc, sat_overlap=ov,
                reason=f"SAT overlap {ov:.3f} < {self.sat_match}",
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
            )

        return GateVerdict(
            allowed=True, cross_coupling=cc, sat_overlap=ov,
            reason="match",
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  Base envelope
# ============================================================
@dataclass
class BaseEnvelope:
    path: str
    sig: bytes
    K: int
    addresses: List[Tuple[float, float]]
    qs: List[int]
    kept_qs: List[int]
    weights: List[float]
    mu: List[int]
    mode: AccessMode
    cpu_fingerprint: str = ""
    claim_summary: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def fingerprint(self) -> bytes:
        h = hashlib.shake_256(self.sig)
        for q, w in zip(self.kept_qs, self.weights):
            h.update(struct.pack('>Id', q, w))
        h.update(self.cpu_fingerprint.encode())
        h.update(str(self.claim_summary.get("timestamp", 0.0)).encode())
        return h.digest(32)

    def summary(self) -> Dict:
        return dict(
            path=self.path,
            mode=self.mode.value,
            K=self.K,
            n_kept=len(self.kept_qs),
            density=len(self.kept_qs) / self.K,
            fingerprint=self.fingerprint().hex()[:16],
            cpu=self.cpu_fingerprint[:16],
            created_at=self.created_at,
        )


# ============================================================
#  READ envelope
# ============================================================
@dataclass
class ReadEnvelope(BaseEnvelope):
    mode: AccessMode = AccessMode.READ

    def allows(self, what: AccessMode) -> bool:
        return what is AccessMode.READ

    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def write(self, *_, **__) -> None:
        raise PermissionError(
            "ReadEnvelope: writes not permitted")


# ============================================================
#  RUN envelope  ·  one-shot 1D path
# ============================================================
@dataclass
class RunEnvelope(BaseEnvelope):
    mode: AccessMode = AccessMode.RUN
    _cursor: int = 0
    _burned: bool = False
    _runs: int = 0

    @property
    def burned(self) -> bool:
        return self._burned

    def step(self) -> Optional[Tuple[int, float]]:
        if self._burned:
            raise PermissionError("RunEnvelope: already burned")
        if self._cursor >= len(self.kept_qs):
            self._burned = True
            return None
        q = self.kept_qs[self._cursor]
        w = self.weights[self._cursor]
        self._cursor += 1
        return q, w

    def run_once(self, executor) -> Dict:
        if self._burned:
            raise PermissionError("RunEnvelope: already burned")
        trace = []
        try:
            while True:
                s = self.step()
                if s is None:
                    break
                q, w = s
                trace.append((q, w))
                executor(len(trace) - 1, q, w)
            with open(self.path, "rb") as f:
                source = f.read()
            code = compile(source, self.path, "exec")
            ns = {"__name__": "__run__", "__file__": self.path}
            exec(code, ns)                                  # noqa: S102
            self._runs += 1
            status = "ok"
        except Exception as e:
            status = f"error: {e}"
        finally:
            self._burned = True
        return dict(path=self.path, status=status,
                    steps=len(trace), burned=self._burned,
                    runs=self._runs)

    def allows(self, what: AccessMode) -> bool:
        return what is AccessMode.RUN and not self._burned

    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def write(self, *_, **__) -> None:
        raise PermissionError("RunEnvelope: writes not permitted")


# ============================================================
#  WRITE envelope
# ============================================================
def _hmac_compare(a: bytes, b: bytes) -> bool:
    if len(a) != len(b):
        return False
    d = 0
    for x, y in zip(a, b):
        d |= x ^ y
    return d == 0


@dataclass
class WriteEnvelope(BaseEnvelope):
    mode: AccessMode = AccessMode.WRITE
    challenge: bytes = b""
    key_hash: bytes = b""
    _writes: int = 0
    _write_log: List[Tuple[float, int]] = field(default_factory=list)

    def authorize(self, key: bytes) -> None:
        self.key_hash = hashlib.shake_256(
            key + self.challenge).digest(32)

    def _verify(self, key: bytes) -> bool:
        if not self.challenge:
            return False
        h = hashlib.shake_256(key + self.challenge).digest(32)
        return _hmac_compare(h, self.key_hash)

    def write(self, key: bytes, data: bytes) -> Dict:
        if not self._verify(key):
            raise PermissionError(
                "WriteEnvelope: invalid key for this challenge")
        with open(self.path, "wb") as f:
            f.write(data)
        self._writes += 1
        self._write_log.append((time.time(), len(data)))
        return dict(path=self.path,
                    bytes_written=len(data),
                    writes=self._writes)

    def allows(self, what: AccessMode) -> bool:
        return what in (AccessMode.READ, AccessMode.WRITE)

    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()


# ============================================================
#  Address extraction
# ============================================================
def content_signature(content: bytes) -> bytes:
    return hashlib.shake_256(content).digest(32)


def addresses_from_signature(sig: bytes, K: int
                             ) -> List[Tuple[float, float]]:
    raw = hashlib.shake_256(sig).digest(K * 16)
    pts = []
    for i in range(K):
        xi = int.from_bytes(raw[16 * i:16 * i + 8], "big")
        yi = int.from_bytes(raw[16 * i + 8:16 * i + 16], "big")
        x = 0.5 + (xi / UINT64_MAX) * 4.5
        y = 0.5 + (yi / UINT64_MAX) * 4.5
        pts.append((x, y))
    return pts


def quadratic_address(x: float, y: float, K: int,
                      A: int = 1, B: int = 1, C: int = 0) -> int:
    h = int.from_bytes(
        hashlib.shake_256(struct.pack('>dd', x, y)).digest(8), 'big')
    return (A * h * h + B * h + C) % K


# ============================================================
#  Geo-fenced envelope factory
# ============================================================
class GeoFencedEnvelopeFactory:
    def __init__(self,
                 cpu_location: CPULocation,
                 gate: Optional[LocationGate] = None,
                 K: int = K_DEFAULT,
                 quad_A: int = 1, quad_B: int = 1, quad_C: int = 0):
        self.cpu = cpu_location
        self.gate = gate or LocationGate()
        self.K = K
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.mu = mobius_sieve(K)
        self.audit: List[Dict] = []

    def _gate(self, claim: LocationClaim) -> GateVerdict:
        v = self.gate.verify(self.cpu, claim)
        self.audit.append(dict(
            t=time.time(),
            allowed=v.allowed,
            reason=v.reason,
            cross=v.cross_coupling,
            sat=v.sat_overlap,
            cpu=self.cpu.fingerprint()[:16],
        ))
        return v

    def _extract(self, path: str):
        with open(path, "rb") as f:
            content = f.read()
        if len(content) > MAX_FILE_B:
            raise ValueError(
                f"file size {len(content)} exceeds MAX_FILE_B")
        sig = content_signature(content)
        addresses = addresses_from_signature(sig, self.K)
        qs = [quadratic_address(x, y, K_FILTER,
                                self.A, self.B, self.C)
              for x, y in addresses]
        kept_qs, weights = [], []
        for (x, y), q in zip(addresses, qs):
            qq = q % len(self.mu)
            if self.mu[qq] != 0:
                kept_qs.append(qq)
                weights.append(elliptic_projection(x, y))
        return sig, addresses, qs, kept_qs, weights

    def _base(self, path: str, claim: LocationClaim):
        v = self._gate(claim)
        if not v.allowed:
            raise PermissionError(
                f"LocationGate: {v.reason} "
                f"(cross={v.cross_coupling:.3f}, "
                f"sat={v.sat_overlap:.3f})")
        sig, addresses, qs, kept, w = self._extract(path)
        return sig, addresses, qs, kept, w

    def read_envelope(self, path: str,
                      claim: LocationClaim) -> ReadEnvelope:
        sig, addresses, qs, kept, w = self._base(path, claim)
        return ReadEnvelope(
            path=path, sig=sig, K=self.K, addresses=addresses,
            qs=qs, kept_qs=kept, weights=w, mu=self.mu,
            cpu_fingerprint=self.cpu.fingerprint(),
            claim_summary=claim.summary())

    def run_envelope(self, path: str,
                     claim: LocationClaim) -> RunEnvelope:
        sig, addresses, qs, kept, w = self._base(path, claim)
        return RunEnvelope(
            path=path, sig=sig, K=self.K, addresses=addresses,
            qs=qs, kept_qs=kept, weights=w, mu=self.mu,
            cpu_fingerprint=self.cpu.fingerprint(),
            claim_summary=claim.summary())

    def write_envelope(self, path: str,
                       claim: LocationClaim,
                       challenge: bytes) -> WriteEnvelope:
        sig, addresses, qs, kept, w = self._base(path, claim)
        return WriteEnvelope(
            path=path, sig=sig, K=self.K, addresses=addresses,
            qs=qs, kept_qs=kept, weights=w, mu=self.mu,
            cpu_fingerprint=self.cpu.fingerprint(),
            claim_summary=claim.summary(),
            challenge=challenge)

    def summary(self) -> Dict:
        n_allowed = sum(1 for e in self.audit if e["allowed"])
        return dict(
            cpu=self.cpu.fingerprint()[:16],
            n_requests=len(self.audit),
            n_allowed=n_allowed,
            n_blocked=len(self.audit) - n_allowed,
        )


# ============================================================
#  OS permission broker
# ============================================================
class OSPermissionBroker:
    def __init__(self, master_secret: bytes):
        self._master = master_secret

    def challenge(self) -> bytes:
        return os.urandom(32)

    def derive_write_key(self, env: WriteEnvelope,
                         master_secret: Optional[bytes] = None) -> bytes:
        secret = master_secret if master_secret is not None else self._master
        return hashlib.shake_256(
            secret + env.challenge + env.fingerprint()).digest(32)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Geo-fenced file envelopes  ·  O(K) Möbius-gated location gate")
    print("=" * 80)
    print(f"  α = 1/(π − e)     = {ALPHA_SYM:.6f}")
    print(f"  A_STEP            = {A_STEP:.6f}")
    print(f"  K (envelope)      = {K_DEFAULT}")
    print(f"  LOC_TOL           = {LOC_TOL}")
    print(f"  SAT_MATCH         = {SAT_MATCH}")
    print(f"  CLAIM_TTL         = {CLAIM_TTL}s")
    print()

    # ---------- CPU establishes its own location ----------
    print("--- CPU location (hardware-derived, O(K) build) ---")
    cpu = CPULocation(
        hardware_id=b"CPU-SERIAL-4A7F-9B12-0001",
        deployment_salt=b"dep-loc-alpha",
        K=K_DEFAULT,
    )
    s = cpu.summary()
    for k, v in s.items():
        print(f"  {k:>14s}: {v}")
    print()

     # ---------- claims ----------
    print("--- claims ---")
    local_claim = LocationClaim(
        claimed_fingerprint=cpu.fingerprint(),
        nonce=os.urandom(8),
        K=K_DEFAULT,
    )
    remote_claim = LocationClaim(
        claimed_fingerprint=os.urandom(32).hex(),
        nonce=os.urandom(8),
        K=K_DEFAULT,
    )
    spoofed_claim = LocationClaim(
        claimed_fingerprint=hashlib.shake_256(
            b"CPU-SERIAL-4A7F-9B12-0002|dep-loc-alpha"
        ).hexdigest(32),                       
        nonce=os.urandom(8),
        K=K_DEFAULT,
    )
    stale_claim = LocationClaim(
        claimed_fingerprint=cpu.fingerprint(),
        nonce=os.urandom(8),
        timestamp=time.time() - 3600.0,
        K=K_DEFAULT,
    )
    print(f"  local   fp        {local_claim.claimed_fingerprint[:32]}…")
    print(f"  remote  fp        {remote_claim.claimed_fingerprint[:32]}…")
    print(f"  spoofed fp        {spoofed_claim.claimed_fingerprint[:32]}…")
    print(f"  stale   age       "
          f"{time.time() - stale_claim.timestamp:.0f}s")
    print()

    # ---------- factory ----------
    factory = GeoFencedEnvelopeFactory(
        cpu_location=cpu, K=K_DEFAULT)

    # ---------- sample file ----------
    tmp_path = "_geo_demo.py"
    with open(tmp_path, "w") as f:
        f.write(
            "print('geo-fenced access granted')\n"
            "x = 6\n"
            "y = 7\n"
            "print('6 * 7 =', x * y)\n"
        )

    try:
        broker = OSPermissionBroker(master_secret=b"os-master-secret")

        # ============================================================
        #  Scenario 1  ·  local claim  →  all envelopes granted
        # ============================================================
        print("=" * 80)
        print("scenario 1  ·  local claim")
        print("=" * 80)
        v = factory.gate.verify(cpu, local_claim)
        print(f"  gate verdict      allowed={v.allowed}")
        print(f"  cross-coupling    {v.cross_coupling:.6f}")
        print(f"  SAT overlap       {v.sat_overlap:.6f}")
        print(f"  reason            {v.reason}")
        print(f"  elapsed           {v.elapsed_ms:.3f} ms")
        print()

        re_ = factory.read_envelope(tmp_path, local_claim)
        print(f"  READ  envelope    fp={re_.fingerprint().hex()[:16]}  "
              f"kept={len(re_.kept_qs)}")
        print(f"        bytes       {len(re_.read())}")

        rn = factory.run_envelope(tmp_path, local_claim)
        print(f"  RUN   envelope    fp={rn.fingerprint().hex()[:16]}  "
              f"1D path={len(rn.kept_qs)}")

        challenge = broker.challenge()
        we = factory.write_envelope(tmp_path, local_claim, challenge)
        key = broker.derive_write_key(we)
        we.authorize(key)
        print(f"  WRITE envelope    fp={we.fingerprint().hex()[:16]}  "
              f"key={key.hex()[:16]}…")
        print()

        # ============================================================
        #  Scenarios 2–4  ·  invalid claims  →  BLOCKED
        # ============================================================
        for name, claim in [
            ("remote (random fingerprint)", remote_claim),
            ("spoofed (near-miss serial)",  spoofed_claim),
            ("stale (1h old)",              stale_claim),
        ]:
            print("=" * 80)
            print(f"scenario  ·  {name}")
            print("=" * 80)
            v = factory.gate.verify(cpu, claim)
            print(f"  gate verdict      allowed={v.allowed}")
            print(f"  cross-coupling    {v.cross_coupling:.6f}")
            print(f"  SAT overlap       {v.sat_overlap:.6f}")
            print(f"  reason            {v.reason}")
            for fn, label in [
                (factory.read_envelope, "READ"),
                (factory.run_envelope,  "RUN"),
            ]:
                try:
                    fn(tmp_path, claim)
                    print(f"  {label:6s}            "
                          f"ALLOWED (unexpected!)")
                except PermissionError as exc:
                    print(f"  {label:6s}            "
                          f"DENIED  → {exc}")
            try:
                factory.write_envelope(tmp_path, claim, challenge)
                print("  WRITE             ALLOWED (unexpected!)")
            except PermissionError as exc:
                print(f"  WRITE             DENIED  → {exc}")
            print()

        # ============================================================
        #  Scenario 5  ·  DNS-style verification table
        # ============================================================
        print("=" * 80)
        print("scenario 5  ·  DNS-style claim verification  (O(K) gate)")
        print("=" * 80)
        print(f"  CPU fingerprint              {cpu.fingerprint()[:32]}…")
        print(f"    {'claim':28s}  {'cross':>9s}  {'sat':>9s}  "
              f"{'age':>8s}  {'verdict':>10s}")
        test_claims = [
            ("local (matches)",  local_claim),
            ("remote IP",        remote_claim),
            ("spoofed serial",   spoofed_claim),
            ("stale (1h old)",   stale_claim),
        ]
        for name, claim in test_claims:
            v = factory.gate.verify(cpu, claim)
            age = time.time() - claim.timestamp
            verdict = "ALLOW" if v.allowed else "DENY"
            print(f"    {name:28s}  "
                  f"{v.cross_coupling:>9.6f}  "
                  f"{v.sat_overlap:>9.6f}  "
                  f"{age:>7.1f}s  "
                  f"{verdict:>10s}")
        print()

        # ============================================================
        #  Timing comparison
        # ============================================================
        print("--- gate timing ---")
        n_iter = 200
        t0 = time.perf_counter()
        for _ in range(n_iter):
            factory.gate.verify(cpu, local_claim)
        dt = (time.perf_counter() - t0) * 1e3 / n_iter
        print(f"  O(K) gate          {dt:.4f} ms per verify  "
              f"({n_iter} samples)")
        print(f"  K                  {K_DEFAULT}")
        print()

        # ============================================================
        #  Audit log
        # ============================================================
        print("=" * 80)
        print("factory audit log")
        print("=" * 80)
        s = factory.summary()
        print(f"  CPU        {s['cpu']}")
        print(f"  requests   {s['n_requests']}")
        print(f"  allowed    {s['n_allowed']}")
        print(f"  blocked    {s['n_blocked']}")
        print()

        # ============================================================
        #  Complexity
        # ============================================================
        print("--- complexity ---")
        print("  CPU location build        O(K log log K) + O(K)")
        print("  claim decode              O(K log log K) + O(K)")
        print("  μ-gated vector            O(K)")
        print("  cross-coupling            O(K)")
        print("  SAT envelope              O(K)")
        print("  SAT overlap               O(K)")
        print("  gate verdict total        O(K)")
        print()
        print(f"  with K={K_DEFAULT}: "
              f"≈ {4 * K_DEFAULT} ops per gate check")
        print("  (no N×N matrix — replaced by μ(n)-gated length-K vectors)")
        print()

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    print("Done.")


if __name__ == "__main__":
    demo()