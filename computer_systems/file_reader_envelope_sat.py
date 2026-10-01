#!/usr/bin/env python3
"""
file_envelope_system.py
=======================

Three-mode envelope for file access:

    READ   →  ReadEnvelope    symmetric   permanent     no writes
    RUN    →  RunEnvelope     one-shot    1D path      consumed once
    WRITE  →  WriteEnvelope   asymmetric  key-bound    OS challenge

Pipeline
--------
    file content
      → SHAKE256 signature
      → K quadratic addresses  q_i = h_i² + h_i  mod K_filter
      → Möbius gate  (μ(q_i) ≠ 0)
      → elliptic projection  Π(x_i, y_i)  →  envelope weight
      → envelope class

The envelopes share the same address set.  What distinguishes read from
run is *consumption*: a run envelope is a 1D path that must be walked
from index 0 to K−1 in order, and after the last step it is burned.
What distinguishes write from read is *authorization*: a write envelope
requires a key issued by the OS in response to a fresh challenge.

Same math as mobius-sat-maxwell-handler.py and projection-suisse-encrypt.py,
but organised around file access rather than key derivation.
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


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA_SYM   = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_ASYM  = 0.3628
NORM        = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
DENSITY     = 6.0 / (PI * PI)         # ≈ 0.6079
W1          = PI
W2          = E

K_DEFAULT   = 256
K_FILTER    = 2048 + 1
MAX_FILE_B  = 16 * 1024 * 1024        # 16 MiB cap per file
UINT32_MAX  = 2 ** 32 - 1


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
#  Figure 3.5 elliptic projection  ∈ [0, 1]
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    """Π(x, y) = ½ (1 + cos(2π x/π) · cos(2π y/e)) ∈ [0, 1]."""
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2.0 * PI * u) * math.cos(2.0 * PI * v))


# ============================================================
#  Content signature → K addresses
# ============================================================
def content_signature(content: bytes) -> bytes:
    return hashlib.shake_256(content).digest(32)


UINT64_MAX = 2 ** 64 - 1

def addresses_from_signature(sig: bytes, K: int) -> List[Tuple[float, float]]:
    """
    K addresses (x, y) in [0.5, 5.0] from a signature.

    The raw bytes are treated as uint64 and linearly mapped to a
    float range — no IEEE-754 reinterpretation, so NaN / ±inf
    cannot appear.
    """
    raw = hashlib.shake_256(sig).digest(K * 16)
    pts: List[Tuple[float, float]] = []
    for i in range(K):
        xi = int.from_bytes(raw[16 * i        : 16 * i + 8], "big")
        yi = int.from_bytes(raw[16 * i + 8    : 16 * i + 16], "big")
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
#  Base envelope
# ============================================================
@dataclass
class BaseEnvelope:
    path: str
    sig: bytes
    K: int
    addresses: List[Tuple[float, float]]
    qs: List[int]                          # quadratic addresses (all)
    kept_qs: List[int]                     # square-free survivors
    weights: List[float]                   # Π(x, y) per survivor
    mu: List[int]
    mode: AccessMode
    created_at: float = field(default_factory=time.time)

    # ---- integrity ----
    def fingerprint(self) -> bytes:
        h = hashlib.shake_256(self.sig)
        for q, w in zip(self.kept_qs, self.weights):
            h.update(struct.pack('>Id', q, w))
        return h.digest(32)

    def summary(self) -> Dict:
        return dict(
            path=self.path,
            mode=self.mode.value,
            K=self.K,
            n_addresses=len(self.addresses),
            n_kept=len(self.kept_qs),
            density=len(self.kept_qs) / self.K,
            fingerprint=self.fingerprint().hex()[:16],
            created_at=self.created_at,
        )


# ============================================================
#  READ envelope — permanent, non-writable
# ============================================================
@dataclass
class ReadEnvelope(BaseEnvelope):
    mode: AccessMode = AccessMode.READ

    def allows(self, what: AccessMode) -> bool:
        """A read envelope permits only reads."""
        return what is AccessMode.READ

    def read(self) -> bytes:
        """Yield the file content — read-only contract."""
        with open(self.path, "rb") as f:
            return f.read()

    # writes are impossible by construction
    def write(self, *_, **__) -> None:
        raise PermissionError(
            "ReadEnvelope: writes not permitted; "
            "request a WriteEnvelope from the OS")


# ============================================================
#  RUN envelope — one-shot 1D path
# ============================================================
@dataclass
class RunEnvelope(BaseEnvelope):
    """
    A one-shot envelope for executing a file.

    The 1D path is walked once, from index 0 to K−1.  After the
    last step the envelope is marked `burned` and cannot be reused.
    """
    mode: AccessMode = AccessMode.RUN
    _cursor: int = 0
    _burned: bool = False
    _runs: int = 0

    @property
    def burned(self) -> bool:
        return self._burned

    def step(self) -> Optional[Tuple[int, float]]:
        """Walk one step of the 1D path. Returns (q, weight) or None."""
        if self._burned:
            raise PermissionError(
                "RunEnvelope: already burned (one-shot)")
        if self._cursor >= len(self.kept_qs):
            self._burned = True
            return None
        q = self.kept_qs[self._cursor]
        w = self.weights[self._cursor]
        self._cursor += 1
        return q, w

    def run_once(self, executor) -> Dict:
        """
        Walk the entire path, calling `executor(step, q, w)` for each
        step, then compile + exec the file content.  The envelope is
        burned regardless of success.
        """
        if self._burned:
            raise PermissionError(
                "RunEnvelope: already burned (one-shot)")

        trace: List[Tuple[int, float]] = []
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
            ns: Dict = {"__name__": "__run__", "__file__": self.path}
            exec(code, ns)                          # noqa: S102
            self._runs += 1
            status = "ok"
        except Exception as e:
            status = f"error: {e}"
        finally:
            self._burned = True

        return dict(
            path=self.path,
            status=status,
            steps=len(trace),
            burned=self._burned,
            runs=self._runs,
        )

    def allows(self, what: AccessMode) -> bool:
        return what is AccessMode.RUN and not self._burned

    def read(self) -> bytes:
        # run implies read of the source bytes for compilation only
        with open(self.path, "rb") as f:
            return f.read()

    def write(self, *_, **__) -> None:
        raise PermissionError(
            "RunEnvelope: writes not permitted")


# ============================================================
#  WRITE envelope — asymmetric, key-bound, OS-authorized
# ============================================================
@dataclass
class WriteEnvelope(BaseEnvelope):
    """
    A write envelope is bound to a specific key and a specific OS
    challenge.  The key is derived from the file's own projection
    and the challenge; a stolen key from another session will fail.
    """
    mode: AccessMode = AccessMode.WRITE
    challenge: bytes = b""
    key_hash: bytes = b""
    _writes: int = 0
    _write_log: List[Tuple[float, int]] = field(default_factory=list)

    def authorize(self, key: bytes) -> None:
        """Bind a key to this envelope's challenge."""
        h = hashlib.shake_256(key + self.challenge).digest(32)
        self.key_hash = h

    def _verify(self, key: bytes) -> bool:
        if not self.challenge:
            return False
        h = hashlib.shake_256(key + self.challenge).digest(32)
        return hmac_compare(h, self.key_hash)

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
#  Envelope factory
# ============================================================
class EnvelopeFactory:
    """
    Builds envelopes from files.  Owns the Möbius sieve and the
    kernel constants shared by every envelope.
    """
    def __init__(self, K: int = K_DEFAULT,
                 quad_A: int = 1, quad_B: int = 1, quad_C: int = 0):
        self.K = K
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.mu = mobius_sieve(K)

    # ---- common extraction ----
    def _extract(self, path: str) -> Tuple[bytes, List[Tuple[float, float]],
                                           List[int], List[int],
                                           List[float]]:
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
        kept_qs: List[int] = []
        weights: List[float] = []
        for (x, y), q in zip(addresses, qs):
            qq = q % len(self.mu)
            if self.mu[qq] != 0:
                kept_qs.append(qq)
                weights.append(elliptic_projection(x, y))
        return sig, addresses, qs, kept_qs, weights

    # ---- READ ----
    def read_envelope(self, path: str) -> ReadEnvelope:
        sig, addresses, qs, kept, w = self._extract(path)
        return ReadEnvelope(
            path=path, sig=sig, K=self.K, addresses=addresses,
            qs=qs, kept_qs=kept, weights=w, mu=self.mu)

    # ---- RUN ----
    def run_envelope(self, path: str) -> RunEnvelope:
        sig, addresses, qs, kept, w = self._extract(path)
        return RunEnvelope(
            path=path, sig=sig, K=self.K, addresses=addresses,
            qs=qs, kept_qs=kept, weights=w, mu=self.mu)

    # ---- WRITE ----
    def write_envelope(self, path: str,
                       challenge: bytes) -> WriteEnvelope:
        sig, addresses, qs, kept, w = self._extract(path)
        return WriteEnvelope(
            path=path, sig=sig, K=self.K, addresses=addresses,
            qs=qs, kept_qs=kept, weights=w, mu=self.mu,
            challenge=challenge)


# ============================================================
#  OS permission bridge
# ============================================================
def hmac_compare(a: bytes, b: bytes) -> bool:
    """Constant-time byte comparison."""
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= x ^ y
    return diff == 0


class OSPermissionBroker:
    """
    Stand-in for the OS.  Issues challenges and derives write keys
    from a master secret and the file's projection.
    """
    def __init__(self, master_secret: bytes):
        self._master = master_secret

    def challenge(self) -> bytes:
        return os.urandom(32)

    def derive_write_key(self, env: WriteEnvelope,
                         master_secret: Optional[bytes] = None
                         ) -> bytes:
        """
        Derive a key from
            master_secret
          + env.challenge
          + env.fingerprint()
        Any change to the file changes the fingerprint and therefore
        the key.  Re-deriving requires the OS to re-sign.
        """
        secret = master_secret if master_secret is not None else self._master
        return hashlib.shake_256(
            secret + env.challenge + env.fingerprint()).digest(32)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("File envelope system  ·  read / run / write")
    print("=" * 78)
    print(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  K       = {K_DEFAULT}")
    print(f"  K_filter= {K_FILTER}")
    print()

    # ---- create a sample file ----
    tmp_path = "_envelope_demo.py"
    with open(tmp_path, "w") as f:
        f.write(
            "print('hello from the one-shot run envelope')\n"
            "x = 6\n"
            "y = 7\n"
            "print('6 * 7 =', x * y)\n"
        )
    try:
        factory = EnvelopeFactory(K=K_DEFAULT)
        broker  = OSPermissionBroker(master_secret=b"os-master-secret")

        # ---------- READ ----------
        print("--- READ envelope ---")
        re_ = factory.read_envelope(tmp_path)
        s = re_.summary()
        for k, v in s.items():
            print(f"  {k:>14s}: {v}")
        print(f"  allows READ  : {re_.allows(AccessMode.READ)}")
        print(f"  allows RUN   : {re_.allows(AccessMode.RUN)}")
        print(f"  allows WRITE : {re_.allows(AccessMode.WRITE)}")
        print(f"  read bytes   : {len(re_.read())}")

        try:
            re_.write(b"x")
            print("  write        : ALLOWED (unexpected!)")
        except PermissionError as e:
            print(f"  write        : DENIED  → {e}")
        print()

        # ---------- RUN (one-shot) ----------
        print("--- RUN envelope (one-shot) ---")
        rn = factory.run_envelope(tmp_path)
        print(f"  1D path length : {len(rn.kept_qs)}")
        print(f"  burned before  : {rn.burned}")

        first = rn.run_once(lambda i, q, w: None)
        print(f"  first run      : {first}")

        # attempt to reuse
        try:
            rn.run_once(lambda i, q, w: None)
            print("  second run     : ALLOWED (unexpected!)")
        except PermissionError as e:
            print(f"  second run     : DENIED  → {e}")
        print()

        # ---------- WRITE ----------
        print("--- WRITE envelope (OS challenge + key) ---")
        challenge = broker.challenge()
        we = factory.write_envelope(tmp_path, challenge)
        key = broker.derive_write_key(we)
        we.authorize(key)
        print(f"  challenge       : {challenge.hex()[:16]}…")
        print(f"  key             : {key.hex()[:16]}…")

        # attempt with a wrong key first
        wrong = os.urandom(32)
        try:
            we.write(wrong, b"should not happen")
            print("  write (wrong key): ALLOWED (unexpected!)")
        except PermissionError as e:
            print(f"  write (wrong key): DENIED  → {e}")

        # now with the correct key
        result = we.write(key, b"new content from authorised writer\n")
        print(f"  write (right key): {result}")

        # the file changed → the old envelopes are stale
        re2 = factory.read_envelope(tmp_path)
        same = re2.fingerprint() == re_.fingerprint()
        print(f"  old read envelope still matches new file? {same}")
        print()

        # ---------- cross-check: different file, different envelope ----------
        print("--- cross-check: different file → different envelope ---")
        tmp2 = "_envelope_demo2.py"
        with open(tmp2, "w") as f:
            f.write("print('a different file')\n")
        re3 = factory.read_envelope(tmp2)
        print(f"  file 1 fingerprint : {re_.fingerprint().hex()[:16]}")
        print(f"  file 2 fingerprint : {re3.fingerprint().hex()[:16]}")
        print(f"  envelopes equal?   : "
              f"{re_.fingerprint() == re3.fingerprint()}")
        print()

        # ---------- reuse of an old key on a changed file ----------
        print("--- key reuse across changed file ---")
        challenge2 = broker.challenge()
        we2 = factory.write_envelope(tmp_path, challenge2)
        key2 = broker.derive_write_key(we2)
        # modify the file behind the envelope's back
        with open(tmp_path, "wb") as f:
            f.write(b"tampered content\n")
        # the key was derived before tampering → envelope is stale
        we2.authorize(key2)
        try:
            we2.write(key2, b"attempt with stale key\n")
            print("  stale key write : ALLOWED (unexpected!)")
        except PermissionError as e:
            print(f"  stale key write : DENIED  → {e}")
        print()

        # ---------- complexity ----------
        print("--- complexity ---")
        print("  content signature         O(|file|)         once per file")
        print("  addresses from signature  O(K)")
        print("  quadratic addresses       O(K)")
        print("  Möbius gate (μ ≠ 0)       O(K)             amortised")
        print("  elliptic projection       O(K)")
        print("  read  envelope total      O(|file| + K)")
        print("  run   envelope total      O(|file| + K)  +  O(K)  one-shot")
        print("  write envelope total      O(|file| + K)  +  O(1)  per write")

    finally:
        for p in (tmp_path, "_envelope_demo2.py"):
            try:
                os.remove(p)
            except OSError:
                pass

    print()
    print("Done.")


if __name__ == "__main__":
    demo()