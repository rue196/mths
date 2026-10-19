#!/usr/bin/env python3
"""
mock_transaction_system.py
==========================

Card = quadratic-address SAT envelope.
Server = two-stage verifier:

    STAGE 1  ·  symmetric  (deterministic)              O(K log log K)
        sieve once, then O(1)/O(K) per check
        · card-holder address fingerprint (hex-DNS)
        · frozen flag (boolean)
        · expiration compare (timestamp)
        · balance sufficiency (deterministic)
        · nonce replay cache (set lookup)

    STAGE 2  ·  asymmetric (dynamic)                    O(K log K)
        inverse-score merge sort
        · envelope similarity vs server reconstruction
        · chiral-bond weight  w_ij = |χ_i − χ_j| / π

    GATE     ·  SAT survivor  (if/else)                 O(K)
        SAT envelope of the card survives iff
            μ(q(a)) ≠ 0  ∧  cross-coupling ≥ τ  ∧  σ_card = σ_server

        if all three → APPROVE
        else         → DECLINE

Transport
---------
    finite-step packet  [s0, d0, …, d_{N-1}],  a = α / 0.3628
    MOSB frame          MAGIC | K | n | cid | S | H | m
    hex-DNS address     fingerprint(domain)
"""

from __future__ import annotations
import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)                # ≈ 2.362
A_STEP   = ALPHA / 0.3628                # ≈ 6.511
DENSITY  = 6.0 / (PI * PI)               # ≈ 0.607927
TAU      = 0.35                          # SAT threshold
UGC_K    = 4                             # label resolution

MAGIC       = b"TXSB"
HEADER_FMT  = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


# ============================================================
#  Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Inverse-score merge sort  ·  O(K log K)
# ============================================================
def _merge_count(arr, tmp, l, m, r) -> int:
    i, j, k, inv = l, m + 1, l, 0
    while i <= m and j <= r:
        if arr[i] <= arr[j]:
            tmp[k] = arr[i]; i += 1
        else:
            tmp[k] = arr[j]; inv += m - i + 1; j += 1
        k += 1
    while i <= m: tmp[k] = arr[i]; i += 1; k += 1
    while j <= r: tmp[k] = arr[j]; j += 1; k += 1
    for i in range(l, r + 1):
        arr[i] = tmp[i]
    return inv


def _merge_sort_inv(arr, tmp, l, r) -> int:
    inv = 0
    if l < r:
        m = (l + r) // 2
        inv += _merge_sort_inv(arr, tmp, l, m)
        inv += _merge_sort_inv(arr, tmp, m + 1, r)
        inv += _merge_count(arr, tmp, l, m, r)
    return inv


def inversion_count(arr) -> int:
    n = len(arr)
    if n < 2:
        return 0
    return _merge_sort_inv(list(arr), [0] * n, 0, n - 1)


def inverse_score(a, b) -> float:
    """
    Returns 0 (identical ordering) … 1 (reversed ordering).
    Symmetric similarity = 1 − inverse_score.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.5
    pairs = sorted(zip(a, b), key=lambda p: p[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    mx = K * (K - 1) // 2
    return inv / mx if mx > 0 else 0.0


# ============================================================
#  SAT envelope  ·  q(a) = a² + a mod K,  μ(q) ≠ 0
# ============================================================
def build_sat_envelope(K: int, mu: np.ndarray,
                       address_bytes: bytes,
                       frozen: bool,
                       expiry_ts: float,
                       nonce: bytes,
                       amount: float) -> np.ndarray:
    """
    Card's quadratic-address SAT envelope.

        q(a) = a² + a mod K,  μ(q) ≠ 0

    Each field deposits amplitude at its own q-addresses:

        address    → bytes tiled across q(a) for a ∈ [0, |addr|)
        frozen     → global gate  (all-frozen = 0 amplitude)
        expiry     → contributes a gradient to the last 8 slots
        nonce      → high-frequency modulation
        amount     → scaled linear contribution
    """
    env = np.zeros(K, dtype=float)
    gate = 0.0 if frozen else 1.0

    # --- address bytes ---
    for i, b in enumerate(address_bytes):
        q = (i * i + i) % K
        if mu[q] != 0:
            env[q] += gate * (b / 255.0)

    # --- expiry gradient (later = cheaper) ---
    # normalize expiry relative to a fixed reference
    expiry_norm = (expiry_ts % 10000.0) / 10000.0
    for i in range(8):
        a = K - 1 - i
        q = (a * a + a) % K
        if mu[q] != 0:
            env[q] += gate * 0.4 * expiry_norm * (1.0 - i / 8.0)

    # --- nonce high-frequency ---
    for i, b in enumerate(nonce):
        q = (i * 7 + 3) % K
        if mu[q] != 0:
            env[q] += gate * 0.15 * (b / 255.0)

    # --- amount (scaled) ---
    amount_scaled = math.tanh(amount / 1000.0)
    for i in range(4):
        a = 100 + i * 11
        q = (a * a + a) % K
        if mu[q] != 0:
            env[q] += gate * 0.2 * amount_scaled

    return env


# ============================================================
#  Cross-coupling + UGC label  ·  O(K)
# ============================================================
def cross_coupling(va: np.ndarray, vb: np.ndarray) -> float:
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.dot(va, vb)) / (na * nb))


def ugc_label(phase: float, k: int = UGC_K) -> int:
    return int(round(k * phase / (2 * PI))) % k


def envelope_phase(env: np.ndarray) -> float:
    """Aggregate phase of an envelope via the parity-weighted sum."""
    s = 0.0
    for i, v in enumerate(env):
        s += float(v) if (i % 2 == 0) else -float(v)
    return math.atan2(s, float(np.sum(np.abs(env))) + 1e-12)


# ============================================================
#  Finite-step packet + MOSB frame  (from mobius_internet.py)
# ============================================================
class FiniteStepPacket:
    a_step = A_STEP

    def encode(self, signal: np.ndarray) -> List[float]:
        if len(signal) < 2:
            raise ValueError("signal must have ≥ 2 samples")
        s0 = float(signal[0])
        d = [(float(signal[n + 1]) - float(signal[n])) / self.a_step
             for n in range(len(signal) - 1)]
        return [s0] + d

    def decode(self, packet: List[float]) -> np.ndarray:
        recon = [packet[0]]
        for dv in packet[1:]:
            recon.append(recon[-1] + self.a_step * dv)
        return np.array(recon)


@dataclass
class Frame:
    cid: int
    payload: bytes
    S: float
    H: float
    m: float
    bytes_: bytes


class MobiusTransport:
    def __init__(self, K: int = 256):
        self.K = K
        self._cid = 0

    @staticmethod
    def _supertrace(data: bytes) -> float:
        S = 0.0
        for t, b in enumerate(data):
            v = (b - 128) / 128.0
            S += v if (t % 2 == 0) else -v
        return S

    @staticmethod
    def _entropy_mass(S: float, n: int):
        if S == 0 or n <= 0:
            return 0.0, 0.0
        p = min(abs(S) / n, 0.9999)
        H = -ALPHA * p * math.log(p)
        return H, abs(S) * math.exp(-H)

    def frame(self, payload: bytes) -> Frame:
        n = len(payload)
        S = self._supertrace(payload)
        H, m = self._entropy_mass(S, n)
        cid = self._cid
        self._cid += 1
        hdr = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        return Frame(cid, payload, S, H, m, hdr + payload)

    def deframe(self, raw: bytes) -> Tuple[bytes, Frame]:
        if len(raw) < HEADER_SIZE:
            raise ValueError("frame too short")
        magic, K, n, cid, S, H, m = struct.unpack(
            HEADER_FMT, raw[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if K != self.K:
            raise ValueError(f"K mismatch: {K} != {self.K}")
        payload = raw[HEADER_SIZE:HEADER_SIZE + n]
        if len(payload) != n:
            raise ValueError("payload length mismatch")
        S2 = self._supertrace(payload)
        if abs(S2 - S) > 1e-9:
            raise ValueError(f"supertrace mismatch: {S2} != {S}")
        return payload, Frame(cid, payload, S, H, m, raw)


# ============================================================
#  Card  ·  quadratic SAT envelope
# ============================================================
@dataclass
class Card:
    holder: str
    address: str
    frozen: bool = False
    expiry_ts: float = 0.0
    balance: float = 1000.0
    K: int = 128

    mu:       np.ndarray = field(init=False)
    nonce:    bytes = field(init=False)
    envelope: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mu = mobius_sieve(self.K)
        self.nonce = hashlib.shake_256(
            (self.holder + self.address).encode()).digest(16)
        self.envelope = build_sat_envelope(
            self.K, self.mu,
            address_bytes=self.address.encode("utf-8"),
            frozen=self.frozen,
            expiry_ts=self.expiry_ts,
            nonce=self.nonce,
            amount=self.balance,
        )

    # ---------- fingerprint ----------
    def fingerprint(self) -> str:
        return hashlib.shake_256(
            self.address.encode("utf-8")
            + struct.pack(">d", self.expiry_ts)
            + struct.pack(">d", self.balance)
        ).hexdigest(32)

    # ---------- build a transaction packet ----------
    def authorize(self, amount: float) -> Dict:
        """Card builds the packet to send to the server."""
        packet = FiniteStepPacket()
        pk = packet.encode(self.envelope)
        payload = struct.pack(f">{len(pk)}d", *pk)
        return {
            "holder": self.holder,
            "address": self.address,
            "frozen": self.frozen,
            "expiry_ts": self.expiry_ts,
            "amount": amount,
            "nonce": self.nonce,
            "envelope": self.envelope,
            "payload": payload,
            "fingerprint": self.fingerprint(),
        }


# ============================================================
#  Server  ·  two-stage verifier + SAT gate
# ============================================================
@dataclass
class Verdict:
    approved: bool
    stage1_ok: bool
    stage2_similarity: float
    sat_cross: float
    sat_labels_match: bool
    sat_survived: bool
    reason: str
    elapsed_ms: float
    stage1_ms: float
    stage2_ms: float


class TransactionServer:
    """
    Server runs:

        STAGE 1  O(K log log K)   (sieve once; O(1)/O(K) per check)
            fingerprint match
            frozen check
            expiry compare
            balance check
            nonce replay check

        STAGE 2  O(K log K)
            inverse_score(card_env, server_reconstruction)
            chiral bond weight w_ij

        GATE     O(K)
            cross_coupling ≥ τ  ∧  σ_card = σ_server  →  APPROVE
            else                                       →  DECLINE
    """

    def __init__(self, K: int = 128, tau: float = TAU):
        self.K = K
        self.tau = tau
        self.mu = mobius_sieve(K)                       # O(K log log K)
        self.transport = MobiusTransport(K=K)
        self.dns: Dict[str, Dict] = {}                  # fingerprint → record
        self.seen_nonces: set = set()
        self.ledger: Dict[str, float] = {}

    # --------------------------------------------------------
    def register(self, holder: str, address: str,
                 balance: float, expiry_ts: float) -> str:
        """Register a card-holder on the server."""
        fp = hashlib.shake_256(
            address.encode("utf-8")
            + struct.pack(">d", expiry_ts)
            + struct.pack(">d", balance)
        ).hexdigest(32)
        self.dns[fp] = dict(
            holder=holder, address=address,
            balance=balance, expiry_ts=expiry_ts,
            frozen=False,
        )
        self.ledger[holder] = balance
        return fp

    # --------------------------------------------------------
    def freeze(self, fp: str) -> bool:
        if fp in self.dns:
            self.dns[fp]["frozen"] = True
            return True
        return False

    # --------------------------------------------------------
    #  STAGE 1  ·  symmetric (deterministic)  O(K log log K) total
    # --------------------------------------------------------
    def _stage1(self, req: Dict, now: float) -> Tuple[bool, str]:
        # fingerprint match  ·  O(1)
        if req["fingerprint"] not in self.dns:
            return False, "unknown card (fingerprint mismatch)"
        record = self.dns[req["fingerprint"]]

        # frozen flag  ·  O(1)
        if record["frozen"]:
            return False, "card is frozen"
        if req["frozen"]:
            return False, "card asserts frozen"

        # expiry compare  ·  O(1)
        if now > record["expiry_ts"]:
            return False, f"expired at {record['expiry_ts']:.0f}"
        if now > req["expiry_ts"]:
            return False, "card reports expired"

        # balance check  ·  O(1)
        if req["amount"] > record["balance"]:
            return False, (f"insufficient: "
                           f"{req['amount']:.2f} > {record['balance']:.2f}")

        # nonce replay  ·  O(1)
        nonce_hex = req["nonce"].hex()
        if nonce_hex in self.seen_nonces:
            return False, "replay detected (nonce reused)"
        self.seen_nonces.add(nonce_hex)

        # address fingerprint match  ·  O(1)
        if record["address"] != req["address"]:
            return False, "holder address mismatch"

        return True, "ok"

    # --------------------------------------------------------
    #  STAGE 2  ·  asymmetric  O(K log K)
    # --------------------------------------------------------
    def _stage2(self, req: Dict, now: float) -> Tuple[float, float]:
        """
        Rebuild the server's expected envelope from the record and
        compare against the card's envelope.
        """
        record = self.dns[req["fingerprint"]]
        server_env = build_sat_envelope(
            self.K, self.mu,
            address_bytes=record["address"].encode("utf-8"),
            frozen=record["frozen"],
            expiry_ts=record["expiry_ts"],
            nonce=req["nonce"],
            amount=record["balance"],
        )

        # inverse-score similarity  ·  O(K log K)
        inv = inverse_score(
            req["envelope"].tolist(), server_env.tolist())
        similarity = 1.0 - inv

        # chiral bond weight  ·  O(K)
        sum_chi = 0.0
        for i in range(self.K):
            chi = math.log(max(i + 1, 1)) % (2 * PI)
            sum_chi += (chi * req["envelope"][i]) ** 2
        w_ij = min(1.0, math.sqrt(sum_chi / self.K))

        return similarity, w_ij

    # --------------------------------------------------------
    #  GATE  ·  SAT survivor  O(K)
    # --------------------------------------------------------
    def _gate(self, req: Dict, similarity: float) -> Tuple[
            bool, float, bool, bool, str]:
        """
        Final if/else: SAT envelope survives?
        """
        # 1. cross-coupling ≥ τ
        # The server reconstructs the reference envelope and
        # measures the cross-coupling.
        record = self.dns[req["fingerprint"]]
        server_env = build_sat_envelope(
            self.K, self.mu,
            address_bytes=record["address"].encode("utf-8"),
            frozen=record["frozen"],
            expiry_ts=record["expiry_ts"],
            nonce=req["nonce"],
            amount=record["balance"],
        )
        cross = cross_coupling(req["envelope"], server_env)

        # 2. σ_card = σ_server
        phi_card   = envelope_phase(req["envelope"])
        phi_server = envelope_phase(server_env)
        lab_card   = ugc_label(phi_card, UGC_K)
        lab_server = ugc_label(phi_server, UGC_K)
        labels_match = (lab_card == lab_server)

        # 3. μ(q) ≠ 0 check (already enforced in build_sat_envelope)
        #    additional: check that the envelope is not empty
        nz = int(np.count_nonzero(req["envelope"]))
        mu_ok = nz > 0

        # if/else
        if cross >= self.tau and labels_match and mu_ok:
            return True, cross, labels_match, True, \
                f"SAT survived (cross={cross:.3f}, σ={lab_card})"
        if not mu_ok:
            return False, cross, labels_match, False, \
                "empty envelope (μ(q)=0 everywhere)"
        if not labels_match:
            return False, cross, labels_match, False, \
                f"label mismatch (card σ={lab_card}, server σ={lab_server})"
        return False, cross, labels_match, False, \
            f"cross-coupling {cross:.3f} < τ={self.tau}"

    # --------------------------------------------------------
    def process(self, req: Dict,
                now: Optional[float] = None) -> Verdict:
        """
        Full transaction pipeline.
        """
        t0 = time.perf_counter()
        now = now if now is not None else time.time()

        # ---- STAGE 1  ·  O(K log log K) ----
        t_s1 = time.perf_counter()
        stage1_ok, s1_reason = self._stage1(req, now)
        stage1_ms = (time.perf_counter() - t_s1) * 1e3
        if not stage1_ok:
            return Verdict(
                approved=False, stage1_ok=False,
                stage2_similarity=0.0,
                sat_cross=0.0, sat_labels_match=False,
                sat_survived=False, reason=s1_reason,
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
                stage1_ms=stage1_ms, stage2_ms=0.0,
            )

        # ---- STAGE 2  ·  O(K log K) ----
        t_s2 = time.perf_counter()
        similarity, _w = self._stage2(req, now)
        stage2_ms = (time.perf_counter() - t_s2) * 1e3

        # ---- GATE  ·  O(K) if/else ----
        survived, cross, labels_match, gate_ok, gate_reason = \
            self._gate(req, similarity)

        # ---- apply (debit) if approved ----
        if gate_ok:
            self.dns[req["fingerprint"]]["balance"] -= req["amount"]
            self.ledger[req["holder"]] = \
                self.dns[req["fingerprint"]]["balance"]

        return Verdict(
            approved=gate_ok,
            stage1_ok=True,
            stage2_similarity=similarity,
            sat_cross=cross,
            sat_labels_match=labels_match,
            sat_survived=survived,
            reason=gate_reason,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
            stage1_ms=stage1_ms,
            stage2_ms=stage2_ms,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Mock transaction system  ·  quadratic SAT envelope card")
    print("=" * 78)
    print(f"  α        = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  a_step   = α / 0.3628   = {A_STEP:.6f}")
    print(f"  6/π²                    = {DENSITY:.6f}")
    print(f"  τ (SAT)                 = {TAU}")
    print()

    server = TransactionServer(K=128, tau=TAU)
    now = time.time()

    # ---- register holders ----
    print("--- register holders ---")
    alice_fp = server.register(
        holder="alice", address="1 Möbius Street, π-City",
        balance=500.0, expiry_ts=now + 86400 * 365)        # 1 yr
    bob_fp = server.register(
        holder="bob", address="42 Fourier Avenue, e-Town",
        balance=200.0, expiry_ts=now + 86400 * 30)         # 30 days
    carol_fp = server.register(
        holder="carol", address="7 Laplace Road, α-Village",
        balance=1000.0, expiry_ts=now - 3600.0)            # expired 1 h ago
    print(f"  alice  fp={alice_fp[:16]}…  balance=500.00  "
          f"expiry=+1yr")
    print(f"  bob    fp={bob_fp[:16]}…  balance=200.00  "
          f"expiry=+30d")
    print(f"  carol  fp={carol_fp[:16]}…  balance=1000.00  "
          f"expiry=−1h  (expired)")
    print()

    # ---- build cards ----
    cards = {
        "alice": Card(
            holder="alice",
            address="1 Möbius Street, π-City",
            frozen=False,
            expiry_ts=now + 86400 * 365,
            balance=500.0,
        ),
        "bob": Card(
            holder="bob",
            address="42 Fourier Avenue, e-Town",
            frozen=False,
            expiry_ts=now + 86400 * 30,
            balance=200.0,
        ),
        "carol": Card(
            holder="carol",
            address="7 Laplace Road, α-Village",
            frozen=False,
            expiry_ts=now - 3600.0,
            balance=1000.0,
        ),
        "dave": Card(
            holder="dave",
            address="99 Unregistered Lane",
            frozen=False,
            expiry_ts=now + 86400 * 10,
            balance=100.0,
        ),
    }

    # ---- transactions ----
    cases = [
        ("alice", 100.0, "valid transaction"),
        ("alice", 100.0, "same card twice (double check)"),
        ("alice", 100.0, "replay same nonce (should fail)"),
        ("bob",   50.0,  "valid small transaction"),
        ("bob",   500.0, "insufficient balance"),
        ("carol", 100.0, "expired card"),
        ("dave",  20.0,  "unknown card"),
        ("alice", 100.0, "frozen card"),
    ]

    print("=" * 78)
    print("TRANSACTIONS")
    print("=" * 78)
    print(f"  {'#':>2}  {'holder':8s}  {'amount':>8s}  "
          f"{'stage1':>7s}  {'sim':>6s}  {'cross':>6s}  "
          f"{'σ':>4s}  {'approved':>9s}  reason")
    print("  " + "-" * 92)

    # freeze alice after her second transaction (for the last case)
    freeze_after = 2

    results = []
    for k, (holder, amount, note) in enumerate(cases):
        # special: freeze alice just before case 7 ("frozen card")
        if k == 7:
            server.freeze(alice_fp)
            cards["alice"].frozen = True
            cards["alice"].__post_init__()   # rebuild envelope

        card = cards[holder]
        req = card.authorize(amount)
        v = server.process(req)

        results.append((holder, amount, v, note))
        stage1_s = "ok" if v.stage1_ok else "FAIL"
        label_s = "=" if v.sat_labels_match else "≠"
        approved_s = "APPROVE" if v.approved else "DECLINE"

        print(f"  {k+1:>2d}  {holder:8s}  {amount:>8.2f}  "
              f"{stage1_s:>7s}  {v.stage2_similarity:>6.3f}  "
              f"{v.sat_cross:>6.3f}  {label_s:>4s}  "
              f"{approved_s:>9s}  {v.reason}")

    print()

    # ---- detailed timing ----
    print("--- timing per transaction (O-stages) ---")
    print(f"  {'holder':8s}  {'stage1 ms':>10s}  "
          f"{'stage2 ms':>10s}  {'total ms':>10s}")
    for holder, amount, v, _ in results:
        print(f"  {holder:8s}  {v.stage1_ms:>10.4f}  "
              f"{v.stage2_ms:>10.4f}  {v.elapsed_ms:>10.4f}")
    print()

    # ---- approval summary ----
    n_app = sum(1 for _, _, v, _ in results if v.approved)
    n_dec = len(results) - n_app
    print("--- summary ---")
    print(f"  approved            : {n_app}")
    print(f"  declined            : {n_dec}")
    print(f"  balance alice       : "
          f"{server.dns[alice_fp]['balance']:.2f}")
    print(f"  balance bob         : "
          f"{server.dns[bob_fp]['balance']:.2f}")
    print(f"  balance carol       : "
          f"{server.dns[carol_fp]['balance']:.2f}  "
          f"(expired, never debited)")
    print()

    # ---- the three complexities on display ----
    print("--- complexity map ---")
    print("  STAGE 1  O(K log log K)  ·  symmetric / deterministic")
    print("    sieve once               O(K log log K)")
    print("    fingerprint match        O(1)")
    print("    frozen check             O(1)")
    print("    expiry compare           O(1)")
    print("    balance check            O(1)")
    print("    nonce replay lookup      O(1)")
    print("    address equality         O(1)")
    print()
    print("  STAGE 2  O(K log K)  ·  asymmetric / dynamic")
    print("    server envelope rebuild  O(K)")
    print("    inverse_score merge sort O(K log K)")
    print("    chiral-bond weight       O(K)")
    print()
    print("  GATE  O(K)  ·  SAT survivor  (if/else)")
    print("    cross_coupling           O(K)")
    print("    UGC label compare        O(K)")
    print("    if cross ≥ τ  ∧ σ_card = σ_server:")
    print("        APPROVE")
    print("    else:")
    print("        DECLINE")
    print()

    # ---- envelope peak stats ----
    print("--- SAT envelope stats (card) ---")
    print(f"  {'holder':8s}  {'|env|':>10s}  {'nz':>5s}  "
          f"{'density':>8s}  {'phase':>8s}  {'σ':>3s}")
    for holder, card in cards.items():
        env = card.envelope
        nz = int(np.count_nonzero(env))
        density = nz / len(env)
        phi = envelope_phase(env)
        sigma = ugc_label(phi, UGC_K)
        print(f"  {holder:8s}  {np.linalg.norm(env):>10.4f}  "
              f"{nz:>5d}  {density:>8.4f}  "
              f"{phi:>+8.4f}  {sigma:>3d}")
    print()


if __name__ == "__main__":
    demo()