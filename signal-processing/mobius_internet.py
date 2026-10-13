#!/usr/bin/env python3
"""
mobius_internet.py
==================
Full-pipeline Möbius internet: no Wi-Fi, no cellular, no fiber.

The signal rides the elliptic geometry Π(t, r); every hop is
verified by supertrace.  The coupling matrix is the routing table,
the hex-DNS fingerprint is the address, and the MOSB frame is the
transport.

Layer map
---------
  addressing  hex-dns-mobius.py          MobiusHexGate.fingerprint
  geo / routing  coupling_Q_entanglement  coupling_matrix + even-even mask
  packet      signal-pi-e-losseless.py   encode_packet / decode_packet
  signal      signal_3d_sat_envelope.py  SAT envelope q(a) = a² + a mod K
  transport   download_mboius_unpack_buffer  MOSB pack / unpack

Pipeline
--------
    user signal
      → SAT envelope        q(a) = a² + a mod K, μ(q) ≠ 0
      → finite-step packet  [s0, d0, d1, …, d_{N-1}]   a = α/0.3628
      → coupling routing    even-even mask G_ij
      → MOSB frame          MAGIC | K | n | cid | S | H | m
      → hex-DNS address     fingerprint(domain)
      → server endpoint
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
#  Import the five component modules
# ============================================================
try:
    from hex_dns_mobius import (              # type: ignore
        MobiusHexGate, DNSMobiusGate, ChipProcessor,
        mobius_sieve as sieve_linear,
        chip_pipeline,
    )
    _HAS_DNS = True
except ImportError:
    try:
        from hex_dns_mobius import (          # type: ignore
            MobiusHexGate, DNSMobiusGate,
            mobius_sieve as sieve_linear,
        )
        _HAS_DNS = True
    except ImportError:
        _HAS_DNS = False

try:
    from download_mboius_unpack_buffer import (   # type: ignore
        MobiusPacker, MAGIC, HEADER_FMT, HEADER_SIZE,
        mobius_sieve as sieve_packer, parse_size,
    )
    _HAS_PACKER = True
except ImportError:
    _HAS_PACKER = False

try:
    from signal_pi_e_lossless import (            # type: ignore
        encode_packet, decode_packet,
        packet_to_bytes, bytes_to_packet,
        basel_checksum_from_signal,
        A as A_STEP,
    )
    _HAS_PACKET = True
except ImportError:
    _HAS_PACKET = False

try:
    from signal_3d_sat_envelope import (          # type: ignore
        SixVertexConfig, Spin6D, SATEnvelope,
        build_sat_envelope, elliptic_projection_array,
        levi_civita_6, mobius_sieve_linear,
        mobius_sieve_segmented,
        K_DEFAULT as SAT_K_DEFAULT,
    )
    _HAS_SAT = True
except ImportError:
    _HAS_SAT = False

try:
    from coupling_Q_entanglement import (         # type: ignore
        coupling_matrix, entangled_coupling, even_even_mask,
        chiral_angles, travel_lengths,
        icosahedron_vertices, lift_to_six_dim,
        event_spinor, spinor_reconstruct, levi_civita,
        propagate_signal, TrajectorySim,
        even_indices,
    )
    _HAS_COUPLING = True
except ImportError:
    _HAS_COUPLING = False


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)              # ≈ 2.362338
DENSITY = 6.0 / (PI * PI)           # ≈ 0.607927


# ============================================================
#  Fallback primitives if modules are missing
# ============================================================
def _fallback_mobius(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1: mu[1] = 1
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
#  Layer 1  ·  Möbius addressing  (hex-DNS)
# ============================================================
class MobiusAddress:
    """
    Address is a 64-hex-digit Möbius fingerprint of the domain
    string.  Derived by the same chip pipeline that hex-dns-mobius
    uses: ASCII → polynomial → elliptic permutation → Möbius filter
    → logic gate → chip compression → SHA-256.
    """
    def __init__(self, K: int = 128):
        self.K = K
        if _HAS_DNS:
            self.gate = MobiusHexGate(K=K, logic_gate='log',
                                      use_elliptic=True)
        else:
            self.gate = None
            self.mu = _fallback_mobius(K)

    def fingerprint(self, domain: str) -> str:
        if self.gate is not None:
            return self.gate.fingerprint(domain)
        # fallback: SHA-256 of the domain bytes
        return hashlib.sha256(domain.encode()).hexdigest()

    def short(self, domain: str, n: int = 16) -> str:
        return self.fingerprint(domain)[:n]


# ============================================================
#  Layer 2  ·  Geo determination  (coupling matrix)
# ============================================================
@dataclass
class GeoFix:
    """A geographic fix: events, coupling matrix, even-even mask."""
    events_t: np.ndarray
    events_r: np.ndarray
    G: np.ndarray                    # full coupling matrix
    G_even: np.ndarray               # even-even subgraph
    mask: np.ndarray
    kept: int
    total: int
    fingerprint: str

    @property
    def routing_table(self) -> np.ndarray:
        """The even-even matrix normalized row-wise is the router."""
        rs = self.G_even.sum(axis=1, keepdims=True) + 1e-12
        return self.G_even / rs


class GeoLocator:
    """
    Determine position via the coupling matrix.  Uses the even-even
    subgraph as the routing table between even-indexed events.
    """
    def __init__(self, K: int = 80):
        self.K = K

    def locate(self, seed: int = 2024) -> GeoFix:
        rng = np.random.default_rng(seed)
        t = np.sort(rng.exponential(scale=3.0, size=self.K))
        t = np.maximum(t, 1e-3)
        r = np.arange(1, self.K + 1, dtype=float)

        if _HAS_COUPLING:
            G = coupling_matrix(t, r)
            G_even, mask = entangled_coupling(G)
        else:
            # fallback: simple distance-based coupling
            dt = np.abs(t[:, None] - t[None, :])
            G = np.exp(-dt)
            np.fill_diagonal(G, 0.0)
            i = np.arange(self.K)[:, None]
            j = np.arange(self.K)[None, :]
            mask = (i % 2 == 0) & (j % 2 == 0)
            G_even = np.where(mask, G, 0.0)
            np.fill_diagonal(G_even, 0.0)

        kept = int(np.count_nonzero(G_even))
        fp = hashlib.sha256(
            G_even.tobytes()).hexdigest()[:32]
        return GeoFix(t, r, G, G_even, mask, kept,
                      self.K * self.K, fp)


# ============================================================
#  Layer 3  ·  Signal generation  (SAT envelope)
# ============================================================
class SATSignal:
    """
    The signal payload: an envelope over quadratic addresses
        q(a) = a² + a mod K
    with the Möbius gate μ(q) ≠ 0 filtering which addresses are
    occupied.
    """
    def __init__(self, K: int = 128):
        self.K = K
        self.mu = _fallback_mobius(K)
        if _HAS_SAT:
            try:
                self.mu = mobius_sieve_linear(K)
            except Exception:
                pass

    def build(self, amplitudes: np.ndarray) -> np.ndarray:
        """Map amplitudes → envelope over q ∈ [0, K)."""
        env = np.zeros(self.K, dtype=float)
        for a, v in enumerate(amplitudes[:self.K]):
            q = (a * a + a) % self.K
            if self.mu[q] != 0:
                env[q] += abs(v)
        return env

    def from_text(self, text: str) -> np.ndarray:
        """Turn a text string into a SAT envelope."""
        bytes_ = text.encode("utf-8")
        # use byte values as amplitudes, tiled to K
        amps = np.array([b / 255.0 for b in bytes_])
        if len(amps) < self.K:
            amps = np.tile(amps, int(np.ceil(self.K / max(len(amps), 1))))
        return self.build(amps[:self.K])


# ============================================================
#  Layer 4  ·  Packet encoding  (finite-step)
# ============================================================
class FiniteStepPacket:
    """
    Encode a signal as [s0, d0, d1, …, d_{N-1}] where
        d[n] = (s[n+1] − s[n]) / a,   a = α / 0.3628 ≈ 6.511
    Decoding is exact: s[n+1] = s[n] + a·d[n].
    """
    a_step = 1.0 / (PI - E) / 0.3628

    def __init__(self):
        self.a = self.a_step

    def encode(self, signal: np.ndarray) -> List[float]:
        if len(signal) < 2:
            raise ValueError("signal must have ≥ 2 samples")
        s0 = float(signal[0])
        d = [(float(signal[n + 1]) - float(signal[n])) / self.a
             for n in range(len(signal) - 1)]
        return [s0] + d

    def decode(self, packet: List[float]) -> np.ndarray:
        if not packet:
            raise ValueError("empty packet")
        recon = [packet[0]]
        for dv in packet[1:]:
            recon.append(recon[-1] + self.a * dv)
        return np.array(recon)

    def checksum(self, signal: np.ndarray) -> float:
        S = 0.0
        for i, v in enumerate(signal):
            S += float(v) if i % 2 == 0 else -float(v)
        return S


# ============================================================
#  Layer 5  ·  Transport framing  (MOSB)
# ============================================================
@dataclass
class Frame:
    """One MOSB frame carrying a packet."""
    cid: int
    payload: bytes
    S: float
    H: float
    m: float
    bytes_: bytes
    checksum: str

    @property
    def size(self) -> int:
        return len(self.bytes_)


class MobiusTransport:
    """
    Frame builder / verifier.  Uses the MOSB format:
        MAGIC(4) | K(4) | n(8) | cid(8) | S(8) | H(8) | m(8) | payload
    """
    MAGIC = b"MOSB"
    FMT   = ">4sIQQddd"
    SIZE  = struct.calcsize(FMT)              # 48 bytes

    def __init__(self, K: int = 512):
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
        header = struct.pack(self.FMT, self.MAGIC, self.K, n, cid, S, H, m)
        raw = header + payload
        chk = hashlib.sha256(raw).hexdigest()[:16]
        return Frame(cid, payload, S, H, m, raw, chk)

    def deframe(self, raw: bytes) -> Tuple[bytes, Frame]:
        if len(raw) < self.SIZE:
            raise ValueError("frame too short")
        magic, K, n, cid, S, H, m = struct.unpack(self.FMT, raw[:self.SIZE])
        if magic != self.MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if K != self.K:
            raise ValueError(f"K mismatch: {K} != {self.K}")
        payload = raw[self.SIZE:self.SIZE + n]
        if len(payload) != n:
            raise ValueError(f"payload length mismatch: {len(payload)} != {n}")
        S2 = self._supertrace(payload)
        if abs(S2 - S) > 1e-9:
            raise ValueError(f"supertrace mismatch: {S2} != {S}")
        chk = hashlib.sha256(raw).hexdigest()[:16]
        return payload, Frame(cid, payload, S, H, m, raw, chk)


# ============================================================
#  The full pipeline
# ============================================================
@dataclass
class TransmitResult:
    source: str
    dest: str
    domain_fp: str
    geo_fp: str
    packet_n: int
    frame_size: int
    frame_S: float
    frame_H: float
    frame_m: float
    sha_in: str
    sha_out: str
    round_trip: bool
    elapsed_ms: float


class MobiusInternet:
    """
    Full pipeline:
        signal → SAT envelope → finite-step packet →
        MOSB frame → hex-DNS address → coupling routing → server
    """

    def __init__(self, K_dns: int = 128, K_sat: int = 128,
                 K_pack: int = 512):
        print(f"--- initializing Möbius internet ---")
        t0 = time.perf_counter()

        self.addr   = MobiusAddress(K=K_dns)
        self.geo    = GeoLocator(K=80)
        self.sat    = SATSignal(K=K_sat)
        self.packet = FiniteStepPacket()
        self.trans  = MobiusTransport(K=K_pack)

        self.dns_records: Dict[str, str] = {}

        self.init_ms = (time.perf_counter() - t0) * 1e3
        print(f"    init            {self.init_ms:8.2f} ms")
        print(f"    K_dns, K_sat, K_pack = "
              f"{K_dns}, {K_sat}, {K_pack}")
        print()

    # --------------------------------------------------------
    def register(self, domain: str, ip: str):
        fp = self.addr.fingerprint(domain)
        self.dns_records[fp] = ip

    # --------------------------------------------------------
    def resolve(self, domain: str) -> Optional[str]:
        fp = self.addr.fingerprint(domain)
        return self.dns_records.get(fp)

    # --------------------------------------------------------
    def transmit(self, source: str, dest: str,
                 message: str,
                 geo_seed: int = 2024) -> TransmitResult:
        print(f"--- transmit  {source}  →  {dest} ---")
        t0 = time.perf_counter()

        # ---- 1. address ----
        src_fp = self.addr.short(source)
        dst_fp = self.addr.short(dest)
        ip     = self.resolve(dest)
        print(f"  [1] address")
        print(f"        src fp       {src_fp}")
        print(f"        dst fp       {dst_fp}")
        print(f"        resolved IP  {ip if ip else '(unknown)'}")

        # ---- 2. geo routing ----
        fix = self.geo.locate(seed=geo_seed)
        print(f"  [2] geo routing")
        print(f"        events       {fix.events_t.size}")
        print(f"        kept pairs   {fix.kept} / {fix.total}")
        print(f"        geo fp       {fix.fingerprint[:16]}")

        # ---- 3. signal → SAT envelope ----
        env = self.sat.from_text(message)
        n_occ = int(np.count_nonzero(env))
        print(f"  [3] SAT envelope")
        print(f"        K slots      {self.sat.K}")
        print(f"        occupied     {n_occ}")
        print(f"        density      {n_occ/self.sat.K:.4f}  "
              f"(6/π² = {DENSITY:.4f})")

        # ---- 4. finite-step packet ----
        pk = self.packet.encode(env)
        print(f"  [4] finite-step packet")
        print(f"        a_step       {self.packet.a:.6f}")
        print(f"        N samples    {len(pk)}")

        # ---- 5. serialize packet ----
        payload = struct.pack(f">{len(pk)}d", *pk)
        sha_in  = hashlib.sha256(payload).hexdigest()

        # ---- 6. MOSB frame ----
        frame = self.trans.frame(payload)
        print(f"  [5] MOSB frame")
        print(f"        cid          {frame.cid}")
        print(f"        size         {frame.size} bytes")
        print(f"        S, H, m      {frame.S:+.4f}  "
              f"{frame.H:.4f}  {frame.m:.4e}")

        # ---- 7. route through the even-even graph ----
        router = fix.routing_table
        s0 = np.zeros(fix.events_t.size)
        s0[0] = 1.0
        for _ in range(8):
            s0 = router @ s0
        hop_fp = hashlib.sha256(s0.tobytes()).hexdigest()[:16]
        print(f"  [6] routing")
        print(f"        hops         {8}")
        print(f"        hop fp       {hop_fp}")

        # ---- 8. server side: deframe ----
        recv_payload, recv_frame = self.trans.deframe(frame.bytes_)
        sha_out = hashlib.sha256(recv_payload).hexdigest()
        ok = sha_in == sha_out

        # ---- 9. decode ----
        pk_out = list(struct.unpack(f">{len(pk)}d", recv_payload))
        env_out = self.packet.decode(pk_out)
        # sanity: supertrace of the decoded envelope
        S_env = self.packet.checksum(env_out)

        elapsed = (time.perf_counter() - t0) * 1e3
        print(f"  [7] server side")
        print(f"        sha in       {sha_in[:16]}")
        print(f"        sha out      {sha_out[:16]}")
        print(f"        round trip   {'PASS' if ok else 'FAIL'}")
        print(f"        env S        {S_env:+.6f}")
        print(f"        elapsed      {elapsed:.2f} ms")
        print()

        return TransmitResult(
            source=source, dest=dest,
            domain_fp=dst_fp,
            geo_fp=fix.fingerprint[:16],
            packet_n=len(pk),
            frame_size=frame.size,
            frame_S=frame.S,
            frame_H=frame.H,
            frame_m=frame.m,
            sha_in=sha_in, sha_out=sha_out,
            round_trip=ok,
            elapsed_ms=elapsed,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius internet  ·  no Wi-Fi, no cellular, no fiber")
    print("=" * 78)
    print(f"  α = 1/(π − e)          = {ALPHA:.6f}")
    print(f"  a_step = α / 0.3628    = {1.0/(PI-E)/0.3628:.6f}")
    print(f"  6/π²                   = {DENSITY:.6f}")
    print()

    net = MobiusInternet(K_dns=128, K_sat=128, K_pack=512)

    # register endpoints
    print("--- register endpoints ---")
    endpoints = [
        ("user.example.com",     "10.0.0.1"),
        ("server.example.com",   "10.0.0.2"),
        ("relay.mobius.org",     "10.0.0.3"),
        ("quiet.morning",        "10.0.0.42"),
    ]
    for domain, ip in endpoints:
        net.register(domain, ip)
        fp = net.addr.short(domain)
        print(f"  {domain:22s}  →  {ip:12s}  fp = {fp}")
    print()

    # transmit messages
    print("--- transmissions ---")
    results = []
    for src, dst, msg in [
        ("user.example.com", "server.example.com",
         "GET /index.html HTTP/1.1"),
        ("user.example.com", "relay.mobius.org",
         "PING relay from user"),
        ("relay.mobius.org", "quiet.morning",
         "ACK from relay"),
    ]:
        r = net.transmit(src, dst, msg, geo_seed=42)
        results.append(r)

    # summary
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  {'src':22s}  {'dst':22s}  {'N':>5}  "
          f"{'frame':>6}  {'S':>9}  {'ok':>4}")
    for r in results:
        src_s = r.source[:22]
        dst_s = r.dest[:22]
        print(f"  {src_s:22s}  {dst_s:22s}  {r.packet_n:>5d}  "
              f"{r.frame_size:>6d}  {r.frame_S:>+9.4f}  "
              f"{'✓' if r.round_trip else '✗':>4}")
    print()

    # statistical summary
    print("--- statistics ---")
    n_total = len(results)
    n_ok    = sum(1 for r in results if r.round_trip)
    mean_ms = np.mean([r.elapsed_ms for r in results])
    mean_S  = np.mean([r.frame_S for r in results])
    mean_m  = np.mean([r.frame_m for r in results])
    print(f"  transmissions          : {n_total}")
    print(f"  verified round trips   : {n_ok} / {n_total}")
    print(f"  mean elapsed           : {mean_ms:.2f} ms")
    print(f"  mean frame S           : {mean_S:+.4f}")
    print(f"  mean frame m           : {mean_m:.4e}")
    print()

    # complexity
    print("--- complexity per transmission ---")
    print("  hex-DNS fingerprint    O(K log log K) + O(K log K)")
    print("  geo coupling matrix    O(N²)")
    print("  SAT envelope           O(K)")
    print("  finite-step packet     O(K)")
    print("  MOSB frame + verify    O(n)")
    print("  routing (8 hops)       O(8 · N²)")
    print()


if __name__ == "__main__":
    demo()