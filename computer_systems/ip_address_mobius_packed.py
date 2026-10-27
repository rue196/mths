#!/usr/bin/env python3
"""
mobius_ip.py
============

Geo-located **ternary** IP addressing.

    Address = K/4 trits, each in {0, 1, 2}.

    trit[0]           mode flag      0 = STATIC / 1 = DYNAMIC / 2 = ANYCAST
    trit[1]           geo sector     3-way from χ = log(t) mod 2π
    trit[2 .. N-2]    geo body       spinor + domain hash (mode-dependent)
    trit[N-1]         checksum       (Σ trits[:-1]) mod 3

Packing: 5 trits per byte (3^5 = 243 < 256).

Capacity table
--------------
    K =  80  →  20 trits  →  4 bytes  →  3.5 × 10⁹ addr   (≈ IPv4)
    K = 160  →  40 trits  →  8 bytes  →  1.2 × 10¹⁹ addr
    K = 324  →  81 trits  → 17 bytes  →  4.4 × 10³⁸ addr   (≈ IPv6)
    K = 640  → 160 trits  → 32 bytes  →  1.8 × 10⁷⁶ addr

Dynamic vs Static
-----------------
    STATIC   router / server — body = SHAKE256("STATIC|" || domain)
             stable forever; same domain → same address.
    DYNAMIC  mobile — body = spinor(geo) ⊕ SHAKE256("DYNAMIC|" || domain || geo)
             new geo-fix → new address (device moved to a new cell).
    ANYCAST  reserved / gateway — body = SHAKE256("ANYCAST|" || domain)
             multiple sites may claim the same address; the router
             picks the geographically nearest responder.

Geo → trits
-----------
    1. events (t, r) from geo via make_geo_events(seed=hash(geo))
    2. coupling G = w·κ → even-even G_ee
    3. spinor reconstruction: |Π(Z_n)| for even events
    4. classify each |Π| into trit 0/1/2 by tertile
    5. hash-mix with geo bytes to remove ordering artifacts
"""

from __future__ import annotations
import hashlib, math, struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- geo-location ----
from coupling_Q_entanglement import (
    coupling_matrix,
    entangled_coupling,
    even_indices,
    spinor_reconstruct,
    make_geo_events,
)

PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)
DENSITY = 6.0 / (PI * PI)


# ============================================================
#  Address modes
# ============================================================
class IPMode(IntEnum):
    STATIC  = 0    # router / server
    DYNAMIC = 1    # mobile
    ANYCAST = 2    # reserved / gateway


# ============================================================
#  Ternary packing  ·  5 trits per byte
# ============================================================
def pack_trits(trits) -> bytes:
    if not all(0 <= t <= 2 for t in trits):
        raise ValueError("all trits must be in {0, 1, 2}")
    out = bytearray()
    for i in range(0, len(trits), 5):
        chunk = list(trits[i:i + 5])
        while len(chunk) < 5:
            chunk.append(0)
        v = 0
        for t in chunk:
            v = v * 3 + t
        out.append(v)                       # 0 ≤ v ≤ 242 < 256
    return bytes(out)


def unpack_trits(data: bytes, n_trits: int) -> List[int]:
    trits = []
    for b in data:
        local = []
        v = b
        for _ in range(5):
            local.append(v % 3)
            v //= 3
        trits.extend(reversed(local))
    return trits[:n_trits]


def packed_size(n_trits: int) -> int:
    return (n_trits + 4) // 5


# ============================================================
#  Geo extraction
# ============================================================
def _events_from_geo(geo: Tuple[float, float],
                     n_events: int) -> Tuple[np.ndarray, np.ndarray]:
    seed = int(hashlib.sha256(
        f"{geo[0]:.6f}|{geo[1]:.6f}|{n_events}".encode()
    ).hexdigest()[:8], 16)
    t, r = make_geo_events(n_events=n_events, seed=seed % (2 ** 32))
    t = t * (1.0 + 0.1 * geo[0])
    r = r * (1.0 + 0.1 * geo[1])
    return t, r


def geo_sector(geo: Tuple[float, float]) -> int:
    """Sector index 0/1/2 from χ = log(t) mod 2π."""
    chi = math.log(max(abs(geo[0]), 1e-9)) % (2 * math.pi)
    return int((chi / (2 * math.pi)) * 3) % 3


def geo_body_trits(geo: Tuple[float, float],
                   n_events: int, n_trits: int) -> List[int]:
    """
    Body trits from the geo coupling structure.

    spinor_reconstruct(t_even, r_even) gives |Π(Z_n)| per even event.
    Tertile-classify each |Π| into {0, 1, 2}, then hash-mix with
    the raw geo bytes to avoid ordering artifacts.
    """
    t, r = _events_from_geo(geo, n_events)
    G = coupling_matrix(t, r)
    G_ee, _ = entangled_coupling(G)

    even_idx = even_indices(len(t))
    Pis, mags = spinor_reconstruct(t[even_idx], r[even_idx])

    if len(mags) >= 3:
        q1, q2 = np.quantile(mags, [1 / 3, 2 / 3])
        classified = [0 if m < q1 else (1 if m < q2 else 2) for m in mags]
    else:
        classified = [0] * max(len(mags), 1)

    mixed = hashlib.shake_256(
        struct.pack(">dd", *geo)).digest(n_trits)

    body = []
    for i in range(n_trits):
        base = classified[i % len(classified)]
        body.append((base + mixed[i]) % 3)
    return body


# ============================================================
#  Address builder
# ============================================================
def build_address(domain: str,
                  geo: Tuple[float, float],
                  mode: IPMode,
                  K: int = 80,
                  n_events: int = 24) -> List[int]:
    n_trits = K // 4
    if n_trits < 4:
        raise ValueError(f"K={K} too small (need K ≥ 16)")

    trits = [0] * n_trits
    trits[0] = int(mode)
    trits[1] = geo_sector(geo)

    n_body = n_trits - 3
    if n_body <= 0:
        trits[-1] = sum(trits[:-1]) % 3
        return trits

    domain_b = domain.encode("utf-8")
    geo_b = struct.pack(">dd", *geo)

    if mode == IPMode.STATIC:
        h = hashlib.shake_256(
            b"STATIC|" + domain_b + bytes([trits[1]])
        ).digest(n_body)
        body = [h[i] % 3 for i in range(n_body)]

    elif mode == IPMode.DYNAMIC:
        spin = geo_body_trits(geo, n_events, n_body)
        h = hashlib.shake_256(
            b"DYNAMIC|" + domain_b + geo_b
        ).digest(n_body)
        body = [(spin[i] + h[i]) % 3 for i in range(n_body)]

    else:  # ANYCAST
        h = hashlib.shake_256(
            b"ANYCAST|" + domain_b
        ).digest(n_body)
        body = [h[i] % 3 for i in range(n_body)]

    for i in range(n_body):
        trits[2 + i] = body[i]
    trits[-1] = sum(trits[:-1]) % 3
    return trits


# ============================================================
#  Address dataclass
# ============================================================
@dataclass(frozen=True)
class MobiusIP:
    trits: Tuple[int, ...]

    def __post_init__(self):
        if not self.trits:
            raise ValueError("empty address")
        if not all(0 <= t <= 2 for t in self.trits):
            raise ValueError("trits must be in {0, 1, 2}")

    # ---- properties ----
    @property
    def mode(self) -> IPMode:
        return IPMode(self.trits[0])

    @property
    def sector(self) -> int:
        return self.trits[1]

    @property
    def n_trits(self) -> int:
        return len(self.trits)

    @property
    def n_bytes(self) -> int:
        return packed_size(self.n_trits)

    def checksum_ok(self) -> bool:
        return sum(self.trits[:-1]) % 3 == self.trits[-1]

    # ---- serialisation ----
    def to_bytes(self) -> bytes:
        return pack_trits(self.trits)

    @classmethod
    def from_bytes(cls, data: bytes, n_trits: int) -> "MobiusIP":
        return cls(tuple(unpack_trits(data, n_trits)))

    def to_dotted(self) -> str:
        return ".".join(str(t) for t in self.trits)

    def to_dotted_prefix(self, prefix_len: int) -> str:
        if not 0 <= prefix_len <= self.n_trits:
            raise ValueError(f"prefix_len {prefix_len} out of range")
        return f"{self.to_dotted()}/{prefix_len}"

    def __str__(self) -> str:
        return self.to_dotted()


# ============================================================
#  Registry
# ============================================================
class MobiusIPRegistry:
    def __init__(self, K: int = 80, n_events: int = 24):
        self.K = K
        self.n_trits = K // 4
        self.n_events = n_events
        self._static:  Dict[str, MobiusIP] = {}
        self._anycast: Dict[str, MobiusIP] = {}
        self._dynamic: Dict[str, List[MobiusIP]] = {}

    # ---- allocation ----
    def register_static(self, domain: str,
                        geo: Tuple[float, float]) -> MobiusIP:
        addr = MobiusIP(tuple(build_address(
            domain, geo, IPMode.STATIC, self.K, self.n_events)))
        self._static[domain] = addr
        return addr

    def register_anycast(self, domain: str) -> MobiusIP:
        addr = MobiusIP(tuple(build_address(
            domain, (0.5, 0.5), IPMode.ANYCAST, self.K, self.n_events)))
        self._anycast[domain] = addr
        return addr

    def alloc_dynamic(self, domain: str,
                      geo: Tuple[float, float]) -> MobiusIP:
        addr = MobiusIP(tuple(build_address(
            domain, geo, IPMode.DYNAMIC, self.K, self.n_events)))
        self._dynamic.setdefault(domain, []).append(addr)
        return addr

    # ---- lookup ----
    def resolve(self, domain: str) -> Optional[MobiusIP]:
        if domain in self._static:
            return self._static[domain]
        return self._anycast.get(domain)

    def known_dynamic(self, domain: str) -> List[MobiusIP]:
        return list(self._dynamic.get(domain, []))

    # ---- verification ----
    def verify_static(self, domain: str, addr: MobiusIP) -> bool:
        expected = self._static.get(domain)
        if expected is None:
            return False
        return (addr.trits == expected.trits) and addr.checksum_ok()

    def verify_dynamic(self, domain: str,
                       geo: Tuple[float, float],
                       addr: MobiusIP) -> bool:
        if not addr.checksum_ok():
            return False
        expected = build_address(domain, geo, IPMode.DYNAMIC,
                                 self.K, self.n_events)
        return tuple(expected) == addr.trits

    def verify_anycast(self, domain: str, addr: MobiusIP) -> bool:
        expected = self._anycast.get(domain)
        if expected is None:
            return False
        return (addr.trits == expected.trits) and addr.checksum_ok()


# ============================================================
#  Router  ·  prefix-based
# ============================================================
class MobiusIPRouter:
    def __init__(self, registry: MobiusIPRegistry):
        self.reg = registry

    def same_network(self, a: MobiusIP, b: MobiusIP,
                     prefix_len: int) -> bool:
        if not 0 <= prefix_len <= a.n_trits:
            raise ValueError(f"prefix_len {prefix_len} out of range")
        return a.trits[:prefix_len] == b.trits[:prefix_len]

    def same_sector(self, a: MobiusIP, b: MobiusIP) -> bool:
        return self.same_network(a, b, 2)

    def route(self, dest: MobiusIP) -> Dict:
        if not dest.checksum_ok():
            return dict(status="ERROR", reason="checksum", dest=str(dest))

        if dest.mode == IPMode.STATIC:
            for domain, a in self.reg._static.items():
                if a.trits == dest.trits:
                    return dict(status="OK", kind="STATIC",
                                domain=domain, sector=dest.sector)
            return dict(status="MISS", kind="STATIC", sector=dest.sector)

        if dest.mode == IPMode.DYNAMIC:
            for domain, addrs in self.reg._dynamic.items():
                if dest.trits in {a.trits for a in addrs}:
                    return dict(status="OK", kind="DYNAMIC",
                                domain=domain, sector=dest.sector)
            return dict(status="MISS", kind="DYNAMIC", sector=dest.sector)

        if dest.mode == IPMode.ANYCAST:
            for domain, a in self.reg._anycast.items():
                if a.trits == dest.trits:
                    return dict(status="OK", kind="ANYCAST",
                                domain=domain, sector=dest.sector)
            return dict(status="MISS", kind="ANYCAST", sector=dest.sector)

        return dict(status="ERROR", reason="mode", dest=str(dest))


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius geo-ternary IP  ·  K/4 trits  ·  router vs mobile")
    print("=" * 78)
    print(f"  α              = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  6/π²                          = {DENSITY:.6f}")
    print(f"  packing                       = 5 trits / byte")
    print()

    # ---- capacity table ----
    print("--- capacity by K ---")
    print(f"  {'K':>5s}  {'N=K/4':>6s}  {'bytes':>6s}  "
          f"{'address space (log10)':>22s}")
    for K in (40, 80, 160, 324, 640):
        N = K // 4
        b = packed_size(N)
        log10 = N * math.log10(3)
        print(f"  {K:>5d}  {N:>6d}  {b:>6d}  {log10:>22.2f}")
    print()

    # ---- registry + router ----
    K = 80                      # 20 trits → 4 bytes
    reg = MobiusIPRegistry(K=K, n_events=24)
    router = MobiusIPRouter(reg)

    # ---------- static: servers ----------
    print("--- static addresses (router / server) ---")
    servers = [
        ("mail.example.com",   (1.2, 0.9)),
        ("dns.example.com",    (1.3, 0.8)),
        ("relay.mobius.org",   (2.7, 1.5)),
        ("storage.mobius.org", (2.5, 1.7)),
    ]
    for domain, geo in servers:
        a = reg.register_static(domain, geo)
        print(f"  {domain:22s}  geo={geo}  "
              f"sector={a.sector}  {a.to_dotted()}  "
              f"({a.n_bytes} B)")
    print()

    # ---------- anycast: CDN ----------
    print("--- anycast addresses (multi-site gateway) ---")
    a_cdn = reg.register_anycast("cdn.mobius.org")
    print(f"  cdn.mobius.org         "
          f"sector={a_cdn.sector}  {a_cdn.to_dotted()}  "
          f"({a_cdn.n_bytes} B)")
    print()

    # ---------- dynamic: mobile ----------
    print("--- dynamic addresses (mobile device) ---")
    mobile_domain = "phone-42@mobius.local"
    path = [(0.8, 0.5), (1.5, 1.0), (2.2, 1.6), (3.1, 2.0)]
    for i, geo in enumerate(path):
        a = reg.alloc_dynamic(mobile_domain, geo)
        print(f"  step {i}  geo={geo}  sector={a.sector}  "
              f"{a.to_dotted()}")
    print(f"  → {len(reg.known_dynamic(mobile_domain))} addresses "
          f"allocated for this device")
    print()

    # ---------- verification ----------
    print("--- verification ---")
    m = reg.resolve("mail.example.com")
    print(f"  static  mail.example.com   "
          f"verified={reg.verify_static('mail.example.com', m)}")
    print(f"  static  dns.example.com    "
          f"verified={reg.verify_static('dns.example.com', m)}  "
          f"(wrong domain)")
    print(f"  anycast cdn.mobius.org     "
          f"verified={reg.verify_anycast('cdn.mobius.org', a_cdn)}")

    last_geo = path[-1]
    last_addr = reg.known_dynamic(mobile_domain)[-1]
    print(f"  dynamic @{last_geo}        "
          f"verified={reg.verify_dynamic(mobile_domain, last_geo, last_addr)}")
    print(f"  dynamic @{(0.8, 0.5)}        "
          f"verified={reg.verify_dynamic(mobile_domain, (0.8, 0.5), last_addr)}"
          f"  (stale geo)")
    print()

    # ---------- routing ----------
    print("--- routing ---")
    for domain, addr in [
        ("mail.example.com",   reg.resolve("mail.example.com")),
        ("cdn.mobius.org",     a_cdn),
        ("phone-42@mobius.local", last_addr),
    ]:
        r = router.route(addr)
        print(f"  {domain:22s}  {r}")
    print()

    # ---------- prefix routing ----------
    print("--- prefix routing ---")
    a = reg.resolve("mail.example.com")
    b = reg.resolve("dns.example.com")
    c = reg.resolve("relay.mobius.org")
    print(f"  mail / dns   same sector  : "
          f"{router.same_sector(a, b)}")
    print(f"  mail / relay same sector  : "
          f"{router.same_sector(a, c)}")
    print(f"  mail         /4 prefix    : "
          f"{a.to_dotted_prefix(4)}")
    print(f"  dns          /4 prefix    : "
          f"{b.to_dotted_prefix(4)}")
    print()

    # ---------- wire round-trip ----------
    print("--- wire round-trip ---")
    for domain in ("mail.example.com", "cdn.mobius.org"):
        addr = reg.resolve(domain)
        wire = addr.to_bytes()
        restored = MobiusIP.from_bytes(wire, n_trits=K // 4)
        print(f"  {domain:22s}  "
              f"{len(wire)} B  "
              f"round-trip={'OK' if restored.trits == addr.trits else 'FAIL'}")
    print()

    # ---------- mobile address changes on move ----------
    print("--- mobile address churn ---")
    a0 = reg.known_dynamic(mobile_domain)[0]
    a3 = reg.known_dynamic(mobile_domain)[-1]
    diff = sum(1 for x, y in zip(a0.trits, a3.trits) if x != y)
    print(f"  start  {a0.to_dotted()}")
    print(f"  end    {a3.to_dotted()}")
    print(f"  churn  {diff} / {a0.n_trits} trits changed  "
          f"({100 * diff / a0.n_trits:.0f}%)")
    print(f"  same mode   : {a0.mode == a3.mode}")
    print(f"  same sector : {router.same_sector(a0, a3)}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  events from geo       O(n_events)")
    print("  coupling matrix       O(n_events²)")
    print("  spinor reconstruct    O(n_even · 720)")
    print("  body trits            O(N)")
    print("  pack trits            O(N)")
    print("  verify static         O(N)")
    print("  verify dynamic        O(n_events² + N)")
    print("  route by prefix       O(N)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()