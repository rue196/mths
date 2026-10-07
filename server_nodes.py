#!/usr/bin/env python3
"""
mobius_dns_server.py
====================

A Möbius DNS server with a split 8/32/64-bit architecture.

Pipeline
--------
    client request
      → 32-bit 1D command tape            O(K)
      → 64-bit walker (if / else routing) O(K)
          ├── rate limit fail       → 8-bit redirect
          ├── DDoS flagged          → 8-bit redirect
          ├── domain unknown        → 404 no envelope
          └── domain resolved       → hex-DNS Möbius gate
                                    → envelope per node
                                    → dispatch to K nodes

    8-bit path   DDoS / rate-limit redirect  O(1) per request
    32-bit path  request tape                O(K) to walk
    64-bit path  hex-DNS + envelope          O(K log log K) + O(K log K)

Hex-DNS
-------
Domain → ASCII → polynomial → elliptic permutation → Möbius filter
       → logic gate → chip compression → SHA-256 hex
       → cache lookup (hex fingerprint → IP)

Envelope
--------
Each node receives a per-node envelope:

    payload + node_id + domain + resolved_ip
    fingerprint = SHAKE256(payload || node_id || domain_fp)

The envelopes are the "letters" that cross the network — nothing
else leaves the server.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA_SYM   = 1.0 / (PI - E)                # ≈ 2.362
DENSITY     = 6.0 / (PI * PI)               # ≈ 0.6079
UINT32_MAX  = 0xFFFFFFFF
UINT8_MAX   = 0xFF
K_DEFAULT   = 128
N_NODES     = 8                             # K nodes in the network


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


def elliptic_projection(x: float, y: float) -> float:
    u = (x / PI) % 1.0
    v = (y / E) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


# ============================================================
#  Chip compression pipeline (from hex-dns-mobius.py)
# ============================================================
class ChipProcessor:
    def __init__(self, max_K: int = 1000):
        self.max_K = max_K
        self.f = np.zeros(max_K, dtype=float)
        self.b = np.zeros(max_K, dtype=float)
        self.conv = np.zeros(max_K, dtype=float)
        self.order = np.zeros(max_K, dtype=int)
        self.mag = np.zeros(max_K, dtype=float)
        self.mu = None
        self._update_mu(max_K)

    def _update_mu(self, K: int) -> None:
        if self.mu is None or len(self.mu) < K + 1:
            self.mu = mobius_sieve(K)

    @staticmethod
    def _supertrace_and_mass(signal, K):
        S = 0.0
        for i in range(K):
            S += abs(signal[i]) if (i % 2 == 0) else -abs(signal[i])
        if S == 0.0:
            return 0.0, 0.0, 0.0
        p = abs(S) / K
        H = -ALPHA_SYM * p * math.log(p) if 0.0 < p < 1.0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
        return S, H, m

    def process(self, signal: np.ndarray):
        K = len(signal)
        if K > self.max_K:
            raise ValueError(f"signal length {K} > max_K {self.max_K}")
        lam = math.exp(-ALPHA_SYM)
        NORM = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
        f = self.f; b = self.b; conv = self.conv
        f[0] = signal[0]
        for i in range(1, K):
            f[i] = signal[i] + lam * f[i - 1]
        b[K - 1] = signal[K - 1]
        for i in range(K - 2, -1, -1):
            b[i] = signal[i] + lam * b[i + 1]
        inv_den = 1.0 / (1.0 - lam * lam)
        inv_norm = 1.0 / NORM
        for i in range(K):
            conv_exp = (f[i] + b[i] - signal[i]) * inv_den
            conv[i] = (1.0 - conv_exp) * inv_norm
        S, H, m = self._supertrace_and_mass(conv, K)
        M = max(1, min(int(abs(S)), K))
        self._update_mu(K)
        mag = np.abs(conv)
        idx_sorted = np.argsort(mag)[::-1]
        kept = []
        count = 0
        for idx in idx_sorted:
            n = int(idx) + 1
            if self.mu[n] != 0:
                kept.append((int(idx), float(conv[idx])))
                count += 1
                if count >= M:
                    break
        return kept, S, H, m, conv[:K].copy()


# ============================================================
#  Hex-DNS Möbius gate (from hex-dns-mobius.py)
# ============================================================
class MobiusHexGate:
    """Domain → hex fingerprint via the Möbius chip pipeline."""
    def __init__(self, K: int = 128):
        self.K = K
        self.mu = mobius_sieve(K)
        self.processor = ChipProcessor(max_K=K)

    def _ascii_to_coeffs(self, text: str) -> np.ndarray:
        sig = np.zeros(self.K, dtype=float)
        for i, ch in enumerate(text[:self.K]):
            sig[i] = (ord(ch) - 32) / 95.0 + 0.1 * math.sin(i * 0.5)
        return sig

    def fingerprint(self, text: str) -> str:
        # 1. ASCII → coefficients
        signal = self._ascii_to_coeffs(text)
        # 2. Möbius filter on the position axis
        keep = np.zeros(self.K, dtype=bool)
        for i in range(1, self.K):
            if self.mu[i] != 0:
                keep[i - 1] = True
        signal = np.where(keep, signal, 0.0)
        # 3. logic gate (log)
        signal = np.log(np.maximum(np.abs(signal) + 1e-12, 1e-12))
        # 4. chip compression
        kept, S, H, m, _ = self.processor.process(signal)
        # 5. pack into hex
        data = bytearray()
        for idx, val in kept:
            data.extend(struct.pack('>Hd', idx, val))
        data.extend(struct.pack('>ddd', S, H, m))
        return hashlib.sha256(data).hexdigest()


class DNSMobiusGate:
    def __init__(self, K: int = 128):
        self.gate = MobiusHexGate(K=K)
        self.cache: Dict[str, str] = {}

    def add(self, domain: str, ip: str) -> str:
        fp = self.gate.fingerprint(domain)
        self.cache[fp] = ip
        return fp

    def resolve(self, domain: str) -> Optional[Tuple[str, str]]:
        fp = self.gate.fingerprint(domain)
        ip = self.cache.get(fp)
        return (fp, ip) if ip else None


# ============================================================
#  Per-node envelope
# ============================================================
@dataclass
class NodeEnvelope:
    """
    A single envelope bound to one node.

        node_id      unique node identifier (0 … K−1)
        domain       requested domain
        ip           resolved IP
        payload      serialized payload (hex)
        fingerprint  SHAKE256(payload || node_id || domain_fp)
        created_at   timestamp
    """
    node_id: int
    domain: str
    ip: str
    payload: bytes
    fingerprint: bytes
    created_at: float = field(default_factory=time.time)

    def summary(self) -> Dict:
        return dict(
            node=self.node_id,
            domain=self.domain,
            ip=self.ip,
            payload_b=len(self.payload),
            fingerprint=self.fingerprint.hex()[:16],
            age_ms=round((time.time() - self.created_at) * 1e3, 3),
        )


def build_node_envelope(node_id: int, domain: str, ip: str,
                        domain_fp: str) -> NodeEnvelope:
    payload = struct.pack('>H', node_id) + ip.encode() + domain.encode()
    h = hashlib.shake_256(
        payload + domain_fp.encode()).digest(32)
    return NodeEnvelope(
        node_id=node_id, domain=domain, ip=ip,
        payload=payload, fingerprint=h,
    )


# ============================================================
#  32-bit request tape
# ============================================================
OP_NOP        = 0x00
OP_METHOD     = 0x01
OP_HOST       = 0x02
OP_PATH       = 0x03
OP_PORT       = 0x04
OP_RATE_CHECK = 0x0A
OP_DDOS_CHECK = 0x0B
OP_DNS_LOOKUP = 0x0C
OP_ENVELOPE   = 0x0D
OP_DISPATCH   = 0x0E
OP_REDIRECT   = 0x0F   # ← 8-bit fallback marker
OP_HALT       = 0xFF

OP_NAMES = {
    OP_NOP: "NOP", OP_METHOD: "METHOD", OP_HOST: "HOST",
    OP_PATH: "PATH", OP_PORT: "PORT",
    OP_RATE_CHECK: "RATE_CHECK", OP_DDOS_CHECK: "DDOS_CHECK",
    OP_DNS_LOOKUP: "DNS_LOOKUP", OP_ENVELOPE: "ENVELOPE",
    OP_DISPATCH: "DISPATCH", OP_REDIRECT: "REDIRECT",
    OP_HALT: "HALT",
}


def pack_word32(op: int, a: int = 0, b: int = 0, c: int = 0) -> int:
    return (((op & 0xFF) << 24) |
            ((a & 0xFF) << 16) |
            ((b & 0xFF) << 8)  |
            (c & 0xFF))


def unpack_word32(word: int) -> Tuple[int, int, int, int]:
    return ((word >> 24) & 0xFF,
            (word >> 16) & 0xFF,
            (word >> 8)  & 0xFF,
            word & 0xFF)


@dataclass
class RequestTape:
    client_id: int
    words: List[int]
    names: Dict[str, int]           # method/host/path → index
    n_requests: int = 0             # cumulative from client_id


def build_request_tape(client_id: int,
                       method: str, host: str, path: str,
                       port: int,
                       n_requests: int) -> RequestTape:
    """
    Build the 32-bit tape for one request.

        METHOD  m         (m = method index 0..255)
        HOST    h         (h = host index)
        PATH    p         (p = path index)
        PORT    q
        RATE_CHECK
        DDOS_CHECK
        DNS_LOOKUP
        ENVELOPE
        DISPATCH
        HALT
    """
    names: Dict[str, int] = {}
    words: List[int] = []

    def _name_index(s: str) -> int:
        if s not in names:
            names[s] = (len(names) + 1) & 0xFF
        return names[s]

    h = _name_index(host)
    p = _name_index(path)
    m = {"GET": 1, "POST": 2, "PUT": 3, "DELETE": 4}.get(
        method.upper(), 0)

    words.append(pack_word32(OP_METHOD, m))
    words.append(pack_word32(OP_HOST, h))
    words.append(pack_word32(OP_PATH, p))
    words.append(pack_word32(OP_PORT, port & 0xFF))
    words.append(pack_word32(OP_RATE_CHECK))
    words.append(pack_word32(OP_DDOS_CHECK))
    words.append(pack_word32(OP_DNS_LOOKUP))
    words.append(pack_word32(OP_ENVELOPE))
    words.append(pack_word32(OP_DISPATCH))
    words.append(pack_word32(OP_HALT))
    return RequestTape(client_id=client_id, words=words,
                       names=names, n_requests=n_requests)


# ============================================================
#  8-bit DDoS redirect path
# ============================================================
class Redirect8:
    """
    A minimal 8-bit 1D path for redirecting suspicious requests
    away from the 64-bit pipeline.  Each redirected request
    becomes one byte on a rolling buffer.
    """
    def __init__(self, size: int = 256):
        self.buf = bytearray(size)
        self.head = 0
        self.count = 0

    def push(self, client_id: int, reason: int) -> int:
        byte = ((client_id & 0x0F) << 4) | (reason & 0x0F)
        self.buf[self.head] = byte
        self.head = (self.head + 1) % len(self.buf)
        self.count += 1
        return byte

    def contents(self) -> bytes:
        return bytes(self.buf)


# ============================================================
#  64-bit walker  ·  executes the tape at full width
# ============================================================
@dataclass
class ServerState:
    """64-bit execution state."""
    client_id: int
    method: int = 0
    host: str = ""
    path: str = ""
    port: int = 80
    rate_ok: bool = True
    ddos_ok: bool = True
    dns_fp: str = ""
    dns_ip: Optional[str] = None
    envelopes: List[NodeEnvelope] = field(default_factory=list)
    redispatched: bool = False
    redirect_reason: int = 0


class RequestWalker:
    """
    Walk the 32-bit tape at 64-bit width.  Every opcode is a
    branch into the 64-bit if/else tree.
    """
    def __init__(self,
                 request: RequestTape,
                 dns: DNSMobiusGate,
                 k_nodes: int = N_NODES,
                 rate_limit: int = 1000,
                 ddos_window_s: float = 1.0,
                 ddos_threshold: int = 200):
        self.request = request
        self.dns = dns
        self.k_nodes = k_nodes
        self.rate_limit = rate_limit
        self.ddos_window_s = ddos_window_s
        self.ddos_threshold = ddos_threshold
        self.state = ServerState(client_id=request.client_id)
        self.pc = 0
        self.trace: List[Tuple[int, int]] = []

    def _name(self, idx: int) -> str:
        for k, v in self.request.names.items():
            if v == idx:
                return k
        return ""

    def step(self) -> bool:
        if self.pc >= len(self.request.words):
            return False
        w = self.request.words[self.pc]
        self.pc += 1
        op, a, b, c = unpack_word32(w)
        self.trace.append((op, self.state.client_id))
        self._exec(op, a, b, c)
        return True

    def _exec(self, op: int, a: int, b: int, c: int) -> None:
        s = self.state
        if op == OP_METHOD:
            s.method = a
        elif op == OP_HOST:
            s.host = self._name(a)
        elif op == OP_PATH:
            s.path = self._name(a)
        elif op == OP_PORT:
            s.port = a
        elif op == OP_RATE_CHECK:
            s.rate_ok = self.request.n_requests < self.rate_limit
        elif op == OP_DDOS_CHECK:
            s.ddos_ok = self.request.n_requests < self.ddos_threshold
        elif op == OP_DNS_LOOKUP:
            if s.rate_ok and s.ddos_ok:
                res = self.dns.resolve(s.host)
                if res is not None:
                    s.dns_fp, s.dns_ip = res
        elif op == OP_ENVELOPE:
            if s.dns_ip is not None:
                for node_id in range(self.k_nodes):
                    s.envelopes.append(build_node_envelope(
                        node_id, s.host, s.dns_ip, s.dns_fp))
        elif op == OP_DISPATCH:
            # dispatch is a no-op in the walker — the envelopes are
            # the outbound payloads; the server collects them.
            pass
        elif op == OP_REDIRECT:
            s.redispatched = True
            s.redirect_reason = b
        elif op == OP_HALT:
            self.pc = len(self.request.words)

    def run(self) -> ServerState:
        while self.pc < len(self.request.words):
            self.step()
        # if rate-limit or DDoS fails, mark for 8-bit redirect
        if not self.state.rate_ok:
            self.state.redispatched = True
            self.state.redirect_reason = 1     # rate limit
        elif not self.state.ddos_ok:
            self.state.redispatched = True
            self.state.redirect_reason = 2     # ddos
        return self.state


# ============================================================
#  Server
# ============================================================
class MobiusDNSServer:
    """
    The server:

        • receives request tapes
        • walks them at 64-bit width
        • dispatches per-node envelopes to K nodes
        • redirects suspicious requests to the 8-bit path
    """
    def __init__(self, K: int = K_DEFAULT, n_nodes: int = N_NODES,
                 rate_limit: int = 1000,
                 ddos_threshold: int = 200):
        self.K = K
        self.n_nodes = n_nodes
        self.rate_limit = rate_limit
        self.ddos_threshold = ddos_threshold
        t0 = time.perf_counter()
        self.mu = mobius_sieve(K)
        self.dns = DNSMobiusGate(K=K)
        self.sieve_ms = (time.perf_counter() - t0) * 1e3
        self.redirect = Redirect8(size=256)

        # per-client request counters (for the DDoS gate)
        self.client_counters: Dict[int, int] = {}

        # outgoing envelopes queue (per node)
        self.outbox: Dict[int, deque] = {
            i: deque(maxlen=1024) for i in range(n_nodes)
        }

        # log
        self.log: List[Dict] = []

    def add_dns_record(self, domain: str, ip: str) -> str:
        return self.dns.add(domain, ip)

    def handle(self, request: RequestTape) -> Dict:
        # increment client counter
        c = self.client_counters.get(request.client_id, 0) + 1
        self.client_counters[request.client_id] = c
        request.n_requests = c

        t0 = time.perf_counter()

        # ---- fast path 1: rate limit hit → 8-bit redirect ----
        if c > self.rate_limit:
            b = self.redirect.push(request.client_id, reason=1)
            self.log.append(dict(client=request.client_id,
                                 verdict="rate_limit",
                                 redirect_byte=b))
            return dict(verdict="rate_limit", envelopes=[],
                        redirect_byte=b,
                        elapsed_ms=(time.perf_counter() - t0) * 1e3)

        # ---- fast path 2: DDoS threshold hit → 8-bit redirect ----
        if c > self.ddos_threshold:
            b = self.redirect.push(request.client_id, reason=2)
            self.log.append(dict(client=request.client_id,
                                 verdict="ddos",
                                 redirect_byte=b))
            return dict(verdict="ddos", envelopes=[],
                        redirect_byte=b,
                        elapsed_ms=(time.perf_counter() - t0) * 1e3)

        # ---- 32-bit tape walk at 64-bit width ----
        walker = RequestWalker(
            request, self.dns, self.n_nodes,
            rate_limit=self.rate_limit,
            ddos_threshold=self.ddos_threshold)
        state = walker.run()

        # ---- resolve verdict ----
        if state.redispatched:
            b = self.redirect.push(request.client_id,
                                   state.redirect_reason)
            verdict = "redirected"
            envelopes = []
        elif state.dns_ip is None:
            verdict = "dns_miss"
            envelopes = []
        else:
            verdict = "ok"
            envelopes = state.envelopes
            for env in envelopes:
                self.outbox[env.node_id].append(env)

        elapsed = (time.perf_counter() - t0) * 1e3
        self.log.append(dict(
            client=request.client_id,
            verdict=verdict,
            host=state.host,
            path=state.path,
            ip=state.dns_ip,
            n_envelopes=len(envelopes),
            steps=len(walker.trace),
            elapsed_ms=elapsed,
        ))
        return dict(
            verdict=verdict,
            state=state,
            envelopes=envelopes,
            walker_steps=len(walker.trace),
            elapsed_ms=elapsed,
        )

    def drain_outbox(self) -> Dict[int, List[NodeEnvelope]]:
        """Return and clear the per-node outbox."""
        out: Dict[int, List[NodeEnvelope]] = {}
        for i in range(self.n_nodes):
            out[i] = list(self.outbox[i])
            self.outbox[i].clear()
        return out

    def summary(self) -> Dict:
        counts: Dict[str, int] = {}
        for entry in self.log:
            counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
        return dict(
            n_requests=len(self.log),
            verdict_counts=counts,
            redirect_bytes=len(self.redirect.contents()),
            sieved_ms=self.sieve_ms,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Möbius DNS server  ·  32-bit tape → 64-bit walker → envelopes → K nodes")
    print("=" * 80)
    print(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  K       = {K_DEFAULT}")
    print(f"  N_nodes = {N_NODES}")
    print()

    server = MobiusDNSServer(K=K_DEFAULT, n_nodes=N_NODES,
                             rate_limit=10, ddos_threshold=50)

    # ---- seed DNS records ----
    print("--- DNS records ---")
    records = [
        ("example.com",  "93.184.216.34"),
        ("mobius.org",   "192.0.2.1"),
        ("localhost",    "127.0.0.1"),
        ("quiet.morning","10.0.0.42"),
    ]
    for d, ip in records:
        fp = server.add_dns_record(d, ip)
        print(f"  {d:15s} → {ip:15s}  fp = {fp[:16]}…")
    print()

    # ---- 32-bit tape walkthrough for one request ----
    print("--- 32-bit request tape  ·  'GET example.com/index.html' ---")
    req = build_request_tape(
        client_id=1, method="GET",
        host="example.com", path="/index.html",
        port=80, n_requests=1)
    print(f"  tape length : {len(req.words)}")
    for w in req.words:
        op, a, b, c = unpack_word32(w)
        print(f"    word 0x{w:08x}  op={OP_NAMES.get(op, '?'):>12s}  "
              f"a={a:3d}  b={b:3d}  c={c:3d}")
    print()

    # ---- single-request walkthrough ----
    print("--- single-request handling ---")
    result = server.handle(req)
    print(f"  verdict      : {result['verdict']}")
    print(f"  host         : {result['state'].host}")
    print(f"  ip           : {result['state'].dns_ip}")
    print(f"  dns_fp       : {result['state'].dns_fp[:16]}…")
    print(f"  envelopes    : {len(result['envelopes'])}")
    print(f"  walker steps : {result['walker_steps']}")
    print(f"  elapsed      : {result['elapsed_ms']:.3f} ms")
    print()
    print("  envelope summaries (first 4 nodes):")
    for env in result["envelopes"][:4]:
        s = env.summary()
        print(f"    node={s['node']}  ip={s['ip']:<15s}  "
              f"payload_b={s['payload_b']}  fp={s['fingerprint']}")
    print()

    # ---- drain the outbox ----
    print("--- drain outbox  ·  per-node deliveries ---")
    out = server.drain_outbox()
    for node_id, envs in out.items():
        print(f"  node {node_id}: {len(envs)} envelope(s)")
    print()

    # ---- batch of requests ----
    print("--- batch of requests ---")
    hosts = ["example.com", "mobius.org", "localhost",
             "quiet.morning", "unknown.com"]
    for i in range(1, 13):
        h = hosts[i % len(hosts)]
        tape = build_request_tape(
            client_id=100 + (i % 3), method="GET",
            host=h, path=f"/p{i}",
            port=80, n_requests=i)
        r = server.handle(tape)
        verdict = r["verdict"]
        ip = r["state"].dns_ip if r["state"] else None
        n_env = len(r["envelopes"])
        print(f"  req {i:>2d}  client={tape.client_id}  "
              f"host={h:15s}  verdict={verdict:<10s}  "
              f"ip={str(ip):<15s}  env={n_env}")
    print()

    # ---- flood the server to trigger DDoS redirect ----
    print("--- DDoS flood  ·  60 requests from one client ---")
    for i in range(60):
        tape = build_request_tape(
            client_id=999, method="GET",
            host="example.com", path=f"/flood{i}",
            port=80, n_requests=i)
        r = server.handle(tape)
    print()
    print("  outbox status:")
    for node_id, envs in server.outbox.items():
        print(f"    node {node_id}: {len(envs)} envelope(s) queued")
    print()
    print("  8-bit redirect buffer  ·  last 32 bytes:")
    buf = server.redirect.contents()
    nonzero = [b for b in buf if b != 0]
    print(f"    {[hex(b) for b in nonzero[-16:]]}")
    print(f"    total redirected: {len(nonzero)}")
    print()

    # ---- summary ----
    print("--- server summary ---")
    s = server.summary()
    print(f"  n_requests       : {s['n_requests']}")
    for k, v in sorted(s["verdict_counts"].items()):
        print(f"    {k:12s} : {v}")
    print(f"  sieve_ms         : {s['sieved_ms']:.2f}")
    print()

    # ---- complexity ----
    print("--- complexity ---")
    print("  build request tape           O(1)")
    print("  walk tape (64-bit)           O(K)")
    print("  hex-DNS lookup               O(K log log K) + O(K log K)")
    print("  envelope build (per node)    O(1)")
    print("  dispatch (K nodes)           O(K)")
    print("  8-bit redirect               O(1)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()