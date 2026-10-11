#!/usr/bin/env python3
"""
server_nodes_extended.py
========================
Extensions for server_nodes.py:

    · HexMobiusIPLimiter        hex-DNS fingerprint → IP → speed limit
    · MobiusReadOnlyBuffer      MOSB-framed read-only chunk buffer
    · SiteExpander              expand / reload state machine
    · ExtendedDNSServer         wires everything into MobiusDNSServer

Import order
------------
    from server_nodes import (
        MobiusDNSServer, DNSMobiusGate, MobiusHexGate,
        build_request_tape, pack_word32, unpack_word32,
        MobiusPacker, mobius_sieve,
        OP_METHOD, OP_HOST, OP_PATH, OP_PORT,
        OP_RATE_CHECK, OP_DDOS_CHECK, OP_DNS_LOOKUP,
        OP_ENVELOPE, OP_DISPATCH, OP_HALT, OP_REDIRECT,
        OP_NAMES, RequestTape,
    )
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
#  Import shims  ·  make this file runnable standalone
# ============================================================
try:
    from server_nodes import (                       # type: ignore
        MobiusDNSServer, DNSMobiusGate, MobiusHexGate,
        build_request_tape, pack_word32, unpack_word32,
        mobius_sieve, RequestTape,
        OP_METHOD, OP_HOST, OP_PATH, OP_PORT,
        OP_RATE_CHECK, OP_DDOS_CHECK, OP_DNS_LOOKUP,
        OP_ENVELOPE, OP_DISPATCH, OP_HALT, OP_REDIRECT,
        OP_NAMES,
        PI, E, ALPHA_SYM, DENSITY, K_DEFAULT, N_NODES,
    )
except ImportError:
    # Fallback: define the constants and the small ones we need
    PI        = math.pi
    E         = math.e
    ALPHA_SYM = 1.0 / (PI - E)
    DENSITY   = 6.0 / (PI * PI)
    K_DEFAULT = 128
    N_NODES   = 8


# ============================================================
#  New opcodes for the extended tape
# ============================================================
OP_EXPAND = 0x10        # request the full page
OP_RELOAD = 0x11        # re-request the current page
OP_MODE   = 0x12        # set session mode (0=minimal, 1=expanded)
OP_BUFFER = 0x13        # read from the read-only buffer
OP_IP_CHECK = 0x14      # hex-DNS IP speed limit gate

OP_NAMES_EXT = {
    **({} if "OP_NAMES" not in dir() else OP_NAMES),
    0x10: "EXPAND", 0x11: "RELOAD", 0x12: "MODE",
    0x13: "BUFFER", 0x14: "IP_CHECK",
}


# ============================================================
#  MOSB  ·  read-only buffer with Möbius framing
# ============================================================
MAGIC       = b"MOSB"
HEADER_FMT  = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)      # 48 bytes


class MobiusReadOnlyBuffer:
    """
    Read-only chunked buffer with per-chunk Möbius invariants.

    Layout (per chunk):
        [ MAGIC(4) | K(4) | n(8) | cid(8) | S(8) | H(8) | m(8) | payload ]

    The buffer is written once (by the site owner) and read-only
    afterwards.  Reads yield a chunk plus its invariants; the reader
    can verify the supertrace without trusting the writer.
    """
    def __init__(self, K: int = 512):
        self.K = int(K)
        self.mu = mobius_sieve(K)
        self.data = bytearray()
        self.chunk_index: List[Tuple[int, int, float, float, float]] = []
        self.read_pos = 0
        self._cid = 0

    # --------------------------------------------------------
    @staticmethod
    def _supertrace(payload: bytes) -> float:
        S = 0.0
        for t, b in enumerate(payload):
            v = (b - 128) / 128.0
            S += v if (t % 2 == 0) else -v
        return S

    @staticmethod
    def _entropy_mass(S: float, n: int) -> Tuple[float, float]:
        if S == 0.0 or n <= 0:
            return 0.0, 0.0
        p = min(abs(S) / n, 0.9999)
        H = -ALPHA_SYM * p * math.log(p) if p > 0 else 0.0
        return H, abs(S) * math.exp(-H)

    # --------------------------------------------------------
    def append(self, payload: bytes) -> int:
        """Write a chunk into the buffer.  Returns the chunk id."""
        n = len(payload)
        S = self._supertrace(payload)
        H, m = self._entropy_mass(S, n)
        cid = self._cid
        self._cid += 1
        header = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        start = len(self.data)
        self.data.extend(header + payload)
        self.chunk_index.append((start, start + HEADER_SIZE + n, S, H, m))
        return cid

    # --------------------------------------------------------
    def read_chunk(self, cid: int) -> Optional[Tuple[bytes, float, float, float]]:
        """Read a chunk by id.  Returns (payload, S, H, m) or None."""
        if cid < 0 or cid >= len(self.chunk_index):
            return None
        start, end, S_hdr, H_hdr, m_hdr = self.chunk_index[cid]
        blob = bytes(self.data[start:end])
        magic, K, n, cid_stored, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])
        if magic != MAGIC or K != self.K or cid_stored != cid:
            return None
        payload = blob[HEADER_SIZE:HEADER_SIZE + n]
        # recompute supertrace locally, don't trust the header
        S_check = self._supertrace(payload)
        if abs(S_check - S) > 1e-9:
            return None
        return payload, S, H, m

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_index)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# ============================================================
#  Hex-DNS IP speed limiter
# ============================================================
class HexMobiusIPLimiter:
    """
    Rate limit keyed by the hex-DNS fingerprint of the client's IP.

    For each request:
        fp = hex_dns_fingerprint(ip)
        bucket = fp[:8]                        (32-bit key)
        check bucket's request count in the sliding window

    Per-client limits:
        rps_limit    requests per second
        burst_limit  instantaneous burst
        cooldown_s   penalty duration after violation
    """
    def __init__(self,
                 hex_gate: MobiusHexGate,
                 rps_limit: float = 20.0,
                 burst_limit: int = 40,
                 cooldown_s: float = 5.0,
                 window_s: float = 1.0):
        self.gate = hex_gate
        self.rps_limit = float(rps_limit)
        self.burst_limit = int(burst_limit)
        self.cooldown_s = float(cooldown_s)
        self.window_s = float(window_s)
        # fp → deque of timestamps
        self.history: Dict[str, deque] = {}
        # fp → cooldown_until
        self.cooldowns: Dict[str, float] = {}

    # --------------------------------------------------------
    def fingerprint(self, ip: str) -> str:
        return self.gate.fingerprint(ip)

    # --------------------------------------------------------
    def check(self, ip: str) -> Dict:
        """
        Returns
        -------
        dict with
            fp          hex-DNS fingerprint of the IP
            allowed     bool
            reason      'ok' | 'cooldown' | 'burst' | 'rps'
            count       count in the current window
            cooldown_s  remaining cooldown seconds (if any)
        """
        now = time.time()
        fp = self.fingerprint(ip)

        # cooldown active?
        cd_until = self.cooldowns.get(fp, 0.0)
        if now < cd_until:
            return dict(fp=fp, allowed=False, reason="cooldown",
                        count=0, cooldown_s=round(cd_until - now, 3))

        # sliding window
        dq = self.history.setdefault(fp, deque())
        cutoff = now - self.window_s
        while dq and dq[0] < cutoff:
            dq.popleft()

        count = len(dq)
        if count + 1 > self.burst_limit:
            self.cooldowns[fp] = now + self.cooldown_s
            return dict(fp=fp, allowed=False, reason="burst",
                        count=count, cooldown_s=self.cooldown_s)
        if count + 1 > self.rps_limit * self.window_s:
            self.cooldowns[fp] = now + self.cooldown_s * 0.5
            return dict(fp=fp, allowed=False, reason="rps",
                        count=count, cooldown_s=self.cooldown_s * 0.5)

        dq.append(now)
        return dict(fp=fp, allowed=True, reason="ok",
                    count=count + 1, cooldown_s=0.0)

    # --------------------------------------------------------
    def summary(self) -> Dict:
        now = time.time()
        active_cd = sum(1 for t in self.cooldowns.values() if t > now)
        return dict(
            tracked_fps=len(self.history),
            active_cooldowns=active_cd,
            rps_limit=self.rps_limit,
            burst_limit=self.burst_limit,
            window_s=self.window_s,
        )


# ============================================================
#  Session  ·  per-client expand / reload state
# ============================================================
@dataclass
class Session:
    client_id: int
    ip: str
    ip_fp: str
    mode: int = 0                          # 0 = minimal, 1 = expanded
    current_path: str = "/"
    expanded_at: Optional[float] = None
    n_reloads: int = 0
    n_expands: int = 0
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()


# ============================================================
#  Site expander  ·  expand / reload state machine
# ============================================================
class SiteExpander:
    """
    State machine:

        MODE 0 (minimal)  →  serve a stub
        EXPAND            →  load the full site, MODE 1
        RELOAD            →  re-request current_path, MODE stays 1
        MODE 1 → MODE 0   →  shrink back to minimal (session idle)

    The read-only buffer holds the page bodies keyed by path.
    """
    def __init__(self,
                 buffer: MobiusReadOnlyBuffer,
                 idle_timeout_s: float = 60.0):
        self.buffer = buffer
        self.idle_timeout_s = idle_timeout_s
        # path → chunk_id (into the read-only buffer)
        self.path_to_chunk: Dict[str, int] = {}
        # minimal stub for MODE 0
        self.stub = b"<html><body>tap to expand</body></html>"

    # --------------------------------------------------------
    def register_path(self, path: str, payload: bytes) -> int:
        cid = self.buffer.append(payload)
        self.path_to_chunk[path] = cid
        return cid

    # --------------------------------------------------------
    def serve_minimal(self, session: Session) -> Dict:
        return dict(
            mode=0,
            status="minimal",
            body=self.stub,
            body_size=len(self.stub),
            S=0.0, H=0.0, m=0.0,
            path=session.current_path,
        )

    # --------------------------------------------------------
    def serve_expand(self, session: Session) -> Dict:
        cid = self.path_to_chunk.get(session.current_path)
        if cid is None:
            # path unknown; return minimal for this request
            return dict(mode=session.mode, status="not_found",
                        body=b"", body_size=0,
                        S=0.0, H=0.0, m=0.0,
                        path=session.current_path)
        chunk = self.buffer.read_chunk(cid)
        if chunk is None:
            return dict(mode=session.mode, status="buffer_error",
                        body=b"", body_size=0,
                        S=0.0, H=0.0, m=0.0,
                        path=session.current_path)
        payload, S, H, m = chunk
        session.mode = 1
        session.expanded_at = time.time()
        session.n_expands += 1
        return dict(mode=1, status="expanded",
                    body=payload, body_size=len(payload),
                    S=S, H=H, m=m, path=session.current_path)

    # --------------------------------------------------------
    def serve_reload(self, session: Session) -> Dict:
        if session.mode == 0:
            # reloading from minimal → treat as expand
            return self.serve_expand(session)
        session.n_reloads += 1
        return self.serve_expand(session)

    # --------------------------------------------------------
    def maybe_shrink(self, session: Session) -> bool:
        """Return True if the session should drop back to MODE 0."""
        if session.mode == 0 or session.expanded_at is None:
            return False
        idle = time.time() - session.last_seen
        if idle > self.idle_timeout_s:
            session.mode = 0
            session.expanded_at = None
            return True
        return False


# ============================================================
#  Extended tape builder  ·  MODE / EXPAND / RELOAD / BUFFER / IP
# ============================================================
def build_extended_tape(client_id: int, ip: str,
                        method: str, host: str, path: str,
                        port: int, n_requests: int,
                        mode: int = 0,
                        action: str = "minimal") -> "RequestTape":
    """
    Extended request tape:

        METHOD, HOST, PATH, PORT
        IP_CHECK  (hex-DNS IP speed limit)
        RATE_CHECK
        DDOS_CHECK
        MODE      (0 = minimal, 1 = expanded)
        <BUFFER | EXPAND | RELOAD | DNS_LOOKUP>
        ENVELOPE
        DISPATCH
        HALT
    """
    # Import from server_nodes if available; else minimal fallback
    try:
        from server_nodes import pack_word32 as pw, RequestTape as RT
    except ImportError:
        def pw(op, a=0, b=0, c=0):
            return (((op & 0xFF) << 24) | ((a & 0xFF) << 16)
                    | ((b & 0xFF) << 8) | (c & 0xFF))
        @dataclass
        class RT:
            client_id: int
            words: List[int]
            names: Dict[str, int]
            n_requests: int

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

    # core identity
    words.append(pw(OP_METHOD, m))
    words.append(pw(OP_HOST, h))
    words.append(pw(OP_PATH, p))
    words.append(pw(OP_PORT, port & 0xFF))
    words.append(pw(OP_IP_CHECK, (hash(ip) >> 8) & 0xFF))
    words.append(pw(OP_RATE_CHECK))
    words.append(pw(OP_DDOS_CHECK))
    words.append(pw(OP_MODE, mode & 0xFF))

    # action branch
    action = action.lower()
    if action == "expand":
        words.append(pw(OP_EXPAND))
    elif action == "reload":
        words.append(pw(OP_RELOAD))
    elif action == "buffer":
        words.append(pw(OP_BUFFER))
    else:
        words.append(pw(OP_DNS_LOOKUP))

    words.append(pw(OP_ENVELOPE))
    words.append(pw(OP_DISPATCH))
    words.append(pw(OP_HALT))

    return RT(client_id=client_id, words=words,
              names=names, n_requests=n_requests)


# ============================================================
#  Extended server  ·  wraps MobiusDNSServer with the new features
# ============================================================
class ExtendedDNSServer:
    """
    Extends the base MobiusDNSServer with:

        · hex-DNS IP speed limiting
        · read-only buffer for page bodies
        · expand / reload state machine
    """
    def __init__(self,
                 K: int = K_DEFAULT,
                 n_nodes: int = N_NODES,
                 rate_limit: int = 1000,
                 ddos_threshold: int = 200,
                 ip_rps: float = 20.0,
                 ip_burst: int = 40,
                 ip_cooldown_s: float = 5.0,
                 idle_timeout_s: float = 60.0):
        # base server
        try:
            self.base = MobiusDNSServer(
                K=K, n_nodes=n_nodes,
                rate_limit=rate_limit,
                ddos_threshold=ddos_threshold)
            self.hex_gate = self.base.dns.gate
        except NameError:
            # standalone fallback: build minimal components
            self.base = None
            self.hex_gate = None

        self.K = K
        self.n_nodes = n_nodes
        self.ip_limiter = HexMobiusIPLimiter(
            hex_gate=self.hex_gate,
            rps_limit=ip_rps,
            burst_limit=ip_burst,
            cooldown_s=ip_cooldown_s)
        self.buffer = MobiusReadOnlyBuffer(K=K)
        self.expander = SiteExpander(self.buffer,
                                     idle_timeout_s=idle_timeout_s)

        self.sessions: Dict[int, Session] = {}
        self.log: List[Dict] = []

    # --------------------------------------------------------
    def _session(self, client_id: int, ip: str) -> Session:
        s = self.sessions.get(client_id)
        if s is None:
            fp = self.ip_limiter.fingerprint(ip)
            s = Session(client_id=client_id, ip=ip, ip_fp=fp)
            self.sessions[client_id] = s
        s.touch()
        return s

    # --------------------------------------------------------
    def register_page(self, path: str, body: bytes) -> int:
        return self.expander.register_path(path, body)

    # --------------------------------------------------------
    def handle_extended(self,
                        client_id: int,
                        ip: str,
                        host: str,
                        path: str,
                        action: str = "minimal") -> Dict:
        """
        Extended request:

            1. IP speed limit (hex-DNS keyed)
            2. session lookup
            3. state machine: minimal → expand → reload
            4. read-only buffer for the page body
        """
        t0 = time.perf_counter()

        # ---- 1. IP speed limit ----
        lim = self.ip_limiter.check(ip)
        if not lim["allowed"]:
            self.log.append(dict(client=client_id, ip=ip,
                                 verdict=f"ip_{lim['reason']}",
                                 fp=lim["fp"][:16]))
            return dict(verdict=f"ip_{lim['reason']}",
                        allowed=False,
                        reason=lim["reason"],
                        ip_fp=lim["fp"][:16],
                        cooldown_s=lim["cooldown_s"],
                        mode=0,
                        body=b"",
                        elapsed_ms=(time.perf_counter() - t0) * 1e3)

        # ---- 2. session ----
        session = self._session(client_id, ip)
        session.current_path = path

        # ---- 3. state machine ----
        action = action.lower()
        if action == "expand":
            result = self.expander.serve_expand(session)
        elif action == "reload":
            result = self.expander.serve_reload(session)
        elif action == "minimal":
            result = self.expander.serve_minimal(session)
        else:
            # default: minimal if not expanded, expand if already 1
            if session.mode == 1:
                result = self.expander.serve_reload(session)
            else:
                result = self.expander.serve_minimal(session)

        result.update(dict(
            client=client_id,
            ip=ip,
            ip_fp=lim["fp"][:16],
            ip_count=lim["count"],
            verdict="ok",
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        ))
        self.log.append(dict(
            client=client_id, ip=ip,
            ip_fp=lim["fp"][:16],
            verdict=result["status"],
            mode=result["mode"],
            body_size=result["body_size"],
            S=result["S"], m=result["m"],
            elapsed_ms=result["elapsed_ms"]))
        return result

    # --------------------------------------------------------
    def summary(self) -> Dict:
        counts: Dict[str, int] = {}
        for e in self.log:
            counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
        return dict(
            n_requests=len(self.log),
            verdict_counts=counts,
            sessions=len(self.sessions),
            buffer_chunks=self.buffer.n_chunks,
            buffer_bytes=self.buffer.size_bytes,
            ip_limiter=self.ip_limiter.summary(),
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Extended DNS server  ·  hex-IP limit  ·  RO buffer  ·  expand/reload")
    print("=" * 80)
    print(f"  α = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  K             = {K_DEFAULT}")
    print(f"  n_nodes       = {N_NODES}")
    print()

    server = ExtendedDNSServer(
        K=K_DEFAULT, n_nodes=N_NODES,
        ip_rps=10.0, ip_burst=15, ip_cooldown_s=3.0,
        idle_timeout_s=5.0,
    )

    # ---- register pages in the read-only buffer ----
    print("--- register pages in the read-only buffer ---")
    pages = {
    "/":            "<html><body>home  ·  minimal</body></html>".encode("utf-8"),
    "/index.html":  ("<html><body>expanded  ·  index</body></html>" * 8).encode("utf-8"),
    "/about":       ("<html><body>about page  ·  full content</body></html>" * 16).encode("utf-8"),
    "/contact":     ("<html><body>contact</body></html>" * 4).encode("utf-8"),
}
    for path, body in pages.items():
        cid = server.register_page(path, body)
        print(f"  {path:15s}  chunk_id={cid}  body={len(body)} B")
    print()
    print(f"  buffer: {server.buffer.n_chunks} chunks  "
          f"{server.buffer.size_bytes} bytes")
    print()

    # ---- scenario 1: minimal, then expand, then reload ----
    print("--- scenario 1  ·  minimal → expand → reload ---")
    client, ip = 1, "198.51.100.7"
    for action in ["minimal", "expand", "reload", "reload"]:
        r = server.handle_extended(client, ip, "example.com",
                                   "/index.html", action=action)
        print(f"  action={action:8s}  mode={r['mode']}  "
              f"status={r['status']:10s}  body={r['body_size']:6d} B  "
              f"S={r['S']:+.4f}  m={r['m']:.4f}  "
              f"({r['elapsed_ms']:.3f} ms)")
    print()

    # ---- scenario 2: IP speed limit ----
    print("--- scenario 2  ·  IP speed limit (burst=15) ---")
    for i in range(20):
        r = server.handle_extended(2, "203.0.113.42", "example.com",
                                   "/index.html", action="expand")
        if not r["allowed"]:
            print(f"  req {i:>2d}  verdict={r['verdict']:14s}  "
                  f"reason={r.get('reason')}  "
                  f"cooldown={r.get('cooldown_s', 0):.2f}s  "
                  f"fp={r['ip_fp']}")
            if i > 16:
                break
        else:
            if i < 3 or i > 14:
                print(f"  req {i:>2d}  allowed=True  "
                      f"count={r['ip_count']}  mode={r['mode']}  "
                      f"body={r['body_size']} B")
    print()

    # ---- scenario 3: many distinct IPs, no collisions ----
    print("--- scenario 3  ·  distinct IPs ---")
    for i in range(5):
        ip_i = f"10.0.0.{10 * i + 5}"
        r = server.handle_extended(100 + i, ip_i, "example.com",
                                   "/", action="minimal")
        print(f"  ip={ip_i:15s}  fp={r['ip_fp']}  "
              f"allowed={r['allowed']}  status={r['status']}")
    print()

    # ---- scenario 4: idle timeout → shrink to minimal ----
    print("--- scenario 4  ·  idle → auto-shrink ---")
    session = server.sessions[1]
    print(f"  before: mode={session.mode}  "
          f"idle={time.time() - session.last_seen:.2f}s")
    time.sleep(5.5)
    shrunk = server.expander.maybe_shrink(session)
    print(f"  after : mode={session.mode}  shrunk={shrunk}")
    print()

    # ---- summary ----
    print("--- summary ---")
    s = server.summary()
    print(f"  n_requests       : {s['n_requests']}")
    for k, v in sorted(s["verdict_counts"].items()):
        print(f"    {k:14s} : {v}")
    print(f"  sessions         : {s['sessions']}")
    print(f"  buffer chunks    : {s['buffer_chunks']}")
    print(f"  buffer bytes     : {s['buffer_bytes']}")
    print(f"  tracked IP FPs   : {s['ip_limiter']['tracked_fps']}")
    print(f"  active cooldowns : {s['ip_limiter']['active_cooldowns']}")
    print()

    # ---- complexity ----
    print("--- complexity ---")
    print("  hex-DNS IP fingerprint     O(K log log K) + O(K log K)   per IP")
    print("  IP speed limit check       O(1)  amortized (sliding window)")
    print("  read-only buffer read      O(1)  per chunk (header + payload)")
    print("  supertrace verify (read)   O(n)  per chunk")
    print("  expand / reload            O(1)  + buffer read")
    print("  idle shrink                O(1)  per session")
    print()


if __name__ == "__main__":
    demo()