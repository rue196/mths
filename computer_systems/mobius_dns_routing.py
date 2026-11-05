#!/usr/bin/env python3
"""
mobius_dns_router_v2.py
=======================

DNS router with:

    · sorting        quicksort  O(K log K)       (replaces TSP 1-D)
    · wifi sense     AC_DC_elliptic_mobius_tau_muon_spectral_decomp
                     → neighbor router positions from spectral peaks
    · client recv    receiver_module
                     → client geo from raw particles
    · response send  mobius_internet.transmit
                     → entangled SAT envelope + MOSB + supertrace

Wire protocol
-------------
    HELLO (0x01) : [u16 id_len][id][u16 nonce_len][nonce]
    ACK   (0x02) : [32B server_fp][16B server_nonce][32B check]
    QUERY (0x40) : [u16 domain_len][domain]
                   [u32 n_particles]
                   [per-particle: u32 idx, 7×double (t,r,Π,x,y,zt,vx)]
    QRESP (0x41) : [u32 n_records][double client_t][double client_r]
                   [per-record: u16 trit_len][trits][u16 ttl][u16 prio]
                   [32B transmit_sha_in][32B transmit_sha_out][u8 round_trip]
    CLOSE (0x04) : [u8]
    ERROR (0xFF) : [u16 msg_len][msg]

Complexity
----------
    quicksort                O(m log m)      m = #records for the domain
    wifi spectral sense      O(N_samples·K² + K log K)  once per sense
    client reception         O(n_particles · K_vec)
    entangled transmit       O(K_sat + K_pack)  via mobius_internet
"""

from __future__ import annotations
import hashlib, hmac, json, math, os, socket, struct, threading, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- project imports (from X import Y) ----
from AC_DC_elliptic_mobius_tau_muon_spectral_decomp import (
    SpectralDensityMobius,
)
from receiver_module import (
    mobius_sieve as mobius_sieve_rcv,
    mu_convolution_H,
    build_particles,
    receive as receiver_receive,
    estimate_device_location,
    PointParticle,
    elliptic_projection,
    elliptic_projection_vec,
)
from mobius_internet import (
    MobiusInternet,
    MobiusTransport,
    TransmitResult,
)
from ip_address_mobius_packed import (
    MobiusIP,
    IPMode,
    build_address,
    pack_trits,
    unpack_trits,
    packed_size,
)


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)
DENSITY = 6.0 / (PI * PI)
A_STEP  = ALPHA / 0.3628

DEFAULT_K         = 80
DEFAULT_N_EVENTS  = 24
K_TRANSPORT       = 512
K_VEC_DEFAULT     = 128
K_MU_TABLE        = 256

MSG_HELLO = 0x01
MSG_ACK   = 0x02
MSG_QUERY = 0x40
MSG_QRESP = 0x41
MSG_CLOSE = 0x04
MSG_ERROR = 0xFF


# ============================================================
#  Quicksort  ·  O(K log K)
# ============================================================
def quicksort_records(records: List, key=None) -> List:
    """
    Iterative quicksort with median-of-three pivot.

    Time:  O(K log K) average
    Space: O(log K) stack depth  (iterative, smaller partition first)

    Replaces tspsp.tsp_1d as the sorting primitive.
    """
    if key is None:
        key = lambda x: x
    arr = list(records)
    if len(arr) < 2:
        return arr

    stack = [(0, len(arr) - 1)]
    while stack:
        lo, hi = stack.pop()
        while lo < hi:
            mid = (lo + hi) // 2
            if key(arr[lo]) > key(arr[mid]):
                arr[lo], arr[mid] = arr[mid], arr[lo]
            if key(arr[mid]) > key(arr[hi]):
                arr[mid], arr[hi] = arr[hi], arr[mid]
            if key(arr[lo]) > key(arr[mid]):
                arr[lo], arr[mid] = arr[mid], arr[lo]

            pivot = key(arr[mid])
            i, j = lo, hi
            while i <= j:
                while key(arr[i]) < pivot:
                    i += 1
                while key(arr[j]) > pivot:
                    j -= 1
                if i <= j:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
                    j -= 1

            if j - lo < hi - i:
                if lo < j:
                    stack.append((lo, j))
                lo = i
            else:
                if i < hi:
                    stack.append((i, hi))
                hi = j

    return arr


# ============================================================
#  DNS record
# ============================================================
@dataclass
class DNSRecord:
    domain:   str
    ip:       MobiusIP
    ttl:      int = 300
    priority: int = 10
    added_at: float = field(default_factory=time.time)

    @property
    def scalar(self) -> int:
        v = 0
        for t in self.ip.trits:
            v = v * 3 + t
        return v

    def summary(self) -> str:
        return (f"{self.domain:<24s}  {self.ip.to_dotted()}  "
                f"ttl={self.ttl}  pri={self.priority}  "
                f"scalar={self.scalar}")


# ============================================================
#  Record store  ·  quicksort, no TSP
# ============================================================
class DNSRecordStore:
    def __init__(self, K: int = DEFAULT_K, n_events: int = DEFAULT_N_EVENTS):
        self.K = K
        self.n_trits = K // 4
        self.n_events = n_events
        self.records: Dict[str, List[DNSRecord]] = {}

    def register(self, domain: str, geo: Tuple[float, float],
                 n_ips: int = 1, mode: IPMode = IPMode.STATIC,
                 ttl: int = 300, priority: int = 10) -> List[DNSRecord]:
        out: List[DNSRecord] = []
        for i in range(max(1, n_ips)):
            variant = f"{domain}#{i}" if n_ips > 1 else domain
            trits = build_address(variant, geo, mode,
                                  self.K, self.n_events)
            ip = MobiusIP(tuple(trits))
            out.append(DNSRecord(domain=domain, ip=ip,
                                 ttl=ttl, priority=priority))
        self.records.setdefault(domain, []).extend(out)
        return out

    def query(self, domain: str) -> List[DNSRecord]:
        recs = list(self.records.get(domain, []))
        return quicksort_records(recs, key=lambda r: r.scalar)

    def query_all(self) -> List[DNSRecord]:
        all_recs = [r for rs in self.records.values() for r in rs]
        return quicksort_records(all_recs, key=lambda r: r.scalar)

    def all_records(self) -> List[DNSRecord]:
        return [r for rs in self.records.values() for r in rs]


# ============================================================
#  WiFi geo-router  ·  senses neighbor routers
# ============================================================
class WiFiGeoRouter:
    """
    Uses SpectralDensityMobius to sense the local RF spectrum and
    extract candidate neighbor router positions from spectral peaks.

    Each peak maps to a router-like position:
        t_router = t[peak_idx]            router's t-coord
        r_router = 1/f_k = λ_k            arc-length proxy
        strength = P_k                    signal power
        channel  = muon / tau             router type
    """
    def __init__(self, K: int = 20, J_C: float = 1.0,
                 omega: float = 1.0, N_samples: int = 256):
        self.spectral = SpectralDensityMobius(
            K=K, J_C=J_C, omega=omega,
            N_samples=N_samples, smooth=True)

    def sense(self) -> List[Dict]:
        spec_f = self.spectral.spec_f
        spec_P = self.spectral.spec_P
        spec_ch = self.spectral.spec_channel
        peak_idx = self.spectral.peak_idx
        t = self.spectral.t

        neighbors: List[Dict] = []
        for k in range(spec_f.size):
            f_k = float(spec_f[k])
            if f_k <= 1e-12:
                continue
            P_k = float(spec_P[k])
            ch = str(spec_ch[k])
            t_k = float(t[peak_idx[k]]) if k < peak_idx.size else 0.0
            lam = 1.0 / f_k
            neighbors.append(dict(
                t=t_k, r=lam,
                f=f_k, strength=P_k, channel=ch,
            ))
        return neighbors

    def sense_summary(self) -> Dict:
        s = self.spectral
        return dict(
            n_peaks=int(s.peak_idx.size),
            n_muon=int(s.muon_idx.size),
            n_tau=int(s.tau_idx.size),
            mean_power=float(s.power.mean()),
            max_power=float(s.power.max()),
            E_max=float(abs(s.E_total).max()),
            n_electrons=int(s.electron_t.size),
        )


# ============================================================
#  Wire-level particle serialization
# ============================================================
PARTICLE_FMT  = ">I7d"
PARTICLE_SIZE = struct.calcsize(PARTICLE_FMT)   # 60 bytes


def _pack_particles(particles: List[PointParticle]) -> bytes:
    out = struct.pack(">I", len(particles))
    for p in particles:
        out += struct.pack(PARTICLE_FMT,
                           p.idx, p.t, p.r, p.Pi,
                           p.x, p.y, p.zt, p.vx)
    return out


def _unpack_particles(data: bytes) -> List[PointParticle]:
    n = struct.unpack(">I", data[:4])[0]
    off = 4
    out: List[PointParticle] = []
    for _ in range(n):
        idx, t, r, Pi, x, y, zt, vx = struct.unpack(
            PARTICLE_FMT, data[off:off + PARTICLE_SIZE])
        off += PARTICLE_SIZE
        out.append(PointParticle(
            idx=int(idx), t=t, r=r, Pi=Pi,
            x=x, y=y, zt=zt, vx=vx,
        ))
    return out


def _pack_query(domain: str,
                particles: List[PointParticle]) -> bytes:
    db = domain.encode("utf-8")
    return (struct.pack(">H", len(db)) + db
            + _pack_particles(particles))


def _unpack_query(payload: bytes
                  ) -> Tuple[str, List[PointParticle]]:
    l = struct.unpack(">H", payload[:2])[0]
    domain = payload[2:2+l].decode("utf-8")
    particles = _unpack_particles(payload[2+l:])
    return domain, particles


def _pack_qresp(records: List[DNSRecord],
                client_geo: Tuple[float, float],
                transmit_summary: Dict) -> bytes:
    out = struct.pack(">I dd",
                      len(records),
                      float(client_geo[0]),
                      float(client_geo[1]))
    for r in records:
        tb = pack_trits(r.ip.trits)
        out += struct.pack(">H", len(tb)) + tb
        out += struct.pack(">HH", r.ttl, r.priority)
    sha_in  = transmit_summary.get("sha_in", "")[:32].encode().ljust(32, b"\x00")
    sha_out = transmit_summary.get("sha_out", "")[:32].encode().ljust(32, b"\x00")
    rt      = 1 if transmit_summary.get("round_trip") else 0
    out += sha_in + sha_out + struct.pack(">B", rt)
    return out


def _unpack_qresp(payload: bytes, n_trits: int
                  ) -> Tuple[List[DNSRecord],
                             Tuple[float, float],
                             Dict]:
    n, ct, cr = struct.unpack(">I dd", payload[:20])
    off = 20
    out: List[DNSRecord] = []
    for _ in range(n):
        l = struct.unpack(">H", payload[off:off+2])[0]; off += 2
        tb = payload[off:off+l]; off += l
        trits = unpack_trits(tb, n_trits)
        ttl, prio = struct.unpack(">HH", payload[off:off+4]); off += 4
        out.append(DNSRecord(domain="",
                             ip=MobiusIP(tuple(trits)),
                             ttl=ttl, priority=prio))
    sha_in  = payload[off:off+32].rstrip(b"\x00").decode(); off += 32
    sha_out = payload[off:off+32].rstrip(b"\x00").decode(); off += 32
    rt      = bool(payload[off]); off += 1
    return out, (ct, cr), dict(
        sha_in=sha_in, sha_out=sha_out, round_trip=rt,
    )


# ============================================================
#  DNS router  ·  combines all four layers
# ============================================================
class DNSRouter:
    def __init__(self,
                 router_domain: str = "router.mobius.local",
                 router_geo: Tuple[float, float] = (1.2, 0.9),
                 K: int = DEFAULT_K,
                 n_events: int = DEFAULT_N_EVENTS,
                 K_wifi: int = 20,
                 N_wifi: int = 256,
                 K_vec: int = K_VEC_DEFAULT,
                 verbose: bool = True):
        self.router_domain = router_domain
        self.router_geo = router_geo
        self.K = K
        self.n_trits = K // 4
        self.K_vec = K_vec
        self.verbose = verbose

        # 1. record store (quicksort)
        self.store = DNSRecordStore(K=K, n_events=n_events)

        # 2. WiFi geo-sensing
        self.wifi = WiFiGeoRouter(K=K_wifi, N_samples=N_wifi)

        # 3. entangled transport
        self.internet = MobiusInternet(
            K_dns=128, K_sat=128, K_pack=K_TRANSPORT)

        # 4. receiver-side tables (loaded once)
        self._F, self._mu_table, self._H = mu_convolution_H(K_MU_TABLE)
        self._mu_vec = mobius_sieve_rcv(K_vec)

        # caches
        self._neighbors: Optional[List[Dict]] = None
        self._sensing_summary: Optional[Dict] = None

        if verbose:
            print(f"--- DNS router {router_domain} @ {router_geo} ---")
            print(f"    K              = {self.K}")
            print(f"    n_trits        = {self.n_trits}")
            print(f"    K_wifi         = {K_wifi}")
            print(f"    K_vec          = {self.K_vec}")

    # --------------------------------------------------------
    def sense_neighbors(self, force: bool = False) -> Dict:
        if self._neighbors is None or force:
            self._neighbors = self.wifi.sense()
            self._sensing_summary = self.wifi.sense_summary()
        return dict(neighbors=self._neighbors,
                    summary=self._sensing_summary)

    # --------------------------------------------------------
    def receive_client(self, particles: List[PointParticle]) -> Dict:
        result = receiver_receive(
            particles, self.K_vec, self._mu_vec,
            k_labels=4, threshold=0.3)
        loc = estimate_device_location(result)
        return dict(
            location=loc,
            n_particles=len(particles),
            n_survivors=len(result["gate_survivors"]),
            even_even_kept=int(np.sum(result["even_even_mask"])),
        )

    # --------------------------------------------------------
    def resolve(self, domain: str) -> List[DNSRecord]:
        return self.store.query(domain)

    # --------------------------------------------------------
    def transmit_response(self,
                          client_domain: str,
                          domain: str,
                          records: List[DNSRecord],
                          client_geo: Tuple[float, float]
                          ) -> TransmitResult:
        payload = {
            "query": domain,
            "n": len(records),
            "records": [r.summary() for r in records],
            "client_geo": list(client_geo),
        }
        message = json.dumps(payload, separators=(",", ":"))
        seed = int(abs(client_geo[0]) * 1000) % (2 ** 31)
        return self.internet.transmit(
            source=self.router_domain,
            dest=client_domain,
            message=message,
            geo_seed=seed,
        )

    # --------------------------------------------------------
    def handle_query(self, client_domain: str, domain: str,
                     particles: List[PointParticle]) -> Dict:
        # 1. locate client via receiver_module
        client = self.receive_client(particles)
        client_geo = (client["location"]["t_hat"],
                     client["location"]["r_hat"])

        # 2. WiFi sense
        sensing = self.sense_neighbors()

        # 3. resolve + quicksort
        records = self.resolve(domain)

        # 4. entangled transmit
        if records:
            tr = self.transmit_response(
                client_domain, domain, records, client_geo)
            tr_summary = dict(
                sha_in=tr.sha_in,
                sha_out=tr.sha_out,
                round_trip=tr.round_trip,
                frame_S=tr.frame_S,
                frame_m=tr.frame_m,
            )
        else:
            tr_summary = dict(sha_in="", sha_out="",
                              round_trip=False,
                              frame_S=0.0, frame_m=0.0)

        return dict(
            client_geo=client_geo,
            client_n_survivors=client["n_survivors"],
            client_even_even=client["even_even_kept"],
            records=records,
            neighbors=sensing["neighbors"],
            sensing_summary=sensing["summary"],
            transmit_summary=tr_summary,
        )


# ============================================================
#  Session key derivation  (from email_server_mobius pattern)
# ============================================================
def derive_session_key(client_id: bytes,
                       client_nonce: bytes,
                       server_fp: bytes,
                       server_nonce: bytes) -> bytes:
    return hashlib.sha256(
        b"SESSION|"
        + client_id + b"|" + client_nonce + b"|"
        + server_fp + b"|" + server_nonce
    ).digest()


# ============================================================
#  Server-side connection handler
# ============================================================
class DNSConnection:
    def __init__(self, sock: socket.socket,
                 server: "MobiusDNSServer",
                 verbose: bool = False):
        self.sock = sock
        self.server = server
        self.router = server.router
        self.transport = MobiusTransport(K=K_TRANSPORT)
        self.session_key: Optional[bytes] = None
        self.client_id: Optional[str] = None
        self.verbose = verbose
        self._nonces: set = set()

    def _log(self, *args):
        if self.verbose:
            print("  " + " ".join(str(a) for a in args))

    def _xor(self, key: bytes, nonce: bytes, data: bytes) -> bytes:
        stream = hashlib.shake_256(key + nonce).digest(len(data))
        return bytes(a ^ b for a, b in zip(data, stream))

    def _send_msg(self, msg_type: int, payload: bytes,
                  session: bool = True) -> None:
        if session and self.session_key:
            nonce = os.urandom(12)
            ct = self._xor(self.session_key, nonce, payload)
            mac = hmac.new(self.session_key,
                           nonce + ct, hashlib.sha256).digest()
            inner = (bytes([msg_type]) + nonce
                     + struct.pack(">H", len(mac)) + mac + ct)
        else:
            inner = bytes([msg_type]) + payload
        frame = self.transport.frame(inner)
        self.sock.sendall(struct.pack(">Q", len(frame.bytes_))
                          + frame.bytes_)

    def _recv_msg(self) -> Tuple[int, bytes]:
        n = struct.unpack(">Q", self._recv_exact(8))[0]
        raw = self._recv_exact(n)
        inner, _ = self.transport.deframe(raw)
        msg_type = inner[0]

        if self.session_key and msg_type not in (MSG_HELLO, MSG_ACK):
            nonce = inner[1:13]
            mac_len = struct.unpack(">H", inner[13:15])[0]
            mac = inner[15:15 + mac_len]
            ct = inner[15 + mac_len:]
            expected = hmac.new(self.session_key,
                                nonce + ct, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                raise RuntimeError("MAC verification failed")
            nh = nonce.hex()
            if nh in self._nonces:
                raise RuntimeError(f"replay nonce {nh[:8]}")
            self._nonces.add(nh)
            return msg_type, self._xor(self.session_key, nonce, ct)
        return msg_type, inner[1:]

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    def handle_hello(self) -> None:
        msg_type, payload = self._recv_msg()
        if msg_type != MSG_HELLO:
            raise RuntimeError(f"expected HELLO, got {msg_type:#x}")
        id_len = struct.unpack(">H", payload[:2])[0]
        client_id = payload[2:2+id_len].decode("utf-8")
        off = 2 + id_len
        nonce_len = struct.unpack(">H", payload[off:off+2])[0]; off += 2
        client_nonce = payload[off:off+nonce_len]
        self.client_id = client_id

        server_nonce = os.urandom(16)
        server_fp = hashlib.sha256(
            self.server.server_id.encode() + server_nonce).digest()
        self.session_key = derive_session_key(
            client_id.encode("utf-8"), client_nonce,
            server_fp, server_nonce)

        check = hashlib.sha256(self.session_key).digest()
        self._send_msg(MSG_ACK, server_fp + server_nonce + check,
                       session=False)
        self._log(f"HELLO  client={client_id}  "
                  f"key={self.session_key.hex()[:16]}…")

    def handle_query(self, payload: bytes) -> bytes:
        domain, particles = _unpack_query(payload)
        r = self.router.handle_query(
            self.client_id, domain, particles)
        recs = r["records"]
        self._log(f"QUERY  {domain}  → {len(recs)} rec  "
                  f"client_geo=({r['client_geo'][0]:+.3f}, "
                  f"{r['client_geo'][1]:+.3f})  "
                  f"n_surv={r['client_n_survivors']}  "
                  f"tr={r['transmit_summary']['round_trip']}")
        return _pack_qresp(recs, r["client_geo"],
                           r["transmit_summary"])

    def run(self) -> None:
        try:
            self.handle_hello()
            while True:
                try:
                    msg_type, payload = self._recv_msg()
                except ConnectionError:
                    break
                if msg_type == MSG_QUERY:
                    self._send_msg(MSG_QRESP, self.handle_query(payload))
                elif msg_type == MSG_CLOSE:
                    break
                else:
                    err = f"unknown message {msg_type:#x}"
                    self._send_msg(
                        MSG_ERROR,
                        struct.pack(">H", len(err)) + err.encode())
        except Exception as e:
            self._log(f"error: {e}")
        finally:
            try:
                self.sock.close()
            except Exception:
                pass


# ============================================================
#  Server
# ============================================================
class MobiusDNSServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 server_id: str = "mobius-dns-01",
                 router_domain: str = "router.mobius.local",
                 router_geo: Tuple[float, float] = (1.2, 0.9),
                 K: int = DEFAULT_K,
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.server_id = server_id
        self.verbose = verbose
        self.router = DNSRouter(
            router_domain=router_domain,
            router_geo=router_geo,
            K=K, verbose=verbose)
        self._sock: Optional[socket.socket] = None
        self._running = False

    def start(self) -> int:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

        if self.verbose:
            print(f"--- DNS TCP server on {self.host}:{self.port} ---")
            s = self.router.sense_neighbors()["summary"]
            print(f"    wifi sense:   "
                  f"peaks={s['n_peaks']}  μ={s['n_muon']}  "
                  f"τ={s['n_tau']}  electrons={s['n_electrons']}")

        return self.port

    def _accept_loop(self):
        while self._running:
            try:
                sock, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle,
                             args=(sock,), daemon=True).start()

    def _handle(self, sock):
        DNSConnection(sock, self, verbose=self.verbose).run()

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass


# ============================================================
#  Client  ·  sends particles + query, receives sorted records
# ============================================================
class MobiusDNSClient:
    def __init__(self, host: str, port: int,
                 client_id: str,
                 client_geo: Tuple[float, float] = (0.4, 0.6),
                 n_particles: int = 96,
                 K_vec: int = K_VEC_DEFAULT,
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.client_geo = client_geo
        self.n_particles = n_particles
        self.K_vec = K_vec
        self.verbose = verbose

        self.sock: Optional[socket.socket] = None
        self.transport = MobiusTransport(K=K_TRANSPORT)
        self.session_key: Optional[bytes] = None
        self._nonces: set = set()

        # tables for particle generation
        self._F, self._mu, self._H = mu_convolution_H(K_MU_TABLE)

    def _log(self, *args):
        if self.verbose:
            print("  " + " ".join(str(a) for a in args))

    def _xor(self, key: bytes, nonce: bytes, data: bytes) -> bytes:
        stream = hashlib.shake_256(key + nonce).digest(len(data))
        return bytes(a ^ b for a, b in zip(data, stream))

    def _send_msg(self, msg_type: int, payload: bytes,
                  session: bool = True) -> None:
        if session and self.session_key:
            nonce = os.urandom(12)
            ct = self._xor(self.session_key, nonce, payload)
            mac = hmac.new(self.session_key,
                           nonce + ct, hashlib.sha256).digest()
            inner = (bytes([msg_type]) + nonce
                     + struct.pack(">H", len(mac)) + mac + ct)
        else:
            inner = bytes([msg_type]) + payload
        frame = self.transport.frame(inner)
        self.sock.sendall(struct.pack(">Q", len(frame.bytes_))
                          + frame.bytes_)

    def _recv_msg(self) -> Tuple[int, bytes]:
        n = struct.unpack(">Q", self._recv_exact(8))[0]
        raw = self._recv_exact(n)
        inner, _ = self.transport.deframe(raw)
        msg_type = inner[0]
        if self.session_key and msg_type not in (MSG_HELLO, MSG_ACK):
            nonce = inner[1:13]
            mac_len = struct.unpack(">H", inner[13:15])[0]
            mac = inner[15:15 + mac_len]
            ct = inner[15 + mac_len:]
            expected = hmac.new(self.session_key,
                                nonce + ct, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                raise RuntimeError("MAC verification failed")
            return msg_type, self._xor(self.session_key, nonce, ct)
        return msg_type, inner[1:]

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        cid = self.client_id.encode("utf-8")
        nonce = os.urandom(16)
        payload = (struct.pack(">H", len(cid)) + cid
                   + struct.pack(">H", len(nonce)) + nonce)
        self._send_msg(MSG_HELLO, payload, session=False)

        msg_type, response = self._recv_msg()
        if msg_type != MSG_ACK:
            raise RuntimeError(f"expected ACK, got {msg_type:#x}")
        server_fp    = response[:32]
        server_nonce = response[32:48]
        check        = response[48:80]

        self.session_key = derive_session_key(
            cid, nonce, server_fp, server_nonce)
        expected = hashlib.sha256(self.session_key).digest()
        if not hmac.compare_digest(check, expected):
            raise RuntimeError("session key check mismatch")
        self._log(f"connected  key={self.session_key.hex()[:16]}…")

    def query(self, domain: str) -> Dict:
        # 1. build particles from client's own location
        particles = build_particles(
            self.client_geo[0], self.client_geo[1],
            self.n_particles,
            self._mu, self._H, self._F,
            noise=0.02, seed=hash(self.client_id) % (2**31))

        # 2. pack + send
        self._send_msg(MSG_QUERY, _pack_query(domain, particles))

        # 3. receive QRESP
        msg_type, payload = self._recv_msg()
        if msg_type != MSG_QRESP:
            raise RuntimeError(f"expected QRESP, got {msg_type:#x}")
        records, client_geo_est, tr = _unpack_qresp(
            payload, self.K_vec // 4)
        for r in records:
            r.domain = domain

        self._log(f"QUERY  {domain}  → {len(records)} rec  "
                  f"client_geo=({client_geo_est[0]:+.3f}, "
                  f"{client_geo_est[1]:+.3f})  "
                  f"tr={tr['round_trip']}")
        return dict(
            records=records,
            client_geo=client_geo_est,
            transmit_summary=tr,
        )

    def close(self) -> None:
        try:
            self._send_msg(MSG_CLOSE, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Möbius DNS router v2  ·  quicksort + WiFi sensing + "
          "receiver + entangled TX")
    print("=" * 80)
    print(f"  α              = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  6/π²                          = {DENSITY:.6f}")
    print(f"  quicksort                     O(K log K)")
    print()

    server = MobiusDNSServer(
        port=0,
        server_id="mobius-dns-01",
        router_domain="router.mobius.local",
        router_geo=(1.2, 0.9),
        K=DEFAULT_K,
        verbose=False)
    port = server.start()
    print()

    # ---- register records ----
    print("--- registering records ---")
    regs = [
        ("mail.example.com",   (1.2, 0.9), 4, IPMode.STATIC,  300),
        ("dns.example.com",    (1.3, 0.8), 2, IPMode.STATIC,  300),
        ("relay.mobius.org",   (2.7, 1.5), 3, IPMode.STATIC,  600),
        ("cdn.mobius.org",     (0.5, 0.5), 3, IPMode.ANYCAST, 60),
    ]
    for domain, geo, n, mode, ttl in regs:
        rs = server.router.store.register(
            domain, geo, n_ips=n, mode=mode, ttl=ttl)
        print(f"  {domain:<22s}  {n} IP  mode={mode.name:<8s}")
    print()

    # ---- client ----
    print("--- client: alice at (0.4, 0.6) ---")
    client = MobiusDNSClient(
        "127.0.0.1", port,
        "alice@mobius.local",
        client_geo=(0.4, 0.6),
        n_particles=96,
        verbose=True)
    client.connect()
    print()

    # ---- query ----
    print("=" * 80)
    print("1  ·  query  mail.example.com  (4 IPs)")
    print("=" * 80)
    r = client.query("mail.example.com")
    print(f"  client geo estimated: "
          f"({r['client_geo'][0]:+.4f}, {r['client_geo'][1]:+.4f})")
    print(f"  records ({len(r['records'])}):")
    for rec in r["records"]:
        print(f"    {rec.ip.to_dotted()}  "
              f"scalar={rec.scalar:<20d}  ttl={rec.ttl}")
        
    qs_scalars  = [x.scalar for x in r["records"]]
    is_sorted   = qs_scalars == sorted(qs_scalars)
    print(f"  quicksort sorted?    {is_sorted}")
    for k, v in r["transmit_summary"].items():
        if isinstance(v, float):
            print(f"    {k:<12s}: {v:.4e}")
        else:
            print(f"    {k:<12s}: {v}")
    print()

    # ---- query across multiple domains ----
    print("=" * 80)
    print("2  ·  queries across all domains")
    print("=" * 80)
    for domain in ("dns.example.com", "relay.mobius.org",
                   "cdn.mobius.org", "unknown.example"):
        r = client.query(domain)
        n = len(r["records"])
        scalars = [x.scalar for x in r["records"]]
        print(f"  {domain:<22s}  n={n}  "
              f"tr={r['transmit_summary']['round_trip']}  "
              f"scalars={scalars[:4]}"
              f"{'…' if len(scalars) > 4 else ''}")
    print()

    # ---- quicksort vs sorted sanity ----
    print("=" * 80)
    print("3  ·  quicksort sanity (vs built-in sorted)")
    print("=" * 80)
    all_recs = server.router.store.all_records()
    qs = quicksort_records(all_recs, key=lambda r: r.scalar)
    py = sorted(all_recs, key=lambda r: r.scalar)
    match = all(a.scalar == b.scalar for a, b in zip(qs, py))
    print(f"  records:          {len(all_recs)}")
    print(f"  quicksort == sorted? {match}")                    # ✓ balanced
    print(f"  first 5 scalars:  {[r.scalar for r in qs[:5]]}")  # ✓ balanced
    print()

    # ---- WiFi geo-sensing ----
    print("=" * 80)
    print("4  ·  WiFi geo-sensing (SpectralDensityMobius)")
    print("=" * 80)
    sensing = server.router.sense_neighbors(force=True)
    summ = sensing["summary"]
    print(f"  peaks              = {summ['n_peaks']}")
    print(f"  μ channel (cosine) = {summ['n_muon']}")
    print(f"  τ channel (sine)   = {summ['n_tau']}")
    print(f"  electrons (mid)    = {summ['n_electrons']}")
    print(f"  mean power         = {summ['mean_power']:.4e}")
    print(f"  E_max              = {summ['E_max']:.4e}")
    print()
    print(f"  neighbor routers detected (first 5):")
    for nbr in sensing["neighbors"][:5]:
        print(f"    t={nbr['t']:.4f}  r={nbr['r']:.4f}  "
              f"f={nbr['f']:.4f}  ch={nbr['channel']:<4s}  "
              f"P={nbr['strength']:.4e}")
    print()

    # ---- complexity ----
    print("=" * 80)
    print("5  ·  complexity")
    print("=" * 80)
    print("  quicksort records     O(m log m)       replaces TSP 1-D")
    print("  WiFi geo-sensing      O(N·K² + K log K)  once per refresh")
    print("  client reception      O(n_particles · K_vec)")
    print("  entangled transmit    O(K_sat + K_pack)")
    print("  MOSB frame + MAC      O(payload bytes)")
    print()

    client.close()
    time.sleep(0.1)
    server.stop()
    print("Done.")


if __name__ == "__main__":
    demo()