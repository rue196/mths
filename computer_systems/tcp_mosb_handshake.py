#!/usr/bin/env python3
"""
mobius_tcp_coupling.py
======================

TCP extension integrating:

    · geo-location   ← coupling_Q_entanglement.py
                       (standard g_ij = w_ij·κ_ij, even-even subgraph,
                        6×6 spinor reconstruction Π(Z_n))

    · clock signalling ← Muon_tau_convolution_tick.py
                       (AtomicClockMuonTau, muon/tau convolution zeros,
                        parity ratio ‖G_ee‖/‖G_eo‖)

    · payload envelope ← UTF_8_dzeta_dt.py
                       (particle cloud, UGC survivors, ALU, fingerprint)

    · receive check   → cross coupling matrix G_eo = G ⊙ [parity(i)≠parity(j)]
                        verified against norms & Π signature on the server

Wire protocol (length-prefixed MOSB frames):

    HELLO (0x01) : [uint8 0x01][uint16 id_len][client_id]
                   [uint16 sig_len][CouplingSignal][clock_snap]
    ACK   (0x02) : [uint8 0x02][double SVP][double parity_ratio][clock_snap]
    DATA  (0x03) : [nonce 12 B][uint16 mac_len][mac][ciphertext]
    CLOSE (0x04) : [uint8 0x04]
"""

from __future__ import annotations
import hashlib, hmac, math, os, socket, struct, threading, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- geo-location (coupling_Q_entanglement.py) ----
from coupling_Q_entanglement import (
    coupling_matrix,
    entangled_coupling,
    even_even_mask,
    even_indices,
    spinor_reconstruct,
    icosahedron_vertices,
    lift_to_six_dim,
    levi_civita,
    event_spinor,
    make_geo_events,
)

# ---- clock (Muon_tau_convolution_tick.py) ----
from Muon_tau_convolution_tick import AtomicClockMuonTau

# ---- payload envelope (UTF_8_dzeta_dt.py) ----
from UTF_8_dzeta_dt import (
    UTF8Table, ReadOnlyBuffer, ReadOnlyALU,
    build_particle_cloud, ugc_survivors,
    K_DEFAULT as K_UTF8_DEF,
)

# ---- MOSB transport (fallback if mobius_internet missing) ----
try:
    from mobius_internet import MobiusTransport
except ImportError:
    @dataclass
    class _MOSBFrame:
        bytes_: bytes
        size:   int
        S:      float
        H:      float
        m:      float

    class MobiusTransport:
        def __init__(self, K: int = 512):
            self.K = K

        def frame(self, data: bytes) -> _MOSBFrame:
            S = 0.0
            for t, b in enumerate(data):
                S += (b - 128) / 128.0 * (1 if t % 2 == 0 else -1)
            H = 0.0
            if S != 0 and len(data):
                p = min(abs(S) / len(data), 0.9999)
                H = -2.362 * p * math.log(p) if p > 0 else 0.0
            m = abs(S) * math.exp(-H) if H < 700 else 0.0
            hdr = struct.pack(">4sIQddd", b"MOSB", self.K, len(data),
                              S, H, m)
            blob = hdr + data
            return _MOSBFrame(bytes_=blob, size=len(blob),
                              S=S, H=H, m=m)

        def deframe(self, blob: bytes):
            hdr_sz = struct.calcsize(">4sIQddd")
            _, _, n, S, H, m = struct.unpack(">4sIQddd", blob[:hdr_sz])
            data = blob[hdr_sz:hdr_sz + n]
            return data, _MOSBFrame(bytes_=blob, size=len(blob),
                                    S=S, H=H, m=m)


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)
A_STEP   = ALPHA / 0.3628
DENSITY  = 6.0 / (PI * PI)

K_HANDSHAKE   = 64
K_UTF8        = 60
K_CLOCK       = 20
N_CLOCK_EVT   = 17
N_CLOCK_SAMP  = 128
N_EVENTS_GEO  = 24

TICK_TOL      = 0.5
SVP_TOL       = 1e-6
MAX_STALE     = 5
TICK_SNAP_FMT = ">dIddIdd"
TICK_SNAP_LEN = struct.calcsize(TICK_SNAP_FMT)

MAGIC_HELLO = 0x01
MAGIC_ACK   = 0x02
MAGIC_DATA  = 0x03
MAGIC_CLOSE = 0x04


# ============================================================
#  Helpers  ·  cross-coupling mask + geo-anchored events
# ============================================================
def _even_odd_mask(N: int) -> np.ndarray:
    """Cross matrix mask: True iff parity(i) ≠ parity(j)."""
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2) != (j % 2)


def _events_from_geo(geo: Tuple[float, float], n_events: int):
    """
    Deterministic event grid anchored at a specific geo-location.

    The seed AND the scale are both derived from geo, so different
    locations give distinct coupling matrices.
    """
    seed = int(hashlib.sha256(
        f"{geo[0]:.6f}|{geo[1]:.6f}|{n_events}".encode()
    ).hexdigest()[:8], 16)
    t, r = make_geo_events(n_events=n_events, seed=seed % (2**32))
    t = t * (1.0 + 0.1 * geo[0])
    r = r * (1.0 + 0.1 * geo[1])
    return t, r


# ============================================================
#  Coupling signal  ·  geo-location through coupling_Q_entanglement
# ============================================================
@dataclass
class CouplingSignal:
    payload:      bytes
    src_geo:      Tuple[float, float]
    dst_geo:      Tuple[float, float]
    K:            int
    n_events:     int
    norm_ee:      float                # ‖G_ee‖
    norm_eo:      float                # ‖G_eo‖  (cross matrix)
    parity_ratio: float                # ‖G_ee‖ / ‖G_eo‖
    pi_signature: np.ndarray           # |Π(Z_n)| per even event
    e_signal:     np.ndarray           # E-parity fold of payload
    b_signal:     np.ndarray           # B-parity fold of payload


def encode_coupling_signal(payload: bytes,
                           src_geo: Tuple[float, float],
                           dst_geo: Tuple[float, float],
                           K: int = K_HANDSHAKE,
                           n_events: int = N_EVENTS_GEO
                           ) -> CouplingSignal:
    """Encode payload + geo via the even-even coupling structure."""

    # 1. events anchored to src_geo
    t, r = _events_from_geo(src_geo, n_events)

    # 2. standard coupling + even-even split (coupling_Q_entanglement)
    G = coupling_matrix(t, r)
    G_ee, _ = entangled_coupling(G)                   # even-even
    G_eo = G * _even_odd_mask(len(t))                 # cross

    norm_ee = float(np.linalg.norm(G_ee))
    norm_eo = float(np.linalg.norm(G_eo))
    parity_ratio = norm_ee / max(norm_eo, 1e-12)

    # 3. 6×6 spinor reconstruction per even event
    even_idx = even_indices(len(t))
    Pis, mags = spinor_reconstruct(t[even_idx], r[even_idx])

    # 4. weight basis from Π (magnitude · cos(phase))
    n   = np.arange(K)
    ang = np.angle(Pis)
    mag = np.abs(Pis)
    basis = np.zeros(K)
    for i in range(min(K, len(mag))):
        basis[i] = mag[i] * math.cos(ang[i])

    # 5. fold payload bytes through the coupling basis
    N = len(payload)
    V = np.zeros((N, K))
    for i, b in enumerate(payload):
        sig = hashlib.shake_256(
            bytes([b])
            + struct.pack(">I", i)
            + struct.pack(">dd", *src_geo)
            + struct.pack(">dd", *dst_geo)
            + struct.pack(">dd", norm_ee, norm_eo)
        ).digest(32)
        s = np.array([bb / 255.0 for bb in sig])
        V[i] = basis * s[n % len(s)]

    # 6. parity split  ·  E = odd-index, B = even-index
    parity = np.arange(N) % 2
    E_signal = np.zeros(K)
    B_signal = np.zeros(K)
    for i in range(N):
        if parity[i] == 1: E_signal += V[i]
        else:              B_signal += V[i]
    E_signal /= max(1, int((parity == 1).sum()))
    B_signal /= max(1, int((parity == 0).sum()))

    return CouplingSignal(
        payload=payload, src_geo=src_geo, dst_geo=dst_geo,
        K=K, n_events=n_events,
        norm_ee=norm_ee, norm_eo=norm_eo,
        parity_ratio=parity_ratio,
        pi_signature=mags,
        e_signal=E_signal, b_signal=B_signal,
    )


SIG_FMT = ">II dddd II ddd I"
SIG_HDR = struct.calcsize(SIG_FMT)


def serialize_coupling_signal(sig: CouplingSignal) -> bytes:
    K = sig.K
    n_pi = len(sig.pi_signature)
    hdr = struct.pack(
        SIG_FMT,
        K, sig.n_events,
        sig.src_geo[0], sig.src_geo[1],
        sig.dst_geo[0], sig.dst_geo[1],
        n_pi, K,
        sig.norm_ee, sig.norm_eo, sig.parity_ratio,
        len(sig.payload),
    )
    return (hdr
            + struct.pack(f">{n_pi}d", *sig.pi_signature.tolist())
            + struct.pack(f">{K}d",    *sig.e_signal.tolist())
            + struct.pack(f">{K}d",    *sig.b_signal.tolist())
            + sig.payload)


def deserialize_coupling_signal(blob: bytes) -> CouplingSignal:
    (K, n_events,
     sx, sy, dx, dy,
     n_pi, n_K,
     nee, neo, ratio,
     plen) = struct.unpack(SIG_FMT, blob[:SIG_HDR])
    off = SIG_HDR
    pi_sig = np.array(struct.unpack(f">{n_pi}d", blob[off:off + 8*n_pi]))
    off += 8 * n_pi
    e_sig  = np.array(struct.unpack(f">{n_K}d",  blob[off:off + 8*n_K]))
    off += 8 * n_K
    b_sig  = np.array(struct.unpack(f">{n_K}d",  blob[off:off + 8*n_K]))
    off += 8 * n_K
    payload = blob[off:off + plen]
    return CouplingSignal(
        payload=payload, src_geo=(sx, sy), dst_geo=(dx, dy),
        K=K, n_events=n_events,
        norm_ee=nee, norm_eo=neo, parity_ratio=ratio,
        pi_signature=pi_sig, e_signal=e_sig, b_signal=b_sig,
    )


# ============================================================
#  Server-side receive check  ·  against cross coupling matrix
# ============================================================
def verify_coupling_signal(sig: CouplingSignal) -> Tuple[bool, Dict]:
    """
    Rebuild G, G_ee, G_eo from the received geo, then check:
        · ‖G_ee‖ matches          (even-even)
        · ‖G_eo‖ matches          (cross matrix)
        · parity protected        (‖G_ee‖ > ‖G_eo‖)
        · Π signature matches
        · cross matrix non-empty
    """
    t, r = _events_from_geo(sig.src_geo, sig.n_events)
    G = coupling_matrix(t, r)
    G_ee, _ = entangled_coupling(G)
    G_eo = G * _even_odd_mask(len(t))

    computed_ee = float(np.linalg.norm(G_ee))
    computed_eo = float(np.linalg.norm(G_eo))

    even_idx = even_indices(len(t))
    _, mags = spinor_reconstruct(t[even_idx], r[even_idx])

    checks = {
        'norm_ee_match':    abs(computed_ee - sig.norm_ee) < 1e-4,
        'norm_eo_match':    abs(computed_eo - sig.norm_eo) < 1e-4,
        'parity_protected': computed_ee > computed_eo,
        'signature_match':  (len(mags) == len(sig.pi_signature)
                             and np.allclose(mags, sig.pi_signature, atol=1e-4)),
        'ratio_match':      abs(computed_ee / max(computed_eo, 1e-12)
                                - sig.parity_ratio) < 1e-4,
        'cross_nonzero':    computed_eo > 1e-9,
    }
    return all(checks.values()), checks


def recover_location(sig: CouplingSignal) -> dict:
    ok, checks = verify_coupling_signal(sig)
    return {
        't_hat':       sig.src_geo[0],
        'r_hat':       sig.src_geo[1],
        'verified':    ok,
        'checks':      checks,
        'norm_ee':     sig.norm_ee,
        'norm_eo':     sig.norm_eo,
        'parity_ratio': sig.parity_ratio,
    }


# ============================================================
#  SVP derivation
# ============================================================
def svp_session_key(t_hat: float, r_hat: float, peer_id: str) -> dict:
    seed = hashlib.sha256(
        f"{peer_id}|{t_hat:.6f}|{r_hat:.6f}".encode()).digest()
    svp = int.from_bytes(seed[:8], "big") / (2 ** 64)
    fp  = seed.hex()[:32]
    key = hashlib.sha256(
        f"{peer_id}|{svp:.12f}|{fp}".encode()).hexdigest()
    return dict(t_hat=t_hat, r_hat=r_hat, SVP=svp,
                fingerprint=fp, session_key=key)


# ============================================================
#  Clock helpers  ·  AtomicClockMuonTau
# ============================================================
def make_clock() -> AtomicClockMuonTau:
    return AtomicClockMuonTau(
        K=K_CLOCK, J_C=1.0,
        N_events=N_CLOCK_EVT, N_samples=N_CLOCK_SAMP)


def clock_snapshot(clock: AtomicClockMuonTau,
                   wall_time: Optional[float] = None) -> dict:
    if wall_time is None:
        wall_time = time.time()
    t_mod = wall_time % clock.T
    idx   = int(np.argmin(np.abs(clock.t - t_mod)))
    return dict(
        wall_time = float(wall_time),
        tick_idx  = idx,
        tick_t    = float(clock.t[idx]),
        tick_rate = float(clock.tick_rate()),
        n_elect   = int(clock.n_electrons),
        parity    = float(clock.parity_ratio),
        T         = float(clock.T),
    )


def pack_clock(s: dict) -> bytes:
    return struct.pack(TICK_SNAP_FMT,
        s["wall_time"], s["tick_idx"], s["tick_t"],
        s["tick_rate"], s["n_elect"], s["parity"], s["T"])


def unpack_clock(blob: bytes) -> dict:
    wall_time, tick_idx, tick_t, rate, n_e, par, T = struct.unpack(
        TICK_SNAP_FMT, blob[:TICK_SNAP_LEN])
    return dict(wall_time=wall_time, tick_idx=tick_idx,
                tick_t=tick_t, tick_rate=rate,
                n_elect=n_e, parity=par, T=T)


def verify_clock_sync(local_clock: AtomicClockMuonTau,
                      remote: dict) -> dict:
    local_t = remote["wall_time"] % local_clock.T
    idx = int(np.argmin(np.abs(local_clock.t - local_t)))
    drift     = abs(float(local_clock.t[idx]) - remote["tick_t"])
    parity_ok = abs(remote["parity"] - local_clock.parity_ratio) < 1e-3
    return dict(
        drift=drift,
        within_tolerance=drift < TICK_TOL,
        parity_agrees=parity_ok,
        local_t=float(local_clock.t[idx]),
        remote_t=remote["tick_t"],
    )


# ============================================================
#  Payload envelope  ·  UTF_8_dzeta_dt
# ============================================================
@dataclass
class UTF8Envelope:
    raw: bytes
    K: int
    n_particles: int
    n_survivors: int
    survival_rate: float
    supertrace: float
    mass: float
    alu_scalar: float
    fingerprint: str


def encode_utf8_pipeline(data: bytes, K: int = K_UTF8) -> UTF8Envelope:
    table = UTF8Table.build(K=K)
    particles = build_particle_cloud(table, data)
    surv, non = ugc_survivors(particles, K, table.mu, 4, threshold=0.3)
    n_p = len(particles)
    rate = len(surv) / max(1, len(surv) + len(non))

    S = 0.0
    for i, p in enumerate(particles):
        S += p.x if (i % 2 == 0) else -p.x

    if abs(S) < 1e-12 or n_p == 0:
        H = 0.0; m = 0.0
    else:
        p = min(abs(S) / n_p, 0.9999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0

    alu = ReadOnlyALU(K=64)
    buf = ReadOnlyBuffer(K=64)
    for p in particles:
        buf.append(struct.pack(">dd", p.x, p.y))
    alu_scalar = 0.0
    if len(buf.chunks) >= 2:
        alu_scalar = alu.dispatch(buf, 0, 1, "ADD")["z"]

    fp = hashlib.sha256(data).hexdigest()
    return UTF8Envelope(raw=data, K=K,
                        n_particles=n_p, n_survivors=len(surv),
                        survival_rate=rate, supertrace=S,
                        mass=m, alu_scalar=alu_scalar, fingerprint=fp)


def verify_utf8_envelope(env: UTF8Envelope) -> Tuple[bool, Dict]:
    rec = encode_utf8_pipeline(env.raw, env.K)
    checks = dict(
        n_particles_ok = rec.n_particles == env.n_particles,
        n_survivors_ok = rec.n_survivors == env.n_survivors,
        rate_ok        = abs(rec.survival_rate - env.survival_rate) < 1e-6,
        supertrace_ok  = abs(rec.supertrace - env.supertrace) < 1e-6,
        alu_ok         = abs(rec.alu_scalar - env.alu_scalar) < 1e-6,
        fingerprint_ok = rec.fingerprint == env.fingerprint,
    )
    return all(checks.values()), checks


ENV_FMT = ">BHHddd32sI"
ENV_HDR = struct.calcsize(ENV_FMT)


def pack_env(env: UTF8Envelope) -> bytes:
    return struct.pack(ENV_FMT, env.K, env.n_particles, env.n_survivors,
                       env.survival_rate, env.supertrace, env.mass,
                       bytes.fromhex(env.fingerprint), len(env.raw)) + env.raw


def unpack_env(blob: bytes) -> Tuple[UTF8Envelope, int]:
    K, n_p, n_s, rate, S, m, fp, rl = struct.unpack(ENV_FMT, blob[:ENV_HDR])
    raw = blob[ENV_HDR:ENV_HDR + rl]
    env = UTF8Envelope(raw=raw, K=K, n_particles=n_p, n_survivors=n_s,
                       survival_rate=rate, supertrace=S, mass=m,
                       alu_scalar=0.0, fingerprint=fp.hex())
    return env, ENV_HDR + rl


# ============================================================
#  Session + crypto
# ============================================================
@dataclass
class TCPSession:
    peer_id: str
    src_geo: Tuple[float, float]
    dst_geo: Tuple[float, float]
    recovered_src: Tuple[float, float]
    SVP: float
    fingerprint: str
    session_key: str
    local_clock:  dict
    remote_clock: dict
    clock_drift: float
    geo_verified: bool
    server_parity: float
    established_at: float = field(default_factory=time.time)
    packets_sent: int = 0
    packets_received: int = 0


def _xor(key: bytes, nonce: bytes, data: bytes) -> bytes:
    stream = hashlib.shake_256(key + nonce).digest(len(data))
    return bytes(a ^ b for a, b in zip(data, stream))


def encrypt(key_hex: str, nonce: bytes, pt: bytes) -> bytes:
    return _xor(bytes.fromhex(key_hex), nonce, pt)


def decrypt(key_hex: str, nonce: bytes, ct: bytes) -> bytes:
    return _xor(bytes.fromhex(key_hex), nonce, ct)


def compute_mac(key_hex: str, nonce: bytes, ct: bytes) -> bytes:
    return hmac.new(bytes.fromhex(key_hex), nonce + ct,
                    hashlib.sha256).digest()


def verify_mac(key_hex: str, nonce: bytes, ct: bytes, mac: bytes) -> bool:
    return hmac.compare_digest(compute_mac(key_hex, nonce, ct), mac)


# ============================================================
#  Framing
# ============================================================
def send_frame(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">Q", len(data)) + data)


def recv_frame(sock: socket.socket) -> bytes:
    n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
    return _recv_exact(sock, n)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


# ============================================================
#  MobiusTCPConnection
# ============================================================
class MobiusTCPConnection:
    def __init__(self, sock: socket.socket,
                 is_server: bool,
                 local_geo: Tuple[float, float] = (0.5, 0.5),
                 remote_geo: Tuple[float, float] = (0.5, 0.5),
                 clock: Optional[AtomicClockMuonTau] = None,
                 verbose: bool = False):
        self.sock = sock
        self.is_server = is_server
        self.local_geo = local_geo
        self.remote_geo = remote_geo
        self.clock = clock or make_clock()
        self.trans = MobiusTransport(K=512)
        self.session: Optional[TCPSession] = None
        self.verbose = verbose
        self.last_tick = -1
        self._nonces: set = set()

    def _log(self, *a):
        if self.verbose:
            print("  " + " ".join(str(x) for x in a))

    # --------------------------------------------------------
    def handshake_client(self, client_id: str) -> TCPSession:
        self._log(f"--- client handshake  id={client_id} ---")

        # 1. local clock snapshot
        cs = clock_snapshot(self.clock)
        self._log(f"clock   idx={cs['tick_idx']}  t={cs['tick_t']:.4f}  "
                  f"rate={cs['tick_rate']:.3f}  "
                  f"parity={cs['parity']:.3f}")

        # 2. encode coupling signal  (geo through coupling_Q_entanglement)
        sig = encode_coupling_signal(
            client_id.encode("utf-8"),
            src_geo=self.local_geo,
            dst_geo=self.remote_geo,
            K=K_HANDSHAKE, n_events=N_EVENTS_GEO)
        sig_bytes = serialize_coupling_signal(sig)
        id_bytes  = client_id.encode("utf-8")

        self._log(f"coupling ‖G_ee‖={sig.norm_ee:.4f}  "
                  f"‖G_eo‖={sig.norm_eo:.4f}  "
                  f"ratio={sig.parity_ratio:.3f}")

        # 3. HELLO
        inner = (struct.pack(">BH", MAGIC_HELLO, len(id_bytes)) + id_bytes
                 + struct.pack(">H", len(sig_bytes)) + sig_bytes
                 + pack_clock(cs))

        frame = self.trans.frame(inner)
        send_frame(self.sock, frame.bytes_)
        self._log(f"HELLO   {frame.size}B  S={frame.S:+.4f}")

        # 4. ACK
        ack_bytes = recv_frame(self.sock)
        ack_inner, _ = self.trans.deframe(ack_bytes)
        if ack_inner[0] != MAGIC_ACK:
            raise RuntimeError(f"expected ACK, got {ack_inner[0]:#x}")

        remote_svp    = struct.unpack(">d", ack_inner[1:9])[0]
        server_parity = struct.unpack(">d", ack_inner[9:17])[0]
        remote_clk    = unpack_clock(ack_inner[17:17 + TICK_SNAP_LEN])

        self._log(f"ACK     SVP={remote_svp:.8f}  "
                  f"parity={server_parity:.3f}  "
                  f"idx={remote_clk['tick_idx']}")

        # 5. SVP match
        my_svp = svp_session_key(self.local_geo[0], self.local_geo[1],
                                 client_id)
        if abs(my_svp["SVP"] - remote_svp) > SVP_TOL:
            raise RuntimeError(
                f"svp mismatch  my={my_svp['SVP']:.8f}  "
                f"remote={remote_svp:.8f}")

        # 6. server parity protection
        if server_parity <= 1.0:
            raise RuntimeError(
                f"server parity unprotected: {server_parity:.4f}")

        # 7. clock
        cc = verify_clock_sync(self.clock, remote_clk)
        if not cc["within_tolerance"]:
            raise RuntimeError(f"clock drift {cc['drift']:.4f} > {TICK_TOL}")
        self._log(f"clock   drift={cc['drift']:.6f}s  "
                  f"parity={'OK' if cc['parity_agrees'] else 'MISMATCH'}")

        # 8. session
        self.session = TCPSession(
            peer_id=client_id,
            src_geo=self.local_geo,
            dst_geo=self.remote_geo,
            recovered_src=(my_svp["t_hat"], my_svp["r_hat"]),
            SVP=my_svp["SVP"],
            fingerprint=my_svp["fingerprint"],
            session_key=my_svp["session_key"],
            local_clock=cs, remote_clock=remote_clk,
            clock_drift=cc["drift"],
            geo_verified=True,
            server_parity=server_parity,
        )
        self.last_tick = cs["tick_idx"]
        self._log(f"session OK  key={self.session.session_key[:16]}…")
        return self.session

    # --------------------------------------------------------
    def handshake_server(self) -> TCPSession:
        self._log("--- server handshake ---")

        # 1. HELLO
        hello_bytes = recv_frame(self.sock)
        inner, hello_frame = self.trans.deframe(hello_bytes)
        if inner[0] != MAGIC_HELLO:
            raise RuntimeError(f"expected HELLO, got {inner[0]:#x}")

        off = 1
        id_len = struct.unpack(">H", inner[off:off + 2])[0]; off += 2
        client_id = inner[off:off + id_len].decode("utf-8"); off += id_len
        sig_len = struct.unpack(">H", inner[off:off + 2])[0]; off += 2
        sig_bytes = inner[off:off + sig_len]; off += sig_len
        client_clk = unpack_clock(inner[off:off + TICK_SNAP_LEN])
        self._log(f"HELLO   id={client_id}  S={hello_frame.S:+.4f}")

        # 2. decode coupling signal
        sig = deserialize_coupling_signal(sig_bytes)

        # 3. receive-check against CROSS coupling matrix
        ok_geo, checks = verify_coupling_signal(sig)
        loc = recover_location(sig)
        self._log(f"geo     t̂={loc['t_hat']:+.4f}  r̂={loc['r_hat']:+.4f}")
        self._log(f"        ‖G_ee‖={sig.norm_ee:.4f}  "
                  f"‖G_eo‖={sig.norm_eo:.4f}  "
                  f"ratio={sig.parity_ratio:.3f}")
        self._log(f"        verified={ok_geo}")
        for k, v in checks.items():
            self._log(f"        {k:>18s} = {v}")

        if not ok_geo:
            raise RuntimeError(f"coupling verification failed: {checks}")

        # 4. derive SVP from recovered location
        svp = svp_session_key(loc["t_hat"], loc["r_hat"], client_id)
        self._log(f"svp     {svp['SVP']:.8f}")

        # 5. own clock + parity
        cs = clock_snapshot(self.clock)
        cc = verify_clock_sync(self.clock, client_clk)
        self._log(f"clock   drift={cc['drift']:.6f}s  "
                  f"{'OK' if cc['within_tolerance'] else 'TOO LARGE'}")

        # 6. ACK  [type][svp][parity_ratio][clock]
        ack_inner = (struct.pack(">Bdd", MAGIC_ACK,
                                 svp["SVP"],
                                 self.clock.parity_ratio)
                     + pack_clock(cs))
        frame = self.trans.frame(ack_inner)
        send_frame(self.sock, frame.bytes_)
        self._log(f"ACK     sent  parity={self.clock.parity_ratio:.3f}")

        # 7. session
        self.session = TCPSession(
            peer_id=client_id,
            src_geo=sig.src_geo,
            dst_geo=sig.dst_geo,
            recovered_src=(loc["t_hat"], loc["r_hat"]),
            SVP=svp["SVP"],
            fingerprint=svp["fingerprint"],
            session_key=svp["session_key"],
            local_clock=cs, remote_clock=client_clk,
            clock_drift=cc["drift"],
            geo_verified=ok_geo,
            server_parity=self.clock.parity_ratio,
        )
        self.last_tick = cs["tick_idx"]
        self._log(f"session OK  key={self.session.session_key[:16]}…")
        return self.session

    # --------------------------------------------------------
    def send_application(self, port: int, payload: str) -> dict:
        if self.session is None:
            raise RuntimeError("no session")
        if not payload:
            raise ValueError("empty payload")

        cs = clock_snapshot(self.clock)
        self.last_tick = max(self.last_tick, cs["tick_idx"])

        port_env    = encode_utf8_pipeline(str(port).encode("utf-8"),
                                           K=K_UTF8)
        payload_env = encode_utf8_pipeline(payload.encode("utf-8"),
                                           K=K_UTF8)

        port_bytes    = pack_env(port_env)
        payload_bytes = pack_env(payload_env)

        inner  = struct.pack(">BH", MAGIC_DATA, len(port_bytes)) + port_bytes
        inner += struct.pack(">I", len(payload_bytes)) + payload_bytes
        inner += struct.pack(">Q", cs["tick_idx"])

        nonce = os.urandom(12)
        ct  = encrypt(self.session.session_key, nonce, inner)
        mac = compute_mac(self.session.session_key, nonce, ct)

        wire = nonce + struct.pack(">H", len(mac)) + mac + ct
        frame = self.trans.frame(wire)
        send_frame(self.sock, frame.bytes_)
        self.session.packets_sent += 1
        self._log(f"DATA    port={port}  plen={len(payload)}  "
                  f"tick={cs['tick_idx']}")
        return dict(port=port, payload_len=len(payload),
                    tick_idx=cs["tick_idx"], frame_size=frame.size)

    # --------------------------------------------------------
    def recv_application(self) -> Tuple[int, str]:
        if self.session is None:
            raise RuntimeError("no session")

        wire_bytes = recv_frame(self.sock)
        wire, _ = self.trans.deframe(wire_bytes)

        nonce   = wire[:12]
        mac_len = struct.unpack(">H", wire[12:14])[0]
        mac     = wire[14:14 + mac_len]
        ct      = wire[14 + mac_len:]

        if not verify_mac(self.session.session_key, nonce, ct, mac):
            raise RuntimeError("MAC verification failed")

        nonce_hex = nonce.hex()
        if nonce_hex in self._nonces:
            raise RuntimeError(f"replay: nonce {nonce_hex[:8]} seen")
        self._nonces.add(nonce_hex)

        inner = decrypt(self.session.session_key, nonce, ct)
        if inner[0] != MAGIC_DATA:
            raise RuntimeError(f"expected DATA, got {inner[0]:#x}")

        off = 1
        plen = struct.unpack(">H", inner[off:off + 2])[0]; off += 2
        port_bytes = inner[off:off + plen]; off += plen
        qlen = struct.unpack(">I", inner[off:off + 4])[0]; off += 4
        payload_bytes = inner[off:off + qlen]; off += qlen
        remote_tick = struct.unpack(">Q", inner[off:off + 8])[0]

        port_env, _ = unpack_env(port_bytes)
        ok1, _ = verify_utf8_envelope(port_env)
        if not ok1:
            raise RuntimeError("port envelope verify failed")
        port = int(port_env.raw.decode("utf-8"))

        payload_env, _ = unpack_env(payload_bytes)
        ok2, _ = verify_utf8_envelope(payload_env)
        if not ok2:
            raise RuntimeError("payload envelope verify failed")
        payload = payload_env.raw.decode("utf-8")

        if remote_tick < self.last_tick - MAX_STALE:
            raise RuntimeError(
                f"stale tick {remote_tick} < {self.last_tick - MAX_STALE}")
        self.last_tick = max(self.last_tick, remote_tick)

        self.session.packets_received += 1
        self._log(f"DATA←   port={port}  plen={len(payload)}  "
                  f"tick={remote_tick}")
        return port, payload

    # --------------------------------------------------------
    def close(self):
        try:
            send_frame(self.sock,
                       self.trans.frame(struct.pack(">B", MAGIC_CLOSE)).bytes_)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ============================================================
#  Server / Client wrappers
# ============================================================
class MobiusTCPServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 server_geo: Tuple[float, float] = (PI / 2, E / 2),
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.server_geo = server_geo
        self.verbose = verbose
        self._sock: Optional[socket.socket] = None
        self._running = False
        self.sessions: List[TCPSession] = []
        self._clients: List[MobiusTCPConnection] = []

    def start(self) -> int:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        if self.verbose:
            print(f"--- TCP server on {self.host}:{self.port} ---")
        return self.port

    def _accept_loop(self):
        while self._running:
            try:
                sock, addr = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(sock, addr),
                             daemon=True).start()

    def _handle(self, sock, addr):
        if self.verbose:
            print(f"--- accept {addr} ---")
        try:
            conn = MobiusTCPConnection(
                sock, is_server=True,
                local_geo=self.server_geo,
                verbose=self.verbose)
            sess = conn.handshake_server()
            self.sessions.append(sess)
            self._clients.append(conn)
            while True:
                port, payload = conn.recv_application()
                response = f"[echo] port={port} payload={payload[:60]}"
                conn.send_application(port=port, payload=response)
        except Exception as e:
            if self.verbose:
                print(f"--- close {addr}: {e} ---")
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass


class MobiusTCPClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 client_geo: Tuple[float, float] = (0.4, 0.6),
                 server_geo: Tuple[float, float] = (PI / 2, E / 2),
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.client_geo = client_geo
        self.server_geo = server_geo
        self.verbose = verbose
        self.conn: Optional[MobiusTCPConnection] = None

    def connect(self, client_id: str) -> TCPSession:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        self.conn = MobiusTCPConnection(
            sock, is_server=False,
            local_geo=self.client_geo,
            remote_geo=self.server_geo,
            verbose=self.verbose)
        return self.conn.handshake_client(client_id)

    def send(self, port: int, payload: str) -> dict:
        return self.conn.send_application(port, payload)

    def recv(self) -> Tuple[int, str]:
        return self.conn.recv_application()

    def close(self):
        if self.conn:
            self.conn.close()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Möbius TCP  ·  coupling geo-location + muon/tau clock  ·  UTF-8 envelope")
    print("=" * 80)
    print(f"  α              = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  a_step         = α / 0.3628  = {A_STEP:.6f}")
    print(f"  6/π²                          = {DENSITY:.6f}")
    print(f"  clock K                       = {K_CLOCK}")
    print(f"  clock events / samples        = {N_CLOCK_EVT} / {N_CLOCK_SAMP}")
    print(f"  geo events (per handshake)    = {N_EVENTS_GEO}")
    print(f"  UTF-8 K                       = {K_UTF8}")
    print()

    # ---------- clock object ----------
    clock = make_clock()
    print(f"--- clock  ·  AtomicClockMuonTau ---")
    print(f"  T              = {clock.T:.6f}")
    print(f"  n_electrons    = {clock.n_electrons}")
    print(f"  tick_rate      = {clock.tick_rate():.4f} / unit-t")
    print(f"  parity_ratio   = {clock.parity_ratio:.4f}  "
          f"(protected = {clock.parity_ratio > 1})")
    print(f"  ‖G_ee‖         = {clock.norm_ee:.6f}")
    print(f"  ‖G_eo‖         = {clock.norm_eo:.6f}")
    print()

    # ---------- geo encoding sanity ----------
    print("--- coupling geo encoding  ·  sanity ---")
    for geo in [(0.4, 0.6), (1.2, 0.9), (PI / 2, E / 2)]:
        sig = encode_coupling_signal(
            b"hello", src_geo=geo, dst_geo=(PI / 2, E / 2),
            K=K_HANDSHAKE, n_events=N_EVENTS_GEO)
        ok, _ = verify_coupling_signal(sig)
        print(f"  geo={geo}  ‖G_ee‖={sig.norm_ee:.4f}  "
              f"‖G_eo‖={sig.norm_eo:.4f}  "
              f"ratio={sig.parity_ratio:.3f}  verified={ok}")
    print()

    # ---------- start server + client ----------
    server = MobiusTCPServer(host="127.0.0.1", port=0,
                             server_geo=(PI / 2, E / 2),
                             verbose=True)
    server.start()
    print()

    client = MobiusTCPClient(host="127.0.0.1", port=server.port,
                             client_geo=(0.4, 0.6),
                             server_geo=(PI / 2, E / 2),
                             verbose=True)
    client.connect(client_id="alice@cell-A")
    print()

    # ---------- data exchange ----------
    print("=" * 80)
    print("EXCHANGE")
    print("=" * 80)
    tests = [
        (8080, "GET /index.html HTTP/1.1\r\nHost: example.com\r\n"),
        (8080, "POST /api/upload HTTP/1.1\r\nContent-Length: 42"),
        (443,  "TLS handshake start"),
        (22,   "SSH-2.0-MobiusClient"),
    ]
    for port, payload in tests:
        print()
        print(f"--- transmit port={port} payload={payload[:40]!r} ---")
        t0 = time.perf_counter()
        client.send(port=port, payload=payload)
        resp_port, resp_payload = client.recv()
        elapsed = (time.perf_counter() - t0) * 1e3
        print(f"  echo:  port={resp_port}  payload={resp_payload[:60]!r}")
        print(f"  RTT:   {elapsed:.2f} ms")
    print()

    # ---------- session ----------
    print("=" * 80)
    print("SESSION")
    print("=" * 80)
    s = client.conn.session
    print(f"  peer            : {s.peer_id}")
    print(f"  client geo      : {s.src_geo}")
    print(f"  server geo      : {s.dst_geo}")
    print(f"  recovered src   : {s.recovered_src}")
    print(f"  geo verified    : {s.geo_verified}")
    print(f"  SVP             : {s.SVP:.12f}")
    print(f"  fingerprint     : {s.fingerprint}")
    print(f"  session_key     : {s.session_key[:32]}…")
    print(f"  clock drift     : {s.clock_drift:.6f}s")
    print(f"  local  tick     : {s.local_clock['tick_idx']}")
    print(f"  remote tick     : {s.remote_clock['tick_idx']}")
    print(f"  server parity   : {s.server_parity:.4f}")
    print(f"  packets sent    : {s.packets_sent}")
    print(f"  packets received: {s.packets_received}")
    print()

    # ---------- server view ----------
    print(f"--- server-side sessions ---")
    for sess in server.sessions:
        print(f"  peer={sess.peer_id}  SVP={sess.SVP:.8f}  "
              f"verified={sess.geo_verified}  "
              f"recv={sess.packets_received}  sent={sess.packets_sent}")
    print()

    client.close()
    time.sleep(0.1)
    server.stop()

    # ---------- complexity ----------
    print("--- complexity per phase ---")
    print("  TCP length prefix            O(1)")
    print("  MOSB frame + verify          O(n)")
    print("  coupling encode (geo)        O(n_events² + K_utf8·|payload|)")
    print("  coupling verify (cross)      O(n_events² + n_pi)")
    print("  spinor reconstruction        O(n_even · 720)")
    print("  clock snapshot               O(1)")
    print("  clock verify                 O(1)")
    print("  UTF-8 envelope               O(|data| · K_utf8)")
    print("  envelope verify              O(|data| · K_utf8)")
    print("  XOR stream + HMAC            O(n)")
    print("  ─────────────────────────────────────────")
    print("  per handshake                O(n_events² + n_even·720)")
    print("  per data packet              O(|data| · K_utf8 + n)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()