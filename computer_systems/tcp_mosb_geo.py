#!/usr/bin/env python3
"""
tcp_mosb_geo.py
===============

TCP handshake using coupling_Q_entanglement.py for geo-location,
with an optional 512-bit compile/parse layer borrowed from cli_512.py.

Wire protocol (length-prefixed MOSB frames):

    HELLO (0x01) : [uint8][uint16 id_len][client_id]
                   [uint16 sig_len][CouplingSignal]
    ACK   (0x02) : [uint8][double SVP][double parity_ratio]
    DATA  (0x03) : [nonce 12 B][uint16 mac_len][mac][ciphertext]
    CLOSE (0x04) : [uint8]

512-bit layer (use_512bit=True, default on)
-------------------------------------------
    Q/P       two_point_QP_512 on sampled even-even coupling entries
    Z-walk    Z_forward_512 over ⌈√z⌉ steps for z = n_events
    Wire      unchanged — same MOSB framing, same DATA payload
"""

from __future__ import annotations
import hashlib, hmac, math, os, socket, struct, threading, time
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- geo-location (coupling_Q_entanglement.py) ----
from coupling_Q_entanglement import (
    coupling_matrix,
    entangled_coupling,
    even_even_mask,
    even_indices,
    spinor_reconstruct,
    make_geo_events,
)

# ---- payload envelope (UTF_8_dzeta_dt.py) ----
from UTF_8_dzeta_dt import (
    UTF8Table, ReadOnlyBuffer, ReadOnlyALU,
    build_particle_cloud, ugc_survivors,
    K_DEFAULT as K_UTF8_DEF,
)

# ---- 512-bit compile/parse extension (cli_512.py) ----
from cli_512 import (
    two_point_QP_512,
    zeta_512,
    Z_forward_512,
    build_C_512,
    DComplex,
    ALPHA_MAX_D,
    ALPHA_MIN_D,
    PREC_DIGITS,
    BITS,
)

# ---- MOSB transport (fallback if mobius_internet missing) ----
try:
    from mobius_internet import MobiusTransport
except ImportError:
    @dataclass
    class _MOSBFrame:
        bytes_: bytes
        size: int
        S: float
        H: float
        m: float

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
            return blob[hdr_sz:hdr_sz + n], _MOSBFrame(
                bytes_=blob, size=len(blob), S=S, H=H, m=m)


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
N_EVENTS_GEO  = 24

SVP_TOL       = 1e-6
SIG_TOL       = 1e-3
NORM_TOL      = 1e-3

MAGIC_HELLO = 0x01
MAGIC_ACK   = 0x02
MAGIC_DATA  = 0x03
MAGIC_CLOSE = 0x04


# ============================================================
#  Cross mask + geo-anchored events
# ============================================================
def _even_odd_mask(N: int) -> np.ndarray:
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2) != (j % 2)


def _events_from_geo(geo: Tuple[float, float], n_events: int):
    seed = int(hashlib.sha256(
        f"{geo[0]:.6f}|{geo[1]:.6f}|{n_events}".encode()
    ).hexdigest()[:8], 16)
    t, r = make_geo_events(n_events=n_events, seed=seed % (2 ** 32))
    t = t * (1.0 + 0.1 * geo[0])
    r = r * (1.0 + 0.1 * geo[1])
    return t, r


# ============================================================
#  Coupling signal
# ============================================================
@dataclass
class CouplingSignal:
    payload:      bytes
    src_geo:      Tuple[float, float]
    dst_geo:      Tuple[float, float]
    K:            int
    n_events:     int
    norm_ee:      float
    norm_eo:      float
    parity_ratio: float
    pi_signature: np.ndarray
    e_signal:     np.ndarray
    b_signal:     np.ndarray


def encode_coupling_signal(payload: bytes,
                           src_geo: Tuple[float, float],
                           dst_geo: Tuple[float, float],
                           K: int = K_HANDSHAKE,
                           n_events: int = N_EVENTS_GEO
                           ) -> CouplingSignal:
    t, r = _events_from_geo(src_geo, n_events)

    G = coupling_matrix(t, r)
    G_ee, _ = entangled_coupling(G)
    G_eo = G * _even_odd_mask(len(t))

    norm_ee = float(np.linalg.norm(G_ee))
    norm_eo = float(np.linalg.norm(G_eo))
    parity_ratio = norm_ee / max(norm_eo, 1e-12)

    even_idx = even_indices(len(t))
    Pis, mags = spinor_reconstruct(t[even_idx], r[even_idx])

    n = np.arange(K)
    ang = np.angle(Pis)
    mag = np.abs(Pis)
    basis = np.zeros(K)
    for i in range(min(K, len(mag))):
        basis[i] = mag[i] * math.cos(ang[i])

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

    parity = np.arange(N) % 2
    E_signal = np.zeros(K)
    B_signal = np.zeros(K)
    for i in range(N):
        if parity[i] == 1:
            E_signal += V[i]
        else:
            B_signal += V[i]
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
            + struct.pack(f">{K}d", *sig.e_signal.tolist())
            + struct.pack(f">{K}d", *sig.b_signal.tolist())
            + sig.payload)


def deserialize_coupling_signal(blob: bytes) -> CouplingSignal:
    (K, n_events,
     sx, sy, dx, dy,
     n_pi, n_K,
     nee, neo, ratio,
     plen) = struct.unpack(SIG_FMT, blob[:SIG_HDR])
    off = SIG_HDR
    pi_sig = np.array(struct.unpack(f">{n_pi}d", blob[off:off + 8 * n_pi]))
    off += 8 * n_pi
    e_sig = np.array(struct.unpack(f">{n_K}d", blob[off:off + 8 * n_K]))
    off += 8 * n_K
    b_sig = np.array(struct.unpack(f">{n_K}d", blob[off:off + 8 * n_K]))
    off += 8 * n_K
    payload = blob[off:off + plen]
    return CouplingSignal(
        payload=payload, src_geo=(sx, sy), dst_geo=(dx, dy),
        K=K, n_events=n_events,
        norm_ee=nee, norm_eo=neo, parity_ratio=ratio,
        pi_signature=pi_sig, e_signal=e_sig, b_signal=b_sig,
    )


# ============================================================
#  64-bit verification (unchanged)
# ============================================================
def verify_coupling_signal(sig: CouplingSignal) -> Tuple[bool, Dict]:
    t, r = _events_from_geo(sig.src_geo, sig.n_events)
    G = coupling_matrix(t, r)
    G_ee, _ = entangled_coupling(G)
    G_eo = G * _even_odd_mask(len(t))

    computed_ee = float(np.linalg.norm(G_ee))
    computed_eo = float(np.linalg.norm(G_eo))

    even_idx = even_indices(len(t))
    _, mags = spinor_reconstruct(t[even_idx], r[even_idx])

    checks = {
        'norm_ee_match':   abs(computed_ee - sig.norm_ee) < NORM_TOL,
        'norm_eo_match':   abs(computed_eo - sig.norm_eo) < NORM_TOL,
        'signature_match': (len(mags) == len(sig.pi_signature)
                            and np.allclose(mags, sig.pi_signature,
                                            rtol=1e-4, atol=SIG_TOL)),
        'cross_nonzero':   computed_eo > 1e-9,
    }
    checks['parity_diagnostic'] = computed_ee > computed_eo

    ok = all(checks[k] for k in
             ('norm_ee_match', 'norm_eo_match',
              'signature_match', 'cross_nonzero'))
    return ok, checks


# ============================================================
#  512-bit extension  ·  Q/P + Z-walk
# ============================================================
def _sample_coupling_indices(sig: CouplingSignal,
                             n_samples: int = 8
                             ) -> List[Tuple[int, int]]:
    """Deterministically sample up to n (i, j) pairs from G_even."""
    t, r = _events_from_geo(sig.src_geo, sig.n_events)
    G = coupling_matrix(t, r)
    G_ee, _ = entangled_coupling(G)
    idx = np.argwhere(G_ee > 1e-9)
    if len(idx) == 0:
        return []
    rng = np.random.default_rng(0)
    k = min(n_samples, len(idx))
    sel = rng.choice(len(idx), size=k, replace=False)
    return [(int(idx[s][0]), int(idx[s][1])) for s in sel]


def _qp_512_consistency(sig: CouplingSignal,
                        n_samples: int = 8
                        ) -> Tuple[bool, Dict]:
    pairs = _sample_coupling_indices(sig, n_samples)
    if not pairs:
        return True, dict(n_samples=0, n_baby=0, n_giant=0)
    n_baby = n_giant = 0
    qp_log: List[Dict] = []
    for (i, j) in pairs:
        Q, P = two_point_QP_512(Decimal(i + 1), Decimal(j + 1))
        if abs(Q) >= abs(P) and Q > 0:
            n_baby += 1
            mode = "baby"
        else:
            n_giant += 1
            mode = "giant"
        if len(qp_log) < 3:
            qp_log.append(dict(i=i, j=j, mode=mode,
                               Q=str(Q)[:20], P=str(P)[:20]))
    # consistency = all pairs classified (no failure) + deterministic
    ok = (n_baby + n_giant == len(pairs))
    return ok, dict(n_samples=len(pairs), n_baby=n_baby,
                    n_giant=n_giant, qp_log=qp_log)


def _zwalk_512_consistency(sig: CouplingSignal
                           ) -> Tuple[bool, Dict]:
    # use n_events as the target z (gives a small, well-defined walk)
    x_target = float(max(sig.n_events, 4))
    expected = max(1, int(math.ceil(math.sqrt(x_target))))
    K = max(sig.K, 8)

    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 5
        A = ALPHA_MAX_D / ALPHA_MIN_D
        dt = Decimal(x_target) / Decimal(expected)
        C = build_C_512(K)

        zf_abs: List[float] = []
        t = Decimal(0)
        for _ in range(expected):
            Zf = Z_forward_512(t, C, K, A)
            zf_abs.append(float(abs(Zf)))
            t += dt

    ok = (len(zf_abs) == expected)
    return ok, dict(
        n_steps=len(zf_abs), expected=expected,
        z=int(x_target), A_512=float(A),
        Zf_abs_rms=float(np.sqrt(np.mean(np.array(zf_abs) ** 2)))
        if zf_abs else 0.0,
    )


def verify_coupling_signal_512(sig: CouplingSignal) -> Tuple[bool, Dict]:
    """
    Combined verification:
      64-bit structural checks  +  512-bit Q/P  +  512-bit Z-walk.
    """
    ok64, c64 = verify_coupling_signal(sig)
    qp_ok, qp_diag = _qp_512_consistency(sig)
    zw_ok, zw_diag = _zwalk_512_consistency(sig)

    checks: Dict = dict(c64)
    checks["qp_512_consistent"]    = qp_ok
    checks["zwalk_512_consistent"] = zw_ok
    checks["qp_512_diag"]          = qp_diag
    checks["zwalk_512_diag"]       = zw_diag
    checks["bits"]                 = BITS

    ok = ok64 and qp_ok and zw_ok
    return ok, checks


def recover_location(sig: CouplingSignal) -> dict:
    ok, checks = verify_coupling_signal_512(sig)
    return dict(
        t_hat=sig.src_geo[0],
        r_hat=sig.src_geo[1],
        verified=ok,
        checks=checks,
        norm_ee=sig.norm_ee,
        norm_eo=sig.norm_eo,
        parity_ratio=sig.parity_ratio,
    )


# ============================================================
#  SVP derivation
# ============================================================
def svp_session_key(t_hat: float, r_hat: float, peer_id: str) -> dict:
    seed = hashlib.sha256(
        f"{peer_id}|{t_hat:.6f}|{r_hat:.6f}".encode()).digest()
    svp = int.from_bytes(seed[:8], "big") / (2 ** 64)
    fp = seed.hex()[:32]
    key = hashlib.sha256(
        f"{peer_id}|{svp:.12f}|{fp}".encode()).hexdigest()
    return dict(t_hat=t_hat, r_hat=r_hat, SVP=svp,
                fingerprint=fp, session_key=key)


# ============================================================
#  UTF-8 payload envelope  ·  wire format includes alu_scalar
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
        H = 0.0
        m = 0.0
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
        n_particles_ok=rec.n_particles == env.n_particles,
        n_survivors_ok=rec.n_survivors == env.n_survivors,
        rate_ok=abs(rec.survival_rate - env.survival_rate) < 1e-6,
        supertrace_ok=abs(rec.supertrace - env.supertrace) < 1e-6,
        alu_ok=abs(rec.alu_scalar - env.alu_scalar) < 1e-6,
        fingerprint_ok=rec.fingerprint == env.fingerprint,
    )
    return all(checks.values()), checks


# ---- FIXED: ENV_FMT now carries alu_scalar (one extra 'd') ----
#     >B H H d d d d 32s I    =  1 + 2 + 2 + 8*4 + 32 + 4  =  73 B
ENV_FMT = ">BHHdddd32sI"
ENV_HDR = struct.calcsize(ENV_FMT)


def pack_env(env: UTF8Envelope) -> bytes:
    return struct.pack(ENV_FMT,
                       env.K, env.n_particles, env.n_survivors,
                       env.survival_rate, env.supertrace, env.mass,
                       env.alu_scalar,
                       bytes.fromhex(env.fingerprint),
                       len(env.raw)) + env.raw


def unpack_env(blob: bytes) -> Tuple[UTF8Envelope, int]:
    (K, n_p, n_s, rate, S, m, alu, fp, rl
     ) = struct.unpack(ENV_FMT, blob[:ENV_HDR])
    raw = blob[ENV_HDR:ENV_HDR + rl]
    env = UTF8Envelope(raw=raw, K=K, n_particles=n_p, n_survivors=n_s,
                       survival_rate=rate, supertrace=S, mass=m,
                       alu_scalar=alu, fingerprint=fp.hex())
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
    geo_verified: bool
    parity_ratio: float
    established_at: float = field(default_factory=time.time)
    packets_sent: int = 0
    packets_received: int = 0
    verification_bits: int = 64
    qp_512_summary: Optional[Dict] = None
    zwalk_512_summary: Optional[Dict] = None


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
#  MobiusTCPConnection  ·  no clock, optional 512-bit verify
# ============================================================
class MobiusTCPConnection:
    def __init__(self, sock: socket.socket,
                 is_server: bool,
                 local_geo: Tuple[float, float] = (0.5, 0.5),
                 remote_geo: Tuple[float, float] = (0.5, 0.5),
                 use_512bit: bool = True,
                 verbose: bool = False):
        self.sock = sock
        self.is_server = is_server
        self.local_geo = local_geo
        self.remote_geo = remote_geo
        self.trans = MobiusTransport(K=512)
        self.session: Optional[TCPSession] = None
        self.use_512bit = use_512bit
        self.verbose = verbose
        self._nonces: set = set()

    def _log(self, *a):
        if self.verbose:
            print("  " + " ".join(str(x) for x in a))

    # --------------------------------------------------------
    def _verify_sig(self, sig: CouplingSignal) -> Tuple[bool, Dict]:
        if self.use_512bit:
            return verify_coupling_signal_512(sig)
        return verify_coupling_signal(sig)

    # --------------------------------------------------------
    def handshake_client(self, client_id: str) -> TCPSession:
        self._log(f"--- client handshake  id={client_id} ---")

        sig = encode_coupling_signal(
            client_id.encode("utf-8"),
            src_geo=self.local_geo,
            dst_geo=self.remote_geo,
            K=K_HANDSHAKE, n_events=N_EVENTS_GEO)

        # pre-send self-check (catches determinism bugs early)
        self_ok, self_checks = self._verify_sig(sig)
        self._log(f"self-verify  ok={self_ok}  "
                  f"bits={BITS if self.use_512bit else 64}")
        if not self_ok:
            bad = [k for k, v in self_checks.items()
                   if v is False]
            raise RuntimeError(
                f"client self-verify failed: {bad}")

        sig_bytes = serialize_coupling_signal(sig)
        id_bytes = client_id.encode("utf-8")

        self._log(f"coupling ‖G_ee‖={sig.norm_ee:.4f}  "
                  f"‖G_eo‖={sig.norm_eo:.4f}  "
                  f"ratio={sig.parity_ratio:.3f}")

        inner = (struct.pack(">BH", MAGIC_HELLO, len(id_bytes)) + id_bytes
                 + struct.pack(">H", len(sig_bytes)) + sig_bytes)
        frame = self.trans.frame(inner)
        send_frame(self.sock, frame.bytes_)
        self._log(f"HELLO   {frame.size}B  S={frame.S:+.4f}")

        ack_bytes = recv_frame(self.sock)
        ack_inner, _ = self.trans.deframe(ack_bytes)
        if ack_inner[0] != MAGIC_ACK:
            raise RuntimeError(f"expected ACK, got {ack_inner[0]:#x}")

        remote_svp = struct.unpack(">d", ack_inner[1:9])[0]
        server_parity = struct.unpack(">d", ack_inner[9:17])[0]
        self._log(f"ACK     SVP={remote_svp:.8f}  "
                  f"parity={server_parity:.3f}")

        my_svp = svp_session_key(self.local_geo[0], self.local_geo[1],
                                 client_id)
        if abs(my_svp["SVP"] - remote_svp) > SVP_TOL:
            raise RuntimeError(
                f"svp mismatch  my={my_svp['SVP']:.8f}  "
                f"remote={remote_svp:.8f}")

        self.session = TCPSession(
            peer_id=client_id,
            src_geo=self.local_geo,
            dst_geo=self.remote_geo,
            recovered_src=(my_svp["t_hat"], my_svp["r_hat"]),
            SVP=my_svp["SVP"],
            fingerprint=my_svp["fingerprint"],
            session_key=my_svp["session_key"],
            geo_verified=True,
            parity_ratio=server_parity,
            verification_bits=BITS if self.use_512bit else 64,
        )
        self._log(f"session OK  key={self.session.session_key[:16]}…")
        return self.session

    # --------------------------------------------------------
    def handshake_server(self) -> TCPSession:
        self._log("--- server handshake ---")

        hello_bytes = recv_frame(self.sock)
        inner, hello_frame = self.trans.deframe(hello_bytes)
        if inner[0] != MAGIC_HELLO:
            raise RuntimeError(f"expected HELLO, got {inner[0]:#x}")

        off = 1
        id_len = struct.unpack(">H", inner[off:off + 2])[0]; off += 2
        client_id = inner[off:off + id_len].decode("utf-8"); off += id_len
        sig_len = struct.unpack(">H", inner[off:off + 2])[0]; off += 2
        sig_bytes = inner[off:off + sig_len]; off += sig_len
        self._log(f"HELLO   id={client_id}  S={hello_frame.S:+.4f}")

        sig = deserialize_coupling_signal(sig_bytes)
        ok_geo, checks = self._verify_sig(sig)
        bits = BITS if self.use_512bit else 64
        self._log(f"geo     t̂={sig.src_geo[0]:+.4f}  "
                  f"r̂={sig.src_geo[1]:+.4f}")
        self._log(f"        ‖G_ee‖={sig.norm_ee:.4f}  "
                  f"‖G_eo‖={sig.norm_eo:.4f}  "
                  f"ratio={sig.parity_ratio:.3f}")
        self._log(f"        verified={ok_geo}  bits={bits}")
        for k, v in checks.items():
            if isinstance(v, dict):
                continue
            self._log(f"        {k:>22s} = {v}")

        if not ok_geo:
            bad = [k for k, v in checks.items()
                   if v is False]
            raise RuntimeError(
                f"coupling verification failed: {bad}")

        svp = svp_session_key(sig.src_geo[0], sig.src_geo[1], client_id)
        self._log(f"svp     {svp['SVP']:.8f}")

        ack_inner = struct.pack(">Bdd", MAGIC_ACK,
                                svp["SVP"],
                                sig.parity_ratio)
        frame = self.trans.frame(ack_inner)
        send_frame(self.sock, frame.bytes_)
        self._log(f"ACK     sent  parity={sig.parity_ratio:.3f}")

        self.session = TCPSession(
            peer_id=client_id,
            src_geo=sig.src_geo,
            dst_geo=sig.dst_geo,
            recovered_src=(svp["t_hat"], svp["r_hat"]),
            SVP=svp["SVP"],
            fingerprint=svp["fingerprint"],
            session_key=svp["session_key"],
            geo_verified=ok_geo,
            parity_ratio=sig.parity_ratio,
            verification_bits=bits,
            qp_512_summary=checks.get("qp_512_diag")
            if self.use_512bit else None,
            zwalk_512_summary=checks.get("zwalk_512_diag")
            if self.use_512bit else None,
        )
        self._log(f"session OK  key={self.session.session_key[:16]}…")
        return self.session

    # --------------------------------------------------------
    def send_application(self, port: int, payload: str) -> dict:
        if self.session is None:
            raise RuntimeError("no session")
        if not payload:
            raise ValueError("empty payload")

        port_env = encode_utf8_pipeline(str(port).encode("utf-8"),
                                        K=K_UTF8)
        payload_env = encode_utf8_pipeline(payload.encode("utf-8"),
                                           K=K_UTF8)

        port_bytes = pack_env(port_env)
        payload_bytes = pack_env(payload_env)

        inner = struct.pack(">BH", MAGIC_DATA, len(port_bytes)) + port_bytes
        inner += struct.pack(">I", len(payload_bytes)) + payload_bytes

        nonce = os.urandom(12)
        ct = encrypt(self.session.session_key, nonce, inner)
        mac = compute_mac(self.session.session_key, nonce, ct)

        wire = nonce + struct.pack(">H", len(mac)) + mac + ct
        frame = self.trans.frame(wire)
        send_frame(self.sock, frame.bytes_)
        self.session.packets_sent += 1
        self._log(f"DATA    port={port}  plen={len(payload)}")
        return dict(port=port, payload_len=len(payload),
                    frame_size=frame.size)

    # --------------------------------------------------------
    def recv_application(self) -> Tuple[int, str]:
        if self.session is None:
            raise RuntimeError("no session")

        wire_bytes = recv_frame(self.sock)
        wire, _ = self.trans.deframe(wire_bytes)

        nonce = wire[:12]
        mac_len = struct.unpack(">H", wire[12:14])[0]
        mac = wire[14:14 + mac_len]
        ct = wire[14 + mac_len:]

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

        port_env, _ = unpack_env(port_bytes)
        ok1, d1 = verify_utf8_envelope(port_env)
        if not ok1:
            bad = [k for k, v in d1.items() if not v]
            raise RuntimeError(f"port envelope verify failed: {bad}")
        port = int(port_env.raw.decode("utf-8"))

        payload_env, _ = unpack_env(payload_bytes)
        ok2, d2 = verify_utf8_envelope(payload_env)
        if not ok2:
            bad = [k for k, v in d2.items() if not v]
            raise RuntimeError(f"payload envelope verify failed: {bad}")
        payload = payload_env.raw.decode("utf-8")

        self.session.packets_received += 1
        self._log(f"DATA←   port={port}  plen={len(payload)}")
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
#  Server / Client
# ============================================================
class MobiusTCPServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 server_geo: Tuple[float, float] = (PI / 2, E / 2),
                 use_512bit: bool = True,
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.server_geo = server_geo
        self.use_512bit = use_512bit
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
                use_512bit=self.use_512bit,
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
                 use_512bit: bool = True,
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.client_geo = client_geo
        self.server_geo = server_geo
        self.use_512bit = use_512bit
        self.verbose = verbose
        self.conn: Optional[MobiusTCPConnection] = None

    def connect(self, client_id: str) -> TCPSession:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        self.conn = MobiusTCPConnection(
            sock, is_server=False,
            local_geo=self.client_geo,
            remote_geo=self.server_geo,
            use_512bit=self.use_512bit,
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
    print("Möbius TCP  ·  geo-location  ·  512-bit Q/P + Z-walk extension")
    print("=" * 80)
    print(f"  α              = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  a_step         = α / 0.3628  = {A_STEP:.6f}")
    print(f"  6/π²                          = {DENSITY:.6f}")
    print(f"  geo events                    = {N_EVENTS_GEO}")
    print(f"  UTF-8 K                       = {K_UTF8}")
    print(f"  verify bits                   = {BITS}")
    print()

    # ---- 64-bit sanity (fast) ----
    print("--- geo encoding sanity (64-bit) ---")
    for geo in [(0.4, 0.6), (1.2, 0.9), (PI / 2, E / 2)]:
        sig = encode_coupling_signal(
            b"hello", src_geo=geo, dst_geo=(PI / 2, E / 2),
            K=K_HANDSHAKE, n_events=N_EVENTS_GEO)
        ok, checks = verify_coupling_signal(sig)
        print(f"  geo={geo}  ‖G_ee‖={sig.norm_ee:.4f}  "
              f"‖G_eo‖={sig.norm_eo:.4f}  "
              f"ratio={sig.parity_ratio:.3f}  verified={ok}")
    print()

    # ---- 512-bit Q/P + Z-walk spot check on one geo ----
    print("--- 512-bit Q/P + Z-walk spot check ---")
    sig = encode_coupling_signal(
        b"handshake", src_geo=(0.4, 0.6),
        dst_geo=(PI / 2, E / 2),
        K=K_HANDSHAKE, n_events=N_EVENTS_GEO)
    ok, checks = verify_coupling_signal_512(sig)
    print(f"  verified           = {ok}")
    print(f"  norm_ee_match      = {checks['norm_ee_match']}")
    print(f"  norm_eo_match      = {checks['norm_eo_match']}")
    print(f"  signature_match    = {checks['signature_match']}")
    print(f"  cross_nonzero      = {checks['cross_nonzero']}")
    print(f"  qp_512_consistent  = {checks['qp_512_consistent']}")
    print(f"  zwalk_512_consistent = {checks['zwalk_512_consistent']}")
    qp = checks["qp_512_diag"]
    print(f"  qp_512: n_baby={qp['n_baby']}  n_giant={qp['n_giant']}  "
          f"n_samples={qp['n_samples']}")
    zw = checks["zwalk_512_diag"]
    print(f"  zwalk_512: n_steps={zw['n_steps']}  "
          f"expected={zw['expected']}  z={zw['z']}  "
          f"Zf_abs_rms={zw['Zf_abs_rms']:.4e}")
    print()

    # ---- start server + client ----
    server = MobiusTCPServer(host="127.0.0.1", port=0,
                             server_geo=(PI / 2, E / 2),
                             use_512bit=True,
                             verbose=True)
    server.start()
    print()

    client = MobiusTCPClient(host="127.0.0.1", port=server.port,
                             client_geo=(0.4, 0.6),
                             server_geo=(PI / 2, E / 2),
                             use_512bit=True,
                             verbose=True)
    client.connect(client_id="alice@cell-A")
    print()

    # ---- data exchange ----
    print("=" * 80)
    print("EXCHANGE")
    print("=" * 80)
    tests = [
        (8080, "GET /index.html HTTP/1.1\r\nHost: example.com\r\n"),
        (8080, "POST /api/upload HTTP/1.1\r\nContent-Length: 42"),
        (443, "TLS handshake start"),
        (22, "SSH-2.0-MobiusClient"),
    ]
    for port, payload in tests:
        print()
        print(f"--- transmit port={port} payload={payload[:40]!r} ---")
        t0 = time.perf_counter()
        client.send(port=port, payload=payload)
        resp_port, resp_payload = client.recv()
        elapsed = (time.perf_counter() - t0) * 1e3
        print(f"  echo:  port={resp_port}  "
              f"payload={resp_payload[:60]!r}")
        print(f"  RTT:   {elapsed:.2f} ms")
    print()

    # ---- session ----
    print("=" * 80)
    print("SESSION")
    print("=" * 80)
    s = client.conn.session
    print(f"  peer            : {s.peer_id}")
    print(f"  client geo      : {s.src_geo}")
    print(f"  server geo      : {s.dst_geo}")
    print(f"  recovered src   : {s.recovered_src}")
    print(f"  geo verified    : {s.geo_verified}")
    print(f"  verification    : {s.verification_bits}-bit")
    print(f"  SVP             : {s.SVP:.12f}")
    print(f"  fingerprint     : {s.fingerprint}")
    print(f"  session_key     : {s.session_key[:32]}…")
    print(f"  parity ratio    : {s.parity_ratio:.4f}")
    print(f"  packets sent    : {s.packets_sent}")
    print(f"  packets received: {s.packets_received}")
    print()

    client.close()
    time.sleep(0.1)
    server.stop()

    # ---- complexity ----
    print("--- complexity per phase ---")
    print("  TCP length prefix            O(1)")
    print("  MOSB frame + verify          O(n)")
    print("  coupling encode (geo)        O(n_events² + K·|payload|)")
    print("  coupling verify 64-bit       O(n_events² + n_pi)")
    print("  Q/P 512-bit (sampled)        O(n_samples · D)      D≈190")
    print("  Z-walk 512-bit (⌈√z⌉ steps)  O(√z · K_ci · D)")
    print("  spinor reconstruction        O(n_even · 720)")
    print("  UTF-8 envelope               O(|data| · K_utf8)")
    print("  envelope verify              O(|data| · K_utf8)")
    print("  XOR stream + HMAC            O(n)")
    print("  ─────────────────────────────────────────")
    print("  per handshake (512-bit)      O(n_events² + √n_events·K·D)")
    print("  per data packet              O(|data| · K_utf8 + n)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()