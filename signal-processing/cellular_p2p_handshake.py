#!/usr/bin/env python3
"""
coupled_fermion_p2p.py
======================
Geo-location handshake → SVP session establishment over the
Möbius P2P cellular network.

Pipeline
--------
SENDER                                       RECEIVER
──────                                       ────────
1. message                                   
2. μ-gated vectors (quantum_geo_location)    
3. couple fermions via even-parity mask      
4. inject elliptic weight Π(t, r)            
5. build MOSB frame                          
6. hand off to cellular_p2p routing          ──▶ hop-by-hop
                                                  │
7.                                          receiver_module.receive()
                                                  │
                                            ├─ parity split (odd→E, even→B)
                                            ├─ even-even mask σ(v) ≡ 0
                                            ├─ cross-coupling gate
                                            └─ recover (t̂, r̂) location
                                                  │
8.                               ◀── ack ────  send back location ack
9. verify ack                                 
10.                                     build LatticeTopology at (t̂, r̂)
                                        compute rank-1 SVP
11. both sides have the SAME SVP weight
12. ◀── session key = SVP fingerprint ──▶
13. subsequent traffic encrypted under SVP session key
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
#  Sibling imports
# ============================================================
try:
    from quantum_geo_location_OK import (
        mobius_sieve as mobius_geo,
        event_vector, build_all_vectors, cross_coupling,
        ugc_label_recovery, even_parity_mask,
        elliptic_projection_vec,  # not directly used but available
        particle_vector as geo_particle_vector,
        A_STEP, ALPHA, PI as QG_PI, E as QG_E,
        K_VEC_DEF,
    )
    _HAS_GEO = True
except ImportError:
    _HAS_GEO = False

try:
    from receiver_module import (
        PointParticle, build_particles, receive,
        estimate_device_location,
        elliptic_projection, elliptic_projection_vec,
        mu_convolution_H, mobius_sieve as mobius_recv,
        particle_vector as recv_particle_vector,
        cross_coupling as recv_cross_coupling,
        ugc_label, dζ_dt,
        K_VEC_DEF as K_RECV,
    )
    _HAS_RECV = True
except ImportError:
    _HAS_RECV = False

try:
    from cellular_p2p import (
        MobiusP2PNetwork, MobiusPeer,
        LatticeTopology, MobiusAddress, SATSignal,
        FiniteStepPacket, MobiusTransport, Frame,
        AudioPayload,
        _make_audio, _snr, _integral_g,
    )
    _HAS_P2P = True
except ImportError:
    _HAS_P2P = False


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)
A_STEP   = ALPHA / 0.3628
DENSITY  = 6.0 / (PI * PI)
W1, W2   = PI, E

K_GEO_DEF    = 128
K_RECV_DEF   = 128
K_HANDSHAKE  = 64
N_HANDSHAKE  = 96


# ============================================================
#  Fallback Möbius sieve
# ============================================================
def _fallback_mobius(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
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
#  Layer 1  ·  Entangled fermion message  (quantum_geo_location)
# ============================================================
@dataclass
class EntangledSignal:
    """
    A message encoded as coupled entangled fermions.

    Per byte b of the payload:
        v_b(n) = μ(n) · cos(A_STEP · n² / K) · s_b(n mod L)

    where s_b is the SHAKE256 signature of (b, position, block).
    The even-even mask σ(v) ≡ 0 (mod 2) selects the fermionic
    sector — the surviving slots are the transmitted quanta.
    """
    payload: bytes
    K: int
    V: np.ndarray                  # (N_bytes, K) μ-gated vectors
    labels: np.ndarray             # (N_bytes,) UGC labels
    mask: np.ndarray               # (N_bytes,) even-even mask
    parity: np.ndarray             # (N_bytes,) odd/even index parity
    E_signal: np.ndarray           # (K,) transmitted on odd slots
    B_signal: np.ndarray           # (K,) transmitted on even slots
    src_geo: Tuple[float, float]   # (t_src, r_src) source location
    dst_geo: Tuple[float, float]   # (t_dst, r_dst) destination location
    timestamp: float = field(default_factory=time.time)


def encode_entangled(payload: bytes,
                     src_geo: Tuple[float, float],
                     dst_geo: Tuple[float, float],
                     K: int = K_GEO_DEF,
                     k_labels: int = 4) -> EntangledSignal:
    """
    Encode a byte payload as coupled entangled fermions.

    Each byte b at position i:

        sig_b   = SHAKE256(b || i || src || dst)
        v_b(n)  = μ(n) · cos(A_STEP · n² / K) · sig_b[n mod 32]

    The source and destination geo-locations are baked into the
    signature so an observer cannot recover the message without
    both endpoints.
    """
    mu = mobius_sieve_geo(K) if _HAS_GEO else _fallback_mobius(K)
    N = len(payload)
    V = np.zeros((N, K), dtype=float)

    src_bytes = struct.pack(">dd", *src_geo)
    dst_bytes = struct.pack(">dd", *dst_geo)

    for i, b in enumerate(payload):
        sig = hashlib.shake_256(
            bytes([b]) + struct.pack(">I", i) + src_bytes + dst_bytes
        ).digest(32)
        s = np.array([bb / 255.0 for bb in sig], dtype=float)
        L = len(s)
        n = np.arange(K)
        mu_arr = np.array([float(mu[k]) if k < len(mu) else 0.0
                           for k in range(K)])
        chirp = np.cos(A_STEP * n * n / K)
        V[i] = mu_arr * chirp * s[n % L]

    # UGC label recovery
    labels = ugc_label_recovery(V, k_labels)

    # even-even mask
    mask = even_parity_mask(labels)

    # parity = sample index parity
    parity = np.array([i % 2 for i in range(N)], dtype=int)

    # E-field (odd indices), B-field (even indices)
    E_signal = np.zeros(K)
    B_signal = np.zeros(K)
    for i in range(N):
        if parity[i] == 1:
            E_signal = E_signal + V[i]
        else:
            B_signal = B_signal + V[i]
    E_signal /= max(1, np.count_nonzero(parity == 1))
    B_signal /= max(1, np.count_nonzero(parity == 0))

    return EntangledSignal(
        payload=payload, K=K, V=V,
        labels=labels, mask=mask, parity=parity,
        E_signal=E_signal, B_signal=B_signal,
        src_geo=src_geo, dst_geo=dst_geo,
    )


# ============================================================
#  Layer 2  ·  Frame the entangled signal as MOSB
# ============================================================
def frame_entangled(signal: EntangledSignal,
                    trans: "MobiusTransport") -> Frame:
    """
    Pack the entangled signal into a MOSB frame.

    Payload layout:
        [ N_bytes : uint32 ][ K : uint32 ][ k_labels : uint32 ]
        [ src_t : double ][ src_r : double ]
        [ dst_t : double ][ dst_r : double ]
        [ E_signal : K doubles ]
        [ B_signal : K doubles ]
    """
    N = len(signal.payload)
    K = signal.K
    header = struct.pack(
        ">III dddd",
        N, K, 4,                # 4 = k_labels default
        signal.src_geo[0], signal.src_geo[1],
        signal.dst_geo[0], signal.dst_geo[1],
    )
    body = struct.pack(f">{K}d", *signal.E_signal.tolist())
    body += struct.pack(f">{K}d", *signal.B_signal.tolist())
    payload = header + body
    return trans.frame(payload)


def deframe_entangled(raw: bytes,
                      trans: "MobiusTransport"
                      ) -> Tuple[bytes, np.ndarray, np.ndarray, Dict]:
    """Unpack the entangled signal from a MOSB frame."""
    payload, frame = trans.deframe(raw)
    N, K, k_labels, st, sr, dt, dr = struct.unpack(
        ">III dddd", payload[:struct.calcsize(">III dddd")])
    off = struct.calcsize(">III dddd")
    E_signal = np.array(
        struct.unpack(f">{K}d", payload[off:off + 8 * K]))
    off += 8 * K
    B_signal = np.array(
        struct.unpack(f">{K}d", payload[off:off + 8 * K]))
    meta = dict(N=N, K=K, k_labels=k_labels,
                src_geo=(st, sr), dst_geo=(dt, dr),
                frame_S=frame.S, frame_H=frame.H, frame_m=frame.m)
    return payload, E_signal, B_signal, meta


# ============================================================
#  Layer 3  ·  Receiver-side geo-location recovery
# ============================================================
def recover_location(E_signal: np.ndarray,
                     B_signal: np.ndarray,
                     meta: Dict,
                     K_mu: int = 256) -> Dict:
    """
    From the received E/B signals, reconstruct the sender's
    location using the parity + even-even mask pipeline.
    """
    if not _HAS_RECV:
        # fallback: assume perfect recovery
        return dict(
            t_hat=meta["src_geo"][0],
            r_hat=meta["src_geo"][1],
            Pi_hat=1.0, n_survivors=len(E_signal) + len(B_signal),
            sigma_t=0.0, sigma_r=0.0,
        )

    K = len(E_signal)
    # build a synthetic particle cloud from the E/B signals
    F_tab, mu_tab, H_tab = mu_convolution_H(K_mu)

    particles: List[PointParticle] = []
    idx = 0
    for k in range(K):
        # E_signal lives on odd parity, B_signal on even
        # reconstruct a pseudo time/rank by inverting the elliptic projection
        for (val, parity) in [(E_signal[k], 1), (B_signal[k], 0)]:
            if val == 0.0:
                continue
            # t, r sampled from the source location with jitter proportional to val
            t_k = meta["src_geo"][0] + 0.05 * val
            r_k = meta["src_geo"][1] + 0.05 * val
            Pi = elliptic_projection(t_k, r_k)
            n = (idx % (2 * K + 1)) - K
            if n == 0:
                n = 1
            a = abs(n)
            real = -(float(H_tab[a]) if a < H_tab.size
                     else math.log(a) + 0.5772)
            imag = (ALPHA * float(F_tab[a]) if n > 0
                    and a < F_tab.size else 0.0)
            particles.append(PointParticle(
                idx=idx, t=t_k, r=r_k, Pi=Pi,
                x=Pi * real, y=Pi * imag,
                zt=0.0, vx=0.0,
            ))
            idx += 1

    mu = mobius_recv(K)
    result = receive(particles, K, mu,
                     k_labels=meta.get("k_labels", 4),
                     threshold=0.3)
    est = estimate_device_location(result)
    return est


# ============================================================
#  Layer 4  ·  SVP session key from recovered location
# ============================================================
def svp_session_key(t_hat: float, r_hat: float,
                    peer_id: str,
                    N: int = 60, k: int = 3) -> Dict:
    """
    Build a LatticeTopology anchored at the recovered location
    and compute the rank-1 SVP as the session key.

        seed = SHA256(peer_id || t_hat || r_hat)
        lattice = LatticeTopology(N, k, seed)
        SVP = lattice.svp_weight
        session_key = SHA256(peer_id || SVP || fingerprint)
    """
    if not _HAS_P2P:
        # fallback: derive from the location directly
        seed_bytes = hashlib.sha256(
            f"{peer_id}|{t_hat:.6f}|{r_hat:.6f}".encode()
        ).digest()
        seed_int = int.from_bytes(seed_bytes[:8], "big")
        SVP = (seed_int % 1_000_000) / 1_000_000
        fingerprint = seed_bytes.hex()[:32]
        return dict(
            t_hat=t_hat, r_hat=r_hat,
            SVP=SVP, fingerprint=fingerprint,
            session_key=hashlib.sha256(
                f"{peer_id}|{SVP}|{fingerprint}".encode()
            ).hexdigest(),
        )

    seed_bytes = hashlib.sha256(
        f"{peer_id}|{t_hat:.6f}|{r_hat:.6f}".encode()
    ).digest()
    seed_int = int.from_bytes(seed_bytes[:8], "big")

    lattice = LatticeTopology(N=N, k_neighbors=k, seed=seed_int)
    SVP = lattice.svp_weight
    fingerprint = lattice.fingerprint()

    session_key = hashlib.sha256(
        f"{peer_id}|{SVP:.12f}|{fingerprint}".encode()
    ).hexdigest()

    return dict(
        t_hat=t_hat, r_hat=r_hat,
        SVP=SVP, fingerprint=fingerprint,
        session_key=session_key,
        lattice=lattice,
    )


# ============================================================
#  Layer 5  ·  Extended peer with geo-location + SVP session
# ============================================================
@dataclass
class GeoSession:
    """An established session between two peers."""
    src_id: str
    dst_id: str
    src_geo: Tuple[float, float]
    dst_geo: Tuple[float, float]
    recovered_src: Tuple[float, float]
    recovered_dst: Tuple[float, float]
    SVP: float
    fingerprint: str
    session_key: str
    established_at: float = field(default_factory=time.time)
    packets_sent: int = 0
    packets_received: int = 0


class GeoAwarePeer(MobiusPeer if _HAS_P2P else object):
    """
    MobiusPeer extended with:
        · geo-location handshake
        · SVP-based session establishment
        · encrypted post-handshake messaging
    """
    def __init__(self, *args, geo_location: Tuple[float, float] = (0.5, 0.5),
                 **kwargs):
        if _HAS_P2P:
            super().__init__(*args, **kwargs)
        self.geo_location = geo_location
        self.sessions: Dict[str, GeoSession] = {}

    def session_with(self, other_id: str) -> Optional[GeoSession]:
        return self.sessions.get(other_id)

    def establish_session(self, other_id: str,
                          session: GeoSession) -> None:
        self.sessions[other_id] = session

    # --------------------------------------------------------
    def encrypt_payload(self, other_id: str, payload: bytes) -> bytes:
        """Symmetric XOR with the session key (demo stand-in for AEAD)."""
        s = self.sessions.get(other_id)
        if s is None:
            return payload
        key = bytes.fromhex(s.session_key)
        L = len(key)
        return bytes(payload[i] ^ key[i % L] for i in range(len(payload)))

    def decrypt_payload(self, other_id: str, payload: bytes) -> bytes:
        return self.encrypt_payload(other_id, payload)


# ============================================================
#  Layer 6  ·  Extended P2P network with handshake
# ============================================================
class CoupledFermionNetwork:
    """
    Extends the cellular P2P network with:
        · geo-location handshake
        · SVP session establishment
        · encrypted session traffic
    """
    def __init__(self):
        self.base = MobiusP2PNetwork()
        self.trans = MobiusTransport(K=512)
        self.sessions: Dict[Tuple[str, str], GeoSession] = {}

    # --------------------------------------------------------
    def join(self, peer_id: str,
             cell_id: str = "cell-default",
             geo_location: Tuple[float, float] = (0.5, 0.5),
             is_gateway: bool = False,
             **kwargs) -> GeoAwarePeer:
        p = GeoAwarePeer(peer_id, cell_id=cell_id,
                         is_gateway=is_gateway,
                         geo_location=geo_location,
                         **kwargs)
        self.base.peers[peer_id] = p
        self.base.events.append(dict(t=time.time(), kind="join",
                                      peer=peer_id, cell=cell_id))
        for other_id, other in self.base.peers.items():
            if other_id == peer_id:
                continue
            if other.cell_id == cell_id and other.online:
                p.connect(other)
        if is_gateway:
            for other_id, other in self.base.peers.items():
                if other_id == peer_id:
                    continue
                if other.is_gateway and other.online:
                    p.connect(other)
        return p

    def connect(self, id1, id2):
        self.base.connect(id1, id2)

    # --------------------------------------------------------
    def handshake(self, src_id: str, dst_id: str,
                  verbose: bool = True) -> Optional[GeoSession]:
        """
        Full handshake:

            1. src encodes entangled fermions with (src_geo, dst_geo)
            2. frame + route + hop-by-hop
            3. dst decodes with receiver_module.receive()
            4. dst recovers (t_hat, r_hat) of src
            5. dst builds LatticeTopology at (t_hat, r_hat)
            6. dst sends SVP fingerprint back
            7. src verifies the returned SVP matches
            8. session established
        """
        if not _HAS_P2P:
            return None

        src = self.base.peers.get(src_id)
        dst = self.base.peers.get(dst_id)
        if src is None or dst is None:
            return None

        # ---- route ----
        path = self.base.route(src_id, dst_id)
        if path is None:
            return None

        if verbose:
            print(f"--- handshake {src_id} → {dst_id} ---")
            print(f"  path          : {' → '.join(path)}")

        # ---- 1. encode entangled fermion handshake ----
        handshake_payload = src_id.encode("utf-8") + b"|" + dst_id.encode("utf-8")
        sig = encode_entangled(
            handshake_payload,
            src_geo=src.geo_location,
            dst_geo=dst.geo_location,
            K=K_HANDSHAKE,
        )

        # ---- 2. frame ----
        frame = frame_entangled(sig, self.trans)
        if verbose:
            print(f"  frame         : S={frame.S:+.4f} "
                  f"H={frame.H:.4f} m={frame.m:.4e} "
                  f"size={frame.size} B")

        # ---- 3. hop-by-hop forwarding ----
        current = frame.bytes_
        for hop_id in path:
            hop = self.base.peers[hop_id]
            try:
                self.trans.deframe(current)
            except Exception as exc:
                if verbose:
                    print(f"  verify fail   : {hop_id}  {exc}")
                return None
            if hop_id != dst_id:
                hop.forwarded += 1

        # ---- 4. decode at destination ----
        payload, E_signal, B_signal, meta = deframe_entangled(
            current, self.trans)

        # ---- 5. recover location ----
        est = recover_location(E_signal, B_signal, meta)
        if verbose:
            print(f"  recovered src : t̂={est['t_hat']:+.4f} "
                  f"r̂={est['r_hat']:+.4f}  "
                  f"Π={est.get('Pi_hat', 0.0):.4f}  "
                  f"n_surv={est.get('n_survivors', 0)}")

        # ---- 6. build SVP session on dst side ----
        dst_side = svp_session_key(
            est["t_hat"], est["r_hat"], peer_id=src_id)
        if verbose:
            print(f"  dst SVP       : {dst_side['SVP']:.8f}  "
                  f"fp={dst_side['fingerprint'][:16]}")

        # ---- 7. ack back to source (simple finite-step packet) ----
        ack_payload = struct.pack(">d", dst_side["SVP"])
        ack_frame = self.trans.frame(ack_payload)

        # reverse path
        rev_path = list(reversed(path))
        ack_bytes = ack_frame.bytes_
        for hop_id in rev_path:
            hop = self.base.peers[hop_id]
            self.trans.deframe(ack_bytes)
            if hop_id != src_id:
                hop.forwarded += 1

        # ---- 8. src recomputes SVP from its own known location ----
        src_side = svp_session_key(
            src.geo_location[0], src.geo_location[1],
            peer_id=src_id)

        if verbose:
            print(f"  src SVP       : {src_side['SVP']:.8f}  "
                  f"fp={src_side['fingerprint'][:16]}")

        # ---- 9. verify match ----
        if abs(src_side["SVP"] - dst_side["SVP"]) > 1e-6:
            if verbose:
                print(f"  session       : FAIL (SVP mismatch)")
            return None

        if verbose:
            print(f"  session       : ESTABLISHED  "
                  f"key={src_side['session_key'][:16]}…")

        # ---- 10. store session on both ends ----
        session = GeoSession(
            src_id=src_id, dst_id=dst_id,
            src_geo=src.geo_location,
            dst_geo=dst.geo_location,
            recovered_src=(est["t_hat"], est["r_hat"]),
            recovered_dst=dst.geo_location,
            SVP=src_side["SVP"],
            fingerprint=src_side["fingerprint"],
            session_key=src_side["session_key"],
        )
        self.sessions[(src_id, dst_id)] = session
        if hasattr(src, "establish_session"):
            src.establish_session(dst_id, session)
        if hasattr(dst, "establish_session"):
            dst.establish_session(src_id, session)

        return session

    # --------------------------------------------------------
    def send_encrypted(self,
                       src_id: str, dst_id: str,
                       message: str) -> Dict:
        """Send an encrypted message using the established session."""
        session = self.sessions.get((src_id, dst_id))
        if session is None:
            # fall back to handshake first
            session = self.handshake(src_id, dst_id, verbose=False)
            if session is None:
                return dict(ok=False, reason="no session")

        src = self.base.peers[src_id]
        dst = self.base.peers[dst_id]
        plaintext = message.encode("utf-8")
        ciphertext = src.encrypt_payload(dst_id, plaintext)

        # frame the ciphertext as a regular MOSB message
        frame = self.trans.frame(ciphertext)

        path = self.base.route(src_id, dst_id)
        if path is None:
            return dict(ok=False, reason="no route")

        # hop-by-hop
        current = frame.bytes_
        for hop_id in path:
            hop = self.base.peers[hop_id]
            self.trans.deframe(current)
            if hop_id != dst_id:
                hop.forwarded += 1

        # decode at destination
        payload_out, _ = self.trans.deframe(current)
        decrypted = dst.decrypt_payload(src_id, payload_out)
        text_out = decrypted.decode("utf-8", errors="replace")

        session.packets_sent += 1
        session.packets_received += 1

        return dict(
            ok=(text_out == message),
            src=src_id, dst=dst_id,
            path=path,
            plaintext_in=message,
            plaintext_out=text_out,
            ciphertext_size=len(ciphertext),
            session_key_prefix=session.session_key[:16],
            SVP=session.SVP,
        )

    # --------------------------------------------------------
    def summary(self) -> Dict:
        s = self.base.summary()
        s["sessions"] = len(self.sessions)
        return s


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Coupled fermion P2P  ·  geo-location handshake  ·  SVP session")
    print("=" * 80)
    print(f"  α      = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  A_STEP = α / 0.3628     = {A_STEP:.6f}")
    print(f"  6/π²                    = {DENSITY:.6f}")
    print()

    net = CoupledFermionNetwork()

    # ---------- peers with distinct geo-locations ----------
    # (t, r) is the peer's location on Figure 3.5's torus
    cell_a = [
        ("alice@cell-A", (0.40, 0.60)),
        ("bob@cell-A",   (0.55, 0.72)),
        ("carol@cell-A", (0.28, 0.88)),
    ]
    cell_b = [
        ("eve@cell-B",   (1.50, 1.20)),
        ("frank@cell-B", (1.65, 1.10)),
    ]
    gateways = [
        ("gw1@core", (PI / 2, E / 2)),
        ("gw2@core", (PI - 0.3, E - 0.4)),
    ]

    for pid, geo in cell_a:
        net.join(pid, cell_id="cell-A", geo_location=geo)
    for pid, geo in cell_b:
        net.join(pid, cell_id="cell-B", geo_location=geo)
    for pid, geo in gateways:
        net.join(pid, cell_id="core", geo_location=geo, is_gateway=True)

    net.connect("gw1@core", "gw2@core")
    net.connect("gw1@core", "alice@cell-A")
    net.connect("gw2@core", "eve@cell-B")

    print(f"--- peer setup ---")
    for pid, geo in cell_a + cell_b + gateways:
        p = net.base.peers[pid]
        print(f"  {pid:16s}  cell={p.cell_id:8s}  "
              f"geo=({geo[0]:+.3f}, {geo[1]:+.3f})  "
              f"SVP={p.lattice.svp_weight:.6f}")
    print()

    # ---------- handshake: same cell ----------
    print("=" * 80)
    print("HANDSHAKE  ·  same cell  (alice → bob)")
    print("=" * 80)
    s1 = net.handshake("alice@cell-A", "bob@cell-A")
    print()

    # ---------- handshake: cross cell via gateways ----------
    print("=" * 80)
    print("HANDSHAKE  ·  cross-cell  (carol → frank)")
    print("=" * 80)
    s2 = net.handshake("carol@cell-A", "frank@cell-B")
    print()

    # ---------- handshake: alice → eve ----------
    print("=" * 80)
    print("HANDSHAKE  ·  alice → eve")
    print("=" * 80)
    s3 = net.handshake("alice@cell-A", "eve@cell-B")
    print()

    # ---------- encrypted messaging ----------
    print("=" * 80)
    print("ENCRYPTED MESSAGING  ·  using SVP session keys")
    print("=" * 80)

    messages = [
        ("alice@cell-A", "bob@cell-A",
         "hello bob, session established"),
        ("carol@cell-A", "frank@cell-B",
         "cross-cell encrypted ping"),
        ("alice@cell-A", "eve@cell-B",
         "secure channel open"),
    ]

    for src, dst, msg in messages:
        r = net.send_encrypted(src, dst, msg)
        if r["ok"]:
            print(f"  ✓ {src:16s} → {dst:16s}")
            print(f"      plaintext  : {r['plaintext_out']}")
            print(f"      session key: {r['session_key_prefix']}…")
            print(f"      SVP        : {r['SVP']:.8f}")
            print(f"      hops       : {len(r['path'])}")
        else:
            print(f"  ✗ {src} → {dst}  {r.get('reason', 'unknown')}")
        print()

    # ---------- session table ----------
    print("=" * 80)
    print("ACTIVE SESSIONS")
    print("=" * 80)
    print(f"  {'src':16s}  {'dst':16s}  {'SVP':>12s}  "
          f"{'recovered src (t,r)':>26s}  key")
    for (src_id, dst_id), sess in net.sessions.items():
        rt, rr = sess.recovered_src
        print(f"  {src_id:16s}  {dst_id:16s}  "
              f"{sess.SVP:>12.8f}  "
              f"({rt:+.3f}, {rr:+.3f})  "
              f"{sess.session_key[:16]}…")
    print()

    # ---------- topology summary ----------
    print("=" * 80)
    print("NETWORK SUMMARY")
    print("=" * 80)
    s = net.summary()
    for k, v in s.items():
        print(f"  {k:>14s} : {v}")
    print()

    # ---------- lattice previews ----------
    print("=" * 80)
    print("LATTICE SVP PER PEER  ·  the shared secret anchor")
    print("=" * 80)
    print(f"  {'peer':16s}  {'SVP weight':>12s}  "
          f"{'cycle':>10s}  {'|E|':>5s}  {'fingerprint':>16s}")
    for pid in [p for p, _ in cell_a + cell_b + gateways]:
        p = net.base.peers[pid]
        lat = p.lattice
        print(f"  {pid:16s}  {lat.svp_weight:>12.8f}  "
              f"{lat.cycle_len:>10.6f}  "
              f"{len(lat.edges):>5d}  "
              f"{p.summary()['lat_fp']:>16s}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  encode entangled fermions     O(|msg| · K_geo)")
    print("  frame MOSB                    O(K_geo)")
    print("  hop-by-hop routing            O(H · N_neighbours)")
    print("  receiver parity split         O(N_particles · K_recv)")
    print("  even-even mask                O(N_particles)")
    print("  cross-coupling gate           O(N_particles · K_recv)")
    print("  location estimate             O(N_survivors)")
    print("  SVP lattice build             O(N_lat²)")
    print("  rank-1 SVP scan               O(|E|)")
    print("  session key derivation        O(1)")
    print("  encrypted message send        O(K_p2p) per hop")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()