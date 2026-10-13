#!/usr/bin/env python3
"""
mobius_p2p_cellular_svp.py
==========================
Peer-to-peer Möbius internet for cellular connections.
Local topology is a graph lattice (shortest-vector-non-linear-graph.py).

  · Points in R^3 per peer
  · k-NN edges       = lattice vectors
  · Weight           w(i,j) = ‖p_i − p_j‖ / I
  · I                = ∫₀^(π+e) exp(−αx) dx,  α = 1/(π − e)
  · Rank-1 SVP       = shortest edge
  · Shortest cycle   = closed lattice vector
  · Reduced basis    = k shortest edges
  · Text payload     uses SAT envelope
  · Audio payload    uses finite_step_audio.MobiusAudioSampler
  · Every hop verifies the MOSB supertrace
"""

from __future__ import annotations

import hashlib
import heapq
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Optional sibling-module imports
# ============================================================
try:
    from hex_dns_mobius import MobiusHexGate  # type: ignore
    _HAS_DNS = True
except ImportError:
    _HAS_DNS = False

try:
    from signal_3d_sat_envelope import (      # type: ignore
        mobius_sieve_linear,
    )
    _HAS_SAT = True
except ImportError:
    _HAS_SAT = False

try:
    from finite_step_audio import (           # type: ignore
        MobiusAudioSampler, SamplingConfig,
        make_audio as audio_make_audio,
        snr as audio_snr,
    )
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)              # ≈ 2.362338
DENSITY = 6.0 / (PI * PI)             # ≈ 0.607927
A_STEP  = ALPHA / 0.3628              # ≈ 6.511
W1, W2  = PI, E


# ============================================================
#  Fallbacks
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
#  Layer 1  ·  Addressing
# ============================================================
class MobiusAddress:
    """Address = hex fingerprint of the peer id."""
    def __init__(self, K: int = 128):
        self.K = K
        if _HAS_DNS:
            self.gate = MobiusHexGate(K=K, logic_gate='log',
                                      use_elliptic=True)
        else:
            self.gate = None

    def fingerprint(self, peer_id: str) -> str:
        if self.gate is not None:
            return self.gate.fingerprint(peer_id)
        return hashlib.sha256(peer_id.encode()).hexdigest()

    def short(self, peer_id: str, n: int = 16) -> str:
        return self.fingerprint(peer_id)[:n]


# ============================================================
#  Layer 2  ·  Lattice topology
#  (shortest-vector-non-linear-graph.py)
# ============================================================
def _integral_g(alpha: float = ALPHA) -> float:
    """I = ∫₀^(π+e) exp(−αx) dx, computed by trapezoid on a fine grid."""
    x = np.linspace(0.0, PI + E, 8192)
    y = np.exp(-alpha * x)
    return float(np.trapezoid(y, x))


def _generate_lattice_points(N: int, seed: int,
                             spread: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((N, 3)) * spread


def _build_knn_edges(points: np.ndarray, k: int = 3
                     ) -> List[Tuple[int, int]]:
    N = points.shape[0]
    edges: set = set()
    for i in range(N):
        d = np.linalg.norm(points - points[i], axis=1)
        order = np.argsort(d)
        for j in order[1:k + 1]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    return sorted(edges)


def _shortest_edge(edges, weights):
    """Rank-1 SVP: the shortest edge in the lattice."""
    idx = int(np.argmin(weights))
    return edges[idx], float(weights[idx]), idx


def _shortest_cycle(points, edges, weights
                    ) -> Tuple[Optional[List[int]], float, Optional[int]]:
    """
    Closed lattice vector: for each edge (u, v) remove it and find
    the shortest u→v path in the remaining graph.
    """
    N = points.shape[0]
    adj: List[List[Tuple[int, float, int]]] = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edges):
        w = float(weights[idx])
        adj[i].append((j, w, idx))
        adj[j].append((i, w, idx))

    best_path: Optional[List[int]] = None
    best_len: float = float("inf")
    best_edge: Optional[int] = None

    for skip_idx, (u, v) in enumerate(edges):
        w_edge = float(weights[skip_idx])
        dist = {u: 0.0}
        pq = [(0.0, u, [u])]
        seen: set = set()
        found = False
        while pq:
            d, node, path = heapq.heappop(pq)
            if node in seen:
                continue
            seen.add(node)
            if node == v:
                total = w_edge + d
                if total < best_len:
                    best_path = path + [u]
                    best_len = total
                    best_edge = skip_idx
                found = True
                break
            for (nb, w, eidx) in adj[node]:
                if eidx == skip_idx:
                    continue
                nd = d + w
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb, path + [nb]))
            if found:
                break
    return best_path, best_len, best_edge


class LatticeTopology:
    """
    Per-peer lattice topology.

        points  = R^3 point cloud (peer's local event manifold)
        edges   = k-NN graph (lattice vectors)
        weights = ‖p_i − p_j‖ / I
        svp     = rank-1 shortest vector
        cycle   = shortest closed lattice vector

    Complexity:
        build points + k-NN edges   O(N²)     once per peer
        rank-1 SVP scan             O(E)
        shortest cycle              O(E · (V log V))
        all Dijkstra queries        O(E log V) per query
    """
    def __init__(self,
                 N: int = 60,
                 k_neighbors: int = 3,
                 seed: int = 42,
                 spread: float = 1.0):
        self.N = N
        self.k = k_neighbors
        self.seed = seed
        self.spread = spread

        # ---- point cloud ----
        self.points = _generate_lattice_points(N, seed=seed, spread=spread)

        # ---- k-NN edges ----
        self.edges = _build_knn_edges(self.points, k=k_neighbors)

        # ---- weights scaled by the normalisation integral ----
        self.I = _integral_g()
        raw = np.array([float(np.linalg.norm(self.points[i]
                                             - self.points[j]))
                        for i, j in self.edges])
        self.raw = raw
        self.weights = raw / self.I

        # ---- rank-1 SVP ----
        self.svp_edge, self.svp_weight, self.svp_idx = \
            _shortest_edge(self.edges, self.weights)

        # ---- shortest cycle ----
        (self.cycle_path,
         self.cycle_len,
         self.cycle_edge) = _shortest_cycle(
            self.points, self.edges, self.weights)

        # ---- adjacency for Dijkstra ----
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
        for idx, (i, j) in enumerate(self.edges):
            w = float(self.weights[idx])
            self.adj[i].append((j, w))
            self.adj[j].append((i, w))

    # --------------------------------------------------------
    def hop_weight(self, i: int, j: int) -> float:
        """Lattice weight of edge (i, j) or inf if not adjacent."""
        for (nb, w) in self.adj[i]:
            if nb == j:
                return w
        return float("inf")

    # --------------------------------------------------------
    def shortest_path(self, src: int, dst: int
                      ) -> Tuple[Optional[List[int]], float]:
        """Dijkstra on the lattice from src to dst."""
        if src == dst:
            return [src], 0.0
        dist = {src: 0.0}
        pq = [(0.0, src, [src])]
        seen: set = set()
        while pq:
            d, node, path = heapq.heappop(pq)
            if node in seen:
                continue
            seen.add(node)
            if node == dst:
                return path, d
            for (nb, w) in self.adj[node]:
                nd = d + w
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb, path + [nb]))
        return None, float("inf")

    # --------------------------------------------------------
    def reduced_basis(self, k: int = 4
                      ) -> List[Tuple[Tuple[int, int], float]]:
        """k shortest edges = a reduced basis of the lattice."""
        order = np.argsort(self.weights)[:k]
        return [(self.edges[i], float(self.weights[i])) for i in order]

    # --------------------------------------------------------
    def fingerprint(self) -> str:
        """Deterministic fingerprint of the lattice."""
        h = hashlib.sha256()
        h.update(self.points.tobytes())
        for (i, j) in self.edges:
            h.update(struct.pack('>II', i, j))
        h.update(struct.pack('>d', self.svp_weight))
        h.update(struct.pack('>d', self.cycle_len if self.cycle_len
                              < 1e12 else 0.0))
        return h.hexdigest()[:32]

    # --------------------------------------------------------
    def summary(self) -> Dict:
        return dict(
            N=self.N,
            k=self.k,
            n_edges=len(self.edges),
            I=self.I,
            svp_edge=self.svp_edge,
            svp_weight=self.svp_weight,
            cycle_len=self.cycle_len,
            cycle_len_edges=(len(self.cycle_path) - 1
                             if self.cycle_path else 0),
            fingerprint=self.fingerprint()[:16],
        )


# ============================================================
#  Layer 3  ·  SAT envelope  (text payload)
# ============================================================
class SATSignal:
    def __init__(self, K: int = 128):
        self.K = K
        if _HAS_SAT:
            try:
                self.mu = mobius_sieve_linear(K)
            except Exception:
                self.mu = _fallback_mobius(K)
        else:
            self.mu = _fallback_mobius(K)

    def from_text(self, text: str) -> np.ndarray:
        raw = text.encode("utf-8")
        amps = np.array([b / 255.0 for b in raw])
        if len(amps) < self.K:
            amps = np.tile(amps,
                           int(np.ceil(self.K / max(len(amps), 1))))
        amps = amps[:self.K]
        env = np.zeros(self.K, dtype=float)
        for a, v in enumerate(amps):
            q = (a * a + a) % self.K
            if self.mu[q] != 0:
                env[q] += abs(v)
        return env


# ============================================================
#  Layer 4  ·  Finite-step packet  (lossless)
# ============================================================
class FiniteStepPacket:
    a_step = A_STEP

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


# ============================================================
#  Layer 5  ·  MOSB framing
# ============================================================
@dataclass
class Frame:
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
    MAGIC = b"MOSB"
    FMT   = ">4sIQQddd"
    SIZE  = struct.calcsize(FMT)

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
        header = struct.pack(self.FMT, self.MAGIC, self.K,
                             n, cid, S, H, m)
        raw = header + payload
        chk = hashlib.sha256(raw).hexdigest()[:16]
        return Frame(cid, payload, S, H, m, raw, chk)

    def deframe(self, raw: bytes) -> Tuple[bytes, Frame]:
        if len(raw) < self.SIZE:
            raise ValueError("frame too short")
        magic, K, n, cid, S, H, m = struct.unpack(
            self.FMT, raw[:self.SIZE])
        if magic != self.MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        payload = raw[self.SIZE:self.SIZE + n]
        if len(payload) != n:
            raise ValueError(
                f"payload length mismatch: {len(payload)} != {n}")
        S2 = self._supertrace(payload)
        if abs(S2 - S) > 1e-9:
            raise ValueError(
                f"supertrace mismatch: {S2} != {S}")
        chk = hashlib.sha256(raw).hexdigest()[:16]
        return payload, Frame(cid, payload, S, H, m, raw, chk)


# ============================================================
#  Audio payload  (finite_step_audio.MobiusAudioSampler)
# ============================================================
class AudioPayload:
    def __init__(self, K: int = 1024, fs: int = 44_100, bps: int = 16):
        if _HAS_AUDIO:
            self.sampler = MobiusAudioSampler(
                SamplingConfig(fs=fs, bps=bps, K=K,
                               use_mobius=True, use_throughput=True))
        else:
            self.sampler = None
        self.K = K
        self.fs = fs

    def encode(self, audio: np.ndarray) -> np.ndarray:
        if self.sampler is None:
            idx = np.linspace(0, len(audio) - 1, self.K)
            return np.interp(idx, np.arange(len(audio)), audio)
        res = self.sampler.sample(audio)
        return np.concatenate([[res["first_sample"]], res["samples"]])

    def decode(self, signal: np.ndarray) -> np.ndarray:
        if self.sampler is None:
            return signal
        first_sample = float(signal[0])
        samples = signal[1:]
        return self.sampler.reconstruct({
            "samples": samples,
            "first_sample": first_sample,
        })

    def bit_rate(self) -> float:
        if self.sampler is None:
            return self.fs * 16
        return self.sampler.bit_rate()


# ============================================================
#  P2P peer  ·  owns a LatticeTopology
# ============================================================
class MobiusPeer:
    """
    One node in the cellular P2P network.

    Local routing primitive = the peer's lattice SVP.
        · svp_edge     shortest lattice vector (routing neighbour)
        · cycle_path   closed lattice vector (loop)
        · reduced      k shortest edges  (basis)
    """
    def __init__(self,
                 peer_id: str,
                 cell_id: str = "cell-default",
                 is_gateway: bool = False,
                 K_lattice: int = 60,
                 K_lattice_k: int = 3,
                 K_sat: int = 128,
                 K_pack: int = 512,
                 K_audio: int = 1024):
        self.peer_id = peer_id
        self.cell_id = cell_id
        self.is_gateway = is_gateway
        self.online = True

        # ---- addressing ----
        self.addr_encoder = MobiusAddress(K=K_sat)
        self.address = self.addr_encoder.fingerprint(peer_id)
        self.short_address = self.address[:16]

        # ---- lattice topology (replaces LocalTopology) ----
        seed = int(hashlib.sha256(peer_id.encode())
                   .hexdigest()[:8], 16)
        self.lattice = LatticeTopology(
            N=K_lattice, k_neighbors=K_lattice_k, seed=seed)

        # ---- payload codecs ----
        self.sat = SATSignal(K=K_sat)
        self.packet = FiniteStepPacket()
        self.trans = MobiusTransport(K=K_pack)
        self.audio = AudioPayload(K=K_audio)

        # ---- P2P links ----
        self.neighbours: Dict[str, "MobiusPeer"] = {}

        # ---- stats ----
        self.received: List[Dict] = []
        self.forwarded: int = 0

    # --------------------------------------------------------
    def connect(self, other: "MobiusPeer") -> None:
        if other.peer_id == self.peer_id:
            return
        self.neighbours[other.peer_id] = other
        other.neighbours[self.peer_id] = self

    def disconnect(self, other_id: str) -> None:
        self.neighbours.pop(other_id, None)

    # --------------------------------------------------------
    def xor_distance(self, other_address: str) -> int:
        return int(self.address, 16) ^ int(other_address, 16)

    # --------------------------------------------------------
    def local_svp_weight(self) -> float:
        """Rank-1 SVP weight of the peer's lattice."""
        return self.lattice.svp_weight

    # --------------------------------------------------------
    def summary(self) -> Dict:
        lat = self.lattice.summary()
        return dict(
            peer_id=self.peer_id,
            cell_id=self.cell_id,
            address=self.short_address,
            is_gateway=self.is_gateway,
            online=self.online,
            n_neighbours=len(self.neighbours),
            n_received=len(self.received),
            n_forwarded=self.forwarded,
            # lattice stats
            lat_N=lat["N"],
            lat_edges=lat["n_edges"],
            lat_svp=lat["svp_weight"],
            lat_cycle=lat["cycle_len"],
            lat_fp=lat["fingerprint"],
        )


# ============================================================
#  P2P network
# ============================================================
class MobiusP2PNetwork:
    def __init__(self):
        self.peers: Dict[str, MobiusPeer] = {}
        self.events: List[Dict] = []

    # --------------------------------------------------------
    def join(self, peer_id: str,
             cell_id: str = "cell-default",
             is_gateway: bool = False,
             **kwargs) -> MobiusPeer:
        p = MobiusPeer(peer_id, cell_id=cell_id,
                       is_gateway=is_gateway, **kwargs)
        self.peers[peer_id] = p
        self.events.append(dict(t=time.time(), kind="join",
                                peer=peer_id, cell=cell_id))
        for other_id, other in self.peers.items():
            if other_id == peer_id:
                continue
            if other.cell_id == cell_id and other.online:
                p.connect(other)
        if is_gateway:
            for other_id, other in self.peers.items():
                if other_id == peer_id:
                    continue
                if other.is_gateway and other.online:
                    p.connect(other)
        return p

    # --------------------------------------------------------
    def connect(self, id1: str, id2: str) -> None:
        p1, p2 = self.peers.get(id1), self.peers.get(id2)
        if p1 and p2:
            p1.connect(p2)

    # --------------------------------------------------------
    def leave(self, peer_id: str) -> None:
        p = self.peers.get(peer_id)
        if p is None:
            return
        p.online = False
        for n_id in list(p.neighbours.keys()):
            n = self.peers.get(n_id)
            if n:
                n.disconnect(peer_id)
        p.neighbours.clear()
        self.events.append(dict(t=time.time(), kind="leave",
                                peer=peer_id, cell=p.cell_id))

    # --------------------------------------------------------
    def rejoin(self, peer_id: str) -> None:
        p = self.peers.get(peer_id)
        if p is None:
            return
        p.online = True
        for other_id, other in self.peers.items():
            if other_id == peer_id:
                continue
            if other.online and (other.cell_id == p.cell_id
                                 or (p.is_gateway and other.is_gateway)):
                p.connect(other)
        self.events.append(dict(t=time.time(), kind="rejoin",
                                peer=peer_id, cell=p.cell_id))

    # --------------------------------------------------------
    def handoff(self, peer_id: str, new_cell_id: str) -> None:
        p = self.peers.get(peer_id)
        if p is None:
            return
        old_cell = p.cell_id
        for n_id in list(p.neighbours.keys()):
            n = self.peers.get(n_id)
            if n and n.cell_id != new_cell_id:
                p.disconnect(n_id)
                n.disconnect(peer_id)
        p.cell_id = new_cell_id
        new_neighbours = [other for o_id, other in self.peers.items()
                          if o_id != peer_id
                          and other.cell_id == new_cell_id
                          and other.online]
        for other in new_neighbours[:3]:
            p.connect(other)
        self.events.append(dict(t=time.time(), kind="handoff",
                                peer=peer_id,
                                from_cell=old_cell,
                                to_cell=new_cell_id))

    # --------------------------------------------------------
    def route(self, src_id: str, dst_id: str,
              use_lattice: bool = True) -> Optional[List[str]]:
        """
        Greedy Kademlia-style routing.

        Tie-break between equidistant candidates using the local
        lattice SVP weight: the peer with the smaller SVP runs on a
        tighter lattice, so it is a shorter hop in the internal
        metric.
        """
        src = self.peers.get(src_id)
        dst = self.peers.get(dst_id)
        if src is None or dst is None:
            return None
        if not (src.online and dst.online):
            return None
        if src_id == dst_id:
            return [src_id]

        path = [src_id]
        visited = {src_id}
        current = src
        for _ in range(32):
            if current.peer_id == dst_id:
                return path
            cands = []
            for n in current.neighbours.values():
                if not n.online or n.peer_id in visited:
                    continue
                xd = n.xor_distance(dst.address)
                svp = n.local_svp_weight() if use_lattice else 0.0
                cands.append((n, xd, svp))
            if not cands:
                return None
            # primary: XOR distance; tie-break: lattice SVP weight
            nxt, _, _ = min(cands, key=lambda x: (x[1], x[2]))
            path.append(nxt.peer_id)
            visited.add(nxt.peer_id)
            current = nxt
        return None if current.peer_id != dst_id else path

    # --------------------------------------------------------
    def send(self,
             src_id: str, dst_id: str,
             message: str = "",
             kind: str = "text",
             audio: Optional[np.ndarray] = None,
             geo_seed: int = 2024) -> Dict:
        t0 = time.perf_counter()

        src = self.peers.get(src_id)
        dst = self.peers.get(dst_id)
        if src is None or dst is None:
            return dict(ok=False, reason="unknown peer")

        # ---- 1. route ----
        path = self.route(src_id, dst_id)
        if path is None:
            return dict(ok=False, reason="no route",
                        src=src_id, dst=dst_id)

        # ---- 2. encode ----
        if kind == "audio":
            if audio is None:
                return dict(ok=False, reason="audio payload missing")
            signal = src.audio.encode(audio)
        else:
            signal = src.sat.from_text(message)

        # ---- 3. finite-step packet ----
        pk = src.packet.encode(signal)
        payload = struct.pack(f">{len(pk)}d", *pk)
        sha_in = hashlib.sha256(payload).hexdigest()

        # ---- 4. MOSB frame ----
        frame = src.trans.frame(payload)

        # ---- 5. hop-by-hop forwarding with per-hop lattice cost ----
        hop_trace: List[Dict] = []
        lattice_cost = 0.0
        current_bytes = frame.bytes_
        for hop_id in path:
            hop = self.peers[hop_id]
            try:
                _, recv_frame = hop.trans.deframe(current_bytes)
            except Exception as exc:
                return dict(ok=False,
                            reason=f"verify failed at {hop_id}: {exc}")
            hop_trace.append(dict(
                hop=hop_id,
                S=recv_frame.S,
                size=recv_frame.size,
                lattice_svp=hop.lattice.svp_weight,
                lattice_cycle=hop.lattice.cycle_len,
            ))
            lattice_cost += hop.lattice.svp_weight
            if hop_id != dst_id:
                hop.forwarded += 1

        # ---- 6. decode at destination ----
        payload_out, recv_frame = dst.trans.deframe(current_bytes)
        sha_out = hashlib.sha256(payload_out).hexdigest()
        ok = (sha_in == sha_out)

        n_floats = len(payload_out) // 8
        pk_out = list(struct.unpack(f">{n_floats}d", payload_out))
        signal_out = dst.packet.decode(pk_out)
        decoded = (dst.audio.decode(signal_out)
                   if kind == "audio" else signal_out)

        dst.received.append(dict(
            t=time.time(), src=src_id, kind=kind,
            bytes=len(payload_out), S=recv_frame.S,
        ))

        elapsed = (time.perf_counter() - t0) * 1e3
        return dict(
            ok=ok, kind=kind,
            src=src_id, dst=dst_id,
            path=path, hops=len(path) - 1,
            sha_in=sha_in, sha_out=sha_out,
            frame_size=frame.size,
            frame_S=frame.S, frame_H=frame.H, frame_m=frame.m,
            decoded=decoded,
            hop_trace=hop_trace,
            lattice_cost=lattice_cost,
            elapsed_ms=elapsed,
        )

    # --------------------------------------------------------
    def summary(self) -> Dict:
        online = sum(1 for p in self.peers.values() if p.online)
        gateways = sum(1 for p in self.peers.values() if p.is_gateway)
        cells = set(p.cell_id for p in self.peers.values())
        return dict(
            n_peers=len(self.peers),
            n_online=online,
            n_gateways=gateways,
            n_cells=len(cells),
            cells=sorted(cells),
            n_events=len(self.events),
        )


# ============================================================
#  Visualisation  ·  one peer's lattice
# ============================================================
def plot_peer_lattice(peer: MobiusPeer, ax=None):
    """Draw the peer's lattice: points, edges, SVP, shortest cycle."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    lat = peer.lattice
    pts = lat.points

    # points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c="#3a7bd5", s=25, depthshade=False)

    # edges colored by weight
    w = lat.weights
    wmin, wmax = float(w.min()), float(w.max())
    span = (wmax - wmin) + 1e-12
    for idx, (i, j) in enumerate(lat.edges):
        p1, p2 = pts[i], pts[j]
        norm = (w[idx] - wmin) / span
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=plt.cm.viridis(1.0 - norm),
                alpha=0.55, linewidth=1.0)

    # SVP edge
    i, j = lat.svp_edge
    p1, p2 = pts[i], pts[j]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color="#e74c3c", linewidth=3.5,
            label=f"SVP = {lat.svp_weight:.4f}")
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
               color="#e74c3c", s=90, edgecolor="k", zorder=5)

    # shortest cycle
    if lat.cycle_path is not None:
        cp = lat.cycle_path
        for a, b in zip(cp[:-1], cp[1:]):
            p1, p2 = pts[a], pts[b]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="#f1c40f", linewidth=2.2, alpha=0.9)
        ax.scatter(pts[cp[:-1], 0], pts[cp[:-1], 1], pts[cp[:-1], 2],
                   color="#f1c40f", s=70, edgecolor="k", zorder=6,
                   label=f"cycle = {lat.cycle_len:.4f}")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"{peer.peer_id}  ·  cell={peer.cell_id}\n"
                 f"|E|={len(lat.edges)}   I={lat.I:.4f}   "
                 f"fp={peer.summary()['lat_fp']}")
    ax.legend(fontsize=8, loc="upper left")
    return ax


# ============================================================
#  Demo
# ============================================================
def _make_audio(K: int = 1024, fs: int = 44_100) -> np.ndarray:
    if _HAS_AUDIO:
        try:
            return audio_make_audio(K, fs)
        except Exception:
            pass
    t = np.arange(K) / fs
    s = (0.6 * np.sin(2 * np.pi * 440.0 * t)
         + 0.3 * np.sin(2 * np.pi * 880.0 * t + 0.7))
    env = 0.5 * (1.0 - np.cos(2 * np.pi * t / (K / fs)))
    return s * env


def _snr(a: np.ndarray, b: np.ndarray) -> float:
    if _HAS_AUDIO:
        try:
            return audio_snr(a, b)
        except Exception:
            pass
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    err = a - b
    num = float(np.sum(a ** 2))
    den = float(np.sum(err ** 2)) + 1e-20
    return 10.0 * math.log10(num / den)


def demo():
    print("=" * 80)
    print("Möbius P2P  ·  lattice topology (SVP)  ·  cellular  ·  audio")
    print("=" * 80)
    print(f"  α = 1/(π − e)       = {ALPHA:.6f}")
    print(f"  a_step = α/0.3628   = {A_STEP:.6f}")
    print(f"  6/π²                = {DENSITY:.6f}")
    print(f"  integral I          = {_integral_g():.6f}")
    print(f"  audio module        = {'loaded' if _HAS_AUDIO else 'fallback'}")
    print()

    net = MobiusP2PNetwork()

    # ---- peers across two cells + 2 gateways ----
    cell_a = ["alice@cell-A", "bob@cell-A",
              "carol@cell-A", "dave@cell-A"]
    cell_b = ["eve@cell-B", "frank@cell-B",
              "grace@cell-B", "heidi@cell-B"]
    gateways = ["gw1@core", "gw2@core"]

    for pid in cell_a:
        net.join(pid, cell_id="cell-A")
    for pid in cell_b:
        net.join(pid, cell_id="cell-B")
    for pid in gateways:
        net.join(pid, cell_id="core", is_gateway=True)
    net.connect(gateways[0], gateways[1])
    net.connect(gateways[0], "alice@cell-A")
    net.connect(gateways[1], "eve@cell-B")

    s = net.summary()
    print(f"--- peer setup ---")
    print(f"  peers {s['n_peers']}  online {s['n_online']}  "
          f"gateways {s['n_gateways']}  cells {s['cells']}")
    print()

    # ---- lattice overview per peer ----
    print("--- per-peer lattice summary ---")
    print(f"  {'peer':20s}  {'cell':8s}  {'|E|':>4s}  "
          f"{'SVP':>10s}  {'cycle':>10s}  {'fp':>16s}")
    for pid in cell_a + cell_b + gateways:
        p = net.peers[pid]
        ps = p.summary()
        print(f"  {pid:20s}  {p.cell_id:8s}  "
              f"{ps['lat_edges']:>4d}  "
              f"{ps['lat_svp']:>10.6f}  "
              f"{ps['lat_cycle']:>10.6f}  "
              f"{ps['lat_fp']:>16s}")
    print()

    # ---- text within a cell ----
    print("--- text: alice → bob (same cell) ---")
    r = net.send("alice@cell-A", "bob@cell-A",
                 message="hello bob, same-cell ping", kind="text")
    print(f"  ok     {r['ok']}   hops {r['hops']}   "
          f"path {' → '.join(r['path'])}")
    print(f"  frame  S {r['frame_S']:+.4f}   size {r['frame_size']} B   "
          f"lattice cost {r['lattice_cost']:.4f}")
    print()

    # ---- text across cells ----
    print("--- text: carol → grace (cross-cell) ---")
    r = net.send("carol@cell-A", "grace@cell-B",
                 message="cross-cell ping", kind="text")
    print(f"  ok     {r['ok']}   hops {r['hops']}")
    print(f"  path   {' → '.join(r['path'])}")
    print(f"  lattice cost {r['lattice_cost']:.4f}")
    print(f"  per-hop lattice SVP:")
    for h in r["hop_trace"]:
        print(f"    {h['hop']:20s}  "
              f"S={h['S']:+.4f}  "
              f"svp={h['lattice_svp']:.6f}  "
              f"cycle={h['lattice_cycle']:.6f}")
    print()

    # ---- audio within a cell ----
    print("--- audio: alice → dave (same cell) ---")
    audio = _make_audio(1024, 44_100)
    r_audio = net.send("alice@cell-A", "dave@cell-A",
                       kind="audio", audio=audio)
    dec = r_audio["decoded"]
    n = min(len(audio), len(dec))
    snr_in = _snr(audio[:n], dec[:n])
    print(f"  ok     {r_audio['ok']}   hops {r_audio['hops']}")
    print(f"  frame  S {r_audio['frame_S']:+.4f}  "
          f"size {r_audio['frame_size']} B")
    print(f"  audio  K_in={len(audio)} K_out={len(dec)}  "
          f"SNR={snr_in:.2f} dB")
    print(f"  lattice cost {r_audio['lattice_cost']:.4f}")
    print()

    # ---- audio across cells ----
    print("--- audio: bob → frank (cross-cell) ---")
    r2 = net.send("bob@cell-A", "frank@cell-B",
                  kind="audio", audio=audio)
    dec2 = r2["decoded"]
    n2 = min(len(audio), len(dec2))
    snr2 = _snr(audio[:n2], dec2[:n2])
    print(f"  ok     {r2['ok']}   hops {r2['hops']}")
    print(f"  path   {' → '.join(r2['path'])}")
    print(f"  audio  SNR={snr2:.2f} dB   lattice cost {r2['lattice_cost']:.4f}")
    print()

    # ---- churn ----
    print("--- churn: bob leaves / rejoins ---")
    net.leave("bob@cell-A")
    r_fail = net.send("alice@cell-A", "bob@cell-A", message="anyone?")
    print(f"  send while offline  ok={r_fail['ok']}  "
          f"reason={r_fail.get('reason')}")
    net.rejoin("bob@cell-A")
    r_ok = net.send("alice@cell-A", "bob@cell-A", message="welcome back")
    print(f"  send after rejoin   ok={r_ok['ok']}  hops={r_ok['hops']}")
    print()

    # ---- handoff ----
    print("--- handoff: carol@cell-A → cell-B ---")
    net.handoff("carol@cell-A", "cell-B")
    carol = net.peers["carol@cell-A"]
    print(f"  carol cell = {carol.cell_id}  nbrs={len(carol.neighbours)}")
    r_handoff = net.send("carol@cell-A", "heidi@cell-B",
                         message="handoff complete")
    print(f"  send  ok={r_handoff['ok']}  "
          f"path={' → '.join(r_handoff['path'])}")
    print()

    # ---- statistics ----
    s = net.summary()
    total_recv = sum(len(p.received) for p in net.peers.values())
    total_fwd = sum(p.forwarded for p in net.peers.values())
    print("--- statistics ---")
    print(f"  peers         {s['n_peers']}   online {s['n_online']}")
    print(f"  cells         {s['cells']}")
    print(f"  events        {s['n_events']}")
    print(f"  delivered     {total_recv}")
    print(f"  forwarded     {total_fwd}")
    print()

    # ---- complexity ----
    print("--- complexity ---")
    print("  lattice build (points + k-NN)    O(N²)        once per peer")
    print("  rank-1 SVP                       O(|E|)       once")
    print("  shortest cycle                   O(|E| · V log V)")
    print("  per-hop Dijkstra                 O(|E| log V)")
    print("  hex-DNS fingerprint              O(K log log K) + O(K log K)")
    print("  SAT envelope (text)              O(K)")
    print("  audio sampler (K=1024)           O(K)         per frame")
    print("  finite-step packet               O(K)")
    print("  MOSB frame + verify              O(n)         per hop")
    print("  routing (greedy XOR)             O(H · N)")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(16, 10))
        # row 1: three peer lattices
        for i, pid in enumerate(["alice@cell-A",
                                 "eve@cell-B",
                                 "gw1@core"]):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            plot_peer_lattice(net.peers[pid], ax=ax)

        # row 2: hop cost trace for the cross-cell audio
        ax = fig.add_subplot(2, 3, 4)
        hops = [h["hop"] for h in r2["hop_trace"]]
        svps = [h["lattice_svp"] for h in r2["hop_trace"]]
        cycles = [h["lattice_cycle"] for h in r2["hop_trace"]]
        ax.plot(range(len(hops)), svps, "o-", color="#e74c3c",
                label="lattice SVP")
        ax.plot(range(len(hops)), cycles, "s--", color="#f1c40f",
                label="lattice cycle")
        ax.set_xticks(range(len(hops)))
        ax.set_xticklabels([h.split("@")[0] for h in hops],
                           rotation=30, fontsize=8)
        ax.set_ylabel("lattice weight")
        ax.set_title("Per-hop lattice cost (bob → frank)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # row 2: bit rate at different frame sizes
        ax = fig.add_subplot(2, 3, 5)
        if _HAS_AUDIO:
            Ks = [128, 256, 512, 1024, 2048]
            br = []
            for K_ in Ks:
                ap = AudioPayload(K=K_, fs=44_100, bps=16)
                br.append(ap.bit_rate() / 1e3)
            ax.plot(Ks, br, "o-", color="#16a085")
            ax.axhline(44_100 * 16 / 1e3, color="k", ls=":",
                       label="uniform 16 bps")
            ax.set_xscale("log")
            ax.set_xlabel("K"); ax.set_ylabel("B (kbps)")
            ax.set_title("Audio bit rate vs K  (Möbius + wire)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, which="both")

        # row 2: peer status bar
        ax = fig.add_subplot(2, 3, 6)
        peers = list(net.peers.keys())
        svps = [net.peers[p].lattice.svp_weight for p in peers]
        colors = ["#e74c3c" if net.peers[p].is_gateway
                  else ("#3498db" if net.peers[p].cell_id == "cell-A"
                        else "#16a085")
                  for p in peers]
        ax.barh(range(len(peers)), svps, color=colors, edgecolor="k")
        ax.set_yticks(range(len(peers)))
        ax.set_yticklabels([p.split("@")[0] for p in peers],
                           fontsize=8)
        ax.set_xlabel("SVP weight")
        ax.set_title("Per-peer lattice SVP  (blue=cell-A, "
                     "green=cell-B, red=gateway)")
        ax.grid(alpha=0.3, axis="x")

        plt.suptitle(
            "Möbius P2P  ·  per-peer lattice (SVP)  ·  "
            "cellular  ·  finite-step audio",
            fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    except Exception as exc:
        print(f"  (plot skipped: {exc})")

    print("Done.")


if __name__ == "__main__":
    demo()