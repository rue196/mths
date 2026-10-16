#!/usr/bin/env python3
"""
ugc_reasoning_net.py
====================

A neural network whose
  · nodes   are UGC discrete-log units        (ugc_discrete_log.py)
  · weights are split-channel spinor blocks   (n_k_decoder_div_weights_compression.py)
  · read path uses the MOSB read-only buffer  (server_nodes_dns_buffer_read_only.py)
  · reasoning layer is the ALU chain          (ALU_CPU_chain_of_reasoning_inference.py)

Symmetry rule
-------------
    W = Wᵀ   (symmetric)  →  bypass buffer.  UGC node solves directly
                             via discrete log on the symmetric graph.
    W ≠ Wᵀ   (asymmetric) →  route through the read-only MOSB buffer.
                             Look up weight block by chunk id, decode,
                             then feed to the UGC node.
"""

from __future__ import annotations

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
ALPHA    = 1.0 / (PI - E)              # ≈ 2.362338
GAMMA    = 0.5772156649015329
DENSITY  = 6.0 / (PI * PI)             # ≈ 0.607927
BLOCK    = 6
N_PERM   = math.factorial(BLOCK)       # 720
A_STEP   = ALPHA / 0.3628              # ≈ 6.511


# ============================================================
#  Primitives
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8)
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


def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


def mu_convolution_H(K: int) -> np.ndarray:
    mu = mobius_sieve(K)
    H = harmonic_numbers(K)
    F = np.zeros(K + 1, dtype=float)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = int(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F


# ============================================================
#  UGC discrete-log node
# ============================================================
class UGCNode:
    """
    A single neural node whose nonlinearity is a UGC discrete log.

    · carries its own prime p and primitive root g
    · activation:  x  →  (exponent, −H_{exponent+1})
    · read path (asymmetric): reads a chunk from the MOSB buffer
    """

    _PRIMES = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
               67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    def __init__(self, node_id: int, K_harmonic: int = 2048):
        self.node_id = int(node_id)
        self.p = self._PRIMES[self.node_id % len(self._PRIMES)]
        self.g = self._find_primitive_root(self.p)
        self.n = self.p - 1
        self.H = harmonic_numbers(max(K_harmonic, self.n + 2))
        self.powers = self._build_powers()
        self.log_table = {int(v): i for i, v in enumerate(self.powers)}

    # ----- primitive root -----
    @staticmethod
    def _find_primitive_root(p: int) -> int:
        if p == 2:
            return 1
        phi = p - 1
        factors, x, d = set(), phi, 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d); x //= d
            d += 1
        if x > 1:
            factors.add(x)
        for g in range(2, p):
            if all(pow(g, phi // q, p) != 1 for q in factors):
                return g
        raise ValueError(f"no primitive root for p={p}")

    def _build_powers(self) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.int64)
        cur = 1
        for i in range(self.n):
            out[i] = cur
            cur = (cur * self.g) % self.p
        return out

    # ----- activation:  dlog as the nonlinearity -----
    def activate(self, x: float) -> Tuple[int, float]:
        h = int(round(abs(x))) % self.p
        if h == 0:
            h = 1
        if h not in self.log_table:
            # snap to the nearest available residue
            best, bd = None, 1 << 30
            for v, i in self.log_table.items():
                d = abs(v - h)
                if d < bd:
                    bd, best = d, i
            exp_idx = best
        else:
            exp_idx = self.log_table[h]
        log_val = -float(self.H[exp_idx + 1])
        if x < 0:
            log_val = -abs(log_val)
        return exp_idx, log_val

    # ----- read path (asymmetric): pull from MOSB buffer -----
    def read_via_buffer(self, buf: "MOSBReadOnlyBuffer",
                        cid: int) -> Optional[float]:
        chunk = buf.read_chunk(cid)
        if chunk is None:
            return None
        payload, *_ = chunk
        if len(payload) < 8:
            return None
        return struct.unpack(">d", payload[:8])[0]


# ============================================================
#  Split-channel spinor weight codec  (64 bits per weight)
# ============================================================
def rank_permutation(perm) -> int:
    n = len(perm); code = 0
    for i in range(n):
        smaller = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
        code += smaller * math.factorial(n - 1 - i)
    return code


def unrank_permutation(code: int, n: int) -> Tuple[int, ...]:
    code = int(code) % math.factorial(n)
    avail = list(range(n)); perm = []
    for i in range(n):
        f = math.factorial(n - 1 - i)
        idx = code // f; code = code % f
        perm.append(avail.pop(idx))
    return tuple(perm)


def pack_real(sign: int, n_idx: int, mag_frac: int) -> int:
    return ((sign & 1) << 31) | ((n_idx & 0xFFFFFF) << 7) | (mag_frac & 0x7F)


def unpack_real(w: int) -> Tuple[int, int, int]:
    return ((w >> 31) & 1, (w >> 7) & 0xFFFFFF, w & 0x7F)


def pack_imag(sign: int, perm_idx: int, F_q: int, gate_q: int) -> int:
    return ((sign & 1) << 31) | ((perm_idx & 0x3FF) << 21) | \
           ((F_q & 0xFFF) << 9) | (gate_q & 0x1FF)


def unpack_imag(w: int) -> Tuple[int, int, int, int]:
    return ((w >> 31) & 1, (w >> 21) & 0x3FF,
            (w >> 9) & 0xFFF, w & 0x1FF)


class SplitChannelCodec:
    """
    Weight codec:

        real half  (32 bits)  =  sign | H-index (24b) | fraction (7b)
        imag half  (32 bits)  =  sign | Lehmer σ* (10b) | F-quant (12b) | gate (9b)
    """
    def __init__(self, K: int = 4096):
        self.K = K
        self.H = harmonic_numbers(K)
        self.F = mu_convolution_H(K)
        self.mu = mobius_sieve(K)
        self.H_max = float(self.H[K])
        Fabs = np.abs(self.F[1:K + 1])
        self.F_max = float(Fabs.max()) if Fabs.size else 1.0

    def encode_scalar(self, v: float) -> Tuple[int, int]:
        sign = 1 if v < 0 else 0
        a = abs(v)
        n_idx = self._harmonic_index(a)
        h_frac = self._frac(a, n_idx)
        w_re = pack_real(sign, n_idx, h_frac)

        F_q = int(min(0xFFF, max(0, round(a * 0xFFF))))
        perm_idx = rank_permutation(tuple(range(BLOCK))) % N_PERM
        gate_q = F_q & 0x1FF
        w_im = pack_imag(sign, perm_idx, F_q, gate_q)
        return w_re, w_im

    def decode_scalar(self, w_re: int, w_im: int) -> float:
        s_re, n_re, h_frac = unpack_real(w_re)
        _, _, F_q, _ = unpack_imag(w_im)
        if n_re <= 0:
            mag = 0.0
        else:
            n_re = min(n_re, self.K - 1)
            lo = self.H[n_re]
            hi = self.H[n_re + 1] if n_re + 1 <= self.K else lo
            mag = lo + (h_frac / 0x7F) * (hi - lo)
        mag_im = F_q / 0xFFF
        val = 0.5 * (mag + mag_im)
        return -val if s_re else val

    # ----- helpers -----
    def _harmonic_index(self, v: float) -> int:
        if v <= 0:
            return 0
        n = int(round(math.exp(v - GAMMA)))
        n = max(1, min(self.K, n))
        while n > 1 and self.H[n] > v:
            n -= 1
        while n < self.K and self.H[n + 1] < v:
            n += 1
        return n

    def _frac(self, v: float, n: int) -> int:
        if n >= self.K or n < 1:
            return 0
        lo, hi = self.H[n], self.H[n + 1]
        if hi <= lo:
            return 0
        f = (v - lo) / (hi - lo)
        return int(min(0x7F, max(0, round(f * 0x7F))))


# ============================================================
#  MOSB read-only buffer
# ============================================================
MAGIC       = b"MOSB"
HEADER_FMT  = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class MOSBReadOnlyBuffer:
    """
    Chunked read-only buffer.

        per chunk:  MAGIC(4) | K(4) | n(8) | cid(8) | S(8) | H(8) | m(8) | payload
    """
    def __init__(self, K: int = 512):
        self.K = int(K)
        self.mu = mobius_sieve(K)
        self.data = bytearray()
        self.chunk_index: List[Tuple[int, int, float, float, float]] = []
        self._cid = 0

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
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        return H, abs(S) * math.exp(-H)

    def append(self, payload: bytes) -> int:
        n = len(payload)
        S = self._supertrace(payload)
        H, m = self._entropy_mass(S, n)
        cid = self._cid; self._cid += 1
        header = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        start = len(self.data)
        self.data.extend(header + payload)
        self.chunk_index.append((start, start + HEADER_SIZE + n, S, H, m))
        return cid

    def read_chunk(self, cid: int
                   ) -> Optional[Tuple[bytes, float, float, float]]:
        if cid < 0 or cid >= len(self.chunk_index):
            return None
        start, end, *_ = self.chunk_index[cid]
        blob = bytes(self.data[start:end])
        magic, K, n, cid_stored, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])
        if magic != MAGIC or K != self.K or cid_stored != cid:
            return None
        payload = blob[HEADER_SIZE:HEADER_SIZE + n]
        if abs(self._supertrace(payload) - S) > 1e-9:
            return None
        return payload, S, H, m

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_index)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# ============================================================
#  Reasoning ALU  (per-node decision)
# ============================================================
@dataclass
class NodeDecision:
    node_id: int
    symmetric: bool
    path: str                    # 'direct' | 'buffer'
    buffer_cid: Optional[int]
    log_x: int
    log_value: float
    accepted: bool
    trace: float


class ReasoningALU:
    """
    Decision layer:
        symmetric   → bypass buffer, solve directly
        asymmetric  → read chunk from MOSB, decode, then activate
    """
    def __init__(self, survival_floor: float = 0.0):
        self.survival_floor = survival_floor

    def decide(self, node: UGCNode, x: float,
               symmetric: bool,
               buf: Optional[MOSBReadOnlyBuffer],
               cid: Optional[int]) -> NodeDecision:
        if symmetric:
            log_x, log_val = node.activate(x)
            return NodeDecision(
                node_id=node.node_id, symmetric=True, path="direct",
                buffer_cid=None, log_x=log_x, log_value=log_val,
                accepted=abs(log_val) > self.survival_floor,
                trace=abs(log_val))

        if buf is None or cid is None:
            return NodeDecision(
                node_id=node.node_id, symmetric=False, path="buffer",
                buffer_cid=cid, log_x=0, log_value=0.0,
                accepted=False, trace=0.0)

        v = node.read_via_buffer(buf, cid)
        if v is None:
            return NodeDecision(
                node_id=node.node_id, symmetric=False, path="buffer",
                buffer_cid=cid, log_x=0, log_value=0.0,
                accepted=False, trace=0.0)

        log_x, log_val = node.activate(v)
        return NodeDecision(
            node_id=node.node_id, symmetric=False, path="buffer",
            buffer_cid=cid, log_x=log_x, log_value=log_val,
            accepted=abs(log_val) > self.survival_floor,
            trace=abs(log_val))


# ============================================================
#  Layer  ·  split-channel weights + MOSB buffer
# ============================================================
@dataclass
class UGCWeightLayer:
    W: np.ndarray
    codec: SplitChannelCodec
    buffer: Optional[MOSBReadOnlyBuffer] = None
    _enc_re: List[int] = field(default_factory=list)
    _enc_im: List[int] = field(default_factory=list)
    _cids:   List[int] = field(default_factory=list)
    _sym:    bool = False
    _built:  bool = False

    def build(self) -> None:
        d_in, d_out = self.W.shape
        self._sym = self.is_symmetric()
        self._enc_re.clear(); self._enc_im.clear(); self._cids.clear()
        for i in range(d_in):
            for j in range(d_out):
                w = float(self.W[i, j])
                wr, wi = self.codec.encode_scalar(w)
                self._enc_re.append(wr)
                self._enc_im.append(wi)
                if self.buffer is not None:
                    cid = self.buffer.append(struct.pack(">d", w))
                    self._cids.append(cid)
                else:
                    self._cids.append(-1)
        self._built = True

    def is_symmetric(self) -> bool:
        if self.W.shape[0] != self.W.shape[1]:
            return False
        return bool(np.allclose(self.W, self.W.T, atol=1e-6))

    def decode(self, i: int, j: int) -> float:
        d_out = self.W.shape[1]
        k = i * d_out + j
        return self.codec.decode_scalar(self._enc_re[k], self._enc_im[k])

    def cid(self, i: int, j: int) -> int:
        return self._cids[i * self.W.shape[1] + j]

    @property
    def symmetric(self) -> bool:
        return self._sym

    @property
    def n_params(self) -> int:
        return self.W.size

    @property
    def n_bytes_packed(self) -> int:
        return 8 * self.n_params


# ============================================================
#  Neural net  ·  reasoning ALU + UGC nodes
# ============================================================
class UGCReasoningNet:
    """
    Forward pass per layer l:
        x_{l+1}[j] = UGC_activate( Σ_i W_l[i, j] · x_l[i] )

    Symmetry rule:
        W_l symmetric   →  direct decode, buffer skipped
        W_l asymmetric  →  read weight from MOSB buffer, then decode
    """
    def __init__(self, dims: List[int], K_codec: int = 2048,
                 K_buffer: int = 512, n_nodes_per_layer: int = 4):
        self.dims = list(dims)
        self.n_layers = len(dims) - 1
        self.codec = SplitChannelCodec(K=K_codec)
        self.buffer = MOSBReadOnlyBuffer(K=K_buffer)
        total_nodes = n_nodes_per_layer * (self.n_layers + 1)
        self.nodes = [UGCNode(i) for i in range(total_nodes)]
        self.n_nodes_per_layer = n_nodes_per_layer
        self.alu = ReasoningALU()
        self.layers: List[UGCWeightLayer] = []

    # --------------------------------------------------------
    def build(self, seed: int = 0, symmetric_layers: Optional[List[int]] = None
              ) -> None:
        symmetric_layers = symmetric_layers or []
        rng = np.random.default_rng(seed)
        self.layers.clear()
        for l in range(self.n_layers):
            d_in, d_out = self.dims[l], self.dims[l + 1]
            W = rng.standard_normal((d_in, d_out)) * 0.5
            if l in symmetric_layers and d_in == d_out:
                W = 0.5 * (W + W.T)
            layer = UGCWeightLayer(W=W, codec=self.codec,
                                    buffer=self.buffer)
            layer.build()
            self.layers.append(layer)

    # --------------------------------------------------------
    def forward(self, x: np.ndarray, verbose: bool = False) -> Dict:
        acts = [np.asarray(x, dtype=float)]
        decisions_per_layer: List[List[NodeDecision]] = []
        for l, layer in enumerate(self.layers):
            d_in, d_out = layer.W.shape
            if verbose:
                path = "direct" if layer.symmetric else "buffer"
                print(f"  layer {l}: {d_in:>3d}→{d_out:<3d}  "
                      f"symmetric={layer.symmetric}  path={path}")
            pre = np.zeros(d_out)
            decisions: List[NodeDecision] = []
            for j in range(d_out):
                s = 0.0
                for i in range(d_in):
                    if layer.symmetric:
                        # direct decode — buffer is skipped
                        w = layer.decode(i, j)
                    else:
                        # buffered read — from MOSB
                        cid = layer.cid(i, j)
                        w = 0.0
                        if layer.buffer is not None and cid >= 0:
                            chunk = layer.buffer.read_chunk(cid)
                            if chunk is not None:
                                w = struct.unpack(">d", chunk[0][:8])[0]
                        if w == 0.0:
                            w = layer.decode(i, j)   # decode fallback
                    s += w * acts[-1][i]
                # choose a UGC node for this output index
                node = self.nodes[(l * d_out + j) % len(self.nodes)]
                dec = self.alu.decide(
                    node, s, layer.symmetric,
                    layer.buffer if not layer.symmetric else None,
                    layer.cid(0, j) if not layer.symmetric else None)
                decisions.append(dec)
                pre[j] = dec.log_value
            acts.append(pre)
            decisions_per_layer.append(decisions)
        return dict(activations=acts, decisions=decisions_per_layer)

    # --------------------------------------------------------
    def solve(self, x: np.ndarray) -> np.ndarray:
        """One-shot solve.  Only valid end-to-end when all layers are symmetric."""
        if not all(l.symmetric for l in self.layers):
            raise RuntimeError("solve() requires all layers symmetric")
        return self.forward(x)["activations"][-1]

    # --------------------------------------------------------
    def summary(self) -> Dict:
        n_bytes_packed = sum(l.n_bytes_packed for l in self.layers)
        n_bytes_f32    = sum(l.W.size * 4 for l in self.layers)
        return dict(
            dims=self.dims,
            n_layers=self.n_layers,
            n_params=sum(l.n_params for l in self.layers),
            n_bytes_packed=n_bytes_packed,
            n_bytes_float32=n_bytes_f32,
            compression=n_bytes_f32 / max(n_bytes_packed, 1),
            symmetric_layers=[l.symmetric for l in self.layers],
            buffer_chunks=self.buffer.n_chunks,
            buffer_bytes=self.buffer.size_bytes,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("UGC reasoning neural net  ·  split-channel weights  ·  MOSB buffer")
    print("=" * 80)
    print(f"  α = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  A_STEP            = {A_STEP:.6f}")
    print(f"  6/π²              = {DENSITY:.6f}")
    print(f"  block             = {BLOCK}×{BLOCK}, perms = {N_PERM}")
    print()

    # ---------- build ----------
    dims = [8, 8, 8, 8]                     # square so layer 0 can be symmetric
    net = UGCReasoningNet(dims=dims, K_codec=1024, K_buffer=256)
    net.build(seed=1, symmetric_layers=[0])  # force layer 0 symmetric
    print("--- network build ---")
    for l, layer in enumerate(net.layers):
        print(f"  layer {l}: {layer.W.shape[0]:>3d}→{layer.W.shape[1]:<3d}  "
              f"symmetric={str(layer.symmetric):5s}  "
              f"params={layer.n_params:>3d}  "
              f"packed={layer.n_bytes_packed:>4d} B")
    print()

    # ---------- symmetry test ----------
    print("--- symmetry rule ---")
    for l, layer in enumerate(net.layers):
        if layer.symmetric:
            print(f"  layer {l}: symmetric → BYPASS buffer, solve directly")
        else:
            print(f"  layer {l}: asymmetric → READ from MOSB buffer")
    print()

    # ---------- forward ----------
    x = np.array([1.0, -0.5, 0.3, 0.8, -1.2, 0.6, 0.1, -0.9])
    print("--- forward pass ---")
    res = net.forward(x, verbose=True)
    print()

    # ---------- per-node decisions ----------
    print("--- per-node decisions ---")
    for l, decisions in enumerate(res["decisions"]):
        path = decisions[0].path if decisions else "—"
        n_acc = sum(1 for d in decisions if d.accepted)
        traces = [d.trace for d in decisions]
        print(f"  layer {l}: path={path:8s}  "
              f"accepted={n_acc:>2d}/{len(decisions):<2d}  "
              f"mean_trace={np.mean(traces):.4f}  "
              f"max_trace={np.max(traces):.4f}")
    print()

    # ---------- final output ----------
    out = res["activations"][-1]
    print("--- final activation ---")
    print(f"  output shape      = {out.shape}")
    print(f"  output            = {np.array2string(out, precision=4)}")
    print()

    # ---------- solve ----------
    print("--- solve path ---")
    n_sym = sum(1 for l in net.layers if l.symmetric)
    print(f"  symmetric layers  = {n_sym} / {net.n_layers}")
    if n_sym == net.n_layers:
        y = net.solve(x)
        print(f"  solved output     = {np.array2string(y, precision=4)}")
    else:
        print("  some layers are asymmetric — buffer path required")
        # demonstrate a single buffered read
        for l, layer in enumerate(net.layers):
            if not layer.symmetric:
                cid = layer.cid(0, 0)
                chunk = layer.buffer.read_chunk(cid)
                if chunk is not None:
                    payload, S, H, m = chunk
                    v = struct.unpack(">d", payload[:8])[0]
                    print(f"  layer {l}: read cid={cid}  "
                          f"value={v:+.6f}  S={S:+.4f}  H={H:.4f}  m={m:.4f}")
                break
        # but layer 0 is symmetric → direct solve of layer 0
        if net.layers[0].symmetric:
            print("  layer 0 is symmetric → direct solve available")
            x1 = res["activations"][1]
            print(f"  layer-0 output    = {np.array2string(x1, precision=4)}")
    print()

    # ---------- buffer summary ----------
    print("--- buffer summary ---")
    s = net.summary()
    print(f"  n layers          = {s['n_layers']}")
    print(f"  n params          = {s['n_params']}")
    print(f"  packed bytes      = {s['n_bytes_packed']}  (64 b / weight)")
    print(f"  float32 bytes     = {s['n_bytes_float32']}")
    print(f"  compression       = {s['compression']:.2f}x")
    print(f"  symmetric layers  = {sum(s['symmetric_layers'])} / {s['n_layers']}")
    print(f"  buffer chunks     = {s['buffer_chunks']}")
    print(f"  buffer bytes      = {s['buffer_bytes']}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  symmetric layer     O(d_in · d_out)  direct decode, no buffer")
    print("  asymmetric layer    O(d_in · d_out)  +  O(1) per buffer read")
    print("  UGC activation      O(1)  log-table lookup")
    print("  buffer read         O(n_chunk)  header + payload")
    print("  dlog solve          O(1)  table lookup")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()