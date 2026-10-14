#!/usr/bin/env python3
"""
mobius_utf8_ugc_alu.py
======================
UTF-8 bytes → discrete/transcendental split → UGC boolean satisfaction
→ read-only ALU → fiber reconstruction.

Concepts
--------
UTF-8 signed-index table   (from n_k_div_finitestep_convergence.py)
    v(-n) = -H_{n+1}                        discrete algebraic (real)
    v(+n) = -H_{n+1} + i·α·(μ*H)(n)         transcendental (imaginary)
    v(0)  = 0

Discrete point particles through transcendental time
    x(i, j) = Re(v(i))          discrete position
    y(i, j) = Im(v(j))          transcendental time
    vx(i)   = dRe/dn            finite-step velocity  (a = α/0.3628)
    ζ'(t)   = spectral derivative  (modulates the transcendental rate)

UGC boolean satisfaction
    edge (a, b) survives iff  g(a,b) ≥ τ  AND  σ(a) = σ(b)

Read-only ALU                 (from alu_finite_step.py)
    input via MOSB buffer     (from server_nodes_dns_buffer_read_only.py)
    ADD / SUB / MUL / DIV via reconstruction coupling

Fiber reconstruction         (from quantum_geo_location_OK.py)
    H(z) = exp(-α|z|)
"""

from __future__ import annotations
import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA      = 1.0 / (PI - E)                   # ≈ 2.362338
ALPHA_ASYM = 0.3628
A_STEP     = ALPHA / ALPHA_ASYM               # ≈ 6.511
GAMMA      = 0.5772156649015329
K_DEFAULT  = 60


# ============================================================
#  1.  Möbius sieve + harmonic numbers + convolution
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
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


def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


def mu_convolution_H(K: int):
    """F = μ * H, computed in O(K log K)."""
    mu = mobius_sieve(K)
    H  = harmonic_numbers(K)
    F  = np.zeros(K + 1, dtype=float)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = int(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F, mu, H


# ============================================================
#  2.  UTF-8 signed-index table
# ============================================================
def utf8_byte_to_signed_index(b: int, K: int) -> int:
    """b ∈ [0,255] → n ∈ [-K,K]  (skip 0)."""
    n = -K + int(round(2 * K * b / 255.0))
    return n if n != 0 else 1


def signed_value(n: int, H: np.ndarray, F: np.ndarray,
                 alpha: float = ALPHA) -> complex:
    """v(n) with discrete real + transcendental imaginary parts."""
    if n == 0:
        return 0.0 + 0.0j
    a = abs(n)
    real = -(float(H[a]) if a < H.size else math.log(a) + GAMMA)
    if n < 0:
        return complex(real, 0.0)
    imag = alpha * (float(F[a]) if a < F.size else 0.0)
    return complex(real, imag)


@dataclass
class UTF8Table:
    K: int
    H: np.ndarray
    F: np.ndarray
    mu: np.ndarray
    v: np.ndarray                        # signed values, n = -K..K

    @classmethod
    def build(cls, K: int = K_DEFAULT) -> "UTF8Table":
        F, mu, H = mu_convolution_H(K + 1)
        ns = np.arange(-K, K + 1)
        v = np.array([signed_value(int(n), H, F) for n in ns])
        return cls(K=K, H=H, F=F, mu=mu, v=v)

    def real_part(self, n: int) -> float:
        return float(self.v[n + self.K].real)

    def imag_part(self, n: int) -> float:
        return float(self.v[n + self.K].imag)

    def dreal_dn(self, n: int, step: float = ALPHA) -> float:
        """Forward finite-step derivative of the discrete part."""
        n_lo = max(-self.K, min(self.K, int(round(n))))
        n_hi = max(-self.K, min(self.K, int(round(n + step))))
        return (self.real_part(n_hi) - self.real_part(n_lo)) / step

    # ---------- spectral derivative dζ/dt ----------
    def dζ_dt(self, t: float, coeffs: Optional[np.ndarray] = None) -> float:
        """
        dζ/dt = -Σ_{j≠0} (j/α) Re(C_j e^{ijt/α})

        If no coefficients given, C_j = v(j).
        """
        if coeffs is None:
            coeffs = self.v
        K = self.K
        j = np.arange(-K, K + 1)
        jnz = j[j != 0]
        Cj = coeffs[jnz + K]
        phases = jnz * t / ALPHA
        return float(-np.sum((jnz / ALPHA) * np.real(Cj * np.exp(1j * phases))))


# ============================================================
#  3.  Discrete point particles through transcendental time
# ============================================================
@dataclass
class DiscreteParticle:
    """
    A discrete point particle at (i, j):

        x  =  Re(v(i))     discrete position
        y  =  Im(v(j))     transcendental time
        vx =  dRe/dn(i)    finite-step velocity
        zt =  dζ/dt        local spectral rate
    """
    i: int
    j: int
    x: float
    y: float
    vx: float
    zt: float

    @classmethod
    def from_table(cls, table: UTF8Table, i: int, j: int,
                   t: float = 0.0) -> "DiscreteParticle":
        return cls(
            i=i, j=j,
            x=table.real_part(i),
            y=table.imag_part(j),
            vx=table.dreal_dn(i),
            zt=table.dζ_dt(t),
        )

    @property
    def theta(self) -> float:
        """Phase in the (discrete, transcendental) plane."""
        return math.atan2(self.y, self.x)

    def __repr__(self):
        return (f"P(i={self.i:>3d}, j={self.j:>3d})  "
                f"x={self.x:+.4f}  y={self.y:+.4f}  "
                f"vx={self.vx:+.4f}")


def build_particle_cloud(table: UTF8Table,
                         stream: bytes) -> List[DiscreteParticle]:
    """Each byte → particle at (i, j)."""
    K = table.K
    particles = []
    for b in stream:
        i = utf8_byte_to_signed_index(b, K)
        j = (i + 1) % (2 * K + 1) - K
        if j == 0:
            j = -1
        particles.append(DiscreteParticle.from_table(table, i, j))
    return particles


# ============================================================
#  4.  μ-gated vector per particle
# ============================================================
def particle_vector(p: DiscreteParticle, K: int,
                    mu: np.ndarray) -> np.ndarray:
    """
        v_p(n) = μ(n) · cos(A_STEP n²/K) · e^{i θ_p}

    where θ_p = atan2(y, x) is the particle's transcendental phase.
    """
    n = np.arange(K)
    mu_arr = np.array([float(mu[k]) if k < len(mu) else 0.0
                       for k in range(K)])
    chirp = np.cos(A_STEP * n * n / K)
    theta = p.theta
    return mu_arr * chirp * complex(math.cos(theta), math.sin(theta))


def cross_coupling(va: np.ndarray, vb: np.ndarray) -> float:
    """Normalized cross-coupling |⟨va, vb⟩| / (‖va‖ · ‖vb‖)."""
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.vdot(va, vb)) / (na * nb))


# ============================================================
#  5.  UGC boolean satisfaction
# ============================================================
def ugc_label(phase: float, k: int) -> int:
    return int(round(k * phase / (2 * PI))) % k


def ugc_satisfied(va: np.ndarray, vb: np.ndarray,
                  k: int, threshold: float = 0.3) -> bool:
    """
    Edge (a, b) survives iff:
        1. cross-coupling ≥ threshold
        2. quantized labels match  σ(a) = σ(b)
    """
    g = cross_coupling(va, vb)
    if g < threshold:
        return False
    phi_a = math.atan2(va.imag.sum(), va.real.sum())
    phi_b = math.atan2(vb.imag.sum(), vb.real.sum())
    return ugc_label(phi_a, k) == ugc_label(phi_b, k)


def ugc_survivors(particles: List[DiscreteParticle],
                  K: int, mu: np.ndarray, k: int,
                  threshold: float = 0.3
                  ) -> Tuple[List[int], List[int]]:
    """Return (survivors, non-survivors) as index lists."""
    N = len(particles)
    vecs = [particle_vector(p, K, mu) for p in particles]
    surv, non = [], []
    for i in range(N - 1):
        (surv if ugc_satisfied(vecs[i], vecs[i + 1], k, threshold)
         else non).append(i)
    return surv, non


# ============================================================
#  6.  Read-only buffer + ALU  (from alu_finite_step.py)
# ============================================================
MAGIC = b"MOSB"
HEADER_FMT = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class ReadOnlyBuffer:
    """MOSB-framed read-only buffer."""
    def __init__(self, K: int):
        self.K = K
        self.data = bytearray()
        self.chunks: List[Tuple[int, int, float, float, float]] = []
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
        H = -ALPHA * p * math.log(p)
        return H, abs(S) * math.exp(-H)

    def append(self, payload: bytes) -> int:
        n = len(payload)
        S = self._supertrace(payload)
        H, m = self._entropy_mass(S, n)
        cid = self._cid
        self._cid += 1
        hdr = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        start = len(self.data)
        self.data.extend(hdr + payload)
        self.chunks.append((start, start + HEADER_SIZE + n, S, H, m))
        return cid

    def read(self, cid: int
             ) -> Optional[Tuple[bytes, float, float, float]]:
        if cid < 0 or cid >= len(self.chunks):
            return None
        start, end, *_ = self.chunks[cid]
        blob = bytes(self.data[start:end])
        magic, K, n, cid_stored, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])
        if magic != MAGIC or K != self.K or cid_stored != cid:
            return None
        payload = blob[HEADER_SIZE:HEADER_SIZE + n]
        S_check = self._supertrace(payload)
        if abs(S_check - S) > 1e-9:
            return None
        return payload, S, H, m


class ReadOnlyALU:
    """
    ALU ops via reconstruction-coupled convolution.

    Universal finite step:
        A_STEP = α / 0.3628 ≈ 6.511
    """
    def __init__(self, K: int = 64):
        self.K = K
        self.a = A_STEP
        self.mu = mobius_sieve(K)
        self.idx = np.arange(K)
        self.gate = (self.mu[1:K + 1] != 0).astype(float)
        self.w = self.gate * np.cos(self.a * self.idx * self.idx / K)
        self.w_sq = self.w ** 2
        self.w_norm = float(np.dot(self.w, self.w))

    def _read(self, buf: ReadOnlyBuffer, cid: int) -> Tuple[float, float]:
        chunk = buf.read(cid)
        if chunk is None:
            return 0.0, 0.0
        payload = chunk[0]
        if len(payload) >= 16:
            x, y = struct.unpack(">dd", payload[:16])
        elif len(payload) >= 8:
            x = struct.unpack(">d", payload[:8])[0]
            y = 0.0
        else:
            x, y = 0.0, 0.0
        return x, y

    def reg2sig(self, x: float) -> np.ndarray:
        return float(x) * self.w

    def travel(self, s: np.ndarray) -> np.ndarray:
        ds = np.abs(np.diff(s, prepend=s[0]))
        return np.cumsum(ds) + 1e-9

    def recon_coupling(self, sx, sy) -> np.ndarray:
        Lx = self.travel(sx); Ly = self.travel(sy)
        L_ref = 0.5 * (Lx[-1] + Ly[-1]) + 1e-9
        return np.clip(np.sqrt(np.outer(Lx, Ly)) / L_ref, 0.0, 1.0)

    # ---------- exact ops ----------
    def ADD(self, x, y):
        sx, sy = self.reg2sig(x), self.reg2sig(y)
        kd = self.recon_coupling(sx, sy).diagonal()
        s_out = kd * (sx + sy)
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(kd, self.w_sq))
        return num / den if den > 1e-12 else 0.0

    def SUB(self, x, y):
        sx, sy = self.reg2sig(x), self.reg2sig(y)
        kd = self.recon_coupling(sx, sy).diagonal()
        s_out = kd * (sx - sy)
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(kd, self.w_sq))
        return num / den if den > 1e-12 else 0.0

    def MUL(self, x, y):
        sx, sy = self.reg2sig(x), self.reg2sig(y)
        Kc = self.recon_coupling(sx, sy)
        G = Kc * np.outer(sx, sy)
        s_out = G @ self.w
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(self.w_sq, Kc @ self.w_sq))
        return num / den if den > 1e-12 else 0.0

    def DIV(self, x, y):
        if abs(y) < 1e-9:
            return float('inf') * np.sign(x)
        sx, sy = self.reg2sig(x), self.reg2sig(y)
        Kc = self.recon_coupling(sx, sy)
        sy_inv = np.where(np.abs(sy) > 1e-9, 1.0 / sy, 0.0)
        s_out = (Kc * np.outer(sx, sy_inv)) @ (sy * self.w)
        num = float(np.dot(s_out, self.w))
        den = float(np.dot(self.w_sq, Kc @ self.w))
        return num / den if den > 1e-12 else 0.0

    def dispatch(self, buf: ReadOnlyBuffer,
                 cid_x: int, cid_y: int, op: str) -> Dict:
        x, _ = self._read(buf, cid_x)
        y, _ = self._read(buf, cid_y)
        fn = {"ADD": self.ADD, "SUB": self.SUB,
              "MUL": self.MUL, "DIV": self.DIV}.get(op)
        z = fn(x, y) if fn else 0.0
        return dict(op=op, x=x, y=y, z=z,
                    cid_x=cid_x, cid_y=cid_y)


# ============================================================
#  7.  Fiber kernel  (from quantum_geo_location_OK.py)
# ============================================================
def fiber_kernel(width: int, alpha: float = ALPHA) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    h = np.exp(-alpha * np.abs(x))
    return h / h.sum()


def fiber_propagate(signal: np.ndarray,
                    width: int = 41) -> np.ndarray:
    h = fiber_kernel(width)
    L = len(signal)
    N = 1 << (2 * L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L))
    ker_pad = np.pad(h, (0, N - len(h)))
    return np.real(np.fft.ifft(
        np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:L])


# ============================================================
#  8.  Full pipeline
# ============================================================
def pipeline(message: str,
             K: int = K_DEFAULT,
             k_labels: int = 4,
             threshold: float = 0.3,
             K_alu: int = 64,
             verbose: bool = True) -> Dict:
    if verbose:
        print("=" * 78)
        print("Möbius UTF-8 → UGC → ALU → fiber")
        print("=" * 78)
        print(f"  α        = 1/(π−e)      = {ALPHA:.6f}")
        print(f"  A_STEP   = α / 0.3628   = {A_STEP:.6f}")
        print(f"  K_table  = {K}    K_alu = {K_alu}")
        print()

    # ---------- 1. UTF-8 table ----------
    t0 = time.perf_counter()
    table = UTF8Table.build(K=K)
    if verbose:
        print(f"[1] UTF-8 table             {(time.perf_counter()-t0)*1e3:.2f} ms")
        print(f"      H[K]={table.H[K]:.4f}  F[K]={table.F[K]:+.4f}  "
              f"dζ/dt(0)={table.dζ_dt(0.0):+.4f}")

    # ---------- 2. particles ----------
    stream = message.encode("utf-8")
    particles = build_particle_cloud(table, stream)
    if verbose:
        print(f"[2] particle cloud          {len(particles)} particles")
        print(f"      stream bytes         = {len(stream)}")
        for p in particles[:3]:
            print(f"      {p!r}")

    # ---------- 3. UGC ----------
    mu = table.mu
    surv, non = ugc_survivors(particles, K, mu, k_labels, threshold)
    if verbose:
        rate = len(surv) / max(1, len(surv) + len(non))
        print(f"[3] UGC boolean")
        print(f"      survivors            = {len(surv)}")
        print(f"      non-survivors        = {len(non)}")
        print(f"      survival rate        = {rate:.4f}")

    # ---------- 4. read-only buffer ----------
    buf = ReadOnlyBuffer(K=K)
    for p in particles:
        buf.append(struct.pack(">dd", p.x, p.y))
    if verbose:
        print(f"[4] read-only buffer        {len(buf.chunks)} chunks  "
              f"({len(buf.data)} bytes)")
        for cid in range(min(2, len(buf.chunks))):
            chunk = buf.read(cid)
            if chunk is not None:
                _, S, H, m = chunk
                print(f"      cid {cid}: S={S:+.4f} H={H:.4f} m={m:.4e}")

    # ---------- 5. ALU ----------
    alu = ReadOnlyALU(K=K_alu)
    alu_results = []
    for cid in range(0, min(8, len(buf.chunks)) - 1, 2):
        for op in ["ADD", "SUB", "MUL", "DIV"]:
            alu_results.append(alu.dispatch(buf, cid, cid + 1, op))
    if verbose:
        print(f"[5] read-only ALU")
        print(f"      μ-gate density       = "
              f"{np.count_nonzero(alu.gate)/alu.K:.4f}  "
              f"(target 6/π²={6/PI**2:.4f})")
        for r in alu_results[:4]:
            print(f"      {r['op']}  x={r['x']:+.4f}  y={r['y']:+.4f}  "
                  f"z={r['z']:+.4f}")

    # ---------- 6. fiber reconstruction ----------
    signal = np.array([p.vx for p in particles])
    signal_fiber = fiber_propagate(signal, width=41)
    if verbose:
        print(f"[6] fiber reconstruction")
        print(f"      ‖signal‖  = {np.linalg.norm(signal):.4f}")
        print(f"      ‖fiber‖   = {np.linalg.norm(signal_fiber):.4f}")
        print()

    return dict(
        table=table, particles=particles,
        survivors=surv, non_survivors=non,
        buffer=buf, alu=alu, alu_results=alu_results,
        signal=signal, signal_fiber=signal_fiber,
        K=K, k_labels=k_labels, threshold=threshold,
    )


# ============================================================
#  9.  Visualization
# ============================================================
def plot_pipeline(result: Dict):
    table = result["table"]
    particles = result["particles"]
    surv = result["survivors"]
    non = result["non_survivors"]
    K = result["K"]

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    # (a) discrete algebraic part
    ax = fig.add_subplot(gs[0, 0])
    ns = np.arange(-K, K + 1)
    ax.plot(ns, table.v.real, color="#2c3e50", lw=1.2,
            label="Re v(n)")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("n"); ax.set_ylabel("Re v")
    ax.set_title("Discrete algebraic part  ·  −H_{n+1}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) transcendental imaginary part
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ns, table.v.imag, color="#e74c3c", lw=1.2,
            label="Im v(n)")
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("n"); ax.set_ylabel("Im v")
    ax.set_title("Transcendental part  ·  α · (μ*H)(n)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) finite-step derivative of the discrete part
    ax = fig.add_subplot(gs[0, 2])
    dreal = np.array([table.dreal_dn(int(n)) for n in ns])
    ax.plot(ns, dreal, color="#8e44ad", lw=1.2)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("n"); ax.set_ylabel("dRe/dn")
    ax.set_title(f"Finite-step velocity  ·  a={ALPHA:.3f}")
    ax.grid(alpha=0.3)

    # (d) particle cloud (discrete position, transcendental time)
    ax = fig.add_subplot(gs[1, 0])
    xs = np.array([p.x for p in particles])
    ys = np.array([p.y for p in particles])
    vxs = np.array([p.vx for p in particles])
    sc = ax.scatter(xs, ys, c=vxs, cmap="viridis",
                    s=25, edgecolors="k", linewidth=0.3)
    ax.set_xlabel("x = Re v(i)   (discrete)")
    ax.set_ylabel("y = Im v(j)   (transcendental)")
    ax.set_title("Particle cloud  ·  discrete × transcendental")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="vx")
    ax.grid(alpha=0.3)

    # (e) UGC survivors / non-survivors
    ax = fig.add_subplot(gs[1, 1])
    idxs = np.arange(len(particles) - 1)
    colors = np.array(["#16a085"] * len(idxs), dtype=object)
    for i in non:
        colors[i] = "#e74c3c"
    ax.bar(idxs, [1] * len(idxs), color=colors, width=1.0)
    ax.set_xlabel("edge index")
    ax.set_ylabel("satisfied?")
    ax.set_title(f"UGC boolean  ·  survival = {len(surv)}/{len(idxs)}")
    ax.set_yticks([]); ax.grid(alpha=0.3, axis="x")

    # (f) ALU results
    ax = fig.add_subplot(gs[1, 2])
    ops = [r["op"] for r in result["alu_results"]]
    zs = [r["z"] for r in result["alu_results"]]
    colors = {"ADD": "#3a7bd5", "SUB": "#e67e22",
              "MUL": "#e74c3c", "DIV": "#16a085"}
    ax.bar(np.arange(len(ops)), zs,
           color=[colors[op] for op in ops], edgecolor="k")
    ax.set_xticks(np.arange(len(ops)))
    ax.set_xticklabels(ops, rotation=45, fontsize=8)
    ax.set_ylabel("z")
    ax.set_title("Read-only ALU outputs")
    ax.grid(alpha=0.3, axis="y")

    # (g) signal before/after fiber
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(result["signal"], color="#2c3e50", lw=1.0,
            label="signal = particle velocities")
    ax.plot(result["signal_fiber"], color="#e74c3c", lw=1.4,
            alpha=0.9, label="fiber reconstruction")
    ax.set_xlabel("particle index")
    ax.set_ylabel("amplitude")
    ax.set_title("Fiber kernel  exp(−α|z|)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (h) summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    v_neg = table.v[0].real
    v_pos = table.v[-1]
    ax.text(0.0, 1.0,
            f"UTF-8 → UGC → ALU → fiber\n\n"
            f"  K_table      = {K}\n"
            f"  K_alu        = {result['alu'].K}\n"
            f"  particles    = {len(particles)}\n"
            f"  survivors    = {len(surv)}\n"
            f"  non-surv     = {len(non)}\n"
            f"  ALU ops      = {len(result['alu_results'])}\n"
            f"  buffer       = {len(result['buffer'].chunks)} chunks\n"
            f"\n"
            f"  v(-K) = {v_neg:+.4f}\n"
            f"  v(+K) = {v_pos.real:+.4f} {v_pos.imag:+.4f}i\n"
            f"  dζ/dt(0) = {table.dζ_dt(0.0):+.4f}\n"
            f"\n"
            f"  A_STEP = {A_STEP:.4f}\n"
            f"  α      = {ALPHA:.4f}",
            va="top", family="monospace", fontsize=9)

    plt.suptitle(
        "Möbius UTF-8 UGC ALU  ·  discrete point particles through "
        "transcendental time  ·  read-only ALU  ·  fiber reconstruction",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================
#  10.  Demo
# ============================================================
def demo():
    message = "Möbius UTF-8 → UGC → ALU → fiber"
    result = pipeline(message,
                      K=K_DEFAULT, k_labels=4,
                      threshold=0.3, K_alu=64,
                      verbose=True)
    plot_pipeline(result)


if __name__ == "__main__":
    demo()