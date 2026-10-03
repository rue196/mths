#!/usr/bin/env python3
"""
gpu_graphics_mobius_game.py
===========================

GPU graphics pipeline where the shader samples:

  • the chip-compressed |Ci| buffer           (from chip-g.py)
  • a ColorMöbiusGate transform LUT           (from color.py)

Fragment shader cost per pixel:

    2 |Ci| loads          O(1)
    2 LUT loads           O(1)
    1 elliptic projection O(1)
    3 tanh                O(1)

The O(K log K) chip pipeline and the ColorMöbiusGate's Möbius
transform run only at build time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
A_CONST     = 1.0 / (PI - E)
ALPHA_SYM   = A_CONST
NORM        = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
K_DEFAULT   = 256
N_FIXED     = 4
LUT_SIZE    = 256


def A_of_k(k: float) -> float:
    return 10.0 * A_CONST * math.sin(math.radians(k * k))


def finite_step_k(k: int, mode: str = "forward") -> float:
    if mode == "forward":
        return A_of_k(k + 1) - A_of_k(k)
    if mode == "backward":
        return A_of_k(k) - A_of_k(k - 1)
    if mode == "central":
        return 0.5 * (A_of_k(k + 1) - A_of_k(k - 1))
    raise ValueError(mode)


A_4       = A_of_k(N_FIXED)
DA_4      = finite_step_k(N_FIXED, "forward")
DC_A_4    = finite_step_k(N_FIXED, "central")


# ============================================================
#  Möbius sieve
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  Chip compression  (from chip-g.py)
# ============================================================
class ChipProcessor:
    def __init__(self, max_K: int = 1000):
        self.max_K = max_K
        self.f = np.zeros(max_K, dtype=float)
        self.b = np.zeros(max_K, dtype=float)
        self.conv = np.zeros(max_K, dtype=float)
        self.order = np.zeros(max_K, dtype=int)
        self.mag = np.zeros(max_K, dtype=float)
        self.idx = np.arange(max_K, dtype=int)
        self.mu = None
        self._update_mu(max_K)

    def _update_mu(self, K: int) -> None:
        if self.mu is None or len(self.mu) < K + 1:
            self.mu = self._mobius_sieve(K)

    @staticmethod
    def _mobius_sieve(K: int):
        mu = [0] * (K + 1)
        mu[1] = 1
        primes, is_comp = [], [False] * (K + 1)
        for i in range(2, K + 1):
            if not is_comp[i]:
                primes.append(i)
                mu[i] = -1
            for p in primes:
                if i * p > K:
                    break
                is_comp[i * p] = True
                if i % p == 0:
                    mu[i * p] = 0
                    break
                else:
                    mu[i * p] = -mu[i]
        return mu

    def _tsp_route(self, signal, K: int) -> None:
        angles = np.zeros(K, dtype=float)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = math.atan2(y, x) + math.pi
        buckets = [[] for _ in range(360)]
        for i in range(K):
            b = int((angles[i] / (2 * math.pi)) * 360) % 360
            buckets[b].append(i)
        order = []
        for b in buckets:
            order.extend(b)
        self.order[:K] = order

    def _conv_exp_kernel(self, signal, K: int, alpha: float = ALPHA_SYM) -> None:
        lam = math.exp(-alpha)
        f = self.f
        f[0] = signal[0]
        for i in range(1, K):
            f[i] = signal[i] + lam * f[i - 1]
        b = self.b
        b[K - 1] = signal[K - 1]
        for i in range(K - 2, -1, -1):
            b[i] = signal[i] + lam * b[i + 1]
        conv = self.conv
        inv_den = 1.0 / (1.0 - lam * lam)
        inv_norm = 1.0 / NORM
        for i in range(K):
            conv_exp = (f[i] + b[i] - signal[i]) * inv_den
            conv[i] = (1.0 - conv_exp) * inv_norm

    def _supertrace_and_mass(self, signal, K: int):
        S = 0.0
        for i in range(K):
            val = signal[i]
            S += abs(val) if (i % 2 == 0) else -abs(val)
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

        self._tsp_route(signal, K)
        order = self.order[:K]
        if (not hasattr(self, "sorted_signal")
                or len(self.sorted_signal) < K):
            self.sorted_signal = np.zeros(K, dtype=signal.dtype)
        self.sorted_signal[:K] = signal[order]

        self._conv_exp_kernel(self.sorted_signal, K)
        S, H, m = self._supertrace_and_mass(self.conv, K)
        M = max(1, min(int(abs(S)), K))

        self._update_mu(K)
        mu = self.mu
        mag = self.mag[:K]
        for i in range(K):
            mag[i] = abs(self.conv[i])
        sorted_idx = np.argsort(mag)[::-1]

        kept = []
        count = 0
        for idx in sorted_idx:
            n = int(idx) + 1
            if mu[n] != 0:
                kept.append((int(idx), float(self.conv[idx])))
                count += 1
                if count >= M:
                    break

        return kept, S, H, m, self.conv[:K].copy()


# ============================================================
#  ColorMöbiusGate  (from color.py)
# ============================================================
class ColorMobiusGate:
    """
    RGB color processing via the Möbius harmonic transform.

    Used here as a build-time LUT generator: for each of the 256
    luminance levels, it produces a transformed RGB triple by
    running the chip pipeline on the RGB signal.
    """
    def __init__(self, K: int = 256, logic_gate: str = 'log',
                 use_elliptic: bool = True):
        self.K = K
        self.logic_gate = logic_gate
        self.use_elliptic = use_elliptic
        self.mu = mobius_sieve(K)
        self.processor = ChipProcessor(max_K=K)

    def _rgb_to_signal(self, rgb_data) -> np.ndarray:
        if isinstance(rgb_data, tuple) and len(rgb_data) == 3:
            signal = np.array(rgb_data, dtype=float)
        elif isinstance(rgb_data, np.ndarray):
            signal = rgb_data.flatten().astype(float)
        else:
            signal = np.array(rgb_data, dtype=float)
        if len(signal) > self.K:
            signal = signal[:self.K]
        elif len(signal) < self.K:
            signal = np.pad(signal, (0, self.K - len(signal)))
        return signal

    def _apply_elliptic_permutation(self, signal: np.ndarray) -> np.ndarray:
        K = len(signal)
        angles = np.zeros(K)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = math.atan2(y, x) + math.pi
        order = np.argsort(angles)
        return signal[order]

    def _apply_logic_gate(self, signal: np.ndarray) -> np.ndarray:
        if self.logic_gate == 'log':
            return np.log(np.maximum(np.abs(signal), 1e-12))
        if self.logic_gate == 'exp':
            return np.exp(np.clip(signal, -50, 50))
        if self.logic_gate == 'sin':
            return np.sin(signal)
        if self.logic_gate == 'cos':
            return np.cos(signal)
        return signal

    def transform_color(self, rgb) -> Tuple[int, int, int]:
        signal = self._rgb_to_signal(rgb)
        if self.use_elliptic:
            signal = self._apply_elliptic_permutation(signal)
        signal = self._apply_logic_gate(signal)
        kept, S, H, m, conv = self.processor.process(signal)
        recon = np.zeros(self.K, dtype=complex)
        for idx, val in kept:
            recon[idx] = val
        out = np.real(recon)[:3]
        out = np.clip(out, 0, 255).astype(np.uint8)
        return tuple(int(c) for c in out)


# ============================================================
#  Flat |Ci| buffer  (buffy.c pattern)
# ============================================================
class CiBuffer:
    __slots__ = ("K", "data")

    def __init__(self, K: int):
        self.K = K
        self.data = np.zeros(2 * K + 1, dtype=np.float32)

    def __getitem__(self, k: int) -> float:
        k = max(-self.K, min(self.K, int(k)))
        return float(self.data[k + self.K])

    def __setitem__(self, k: int, v: float) -> None:
        k = max(-self.K, min(self.K, int(k)))
        self.data[k + self.K] = np.float32(v)

    def as_bytes(self) -> bytes:
        return self.data.tobytes()


def build_ci_buffer(K: int, n: int = N_FIXED,
                    step_mode: str = "forward") -> CiBuffer:
    mu = mobius_sieve(K)
    buf = CiBuffer(K)
    ref_step = finite_step_k(n, step_mode)
    step_scale = DA_4 / (ref_step if ref_step != 0.0 else 1.0)

    for k in range(1, K + 1):
        gated = 1.0 if mu[k] != 0 else 0.0
        c_k = A_of_k(k) * step_scale * gated
        buf[k] = c_k
        buf[-k] = -c_k
    buf[0] = 0.0
    return buf


def compress_ci_buffer(raw: CiBuffer,
                       processor: Optional[ChipProcessor] = None
                       ) -> Tuple[CiBuffer, Dict]:
    K = raw.K
    if processor is None or processor.max_K < 2 * K + 1:
        processor = ChipProcessor(max_K=2 * K + 1)
    signal = raw.data.astype(float)
    kept, S, H, m, conv = processor.process(signal)
    comp = CiBuffer(K)
    for i, v in enumerate(conv):
        comp.data[i] = np.float32(v)
    return comp, dict(S=S, H=H, m=m,
                      n_kept=len(kept), n_total=len(conv))


# ============================================================
#  Build the ColorMöbiusGate LUT
# ============================================================
def build_color_lut(K: int = 256,
                    logic_gate: str = 'log') -> np.ndarray:
    """
    For each luminance level v ∈ [0, 255], build a transformed RGB
    triple by running the ColorMöbiusGate on the RGB signal
    (v, v, v).

    Returns
    -------
    lut : uint8 array of shape (256, 3)
    """
    gate = ColorMobiusGate(K=K, logic_gate=logic_gate,
                           use_elliptic=True)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for v in range(256):
        lut[v] = gate.transform_color((v, v, v))
    return lut


def build_color_lut_chromatic(K: int = 256,
                              logic_gate: str = 'log'
                              ) -> np.ndarray:
    """
    Build a 3-channel chromatic LUT:
        R-channel LUT: gate( (v, 0, 0) )[0]
        G-channel LUT: gate( (0, v, 0) )[1]
        B-channel LUT: gate( (0, 0, v) )[2]
    """
    gate = ColorMobiusGate(K=K, logic_gate=logic_gate,
                           use_elliptic=True)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for v in range(256):
        lut[v, 0] = gate.transform_color((v,   0,   0))[0]
        lut[v, 1] = gate.transform_color((0,   v,   0))[1]
        lut[v, 2] = gate.transform_color((0,   0,   v))[2]
    return lut


# ============================================================
#  Elliptic projection
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    u = (x / PI) % 1.0
    v = (y / E) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


# ============================================================
#  Reference fragment shader (Python)
# ============================================================
def shade_fragment(buf: CiBuffer, lut: np.ndarray,
                   i: int, j: int, W: int, H: int
                   ) -> Tuple[float, float, float]:
    K = buf.K
    x = i / max(W - 1, 1)
    y = j / max(H - 1, 1)

    ku = int(x * (2 * K)) - K
    kv = int(y * (2 * K)) - K
    ku = ku if ku != 0 else 1
    kv = kv if kv != 0 else 1

    cu = buf[ku]
    cv = buf[kv]
    Pi = elliptic_projection(x * PI, y * E)

    # --- LUT-sampled color channels ---
    # map the |Ci| samples and Π into LUT index space [0, 255]
    idx_u = int(np.clip((math.tanh(cu * Pi) * 0.5 + 0.5) * 255, 0, 255))
    idx_v = int(np.clip((math.tanh(cv * Pi) * 0.5 + 0.5) * 255, 0, 255))
    idx_p = int(np.clip(Pi * 255, 0, 255))

    r = lut[idx_u, 0] / 255.0
    g = lut[idx_v, 1] / 255.0
    b = lut[idx_p, 2] / 255.0
    return r, g, b


def render_image(buf: CiBuffer, lut: np.ndarray,
                 W: int, H: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.float32)
    for j in range(H):
        for i in range(W):
            img[j, i] = shade_fragment(buf, lut, i, j, W, H)
    return img


# ============================================================
#  GLSL shader emitter
# ============================================================
GLSL_TEMPLATE = r"""
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

// ---- |Ci| buffer (chip-compressed) ----
layout(std430, binding = 0) readonly buffer CiBlock {
    float c[2 * K + 1];
};

// ---- ColorMöbiusGate LUT ----
layout(std430, binding = 1) readonly buffer LutBlock {
    uint lut[256];        // packed RGB as 0x00BBGGRR
};

layout(binding = 2, rgba32f) uniform image2D out_image;

// ---- fixed n = 4 block, computed on the host and pasted here ----
const int   K        = __K__;
const float A_4      = __A_4__;
const float DA_4     = __DA_4__;
const float DC_A_4   = __DC_A_4__;
const float PI       = 3.141592653589793;
const float E        = 2.718281828459045;

// ---- elliptic projection Π(x, y) ∈ [0, 1] ----
float elliptic_projection(float x, float y) {
    float u = mod(x / PI, 1.0);
    float v = mod(y / E,  1.0);
    return 0.5 * (1.0 + cos(2.0 * PI * u) * cos(2.0 * PI * v));
}

// ---- unpack a LUT entry (0x00BBGGRR) into normalized RGB ----
vec3 unpack_lut(uint packed) {
    float r = float( packed        & 0xFFu) / 255.0;
    float g = float((packed >>  8) & 0xFFu) / 255.0;
    float b = float((packed >> 16) & 0xFFu) / 255.0;
    return vec3(r, g, b);
}

void main() {
    ivec2 gid  = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(out_image);
    if (gid.x >= size.x || gid.y >= size.y) return;

    float x = float(gid.x) / float(size.x - 1);
    float y = float(gid.y) / float(size.y - 1);

    // ---- O(1) |Ci| access ----
    int ku = int(x * float(2 * K)) - K;
    int kv = int(y * float(2 * K)) - K;
    ku = (ku == 0) ? 1 : ku;
    kv = (kv == 0) ? 1 : kv;
    ku = clamp(ku, -K, K);
    kv = clamp(kv, -K, K);

    float cu = c[K + ku];
    float cv = c[K + kv];

    // ---- elliptic projection ----
    float Pi = elliptic_projection(x * PI, y * E);

    // ---- LUT indices ----
    int idx_u = int(clamp((tanh(cu * Pi) * 0.5 + 0.5) * 255.0, 0.0, 255.0));
    int idx_v = int(clamp((tanh(cv * Pi) * 0.5 + 0.5) * 255.0, 0.0, 255.0));
    int idx_p = int(clamp(Pi * 255.0, 0.0, 255.0));

    // ---- sample the ColorMöbiusGate LUT ----
    vec3 col_u = unpack_lut(lut[idx_u]);
    vec3 col_v = unpack_lut(lut[idx_v]);
    vec3 col_p = unpack_lut(lut[idx_p]);

    // ---- channel assignment ----
    vec3 rgb;
    rgb.r = col_u.r;
    rgb.g = col_v.g;
    rgb.b = col_p.b;

    imageStore(out_image, gid, vec4(rgb, 1.0));
}
"""


def emit_glsl(K: int) -> str:
    return (GLSL_TEMPLATE
            .replace("__K__",      str(K))
            .replace("__A_4__",    f"{A_4:.10f}")
            .replace("__DA_4__",   f"{DA_4:.10f}")
            .replace("__DC_A_4__", f"{DC_A_4:.10f}"))


def pack_lut_for_gpu(lut: np.ndarray) -> np.ndarray:
    """Pack (256, 3) uint8 into uint32 words: 0x00BBGGRR."""
    out = np.zeros(256, dtype=np.uint32)
    for i in range(256):
        r = int(lut[i, 0])
        g = int(lut[i, 1])
        b = int(lut[i, 2])
        out[i] = (b << 16) | (g << 8) | r
    return out


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("GPU graphics  ·  chip compression + ColorMöbiusGate LUT")
    print("=" * 78)
    print(f"  a = 1/(π − e)  = {A_CONST:.8f}")
    print()

    # ---------- fixed n = 4 block ----------
    print("--- fixed n = 4 block ---")
    print(f"  A(4)            = {A_4:.10f}")
    print(f"  ΔA(4) forward   = {DA_4:.10f}")
    print(f"  Δ_c A(4) central= {DC_A_4:.10f}   (≈ π)")
    print()

    # ---------- raw |Ci| ----------
    K = K_DEFAULT
    t0 = time.perf_counter()
    raw = build_ci_buffer(K, n=N_FIXED)
    t_raw = (time.perf_counter() - t0) * 1e3
    print(f"--- raw |Ci|  ·  K = {K} ---")
    print(f"  build time      = {t_raw:.2f} ms")
    print(f"  length          = {len(raw.data)} float32 "
          f"= {raw.data.nbytes} bytes")
    print(f"  buf[1]          = {raw[1]:+.6f}")
    print(f"  buf[K]          = {raw[K]:+.6f}")
    print(f"  buf.data[-1]    = {raw.data[-1]:+.6f}")
    print()

    # ---------- chip compression ----------
    print("--- chip compression pipeline  ·  O(K log K) ---")
    t0 = time.perf_counter()
    comp, rep = compress_ci_buffer(raw, ChipProcessor(max_K=2 * K + 1))
    t_comp = (time.perf_counter() - t0) * 1e3
    print(f"  compress time   = {t_comp:.2f} ms")
    print(f"  supertrace S    = {rep['S']:+.6f}")
    print(f"  entropy H       = {rep['H']:.6f}")
    print(f"  mass m          = {rep['m']:.6f}")
    print(f"  kept            = {rep['n_kept']} / {rep['n_total']} "
          f"({100 * rep['n_kept'] / rep['n_total']:.1f} %)")
    print()

    # ---------- ColorMöbiusGate LUT ----------
    print("--- ColorMöbiusGate LUT  ·  chromatic (256 entries) ---")
    t0 = time.perf_counter()
    lut = build_color_lut_chromatic(K=K, logic_gate='log')
    t_lut = (time.perf_counter() - t0) * 1e3
    print(f"  build time      = {t_lut:.2f} ms")
    print(f"  LUT shape       = {lut.shape}  ({lut.nbytes} bytes)")
    print(f"  sample entries:")
    for v in [0, 32, 64, 128, 192, 255]:
        print(f"    LUT[{v:>3}]  = "
              f"({lut[v,0]:>3}, {lut[v,1]:>3}, {lut[v,2]:>3})")
    print()

    # ---------- render ----------
    W, H = 96, 96
    print(f"--- reference render  ·  {W}×{H} ---")
    t0 = time.perf_counter()
    img = render_image(comp, lut, W, H)
    t_render = (time.perf_counter() - t0) * 1e3
    per_frag = t_render / (W * H)
    print(f"  total           = {t_render:.2f} ms")
    print(f"  per fragment    = {per_frag * 1e3:.4f} μs")
    print(f"  throughput      = {W * H / (t_render * 1e-3):.0f} frag/s")
    print()

    # ---------- GLSL ----------
    glsl = emit_glsl(K)
    print("--- GLSL compute shader  ·  first 24 lines ---")
    for line in glsl.splitlines()[:24]:
        print(f"  {line}")
    print(f"  ...  ({len(glsl.splitlines())} lines total)")
    print()

    # ---------- GPU payload sizes ----------
    packed_lut = pack_lut_for_gpu(lut)
    print("--- GPU payloads ---")
    print(f"  |Ci| buffer     : {comp.data.nbytes} bytes "
          f"({len(comp.data)} float32)")
    print(f"  LUT buffer      : {packed_lut.nbytes} bytes "
          f"({len(packed_lut)} uint32)")
    print(f"  total           : "
          f"{comp.data.nbytes + packed_lut.nbytes} bytes")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  Möbius sieve            O(K log log K)   once (build)")
    print("  raw |Ci| build          O(K)             once")
    print("  chip compression        O(K log K)       once")
    print("  ColorMöbiusGate LUT     O(256 · K log K) once")
    print("  buf[k] access           O(1)             per fragment")
    print("  lut[i] access           O(1)             per fragment")
    print("  fragment shade          O(1)             per fragment")
    print("  render W×H              O(W·H)")

    # ---------- plot ----------
    if HAS_MPL:
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig)

        # (a) A(k) and ΔA(k)
        ax = fig.add_subplot(gs[0, 0])
        ks = np.arange(1, 25)
        ax.plot(ks, [A_of_k(k) for k in ks], "o-",
                color="#2c3e50", label="A(k)")
        ax.bar(ks + 0.15,
               [finite_step_k(int(k), "forward") for k in ks],
               width=0.3, color="#e67e22", alpha=0.75,
               label="ΔA(k)")
        ax.axvline(N_FIXED, color="#e74c3c", ls=":", label="n = 4")
        ax.set_title("A(k) = 10·a·sin(k²°)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (b) |Ci| raw vs compressed
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(np.arange(-K, K + 1), raw.data,
                color="#3a7bd5", lw=0.6, label="raw |Ci|")
        ax.plot(np.arange(-K, K + 1), comp.data,
                color="#e74c3c", lw=0.6, alpha=0.75,
                label="compressed")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"|Ci|  ·  {len(raw.data)} slots")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (c) LUT channels
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(lut[:, 0], color="#e74c3c", lw=1.4, label="R LUT")
        ax.plot(lut[:, 1], color="#2ecc71", lw=1.4, label="G LUT")
        ax.plot(lut[:, 2], color="#3498db", lw=1.4, label="B LUT")
        ax.set_xlabel("input luminance v")
        ax.set_ylabel("output value")
        ax.set_title("ColorMöbiusGate LUT (per channel)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (d) rendered image
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(img, origin="lower")
        ax.set_title(f"Render {W}×{H}  ·  LUT-sampled")

        # (e) Π(x, y)
        ax = fig.add_subplot(gs[1, 1])
        xs = np.linspace(0, PI, 64)
        ys = np.linspace(0, E, 64)
        XX, YY = np.meshgrid(xs, ys)
        PU = (XX / PI) % 1.0
        PV = (YY / E) % 1.0
        PP = 0.5 * (1 + np.cos(2 * np.pi * PU) * np.cos(2 * np.pi * PV))
        ax.imshow(PP, origin="lower", extent=[0, PI, 0, E],
                  cmap="inferno", aspect="auto")
        ax.set_title("Π(x, y)  ·  Figure 3.5")

        # (f) 3D vertex grid using the compressed |Ci|
        ax = fig.add_subplot(gs[1, 2], projection="3d")
        W3 = H3 = 32
        Xs = np.zeros((W3, H3))
        Ys = np.zeros((W3, H3))
        Zs = np.zeros((W3, H3))
        for jj in range(H3):
            for ii in range(W3):
                x = ii / (W3 - 1)
                y = jj / (H3 - 1)
                k = int((x - 0.5) * 2 * K) or 1
                Xs[ii, jj] = x
                Ys[ii, jj] = y
                Zs[ii, jj] = A_4 + DA_4 * comp[k] * 0.1
        ax.plot_surface(Xs, Ys, Zs, cmap="viridis",
                        alpha=0.9, linewidth=0)
        ax.set_title("vertex grid  z = A(4) + ΔA(4)·c[k]")

        plt.suptitle(
            "GPU graphics  ·  chip compression + ColorMöbiusGate LUT  ·  "
            "A(k) = 10·a·sin(k²°)",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print()
    print("Done.")


if __name__ == "__main__":
    demo()