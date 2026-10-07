#!/usr/bin/env python3
"""
spinor_propagation.py
=====================

Propagating signals with:

  • 6-vertex configuration    →  symmetric   O(K log log K)   linear sieve
  • 6D spinor Z_ij(t)          →  asymmetric  O(K log K)       segmented sieve
  • electron probability cloud  |ψ|² = |E + iB|²
  • packets as SAT envelopes    q(a) = a² + a  mod K,  μ(q) ≠ 0

Pipeline
--------
    1. linear sieve          O(K log log K)      once, symmetric path
    2. segmented sieve       O(K log K)          once, asymmetric path
    3. 6-vertex config       O(K log log K)      per frame, symmetric
    4. 6D spinor Z_ij(t)     O(K log K)          per frame, asymmetric
    5. SAT envelope packets  O(K)                 per frame
    6. Maxwell fields        O(K · N_t)           per frame
    7. electron cloud |ψ|²   O(K)                 per frame
    8. fiber propagation     O(K · W)             per frame
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Tuple

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
ALPHA_SYM   = 1.0 / (PI - E)                # ≈ 2.362
ALPHA_ASYM  = 0.3628
NORM        = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
DENSITY     = 6.0 / (PI * PI)               # ≈ 0.6079
W1          = PI
W2          = E

K_DEFAULT   = 128
N_VERT      = 6
N_EVENTS    = 256
BARRIER_W   = 31


# ============================================================
#  Two sieves
# ============================================================
def mobius_sieve_linear(K: int) -> np.ndarray:
    """Linear sieve  ·  O(K log log K)  ·  symmetric path."""
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes: List[int] = []
    is_comp = np.zeros(K + 1, dtype=bool)
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


def mobius_sieve_segmented(K: int) -> np.ndarray:
    """Segmented sieve  ·  O(K log K)  ·  asymmetric path."""
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    mu[1] = 1
    limit = int(math.isqrt(K)) + 1
    base_primes: List[int] = []
    is_comp = np.zeros(limit + 1, dtype=bool)
    for i in range(2, limit + 1):
        if not is_comp[i]:
            base_primes.append(i)
            for j in range(i * i, limit + 1, i):
                is_comp[j] = True
    for p in base_primes:
        mu[p::p] = -mu[p::p]
        p2 = p * p
        if p2 <= K:
            mu[p2::p2] = 0
    return mu


# ============================================================
#  Elliptic projection  ·  Figure 3.5
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


def elliptic_projection_array(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    u = (x / W1) % 1.0
    v = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * PI * u) * math.cos(2 * PI * v))


# ============================================================
#  Levi-Civita rank-6
# ============================================================
def _levi_civita_terms(d: int) -> List[Tuple[Tuple[int, ...], int]]:
    out = []
    for perm in permutations(range(d)):
        inv = sum(1 for i in range(d) for j in range(i + 1, d)
                  if perm[i] > perm[j])
        out.append((perm, (-1) ** inv))
    return out


TERMS_6 = _levi_civita_terms(6)             # 720 terms


def levi_civita_6(z: np.ndarray) -> complex:
    total = 0.0 + 0.0j
    for perm, sign in TERMS_6:
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= z[k, c]
        total += sign * prod
    return total


# ============================================================
#  1.  6-vertex configuration  ·  symmetric  O(K log log K)
# ============================================================
@dataclass
class SixVertexConfig:
    """
    The symmetric path.

    Uses the linear sieve (O(K log log K)) to place six vertices on
    the elliptic torus, compute their amplitude envelope, and
    contract them by the rank-6 Levi-Civita tensor.

    Everything here is deterministic: the same K always gives the
    same V6, the same row norms, the same Π_6_ref.
    """
    K: int
    mu: np.ndarray                       # linear sieve
    positions: np.ndarray                # (6, 2)
    amplitudes: np.ndarray               # (6,)
    V6: np.ndarray                       # (6, 6) complex
    Pi6_ref: complex
    phase_ref: float
    amplitude_profile: np.ndarray        # (K,)  — per-slot envelope

    @classmethod
    def build(cls, K: int, seed: int = 42) -> "SixVertexConfig":
        mu = mobius_sieve_linear(K)

        # --- 6 vertices on a ring, gated by μ(k) ≠ 0 ---
        kept = np.array([k for k in range(1, K + 1) if mu[k] != 0])
        rng = np.random.default_rng(seed)
        angles = np.linspace(0.0, 2 * PI, N_VERT, endpoint=False)
        R = 0.8
        positions = np.column_stack([R * np.cos(angles),
                                     R * np.sin(angles)])

        # --- V6 matrix on the 6×6 block ---
        V = (rng.standard_normal((6, 6))
             + 1j * rng.standard_normal((6, 6)))
        V = V / (np.max(np.abs(V)) + 1e-12) * 0.5
        V = V - V.mean()
        row_norms = np.linalg.norm(V, axis=1)
        row_norms = row_norms / (row_norms.max() + 1e-12)

        Pi6_ref = levi_civita_6(V)
        phase_ref = math.atan2(Pi6_ref.imag, Pi6_ref.real)

        # --- per-slot amplitude envelope over the K slots ---
        # Each kept slot k gets amplitude = row_norm[k mod 6] *
        # Π(x_k, y0)  — a deterministic envelope sampled from the
        # six amplitudes modulated by the elliptic projection
        x = np.linspace(0, W1, K)
        profile = elliptic_projection_array(x, y0=0.5 * W2)
        amp_profile = np.zeros(K)
        for idx, k in enumerate(kept):
            amp_profile[k - 1] = (row_norms[(k - 1) % 6]
                                   * profile[k - 1])
        return cls(
            K=K, mu=mu, positions=positions,
            amplitudes=row_norms, V6=V,
            Pi6_ref=Pi6_ref, phase_ref=phase_ref,
            amplitude_profile=amp_profile,
        )

    # ---- cost: O(K) per call, since the sieve is already built ----
    def apply_to(self, channel: np.ndarray) -> np.ndarray:
        """
        Modulate a 1D channel by the 6-vertex envelope.

        Returns the symmetrically-modulated channel:
            channel * amplitude_profile (both length K)
        """
        return channel * self.amplitude_profile


# ============================================================
#  2.  6D spinor dynamics  ·  asymmetric  O(K log K)
# ============================================================
@dataclass
class Spin6D:
    """
    The asymmetric path.

    Uses the segmented sieve (O(K log K)) to evolve a 6×6 spinor
    Z_ij(t) under a chirped phase.  The spinor is gated by μ(q) ≠ 0
    at the quadratic addresses q = a² + a mod K.

    The asymmetry comes from the chirp:
        θ(t) = a·t² + b·t + c        with a = 1e-4
    """
    K: int
    mu: np.ndarray                       # segmented sieve
    a: float = 1e-4
    b: float = 0.0
    c: float = 0.0

    @classmethod
    def build(cls, K: int,
              a: float = 1e-4, b: float = 0.0, c: float = 0.0
              ) -> "Spin6D":
        mu = mobius_sieve_segmented(K)
        return cls(K=K, mu=mu, a=a, b=b, c=c)

    def chirp(self, t: float) -> float:
        return self.a * t * t + self.b * t + self.c

    # ---- O(K log K) per call: the segmented sieve cost is the log factor ----
    def evolve(self, t: float, V6: np.ndarray) -> np.ndarray:
        """
        Produce the 6×6 spinor at time t:

            Z_ij(t) = V6_ij · exp(i · θ(t) · μ-shape_ij)

        The μ-shape is a small deterministic pattern from the
        segmented sieve — the asymmetric modulation that makes
        the spin 'creative' rather than fixed.
        """
        theta = self.chirp(t)
        # per-entry phase jitter driven by the segmented sieve
        phase = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                k = (i * 6 + j) % self.K
                phase[i, j] = math.cos(theta
                                       + self.mu[k + 1] * 0.1)
        return V6 * np.exp(1j * theta * phase)

    def contract_and_project(self, Z: np.ndarray) -> Dict:
        """
        Rank-6 contraction + 6D→3D reduction.

            Π6 = Π_6(Z)
            z3 = [Z_00, Z_11, Z_22] · e^{i Π6 / 6}
        """
        Pi6 = levi_civita_6(Z)
        phase = math.atan2(Pi6.imag, Pi6.real)
        diag3 = np.array([Z[0, 0], Z[1, 1], Z[2, 2]], dtype=complex)
        z3 = diag3 * np.exp(1j * phase / 6.0)
        return dict(Pi6=Pi6, z3=z3)


# ============================================================
#  3.  SAT envelope packets
# ============================================================
def quadratic_address(a: int, A: int = 1, B: int = 1, C: int = 0,
                      K: int = 32) -> int:
    return (A * a * a + B * a + C) % K


@dataclass
class SATEnvelope:
    """
    A packet: envelope of |amplitude| over quadratic addresses
    gated by μ(q) ≠ 0.
    """
    K: int
    env: np.ndarray                      # (K,)
    q_of_a: Dict[int, int]               # a → q(a)
    n_packets: int

    def as_channel(self) -> np.ndarray:
        return self.env.copy()


def build_sat_envelope(amplitudes: np.ndarray,
                       K: int,
                       mu: np.ndarray,
                       A: int = 1, B: int = 1, C: int = 0
                       ) -> SATEnvelope:
    """
    Each amplitude event a is mapped to q(a) = a² + a mod K and
    accumulated iff μ(q) ≠ 0.  Returns the envelope over q ∈ [0, K).
    """
    env = np.zeros(K, dtype=float)
    q_of_a: Dict[int, int] = {}
    for a, v in enumerate(amplitudes):
        q = quadratic_address(a, A, B, C, K)
        q_of_a[a] = q
        if mu[q] != 0:
            env[q] += abs(v)
    return SATEnvelope(K=K, env=env, q_of_a=q_of_a,
                       n_packets=int(np.count_nonzero(env)))


# ============================================================
#  4.  Maxwell fields from the Möbius spectral sum  ζ(t)
# ============================================================
def zeta_mobius(t: float, mu: np.ndarray, K: int,
                alpha: float = ALPHA_SYM) -> complex:
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    theta = t * n / alpha
    return complex(np.sum(m * np.cos(theta)),
                   np.sum(m * np.sin(theta)))


def maxwell_fields(t_vals: np.ndarray, mu: np.ndarray, K: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    E(t) = Re ζ(t), B(t) = Im ζ(t), ∇·E — the signed charge density.
    """
    E_v = np.empty(len(t_vals))
    B_v = np.empty(len(t_vals))
    div_v = np.empty(len(t_vals))
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    signs = np.where(n % 2 == 0, 1.0, -1.0)
    for i, t in enumerate(t_vals):
        theta = t * n / ALPHA_SYM
        E_v[i] = float(np.sum(m * np.cos(theta)))
        B_v[i] = float(np.sum(m * np.sin(theta)))
        div_v[i] = float(np.sum(signs * m * np.cos(theta)))
    return E_v, B_v, div_v


def electron_probability_cloud(E_v: np.ndarray,
                                B_v: np.ndarray,
                                div_v: np.ndarray
                                ) -> Dict[str, np.ndarray]:
    """
    |ψ|² = |E + iB|²  — the electron probability cloud.

    Also returns the charge density (∇·E) and the phase arg(E + iB).
    """
    psi = E_v + 1j * B_v
    prob = np.abs(psi) ** 2
    phase = np.angle(psi)
    return dict(prob=prob, phase=phase, charge=div_v,
                E=E_v, B=B_v)


# ============================================================
#  5.  Fiber propagation  ·  exponential kernel
# ============================================================
def barrier_kernel(width: int, alpha: float = ALPHA_SYM) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    ker = np.exp(-alpha * np.abs(x))
    return ker / ker.sum()


def propagate_fiber(field: np.ndarray, width: int = BARRIER_W
                    ) -> np.ndarray:
    """
    Apply the exponential barrier kernel exp(−α|z|) via FFT.
    """
    ker = barrier_kernel(width)
    L = len(field)
    N = 1 << (2 * L - 1).bit_length()
    sig_pad = np.pad(field, (0, N - L))
    ker_pad = np.pad(ker, (0, N - len(ker)))
    y = np.fft.ifft(np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:L]
    return np.real(y)


# ============================================================
#  Full pipeline
# ============================================================
def simulate(K: int = K_DEFAULT,
             N_events: int = N_EVENTS,
             N_steps: int = 8,
             seed_vertices: int = 42) -> Dict:
    print("=" * 82)
    print("Spinor propagation  ·  6-vertex (sym) × 6D spin (asym) × SAT envelopes")
    print("=" * 82)
    print(f"  α_sym  = 1/(π − e)  = {ALPHA_SYM:.6f}")
    print(f"  α_asym = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K                   = {K}")
    print(f"  N_events            = {N_events}")
    print(f"  N_steps             = {N_steps}")
    print()

    # ---- 1. build the two sieves once ----
    t0 = time.perf_counter()
    sv  = SixVertexConfig.build(K, seed=seed_vertices)
    t_sym = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    sp  = Spin6D.build(K, a=1e-4)
    t_asym = (time.perf_counter() - t0) * 1e3
    print(f"[1] symmetric  6-vertex build   {t_sym:8.2f} ms  (O(K log log K))")
    print(f"[2] asymmetric 6D spin build    {t_asym:8.2f} ms  (O(K log K))")
    print(f"      V6 ||  = {np.linalg.norm(sv.V6):.4f}")
    print(f"      Π6_ref = {sv.Pi6_ref.real:+.4e} "
          f"{sv.Pi6_ref.imag:+.4e}i")
    print(f"      |Π6_ref| = {abs(sv.Pi6_ref):.4e}")
    print()

    # ---- 2. per-step evolution ----
    t_vals = np.linspace(0.0, 2 * PI * ALPHA_SYM, N_events)
    history: List[Dict] = []
    for step in range(N_steps):
        t_anchor = step * (2 * PI * ALPHA_SYM) / N_steps

        # --- asymmetric: 6D spinor at t_anchor ---
        Z6 = sp.evolve(t_anchor, sv.V6)
        spin = sp.contract_and_project(Z6)

        # --- symmetric: 6-vertex envelope modulating the spin ---
        amplitudes = np.abs(spin["z3"])                 # length 3
        tile = int(np.ceil(N_events / len(amplitudes)))
        amps_stream = np.tile(amplitudes, tile)[:N_events]

        # resample the symmetric envelope to N_events before multiply
        profile_resampled = np.interp(
            np.linspace(0, 1, N_events),
            np.linspace(0, 1, K),
            sv.amplitude_profile,
        )
        amps_stream = amps_stream * profile_resampled

        # --- SAT envelope packet ---
        packet = build_sat_envelope(amps_stream, K, sv.mu)

        # --- Maxwell fields + electron cloud ---
        E_v, B_v, div_v = maxwell_fields(t_vals, sv.mu, K)
        cloud = electron_probability_cloud(E_v, B_v, div_v)

        # --- pack the packet through the fiber ---
        packet_1d   = packet.as_channel()               # length K
        packet_prop = propagate_fiber(packet_1d, width=BARRIER_W)

        history.append(dict(
            step=step, t_anchor=t_anchor,
            Z6=Z6, spin=spin,
            amplitudes=amplitudes,
            packet=packet, packet_prop=packet_prop,
            cloud=cloud,
            E=E_v, B=B_v, div=div_v,
        ))

        
    print(f"[3] {N_steps} evolution steps produced")
    print(f"      |Π6| range      : "
          f"[{min(abs(h['spin']['Pi6']) for h in history):.3e}, "
          f"{max(abs(h['spin']['Pi6']) for h in history):.3e}]")
    print(f"      packet density  : "
          f"{np.mean([h['packet'].n_packets for h in history]) / K:.4f} "
          f"(6/π² = {DENSITY:.4f})")
    print()

    # ---- 3. aggregate statistics ----
    first = history[0]
    last  = history[-1]
    print("--- evolution summary ---")
    print(f"  step 0 :  |Π6| = {abs(first['spin']['Pi6']):.4e}   "
          f"packets = {first['packet'].n_packets}")
    print(f"  step {N_steps-1} :  "
          f"|Π6| = {abs(last['spin']['Pi6']):.4e}   "
          f"packets = {last['packet'].n_packets}")
    print()

    # ---- 4. complexity ----
    print("--- complexity ---")
    print("  linear sieve  (sym)          O(K log log K)   once")
    print("  segmented     (asym)         O(K log K)       once")
    print("  6-vertex apply (sym)         O(K)             per frame")
    print("  6D spin evolve (asym)        O(K log K)       per frame")
    print("  SAT envelope                 O(N_events)      per frame")
    print("  Maxwell + electron cloud     O(K · N_events)")
    print("  fiber propagation            O(K log K)       per frame")
    print()

    return dict(
        sv=sv, sp=sp, history=history,
        t_vals=t_vals, t_sym=t_sym, t_asym=t_asym,
    )


# ============================================================
#  Plot
# ============================================================
def plot_sim(res: Dict) -> None:
    if not HAS_MPL:
        return

    sv = res["sv"]
    sp = res["sp"]
    h = res["history"]
    K = sv.K

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

    # (a) 6-vertex amplitudes
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(6), sv.amplitudes, color="#16a085",
           edgecolor="k", width=0.6)
    ax.set_xlabel("vertex"); ax.set_ylabel("amplitude")
    ax.set_title(f"6-vertex amplitudes (sym)\n"
                 f"|Π6_ref| = {abs(sv.Pi6_ref):.3e}")
    ax.grid(alpha=0.3)

    # (b) symmetric envelope profile (linear sieve)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sv.amplitude_profile[:256], color="#16a085", lw=1.2)
    ax.set_xlabel("slot k"); ax.set_ylabel("amplitude")
    ax.set_title("Symmetric envelope  (linear sieve)\n"
                 "O(K log log K)")
    ax.grid(alpha=0.3)

    # (c) asymmetric envelope profile (segmented sieve)
    ax = fig.add_subplot(gs[0, 2])
    mu_seg = sp.mu[1:257]
    ax.plot(mu_seg, "o-", color="#c0392b", ms=3)
    ax.set_xlabel("slot k"); ax.set_ylabel("μ(k)")
    ax.set_title("Asymmetric μ pattern  (segmented sieve)\n"
                 "O(K log K)")
    ax.grid(alpha=0.3)

    # (d) 6D spinor phases across steps
    ax = fig.add_subplot(gs[1, 0])
    for i, h_i in enumerate(h):
        diag3 = h_i["spin"]["z3"]
        ax.plot(np.arange(3) + i * 3, np.angle(diag3),
                "o-", label=f"step {i}")
    ax.set_xlabel("mode index"); ax.set_ylabel("arg z3")
    ax.set_title("6D → 3D phases across steps")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) SAT envelope packet (first step)
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(np.arange(K), h[0]["packet"].env, color="#8e44ad",
           width=0.85)
    ax.set_xlabel("q = a² + a mod K")
    ax.set_ylabel("env[q]")
    ax.set_title(f"SAT envelope packet  ·  "
                 f"occupied = {h[0]['packet'].n_packets}/{K}")
    ax.grid(alpha=0.3)

    # (f) packet propagation
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(h[0]["packet"].as_channel(), color="#8e44ad", lw=1.0,
            alpha=0.7, label="input packet")
    ax.plot(h[0]["packet_prop"], color="#e74c3c", lw=1.4,
            label="propagated")
    ax.set_xlabel("slot q"); ax.set_ylabel("amplitude")
    ax.set_title("Packet through the fiber  (exp(−α|z|))")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (g) Maxwell fields
    ax = fig.add_subplot(gs[2, :2])
    t_vals = res["t_vals"]
    h0 = h[0]
    ax.plot(t_vals, h0["E"], color="#2ecc71", lw=1.2,
            label="E(t) = Re ζ(t)")
    ax.plot(t_vals, h0["B"], color="#3498db", lw=1.0,
            alpha=0.75, label="B(t) = Im ζ(t)")
    ax.plot(t_vals, h0["div"], color="#e74c3c", lw=0.9,
            alpha=0.65, label="∇·E  charge density")
    ax.set_xlabel("t"); ax.set_ylabel("field")
    ax.set_title("Maxwell fields from the Möbius spectral sum")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (h) electron probability cloud
    ax = fig.add_subplot(gs[2, 2])
    prob = h0["cloud"]["prob"]
    ax.plot(t_vals, prob, color="#16a085", lw=1.4)
    ax.fill_between(t_vals, 0, prob, color="#16a085", alpha=0.2)
    ax.set_xlabel("t"); ax.set_ylabel("|ψ|²")
    ax.set_title(f"Electron probability cloud\n"
                 f"|ψ|² = |E + iB|²  ·  mean = {prob.mean():.3f}")
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Spinor propagation  ·  6-vertex (O(K log log K))  ×  "
        "6D spin (O(K log K))  ·  SAT envelope packets",
        fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    res = simulate(K=K_DEFAULT, N_events=N_EVENTS, N_steps=8,
                   seed_vertices=42)
    plot_sim(res)