#!/usr/bin/env python3
"""
wire_throughput_envelope.py

Signal transmission through wires using **quadratic envelopes** on a
**1D throughput path**.

Model
-----
A wire is a 1D channel of length L, discretised into K segments.
Each segment has a throughput τ_k ∈ [0, 1] computed from the elliptic
projection Π (Figure 3.5) and the adaptive-Simpson entropy scaling of
throughput-supertrace.py.

The signal on the wire is carried as a **quadratic envelope**:

        q(a) = A·a² + B·a + C  mod K          (SAT-mobius-gate.py)
        env[q] += penalty(a)                   (envelope over addresses)

where each voltage/current event a is gated by the Möbius sieve
μ(q(a)) ≠ 0, so only square‑free addresses survive.

Transmission along the wire is a **throughput‑weighted shift**:

        sig_k ← τ_k · sig_{k−1} + (1 − τ_k) · env_k

so the wire attenuates the envelope segment by segment.  The
M‑matrix in throughput-supertrace.py becomes the 2×2 per‑segment
transfer matrix

        M_k = [[Re env_k,  −Im env_k],
               [Im env_k,   Re env_k]]

and the supertrace S = Σ (−1)^k tr(M_k) gives the wire's end‑to‑end
invariant, from which entropy H and invariant mass m follow.

All per‑segment operations are O(K); the sieve is O(K log log K).
"""

from __future__ import annotations

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_ASYM = 0.3628
DENSITY    = 6.0 / (PI * PI)         # ≈ 0.6079271018
W1         = PI
W2         = E


# ============================================================
#  1. Möbius sieve  ·  O(K log log K)
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
#  2. Figure 3.5  ·  bounded elliptic projection
# ============================================================
def elliptic_projection(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    """Π(x, y0) ∈ [0, 1] on the torus ℂ/(πℤ + eℤ)."""
    u = (x / W1) % 1.0
    v = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * u) * math.cos(2 * np.pi * v))


def throughput_1d(K: int) -> np.ndarray:
    """
    1D throughput path along the wire:
    τ_k = Π(x_k, y0) sampled at K equally‑spaced points on [0, π).
    """
    x = np.linspace(0, W1, K)
    return elliptic_projection(x, y0=0.5 * W2)


# ============================================================
#  3. Supertrace entropy of a sampled signal  (throughput‑supertrace.py)
# ============================================================
def supertrace_entropy(values: np.ndarray, alpha: float = ALPHA_SYM) -> float:
    N = len(values)
    if N == 0:
        return 0.0
    S = 0.0
    for i, v in enumerate(values):
        if (i + 1) % 2 == 0:
            S += v
        else:
            S -= v
    ratio = abs(S) / N
    if 0.0 < ratio < 1.0:
        return -alpha * ratio * math.log(ratio)
    return 0.0


# ============================================================
#  4. Adaptive Simpson with entropy scaling
# ============================================================
def adaptive_simpson(f, a, b, tol=1e-6, max_depth=20,
                     alpha_scale=1.0) -> float:
    def simpson(f, a, b):
        return (b - a) / 6.0 * (f(a) + 4.0 * f((a + b) / 2.0) + f(b))

    def recursive(a, b, fa, fm, fb, S, depth):
        m = (a + b) / 2.0
        lm = (a + m) / 2.0
        rm = (m + b) / 2.0
        flm = f(lm)
        frm = f(rm)
        S_left = simpson(f, a, m)
        S_right = simpson(f, m, b)
        S_total = S_left + S_right
        scaled_tol = tol * (1.0 / (1.0 + alpha_scale * 0.1))
        error = abs(S_total - S)
        if depth <= 0 or error < 15.0 * scaled_tol:
            return S_total + (S_total - S) / 15.0
        return (recursive(a, m, fa, flm, fm, S_left, depth - 1) +
                recursive(m, b, fm, frm, fb, S_right, depth - 1))

    fa = f(a)
    fb = f(b)
    fm = f((a + b) / 2.0)
    S = simpson(f, a, b)
    return recursive(a, b, fa, fm, fb, S, max_depth)


def integrate_with_alpha(f, a, b, tol=1e-6, use_entropy=True):
    if use_entropy:
        x_samples = np.linspace(a, b, 100)
        y_samples = np.array([f(x) for x in x_samples])
        H = supertrace_entropy(y_samples)
        H = max(0.0, min(H, 1.0))
        alpha_scale = 1.0 + 2.0 * H
    else:
        alpha_scale = 1.0
    return adaptive_simpson(f, a, b, tol=tol, alpha_scale=alpha_scale)


# ============================================================
#  5. Quadratic envelope  ·  SAT-mobius-gate.py style
# ============================================================
def quadratic_address(a: int, A: int = 1, B: int = 1, C: int = 0,
                      K: int = 32) -> int:
    return (A * a * a + B * a + C) % K


def build_quadratic_envelope(events: np.ndarray,
                             K: int,
                             mu: np.ndarray,
                             A: int = 1, B: int = 1, C: int = 0
                             ) -> tuple[np.ndarray, dict[int, int]]:
    """
    Given a stream of voltage events (one per wire segment), build the
    quadratic envelope:
        q(a) = A·a² + B·a + C mod K
        env[q] += |event[a]|   if μ(q) ≠ 0
    Returns env (length K) and the map a → q(a).
    """
    env = np.zeros(K, dtype=np.float64)
    q_of_a: dict[int, int] = {}
    for a, v in enumerate(events):
        q = quadratic_address(a, A, B, C, K)
        q_of_a[a] = q
        if q < len(mu) and mu[q] != 0:
            env[q] += abs(v)
    return env, q_of_a


# ============================================================
#  6. Per‑segment 2×2 M‑matrix  (throughput-supertrace.py)
# ============================================================
def segment_M(env_k: float, tau_k: float) -> np.ndarray:
    """
    M_k = [[Re env_k,  −Im env_k],
           [Im env_k,   Re env_k]] · τ_k

    The factor τ_k is the throughput of segment k (elliptic
    projection slice).  This keeps the transfer bounded and makes
    the supertrace sensitive to the local channel opening.
    """
    re = tau_k * env_k
    im = tau_k * math.sqrt(max(env_k, 0.0))
    return np.array([[re, -im], [im, re]])


def supertrace_of_M_list(M_list: list[np.ndarray]) -> float:
    """S = Σ_k (−1)^k · tr(M_k)."""
    S = 0.0
    for k, M in enumerate(M_list):
        sign = 1.0 if (k % 2 == 0) else -1.0
        S += sign * float(np.trace(M).real)
    return S


def entropy(S: float, N: int, alpha: float = ALPHA_SYM) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def invariant_mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  7. Wire transmission  (throughput‑weighted shift)
# ============================================================
def transmit(env: np.ndarray,
             tau: np.ndarray,
             input_signal: np.ndarray) -> np.ndarray:
    """
    Propagate the quadratic envelope down the wire.

    At each segment k:
        sig_k = τ_k · sig_{k−1} + (1 − τ_k) · env_k

    τ_k is the throughput of segment k.  Where the wire is fully
    open (τ_k ≈ 1), the previous signal dominates and the envelope
    contribution is small; where the wire is depleted (τ_k ≈ 0),
    the envelope is injected almost completely.
    """
    K = len(tau)
    sig = np.zeros(K, dtype=np.float64)
    prev = float(input_signal[0]) if len(input_signal) > 0 else 0.0
    for k in range(K):
        e = float(env[k]) if k < len(env) else 0.0
        sig[k] = tau[k] * prev + (1.0 - tau[k]) * e
        prev = sig[k]
    return sig


# ============================================================
#  8. Full simulation
# ============================================================
def simulate(K: int = 64,
             n_events: int = 128,
             quad_A: int = 1,
             quad_B: int = 1,
             quad_C: int = 0,
             seed: int = 2024):
    """
    Simulate a signal sent as a quadratic envelope through a 1D wire
    whose throughput path is given by the elliptic projection.
    """
    print("=" * 80)
    print("Signal transmission through wires  ·  quadratic envelope on 1D path")
    print("=" * 80)
    print(f"  α_sym = 1/(π − e)  = {ALPHA_SYM:.6f}")
    print(f"  K                  = {K}")
    print(f"  wire segments      = {K}")
    print(f"  events             = {n_events}")
    print(f"  quadratic address  q(a) = {quad_A}·a² + {quad_B}·a + {quad_C} mod K")
    print()

    # ---------- 1. Möbius sieve ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(K)
    t_sieve = (time.perf_counter() - t0) * 1e3
    print(f"[1] Möbius sieve      {t_sieve:.2f} ms   (O(K log log K))")

    # ---------- 2. 1D throughput path ----------
    t0 = time.perf_counter()
    tau = throughput_1d(K)
    t_tau = (time.perf_counter() - t0) * 1e3
    print(f"[2] Throughput path   {t_tau:.2f} ms   "
          f"(τ range [{tau.min():.3f}, {tau.max():.3f}])")

    # ---------- 3. Generate events ----------
    rng = np.random.default_rng(seed)
    events = rng.standard_normal(n_events) * 0.5 + 0.8 * np.sin(
        2 * np.pi * np.arange(n_events) / n_events)
    # optional DC lift so envelope is non‑negative
    events = np.abs(events)

    # ---------- 4. Quadratic envelope ----------
    t0 = time.perf_counter()
    env, q_of_a = build_quadratic_envelope(
        events, K, mu, A=quad_A, B=quad_B, C=quad_C)
    t_env = (time.perf_counter() - t0) * 1e3
    occupied = int(np.count_nonzero(env))
    print(f"[3] Quadratic envelope{t_env:6.2f} ms   "
          f"({occupied}/{K} addresses occupied, "
          f"density = {occupied/K:.3f}, expected 6/π² = {DENSITY:.3f})")

    # ---------- 5. Transmit envelope down the wire ----------
    t0 = time.perf_counter()
    sig = transmit(env, tau, input_signal=env[:1])
    t_tx = (time.perf_counter() - t0) * 1e3
    print(f"[4] Transmission      {t_tx:.2f} ms   (throughput‑weighted shift)")

    # ---------- 6. Per‑segment M‑matrices ----------
    t0 = time.perf_counter()
    M_list = [segment_M(sig[k], tau[k]) for k in range(K)]
    t_M = (time.perf_counter() - t0) * 1e3
    print(f"[5] M‑matrix build    {t_M:.2f} ms   ({K} × 2×2)")

    # ---------- 7. Supertrace and invariants ----------
    S = supertrace_of_M_list(M_list)
    H = entropy(S, K)
    m = invariant_mass(S, K)
    print(f"[6] Supertrace        S = {S:+.6f}   "
          f"H = {H:.6f}   m = {m:.6f}")

    # ---------- 8. Adaptive Simpson integrals with entropy scaling ----
    t0 = time.perf_counter()
    x_cont = np.linspace(0, 1, 200)

    def sig_fn(u: float) -> float:
        k = int(round(u * (K - 1)))
        k = max(0, min(K - 1, k))
        return float(sig[k])

    def tau_fn(u: float) -> float:
        k = int(round(u * (K - 1)))
        k = max(0, min(K - 1, k))
        return float(tau[k])

    I_sig = integrate_with_alpha(sig_fn, 0.0, 1.0, tol=1e-6)
    I_tau = integrate_with_alpha(tau_fn, 0.0, 1.0, tol=1e-6)
    t_int = (time.perf_counter() - t0) * 1e3
    print(f"[7] Adaptive Simpson  {t_int:.2f} ms   "
          f"∫sig = {I_sig:.6f}   ∫τ = {I_tau:.6f}")

    # ---------- 9. Compare to theoretical bounds ----------
    theory_amp = 2 * (6.0 / PI) * K
    theory_pow = 2 * (6.0 / PI**2) * K
    print()
    print("--- Theoretical bounds (throughput-supertrace.py) ---")
    print(f"  6‑vertex  ∫|ζ|       theory = {theory_amp:.4f}")
    print(f"  12‑vertex ∫|ζ|²      theory = {theory_pow:.4f}")
    print(f"  Our ∫ sig            = {I_sig:.4f}")
    print(f"  Our ∫ τ              = {I_tau:.4f}")
    print(f"  S / theory_amp       = {S/theory_amp:+.4e}")
    print(f"  m / theory_pow       = {m/theory_pow:+.4e}")

    # ---------- 10. Throughput per segment summary ----------
    print()
    print("--- Wire segments (first 12) ---")
    print(f"  {'k':>3s}  {'τ_k':>8s}  {'env_k':>10s}  "
          f"{'sig_k':>10s}  {'tr(M_k)':>10s}")
    for k in range(min(12, K)):
        print(f"  {k:>3d}  {tau[k]:>8.4f}  {env[k]:>10.4f}  "
              f"{sig[k]:>10.4f}  {float(np.trace(M_list[k]).real):>10.4f}")

    # ---------- 11. Plot ----------
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # (a) throughput path
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.arange(K), tau, color="#2980b9", lw=1.6)
    ax.fill_between(np.arange(K), 0, tau, color="#2980b9", alpha=0.25)
    ax.set_xlabel("wire segment k")
    ax.set_ylabel("throughput τ_k")
    ax.set_title("1D throughput path  (elliptic projection slice)")
    ax.grid(True, alpha=0.3)

    # (b) input events
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(n_events), events, color="#c0392b", lw=1.1)
    ax.set_xlabel("event index a")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Input events ({n_events})")
    ax.grid(True, alpha=0.3)

    # (c) quadratic envelope
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(K), env, color="#16a085", width=0.85)
    ax.set_xlabel("quadratic address  q = a² + a  (mod K)")
    ax.set_ylabel("env[q]")
    ax.set_title(f"Quadratic envelope  (K = {K})")
    ax.grid(True, alpha=0.3)

    # (d) transmitted signal
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(np.arange(K), sig, color="#8e44ad", lw=1.6,
            label="transmitted signal sig_k")
    ax.plot(np.arange(K), env[:K], color="#16a085", lw=1.0, ls="--",
            alpha=0.7, label="envelope env_k")
    ax.plot(np.arange(K), tau, color="#2980b9", lw=1.0, ls=":",
            alpha=0.7, label="throughput τ_k")
    ax.set_xlabel("wire segment k")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Signal transmission  ·  S = {S:+.4f}  H = {H:.4f}  "
                 f"m = {m:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) M‑matrix traces
    ax = fig.add_subplot(gs[1, 2])
    traces = [float(np.trace(M).real) for M in M_list]
    colors = ["#2ecc71" if k % 2 == 0 else "#e74c3c"
              for k in range(K)]
    ax.bar(np.arange(K), traces, color=colors, width=0.85)
    ax.set_xlabel("segment k")
    ax.set_ylabel("tr(M_k)")
    ax.set_title("M‑matrix traces  ·  green=even, red=odd")
    ax.grid(True, alpha=0.3)

    # (f) adaptive Simpson integrand (sampled)
    ax = fig.add_subplot(gs[2, :2])
    u_grid = np.linspace(0, 1, 200)
    sig_u = np.array([sig_fn(u) for u in u_grid])
    tau_u = np.array([tau_fn(u) for u in u_grid])
    ax.plot(u_grid, sig_u, color="#8e44ad", lw=1.4,
            label=f"sig(u)  ∫ = {I_sig:.4f}")
    ax.plot(u_grid, tau_u, color="#2980b9", lw=1.2, alpha=0.7,
            label=f"τ(u)  ∫ = {I_tau:.4f}")
    ax.set_xlabel("normalised wire coordinate u ∈ [0, 1]")
    ax.set_ylabel("amplitude")
    ax.set_title("Continuous view of the wire  (adaptive Simpson integrand)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (g) supertrace sign pattern
    ax = fig.add_subplot(gs[2, 2])
    signs = np.array([1.0 if k % 2 == 0 else -1.0 for k in range(K)])
    contribs = signs * np.array(traces)
    ax.bar(np.arange(K), contribs,
           color=["#2ecc71" if k % 2 == 0 else "#e74c3c"
                  for k in range(K)], width=0.85)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("segment k")
    ax.set_ylabel("(−1)^k · tr(M_k)")
    ax.set_title(f"Signed supertrace contributions  (S = {S:+.4f})")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Wire signal transmission  ·  quadratic envelope on 1D "
                 "throughput path", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # ---------- Summary ----------
    print()
    print("--- Summary ---")
    print(f"  Wire length (segments)     : {K}")
    print(f"  Throughput range           : [{tau.min():.4f}, {tau.max():.4f}]")
    print(f"  Envelope occupied          : {occupied}/{K} "
          f"(density {occupied/K:.4f})")
    print(f"  Transmitted signal range   : [{sig.min():+.4f}, {sig.max():+.4f}]")
    print(f"  Supertrace S               : {S:+.6f}")
    print(f"  Entropy H                  : {H:.6f}")
    print(f"  Invariant mass m           : {m:.6f}")
    print(f"  ∫ sig du                   : {I_sig:.6f}")
    print(f"  ∫ τ   du                   : {I_tau:.6f}")
    print()
    print("--- Complexity ---")
    print("  Möbius sieve                 O(K log log K)   once")
    print("  Throughput path              O(K)")
    print("  Quadratic envelope           O(n_events)")
    print("  Transmission (shift)         O(K)")
    print("  M‑matrix build                O(K)")
    print("  Supertrace + entropy         O(K)")
    print("  Adaptive Simpson             O(K · depth)")
    print("  ─────────────────────────────────────────")
    print(f"  Total per run                O(K log log K + K · depth)")

    return dict(
        tau=tau, env=env, sig=sig, M_list=M_list,
        S=S, H=H, m=m,
        I_sig=I_sig, I_tau=I_tau,
        theory_amp=theory_amp, theory_pow=theory_pow,
        events=events,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=64, n_events=128,
             quad_A=1, quad_B=1, quad_C=0, seed=2024)