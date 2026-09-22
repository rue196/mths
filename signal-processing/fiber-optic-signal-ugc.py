#!/usr/bin/env python3
"""
fiber_ugc_transmission.py

Signals through a fiber-optic cable using the 6-vertex configuration
as the **set signal** and the Unique Games instance as the **message**.

Pipeline
--------
    6-vertex configuration Π  →  initial optical field   (set signal)
    UGC label assignment σ(v) →  phase modulation        (message)
    Maxwell ζ(t) in Möbius basis  →  E(z, t), B(z, t)    (propagation)
    Exponential fiber kernel       →  dispersion          (channel)
    Möbius label decoder           →  recovered σ̂(v)      (receiver)

The 6-vertex configuration is the spinor projection on the first 6
of the 12 vertices in ℝ⁶ (single Levi‑Civita contraction, the 32‑bit
regime).  Its contraction scalar seeds the optical carrier, so every
transmission starts from a *deterministic* set signal.

The Unique Games instance is transmitted as a k‑level phase modulation:
each vertex label λ ∈ [0, k) is encoded as a phase shift on one of k
optical modes.  After the fiber, the receiver recovers the labels by
the same Möbius gate used in `un_q_games.py`.

The fiber channel is modelled by
        H(z, ω) = exp(−α |z|) · exp(i ω z / c)
so that the received field is the convolution of the input field with
the exponential dispersion kernel.
"""

from __future__ import annotations

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import convolve


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_ASYM = 0.3628
DENSITY    = 6.0 / (PI * PI)           # ≈ 0.6079271018
W1         = PI
W2         = E
C_LIGHT    = 1.0                        # normalized units


# ============================================================
#  1. Möbius sieve  (shared O(K log log K))
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
#  2. Figure 3.5 — bounded elliptic projection
# ============================================================
def elliptic_projection_2d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))


def elliptic_projection_1d(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


# ============================================================
#  3. 6-vertex configuration  (the SET signal)
# ============================================================
def generate_vertices(seed: int = 42) -> np.ndarray:
    """12 vertices in ℝ⁶ — the projection operator source."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((12, 6))


def levi_civita_contraction(vertices_6x6: np.ndarray) -> float:
    """
    Rank‑6 Levi‑Civita contraction on the first 6 vertices:
        Π(z) = Σ_σ ε(σ) Π_k z_{k, σ(k)}
    This is the **set signal scalar** — a deterministic invariant of
    the 6‑vertex configuration.
    """
    from itertools import permutations
    scalar = 0.0
    for perm in permutations(range(6)):
        inv = sum(1 for i in range(6) for j in range(i + 1, 6)
                  if perm[i] > perm[j])
        sign = (-1) ** inv
        prod = 1.0
        for k, c in enumerate(perm):
            prod *= vertices_6x6[k, c]
        scalar += sign * prod
    return scalar


def six_vertex_set_signal(seed: int = 42, n_modes: int = 6) -> dict:
    """
    Extract the 6‑vertex set signal from the 12‑vertex configuration:

      • take the first 6 vertices (32‑bit regime)
      • project onto 6 optical modes by the row norms
      • normalise to unit amplitude per mode
      • record the Levi‑Civita scalar as the global phase
    """
    V = generate_vertices(seed)
    V6 = V[:6, :6]                              # 6 vertices × 6 coords
    row_norms = np.linalg.norm(V6, axis=1)      # one amplitude per vertex
    row_norms = row_norms / (row_norms.max() + 1e-12)
    scalar = levi_civita_contraction(V6)
    phase = math.atan2(0.0, scalar) if scalar > 0 else math.pi
    return dict(
        vertices=V6,
        amplitudes=row_norms,
        scalar=scalar,
        phase=phase,
        n_modes=n_modes,
    )


# ============================================================
#  4. Unique Games instance  (the MESSAGE)
# ============================================================
def generate_unique_games_instance(N_left: int, N_right: int,
                                   k: int, degree: int = 3,
                                   seed: int = 2024):
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(N_left):
        js = rng.choice(N_right, size=degree, replace=False)
        for j in js:
            perm = tuple(int(x) for x in rng.permutation(k))
            edges.append((int(i), int(j), perm))
    return edges


def vertex_hash(v: int, side: str) -> float:
    h = (v * 2654435761) ^ (0 if side == "L" else 0x9E3779B9)
    return 0.5 + ((h & 0xFFFF) / 0xFFFF) * 4.5


def quadratic_address(a: int, A: int = 1, B: int = 1, C: int = 0,
                      K: int = 32) -> int:
    return (A * a * a + B * a + C) % K


def ugc_label_assignment(N_left: int, N_right: int, k: int,
                         mu: np.ndarray,
                         K_filter: int = 2 ** 10 + 1):
    """Möbius‑gated label assignment σ(v) ∈ [0, k)."""
    N_v = N_left + N_right
    scores = np.zeros((N_v, k), dtype=np.float64)
    for v in range(N_v):
        side = "L" if v < N_left else "R"
        v_local = v if v < N_left else v - N_left
        x = vertex_hash(v_local, side)
        y = vertex_hash(v_local + 1, side)
        Pi = elliptic_projection_1d(np.array([x]), y0=y)[0]
        Pi = min(max(Pi, 0.0), 1.0)
        for lam in range(k):
            a = v * k + lam
            q = quadratic_address(a, 1, 1, 0, K_filter)
            if q < len(mu) and mu[q] != 0:
                scores[v, lam] += Pi
    return np.argmax(scores, axis=1).astype(int), scores


# ============================================================
#  5. Optical modulation  (labels → phases)
# ============================================================
def encode_labels_as_phases(labels: np.ndarray, k: int,
                            set_signal: dict) -> dict:
    """
    Encode the UGC labels as phase modulations on the 6 optical modes.

    Each label λ ∈ [0, k) is mapped to a phase φ = 2π λ / k.  The
    amplitudes come from the 6‑vertex set signal, so every transmission
    starts from the same deterministic carrier.

    Returns
    -------
    E0 : initial electric field per mode
    B0 : initial magnetic field per mode
    phi : per‑vertex phase
    """
    N_v = len(labels)
    amps = set_signal["amplitudes"]
    phase0 = set_signal["phase"]
    scalar = set_signal["scalar"]

    # per‑vertex phase: label λ → 2π λ / k
    phi = 2.0 * PI * labels / k

    # distribute the 6 amplitudes across the N_v vertices
    # by repeating the 6‑mode carrier
    carrier_amp = np.tile(amps, int(np.ceil(N_v / len(amps))))[:N_v]

    # electric field: amplitude × cos(phase)
    E0 = carrier_amp * np.cos(phi + phase0)
    # magnetic field: amplitude × sin(phase) (90° lag)
    B0 = carrier_amp * np.sin(phi + phase0)

    # Levi‑Civita scalar modulates the global envelope
    envelope = math.tanh(abs(scalar))
    E0 *= envelope
    B0 *= envelope

    return dict(E0=E0, B0=B0, phi=phi,
                envelope=envelope, carrier_amp=carrier_amp)


# ============================================================
#  6. Maxwell propagation in the Möbius basis
# ============================================================
def zeta_mobius(t: float, mu: np.ndarray, K: int,
                alpha: float = ALPHA_SYM) -> complex:
    """ζ(t) = Σ_{1≤n≤K} μ(n) · e^{i t n / α}."""
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    theta = t * n / alpha
    return complex(np.sum(m * np.cos(theta)), np.sum(m * np.sin(theta)))


def propagate_fiber(E0: np.ndarray, B0: np.ndarray,
                    mu: np.ndarray, K: int,
                    N_samples: int = 512,
                    barrier_width: int = 41) -> dict:
    """
    Propagate the encoded signal through the fiber.

    The fiber acts as a dispersive channel:

        H(z, ω) = exp(−α |z|) · exp(i ω z / c)

    which, in the time domain, is the convolution of the input field
    with the exponential dispersion kernel exp(−α |z|).

    Additionally, the Maxwell ζ(t) amplitude modulates the envelope
    over one period 2πα, so the transmitted signal carries the full
    Möbius spectral content.
    """
    N_v = len(E0)
    period = 2.0 * PI * ALPHA_SYM

    # --- 1. sample the Maxwell field over one period ---
    t_vals = np.linspace(0, period, N_samples)
    zeta_vals = np.array([zeta_mobius(t, mu, K) for t in t_vals])
    E_maxwell = zeta_vals.real
    B_maxwell = zeta_vals.imag

    # --- 2. build the input field along the fiber coordinate ---
    # interpolate the discrete vertex signals to the fiber grid
    z_grid = np.linspace(0, 1, N_samples)
    E_input = np.interp(z_grid, np.linspace(0, 1, N_v), E0)
    B_input = np.interp(z_grid, np.linspace(0, 1, N_v), B0)

    # --- 3. modulate by the Maxwell envelope ---
    E_carrier = E_input * (1.0 + 0.5 * E_maxwell)
    B_carrier = B_input * (1.0 + 0.5 * B_maxwell)

    # --- 4. fiber dispersion kernel ---
    x = np.arange(-barrier_width // 2, barrier_width // 2)
    kernel = np.exp(-ALPHA_SYM * np.abs(x))
    kernel = kernel / kernel.sum()

    # --- 5. propagate through the fiber ---
    E_out = convolve(E_carrier, kernel, mode="same")
    B_out = convolve(B_carrier, kernel, mode="same")

    # --- 6. energy and phase ---
    energy = E_out ** 2 + B_out ** 2
    intensity = np.abs(E_out + 1j * B_out)

    return dict(
        z_grid=z_grid,
        t_vals=t_vals,
        E_input=E_input, B_input=B_input,
        E_maxwell=E_maxwell, B_maxwell=B_maxwell,
        E_carrier=E_carrier, B_carrier=B_carrier,
        E_out=E_out, B_out=B_out,
        energy=energy, intensity=intensity,
        kernel=kernel,
    )


# ============================================================
#  7. Receiver — decode labels from the received field
# ============================================================
def decode_labels_from_field(E_out: np.ndarray,
                             B_out: np.ndarray,
                             N_v: int,
                             k: int,
                             phi0: float = 0.0) -> np.ndarray:
    """
    Recover the per‑vertex labels by sampling the received intensity
    at the vertex positions and inverting the phase map.
    """
    samples_E = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, len(E_out)),
                          E_out)
    samples_B = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, len(B_out)),
                          B_out)

    # recovered phase per vertex
    phi_hat = np.arctan2(samples_B, samples_E) - phi0

    # label = round( k · φ / 2π ) mod k
    labels_hat = np.round(k * phi_hat / (2.0 * PI)).astype(int) % k
    return labels_hat


# ============================================================
#  8. Evaluation
# ============================================================
def ugc_evaluate(edges, labels_true, labels_hat, N_left):
    total = len(edges)
    if total == 0:
        return dict(satisfied_true=0, satisfied_hat=0,
                    completeness_true=0.0, completeness_hat=0.0,
                    decode_accuracy=0.0)
    sat_true = 0
    sat_hat = 0
    for i, j, perm in edges:
        lt_i, lt_j = labels_true[i], labels_true[N_left + j]
        lh_i, lh_j = labels_hat[i], labels_hat[N_left + j]
        if perm[lt_i] == lt_j:
            sat_true += 1
        if perm[lh_i] == lh_j:
            sat_hat += 1
    agree = int(np.sum(labels_true == labels_hat))
    return dict(
        satisfied_true=sat_true,
        satisfied_hat=sat_hat,
        completeness_true=sat_true / total,
        completeness_hat=sat_hat / total,
        decode_accuracy=agree / len(labels_true),
    )


# ============================================================
#  9. Full simulation
# ============================================================
def simulate(K: int = 24,
             N_left: int = 60,
             N_right: int = 60,
             k_labels: int = 4,
             degree: int = 3,
             N_samples: int = 512,
             barrier_width: int = 41,
             seed_vertices: int = 42):
    print("=" * 80)
    print("Fiber‑optic UGC transmission  ·  6‑vertex set signal + Maxwell + Möbius")
    print("=" * 80)
    print(f"  α_sym        = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  α_asym       = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K            = {K}")
    print(f"  N_left       = {N_left}    N_right = {N_right}")
    print(f"  k labels     = {k_labels}  degree = {degree}")
    print(f"  N_samples    = {N_samples}   barrier_width = {barrier_width}")
    print()

    # ---------- 1. Möbius sieve ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(max(K, 2 ** 10 + 1))
    print(f"[1] Möbius sieve        {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(O(K log log K))")

    # ---------- 2. 6‑vertex set signal ----------
    t0 = time.perf_counter()
    set_signal = six_vertex_set_signal(seed=seed_vertices)
    print(f"[2] 6‑vertex set signal {(time.perf_counter()-t0)*1e3:8.2f} ms")
    print(f"      amplitudes  : {np.round(set_signal['amplitudes'], 4)}")
    print(f"      Π‑scalar    : {set_signal['scalar']:+.6f}")
    print(f"      global phase: {set_signal['phase']:+.6f}")

    # ---------- 3. UGC instance ----------
    t0 = time.perf_counter()
    edges = generate_unique_games_instance(
        N_left, N_right, k_labels, degree=degree, seed=2024)
    N_v = N_left + N_right
    print(f"[3] UGC instance        {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"({len(edges)} edges)")

    # ---------- 4. Label assignment ----------
    t0 = time.perf_counter()
    labels_true, scores = ugc_label_assignment(
        N_left, N_right, k_labels, mu, K_filter=2 ** 10 + 1)
    print(f"[4] Label assignment    {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(labels ∈ [0, {k_labels}))")

    # ---------- 5. Encode labels as optical phases ----------
    t0 = time.perf_counter()
    encoded = encode_labels_as_phases(labels_true, k_labels, set_signal)
    print(f"[5] Optical encoding    {(time.perf_counter()-t0)*1e3:8.2f} ms")
    print(f"      envelope    : {encoded['envelope']:.6f}")
    print(f"      phase range : [{encoded['phi'].min():.4f}, "
          f"{encoded['phi'].max():.4f}]")

    # ---------- 6. Fiber propagation ----------
    t0 = time.perf_counter()
    prop = propagate_fiber(
        encoded["E0"], encoded["B0"], mu, K,
        N_samples=N_samples, barrier_width=barrier_width,
    )
    print(f"[6] Fiber propagation   {(time.perf_counter()-t0)*1e3:8.2f} ms")
    print(f"      E_in  range : [{prop['E_input'].min():+.4f}, "
          f"{prop['E_input'].max():+.4f}]")
    print(f"      E_out range : [{prop['E_out'].min():+.4f}, "
          f"{prop['E_out'].max():+.4f}]")
    print(f"      max |ζ|     : {np.max(np.abs(prop['E_maxwell'] + 1j * prop['B_maxwell'])):.4f}")

    # ---------- 7. Receiver decoding ----------
    t0 = time.perf_counter()
    labels_hat = decode_labels_from_field(
        prop["E_out"], prop["B_out"], N_v, k_labels,
        phi0=set_signal["phase"])
    print(f"[7] Receiver decoding   {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # ---------- 8. Evaluation ----------
    eval_result = ugc_evaluate(edges, labels_true, labels_hat, N_left)

    print()
    print("--- UGC transmission result ---")
    print(f"  edges                    : {len(edges)}")
    print(f"  satisfied (true labels)  : {eval_result['satisfied_true']}  "
          f"→ completeness = {eval_result['completeness_true']:.6f}")
    print(f"  satisfied (decoded)      : {eval_result['satisfied_hat']}  "
          f"→ completeness = {eval_result['completeness_hat']:.6f}")
    print(f"  soundness bound (1/k)    : {1.0 / k_labels:.6f}")
    print(f"  gap (true)               : "
          f"{eval_result['completeness_true'] - 1.0/k_labels:+.6f}")
    print(f"  gap (decoded)            : "
          f"{eval_result['completeness_hat'] - 1.0/k_labels:+.6f}")
    print(f"  decode accuracy          : "
          f"{eval_result['decode_accuracy']*100:.2f} %")

    # ---------- 9. Plot ----------
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.4)

    # (a) 6‑vertex set signal
    ax = fig.add_subplot(gs[0, 0])
    ax.bar(np.arange(6), set_signal["amplitudes"],
           color="#3a7bd5", width=0.7)
    ax.set_xlabel("optical mode")
    ax.set_ylabel("amplitude")
    ax.set_title(f"6‑vertex set signal\n(Π = {set_signal['scalar']:+.4f})")
    ax.grid(True, alpha=0.3)

    # (b) Figure 3.5 heat map
    xs = np.linspace(0, W1, 128)
    ys = np.linspace(0, W2, 128)
    X, Y = np.meshgrid(xs, ys)
    P2D = elliptic_projection_2d(X, Y)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(P2D, extent=[0, W1, 0, W2], origin="lower",
                   cmap="inferno", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("x (mod π)")
    ax.set_ylabel("y (mod e)")
    ax.set_title("Figure 3.5 · label density")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) UGC instance graph
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(2024)
    pos_L = np.column_stack([np.zeros(N_left), np.linspace(0, 1, N_left)])
    pos_R = np.column_stack([np.ones(N_right), np.linspace(0, 1, N_right)])
    cmap = plt.cm.tab10
    for e_idx, (i, j, perm) in enumerate(edges):
        sat = (perm[labels_true[i]] == labels_true[N_left + j])
        color = "#2ecc71" if sat else "#e74c3c"
        ax.plot([pos_L[i, 0], pos_R[j, 0]],
                [pos_L[i, 1], pos_R[j, 1]],
                color=color, lw=0.4, alpha=0.5)
    ax.scatter(pos_L[:, 0], pos_L[:, 1],
               c=[cmap(labels_true[i] / k_labels) for i in range(N_left)],
               s=20, edgecolors="k", zorder=5)
    ax.scatter(pos_R[:, 0], pos_R[:, 1],
               c=[cmap(labels_true[N_left + j] / k_labels)
                  for j in range(N_right)],
               s=20, edgecolors="k", zorder=5)
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UGC instance  (|E| = {len(edges)})")

    # (d) input optical field
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(prop["z_grid"], prop["E_input"], color="#2ecc71",
            lw=1.2, label="E_input (set signal)")
    ax.plot(prop["z_grid"], prop["B_input"], color="#3498db",
            lw=1.0, alpha=0.7, label="B_input")
    ax.plot(prop["z_grid"], prop["E_carrier"], color="#e67e22",
            lw=0.9, alpha=0.6, label="E_carrier (× Maxwell)")
    ax.set_xlabel("position along fiber")
    ax.set_ylabel("field amplitude")
    ax.set_title("Encoded signal at the transmitter")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Maxwell fields
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(prop["t_vals"], prop["E_maxwell"], color="#2ecc71",
            lw=1.0, label="E(t) = Re ζ")
    ax.plot(prop["t_vals"], prop["B_maxwell"], color="#3498db",
            lw=1.0, alpha=0.7, label="B(t) = Im ζ")
    ax.set_xlabel("t")
    ax.set_ylabel("amplitude")
    ax.set_title("Maxwell field over one period")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) fiber kernel
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(np.arange(-barrier_width // 2, barrier_width // 2),
            prop["kernel"], color="#8e44ad", lw=1.5)
    ax.set_xlabel("z")
    ax.set_ylabel("H(z)")
    ax.set_title(f"Fiber kernel exp(−α|z|)")
    ax.grid(True, alpha=0.3)

    # (g) received field
    ax = fig.add_subplot(gs[2, 1:])
    ax.plot(prop["z_grid"], prop["E_out"], color="#e74c3c",
            lw=1.4, label="E_out")
    ax.plot(prop["z_grid"], prop["B_out"], color="#8e44ad",
            lw=1.2, alpha=0.7, label="B_out")
    ax.plot(prop["z_grid"], prop["intensity"], color="#16a085",
            lw=1.0, ls="--", label="|E + iB|")
    ax.set_xlabel("position along fiber")
    ax.set_ylabel("field amplitude")
    ax.set_title("Received signal after fiber propagation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (h) label comparison
    ax = fig.add_subplot(gs[3, :2])
    z_v = np.arange(N_v)
    ax.scatter(z_v, labels_true, s=40, c="#3a7bd5",
               edgecolors="k", label="true labels σ(v)")
    ax.scatter(z_v, labels_hat, s=20, c="#e74c3c",
               marker="x", label="decoded labels σ̂(v)")
    ax.set_xlabel("vertex index")
    ax.set_ylabel("label")
    ax.set_title(f"Label recovery  ·  accuracy = "
                 f"{eval_result['decode_accuracy']*100:.1f} %")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (i) completeness comparison
    ax = fig.add_subplot(gs[3, 2])
    bars = [
        eval_result["completeness_true"],
        eval_result["completeness_hat"],
        1.0 / k_labels,
    ]
    colors = ["#3a7bd5", "#e74c3c", "#95a5a6"]
    labels = ["true", "decoded", "soundness 1/k"]
    ax.bar(np.arange(3), bars, color=colors, width=0.7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("completeness")
    ax.set_title("UGC completeness")
    for i, v in enumerate(bars):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Fiber‑optic UGC transmission  ·  "
        f"completeness true = {eval_result['completeness_true']:.3f},  "
        f"decoded = {eval_result['completeness_hat']:.3f},  "
        f"accuracy = {eval_result['decode_accuracy']*100:.1f}%",
        fontsize=13, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # ---------- Summary ----------
    print()
    print("--- Summary ---")
    print(f"  6‑vertex Π scalar            : {set_signal['scalar']:+.6f}")
    print(f"  UGC edges                    : {len(edges)}")
    print(f"  Completeness (true)          : {eval_result['completeness_true']:.6f}")
    print(f"  Completeness (decoded)       : {eval_result['completeness_hat']:.6f}")
    print(f"  Soundness bound (1/k)        : {1.0 / k_labels:.6f}")
    print(f"  Decode accuracy              : {eval_result['decode_accuracy']*100:.2f} %")
    print(f"  Received |E|²  range         : [{prop['energy'].min():.4e}, "
          f"{prop['energy'].max():.4e}]")
    print()

    return dict(
        set_signal=set_signal,
        labels_true=labels_true,
        labels_hat=labels_hat,
        scores=scores,
        edges=edges,
        propagation=prop,
        eval_result=eval_result,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24,
             N_left=60, N_right=60,
             k_labels=4, degree=3,
             N_samples=512, barrier_width=41,
             seed_vertices=42)