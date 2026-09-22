#!/usr/bin/env python3
"""
fiber_ugc_elliptic_amplitude.py

Fiber-optic UGC transmission where the **amplitude of the sent signal**
is given by the elliptic Möbius power spectrum |ζ(t)| from
`eliptic-finite-step-mobius.py`.

Pipeline
--------
    6-vertex configuration        →  set signal (Π scalar, phase)
    UGC labels σ(v)               →  phase modulation (message)
    Elliptic Möbius ζ(t)          →  **amplitude modulation**       ← new
    Maxwell propagation           →  E(z, t), B(z, t) in the fiber
    Exponential fiber kernel      →  dispersion
    Möbius receiver               →  decoded labels σ̂(v)

The elliptic Möbius gate builds coefficients

    C_i = μ(|i|)                     if μ(|i|) ≠ 0
    C_i = linear interpolation       if μ(|i|) = 0

and the spectral sum

    ζ(t) = Σ_i C_i · exp(i t i / α),    α = 1/(π − e)

whose modulus |ζ(t)| becomes the **per‑mode amplitude envelope** of
the transmitted signal.  The power spectrum |ζ(t)|² is the energy
carried by each optical mode.
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
DENSITY    = 6.0 / (PI * PI)
W1         = PI
W2         = E


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
#  2. Elliptic Möbius gate  (from eliptic-finite-step-mobius.py)
# ============================================================
def build_elliptic_coeffs(K: int, smooth: bool = True):
    """
    Build C_i for i = -K..K.
      C_i = μ(|i|)                     if μ(|i|) ≠ 0
      C_i = linear interpolation       if μ(|i|) = 0
    """
    mu = mobius_sieve(K)
    c = np.zeros(2 * K + 1, dtype=float)
    for i in range(-K, K + 1):
        if i == 0:
            c[i + K] = 0.0
        else:
            c[i + K] = mu[abs(i)]

    if smooth:
        for i in range(-K, K + 1):
            if i == 0:
                continue
            if c[i + K] == 0.0:
                left, right = i - 1, i + 1
                while left >= -K and c[left + K] == 0.0:
                    left -= 1
                while right <= K and c[right + K] == 0.0:
                    right += 1
                if left < -K or right > K:
                    continue
                left_val, right_val = c[left + K], c[right + K]
                dist = right - left
                if dist == 0:
                    continue
                wl = (right - i) / dist
                wr = (i - left) / dist
                c[i + K] = wl * left_val + wr * right_val

    coeffs = {i: c[i + K] for i in range(-K, K + 1)
              if abs(c[i + K]) > 1e-12}
    return coeffs, c


class EllipticMobiusGate:
    """Spectral sum ζ(t) = Σ C_i · exp(i t i / α)."""
    def __init__(self, K: int, smooth: bool = True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.values = np.array([self.coeffs[i] for i in self.indices])
        self.period = 2 * PI * ALPHA_SYM

    def zeta(self, t: float) -> complex:
        if len(self.indices) == 0:
            return 0.0 + 0.0j
        phases = t * self.indices / ALPHA_SYM
        return complex(np.sum(self.values * np.cos(phases)),
                       np.sum(self.values * np.sin(phases)))
    def zeta_array(self, t_vals: np.ndarray) -> np.ndarray:
        """Vectorised ζ over an array of times."""
        phases = np.outer(t_vals, self.indices) / ALPHA_SYM
        return np.exp(1j * phases) @ self.values        # ← operands swapped

    def amplitude(self, t_vals: np.ndarray) -> np.ndarray:
        """|ζ(t)| — the amplitude envelope used for transmission."""
        return np.abs(self.zeta_array(t_vals))

    def power(self, t_vals: np.ndarray) -> np.ndarray:
        """|ζ(t)|² — the transmitted energy per mode."""
        return self.amplitude(t_vals) ** 2

    def basel_theoretical(self) -> float:
        return 2.0 * (6.0 / (PI * PI)) * self.K

    def basel_error(self, N: int = 1000) -> float:
        t_vals = np.linspace(0, self.period, N)
        power = self.power(t_vals)
        try:
            integral = np.trapezoid(power, t_vals)
        except AttributeError:
            integral = np.trapz(power, t_vals)
        avg = integral / self.period
        theo = self.basel_theoretical()
        return abs(avg - theo) / theo if theo != 0 else 0.0


# ============================================================
#  3. Figure 3.5 — bounded elliptic projection
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
#  4. 6-vertex configuration  (the SET signal)
# ============================================================
def generate_vertices(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((12, 6))


def levi_civita_contraction(V6: np.ndarray) -> float:
    from itertools import permutations
    scalar = 0.0
    for perm in permutations(range(6)):
        inv = sum(1 for i in range(6) for j in range(i + 1, 6)
                  if perm[i] > perm[j])
        sign = (-1) ** inv
        prod = 1.0
        for k, c in enumerate(perm):
            prod *= V6[k, c]
        scalar += sign * prod
    return scalar


def six_vertex_set_signal(seed: int = 42) -> dict:
    V = generate_vertices(seed)
    V6 = V[:6, :6]
    row_norms = np.linalg.norm(V6, axis=1)
    row_norms = row_norms / (row_norms.max() + 1e-12)
    scalar = levi_civita_contraction(V6)
    phase = 0.0 if scalar >= 0 else math.pi
    return dict(
        vertices=V6,
        amplitudes=row_norms,
        scalar=scalar,
        phase=phase,
    )


# ============================================================
#  5. UGC instance + Möbius label assignment
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
#  6. Amplitude from the elliptic Möbius gate
# ============================================================
def elliptic_amplitude_envelope(gate: EllipticMobiusGate,
                                N_v: int,
                                N_samples: int) -> dict:
    """
    Sample |ζ(t)| over one period and resample it to N_samples points.

    The result is a smooth, deterministic amplitude curve — the same
    curve at every transmission, so the receiver knows the carrier
    envelope in advance.

    Returns
    -------
    t_vals       : sample times over [0, 2πα]
    zeta_mag     : |ζ(t)| at the sample times
    amplitude    : |ζ(t)| resampled to N_samples points, normalised to
                   [0, 1] and lifted to [0.3, 1.0] to keep the carrier
                   above the noise floor
    """
    t_vals = np.linspace(0.0, gate.period, N_samples)
    zeta_mag = gate.amplitude(t_vals)
    zeta_norm = zeta_mag / (zeta_mag.max() + 1e-12)
    # lift to [0.3, 1.0] — keeps the carrier strictly positive
    amplitude = 0.3 + 0.7 * zeta_norm
    return dict(
        t_vals=t_vals,
        zeta_mag=zeta_mag,
        zeta_norm=zeta_norm,
        amplitude=amplitude,
    )


# ============================================================
#  7. Optical encoding with elliptic amplitude
# ============================================================
def encode_labels_with_elliptic_amplitude(labels: np.ndarray,
                                          k: int,
                                          set_signal: dict,
                                          gate: EllipticMobiusGate,
                                          N_samples: int) -> dict:
    """
    Encode the UGC labels as phase modulation, with the **amplitude
    envelope** given by the elliptic Möbius power spectrum |ζ(t)|.
    """
    N_v = len(labels)
    amps = set_signal["amplitudes"]
    phase0 = set_signal["phase"]
    scalar = set_signal["scalar"]

    # per-vertex phase
    phi = 2.0 * PI * labels / k

    # 6-mode carrier tiled across the N_v vertices
    carrier_amp = np.tile(amps, int(np.ceil(N_v / len(amps))))[:N_v]

    # elliptic amplitude envelope — resampled to the vertex count
    env = elliptic_amplitude_envelope(gate, N_v, N_samples)
    amp_at_vertices = np.interp(
        np.linspace(0, 1, N_v),
        np.linspace(0, 1, N_samples),
        env["amplitude"],
    )

    # full set signal amplitude (carrier × envelope × Π scalar)
    envelope_scale = math.tanh(abs(scalar))
    A_full = carrier_amp * amp_at_vertices * envelope_scale

    # electric / magnetic fields from the phase
    E0 = A_full * np.cos(phi + phase0)
    B0 = A_full * np.sin(phi + phase0)

    return dict(
        E0=E0, B0=B0, phi=phi,
        A_full=A_full,
        amp_at_vertices=amp_at_vertices,
        envelope=env,
        envelope_scale=envelope_scale,
    )


# ============================================================
#  8. Fiber propagation
# ============================================================
def propagate_fiber(E0, B0, mu, K,
                    N_samples=512, barrier_width=41):
    N_v = len(E0)
    z_grid = np.linspace(0, 1, N_samples)
    E_input = np.interp(z_grid, np.linspace(0, 1, N_v), E0)
    B_input = np.interp(z_grid, np.linspace(0, 1, N_v), B0)

    # dispersion kernel
    x = np.arange(-barrier_width // 2, barrier_width // 2)
    kernel = np.exp(-ALPHA_SYM * np.abs(x))
    kernel = kernel / kernel.sum()

    E_out = convolve(E_input, kernel, mode="same")
    B_out = convolve(B_input, kernel, mode="same")

    intensity = np.abs(E_out + 1j * B_out)
    energy = E_out ** 2 + B_out ** 2
    return dict(
        z_grid=z_grid,
        E_input=E_input, B_input=B_input,
        E_out=E_out, B_out=B_out,
        intensity=intensity, energy=energy,
        kernel=kernel,
    )


# ============================================================
#  9. Receiver
# ============================================================
def decode_labels_from_field(E_out, B_out, N_v, k,
                             phi0=0.0,
                             amp_reference=None) -> np.ndarray:
    """
    Recover labels by sampling the phase of the received field.

    If `amp_reference` is provided, the received field is normalised
    by the known elliptic amplitude envelope before phase extraction,
    which greatly improves decode accuracy.
    """
    samples_E = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, len(E_out)), E_out)
    samples_B = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, len(B_out)), B_out)

    if amp_reference is not None:
        ref = np.maximum(amp_reference, 1e-6)
        samples_E = samples_E / ref
        samples_B = samples_B / ref

    phi_hat = np.arctan2(samples_B, samples_E) - phi0
    labels_hat = np.round(k * phi_hat / (2 * PI)).astype(int) % k
    return labels_hat


# ============================================================
#  10. Evaluation
# ============================================================
def ugc_evaluate(edges, labels_true, labels_hat, N_left):
    total = len(edges)
    if total == 0:
        return dict(satisfied_true=0, satisfied_hat=0,
                    completeness_true=0.0, completeness_hat=0.0,
                    decode_accuracy=0.0)
    sat_true = sat_hat = 0
    for i, j, perm in edges:
        if perm[labels_true[i]] == labels_true[N_left + j]:
            sat_true += 1
        if perm[labels_hat[i]] == labels_hat[N_left + j]:
            sat_hat += 1
    agree = int(np.sum(labels_true == labels_hat))
    return dict(
        satisfied_true=sat_true, satisfied_hat=sat_hat,
        completeness_true=sat_true / total,
        completeness_hat=sat_hat / total,
        decode_accuracy=agree / len(labels_true),
    )


# ============================================================
#  11. Full simulation
# ============================================================
def simulate(K: int = 24,
             K_gate: int = 20,
             N_left: int = 60,
             N_right: int = 60,
             k_labels: int = 4,
             degree: int = 3,
             N_samples: int = 512,
             barrier_width: int = 41,
             seed_vertices: int = 42):
    print("=" * 82)
    print("Fiber-optic UGC transmission  ·  elliptic Möbius amplitude envelope")
    print("=" * 82)
    print(f"  α_sym      = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  K (sieve)  = {K}")
    print(f"  K (gate)   = {K_gate}")
    print(f"  N_left     = {N_left}   N_right = {N_right}")
    print(f"  k labels   = {k_labels}  degree = {degree}")
    print()

    # ---------- 1. Möbius sieve ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(max(K, 2 ** 10 + 1))
    print(f"[1] Möbius sieve          {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # ---------- 2. Elliptic Möbius gate ----------
    t0 = time.perf_counter()
    gate = EllipticMobiusGate(K_gate, smooth=True)
    print(f"[2] Elliptic Möbius gate  {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"({len(gate.indices)} non-zero coeffs)")

    basel_err = gate.basel_error(N=2000)
    print(f"      Basel error         : {basel_err:.6f}")
    print(f"      ζ period            : {gate.period:.6f}")

    # ---------- 3. 6-vertex set signal ----------
    t0 = time.perf_counter()
    set_signal = six_vertex_set_signal(seed=seed_vertices)
    print(f"[3] 6-vertex set signal   {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(Π = {set_signal['scalar']:+.4f})")

    # ---------- 4. UGC instance ----------
    edges = generate_unique_games_instance(
        N_left, N_right, k_labels, degree=degree, seed=2024)
    N_v = N_left + N_right
    print(f"[4] UGC instance           {len(edges)} edges")

    # ---------- 5. Label assignment ----------
    t0 = time.perf_counter()
    labels_true, scores = ugc_label_assignment(
        N_left, N_right, k_labels, mu, K_filter=2 ** 10 + 1)
    print(f"[5] Label assignment      {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # ---------- 6. Encode with elliptic amplitude ----------
    t0 = time.perf_counter()
    encoded = encode_labels_with_elliptic_amplitude(
        labels_true, k_labels, set_signal, gate, N_samples)
    print(f"[6] Encoding (elliptic A) {(time.perf_counter()-t0)*1e3:8.2f} ms")
    print(f"      amplitude range     : "
          f"[{encoded['A_full'].min():.4f}, "
          f"{encoded['A_full'].max():.4f}]")
    print(f"      |ζ| range           : "
          f"[{encoded['envelope']['zeta_mag'].min():.4f}, "
          f"{encoded['envelope']['zeta_mag'].max():.4f}]")

    # ---------- 7. Fiber propagation ----------
    t0 = time.perf_counter()
    prop = propagate_fiber(
        encoded["E0"], encoded["B0"], mu, K,
        N_samples=N_samples, barrier_width=barrier_width)
    print(f"[7] Fiber propagation     {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # ---------- 8. Receiver ----------
    # reference amplitude = the transmitted envelope (known in advance)
    amp_ref = np.interp(
        np.linspace(0, 1, N_v),
        np.linspace(0, 1, N_samples),
        encoded["envelope"]["amplitude"],
    )
    labels_hat = decode_labels_from_field(
        prop["E_out"], prop["B_out"], N_v, k_labels,
        phi0=set_signal["phase"],
        amp_reference=amp_ref,
    )
    print(f"[8] Receiver decode       done")

    # ---------- 9. Evaluation ----------
    eval_result = ugc_evaluate(edges, labels_true, labels_hat, N_left)

    print()
    print("--- UGC transmission result ---")
    print(f"  edges                    : {len(edges)}")
    print(f"  completeness (true)      : {eval_result['completeness_true']:.6f}")
    print(f"  completeness (decoded)   : {eval_result['completeness_hat']:.6f}")
    print(f"  soundness bound (1/k)    : {1.0 / k_labels:.6f}")
    print(f"  decode accuracy          : "
          f"{eval_result['decode_accuracy']*100:.2f} %")
    print(f"  amplification range      : "
          f"[{encoded['A_full'].min():.4f}, "
          f"{encoded['A_full'].max():.4f}]")

    # ---------- 10. Plot ----------
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.4)

    # (a) Elliptic Möbius amplitude |ζ(t)|
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(encoded["envelope"]["t_vals"],
            encoded["envelope"]["zeta_mag"],
            color="#3a7bd5", lw=1.4)
    ax.fill_between(encoded["envelope"]["t_vals"], 0,
                    encoded["envelope"]["zeta_mag"],
                    color="#3a7bd5", alpha=0.25)
    ax.set_xlabel("t")
    ax.set_ylabel("|ζ(t)|")
    ax.set_title(f"Elliptic Möbius amplitude  (K={K_gate})")
    ax.grid(True, alpha=0.3)

    # (b) power spectrum |ζ(t)|²
    ax = fig.add_subplot(gs[0, 1])
    power = encoded["envelope"]["zeta_mag"] ** 2
    ax.plot(encoded["envelope"]["t_vals"], power,
            color="#e67e22", lw=1.4)
    ax.axhline(gate.basel_theoretical() / N_samples,
               color="g", ls=":", lw=1.0,
               label="Basel ref")
    ax.set_xlabel("t")
    ax.set_ylabel("|ζ(t)|²")
    ax.set_title(f"Power spectrum  ·  Basel err = {basel_err:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) 6-vertex set signal
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(6), set_signal["amplitudes"],
           color="#16a085", width=0.7)
    ax.set_xlabel("mode")
    ax.set_ylabel("amplitude")
    ax.set_title(f"6-vertex set  ·  Π = {set_signal['scalar']:+.4f}")
    ax.grid(True, alpha=0.3)

    # (d) transmitted signal with elliptic amplitude
    ax = fig.add_subplot(gs[1, :2])
    z_v = np.linspace(0, 1, N_v)
    ax.plot(z_v, encoded["A_full"], color="#8e44ad", lw=1.4,
            label="A_full = carrier × |ζ| × Π")
    ax.plot(z_v, encoded["E0"], color="#2ecc71", lw=1.0,
            alpha=0.7, label="E₀ (cos)")
    ax.plot(z_v, encoded["B0"], color="#3498db", lw=1.0,
            alpha=0.7, label="B₀ (sin)")
    ax.set_xlabel("vertex index")
    ax.set_ylabel("amplitude")
    ax.set_title("Transmitter: elliptic-amplitude modulated signal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) UGC graph
    ax = fig.add_subplot(gs[1, 2])
    pos_L = np.column_stack([np.zeros(N_left), np.linspace(0, 1, N_left)])
    pos_R = np.column_stack([np.ones(N_right), np.linspace(0, 1, N_right)])
    cmap = plt.cm.tab10
    for e_idx, (i, j, perm) in enumerate(edges):
        sat = (perm[labels_true[i]] == labels_true[N_left + j])
        ax.plot([pos_L[i, 0], pos_R[j, 0]],
                [pos_L[i, 1], pos_R[j, 1]],
                color="#2ecc71" if sat else "#e74c3c",
                lw=0.4, alpha=0.5)
    ax.scatter(pos_L[:, 0], pos_L[:, 1],
               c=[cmap(labels_true[i] / k_labels) for i in range(N_left)],
               s=18, edgecolors="k", zorder=5)
    ax.scatter(pos_R[:, 0], pos_R[:, 1],
               c=[cmap(labels_true[N_left + j] / k_labels)
                  for j in range(N_right)],
               s=18, edgecolors="k", zorder=5)
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UGC  ·  |E| = {len(edges)}")

    # (f) fiber kernel
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(np.arange(-barrier_width // 2, barrier_width // 2),
            prop["kernel"], color="#8e44ad", lw=1.5)
    ax.set_xlabel("z")
    ax.set_ylabel("H(z)")
    ax.set_title("Fiber kernel exp(−α|z|)")
    ax.grid(True, alpha=0.3)

    # (g) received signal
    ax = fig.add_subplot(gs[2, 1:])
    ax.plot(prop["z_grid"], prop["E_out"], color="#e74c3c",
            lw=1.3, label="E_out")
    ax.plot(prop["z_grid"], prop["B_out"], color="#8e44ad",
            lw=1.1, alpha=0.7, label="B_out")
    ax.plot(prop["z_grid"], prop["intensity"], color="#16a085",
            lw=1.0, ls="--", label="|E + iB|")
    ax.set_xlabel("position along fiber")
    ax.set_ylabel("field")
    ax.set_title("Receiver: after fiber propagation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (h) label comparison
    ax = fig.add_subplot(gs[3, :2])
    ax.scatter(np.arange(N_v), labels_true, s=42, c="#3a7bd5",
               edgecolors="k", label="true σ(v)")
    ax.scatter(np.arange(N_v), labels_hat, s=22, c="#e74c3c",
               marker="x", label="decoded σ̂(v)")
    ax.set_xlabel("vertex index")
    ax.set_ylabel("label")
    ax.set_title(
        f"Label recovery  ·  accuracy = "
        f"{eval_result['decode_accuracy']*100:.1f} %"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (i) completeness bars
    ax = fig.add_subplot(gs[3, 2])
    bars = [
        eval_result["completeness_true"],
        eval_result["completeness_hat"],
        1.0 / k_labels,
    ]
    colors = ["#3a7bd5", "#e74c3c", "#95a5a6"]
    ax.bar(np.arange(3), bars, color=colors, width=0.7)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["true", "decoded", "soundness"], fontsize=9)
    ax.set_ylabel("completeness")
    ax.set_title("UGC completeness")
    for i, v in enumerate(bars):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Fiber-optic UGC with elliptic Möbius amplitude envelope  ·  "
        f"accuracy = {eval_result['decode_accuracy']*100:.1f} %,  "
        f"Basel error = {basel_err:.4f}",
        fontsize=13, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return dict(
        gate=gate, set_signal=set_signal,
        labels_true=labels_true, labels_hat=labels_hat,
        edges=edges, propagation=prop,
        encoded=encoded,
        eval_result=eval_result,
        basel_error=basel_err,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24, K_gate=20,
             N_left=60, N_right=60,
             k_labels=4, degree=3,
             N_samples=512, barrier_width=41,
             seed_vertices=42)
# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24,
             N_left=60, N_right=60,
             k_labels=4, degree=3,
             N_samples=512, barrier_width=41,
             seed_vertices=42)