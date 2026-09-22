#!/usr/bin/env python3
"""
photonic_chip_ugc.py

A photonic chip simulator that fuses:

    • fiber-optic-signal-ugc.py   →  optical signal path (waveguides,
                                     phase modulators, elliptic amplitude)
    • 64-BIT.py                   →  dual-envelope readout (positive
                                     half = symmetric/real modes,
                                     negative half = asymmetric/chirped
                                     modes)
    • Figure 3.5 / UGC            →  label routing and gate logic
    • chip-g.py TSP route          →  on-chip waveguide ordering

Physical picture
----------------
    +--------------------------------------------------------------+
    |                     PHOTONIC CHIP                            |
    |                                                              |
    |  input laser →  6-vertex carrier  →  12 phase modulators     |
    |                                          ↓                    |
    |                   waveguide array (TSP-routed)                |
    |                                          ↓                    |
    |             output couplers  →  photodetector bank            |
    |                                          ↓                    |
    |                dual-envelope readout (64-bit layout)          |
    |                                          ↓                    |
    |             decoded UGC labels σ̂(v)  →  completeness          |
    +--------------------------------------------------------------+

Everything runs on the standard CPU; the photonic chip is a
numerical model of the field evolution along the waveguides.
"""

from __future__ import annotations

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
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
#  2. Elliptic Möbius gate  (amplitude source)
# ============================================================
def build_elliptic_coeffs(K: int, smooth: bool = True):
    mu = mobius_sieve(K)
    c = np.zeros(2 * K + 1, dtype=float)
    for i in range(-K, K + 1):
        c[i + K] = 0.0 if i == 0 else mu[abs(i)]

    if smooth:
        for i in range(-K, K + 1):
            if i == 0 or c[i + K] != 0.0:
                continue
            left, right = i - 1, i + 1
            while left >= -K and c[left + K] == 0.0:
                left -= 1
            while right <= K and c[right + K] == 0.0:
                right += 1
            if left < -K or right > K:
                continue
            dist = right - left
            if dist == 0:
                continue
            wl = (right - i) / dist
            wr = (i - left) / dist
            c[i + K] = wl * c[left + K] + wr * c[right + K]

    coeffs = {i: c[i + K] for i in range(-K, K + 1)
              if abs(c[i + K]) > 1e-12}
    return coeffs, c


class EllipticMobiusGate:
    """ζ(t) = Σ_i C_i · exp(i t i / α)."""
    def __init__(self, K: int, smooth: bool = True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.values  = np.array([self.coeffs[i] for i in self.indices])
        self.period  = 2 * PI * ALPHA_SYM

    def zeta_array(self, t_vals: np.ndarray) -> np.ndarray:
        phases = np.outer(t_vals, self.indices) / ALPHA_SYM
        return np.exp(1j * phases) @ self.values

    def amplitude(self, t_vals: np.ndarray) -> np.ndarray:
        return np.abs(self.zeta_array(t_vals))

    def power(self, t_vals: np.ndarray) -> np.ndarray:
        return self.amplitude(t_vals) ** 2

    def basel_theoretical(self) -> float:
        return 2.0 * (6.0 / (PI * PI)) * self.K

    def basel_error(self, N: int = 2000) -> float:
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
def elliptic_projection_2d(X, Y):
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))


def elliptic_projection_1d(x, y0=1.0):
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


# ============================================================
#  4. 6-vertex set signal  (the input carrier)
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
    return dict(vertices=V6, amplitudes=row_norms,
                scalar=scalar, phase=phase)


# ============================================================
#  5. Chip angle sort  (chip-g.py TSP route)
# ============================================================
def tsp_route_1d(K: int, offset: float = 0.0) -> np.ndarray:
    angles = np.zeros(K, dtype=float)
    for i in range(K):
        x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
        y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
        angles[i] = (math.atan2(y, x) + math.pi + offset) % (2 * math.pi)
    buckets = [[] for _ in range(360)]
    for i, a in enumerate(angles):
        idx = int((a / (2 * math.pi)) * 360) % 360
        buckets[idx].append(i)
    order = []
    for b in buckets:
        order.extend(b)
    return np.array(order, dtype=int)


# ============================================================
#  6. UGC instance + Möbius label assignment
# ============================================================
def generate_unique_games_instance(N_left, N_right, k,
                                   degree=3, seed=2024):
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


def quadratic_address(a, A=1, B=1, C=0, K=32):
    return (A * a * a + B * a + C) % K


def ugc_label_assignment(N_left, N_right, k, mu, K_filter=2**10 + 1):
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
#  7. Photonic chip layout
# ============================================================
class PhotonicChip:
    """
    A photonic chip is an array of waveguides that carry the
    optical signal from input modulators to output couplers.

    Geometry
    --------
    • N_wg waveguides laid out in 1D (the chip is 1D in the demo,
      but the layout can be extruded).
    • Each waveguide has:
        - a length L_wg (in cm)
        - a propagation loss α_loss (in dB/cm)
        - a dispersion coefficient D (in ps²/km)
    • Input modulators set the phase φ_n and amplitude A_n of each
      waveguide.
    • Output couplers sample the field.
    • Waveguides are TSP-routed using chip-g.py bucket sort.
    """

    def __init__(self, N_wg: int = 12, L_wg: float = 1.0,
                 loss_dB_cm: float = 3.0, D_ps2_km: float = 17.0):
        self.N_wg     = N_wg
        self.L_wg     = L_wg
        self.loss     = loss_dB_cm
        self.D        = D_ps2_km

        # TSP route by chip-g.py bucket sort
        self.tsp_route = tsp_route_1d(N_wg, offset=math.pi / 2)

        # physical metadata
        self.lengths = np.full(N_wg, L_wg)
        self.losses  = np.full(N_wg, loss_dB_cm)
        self.D_vals  = np.full(N_wg, D_ps2_km)

    def transmit(self, A_in: np.ndarray, phase_in: np.ndarray,
                 mu: np.ndarray, K: int,
                 N_samples: int = 512) -> dict:
        """
        Transmit the encoded signal through the waveguides.

        A_in, phase_in : amplitudes and phases at the input
        Output : E_out, B_out, intensity, energy at the output couplers
        """
        # sort input by the TSP route
        A_sorted     = A_in[self.tsp_route]
        phase_sorted = phase_in[self.tsp_route]

        # up-sample to the waveguide propagation grid
        z = np.linspace(0, 1, N_samples)
        E_in = np.interp(z, np.linspace(0, 1, self.N_wg),
                         A_sorted * np.cos(phase_sorted))
        B_in = np.interp(z, np.linspace(0, 1, self.N_wg),
                         A_sorted * np.sin(phase_sorted))

        # loss factor along z
        loss_z = 10 ** (-self.loss * self.L_wg * z / 10.0)

        # dispersion kernel
        width = 41
        x = np.arange(-width // 2, width // 2)
        kernel = np.exp(-ALPHA_SYM * np.abs(x))
        kernel = kernel / kernel.sum()

        E_out = loss_z * convolve(E_in, kernel, mode="same")
        B_out = loss_z * convolve(B_in, kernel, mode="same")

        intensity = np.abs(E_out + 1j * B_out)
        energy    = E_out ** 2 + B_out ** 2
        return dict(
            z=z, E_in=E_in, B_in=B_in,
            E_out=E_out, B_out=B_out,
            intensity=intensity, energy=energy,
            loss_z=loss_z, kernel=kernel,
            tsp_route=self.tsp_route,
        )


# ============================================================
#  8. 64-bit dual-envelope readout
# ============================================================
class DualEnvelopeReadout:
    """
    Positive half of the |Ci| array  →  symmetric (real) modes
    Negative half                    →  asymmetric (chirped) modes

    The readout computes S_pos, S_neg, S_all, H, m from the field
    at the output of the photonic chip.
    """
    def __init__(self, K: int, chirp_a: float = 1e-4,
                 chirp_b: float = 0.0, chirp_c: float = 0.0):
        self.K = K
        self.chirp_a, self.chirp_b, self.chirp_c = chirp_a, chirp_b, chirp_c

    def readout(self, E_out: np.ndarray, B_out: np.ndarray) -> dict:
        # split into positive (first half) and negative (second half)
        N = len(E_out)
        half = N // 2
        E_pos, E_neg = E_out[:half], E_out[half:]
        B_pos, B_neg = B_out[:half], B_out[half:]

        mag_pos = np.abs(E_pos + 1j * B_pos)
        mag_neg = np.abs(E_neg + 1j * B_neg)

        chirp_neg = np.array([
            quadratic_chirp(i, self.chirp_a,
                            self.chirp_b, self.chirp_c)
            for i in range(1, len(mag_neg) + 1)
        ])

        # supertraces
        S_pos = float(np.sum(np.where(np.arange(len(mag_pos)) % 2 == 0,
                                      mag_pos, -mag_pos)))
        S_neg = float(np.sum(np.where(np.arange(len(mag_neg)) % 2 == 0,
                                      mag_neg * chirp_neg,
                                      -mag_neg * chirp_neg)))
        S_all = S_pos + S_neg

        H = self._entropy(S_all, N)
        m = self._mass(S_all, N)

        return dict(S_pos=S_pos, S_neg=S_neg, S_all=S_all,
                    H=H, m=m, mag_pos=mag_pos, mag_neg=mag_neg,
                    chirp_neg=chirp_neg)

    @staticmethod
    def _entropy(S: float, N: int) -> float:
        if N <= 0 or S == 0.0:
            return 0.0
        p = abs(S) / N
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -ALPHA_SYM * p * math.log(p)

    @staticmethod
    def _mass(S: float, N: int) -> float:
        H = DualEnvelopeReadout._entropy(S, N)
        return abs(S) * math.exp(-H) if H < 700 else 0.0


def quadratic_chirp(n, a, b, c):
    return math.cos(a * n * n + b * n + c)


# ============================================================
#  9. Full photonic chip simulation
# ============================================================
def simulate(K: int = 24,
             K_gate: int = 20,
             N_left: int = 60,
             N_right: int = 60,
             k_labels: int = 4,
             degree: int = 3,
             N_wg: int = 12,
             N_samples: int = 512,
             seed_vertices: int = 42):
    print("=" * 82)
    print("Photonic chip  ·  6-vertex carrier + elliptic Möbius amplitude + UGC")
    print("=" * 82)
    print(f"  α_sym      = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  α_asym     = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K (sieve)  = {K}")
    print(f"  K (gate)   = {K_gate}")
    print(f"  waveguides = {N_wg}")
    print(f"  N_left     = {N_left}   N_right = {N_right}")
    print(f"  k labels   = {k_labels}  degree = {degree}")
    print()

    # ---------- 1. Möbius sieve ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(max(K, 2 ** 10 + 1))
    print(f"[1] Möbius sieve            {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # ---------- 2. Elliptic Möbius gate ----------
    t0 = time.perf_counter()
    gate = EllipticMobiusGate(K_gate, smooth=True)
    print(f"[2] Elliptic Möbius gate    {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"({len(gate.indices)} non-zero coeffs)")
    basel_err = gate.basel_error(N=2000)
    print(f"      Basel error           : {basel_err:.6f}")

    # ---------- 3. 6-vertex set signal ----------
    t0 = time.perf_counter()
    set_signal = six_vertex_set_signal(seed=seed_vertices)
    print(f"[3] 6-vertex set signal     {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(Π = {set_signal['scalar']:+.4f})")

    # ---------- 4. UGC instance ----------
    edges = generate_unique_games_instance(
        N_left, N_right, k_labels, degree=degree, seed=2024)
    N_v = N_left + N_right
    print(f"[4] UGC instance             {len(edges)} edges")

    # ---------- 5. Label assignment ----------
    t0 = time.perf_counter()
    labels_true, scores = ugc_label_assignment(
        N_left, N_right, k_labels, mu, K_filter=2 ** 10 + 1)
    print(f"[5] Label assignment        {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # ---------- 6. Amplitude envelope from the elliptic gate ----------
    t0 = time.perf_counter()
    t_env = np.linspace(0, gate.period, N_samples)
    zeta_mag = gate.amplitude(t_env)
    amp_env = 0.3 + 0.7 * (zeta_mag / (zeta_mag.max() + 1e-12))
    print(f"[6] Amplitude envelope      {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(|ζ| ∈ [{zeta_mag.min():.4f}, {zeta_mag.max():.4f}])")

    # ---------- 7. Encode into the photonic chip ----------
    # we tile the 6 carrier amplitudes across the N_wg waveguides
    amps = set_signal["amplitudes"]
    carrier_amp = np.tile(amps, int(np.ceil(N_wg / 6)))[:N_wg]
    amp_at_wg = np.interp(
        np.linspace(0, 1, N_wg),
        np.linspace(0, 1, N_samples),
        amp_env,
    )
    envelope_scale = math.tanh(abs(set_signal["scalar"]))
    A_full = carrier_amp * amp_at_wg * envelope_scale

    # per-waveguide phase from the UGC labels (first N_wg labels)
    phases_wg = 2.0 * PI * labels_true[:N_wg] / k_labels
    phase0 = set_signal["phase"]

    print(f"[7] Encoded {N_wg} waveguides  "
          f"(A ∈ [{A_full.min():.4f}, {A_full.max():.4f}])")

    # ---------- 8. Photonic chip propagation ----------
    t0 = time.perf_counter()
    chip = PhotonicChip(N_wg=N_wg, L_wg=1.0,
                        loss_dB_cm=3.0, D_ps2_km=17.0)
    prop = chip.transmit(A_full, phases_wg + phase0, mu, K,
                         N_samples=N_samples)
    print(f"[8] Chip propagation        {(time.perf_counter()-t0)*1e3:8.2f} ms")
    print(f"      TSP route (first 8)   : {prop['tsp_route'][:8].tolist()}")
    print(f"      loss at z=1           : {prop['loss_z'][-1]:.4f}")

    # ---------- 9. 64-bit dual-envelope readout ----------
    t0 = time.perf_counter()
    readout = DualEnvelopeReadout(K_gate,
                                  chirp_a=1e-4,
                                  chirp_b=0.0, chirp_c=0.0)
    dual = readout.readout(prop["E_out"], prop["B_out"])
    print(f"[9] Dual-envelope readout   {(time.perf_counter()-t0)*1e3:8.2f} ms")
    print(f"      S_pos  (real plane)   : {dual['S_pos']:+.6f}")
    print(f"      S_neg  (imag plane)   : {dual['S_neg']:+.6f}")
    print(f"      S_all                 : {dual['S_all']:+.6f}")
    print(f"      H (entropy)           : {dual['H']:.6f}")
    print(f"      m (mass)              : {dual['m']:.6f}")

    # ---------- 10. Receiver — decode labels from the output field ----------
    samples_E = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, N_samples),
                          prop["E_out"])
    samples_B = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, N_samples),
                          prop["B_out"])
    # normalise by the envelope before phase extraction
    ref = np.interp(np.linspace(0, 1, N_v),
                    np.linspace(0, 1, N_samples), amp_env)
    ref = np.maximum(ref, 1e-6)
    phi_hat = np.arctan2(samples_B / ref, samples_E / ref) - phase0
    labels_hat = np.round(k_labels * phi_hat / (2 * PI)).astype(int) % k_labels

    # ---------- 11. UGC evaluation ----------
    sat_true = sat_hat = 0
    for i, j, perm in edges:
        if perm[labels_true[i]] == labels_true[N_left + j]:
            sat_true += 1
        if perm[labels_hat[i]] == labels_hat[N_left + j]:
            sat_hat += 1
    completeness_true = sat_true / len(edges)
    completeness_hat  = sat_hat / len(edges)
    decode_accuracy   = int(np.sum(labels_true == labels_hat)) / len(labels_true)

    print()
    print("--- UGC transmission result ---")
    print(f"  edges                    : {len(edges)}")
    print(f"  completeness (true)      : {completeness_true:.6f}")
    print(f"  completeness (decoded)   : {completeness_hat:.6f}")
    print(f"  soundness bound (1/k)    : {1.0/k_labels:.6f}")
    print(f"  decode accuracy          : {decode_accuracy*100:.2f} %")

    # ---------- 12. Plot ----------
    fig = plt.figure(figsize=(17, 12))
    gs  = GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.4)

    # (a) Chip layout
    ax = fig.add_subplot(gs[0, :2])
    ax.set_xlim(-0.5, N_wg - 0.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(range(N_wg))
    ax.set_yticks([])
    ax.set_xlabel("waveguide index (TSP-ordered)")
    ax.set_title(f"Photonic chip layout  ·  {N_wg} waveguides (TSP-routed)")

    # input modulators, waveguides, output couplers
    for k_wg in range(N_wg):
        # input modulator
        ax.add_patch(FancyBboxPatch((k_wg - 0.35, 2.5), 0.7, 0.6,
                                    boxstyle="round,pad=0.02",
                                    facecolor="#3a7bd5", edgecolor="k"))
        # waveguide
        ax.add_patch(Rectangle((k_wg - 0.05, 1.0), 0.1, 1.4,
                               facecolor="#16a085", edgecolor="k"))
        # output coupler
        ax.add_patch(FancyBboxPatch((k_wg - 0.35, 0.2), 0.7, 0.6,
                                    boxstyle="round,pad=0.02",
                                    facecolor="#e67e22", edgecolor="k"))

    # highlight TSP order with arrows
    order = prop["tsp_route"]
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        ax.annotate("", xy=(b, 3.2), xytext=(a, 3.2),
                    arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2))
    ax.scatter(order, [3.2] * len(order), c="#8e44ad", s=20, zorder=5)

    # (b) Elliptic Möbius amplitude
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t_env, zeta_mag, color="#3a7bd5", lw=1.4)
    ax.fill_between(t_env, 0, zeta_mag, color="#3a7bd5", alpha=0.25)
    ax.set_xlabel("t"); ax.set_ylabel("|ζ(t)|")
    ax.set_title(f"Elliptic Möbius amplitude\n(K={K_gate}, Basel err = {basel_err:.4f})")
    ax.grid(True, alpha=0.3)

    # (c) 6-vertex set signal
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(np.arange(6), set_signal["amplitudes"], color="#16a085", width=0.7)
    ax.set_xlabel("mode"); ax.set_ylabel("amplitude")
    ax.set_title(f"6-vertex set  ·  Π = {set_signal['scalar']:+.4f}")
    ax.grid(True, alpha=0.3)

    # (d) Transmitted fields along the chip
    ax = fig.add_subplot(gs[1, 1:])
    ax.plot(prop["z"], prop["E_in"], color="#2ecc71", lw=1.0,
            alpha=0.6, label="E_in")
    ax.plot(prop["z"], prop["E_out"], color="#e74c3c", lw=1.3,
            label="E_out (after loss + dispersion)")
    ax.plot(prop["z"], prop["loss_z"], color="#8e44ad", lw=1.0,
            ls="--", label="loss factor")
    ax.set_xlabel("position along waveguide (z/L)")
    ax.set_ylabel("field")
    ax.set_title("Field propagation through the chip")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (e) Chip angle sort
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(np.arange(N_wg), order, c="#8e44ad", s=40)
    ax.plot(np.arange(N_wg), order, "-", color="#8e44ad", alpha=0.5)
    ax.set_xlabel("rank"); ax.set_ylabel("waveguide index")
    ax.set_title("TSP route (chip-g bucket sort)")
    ax.grid(True, alpha=0.3)

    # (f) Dual-envelope readout
    ax = fig.add_subplot(gs[2, 1:])
    half = len(dual["mag_pos"])
    ax.plot(np.arange(half), dual["mag_pos"], color="#3a7bd5",
            lw=1.2, label="positive half (symmetric)")
    ax.plot(np.arange(half), dual["mag_neg"], color="#e74c3c",
            lw=1.2, label="negative half (asymmetric)")
    ax.plot(np.arange(half), dual["mag_neg"] * dual["chirp_neg"],
            color="#e67e22", lw=1.0, ls="--", label="negative × chirp")
    ax.set_xlabel("mode index within half")
    ax.set_ylabel("|field|")
    ax.set_title(f"Dual-envelope readout  ·  "
                 f"S_pos = {dual['S_pos']:+.3f}, "
                 f"S_neg = {dual['S_neg']:+.3f}, "
                 f"S_all = {dual['S_all']:+.3f}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (g) Label comparison
    ax = fig.add_subplot(gs[3, :2])
    ax.scatter(np.arange(N_v), labels_true, s=42, c="#3a7bd5",
               edgecolors="k", label="true σ(v)")
    ax.scatter(np.arange(N_v), labels_hat, s=22, c="#e74c3c",
               marker="x", label="decoded σ̂(v)")
    ax.set_xlabel("vertex index"); ax.set_ylabel("label")
    ax.set_title(f"Label recovery  ·  accuracy = {decode_accuracy*100:.1f} %")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (h) Completeness bars
    ax = fig.add_subplot(gs[3, 2])
    bars = [completeness_true, completeness_hat, 1.0 / k_labels]
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
        "Photonic chip simulation  ·  UGC + elliptic Möbius amplitude "
        f"+ 64-bit dual envelope  ·  accuracy = {decode_accuracy*100:.1f} %",
        fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return dict(
        gate=gate, set_signal=set_signal,
        labels_true=labels_true, labels_hat=labels_hat,
        edges=edges, chip=chip, prop=prop,
        dual=dual,
        completeness_true=completeness_true,
        completeness_hat=completeness_hat,
        decode_accuracy=decode_accuracy,
        basel_error=basel_err,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24, K_gate=20,
             N_left=60, N_right=60,
             k_labels=4, degree=3,
             N_wg=12, N_samples=512,
             seed_vertices=42)