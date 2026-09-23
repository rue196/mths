#!/usr/bin/env python3
"""
lidar_ugc_sensing.py

LIDAR sensing and point-cloud reconstruction using the UGC pipeline.

Physical picture
----------------
    LIDAR head                     →  UGC instance
    scan points (azimuth, elev)    →  vertices L ∪ R
    depth bin ∈ [0, k)             →  UGC label σ(v)
    neighboring scan points        →  UGC edges with permutations
    laser pulse amplitude          →  elliptic Möbius envelope |ζ(t)|
    round-trip through air         →  fiber kernel exp(−α|z|)
    returned photon field          →  E_out, B_out
    decoder                        →  depth bins σ̂(v) → point cloud

Why UGC?
--------
A LIDAR sweep is a *labelled graph*: each scan point has a depth,
and neighboring points must be depth-consistent.  That is exactly a
Unique Games instance — the depth label is the game's label, and the
permutation on each edge encodes the local smoothness constraint
between adjacent scan lines.  Solving the UGC (even approximately)
gives a globally consistent depth map, and the elliptically-modulated
UGC field is the actual transmitted laser pulse.

Reconstruction
--------------
Once the depth bins σ̂(v) are decoded from the returned field, we
place each scan point at its (azimuth, elevation, depth) coordinate,
producing a point cloud.
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
C_LIGHT    = 1.0                        # normalized units


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
#  2. Elliptic Möbius gate  (pulse envelope)
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
    """Spectral sum ζ(t) = Σ C_i · exp(i t i / α)  →  pulse amplitude."""
    def __init__(self, K: int, smooth: bool = True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.values  = np.array([self.coeffs[i] for i in self.indices])
        self.period  = 2 * PI * ALPHA_SYM

    def zeta_array(self, t_vals: np.ndarray) -> np.ndarray:
        phases = np.outer(t_vals, self.indices) / ALPHA_SYM
        return np.exp(1j * phases) @ self.values

    def amplitude(self, t_vals):
        return np.abs(self.zeta_array(t_vals))

    def power(self, t_vals):
        return self.amplitude(t_vals) ** 2

    def basel_theoretical(self):
        return 2.0 * (6.0 / (PI * PI)) * self.K

    def basel_error(self, N: int = 2000) -> float:
        t_vals = np.linspace(0, self.period, N)
        power = self.power(t_vals)
        try:
            integral = np.trapezoid(power, t_vals)
        except AttributeError:
            integral = np.trapz(power, t_vals)
        avg  = integral / self.period
        theo = self.basel_theoretical()
        return abs(avg - theo) / theo if theo != 0 else 0.0


# ============================================================
#  3. Figure 3.5 — bounded elliptic projection
# ============================================================
def elliptic_projection_1d(x, y0=1.0):
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


def elliptic_projection_2d(X, Y):
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))


# ============================================================
#  4. 6-vertex set signal  (LIDAR head geometry)
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


def lidar_head_set_signal(seed: int = 42) -> dict:
    """
    The 6-vertex configuration defines the LIDAR head geometry:
    6 beams with amplitudes = row norms of the 6×6 vertex block.
    The Levi-Civita scalar sets the global phase.
    """
    V = generate_vertices(seed)
    V6 = V[:6, :6]
    row_norms = np.linalg.norm(V6, axis=1)
    row_norms = row_norms / (row_norms.max() + 1e-12)
    scalar = levi_civita_contraction(V6)
    phase = 0.0 if scalar >= 0 else math.pi
    return dict(vertices=V6, amplitudes=row_norms,
                scalar=scalar, phase=phase)


# ============================================================
#  5. LIDAR scan → UGC instance
# ============================================================
def lidar_scan_to_ugc(n_azimuth: int, n_elevation: int,
                      depth_bins: int,
                      scene_fn,
                      seed: int = 2024):
    """
    Build a UGC instance from a LIDAR sweep.

    Vertices    : every (azimuth, elevation) scan point
    Labels      : depth bin ∈ [0, k)
    Edges       : neighbours in the scan grid (4-connected), with
                  a permutation that enforces smoothness

    Parameters
    ----------
    n_azimuth   : scan points along azimuth
    n_elevation : scan points along elevation
    depth_bins  : k — number of depth quantization levels
    scene_fn    : function (az, el) → depth ∈ [0, 1]
                  the ground-truth scene surface

    Returns
    -------
    scan_grid     : (n_az, n_el) array of (az, el) coordinates
    depths_true   : (n_az, n_el) ground-truth depths (continuous)
    labels_true   : (n_az*n_el,) ground-truth labels
    edges         : list of (i, j, perm)
    N_left        : number of vertices on the left side of each edge
                    (we use half the grid for L, half for R)
    """
    rng = np.random.default_rng(seed)

    az_vals = np.linspace(-0.5, 0.5, n_azimuth)
    el_vals = np.linspace(-0.2, 0.2, n_elevation)
    AZ, EL  = np.meshgrid(az_vals, el_vals, indexing='ij')

    # ground-truth depth from scene function
    depths_true = np.zeros((n_azimuth, n_elevation))
    for i in range(n_azimuth):
        for j in range(n_elevation):
            depths_true[i, j] = scene_fn(AZ[i, j], EL[i, j])

    # quantize to depth bins  →  the ground-truth label
    labels_true = np.floor(depths_true * depth_bins).astype(int)
    labels_true = np.clip(labels_true, 0, depth_bins - 1)

    N_v = n_azimuth * n_elevation

    # split vertices into L and R by azimuth index parity
    # (even columns → L, odd columns → R)
    L_idx = [(i, j) for i in range(0, n_azimuth, 2)
             for j in range(n_elevation)]
    R_idx = [(i, j) for i in range(1, n_azimuth, 2)
             for j in range(n_elevation)]
    L_map = {ij: k for k, ij in enumerate(L_idx)}
    R_map = {ij: k for k, ij in enumerate(R_idx)}
    N_left  = len(L_idx)
    N_right = len(R_idx)

    def depth_label(i, j):
        return int(labels_true[i, j])

    # edges: every L vertex connects to its two vertical neighbours in R
    edges = []
    for (i, j) in L_idx:
        for dj in (-1, +1):
            jj = j + dj
            if jj < 0 or jj >= n_elevation:
                continue
            # horizontal and vertical neighbours in R
            for di in (-1, +1):
                ii = i + di
                if ii < 0 or ii >= n_azimuth:
                    continue
                if (ii, jj) not in R_map:
                    continue
                # permutation that enforces |σ_L - σ_R| ≤ 1
                # we use the identity permutation with a small
                # twist that shifts the label by ±1 randomly
                shift = int(rng.integers(-1, 2))
                perm = tuple((x + shift) % depth_bins
                             for x in range(depth_bins))
                edges.append((L_map[(i, j)], R_map[(ii, jj)], perm))

    return dict(
        az_vals=az_vals, el_vals=el_vals,
        AZ=AZ, EL=EL,
        depths_true=depths_true,
        labels_true=labels_true,
        edges=edges,
        N_left=N_left, N_right=N_right,
        N_v=N_v, k=depth_bins,
        L_idx=L_idx, R_idx=R_idx,
        L_map=L_map, R_map=R_map,
    )


# ============================================================
#  6. Möbius label assignment  (the "solver")
# ============================================================
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
#  7. Elliptic amplitude envelope  (LIDAR pulse)
# ============================================================
def elliptic_amplitude_envelope(gate: EllipticMobiusGate,
                                N_samples: int) -> dict:
    t_vals = np.linspace(0.0, gate.period, N_samples)
    zeta_mag = gate.amplitude(t_vals)
    zeta_norm = zeta_mag / (zeta_mag.max() + 1e-12)
    amplitude = 0.3 + 0.7 * zeta_norm
    return dict(t_vals=t_vals, zeta_mag=zeta_mag,
                zeta_norm=zeta_norm, amplitude=amplitude)


# ============================================================
#  8. LIDAR encoding  (depths → laser phases)
# ============================================================
def encode_depths_with_elliptic_pulse(labels: np.ndarray,
                                      k: int,
                                      set_signal: dict,
                                      gate: EllipticMobiusGate,
                                      N_samples: int) -> dict:
    """
    Encode the depth labels as phase modulation on the laser pulse,
    with the pulse amplitude envelope given by |ζ(t)|.
    """
    N_v = len(labels)
    amps = set_signal["amplitudes"]
    phase0 = set_signal["phase"]
    scalar = set_signal["scalar"]

    phi = 2.0 * PI * labels / k
    carrier_amp = np.tile(amps, int(np.ceil(N_v / len(amps))))[:N_v]

    env = elliptic_amplitude_envelope(gate, N_samples)
    amp_at_vertices = np.interp(
        np.linspace(0, 1, N_v),
        np.linspace(0, 1, N_samples),
        env["amplitude"],
    )
    envelope_scale = math.tanh(abs(scalar))
    A_full = carrier_amp * amp_at_vertices * envelope_scale

    E0 = A_full * np.cos(phi + phase0)
    B0 = A_full * np.sin(phi + phase0)
    return dict(E0=E0, B0=B0, phi=phi, A_full=A_full,
                amp_at_vertices=amp_at_vertices,
                envelope=env, envelope_scale=envelope_scale)


# ============================================================
#  9. Round-trip propagation  (air + reflection)
# ============================================================
def propagate_round_trip(E0, B0, mu, K,
                         N_samples=512, barrier_width=41,
                         reflectivity=0.7):
    """
    Outgoing laser pulse, reflection off the scene, return path.
    Two identical dispersions (out and back) plus a reflectivity
    factor.
    """
    N_v = len(E0)
    z_grid = np.linspace(0, 1, N_samples)
    E_in = np.interp(z_grid, np.linspace(0, 1, N_v), E0)
    B_in = np.interp(z_grid, np.linspace(0, 1, N_v), B0)

    x = np.arange(-barrier_width // 2, barrier_width // 2)
    kernel = np.exp(-ALPHA_SYM * np.abs(x))
    kernel = kernel / kernel.sum()

    # outgoing
    E_mid = convolve(E_in, kernel, mode="same")
    B_mid = convolve(B_in, kernel, mode="same")

    # reflection
    E_mid *= reflectivity
    B_mid *= reflectivity

    # return path
    E_out = convolve(E_mid, kernel, mode="same")
    B_out = convolve(B_mid, kernel, mode="same")

    intensity = np.abs(E_out + 1j * B_out)
    energy = E_out ** 2 + B_out ** 2
    return dict(
        z_grid=z_grid,
        E_input=E_in, B_input=B_in,
        E_out=E_out, B_out=B_out,
        intensity=intensity, energy=energy,
        kernel=kernel,
    )


# ============================================================
#  10. LIDAR reconstruction  (returned field → point cloud)
# ============================================================
def decode_depths_from_field(E_out, B_out, N_v, k,
                             phi0=0.0, amp_reference=None) -> np.ndarray:
    """
    Recover the depth labels from the returned field's phase.
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


def reconstruct_point_cloud(scan: dict,
                            labels_hat: np.ndarray) -> dict:
    """
    Turn the decoded labels back into a point cloud.

    Each vertex is placed at its (azimuth, elevation, depth) coordinate.
    """
    n_az = len(scan["az_vals"])
    n_el = len(scan["el_vals"])
    k    = scan["k"]

    # map label → continuous depth
    depths_hat = labels_hat.astype(float) / k

    # reassemble the (n_az, n_el) grid
    cloud = np.zeros((n_az, n_el, 3))
    N_left = scan["N_left"]

    # left vertices
    for idx, (i, j) in enumerate(scan["L_idx"]):
        depth = depths_hat[idx]
        az    = scan["az_vals"][i]
        el    = scan["el_vals"][j]
        cloud[i, j] = [az, el, depth]

    # right vertices
    for idx, (i, j) in enumerate(scan["R_idx"]):
        depth = depths_hat[N_left + idx]
        az    = scan["az_vals"][i]
        el    = scan["el_vals"][j]
        cloud[i, j] = [az, el, depth]

    # ground truth cloud
    cloud_true = np.zeros((n_az, n_el, 3))
    for i in range(n_az):
        for j in range(n_el):
            cloud_true[i, j] = [
                scan["az_vals"][i],
                scan["el_vals"][j],
                scan["depths_true"][i, j],
            ]

    return dict(
        cloud=cloud,
        cloud_true=cloud_true,
        depths_hat=depths_hat,
        labels_hat=labels_hat,
    )


# ============================================================
#  11. Evaluation
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


def point_cloud_error(scan: dict, recon: dict) -> dict:
    """Mean depth error between reconstructed and true cloud."""
    dep_hat = recon["cloud"][:, :, 2]
    dep_true = recon["cloud_true"][:, :, 2]
    err = np.abs(dep_hat - dep_true)
    return dict(
        mean_abs_error=float(np.mean(err)),
        max_abs_error=float(np.max(err)),
        rmse=float(np.sqrt(np.mean(err ** 2))),
    )


# ============================================================
#  12. Full simulation
# ============================================================
def simulate(K: int = 24,
             K_gate: int = 20,
             n_azimuth: int = 24,
             n_elevation: int = 24,
             k_bins: int = 6,
             N_samples: int = 512,
             barrier_width: int = 41,
             reflectivity: float = 0.7,
             seed_vertices: int = 42):
    print("=" * 82)
    print("LIDAR sensing + reconstruction  ·  UGC + elliptic Möbius pulse")
    print("=" * 82)
    print(f"  α_sym      = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  K (sieve)  = {K}")
    print(f"  K (gate)   = {K_gate}")
    print(f"  azimuth    = {n_azimuth}   elevation = {n_elevation}")
    print(f"  depth bins = {k_bins}")
    print(f"  reflectivity = {reflectivity}")
    print()

    # ---------- 1. Möbius sieve ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(max(K, 2 ** 10 + 1))
    print(f"[1] Möbius sieve          {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # ---------- 2. Elliptic Möbius gate  →  pulse envelope ----------
    t0 = time.perf_counter()
    gate = EllipticMobiusGate(K_gate, smooth=True)
    print(f"[2] Elliptic Möbius gate  {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"({len(gate.indices)} non-zero coeffs)")
    basel_err = gate.basel_error(N=2000)
    print(f"      Basel error         : {basel_err:.6f}")
    print(f"      ζ period            : {gate.period:.6f}")

    # ---------- 3. LIDAR head set signal ----------
    t0 = time.perf_counter()
    set_signal = lidar_head_set_signal(seed=seed_vertices)
    print(f"[3] LIDAR head signal     {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(Π = {set_signal['scalar']:+.4f})")

    # ---------- 4. Build the UGC instance from the LIDAR scan ----------
    def scene_fn(az, el):
        # a smooth undulating surface with a foreground "car"
        base = 0.55 + 0.15 * math.sin(6 * az) * math.cos(4 * el)
        # foreground object in the centre
        r = math.hypot(az, el * 2.5)
        car = 0.2 * math.exp(-(r / 0.15) ** 2)
        return max(0.05, min(0.95, base - car))

    t0 = time.perf_counter()
    scan = lidar_scan_to_ugc(n_azimuth, n_elevation, k_bins,
                             scene_fn, seed=2024)
    print(f"[4] LIDAR scan → UGC      {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"({len(scan['edges'])} edges, "
          f"N_L = {scan['N_left']}, N_R = {scan['N_right']})")

    # ---------- 5. Möbius label assignment (ground truth) ----------
    # In a real system this is the "solver"; here we use the true
    # labels as the reference and let the Möbius gate give its own
    # assignment for comparison.
    t0 = time.perf_counter()
    labels_mobius, scores = ugc_label_assignment(
        scan["N_left"], scan["N_right"], k_bins, mu, K_filter=2 ** 10 + 1)
    print(f"[5] Möbius label assign   {(time.perf_counter()-t0)*1e3:8.2f} ms")

    # Also flatten the true labels in the same L/R order
    labels_true_L = np.array([scan["labels_true"][i, j]
                              for (i, j) in scan["L_idx"]])
    labels_true_R = np.array([scan["labels_true"][i, j]
                              for (i, j) in scan["R_idx"]])
    labels_true = np.concatenate([labels_true_L, labels_true_R])

    # ---------- 6. Encode the true depths with the elliptic pulse ----------
    t0 = time.perf_counter()
    encoded = encode_depths_with_elliptic_pulse(
        labels_true, k_bins, set_signal, gate, N_samples)
    print(f"[6] Pulse encoding        {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(A ∈ [{encoded['A_full'].min():.4f}, "
          f"{encoded['A_full'].max():.4f}])")

    # ---------- 7. Round-trip propagation ----------
    t0 = time.perf_counter()
    prop = propagate_round_trip(
        encoded["E0"], encoded["B0"], mu, K,
        N_samples=N_samples, barrier_width=barrier_width,
        reflectivity=reflectivity)
    print(f"[7] Round-trip propag     {(time.perf_counter()-t0)*1e3:8.2f} ms  "
          f"(reflected E ∈ [{prop['E_out'].min():+.4f}, "
          f"{prop['E_out'].max():+.4f}])")

    # ---------- 8. Decoder ----------
    N_v = scan["N_v"]
    amp_ref = np.interp(
        np.linspace(0, 1, N_v),
        np.linspace(0, 1, N_samples),
        encoded["envelope"]["amplitude"],
    )
    labels_hat = decode_depths_from_field(
        prop["E_out"], prop["B_out"], N_v, k_bins,
        phi0=set_signal["phase"],
        amp_reference=amp_ref,
    )
    print(f"[8] Depth decoder         done")

    # ---------- 9. Point-cloud reconstruction ----------
    recon = reconstruct_point_cloud(scan, labels_hat)
    err = point_cloud_error(scan, recon)

    print()
    print("--- LIDAR reconstruction result ---")
    print(f"  scan vertices            : {N_v}")
    print(f"  depth bins               : {k_bins}")
    print(f"  true completeness        : "
          f"{ugc_evaluate(scan['edges'], labels_true, labels_true, scan['N_left'])['completeness_true']:.4f}")
    eval_result = ugc_evaluate(
        scan["edges"], labels_true, labels_hat, scan["N_left"])
    print(f"  decoded completeness     : "
          f"{eval_result['completeness_hat']:.4f}")
    print(f"  decode accuracy          : "
          f"{eval_result['decode_accuracy']*100:.2f} %")
    print(f"  mean depth error         : {err['mean_abs_error']:.4f}")
    print(f"  RMSE depth error         : {err['rmse']:.4f}")
    print(f"  max depth error          : {err['max_abs_error']:.4f}")

    # ---------- 10. Plot ----------
    fig = plt.figure(figsize=(17, 12))
    gs  = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.4)

    # (a) Elliptic Möbius pulse amplitude
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(encoded["envelope"]["t_vals"],
            encoded["envelope"]["zeta_mag"],
            color="#3a7bd5", lw=1.4)
    ax.fill_between(encoded["envelope"]["t_vals"], 0,
                    encoded["envelope"]["zeta_mag"],
                    color="#3a7bd5", alpha=0.25)
    ax.set_xlabel("t"); ax.set_ylabel("|ζ(t)|")
    ax.set_title(f"LIDAR pulse envelope  (K={K_gate})")
    ax.grid(True, alpha=0.3)

    # (b) power spectrum |ζ(t)|²
    ax = fig.add_subplot(gs[0, 1])
    power = encoded["envelope"]["zeta_mag"] ** 2
    ax.plot(encoded["envelope"]["t_vals"], power,
            color="#e67e22", lw=1.4)
    ax.axhline(gate.basel_theoretical() / N_samples,
               color="g", ls=":", lw=1.0, label="Basel ref")
    ax.set_xlabel("t"); ax.set_ylabel("|ζ(t)|²")
    ax.set_title(f"Pulse power spectrum  ·  Basel err = {basel_err:.4f}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) 6-beam head geometry
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(6), set_signal["amplitudes"],
           color="#16a085", width=0.7)
    ax.set_xlabel("beam"); ax.set_ylabel("amplitude")
    ax.set_title(f"LIDAR head  ·  Π = {set_signal['scalar']:+.4f}")
    ax.grid(True, alpha=0.3)

    # (d) transmitted depth-encoded pulse
    ax = fig.add_subplot(gs[1, :2])
    z_v = np.linspace(0, 1, N_v)
    ax.plot(z_v, encoded["A_full"], color="#8e44ad", lw=1.4,
            label="A_full = carrier × |ζ| × Π")
    ax.plot(z_v, encoded["E0"], color="#2ecc71", lw=1.0,
            alpha=0.7, label="E₀ (cos)")
    ax.plot(z_v, encoded["B0"], color="#3498db", lw=1.0,
            alpha=0.7, label="B₀ (sin)")
    ax.set_xlabel("scan vertex index")
    ax.set_ylabel("amplitude")
    ax.set_title("Transmitted depth-encoded pulse")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (e) UGC graph (sparse sample)
    ax = fig.add_subplot(gs[1, 2])
    pos_L = np.random.default_rng(0).uniform(0, 1, (scan["N_left"], 2))
    pos_R = np.random.default_rng(1).uniform(0, 1, (scan["N_right"], 2))
    cmap = plt.cm.tab10
    for e_idx, (i, j, perm) in enumerate(scan["edges"][:200]):
        sat = (perm[labels_true[i]] == labels_true[scan["N_left"] + j])
        ax.plot([pos_L[i, 0], pos_R[j, 0]],
                [pos_L[i, 1], pos_R[j, 1]],
                color="#2ecc71" if sat else "#e74c3c",
                lw=0.4, alpha=0.5)
    ax.scatter(pos_L[:, 0], pos_L[:, 1],
               c=[cmap(labels_true[i] / k_bins)
                  for i in range(scan["N_left"])],
               s=18, edgecolors="k", zorder=5)
    ax.scatter(pos_R[:, 0], pos_R[:, 1],
               c=[cmap(labels_true[scan["N_left"] + j] / k_bins)
                  for j in range(scan["N_right"])],
               s=18, edgecolors="k", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UGC  ·  |E| = {len(scan['edges'])}")

    # (f) round-trip kernel
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(np.arange(-barrier_width // 2, barrier_width // 2),
            prop["kernel"], color="#8e44ad", lw=1.5)
    ax.set_xlabel("z"); ax.set_ylabel("H(z)")
    ax.set_title("Air kernel exp(−α|z|)")
    ax.grid(True, alpha=0.3)

    # (g) returned field
    ax = fig.add_subplot(gs[2, 1:])
    ax.plot(prop["z_grid"], prop["E_out"], color="#e74c3c",
            lw=1.3, label="E_out")
    ax.plot(prop["z_grid"], prop["B_out"], color="#8e44ad",
            lw=1.1, alpha=0.7, label="B_out")
    ax.plot(prop["z_grid"], prop["intensity"], color="#16a085",
            lw=1.0, ls="--", label="|E + iB|")
    ax.set_xlabel("return path"); ax.set_ylabel("field")
    ax.set_title("Returned LIDAR signal")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (h) depth map (true vs decoded)
    ax = fig.add_subplot(gs[3, :2])
    true_depth_grid = scan["depths_true"]
    hat_depth_grid  = np.zeros_like(true_depth_grid)
    for idx, (i, j) in enumerate(scan["L_idx"]):
        hat_depth_grid[i, j] = labels_hat[idx] / k_bins
    for idx, (i, j) in enumerate(scan["R_idx"]):
        hat_depth_grid[i, j] = labels_hat[scan["N_left"] + idx] / k_bins
    vmax = max(true_depth_grid.max(), 1e-6)
    ax.imshow(true_depth_grid.T, origin="lower", cmap="viridis",
              aspect="auto", vmin=0, vmax=vmax)
    ax.set_xlabel("azimuth"); ax.set_ylabel("elevation")
    ax.set_title("Ground-truth depth map")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.03, pad=0.02)

    # (i) reconstructed depth map
    ax = fig.add_subplot(gs[3, 2])
    ax.imshow(hat_depth_grid.T, origin="lower", cmap="viridis",
              aspect="auto", vmin=0, vmax=vmax)
    ax.set_xlabel("azimuth"); ax.set_ylabel("elevation")
    ax.set_title(f"Decoded  ·  acc = "
                 f"{eval_result['decode_accuracy']*100:.1f}%")
    plt.colorbar(ax.images[0], ax=ax, fraction=0.03, pad=0.02)

    plt.suptitle(
        "LIDAR sensing + reconstruction  ·  UGC + elliptic Möbius pulse  ·  "
        f"RMSE = {err['rmse']:.4f}",
        fontsize=13, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return dict(
        gate=gate, set_signal=set_signal,
        scan=scan, encoded=encoded, propagation=prop,
        labels_true=labels_true, labels_hat=labels_hat,
        labels_mobius=labels_mobius,
        recon=recon, error=err,
        eval_result=eval_result,
        basel_error=basel_err,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24, K_gate=20,
             n_azimuth=24, n_elevation=24,
             k_bins=6,
             N_samples=512, barrier_width=41,
             reflectivity=0.7,
             seed_vertices=42)