#!/usr/bin/env python3
"""
quantum_geo_location.py
=======================

Earth as a boson blob.  Quantum geo-location via:
    • 6-vertex configuration on the Earth boson field
    • signals transmitted at 2c (twice the speed of light)
    • 6D spinor contraction to 3D (Levi-Civita rank-6)
    • NORM-based tunneling reconstruction
      (from tunneling_prob_diriclet_norm.py)

Layout
------
Boson blob      Earth as 2D Gaussian on a lat/lon grid
Fermion lines   six world lines from the vertices, passing through
                the blob (M_BF = 0, no scattering)
6D spinor       Z_ij, i,j ∈ [1,6], one 6×6 complex matrix per vertex
Contraction     Π_6(z) = Σ_σ ε(σ) Π_k z_{k,σ(k)}     (720 terms)
6D → 3D         keep the first 3 diagonal modes of the contracted spinor
Transmission    t = d / (2c)   — twice the speed of light
Reception       Z_eff(θ) = √NORM · Z(θ) + √(1−NORM) · e^{iπ/2} · Z(θ)
                (from the tunneling file: NORM = 1 − e^{−α(π+e)})
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
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)              # ≈ 2.362
DENSITY    = 6.0 / (PI * PI)             # ≈ 0.6079

# --- NORM from tunneling_prob_diriclet_norm.py ---
NORM        = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
TUNNEL_PROB = 1.0 - NORM
CONFINE_PROB = NORM
TUNNEL_AMP  = math.sqrt(TUNNEL_PROB)
CONFINE_AMP = math.sqrt(CONFINE_PROB)
TUNNEL_PHASE = PI / 2.0

# --- physical / world constants ---
C_LIGHT     = 3.0e8                      # m/s
SIGNAL_MULT = 2.0                        # 2c  — twice the speed of light
R_EARTH_KM  = 6371.0
R_EARTH_M   = R_EARTH_KM * 1000.0


# ============================================================
#  Levi-Civita rank-6
# ============================================================
def _levi_civita_terms(d: int) -> List[Tuple[Tuple[int, ...], int]]:
    terms = []
    for perm in permutations(range(d)):
        inv = sum(1 for i in range(d) for j in range(i + 1, d)
                  if perm[i] > perm[j])
        terms.append((perm, (-1) ** inv))
    return terms


TERMS_6 = _levi_civita_terms(6)          # 720 terms


def levi_civita_6(z: np.ndarray) -> complex:
    """Π_6(z) = Σ_σ ε(σ) · Π_k z[k, σ(k)]."""
    total = 0.0 + 0.0j
    for perm, sign in TERMS_6:
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= z[k, c]
        total += sign * prod
    return total


def contract_6d_to_3d(z6: np.ndarray) -> np.ndarray:
    """
    6D → 3D reduction: keep the first 3 diagonal modes.

    A 6×6 spinor Z has 36 entries.  The 3D reduction Z3 is
    the first 3 diagonal entries, rescaled by the rank-6
    Levi-Civita contraction to preserve phase.

    Returns a length-3 complex vector.
    """
    Pi6 = levi_civita_6(z6)
    phase = math.atan2(Pi6.imag, Pi6.real)
    diag3 = np.array([z6[0, 0], z6[1, 1], z6[2, 2]], dtype=complex)
    return diag3 * np.exp(1j * phase / 6.0)     # /6 spreads the rank-6 phase


# ============================================================
#  Earth boson blob  (2D Gaussian on a lat/lon grid)
# ============================================================
@dataclass
class EarthBlob:
    N: int = 200                                  # grid size
    R_e: float = 1.0                              # Earth radius, in grid units
    def __post_init__(self):
        lin = np.linspace(-1.5 * self.R_e, 1.5 * self.R_e, self.N)
        self.x = lin
        self.y = lin
        self.X, self.Y = np.meshgrid(lin, lin, indexing="ij")
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.b = np.exp(-self.R**2 / (2 * self.R_e**2))
        # fermion gas lives in the outer shell
        self.f = 0.15 * (1.0 - self.b) * np.exp(
            -self.R**2 / (2 * (1.5 * self.R_e)**2))
        # total boson mass (discrete)
        dx = (lin[1] - lin[0])
        self.M_bh = float(np.sum(self.b) * dx * dx)


# ============================================================
#  6-vertex configuration on the sphere
# ============================================================
@dataclass
class VertexConfig:
    positions: np.ndarray     # (6, 2) on the lat/lon grid
    amplitudes: np.ndarray    # (6,)  row norms of the 6×6 matrix
    V6: np.ndarray            # 6×6 vertex matrix
    Pi6_ref: complex          # reference contraction
    phase_ref: float


def make_vertex_config(seed: int = 42,
                       earth: EarthBlob = None) -> VertexConfig:
    """
    Six vertices on the sphere.  The 6×6 matrix V6 provides the
    per-vertex amplitude (row norms) and the reference contraction
    Π_6 (rank-6 Levi-Civita).
    """
    rng = np.random.default_rng(seed)
    # vertex positions on the grid — spread around the globe
    angles = np.linspace(0.0, 2.0 * PI, 6, endpoint=False)
    R = (earth.R_e if earth is not None else 1.0) * 0.8
    xs = R * np.cos(angles)
    ys = R * np.sin(angles)
    positions = np.column_stack([xs, ys])

    # 6×6 vertex matrix — bounded Gaussian
    V = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    V = V / (np.max(np.abs(V)) + 1e-12) * 0.5
    # center for zero-sum
    V = V - V.mean()
    row_norms = np.linalg.norm(V, axis=1)
    row_norms = row_norms / (row_norms.max() + 1e-12)

    Pi6_ref = levi_civita_6(V)
    phase_ref = math.atan2(Pi6_ref.imag, Pi6_ref.real)

    return VertexConfig(
        positions=positions,
        amplitudes=row_norms,
        V6=V,
        Pi6_ref=Pi6_ref,
        phase_ref=phase_ref,
    )


# ============================================================
#  Transmission at 2c
# ============================================================
@dataclass
class TransmissionPath:
    src_idx: int
    dst_idx: int
    distance_km: float
    flight_time_s: float          # d / (2c)
    alpha_phase: float            # ALPHA_SYM · flight_time · R


def flight_time(distance_km: float,
                c_mult: float = SIGNAL_MULT) -> float:
    """t = d / (c_mult · c)."""
    d_m = distance_km * 1000.0
    return d_m / (c_mult * C_LIGHT)


def build_paths(cfg: VertexConfig,
                c_mult: float = SIGNAL_MULT) -> List[TransmissionPath]:
    """Full mesh of 6×5 = 30 directed paths between the 6 vertices."""
    paths: List[TransmissionPath] = []
    N = len(cfg.positions)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = cfg.positions[j, 0] - cfg.positions[i, 0]
            dy = cfg.positions[j, 1] - cfg.positions[i, 1]
            d_grid = math.hypot(dx, dy)            # in grid units (R_e)
            d_km = d_grid * R_EARTH_KM
            t_s = flight_time(d_km, c_mult)
            R_loc = (math.hypot(cfg.positions[i, 0], cfg.positions[i, 1])
                     + 1e-6)
            phase = ALPHA_SYM * t_s * R_loc * 1e3   # scale so phase is O(1)
            paths.append(TransmissionPath(
                src_idx=i, dst_idx=j,
                distance_km=d_km,
                flight_time_s=t_s,
                alpha_phase=phase,
            ))
    return paths


# ============================================================
#  Signal at each path
# ============================================================
def vertex_spinor(cfg: VertexConfig, idx: int,
                  phase: float) -> np.ndarray:
    """
    6×6 spinor for vertex `idx`:
        Z[k, c] = a_idx · V6[k, c] · exp(i · phase / 6)
    The /6 spreads the total phase over the 6 rows.
    """
    a = cfg.amplitudes[idx]
    row_phase = np.exp(1j * phase / 6.0)
    return a * cfg.V6 * row_phase


def received_spinor(cfg: VertexConfig, path: TransmissionPath,
                    noise: float = 0.0,
                    rng: np.random.Generator = None) -> np.ndarray:
    """
    Signal delivered at `path.dst_idx` from `path.src_idx` at 2c.
    Additive noise on the complex entries.
    """
    Z = vertex_spinor(cfg, path.src_idx, path.alpha_phase)
    if noise > 0.0 and rng is not None:
        Z = Z + noise * (rng.standard_normal(Z.shape)
                         + 1j * rng.standard_normal(Z.shape))
    return Z


# ============================================================
#  NORM-based reception
# ============================================================
def norm_reconstruct(Z: np.ndarray) -> np.ndarray:
    """
    NORM superposition from tunneling_prob_diriclet_norm.py:

        Z_eff(θ) = √NORM · Z(θ) + √(1−NORM) · e^{iπ/2} · Z(θ)

    The tunneling channel carries a π/2 phase shift relative to the
    confined channel.  Returns the reconstructed spinor.
    """
    return CONFINE_AMP * Z + TUNNEL_AMP * 1j * Z


def project_3d(Z: np.ndarray) -> np.ndarray:
    """6D → 3D by the Levi-Civita contraction + diagonal retention."""
    return contract_6d_to_3d(Z)


# ============================================================
#  Localization
# ============================================================
@dataclass
class GeoFix:
    """A single received path and its reconstructed 3D position."""
    src_idx: int
    dst_idx: int
    distance_km: float
    t_flight_ms: float
    Pi6_raw: complex
    Pi6_norm: complex
    z3: np.ndarray
    position_est_km: Tuple[float, float]    # (x, y) in km
    position_true_km: Tuple[float, float]
    error_km: float


def localize_path(cfg: VertexConfig,
                  path: TransmissionPath,
                  Z_received: np.ndarray) -> GeoFix:
    """
    Reconstruct the source position from a received spinor.

        raw contraction        Π_6(Z_received)
        NORM reconstruction    Z_eff = √NORM·Z + √(1−NORM)·e^{iπ/2}·Z
        NORM contraction       Π_6(Z_eff)
        6D → 3D                z3 = contract_6d_to_3d(Z_eff)
        position               recovered from |z3| and phase
    """
    Pi6_raw = levi_civita_6(Z_received)

    Z_eff = norm_reconstruct(Z_received)
    Pi6_norm = levi_civita_6(Z_eff)

    z3 = project_3d(Z_eff)

    # position estimate: the recovered phase and amplitude localize
    # the source by decoding the transit phase
    amp = float(np.mean(np.abs(z3)))
    phase = math.atan2(Pi6_norm.imag, Pi6_norm.real)

    # the transit phase encodes the source angle from the vertex
    r_est = (cfg.positions[path.dst_idx]
             + amp * np.array([math.cos(phase), math.sin(phase)]))
    true_src = cfg.positions[path.src_idx]

    err = float(np.hypot(r_est[0] - true_src[0],
                         r_est[1] - true_src[1]) * R_EARTH_KM)
    pos_est_km = (r_est[0] * R_EARTH_KM, r_est[1] * R_EARTH_KM)
    pos_true_km = (true_src[0] * R_EARTH_KM, true_src[1] * R_EARTH_KM)
    return GeoFix(
        src_idx=path.src_idx, dst_idx=path.dst_idx,
        distance_km=path.distance_km,
        t_flight_ms=path.flight_time_s * 1e3,
        Pi6_raw=Pi6_raw, Pi6_norm=Pi6_norm, z3=z3,
        position_est_km=pos_est_km,
        position_true_km=pos_true_km,
        error_km=err,
    )


# ============================================================
#  Full pipeline
# ============================================================
def run_geo_location(seed: int = 42,
                     c_mult: float = SIGNAL_MULT,
                     noise: float = 0.002,
                     N: int = 200) -> Dict:
    earth = EarthBlob(N=N, R_e=1.0)
    cfg   = make_vertex_config(seed=seed, earth=earth)
    paths = build_paths(cfg, c_mult=c_mult)
    rng   = np.random.default_rng(seed + 100)

    fixes: List[GeoFix] = []
    for p in paths:
        Z_recv = received_spinor(cfg, p, noise=noise, rng=rng)
        fixes.append(localize_path(cfg, p, Z_recv))

    errors = np.array([f.error_km for f in fixes])
    pi6_raw  = np.array([f.Pi6_raw  for f in fixes])
    pi6_norm = np.array([f.Pi6_norm for f in fixes])
    t_ms     = np.array([f.t_flight_ms for f in fixes])
    dist_km  = np.array([f.distance_km for f in fixes])

    return dict(
        earth=earth, cfg=cfg, paths=paths, fixes=fixes,
        errors=errors, pi6_raw=pi6_raw, pi6_norm=pi6_norm,
        t_ms=t_ms, dist_km=dist_km,
        c_mult=c_mult, noise=noise,
    )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Quantum geo-location  ·  Earth as boson blob  ·  signals at 2c")
    print("=" * 80)
    print(f"  α_sym   = 1/(π − e)       = {ALPHA_SYM:.6f}")
    print(f"  NORM    = 1 − e^(−α(π+e)) = {NORM:.10f}")
    print(f"  1−NORM  (tunnel prob.)    = {TUNNEL_PROB:.6e}")
    print(f"  A_conf  = √NORM           = {CONFINE_AMP:.10f}")
    print(f"  A_tun   = √(1−NORM)       = {TUNNEL_AMP:.6e}")
    print(f"  c_mult                    = {SIGNAL_MULT}c  (twice light speed)")
    print(f"  R_earth                   = {R_EARTH_KM:.0f} km")
    print()

    t0 = time.perf_counter()
    res = run_geo_location(seed=42, c_mult=SIGNAL_MULT, noise=0.002, N=200)
    dt = (time.perf_counter() - t0) * 1e3

    earth = res["earth"]
    cfg = res["cfg"]
    fixes = res["fixes"]

    print(f"--- setup  ·  {dt:.1f} ms ---")
    print(f"  Earth boson mass M_bh     = {earth.M_bh:.6f}")
    print(f"  grid                      = {earth.N}×{earth.N}")
    print(f"  vertices                  = 6")
    print(f"  directed paths            = {len(res['paths'])}")
    print(f"  reference |Π_6|           = {abs(cfg.Pi6_ref):.6e}")
    print(f"  reference arg Π_6         = {cfg.phase_ref:+.6f} rad")
    print()

    # ---- transit times ----
    print("--- transit times at 2c ---")
    print(f"  min d (km)   = {res['dist_km'].min():.1f}")
    print(f"  max d (km)   = {res['dist_km'].max():.1f}")
    print(f"  min t (ms)   = {res['t_ms'].min():.6f}")
    print(f"  max t (ms)   = {res['t_ms'].max():.6f}")
    print(f"  c_light      = {C_LIGHT:.3e} m/s")
    print(f"  signal speed = {SIGNAL_MULT*C_LIGHT:.3e} m/s")
    print()

    # ---- contraction comparison: raw vs NORM ----
    raw_abs  = np.abs(res["pi6_raw"])
    norm_abs = np.abs(res["pi6_norm"])
    print("--- 6D contraction: raw vs NORM reconstruction ---")
    print(f"  {'src':>3s}  {'dst':>3s}  {'d(km)':>8s}  "
          f"{'t(ms)':>10s}  {'|Π_raw|':>10s}  {'|Π_NORM|':>10s}  "
          f"{'err(km)':>10s}")
    print("  " + "-" * 76)
    for f in fixes[:12]:
        print(f"  {f.src_idx:>3d}  {f.dst_idx:>3d}  "
              f"{f.distance_km:>8.1f}  {f.t_flight_ms:>10.6f}  "
              f"{abs(f.Pi6_raw):>10.4e}  {abs(f.Pi6_norm):>10.4e}  "
              f"{f.error_km:>10.3f}")
    print(f"  ...  ({len(fixes) - 12} more paths)")
    print()

    # ---- error statistics ----
    e = res["errors"]
    print("--- localization error statistics ---")
    print(f"  mean   = {e.mean():.3f} km")
    print(f"  median = {np.median(e):.3f} km")
    print(f"  min    = {e.min():.3f} km")
    print(f"  max    = {e.max():.3f} km")
    print(f"  std    = {e.std():.3f} km")
    print()

    # ---- NORM reconstruction effect ----
    ratio = norm_abs / np.maximum(raw_abs, 1e-12)
    print("--- NORM reconstruction effect  |Π_NORM| / |Π_raw| ---")
    print(f"  mean ratio     = {ratio.mean():.6f}")
    print(f"  expected √NORM = {CONFINE_AMP:.6f}")
    print(f"  (the reconstruction is dominated by the confined channel;")
    print(f"   the tunneling channel contributes at the {TUNNEL_AMP:.1e} level)")
    print()

    # ---- complexity ----
    print("--- complexity ---")
    print("  Earth boson blob       O(N²)")
    print("  6-vertex config        O(1)")
    print("  6×5 path build         O(1)  —  30 directed paths")
    print("  6×6 spinor per path    O(36)")
    print("  Levi-Civita rank-6     O(6!) = O(720)")
    print("  6D → 3D reduction      O(9)")
    print("  NORM superposition     O(36)")
    print("  ─────────────────────────────────")
    print(f"  per path               O(720)")
    print(f"  total                  O(720 · 30) = O(21600)")

    # ---- plot ----
    if HAS_MPL:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

        # (a) Earth boson blob + vertices
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(earth.b.T, origin="lower",
                  extent=[earth.x.min(), earth.x.max(),
                          earth.y.min(), earth.y.max()],
                  cmap="inferno")
        ax.scatter(cfg.positions[:, 0], cfg.positions[:, 1],
                   s=80, c="#16a085", edgecolors="k",
                   label="vertices")
        for i, (x, y) in enumerate(cfg.positions):
            ax.annotate(f"V{i}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color="white")
        ax.set_xlabel("x (R_earth)")
        ax.set_ylabel("y (R_earth)")
        ax.set_title("Earth boson blob  b(x, y)")
        ax.legend(fontsize=8)

        # (b) 6-vertex amplitudes and reference contraction
        ax = fig.add_subplot(gs[0, 1])
        ax.bar(np.arange(6), cfg.amplitudes, color="#16a085",
               edgecolor="k", width=0.6)
        ax.set_xlabel("vertex"); ax.set_ylabel("amplitude")
        ax.set_title(f"6-vertex amplitudes\n|Π₆_ref| = "
                     f"{abs(cfg.Pi6_ref):.3e}")
        ax.grid(alpha=0.3)

        # (c) transit times
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(np.arange(len(res["t_ms"])), res["t_ms"],
                "o-", color="#2980b9", lw=1.2, ms=4)
        ax.set_xlabel("path index"); ax.set_ylabel("t (ms)")
        ax.set_title(f"Transit time at {SIGNAL_MULT}c  "
                     f"(min {res['t_ms'].min():.3f}, "
                     f"max {res['t_ms'].max():.3f} ms)")
        ax.grid(alpha=0.3)

        # (d) world lines
        ax = fig.add_subplot(gs[1, 0])
        s_par = np.linspace(-1.5, 1.5, 200)
        for i, (x, y) in enumerate(cfg.positions):
            # straight lines through each vertex (fermion world lines)
            for theta in (0.0, PI / 3.0, 2 * PI / 3.0):
                ax.plot(x + s_par * math.cos(theta),
                        y + s_par * math.sin(theta),
                        color="steelblue", lw=0.5, alpha=0.5)
        ax.imshow(earth.b.T, origin="lower",
                  extent=[earth.x.min(), earth.x.max(),
                          earth.y.min(), earth.y.max()],
                  cmap="inferno", alpha=0.35)
        ax.scatter(cfg.positions[:, 0], cfg.positions[:, 1],
                   s=60, c="#16a085", edgecolors="k", zorder=5)
        ax.set_xlim(earth.x.min(), earth.x.max())
        ax.set_ylim(earth.y.min(), earth.y.max())
        ax.set_aspect("equal")
        ax.set_title("Fermion world lines through the blob\n"
                     "(M_BF = 0, no scattering)")

        # (e) raw vs NORM contraction magnitudes
        ax = fig.add_subplot(gs[1, 1])
        idx = np.arange(len(fixes))
        ax.semilogy(idx, raw_abs, "o-", color="#c0392b",
                    lw=1.0, ms=4, label="|Π_raw|")
        ax.semilogy(idx, norm_abs, "s-", color="#8e44ad",
                    lw=1.0, ms=4, label="|Π_NORM|")
        ax.set_xlabel("path index"); ax.set_ylabel("|Π₆|")
        ax.set_title(f"6D contraction magnitude  ·  "
                     f"NORM ratio = {ratio.mean():.6f}")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

        # (f) localization errors
        ax = fig.add_subplot(gs[1, 2])
        ax.hist(res["errors"], bins=15, color="#2980b9",
                edgecolor="k", alpha=0.85)
        ax.axvline(e.mean(), color="r", ls="--",
                   label=f"mean = {e.mean():.2f} km")
        ax.set_xlabel("localization error (km)")
        ax.set_ylabel("# paths")
        ax.set_title(f"Position error  ·  n = {len(fixes)}")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (g) reconstructed positions vs true
        ax = fig.add_subplot(gs[2, :2])
        ax.imshow(earth.b.T, origin="lower",
                  extent=[earth.x.min(), earth.x.max(),
                          earth.y.min(), earth.y.max()],
                  cmap="inferno", alpha=0.5)
        # plot all reconstructions
        xs_est = np.array([f.position_est_km[0] / R_EARTH_KM for f in fixes])
        ys_est = np.array([f.position_est_km[1] / R_EARTH_KM for f in fixes])
        xs_true = cfg.positions[:, 0]
        ys_true = cfg.positions[:, 1]
        ax.scatter(xs_est, ys_est, s=14, c="#3a7bd5", alpha=0.55,
                   label="reconstructed")
        ax.scatter(xs_true, ys_true, s=100, c="#16a085",
                   edgecolors="k", label="true vertices", zorder=5)
        for i, (x, y) in enumerate(cfg.positions):
            ax.annotate(f"V{i}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color="k")
        ax.set_xlim(earth.x.min(), earth.x.max())
        ax.set_ylim(earth.y.min(), earth.y.max())
        ax.set_aspect("equal")
        ax.set_title(f"Localization on the Earth surface  ·  "
                     f"mean error {e.mean():.2f} km")
        ax.legend(fontsize=9)

        # (h) NORM breakdown
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        ax.text(0.0, 1.0,
                "NORM tunneling reconstruction\n\n"
                f"  NORM        = {NORM:.10f}\n"
                f"  1 − NORM    = {TUNNEL_PROB:.6e}\n"
                f"  A_confine   = {CONFINE_AMP:.10f}\n"
                f"  A_tunnel    = {TUNNEL_AMP:.6e}\n\n"
                f"  Z_eff(θ)    = √NORM · Z(θ)\n"
                f"              + √(1−NORM)·e^(iπ/2)·Z(θ)\n\n"
                f"  Effect: the NORM reconstruction\n"
                f"  amplifies |Π₆| by √NORM ≈ {CONFINE_AMP:.6f}\n"
                f"  and introduces a small\n"
                f"  tunneling shift at the {TUNNEL_AMP:.2e} level.",
                va="top", family="monospace", fontsize=9)

        plt.suptitle(
            "Quantum geo-location  ·  Earth as boson blob  ·  "
            f"signals at {SIGNAL_MULT}c  ·  NORM reconstruction",
            fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    print()
    print("Done.")


if __name__ == "__main__":
    demo()