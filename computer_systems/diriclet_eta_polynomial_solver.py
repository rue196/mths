#!/usr/bin/env python3
"""
diriclet_eta_polynomial_solver.py  ·  chirp trajectory edition
===============================================================

Spectral sum  ζ(t) = Σ C_i exp(i t i / α)  with α = 1/(π − e).

Additions
---------
• DC chirp trajectory      (DC_chirp.py)   — entangled, deterministic
• Dirichlet eta function   η(a) = Σ_{n≥1} (−1)^{n−1} / n^a
• Asymmetric polynomial solver on the chirp trajectory, using the
  eta weight  η(0.3628) ≈ 0.659538886352  as the target norm.

Chirp trajectory  (from DC_chirp.py, unchanged)
-----------------------------------------------
    DC path      x(s)        = exp(A_STEP · s / N)
    universal    w(s)        = cos(A_STEP · s² / N)
    entangled    φ(s)        = arg ζ(s/N · period)
    chirp        chirp_ent(s)= w(s) · cos φ(s)
    coupling     g_ij        = w_ij · κ_ij
    even-even    M_ij        = [i even ∧ j even]
    propagation  s_{t+1}     = G_even · s_t / row_sum

The entangled chirp replaces the 1D TSP route.  The polynomial is
evaluated at the chirp trajectory positions, multiplied by η and by
chirp_ent, then root-bracketed.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---- DC chirp trajectory  (unchanged parent) ----
from DC_chirp import (
    DCMobiusGate,
    mobius_sieve,
    coupling_matrix_1d,
    even_even_mask,
    propagate_signal,
    A_STEP,
    ALPHA,
    PI,
    E,
)


# ============================================================
#  Constants
# ============================================================
EXP         = math.exp(0.3628)               # e^0.3628 ≈ 1.4371
A           = 1.0 / (PI - EXP)               # ≈ 0.5866
ALPHA_ASYM  = 0.3628
ETA_TARGET  = 0.659538886352                 # η(0.3628)


# ============================================================
#  1.  Spectral sum  ζ(t)  ·  unchanged
# ============================================================
def zeta(t, coeffs, alpha=ALPHA):
    total = 0.0 + 0.0j
    K = len(coeffs) - 1
    for i in range(1, K + 1):
        if coeffs[i] != 0:
            total += coeffs[i] * np.exp(1j * t * i / alpha)
    return total


def integrate_abs_zeta(coeffs, N_points=1000, alpha=ALPHA):
    period = 2 * PI * alpha
    t_vals = np.linspace(0, period, N_points)
    s = 0.0
    for t in t_vals:
        s += abs(zeta(t, coeffs, alpha))
    dt = t_vals[1] - t_vals[0]
    return s * dt


# ============================================================
#  2.  DC chirp trajectory  ·  from DC_chirp.py
# ============================================================
def dc_chirp_trajectory(N_path: int = 80,
                        K_gate: int = 20,
                        propagate_steps: int = 40,
                        propagate_decay: float = 0.02) -> dict:
    """
    Full chirp trajectory from DC_chirp.py primitives.

    Returns
    -------
    dict with keys:
        s         : np.arange(N_path)                   [0, N)
        x_path    : exp(A_STEP · s/N)                   DC path
        w_chirp   : cos(A_STEP · s²/N)                  universal chirp
        phis      : arg ζ(s/N · period)                 entangled phase
        chirp_ent : w_chirp · cos φ                     entangled chirp
        G         : coupling_matrix_1d(x_path)
        mask      : even_even_mask(N_path)
        G_even    : even-even masked G
        traj      : propagate_signal(G_even, s0)        time evolution
        gate      : DCMobiusGate
        period    : gate.period
    """
    gate = DCMobiusGate(K_gate, smooth=True)
    period = gate.period

    s = np.arange(N_path, dtype=float)
    x_path = np.exp(A_STEP * s / N_path)
    w_chirp = np.cos(A_STEP * s * s / N_path)

    phis = np.array([np.angle(gate.zeta(si / N_path * period))
                     for si in s])
    chirp_ent = w_chirp * np.cos(phis)

    G = coupling_matrix_1d(x_path)
    mask = even_even_mask(N_path)
    G_even = np.where(mask, G, 0.0)
    np.fill_diagonal(G_even, 0.0)

    s0 = np.zeros(N_path)
    if N_path > 0:
        s0[0] = chirp_ent[0] if abs(chirp_ent[0]) > 1e-9 else 1.0
    traj = propagate_signal(G_even, s0,
                            steps=propagate_steps,
                            decay=propagate_decay)

    return dict(
        s=s, x_path=x_path, w_chirp=w_chirp,
        phis=phis, chirp_ent=chirp_ent,
        G=G, mask=mask, G_even=G_even,
        traj=traj, gate=gate, period=period,
    )


# ============================================================
#  3.  Dirichlet eta function  η(a)
# ============================================================
def dirichlet_eta(a=ALPHA_ASYM, N=2000):
    s = 0.0
    for n in range(1, N + 1):
        s += ((-1) ** (n - 1)) / (n ** a)
    return s


def _zeta_scalar(a, N=20000):
    """ζ(a) = Σ_{n≥1} n^{-a}  (real, a > 1)."""
    s = 0.0
    for n in range(1, N + 1):
        s += 1.0 / (n ** a)
    return s


# ============================================================
#  4.  Asymmetric polynomial
# ============================================================
def asymmetric_poly(coeffs, x):
    return sum(c * (x ** k) for k, c in enumerate(coeffs))


# ============================================================
#  5.  Solver on the chirp trajectory
# ============================================================
def solve_asym_poly_on_chirp(coeffs, chirp_data,
                             eta_weight=None,
                             x_mode: str = "norm"):
    """
    Solve an asymmetric polynomial P(x) along the DC chirp trajectory.

    Parameters
    ----------
    coeffs      : polynomial coefficients  P(x) = Σ c_k x^k
    chirp_data  : dict from dc_chirp_trajectory()
    eta_weight  : Dirichlet eta weight (default η(0.3628) ≈ 0.6595)
    x_mode      : polynomial domain choice
        "path"  →  x = exp(A_STEP · s/N)       raw DC path
        "norm"  →  x = x_path / max(x_path)    normalized to (0, 1]
        "param" →  x = s / (N−1)                chirp parameter [0, 1]

    Steps
    -----
    1. Take x_r = domain values from chirp trajectory
    2. Evaluate P(x) at each position
    3. Multiply by η weight and entangled chirp
    4. Root-bracket the signed stream along the chirp path
    5. Compute supertrace S_asym over the signed stream
    """
    s         = chirp_data["s"]
    x_path    = chirp_data["x_path"]
    chirp_ent = chirp_data["chirp_ent"]
    N         = len(s)

    if x_mode == "path":
        x_r = x_path
    elif x_mode == "norm":
        x_r = x_path / max(x_path.max(), 1e-12)
    elif x_mode == "param":
        x_r = s / max(N - 1, 1)
    else:
        raise ValueError(f"unknown x_mode: {x_mode}")

    y_r = np.array([asymmetric_poly(coeffs, float(xi)) for xi in x_r])

    if eta_weight is None:
        eta_weight = dirichlet_eta(ALPHA_ASYM)

    # signed stream: η · chirp_ent · P(x)
    signed = eta_weight * chirp_ent * y_r

    # ---- root bracketing ----
    sign = np.sign(signed)
    sign[sign == 0] = 1.0
    crossings = np.where(np.diff(sign) != 0)[0]
    roots = []
    for c in crossings:
        x0, x1 = x_r[c], x_r[c + 1]
        y0, y1 = signed[c], signed[c + 1]
        if y1 - y0 != 0.0:
            roots.append(x0 - y0 * (x1 - x0) / (y1 - y0))
        else:
            roots.append(0.5 * (x0 + x1))

    # ---- supertrace over signed stream ----
    S_asym = float(np.sum(np.where(np.arange(N) % 2 == 0,
                                    signed, -signed)))

    return dict(
        s=s, x_r=x_r, y_r=y_r, signed=signed,
        chirp_ent=chirp_ent, w_chirp=chirp_data["w_chirp"],
        phis=chirp_data["phis"],
        roots=np.array(roots), eta_weight=eta_weight,
        S_asym=S_asym, x_mode=x_mode,
    )


# ============================================================
#  MAIN  ·  spectral sum + asymmetric chirp polynomial
# ============================================================
def main():
    # ---------------------------------------------------------
    #  1. Spectral sum  ·  scaled to η target
    # ---------------------------------------------------------
    K = 100
    mu = mobius_sieve(K)
    coeffs = [int(mu[i]) if i > 0 else 0 for i in range(K + 1)]

    I_abs = integrate_abs_zeta(coeffs, N_points=2000)
    scaling = ETA_TARGET / I_abs
    scaled_coeffs = [c * scaling for c in coeffs]

    period = 2 * PI * ALPHA
    t_vals = np.linspace(0, period, 500)
    z_vals = np.array([zeta(t, scaled_coeffs, ALPHA) for t in t_vals])
    real_vals = np.real(z_vals)
    imag_vals = np.imag(z_vals)
    abs_vals  = np.abs(z_vals)
    avg_abs   = np.mean(abs_vals)

    print("=" * 72)
    print("Dirichlet eta polynomial solver  ·  chirp trajectory edition")
    print("=" * 72)
    print(f"  A (solver α)        = 1/(π − e^0.3628)  = {A:.8f}")
    print(f"  α (spectral)        = 1/(π − e)         = {ALPHA:.8f}")
    print(f"  A_STEP              = α / 0.3628       = {A_STEP:.8f}")
    print(f"  η target            = {ETA_TARGET:.12f}")
    print(f"  avg |ζ(t)|          = {avg_abs:.8f}")
    print(f"  integral over period= {avg_abs * period:.8f}")
    print()

    # ---------------------------------------------------------
    #  2. Dirichlet eta verification
    # ---------------------------------------------------------
    eta_num    = dirichlet_eta(ALPHA_ASYM, N=20000)
    eta_closed = (1.0 - 2.0 ** (1.0 - ALPHA_ASYM)) * \
                 _zeta_scalar(ALPHA_ASYM)
    print("--- Dirichlet eta  η(a) = Σ (−1)^(n−1) / n^a ---")
    print(f"  η(0.3628) numerical = {eta_num:.12f}")
    print(f"  η(0.3628) closed    = {eta_closed:.12f}")
    print(f"  difference          = {abs(eta_num - eta_closed):.3e}")
    print()

    # ---------------------------------------------------------
    #  3. DC chirp trajectory
    # ---------------------------------------------------------
    N_path = 80
    print(f"--- DC chirp trajectory  (N_path = {N_path}) ---")
    chirp = dc_chirp_trajectory(
        N_path=N_path, K_gate=20,
        propagate_steps=40, propagate_decay=0.02)
    gate = chirp["gate"]
    dc_tp = gate.dc_throughput()
    S_dc, H_dc, m_dc = gate.supertrace()

    print(f"  gate K              = {gate.K}")
    print(f"  period              = {chirp['period']:.6f}")
    print(f"  DC throughput       = {dc_tp:.6f}")
    print(f"  supertrace S_DC     = {S_dc:+.6f}")
    print(f"  entropy H_DC        = {H_dc:.6f}")
    print(f"  mass m_DC           = {m_dc:.6e}")
    print()
    print(f"  x_path range        = "
          f"[{chirp['x_path'].min():.4f}, {chirp['x_path'].max():.4f}]")
    print(f"  w_chirp range       = "
          f"[{chirp['w_chirp'].min():+.4f}, "
          f"{chirp['w_chirp'].max():+.4f}]")
    print(f"  chirp_ent range     = "
          f"[{chirp['chirp_ent'].min():+.4f}, "
          f"{chirp['chirp_ent'].max():+.4f}]")
    print(f"  ⟨chirp_ent⟩         = {chirp['chirp_ent'].mean():+.6f}")
    print(f"  G kept (even-even)  = "
          f"{int(np.count_nonzero(chirp['G_even']))} / {N_path ** 2}")
    print(f"  trajectory shape    = {chirp['traj'].shape}")
    print()

    # ---------------------------------------------------------
    #  4. Asymmetric polynomial on the chirp trajectory
    # ---------------------------------------------------------
    print("--- Asymmetric polynomial on the chirp trajectory ---")
    rng = np.random.default_rng(2024)
    degree = 12
    poly_coeffs = rng.standard_normal(degree + 1) * \
                  (0.8 ** np.arange(degree + 1))
    poly_coeffs[0] = -0.4
    print(f"  polynomial degree   = {degree}")

    for mode in ("param", "norm", "path"):
        result = solve_asym_poly_on_chirp(
            poly_coeffs, chirp,
            eta_weight=ETA_TARGET, x_mode=mode)
        print(f"  x_mode = {mode:6s}  "
              f"x range = [{result['x_r'].min():+.4f}, "
              f"{result['x_r'].max():+.4f}]  "
              f"roots = {len(result['roots']):>2d}  "
              f"S_asym = {result['S_asym']:+.4f}")
        for r in result["roots"][:4]:
            print(f"    x ≈ {r:+.6f}")
    print()

    # ---------------------------------------------------------
    #  5. Visualization  ·  uses normalized chirp domain
    # ---------------------------------------------------------
    result = solve_asym_poly_on_chirp(
        poly_coeffs, chirp,
        eta_weight=ETA_TARGET, x_mode="norm")

    fig, axes = plt.subplots(5, 1, figsize=(11, 16))

    # (1) Re ζ(t)
    axes[0].plot(t_vals, real_vals, "b-", label="Re ζ(t)")
    axes[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Real")
    axes[0].legend(); axes[0].grid(True)

    # (2) Im ζ(t)
    axes[1].plot(t_vals, imag_vals, "r-", label="Im ζ(t)")
    axes[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Imag")
    axes[1].legend(); axes[1].grid(True)

    # (3) |ζ(t)| + η target
    axes[2].plot(t_vals, abs_vals, "g-", label="|ζ(t)|")
    axes[2].axhline(y=avg_abs, color="m", linestyle="--",
                    label=f"Average = {avg_abs:.6f}")
    axes[2].axhline(y=ETA_TARGET, color="orange", linestyle=":",
                    label=f"η(a) = {ETA_TARGET:.6f}")
    axes[2].set_xlabel("t"); axes[2].set_ylabel("|ζ|")
    axes[2].legend(); axes[2].grid(True)

    # (4) chirp trajectory components
    s = chirp["s"]
    axes[3].plot(s, chirp["x_path"] / chirp["x_path"].max(),
                 color="#16a085", lw=1.2,
                 label="x_path / max  (normalized DC path)")
    axes[3].plot(s, chirp["w_chirp"], color="#3498db", lw=1.0,
                 ls="--", alpha=0.7,
                 label="w(s) = cos(A_STEP · s² / N)")
    axes[3].plot(s, chirp["chirp_ent"], color="#e74c3c", lw=1.3,
                 label="chirp_ent(s) = w(s) · cos φ(s)")
    axes[3].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[3].set_xlabel("path position  s")
    axes[3].set_ylabel("amplitude")
    axes[3].legend(fontsize=9); axes[3].grid(True)

    # (5) polynomial on the chirp trajectory
    axes[4].plot(result["x_r"], result["y_r"],
                 color="#34495e", lw=1.0, alpha=0.55,
                 label="P(x) along chirp trajectory")
    axes[4].plot(result["x_r"], result["signed"],
                 color="#c0392b", lw=1.5,
                 label=f"η · chirp_ent · P(x)   (η = {ETA_TARGET:.6f})")
    # overlay scaled chirp
    ymax = max(abs(result["signed"]).max(), 1e-9)
    axes[4].plot(result["x_r"], result["chirp_ent"] * ymax,
                 color="#16a085", lw=1.0, ls="--", alpha=0.7,
                 label="chirp_ent (scaled)")
    axes[4].axhline(0, color="k", linestyle="--", alpha=0.5)
    if len(result["roots"]) > 0:
        axes[4].scatter(
            result["roots"], [0.0] * len(result["roots"]),
            color="#f1c40f", edgecolor="k", s=90, zorder=5,
            label=f"roots ({len(result['roots'])})")
    axes[4].set_xlabel("x  (chirp trajectory, normalized)")
    axes[4].set_ylabel("amplitude")
    axes[4].legend(fontsize=9)
    axes[4].grid(True)

    plt.suptitle(
        f"Dirichlet eta polynomial solver  ·  chirp trajectory\n"
        f"α = {ALPHA:.4f}   A_STEP = {A_STEP:.4f}   "
        f"η(0.3628) = {ETA_TARGET:.6f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # ---------------------------------------------------------
    #  6. Summary
    # ---------------------------------------------------------
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  Spectral sum       ζ(t) = Σ C_i exp(i·t·i/α)")
    print(f"    A (solver)            = {A:.8f}")
    print(f"    α (spectral)          = {ALPHA:.8f}")
    print()
    print("  Dirichlet eta      η(a) = Σ (−1)^(n−1) / n^a")
    print(f"    η(0.3628)             = {eta_num:.12f}")
    print()
    print("  DC chirp trajectory  ·  from DC_chirp.py")
    print("    x(s)      = exp(A_STEP · s/N)")
    print("    w(s)      = cos(A_STEP · s²/N)")
    print("    φ(s)      = arg ζ(s/N · period)")
    print("    chirp_ent = w(s) · cos φ(s)")
    print("    g_ij      = w_ij · κ_ij")
    print("    even-even = [i even ∧ j even]")
    print("    propagate = s_{t+1} = G_even · s_t / row_sum")
    print()
    print("  Solver uses the entangled chirp as the routing primitive")
    print("  (replaces the 1D TSP route).  The η weight anchors the")
    print("  target norm.  Polynomial is evaluated at x_r = domain(s)")
    print("  for s = 0, 1, …, N−1, then modulated by η·chirp_ent.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()