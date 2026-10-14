#!/usr/bin/env python3
"""
device_locator.py
=================
Locate a device by its signal through the elliptic-Möbius framework.

Signal source (device) — a point particle at Figure 3.5
    Π(t, r) ∈ [0, 1]         elliptic projection weight
    dζ/dt                    spectral rate (discrete + transcendental)
    parity of index          odd  → E-field  (fermionic)
                             even → B-field  (bosonic)
    μ-gated vector           v_n = μ(n)·cos(A_STEP·n²/K)·s_n
    even-even mask           σ(v) ≡ 0 (mod 2)  →  true information

Receiver
    · takes the raw signal
    · assigns E (odd t) and B (even t) slots
    · keeps only even-parity samples whose μ-gate matches
    · reconstructs location from the surviving cluster
"""

from __future__ import annotations
import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA       = 1.0 / (PI - E)              # ≈ 2.362338
ALPHA_ASYM  = 0.3628
A_STEP      = ALPHA / ALPHA_ASYM          # ≈ 6.511
PHI         = (1.0 + math.sqrt(5.0)) / 2.0
W1          = PI
W2          = E
GAMMA       = 0.5772156649015329

K_VEC_DEF   = 128
K_FILTER    = 2 ** 10 + 1


# ============================================================
#  Möbius sieve  ·  O(K log log K)
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
#  Figure 3.5  ·  elliptic projection  Π(t, r)
# ============================================================
def elliptic_projection(t: float, r: float) -> float:
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


def elliptic_projection_vec(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * PI * u) * np.cos(2 * PI * v))


# ============================================================
#  dζ/dt  ·  discrete vs transcendental split
# ============================================================
def discrete_contribution(v: complex) -> float:
    """Discrete algebraic part of a signed value — Re v."""
    return float(v.real)


def transcendental_contribution(v: complex,
                                 alpha: float = ALPHA) -> float:
    """Transcendental part of a signed value — Im v."""
    return float(v.imag)


def dζ_dt(t: float, coeffs: np.ndarray,
          K: int, alpha: float = ALPHA) -> float:
    """
    dζ/dt = -Σ_{j≠0} (j/α) Re(C_j e^{ijt/α})

    Coeffs indexed by j ∈ [-K, K] via coeffs[K + j].
    """
    j = np.arange(-K, K + 1)
    jnz = j[j != 0]
    Cj = coeffs[jnz + K]
    phases = jnz * t / alpha
    return float(-np.sum((jnz / alpha) * np.real(Cj * np.exp(1j * phases))))


# ============================================================
#  Point particle at Figure 3.5
# ============================================================
@dataclass
class PointParticle:
    """
    Device emission sample at (t, r):
        Pi      = elliptic projection weight
        x       = Π(t, r) · Re v(n)        discrete position
        y       = Π(t, r) · Im v(n)        transcendental time
        zt      = dζ/dt                     spectral rate
        vx      = finite-step velocity
        idx     = global sample index (parity)
    """
    idx: int
    t: float
    r: float
    Pi: float
    x: float
    y: float
    zt: float
    vx: float

    @property
    def parity(self) -> int:
        return self.idx % 2

    @property
    def field(self) -> str:
        return "B" if self.parity == 0 else "E"

    def __repr__(self):
        return (f"P[{self.idx:>3d}]  t={self.t:+.3f} r={self.r:+.3f} "
                f"Π={self.Pi:.3f} → {self.field}")


def build_particles(device_t: float, device_r: float,
                    n_samples: int,
                    table_mu: np.ndarray,
                    table_H: np.ndarray,
                    table_F: np.ndarray,
                    noise: float = 0.02,
                    seed: int = 42) -> List[PointParticle]:
    """
    Emit n_samples point particles from a device located at
    (device_t, device_r).

    Each sample:
        t_sample = device_t + small jitter
        r_sample = device_r + small jitter
        Pi       = Π(t_sample, r_sample)
        n        = sample index mapped to a signed index
        x, y     = Π · Re v(n), Π · Im v(n)
        zt       = dζ/dt
        vx       = finite-step derivative of Re v(n)
    """
    rng = np.random.default_rng(seed)
    K = len(table_H) - 1

    particles: List[PointParticle] = []
    for k in range(n_samples):
        # local time/rank with jitter (device drift)
        tk = device_t + rng.normal(0.0, 0.05)
        rk = device_r + rng.normal(0.0, 0.05)

        Pi = elliptic_projection(tk, rk)

        # map k → signed index n ∈ [-K, K]
        n = (k % (2 * K + 1)) - K
        if n == 0:
            n = 1

        a = abs(n)
        real = -(float(table_H[a]) if a < table_H.size
                 else math.log(a) + GAMMA)
        imag = (ALPHA * float(table_F[a]) if n > 0
                and a < table_F.size else 0.0)

        x = Pi * real + rng.normal(0.0, noise)
        y = Pi * imag + rng.normal(0.0, noise)

        # spectral rate — approximate from local signed values
        coeffs = np.zeros(2 * K + 1, dtype=complex)
        for j in range(-K, K + 1):
            nj = abs(j) if j != 0 else 1
            coeffs[K + j] = (-(table_H[nj]) if j < 0
                             else -(table_H[nj]) + 1j * ALPHA * table_F[nj])
        zt = dζ_dt(tk, coeffs, K)

        # finite-step velocity of the discrete part
        a_hi = max(-K, min(K, n + 1))
        a_lo = max(-K, min(K, n - 1))
        vx = (table_H[abs(a_hi)] - table_H[abs(a_lo)]) / 2.0 if a_lo != 0 else 0.0

        particles.append(PointParticle(
            idx=k, t=tk, r=rk, Pi=Pi,
            x=x, y=y, zt=zt, vx=vx,
        ))
    return particles


# ============================================================
#  Receiver  ·  parity E/B split + even-even info mask
# ============================================================
def particle_vector(p: PointParticle, K: int,
                    mu: np.ndarray) -> np.ndarray:
    """μ-gated vector for a particle, using its Π(t, r) phase."""
    n = np.arange(K)
    mu_arr = np.array([float(mu[k]) if k < len(mu) else 0.0
                       for k in range(K)])
    chirp = np.cos(A_STEP * n * n / K)
    theta = math.atan2(p.y, p.x) if (p.x or p.y) else 0.0
    return mu_arr * chirp * p.Pi * complex(
        math.cos(theta), math.sin(theta))


def cross_coupling(va: np.ndarray, vb: np.ndarray) -> float:
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.vdot(va, vb)) / (na * nb))


def ugc_label(phase: float, k: int) -> int:
    return int(round(k * phase / (2 * PI))) % k


def receive(particles: List[PointParticle],
            K: int, mu: np.ndarray,
            k_labels: int = 4,
            threshold: float = 0.3
            ) -> Dict:
    """
    Receiver pipeline:

        1. split incoming samples by parity
               odd  → E-field (fermionic)
               even → B-field (bosonic)
        2. compute μ-gated vectors per particle
        3. even-even mask:  σ(v) ≡ 0 (mod 2)
        4. cross-coupling gate between consecutive survivors
        5. keep only edges above threshold
    """
    N = len(particles)
    vecs = [particle_vector(p, K, mu) for p in particles]

    # parity split
    odd_idx  = [i for i, p in enumerate(particles) if p.parity == 1]
    even_idx = [i for i, p in enumerate(particles) if p.parity == 0]

    # UGC labels
    labels = np.array([
        ugc_label(math.atan2(vecs[i].imag.sum(), vecs[i].real.sum()),
                  k_labels)
        for i in range(N)
    ])
    even_even_mask = (labels % 2) == 0

    # gate: consecutive even-even pairs whose coupling ≥ τ
    gate_survivors: List[int] = []
    for i in range(N - 1):
        if not (even_even_mask[i] and even_even_mask[i + 1]):
            continue
        g = cross_coupling(vecs[i], vecs[i + 1])
        if g >= threshold:
            gate_survivors.append(i)

    # E-field and B-field signals
    E_signal = np.zeros(N)
    B_signal = np.zeros(N)
    for i in odd_idx:
        E_signal[i] = particles[i].x * particles[i].Pi
    for i in even_idx:
        B_signal[i] = particles[i].y * particles[i].Pi

    # spectral rate signal (transcendental clock)
    zt_signal = np.array([p.zt for p in particles])

    return dict(
        particles=particles,
        odd_idx=odd_idx, even_idx=even_idx,
        labels=labels, even_even_mask=even_even_mask,
        gate_survivors=gate_survivors,
        E_signal=E_signal, B_signal=B_signal,
        zt_signal=zt_signal,
    )


# ============================================================
#  Location estimate from surviving samples
# ============================================================
def estimate_device_location(result: Dict) -> Dict:
    """
    Recover device location (t_device, r_device) from the
    surviving even-parity samples:

        t_hat = weighted mean of t over gate survivors
        r_hat = weighted mean of r over gate survivors
        Pi_hat = weighted mean of Π over gate survivors
    """
    particles = result["particles"]
    surv = result["gate_survivors"]
    if not surv:
        return dict(t_hat=0.0, r_hat=0.0, Pi_hat=0.0,
                    sigma_t=0.0, sigma_r=0.0, n_survivors=0)

    idxs = np.array(surv)
    ts = np.array([particles[i].t for i in surv])
    rs = np.array([particles[i].r for i in surv])
    ps = np.array([particles[i].Pi for i in surv])
    w = ps / (ps.sum() + 1e-12)

    t_hat = float(np.sum(w * ts))
    r_hat = float(np.sum(w * rs))
    Pi_hat = float(np.mean(ps))
    return dict(
        t_hat=t_hat, r_hat=r_hat, Pi_hat=Pi_hat,
        sigma_t=float(np.std(ts)), sigma_r=float(np.std(rs)),
        n_survivors=len(surv),
    )


# ============================================================
#  Full pipeline  ·  multi-device scenario
# ============================================================
def pipeline(devices: List[Tuple[float, float, str]],
             n_samples: int = 96,
             K_vec: int = K_VEC_DEF,
             k_labels: int = 4,
             threshold: float = 0.3,
             noise: float = 0.02,
             verbose: bool = True) -> Dict:
    if verbose:
        print("=" * 78)
        print("Device locator  ·  Figure 3.5 + parity split + even-even mask")
        print("=" * 78)
        print(f"  α       = 1/(π−e)      = {ALPHA:.6f}")
        print(f"  A_STEP  = α / 0.3628   = {A_STEP:.6f}")
        print(f"  K_vec   = {K_vec}")
        print(f"  n_samples (per device) = {n_samples}")
        print()

    # ---------- Möbius table ----------
    K_mu = 256
    F_tab, mu_tab, H_tab = mu_convolution_H(K_mu)

    all_particles: List[PointParticle] = []
    per_device: List[Dict] = []

    for (dev_t, dev_r, name) in devices:
        parts = build_particles(dev_t, dev_r, n_samples,
                                mu_tab, H_tab, F_tab,
                                noise=noise)
        # offset sample indices so parity is unique per device
        off = len(all_particles)
        for p in parts:
            p.idx += off
        all_particles.extend(parts)
        if verbose:
            print(f"  device {name:>8s}  at (t={dev_t:+.3f}, "
                  f"r={dev_r:+.3f})   {len(parts)} samples")

    if verbose:
        print()

    # ---------- receive ----------
    mu = mobius_sieve(K_vec)
    result = receive(all_particles, K_vec, mu,
                     k_labels=k_labels, threshold=threshold)

    # ---------- locate each device by clustering survivors ----------
    surv = result["gate_survivors"]
    if verbose:
        print(f"  total particles        = {len(all_particles)}")
        print(f"  even-parity survivors  = "
              f"{int(np.sum(result['even_even_mask']))}")
        print(f"  gate survivors (edges) = {len(surv)}")
        print()

    # cluster survivors by device (nearest device in (t, r))
    estimates = []
    for (dev_t, dev_r, name) in devices:
        idxs = [i for i in surv
                if abs(all_particles[i].t - dev_t) < 0.2
                and abs(all_particles[i].r - dev_r) < 0.2]
        if idxs:
            ts = np.array([all_particles[i].t for i in idxs])
            rs = np.array([all_particles[i].r for i in idxs])
            ps = np.array([all_particles[i].Pi for i in idxs])
            w = ps / (ps.sum() + 1e-12)
            est = dict(
                name=name,
                t_true=dev_t, r_true=dev_r,
                t_hat=float(np.sum(w * ts)),
                r_hat=float(np.sum(w * rs)),
                n=len(idxs),
                sigma_t=float(np.std(ts)),
                sigma_r=float(np.std(rs)),
            )
        else:
            est = dict(name=name, t_true=dev_t, r_true=dev_r,
                       t_hat=float("nan"), r_hat=float("nan"),
                       n=0, sigma_t=0.0, sigma_r=0.0)
        estimates.append(est)
        if verbose:
            dt = abs(est["t_hat"] - dev_t) if not math.isnan(est["t_hat"]) else float("nan")
            dr = abs(est["r_hat"] - dev_r) if not math.isnan(est["r_hat"]) else float("nan")
            print(f"  {name:>8s}  →  "
                  f"t̂={est['t_hat']:+.3f}  r̂={est['r_hat']:+.3f}  "
                  f"|Δt|={dt:.4f}  |Δr|={dr:.4f}  "
                  f"n_surv={est['n']:>2d}")

    return dict(
        particles=all_particles,
        result=result,
        estimates=estimates,
        devices=devices,
        K_vec=K_vec, k_labels=k_labels,
    )


# ============================================================
#  Visualization
# ============================================================
def plot_result(data: Dict):
    particles = data["particles"]
    result = data["result"]
    estimates = data["estimates"]
    K_vec = data["K_vec"]

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    # (a) Figure 3.5  ·  elliptic projection surface
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, W1, 200)
    ys = np.linspace(0, W2, 200)
    X, Y = np.meshgrid(xs, ys)
    P = elliptic_projection_vec(X, Y)
    im = ax.imshow(P, extent=[0, W1, 0, W2], origin="lower",
                   cmap="viridis", aspect="auto", vmin=0, vmax=1)
    for (dev_t, dev_r, name) in data["devices"]:
        ax.scatter([dev_t % W1], [dev_r % W2], s=120,
                   c="crimson", edgecolors="k", zorder=5)
        ax.annotate(name, (dev_t % W1, dev_r % W2),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color="crimson")
    ax.set_xlabel("t (mod π)"); ax.set_ylabel("r (mod e)")
    ax.set_title("Figure 3.5  ·  Π(t, r)   +  device positions")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) particle cloud  ·  discrete (x) vs transcendental (y)
    ax = fig.add_subplot(gs[0, 1])
    xs_p = np.array([p.x for p in particles])
    ys_p = np.array([p.y for p in particles])
    ps_p = np.array([p.Pi for p in particles])
    sc = ax.scatter(xs_p, ys_p, c=ps_p, cmap="plasma",
                    s=18, edgecolors="k", linewidth=0.2)
    ax.set_xlabel("x = Π · Re v(n)   discrete")
    ax.set_ylabel("y = Π · Im v(n)   transcendental")
    ax.set_title("Particle cloud  ·  Π-weighted")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Π")
    ax.grid(alpha=0.3)

    # (c) parity split  ·  E vs B
    ax = fig.add_subplot(gs[0, 2])
    E_sig = result["E_signal"]
    B_sig = result["B_signal"]
    idxs = np.arange(len(particles))
    ax.plot(idxs, E_sig, color="#e74c3c", lw=0.8, label="E (odd t)")
    ax.plot(idxs, B_sig, color="#3a7bd5", lw=0.8, label="B (even t)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("sample index"); ax.set_ylabel("field")
    ax.set_title("Parity split  ·  odd → E, even → B")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) UGC labels  ·  even-even mask highlighted
    ax = fig.add_subplot(gs[1, 0])
    labels = result["labels"]
    colors = ["#16a085" if m else "#e74c3c"
              for m in result["even_even_mask"]]
    ax.bar(idxs, labels, color=colors, width=1.0)
    ax.set_xlabel("sample index"); ax.set_ylabel("σ(v)")
    ax.set_title(f"UGC labels  ·  even-even mask "
                 f"({int(np.sum(result['even_even_mask']))} kept)")
    ax.set_yticks(range(0, data["k_labels"]))
    ax.grid(alpha=0.3, axis="y")

    # (e) spectral rate  ·  discrete vs transcendental
    ax = fig.add_subplot(gs[1, 1])
    zt = result["zt_signal"]
    ax.plot(idxs, zt, color="#8e44ad", lw=1.0)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("sample index"); ax.set_ylabel("dζ/dt")
    ax.set_title("Spectral rate  ·  transcendental clock")
    ax.grid(alpha=0.3)

    # (f) gate survivors
    ax = fig.add_subplot(gs[1, 2])
    surv = set(result["gate_survivors"])
    gate = np.array([1 if i in surv else 0 for i in range(len(particles) - 1)])
    ax.bar(np.arange(len(gate)), gate,
           color=["#16a085" if g else "#bdc3c7" for g in gate],
           width=1.0)
    ax.set_xlabel("edge index"); ax.set_ylabel("survived?")
    ax.set_title(f"Cross-coupling gate  ·  {len(surv)} survivors")
    ax.set_yticks([]); ax.grid(alpha=0.3, axis="x")

    # (g) located device positions  ·  true vs estimated
    ax = fig.add_subplot(gs[2, :2])
    for (dev_t, dev_r, name) in data["devices"]:
        ax.scatter([dev_t], [dev_r], s=200, marker="*",
                   c="crimson", edgecolors="k", zorder=5,
                   label=f"{name}  true" if name == data["devices"][0][2] else None)
    for est in estimates:
        if not math.isnan(est["t_hat"]):
            ax.scatter([est["t_hat"]], [est["r_hat"]],
                       s=90, marker="o", c="#3a7bd5",
                       edgecolors="k", zorder=6,
                       label="estimated" if est == estimates[0] else None)
            ax.plot([est["t_true"], est["t_hat"]],
                    [est["r_true"], est["r_hat"]],
                    "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("t"); ax.set_ylabel("r")
    ax.set_title("Device locations  ·  true (★) vs recovered (●)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (h) summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    lines = ["Locator summary\n"]
    for est in estimates:
        lines.append(
            f"  {est['name']:>8s}\n"
            f"    true  t={est['t_true']:+.3f}  r={est['r_true']:+.3f}\n"
            f"    est   t={est['t_hat']:+.3f}  r={est['r_hat']:+.3f}\n"
            f"    n_surv = {est['n']}"
        )
    ax.text(0.0, 1.0, "\n".join(lines),
            va="top", family="monospace", fontsize=9)

    plt.suptitle(
        "Device locator  ·  Figure 3.5 point particles  ·  "
        "odd→E / even→B parity  ·  even-even info mask  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    devices = [
        (0.40, 0.60, "alice"),
        (1.50, 1.20, "bob"),
        (2.60, 1.80, "carol"),
    ]
    data = pipeline(devices,
                    n_samples=96,
                    K_vec=K_VEC_DEF,
                    k_labels=4,
                    threshold=0.3,
                    noise=0.02,
                    verbose=True)
    plot_result(data)


if __name__ == "__main__":
    demo()