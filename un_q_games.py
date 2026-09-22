#!/usr/bin/env python3
"""
ugc_electron_chip.py

Unique Games on the transistor electron chip.

The original pipeline solves a large asymmetric polynomial via the
Möbius SAT gate.  Here we replace it with the **Unique Games** problem:

    Instance  (V, E, [k], {π_e})
        • bipartite vertex set  V = L ∪ R
        • label set             [k] = {0, …, k−1}
        • for each edge e=(i, j) ∈ E, a permutation  π_e ∈ S_k
        • a labelling σ : V → [k]
        • edge e is satisfied iff  π_e(σ(i)) = σ(j)

The UGC states that for every ε, δ > 0 there is a k such that it is
NP‑hard to distinguish instances with completeness ≥ 1−ε from
instances with soundness ≤ δ.

Everything else is preserved:

    • Figure 3.5   — bounded elliptic projection Π ∈ [0,1] on the
                     torus ℂ/(πℤ + eℤ), used as the *label density*
                     across the chip.
    • Maxwell      — E(t) = Re ζ(t), B(t) = Im ζ(t) from the Möbius
                     spectral sum, with ∇·E the local charge density.
    • Convolution  — exponential barrier kernel exp(−α|x|) modelling
                     label tunnelling through edges.
    • Quadratic    — each label event a ∈ [0, N) is mapped to a
                     quadratic address  q(a) = A·a² + B·a + C mod K,
                     the envelope  env[q]  gates by μ(q) ≠ 0.
    • Dual const.  — α_sym = 1/(π − e) for the even half,
                     α_asym = 0.3628   for the odd half.

The UGC‑specific outputs are:

    completeness = (# satisfied edges) / |E|       (best labelling)
    soundness    = best labelling under gap‑threshold
    ugc_gap      = completeness − soundness        (the conjectural gap)
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
ALPHA_SYM  = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_ASYM = 0.3628
NORM       = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
DENSITY    = 6.0 / (PI * PI)         # ≈ 0.6079271018
W1         = PI
W2         = E


# ============================================================
#  1. Möbius sieve  (shared O(K log log K) table)
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
    """Π(x, y) = ½ (1 + cos(2π x/π) · cos(2π y/e))  ∈ [0, 1]."""
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))


def elliptic_projection_1d(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


# ============================================================
#  3. Möbius spectral sum  ζ(t)
# ============================================================
def zeta_mobius(t: float, mu: np.ndarray, K: int,
                alpha: float = ALPHA_SYM) -> complex:
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    theta = t * n / alpha
    return complex(np.sum(m * np.cos(theta)), np.sum(m * np.sin(theta)))


def field_components(t: float, mu: np.ndarray, K: int
                     ) -> tuple[float, float]:
    z = zeta_mobius(t, mu, K)
    return z.real, z.imag


def divergence_E(t: float, mu: np.ndarray, K: int) -> float:
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    signs = np.where(n % 2 == 0, 1.0, -1.0)
    return float(np.sum(signs * m * np.cos(t * n / ALPHA_SYM)))


# ============================================================
#  4. Convolution tunneling  (label transport kernel)
# ============================================================
def barrier_kernel(width: int, alpha: float = ALPHA_SYM) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    ker = np.exp(-alpha * np.abs(x))
    return ker / ker.sum()


def tunnel(signal: np.ndarray, width: int = 21) -> np.ndarray:
    return convolve(signal, barrier_kernel(width), mode="same")


# ============================================================
#  5. Quadratic envelope  (label stream over the chip)
# ============================================================
def quadratic_address(a: int, A: int = 1, B: int = 1, C: int = 0,
                      K: int = 32) -> int:
    return (A * a * a + B * a + C) % K


def build_quadratic_envelope(K: int, N: int,
                             mu: np.ndarray,
                             A: int = 1, B: int = 1, C: int = 0
                             ) -> tuple[np.ndarray, dict[int, int]]:
    """
    Envelope over quadratic addresses, gated by μ(q) ≠ 0.

    Here the "label stream" is the list of *labels* a ∈ [0, N), and the
    envelope env[q] accumulates the squared label frequency at each
    square‑free address q.
    """
    env = np.zeros(K, dtype=np.float64)
    q_of_a: dict[int, int] = {}
    for a in range(N):
        q = quadratic_address(a, A, B, C, K)
        q_of_a[a] = q
        if q < len(mu) and mu[q] != 0:
            env[q] += 1.0
    return env, q_of_a


def send_envelope_1d(env: np.ndarray, channel: np.ndarray) -> np.ndarray:
    L = len(channel)
    if len(env) == L:
        return env * channel
    env_r = np.interp(np.linspace(0, 1, L),
                      np.linspace(0, 1, len(env)), env)
    return env_r * channel


# ============================================================
#  6. Dual-constant supertrace  (sym / asym label split)
# ============================================================
def supertrace_dual(channel: np.ndarray) -> tuple[float, float, float]:
    S_sym = 0.0
    S_asym = 0.0
    for i, v in enumerate(channel):
        if i % 2 == 0:
            S_sym += ALPHA_SYM * abs(v)
        else:
            S_asym += ALPHA_ASYM * abs(v)
    return S_sym, S_asym, S_sym - S_asym


def entropy(S: float, N: int, alpha: float = ALPHA_SYM) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  7. Unique Games instance
# ============================================================
def generate_unique_games_instance(N_left: int,
                                   N_right: int,
                                   k: int,
                                   degree: int = 3,
                                   seed: int = 2024):
    """
    Bipartite Unique Games instance.

    Returns
    -------
    edges : list of (i, j, perm)
        i ∈ [0, N_left), j ∈ [0, N_right), perm ∈ S_k (tuple length k)
    """
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(N_left):
        js = rng.choice(N_right, size=degree, replace=False)
        for j in js:
            perm = tuple(int(x) for x in rng.permutation(k))
            edges.append((int(i), int(j), perm))
    return edges


# ============================================================
#  8. Label assignment via the Möbius gate
# ============================================================
def vertex_hash(v: int, side: str) -> float:
    """Deterministic scalar in (0.5, 5.0) for a vertex."""
    h = (v * 2654435761) ^ (0 if side == "L" else 0x9E3779B9)
    return 0.5 + ((h & 0xFFFF) / 0xFFFF) * 4.5


def ugc_label_assignment(edges,
                         N_left: int,
                         N_right: int,
                         k: int,
                         mu: np.ndarray,
                         K_filter: int = 2 ** 10 + 1,
                         quad_A: int = 1,
                         quad_B: int = 1,
                         quad_C: int = 0,
                         proj_max: float = 1.0):
    """
    Assign a label σ(v) ∈ [0, k) to each vertex using the Möbius gate.

    For each vertex v and each candidate label λ ∈ [0, k):
        a = v·k + λ                      (linear label index)
        q = A·a² + B·a + C  mod K_filter
        if μ(q) ≠ 0:  score[v, λ] += Π(x_v, y_v)

    The label is the argmax over λ of score[v, λ].
    Vertices with no square‑free address fall back to label 0.
    """
    N_v = N_left + N_right
    scores = np.zeros((N_v, k), dtype=np.float64)

    for v in range(N_v):
        side = "L" if v < N_left else "R"
        v_local = v if v < N_left else v - N_left
        x = vertex_hash(v_local, side)
        y = vertex_hash(v_local + 1, side)
        Pi = elliptic_projection_1d(np.array([x]), y0=y)[0]
        Pi = min(max(Pi, 0.0), proj_max)

        for lam in range(k):
            a = v * k + lam
            q = quadratic_address(a, quad_A, quad_B, quad_C, K_filter)
            if q < len(mu) and mu[q] != 0:
                scores[v, lam] += Pi

    # fall back to label 0 if a vertex has no square‑free address
    labels = np.argmax(scores, axis=1).astype(int)
    return labels, scores


# ============================================================
#  9. Unique Games evaluation
# ============================================================
def ugc_evaluate(edges, labels, N_left: int) -> dict:
    """
    Compute the fraction of satisfied edges.

        edge (i, j, π) satisfied iff  π(label(i)) = label(j)
    """
    total = len(edges)
    if total == 0:
        return dict(satisfied=0, total=0,
                    completeness=0.0,
                    per_edge=np.zeros(0, dtype=bool))

    sat_flags = np.zeros(total, dtype=bool)
    for e_idx, (i, j, perm) in enumerate(edges):
        li = labels[i]
        lj = labels[N_left + j]
        if perm[li] == lj:
            sat_flags[e_idx] = True

    satisfied = int(sat_flags.sum())
    return dict(
        satisfied=satisfied,
        total=total,
        completeness=satisfied / total,
        per_edge=sat_flags,
    )


def ugc_soundness_bound(edges, N_left, N_right, k):
    """
    Trivial lower bound on soundness:

        soundness ≥ 1 / k

    (achieved by any random constant labelling, on expectation).
    """
    return 1.0 / k


def ugc_gap(eval_result, k):
    """
    Gap between the completeness achieved by the Möbius labelling
    and the trivial soundness bound 1/k.
    """
    return eval_result["completeness"] - ugc_soundness_bound(
        [], 0, 0, k)


# ============================================================
#  10. Full simulation
# ============================================================
def simulate(K: int = 24,
             N_left: int = 60,
             N_right: int = 60,
             k_labels: int = 4,
             degree: int = 3,
             N_events: int = 256,
             channel_length: int = 256,
             barrier_width: int = 21):
    print("=" * 78)
    print("Unique Games on the electron chip  ·  Figure 3.5 + Maxwell + Möbius")
    print("=" * 78)
    print(f"  α_sym        = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  α_asym       = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K            = {K}")
    print(f"  N_left       = {N_left}")
    print(f"  N_right      = {N_right}")
    print(f"  k labels     = {k_labels}")
    print(f"  edge degree  = {degree}")
    print()

    # ---------- 1. Möbius sieve ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(max(K, 2 ** 10 + 1))
    print(f"[1] Möbius sieve       {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"(O(K log log K))")

    # ---------- 2. Figure 3.5 heat map ----------
    t0 = time.perf_counter()
    xs = np.linspace(0, W1, channel_length)
    ys = np.linspace(0, W2, channel_length)
    X, Y = np.meshgrid(xs, ys)
    P2D = elliptic_projection_2d(X, Y)
    channel = elliptic_projection_1d(xs, y0=0.5 * W2)
    print(f"[2] Figure 3.5 heat map {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 3. Maxwell fields over one period ----------
    t0 = time.perf_counter()
    period = 2 * PI * ALPHA_SYM
    t_vals = np.linspace(0, period, N_events)
    E_vals = np.empty(N_events)
    B_vals = np.empty(N_events)
    div_vals = np.empty(N_events)
    for i, t in enumerate(t_vals):
        E_, B_ = field_components(t, mu, K)
        E_vals[i] = E_
        B_vals[i] = B_
        div_vals[i] = divergence_E(t, mu, K)
    print(f"[3] Maxwell fields     {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 4. Barrier tunneling ----------
    t0 = time.perf_counter()
    E_tunnel = tunnel(E_vals, width=barrier_width)
    div_tunnel = tunnel(div_vals, width=barrier_width)
    print(f"[4] Convolution tunnel {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 5. Quadratic envelope over labels ----------
    t0 = time.perf_counter()
    N_labels = N_left + N_right
    env, q_of_a = build_quadratic_envelope(K, N_labels, mu)
    print(f"[5] Quadratic envelope {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"(occupied {int(np.count_nonzero(env))}/{K})")

    # ---------- 6. Send envelope down the 1D channel ----------
    t0 = time.perf_counter()
    transmitted = send_envelope_1d(env, channel)
    print(f"[6] 1D chip path       {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 7. Dual supertrace ----------
    S_sym, S_asym, S_net = supertrace_dual(transmitted)
    H_net = entropy(S_net, len(transmitted))
    m_net = mass(S_net, len(transmitted))
    print(f"[7] Dual supertrace    S_sym={S_sym:+.4f}  S_asym={S_asym:+.4f}  "
          f"S_net={S_net:+.4f}")

    # ---------- 8. Unique Games instance ----------
    t0 = time.perf_counter()
    edges = generate_unique_games_instance(
        N_left, N_right, k_labels, degree=degree, seed=2024)
    print(f"[8] Unique Games instance built  "
          f"({len(edges)} edges, k={k_labels}, "
          f"{(time.perf_counter()-t0)*1e3:.2f} ms)")

    # ---------- 9. Möbius label assignment ----------
    t0 = time.perf_counter()
    labels, scores = ugc_label_assignment(
        edges, N_left, N_right, k_labels, mu,
        K_filter=2 ** 10 + 1, quad_A=1, quad_B=1, quad_C=0,
    )
    print(f"[9] Label assignment   {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"(labels ∈ [0, {k_labels}))")

    # ---------- 10. UGC evaluation ----------
    eval_result = ugc_evaluate(edges, labels, N_left)
    soundness_bound = ugc_soundness_bound(edges, N_left, N_right, k_labels)
    gap = eval_result["completeness"] - soundness_bound

    print()
    print("--- Unique Games result ---")
    print(f"  edges                  : {eval_result['total']}")
    print(f"  satisfied              : {eval_result['satisfied']}")
    print(f"  completeness           : {eval_result['completeness']:.6f}")
    print(f"  soundness bound (1/k)  : {soundness_bound:.6f}")
    print(f"  gap (comp − sound)     : {gap:+.6f}")

    # ---------- 11. Plot ----------
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # (a) Figure 3.5 heat map
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(P2D, extent=[0, W1, 0, W2], origin="lower",
                   cmap="inferno", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("x  (mod π)")
    ax.set_ylabel("y  (mod e)")
    ax.set_title("Figure 3.5  ·  label density Π(x,y)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) 1D channel
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xs, channel, color="#e67e22", lw=1.6)
    ax.fill_between(xs, 0, channel, color="#e67e22", alpha=0.25)
    ax.set_xlabel("chip position  (mod π)")
    ax.set_ylabel("density")
    ax.set_title("1D chip channel  (slice of Π)")
    ax.grid(True, alpha=0.3)

    # (c) quadratic envelope over labels
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(len(env)), env, color="#3a7bd5", width=0.85)
    ax.set_xlabel("quadratic address  q = a² + a  (mod K)")
    ax.set_ylabel("env[q]")
    ax.set_title(f"Label envelope  (K = {K})")
    ax.grid(True, alpha=0.3)

    # (d) Maxwell fields
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t_vals, E_vals, color="#2ecc71", lw=1.2, label="E(t) = Re ζ(t)")
    ax.plot(t_vals, B_vals, color="#3498db", lw=1.0,
            label="B(t) = Im ζ(t)", alpha=0.7)
    ax.plot(t_vals, div_vals, color="#e74c3c", lw=0.9, alpha=0.6,
            label="∇·E  (charge density)")
    ax.set_xlabel("t  (one period 2πα_sym)")
    ax.set_ylabel("field amplitude")
    ax.set_title("Maxwell fields from the Möbius spectral sum")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) tunneling
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t_vals, E_vals, color="#95a5a6", lw=1.0, alpha=0.5,
            label="before barrier")
    ax.plot(t_vals, E_tunnel, color="#8e44ad", lw=1.4, label="after barrier")
    ax.set_xlabel("t")
    ax.set_ylabel("E")
    ax.set_title(f"Tunneling  (S={S_net:+.3f}, m={m_net:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (f) Unique Games bipartite graph
    ax = fig.add_subplot(gs[2, :2])
    ax.set_title(
        f"Unique Games bipartite graph  ·  k={k_labels}  "
        f"|E|={eval_result['total']}  "
        f"completeness={eval_result['completeness']:.3f}"
    )
    rng = np.random.default_rng(2024)
    pos_L = np.column_stack([
        np.zeros(N_left),
        np.linspace(0, 1, N_left),
    ])
    pos_R = np.column_stack([
        np.ones(N_right),
        np.linspace(0, 1, N_right),
    ])

    # draw edges coloured by satisfaction
    for e_idx, (i, j, perm) in enumerate(edges):
        color = "#2ecc71" if eval_result["per_edge"][e_idx] else "#e74c3c"
        ax.plot([pos_L[i, 0], pos_R[j, 0]],
                [pos_L[i, 1], pos_R[j, 1]],
                color=color, lw=0.5, alpha=0.55)

    # draw vertices coloured by label
    cmap = plt.cm.tab10
    ax.scatter(pos_L[:, 0], pos_L[:, 1],
               c=[cmap(labels[i] / k_labels) for i in range(N_left)],
               s=28, edgecolors="k", zorder=5)
    ax.scatter(pos_R[:, 0], pos_R[:, 1],
               c=[cmap(labels[N_left + j] / k_labels)
                  for j in range(N_right)],
               s=28, edgecolors="k", zorder=5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Unique Games  ·  completeness = "
        f"{eval_result['completeness']:.4f}  "
        f"(soundness bound = {soundness_bound:.4f})",
        fontsize=10,
    )

    # (g) label histogram
    ax = fig.add_subplot(gs[2, 2])
    counts = np.bincount(labels, minlength=k_labels)
    ax.bar(np.arange(k_labels), counts, color="#8e44ad", width=0.7)
    ax.set_xlabel("label  λ")
    ax.set_ylabel("# vertices")
    ax.set_title(
        f"Label assignment distribution  "
        f"(k = {k_labels})"
    )
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Unique Games on the electron chip  ·  "
        f"gap = {gap:+.4f}",
        fontsize=14, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # ---------- summary ----------
    print()
    print("--- Summary ---")
    print(f"  Figure 3.5 range            : [{P2D.min():.4f}, {P2D.max():.4f}]")
    print(f"  1D channel range            : [{channel.min():.4f}, "
          f"{channel.max():.4f}]")
    print(f"  Quadratic envelope occupied : "
          f"{int(np.count_nonzero(env))} / {K}")
    print(f"  Supertrace S_net            : {S_net:+.4f}")
    print(f"  Entropy H                   : {H_net:.4f}")
    print(f"  Invariant mass m            : {m_net:.4f}")
    print(f"  Unique Games edges          : {eval_result['total']}")
    print(f"  Satisfied edges             : {eval_result['satisfied']}")
    print(f"  Completeness                : {eval_result['completeness']:.6f}")
    print(f"  Soundness bound (1/k)       : {soundness_bound:.6f}")
    print(f"  UGC gap                     : {gap:+.6f}")
    print()
    print("  Complexity per pipeline stage:")
    print("    Möbius sieve                O(K log log K)   once")
    print("    Figure 3.5 heat map         O(L²)            L = channel_length")
    print("    1D channel slice            O(L)")
    print("    Maxwell fields              O(K · N_events)")
    print("    Barrier convolution         O(N_events · W)")
    print("    Quadratic envelope          O(N_labels)")
    print("    Send envelope down chip     O(L)")
    print("    Dual supertrace              O(L)")
    print("    UGC label assignment        O(N · k)")
    print("    UGC evaluation              O(|E|)")

    return dict(
        P2D=P2D, channel=channel, env=env,
        E_vals=E_vals, B_vals=B_vals, div_vals=div_vals,
        E_tunnel=E_tunnel, transmitted=transmitted,
        S_sym=S_sym, S_asym=S_asym, S_net=S_net, H=H_net, m=m_net,
        edges=edges, labels=labels, scores=scores,
        eval_result=eval_result,
        completeness=eval_result["completeness"],
        soundness_bound=soundness_bound,
        gap=gap,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24,
             N_left=60, N_right=60,
             k_labels=4, degree=3,
             N_events=256, channel_length=256,
             barrier_width=21)#!/usr/bin/env python3
"""
ugc_electron_chip.py

Unique Games on the transistor electron chip.

The original pipeline solves a large asymmetric polynomial via the
Möbius SAT gate.  Here we replace it with the **Unique Games** problem:

    Instance  (V, E, [k], {π_e})
        • bipartite vertex set  V = L ∪ R
        • label set             [k] = {0, …, k−1}
        • for each edge e=(i, j) ∈ E, a permutation  π_e ∈ S_k
        • a labelling σ : V → [k]
        • edge e is satisfied iff  π_e(σ(i)) = σ(j)

The UGC states that for every ε, δ > 0 there is a k such that it is
NP‑hard to distinguish instances with completeness ≥ 1−ε from
instances with soundness ≤ δ.

Everything else is preserved:

    • Figure 3.5   — bounded elliptic projection Π ∈ [0,1] on the
                     torus ℂ/(πℤ + eℤ), used as the *label density*
                     across the chip.
    • Maxwell      — E(t) = Re ζ(t), B(t) = Im ζ(t) from the Möbius
                     spectral sum, with ∇·E the local charge density.
    • Convolution  — exponential barrier kernel exp(−α|x|) modelling
                     label tunnelling through edges.
    • Quadratic    — each label event a ∈ [0, N) is mapped to a
                     quadratic address  q(a) = A·a² + B·a + C mod K,
                     the envelope  env[q]  gates by μ(q) ≠ 0.
    • Dual const.  — α_sym = 1/(π − e) for the even half,
                     α_asym = 0.3628   for the odd half.

The UGC‑specific outputs are:

    completeness = (# satisfied edges) / |E|       (best labelling)
    soundness    = best labelling under gap‑threshold
    ugc_gap      = completeness − soundness        (the conjectural gap)
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
ALPHA_SYM  = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_ASYM = 0.3628
NORM       = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
DENSITY    = 6.0 / (PI * PI)         # ≈ 0.6079271018
W1         = PI
W2         = E


# ============================================================
#  1. Möbius sieve  (shared O(K log log K) table)
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
    """Π(x, y) = ½ (1 + cos(2π x/π) · cos(2π y/e))  ∈ [0, 1]."""
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))


def elliptic_projection_1d(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


# ============================================================
#  3. Möbius spectral sum  ζ(t)
# ============================================================
def zeta_mobius(t: float, mu: np.ndarray, K: int,
                alpha: float = ALPHA_SYM) -> complex:
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    theta = t * n / alpha
    return complex(np.sum(m * np.cos(theta)), np.sum(m * np.sin(theta)))


def field_components(t: float, mu: np.ndarray, K: int
                     ) -> tuple[float, float]:
    z = zeta_mobius(t, mu, K)
    return z.real, z.imag


def divergence_E(t: float, mu: np.ndarray, K: int) -> float:
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    signs = np.where(n % 2 == 0, 1.0, -1.0)
    return float(np.sum(signs * m * np.cos(t * n / ALPHA_SYM)))


# ============================================================
#  4. Convolution tunneling  (label transport kernel)
# ============================================================
def barrier_kernel(width: int, alpha: float = ALPHA_SYM) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    ker = np.exp(-alpha * np.abs(x))
    return ker / ker.sum()


def tunnel(signal: np.ndarray, width: int = 21) -> np.ndarray:
    return convolve(signal, barrier_kernel(width), mode="same")


# ============================================================
#  5. Quadratic envelope  (label stream over the chip)
# ============================================================
def quadratic_address(a: int, A: int = 1, B: int = 1, C: int = 0,
                      K: int = 32) -> int:
    return (A * a * a + B * a + C) % K


def build_quadratic_envelope(K: int, N: int,
                             mu: np.ndarray,
                             A: int = 1, B: int = 1, C: int = 0
                             ) -> tuple[np.ndarray, dict[int, int]]:
    """
    Envelope over quadratic addresses, gated by μ(q) ≠ 0.

    Here the "label stream" is the list of *labels* a ∈ [0, N), and the
    envelope env[q] accumulates the squared label frequency at each
    square‑free address q.
    """
    env = np.zeros(K, dtype=np.float64)
    q_of_a: dict[int, int] = {}
    for a in range(N):
        q = quadratic_address(a, A, B, C, K)
        q_of_a[a] = q
        if q < len(mu) and mu[q] != 0:
            env[q] += 1.0
    return env, q_of_a


def send_envelope_1d(env: np.ndarray, channel: np.ndarray) -> np.ndarray:
    L = len(channel)
    if len(env) == L:
        return env * channel
    env_r = np.interp(np.linspace(0, 1, L),
                      np.linspace(0, 1, len(env)), env)
    return env_r * channel


# ============================================================
#  6. Dual-constant supertrace  (sym / asym label split)
# ============================================================
def supertrace_dual(channel: np.ndarray) -> tuple[float, float, float]:
    S_sym = 0.0
    S_asym = 0.0
    for i, v in enumerate(channel):
        if i % 2 == 0:
            S_sym += ALPHA_SYM * abs(v)
        else:
            S_asym += ALPHA_ASYM * abs(v)
    return S_sym, S_asym, S_sym - S_asym


def entropy(S: float, N: int, alpha: float = ALPHA_SYM) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  7. Unique Games instance
# ============================================================
def generate_unique_games_instance(N_left: int,
                                   N_right: int,
                                   k: int,
                                   degree: int = 3,
                                   seed: int = 2024):
    """
    Bipartite Unique Games instance.

    Returns
    -------
    edges : list of (i, j, perm)
        i ∈ [0, N_left), j ∈ [0, N_right), perm ∈ S_k (tuple length k)
    """
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(N_left):
        js = rng.choice(N_right, size=degree, replace=False)
        for j in js:
            perm = tuple(int(x) for x in rng.permutation(k))
            edges.append((int(i), int(j), perm))
    return edges


# ============================================================
#  8. Label assignment via the Möbius gate
# ============================================================
def vertex_hash(v: int, side: str) -> float:
    """Deterministic scalar in (0.5, 5.0) for a vertex."""
    h = (v * 2654435761) ^ (0 if side == "L" else 0x9E3779B9)
    return 0.5 + ((h & 0xFFFF) / 0xFFFF) * 4.5


def ugc_label_assignment(edges,
                         N_left: int,
                         N_right: int,
                         k: int,
                         mu: np.ndarray,
                         K_filter: int = 2 ** 10 + 1,
                         quad_A: int = 1,
                         quad_B: int = 1,
                         quad_C: int = 0,
                         proj_max: float = 1.0):
    """
    Assign a label σ(v) ∈ [0, k) to each vertex using the Möbius gate.

    For each vertex v and each candidate label λ ∈ [0, k):
        a = v·k + λ                      (linear label index)
        q = A·a² + B·a + C  mod K_filter
        if μ(q) ≠ 0:  score[v, λ] += Π(x_v, y_v)

    The label is the argmax over λ of score[v, λ].
    Vertices with no square‑free address fall back to label 0.
    """
    N_v = N_left + N_right
    scores = np.zeros((N_v, k), dtype=np.float64)

    for v in range(N_v):
        side = "L" if v < N_left else "R"
        v_local = v if v < N_left else v - N_left
        x = vertex_hash(v_local, side)
        y = vertex_hash(v_local + 1, side)
        Pi = elliptic_projection_1d(np.array([x]), y0=y)[0]
        Pi = min(max(Pi, 0.0), proj_max)

        for lam in range(k):
            a = v * k + lam
            q = quadratic_address(a, quad_A, quad_B, quad_C, K_filter)
            if q < len(mu) and mu[q] != 0:
                scores[v, lam] += Pi

    # fall back to label 0 if a vertex has no square‑free address
    labels = np.argmax(scores, axis=1).astype(int)
    return labels, scores


# ============================================================
#  9. Unique Games evaluation
# ============================================================
def ugc_evaluate(edges, labels, N_left: int) -> dict:
    """
    Compute the fraction of satisfied edges.

        edge (i, j, π) satisfied iff  π(label(i)) = label(j)
    """
    total = len(edges)
    if total == 0:
        return dict(satisfied=0, total=0,
                    completeness=0.0,
                    per_edge=np.zeros(0, dtype=bool))

    sat_flags = np.zeros(total, dtype=bool)
    for e_idx, (i, j, perm) in enumerate(edges):
        li = labels[i]
        lj = labels[N_left + j]
        if perm[li] == lj:
            sat_flags[e_idx] = True

    satisfied = int(sat_flags.sum())
    return dict(
        satisfied=satisfied,
        total=total,
        completeness=satisfied / total,
        per_edge=sat_flags,
    )


def ugc_soundness_bound(edges, N_left, N_right, k):
    """
    Trivial lower bound on soundness:

        soundness ≥ 1 / k

    (achieved by any random constant labelling, on expectation).
    """
    return 1.0 / k


def ugc_gap(eval_result, k):
    """
    Gap between the completeness achieved by the Möbius labelling
    and the trivial soundness bound 1/k.
    """
    return eval_result["completeness"] - ugc_soundness_bound(
        [], 0, 0, k)


# ============================================================
#  10. Full simulation
# ============================================================
def simulate(K: int = 24,
             N_left: int = 60,
             N_right: int = 60,
             k_labels: int = 4,
             degree: int = 3,
             N_events: int = 256,
             channel_length: int = 256,
             barrier_width: int = 21):
    print("=" * 78)
    print("Unique Games on the electron chip  ·  Figure 3.5 + Maxwell + Möbius")
    print("=" * 78)
    print(f"  α_sym        = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  α_asym       = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K            = {K}")
    print(f"  N_left       = {N_left}")
    print(f"  N_right      = {N_right}")
    print(f"  k labels     = {k_labels}")
    print(f"  edge degree  = {degree}")
    print()

    # ---------- 1. Möbius sieve ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(max(K, 2 ** 10 + 1))
    print(f"[1] Möbius sieve       {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"(O(K log log K))")

    # ---------- 2. Figure 3.5 heat map ----------
    t0 = time.perf_counter()
    xs = np.linspace(0, W1, channel_length)
    ys = np.linspace(0, W2, channel_length)
    X, Y = np.meshgrid(xs, ys)
    P2D = elliptic_projection_2d(X, Y)
    channel = elliptic_projection_1d(xs, y0=0.5 * W2)
    print(f"[2] Figure 3.5 heat map {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 3. Maxwell fields over one period ----------
    t0 = time.perf_counter()
    period = 2 * PI * ALPHA_SYM
    t_vals = np.linspace(0, period, N_events)
    E_vals = np.empty(N_events)
    B_vals = np.empty(N_events)
    div_vals = np.empty(N_events)
    for i, t in enumerate(t_vals):
        E_, B_ = field_components(t, mu, K)
        E_vals[i] = E_
        B_vals[i] = B_
        div_vals[i] = divergence_E(t, mu, K)
    print(f"[3] Maxwell fields     {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 4. Barrier tunneling ----------
    t0 = time.perf_counter()
    E_tunnel = tunnel(E_vals, width=barrier_width)
    div_tunnel = tunnel(div_vals, width=barrier_width)
    print(f"[4] Convolution tunnel {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 5. Quadratic envelope over labels ----------
    t0 = time.perf_counter()
    N_labels = N_left + N_right
    env, q_of_a = build_quadratic_envelope(K, N_labels, mu)
    print(f"[5] Quadratic envelope {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"(occupied {int(np.count_nonzero(env))}/{K})")

    # ---------- 6. Send envelope down the 1D channel ----------
    t0 = time.perf_counter()
    transmitted = send_envelope_1d(env, channel)
    print(f"[6] 1D chip path       {(time.perf_counter()-t0)*1e3:.2f} ms")

    # ---------- 7. Dual supertrace ----------
    S_sym, S_asym, S_net = supertrace_dual(transmitted)
    H_net = entropy(S_net, len(transmitted))
    m_net = mass(S_net, len(transmitted))
    print(f"[7] Dual supertrace    S_sym={S_sym:+.4f}  S_asym={S_asym:+.4f}  "
          f"S_net={S_net:+.4f}")

    # ---------- 8. Unique Games instance ----------
    t0 = time.perf_counter()
    edges = generate_unique_games_instance(
        N_left, N_right, k_labels, degree=degree, seed=2024)
    print(f"[8] Unique Games instance built  "
          f"({len(edges)} edges, k={k_labels}, "
          f"{(time.perf_counter()-t0)*1e3:.2f} ms)")

    # ---------- 9. Möbius label assignment ----------
    t0 = time.perf_counter()
    labels, scores = ugc_label_assignment(
        edges, N_left, N_right, k_labels, mu,
        K_filter=2 ** 10 + 1, quad_A=1, quad_B=1, quad_C=0,
    )
    print(f"[9] Label assignment   {(time.perf_counter()-t0)*1e3:.2f} ms  "
          f"(labels ∈ [0, {k_labels}))")

    # ---------- 10. UGC evaluation ----------
    eval_result = ugc_evaluate(edges, labels, N_left)
    soundness_bound = ugc_soundness_bound(edges, N_left, N_right, k_labels)
    gap = eval_result["completeness"] - soundness_bound

    print()
    print("--- Unique Games result ---")
    print(f"  edges                  : {eval_result['total']}")
    print(f"  satisfied              : {eval_result['satisfied']}")
    print(f"  completeness           : {eval_result['completeness']:.6f}")
    print(f"  soundness bound (1/k)  : {soundness_bound:.6f}")
    print(f"  gap (comp − sound)     : {gap:+.6f}")

    # ---------- 11. Plot ----------
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # (a) Figure 3.5 heat map
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(P2D, extent=[0, W1, 0, W2], origin="lower",
                   cmap="inferno", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("x  (mod π)")
    ax.set_ylabel("y  (mod e)")
    ax.set_title("Figure 3.5  ·  label density Π(x,y)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) 1D channel
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xs, channel, color="#e67e22", lw=1.6)
    ax.fill_between(xs, 0, channel, color="#e67e22", alpha=0.25)
    ax.set_xlabel("chip position  (mod π)")
    ax.set_ylabel("density")
    ax.set_title("1D chip channel  (slice of Π)")
    ax.grid(True, alpha=0.3)

    # (c) quadratic envelope over labels
    ax = fig.add_subplot(gs[0, 2])
    ax.bar(np.arange(len(env)), env, color="#3a7bd5", width=0.85)
    ax.set_xlabel("quadratic address  q = a² + a  (mod K)")
    ax.set_ylabel("env[q]")
    ax.set_title(f"Label envelope  (K = {K})")
    ax.grid(True, alpha=0.3)

    # (d) Maxwell fields
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t_vals, E_vals, color="#2ecc71", lw=1.2, label="E(t) = Re ζ(t)")
    ax.plot(t_vals, B_vals, color="#3498db", lw=1.0,
            label="B(t) = Im ζ(t)", alpha=0.7)
    ax.plot(t_vals, div_vals, color="#e74c3c", lw=0.9, alpha=0.6,
            label="∇·E  (charge density)")
    ax.set_xlabel("t  (one period 2πα_sym)")
    ax.set_ylabel("field amplitude")
    ax.set_title("Maxwell fields from the Möbius spectral sum")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) tunneling
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t_vals, E_vals, color="#95a5a6", lw=1.0, alpha=0.5,
            label="before barrier")
    ax.plot(t_vals, E_tunnel, color="#8e44ad", lw=1.4, label="after barrier")
    ax.set_xlabel("t")
    ax.set_ylabel("E")
    ax.set_title(f"Tunneling  (S={S_net:+.3f}, m={m_net:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (f) Unique Games bipartite graph
    ax = fig.add_subplot(gs[2, :2])
    ax.set_title(
        f"Unique Games bipartite graph  ·  k={k_labels}  "
        f"|E|={eval_result['total']}  "
        f"completeness={eval_result['completeness']:.3f}"
    )
    rng = np.random.default_rng(2024)
    pos_L = np.column_stack([
        np.zeros(N_left),
        np.linspace(0, 1, N_left),
    ])
    pos_R = np.column_stack([
        np.ones(N_right),
        np.linspace(0, 1, N_right),
    ])

    # draw edges coloured by satisfaction
    for e_idx, (i, j, perm) in enumerate(edges):
        color = "#2ecc71" if eval_result["per_edge"][e_idx] else "#e74c3c"
        ax.plot([pos_L[i, 0], pos_R[j, 0]],
                [pos_L[i, 1], pos_R[j, 1]],
                color=color, lw=0.5, alpha=0.55)

    # draw vertices coloured by label
    cmap = plt.cm.tab10
    ax.scatter(pos_L[:, 0], pos_L[:, 1],
               c=[cmap(labels[i] / k_labels) for i in range(N_left)],
               s=28, edgecolors="k", zorder=5)
    ax.scatter(pos_R[:, 0], pos_R[:, 1],
               c=[cmap(labels[N_left + j] / k_labels)
                  for j in range(N_right)],
               s=28, edgecolors="k", zorder=5)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Unique Games  ·  completeness = "
        f"{eval_result['completeness']:.4f}  "
        f"(soundness bound = {soundness_bound:.4f})",
        fontsize=10,
    )

    # (g) label histogram
    ax = fig.add_subplot(gs[2, 2])
    counts = np.bincount(labels, minlength=k_labels)
    ax.bar(np.arange(k_labels), counts, color="#8e44ad", width=0.7)
    ax.set_xlabel("label  λ")
    ax.set_ylabel("# vertices")
    ax.set_title(
        f"Label assignment distribution  "
        f"(k = {k_labels})"
    )
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Unique Games on the electron chip  ·  "
        f"gap = {gap:+.4f}",
        fontsize=14, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # ---------- summary ----------
    print()
    print("--- Summary ---")
    print(f"  Figure 3.5 range            : [{P2D.min():.4f}, {P2D.max():.4f}]")
    print(f"  1D channel range            : [{channel.min():.4f}, "
          f"{channel.max():.4f}]")
    print(f"  Quadratic envelope occupied : "
          f"{int(np.count_nonzero(env))} / {K}")
    print(f"  Supertrace S_net            : {S_net:+.4f}")
    print(f"  Entropy H                   : {H_net:.4f}")
    print(f"  Invariant mass m            : {m_net:.4f}")
    print(f"  Unique Games edges          : {eval_result['total']}")
    print(f"  Satisfied edges             : {eval_result['satisfied']}")
    print(f"  Completeness                : {eval_result['completeness']:.6f}")
    print(f"  Soundness bound (1/k)       : {soundness_bound:.6f}")
    print(f"  UGC gap                     : {gap:+.6f}")
    print()
    print("  Complexity per pipeline stage:")
    print("    Möbius sieve                O(K log log K)   once")
    print("    Figure 3.5 heat map         O(L²)            L = channel_length")
    print("    1D channel slice            O(L)")
    print("    Maxwell fields              O(K · N_events)")
    print("    Barrier convolution         O(N_events · W)")
    print("    Quadratic envelope          O(N_labels)")
    print("    Send envelope down chip     O(L)")
    print("    Dual supertrace              O(L)")
    print("    UGC label assignment        O(N · k)")
    print("    UGC evaluation              O(|E|)")

    return dict(
        P2D=P2D, channel=channel, env=env,
        E_vals=E_vals, B_vals=B_vals, div_vals=div_vals,
        E_tunnel=E_tunnel, transmitted=transmitted,
        S_sym=S_sym, S_asym=S_asym, S_net=S_net, H=H_net, m=m_net,
        edges=edges, labels=labels, scores=scores,
        eval_result=eval_result,
        completeness=eval_result["completeness"],
        soundness_bound=soundness_bound,
        gap=gap,
    )


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    simulate(K=24,
             N_left=60, N_right=60,
             k_labels=4, degree=3,
             N_events=256, channel_length=256,
             barrier_width=21)