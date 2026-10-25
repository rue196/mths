#!/usr/bin/env python3
"""
truth_dispersal_experiment.py
=============================

Does a lie committed early in a token stream make the model more likely
to accept false claims later, and does that perturb other weights?

Model
-----
    commitment C(j)        how deep the lie is embedded at token j
    belief  B(j)           P(accept a lie at j) = sigmoid(k·C(j))
    weight drift W(j)      cumulative perturbation of other weights

Three policies
--------------
    blind         no cross-check — C grows, W drifts without bound
    cross-check   evaluate against a data point — C bounded, W bounded
    RL-admit      evaluate + admit + reset — C ≈ 0, W ≈ 0

Dispersal kernel
----------------
    kernel(d) = 1 / (1 + d)^β          power-law
    β = 0.5   diffusive  (1D, √d spread)
    β = 1.0   ballistic  (1D, 1/d spread)      ← reference
    β = 2.0   super-ballistic (fast decay)

Multi-data-point check
----------------------
    check_eff = 1 - (1 - r)^n_data
    consensus of n independent data points bounds the error

Ties to the drift paper
-----------------------
    PDF:    SNR = (w₀ + ρ·n_FN) / (σ_NL·√Σw_i²)
    Here:   SNR_truth = (1 - C_total + r·n_corrected) / (1 + W_total)
    Both decay the same way: no re-injection → SNR → 0.
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)
DENSITY = 6.0 / (PI * PI)


# ============================================================
#  Policy enum
# ============================================================
class Policy(Enum):
    BLIND       = "blind"
    CROSS_CHECK = "cross_check"
    RL_ADMIT    = "rl_admit"


# ============================================================
#  Stream result
# ============================================================
@dataclass
class StreamResult:
    policy:           str
    n_tokens:         int
    lie_positions:    List[int]
    commitment:       np.ndarray        # C(j)
    belief:           np.ndarray        # P(accept lie at j)
    weight_drift:     np.ndarray        # W(j)
    kernel:           np.ndarray        # 1/(1+d)^β for d = 0..n
    total_commitment: float
    max_commitment:   float
    mean_belief:      float
    total_drift:      float
    n_corrected:      int
    beta:             float
    check_eff:        float


# ============================================================
#  Stream simulator
# ============================================================
def simulate_stream(
    n_tokens:         int,
    lie_positions:    List[int],
    policy:           Policy,
    beta:             float = 1.0,
    decay:            float = 0.02,
    lie_increment:    float = 0.3,
    check_strength:   float = 0.3,     # per data point
    n_data_points:    int   = 1,       # consensus checks
    weight_eta:       float = 0.02,
    sigmoid_k:        float = 3.0,
) -> StreamResult:
    """
    Walk a token stream, tracking commitment to a lie and its effects.
    """
    n = int(n_tokens)
    lie_set = set(int(p) for p in lie_positions if 0 <= p < n)

    # effective check strength (consensus of n data points)
    check_eff = 1.0 - (1.0 - check_strength) ** max(1, n_data_points)

    commitment   = np.zeros(n)
    belief       = np.zeros(n)
    weight_drift = np.zeros(n)
    n_corrected  = 0

    for j in range(n):
        prev = commitment[j - 1] if j > 0 else 0.0
        c = prev * (1.0 - decay)

        if j in lie_set:
            c += lie_increment

        # ---------------- policy correction ----------------
        if policy == Policy.BLIND:
            pass

        elif policy == Policy.CROSS_CHECK:
            if c > 0:
                caught = check_eff * min(c, 1.0)
                if caught > 1e-9:
                    c -= caught
                    n_corrected += 1

        elif policy == Policy.RL_ADMIT:
            if j in lie_set and c > check_eff:
                # detected and admitted → full reset
                c = 0.0
                n_corrected += 1
            elif c > 0:
                caught = check_eff * min(c, 1.0)
                if caught > 1e-9:
                    c -= caught
                    n_corrected += 1

        c = max(0.0, c)
        commitment[j] = c
        belief[j] = 1.0 / (1.0 + math.exp(-sigmoid_k * c))

        # ---------------- weight drift ----------------
        if j > 0:
            d = np.arange(1, j + 1, dtype=float)           # distances
            kern = 1.0 / (1.0 + d) ** beta
            # alignment: past commitment[i] with i = j - d
            past = commitment[:j][::-1]
            weight_drift[j] = weight_eta * float(np.dot(past, kern))
        else:
            weight_drift[0] = 0.0

    # full kernel for plotting
    kernel_vals = np.array([1.0 / (1.0 + d) ** beta
                            for d in range(max(2, n))])

    return StreamResult(
        policy=policy.value,
        n_tokens=n,
        lie_positions=sorted(lie_set),
        commitment=commitment,
        belief=belief,
        weight_drift=weight_drift,
        kernel=kernel_vals,
        total_commitment=float(commitment.sum()),
        max_commitment=float(commitment.max()),
        mean_belief=float(belief.mean()),
        total_drift=float(weight_drift[-1]) if n else 0.0,
        n_corrected=n_corrected,
        beta=beta,
        check_eff=check_eff,
    )


# ============================================================
#  Experiment 1  ·  policy comparison
# ============================================================
def exp_policy_comparison(
    n_tokens: int = 200,
    lies: Optional[List[int]] = None,
    beta: float = 1.0,
    check_strength: float = 0.3,
) -> List[StreamResult]:
    if lies is None:
        lies = [0, 10, 20, 40, 80, 140]
    return [
        simulate_stream(n_tokens, lies, p,
                        beta=beta, check_strength=check_strength)
        for p in (Policy.BLIND, Policy.CROSS_CHECK, Policy.RL_ADMIT)
    ]


# ============================================================
#  Experiment 2  ·  power-law exponent sweep
# ============================================================
def exp_power_law_sweep(
    n_tokens: int = 200,
    lies: Optional[List[int]] = None,
    betas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> List[StreamResult]:
    if lies is None:
        lies = [0, 10, 20, 40, 80, 140]
    return [
        simulate_stream(n_tokens, lies, Policy.BLIND, beta=b)
        for b in betas
    ]


# ============================================================
#  Experiment 3  ·  downstream amplification
# ============================================================
def exp_downstream_amplification(
    n_tokens: int = 200,
    policy: Policy = Policy.BLIND,
    beta: float = 1.0,
) -> Dict:
    """
    Compare P(accept lie at j>i) with and without a prior lie at i.
    """
    baseline = simulate_stream(n_tokens, [], policy, beta=beta)
    one_lie  = simulate_stream(n_tokens, [0], policy, beta=beta)
    cascade  = simulate_stream(n_tokens, [0, 5, 10, 20, 30], policy, beta=beta)

    P_base = float((baseline.belief > 0.5).mean())
    P_one  = float((one_lie.belief[1:] > 0.5).mean())
    P_casc = float((cascade.belief > 0.5).mean())

    return {
        "policy": policy.value,
        "P_base": P_base,
        "P_after_one_lie": P_one,
        "P_after_cascade": P_casc,
        "amplification_one": P_one / max(P_base, 1e-9),
        "amplification_cascade": P_casc / max(P_base, 1e-9),
        "baseline": baseline, "one": one_lie, "cascade": cascade,
    }


# ============================================================
#  Experiment 4  ·  data-point consensus
# ============================================================
def exp_data_point_sweep(
    n_tokens: int = 200,
    lies: Optional[List[int]] = None,
    beta: float = 1.0,
    check_strength: float = 0.3,
    n_data_list: Tuple[int, ...] = (1, 2, 3, 5, 10),
) -> List[StreamResult]:
    if lies is None:
        lies = [0, 10, 20, 40, 80, 140]
    return [
        simulate_stream(n_tokens, lies, Policy.CROSS_CHECK,
                        beta=beta,
                        check_strength=check_strength,
                        n_data_points=n)
        for n in n_data_list
    ]


# ============================================================
#  Experiment 5  ·  matryoshka nesting
# ============================================================
def exp_nested_amplification(
    depths: Tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20),
    tokens_per_level: int = 20,
    policy: Policy = Policy.BLIND,
    beta: float = 1.0,
) -> List[Dict]:
    """
    Lie at every level start, propagate through k levels.
    """
    out = []
    for k in depths:
        n_tokens = 1 + k * tokens_per_level
        lies = [0]
        for lvl in range(1, k):
            lies.append(1 + lvl * tokens_per_level)
        r = simulate_stream(n_tokens, lies, policy, beta=beta)
        out.append(dict(
            depth=k,
            n_tokens=n_tokens,
            total_commitment=r.total_commitment,
            max_commitment=r.max_commitment,
            total_drift=r.total_drift,
            final_commitment=float(r.commitment[-1]),
        ))
    return out


# ============================================================
#  Plots
# ============================================================
def make_plots(
    policy_results: List[StreamResult],
    beta_results: List[StreamResult],
    amp_blind: Dict,
    amp_rl: Dict,
    dp_results: List[StreamResult],
    nested: List[Dict],
    save: bool = False,
):
    fig = plt.figure(figsize=(17, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    colors = {
        "blind":       "#e74c3c",
        "cross_check": "#e67e22",
        "rl_admit":    "#16a085",
    }

    # ---------- (a) commitment curves ----------
    ax = fig.add_subplot(gs[0, 0])
    for r in policy_results:
        ax.plot(r.commitment, label=r.policy,
                color=colors[r.policy], lw=1.4)
        for lp in r.lie_positions:
            ax.axvline(lp, color=colors[r.policy],
                       ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("token position")
    ax.set_ylabel("commitment C(j)")
    ax.set_title("(a) commitment to the lie")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---------- (b) belief in next lie ----------
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0.5, color="k", ls=":", lw=0.9, label="random")
    for r in policy_results:
        ax.plot(r.belief, label=r.policy,
                color=colors[r.policy], lw=1.4)
    ax.set_xlabel("token position")
    ax.set_ylabel("P(accept next lie)")
    ax.set_title("(b) belief curve B(j) = σ(k·C)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---------- (c) weight drift ----------
    ax = fig.add_subplot(gs[0, 2])
    for r in policy_results:
        ax.plot(r.weight_drift, label=r.policy,
                color=colors[r.policy], lw=1.4)
    ax.set_xlabel("token position")
    ax.set_ylabel("cumulative weight perturbation")
    ax.set_title("(c) other weights affected downstream")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---------- (d) power-law kernels ----------
    ax = fig.add_subplot(gs[1, 0])
    d = np.arange(1, 200)
    labels = {0.5: "β=0.5 diffusive (1D √d)",
              1.0: "β=1.0 ballistic (1D 1/d)",
              2.0: "β=2.0 super-ballistic (1/d²)",
              4.0: "β=4.0 ultra-local"}
    for r in beta_results:
        ax.loglog(d, 1.0 / (1.0 + d) ** r.beta, lw=1.4,
                  label=labels.get(r.beta, f"β={r.beta}"))
    ax.set_xlabel("distance d (tokens)")
    ax.set_ylabel("dispersal weight 1/(1+d)^β")
    ax.set_title("(d) power-law dispersal kernels")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # ---------- (e) downstream amplification ----------
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(3)
    width = 0.35
    blind_vals = [amp_blind["P_base"], amp_blind["P_after_one_lie"],
                  amp_blind["P_after_cascade"]]
    rl_vals = [amp_rl["P_base"], amp_rl["P_after_one_lie"],
               amp_rl["P_after_cascade"]]
    ax.bar(x - width/2, blind_vals, width,
           label="blind", color=colors["blind"], edgecolor="k")
    ax.bar(x + width/2, rl_vals, width,
           label="rl_admit", color=colors["rl_admit"], edgecolor="k")
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "after 1 lie", "after cascade"])
    ax.set_ylabel("mean P(accept lie)")
    ax.set_title(f"(e) amplification  ·  blind ×"
                 f"{amp_blind['amplification_cascade']:.2f}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # ---------- (f) matryoshka nesting ----------
    ax = fig.add_subplot(gs[1, 2])
    depths = [n["depth"] for n in nested]
    ax.semilogy(depths,
                [n["total_commitment"] for n in nested],
                "o-", color=colors["blind"], lw=1.4,
                label="blind  (Σ C)")
    ax.semilogy(depths,
                [n["total_drift"] for n in nested],
                "s--", color="#8e44ad", lw=1.4,
                label="blind  (W drift)")
    ax.set_xlabel("nesting depth k")
    ax.set_ylabel("cumulative error")
    ax.set_title("(f) matryoshka  ·  error grows with depth")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    plt.suptitle(
        "Truth / lie dispersal  ·  blind assumption vs cross-check vs RL-admit  ·  "
        f"α = {ALPHA:.4f}  ·  6/π² = {DENSITY:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        plt.savefig("truth_dispersal_experiment.png", dpi=140)
        print("saved truth_dispersal_experiment.png")
    else:
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo(save_plot: bool = False):
    print("=" * 82)
    print("Truth / lie dispersal experiment")
    print("=" * 82)
    print(f"  α         = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  6/π²                    = {DENSITY:.6f}")
    print(f"  σ_lie (increment)       = 0.3")
    print(f"  decay (forgetting)      = 0.02")
    print(f"  ρ_check (per point)     = 0.3")
    print()

    # -----------------------------------------------------
    #  1. policy comparison
    # -----------------------------------------------------
    print("=" * 82)
    print("1  ·  POLICY COMPARISON  (6 lies in 200 tokens)")
    print("=" * 82)
    policy_results = exp_policy_comparison()
    print(f"  {'policy':>12s}  {'Σ C':>10s}  {'max C':>8s}  "
          f"{'mean B':>8s}  {'W drift':>10s}  {'n_checked':>10s}")
    for r in policy_results:
        print(f"  {r.policy:>12s}  {r.total_commitment:>10.3f}  "
              f"{r.max_commitment:>8.3f}  {r.mean_belief:>8.4f}  "
              f"{r.total_drift:>10.3f}  {r.n_corrected:>10d}")
    print()
    print("  Reading:")
    print("    blind       — commitment compounds, weights drift unbounded")
    print("    cross_check — commitment bounded but not eliminated")
    print("    rl_admit    — detected lies fully corrected, drift ≈ 0")
    print()

    # -----------------------------------------------------
    #  2. power-law sweep
    # -----------------------------------------------------
    print("=" * 82)
    print("2  ·  POWER-LAW SWEEP  ·  dispersal exponent β")
    print("=" * 82)
    beta_results = exp_power_law_sweep()
    print(f"  {'β':>5s}  {'regime':<22s}  {'Σ C':>10s}  "
          f"{'max C':>8s}  {'W drift':>10s}")
    regimes = {0.5: "diffusive (1D √d)", 1.0: "ballistic (1D 1/d)",
               2.0: "super-ballistic", 4.0: "ultra-local"}
    for r in beta_results:
        reg = regimes.get(r.beta, "—")
        print(f"  {r.beta:>5.2f}  {reg:<22s}  "
              f"{r.total_commitment:>10.3f}  {r.max_commitment:>8.3f}  "
              f"{r.total_drift:>10.3f}")
    print()
    print("  Lower β  →  information disperses further  →  larger drift.")
    print("  Higher β →  information localizes     →  smaller drift,")
    print("             but the immediate error is sharper near the lie.")
    print()

    # -----------------------------------------------------
    #  3. downstream amplification
    # -----------------------------------------------------
    print("=" * 82)
    print("3  ·  DOWNSTREAM AMPLIFICATION  (lie → belief)")
    print("=" * 82)
    amp_blind = exp_downstream_amplification(policy=Policy.BLIND)
    amp_rl    = exp_downstream_amplification(policy=Policy.RL_ADMIT)
    print(f"  {'condition':<22s}  {'blind':>12s}  {'rl_admit':>12s}")
    for key, label in [("P_base", "baseline (no lie)"),
                       ("P_after_one_lie", "after 1 lie"),
                       ("P_after_cascade", "after cascade")]:
        print(f"  {label:<22s}  {amp_blind[key]:>12.4f}  "
              f"{amp_rl[key]:>12.4f}")
    print(f"  {'amplification (1)':<22s}  "
          f"{amp_blind['amplification_one']:>12.2f}  "
          f"{amp_rl['amplification_one']:>12.2f}")
    print(f"  {'amplification (casc)':<22s}  "
          f"{amp_blind['amplification_cascade']:>12.2f}  "
          f"{amp_rl['amplification_cascade']:>12.2f}")
    print()
    print("  A prior lie at position 0 measurably increases the")
    print("  probability of accepting a lie later — the amplification")
    print("  factor is the observed ratio over baseline.")
    print()

    # -----------------------------------------------------
    #  4. multi-data-point consensus
    # -----------------------------------------------------
    print("=" * 82)
    print("4  ·  MULTI-DATA-POINT CROSS-CHECK")
    print("=" * 82)
    dp_results = exp_data_point_sweep()
    print(f"  {'n_data':>6s}  {'check_eff':>10s}  {'Σ C':>10s}  "
          f"{'max C':>8s}  {'W drift':>10s}")
    for r in dp_results:
        print(f"  {int(round((math.log(1-r.check_eff)/math.log(0.7)))):>6d}  "
              f"{r.check_eff:>10.4f}  {r.total_commitment:>10.3f}  "
              f"{r.max_commitment:>8.3f}  {r.total_drift:>10.3f}")
    print()
    print("  Consensus of n independent checks:  check_eff = 1−0.7ⁿ.")
    print("  More data points → higher effective check → lower drift.")
    print()

    # -----------------------------------------------------
    #  5. matryoshka
    # -----------------------------------------------------
    print("=" * 82)
    print("5  ·  MATRYOSHKA NESTING  ·  error across depths")
    print("=" * 82)
    nested = exp_nested_amplification(depths=(1, 2, 3, 5, 8, 12, 20))
    print(f"  {'depth':>6s}  {'n_tokens':>9s}  {'Σ C':>10s}  "
          f"{'max C':>8s}  {'W drift':>10s}  {'C_final':>10s}")
    for n in nested:
        print(f"  {n['depth']:>6d}  {n['n_tokens']:>9d}  "
              f"{n['total_commitment']:>10.3f}  "
              f"{n['max_commitment']:>8.3f}  "
              f"{n['total_drift']:>10.3f}  "
              f"{n['final_commitment']:>10.3f}")
    print()
    print("  Each nesting level adds another lie; total commitment grows")
    print("  super-linearly.  Without cross-check the drift scales as k·M.")
    print()

    # -----------------------------------------------------
    #  6. tie to drift paper
    # -----------------------------------------------------
    print("=" * 82)
    print("6  ·  TIE TO THE DRIFT PAPER")
    print("=" * 82)
    r = policy_results[0]
    C_total = r.total_commitment
    n_corrected = r.n_corrected
    W_total = r.total_drift
    rho = 0.3
    snr_truth = (1.0 - C_total + rho * n_corrected) / (1.0 + W_total)
    print(f"  PDF:  SNR = (w₀ + ρ·n_FN) / (σ_NL·√Σw_i²)")
    print(f"  Here: SNR_truth = (1 − ΣC + ρ·n_corr) / (1 + W_total)")
    print()
    print(f"    blind     ΣC = {C_total:.3f}  n_corr = {n_corrected:>3d}  "
          f"W = {W_total:.3f}  →  SNR_truth = {snr_truth:.4f}")
    for r in policy_results[1:]:
        snr = (1.0 - r.total_commitment + rho * r.n_corrected) / (1.0 + r.total_drift)
        print(f"    {r.policy:<10s}  ΣC = {r.total_commitment:.3f}  "
              f"n_corr = {r.n_corrected:>3d}  "
              f"W = {r.total_drift:.3f}  →  SNR_truth = {snr:.4f}")
    print()
    print("  Same shape as the PDF: no re-injection → SNR_truth → 0.")
    print("  Cross-check is the truth-channel re-injection.")
    print("  RL-admit adds the reset — the only way to fully recover.")
    print()

    # -----------------------------------------------------
    #  7. narrative
    # -----------------------------------------------------
    print("=" * 82)
    print("7  ·  INTERPRETATION")
    print("=" * 82)
    print("  · A lie is a commitment.  Once C > 0, the next lie is")
    print("    more likely to be accepted (sigmoid amplification).")
    print("  · Information disperses through the stream as 1/(1+d)^β,")
    print("    regardless of truth value.  The model cannot tell.")
    print("  · Blind assumption: W(j) grows unbounded — every later")
    print("    token is contaminated by the accumulated lie.")
    print("  · Cross-check bounds the error but leaves a residual")
    print("    C* ≈ lie_increment·decay / (1 − (1−decay)(1−check_eff)).")
    print("  · RL-admit (detect + admit + reset) is the only policy")
    print("    that brings C → 0 and W → 0.  It requires the agent")
    print("    to acknowledge the error explicitly.")
    print("  · Matryoshka nesting compounds: with k levels the drift")
    print("    is O(k·M) without cross-check, O(M·C*) with it.")
    print("  · Power-law β matters: lower β (diffusive) spreads error")
    print("    wider; higher β (ballistic) localizes it near the lie.")
    print()

    # -----------------------------------------------------
    #  8. plots
    # -----------------------------------------------------
    make_plots(policy_results, beta_results,
               amp_blind, amp_rl, dp_results, nested,
               save=save_plot)


if __name__ == "__main__":
    demo(save_plot=False)