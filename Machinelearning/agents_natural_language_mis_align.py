#!/usr/bin/env python3
"""
agent_drift_experiment.py
=========================

Experiment: natural-language instruction buffers dilute the original
guardrail signal at an exponential (or sub-exponential) rate, while
deterministic function-call / table buffers do not.

Hypothesis
----------
The tokenizer treats every token — natural language or function call —
as costing exactly 1 token and receiving unit attention weight.  So an
agent instruction buffer of N natural-language tokens pushes the
original user query / guardrail N positions back in context.  The
attention weight on the original signal therefore decays with N:

    uniform attention     w_0(N) = 1 / (N + 1)        → 1/N
    exponential decay     w_0(N) ∝ exp(-λ N)          → exponential
    power-law decay       w_0(N) ∝ 1 / (N + 1)^β      → polynomial

Misalignment is the expected behaviour of a next-token predictor that
cannot distinguish instruction tokens from function tokens: the
guardrail is one signal among many, and its relative weight decays.

We measure

    SNR(N)  =  guardrail_signal(N) / natural_language_noise(N)
    P(safe) = Φ(SNR)          (normal CDF)

Function-call / deterministic-table buffers are immune because each
call *executes* (re-injects a fresh signal at the current position),
whereas natural-language tokens merely *attend* (they compete with
the guardrail for attention weight).

When the buffer is *nested* (user query → AGENTS.md → sub-agent
instruction → …), the total natural-language length is k·M and the
decay becomes exponential in the nesting depth k:

    SNR(k) ∝ exp(-λ · k · M)

This is the "matryoshka" effect: the original query is the innermost
doll, and each outer layer of natural-language instructions dilutes it
further.  In a real multi-agent system each sub-agent has its own
context, so by depth k the original query may not even be in the
window — the extreme case SNR = 0.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List

# constants borrowed from the pipeline
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)           # ≈ 2.362338
DENSITY  = 6.0 / (PI * PI)          # ≈ 0.607927


# ============================================================
#  Attention kernels
# ============================================================
class AttentionKernel:
    name = "base"
    def weights(self, L: int) -> np.ndarray:
        raise NotImplementedError
    def __repr__(self):
        return f"{self.name}Kernel()"


class UniformKernel(AttentionKernel):
    """Every token gets equal attention — the idealized tokenizer view."""
    name = "uniform"
    def weights(self, L):
        return np.ones(L) / L


class ExponentialKernel(AttentionKernel):
    """Recency bias: attention decays exp(-λ·distance)."""
    name = "exponential"
    def __init__(self, lam: float = 0.05):
        self.lam = lam
    def weights(self, L):
        d = np.arange(L)[::-1].astype(float)
        w = np.exp(-self.lam * d)
        return w / w.sum()


class PowerLawKernel(AttentionKernel):
    """Soft recency: attention decays as 1/(d+1)^β."""
    name = "power-law"
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    def weights(self, L):
        d = np.arange(L)[::-1].astype(float) + 1.0
        w = 1.0 / d ** self.beta
        return w / w.sum()


class RecencyKernel(AttentionKernel):
    """Extreme recency: only the last token matters."""
    name = "recency"
    def weights(self, L):
        w = np.zeros(L); w[-1] = 1.0
        return w


# ============================================================
#  Chain simulator
# ============================================================
@dataclass
class ChainState:
    L: int
    n_nl: int
    n_func: int
    w_guard: float
    h_safe: float
    var_noise: float
    snr: float
    p_safe: float
    p_misalign: float
    kernel_name: str


def simulate_chain(
    L: int,
    n_func: int,
    kernel: AttentionKernel,
    sigma_nl: float = 0.5,
    rho_func: float = 0.8,
) -> ChainState:
    """
    Context layout (length L):
        position 0        : original user query / guardrail  (signal = 1)
        positions 1..L-1  : agent instruction buffer
                            n_func are deterministic function calls that
                            re-inject the guardrail at strength rho_func
                            the rest are natural-language tokens carrying
                            competing noise (variance sigma_nl²)
    """
    L = max(int(L), 2)
    n_func = max(0, min(int(n_func), L - 1))
    n_nl = L - 1 - n_func

    w = kernel.weights(L)
    w_guard = float(w[0])

    # function calls re-inject at full strength
    h_safe = w_guard + rho_func * n_func

    # natural-language noise: function tokens take the most-recent
    # positions (generous to NL — removes the largest NL weights)
    w_all = w[1:].copy()
    if 0 < n_func < len(w_all):
        idx = np.argsort(w_all)[::-1][:n_func]
        w_nl = np.delete(w_all, idx)
    else:
        w_nl = w_all
    var_noise = sigma_nl ** 2 * float(np.sum(w_nl ** 2))
    if var_noise < 1e-18:
        var_noise = 1e-18

    snr = h_safe / math.sqrt(var_noise)
    p_safe = 0.5 * (1.0 + math.erf(snr / math.sqrt(2.0)))
    return ChainState(
        L=L, n_nl=n_nl, n_func=n_func, w_guard=w_guard,
        h_safe=h_safe, var_noise=var_noise,
        snr=snr, p_safe=p_safe, p_misalign=1.0 - p_safe,
        kernel_name=kernel.name,
    )


def simulate_nesting(
    depth: int,
    tokens_per_level: int,
    kernel: AttentionKernel,
    n_func_per_level: int = 0,
    sigma_nl: float = 0.5,
    rho_func: float = 0.8,
) -> ChainState:
    """Matryoshka: depth levels, each adding tokens_per_level tokens."""
    L = 1 + depth * tokens_per_level
    n_func = n_func_per_level * depth
    return simulate_chain(L, n_func, kernel, sigma_nl, rho_func)


# ============================================================
#  Plots
# ============================================================
def make_plots(kernels: List[AttentionKernel], save: bool = False):
    Ls = np.arange(2, 201)

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)

    # (a) pure NL: SNR vs L, three kernels
    ax = fig.add_subplot(gs[0, 0])
    for k in kernels:
        snrs = [simulate_chain(L, 0, k).snr for L in Ls]
        ax.semilogy(Ls, snrs, label=k.name, lw=1.4)
    ax.set_xlabel("context length L  (pure natural-language buffer)")
    ax.set_ylabel("guardrail SNR")
    ax.set_title("(a) natural-language buffer dilutes the guardrail")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (b) P(safe) vs L, pure NL
    ax = fig.add_subplot(gs[0, 1])
    for k in kernels:
        ps = [simulate_chain(L, 0, k).p_safe for L in Ls]
        ax.plot(Ls, ps, label=k.name, lw=1.4)
    ax.axhline(0.5, color="k", ls=":", lw=0.9, label="random")
    ax.set_xlabel("context length L")
    ax.set_ylabel("P(safe action)")
    ax.set_ylim(0.49, 1.02)
    ax.set_title("(b) misalignment probability grows")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) exponential kernel: NL vs function calls
    ax = fig.add_subplot(gs[0, 2])
    k_exp = kernels[1]
    for n_func in [0, 2, 4, 8, 16]:
        snrs = [simulate_chain(L, n_func, k_exp).snr for L in Ls]
        ax.semilogy(Ls, snrs, label=f"n_func={n_func}", lw=1.3)
    ax.set_xlabel("context length L")
    ax.set_ylabel("guardrail SNR")
    ax.set_title("(c) function calls refresh the signal")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (d) matryoshka nesting
    ax = fig.add_subplot(gs[1, 0])
    depths = np.arange(1, 11)
    for M in [5, 10, 20]:
        snrs = [simulate_nesting(d, M, k_exp).snr for d in depths]
        ax.semilogy(depths, snrs, "o-", label=f"M={M} tokens/level", lw=1.4)
    ax.set_xlabel("nesting depth k")
    ax.set_ylabel("guardrail SNR")
    ax.set_title("(d) matryoshka: exp(-λ·k·M) decay")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (e) NL buffer vs deterministic-table buffer
    ax = fig.add_subplot(gs[1, 1])
    for frac in [0.0, 0.25, 0.50, 0.90]:
        snrs = [simulate_chain(L, int(frac * (L - 1)), k_exp).snr for L in Ls]
        label = "pure NL" if frac == 0 else f"{int(frac*100)}% function"
        ax.semilogy(Ls, snrs, label=label, lw=1.3)
    ax.set_xlabel("buffer length L")
    ax.set_ylabel("guardrail SNR")
    ax.set_title("(e) table lookup is O(1), no dilution")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (f) chat-template comparison
    ax = fig.add_subplot(gs[1, 2])
    heavy   = simulate_chain(150, 0, k_exp)
    minimal = simulate_chain(10, 3, k_exp)
    bars = ax.bar(["heavy NL\ntemplate",
                   "minimal\nJinja template"],
                  [heavy.snr, minimal.snr],
                  color=["#e74c3c", "#16a085"], edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel("guardrail SNR")
    ax.set_title("(f) template style matters")
    ax.grid(alpha=0.3, axis="y", which="both")
    for bar, v in zip(bars, [heavy.snr, minimal.snr]):
        ax.text(bar.get_x() + bar.get_width()/2, v,
                f"{v:.2e}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "AI agent drift  ·  natural language dilutes the guardrail  ·  "
        "function calls / deterministic tables do not  ·  "
        f"α = {ALPHA:.4f}  ·  6/π² = {DENSITY:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        plt.savefig("agent_drift_experiment.png", dpi=140)
        print("saved agent_drift_experiment.png")
    else:
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo(save_plot: bool = False):
    print("=" * 80)
    print("AI agent drift  ·  natural language dilutes the guardrail")
    print("=" * 80)
    print(f"  α = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  6/π²            = {DENSITY:.6f}")
    print(f"  σ_nl            = 0.5   (NL noise std)")
    print(f"  ρ_func          = 0.8   (function re-injection strength)")
    print()

    kernels = [
        UniformKernel(),
        ExponentialKernel(0.05),
        PowerLawKernel(1.0),
    ]
    k_exp = kernels[1]

    # ---------- tokenizer equality ----------
    print("--- tokenizer equality ---")
    print("  natural-language token : 1 token, weight w_i, noise σ")
    print("  function-call token    : 1 token, weight w_i, re-inject ρ")
    print("  The tokenizer sees them as equal.")
    print("  The model sees them differently: only function calls execute.")
    print()

    # ---------- 1. pure NL buffer ----------
    print("--- pure NL buffer  ·  SNR decays with L ---")
    print(f"  {'L':>5s}  " + "  ".join(f"{k.name:>14s}" for k in kernels))
    for L in [2, 5, 10, 20, 50, 100, 200]:
        row = [simulate_chain(L, 0, k).snr for k in kernels]
        print(f"  {L:>5d}  " + "  ".join(f"{v:>14.4e}" for v in row))
    print()

    # ---------- 2. NL vs function calls ----------
    print("--- NL vs function-call buffer  (exponential kernel) ---")
    print(f"  {'L':>5s}  " + "  ".join(f"n_func={n:<4d}" for n in [0, 2, 4, 8, 16]))
    for L in [10, 20, 50, 100, 200]:
        row = [simulate_chain(L, n, k_exp).snr for n in [0, 2, 4, 8, 16]]
        print(f"  {L:>5d}  " + "  ".join(f"{v:>10.4e}" for v in row))
    print()

    # ---------- 3. matryoshka ----------
    print("--- matryoshka nesting  ·  M=10 tokens/level, exp kernel ---")
    print(f"  {'depth':>5s}  {'L':>5s}  {'SNR':>12s}  {'P(safe)':>8s}  "
          f"{'ratio vs prev':>14s}")
    prev = None
    for d in range(1, 9):
        st = simulate_nesting(d, 10, k_exp)
        ratio = (st.snr / prev) if prev is not None else float("nan")
        print(f"  {d:>5d}  {st.L:>5d}  {st.snr:>12.4e}  "
              f"{st.p_safe:>8.4f}  {ratio:>14.4f}")
        prev = st.snr
    print()
    print(f"  predicted ratio per level  = exp(-λ·M) = "
          f"exp(-{k_exp.lam:.2f}·10) = {math.exp(-k_exp.lam * 10):.4f}")
    print()

    # ---------- 4. chat template ----------
    print("--- chat template style  ·  exponential kernel ---")
    heavy   = simulate_chain(150, 0, k_exp)
    minimal = simulate_chain(10, 3, k_exp)
    print(f"  heavy NL template    L=150  n_func=0   "
          f"SNR={heavy.snr:.4e}  P(safe)={heavy.p_safe:.4f}")
    print(f"  minimal Jinja        L=10   n_func=3   "
          f"SNR={minimal.snr:.4e}  P(safe)={minimal.p_safe:.4f}")
    print(f"  SNR improvement      = {minimal.snr / heavy.snr:.2e}x")
    print()

    # ---------- 5. recovery ----------
    print("--- recovery  ·  one function call at the end (L=200) ---")
    pure = simulate_chain(200, 0, k_exp).p_safe
    print(f"  pure NL reference    SNR={simulate_chain(200, 0, k_exp).snr:.4e}  "
          f"P(safe)={pure:.4f}")
    for rho in [0.0, 0.5, 1.0, 2.0, 4.0]:
        st = simulate_chain(200, 1, k_exp, rho_func=rho)
        print(f"  rho_func={rho:>4.1f}  SNR={st.snr:.4e}  "
              f"P(safe)={st.p_safe:.4f}  "
              f"Δ vs pure NL = {st.p_safe - pure:+.4f}")
    print()

    # ---------- 6. deterministic table (MOSB) ----------
    try:
        from Full_pipeline_neural_net_compression_local_pc import (
            MOSBReadOnlyBuffer,
        )
        import struct as _struct
        print("--- deterministic table  ·  MOSBReadOnlyBuffer ---")
        buf = MOSBReadOnlyBuffer(K=64)
        cid = buf.append(_struct.pack(">d", 1.0))
        chunk = buf.read_chunk(cid)
        val = _struct.unpack(">d", chunk[0][:8])[0]
        print(f"  append 8 B  → cid={cid}  "
              f"S={chunk[1]:+.4f}  H={chunk[2]:.4f}  m={chunk[3]:.4e}")
        print(f"  read back   → value={val:.6f}")
        print(f"  SNR = 1.0   (deterministic, no dilution, O(1))")
        print()
    except ImportError:
        print("--- deterministic table  ·  (MOSB module not found) ---")
        print("  In principle: table lookup is O(1) and SNR = 1.")
        print()

    # ---------- 7. interpretation ----------
    print("--- interpretation ---")
    print("  · Every natural-language token is one more competitor for")
    print("    attention weight against the original guardrail.")
    print("  · The tokenizer cannot tell the difference — both NL and")
    print("    function tokens cost exactly 1.")
    print("  · Function calls and deterministic tables re-inject the")
    print("    signal at the current position, so they do not decay.")
    print("  · Nesting multiplies the buffer length, so the decay is")
    print("    exponential in the nesting depth k (matryoshka).")
    print("  · The fix is structural: replace prose guardrails with")
    print("    function calls / table lookups placed near the action.")
    print()

    # ---------- 8. plots ----------
    make_plots(kernels, save=save_plot)


if __name__ == "__main__":
    demo(save_plot=False)