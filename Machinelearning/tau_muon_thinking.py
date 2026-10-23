#!/usr/bin/env python3
"""
alu_reasoning_chain.py
======================

ALU reasoning chain with muon/tau start/end prediction.

    Initial plan      = start peak of the Möbius smoothing elliptic curve
    i = 0 electron    = unchanged goal (invariant anchor)
    Final output      = end peak (μ/τ mirror of the start)

    Reasoning steps   = zeros of the muon–tau convolution (electrons)
    Each step fires an ALU ADD weighted by the probability flow
    Muon  (S_μ)       = start prediction  (plan-aligned)
    Tau   (S_τ)       = end prediction    (output-aligned)

    Symmetry rule (from Full_pipeline_neural_net_compression):
        symmetric  → direct bypass (UGC prime-real node)
        asymmetric → MOSB read + UGC discrete log

    Everything imported via explicit `from X import Y`.
"""

from __future__ import annotations
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
#  from X import Y   —  all modules used
# ============================================================
from Muon_tau_convolution_tick import (
    AtomicClockMuonTau,
    spin_grade,
    pi_e,
    ALPHA     as MT_ALPHA,
    A_STEP    as MT_ASTEP,
    DENSITY   as MT_DENSITY,
    PI        as MT_PI,
    E         as MT_E,
)

from planning_harness import (
    PlanningHarness,
    PlanReport,
    TokenLogRank,
    tokenize           as plan_tokenize,
    classify_token,
    log_rank_between,
    elliptic_projection,
    mobius_sieve       as plan_mobius,
    ALPHA_SYM,
    ALPHA_ASYM,
    K_DEFAULT          as PLAN_K,
)

from Full_pipeline_neural_net_compression_local_pc import (
    UGCNode,
    ReasoningALU,
    NodeDecision,
    UGCWeightLayer,
    SplitChannelCodec,
    MOSBReadOnlyBuffer,
    mobius_sieve       as ugc_mobius,
    harmonic_numbers,
    A_STEP             as UGC_ASTEP,
    ALPHA              as UGC_ALPHA,
    BLOCK,
    N_PERM,
)

from UTF_8_dzeta_dt import (
    ReadOnlyALU,
    ReadOnlyBuffer,
    ALPHA              as UTF8_ALPHA,
    A_STEP             as UTF8_ASTEP,
)


# ============================================================
#  Dataclasses
# ============================================================
@dataclass
class ReasoningStep:
    step_idx:    int
    k:           int
    t_k:         float
    S_mu:        float          # start-prediction channel
    S_tau:       float          # end-prediction channel (= −S_mu)
    K_sum:       float          # S_μ + c_0 + S_τ
    weight:      float          # probability flow through this tick
    alu_add:     float          # ReadOnlyALU ADD(muon_window, tau_window)
    muon_window: float          # integral of S_μ before the tick
    tau_window:  float          # integral of S_τ after the tick
    plan_token:  str
    phase:       str            # 'start' | 'goal' | 'end'
    is_goal:     bool


@dataclass
class ReasoningTrace:
    plan:             PlanReport
    clock_info:       Dict
    steps:            List[ReasoningStep]
    start_prediction: np.ndarray
    end_prediction:   np.ndarray
    goal_tokens:      List[str]
    goal_k:           int
    user_output:      str
    output_tokens:    List[str]
    elapsed_ms:       float
    geometric_data:   Dict = field(default_factory=dict)


# ============================================================
#  ALU reasoning chain
# ============================================================
class ALUReasoningChain:
    """
    Walks the electron ticks of the muon–tau clock.  Each tick is
    one reasoning step.  At each tick:

        muon_window = ∫ S_μ over the interval before the tick  (start)
        tau_window  = ∫ S_τ over the interval after the tick   (end)
        alu_add     = ReadOnlyALU.ADD(muon_window, tau_window)
        weight      = probability flow through this interval

    The start peak (first electron) is where the plan is anchored.
    The end peak (last electron) is where the user output surfaces.
    i = 0 (K_sum ≈ 0) is the invariant goal axis.
    """

    def __init__(self,
                 K_clock: int = 20,
                 K_alu:   int = 64,
                 J_C:     float = 1.0,
                 n_events:  int = 41,
                 n_samples: int = 2048,
                 plan_K:    int = PLAN_K):
        self.K_clock = K_clock
        self.K_alu   = K_alu
        self.clock   = AtomicClockMuonTau(
            K=K_clock, J_C=J_C,
            N_events=n_events, N_samples=n_samples)
        self.alu     = ReadOnlyALU(K=K_alu)
        self.buf     = ReadOnlyBuffer(K=K_alu)
        self.planner = PlanningHarness(K=plan_K)

    # --------------------------------------------------------
    def _find_peaks(self) -> Tuple[int, int, int]:
        """
        start peak   = first electron tick
        goal (i=0)   = middle electron tick (min |conv| in central third)
        end peak     = last electron tick
        """
        electron_idx = self.clock.electron_idx
        conv         = self.clock.conv
        if len(electron_idx) < 3:
            n = len(conv)
            return 0, n // 2, n - 1

        start_peak = int(electron_idx[0])
        end_peak   = int(electron_idx[-1])
        lo = len(electron_idx) // 3
        hi = 2 * len(electron_idx) // 3
        mid = electron_idx[lo:hi] if hi > lo else electron_idx
        goal_k = int(mid[np.argmin(np.abs(conv[mid]))])
        return start_peak, goal_k, end_peak

    # --------------------------------------------------------
    def _window_integral(self, signal: np.ndarray,
                        k_start: int, k_end: int) -> float:
        lo = max(0, min(k_start, len(signal) - 1))
        hi = max(lo + 1, min(k_end + 1, len(signal)))
        seg = signal[lo:hi]
        return float(np.sum(seg)) if seg.size else 0.0

    # --------------------------------------------------------
    def _probability_weights(self) -> Dict[int, float]:
        """
        Weight each electron tick by the RMS of the muon channel in
        the interval between this tick and the next.  Normalised to 1.
        """
        electron_idx = self.clock.electron_idx
        S_mu         = self.clock.S_mu
        if len(electron_idx) == 0:
            return {}
        if len(electron_idx) == 1:
            return {int(electron_idx[0]): 1.0}

        weights: Dict[int, float] = {}
        total = 0.0
        for i in range(len(electron_idx) - 1):
            k_start = int(electron_idx[i])
            k_end   = int(electron_idx[i + 1])
            seg = S_mu[k_start:k_end + 1]
            rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
            weights[k_start] = rms
            total += rms

        # last tick → interval to end of signal
        last_k = int(electron_idx[-1])
        seg = S_mu[last_k:]
        rms = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
        weights[last_k] = rms
        total += rms

        if total > 1e-12:
            for k in weights:
                weights[k] /= total
        return weights

    # --------------------------------------------------------
    def run(self, requirement: str, code: str) -> ReasoningTrace:
        t0 = time.perf_counter()

        # ---- 1. plan = start peak ----
        plan_rep = self.planner.plan(requirement, code, top_k=12)

        # ---- 2. clock peaks + weights ----
        start_peak, goal_k, end_peak = self._find_peaks()
        weights  = self._probability_weights()
        electron = list(self.clock.electron_idx)

        # ---- 3. walk the ticks ----
        steps: List[ReasoningStep] = []
        n_elec = len(electron)
        for i, k_raw in enumerate(electron):
            k = int(k_raw)

            S_mu_k  = float(self.clock.S_mu[k])
            S_tau_k = float(self.clock.S_tau[k])       # = −S_mu_k
            c_0     = 0.0                              # smoothed electron
            K_sum_k = S_mu_k + c_0 + S_tau_k

            k_prev = int(electron[i - 1]) if i > 0       else 0
            k_next = int(electron[i + 1]) if i < n_elec - 1 else len(self.clock.t) - 1

            # start window = muon channel BEFORE this tick
            muon_window = self._window_integral(self.clock.S_mu, k_prev, k)
            # end window = tau channel AFTER this tick
            tau_window  = self._window_integral(self.clock.S_tau, k, k_next)

            # ALU ADD (reconstruction-coupled, not plain +)
            xb = struct.pack(">dd", muon_window, float(k))
            yb = struct.pack(">dd", tau_window,  float(k))
            cid_x = self.buf.append(xb)
            cid_y = self.buf.append(yb)
            add_z = float(self.alu.dispatch(self.buf, cid_x, cid_y, "ADD")["z"])

            # phase
            if   k < goal_k: phase = "start"
            elif k == goal_k: phase = "goal"
            else:             phase = "end"

            # attach a plan token to each step
            plan_tok = ""
            if plan_rep.needed:
                idx = min(i, len(plan_rep.needed) - 1)
                plan_tok = plan_rep.needed[idx].token

            steps.append(ReasoningStep(
                step_idx=i, k=k,
                t_k=float(self.clock.t[k]) if k < len(self.clock.t) else 0.0,
                S_mu=S_mu_k, S_tau=S_tau_k, K_sum=K_sum_k,
                weight=float(weights.get(k, 0.0)),
                alu_add=add_z,
                muon_window=muon_window,
                tau_window=tau_window,
                plan_token=plan_tok,
                phase=phase,
                is_goal=(phase == "goal"),
            ))

        # ---- 4. split start / end predictions ----
        start_steps = [s for s in steps if s.phase == "start"]
        goal_step   = next((s for s in steps if s.is_goal), None)
        end_steps   = [s for s in steps if s.phase == "end"]

        start_pred = np.array(
            [s.S_mu  for s in start_steps]
            + ([goal_step.S_mu] if goal_step else []))
        end_pred = np.array(
            ([goal_step.S_tau] if goal_step else [])
            + [s.S_tau for s in end_steps])

        # ---- 5. output tokens (end peak) ----
        output_tokens = self._build_output(plan_rep, steps, goal_step)
        user_output   = " → ".join(output_tokens[:8]) if output_tokens else "(none)"

        # ---- 6. goal tokens (i = 0, unchanged) ----
        goal_tokens = [goal_step.plan_token] if goal_step and goal_step.plan_token else []
        if goal_step:
            goal_tokens.append(f"k={goal_k}")

        elapsed = (time.perf_counter() - t0) * 1e3

        return ReasoningTrace(
            plan=plan_rep,
            clock_info=dict(
                K=self.K_clock,
                T=float(self.clock.T),
                n_electrons=int(self.clock.n_electrons),
                tick_rate=float(self.clock.tick_rate()),
                parity_ratio=float(self.clock.parity_ratio),
                norm_ee=float(self.clock.norm_ee),
                norm_eo=float(self.clock.norm_eo),
                start_peak=int(start_peak),
                goal_k=int(goal_k),
                end_peak=int(end_peak),
                A_STEP=float(MT_ASTEP),
                alpha=float(MT_ALPHA),
            ),
            steps=steps,
            start_prediction=start_pred,
            end_prediction=end_pred,
            goal_tokens=goal_tokens,
            goal_k=int(goal_k),
            user_output=user_output,
            output_tokens=output_tokens,
            elapsed_ms=elapsed,
            geometric_data=dict(
                norm_start=float(np.linalg.norm(start_pred)),
                norm_end=float(np.linalg.norm(end_pred)),
                n_start=len(start_steps),
                n_end=len(end_steps),
                has_goal=goal_step is not None,
                mean_weight=float(np.mean([s.weight for s in steps])) if steps else 0.0,
                sum_alu=float(np.sum([s.alu_add for s in steps])) if steps else 0.0,
            ),
        )

    # --------------------------------------------------------
    def _build_output(self, plan_rep: PlanReport,
                      steps: List[ReasoningStep],
                      goal_step: Optional[ReasoningStep]) -> List[str]:
        """
        End-peak output:
          1. goal token anchors the list (i = 0)
          2. highest-weight end-phase tokens are next
          3. plan tokens (start) fill the rest
        """
        output: List[str] = []

        # 1. anchor with goal token
        if goal_step and goal_step.plan_token:
            output.append(goal_step.plan_token)

        # 2. highest-weight end-phase tokens (tau side)
        end_ranked = sorted(
            [s for s in steps if s.phase == "end" and s.plan_token],
            key=lambda s: -s.weight)[:6]
        for s in end_ranked:
            if s.plan_token not in output:
                output.append(s.plan_token)

        # 3. fall back / fill from plan (muon side)
        for r in plan_rep.needed[:10]:
            if r.token not in output:
                output.append(r.token)

        return output[:8]

    # --------------------------------------------------------
    def report(self, trace: ReasoningTrace) -> str:
        L: List[str] = []
        L.append("=" * 80)
        L.append("ALU reasoning chain  ·  μ/τ start→end prediction")
        L.append("=" * 80)
        L.append(f"  α       = 1/(π − e)  = {MT_ALPHA:.6f}")
        L.append(f"  A_STEP               = {MT_ASTEP:.6f}")
        L.append(f"  6/π²                 = {MT_DENSITY:.6f}")
        L.append("")

        ci = trace.clock_info
        L.append("--- clock  ·  AtomicClockMuonTau ---")
        L.append(f"  K                    = {ci['K']}")
        L.append(f"  T                    = {ci['T']:.6f}")
        L.append(f"  n_electrons          = {ci['n_electrons']}")
        L.append(f"  tick_rate            = {ci['tick_rate']:.4f}")
        L.append(f"  parity ratio         = {ci['parity_ratio']:.4f}  "
                 f"(‖G_ee‖={ci['norm_ee']:.4f}, ‖G_eo‖={ci['norm_eo']:.4f})")
        L.append(f"  start peak k         = {ci['start_peak']}")
        L.append(f"  goal k   (i = 0)     = {ci['goal_k']}")
        L.append(f"  end peak k           = {ci['end_peak']}")
        L.append("")

        p = trace.plan
        L.append("--- plan (start peak) ---")
        L.append(f"  Σ Z (log-rank)       = {p.Z_sum:+.4f}")
        L.append(f"  Σ L (weighted)       = {p.L_sum:+.4f}")
        L.append(f"  mean Π               = {p.Pi_mean:.4f}")
        L.append(f"  needed tokens        = {[r.token for r in p.needed[:8]]}")
        L.append("")

        L.append("--- reasoning steps  ·  ALU ADD chain ---")
        hdr = (f"  {'#':>3}  {'k':>4}  {'phase':>5}  "
               f"{'S_μ':>10}  {'S_τ':>10}  {'K_sum':>10}  "
               f"{'weight':>7}  {'ADD':>11}  token")
        L.append(hdr)
        for s in trace.steps:
            flag = " ← goal" if s.is_goal else ""
            L.append(
                f"  {s.step_idx:>3d}  {s.k:>4d}  {s.phase:>5s}  "
                f"{s.S_mu:>+10.4f}  {s.S_tau:>+10.4f}  "
                f"{s.K_sum:>+10.6f}  {s.weight:>7.4f}  "
                f"{s.alu_add:>+11.4f}  {s.plan_token[:14]:<14s}{flag}")
        L.append("")

        L.append("--- start prediction  (μ channel, plan-aligned) ---")
        sp = trace.start_prediction
        L.append(f"  |start|              = {np.linalg.norm(sp):.4f}")
        L.append(f"  mean                 = {np.mean(sp):+.6f}" if sp.size else "  mean = —")
        L.append(f"  n steps              = {sp.size}")
        L.append("")

        L.append("--- end prediction  (τ channel, output-aligned) ---")
        ep = trace.end_prediction
        L.append(f"  |end|                = {np.linalg.norm(ep):.4f}")
        L.append(f"  mean                 = {np.mean(ep):+.6f}" if ep.size else "  mean = —")
        L.append(f"  n steps              = {ep.size}")
        L.append("")

        L.append("--- i = 0 goal  (unchanged real part) ---")
        L.append(f"  goal tokens          = {trace.goal_tokens}")
        L.append(f"  goal k               = {trace.goal_k}")
        L.append("")

        L.append("--- user output  (end peak of the elliptic curve) ---")
        L.append(f"  {trace.user_output}")
        L.append("")
        L.append(f"  elapsed              = {trace.elapsed_ms:.2f} ms")
        return "\n".join(L)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("ALU reasoning chain  ·  plan = start peak, output = end peak")
    print("=" * 80)
    print(f"  α      = 1/(π − e)    = {MT_ALPHA:.6f}")
    print(f"  A_STEP                = {MT_ASTEP:.6f}")
    print(f"  6/π²                  = {MT_DENSITY:.6f}")
    print()

    requirement = (
        "Add multiply and subtract functions to the calculator. "
        "Both should take two arguments a and b. "
        "Keep the existing add function working."
    )
    code = """
def add(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value

def main():
    c = Calculator()
    c.add(5)
    print(c.value)
"""

    chain = ALUReasoningChain(
        K_clock=20, K_alu=64, J_C=1.0,
        n_events=41, n_samples=2048)
    trace = chain.run(requirement, code)
    print(chain.report(trace))
    print()

    # ---- compact geometric summary ----
    g = trace.geometric_data
    print("=" * 80)
    print("Geometric view")
    print("=" * 80)
    print(f"  start peak  → |start| = {g['norm_start']:.4f}   "
          f"({g['n_start']} steps)")
    print(f"  end   peak  → |end|   = {g['norm_end']:.4f}   "
          f"({g['n_end']} steps)")
    print(f"  mean weight           = {g['mean_weight']:.4f}")
    print(f"  Σ ALU ADD             = {g['sum_alu']:+.4f}")
    print(f"  goal locked (i=0)     = {g['has_goal']}")
    print()
    print(f"  User output:  {trace.user_output}")
    print()

    # ---- complexity ----
    print("--- complexity ---")
    print("  planning log-rank        O(K log K)")
    print("  clock build (Atomic)     O(N_events · N_samples)")
    print("  probability weights      O(N_samples)")
    print("  ALU ADD per tick         O(K_alu²)  (reconstruction)")
    print("  trace walk               O(n_electrons)")
    print("  ─────────────────────────────────────")
    print("  total reasoning          O(N_samples + n_electrons · K_alu²)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()