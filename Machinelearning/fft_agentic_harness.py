#!/usr/bin/env python3
"""
agentic_fft_mobius_harness.py
=============================

Deterministic agentic harness whose history lives in the FFT Möbius
memory from memory-okloglogk-mobius.py.

Storage model
-------------
The |Ci| array of length 2K+1 has two kinds of contributions:

    symmetric tokens   →  add to both K+i and K-i  (balanced)
    asymmetric tokens  →  add only to K+i           (unbalanced)

The memory's split_symmetric_asymmetric classifies each index pair
| c[K+i] | vs | c[K-i] |:
    equal magnitude → symmetric   → c_sym
    unequal         → asymmetric  → c_asym

Each part is then compressed by FFT (top-M coefficients) and
decompressed; the symmetric part is smoothed by the exponential
kernel from memory-okloglogk-mobius.py before recombination.

Loop
----
    while not complete:
        next action  (symmetric or asymmetric)
        run action  →  tokens + is_symmetric flag
        append tokens to the corresponding pool

Termination
-----------
    complete() when  len(sym_tokens) + len(asym_tokens) ≥ target_entries

Imports from the sibling .py files
----------------------------------
    memory-okloglogk-mobius.py
        FFTMobiusMemory, CompressedMemory, FFTPart,
        fft_compress_part, fft_decompress_part,
        split_symmetric_asymmetric, conv_exp_kernel, mobius_sieve
    deterministic-asymmetric-linear-regression-OKlogK.py
        MobiusMLRouter, classify
    quick-reader-analyzer.py
        StrikeFlagReader, tokens_to_k_sum
    planning_harness.py
        PlanningHarness
"""

from __future__ import annotations
import hashlib
import importlib.util
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Load sibling modules  (hyphenated names → importlib)
# ============================================================
HERE = pathlib.Path(__file__).resolve().parent


def _load(mod_name: str, filename: str):
    path = HERE / filename
    if not path.exists():
        raise FileNotFoundError(f"expected {filename} next to "
                                f"{pathlib.Path(__file__).name}")
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fft_mod    = _load("fft_mobius",    "memory-okloglogk-mobius.py")
router_mod = _load("mobius_router",
                   "deterministic-asymmetric-linear-regression-OKlogK.py")
reader_mod = _load("quick_reader",  "quick-reader-analyzer.py")
plan_mod   = _load("planning",      "planning_harness.py")

FFTMobiusMemory          = fft_mod.FFTMobiusMemory
CompressedMemory         = fft_mod.CompressedMemory
FFTPart                  = fft_mod.FFTPart
fft_compress_part        = fft_mod.fft_compress_part
fft_decompress_part      = fft_mod.fft_decompress_part
split_symmetric_asymmetric = fft_mod.split_symmetric_asymmetric
conv_exp_kernel          = fft_mod.conv_exp_kernel
mobius_sieve             = fft_mod.mobius_sieve

MobiusMLRouter = router_mod.MobiusMLRouter
classify_query = router_mod.classify

StrikeFlagReader = reader_mod.StrikeFlagReader
tokens_to_k_sum  = reader_mod.tokens_to_k_sum

PlanningHarness = plan_mod.PlanningHarness


# ============================================================
#  |Ci| construction with explicit symmetric / asymmetric pool
# ============================================================
def _hash_index(token, K: int) -> int:
    h = int(hashlib.md5(str(token).encode()).hexdigest()[:8], 16)
    return (h % K) + 1


def build_mixed_ci(sym_tokens: List,
                   asym_tokens: List,
                   K: int,
                   mu: List[int]) -> np.ndarray:
    """
    Symmetric tokens contribute to both ±i  (balanced |Ci| entry).
    Asymmetric tokens contribute only to +i (unbalanced |Ci| entry).
    Square-free filter: an index with μ=0 is skipped.
    """
    c = np.zeros(2 * K + 1, dtype=float)
    for t in sym_tokens:
        i = _hash_index(t, K)
        if 0 < i <= K and mu[i] != 0:
            c[K + i] += 1.0
            c[K - i] += 1.0
    for t in asym_tokens:
        i = _hash_index(t, K)
        if 0 < i <= K and mu[i] != 0:
            c[K + i] += 1.0     # only the positive side
    return c


# ============================================================
#  Step record
# ============================================================
@dataclass
class FFTStep:
    step: int
    action: str
    is_symmetric: bool
    n_tokens: int
    value: float
    detail: Dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


# ============================================================
#  The agentic harness
# ============================================================
class AgenticFFTHarness:
    """
    Deterministic agent whose history is stored in the FFT Möbius
    memory.  Symmetric and asymmetric actions feed two separate
    token pools, and the memory's split function classifies the
    resulting |Ci| entries automatically.
    """

    def __init__(self,
                 K: int,
                 task: str,
                 target_entries: int = 128,
                 M_sym: Optional[int] = None,
                 M_asym: Optional[int] = None):
        self.K = K
        self.task = task
        self.target_entries = int(target_entries)
        self.M_sym = M_sym
        self.M_asym = M_asym

        self.memory = FFTMobiusMemory(K)
        self.mu = self.memory.mu

        # token pools
        self.sym_tokens: List = []
        self.asym_tokens: List = []

        # history
        self.history: List[FFTStep] = []
        self.completed: bool = False

        # lazy sub-modules
        self._router: Optional[MobiusMLRouter] = None
        self._reader: Optional[StrikeFlagReader] = None
        self._planner: Optional[PlanningHarness] = None

        # results after compression
        self.c_orig: Optional[np.ndarray] = None
        self.c_recon: Optional[np.ndarray] = None
        self.part_sym: Optional[FFTPart] = None
        self.part_asym: Optional[FFTPart] = None
        self.n_sym: int = 0
        self.n_asym: int = 0
        self.rel_err: float = 0.0
        self.fingerprint: str = ""

    # ---------- lazy sub-module getters ----------
    def router(self) -> MobiusMLRouter:
        if self._router is None:
            self._router = MobiusMLRouter(K=64)
        return self._router

    def reader(self) -> StrikeFlagReader:
        if self._reader is None:
            self._reader = StrikeFlagReader(strike_threshold=2,
                                            entropy_tolerance=0.12)
            x = np.arange(64)
            ref = np.abs(np.sin(x * 0.05) + 0.3 * np.cos(x * 0.13))
            self._reader.set_reference(ref)
        return self._reader

    def planner(self) -> PlanningHarness:
        if self._planner is None:
            self._planner = PlanningHarness(K=self.K)
        return self._planner

    # ---------- progress ----------
    def total_tokens(self) -> int:
        return len(self.sym_tokens) + len(self.asym_tokens)

    def complete(self) -> bool:
        return self.total_tokens() >= self.target_entries

    # ---------- symmetric actions ----------
    def act_classify(self, step: int) -> Tuple[float, List, Dict]:
        q = f"{self.task}  step {step}"
        cls = classify_query(q)
        tokens = [f"cls_{cls.kind}", f"det_{cls.deterministic}"]
        return float(cls.projection_gap), tokens, {
            "kind": cls.kind, "deterministic": cls.deterministic}

    def act_arith(self, step: int) -> Tuple[float, List, Dict]:
        q = f"({step + 1}) * {self.K} - {step * 3}"
        a = self.router().ask(q)
        tokens = [f"arith_{q}", f"ans_{a.answer}"]
        return float(a.elapsed_ms), tokens, {
            "query": q, "answer": a.answer, "alpha": a.alpha_used}

    def act_plan(self, step: int) -> Tuple[float, List, Dict]:
        code = "def add(a,b):\n    return a+b\n"
        req  = f"{self.task}; step {step}"
        rep  = self.planner().plan(req, code, top_k=3)
        tokens = [f"plan_nreq_{rep.n_req_tokens}",
                  f"plan_Z_{round(rep.Z_sum, 3)}",
                  f"plan_L_{round(rep.L_sum, 3)}"]
        return float(rep.L_sum), tokens, {
            "Z": round(rep.Z_sum, 3), "n_req": rep.n_req_tokens}

    def act_hash(self, step: int) -> Tuple[float, List, Dict]:
        h = int(hashlib.md5(
            f"{self.task}:{step}".encode()).hexdigest()[:8], 16)
        tokens = [f"hash_{h}"]
        return (h % 100_000) / 100_000.0, tokens, {"hash": h}

    # ---------- asymmetric actions ----------
    def act_asym(self, step: int) -> Tuple[float, List, Dict]:
        q = f"describe step {step} of {self.task}"
        a = self.router().ask(q)
        # use the words of the asymmetric answer as the token stream
        words = a.answer.split()[:4] or ["silence"]
        tokens = [f"asym_{w}" for w in words]
        v = float(a.detail.get("supertrace_q", 0.0))
        return v, tokens, {
            "answer": a.answer[:40],
            "alpha": a.alpha_used,
            "words": words}

    def act_read(self, step: int) -> Tuple[float, List, Dict]:
        raw = [(step * 31 + i * 7) % 10_000 for i in range(16)]
        c = tokens_to_k_sum(raw, K=64)
        emo = np.array([0.5] * 6)
        prof = self.reader().read(
            f"step_{step}", c, emotional_vector=emo,
            symmetric_only=False)
        tokens = [f"read_H_{round(prof.entropy_H, 3)}",
                  f"read_S_{round(prof.supertrace_S, 3)}",
                  f"read_strikes_{prof.strikes}"]
        return float(prof.entropy_H), tokens, {
            "S": round(prof.supertrace_S, 3),
            "strikes": prof.strikes,
            "flagged": prof.flagged}

    # ---------- action schedule (deterministic) ----------
    def schedule(self
                 ) -> List[Tuple[str, bool,
                                 Callable[[int],
                                          Tuple[float, List, Dict]]]]:
        """
        One full cycle of the agent.
        Returns (name, is_symmetric, fn).
        """
        return [
            ("classify", True,  self.act_classify),
            ("arith",    True,  self.act_arith),
            ("asym",     False, self.act_asym),
            ("plan",     True,  self.act_plan),
            ("read",     False, self.act_read),
            ("hash",     True,  self.act_hash),
        ]

    # ---------- one step ----------
    def step(self, verbose: bool = False) -> FFTStep:
        sched = self.schedule()
        step_num = len(self.history)
        name, is_sym, fn = sched[step_num % len(sched)]

        t0 = time.perf_counter()
        try:
            value, tokens, detail = fn(step_num)
        except Exception as exc:
            value, tokens, detail = 0.0, [], {"error": str(exc)}
        dt = (time.perf_counter() - t0) * 1e3

        if is_sym:
            self.sym_tokens.extend(tokens)
        else:
            self.asym_tokens.extend(tokens)

        rec = FFTStep(step=step_num, action=name,
                      is_symmetric=is_sym,
                      n_tokens=len(tokens),
                      value=float(value), detail=detail,
                      elapsed_ms=dt)
        self.history.append(rec)

        if verbose:
            tag = "SYM" if is_sym else "ASYM"
            print(f"  [step {step_num:>3}]  {tag:<4}  "
                  f"{name:<8}  tokens={len(tokens):>2}  "
                  f"value={value:>+10.4f}  "
                  f"({dt:5.2f} ms)")

        return rec

    # ---------- full loop ----------
    def run(self, max_steps: int = 4096,
            verbose: bool = False) -> None:
        guard = 0
        while not self.complete() and guard < max_steps:
            self.step(verbose=verbose)
            guard += 1
        self.completed = self.complete()
        self._compress_and_validate()

    # ---------- compress / decompress / validate ----------
    def _compress_and_validate(self) -> None:
        K = self.K
        # 1. Build |Ci| from both pools
        c_orig = build_mixed_ci(self.sym_tokens, self.asym_tokens,
                                K, self.mu)
        self.c_orig = c_orig

        # 2. Split into symmetric and asymmetric parts
        c_sym, c_asym, n_sym, n_asym = split_symmetric_asymmetric(
            c_orig, K)
        self.n_sym = n_sym
        self.n_asym = n_asym

        # 3. FFT compress each part
        part_sym  = fft_compress_part(c_sym,  self.M_sym)
        part_asym = fft_compress_part(c_asym, self.M_asym)
        self.part_sym  = part_sym
        self.part_asym = part_asym

        # 4. Decompress
        c_sym_rec  = fft_decompress_part(part_sym)
        c_asym_rec = fft_decompress_part(part_asym)

        # 5. Exponential-kernel smoothing on the symmetric part
        c_sym_rec = conv_exp_kernel(c_sym_rec)

        # 6. Recombine
        n = 2 * K + 1
        out = np.zeros(n, dtype=float)
        for i in range(1, K + 1):
            pos, neg = K + i, K - i
            out[pos] = c_sym_rec[pos] + c_asym_rec[pos]
            out[neg] = c_sym_rec[neg] - c_asym_rec[neg]
        out[K] = c_sym_rec[K]

        # 7. Enforce Möbius support
        for i in range(1, K + 1):
            if self.mu[i] == 0:
                out[K + i] = 0.0
                out[K - i] = 0.0
        self.c_recon = out

        # 8. Relative L2 error
        denom = float(np.linalg.norm(c_orig)) + 1e-12
        self.rel_err = float(np.linalg.norm(out - c_orig) / denom)

        # 9. Fingerprint
        h = hashlib.sha256()
        h.update(b"agentic_fft")
        h.update(str(K).encode())
        for p in (part_sym, part_asym):
            h.update(np.asarray(p.kept_indices).tobytes())
            h.update(np.asarray(p.kept_values).tobytes())
        self.fingerprint = h.hexdigest()

    # ---------- supertrace of the reconstruction ----------
    @staticmethod
    def _supertrace(sig: np.ndarray) -> float:
        s = 0.0
        for i, v in enumerate(sig):
            s += v if (i % 2 == 0) else -v
        return float(s)

    # ---------- report ----------
    def report(self) -> str:
        L = []
        L.append("=" * 78)
        L.append("Agentic FFT Möbius harness  ·  symmetric / asymmetric split")
        L.append("=" * 78)
        L.append(f"  task                    : {self.task!r}")
        L.append(f"  K                       : {self.K}")
        L.append(f"  target entries          : {self.target_entries}")
        L.append(f"  sym token count         : {len(self.sym_tokens)}")
        L.append(f"  asym token count        : {len(self.asym_tokens)}")
        L.append(f"  total tokens            : {self.total_tokens()}")
        L.append(f"  complete                : {self.completed}")
        L.append("")
        if self.part_sym is not None and self.part_asym is not None:
            L.append("--- FFT compression ---")
            L.append(f"  symmetric entries       : {self.n_sym}")
            L.append(f"  asymmetric entries      : {self.n_asym}")
            L.append(f"  kept (sym part)         : "
                     f"{self.part_sym.n_kept} of {self.part_sym.n_full}")
            L.append(f"  kept (asym part)        : "
                     f"{self.part_asym.n_kept} of {self.part_asym.n_full}")
            L.append(f"  S_sym                   : "
                     f"{self.part_sym.supertrace:+.6f}")
            L.append(f"  S_asym                  : "
                     f"{self.part_asym.supertrace:+.6f}")
            L.append(f"  H_sym                   : "
                     f"{self.part_sym.entropy:.6f}")
            L.append(f"  H_asym                  : "
                     f"{self.part_asym.entropy:.6f}")
            L.append(f"  mass_sym                : "
                     f"{self.part_sym.mass:.6e}")
            L.append(f"  mass_asym               : "
                     f"{self.part_asym.mass:.6e}")
            L.append(f"  fingerprint             : "
                     f"{self.fingerprint[:32]}…")
            L.append("")
            L.append("--- Reconstruction ---")
            S_o = self._supertrace(self.c_orig)
            S_r = self._supertrace(self.c_recon)
            L.append(f"  ||c_orig||₂             : "
                     f"{float(np.linalg.norm(self.c_orig)):.6f}")
            L.append(f"  ||c_recon||₂            : "
                     f"{float(np.linalg.norm(self.c_recon)):.6f}")
            L.append(f"  relative L2 error       : {self.rel_err:.4e}")
            L.append(f"  supertrace original     : {S_o:+.6f}")
            L.append(f"  supertrace reconstructed: {S_r:+.6f}")
            L.append(f"  Δ supertrace            : {S_r - S_o:+.3e}")
        L.append("")
        L.append("--- Timeline ---")
        L.append("  step  action     sym?   n_tok  value        detail")
        L.append("  " + "-" * 68)
        for r in self.history:
            tag = "S" if r.is_symmetric else "A"
            d = ", ".join(f"{k}={v}"
                          for k, v in list(r.detail.items())[:2])
            L.append(f"  {r.step:>4}  {r.action:<9}  {tag}     "
                     f"{r.n_tokens:>3}   {r.value:>+10.4f}  {d}")
        return "\n".join(L)


# ============================================================
#  Plot
# ============================================================
def plot_history(agent: AgenticFFTHarness) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (a) value timeline split by channel
    ax = axes[0, 0]
    sym = [(r.step, r.value) for r in agent.history if r.is_symmetric]
    asy = [(r.step, r.value) for r in agent.history if not r.is_symmetric]
    if sym:
        ax.scatter([s for s, _ in sym], [v for _, v in sym],
                   color="#2c3e50", s=26, label="symmetric")
    if asy:
        ax.scatter([s for s, _ in asy], [v for _, v in asy],
                   color="#e74c3c", s=26, marker="^",
                   label="asymmetric")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("step"); ax.set_ylabel("value")
    ax.set_title("Action values by channel")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (b) token accumulation
    ax = axes[0, 1]
    steps = [r.step for r in agent.history]
    cum_sym = np.cumsum([r.n_tokens if r.is_symmetric else 0
                         for r in agent.history])
    cum_asy = np.cumsum([r.n_tokens if not r.is_symmetric else 0
                         for r in agent.history])
    ax.plot(steps, cum_sym, color="#2c3e50", lw=1.8,
            label="symmetric tokens")
    ax.plot(steps, cum_asy, color="#e74c3c", lw=1.8,
            label="asymmetric tokens")
    ax.axhline(agent.target_entries, color="g", ls=":",
               lw=1.2, label="target")
    ax.set_xlabel("step"); ax.set_ylabel("cumulative tokens")
    ax.set_title("Token accumulation")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (c) original vs reconstructed |Ci|
    ax = axes[1, 0]
    K = agent.K
    ax.plot(np.arange(-K, K + 1), agent.c_orig,
            color="#2c3e50", lw=1.4, label="|Ci| original")
    ax.plot(np.arange(-K, K + 1), agent.c_recon,
            color="#e74c3c", lw=1.2, ls="--",
            label="reconstruction")
    ax.set_xlabel("index"); ax.set_ylabel("value")
    ax.set_title(f"Memory  ·  rel L2 = {agent.rel_err:.2e}")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (d) spectrum of the two parts
    ax = axes[1, 1]
    sym_spec  = np.abs(np.fft.rfft(agent.part_sym.kept_values,
                                   n=agent.part_sym.n_full))
    asym_spec = np.abs(np.fft.rfft(agent.part_asym.kept_values,
                                   n=agent.part_asym.n_full))
    ax.semilogy(sym_spec + 1e-9, color="#2c3e50", lw=1.4,
                label="symmetric kept spectrum")
    ax.semilogy(asym_spec + 1e-9, color="#e74c3c", lw=1.4,
                label="asymmetric kept spectrum")
    ax.set_xlabel("frequency"); ax.set_ylabel("magnitude")
    ax.set_title("Kept FFT coefficients")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Agentic FFT Möbius harness  ·  "
        "symmetric actions → c_sym,  asymmetric actions → c_asym",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================
#  Demo
# ============================================================
def main():
    print("=" * 78)
    print("Agentic harness  ·  FFT Möbius memory as history")
    print("=" * 78)
    print()
    print("  Modules imported:")
    print("    · memory-okloglogk-mobius.py")
    print("        FFTMobiusMemory, split_symmetric_asymmetric, "
          "conv_exp_kernel")
    print("    · deterministic-asymmetric-linear-regression-OKlogK.py")
    print("        MobiusMLRouter, classify")
    print("    · quick-reader-analyzer.py")
    print("        StrikeFlagReader, tokens_to_k_sum")
    print("    · planning_harness.py")
    print("        PlanningHarness")
    print()

    K = 128
    task = ("Write a calculator with add, multiply, "
            "subtract operations")

    agent = AgenticFFTHarness(
        K=K,
        task=task,
        target_entries=96,          # completeness target
        M_sym=None,                 # top-M = floor(|S|)
        M_asym=None,
    )
    print(f"  K                       = {K}")
    print(f"  target entries          = {agent.target_entries}")
    print(f"  task                    = {task!r}")
    print()
    print("  running agent loop ...")

    t0 = time.perf_counter()
    agent.run(verbose=False)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  loop finished in {dt:.2f} ms")
    print()

    print(agent.report())

    # ---------- determinism check ----------
    print()
    print("--- Determinism check ---")
    a2 = AgenticFFTHarness(K=K, task=task, target_entries=96)
    a2.run()
    same_count  = (len(a2.sym_tokens)  == len(agent.sym_tokens) and
                   len(a2.asym_tokens) == len(agent.asym_tokens))
    same_orig   = np.allclose(a2.c_orig, agent.c_orig)
    same_recon  = np.allclose(a2.c_recon, agent.c_recon)
    same_fp     = (a2.fingerprint == agent.fingerprint)
    print(f"  same token counts        : {same_count}")
    print(f"  same original |Ci|       : {same_orig}")
    print(f"  same reconstruction      : {same_recon}")
    print(f"  same fingerprint         : {same_fp}")
    print(f"  → run is deterministic   : "
          f"{same_count and same_orig and same_recon and same_fp}")
    print()

    # ---------- complexity ----------
    print("--- Complexity ---")
    print("  load sibling modules       once")
    print("  one step (action)          O(K) or O(K log K)")
    print("  build_mixed_ci             O(|tokens|)")
    print("  split_symmetric_asymmetric O(K)")
    print("  FFT compress (each part)   O(K log K)")
    print("  inverse FFT + conv kernel  O(K log K)")
    print("  ─────────────────────────────────────────────")
    print(f"  total loop                 "
          f"O(|steps| · K log K) + O(K log K)")

    try:
        plot_history(agent)
    except Exception as exc:
        print(f"\n  (plot skipped: {exc})")

    print("\nDone.")


if __name__ == "__main__":
    main()