#!/usr/bin/env python3
"""
ugc_clock_scheduler.py
======================

UGC discrete log as integrated circuit scheduler for the muon/tau clock.

    Cycle space      ugc_discrete_log.UGCDiscreteLog
                     →  n = p-1 cycle slots  (L side)
                     →  work → cycle via dlog(hash(work) mod p)
                     →  phase = -H_{cycle+1}  (continuous time offset)
                     →  BSGS reverse lookup  O(√n)

    Clock domain     O(K)_conv_tau_muon.OKRealClock
                     →  tick events (K_sum = 0)  +  electron (i = 0)
                     →  parity check (Figure 4)
                     →  frequency, tick_rate

Scheduling map
--------------
    UGC L_i          =  IC cycle slot i
    UGC R_i          =  work identifier  (g^i mod p)
    dlog(hash(w))    =  which cycle w runs on
    -H_{x+1}         =  clock phase offset at cycle x
    Clock ticks      =  synchronization barriers
    Parity ratio     =  policy: symmetric (>1) vs asymmetric (≤1)

Modes
-----
    direct     — work fires at its assigned dlog cycle
    barrier    — work fires at the next clock barrier ≥ assigned cycle

Complexity
----------
    submit                 O(1)         dlog table lookup (symmetric)
                           O(√n)        BSGS               (asymmetric)
    run                    O(n)         single cycle scan
    verify_assignment      O(log p)     pow check
"""

from __future__ import annotations
import hashlib
import importlib.util
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- UGC discrete log  (no special filename chars) ----
from ugc_discrete_log import (
    UGCDiscreteLog,
    GAMMA,
    ALPHA,
)


# ---- O(K)_conv_tau_muon.py  (filename contains parens) ----
def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ok = _load("ok_clock", "O(K)_conv_tau_muon.py")
OKRealClock = _ok.OKRealClock


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
DENSITY = 6.0 / (PI * PI)


# ============================================================
#  WorkItem
# ============================================================
@dataclass
class WorkItem:
    name:        str
    hash_val:    int              # hash(name) mod p
    cycle:       int              # dlog cycle assignment
    phase:       float            # -H_{cycle+1}
    priority:    int = 2          # lower = higher priority
    work_class:  str = "regular"  # 'electron' | 'k-sum' | 'regular'
    executed:    bool = False
    executed_at: Optional[int] = None

    def is_done(self) -> bool:
        return self.executed


# ============================================================
#  UGCScheduler
# ============================================================
class UGCScheduler:
    """
    Integrated circuit scheduler.

    The clock provides tick events, parity protection, and frequency.
    The UGC graph provides the cycle space and dlog-based assignment.
    """
    def __init__(self,
                 p: int = 10007,
                 g: int = 5,
                 K_clock: int = 20,
                 J_C: float = 1.0,
                 verbose: bool = True):
        self.p = p
        self.g = g
        self.K_clock = K_clock

        # ---- cycle space ----
        self.ugc = UGCDiscreteLog(p=p, g=g)

        # ---- tick source ----
        self.clock = OKRealClock(K=K_clock, J_C=J_C,
                                 smooth=True, tick_tol=1e-9)

        # ---- local phase classification ----
        self._window = K_clock + 1
        self._electron_local = set(self.clock.tick_electron)
        self._ksum_local     = set(self.clock.tick_ksum)
        self._ticks_local    = set(self.clock.ticks)

        # ---- policy ----
        self.parity_protected = self.clock.parity_ratio > 1.0
        self.policy = "symmetric" if self.parity_protected else "asymmetric"

        # ---- state ----
        self.slots: List[List[WorkItem]] = [[] for _ in range(self.ugc.n)]
        self.registry: Dict[str, WorkItem] = {}
        self.stats = dict(submitted=0, executed=0, cycles_run=0)

        if verbose:
            print(f"--- UGC scheduler  p={p}  g={g}  n={self.ugc.n} ---")
            print(f"    clock K={K_clock}   J_C={J_C}")
            print(f"    parity ratio    = {self.clock.parity_ratio:.4f}  "
                  f"policy = {self.policy}")
            print(f"    electron tick   = cycle 0 (local i=0)")
            print(f"    K-sum ticks     = {self.clock.tick_ksum}")

    # --------------------------------------------------------
    def _hash_work(self, name: str) -> int:
        h = int.from_bytes(
            hashlib.sha256(name.encode()).digest()[:8], "big")
        h = h % self.ugc.p
        return h if h != 0 else 1

    # --------------------------------------------------------
    def _classify(self, cycle: int) -> str:
        local = cycle % self._window
        if local in self._electron_local:
            return "electron"
        if local in self._ksum_local:
            return "k-sum"
        return "regular"

    # --------------------------------------------------------
    def submit(self, name: str, priority: int = 2) -> WorkItem:
        """Assign work to a cycle via dlog(hash(name))."""
        if name in self.registry:
            return self.registry[name]

        h = self._hash_work(name)

        # symmetric → direct dlog table lookup
        # asymmetric → BSGS reverse search
        if self.parity_protected:
            x, lv = self.ugc.dlog(h)
        else:
            x, lv, _ = self.ugc.dlog_bsgs(h)

        if x is None:                                     # defensive
            x = h % self.ugc.n
            lv = (-self.ugc.H[x + 1]
                  if x + 1 < len(self.ugc.H) else 0.0)

        w = WorkItem(
            name=name, hash_val=h, cycle=x, phase=lv,
            priority=priority,
            work_class=self._classify(x),
        )
        self.slots[x].append(w)
        self.registry[name] = w
        self.stats["submitted"] += 1
        return w

    # --------------------------------------------------------
    def lookup(self, name: str,
               use_bsgs: Optional[bool] = None) -> Optional[int]:
        """Return the cycle assigned to a given work name."""
        h = self._hash_work(name)
        if use_bsgs is None:
            use_bsgs = not self.parity_protected
        if use_bsgs:
            x, _, _ = self.ugc.dlog_bsgs(h)
        else:
            x, _ = self.ugc.dlog(h)
        return x

    # --------------------------------------------------------
    def verify_assignment(self, name: str) -> bool:
        """Check pow(g, cycle, p) == hash via UGC.verify."""
        w = self.registry.get(name)
        if w is None:
            return False
        return self.ugc.verify(w.hash_val, w.cycle)

    # --------------------------------------------------------
    def schedule_at(self, cycle: int) -> List[WorkItem]:
        if cycle < 0 or cycle >= self.ugc.n:
            return []
        return list(self.slots[cycle])

    # --------------------------------------------------------
    def barrier_cycles(self, max_cycle: Optional[int] = None
                       ) -> List[int]:
        """
        Cycles that are synchronization barriers.

            symmetric    — every clock tick is a barrier
            asymmetric   — only electron tick (i = 0) is a barrier
        """
        if max_cycle is None:
            max_cycle = self.ugc.n
        local_set = (self._ticks_local if self.parity_protected
                     else self._electron_local)
        return [c for c in range(max_cycle)
                if (c % self._window) in local_set]

    # --------------------------------------------------------
    def run(self, max_cycles: Optional[int] = None,
            mode: str = "direct") -> List[WorkItem]:
        """
        Run the schedule.

            mode = 'direct'    — work fires at its assigned cycle
            mode = 'barrier'   — work fires at the next barrier cycle
                                 (queue then flush at barrier)
        """
        n = self.ugc.n
        max_c = min(n, max_cycles) if max_cycles else n

        if mode == "direct":
            fired = []
            for cycle in range(max_c):
                for w in self.slots[cycle]:
                    if not w.executed:
                        w.executed = True
                        w.executed_at = cycle
                        fired.append(w)
                        self.stats["executed"] += 1
                self.stats["cycles_run"] += 1
            return fired

        if mode == "barrier":
            barriers = set(self.barrier_cycles(max_cycle=max_c))
            pending: Dict[int, List[WorkItem]] = {}
            fired = []
            for cycle in range(max_c):
                # queue any unexecuted work at this cycle
                for w in self.slots[cycle]:
                    if not w.executed:
                        pending.setdefault(cycle, []).append(w)

                # if this cycle is a barrier, flush everything queued
                if cycle in barriers:
                    flush = []
                    for q_cycle in sorted(pending):
                        if q_cycle <= cycle:
                            flush.extend(pending.pop(q_cycle))
                    for w in flush:
                        w.executed = True
                        w.executed_at = cycle
                        fired.append(w)
                        self.stats["executed"] += 1
                self.stats["cycles_run"] += 1
            return fired

        raise ValueError(f"unknown mode {mode!r}")

    # --------------------------------------------------------
    def pending(self) -> List[WorkItem]:
        return [w for w in self.registry.values() if not w.executed]

    def report(self) -> str:
        lines = [
            "=" * 76,
            "UGC scheduler  ·  integrated circuit over muon/tau clock",
            "=" * 76,
            f"  p                = {self.p}",
            f"  g                = {self.g}",
            f"  n = p − 1        = {self.ugc.n}",
            f"  clock K          = {self.K_clock}",
            f"  parity ratio     = {self.clock.parity_ratio:.4f}  "
            f"({self.policy})",
            f"  tick rate        = {self.clock.tick_rate():.4f}",
            f"  freq             = {self.clock.frequency():.6f}",
            "",
            f"  submitted        = {self.stats['submitted']}",
            f"  executed         = {self.stats['executed']}",
            f"  pending          = "
            f"{self.stats['submitted'] - self.stats['executed']}",
            f"  cycles run       = {self.stats['cycles_run']}",
        ]
        by_cls: Dict[str, int] = {}
        for w in self.registry.values():
            by_cls[w.work_class] = by_cls.get(w.work_class, 0) + 1
        lines.append("")
        for cls in ("electron", "k-sum", "regular"):
            lines.append(f"  {cls:>10s} class = {by_cls.get(cls, 0)}")
        lines.append("")
        n_barriers = len(self.barrier_cycles())
        lines.append(f"  barriers         = {n_barriers} cycles")
        return "\n".join(lines)

    # --------------------------------------------------------
    def visualize(self) -> None:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.32)

        # (a) clock K_sum + ticks
        ax = fig.add_subplot(gs[0, 0])
        k = np.arange(self.clock.K + 1)
        ax.plot(k, self.clock.K_sum, color="#16a085", lw=1.4,
                label=r"$K_{sum}(k)$")
        ax.axhline(0, color="k", lw=0.4)
        for t in self.clock.tick_electron:
            ax.axvline(t, color="#f1c40f", ls="--", lw=1.0, alpha=0.7,
                       label="electron" if t == 0 else "")
        for i, t in enumerate(self.clock.tick_ksum):
            ax.axvline(t, color="#8e44ad", ls=":", lw=1.0, alpha=0.7,
                       label="K-sum tick" if i == 0 else "")
        ax.set_xlabel("clock phase k")
        ax.set_ylabel(r"$K_{sum}$")
        ax.set_title(f"Clock tick source  ·  K = {self.clock.K}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (b) parity check
        ax = fig.add_subplot(gs[0, 1])
        ax.bar(["‖G_ee‖", "‖G_eo‖"],
               [self.clock.norm_ee, self.clock.norm_eo],
               color=["#16a085", "#e74c3c"], edgecolor="k")
        ax.text(0, self.clock.norm_ee, f"{self.clock.norm_ee:.3f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(1, self.clock.norm_eo, f"{self.clock.norm_eo:.3f}",
                ha="center", va="bottom", fontsize=9)
        prot = "protected" if self.parity_protected else "unprotected"
        ax.set_ylabel("‖G‖")
        ax.set_title(f"Parity check\n"
                     f"ratio = {self.clock.parity_ratio:.4f} ({prot})")
        ax.grid(alpha=0.3, axis="y")

        # (c) policy summary
        ax = fig.add_subplot(gs[0, 2])
        ax.axis("off")
        color = ("#16a085" if self.parity_protected else "#e74c3c")
        ax.text(0.5, 0.75, f"policy: {self.policy}",
                ha="center", va="center", fontsize=15, weight="bold",
                color=color)
        ax.text(0.5, 0.55,
                f"tick rate = {self.clock.tick_rate():.4f}",
                ha="center", fontsize=11)
        ax.text(0.5, 0.42,
                f"freq      = {self.clock.frequency():.6f}",
                ha="center", fontsize=11)
        ax.text(0.5, 0.29,
                f"n cycles  = {self.ugc.n}",
                ha="center", fontsize=11)
        ax.text(0.5, 0.16,
                f"executed  = {self.stats['executed']} / "
                f"{self.stats['submitted']}",
                ha="center", fontsize=11, weight="bold")

        # (d) work → cycle histogram
        ax = fig.add_subplot(gs[1, 0])
        colors = {"electron": "#f1c40f",
                  "k-sum":    "#8e44ad",
                  "regular":  "#3498db"}
        cycles = [w.cycle for w in self.registry.values()]
        classes = [w.work_class for w in self.registry.values()]
        for cls in ("electron", "k-sum", "regular"):
            cy = [c for c, cl in zip(cycles, classes) if cl == cls]
            if cy:
                ax.hist(cy, bins=40, color=colors[cls], alpha=0.7,
                        edgecolor="k", label=f"{cls} ({len(cy)})")
        ax.set_xlabel("cycle"); ax.set_ylabel("work count")
        ax.set_title("Work → cycle assignment (via dlog)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (e) phase vs cycle
        ax = fig.add_subplot(gs[1, 1])
        pts = sorted([(w.cycle, w.phase) for w in self.registry.values()])
        if pts:
            xs = [c for c, _ in pts]
            ys = [p for _, p in pts]
            ax.scatter(xs, ys, s=14, alpha=0.7, color="#3498db",
                       edgecolors="k", linewidths=0.3)
            x_max = max(xs)
            xx = np.linspace(1, max(x_max, 10), 200)
            yy = -np.array([
                self.ugc.H[int(x)] if int(x) < len(self.ugc.H)
                else math.log(x) + GAMMA
                for x in xx
            ])
            ax.plot(xx, yy, "r--", lw=1.0, alpha=0.6,
                    label=r"$-H_{x+1}$ reference")
            ax.set_xlabel("cycle x")
            ax.set_ylabel(r"phase  $-H_{x+1}$")
            ax.set_title("Phase decay  ·  work follows the log curve")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (f) execution timeline
        ax = fig.add_subplot(gs[1, 2])
        by_cls: Dict[str, List[int]] = {
            "electron": [], "k-sum": [], "regular": []
        }
        for w in self.registry.values():
            if w.executed:
                by_cls[w.work_class].append(w.executed_at)
        for cls, times in by_cls.items():
            if times:
                ax.vlines(times, 0, 1, color=colors[cls], alpha=0.65,
                          lw=0.7, label=f"{cls} ({len(times)})")
        ax.set_xlabel("execution cycle")
        ax.set_yticks([])
        ax.set_title(f"Execution timeline\n"
                     f"{self.stats['executed']} fired / "
                     f"{self.stats['submitted']} submitted")
        if any(by_cls.values()):
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="x")

        plt.suptitle(
            f"UGC integrated circuit scheduler  ·  "
            f"p={self.p}  g={self.g}  K={self.K_clock}  ·  "
            f"policy = {self.policy}  ·  α = {ALPHA:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("UGC integrated circuit scheduler  ·  over the muon/tau clock")
    print("=" * 78)
    print(f"  α = 1/(π − e)  = {ALPHA:.6f}")
    print(f"  6/π²           = {DENSITY:.6f}")
    print()

    sched = UGCScheduler(p=10007, g=5, K_clock=20, J_C=1.0,
                         verbose=True)
    print()

    work_names = [
        "load_image", "fft_forward", "squarefree_mask", "ifft_backward",
        "compress_frame", "decompress_frame", "write_output",
        "update_log", "send_response", "ack_request",
        "read_config", "parse_args", "open_socket", "close_socket",
        "hash_check", "verify_sig", "sync_clock", "flush_buffer",
    ]

    print("--- submit ---")
    print(f"  {'name':<22s}  {'hash':>8s}  {'cycle':>7s}  "
          f"{'phase':>10s}  {'class':>9s}")
    for name in work_names:
        w = sched.submit(name)
        print(f"  {w.name:<22s}  {w.hash_val:>8d}  "
              f"{w.cycle:>7d}  {w.phase:>+10.4f}  "
              f"{w.work_class:>9s}")
    print()

    # verify dlog correctness
    print("--- verify assignments  (g^x mod p == hash) ---")
    ok_all = True
    for name in work_names:
        ok = sched.verify_assignment(name)
        ok_all = ok_all and ok
    print(f"  all verified       = {ok_all}")
    print()

    # compare direct vs barrier mode
    print("--- direct mode ---")
    sched_direct = UGCScheduler(p=10007, g=5, K_clock=20, verbose=False)
    for name in work_names:
        sched_direct.submit(name)
    fired_direct = sched_direct.run(mode="direct")
    print(f"  executed = {len(fired_direct)} / {len(work_names)}")

    print()
    print("--- barrier mode ---")
    sched_barrier = UGCScheduler(p=10007, g=5, K_clock=20, verbose=False)
    for name in work_names:
        sched_barrier.submit(name)
    fired_barrier = sched_barrier.run(mode="barrier")
    print(f"  executed = {len(fired_barrier)} / {len(work_names)}")

    barriers = sched_barrier.barrier_cycles()
    print(f"  barriers = {len(barriers)} cycles  "
          f"(first 10 = {barriers[:10]})")

    # execution cycle comparison
    print()
    print("--- execution cycle shifts  (direct → barrier) ---")
    print(f"  {'name':<22s}  {'cycle':>7s}  "
          f"{'direct':>7s}  {'barrier':>8s}  {'Δ':>5s}")
    for name in work_names[:8]:
        w_d = sched_direct.registry[name]
        w_b = sched_barrier.registry[name]
        d_at = w_d.executed_at if w_d.executed else -1
        b_at = w_b.executed_at if w_b.executed else -1
        delta = b_at - d_at if (d_at >= 0 and b_at >= 0) else 0
        print(f"  {name:<22s}  {w_d.cycle:>7d}  "
              f"{d_at:>7d}  {b_at:>8d}  {delta:>5d}")
    print()

    print(sched.report())
    print()

    # also demonstrate run on sched_direct for visualization
    sched_direct.run(mode="direct")
    sched_direct.visualize()


if __name__ == "__main__":
    demo()