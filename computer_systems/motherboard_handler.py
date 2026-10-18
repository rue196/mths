#!/usr/bin/env python3
"""
motherboard.py
==============

Motherboard-level handler connecting:

    ClockChip         ← DC_chirp.py     (DCMobiusGate)
    MemoryController  ← DRAM.py          (FermionDRAM)
    CPU               ← CPU_Lin_Alg.py   (ChirpBridgePipeline)

System bus
----------
    Address bus   μ-gated:   addr live iff μ(addr) ≠ 0
    Data bus      coupling:  raw G_ij, no normalization
    Control bus   finite step A_STEP = α / 0.3628
    Clock         DC throughput of ζ(t), closed form

Transaction flow
----------------
    1. CPU issues request(addr, op, payload)
    2. Address decoder: μ(addr) ≠ 0 ?
         no  → NAK (bad address), cycle still ticks
         yes → 3.
    3. Clock tick: bump ζ(t) phase, propagate dc_throughput
    4. Memory read / write via coupling matrix
    5. Coherence register: signed sum of valid couplings
    6. Return transaction record + bus cycle telemetry

Boot sequence
-------------
    clock.boot()  → dc_throughput, supertrace S/H/m, clock period
    memory.boot() → populate cells at square-free addresses
    cpu.boot()    → chirp bridge, buffer sealed, chirp-z ready
    self_test()   → round-trip on live addresses + NAK on dead ones
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from DC_chirp import (
    DCMobiusGate, mobius_sieve, ALPHA, A_STEP, DENSITY,
)
from CPU_Lin_Alg import (
    ChirpBridgePipeline, ReadOnlyCiBuffer,
    base_chirp, chirp_bridge,
)
from DRAM import FermionDRAM


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
NORM  = 1.0 - math.exp(-ALPHA * (PI + E))


# ============================================================
#  Bus-cycle telemetry
# ============================================================
@dataclass
class BusCycle:
    t: float
    cycle: int
    op: str
    addr: int
    mu: int
    data_in: Optional[int]
    data_out: Optional[int]
    valid: bool
    coupling: float
    dc_phase: float


@dataclass
class Transaction:
    op: str
    addr: int
    valid: bool
    reason: str
    cycle: int
    latency: int
    data: Optional[int] = None
    coupling: float = 0.0
    S: float = 0.0
    H: float = 0.0
    m: float = 0.0


# ============================================================
#  Clock chip  ·  DC_chirp.py
# ============================================================
class ClockChip:
    """
    System clock derived from the DC Möbius throughput.

        dc_throughput = (1 − exp(−α x)) / NORM
        ζ(t)          = Σ c_i exp(i·t·i/α)
        S, H, m       = supertrace / entropy / mass invariants
    """
    def __init__(self, K: int = 20, smooth: bool = True):
        self.K = int(K)
        self.gate = DCMobiusGate(K=K, smooth=smooth)
        self.period = self.gate.period
        self.dc_throughput = self.gate.dc_throughput()
        self.S, self.H, self.m = self.gate.supertrace()
        self.cycle = 0
        self.t = 0.0

    # ---- one clock tick ----
    def tick(self, dt: float = 0.02) -> Dict:
        self.cycle += 1
        self.t = (self.t + dt) % self.period
        z = self.gate.zeta(self.t)
        return dict(
            cycle=self.cycle,
            t=self.t,
            zeta=abs(z),
            power=abs(z) ** 2,
            throughput=self.dc_throughput,
        )

    # ---- invariants ----
    def report(self) -> Dict:
        return dict(
            K=self.K,
            period=self.period,
            dc_throughput=self.dc_throughput,
            S=self.S,
            H=self.H,
            m=self.m,
            cycle=self.cycle,
        )


# ============================================================
#  Memory controller  ·  DRAM.py
# ============================================================
class MemoryController:
    """
    Address space  =  square-free integers k with μ(k) ≠ 0.
    Cell value     =  coupling strength (integer).
    Read           =  Möbius-gated (no N×N matrix), optionally
                      via the dzeta particle search.
    """
    def __init__(self, K: int = 120, coupling_strength: int = 1):
        self.K = int(K)
        self.coupling_strength = int(coupling_strength)
        self.dram = FermionDRAM(K=K, coupling_strength=coupling_strength)
        self.mu = self.dram.mu
        self._refresh_coupling()

    # ---- coupling cache ----
    def _refresh_coupling(self) -> None:
        self.G, self.addrs = self.dram.coupling_matrix()
        self.addr_to_idx = {a: i for i, a in enumerate(self.addrs)}
        self.row_sum = {a: float(self.G[i].sum())
                        for a, i in self.addr_to_idx.items()}

    # ---- address decode ----
    def is_live(self, addr: int) -> bool:
        if not (1 <= addr < len(self.mu)):
            return False
        return int(self.mu[addr]) != 0

    def mu_at(self, addr: int) -> int:
        if not (1 <= addr < len(self.mu)):
            return 0
        return int(self.mu[addr])

    # ---- read / write ----
    def read(self, addr: int) -> Optional[int]:
        if not self.is_live(addr):
            return None
        return self.dram.read(addr)

    def write(self, addr: int, value: int) -> bool:
        if not self.is_live(addr):
            return False
        self.dram.write(addr, value)
        self._refresh_coupling()
        return True

    def coupling_at(self, addr: int) -> float:
        if addr not in self.row_sum:
            return 0.0
        return self.row_sum[addr]

    # ---- particle read (dzeta flow) ----
    def particle_read(self, s0: complex,
                      steps: int = 100, dt: float = 0.02
                      ) -> List[Tuple[int, int]]:
        _, readings = self.dram.read_via_particle(
            s0, steps=steps, dt=dt)
        return readings

    # ---- summary ----
    def report(self) -> Dict:
        live = int(sum(1 for a in self.addrs))
        return dict(
            K=self.K,
            live_cells=live,
            density=live / max(self.K, 1),
            coupling_max=float(self.G.max()) if self.G.size else 0.0,
            coupling_mean=float(self.G[self.G > 0].mean())
                          if np.any(self.G > 0) else 0.0,
        )


# ============================================================
#  CPU  ·  CPU_Lin_Alg.py
# ============================================================
class CPU:
    """
    Executes the chirp bridge pipeline.

        build:  μ sieve + |Ci| buffer + sym/asym planes
        solve:  chirp bridge + chirp-z transcendental transform
    """
    def __init__(self, K: int = 64):
        self.K = int(K)
        self.pipe = ChirpBridgePipeline(K=K)
        self.buffer = self.pipe.build()
        self.sym, self.asym = self.buffer.read_planes()
        self.w = base_chirp(K, self.pipe.mu)
        self.bridge = chirp_bridge(self.sym, self.asym, self.w)
        self._last_result = None

    def solve(self, t_samples: Optional[np.ndarray] = None):
        self._last_result = self.pipe.solve(self.buffer, t_samples)
        return self._last_result

    def report(self) -> Dict:
        r = self._last_result
        base = dict(
            K=self.K,
            buffer_live=int(self.pipe.live.size),
            buffer_dc=float(self.buffer.dc()),
            bridge_norm=float(self.bridge['mix_norm']),
        )
        if r is not None:
            base.update(
                S=r.S, H=r.H, m=r.m,
                chirp_z_ms=r.t_czt_ms,
                z_max=float(np.max(np.abs(r.Z_trans))),
            )
        return base


# ============================================================
#  Motherboard
# ============================================================
class Motherboard:
    """
    System bus + transaction arbiter.

        address bus   μ-gated
        data bus      coupling strength
        control bus   finite step A_STEP
        coherence     signed running sum of valid couplings
    """
    def __init__(self,
                 K_clock: int = 20,
                 K_mem: int = 120,
                 K_cpu: int = 64,
                 coupling_strength: int = 7,
                 dt_clock: float = 0.02):
        # ---- boot components ----
        t0 = time.perf_counter()
        self.clock = ClockChip(K=K_clock)
        self.memory = MemoryController(K=K_mem,
                                       coupling_strength=coupling_strength)
        self.cpu = CPU(K=K_cpu)
        self.boot_time_ms = (time.perf_counter() - t0) * 1e3
        self.dt_clock = float(dt_clock)

        # ---- bus state ----
        self.cycle = 0
        self.coherence_register = 0.0
        self.transactions: List[Transaction] = []
        self.cycles: List[BusCycle] = []

    # --------------------------------------------------------
    def decode(self, addr: int) -> Tuple[bool, int]:
        live = self.memory.is_live(addr)
        mu = self.memory.mu_at(addr)
        return live, mu

    # --------------------------------------------------------
    def access(self,
               op: str,
               addr: int,
               value: Optional[int] = None,
               dzeta_seed: Optional[complex] = None) -> Transaction:
        """
        One bus transaction.  Always consumes one clock tick.
        """
        self.cycle += 1
        tick = self.clock.tick(dt=self.dt_clock)
        live, mu = self.decode(addr)

        # ---- bus cycle telemetry ----
        cyc = BusCycle(
            t=tick['t'],
            cycle=self.cycle,
            op=op,
            addr=addr,
            mu=mu,
            data_in=value,
            data_out=None,
            valid=live,
            coupling=0.0,
            dc_phase=tick['zeta'],
        )

        # ---- address decoder reject ----
        if not live:
            self.cycles.append(cyc)
            tx = Transaction(
                op=op, addr=addr,
                valid=False,
                reason=f"NAK: μ({addr}) = 0",
                cycle=self.cycle,
                latency=1,
            )
            self.transactions.append(tx)
            return tx

        # ---- route by opcode ----
        if op == "READ":
            data = self.memory.read(addr)
            cpl = self.memory.coupling_at(addr)
            cyc.data_out = data
            cyc.coupling = cpl
            self.coherence_register += (
                cpl if self.cycle % 2 == 0 else -cpl)
            tx = Transaction(
                op="READ", addr=addr, valid=True, reason="ok",
                cycle=self.cycle, latency=1,
                data=data, coupling=cpl,
            )

        elif op == "WRITE":
            ok = self.memory.write(addr, value if value is not None else 0)
            cpl = self.memory.coupling_at(addr)
            cyc.coupling = cpl
            self.coherence_register += (
                cpl if self.cycle % 2 == 0 else -cpl)
            tx = Transaction(
                op="WRITE", addr=addr, valid=ok,
                reason="ok" if ok else "write failed",
                cycle=self.cycle, latency=1,
                data=value, coupling=cpl,
            )

        elif op == "SOLVE":
            # CPU sub-cycle — consumes several system cycles
            n_sub = self.cpu.K // 32
            for _ in range(max(1, n_sub)):
                self.clock.tick(dt=self.dt_clock)
                self.cycle += 1
            res = self.cpu.solve()
            tx = Transaction(
                op="SOLVE", addr=addr, valid=True, reason="ok",
                cycle=self.cycle, latency=max(1, n_sub),
                S=res.S, H=res.H, m=res.m,
            )

        elif op == "PARTICLE_READ":
            s0 = dzeta_seed if dzeta_seed is not None else (1.5 + 5j)
            readings = self.memory.particle_read(s0, steps=100)
            hit = [(a, v) for a, v in readings if a == addr]
            data = hit[0][1] if hit else None
            tx = Transaction(
                op="PARTICLE_READ", addr=addr,
                valid=data is not None,
                reason="hit" if data is not None else "miss",
                cycle=self.cycle,
                latency=100,
                data=data,
            )

        else:
            tx = Transaction(
                op=op, addr=addr, valid=False,
                reason=f"unknown opcode {op}",
                cycle=self.cycle, latency=1,
            )

        self.cycles.append(cyc)
        self.transactions.append(tx)
        return tx

    # --------------------------------------------------------
    #  Convenience wrappers
    # --------------------------------------------------------
    def read(self, addr: int) -> Transaction:
        return self.access("READ", addr)

    def write(self, addr: int, value: int) -> Transaction:
        return self.access("WRITE", addr, value=value)

    def solve(self, addr: int = 1) -> Transaction:
        return self.access("SOLVE", addr)

    def particle_read(self, addr: int, s0: complex = 1.5 + 5j
                      ) -> Transaction:
        return self.access("PARTICLE_READ", addr, dzeta_seed=s0)

    # --------------------------------------------------------
    #  Boot self-test
    # --------------------------------------------------------
    def self_test(self) -> Dict:
        """
        Round-trip write/read on live addresses,
        NAK on a known dead address.
        """
        addrs = self.memory.dram.address_list()[:6]
        round_trip_ok = 0
        round_trip_total = 0
        for a in addrs:
            v = (a * 13 + 7) % 256
            w = self.write(a, v)
            r = self.read(a)
            round_trip_total += 1
            if (w.valid and r.valid and r.data == v):
                round_trip_ok += 1

        # find a dead address
        dead_addr = None
        for k in range(1, self.memory.K):
            if not self.memory.is_live(k):
                dead_addr = k
                break
        nak_ok = False
        if dead_addr is not None:
            t = self.read(dead_addr)
            nak_ok = not t.valid

        return dict(
            round_trip_ok=round_trip_ok,
            round_trip_total=round_trip_total,
            dead_addr=dead_addr,
            nak_ok=nak_ok,
            boot_cycles=self.cycle,
        )

    # --------------------------------------------------------
    #  Coherence and bus statistics
    # --------------------------------------------------------
    def bus_stats(self) -> Dict:
        if not self.cycles:
            return dict(n=0)
        n = len(self.cycles)
        n_valid = sum(1 for c in self.cycles if c.valid)
        n_nak = n - n_valid
        latencies = {}
        for t in self.transactions:
            latencies.setdefault(t.op, []).append(t.latency)
        return dict(
            cycles=self.cycle,
            valid_transactions=n_valid,
            naks=n_nak,
            utilization=n_valid / max(n, 1),
            coherence=self.coherence_register,
            mean_latency={
                op: float(np.mean(lat)) for op, lat in latencies.items()
            },
        )

    # --------------------------------------------------------
    #  Full system report
    # --------------------------------------------------------
    def report(self) -> str:
        L = ["=" * 78,
             "Motherboard report",
             "=" * 78]
        clk = self.clock.report()
        mem = self.memory.report()
        cpu = self.cpu.report()
        bus = self.bus_stats()

        L.append(f"  boot time         : {self.boot_time_ms:.2f} ms")
        L.append(f"  cycles            : {bus['cycles']}")
        L.append(f"  coherence reg     : {bus['coherence']:+.6f}")
        L.append(f"  bus utilization   : {bus['utilization']:.4f}")
        L.append(f"  NAKs              : {bus['naks']}")
        L.append("")
        L.append("--- ClockChip  (DC_chirp.py) ---")
        L.append(f"  K                 : {clk['K']}")
        L.append(f"  period            : {clk['period']:.6f}")
        L.append(f"  dc throughput     : {clk['dc_throughput']:.6f}")
        L.append(f"  supertrace S      : {clk['S']:+.6f}")
        L.append(f"  entropy H         : {clk['H']:.6f}")
        L.append(f"  mass m            : {clk['m']:.4e}")
        L.append("")
        L.append("--- MemoryController  (DRAM.py) ---")
        L.append(f"  K                 : {mem['K']}")
        L.append(f"  live cells        : {mem['live_cells']}")
        L.append(f"  density           : {mem['density']:.4f}  "
                 f"(target 6/π² = {DENSITY:.4f})")
        L.append(f"  coupling max      : {mem['coupling_max']:.4f}")
        L.append(f"  coupling mean     : {mem['coupling_mean']:.4f}")
        L.append("")
        L.append("--- CPU  (CPU_Lin_Alg.py) ---")
        L.append(f"  K                 : {cpu['K']}")
        L.append(f"  buffer live       : {cpu['buffer_live']}")
        L.append(f"  buffer DC         : {cpu['buffer_dc']:+.4f}")
        L.append(f"  bridge ‖mix‖      : {cpu['bridge_norm']:.4f}")
        if "S" in cpu:
            L.append(f"  last solve S/H/m  : {cpu['S']:+.4f} / "
                     f"{cpu['H']:.4f} / {cpu['m']:.4e}")
            L.append(f"  chirp-z time      : {cpu['chirp_z_ms']:.2f} ms")
        L.append("")
        if bus["mean_latency"]:
            L.append("--- transaction latency ---")
            for op, lat in bus["mean_latency"].items():
                L.append(f"  {op:14s}    {lat:.2f} cycles")
        return "\n".join(L)

    # --------------------------------------------------------
    #  Visualization
    # --------------------------------------------------------
    def visualize(self) -> None:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

        # (a) address bus state
        ax = fig.add_subplot(gs[0, 0])
        mu = self.memory.mu[1:self.memory.K + 1]
        ax.vlines(np.arange(1, self.memory.K + 1), 0, np.abs(mu),
                  color="#e74c3c", lw=1.0)
        ax.set_xlabel("address k"); ax.set_ylabel("|μ(k)|")
        ax.set_title(f"Address bus  ·  μ-gated\n"
                     f"live = {self.memory.report()['live_cells']} / "
                     f"{self.memory.K}")
        ax.grid(alpha=0.3)

        # (b) clock waveform
        ax = fig.add_subplot(gs[0, 1])
        tt = np.linspace(0, self.clock.period, 512)
        zz = np.array([abs(self.clock.gate.zeta(t)) ** 2 for t in tt])
        ax.plot(tt, zz, color="#3498db", lw=1.2, label=r"$|\zeta(t)|^2$")
        ax.axhline(self.clock.dc_throughput,
                   color="#e74c3c", ls="--", lw=1.0,
                   label=f"dc = {self.clock.dc_throughput:.4f}")
        ax.set_xlabel("t"); ax.set_ylabel("power")
        ax.set_title("Clock chip  ·  DC throughput")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (c) transaction timeline
        ax = fig.add_subplot(gs[0, 2])
        if self.cycles:
            cyc_idx = [c.cycle for c in self.cycles]
            valid   = [c.valid for c in self.cycles]
            ys      = np.where(valid, 1.0, -1.0)
            colors  = ["#16a085" if v else "#e74c3c" for v in valid]
            ax.bar(cyc_idx, ys, color=colors, width=0.9)
            ax.axhline(0, color="k", lw=0.4)
            ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("cycle")
        ax.set_yticks([-1, 1]); ax.set_yticklabels(["NAK", "ok"])
        ax.set_title("Transaction timeline")
        ax.grid(alpha=0.3, axis="x")

        # (d) coupling matrix (memory)
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(self.memory.G, origin="lower",
                       cmap="inferno", aspect="equal")
        ax.set_xlabel("cell j"); ax.set_ylabel("cell i")
        ax.set_title("Memory coupling matrix  G_ij = κ_ij v_i v_j")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # (e) coherence register evolution
        ax = fig.add_subplot(gs[1, 1])
        coh = [0.0]
        for c in self.cycles:
            if c.valid:
                coh.append(coh[-1] + (
                    c.coupling if c.cycle % 2 == 0 else -c.coupling))
            else:
                coh.append(coh[-1])
        ax.plot(coh, color="#8e44ad", lw=1.2)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("cycle"); ax.set_ylabel("coherence")
        ax.set_title(f"Coherence register\n"
                     f"final = {self.coherence_register:+.4f}")
        ax.grid(alpha=0.3)

        # (f) CPU solve output (if any)
        ax = fig.add_subplot(gs[1, 2])
        if self.cpu._last_result is not None:
            Z = self.cpu._last_result.Z_trans
            ax.plot(np.real(Z), color="#3498db", lw=1.0, label="Re ζ")
            ax.plot(np.imag(Z), color="#e74c3c", lw=1.0, label="Im ζ")
            ax.set_xlabel("t sample")
            ax.set_title(f"CPU chirp-z output\n"
                         f"|ζ| max = {np.max(np.abs(Z)):.4f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "(no solve issued)", ha="center",
                    va="center", family="monospace")
            ax.set_axis_off()
        ax.grid(alpha=0.3)

        plt.suptitle(
            f"Motherboard  ·  clock + memory + CPU over the μ-gated bus  ·  "
            f"A_STEP = {A_STEP:.4f}  ·  α = {ALPHA:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Motherboard  ·  clock + memory + CPU over the μ-gated bus")
    print("=" * 78)
    print(f"  α      = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP               = {A_STEP:.6f}")
    print(f"  NORM                 = {NORM:.6f}")
    print(f"  6/π²                 = {DENSITY:.6f}")
    print()

    mb = Motherboard(
        K_clock=20, K_mem=120, K_cpu=64,
        coupling_strength=7, dt_clock=0.02,
    )
    print(f"--- boot time  : {mb.boot_time_ms:.2f} ms ---")
    print()

    # ---------- self-test ----------
    st = mb.self_test()
    print("--- self-test ---")
    print(f"  round-trip OK     : "
          f"{st['round_trip_ok']} / {st['round_trip_total']}")
    print(f"  dead address NAK  : {st['nak_ok']}  "
          f"(addr = {st['dead_addr']})")
    print(f"  boot cycles used  : {st['boot_cycles']}")
    print()

    # ---------- live transaction sweep ----------
    addrs = mb.memory.dram.address_list()[:8]
    print("--- live transactions  (READ/WRITE) ---")
    for a in addrs:
        v = (a * 37 + 11) % 200
        mb.write(a, v)
        r = mb.read(a)
        print(f"  addr={a:>4d}  μ={mb.memory.mu_at(a):+d}  "
              f"W({v:>3d})  R({r.data})  "
              f"coupling={r.coupling:.4f}")
    print()

    # ---------- NAK on dead address ----------
    dead_addr = st['dead_addr']
    print("--- NAK test on dead address ---")
    r = mb.read(dead_addr)
    print(f"  addr={dead_addr}  μ={mb.memory.mu_at(dead_addr):+d}  "
          f"valid={r.valid}  reason='{r.reason}'")
    print()

    # ---------- CPU solve cycle ----------
    print("--- CPU solve cycle ---")
    r = mb.solve(addr=1)
    print(f"  S = {r.S:+.4f}  H = {r.H:.4f}  m = {r.m:.4e}  "
          f"latency = {r.latency} cycles")
    print()

    # ---------- particle read ----------
    print("--- particle read through dzeta flow ---")
    r = mb.particle_read(addrs[0], s0=1.5 + 5.0j)
    print(f"  addr={r.addr}  valid={r.valid}  data={r.data}  "
          f"reason='{r.reason}'")
    print()

    # ---------- full report ----------
    print(mb.report())
    print()

    # ---------- visualization ----------
    mb.visualize()


if __name__ == "__main__":
    demo()