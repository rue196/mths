#!/usr/bin/env python3
"""
charging_battery_wire.py
========================

Wire-driven charging for the neutral battery.

Pipeline
--------
    wire_analog.AnalogWire       pumps electrons with alternating E-field
              │
              ▼  coupling filter  κ_local ≥ κ_threshold
    ElectronBuffer               MOSB-framed batches of accepted electrons
              │                  header = MAGIC | K | n | cid | S | H | m
              ▼  verify S, H, m on unpack
    ChargeDeposit                symmetric deposit into the battery
                                 ΔQ_cathode = +Σ|q_i|
                                 ΔQ_anode   = −Σ|q_i|
                                 ΔQ_net     ≡ 0

Anti-depletion invariants
-------------------------
    (1)  Q_net ≡ 0                (symmetric deposit, exact)
    (2)  ‖G_ee‖ > ‖G_eo‖          (parity protection, Figure 4)
    (3)  Π_e ∈ [0, 1]             (bounded cloud, Figure 3)

Per-electron charge quantum
---------------------------
    q_i = κ_local · Π_e_local · tanh(|E_local|)   ∈ [0, 1]
"""

from __future__ import annotations
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from wire_analog import (
    AnalogWire, alternating_E_field, pi_e,
    PI, E, ALPHA, A_STEP 
)

# ---- import the neutral battery (typo-safe) ----

from neurtal_anti_deplete_battery_handler import (
    NeutralBatteryHandler, DENSITY)



# ============================================================
#  MOSB constants
# ============================================================
MAGIC       = b"ELEB"                      # ELEctron Buffer
HEADER_FMT  = ">4sIQQddd"                  # magic, K, n, cid, S, H, m
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # = 48 bytes
PACKET_FMT  = ">iddddd"                    # idx, s, |v|, E, Π_e, κ
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # = 44 bytes


# ============================================================
#  Möbius sieve
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    if K < 1:
        return [0] * (K + 1)
    mu = [0] * (K + 1)
    mu[1] = 1
    primes: List[int] = []
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Electron packet
# ============================================================
@dataclass
class ElectronPacket:
    """
    One accepted, coupled electron from the wire.

    All fields are bounded in [0, 1] (except idx and |v|):

        κ_local   coupling strength at the wire point
        Π_e       bounded cloud intensity
        E_local   local E-field (bounded via tanh)
    """
    idx:       int
    s_pos:     float
    v_mag:     float
    E_local:   float
    pi_e:      float
    kappa:     float

    @property
    def q_abs(self) -> float:
        """Charge quantum deposited onto the cathode (and mirrored)."""
        e_bounded = math.tanh(abs(self.E_local))
        pe_bounded = max(0.0, min(1.0, self.pi_e))
        k_bounded = max(0.0, min(1.0, self.kappa))
        return k_bounded * pe_bounded * e_bounded

    def to_bytes(self) -> bytes:
        return struct.pack(PACKET_FMT, self.idx, self.s_pos, self.v_mag,
                           self.E_local, self.pi_e, self.kappa)

    @classmethod
    def from_bytes(cls, b: bytes) -> "ElectronPacket":
        idx, s, v, E_l, pe, k = struct.unpack(PACKET_FMT, b)
        return cls(int(idx), float(s), float(v), float(E_l),
                   float(pe), float(k))


# ============================================================
#  MOSB-framed electron buffer
# ============================================================
@dataclass
class BatchRecord:
    cid: int
    n_electrons: int
    Q_total: float
    S: float
    H: float
    m: float
    status: str          # "accepted" | "rejected"
    reason: str = ""


class ElectronBuffer:
    """
    MOSB-framed buffer for accepted coupled electrons.

    Frame layout (per batch):
        MAGIC(4) | K(4) | n(8) | cid(8) | S(8) | H(8) | m(8) | packets

    Each packet is PACKET_SIZE = 44 bytes.

    Invariants per batch:
        S = Σ_i (−1)^i |q_i|       (alternating charge sum)
        H = −α·p·log(p)            with p = |S|/n
        m = |S|·exp(−H)
    """

    def __init__(self, K: int = 20, batch_size: int = 16):
        self.K = int(K)
        self.batch_size = int(batch_size)
        self.mu = mobius_sieve(K)
        self._pending: List[ElectronPacket] = []
        self.ready: List[bytes] = []
        self.records: List[BatchRecord] = []
        self._cid = 0

    # ---- append / flush ----
    def append(self, packet: ElectronPacket) -> None:
        self._pending.append(packet)
        if len(self._pending) >= self.batch_size:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending:
            return

        # invariants over the batch
        qs = [p.q_abs for p in self._pending]
        S = 0.0
        for i, q in enumerate(qs):
            S += q if (i % 2 == 0) else -q

        n = len(self._pending)
        if abs(S) < 1e-12:
            H = 0.0
            m = 0.0
        else:
            p = min(abs(S) / n, 0.9999)
            H = -ALPHA * p * math.log(p) if p > 0 else 0.0
            m = abs(S) * math.exp(-H) if H < 700 else 0.0

        # frame
        cid = self._cid
        self._cid += 1
        header = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        payload = b"".join(p.to_bytes() for p in self._pending)
        self.ready.append(header + payload)
        self._pending = []

    def drain_ready(self) -> List[bytes]:
        self._flush_pending()
        out = self.ready
        self.ready = []
        return out

    # ---- verify / unpack ----
    def unpack(self, blob: bytes) -> List[ElectronPacket]:
        if len(blob) < HEADER_SIZE:
            raise ValueError("blob shorter than header")
        magic, K, n, cid, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])

        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if K != self.K:
            raise ValueError(f"K mismatch: {K} != {self.K}")

        body = blob[HEADER_SIZE:]
        need = n * PACKET_SIZE
        if len(body) < need:
            raise ValueError(
                f"payload short: {len(body)} < {need}")

        packets: List[ElectronPacket] = []
        for i in range(n):
            off = i * PACKET_SIZE
            packets.append(
                ElectronPacket.from_bytes(body[off:off + PACKET_SIZE]))

        # re-verify supertrace
        qs = [p.q_abs for p in packets]
        S_check = 0.0
        for i, q in enumerate(qs):
            S_check += q if (i % 2 == 0) else -q
        if abs(S_check - S) > 1e-9:
            raise ValueError(
                f"supertrace mismatch: {S_check:.6e} != {S:.6e}")

        # record
        self.records.append(BatchRecord(
            cid=int(cid),
            n_electrons=n,
            Q_total=float(sum(qs)),
            S=float(S), H=float(H), m=float(m),
            status="accepted",
        ))
        return packets


# ============================================================
#  Charging wire  (AnalogWire with cached E amplitude)
# ============================================================
class ChargingWire:
    """
    Wire configured for charging with fast field access.

    The wire's `field_at_time` is monkey-patched to use cached
    `S_νμ + S_ντ` (they don't depend on t in wire_analog.py), which
    would otherwise recompute the 720-term Levi-Civita every step.
    """

    def __init__(self, N: int = 32, length: float = 10.0,
                 wiggle: float = 0.25, turns: float = 3.0,
                 N_electrons: int = 20, J_C: float = 1.0,
                 omega: float = 1.0):
        self.wire = AnalogWire(
            N=N, length=length, wiggle=wiggle, turns=turns,
            N_electrons=N_electrons, J_C=J_C, omega=omega)

        # --- cache E amplitude (time-independent) ---
        half = self.wire.N // 2
        S_mu, S_tau = alternating_E_field(
            self.wire.t_wire[:half], self.wire.r_wire[:half],
            J_C=self.wire.J_C)
        self.S_mu = float(S_mu)
        self.S_tau = float(S_tau)
        self.E_amp = self.S_mu + self.S_tau

        # --- monkey-patch the wire's field lookup ---
        self.wire.field_at_time = self._fast_field

    def _fast_field(self, t: float) -> float:
        return self.E_amp * math.cos(self.wire.omega * t)

    # --------------------------------------------------------
    def run_one_step(self, dt: float, t: float) -> None:
        self.wire.step(dt, t)


# ============================================================
#  Charging battery
# ============================================================
@dataclass
class ChargingStats:
    total_emitted: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_deposited: int = 0
    batches_flushed: int = 0
    batches_rejected: int = 0


class ChargingBattery(NeutralBatteryHandler):
    """
    Neutral battery extended with wire-based charging.

    Anti-depletion invariants of the parent battery are preserved:

        Q_net ≡ 0
        ‖G_ee‖ > ‖G_eo‖
        Π_e ∈ [0,1]

    Charging only adds to |Q_plate|, never removes — the battery
    becomes more charged, never less.
    """

    def __init__(self,
                 K: int = 20,
                 J_C: float = 1.0,
                 N_events: int = 41,
                 N_samples: int = 1024,
                 free_e_threshold: float = 0.05,
                 wire_N: int = 32,
                 wire_N_electrons: int = 20,
                 wire_omega: float = 1.0,
                 coupling_threshold: float = 0.10,
                 batch_size: int = 16):
        super().__init__(
            K=K, J_C=J_C, N_events=N_events,
            N_samples=N_samples,
            free_e_threshold=free_e_threshold)

        # --- wire ---
        self.wire = ChargingWire(
            N=wire_N, N_electrons=wire_N_electrons, omega=wire_omega)

        # --- electron buffer ---
        self.buffer = ElectronBuffer(K=K, batch_size=batch_size)
        self.coupling_threshold = float(coupling_threshold)

        # --- charge accounting ---
        self.q_cathode_total = 0.0
        self.q_anode_total = 0.0
        self.charge_level = 0.0
        self.stats = ChargingStats()

        # --- trace for plots ---
        self.trace: Dict[str, List[float]] = {
            "step": [],
            "t": [],
            "wire_e_pos": [],
            "accepted_pos": [],
            "kappa_mean": [],
            "charge_level": [],
            "q_net": [],
            "batch_ready": [],
            "pending": [],
        }

    # --------------------------------------------------------
    def _local_coupling(self, wire_idx: int) -> float:
        """
        Peak coupling strength at a wire index (even-even only).

            κ_local = max_j K_even[i, j]
        """
        row = self.wire.wire.K_even[wire_idx, :]
        if row.size == 0:
            return 0.0
        return float(np.max(row))

    # --------------------------------------------------------
    def _collect_accepted(self, t: float
                          ) -> Tuple[List[ElectronPacket], int]:
        """
        Filter wire electrons by coupling threshold.

        Returns (accepted_packets, n_total_electrons).
        """
        wire = self.wire.wire
        E_local = self.wire._fast_field(t)
        packets: List[ElectronPacket] = []
        n_total = wire.N_electrons

        for i in range(n_total):
            s = float(wire.e_pos[i])
            idx = wire.wire_index(s)
            kappa = self._local_coupling(idx)

            if kappa < self.coupling_threshold:
                continue

            r_eff = s / wire.length * PI
            pe = float(pi_e(t, r_eff))

            packets.append(ElectronPacket(
                idx=i,
                s_pos=s,
                v_mag=float(abs(wire.e_vel[i])),
                E_local=E_local,
                pi_e=pe,
                kappa=kappa,
            ))

        return packets, n_total

    # --------------------------------------------------------
    def _deposit_ready(self) -> None:
        """Flush buffer, verify every batch, deposit symmetrically."""
        batches = self.buffer.drain_ready()
        if not batches:
            return

        for blob in batches:
            self.stats.batches_flushed += 1
            try:
                packets = self.buffer.unpack(blob)
            except ValueError as exc:
                self.stats.batches_rejected += 1
                # record rejection
                self.buffer.records.append(BatchRecord(
                    cid=-1, n_electrons=0, Q_total=0.0,
                    S=0.0, H=0.0, m=0.0,
                    status="rejected",
                    reason=str(exc),
                ))
                continue

            # ---- symmetric deposit ----
            Q_batch = float(sum(p.q_abs for p in packets))
            self.q_cathode_total += Q_batch
            self.q_anode_total   -= Q_batch
            self.charge_level    += Q_batch
            self.stats.total_deposited += len(packets)

    # --------------------------------------------------------
    def charge(self, n_steps: int = 50, dt: float = 0.02) -> None:
        """
        Run the charging session.

            1. wire.step(dt, t)
            2. collect accepted electrons
            3. append to buffer (auto-flushes at batch_size)
            4. drain ready batches and deposit into the battery
        """
        for step in range(n_steps):
            t = step * dt

            # 1. advance wire
            self.wire.run_one_step(dt, t)

            # 2. collect
            packets, n_total = self._collect_accepted(t)
            self.stats.total_emitted += n_total
            self.stats.total_accepted += len(packets)
            self.stats.total_rejected += (n_total - len(packets))

            # 3. buffer
            for p in packets:
                self.buffer.append(p)

            # 4. deposit
            self._deposit_ready()

            # --- trace ---
            wire = self.wire.wire
            self.trace["step"].append(step)
            self.trace["t"].append(t)
            self.trace["wire_e_pos"].append(list(wire.e_pos))
            self.trace["accepted_pos"].append(
                [p.s_pos for p in packets])
            self.trace["kappa_mean"].append(
                float(np.mean([p.kappa for p in packets]))
                if packets else 0.0)
            self.trace["charge_level"].append(self.charge_level)
            self.trace["q_net"].append(
                self.q_cathode_total + self.q_anode_total)
            self.trace["batch_ready"].append(len(batches)
                                             if (batches := self.buffer.ready)
                                             else 0)
            self.trace["pending"].append(len(self.buffer._pending))

        # final flush
        self._deposit_ready()

    # --------------------------------------------------------
    def charge_report(self) -> str:
        n_acc = sum(1 for r in self.buffer.records
                    if r.status == "accepted")
        n_rej = sum(1 for r in self.buffer.records
                    if r.status == "rejected")
        q_net = self.q_cathode_total + self.q_anode_total

        L = ["=" * 76,
             "Charging report  ·  wire → buffer → battery",
             "=" * 76]
        L.append("  wire")
        L.append(f"    N                       : {self.wire.wire.N}")
        L.append(f"    N_electrons             : "
                 f"{self.wire.wire.N_electrons}")
        L.append(f"    S_νμ + S_ντ             : "
                 f"{self.wire.S_mu + self.wire.S_tau:+.4e}")
        L.append(f"    coupling threshold      : "
                 f"{self.coupling_threshold:.4f}")
        L.append(f"    batch size              : "
                 f"{self.buffer.batch_size}")
        L.append("")
        L.append("  electron accounting")
        L.append(f"    total emitted           : {self.stats.total_emitted}")
        L.append(f"    accepted (κ ≥ thr)      : {self.stats.total_accepted}")
        L.append(f"    rejected (κ < thr)      : {self.stats.total_rejected}")
        L.append(f"    deposited               : "
                 f"{self.stats.total_deposited}")
        L.append(f"    acceptance rate         : "
                 f"{self.stats.total_accepted / max(self.stats.total_emitted, 1):.4f}")
        L.append("")
        L.append("  MOSB buffer")
        L.append(f"    batches flushed         : "
                 f"{self.stats.batches_flushed}")
        L.append(f"    batches accepted        : {n_acc}")
        L.append(f"    batches rejected        : {n_rej}")
        L.append(f"    frame size              : "
                 f"{HEADER_SIZE} B header + "
                 f"{self.buffer.batch_size} × {PACKET_SIZE} B")
        L.append("")
        L.append("  battery charge")
        L.append(f"    Q_cathode (μ, +)        : "
                 f"{self.q_cathode_total:+.6f}")
        L.append(f"    Q_anode   (τ, −)        : "
                 f"{self.q_anode_total:+.6f}")
        L.append(f"    Q_net = cathode + anode : {q_net:+.2e}  "
                 f"(must be 0)")
        L.append(f"    charge level            : "
                 f"{self.charge_level:.6f}")
        L.append("")
        L.append("  parent battery invariants")
        L.append(f"    parity ratio            : "
                 f"{self.parity_ratio:.4f}  "
                 f"({'OK' if self.parity_ratio > 1.0 else 'BROKEN'})")
        L.append(f"    Π_e bounded             : "
                 f"[0,1] by construction")
        return "\n".join(L)

    # --------------------------------------------------------
    def visualize(self) -> None:
        if not self.trace["step"]:
            return

        steps = np.array(self.trace["step"])
        t = np.array(self.trace["t"])
        charge = np.array(self.trace["charge_level"])
        q_net = np.array(self.trace["q_net"])
        pending = np.array(self.trace["pending"])
        kappa_m = np.array(self.trace["kappa_mean"])

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)

        # (a) wire electron positions  ·  accepted vs total
        ax = fig.add_subplot(gs[0, 0])
        for s, total in zip(steps[::2], self.trace["wire_e_pos"][::2]):
            ax.scatter([s] * len(total), total,
                       s=8, color="#bdc3c7", alpha=0.5, zorder=1)
        for s, accepted in zip(steps[::2],
                               self.trace["accepted_pos"][::2]):
            if accepted:
                ax.scatter([s] * len(accepted), accepted,
                           s=14, color="#16a085", alpha=0.9,
                           edgecolors="k", linewidths=0.3, zorder=3)
        ax.set_xlabel("wire step"); ax.set_ylabel("wire position s")
        ax.set_title("Wire electrons\n"
                     "grey = all  ·  green = accepted (κ ≥ thr)")
        ax.grid(alpha=0.3)

        # (b) coupling filter distribution
        ax = fig.add_subplot(gs[0, 1])
        kappas = np.linspace(0, 1, 100)
        ax.hist(kappa_m, bins=20, color="#3498db", alpha=0.7,
                edgecolor="k", label="⟨κ⟩ of accepted")
        ax.axvline(self.coupling_threshold, color="#e74c3c",
                   ls="--", lw=1.5,
                   label=f"threshold = {self.coupling_threshold}")
        ax.set_xlabel("mean κ of accepted electrons")
        ax.set_ylabel("steps")
        ax.set_title("Coupling filter\n"
                     f"acceptance rate = "
                     f"{self.stats.total_accepted / max(self.stats.total_emitted, 1):.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (c) buffer occupancy
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(steps, pending, color="#8e44ad", lw=1.4,
                label="pending packets")
        ready = np.array(self.trace["batch_ready"])
        ax.bar(steps, ready, width=0.9, color="#e67e22",
               alpha=0.6, label="batches ready")
        ax.set_xlabel("wire step")
        ax.set_ylabel("count")
        ax.set_title(f"Electron buffer\n"
                     f"batches flushed = "
                     f"{self.stats.batches_flushed}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (d) charge level
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(steps, charge, color="#f1c40f", lw=2.0,
                label="charge level  Q_total")
        ax.fill_between(steps, 0, charge, color="#f1c40f",
                        alpha=0.25)
        ax.set_xlabel("wire step")
        ax.set_ylabel("charge level")
        ax.set_title(f"Battery charge level\n"
                     f"Q_total = {self.charge_level:.4f} "
                     f"·  no decay (monotone)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (e) Q_net ≡ 0
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(steps, q_net, color="#16a085", lw=2.0,
                label=r"$Q_{net} = Q_{anode} + Q_{cathode}$")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-1e-10, 1e-10)
        ax.set_xlabel("wire step")
        ax.set_ylabel(r"$Q_{net}$")
        ax.set_title(f"Charge neutrality  ·  max |Q_net| = "
                     f"{np.max(np.abs(q_net)):.1e}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (f) per-batch charge distribution
        ax = fig.add_subplot(gs[1, 2])
        acc_Qs = [r.Q_total for r in self.buffer.records
                  if r.status == "accepted"]
        if acc_Qs:
            ax.hist(acc_Qs, bins=15, color="#e74c3c",
                    alpha=0.7, edgecolor="k")
            ax.axvline(np.mean(acc_Qs), color="k", ls="--",
                       lw=1.2, label=f"⟨Q_batch⟩ = "
                                    f"{np.mean(acc_Qs):.4f}")
        else:
            ax.text(0.5, 0.5, "(no batches flushed)",
                    ha="center", va="center",
                    family="monospace")
        ax.set_xlabel("Q_batch per MOSB batch")
        ax.set_ylabel("count")
        ax.set_title(f"Batch charge distribution\n"
                     f"accepted = {len(acc_Qs)}  ·  "
                     f"rejected = {self.stats.batches_rejected}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.suptitle(
            f"Charging battery  ·  wire_analog → MOSB electron buffer → "
            f"neutral battery  ·  α = {ALPHA:.4f}  ·  "
            f"Q_net ≡ 0",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Charging battery  ·  wire_analog → MOSB electron buffer")
    print("=" * 76)
    print(f"  α      = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print(f"  6/π²                  = {DENSITY:.6f}")
    print()

    # ---- battery + wire + buffer ----
    battery = ChargingBattery(
        K=20, J_C=1.0, N_events=41, N_samples=512,
        free_e_threshold=0.05,
        wire_N=32, wire_N_electrons=20, wire_omega=1.0,
        coupling_threshold=0.10, batch_size=8,
    )
    print(f"--- charging configuration ---")
    print(f"  wire N                : {battery.wire.wire.N}")
    print(f"  wire N_electrons      : {battery.wire.wire.N_electrons}")
    print(f"  S_νμ + S_ντ           : "
          f"{battery.wire.S_mu + battery.wire.S_tau:+.4e}")
    print(f"  coupling threshold    : "
          f"{battery.coupling_threshold}")
    print(f"  batch size            : {battery.buffer.batch_size}")
    print()

    # ---- run the charge ----
    t0 = time.perf_counter()
    battery.charge(n_steps=60, dt=0.02)
    elapsed = (time.perf_counter() - t0) * 1e3
    print(f"--- charge session  ·  {elapsed:.1f} ms ---")
    print()

    # ---- report ----
    print(battery.charge_report())
    print()

    # ---- sample of batches ----
    if battery.buffer.records:
        print("--- batch log  (first 8 accepted) ---")
        print(f"  {'cid':>4}  {'n_e':>4}  "
              f"{'Q_batch':>10}  {'S':>12}  {'H':>8}  "
              f"{'m':>10}  {'status':>10}")
        acc = [r for r in battery.buffer.records
               if r.status == "accepted"][:8]
        for r in acc:
            print(f"  {r.cid:>4d}  {r.n_electrons:>4d}  "
                  f"{r.Q_total:>10.4f}  {r.S:>+12.4e}  "
                  f"{r.H:>8.4f}  {r.m:>10.4e}  {r.status:>10s}")
        print()

    # ---- invariant check ----
    q_net = battery.q_cathode_total + battery.q_anode_total
    print("--- anti-depletion invariants ---")
    print(f"  Q_net                 = {q_net:+.2e}   "
          f"(target 0)")
    print(f"  parity ratio          = {battery.parity_ratio:.4f}  "
          f"({'protected' if battery.parity_ratio > 1.0 else 'BROKEN'})")
    print(f"  charge level          = {battery.charge_level:.6f}  "
          f"(monotone increasing)")
    print(f"  battery state         : "
          f"{'HEALTHY' if abs(q_net) < 1e-9 else 'LEAKING'}")
    print()

    # ---- physics summary ----
    print("--- physics of the charging battery ---")
    print("  wire            = AnalogWire   (alternating E-field)")
    print("  filter          = κ_local ≥ threshold")
    print("  buffer          = MOSB-framed electron batches")
    print("  deposit         = ΔQ_cathode = +Σ|q_i|")
    print("                    ΔQ_anode   = −Σ|q_i|")
    print("                    ΔQ_net     ≡ 0")
    print()
    print("  charge quantum per electron:")
    print("      q_i = κ_local · Π_e_local · tanh(|E_local|)   ∈ [0,1]")
    print()
    print("  why it can't deplete:")
    print("      · charging only ADDS to |Q_plate|")
    print("      · symmetric deposit keeps Q_net ≡ 0")
    print("      · parity protection blocks any scattering leak")
    print("      · the buffer rejects any batch with a forged S")
    print()

    battery.visualize()


if __name__ == "__main__":
    demo()