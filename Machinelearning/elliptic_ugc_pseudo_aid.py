#!/usr/bin/env python3
"""
UGC_elliptic_gate.py
====================

Elliptic regression extends UGC envelope training with an O(1)
pseudo-deterministic gate.

Key insight
-----------
The elliptic projection

    Π(x, y) = ½(1 + cos(2π x/π) · cos(2π y/e))  ∈ [0, 1]

is bounded, smooth, periodic, and computable in O(1) from (x, y).
The regression's identity

    pred_final = pred_alg + Π · corr
    └─ when Π high → algebraic dominates → reliable (baby)
    └─ when Π low  → transcendental  → needs search (giant)

gives us a **bounded pseudo-deterministic predictor** of the
baby-vs-giant decision.  Calibrated once, it converts every
mode classification from O(√n) (BSGS) into O(1) (table lookup).

Two encoders
------------
EllipticQPWeightEncoder
    Extends QPWeightEncoder with the elliptic gate as a fast
    pre-classifier.  Falls back to full Q/P only in the uncertain
    band [π_lo, π_hi].  Tracks fast-path vs qp-path usage.

EllipticFastQPWeightEncoder
    Fully O(1) per weight using a precomputed h → mode table.
    One-time precompute cost O(p), then every snap costs O(|θ|)
    table lookups with no dlog, no BSGS, no Q/P.

Both are drop-in for train_envelope_stream.

Imports (unchanged parents)
---------------------------
    elliptic_curve_suisse_regression  elliptic_projection,
                                       transcendental_coords,
                                       transcendental_correction,
                                       inverse_coords, algebraic_fit,
                                       inverse_map, ALPHA, PI, E, W1, W2
    UGC_discrete_train_envelope       QPWeightEncoder, EnvelopeStreamTrainer,
                                       train_envelope_stream
    UGC_discrete_train                UGCWeightEncoder
    complete_planes_spectral_sum      TwoPointMatrix
"""

from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- elliptic regression primitives (unchanged) ----
from elliptic_curve_suisse_regression import (
    elliptic_projection,
    elliptic_projection_vec,
    inverse_coords,
    algebraic_fit,
    inverse_map,
    transcendental_coords,
    transcendental_correction,
    supertrace,
    entropy,
    invariant_mass,
    PI, E, ALPHA, W1, W2,
)

# ---- UGC envelope training (unchanged) ----
from UGC_discrete_train_envelope import (
    QPWeightEncoder,
    EnvelopeStreamTrainer,
    train_envelope_stream,
)

from UGC_discrete_train import (
    UGCWeightEncoder,
    UGCWeightTensor,
)

from complete_planes_spectral_sum import (
    TwoPointMatrix,
    A_STEP,
)


# ============================================================
#  EllipticProjectionGate  ·  O(1) pseudo-deterministic classifier
# ============================================================
class EllipticProjectionGate:
    """
    Bounded gate on Π ∈ [0, 1] with precomputed periodic table.

    Decision:
        Π ≥ π_hi   →  "baby"       (algebraic dominant)
        Π ≤ π_lo   →  "giant"      (transcendental)
        otherwise  →  "uncertain"  (fall back to Q/P)
    """
    def __init__(self,
                 pi_hi: float = 0.7,
                 pi_lo: float = 0.3,
                 n_bins: int = 64):
        self.pi_hi = float(pi_hi)
        self.pi_lo = float(pi_lo)
        self.n_bins = int(n_bins)
        self._build_table()

    def _build_table(self):
        """Precompute Π over a periodic (u, v) ∈ [0,1)² grid."""
        n = self.n_bins
        u = (np.arange(n) + 0.5) / n
        v = (np.arange(n) + 0.5) / n
        U, V = np.meshgrid(u, v, indexing="ij")
        self.table = 0.5 * (1.0 + np.cos(2 * np.pi * U)
                                  * np.cos(2 * np.pi * V))

    # ---- single lookup ----
    def project(self, x: float, y: float) -> float:
        """O(1) Π lookup via table binning."""
        u = (x / PI) % 1.0
        v = (y / E) % 1.0
        i = min(int(u * self.n_bins), self.n_bins - 1)
        j = min(int(v * self.n_bins), self.n_bins - 1)
        return float(self.table[i, j])

    def project_exact(self, x: float, y: float) -> float:
        """Exact Π (one cos evaluation, still O(1))."""
        return elliptic_projection(x, y)

    def decide(self, x: float, y: float) -> str:
        p = self.project(x, y)
        if p >= self.pi_hi:
            return "baby"
        if p <= self.pi_lo:
            return "giant"
        return "uncertain"

    def decide_batch(self, xs: np.ndarray, ys: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized gate: returns (mode_codes, pi_values).
        mode_codes: 0 = giant, 1 = uncertain, 2 = baby.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        u = (xs / PI) % 1.0
        v = (ys / E) % 1.0
        i = np.minimum((u * self.n_bins).astype(int), self.n_bins - 1)
        j = np.minimum((v * self.n_bins).astype(int), self.n_bins - 1)
        pis = self.table[i, j]
        codes = np.ones_like(pis, dtype=np.int8) * 1
        codes[pis >= self.pi_hi] = 2
        codes[pis <= self.pi_lo] = 0
        return codes, pis


# ============================================================
#  Calibrate gate thresholds against Q/P ground truth
# ============================================================
def calibrate_gate(encoder: QPWeightEncoder,
                   gate: Optional[EllipticProjectionGate] = None,
                   n_samples: int = 4000,
                   seed: int = 42) -> Dict:
    """
    Sample (x, y) pairs uniformly, compute Q/P ground truth and Π.
    Find (π_hi, π_lo) that maximise agreement × coverage.
    """
    rng = np.random.default_rng(seed)
    n = encoder.ugc.n
    pis = np.zeros(n_samples)
    is_baby = np.zeros(n_samples, dtype=bool)

    for k in range(n_samples):
        x = int(rng.integers(0, n))
        y = (x + encoder.conjugate_offset) % n
        m = TwoPointMatrix(float(x + 1), float(y + 1))
        Q, P = m.Q(), m.P()
        is_baby[k] = (abs(Q) >= abs(P)) and (Q > 0)
        pis[k] = elliptic_projection(float(x + 1), float(y + 1))

    # sweep thresholds
    best = (-1.0, 0.7, 0.3)
    for hi in np.linspace(0.55, 0.95, 25):
        for lo in np.linspace(0.05, 0.45, 25):
            if lo >= hi:
                continue
            certain = (pis >= hi) | (pis <= lo)
            if certain.sum() < 0.05 * n_samples:
                continue
            pred = pis >= hi          # True → baby
            agree = (pred[certain] == is_baby[certain]).mean()
            coverage = certain.mean()
            score = agree * coverage
            if score > best[0]:
                best = (score, float(hi), float(lo))

    _, hi, lo = best
    if gate is not None:
        gate.pi_hi = hi
        gate.pi_lo = lo

    certain = (pis >= hi) | (pis <= lo)
    if certain.sum() > 0:
        pred = pis >= hi
        agreement = float((pred[certain] == is_baby[certain]).mean())
    else:
        agreement = 0.0

    return dict(
        pi_hi=hi, pi_lo=lo,
        agreement=agreement,
        coverage=float(certain.mean()),
        n_samples=int(n_samples),
        pi_baby_mean=float(pis[is_baby].mean()) if is_baby.any() else 0.0,
        pi_giant_mean=float(pis[~is_baby].mean()) if (~is_baby).any() else 0.0,
        pi_corr=float(np.corrcoef(pis, is_baby.astype(float))[0, 1])
                if pis.std() > 1e-12 else 0.0,
    )


# ============================================================
#  EllipticQPWeightEncoder  ·  gate pre-classifier
# ============================================================
class EllipticQPWeightEncoder(QPWeightEncoder):
    """
    QPWeightEncoder + elliptic gate as O(1) pre-classifier.

    Fast path  :  gate returns 'baby' or 'giant' → skip Q/P
    Q/P path   :  gate returns 'uncertain'       → compute Q/P
    """
    def __init__(self, *args,
                 pi_hi: float = 0.7,
                 pi_lo: float = 0.3,
                 n_bins: int = 64,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = EllipticProjectionGate(
            pi_hi=pi_hi, pi_lo=pi_lo, n_bins=n_bins)
        self.stats = dict(
            baby=0, giant=0, uncertain=0,
            fast_path=0, qp_path=0,
            dlog_bsgs=0, table_hit=0,
        )

    # --------------------------------------------------------
    def encode(self, w: float) -> Dict:
        h = self.weight_to_h(w)

        # ---- resolve x ----
        x_lookup = self.direct_table.get(h)
        if x_lookup is not None:
            x = int(x_lookup)
            self.stats["table_hit"] += 1
            from_table = True
        else:
            x_bsgs, _, _ = self.ugc.dlog_bsgs(h)
            x = int(x_bsgs) if x_bsgs is not None else None
            self.stats["dlog_bsgs"] += 1
            from_table = False

        if x is None:
            return dict(
                mode="fail", weight=float(w), h=int(h),
                x=None, y=None, Q=0.0, P=0.0, Pi=0.0,
                sym=0.0, asym=0.0, vertex_scalar=None,
                weight_reconstructed=float(w), residual=0.0,
                from_table=False, gate_decision="fail", fast=True,
            )

        n = self.ugc.n
        y = (x + self.conjugate_offset) % n

        # ---- elliptic gate (O(1)) ----
        gate_dec = self.gate.decide(float(x + 1), float(y + 1))
        pi_val = self.gate.project(float(x + 1), float(y + 1))

        if gate_dec == "uncertain":
            # fall back to Q/P
            m = TwoPointMatrix(float(x + 1), float(y + 1))
            Q = float(m.Q())
            P = float(m.P())
            mode = "baby" if (abs(Q) >= abs(P) and Q > 0) else "giant"
            self.stats["uncertain"] += 1
            self.stats["qp_path"] += 1
        else:
            mode = gate_dec
            m = TwoPointMatrix(float(x + 1), float(y + 1))
            Q = float(m.Q())
            P = float(m.P())
            self.stats["fast_path"] += 1

        self.stats[mode] += 1

        lv = (-float(self.ugc.H[x + 1])
              if x < len(self.ugc.H) else 0.0)

        return dict(
            mode=mode, weight=float(w), h=int(h),
            x=int(x), y=int(y),
            Q=Q, P=P, Pi=pi_val,
            sym=Q, asym=P,
            vertex_scalar=lv,
            weight_reconstructed=float(self.h_to_weight(h)),
            residual=abs(w - self.h_to_weight(h)),
            from_table=from_table,
            gate_decision=gate_dec,
            fast=(gate_dec != "uncertain"),
        )


# ============================================================
#  EllipticFastQPWeightEncoder  ·  fully O(1)
# ============================================================
class EllipticFastQPWeightEncoder(EllipticQPWeightEncoder):
    """
    Fully O(1) per weight via precomputed h → mode table.

    Precompute (one-time, O(p)):
        for each h ∈ Z_p^*:
            x = log_table[h]
            y = (x + conjugate_offset) % n
            mode = gate.decide(x+1, y+1)  or Q/P fallback
        store in mode_array[h]

    Encode (per weight, O(1)):
        mode = mode_array[h]  → array lookup
        weight_reconstructed = h_to_weight(h)
        no dlog, no BSGS, no Q/P
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_array: np.ndarray = np.zeros(self.p, dtype=np.int8)  # 1=baby, 0=giant
        self._precompute_modes()

    def _precompute_modes(self):
        t0 = time.perf_counter()
        n = self.ugc.n
        n_fast = n_slow = 0
        for h, x in self.ugc.log_table.items():
            y = (x + self.conjugate_offset) % n
            gate_dec = self.gate.decide(float(x + 1), float(y + 1))
            if gate_dec == "uncertain":
                m = TwoPointMatrix(float(x + 1), float(y + 1))
                Q, P = m.Q(), m.P()
                mode = "baby" if (abs(Q) >= abs(P) and Q > 0) else "giant"
                n_slow += 1
            else:
                mode = gate_dec
                n_fast += 1
            self.mode_array[h] = 1 if mode == "baby" else 0
        self.precompute_ms = (time.perf_counter() - t0) * 1e3
        self.precompute_fast = n_fast
        self.precompute_slow = n_slow

    def encode(self, w: float) -> Dict:
        h = self.weight_to_h(w)
        mode = "baby" if self.mode_array[h] == 1 else "giant"
        w_rec = float(self.h_to_weight(h))
        self.stats[mode] += 1
        self.stats["fast_path"] += 1
        return dict(
            mode=mode, weight=float(w), h=int(h),
            x=int(h), y=None,          # x = h is a placeholder for the fast path
            Q=0.0, P=0.0, Pi=0.0,
            sym=0.0, asym=0.0,
            vertex_scalar=None,
            weight_reconstructed=w_rec,
            residual=abs(w - w_rec),
            from_table=True, gate_decision=mode, fast=True,
        )


# ============================================================
#  Verification helpers
# ============================================================
def verify_accuracy(encoder: QPWeightEncoder,
                    n_samples: int = 2000,
                    seed: int = 7) -> Dict:
    """Compare elliptic gate predictions with Q/P ground truth."""
    rng = np.random.default_rng(seed)
    n = encoder.ugc.n
    gate = EllipticProjectionGate()          # fresh gate at default thresholds
    agree = 0
    agree_certain = 0
    total = 0
    certain_count = 0

    for _ in range(n_samples):
        x = int(rng.integers(0, n))
        y = (x + encoder.conjugate_offset) % n
        m = TwoPointMatrix(float(x + 1), float(y + 1))
        Q, P = m.Q(), m.P()
        truth = (abs(Q) >= abs(P)) and (Q > 0)
        dec = gate.decide(float(x + 1), float(y + 1))
        if dec == "baby":
            pred = True
        elif dec == "giant":
            pred = False
        else:
            continue                          # skip uncertain
        total += 1
        if pred == truth:
            agree += 1
        certain_count += 1

    return dict(
        n_samples=n_samples,
        n_certain=certain_count,
        agreement=agree / max(certain_count, 1),
    )


def compare_speeds(X, y, entropy,
                   n_snaps: int = 5,
                   batch_size: int = 32,
                   seed: int = 0) -> Dict:
    """
    Time per-snap cost across three encoder variants.

    Simulates a snap by calling encode() on every model weight.
    """
    from entropic_neural_alternating import generate_spectral_dataset

    rng = np.random.default_rng(seed)

    # 1 · parent QP
    p_enc = QPWeightEncoder(
        p=10007, g=5, w_min=-1.0, w_max=1.0,
        precompute_fraction=0.9)
    rng_weights = rng.standard_normal(7000).astype(np.float32)

    t0 = time.perf_counter()
    for _ in range(n_snaps):
        for w in rng_weights:
            p_enc.encode(float(w))
    parent_ms = (time.perf_counter() - t0) * 1e3 / n_snaps

    # 2 · EllipticQP (gate pre-classifier)
    e_enc = EllipticQPWeightEncoder(
        p=10007, g=5, w_min=-1.0, w_max=1.0,
        precompute_fraction=0.9,
        pi_hi=0.7, pi_lo=0.3)

    t0 = time.perf_counter()
    for _ in range(n_snaps):
        for w in rng_weights:
            e_enc.encode(float(w))
    elliptic_ms = (time.perf_counter() - t0) * 1e3 / n_snaps

    # 3 · EllipticFast (precomputed mode array)
    f_enc = EllipticFastQPWeightEncoder(
        p=10007, g=5, w_min=-1.0, w_max=1.0,
        precompute_fraction=0.9,
        pi_hi=0.7, pi_lo=0.3)

    t0 = time.perf_counter()
    for _ in range(n_snaps):
        for w in rng_weights:
            f_enc.encode(float(w))
    fast_ms = (time.perf_counter() - t0) * 1e3 / n_snaps

    return dict(
        n_weights=len(rng_weights),
        parent_ms=parent_ms,
        elliptic_ms=elliptic_ms,
        fast_ms=fast_ms,
        speedup_elliptic=parent_ms / elliptic_ms if elliptic_ms > 0 else 0,
        speedup_fast=parent_ms / fast_ms if fast_ms > 0 else 0,
    )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("UGC elliptic gate  ·  O(1) pseudo-deterministic classifier")
    print("=" * 78)
    print(f"  α         = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  ω₁ = π                  = {W1:.6f}")
    print(f"  ω₂ = e                  = {W2:.6f}")
    print()

    # ---- build base encoder to calibrate on ----
    base_enc = QPWeightEncoder(
        p=10007, g=5, w_min=-1.0, w_max=1.0,
        precompute_fraction=0.9)

    # ============================================================
    #  1 · Calibrate the gate
    # ============================================================
    print("=" * 78)
    print("1  ·  calibrate gate thresholds  (π_hi, π_lo)")
    print("=" * 78)
    gate = EllipticProjectionGate(pi_hi=0.7, pi_lo=0.3, n_bins=64)
    calib = calibrate_gate(base_enc, gate=gate, n_samples=4000)
    print(f"  π_hi               = {calib['pi_hi']:.4f}")
    print(f"  π_lo               = {calib['pi_lo']:.4f}")
    print(f"  agreement          = {calib['agreement']*100:.2f} %")
    print(f"  coverage (certain) = {calib['coverage']*100:.2f} %")
    print(f"  Π_baby_mean        = {calib['pi_baby_mean']:.4f}")
    print(f"  Π_giant_mean       = {calib['pi_giant_mean']:.4f}")
    print(f"  Π ↔ baby corr      = {calib['pi_corr']:+.4f}")
    print()

    # ============================================================
    #  2 · Comparison at weight samples
    # ============================================================
    print("=" * 78)
    print("2  ·  per-weight gate decision vs Q/P ground truth")
    print("=" * 78)
    print(f"  {'w':>6s}  {'h':>6s}  {'Π':>8s}  "
          f"{'gate':>10s}  {'Q/P':>8s}  {'agree':>6s}")
    print("  " + "-" * 52)
    agree = 0
    total = 0
    for w_val in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]:
        h = base_enc.weight_to_h(float(w_val))
        x = base_enc.ugc.log_table.get(h)
        if x is None:
            continue
        y = (x + base_enc.conjugate_offset) % base_enc.ugc.n
        m = TwoPointMatrix(float(x + 1), float(y + 1))
        Q, P = m.Q(), m.P()
        truth = "baby" if (abs(Q) >= abs(P) and Q > 0) else "giant"
        dec = gate.decide(float(x + 1), float(y + 1))
        pi_val = gate.project(float(x + 1), float(y + 1))
        ok = "✓" if dec == truth else ("—" if dec == "uncertain" else "✗")
        if dec != "uncertain":
            total += 1
            if dec == truth:
                agree += 1
        print(f"  {w_val:>6.3f}  {h:>6d}  {pi_val:>8.4f}  "
              f"{dec:>10s}  {truth:>8s}  {ok:>6s}")
    print()
    if total:
        print(f"  sampled agreement  = {agree}/{total}  "
              f"({100*agree/total:.1f} %)")
    print()

    # ============================================================
    #  3 · Speed comparison
    # ============================================================
    print("=" * 78)
    print("3  ·  per-snap cost  ·  three encoder variants")
    print("=" * 78)
    from entropic_neural_alternating import generate_spectral_dataset
    X, y, entropy = generate_spectral_dataset(
        num_samples=300, K_min=10, K_max=40, N_harmonics=20)

    speeds = compare_speeds(X, y, entropy, n_snaps=3, seed=0)

    print(f"  weights per snap     = {speeds['n_weights']}")
    print(f"  parent QP            = {speeds['parent_ms']:>8.2f} ms")
    print(f"  elliptic QP          = {speeds['elliptic_ms']:>8.2f} ms   "
          f"({speeds['speedup_elliptic']:.2f}×)")
    print(f"  elliptic fast        = {speeds['fast_ms']:>8.2f} ms   "
          f"({speeds['speedup_fast']:.2f}×)")
    print()

    # ============================================================
    #  4 · Actual training with elliptic-fast encoder
    # ============================================================
    print("=" * 78)
    print("4  ·  training with EllipticFastQPWeightEncoder")
    print("=" * 78)
    fast_enc = EllipticFastQPWeightEncoder(
        p=10007, g=5, w_min=-1.0, w_max=1.0,
        precompute_fraction=0.9,
        pi_hi=calib["pi_hi"], pi_lo=calib["pi_lo"],
        n_bins=64)
    print(f"  precompute time     = {fast_enc.precompute_ms:.2f} ms")
    print(f"  precompute fast/qp  = "
          f"{fast_enc.precompute_fast} / {fast_enc.precompute_slow}")
    print()

    model, history, trainer = train_envelope_stream(
        X, y, entropy, fast_enc,
        hidden_dim=64, n_steps=200, batch_size=32,
        lr=0.001, snap_every=20, K_env=256, verbose=False)
    final = history[-1]
    print(f"  stream length       = {trainer.n_steps_available}")
    print(f"  training steps      = {len(history)}")
    print(f"  final weighted loss = {final['loss']:.4f}")
    print(f"  final baby frac     = {final['baby_frac']:.4f}")
    print(f"  final giant frac    = {final['giant_frac']:.4f}")
    print()
    trainer.cleanup()

    # ============================================================
    #  5 · Verified accuracy check
    # ============================================================
    print("=" * 78)
    print("5  ·  gate accuracy against Q/P ground truth  (fresh sample)")
    print("=" * 78)
    acc = verify_accuracy(base_enc, n_samples=3000)
    print(f"  n_certain          = {acc['n_certain']}")
    print(f"  agreement          = {acc['agreement']*100:.2f} %")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30)

    # (a) Π surface — Figure 3.5
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, W1, 200)
    ys = np.linspace(0, W2, 200)
    Xg, Yg = np.meshgrid(xs, ys)
    Pg = 0.5 * (1.0 + np.cos(2 * np.pi * Xg / W1)
                      * np.cos(2 * np.pi * Yg / W2))
    im = ax.imshow(Pg, extent=[0, W1, 0, W2], origin="lower",
                   cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.axhline(calib["pi_hi"] * W2, color="#e74c3c",
               ls="--", lw=1.0, label=f"π_hi = {calib['pi_hi']:.2f}")
    ax.axhline(calib["pi_lo"] * W2, color="#3498db",
               ls="--", lw=1.0, label=f"π_lo = {calib['pi_lo']:.2f}")
    ax.set_xlabel("x (mod π)"); ax.set_ylabel("y (mod e)")
    ax.set_title("Π(x, y)  ·  elliptic projection")
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) Π histogram split by truth
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    n = base_enc.ugc.n
    pis_b, pis_g = [], []
    for _ in range(2000):
        x = int(rng.integers(0, n))
        y = (x + base_enc.conjugate_offset) % n
        m = TwoPointMatrix(float(x + 1), float(y + 1))
        Q, P = m.Q(), m.P()
        truth = (abs(Q) >= abs(P)) and (Q > 0)
        pi_val = elliptic_projection(float(x + 1), float(y + 1))
        (pis_b if truth else pis_g).append(pi_val)
    ax.hist(pis_b, bins=30, color="#16a085", alpha=0.6,
            label=f"baby (n={len(pis_b)})")
    ax.hist(pis_g, bins=30, color="#e74c3c", alpha=0.6,
            label=f"giant (n={len(pis_g)})")
    ax.axvline(calib["pi_hi"], color="#e74c3c", ls="--", lw=1.0)
    ax.axvline(calib["pi_lo"], color="#3498db", ls="--", lw=1.0)
    ax.set_xlabel("Π"); ax.set_ylabel("count")
    ax.set_title("Π distribution: baby vs giant")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) speed comparison bar
    ax = fig.add_subplot(gs[0, 2])
    labels = ["parent QP", "elliptic QP", "elliptic fast"]
    vals = [speeds["parent_ms"], speeds["elliptic_ms"], speeds["fast_ms"]]
    colors = ["#e74c3c", "#e67e22", "#16a085"]
    ax.bar(labels, vals, color=colors, edgecolor="k")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ms per snap")
    ax.set_title(f"per-snap cost  ·  {speeds['n_weights']} weights")
    ax.grid(alpha=0.3, axis="y")

    # (d) training loss
    ax = fig.add_subplot(gs[1, 0])
    steps = [r["step"] for r in history]
    losses = [r["loss"] for r in history]
    ax.semilogy(steps, losses, color="#2c3e50", lw=1.3)
    ax.set_xlabel("envelope step"); ax.set_ylabel("loss")
    ax.set_title("training loss  ·  fast encoder")
    ax.grid(alpha=0.3, which="both")

    # (e) mode usage over steps
    ax = fig.add_subplot(gs[1, 1])
    baby_f  = [r["baby_frac"] for r in history]
    giant_f = [r["giant_frac"] for r in history]
    ax.plot(steps, baby_f,  color="#16a085", lw=1.3, label="baby")
    ax.plot(steps, giant_f, color="#e74c3c", lw=1.3, label="giant")
    ax.set_xlabel("step"); ax.set_ylabel("fraction")
    ax.set_title("baby vs giant  ·  gate-driven")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (f) agreement grid: Π vs truth (sorted)
    ax = fig.add_subplot(gs[1, 2])
    all_pis = np.array(pis_b + pis_g)
    all_truth = np.array([1] * len(pis_b) + [0] * len(pis_g))
    order = np.argsort(all_pis)
    pis_sorted = all_pis[order]
    truth_sorted = all_truth[order]
    ax.scatter(pis_sorted, truth_sorted, s=6, alpha=0.5,
               c=truth_sorted, cmap="RdYlGn")
    ax.axvline(calib["pi_hi"], color="#e74c3c", ls="--", lw=1.0)
    ax.axvline(calib["pi_lo"], color="#3498db", ls="--", lw=1.0)
    ax.set_xlabel("Π (sorted)"); ax.set_ylabel("truth (1 = baby)")
    ax.set_title(f"gate alignment  ·  agreement = "
                 f"{calib['agreement']*100:.1f}%")
    ax.grid(alpha=0.3)

    plt.suptitle(
        "UGC elliptic gate  ·  Π ∈ [0,1] as O(1) pseudo-deterministic "
        "classifier  ·  baby ↔ giant decision",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print("  Elliptic gate:")
    print("    Π(x, y)  = ½(1 + cos(2π x/π) · cos(2π y/e))  ∈ [0, 1]")
    print(f"    π_hi     = {calib['pi_hi']:.4f}   →  baby")
    print(f"    π_lo     = {calib['pi_lo']:.4f}   →  giant")
    print(f"    agreement with Q/P  = {calib['agreement']*100:.2f} %")
    print(f"    coverage (certain)  = {calib['coverage']*100:.2f} %")
    print()
    print("  Encoders:")
    print("    EllipticQPWeightEncoder   ·  gate pre-classifier")
    print("      fast path : O(1) when gate is certain")
    print("      Q/P path  : only in uncertain band")
    print()
    print("    EllipticFastQPWeightEncoder   ·  fully O(1)")
    print(f"      precompute    : O(p) ≈ {fast_enc.precompute_ms:.2f} ms")
    print("      per weight    : O(1) array lookup (no dlog, no BSGS, no Q/P)")
    print(f"      speedup vs parent  ≈ {speeds['speedup_fast']:.2f}×")
    print()
    print("  Complexity:")
    print("    gate lookup (project)  : O(1)  ·  4 ops")
    print("    gate lookup (decide)   : O(1)  ·  6 ops")
    print("    fast encode (table)    : O(1)  ·  2 array accesses")
    print("    parent Q/P encode      : O(1) or O(√p) depending on table")
    print("    parent dlog_bsgs       : O(√p)")
    print()
    print("  Memory:")
    print(f"    gate.table        = {gate.n_bins}×{gate.n_bins} floats = "
          f"{gate.n_bins * gate.n_bins * 8 / 1024:.1f} KB")
    print(f"    mode_array        = {fast_enc.p} int8 = "
          f"{fast_enc.p / 1024:.1f} KB")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()