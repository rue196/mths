#!/usr/bin/env python3
"""
UGC_discrete_train_envelope.py
==============================

Extends UGC_discrete_train.py with:

  1. Envelope-streamed training     (file_reader_envelope_sat.READ)
  2. Q/P decomposition              (complete_planes_spectral_sum.TwoPointMatrix)
  3. Z-walk finite-step             (complete_planes_spectral_sum.Z_forward/central)

Per weight w:
    h   = quantise(w)           ∈ Z_p^*
    x   = dlog_g(h)             UGC vertex (from table or BSGS)
    y   = (x + n//2) % n        paired vertex

    M   = TwoPointMatrix(x+1, y+1)     # positive real arguments for ln

    Q   = Re(tr M) = x^-1 cos ln x + y^-1 cos ln y    algebraic, real
    P   = Im(tr M) = x^-1 sin ln x + y^-1 sin ln y    transcendental, imag

Decision:
    baby step   ⇔  |Q| ≥ |P|  AND  Q > 0     (algebraic-real dominant)
    giant step  ⇔  |P| > |Q|  OR   Q < 0     (transcendental / cos < 0 band)

|Ci| split:
    sym  = Q  →  positive half  (algebraic-real)
    asym = P  →  negative half  (transcendental-imag)

Giant-step Z-walk  O(√z):
    z        = |x_target − x_start|
    n_steps  = ⌈√z⌉          matches BSGS O(√n)
    at each step:
        Z_forward(t) = (ζ(t+A) − ζ(t)) / A
        Z_central(t) = (ζ(t+A) − ζ(t−A)) / (2A)

Imports (unchanged parents)
---------------------------
    UGC_discrete_train              UGCWeightEncoder, UGCWeightTensor,
                                    UGCWeightsLinear, SpectralNNWithUGCWeights
    complete_planes_spectral_sum    TwoPointMatrix, Z_forward, Z_central,
                                    A_STEP, build_spectral_coefficients
    file_reader_envelope_sat        EnvelopeFactory, AccessMode
"""

from __future__ import annotations
import math
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- parent encoders ----
from UGC_discrete_train import (
    UGCWeightEncoder,
    UGCWeightTensor,
    UGCWeightsLinear,
    SpectralNNWithUGCWeights,
)

# ---- envelope streaming ----
from file_reader_envelope_sat import (
    EnvelopeFactory,
    AccessMode,
)

# ---- Q/P decomposition + Z-walk ----
from complete_planes_spectral_sum import (
    TwoPointMatrix,
    Z_forward,
    Z_central,
    A_STEP,
    build_spectral_coefficients,
)


# ============================================================
#  QPWeightEncoder  ·  Q/P-based baby vs giant decision
# ============================================================
class QPWeightEncoder(UGCWeightEncoder):
    """
    Extends UGCWeightEncoder with the Q/P physics decision.

    Baby step  ⇔  |Q| ≥ |P|  AND  Q > 0   (algebraic-real dominant)
    Giant step ⇔  |P| > |Q|  OR   Q < 0   (transcendental / cos<0 band)
    """

    def __init__(self, p: int = 10007, g: int = 5,
                 w_min: float = -1.0, w_max: float = 1.0,
                 precompute_fraction: float = 0.9,
                 conjugate_offset: Optional[int] = None,
                 K_sieve: Optional[int] = None):
        super().__init__(
            p=p, g=g, w_min=w_min, w_max=w_max,
            precompute_fraction=precompute_fraction,
            K_sieve=K_sieve)
        self.conjugate_offset = (
            int(conjugate_offset) if conjugate_offset is not None
            else self.ugc.n // 2)

    # --------------------------------------------------------
    def encode(self, w: float) -> Dict:
        h = self.weight_to_h(w)

        # ---- resolve x (vertex position) ----
        x_lookup = self.direct_table.get(h)
        if x_lookup is not None:
            x = int(x_lookup)
        else:
            x_bsgs, _, _ = self.ugc.dlog_bsgs(h)
            x = int(x_bsgs) if x_bsgs is not None else None

        if x is None:
            return dict(
                mode='fail', weight=float(w), h=int(h),
                x=None, y=None, Q=0.0, P=0.0,
                sym=0.0, asym=0.0,
                vertex_scalar=None,
                weight_reconstructed=float(w), residual=0.0,
            )

        # ---- paired vertex ----
        n = self.ugc.n
        y = (x + self.conjugate_offset) % n

        # ---- Q/P decomposition via TwoPointMatrix ----
        x_real = float(x + 1)          # positive reals for ln()
        y_real = float(y + 1)
        m = TwoPointMatrix(x_real, y_real)
        Q = float(m.Q())
        P = float(m.P())

        # ---- physics decision ----
        algebraic_dominant = (abs(Q) >= abs(P)) and (Q > 0.0)
        decided_mode = 'baby' if algebraic_dominant else 'giant'

        # ---- record ----
        lv = (-float(self.ugc.H[x + 1])
              if x < len(self.ugc.H) else 0.0)

        return dict(
            mode=decided_mode,
            weight=float(w), h=int(h),
            x=int(x), y=int(y),
            Q=Q, P=P,
            sym=Q,     # |Ci| positive half  (algebraic-real)
            asym=P,    # |Ci| negative half  (transcendental-imag)
            vertex_scalar=lv,
            weight_reconstructed=float(self.h_to_weight(h)),
            residual=abs(w - self.h_to_weight(h)),
        )


# ============================================================
#  Giant-step Z-walk  ·  O(√z)
# ============================================================
def giant_step_walk(x_start: float, x_target: float,
                    C: np.ndarray,
                    K: Optional[int] = None,
                    max_steps: Optional[int] = None) -> List[Dict]:
    """
    Walk from x_start to x_target on the Z-plane using
        Z_forward(t) = (ζ(t+A) − ζ(t)) / A
        Z_central(t) = (ζ(t+A) − ζ(t−A)) / (2A)

    Steps = ⌈√z⌉ where z = |x_target − x_start|   (matches BSGS).
    """
    if K is None:
        K = (len(C) - 1) // 2

    z = max(1.0, abs(x_target - x_start))
    n_steps = max(1, int(math.ceil(math.sqrt(z))))
    if max_steps is not None:
        n_steps = min(n_steps, max_steps)

    direction = 1.0 if x_target >= x_start else -1.0
    dt = z / n_steps * direction

    t = float(x_start)
    path: List[Dict] = []
    for _ in range(n_steps):
        Zf = Z_forward(t, C, K=K)
        Zc = Z_central(t, C, K=K)
        path.append(dict(
            t=t,
            Z_forward_real=float(Zf.real),
            Z_forward_imag=float(Zf.imag),
            Z_central_real=float(Zc.real),
            Z_central_imag=float(Zc.imag),
        ))
        t += dt
    return path


# ============================================================
#  EnvelopeStreamTrainer  ·  constant-rate batch streaming
# ============================================================
class EnvelopeStreamTrainer:
    """
    Streams training batches from a READ envelope's 1D path.

    Each step:
        q, w_ell      = envelope.kept_qs[i], envelope.weights[i]
        batch_start   = q mod N
        batch_idx     = [batch_start, …, batch_start + B − 1] mod N
        importance    = 1 + w_ell
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 entropy: np.ndarray,
                 encoder: UGCWeightEncoder,
                 model: nn.Module,
                 K_env: int = 256,
                 temp_prefix: str = "ugc_train_"):
        self.X = X
        self.y = y
        self.entropy = entropy
        self.encoder = encoder
        self.model = model
        self.N = len(X)

        # ---- serialize training data to temp file for EnvelopeFactory ----
        fd, self.temp_path = tempfile.mkstemp(
            prefix=temp_prefix, suffix=".bin")
        os.close(fd)
        with open(self.temp_path, "wb") as f:
            f.write(np.ascontiguousarray(X, np.float32).tobytes())
            f.write(np.ascontiguousarray(y, np.float32).tobytes())
            f.write(np.ascontiguousarray(entropy, np.float32).tobytes())

        # ---- build READ envelope ----
        self.factory = EnvelopeFactory(K=K_env)
        self.envelope = self.factory.read_envelope(self.temp_path)
        self.n_steps_available = len(self.envelope.kept_qs)

    def cleanup(self):
        if os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except OSError:
                pass

    def stream_step(self, step: int, batch_size: int,
                    criterion, optimizer) -> Dict:
        """One training step at envelope index `step`."""
        i       = step % self.n_steps_available
        q       = self.envelope.kept_qs[i]
        w_ell   = self.envelope.weights[i]

        idx       = int(q) % self.N
        batch_idx = [(idx + k) % self.N for k in range(batch_size)]

        bX = torch.tensor(self.X[batch_idx], dtype=torch.float32)
        bY = torch.tensor(
            np.hstack([self.y[batch_idx], self.entropy[batch_idx]]),
            dtype=torch.float32)

        self.model.train()
        optimizer.zero_grad()
        pred = self.model(bX)
        loss = criterion(pred, bY) * (1.0 + float(w_ell))
        loss.backward()
        optimizer.step()

        return dict(
            step=step, q=int(q), w_ell=float(w_ell),
            loss=float(loss.item()),
        )


# ============================================================
#  train_envelope_stream
# ============================================================
def train_envelope_stream(
        X: np.ndarray,
        y: np.ndarray,
        entropy: np.ndarray,
        encoder: UGCWeightEncoder,
        hidden_dim: int = 64,
        n_steps: int = 300,
        batch_size: int = 32,
        lr: float = 0.001,
        snap_every: int = 20,
        K_env: int = 256,
        verbose: bool = True,
) -> Tuple[nn.Module, List[Dict], EnvelopeStreamTrainer]:
    """
    Train at envelope rate — one step per envelope (q, w) pair.
    Model snaps to UGC weights every `snap_every` steps.
    """
    model = SpectralNNWithUGCWeights(
        input_dim=X.shape[1], hidden_dim=hidden_dim,
        encoder=encoder, output_dim=2, snap_every=snap_every)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = EnvelopeStreamTrainer(
        X, y, entropy, encoder, model, K_env=K_env)

    history: List[Dict] = []
    for step in range(n_steps):
        rec = trainer.stream_step(step, batch_size, criterion, optimizer)
        st = model.aggregate_stats()
        rec['baby_frac']  = st['baby_frac']
        rec['giant_frac'] = st['giant_frac']
        history.append(rec)
        if verbose and (step + 1) % 50 == 0:
            print(f"  step {step+1:>4d}/{n_steps}  "
                  f"loss={rec['loss']:.4f}  "
                  f"q={rec['q']:>5d}  "
                  f"w={rec['w_ell']:.3f}  "
                  f"baby={rec['baby_frac']:.2%}  "
                  f"giant={rec['giant_frac']:.2%}")

    return model, history, trainer


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("UGC envelope training  ·  Q/P decomposition  ·  Z-walk O(√z)")
    print("=" * 78)
    print()

    # ============================================================
    #  1 · QPWeightEncoder  ·  Q/P decomposition table
    # ============================================================
    print("=" * 78)
    print("1  ·  QPWeightEncoder  ·  per-weight Q/P decomposition")
    print("=" * 78)
    encoder = QPWeightEncoder(
        p=10007, g=5,
        w_min=-1.0, w_max=1.0,
        precompute_fraction=0.9)

    print(f"  p                  = {encoder.p}")
    print(f"  g                  = {encoder.g}")
    print(f"  n                  = {encoder.ugc.n}")
    print(f"  conjugate_offset   = {encoder.conjugate_offset}")
    print()
    print(f"  {'w':>6s}  {'h':>6s}  {'x':>6s}  {'y':>6s}  "
          f"{'Q':>11s}  {'P':>11s}  {'|Q|-|P|':>11s}  "
          f"{'mode':>6s}  {'sym':>9s}  {'asym':>9s}")
    print("  " + "-" * 94)
    for w_val in [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]:
        enc = encoder.encode(float(w_val))
        if enc['x'] is None:
            continue
        balance = abs(enc['Q']) - abs(enc['P'])
        print(f"  {w_val:>6.3f}  {enc['h']:>6d}  "
              f"{enc['x']:>6d}  {enc['y']:>6d}  "
              f"{enc['Q']:>+11.6f}  {enc['P']:>+11.6f}  "
              f"{balance:>+11.6f}  {enc['mode']:>6s}  "
              f"{enc['sym']:>+9.4f}  {enc['asym']:>+9.4f}")
    print()

    # ============================================================
    #  2 · Giant-step Z-walk
    # ============================================================
    print("=" * 78)
    print("2  ·  giant-step Z-walk  ·  O(√z)")
    print("=" * 78)
    K_ci = 20
    C = build_spectral_coefficients(K_ci)

    # pick a giant-step target (large-x end of the weight range)
    enc_giant = encoder.encode(0.95)
    if enc_giant['mode'] != 'giant' or enc_giant['x'] is None:
        enc_giant = encoder.encode(-0.95)

    x_start = 0.0
    x_target = float(enc_giant['x']) if enc_giant['x'] is not None else 100.0

    t0 = time.perf_counter()
    path = giant_step_walk(x_start, x_target, C, K=K_ci)
    t_walk = (time.perf_counter() - t0) * 1e3

    print(f"  x_start            = {x_start:.0f}")
    print(f"  x_target           = {x_target:.0f}")
    print(f"  z = |Δx|           = {abs(x_target - x_start):.0f}")
    print(f"  n_steps = ⌈√z⌉     = {len(path)}")
    print(f"  walk time          = {t_walk:.2f} ms")
    print()
    print(f"  {'step':>4s}  {'t':>10s}  {'Re Z_fwd':>13s}  "
          f"{'Im Z_fwd':>13s}  {'Re Z_ctr':>13s}")
    print("  " + "-" * 62)
    for k in range(min(8, len(path))):
        p = path[k]
        print(f"  {k:>4d}  {p['t']:>10.4f}  "
              f"{p['Z_forward_real']:>+13.6f}  "
              f"{p['Z_forward_imag']:>+13.6f}  "
              f"{p['Z_central_real']:>+13.6f}")
    print()

    # ============================================================
    #  3 · Dataset + envelope-stream training
    # ============================================================
    print("=" * 78)
    print("3  ·  envelope-stream training  (file_reader_envelope_sat READ)")
    print("=" * 78)
    from entropic_neural_alternating import generate_spectral_dataset
    X, y, entropy = generate_spectral_dataset(
        num_samples=300, K_min=10, K_max=40, N_harmonics=20)
    print(f"  X shape            = {X.shape}")
    print(f"  y shape            = {y.shape}")
    print(f"  entropy shape      = {entropy.shape}")
    print()

    model, history, trainer = train_envelope_stream(
        X, y, entropy, encoder,
        hidden_dim=64, n_steps=300, batch_size=32,
        lr=0.001, snap_every=20, K_env=256,
        verbose=True)

    final = history[-1]
    print()
    print(f"  envelope stream length  = {trainer.n_steps_available}  "
          f"(kept_qs)")
    print(f"  training steps          = {len(history)}")
    print(f"  final weighted loss     = {final['loss']:.4f}")
    print(f"  final baby frac         = {final['baby_frac']:.4f}")
    print(f"  final giant frac        = {final['giant_frac']:.4f}")
    print()

    trainer.cleanup()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30)

    # (a) Q / P along weight axis
    ax = fig.add_subplot(gs[0, 0])
    w_grid = np.linspace(-1.0, 1.0, 80)
    Qs, Ps = [], []
    for w_val in w_grid:
        enc = encoder.encode(float(w_val))
        Qs.append(enc['Q'])
        Ps.append(enc['P'])
    ax.plot(w_grid, Qs, color="#16a085", lw=1.3, label="Q (algebraic)")
    ax.plot(w_grid, Ps, color="#e74c3c", lw=1.3, label="P (transcendental)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("weight w")
    ax.set_ylabel("Q / P")
    ax.set_title("Q vs P along weight axis")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) decision regions
    ax = fig.add_subplot(gs[0, 1])
    absQ = np.abs(Qs); absP = np.abs(Ps)
    colors = ["#16a085" if (abs(q) >= abs(p) and q > 0) else "#e74c3c"
              for q, p in zip(Qs, Ps)]
    ax.scatter(w_grid, absQ, c=colors, s=22, edgecolors="k",
               linewidths=0.4)
    ax.plot(w_grid, absP, color="gray", lw=1.0, alpha=0.6,
            label="|P|")
    ax.set_xlabel("weight w")
    ax.set_ylabel("|Q|")
    ax.set_title("Decision: green = baby, red = giant")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) Z-walk trajectory
    ax = fig.add_subplot(gs[0, 2])
    if len(path) > 0:
        ts   = [p['t'] for p in path]
        zfr  = [p['Z_forward_real'] for p in path]
        zcr  = [p['Z_central_real'] for p in path]
        ax.plot(ts, zfr, "o-", color="#3498db", lw=1.2, ms=3,
                label="Re Z_forward")
        ax.plot(ts, zcr, "s--", color="#e74c3c", lw=1.0, ms=3,
                alpha=0.7, label="Re Z_central")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("Re Z")
    ax.set_title(f"Z-walk  ·  0→{x_target:.0f}  ·  "
                 f"{len(path)} steps  = ⌈√z⌉")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) training loss
    ax = fig.add_subplot(gs[1, 0])
    steps = [r['step'] for r in history]
    losses = [r['loss'] for r in history]
    ax.semilogy(steps, losses, color="#2c3e50", lw=1.3)
    ax.set_xlabel("envelope step"); ax.set_ylabel("loss")
    ax.set_title("Envelope-stream training loss")
    ax.grid(alpha=0.3, which="both")

    # (e) baby/giant fractions over steps
    ax = fig.add_subplot(gs[1, 1])
    baby_f  = [r['baby_frac']  for r in history]
    giant_f = [r['giant_frac'] for r in history]
    ax.plot(steps, baby_f,  color="#16a085", lw=1.3, label="baby")
    ax.plot(steps, giant_f, color="#e74c3c", lw=1.3, label="giant")
    ax.set_xlabel("envelope step"); ax.set_ylabel("fraction")
    ax.set_title("Baby vs giant fraction (physics-based)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (f) per-step envelope (q, w_ell)
    ax = fig.add_subplot(gs[1, 2])
    qs = [r['q'] for r in history]
    ws = [r['w_ell'] for r in history]
    ax2 = ax.twinx()
    ax.plot(steps, qs, color="#8e44ad", lw=1.0, alpha=0.7,
            label="q (address)")
    ax2.plot(steps, ws, color="#e67e22", lw=1.0, alpha=0.7,
             label="w_ell")
    ax.set_xlabel("envelope step")
    ax.set_ylabel("q", color="#8e44ad")
    ax2.set_ylabel("w_ell", color="#e67e22")
    ax.set_title("Envelope stream  (q, w_ell) per step")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(
        "UGC envelope training  ·  Q = algebraic-real → baby, "
        "P = transcendental-imag → giant  ·  Z-walk O(√z)",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print("  Q/P decomposition  (TwoPointMatrix):")
    print("    Q = Re(tr M) = x^-1 cos ln x + y^-1 cos ln y   algebraic-real")
    print("    P = Im(tr M) = x^-1 sin ln x + y^-1 sin ln y   transcendental-imag")
    print()
    print("  decision:")
    print("    baby step   ⇔  |Q| ≥ |P|  AND  Q > 0     (algebraic-real dominant)")
    print("    giant step  ⇔  |P| > |Q|  OR   Q < 0     (transcendental / cos<0)")
    print()
    print("  |Ci| split:")
    print("    sym   =  Q   →  positive half   (algebraic-real)")
    print("    asym  =  P   →  negative half   (transcendental-imag)")
    print()
    print("  envelope streaming  (file_reader_envelope_sat READ):")
    print(f"    stream length   = {trainer.n_steps_available}  (kept_qs)")
    print(f"    training steps  = {len(history)}")
    print("    batch start     = q_i mod N")
    print("    importance      = 1 + w_i")
    print()
    print("  giant-step Z-walk:")
    print("    Z_forward(t)  = (ζ(t+A) − ζ(t)) / A")
    print("    Z_central(t)  = (ζ(t+A) − ζ(t−A)) / (2A)")
    print(f"    A_STEP        = {A_STEP:.6f}")
    print("    steps         = ⌈√z⌉  (matches BSGS O(√n))")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()