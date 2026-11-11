#!/usr/bin/env python3
"""
entropic_ugc_neural.py
======================

Extends entropic-neural-alternating.py with discrete log training
tensors on the UGC graph.

Core idea
---------
Every network weight becomes a vertex on the UGC graph:

    weight w  →  target h = quantise(w) ∈ Z_p^*
    h         →  discrete log x, via  L_i · g^{shift} = R_j
    vertex x  →  scalar  v(x) = −H_{x+1}   (negative harmonic)

The vertex scalar is the *physical amplitude* of the weight in
harmonic space; the discrete log position x is how it is stored and
transmitted.

Two problem-solving modes
-------------------------
    baby step   h is in the restricted precomputed table (small x)
                → O(1) direct lookup  (pseudo-deterministic)
    giant step  h needs BSGS search (large x)
                → O(√n) walk through  L_i · g^shift = R_j
                  (complex problem solving)

`precompute_fraction` splits the table: weights whose h lands
inside keep baby-step; the rest require giant-step.

Imports (unchanged parents)
---------------------------
    entropic-neural-alternating.py    mobius_sieve, harmonic_numbers,
                                      compute_supertrace_and_entropy,
                                      generate_spectral_dataset,
                                      SpectralNNWithEntropy, train_model
    ugc_discrete_log.py               UGCDiscreteLog, UGCVertex,
                                      harmonic_numbers, modinv, egcd
"""

from __future__ import annotations
import importlib.util
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from entropic_neural_alternating import(
    mobius_sieve,
    harmonic_numbers,
    build_coeffs,
    zeta_at,
    compute_supertrace_and_entropy,
    generate_spectral_dataset,
    SpectralNNWithEntropy,
    train_model,
    ALPHA,
    PI,
    E
   
)

# ---- UGC discrete log ----
from ugc_discrete_log import (
    UGCDiscreteLog,
    UGCVertex,
    harmonic_numbers as ugc_harmonic_numbers,
    modinv,
    egcd,
)


# ============================================================
#  UGCWeightEncoder
# ============================================================
class UGCWeightEncoder:
    """
    Encode network weights as discrete log positions on the UGC graph.

    Each weight w ∈ [w_min, w_max] is mapped to a target h ∈ Z_p^*;
    its discrete log x = dlog_g(h) gives the UGC vertex whose
    scalar is  −H_{x+1}.
    """

    def __init__(self,
                 p: int = 10007,
                 g: int = 5,
                 w_min: float = -1.0,
                 w_max: float = 1.0,
                 precompute_fraction: float = 0.9,
                 K_sieve: Optional[int] = None):
        self.ugc = UGCDiscreteLog(p=p, g=g, K_sieve=K_sieve)
        self.p   = int(p)
        self.g   = int(g)
        self.w_min = float(w_min)
        self.w_max = float(w_max)

        if not 0.0 < precompute_fraction <= 1.0:
            raise ValueError(
                f"precompute_fraction must be in (0, 1]; "
                f"got {precompute_fraction}")
        self.precompute_fraction = float(precompute_fraction)

        # restrict direct lookup to small exponents
        n_total = len(self.ugc.log_table)
        n_keep  = max(1, int(n_total * precompute_fraction))
        self.n_keep = n_keep
        self.direct_table = {
            k: v for k, v in self.ugc.log_table.items() if v < n_keep
        }

    # ---------- mapping ----------
    def weight_to_h(self, w: float) -> int:
        w_norm = (w - self.w_min) / (self.w_max - self.w_min + 1e-12)
        w_norm = float(np.clip(w_norm, 0.0, 1.0))
        return 1 + int(round(w_norm * (self.p - 2)))

    def h_to_weight(self, h: int) -> float:
        return self.w_min + ((h - 1) / (self.p - 2)) * \
               (self.w_max - self.w_min)

    # ---------- encode ----------
    def encode(self, w: float) -> Dict:
        """Encode one weight (auto-selects baby vs giant)."""
        h = self.weight_to_h(w)

        # ----- baby step  ·  direct table lookup -----
        x = self.direct_table.get(h)
        if x is not None:
            lv = -self.ugc.H[x + 1]
            return dict(
                mode='baby',
                weight=float(w),
                h=int(h), x=int(x), ij=None,
                vertex_scalar=float(lv),
                weight_reconstructed=float(self.h_to_weight(h)),
                residual=abs(w - self.h_to_weight(h)),
            )

        # ----- giant step  ·  BSGS search -----
        x, lv, ij = self.ugc.dlog_bsgs(h)
        if x is not None:
            return dict(
                mode='giant',
                weight=float(w),
                h=int(h), x=int(x), ij=ij,
                vertex_scalar=float(lv) if lv is not None else 0.0,
                weight_reconstructed=float(self.h_to_weight(h)),
                residual=abs(w - self.h_to_weight(h)),
            )

        # fallback (g is a generator, so this should not happen)
        return dict(
            mode='fail',
            weight=float(w),
            h=int(h), x=None, ij=None,
            vertex_scalar=None,
            weight_reconstructed=float(w),
            residual=0.0,
        )

    def decode(self, x: int) -> float:
        """Recover weight from discrete log position x."""
        if x < 0 or x >= len(self.ugc.powers):
            return self.w_min
        return self.h_to_weight(int(self.ugc.powers[x]))

    def encode_array(self, W: np.ndarray) -> List[Dict]:
        flat = np.asarray(W, dtype=np.float64).flatten().tolist()
        return [self.encode(float(w)) for w in flat]


# ============================================================
#  UGCWeightTensor
# ============================================================
class UGCWeightTensor:
    """
    2D tensor of weights stored as UGC vertex encodings.

    Each entry carries (mode, x, vertex_scalar).  The vertex scalar
    −H_{x+1} is the "physical amplitude" of the weight in harmonic
    space.
    """

    def __init__(self, encodings: List[Dict], shape: Tuple[int, int]):
        self.encodings = encodings
        self.shape = shape

    @classmethod
    def from_weights(cls, W: np.ndarray,
                     encoder: UGCWeightEncoder) -> "UGCWeightTensor":
        flat = np.asarray(W, dtype=np.float64).flatten().tolist()
        enc  = [encoder.encode(float(w)) for w in flat]
        return cls(enc, tuple(np.asarray(W).shape))

    def to_weights(self) -> np.ndarray:
        flat = [e['weight_reconstructed'] for e in self.encodings]
        return np.array(flat, dtype=np.float32).reshape(self.shape)

    def vertex_scalars(self) -> np.ndarray:
        flat = [e['vertex_scalar'] if e['vertex_scalar'] is not None
                else 0.0 for e in self.encodings]
        return np.array(flat, dtype=np.float32).reshape(self.shape)

    def vertex_positions(self) -> np.ndarray:
        flat = [e['x'] if e['x'] is not None else -1
                for e in self.encodings]
        return np.array(flat, dtype=np.int64).reshape(self.shape)

    def baby_fraction(self) -> float:
        n = len(self.encodings)
        return sum(1 for e in self.encodings
                   if e['mode'] == 'baby') / max(n, 1)

    def giant_fraction(self) -> float:
        n = len(self.encodings)
        return sum(1 for e in self.encodings
                   if e['mode'] == 'giant') / max(n, 1)

    def mode_counts(self) -> Dict[str, int]:
        counts = {'baby': 0, 'giant': 0, 'fail': 0}
        for e in self.encodings:
            counts[e['mode']] = counts.get(e['mode'], 0) + 1
        return counts

    def residual_norm(self) -> float:
        return math.sqrt(sum(e['residual'] ** 2
                             for e in self.encodings))


# ============================================================
#  UGCWeightsLinear
# ============================================================
class UGCWeightsLinear(nn.Module):
    """
    Linear layer whose weights are periodically snapped to UGC values.

    After every `snap_every` forward passes, each weight is
    re-encoded via the UGCWeightEncoder and reset to the nearest
    UGC-representable value.  Baby vs giant step counts are tracked.
    """

    def __init__(self, in_features: int, out_features: int,
                 encoder: UGCWeightEncoder,
                 snap_every: int = 100):
        super().__init__()
        self.linear    = nn.Linear(in_features, out_features)
        self.encoder   = encoder
        self.snap_every = int(snap_every)
        self.step = 0
        self.baby_count  = 0
        self.giant_count = 0
        self.fail_count  = 0
        self.snap_history: List[Dict] = []

    def forward(self, x):
        self.step += 1
        if self.snap_every > 0 and self.step % self.snap_every == 0:
            self._snap_weights()
        return self.linear(x)

    def _snap_weights(self):
        with torch.no_grad():
            W = self.linear.weight
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    enc = self.encoder.encode(float(W[i, j]))
                    self._tally(enc['mode'])
                    if enc['x'] is not None:
                        W[i, j] = enc['weight_reconstructed']
            b = self.linear.bias
            if b is not None:
                for i in range(b.shape[0]):
                    enc = self.encoder.encode(float(b[i]))
                    self._tally(enc['mode'])
                    if enc['x'] is not None:
                        b[i] = enc['weight_reconstructed']
        self.snap_history.append(self.stats())

    def _tally(self, mode: str):
        if mode == 'baby':
            self.baby_count += 1
        elif mode == 'giant':
            self.giant_count += 1
        else:
            self.fail_count += 1

    def stats(self) -> Dict:
        total = self.baby_count + self.giant_count + self.fail_count
        return dict(
            total=total,
            baby=self.baby_count,
            giant=self.giant_count,
            fail=self.fail_count,
            baby_frac=self.baby_count / max(total, 1),
            giant_frac=self.giant_count / max(total, 1),
        )

    def as_ugc_tensor(self) -> UGCWeightTensor:
        W = self.linear.weight.detach().cpu().numpy()
        return UGCWeightTensor.from_weights(W, self.encoder)


# ============================================================
#  SpectralNNWithUGCWeights
# ============================================================
class SpectralNNWithUGCWeights(nn.Module):
    """
    Multi-output NN (sum_z, entropy) whose linear layers are
    UGC-encoded with periodic weight snapping.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 encoder: UGCWeightEncoder,
                 output_dim: int = 2,
                 snap_every: int = 100):
        super().__init__()
        self.encoder = encoder
        self.layers = nn.ModuleList([
            UGCWeightsLinear(input_dim, hidden_dim,
                             encoder, snap_every),
            UGCWeightsLinear(hidden_dim, hidden_dim,
                             encoder, snap_every),
            UGCWeightsLinear(hidden_dim, output_dim,
                             encoder, snap_every),
        ])
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.layers[0](x))
        x = self.act(self.layers[1](x))
        return self.layers[2](x)

    def aggregate_stats(self) -> Dict:
        total_baby  = sum(l.baby_count  for l in self.layers)
        total_giant = sum(l.giant_count for l in self.layers)
        total_fail  = sum(l.fail_count  for l in self.layers)
        total = total_baby + total_giant + total_fail
        return dict(
            total=total,
            baby=total_baby,
            giant=total_giant,
            fail=total_fail,
            baby_frac=total_baby  / max(total, 1),
            giant_frac=total_giant / max(total, 1),
        )


# ============================================================
#  Training with UGC tracking
# ============================================================
def train_model_ugc(X: np.ndarray,
                    y: np.ndarray,
                    entropy: np.ndarray,
                    encoder: UGCWeightEncoder,
                    hidden_dim: int = 64,
                    epochs: int = 100,
                    batch_size: int = 32,
                    lr: float = 0.001,
                    snap_every: int = 100
                    ) -> Tuple[nn.Module, List[float], List[float], List[Dict]]:
    """Train with UGC-snapped weights + per-epoch baby/giant tracking."""
    Y = np.hstack([y, entropy])

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32))
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = SpectralNNWithUGCWeights(
        input_dim=X.shape[1], hidden_dim=hidden_dim,
        encoder=encoder, output_dim=2, snap_every=snap_every)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses: List[float] = []
    test_losses:  List[float] = []
    epoch_stats:  List[Dict]  = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_Y)
                test_loss += loss.item() * batch_X.size(0)
        test_losses.append(test_loss / len(test_loader.dataset))

        st = model.aggregate_stats()
        st['epoch'] = epoch + 1
        st['train_loss'] = train_losses[-1]
        st['test_loss']  = test_losses[-1]
        epoch_stats.append(st)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3d}/{epochs}  "
                  f"Train={train_losses[-1]:.4f}  "
                  f"Test={test_losses[-1]:.4f}  "
                  f"baby={st['baby_frac']:.2%}  "
                  f"giant={st['giant_frac']:.2%}")

    return model, train_losses, test_losses, epoch_stats


# ============================================================
#  Demo
# ============================================================
def main():
    print("=" * 78)
    print("Entropic UGC neural  ·  discrete log training tensors")
    print("=" * 78)
    print(f"  α = 1/(π − e)  = {ALPHA:.6f}")
    print()

    # ============================================================
    #  1 · encoder sanity
    # ============================================================
    print("=" * 78)
    print("1  ·  UGC weight encoder  ·  baby vs giant step split")
    print("=" * 78)
    encoder = UGCWeightEncoder(
        p=10007, g=5,
        w_min=-1.0, w_max=1.0,
        precompute_fraction=0.9)

    print(f"  p                    = {encoder.p}")
    print(f"  g                    = {encoder.g}")
    print(f"  n                    = {encoder.ugc.n}")
    print(f"  precompute_fraction  = {encoder.precompute_fraction}")
    print(f"  direct_table size    = {len(encoder.direct_table)}")
    print(f"  n_keep (max baby x)  = {encoder.n_keep}")
    print()

    test_weights = [-0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9]
    print(f"  {'w':>6s}  {'h':>6s}  {'mode':>6s}  {'x':>6s}  "
          f"{'vertex_scalar':>14s}  {'w_rec':>8s}  {'resid':>10s}")
    print("  " + "-" * 68)
    for w in test_weights:
        enc  = encoder.encode(w)
        x_s  = str(enc['x']) if enc['x'] is not None else "—"
        vs_s = f"{enc['vertex_scalar']:+.6f}" \
               if enc['vertex_scalar'] is not None else "—"
        print(f"  {w:>6.3f}  {enc['h']:>6d}  {enc['mode']:>6s}  "
              f"{x_s:>6s}  {vs_s:>14s}  "
              f"{enc['weight_reconstructed']:>8.4f}  "
              f"{enc['residual']:>10.2e}")
    print()

    # ============================================================
    #  2 · UGCWeightTensor batch demo
    # ============================================================
    print("=" * 78)
    print("2  ·  UGCWeightTensor  ·  full matrix as UGC vertices")
    print("=" * 78)

    rng = np.random.default_rng(0)
    W_demo = rng.uniform(-1.0, 1.0, size=(8, 16)).astype(np.float32)

    t0 = time.perf_counter()
    tensor_ugc = UGCWeightTensor.from_weights(W_demo, encoder)
    t_enc = (time.perf_counter() - t0) * 1e3

    print(f"  shape            = {W_demo.shape}")
    print(f"  encode time      = {t_enc:.2f} ms")
    print(f"  mode counts      = {tensor_ugc.mode_counts()}")
    print(f"  baby fraction    = {tensor_ugc.baby_fraction():.4f}")
    print(f"  giant fraction   = {tensor_ugc.giant_fraction():.4f}")
    print(f"  residual ‖r‖₂    = {tensor_ugc.residual_norm():.4e}")

    vs = tensor_ugc.vertex_scalars()
    print(f"  vertex scalars:")
    print(f"    min            = {vs.min():.4f}")
    print(f"    max            = {vs.max():.4f}")
    print(f"    mean           = {vs.mean():.4f}")
    print(f"    std            = {vs.std():.4f}")
    print()

    # ============================================================
    #  3 · dataset
    # ============================================================
    print("=" * 78)
    print("3  ·  generate spectral dataset  (parent module)")
    print("=" * 78)
    X, y, entropy = generate_spectral_dataset(
        num_samples=500, K_min=10, K_max=60, N_harmonics=30)
    print(f"  features shape    = {X.shape}")
    print(f"  targets shape     = {y.shape}")
    print(f"  entropy range     = [{entropy.min():.4f}, "
          f"{entropy.max():.4f}]")
    print()

    # ============================================================
    #  4 · train
    # ============================================================
    print("=" * 78)
    print("4  ·  train SpectralNNWithUGCWeights")
    print("=" * 78)
    model, train_l, test_l, stats = train_model_ugc(
        X, y, entropy, encoder,
        hidden_dim=64, epochs=100,
        batch_size=32, lr=0.001,
        snap_every=100)

    final = stats[-1]
    print()
    print(f"  final train MSE   = {final['train_loss']:.4f}")
    print(f"  final test  MSE   = {final['test_loss']:.4f}")
    print(f"  final baby frac   = {final['baby_frac']:.4f}")
    print(f"  final giant frac  = {final['giant_frac']:.4f}")
    print(f"  total snaps       = {final['total']}")
    print()

    # ============================================================
    #  5 · inspect final weight tensors
    # ============================================================
    print("=" * 78)
    print("5  ·  per-layer UGC weight tensors")
    print("=" * 78)
    for i, layer in enumerate(model.layers):
        W = layer.linear.weight.detach().cpu().numpy()
        ut = UGCWeightTensor.from_weights(W, encoder)
        print(f"  layer {i}: shape = {W.shape}")
        print(f"    baby frac       = {ut.baby_fraction():.4f}")
        print(f"    giant frac      = {ut.giant_fraction():.4f}")
        print(f"    residual ‖r‖₂   = {ut.residual_norm():.4e}")
        print(f"    vertex x range  = "
              f"[{ut.vertex_positions().min()}, "
              f"{ut.vertex_positions().max()}]")
        print()

    # ============================================================
    #  6 · sample-encoding complexity vs entropy quartile
    # ============================================================
    print("=" * 78)
    print("6  ·  target encoding vs entropy quartile")
    print("=" * 78)
    # normalise targets to encoder range, then encode
    y_flat = y.flatten()
    y_min, y_max = float(y_flat.min()), float(y_flat.max())
    y_norm = (y_flat - y_min) / (y_max - y_min + 1e-12) * 2.0 - 1.0
    y_modes = [encoder.encode(float(v))['mode'] for v in y_norm]

    ent_flat = entropy.flatten()
    quartiles = np.quantile(ent_flat, [0.0, 0.25, 0.5, 0.75, 1.0])
    groups: Dict[str, List[str]] = {f"Q{i+1}": [] for i in range(4)}
    for ent_val, mode in zip(ent_flat, y_modes):
        idx = int(np.searchsorted(quartiles, ent_val, side='right')) - 1
        idx = max(0, min(3, idx))
        groups[f"Q{idx+1}"].append(mode)

    print(f"  {'quartile':>8s}  {'H_ent range':>22s}  "
          f"{'n':>5s}  {'baby':>5s}  {'giant':>6s}  "
          f"{'giant_frac':>10s}")
    print("  " + "-" * 66)
    for i, (name, modes) in enumerate(groups.items()):
        lo, hi = quartiles[i], quartiles[i + 1]
        n     = len(modes)
        baby  = sum(1 for m in modes if m == 'baby')
        giant = sum(1 for m in modes if m == 'giant')
        print(f"  {name:>8s}  [{lo:.4f}, {hi:.4f}]      "
              f"{n:>5d}  {baby:>5d}  {giant:>6d}  "
              f"{giant/max(n,1):>10.4f}")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

    # (a) encoder split along weight axis
    ax = fig.add_subplot(gs[0, 0])
    w_grid = np.linspace(-1.0, 1.0, 400)
    modes_grid = [encoder.encode(float(w))['mode'] for w in w_grid]
    baby_mask = np.array([m == 'baby' for m in modes_grid])
    ax.scatter(w_grid[baby_mask],
               np.full(baby_mask.sum(), 0.0),
               s=10, c='#16a085',
               label=f'baby ({baby_mask.sum()})')
    ax.scatter(w_grid[~baby_mask],
               np.full((~baby_mask).sum(), 0.0),
               s=10, c='#e74c3c',
               label=f'giant ({(~baby_mask).sum()})')
    ax.set_xlabel("weight w")
    ax.set_title(f"Baby / giant split along w\n"
                 f"precompute_fraction = {encoder.precompute_fraction}")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.grid(alpha=0.3, axis='x')

    # (b) demo weight matrix — vertex scalars
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(tensor_ugc.vertex_scalars(),
                   cmap="viridis", aspect="auto")
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.set_title("vertex scalars  −H_{x+1}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) vertex positions
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(tensor_ugc.vertex_positions(),
                   cmap="plasma", aspect="auto")
    ax.set_xlabel("col"); ax.set_ylabel("row")
    ax.set_title("discrete log positions  x")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) training losses
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(train_l, color='#3498db', lw=1.4, label='Train')
    ax.plot(test_l,  color='#e74c3c', lw=1.4, label='Test')
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    ax.set_title("Training losses  ·  UGC-snapped weights")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (e) baby / giant fractions over epochs
    ax = fig.add_subplot(gs[1, 1])
    ep = [s['epoch'] for s in stats]
    ax.plot(ep, [s['baby_frac']  for s in stats],
            color='#16a085', lw=1.4, label='baby')
    ax.plot(ep, [s['giant_frac'] for s in stats],
            color='#e74c3c', lw=1.4, label='giant')
    ax.set_xlabel("epoch"); ax.set_ylabel("fraction")
    ax.set_title("Baby vs giant step usage over training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (f) giant fraction by entropy quartile
    ax = fig.add_subplot(gs[1, 2])
    names = list(groups.keys())
    giant_fracs_q = [
        sum(1 for m in groups[n] if m == 'giant') / max(len(groups[n]), 1)
        for n in names
    ]
    ax.bar(names, giant_fracs_q,
           color=['#16a085', '#3498db', '#e67e22', '#e74c3c'],
           edgecolor='k')
    for i, v in enumerate(giant_fracs_q):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center",
                va="bottom", fontsize=9)
    ax.set_xlabel("entropy quartile")
    ax.set_ylabel("giant fraction")
    ax.set_title("Target encoding vs entropy quartile\n"
                 "(Q4 = highest H_ent)")
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        "Entropic UGC neural  ·  discrete log training tensors  ·  "
        "baby = pseudo-deterministic  ·  giant = complex  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  UGC encoder:")
    print(f"    p                = {encoder.p}")
    print(f"    g                = {encoder.g}")
    print(f"    precompute_frac  = {encoder.precompute_fraction}")
    print(f"    direct_table     = {len(encoder.direct_table)} entries")
    print()
    print(f"  Training:")
    print(f"    epochs           = {len(stats)}")
    print(f"    final train MSE  = {stats[-1]['train_loss']:.4f}")
    print(f"    final test  MSE  = {stats[-1]['test_loss']:.4f}")
    print(f"    final baby frac  = {stats[-1]['baby_frac']:.4f}")
    print(f"    final giant frac = {stats[-1]['giant_frac']:.4f}")
    print()
    print(f"  Key ideas:")
    print(f"    · each weight → discrete log position x on UGC graph")
    print(f"    · vertex scalar = −H_{{x+1}}  (harmonic amplitude)")
    print(f"    · baby step  = direct lookup in precomputed table")
    print(f"                   → O(1), pseudo-deterministic")
    print(f"    · giant step = BSGS search via  L_i · g^shift = R_j")
    print(f"                   → O(√n), complex problem solving")
    print()
    print("Done.")


if __name__ == "__main__":
    main()