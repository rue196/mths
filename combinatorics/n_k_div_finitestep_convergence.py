#!/usr/bin/env python3
"""
signed_index_ugc.py
===================
Signed-index UGC discrete log with real/imaginary split.

    k < 0  →  v(k) = −H_{|k|+1}                     (R-plane, real negative)
    k > 0  →  v(k) = −H_{k+1} + i·α·(μ*H)(k)        (L-plane, complex)
    k = 0  →  v(0) = 0

The UGC edge (L_i, R_i) enforces:

    Re(v(+i)) = v(−i)                  ∀ i ≥ 1

i.e. the negative-index real axis is the real projection of the
positive-index complex curve.

Finite-step derivative (a = 1/(π−e)) normalizes the whole family so
it stays bounded as K grows: O(n) integer precision ↔ O(K log K)
Möbius precision.
"""

from __future__ import annotations
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Constants
# ============================================================
PI     = math.pi
E      = math.e
ALPHA  = 1.0 / (PI - E)                 # ≈ 2.362338
A      = ALPHA                          # finite-step size
GAMMA  = 0.5772156649015329             # Euler–Mascheroni

# ============================================================
#  O(K) linear Möbius sieve
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
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
#  O(K) harmonic numbers
# ============================================================
def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H

# ============================================================
#  O(K log K) Möbius convolution  F = μ * H
# ============================================================
def mu_convolution_H(K: int):
    mu = mobius_sieve(K)
    H  = harmonic_numbers(K)
    F  = np.zeros(K + 1, dtype=float)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = int(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F, mu, H

# ============================================================
#  Signed-index value
# ============================================================
def signed_value(k: int,
                 H: np.ndarray,
                 F: np.ndarray,
                 imag_scale: float = ALPHA) -> complex:
    """
    v(k):
        k < 0  →  −H_{|k|+1}                 (purely real, negative)
        k > 0  →  −H_{k+1} + i·scale·F(k)    (imaginary part from O(K log K))
        k = 0  →  0
    """
    if k == 0:
        return 0.0 + 0.0j
    n = abs(k)
    real = -(float(H[n]) if n < H.size else math.log(n) + GAMMA)
    if k < 0:
        return complex(real, 0.0)
    imag = imag_scale * float(F[n]) if n < F.size else 0.0
    return complex(real, imag)

def build_signed_grid(K: int, imag_scale: float = ALPHA):
    t0 = time.perf_counter()
    F, mu, H = mu_convolution_H(K + 1)
    t_mu = (time.perf_counter() - t0) * 1e3

    ks  = np.arange(-K, K + 1)
    vals = np.array([signed_value(int(k), H, F, imag_scale) for k in ks])
    return ks, vals, H, F, mu, t_mu

# ============================================================
#  Finite-step normalization  (Euler–Maclaurin, step a)
# ============================================================
def finite_step_normalize(vals: np.ndarray, a: float = A) -> np.ndarray:
    """
    Apply finite-step Euler–Maclaurin correction to a complex sequence:

        f_eff = f − ½(f_0 + f_N) + (a/12)(f'_N − f'_0)

    with f'(x) ≈ [f(x+a) − f(x−a)] / (2a).
    Applied to real and imaginary parts separately.
    """
    N = vals.size
    if N < 3:
        return vals.copy()

    def _corr(f):
        fp = np.zeros_like(f)
        fp[1:-1] = (f[2:] - f[:-2]) / (2.0 * a)
        return f - 0.5 * (f[0] + f[-1]) + (a / 12.0) * (fp[-1] - fp[0])

    return _corr(vals.real) + 1j * _corr(vals.imag)

# ============================================================
#  Supertrace / entropy / invariant mass
# ============================================================
def supertrace(vals: np.ndarray) -> float:
    ks = np.arange(vals.size) - vals.size // 2
    signs = np.where(ks % 2 == 0, 1.0, -1.0)
    return float(np.sum(signs * np.abs(vals)))

def entropy(S: float, N: int) -> float:
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -ALPHA * p * math.log(p)

def invariant_mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0

# ============================================================
#  UGC discrete log on Z_p^* with signed harmonic values
# ============================================================
class SignedUGC:
    """
    Discrete log on a UGC graph whose vertices carry signed-index values.

    L side (positive index):  complex value  v(+i) = −H_{i+1} + i·α·F(i)
    R side (negative index):  real value     v(−i) = −H_{i+1}

    Edge  (L_i, R_i)  ⟺  Re(v(+i)) = v(−i)

    dlog_g(h) = (x, v(x))   with   g^x ≡ h (mod p).
    """
    def __init__(self, p: int, g: int, K: int | None = None):
        if not self._is_prime(p):
            raise ValueError(f"{p} is not prime")
        self.p = p
        self.g = g % p
        self.n = p - 1
        self.K = K or self.n

        self.F, self.mu, self.H = mu_convolution_H(self.K + 1)
        self.powers = self._powers()
        self.log_table = {int(v): i for i, v in enumerate(self.powers)}

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2: return False
        if n % 2 == 0: return n == 2
        return all(n % q for q in range(3, math.isqrt(n) + 1, 2))

    def _powers(self) -> np.ndarray:
        powers = np.zeros(self.n, dtype=np.int64)
        cur, seen = 1, set()
        for i in range(self.n):
            if cur in seen:
                raise ValueError(f"g = {self.g} is not a generator mod {self.p}")
            seen.add(cur)
            powers[i] = cur
            cur = (cur * self.g) % self.p
        return powers

    def value_at(self, x: int, imag_scale: float = ALPHA) -> complex:
        return signed_value(x, self.H, self.F, imag_scale)

    def dlog(self, h: int):
        h = h % self.p
        if h == 0:
            return None, None
        x = self.log_table.get(int(h))
        if x is None:
            return None, None
        return x, self.value_at(x)

    def verify(self, h: int, x: int) -> bool:
        return pow(self.g, x, self.p) == (h % self.p)

    # ---------- the UGC correspondence ----------
    def correspondence_table(self, n_max: int = 12):
        """
        Verify Re(v(+n)) = v(−n) for every n.
        """
        rows = []
        for n in range(1, n_max + 1):
            v_pos = self.value_at(+n)
            v_neg = self.value_at(-n)
            rows.append(dict(n=n,
                             v_pos=v_pos, v_neg=v_neg,
                             real_match=abs(v_pos.real - v_neg.real) < 1e-12))
        return rows

# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 84)
    print("Signed-index UGC  ·  real/imaginary harmonic split")
    print("=" * 84)
    print(f"  α = 1/(π−e) = {ALPHA:.6f}")
    print(f"  γ           = {GAMMA:.6f}")
    print()
    print("  v(−n) = −H_{n+1}                          (R-plane, real)")
    print("  v(+n) = −H_{n+1} + i·α·(μ*H)(n)           (L-plane, complex)")
    print("  Edge (L_n, R_n)  ⟺  Re(v(+n)) = v(−n)")
    print()

    K = 60
    ks, vals, H, F, mu, t_mu = build_signed_grid(K)
    print(f"  K = {K}   |  (μ*H) built in {t_mu:.2f} ms  (O(K log K))")
    print()

    # ---------- correspondence table ----------
    print("--- v(−n)  vs  Re(v(+n))  for n = 1…12 ---")
    print(f"  {'n':>3s}  {'v(−n)':>12s}  {'Re(v(+n))':>12s}  "
          f"{'Im(v(+n))':>12s}  match")
    print("  " + "-" * 60)
    for n in range(1, 13):
        vn = vals[K - n]        # index k = −n
        vp = vals[K + n]        # index k = +n
        match = "✓" if abs(vn.real - vp.real) < 1e-12 else "✗"
        print(f"  {n:>3d}  {vn.real:>+12.6f}  {vp.real:>+12.6f}  "
              f"{vp.imag:>+12.6f}  {match}")
    print()

    # ---------- invariants ----------
    S = supertrace(vals)
    H_ent = entropy(S, vals.size)
    m = invariant_mass(S, vals.size)
    print(f"--- invariants of the signed grid  |k| ≤ {K} ---")
    print(f"  S (supertrace)     = {S:+.6f}")
    print(f"  H (entropy)        = {H_ent:.6f}")
    print(f"  m (invariant mass) = {m:.6e}")
    print()

    # ---------- finite-step normalization ----------
    norm = finite_step_normalize(vals)
    S_n = supertrace(norm)
    H_n = entropy(S_n, norm.size)
    m_n = invariant_mass(S_n, norm.size)
    print(f"--- after finite-step normalization (a = {A:.4f}) ---")
    print(f"  S (supertrace)     = {S_n:+.6f}")
    print(f"  H (entropy)        = {H_n:.6f}")
    print(f"  m (invariant mass) = {m_n:.6e}")
    print()

    # ---------- UGC discrete log verification ----------
    print("--- UGC discrete log  ·  Z_101^*, g = 2 ---")
    ugc = SignedUGC(p=101, g=2, K=K)
    rng = np.random.default_rng(0)
    tests = rng.integers(0, ugc.n, size=5)
    print(f"  {'x_true':>7s}  {'h = g^x':>9s}  {'x_hat':>7s}  "
          f"{'Re(v(x_hat))':>13s}  {'Im(v(x_hat))':>13s}  ok")
    for x_true in tests:
        h = int(ugc.powers[x_true])
        x_hat, v = ugc.dlog(h)
        ok = ugc.verify(h, x_hat)
        print(f"  {int(x_true):>7d}  {h:>9d}  {x_hat:>7d}  "
              f"{v.real:>+13.6f}  {v.imag:>+13.6f}  "
              f"{'✓' if ok else '✗'}")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # (a) complex plane — negative indices on real axis, positive in upper half
    ax = fig.add_subplot(gs[0, 0])
    neg = vals[:K]                 # k = −K … −1
    pos = vals[K + 1:]             # k = +1 … +K
    ax.scatter(neg.real, neg.imag, c="crimson", s=14, alpha=0.8,
               label="negative k  (R-plane, real)")
    ax.scatter(pos.real, pos.imag, c="royalblue", s=14, alpha=0.8,
               label="positive k  (L-plane, complex)")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("Re v"); ax.set_ylabel("Im v")
    ax.set_title("Signed index values in the complex plane")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) real part of v(k) across k = −K … +K
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ks, vals.real, color="purple", lw=1.3)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("k"); ax.set_ylabel("Re v(k)")
    ax.set_title("Real part  ·  symmetric in k ↔ −k")
    ax.grid(alpha=0.3)

    # (c) imaginary part of v(k) — only nonzero for k > 0
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ks, vals.imag, color="darkorange", lw=1.3)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("k"); ax.set_ylabel("Im v(k)")
    ax.set_title("Imaginary part  ·  O(K log K) Möbius sector")
    ax.grid(alpha=0.3)

    # (d) correspondence: v(−n) vs Re(v(+n))
    ax = fig.add_subplot(gs[1, 0])
    ns = np.arange(1, K + 1)
    v_neg = vals[K - ns].real
    v_pos_re = vals[K + ns].real
    ax.plot(ns, v_neg, "crimson", lw=1.6, label=r"$v(-n)$")
    ax.plot(ns, v_pos_re, "royalblue", ls="--", lw=1.4,
            label=r"$\mathrm{Re}(v(+n))$")
    ax.set_xlabel("n"); ax.set_ylabel("value")
    ax.set_title("UGC edge:  Re(v(+n)) = v(−n)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (e) |v| before / after finite-step normalization
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ks, np.abs(vals), color="gray", lw=1.2, label="raw |v|")
    ax.plot(ks, np.abs(norm), color="teal", lw=1.4,
            label="finite-step normalized")
    ax.set_xlabel("k"); ax.set_ylabel("|v|")
    ax.set_title("Finite-step normalization  ·  a = 1/(π−e)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (f) F(n) = (μ*H)(n) — the O(K log K) source
    ax = fig.add_subplot(gs[1, 2])
    ns_F = np.arange(1, K + 1)
    ax.stem(ns_F, F[1:K + 1], linefmt="C0-", markerfmt="C0.",
            basefmt="k-")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("n"); ax.set_ylabel("F(n) = (μ*H)(n)")
    ax.set_title("O(K log K) imaginary source")
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Signed-index UGC  ·  negative real harmonics ↔ "
        "positive complex indices  ·  step a = 1/(π−e)",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    demo()