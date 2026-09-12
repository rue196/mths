#!/usr/bin/env python3
"""
llm_collatz_sandbox.py

Sandboxed Collatz‑escape guard for LLM decoding / K‑sum accumulation.

Motivation
----------
When an LLM accumulates a K‑sum (e.g. |Ci| arrays, suisse traces,
spectral coefficients), an "escape" can occur: the magnitude grows
without bound, the sieve buffer grows past K, or the collatz iteration
runs forever.  This sandbox detects escape and applies:

    1. Collatz step           (n → n/2 if even, 3n+1 if odd)
       on integer K‑index values, or on the IEEE‑754 bit pattern
       when the value is a float.
    2. Rational cap           (from rational_maybe.py)
       approximates the running scalar as  r + s·e  with small
       rationals; if the residual grows the sandbox snaps the
       value to the rational boundary, preventing ∞ growth.
    3. Escape budget          a hard O(K log K) bound; once the
       sandbox has spent the budget, every further step is forced
       through the rational cap.

Public API
----------
    sandbox = CollatzEscapeSandbox(K=1024, max_steps=64)
    sandbox.push(value)              # push a new K‑sum sample
    sandbox.status()                 # current state
    sandbox.wrap(decoder_fn)         # wrap an LLM step function
    sandbox.cap(value)               # force a rational cap

All operations are O(K) or O(K log K).
"""

import math
import struct
import hashlib
import numpy as np
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)             # ≈ 2.362
ALPHA_USER = 0.3628
A_STEP    = ALPHA / ALPHA_USER         # ≈ 6.511
DENSITY   = 6.0 / (PI * PI)            # ≈ 0.6079 square‑free density

# ============================================================
#  Rational search (from rational_maybe.py)
# ============================================================
def exp_series(k: int) -> float:
    s, t = 1.0, 1.0
    for n in range(1, k + 1):
        t /= n
        s += t
    return s


def pi_series(k: int) -> float:
    s, sign = 0.0, 1.0
    for n in range(k + 1):
        s += sign / (2 * n + 1)
        sign = -sign
    return 4.0 * s


def search_rational_combination(y: float, e_ap: float,
                                r_range=(-5, 5),
                                s_range=(-5, 5),
                                denom_limit: int = 4
                                ) -> Tuple[Fraction, Fraction, float]:
    """
    Return (best_r, best_s, err) such that  y ≈ r + s·e_ap.
    O(D²) with D = number of candidate rationals.
    """
    rationals = set()
    for den in range(1, denom_limit + 1):
        for num in range(r_range[0] * den, r_range[1] * den + 1):
            rationals.add(Fraction(num, den))
    rationals = list(rationals)

    best_r, best_s, best_err = Fraction(0, 1), Fraction(0, 1), float('inf')
    for r in rationals:
        for s in rationals:
            pred = float(r) + float(s) * e_ap
            err = abs(y - pred)
            if err < best_err:
                best_err, best_r, best_s = err, r, s
    return best_r, best_s, best_err


# ============================================================
#  Collatz primitives
# ============================================================
def collatz_step_int(n: int) -> int:
    """Classical Collatz step on a positive integer."""
    if n <= 0:
        return 0
    return n // 2 if n % 2 == 0 else 3 * n + 1


def collatz_step_bits(value: float, steps: int = 1) -> float:
    """
    Collatz on the IEEE‑754 bit pattern of a float.
    Used in projection-suisse-encrypt.py.
    """
    bits = struct.unpack('>Q', struct.pack('>d', float(value)))[0]
    for _ in range(steps):
        if bits & 1 == 0:
            bits >>= 1
        else:
            bits = (bits << 1) + bits + 1
        # mask back into 64 bits
        bits &= 0xFFFFFFFFFFFFFFFF
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


# ============================================================
#  Escape detection
# ============================================================
@dataclass
class EscapeState:
    step: int = 0
    k_sum: float = 0.0
    magnitude: float = 0.0
    escape: bool = False
    reason: str = ""
    collatz_steps_applied: int = 0
    rational_caps_applied: int = 0
    last_rational_fit: Optional[Tuple[Fraction, Fraction, float]] = None


# ============================================================
#  The sandbox
# ============================================================
class CollatzEscapeSandbox:
    """
    Guard‑rail around a streaming K‑sum.

    Parameters
    ----------
    K                : maximum allowed K‑sum buffer size
    max_steps        : hard budget on collatz steps (O(K log K) total)
    escape_threshold : magnitude above which escape is declared
    growth_tol       : relative growth per step tolerated before escape
    rational_mode    : 'snap'   → snap to the rational fit when escaping
                       'clip'   → clip to the rational boundary
                       'pass'   → pass through but log
    e_series_terms   : series length used to approximate e in the rational search
    pi_series_terms  : series length used to approximate π
    """

    def __init__(self,
                 K: int = 1024,
                 max_steps: int = 64,
                 escape_threshold: float = 1e6,
                 growth_tol: float = 1.5,
                 rational_mode: str = 'snap',
                 e_series_terms: int = 120,
                 pi_series_terms: int = 120):
        self.K = K
        self.max_steps = max_steps
        self.escape_threshold = escape_threshold
        self.growth_tol = growth_tol
        self.rational_mode = rational_mode

        # Rational cache
        self.e_ap = exp_series(e_series_terms)
        self.pi_ap = pi_series(pi_series_terms)
        self.a_ap = 1.0 / (self.pi_ap - self.e_ap)

        # State
        self.state = EscapeState()
        self.prev_magnitude: Optional[float] = None
        self.k_sum_values: List[float] = []
        self.log: List[Dict] = []

    # ---------- diagnostics ----------
    def _log(self, **kwargs):
        entry = dict(step=self.state.step, **kwargs)
        self.log.append(entry)
        return entry

    # ---------- escape detection ----------
    def _detect_escape(self, value: float) -> Tuple[bool, str]:
        mag = abs(value)

        # 1. Hard magnitude bound
        if mag > self.escape_threshold:
            return True, "magnitude_exceeds_threshold"

        # 2. Unbounded growth vs previous step
        if self.prev_magnitude is not None:
            if self.prev_magnitude > 0:
                growth = mag / (self.prev_magnitude + 1e-30)
                if growth > self.growth_tol and mag > self.prev_magnitude:
                    return True, "unbounded_growth"

        # 3. Budget exhausted
        if self.state.step >= self.max_steps:
            return True, "step_budget_exhausted"

        return False, ""

    # ---------- rational cap ----------
    def _rational_cap(self, value: float) -> float:
        """
        Approximate  a·value  as  r + s·e  with small rationals,
        then reconstruct the capped scalar.
        """
        y = self.a_ap * value
        r, s, err = search_rational_combination(y, self.e_ap)
        self.state.last_rational_fit = (r, s, err)

        # r + s·e  gives the rational boundary in the scaled space
        boundary_scaled = float(r) + float(s) * self.e_ap
        boundary = boundary_scaled / self.a_ap

        if self.rational_mode == 'snap':
            return boundary
        elif self.rational_mode == 'clip':
            return math.copysign(min(abs(value), abs(boundary)), value)
        else:
            return value

    # ---------- the main step ----------
    def push(self, value: float) -> float:
        """
        Push a new K‑sum sample.

        Returns the (possibly transformed) value.
        """
        self.state.step += 1
        self.state.k_sum += value
        value = float(value)

        escape, reason = self._detect_escape(value)

        if not escape:
            self.prev_magnitude = abs(value)
            self.k_sum_values.append(value)
            self._log(value=value, escape=False)
            return value

        # --- escape declared ---
        self.state.escape = True
        self.state.reason = reason

        # 1. Collatz step (bits level for floats)
        collatz_val = collatz_step_bits(value, steps=1)
        # Use the smaller of {original, collatz} to avoid growth
        transformed = min(abs(value), abs(collatz_val))
        transformed = math.copysign(transformed, value)
        self.state.collatz_steps_applied += 1

        # 2. Rational cap
        capped = self._rational_cap(transformed)
        self.state.rational_caps_applied += 1

        # Final value
        out = capped

        self.prev_magnitude = abs(out)
        self.k_sum_values.append(out)
        self._log(
            value=value,
            escape=True,
            reason=reason,
            collatz=collatz_val,
            capped=capped,
        )
        return out

    # ---------- forcing a cap ----------
    def cap(self, value: float) -> float:
        """Explicitly force a rational cap (bypasses escape detection)."""
        capped = self._rational_cap(value)
        self.state.rational_caps_applied += 1
        self._log(value=value, forced_cap=True, capped=capped)
        return capped

    # ---------- status ----------
    def status(self) -> Dict:
        return dict(
            step=self.state.step,
            escape=self.state.escape,
            reason=self.state.reason,
            k_sum=self.state.k_sum,
            magnitude=abs(self.state.k_sum),
            collatz_steps=self.state.collatz_steps_applied,
            rational_caps=self.state.rational_caps_applied,
            last_rational_fit=(
                (str(self.state.last_rational_fit[0]),
                 str(self.state.last_rational_fit[1]),
                 self.state.last_rational_fit[2])
                if self.state.last_rational_fit else None
            ),
            n_samples=len(self.k_sum_values),
        )

    def reset(self):
        self.state = EscapeState()
        self.prev_magnitude = None
        self.k_sum_values.clear()
        self.log.clear()

    # ---------- wrap an LLM step ----------
    def wrap(self, step_fn: Callable[[np.ndarray], np.ndarray]):
        """
        Wrap an LLM step function:
            (x) -> x'
        Every output sample goes through the escape guard.
        """
        def wrapped(x):
            y = step_fn(x)
            y = np.asarray(y, dtype=float).ravel()
            out = np.empty_like(y)
            for i, v in enumerate(y):
                out[i] = self.push(v)
            return out.reshape(np.asarray(y).shape)
        return wrapped

    # ---------- streaming K‑sum reader (O(K)) ----------
    def read_stream(self, source: Iterable[float]) -> np.ndarray:
        """Feed an iterable of K‑sum samples through the guard."""
        return np.array([self.push(v) for v in source], dtype=float)


# ============================================================
#  Integration with an LLM decoding loop (mock example)
# ============================================================
def mock_llm_step(x: np.ndarray) -> np.ndarray:
    """
    A mock LLM step that occasionally produces runaway magnitudes.
    Replace with a real decoding step (logits → hidden → |Ci|).
    """
    rng = np.random.default_rng(seed=int(abs(x.sum())) % (2**31))
    step = 0.1 * x + rng.standard_normal(x.size) * 0.05
    # occasional runaway
    if rng.random() < 0.15:
        step *= 50.0
    return step


def demo():
    print("=" * 68)
    print("Collatz escape sandbox for LLM K‑sums")
    print("=" * 68)

    sandbox = CollatzEscapeSandbox(
        K=256,
        max_steps=32,
        escape_threshold=50.0,
        growth_tol=1.4,
        rational_mode='snap',
        e_series_terms=140,
        pi_series_terms=140,
    )

    # --- wrap the mock LLM step ---
    wrapped = sandbox.wrap(mock_llm_step)

    x = np.ones(8) * 0.5
    for step in range(20):
        x = wrapped(x)
        mag = float(np.linalg.norm(x))
        if step % 4 == 0 or sandbox.state.escape:
            print(f"  step {step:2d}  |x|={mag:9.4f}  "
                  f"escape={sandbox.state.escape}  "
                  f"collatz={sandbox.state.collatz_steps_applied}  "
                  f"caps={sandbox.state.rational_caps_applied}")

    print("\n--- status ---")
    for k, v in sandbox.status().items():
        print(f"  {k}: {v}")

    # --- explicit capping ---
    print("\n--- explicit rational cap ---")
    for v in [1e8, -3.2e7, 42.0]:
        capped = sandbox.cap(v)
        print(f"  {v:+.3e}  →  {capped:+.6f}")

    # --- last rational fit ---
    print("\n--- last rational fit ---")
    fit = sandbox.state.last_rational_fit
    if fit:
        r, s, err = fit
        print(f"  y ≈ {r} + {s}·e  (error={err:.4e})")

    # --- log tail ---
    print("\n--- escape log (last 5 entries) ---")
    for entry in sandbox.log[-5:]:
        print("  " + str(entry))

    print("\nDone.")


if __name__ == "__main__":
    demo()
    input("Press ENTER to exit")