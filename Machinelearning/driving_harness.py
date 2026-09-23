#!/usr/bin/env python3
"""
driving_harness.py

A driving harness where sensors + camera quick-scan a scene and route
every detection through the symmetric / asymmetric pipelines.

Physical picture
----------------
    LIDAR + camera  →  vectorized detections  (x, y, v, class)
    algebraic cars  →  symmetric objects, μ-filtered  →  O(K log log K)
    time evolution  →  transcendental entries (phase e^{i·α·t·R})
    unknown objects →  asymmetric entries  →  O(K log K) inverse score
    braking scalar  →  slow-down → complete stop
    road commands   →  deterministic Möbius-compressed command list
                        (stop signs, lights, intersections)

Pipeline per frame
------------------
    scan()          → detections
    classify()      → algebraic_cars | asymmetric_objects | road_commands
    symmetric path  → O(K log log K)
    asymmetric path → O(K log K)  →  flag → brake scalar
    transcendental  → O(K) forward integration of car positions
    final control   → throttle / brake / steering from scalar
"""

from __future__ import annotations

import math
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_ASYM = 0.3628
DENSITY    = 6.0 / (PI * PI)           # ≈ 0.6079271018
K_DEFAULT  = 128


# ============================================================
#  1. Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  2. Merge-sort inverse score  ·  O(K log K)
# ============================================================
def _merge_count(a, t, l, m, r):
    i, j, k, inv = l, m + 1, l, 0
    while i <= m and j <= r:
        if a[i] <= a[j]:
            t[k] = a[i]; i += 1
        else:
            t[k] = a[j]; inv += m - i + 1; j += 1
        k += 1
    while i <= m: t[k] = a[i]; i += 1; k += 1
    while j <= r: t[k] = a[j]; j += 1; k += 1
    for i in range(l, r + 1):
        a[i] = t[i]
    return inv


def _merge_sort_count(a, t, l, r):
    inv = 0
    if l < r:
        m = (l + r) // 2
        inv += _merge_sort_count(a, t, l, m)
        inv += _merge_sort_count(a, t, m + 1, r)
        inv += _merge_count(a, t, l, m, r)
    return inv


def inversion_count(arr) -> int:
    n = len(arr)
    if n < 2:
        return 0
    return _merge_sort_count(list(arr), [0] * n, 0, n - 1)


def inverse_score(a, b) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.5
    pairs = sorted(zip(a, b), key=lambda p: p[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    mx = K * (K - 1) // 2
    return inv / mx if mx > 0 else 0.0


# ============================================================
#  3. Supertrace / entropy
# ============================================================
def supertrace(x) -> float:
    S = 0.0
    for i, v in enumerate(x):
        S += v if (i % 2 == 0) else -v
    return float(S)


def entropy(S: float, K: int, alpha: float) -> float:
    if K <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / K
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


# ============================================================
#  4. Detection dataclass  ·  a vectorized object
# ============================================================
@dataclass
class Detection:
    """A single vectorized object from the sensor scan."""
    x: float                     # position in metres (forward)
    y: float                     # lateral position in metres
    vx: float                    # forward velocity
    vy: float                    # lateral velocity
    label: str = "unknown"       # 'car' | 'pedestrian' | 'sign' | 'light'
    confidence: float = 1.0

    def vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy], dtype=float)


# ============================================================
#  5. Algebraic (symmetric) car bank  ·  O(K log log K)
# ============================================================
class AlgebraicCarBank:
    """
    Cars are known algebraic objects: their vectorized signature lies
    on the Möbius-filtered manifold, so a symmetric scan costs
    O(K log log K).
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)

    def embed(self, det: Detection) -> np.ndarray:
        """
        Embed the car's vector into a |Ci| array using the Möbius
        sieve: only square-free indices carry energy.
        """
        v = det.vector()
        c = np.zeros(2 * self.K + 1, dtype=float)
        for j, val in enumerate(v):
            i = (int(abs(val) * 1000) % self.K) + 1
            if self.mu[i] == 0:
                continue
            c[self.K + i] += val
            c[self.K - i] += val
        return c

    def scan(self, cars: List[Detection]) -> dict:
        t0 = time.perf_counter()
        if not cars:
            return dict(S=0.0, H=0.0, m=0.0, n_sym=0,
                        elapsed_ms=(time.perf_counter() - t0) * 1e3)
        concat = np.concatenate([self.embed(d) for d in cars])
        S = supertrace(concat)
        K = len(concat)
        H = entropy(S, K, ALPHA_SYM)
        m = abs(S) * math.exp(-H) if 0.0 < H < 700 else 0.0
        return dict(
            S=S, H=H, m=m, n_sym=len(cars),
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  6. Asymmetric unknown objects  ·  O(K log K)
# ============================================================
class AsymmetricObjectBank:
    """
    Random unknown vectorized objects (pedestrians, obstacles) have
    no algebraic signature, so we embed them by hash and compare by
    inverse score.  The resulting deviation is the braking scalar.
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)
        self.reference = self._reference_embedding()

    def _reference_embedding(self) -> np.ndarray:
        """Reference = a stationary object directly ahead."""
        h = int(hashlib.md5(b"static_ahead").hexdigest()[:8], 16)
        i = (h % self.K) + 1
        c = np.zeros(2 * self.K + 1, dtype=float)
        if self.mu[i] != 0:
            c[self.K + i] = 1.0
            c[self.K - i] = 1.0
        return c

    def embed(self, det: Detection) -> np.ndarray:
        h = hashlib.md5(f"{det.x:.3f}|{det.y:.3f}|{det.label}"
                        .encode()).hexdigest()
        idx = (int(h[:8], 16) % self.K) + 1
        c = np.zeros(2 * self.K + 1, dtype=float)
        if self.mu[idx] != 0:
            c[self.K + idx] += det.confidence
            c[self.K - idx] += det.confidence
        return c

    def scan(self, objects: List[Detection]) -> dict:
        t0 = time.perf_counter()
        if not objects:
            return dict(n_asym=0, scores=[], S=0.0, H=0.0, m=0.0,
                        flag=0.0, brake=0.0,
                        elapsed_ms=(time.perf_counter() - t0) * 1e3)
        scores = []
        for det in objects:
            emb = self.embed(det)
            sc = inverse_score(emb.tolist(), self.reference.tolist())
            scores.append((det, sc))

        # The deviation from 0.5 is the *raw* asymmetric flag.
        mean_score = float(np.mean([s for _, s in scores]))
        # Inverse score ∈ [0,1].  > 0.5 means "not matching reference".
        # Flag = max over objects, so a single close pedestrian dominates.
        flag = max(abs(s - 0.5) * 2.0 for _, s in scores)
        # Braking scalar: 0 (safe) → 1 (full stop).
        # Threshold:  flag < 0.25  →  no brake
        #             flag ∈ [0.25, 0.6]  →  linear slowdown
        #             flag > 0.6  →  full stop
        if flag < 0.25:
            brake = 0.0
        elif flag < 0.6:
            brake = (flag - 0.25) / 0.35
        else:
            brake = 1.0

        concat = np.concatenate([self.embed(d) for d in objects])
        S = supertrace(concat)
        K = len(concat)
        H = entropy(S, K, ALPHA_ASYM)
        m = abs(S) * math.exp(-H) if 0.0 < H < 700 else 0.0
        return dict(
            n_asym=len(objects), scores=scores,
            S=S, H=H, m=m,
            flag=flag, brake=brake,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  7. Transcendental time evolution  ·  O(K)
# ============================================================
def evolve_cars(cars: List[Detection], dt: float,
                horizon: float = 1.0) -> List[Detection]:
    """
    Move every car through the transcendental entries:

        x(t) = x_0 + vx · t · e^{i α R}
        y(t) = y_0 + vy · t · e^{i α R}

    We integrate the *real* trajectory and accumulate the *imaginary*
    phase as an uncertainty flag.  Cars whose phase uncertainty grows
    beyond a threshold are re-classified as asymmetric.
    """
    if dt <= 0 or not cars:
        return cars
    steps = max(1, int(horizon / dt))
    moved = []
    for det in cars:
        x, y = det.x, det.y
        vx, vy = det.vx, det.vy
        R = math.hypot(x, y) + 1e-6
        phase = 0.0
        for _ in range(steps):
            # transcendental phase accumulates
            phase += ALPHA_SYM * dt * R
            x += vx * dt
            y += vy * dt
            R = math.hypot(x, y) + 1e-6
        # imaginary uncertainty
        unc = abs(math.sin(phase))
        if unc > 0.9:
            # too uncertain — reclassify as asymmetric
            moved.append(Detection(x=x, y=y, vx=vx, vy=vy,
                                   label="reclass",
                                   confidence=det.confidence))
        else:
            moved.append(Detection(x=x, y=y, vx=vx, vy=vy,
                                   label=det.label,
                                   confidence=det.confidence))
    return moved


# ============================================================
#  8. Deterministic road command list  ·  Möbius-compressed
# ============================================================
class RoadCommandGate:
    """
    Stop signs, traffic lights, and intersections are deterministic
    commands.  They live on a fixed Möbius-compressed list.  Every
    frame, the gate scans the compressed list against the incoming
    detections; a match forces the corresponding command.
    """
    COMMANDS = {
        "STOP":       "full_stop",
        "YIELD":      "slow_down",
        "LIGHT_RED":  "full_stop",
        "LIGHT_YELLOW": "slow_down",
        "LIGHT_GREEN": "proceed",
        "CROSSWALK":  "slow_down",
        "INTERSECTION": "proceed",
    }

    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)
        self.packed = self._pack_commands()

    def _pack_commands(self) -> Dict[str, int]:
        """
        2-bit-style packing of the deterministic command list:
        each command gets a square-free Möbius address.  This is
        the same packing used for the Möbius memory.
        """
        packed = {}
        for i, key in enumerate(sorted(self.COMMANDS.keys())):
            addr = (i * i + i + 1) % max(self.K, 1)
            if addr == 0 or self.mu[addr] == 0:
                addr = (addr + 1) % max(self.K, 1)
            packed[key] = addr
        return packed

    def scan(self, detections: List[Detection]) -> Optional[str]:
        """
        Scan the deterministic list against the detections; return the
        forced command if a known sign or light is present, else None.
        """
        for det in detections:
            lab = det.label.upper()
            if lab in self.COMMANDS:
                return self.COMMANDS[lab]
        return None


# ============================================================
#  9. The driving harness
# ============================================================
@dataclass
class ControlOutput:
    throttle: float
    brake: float
    steering: float
    forced_command: Optional[str]
    symmetric_S: float
    asymmetric_S: float
    brake_scalar: float
    n_sym: int
    n_asym: int
    flag: float
    elapsed_ms: float


class DrivingHarness:
    """
    End-to-end frame processor:

        scan  →  classify  →  symmetric  →  transcendental
              →  asymmetric  →  brake scalar  →  command list
              →  ControlOutput
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.car_bank = AlgebraicCarBank(K)
        self.obj_bank = AsymmetricObjectBank(K)
        self.cmd_gate = RoadCommandGate(K)
        self.frame_count = 0
        self.history: List[ControlOutput] = []

    # ---------- scan + classify ----------
    def classify_detections(self, dets: List[Detection]
                            ) -> Tuple[List[Detection], List[Detection],
                                       List[Detection]]:
        cars, obstacles, cmds = [], [], []
        for d in dets:
            lab = d.label.lower()
            if lab in ("car", "truck", "bus", "motorcycle"):
                cars.append(d)
            elif lab in ("pedestrian", "cyclist", "obstacle",
                         "reclass", "unknown"):
                obstacles.append(d)
            elif lab in ("stop", "yield", "light_red", "light_yellow",
                         "light_green", "crosswalk", "intersection"):
                cmds.append(d)
            else:
                obstacles.append(d)   # conservative default
        return cars, obstacles, cmds

    # ---------- main frame ----------
    def process_frame(self, dets: List[Detection],
                      dt: float = 0.1) -> ControlOutput:
        t0 = time.perf_counter()
        self.frame_count += 1

        cars, obstacles, cmds = self.classify_detections(dets)

        # 1. symmetric scan of algebraic cars  O(K log log K)
        car_res = self.car_bank.scan(cars)

        # 2. transcendental time evolution of cars  O(K)
        cars_evolved = evolve_cars(cars, dt)
        # any car that became uncertain is moved to asymmetric
        stable_cars, reclassed = [], []
        for c in cars_evolved:
            if c.label == "reclass":
                reclassed.append(c)
            else:
                stable_cars.append(c)
        obstacles = obstacles + reclassed
        car_res = self.car_bank.scan(stable_cars)   # re-scan after evolution

        # 3. asymmetric scan of unknown objects  O(K log K)
        obj_res = self.obj_bank.scan(obstacles)

        # 4. deterministic command list
        forced = self.cmd_gate.scan(cmds)

        # 5. control synthesis
        #    brake scalar comes straight from the asymmetric flag
        brake_scalar = obj_res["brake"]
        throttle = 1.0 - brake_scalar
        brake    = brake_scalar
        steering = 0.0
        for c in stable_cars:
            steering += -0.02 * c.y + 0.05 * c.vy
        for o in obstacles:
            steering += -0.05 * o.y
        steering = float(np.tanh(steering))

        # forced command overrides everything
        if forced == "full_stop":
            throttle, brake = 0.0, 1.0
        elif forced == "slow_down":
            throttle = min(throttle, 0.3)
            brake    = max(brake, 0.5)
        elif forced == "proceed":
            pass  # keep the current values

        out = ControlOutput(
            throttle=throttle,
            brake=brake,
            steering=steering,
            forced_command=forced,
            symmetric_S=car_res["S"],
            asymmetric_S=obj_res["S"],
            brake_scalar=brake_scalar,
            n_sym=car_res["n_sym"],
            n_asym=obj_res["n_asym"],
            flag=obj_res["flag"],
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )
        self.history.append(out)
        return out


# ============================================================
#  10. Scene generator  ·  synthetic sensor frames
# ============================================================
def make_frame(frame_id: int, rng: np.random.Generator) -> List[Detection]:
    """
    Build a synthetic sensor frame:

        • 2–5 algebraic cars in the forward corridor
        • 0–2 unknown obstacles
        • occasional deterministic road commands
    """
    dets: List[Detection] = []

    # algebraic cars
    n_cars = rng.integers(2, 6)
    for _ in range(n_cars):
        dets.append(Detection(
            x=float(rng.uniform(5, 60)),
            y=float(rng.uniform(-3, 3)),
            vx=float(rng.uniform(0.5, 3.0)),
            vy=float(rng.uniform(-0.3, 0.3)),
            label="car",
            confidence=float(rng.uniform(0.7, 1.0)),
        ))

    # unknown objects
    n_obj = rng.integers(0, 3)
    for _ in range(n_obj):
        dets.append(Detection(
            x=float(rng.uniform(3, 25)),
            y=float(rng.uniform(-2, 2)),
            vx=float(rng.uniform(0, 1.5)),
            vy=float(rng.uniform(-0.5, 0.5)),
            label=str(rng.choice(["pedestrian", "cyclist", "obstacle"])),
            confidence=float(rng.uniform(0.6, 0.95)),
        ))

    # deterministic commands
    if frame_id % 7 == 0:
        dets.append(Detection(x=80.0, y=0.0, vx=0.0, vy=0.0,
                              label="STOP"))
    elif frame_id % 11 == 0:
        dets.append(Detection(x=60.0, y=0.0, vx=0.0, vy=0.0,
                              label="LIGHT_RED"))
    elif frame_id % 5 == 0:
        dets.append(Detection(x=40.0, y=0.0, vx=0.0, vy=0.0,
                              label="CROSSWALK"))

    return dets


# ============================================================
#  11. Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Driving harness  ·  symmetric cars + asymmetric objects + commands")
    print("=" * 78)
    print(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  α_asym  = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K       = {K_DEFAULT}")
    print()

    harness = DrivingHarness(K=K_DEFAULT)
    rng = np.random.default_rng(2024)

    print(f"{'frame':>5s}  {'n_sym':>5s}  {'n_asym':>6s}  "
          f"{'S_sym':>8s}  {'S_asym':>8s}  {'flag':>6s}  "
          f"{'brake':>6s}  {'steer':>7s}  {'cmd':>12s}  {'ms':>6s}")
    print("-" * 90)

    for fid in range(1, 31):
        dets = make_frame(fid, rng)
        out = harness.process_frame(dets, dt=0.1)
        cmd = out.forced_command if out.forced_command else "—"
        print(f"{fid:>5d}  {out.n_sym:>5d}  {out.n_asym:>6d}  "
              f"{out.symmetric_S:+8.3f}  {out.asymmetric_S:+8.3f}  "
              f"{out.flag:6.3f}  {out.brake:6.3f}  "
              f"{out.steering:+7.3f}  {cmd:>12s}  {out.elapsed_ms:6.2f}")

    # ---------- aggregate ----------
    print()
    print("--- Aggregate over the run ---")
    n = len(harness.history)
    mean_sym_S   = np.mean([o.symmetric_S for o in harness.history])
    mean_asym_S  = np.mean([o.asymmetric_S for o in harness.history])
    mean_brake   = np.mean([o.brake for o in harness.history])
    mean_flag    = np.mean([o.flag for o in harness.history])
    mean_time    = np.mean([o.elapsed_ms for o in harness.history])
    stops = sum(1 for o in harness.history if o.brake >= 1.0 - 1e-9)
    slow  = sum(1 for o in harness.history
                if 0.0 < o.brake < 1.0 - 1e-9)

    print(f"  frames                : {n}")
    print(f"  mean symmetric S      : {mean_sym_S:+.4f}")
    print(f"  mean asymmetric S     : {mean_asym_S:+.4f}")
    print(f"  mean flag             : {mean_flag:.4f}")
    print(f"  mean brake            : {mean_brake:.4f}")
    print(f"  full-stop frames      : {stops}")
    print(f"  partial-slow frames   : {slow}")
    print(f"  mean frame time       : {mean_time:.2f} ms")
    print()

    # ---------- packing of command list ----------
    print("--- Deterministic road commands (Möbius-packed) ---")
    for key, addr in harness.cmd_gate.packed.items():
        mu_val = harness.cmd_gate.mu[addr] if addr < len(harness.cmd_gate.mu) else 0
        tag = "μ ≠ 0" if mu_val != 0 else "μ = 0"
        print(f"  {key:<14s}  address = {addr:>4d}  {tag}")
    print()

    # ---------- complexity ----------
    print("--- Pipeline complexity ---")
    print("  algebraic cars  ·  symmetric scan       O(K log log K)  per frame")
    print("  transcendental time evolution           O(K · steps)")
    print("  unknown objects ·  asymmetric scan      O(K log K)      per frame")
    print("  deterministic command scan              O(K)")
    print("  total per frame                         O(K log K)")

    print("\nDone.")


if __name__ == "__main__":
    demo()