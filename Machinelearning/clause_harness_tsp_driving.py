#!/usr/bin/env python3
"""
tsp_map_harness.py

A TSP map harness with two routing regimes:

    symmetric (unchanged direction)              O(K log log K)
        • normal TSP path over the destination set
        • used when no obstruction is present
        • direction is preserved — the "straightest" 1D TSP order

    asymmetric (obstruction-aware)               O(K log K)
        • triggered by unknown obstacles on the path
        • finds the next best path (re-plan)
        • possible actions: follow traffic, safe U-turn, turn aside
        • conditional clauses evaluate in order

    deterministic (algebraic)                    O(K)
        • used when there are no lights, stop signs, or commands
        • linear-regression if/else over car states
        • this is the algebraic part of the harness

Clauses in the command language:
    if <predicate> then <action>
    while <predicate> do <action>
    else <action>

Predicates are formed from the sensor detections and the current map state.

Complexity
----------
    TSP sort (symmetric)                O(K log K)      once per plan
    update plan (asymmetric)            O(K log K)      per obstruction
    linear-regression fallback          O(K)            per frame
    total per frame                     O(K log K)
"""

from __future__ import annotations

import math
import time
import random
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA_SYM   = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_ASYM  = 0.3628
DENSITY     = 6.0 / (PI * PI)
K_DEFAULT   = 128


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
#  2. Merge sort (symmetric, unchanged direction)
# ============================================================
def merge_sort_1d(points: List[float]) -> List[float]:
    """Deterministic 1D merge sort preserving the left-to-right direction."""
    if len(points) <= 1:
        return points[:]
    mid = len(points) // 2
    left = merge_sort_1d(points[:mid])
    right = merge_sort_1d(points[mid:])
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


# ============================================================
#  3. TSP map: destinations on a 1D road
# ============================================================
@dataclass
class Destination:
    """A point on the map with a 1D position and a target id."""
    pos: float
    dest_id: int
    kind: str = "waypoint"      # 'waypoint' | 'traffic_light' | 'stop_sign'


@dataclass
class Obstruction:
    """A detected obstruction on the map."""
    pos: float
    width: float
    label: str                  # 'car' | 'pedestrian' | 'unknown'
    moving: bool = False
    velocity: float = 0.0


# ============================================================
#  4. Symmetric TSP plan  ·  O(K log K)  (unchanged direction)
# ============================================================
class SymmetricTSPPlan:
    """
    The symmetric plan sorts destinations by position and keeps the
    direction unchanged: the vehicle always drives forward along the
    sorted 1D order.  This is O(K log K) once per plan.
    """
    def __init__(self, destinations: List[Destination]):
        self.destinations = destinations
        self.order: List[Destination] = []
        self.total_distance = 0.0

    def plan(self) -> Dict:
        t0 = time.perf_counter()
        # sort by position, preserving direction (left to right)
        positions = [d.pos for d in self.destinations]
        sorted_positions = merge_sort_1d(positions)
        pos_to_dest = {d.pos: d for d in self.destinations}
        self.order = [pos_to_dest[p] for p in sorted_positions]
        if self.order:
            self.total_distance = (self.order[-1].pos - self.order[0].pos)
        return dict(
            order=self.order,
            total_distance=self.total_distance,
            n=len(self.destinations),
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  5. Asymmetric reroute  ·  O(K log K)
# ============================================================
@dataclass
class RerouteAction:
    kind: str                    # 'follow' | 'u_turn' | 'turn_aside' | 'wait' | 'proceed'
    new_pos: float = 0.0
    direction: int = +1          # +1 forward, -1 backward
    reason: str = ""

    def as_tuple(self) -> Tuple[str, int]:
        return (self.kind, self.direction)


class AsymmetricRerouter:
    """
    When an obstruction blocks the symmetric path, the asymmetric
    rerouter finds the next best route in O(K log K):

        1. If a moving car leads the way and no lights exist → follow.
        2. If a stationary obstruction is close → safe U-turn.
        3. If a turn-aside lane exists → turn aside.
        4. Otherwise → wait.

    The reroute preserves the destination ordering but may flip the
    direction of travel to the nearest open waypoint.
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)

    def _next_open(self, current_pos: float, obstructions: List[Obstruction],
                   horizon: float = 30.0) -> Optional[float]:
        """Find the nearest open waypoint beyond the obstruction."""
        blocked = sorted([o.pos for o in obstructions])
        # look ahead
        for step in (1.0, 2.0, 5.0, 10.0, 20.0):
            candidate = current_pos + step
            if candidate > horizon:
                break
            if all(abs(candidate - b) > 2.0 for b in blocked):
                return candidate
        return None

    def _nearest_back(self, current_pos: float,
                      obstructions: List[Obstruction]) -> Optional[float]:
        """Find the nearest open waypoint behind."""
        blocked = sorted([o.pos for o in obstructions])
        for step in (1.0, 2.0, 5.0, 10.0, 20.0):
            candidate = current_pos - step
            if candidate < 0.0:
                break
            if all(abs(candidate - b) > 2.0 for b in blocked):
                return candidate
        return None

    def reroute(self, current_pos: float,
                obstructions: List[Obstruction],
                traffic: List[Obstruction],
                has_lights: bool,
                has_signs: bool) -> RerouteAction:
        t0 = time.perf_counter()

        # -- 1. no lights, no signs, but traffic ahead → follow
        if not has_lights and not has_signs and traffic:
            leaders = [t for t in traffic
                       if t.pos > current_pos and t.moving]
            if leaders:
                leaders.sort(key=lambda t: t.pos)
                leader = leaders[0]
                # follow at a safe offset
                new_pos = max(current_pos, leader.pos - 5.0)
                return RerouteAction(
                    kind="follow", new_pos=new_pos, direction=+1,
                    reason=f"traffic follow {leader.label} @ {leader.pos:.1f}",
                )

        # -- 2. stationary obstruction close → safe U-turn
        stationary = [o for o in obstructions
                      if not o.moving and 0.0 < o.pos - current_pos < 15.0]
        if stationary:
            back = self._nearest_back(current_pos, obstructions)
            if back is not None and back >= 0.0:
                return RerouteAction(
                    kind="u_turn", new_pos=back, direction=-1,
                    reason=f"safe U-turn, back to {back:.1f}",
                )

        # -- 3. turn aside: open lane exists beyond
        open_pos = self._next_open(current_pos, obstructions)
        if open_pos is not None:
            return RerouteAction(
                kind="turn_aside", new_pos=open_pos, direction=+1,
                reason=f"turn aside to {open_pos:.1f}",
            )

        # -- 4. otherwise wait
        return RerouteAction(
            kind="wait", new_pos=current_pos, direction=0,
            reason="no safe route found",
        )


# ============================================================
#  6. Deterministic algebraic fallback  ·  O(K)
# ============================================================
def linear_regression_fallback(destinations: List[Destination],
                               current_pos: float,
                               car_states: List[Obstruction]
                               ) -> float:
    """
    Deterministic linear-regression if/else over the current map state.
    Used when there are no lights, stop signs, or commands.

    Fits pos = a·dest_id + b on the destination set and returns the
    next best position along the fitted line.
    """
    if not destinations:
        return current_pos
    xs = np.array([d.dest_id for d in destinations], dtype=float)
    ys = np.array([d.pos for d in destinations], dtype=float)
    if len(xs) < 2:
        return float(ys[0])
    A = np.vstack([xs, np.ones_like(xs)]).T
    a, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    # predict the nearest destination ahead on the fitted line
    preds = a * xs + b
    ahead = preds[preds > current_pos]
    if ahead.size == 0:
        # if no destination ahead, take the farthest
        return float(preds.max())
    return float(ahead.min())


# ============================================================
#  7. Command clause language  ·  deterministic + algebraic
# ============================================================
@dataclass
class Clause:
    """A single clause: `if <pred> then <action>` or `while ... do ...`"""
    keyword: str               # 'if' | 'while' | 'else'
    predicate: Callable[[dict], bool]
    action: Callable[[dict], RerouteAction]

    def evaluate(self, ctx: dict) -> Optional[RerouteAction]:
        if self.keyword in ("if", "while"):
            if self.predicate(ctx):
                return self.action(ctx)
        elif self.keyword == "else":
            return self.action(ctx)
        return None


class CommandClauseEngine:
    """
    Holds a list of clauses; evaluates them in order; the first
    matching clause wins.  Unmatched situations fall back to the
    algebraic (linear-regression) rule.
    """
    def __init__(self, clauses: List[Clause]):
        self.clauses = clauses

    def evaluate(self, ctx: dict) -> Optional[RerouteAction]:
        for c in self.clauses:
            res = c.evaluate(ctx)
            if res is not None:
                return res
        return None


# ============================================================
#  8. The TSP map harness
# ============================================================
@dataclass
class MapOutput:
    current_pos: float
    destination: float
    action: RerouteAction
    clause_used: str
    symmetric_distance: float
    reroute_distance: float
    has_lights: bool
    has_signs: bool
    n_obstructions: int
    n_traffic: int
    elapsed_ms: float


class TSPMapHarness:
    """
    Full TSP map harness:

        plan (symmetric)                O(K log K)     once
        detect obstructions             O(K)
        evaluate clauses                O(K)           per frame
        asymmetric reroute              O(K log K)     if needed
        algebraic fallback              O(K)           if no commands
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)
        self.planner = SymmetricTSPPlan([])
        self.rerouter = AsymmetricRerouter(K)
        self.clauses: List[Clause] = []
        self.clause_engine: Optional[CommandClauseEngine] = None
        self.destinations: List[Destination] = []
        self.current_pos: float = 0.0
        self.history: List[MapOutput] = []

    # ---------- planning ----------
    def load_destinations(self, destinations: List[Destination]) -> Dict:
        self.destinations = destinations
        self.planner = SymmetricTSPPlan(destinations)
        return self.planner.plan()

    # ---------- clause registration ----------
    def register_clause(self, clause: Clause) -> None:
        self.clauses.append(clause)
        self.clause_engine = CommandClauseEngine(self.clauses)

    # ---------- frame processing ----------
    def process_frame(self,
                      obstructions: List[Obstruction],
                      traffic: List[Obstruction],
                      has_lights: bool,
                      has_signs: bool,
                      dt: float = 0.1) -> MapOutput:
        t0 = time.perf_counter()

        # 1. symmetric plan — unchanged direction
        plan = self.planner.plan()
        sym_dist = plan["total_distance"]

        # 2. build the current-context dict for clause evaluation
        ctx = dict(
            current_pos=self.current_pos,
            destinations=self.destinations,
            obstructions=obstructions,
            traffic=traffic,
            has_lights=has_lights,
            has_signs=has_signs,
            dt=dt,
            K=self.K,
        )

        # 3. evaluate clauses
        action: Optional[RerouteAction] = None
        clause_used = ""
        if self.clause_engine is not None:
            action = self.clause_engine.evaluate(ctx)
            if action is not None:
                # find which clause fired
                for c in self.clauses:
                    if c.keyword == "else":
                        clause_used = "else"
                        break
                    if c.predicate(ctx):
                        clause_used = c.keyword
                        break

        # 4. asymmetric reroute if no clause fired
        if action is None:
            action = self.rerouter.reroute(
                self.current_pos,
                obstructions=obstructions,
                traffic=traffic,
                has_lights=has_lights,
                has_signs=has_signs,
            )
            clause_used = action.kind

        # 5. algebraic fallback: no lights, no signs, no traffic → use
        #    the deterministic linear regression to pick the next position
        if not has_lights and not has_signs and not traffic and not obstructions:
            next_pos = linear_regression_fallback(
                self.destinations, self.current_pos, [])
            action = RerouteAction(
                kind="proceed", new_pos=next_pos, direction=+1,
                reason="algebraic linear-regression fallback",
            )
            clause_used = "algebraic"

        # 6. update position
        if action.direction != 0:
            self.current_pos = action.new_pos

        out = MapOutput(
            current_pos=self.current_pos,
            destination=self.destinations[-1].pos if self.destinations else 0.0,
            action=action,
            clause_used=clause_used,
            symmetric_distance=sym_dist,
            reroute_distance=abs(action.new_pos - self.current_pos),
            has_lights=has_lights,
            has_signs=has_signs,
            n_obstructions=len(obstructions),
            n_traffic=len(traffic),
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )
        self.history.append(out)
        return out


# ============================================================
#  9. Scene generator  ·  synthetic map frames
# ============================================================
def make_map_scene(frame_id: int, rng: random.Random,
                   n_dest: int = 12) -> Dict:
    """Build a synthetic map frame with destinations, obstructions,
    traffic, and command states."""
    # destinations: sorted positions along the road
    positions = sorted(rng.uniform(0, 200) for _ in range(n_dest))
    destinations = [
        Destination(pos=float(p),
                    dest_id=i,
                    kind="waypoint")
        for i, p in enumerate(positions)
    ]

    # obstructions
    obstructions = []
    if rng.random() < 0.4:
        obstructions.append(Obstruction(
            pos=float(rng.uniform(20, 150)),
            width=2.0,
            label="obstacle",
            moving=False,
        ))
    if rng.random() < 0.2:
        obstructions.append(Obstruction(
            pos=float(rng.uniform(20, 150)),
            width=2.5,
            label="pedestrian",
            moving=False,
        ))

    # traffic
    traffic = []
    if rng.random() < 0.5:
        n_traffic = rng.randint(1, 3)
        for _ in range(n_traffic):
            traffic.append(Obstruction(
                pos=float(rng.uniform(10, 180)),
                width=2.0,
                label="car",
                moving=True,
                velocity=float(rng.uniform(1.0, 3.0)),
            ))

    # commands (lights / signs)
    has_lights = (frame_id % 11 == 0)
    has_signs  = (frame_id % 7 == 0)

    return dict(
        destinations=destinations,
        obstructions=obstructions,
        traffic=traffic,
        has_lights=has_lights,
        has_signs=has_signs,
    )


# ============================================================
#  10. Demo
# ============================================================
def demo():
    print("=" * 82)
    print("TSP map harness  ·  symmetric + asymmetric + algebraic fallback")
    print("=" * 82)
    print(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  α_asym  = 0.3628     = {ALPHA_ASYM:.6f}")
    print(f"  K       = {K_DEFAULT}")
    print()

    harness = TSPMapHarness(K=K_DEFAULT)
    rng = random.Random(2024)

    # ---- register a few clauses ----
    def _red_light(ctx):
        return ctx["has_lights"] and any(
            o.label == "light_red" for o in ctx["obstructions"])

    def _stop_sign(ctx):
        return ctx["has_signs"] and any(
            o.label == "stop_sign" for o in ctx["obstructions"])

    def _obstruction_close(ctx):
        return any(0.0 < o.pos - ctx["current_pos"] < 15.0
                   for o in ctx["obstructions"])

    def _traffic_ahead(ctx):
        return any(t.moving and t.pos > ctx["current_pos"]
                   for t in ctx["traffic"])

    # -- clause 1: red light → full stop
    harness.register_clause(Clause(
        "if", _red_light,
        lambda ctx: RerouteAction(
            kind="wait", new_pos=ctx["current_pos"], direction=0,
            reason="red light"),
    ))

    # -- clause 2: stop sign → full stop
    harness.register_clause(Clause(
        "if", _stop_sign,
        lambda ctx: RerouteAction(
            kind="wait", new_pos=ctx["current_pos"], direction=0,
            reason="stop sign"),
    ))

    # -- clause 3: obstruction close → asymmetric reroute
    harness.register_clause(Clause(
        "if", _obstruction_close,
        lambda ctx: RerouteAction(
            kind="u_turn",
            new_pos=max(0.0, ctx["current_pos"] - 8.0),
            direction=-1,
            reason="obstruction close"),
    ))

    # -- clause 4: traffic ahead → follow
    harness.register_clause(Clause(
        "while", _traffic_ahead,
        lambda ctx: RerouteAction(
            kind="follow",
            new_pos=ctx["current_pos"] + 5.0,
            direction=+1,
            reason="follow traffic"),
    ))

    # ---- initial destination plan (symmetric) ----
    scene0 = make_map_scene(0, rng, n_dest=12)
    plan = harness.load_destinations(scene0["destinations"])
    print("--- Symmetric TSP plan ---")
    print(f"  destinations        : {plan['n']}")
    print(f"  total distance      : {plan['total_distance']:.2f}")
    print(f"  plan time           : {plan['elapsed_ms']:.2f} ms")
    print()

    # ---- frames ----
    print(f"{'frame':>5s}  {'pos':>7s}  {'cmd':>12s}  {'action':>11s}  "
          f"{'dir':>4s}  {'lights':>6s}  {'signs':>6s}  "
          f"{'obstr':>5s}  {'traf':>5s}  {'ms':>6s}")
    print("-" * 82)

    for fid in range(1, 41):
        scene = make_map_scene(fid, rng, n_dest=12)
        # re-plan from the current map if destinations changed
        if fid == 1 or fid % 10 == 0:
            harness.load_destinations(scene["destinations"])
        out = harness.process_frame(
            obstructions=scene["obstructions"],
            traffic=scene["traffic"],
            has_lights=scene["has_lights"],
            has_signs=scene["has_signs"],
        )
        print(f"{fid:>5d}  {out.current_pos:>7.2f}  "
              f"{out.clause_used:>12s}  {out.action.kind:>11s}  "
              f"{out.action.direction:>+4d}  "
              f"{'yes' if out.has_lights else 'no':>6s}  "
              f"{'yes' if out.has_signs else 'no':>6s}  "
              f"{out.n_obstructions:>5d}  {out.n_traffic:>5d}  "
              f"{out.elapsed_ms:>6.2f}")

    # ---- aggregate ----
    print()
    print("--- Aggregate over the run ---")
    n = len(harness.history)
    kinds = {}
    for o in harness.history:
        kinds[o.action.kind] = kinds.get(o.action.kind, 0) + 1
    clauses = {}
    for o in harness.history:
        clauses[o.clause_used] = clauses.get(o.clause_used, 0) + 1

    print(f"  frames                  : {n}")
    print(f"  action distribution     : {kinds}")
    print(f"  clause distribution     : {clauses}")
    print(f"  mean frame time         : "
          f"{np.mean([o.elapsed_ms for o in harness.history]):.2f} ms")
    print(f"  final position          : {harness.current_pos:.2f}")

    # ---- complexity ----
    print()
    print("--- Pipeline complexity ---")
    print("  symmetric TSP plan              O(K log K)      per plan")
    print("  clause evaluation               O(K)            per frame")
    print("  asymmetric reroute              O(K log K)      per obstruction")
    print("  algebraic fallback              O(K)            per frame")
    print("  total per frame                 O(K log K)")

    print("\nDone.")


if __name__ == "__main__":
    demo()