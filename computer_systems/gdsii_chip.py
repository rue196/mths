#!/usr/bin/env python3
"""
alu_chip_gdsii.py
=================

GDSII chip layout for the Möbius ALU pipeline.

Layout priorities (highest → lowest)
------------------------------------
    P1  ALU cells          QuantumALU  (alu_finite_step.py)
    P2  Logic gate cells   ChipLogicGate  (mobius_gates.py)
    P3  Arithmetic cores   OKRealClock  (O(K)_conv_tau_muon.py)

Placement
---------
    ALU cells      → priority row, packed left-to-right
    Logic gates    → priority row, packed left-to-right
    Cores          → 1D TSP order  (tspsp.tsp_1d)

Clock
-----
    The clock tree is generated from OKRealClock.conv_speed
    = |dM/dk| where  M[k] = S_mu[k] · S_tau[k].

    Both the convolution and the derivative are O(K)  →
    the entire clock tree is O(K).

Layers
------
    1   ALU cells (poly/diffusion body + chirp strip)
    2   Logic gate cells
    3   Arithmetic cores
    4   Clock distribution network
    5   Power / ground rails
    10  Text labels

Fallback
--------
    If gdspy is not installed, an SVG file is written instead —
    identical geometry, human-viewable, and still shows the
    O(K) convolution profile as a clock bar chart.
"""

from __future__ import annotations
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- gdspy optional ----
try:
    import gdspy
    HAS_GDSPY = True
except ImportError:
    HAS_GDSPY = False

# ---- building blocks ----
from alu_finite_step import (
    QuantumALU, A_STEP, ALPHA, K_DEFAULT as ALU_K_DEFAULT,
)
from mobius_gates import ChipLogicGate, ChipProcessor
import tspsp


from OK_conv_tau_muon import OKRealClock

# ============================================================
#  GDSII layers
# ============================================================
L_ALU      = 1
L_LOGIC    = 2
L_CORE     = 3
L_CLOCK    = 4
L_POWER    = 5
L_LABEL    = 10

_LAYER_COLOR = {
    L_ALU:   "#3a7bd5",
    L_LOGIC: "#e67e22",
    L_CORE:  "#16a085",
    L_CLOCK: "#e74c3c",
    L_POWER: "#888888",
}


# ============================================================
#  CellSpec
# ============================================================
@dataclass
class CellSpec:
    name:     str
    w_um:     float
    h_um:     float
    layer:    int
    priority: int
    ideal_x:  float = 0.0
    ideal_y:  float = 0.0
    meta:     Dict  = field(default_factory=dict)


# ============================================================
#  Cell builders
# ============================================================
def build_alu_cells(alu: QuantumALU, n_cells: int = 16) -> List[CellSpec]:
    """ALU cells — one per chirp segment, sized by local chirp amplitude."""
    cells: List[CellSpec] = []
    K = alu.K
    for i in range(n_cells):
        lo = i * K // n_cells
        hi = (i + 1) * K // n_cells
        seg = alu.w[lo:hi]
        amp = float(np.max(np.abs(seg))) if seg.size else 0.5
        w_um = 8.0 + 2.0 * amp
        h_um = 6.0
        cells.append(CellSpec(
            name=f"ALU_{i:02d}",
            w_um=w_um, h_um=h_um,
            layer=L_ALU, priority=1,
            meta=dict(chirp_amp=amp, A_STEP=A_STEP, K=K),
        ))
    return cells


def build_logic_cells(gate: ChipLogicGate,
                      n_cells: int = 8) -> List[CellSpec]:
    """Logic gate cells — bounded log / exp / sin / cos / trace."""
    fns = ['log', 'exp', 'sin', 'cos', 'trace']
    cells: List[CellSpec] = []
    for i in range(n_cells):
        fn = fns[i % len(fns)]
        cells.append(CellSpec(
            name=f"GATE_{i:02d}_{fn}",
            w_um=5.0, h_um=4.0,
            layer=L_LOGIC, priority=2,
            meta=dict(func=fn),
        ))
    return cells


def build_core_cells(clock: OKRealClock,
                     n_cores: int = 4) -> List[CellSpec]:
    """Arithmetic cores — one per OKRealClock block."""
    span = float(clock.S_mu.max() - clock.S_mu.min())
    cells: List[CellSpec] = []
    for i in range(n_cores):
        w_um = 15.0 + 0.5 * abs(span)
        h_um = 12.0
        cells.append(CellSpec(
            name=f"CORE_{i:02d}",
            w_um=w_um, h_um=h_um,
            layer=L_CORE, priority=3,
            meta=dict(freq=clock.frequency(), K=clock.K),
        ))
    return cells


# ============================================================
#  Placement
# ============================================================
def place_cores_tsp(cores: List[CellSpec],
                    chip_w_um: float) -> Tuple[List[CellSpec], float]:
    """
    Place cores along X using 1D TSP  (tspsp.tsp_1d).

    The TSP gives the optimal cycle over the ideal positions;
    cores are then assigned to evenly spaced slots in TSP order.
    """
    n = len(cores)
    if n == 0:
        return cores, 0.0

    slot_w = chip_w_um / n
    ideal_xs = np.array([(i + 0.5) * slot_w for i in range(n)])
    for c, x in zip(cores, ideal_xs):
        c.ideal_x = float(x)

    _, dist = tspsp.tsp_1d(ideal_xs.tolist())

    # TSP tours 1D points in sorted order
    cores_sorted = sorted(cores, key=lambda c: c.ideal_x)
    for rank, c in enumerate(cores_sorted):
        c.ideal_x = (rank + 0.5) * slot_w
    return cores_sorted, float(dist)


def place_priority_row(cells: List[CellSpec],
                       chip_w_um: float, y_um: float) -> None:
    """Pack cells left-to-right in one row at y_um, priority ordered."""
    if not cells:
        return
    total_w = sum(c.w_um for c in cells) + 1.0 * (len(cells) - 1)
    x = (chip_w_um - total_w) / 2.0
    for c in sorted(cells, key=lambda c: c.priority):
        c.ideal_x = x + c.w_um / 2.0
        c.ideal_y = y_um
        x += c.w_um + 1.0


# ============================================================
#  O(K) clock tree from convolution speed
# ============================================================
def clock_peaks(clock: OKRealClock,
                threshold: float = 0.5) -> np.ndarray:
    """Local maxima of conv_speed above a fraction of its peak."""
    cs = clock.conv_speed
    if cs.size < 3:
        return np.array([], dtype=int)
    thresh = float(cs.max()) * threshold
    peaks = []
    for i in range(1, cs.size - 1):
        if (cs[i] > cs[i-1] and cs[i] > cs[i+1]
                and cs[i] >= thresh):
            peaks.append(i)
    return np.array(peaks, dtype=int)


# ============================================================
#  GDSII drawing  ·  coordinates in µm (user units)
# ============================================================
def _draw_alu_cell(top, spec: CellSpec) -> None:
    """ALU cell: body rect + chirp strip inside."""
    x0 = spec.ideal_x - spec.w_um / 2.0
    x1 = spec.ideal_x + spec.w_um / 2.0
    y0 = spec.ideal_y - spec.h_um / 2.0
    y1 = spec.ideal_y + spec.h_um / 2.0
    top.add(gdspy.Rectangle((x0, y0), (x1, y1), layer=L_ALU))

    # chirp waveform strip
    n_seg = 24
    seg_w = (x1 - x0) / n_seg
    amp = spec.meta.get("chirp_amp", 0.5)
    for i in range(n_seg):
        phase = (i + 0.5) / n_seg * 4 * math.pi
        h = amp * math.cos(A_STEP * phase * phase / 4.0) * 1.4
        sx = x0 + (i + 0.5) * seg_w
        top.add(gdspy.Rectangle(
            (sx - seg_w * 0.25, spec.ideal_y + h - 0.15),
            (sx + seg_w * 0.25, spec.ideal_y + h + 0.15),
            layer=L_ALU,
        ))


def _draw_logic_cell(top, spec: CellSpec) -> None:
    x0 = spec.ideal_x - spec.w_um / 2.0
    x1 = spec.ideal_x + spec.w_um / 2.0
    y0 = spec.ideal_y - spec.h_um / 2.0
    y1 = spec.ideal_y + spec.h_um / 2.0
    top.add(gdspy.Rectangle((x0, y0), (x1, y1), layer=L_LOGIC))
    # inner "gate symbol" box
    m = 0.8
    top.add(gdspy.Rectangle(
        (x0 + m, y0 + m), (x1 - m, y1 - m),
        layer=L_LOGIC,
    ))


def _draw_core_cell(top, spec: CellSpec) -> None:
    x0 = spec.ideal_x - spec.w_um / 2.0
    x1 = spec.ideal_x + spec.w_um / 2.0
    y0 = spec.ideal_y - spec.h_um / 2.0
    y1 = spec.ideal_y + spec.h_um / 2.0
    top.add(gdspy.Rectangle((x0, y0), (x1, y1), layer=L_CORE))
    # register-bank strips
    n_bank = 6
    bh = (y1 - y0) / n_bank
    for i in range(n_bank):
        top.add(gdspy.Rectangle(
            (x0 + 0.5, y0 + i * bh + 0.3),
            (x1 - 0.5, y0 + (i + 1) * bh - 0.3),
            layer=L_CORE,
        ))


def _draw_clock_profile(top, clock: OKRealClock,
                        chip_w_um: float, y_um: float
                        ) -> List[float]:
    """
    Draw the O(K) convolution clock profile along the chip width.

        M[k] = S_mu[k] · S_tau[k]          (mirror product, O(K))
        speed[k] = |dM/dk|                  (O(K) derivative)
    """
    cs = clock.conv_speed
    if cs.size == 0:
        return []
    cmax = float(cs.max()) if cs.max() > 0 else 1.0
    heights_um = (cs / cmax) * 15.0

    n = cs.size
    xs = np.arange(n) / max(n - 1, 1) * chip_w_um

    # profile bars
    for x, h in zip(xs, heights_um):
        if h < 0.1:
            continue
        top.add(gdspy.Rectangle(
            (x - 0.5, y_um), (x + 0.5, y_um + h),
            layer=L_CLOCK,
        ))
    # baseline rail
    top.add(gdspy.Rectangle(
        (0, y_um - 0.3), (chip_w_um, y_um + 0.3),
        layer=L_CLOCK,
    ))

    # clock buffers at peaks
    peaks = clock_peaks(clock, threshold=0.5)
    pts: List[float] = []
    for p in peaks:
        x = float(xs[p])
        pts.append(x)
        top.add(gdspy.Rectangle(
            (x - 0.8, y_um - 0.8), (x + 0.8, y_um + 0.8),
            layer=L_CLOCK,
        ))
    return pts


def _draw_power_rails(top, chip_w_um: float, chip_h_um: float) -> None:
    for y in (chip_h_um * 0.04, chip_h_um * 0.96):
        top.add(gdspy.Rectangle(
            (chip_w_um * 0.02, y - 1.5),
            (chip_w_um * 0.98, y + 1.5),
            layer=L_POWER,
        ))


def _draw_labels(top, cells: List[CellSpec]) -> None:
    for c in cells:
        top.add(gdspy.Label(
            c.name,
            (c.ideal_x, c.ideal_y + c.h_um / 2.0 + 1.5),
            layer=L_LABEL,
        ))


# ============================================================
#  Main layout function
# ============================================================
def layout_chip(
        out_path: str = "alu_chip.gds",
        chip_w_um: float = 500.0,
        chip_h_um: float = 500.0,
        alu_K: int = 64,
        n_alu: int = 16,
        n_gates: int = 8,
        n_cores: int = 4,
        clock_K: int = 20,
        seed: int = 42,
        verbose: bool = True,
) -> Dict:
    """
    Build the chip layout.  Returns a dict of layout statistics.

    If gdspy is not installed, writes an SVG file instead.
    """
    # ---- 1. component instances ----
    alu    = QuantumALU(K=alu_K)
    gate   = ChipLogicGate(max_K=alu_K * 2)
    clock  = OKRealClock(K=clock_K, J_C=1.0, smooth=True,
                         tick_tol=1e-9, use_fft=False)

    alu_cells   = build_alu_cells(alu, n_cells=n_alu)
    logic_cells = build_logic_cells(gate, n_cells=n_gates)
    core_cells  = build_core_cells(clock, n_cores=n_cores)

    # ---- 2. TSP core placement ----
    # shuffle first so the TSP is doing real work
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(core_cells))
    core_cells = [core_cells[i] for i in order]

    core_cells, tsp_dist = place_cores_tsp(core_cells, chip_w_um)

    # ---- 3. priority rows ----
    y_alu   = chip_h_um * 0.22
    y_logic = chip_h_um * 0.42
    y_core  = chip_h_um * 0.78
    place_priority_row(alu_cells, chip_w_um, y_alu)
    place_priority_row(logic_cells, chip_w_um, y_logic)
    for c in core_cells:
        c.ideal_y = y_core

    y_clock = chip_h_um * 0.58

    # ---- 4. write GDSII or SVG ----
    all_cells = alu_cells + logic_cells + core_cells
    clock_pts: List[float] = []
    out_format: str

    if HAS_GDSPY:
        lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
        top = lib.new_cell("TOP")
        for c in alu_cells:
            _draw_alu_cell(top, c)
        for c in logic_cells:
            _draw_logic_cell(top, c)
        for c in core_cells:
            _draw_core_cell(top, c)
        clock_pts = _draw_clock_profile(top, clock, chip_w_um, y_clock)
        _draw_power_rails(top, chip_w_um, chip_h_um)
        _draw_labels(top, all_cells)
        lib.write_gds(out_path)
        out_format = "GDSII"
    else:
        _write_svg_fallback(
            out_path.replace(".gds", ".svg"),
            all_cells, clock, chip_w_um, chip_h_um, y_clock,
        )
        clock_pts = [
            float(p) / max(clock.conv_speed.size - 1, 1) * chip_w_um
            for p in clock_peaks(clock, threshold=0.5)
        ]
        out_format = "SVG (gdspy not installed)"
        out_path = out_path.replace(".gds", ".svg")

    stats = dict(
        out_path=out_path,
        out_format=out_format,
        chip_w_um=chip_w_um,
        chip_h_um=chip_h_um,
        n_alu=len(alu_cells),
        n_logic=len(logic_cells),
        n_cores=len(core_cells),
        tsp_distance_um=tsp_dist,
        clock_peaks_x_um=clock_pts,
        n_clock_peaks=len(clock_pts),
        clock_frequency=clock.frequency(),
        clock_K=clock.K,
        clock_convolution_mode="mirror (O(K))",
        alu_K=alu.K,
    )

    if verbose:
        _print_summary(stats, alu, gate, clock, core_cells)
    return stats


# ============================================================
#  SVG fallback
# ============================================================
def _write_svg_fallback(path: str, cells: List[CellSpec],
                        clock: OKRealClock,
                        w: float, h: float, y_clock: float) -> None:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}mm" height="{h}mm" '
        f'viewBox="0 0 {w} {h}" font-family="monospace">',
        f'<rect width="100%" height="100%" fill="#fafafa"/>',
    ]
    for c in cells:
        x = c.ideal_x - c.w_um / 2.0
        y = h - (c.ideal_y + c.h_um / 2.0)
        color = _LAYER_COLOR.get(c.layer, "#888")
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" '
            f'width="{c.w_um:.2f}" height="{c.h_um:.2f}" '
            f'fill="{color}" fill-opacity="0.55" '
            f'stroke="#111" stroke-width="0.15"/>')
        lines.append(
            f'<text x="{c.ideal_x:.2f}" '
            f'y="{h - c.ideal_y + 1.5:.2f}" '
            f'font-size="2.2" text-anchor="middle">{c.name}</text>')

    # clock profile
    cs = clock.conv_speed
    cmax = float(cs.max()) if cs.size and cs.max() > 0 else 1.0
    n = cs.size
    for i, v in enumerate(cs):
        x = i / max(n - 1, 1) * w
        bar_h = v / cmax * 15.0
        yb = h - (y_clock + bar_h)
        lines.append(
            f'<rect x="{x - 0.4:.2f}" y="{yb:.2f}" '
            f'width="0.8" height="{bar_h:.2f}" '
            f'fill="#e74c3c" fill-opacity="0.7"/>')
    lines.append(
        f'<line x1="0" y1="{h - y_clock:.2f}" '
        f'x2="{w}" y2="{h - y_clock:.2f}" '
        f'stroke="#e74c3c" stroke-width="0.3"/>')
    lines.append('</svg>')

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ============================================================
#  Summary printer
# ============================================================
def _print_summary(stats: Dict, alu: QuantumALU,
                   gate: ChipLogicGate, clock: OKRealClock,
                   core_cells: List[CellSpec]) -> None:
    print("=" * 78)
    print("ALU chip GDSII layout")
    print("=" * 78)
    print(f"  output             = {stats['out_path']}")
    print(f"  format             = {stats['out_format']}")
    print(f"  chip area          = {stats['chip_w_um']} µm × "
          f"{stats['chip_h_um']} µm")
    print()

    print("--- layout priority ---")
    print(f"  P1  ALU cells      = {stats['n_alu']}   "
          f"(QuantumALU, A_STEP={A_STEP:.4f}, K={alu.K})")
    print(f"  P2  Logic gates    = {stats['n_logic']}   "
          f"(log/exp/sin/cos/trace)")
    print(f"  P3  Arithmetic     = {stats['n_cores']}   "
          f"(OKRealClock, K={clock.K})")
    print()

    print("--- 1D TSP core placement ---")
    print(f"  cores              = {len(core_cells)}")
    print(f"  TSP tour length    = {stats['tsp_distance_um']:.2f} µm")
    for c in core_cells:
        print(f"    {c.name:>8s}  x = {c.ideal_x:7.2f} µm   "
              f"freq = {c.meta.get('freq', 0):.4f}")
    print()

    print("--- O(K) clock tree  ·  convolution speed  |dM/dk| ---")
    print(f"  clock K            = {clock.K}")
    print(f"  convolution mode   = {stats['clock_convolution_mode']}")
    print(f"  clock frequency    = {stats['clock_frequency']:.6f}")
    print(f"  clock speed mean   = {clock.clock_speed():.6f}")
    print(f"  clock speed peak   = {clock.peak_speed():.6f}")
    print(f"  clock peaks        = {stats['n_clock_peaks']}  "
          f"at x = [{', '.join(f'{x:.1f}' for x in stats['clock_peaks_x_um'])}] µm")
    print()

    print("--- invariants ---")
    n_gate = int(np.count_nonzero(alu.gate))
    print(f"  ALU μ-gate density = {n_gate}/{alu.K} = "
          f"{n_gate / alu.K:.4f}   (6/π² = {6/math.pi**2:.4f})")
    S, H, m = clock.supertrace() if hasattr(clock, "supertrace") else (0, 0, 0)
    print(f"  ALU ⟨w,w⟩          = {alu.w_norm:.6f}")
    print(f"  clock ‖G_ee‖       = {clock.norm_ee:.4f}   "
          f"‖G_eo‖ = {clock.norm_eo:.4f}   "
          f"ratio = {clock.parity_ratio:.3f}")
    print()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("ALU chip GDSII  ·  1D-TSP cores  ·  O(K) convolution clock")
    print("=" * 78)
    print(f"  α        = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  A_STEP   = α / 0.3628    = {A_STEP:.6f}")
    print(f"  gdspy    = {'available' if HAS_GDSPY else 'MISSING → SVG fallback'}")
    print()

    stats = layout_chip(
        out_path="alu_chip.gds",
        chip_w_um=500.0, chip_h_um=500.0,
        alu_K=64, n_alu=16,
        n_gates=8,
        n_cores=4,
        clock_K=20,
        seed=42,
        verbose=True,
    )

    # ---- verify the output file ----
    print("--- verification ---")
    if os.path.exists(stats["out_path"]):
        size = os.path.getsize(stats["out_path"])
        print(f"  file written       = {stats['out_path']}")
        print(f"  size               = {size:,} bytes")
    else:
        print(f"  MISSING: {stats['out_path']}")
    print()

    # ---- if gdspy, read back and count layers ----
    if HAS_GDSPY and stats["out_format"] == "GDSII":
        print("--- GDSII read-back ---")
        lib = gdspy.GdsLibrary(infile=stats["out_path"])
        top = lib.top_level()[0]
        layers_seen = sorted({p.layer for p in top.get_polygons()})
        labels_seen = len(top.labels) if hasattr(top, "labels") else 0
        print(f"  cells              = {len(lib.cells)}")
        print(f"  polygons (top)     = {len(top.get_polygons())}")
        print(f"  layers present     = {layers_seen}")
        print(f"  labels             = {labels_seen}")
        expected_layers = [L_ALU, L_LOGIC, L_CORE, L_CLOCK, L_POWER]
        present = all(l in layers_seen for l in expected_layers)
        print(f"  all P1–P5 layers   = {'OK' if present else 'MISSING'}")
        print()

    # ---- complexity ----
    print("=" * 78)
    print("Complexity")
    print("=" * 78)
    print("  cell build (ALU)        O(K_alu)             per cell")
    print("  cell build (logic)      O(1)                 per cell")
    print("  cell build (cores)      O(1)                 per cell")
    print("  1D TSP placement        O(n log n)           tspsp.tsp_1d")
    print("  priority-row packing    O(n)                 one pass")
    print("  clock convolution       O(K_clock)           S_mu · S_tau")
    print("  clock derivative        O(K_clock)           |dM/dk|")
    print("  clock peak detect       O(K_clock)           single pass")
    print("  GDSII emit              O(n_cells + K_clock) rectangles")
    print()
    print("  Total layout time:  O(n log n) + O(K_clock)  →  O(n log n)")
    print()
    print(f"  The clock tree is driven by  |dM/dk|  where")
    print(f"  M[k] = S_mu[k] · S_tau[k]  — both O(K), so the entire")
    print(f"  clock distribution network is O(K) to generate.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()