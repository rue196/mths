#!/usr/bin/env python3
"""
coding_pipeline_orchestrator.py
===============================

Full coding pipeline:

    STAGE 0  PLAN       planning_harness.py       log-rank residual
    STAGE 1  STORE      MOSB read-only buffer      plan residuals
    STAGE 2  ENCODE     prime_encoding_UTF_…       prime/composite
    STAGE 3  READ       MOSB read-only buffer      modular vector
    STAGE 4  CMD        coding_harness_3.py        8-bit + DL gate
    STAGE 5  ORCHESTRATE 6×6 block grid (B_in × B_out)

    Each stage → one row of the grid (B_in = n_stages)
    Each stage's encoding is partitioned into B_out blocks of 6×6
    Möbius gate μ(k_lin) ≠ 0 keeps ~6/π² of blocks

    Block classification:
        symmetric block   →  prime real encoding, direct bypass
        asymmetric block  →  composite complex encoding, MOSB lookup

    Jinja template renders the grid + per-block descriptors.
"""

from __future__ import annotations
import importlib.util
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Module loading (files have periods / parens)
# ============================================================
def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


planning   = _load("planning_harness", "planning_harness.py")
prime_utf  = _load("prime_enc",        "prime_encoding_UTF_coding_languages.py")
coding_h   = _load("coding_h",         "coding_harness_3.py")
wc         = _load("wc",               "weights_compression.py")

try:
    from jinja2 import Template
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False


# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)
A_STEP    = ALPHA / 0.3628
DENSITY   = 6.0 / (PI * PI)
BLOCK     = 6

K_ENCODE  = 60      # encoding table size
B_OUT_DEF = 10      # blocks per stage row


# ============================================================
#  Block descriptor
# ============================================================
@dataclass
class BlockInfo:
    bi:        int              # grid row   (stage index)
    bj:        int              # grid col   (block within stage)
    k_lin:     int              # linear gate index
    symmetric: bool             # Z = Zᵀ ?
    word:      int              # 32-bit spinor word
    cid:       int              # MOSB chunk id (−1 if symmetric)
    Z:         np.ndarray       # 6×6 decoded matrix
    Z_val:     float            # scalar signature (trace)
    X:         int              # grid X coord (bi * BLOCK)
    Y:         int              # grid Y coord (bj * BLOCK)
    label:     str              # short label for the template

    def tag(self) -> str:
        return "S" if self.symmetric else "A"

    def word_hex(self) -> str:
        return f"{self.word:08x}"


@dataclass
class StageResult:
    idx:           int
    name:          str
    plan_Z:        float
    plan_L:        float
    needed:        List[str]
    n_function:    int
    n_modular:     int
    mod_cid:       int
    ugc_survival:  float
    dl_blocked:    int
    bugs:          int
    blocks_placed: int
    blocks_total:  int
    dc_throughput: float
    elapsed_ms:    float


# ============================================================
#  O(n) 6×6 block grid orchestrator
# ============================================================
class BlockGrid:
    """
    Layered 6×6 block grid.

        rows   bi = stage index  (B_in = n_stages)
        cols   bj = block within stage (B_out)

    Grid coordinates:
        X = bi * BLOCK
        Y = bj * BLOCK

    Total blocks  =  B_in · B_out
    Total values  =  36 · B_in · B_out   (O(n) in grid size)

    Möbius gate:  block (bi, bj) stored iff μ(bi·B_out + bj) ≠ 0
    Symmetry rule: symmetric block → direct, asymmetric → MOSB
    """

    def __init__(self, B_in: int, B_out: int, mosb_buf, proj):
        self.B_in = B_in
        self.B_out = B_out
        self.mosb = mosb_buf
        self.proj = proj
        self.mu = planning.mobius_sieve(B_in * B_out + 1)
        self.blocks: Dict[Tuple[int, int], BlockInfo] = {}
        self._perm_cache: Dict[int, Tuple[int, ...]] = {}
        self._n_gated = 0

    # --------------------------------------------------------
    def stage_to_matrices(self, enc) -> np.ndarray:
        """
        Flatten one stage's encoding into B_out blocks of 6×6.

            flat = concat(func_vec.real, mod_vec.imag)
            pad to B_out · 36 values
            reshape → (B_out, 6, 6)
        """
        flat = np.concatenate([enc.func_vec.real, enc.mod_vec.imag])
        needed = self.B_out * 36
        if flat.size < needed:
            flat = np.pad(flat, (0, needed - flat.size))
        else:
            flat = flat[:needed]
        return flat.reshape(self.B_out, BLOCK, BLOCK)

    # --------------------------------------------------------
    def place_stage(self, stage_idx: int, enc) -> Tuple[int, int]:
        """Place one stage's encoding into row `stage_idx`."""
        matrices = self.stage_to_matrices(enc)
        placed = 0
        total = self.B_out

        for bj in range(self.B_out):
            k_lin = stage_idx * self.B_out + bj
            # Möbius gate
            if k_lin < self.mu.size and self.mu[k_lin] == 0:
                self._n_gated += 1
                continue

            Z = matrices[bj]
            # symmetry: is this 6×6 block symmetric?
            symmetric = bool(np.allclose(Z, Z.T, atol=1e-6))

            # project to 32-bit spinor word
            info = self.proj.project(Z)
            word = self.proj.encode(info)

            cid = -1
            if not symmetric:
                # store in MOSB
                payload = (struct.pack(">I", word)
                           + Z.astype(np.float64).tobytes())
                cid = self.mosb.append(payload)

            bi = stage_idx
            blk = BlockInfo(
                bi=bi, bj=bj, k_lin=k_lin,
                symmetric=symmetric, word=word, cid=cid,
                Z=Z, Z_val=float(np.trace(Z)),
                X=bi * BLOCK, Y=bj * BLOCK,
                label=f"s{bi}.b{bj}",
            )
            self.blocks[(bi, bj)] = blk
            placed += 1
        return placed, total

    # --------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x  : length B_in · 6, one 6-vector per stage
        out: length B_out · 6
        """
        x = np.asarray(x, dtype=np.float64).reshape(self.B_in, BLOCK)
        y = np.zeros((self.B_out, BLOCK), dtype=np.float64)

        for (bi, bj), blk in self.blocks.items():
            x_blk = x[bi]

            if blk.symmetric:
                W_blk = blk.Z
            else:
                chunk = self.mosb.read(blk.cid)
                if chunk is None:
                    continue
                payload = chunk[0]
                word_stored = struct.unpack(">I", payload[:4])[0]
                # decode to recover the 6×6
                W_blk = self.proj.decode(word_stored, self._perm_cache)

            y[bj] += W_blk.T @ x_blk
        return y.ravel()

    # --------------------------------------------------------
    def grid_rows(self) -> List[Dict]:
        """Grid view for the Jinja template."""
        rows = []
        for bi in range(self.B_in):
            row_blocks = []
            for bj in range(self.B_out):
                blk = self.blocks.get((bi, bj))
                if blk is None:
                    row_blocks.append(dict(
                        label=f"s{bi}.b{bj}",
                        tag=".",
                        word="--------",
                        Z_val=0.0,
                        symmetric=False,
                        cid=-1,
                    ))
                else:
                    row_blocks.append(dict(
                        label=blk.label,
                        tag=blk.tag(),
                        word=blk.word_hex(),
                        Z_val=blk.Z_val,
                        symmetric=blk.symmetric,
                        cid=blk.cid,
                    ))
            rows.append(dict(bi=bi, blocks=row_blocks))
        return rows

    # --------------------------------------------------------
    def summary(self) -> Dict:
        n_placed = len(self.blocks)
        n_sym = sum(1 for b in self.blocks.values() if b.symmetric)
        n_asym = n_placed - n_sym
        n_total = self.B_in * self.B_out
        return dict(
            B_in=self.B_in, B_out=self.B_out,
            n_blocks_total=n_total,
            n_blocks_placed=n_placed,
            n_gated=self._n_gated,
            n_sym=n_sym, n_asym=n_asym,
            density=n_placed / max(n_total, 1),
            ascii_grid=self.ascii_grid(),
        )

    def ascii_grid(self) -> str:
        L = ["    " + " ".join(f"b{bj:0>1}" for bj in range(self.B_out))]
        for bi in range(self.B_in):
            row = [f"r{bi}"]
            for bj in range(self.B_out):
                blk = self.blocks.get((bi, bj))
                row.append(blk.tag() if blk else ".")
            L.append(" ".join(row))
        return "\n".join(L)


# ============================================================
#  Full coding pipeline
# ============================================================
class CodingPipeline:
    def __init__(self,
                 K_encode: int = K_ENCODE,
                 B_out: int = B_OUT_DEF,
                 K_buffer: int = 256,
                 K_gate: int = 20):
        self.K = K_encode
        self.B_out = B_out

        # ---- planning ----
        self.planner = planning.PlanningHarness(K=512)

        # ---- encoding table ----
        self.table = prime_utf.CodingLanguageTable(K=K_encode)

        # ---- MOSB buffer (from the UTF-8 pipeline, has read) ----
        self.mosb = prime_utf.ReadOnlyBuffer(K=K_buffer)

        # ---- cmd harness ----
        self.harness = coding_h.BugRunHarness(
            K_utf8=K_encode, K_buffer=K_buffer, K_gate=K_gate)

        # ---- spinor projector ----
        self.proj = wc.SpinorProjector(mag_max=1.0, tr_max=1.0)

    # --------------------------------------------------------
    def run(self, requirement: str, code: str) -> Dict:
        t_start = time.perf_counter()
        stages: List[StageResult] = []
        B_in = 4
        grid = BlockGrid(B_in=B_in, B_out=self.B_out,
                         mosb_buf=self.mosb, proj=self.proj)

        # ---------- STAGE 0: PLAN ----------
        t0 = time.perf_counter()
        plan_rep = self.planner.plan(requirement, code, top_k=12)
        t_plan = (time.perf_counter() - t0) * 1e3

        # ---------- STAGE 1: STORE plan residuals in MOSB ----------
        plan_cids: Dict[str, int] = {}
        for r in plan_rep.needed:
            payload = struct.pack(">f", r.L) + r.token.encode()[:32]
            cid = self.mosb.append(payload)
            plan_cids[r.token] = cid

        stages.append(StageResult(
            idx=0, name="PLAN",
            plan_Z=float(plan_rep.Z_sum),
            plan_L=float(plan_rep.L_sum),
            needed=[r.token for r in plan_rep.needed[:8]],
            n_function=0, n_modular=0, mod_cid=-1,
            ugc_survival=0.0, dl_blocked=0, bugs=0,
            blocks_placed=0, blocks_total=0,
            dc_throughput=0.0,
            elapsed_ms=t_plan,
        ))

        # ---------- STAGE 2: ENCODE (reads plan from MOSB) ----------
        t0 = time.perf_counter()
        # read back the plan residuals to focus the encoding
        needed_set = set()
        for tok, cid in plan_cids.items():
            chunk = self.mosb.read(cid)
            if chunk is not None:
                L = struct.unpack(">f", chunk[0][:4])[0]
                if L > 0:
                    needed_set.add(tok)

        enc = prime_utf.encode_program(code, table=self.table)
        t_encode = (time.perf_counter() - t0) * 1e3

        stages.append(StageResult(
            idx=1, name="ENCODE",
            plan_Z=0.0, plan_L=0.0,
            needed=sorted(needed_set)[:8],
            n_function=enc.n_function,
            n_modular=enc.n_modular,
            mod_cid=-1,
            ugc_survival=0.0, dl_blocked=0, bugs=0,
            blocks_placed=0, blocks_total=0,
            dc_throughput=0.0,
            elapsed_ms=t_encode,
        ))

        # ---------- STAGE 3: STORE modular vector in MOSB ----------
        t0 = time.perf_counter()
        mod_payload = enc.mod_vec.astype(np.float64).tobytes()
        mod_cid = self.mosb.append(mod_payload)
        # read back to confirm
        chunk = self.mosb.read(mod_cid)
        mod_ok = chunk is not None and len(chunk[0]) == len(mod_payload)
        t_store = (time.perf_counter() - t0) * 1e3

        stages.append(StageResult(
            idx=2, name="MOSB READ",
            plan_Z=0.0, plan_L=0.0,
            needed=[],
            n_function=enc.n_function,
            n_modular=enc.n_modular,
            mod_cid=mod_cid,
            ugc_survival=1.0 if mod_ok else 0.0,
            dl_blocked=0, bugs=0,
            blocks_placed=0, blocks_total=0,
            dc_throughput=0.0,
            elapsed_ms=t_store,
        ))

        # ---------- STAGE 4: CMD run ----------
        t0 = time.perf_counter()
        bug_rep = self.harness.run(code)
        t_cmd = (time.perf_counter() - t0) * 1e3

        stages.append(StageResult(
            idx=3, name="CMD RUN",
            plan_Z=0.0, plan_L=0.0,
            needed=bug_rep.bug_tokens[:8],
            n_function=enc.n_function,
            n_modular=enc.n_modular,
            mod_cid=mod_cid,
            ugc_survival=bug_rep.ugc_survival_rate,
            dl_blocked=bug_rep.dl_blocked,
            bugs=bug_rep.bugs_detected,
            blocks_placed=0, blocks_total=0,
            dc_throughput=bug_rep.dc_throughput,
            elapsed_ms=t_cmd,
        ))

        # ---------- STAGE 5: ORCHESTRATE GRID ----------
        t0 = time.perf_counter()
        # fill grid rows: one per stage (padded with empty encoding if needed)
        encodings = [enc, enc, enc, enc]   # simple: all 4 stages use the same
        for bi, e in enumerate(encodings):
            placed, total = grid.place_stage(bi, e)
            stages[bi].blocks_placed = placed
            stages[bi].blocks_total = total

        # sample forward pass
        rng = np.random.default_rng(42)
        x = rng.standard_normal(B_in * BLOCK) * 0.5
        y = grid.forward(x)
        t_grid = (time.perf_counter() - t0) * 1e3

        # ---------- assemble report ----------
        total_ms = (time.perf_counter() - t_start) * 1e3
        return dict(
            requirement=requirement,
            code=code,
            plan=plan_rep,
            encoding=enc,
            bug_report=bug_rep,
            grid=grid,
            stages=stages,
            forward_out=y,
            forward_norm=float(np.linalg.norm(y)),
            grid_summary=grid.summary(),
            grid_rows=grid.grid_rows(),
            total_ms=total_ms,
            plan_cids=plan_cids,
            mod_cid=mod_cid,
        )


# ============================================================
#  Jinja template
# ============================================================
TEMPLATE_STR = """\
# Coding pipeline report
========================

Requirement: {{ result.requirement|truncate(80) }}
Code length: {{ result.code|length }} chars

## Stages

{% for s in result.stages %}
### Stage {{ s.idx }}: {{ s.name }}

| metric | value |
|---|---|
| plan Z residual  | {{ "%.4f"|format(s.plan_Z) }} |
| plan L weighted  | {{ "%.4f"|format(s.plan_L) }} |
| function tokens  | {{ s.n_function }} |
| modular tokens   | {{ s.n_modular }} |
| MOSB chunk id    | {{ s.mod_cid }} |
| UGC survival     | {{ "%.4f"|format(s.ugc_survival) }} |
| DL blocked       | {{ s.dl_blocked }} |
| bugs detected    | {{ s.bugs }} |
| DC throughput    | {{ "%.6f"|format(s.dc_throughput) }} |
| blocks placed    | {{ s.blocks_placed }} / {{ s.blocks_total }} |
| elapsed ms       | {{ "%.2f"|format(s.elapsed_ms) }} |

{% if s.needed %}
Needed tokens: `{{ s.needed|join("`, `") }}`
{% endif %}

{% endfor %}

## Grid  ({{ result.grid_summary.B_in }} × {{ result.grid_summary.B_out }} blocks)



Legend: `S` = symmetric (prime real, direct), `A` = asymmetric (composite complex, MOSB), `.` = μ-gated out

## Grid summary

- total blocks:    {{ result.grid_summary.n_blocks_total }}
- placed blocks:   {{ result.grid_summary.n_blocks_placed }}
- gated by μ=0:    {{ result.grid_summary.n_gated }}
- symmetric:       {{ result.grid_summary.n_sym }}
- asymmetric:      {{ result.grid_summary.n_asym }}
- density:         {{ "%.4f"|format(result.grid_summary.density) }}  (target 6/π² = 0.6079)

## Forward pass

- output norm:     {{ "%.4f"|format(result.forward_norm) }}
- total time:      {{ "%.2f"|format(result.total_ms) }} ms
"""


def render_report(result: Dict) -> str:
    if HAS_JINJA:
        return Template(TEMPLATE_STR).render(result=result)
    # fallback: manual format
    L = []
    L.append("# Coding pipeline report")
    L.append("=" * 30)
    L.append(f"Requirement: {result['requirement'][:80]}")
    L.append(f"Code length: {len(result['code'])}")
    L.append("")
    L.append("## Stages")
    for s in result['stages']:
        L.append(f"### Stage {s.idx}: {s.name}")
        L.append(f"  plan Z       = {s.plan_Z:+.4f}")
        L.append(f"  plan L       = {s.plan_L:+.4f}")
        L.append(f"  functions    = {s.n_function}")
        L.append(f"  modular      = {s.n_modular}")
        L.append(f"  mod_cid      = {s.mod_cid}")
        L.append(f"  UGC survival = {s.ugc_survival:.4f}")
        L.append(f"  DL blocked   = {s.dl_blocked}")
        L.append(f"  bugs         = {s.bugs}")
        L.append(f"  blocks       = {s.blocks_placed}/{s.blocks_total}")
        L.append("")
    L.append("## Grid")
    L.append("```")
    L.append(result['grid_summary']['ascii_grid'])
    L.append("```")
    return "\n".join(L)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Coding pipeline orchestrator  ·  plan → encode → MOSB → cmd → grid")
    print("=" * 78)
    print(f"  α       = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print(f"  6/π²                  = {DENSITY:.6f}")
    print(f"  Jinja available       = {HAS_JINJA}")
    print()

    # ---------- inputs ----------
    requirement = (
        "Add multiply and subtract functions to the calculator. "
        "Both should take two arguments a and b. "
        "Keep the existing add function working."
    )
    code = """
def add(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value

def main():
    c = Calculator()
    c.add(5)
    print(c.value)
"""

    print(f"--- requirement ---")
    print(f"  {requirement[:70]}…" if len(requirement) > 70 else f"  {requirement}")
    print(f"--- code ---")
    print(f"  {len(code)} chars, {code.count(chr(10))} lines")
    print()

    # ---------- run pipeline ----------
    pipe = CodingPipeline(K_encode=60, B_out=10, K_buffer=256, K_gate=20)
    print(f"--- running pipeline ---")
    t0 = time.perf_counter()
    result = pipe.run(requirement, code)
    total = (time.perf_counter() - t0) * 1e3

    # ---------- print stages ----------
    print(f"--- stage timings ---")
    print(f"  {'stage':<14s}  {'ms':>8s}  notes")
    for s in result["stages"]:
        note = ""
        if s.name == "PLAN":
            note = f"Z={s.plan_Z:+.2f}  L={s.plan_L:+.2f}"
        elif s.name == "ENCODE":
            note = (f"func={s.n_function}  mod={s.n_modular}  "
                    f"needed={len(s.needed)}")
        elif s.name == "MOSB READ":
            note = f"mod_cid={s.mod_cid}  ok={s.ugc_survival > 0.5}"
        elif s.name == "CMD RUN":
            note = (f"survival={s.ugc_survival:.3f}  "
                    f"blocked={s.dl_blocked}  bugs={s.bugs}")
        print(f"  {s.name:<14s}  {s.elapsed_ms:>8.2f}  {note}")
    print(f"  {'TOTAL':<14s}  {total:>8.2f} ms")
    print()

    # ---------- MOSB state ----------
    print(f"--- MOSB buffer ---")
    print(f"  chunks            = {result['grid_summary']['n_blocks_placed']}")
    print(f"  plan cids         = {len(result['plan_cids'])}")
    print(f"  mod cid           = {result['mod_cid']}")
    print()

    # ---------- grid ----------
    print(f"--- grid  ({result['grid_summary']['B_in']} × "
          f"{result['grid_summary']['B_out']}) ---")
    print(result["grid_summary"]["ascii_grid"])
    print()
    print(f"  total blocks      = {result['grid_summary']['n_blocks_total']}")
    print(f"  placed blocks     = {result['grid_summary']['n_blocks_placed']}")
    print(f"  μ-gated out       = {result['grid_summary']['n_gated']}")
    print(f"  symmetric         = {result['grid_summary']['n_sym']}")
    print(f"  asymmetric        = {result['grid_summary']['n_asym']}")
    print(f"  density           = {result['grid_summary']['density']:.4f}  "
          f"(target 6/π² = {DENSITY:.4f})")
    print()

    # ---------- forward ----------
    print(f"--- grid forward pass ---")
    print(f"  output |y|        = {result['forward_norm']:.4f}")
    print(f"  output shape      = {result['forward_out'].shape}")
    print()

    # ---------- Jinja template ----------
    print("=" * 78)
    print("JINJA-TEMPLATED REPORT")
    print("=" * 78)
    print(render_report(result))

    # ---------- complexity ----------
    print()
    print("=" * 78)
    print("Complexity")
    print("=" * 78)
    print("  STAGE 0  PLAN")
    print("    tokenize + log-rank      O(K)")
    print("    elliptic projection      O(K)")
    print("    residual projection      O(K log K)")
    print("  STAGE 1  STORE")
    print("    MOSB append per token    O(1) each, O(K) total")
    print("  STAGE 2  ENCODE")
    print("    prime/composite map      O(1) per token")
    print("    |Ci| build               O(|tokens|)")
    print("    finite-step normalise    O(K)")
    print("  STAGE 3  MOSB READ")
    print("    chunk read               O(|chunk|)  =  O(K)")
    print("  STAGE 4  CMD")
    print("    UGC boolean              O(K · K_utf8)")
    print("    DL gate                  O(K · K_buffer)")
    print("  STAGE 5  GRID")
    print("    per block proj/encode    O(720)      (6×6 Hungarian)")
    print("    per block MOSB append    O(36)")
    print("    forward per block        O(36)       (6×6 matmul)")
    print("    ─────────────────────────────────────────────")
    print("  total blocks               B_in · B_out = O(n)")
    print("  total forward              O(B_in · B_out · 36) = O(n)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()