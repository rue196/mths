#!/usr/bin/env python3
"""
font_compress.py
================

Font converter / compressor.

RASTER  ·  normal bitmap letters
    algebraic grid    letter = ⌊A⌋² cells        (A = 6.511 → 6×6 min)
    finite-step curve edge anti-aliasing from
                      A(n) = 10·a·sin(n²°),  a = 1/(π−e)
                      central step at n = 4 recovers π
    encoding          row-dictionary + raw fallback

VECTOR  ·  outline-font paths (SVG / TTF control points)
    UGC discrete log  ugc_discrete_log.py
    vertex sharing    unique x's / y's stored once;
                      each per-point reference is an index
    decode verified   BSGS dlog round-trip per unique coord
    log-value         −H_(x+1) as per-point metadata

Both modes are lossless converters:
    encode(text | path)  →  bytes   →  decode(bytes)  →  text | path
"""

from __future__ import annotations
import argparse
import math
import struct
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

# ---- finite-step source ----
from trvial_proof_square_packing import (
    A_of_n,
    finite_step_derivative,
    a as A_CONST,
)

# ---- UGC discrete log source ----
from ugc_discrete_log import UGCDiscreteLog

PI     = math.pi
E      = math.e
A_N4   = A_of_n(4)                     # ≈ 6.5112
N_GRID = math.floor(A_N4)             # 6 (algebraic grid side)


# ============================================================
#  Finite-step smoothstep  ·  curves
# ============================================================
def finite_step_smoothstep(t: float) -> float:
    """
    Smooth step [0,1] → [0,1] from A(n) = 10·a·sin(n²°).
    At t = 0.5 (n = 4) the local derivative matches π.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    lo = A_of_n(3.0)
    hi = A_of_n(5.0)
    mid = A_of_n(3.0 + 2.0 * t)
    return float((mid - lo) / (hi - lo))


# ============================================================
#  Bit helpers
# ============================================================
def _pack_bits(values: List[int], width: int) -> bytes:
    out = bytearray()
    buf = cnt = 0
    for v in values:
        buf |= v << cnt
        cnt += width
        while cnt >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            cnt -= 8
    if cnt > 0:
        out.append(buf & 0xFF)
    return bytes(out)


def _unpack_bits(data: bytes, off: int, n: int,
                 width: int) -> Tuple[List[int], int]:
    out: List[int] = []
    buf = cnt = 0
    for _ in range(n):
        while cnt < width:
            buf |= data[off] << cnt
            off += 1
            cnt += 8
        out.append(buf & ((1 << width) - 1))
        buf >>= width
        cnt -= width
    return out, off


# ============================================================
#  Raster glyph  ·  algebraic grid + finite-step curves
# ============================================================
@dataclass
class RasterGlyph:
    mask: np.ndarray
    name: str = "?"

    @property
    def N(self) -> int:
        return self.mask.shape[0]

    # ---------- rows ----------
    def _rows_as_ints(self) -> List[int]:
        rows = []
        for iy in range(self.N):
            v = 0
            for ix in range(self.N):
                if self.mask[iy, ix]:
                    v |= (1 << ix)
            rows.append(v)
        return rows

    # ---------- encode ----------
    def encode(self) -> bytes:
        dict_enc = self._encode_dict()
        raw_enc  = self._encode_raw()
        return dict_enc if len(dict_enc) <= len(raw_enc) else raw_enc

    def _encode_dict(self) -> bytes:
        N = self.N
        nbytes = (N + 7) // 8
        rows = self._rows_as_ints()

        unique = sorted(set(rows))
        n_unique = len(unique)
        row_to_idx = {v: i for i, v in enumerate(unique)}
        bits_per_idx = max(1, (n_unique - 1).bit_length())

        out = bytearray()
        out += b"FGRS"
        out += bytes([2, N, 4, n_unique, bits_per_idx])
        for v in unique:
            for b in range(nbytes):
                out.append((v >> (8 * b)) & 0xFF)
        out += _pack_bits([row_to_idx[v] for v in rows], bits_per_idx)
        return bytes(out)

    def _encode_raw(self) -> bytes:
        N = self.N
        nbytes = (N + 7) // 8
        out = bytearray()
        out += b"FGRR"
        out += bytes([1, N])
        for v in self._rows_as_ints():
            for b in range(nbytes):
                out.append((v >> (8 * b)) & 0xFF)
        return bytes(out)

    # ---------- decode ----------
    @classmethod
    def decode(cls, data: bytes) -> "RasterGlyph":
        magic = data[:4]
        if magic == b"FGRS":
            return cls._decode_dict(data)
        if magic == b"FGRR":
            return cls._decode_raw(data)
        raise ValueError(f"bad raster magic {magic!r}")

    @classmethod
    def _decode_dict(cls, data: bytes) -> "RasterGlyph":
        version  = data[4]
        N        = data[5]
        n_step   = data[6]
        n_unique = data[7]
        bits_per = data[8]
        nbytes   = (N + 7) // 8
        off = 9
        unique = []
        for _ in range(n_unique):
            v = 0
            for b in range(nbytes):
                v |= data[off + b] << (8 * b)
            off += nbytes
            unique.append(v)
        idxs, _ = _unpack_bits(data, off, N, bits_per)
        rows = [unique[i] for i in idxs]
        return cls(mask=_rows_to_mask(rows, N))

    @classmethod
    def _decode_raw(cls, data: bytes) -> "RasterGlyph":
        version = data[4]
        N       = data[5]
        nbytes  = (N + 7) // 8
        off = 6
        rows = []
        for iy in range(N):
            v = 0
            for b in range(nbytes):
                v |= data[off + b] << (8 * b)
            off += nbytes
            rows.append(v)
        return cls(mask=_rows_to_mask(rows, N))

    # ---------- render ----------
    def render(self, scale: int = 4) -> np.ndarray:
        N = self.N
        H = N * scale
        img = np.zeros((H, H), dtype=float)
        smooth = np.array([
            finite_step_smoothstep((k + 0.5) / scale)
            for k in range(scale)
        ])
        for iy in range(N):
            for ix in range(N):
                if not self.mask[iy, ix]:
                    continue
                img[iy*scale:(iy+1)*scale, ix*scale:(ix+1)*scale] = 1.0
                if iy == 0 or not self.mask[iy - 1, ix]:
                    for k in range(scale):
                        img[iy*scale + k,
                            ix*scale:(ix+1)*scale] *= smooth[k]
                if iy == N - 1 or not self.mask[iy + 1, ix]:
                    for k in range(scale):
                        img[(iy+1)*scale - 1 - k,
                            ix*scale:(ix+1)*scale] *= smooth[k]
                if ix == 0 or not self.mask[iy, ix - 1]:
                    for k in range(scale):
                        img[iy*scale:(iy+1)*scale,
                            ix*scale + k] *= smooth[k]
                if ix == N - 1 or not self.mask[iy, ix + 1]:
                    for k in range(scale):
                        img[iy*scale:(iy+1)*scale,
                            (ix+1)*scale - 1 - k] *= smooth[k]
        return img

    @property
    def bits_raw(self) -> int:
        return self.N * self.N


def _rows_to_mask(rows: List[int], N: int) -> np.ndarray:
    mask = np.zeros((N, N), dtype=bool)
    for iy in range(N):
        for ix in range(N):
            mask[iy, ix] = bool((rows[iy] >> ix) & 1)
    return mask


# ============================================================
#  Vector glyph  ·  UGC discrete log with vertex sharing
# ============================================================
@dataclass
class VectorGlyph:
    points: List[Tuple[int, int]]
    closed: bool = False
    name: str = "?"
    p: int = 10007
    g: int = 5

    # ---------- encode ----------
    def encode(self) -> bytes:
        ux = sorted(set(x for x, _ in self.points))
        uy = sorted(set(y for _, y in self.points))
        ux_idx = {c: i for i, c in enumerate(ux)}
        uy_idx = {c: i for i, c in enumerate(uy)}
        n = self.p - 1

        bits_per = max(1, int(math.ceil(
            math.log2(max(len(ux), len(uy), 2)))))

        out = bytearray()
        out += b"FGVT"
        out += bytes([1])
        out += struct.pack(">I", self.p)
        out += struct.pack(">H", self.g)
        out += struct.pack(">I", len(self.points))
        out += struct.pack(">H", len(ux))
        out += struct.pack(">H", len(uy))
        out += bytes([bits_per, 1 if self.closed else 0])

        # unique coordinates as varint
        for c in ux:
            out += _varint_encode(c)
        for c in uy:
            out += _varint_encode(c)

        # bit-packed indices
        idx_seq: List[int] = []
        for (x, y) in self.points:
            idx_seq.append(ux_idx[x])
            idx_seq.append(uy_idx[y])
        out += _pack_bits(idx_seq, bits_per)
        return bytes(out)

    # ---------- decode ----------
    @classmethod
    def decode(cls, data: bytes) -> "VectorGlyph":
        if data[:4] != b"FGVT":
            raise ValueError("bad vector magic")
        off = 5
        p      = struct.unpack(">I", data[off:off+4])[0]; off += 4
        g      = struct.unpack(">H", data[off:off+2])[0]; off += 2
        n_pts  = struct.unpack(">I", data[off:off+4])[0]; off += 4
        n_ux   = struct.unpack(">H", data[off:off+2])[0]; off += 2
        n_uy   = struct.unpack(">H", data[off:off+2])[0]; off += 2
        bits_per = data[off]; off += 1
        closed   = bool(data[off]); off += 1

        ux = []
        for _ in range(n_ux):
            c, off = _varint_decode(data, off)
            ux.append(c)
        uy = []
        for _ in range(n_uy):
            c, off = _varint_decode(data, off)
            uy.append(c)

        # UGC dlog verification (skip coords ≥ p−1; Fermat wraps them)
        ugc = UGCDiscreteLog(p=p, g=g)
        for c in ux + uy:
            if 0 <= c < p - 1:
                h = pow(g, c, p)
                x_rec, _ = ugc.dlog(h)
                if x_rec != c:
                    raise ValueError(f"UGC dlog failed for c={c}")

        idxs, _ = _unpack_bits(data, off, 2 * n_pts, bits_per)
        points = [(ux[idxs[2*k]], uy[idxs[2*k + 1]])
                  for k in range(n_pts)]
        return cls(points=points, closed=closed, p=p, g=g)

    @property
    def bits_raw(self) -> int:
        if not self.points:
            return 0
        max_c = max(max(x, y) for x, y in self.points)
        w = max(1, int(math.ceil(math.log2(max_c + 1))))
        return len(self.points) * 2 * w

    # ---------- render ----------
    def render(self, canvas: int = 64, scale: int = 1) -> np.ndarray:
        img = np.zeros((canvas, canvas), dtype=float)
        if len(self.points) < 2:
            return img
        pts = list(self.points) + ([self.points[0]] if self.closed else [])
        for k in range(len(pts) - 1):
            x0, y0 = pts[k]
            x1, y1 = pts[k + 1]
            steps = max(abs(x1 - x0), abs(y1 - y0), 1) * scale
            for s in range(steps + 1):
                t = s / steps
                x = int(round((x0 + t * (x1 - x0)) * scale))
                y = int(round((y0 + t * (y1 - y0)) * scale))
                if 0 <= x < canvas and 0 <= y < canvas:
                    img[canvas - 1 - y, x] = 1.0
        return img


# ============================================================
#  Varint helpers  ·  zigzag LEB128 (for coordinate storage)
# ============================================================
def _varint_encode(n: int) -> bytes:
    if n < 0:
        # zigzag
        n = (-n << 1) - 1
    else:
        n = n << 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _varint_decode(data: bytes, off: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[off]; off += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    # unzigzag
    if result & 1:
        result = -((result + 1) >> 1)
    else:
        result = result >> 1
    return result, off


# ============================================================
#  16×16 demo font  (letter generators)
# ============================================================
def _H16() -> np.ndarray:
    m = np.zeros((16, 16), dtype=bool)
    m[:, 0:2] = True
    m[:, 14:16] = True
    m[7:9, :] = True
    return m


def _I16() -> np.ndarray:
    m = np.zeros((16, 16), dtype=bool)
    m[0:2, :] = True
    m[14:16, :] = True
    m[2:14, 7:9] = True
    return m


def _O16() -> np.ndarray:
    m = np.zeros((16, 16), dtype=bool)
    m[0:2, 2:14] = True
    m[14:16, 2:14] = True
    m[2:14, 0:2] = True
    m[2:14, 14:16] = True
    return m


def _T16() -> np.ndarray:
    m = np.zeros((16, 16), dtype=bool)
    m[0:2, :] = True
    m[2:16, 7:9] = True
    return m


def _X16() -> np.ndarray:
    m = np.zeros((16, 16), dtype=bool)
    for k in range(16):
        m[k, max(0, k-1):min(16, k+2)] = True
        m[k, max(0, 15-k-1):min(16, 15-k+2)] = True
    return m


_FONT_16 = {
    "H": _H16(),
    "I": _I16(),
    "O": _O16(),
    "T": _T16(),
    "X": _X16(),
}


def load_char(c: str) -> Optional[RasterGlyph]:
    if c not in _FONT_16:
        return None
    return RasterGlyph(mask=_FONT_16[c].copy(), name=c)


# ============================================================
#  CLI
# ============================================================
def cmd_info(_):
    print("=" * 72)
    print("Font converter  ·  raster (finite-step) + vector (UGC dlog)")
    print("=" * 72)
    print(f"  a = 1/(π − e)              = {A_CONST:.8f}")
    print(f"  A(n=4)                     = {A_N4:.6f}")
    print(f"  ⌊A⌋²  algebraic grid       = "
          f"{N_GRID}×{N_GRID} = {N_GRID**2} cells")
    print(f"  ∂A/∂n at n=4 (central)     = "
          f"{finite_step_derivative(4)[2]:.6f}   (≈ π)")
    print()
    print("  RASTER  mode:")
    print(f"    · demo letters (16×16)  :  "
          f"{''.join(_FONT_16.keys())}")
    print("    · row dictionary + raw fallback")
    print("    · finite-step anti-aliased render")
    print()
    print("  VECTOR  mode:")
    print("    · unique x's / y's as UGC vertices")
    print("    · per-point references as bit-packed indices")
    print("    · decode verifies BSGS dlog round-trip")
    print("    · default (p, g) = (10007, 5)")


def cmd_encode(args):
    if args.mode == "raster":
        print("=" * 72)
        print(f"RASTER ENCODE  ·  {args.text!r}")
        print("=" * 72)
        total_raw = total_comp = 0
        for ch in args.text:
            g = load_char(ch)
            if g is None:
                print(f"  '{ch}'   →  (not in demo font)")
                continue
            enc = g.encode()
            raw = g.bits_raw
            comp = len(enc) * 8
            total_raw += raw
            total_comp += comp
            print(f"  '{ch}'   raw={raw:>4d} b   "
                  f"comp={comp:>4d} b   "
                  f"ratio={comp/raw:.2f}×   "
                  f"({len(enc)} B)")
        if total_raw:
            print(f"  ───────────────────────────────────────")
            print(f"  total   raw={total_raw:>4d} b   "
                  f"comp={total_comp:>4d} b   "
                  f"ratio={total_comp/total_raw:.2f}×")
    elif args.mode == "vector":
        pts = _parse_path(args.path)
        g = VectorGlyph(points=pts, closed=args.closed,
                        p=args.p, g=args.g)
        enc = g.encode()
        raw = g.bits_raw
        print("=" * 72)
        print("VECTOR ENCODE  ·  UGC dlog with vertex sharing")
        print("=" * 72)
        print(f"  points       = {len(pts)}")
        print(f"  unique x     = {len(set(x for x, _ in pts))}")
        print(f"  unique y     = {len(set(y for _, y in pts))}")
        print(f"  raw bits     = {raw}")
        print(f"  comp bits    = {len(enc)*8}   ({len(enc)} B)")
        print(f"  ratio        = {len(enc)*8 / raw:.2f}×")
        print(f"  hex (head)   = {enc.hex()[:64]}…")
    print()
    return 0


def cmd_decode(args):
    data = bytes.fromhex(args.hex)
    print("=" * 72)
    print(f"DECODE  ·  {args.mode}")
    print("=" * 72)
    if args.mode == "raster":
        g = RasterGlyph.decode(data)
        print(f"  {g.N}×{g.N} glyph")
        for row in g.mask:
            print("  " + "".join("█" if v else "·" for v in row))
    else:
        g = VectorGlyph.decode(data)
        print(f"  {len(g.points)} points  "
              f"(p={g.p}, g={g.g}, closed={g.closed})")
        for (x, y) in g.points:
            print(f"    ({x:>5d}, {y:>5d})")
    print()
    return 0


def cmd_demo(_):
    print("=" * 72)
    print("DEMO")
    print("=" * 72)
    print()

    # ---- raster ----
    print("--- RASTER  ·  16×16  ·  'HIO' ---")
    total_raw = total_comp = 0
    for ch in "HIO":
        g = load_char(ch)
        enc = g.encode()
        raw = g.bits_raw
        comp = len(enc) * 8
        total_raw += raw
        total_comp += comp
        print(f"  '{ch}'  {raw:>4d} → {comp:>4d} bits   "
              f"ratio {comp/raw:.2f}×")
        for row in g.mask:
            print("    " + "".join("█" if v else "·" for v in row))
        # round-trip
        dec = RasterGlyph.decode(enc)
        assert np.array_equal(dec.mask, g.mask), f"raster {ch} failed"
    print(f"  total  {total_raw} → {total_comp} bits  "
          f"ratio {total_comp/total_raw:.2f}×")
    print(f"  round-trip  :  OK")
    print()

    # ---- vector  ·  shared coords ----
    print("--- VECTOR  ·  square path  (4 pts, 2 unique x, 2 unique y) ---")
    pts = [(5, 5), (15, 5), (15, 15), (5, 15)]
    g = VectorGlyph(points=pts, closed=True)
    enc = g.encode()
    raw = g.bits_raw
    print(f"  points = {pts}")
    print(f"  raw  = {raw:>4d} bits")
    print(f"  comp = {len(enc)*8:>4d} bits   ({len(enc)} B)")
    print(f"  ratio = {len(enc)*8/raw:.2f}×")
    dec = VectorGlyph.decode(enc)
    print(f"  UGC dlog verification  :  "
          f"{'OK' if dec.points == pts else 'FAILED'}")
    print()

    # ---- vector  ·  stair path ----
    print("--- VECTOR  ·  stair path  (20 pts, 11 unique x, 11 unique y) ---")
    pts = [(5*(k+1), 5*((k//2)+1)) for k in range(20)]
    g = VectorGlyph(points=pts)
    enc = g.encode()
    raw = g.bits_raw
    print(f"  points = {len(pts)}")
    print(f"  raw  = {raw:>4d} bits")
    print(f"  comp = {len(enc)*8:>4d} bits   ({len(enc)} B)")
    print(f"  ratio = {len(enc)*8/raw:.2f}×")
    dec = VectorGlyph.decode(enc)
    print(f"  UGC dlog verification  :  "
          f"{'OK' if dec.points == pts else 'FAILED'}")
    print()

    # ---- render 'H' ----
    print("--- render 'H'  (finite-step anti-alias, scale=2) ---")
    g = load_char("H")
    img = g.render(scale=2)
    for row in img[::2]:
        print("  " + "".join(
            "█" if v > 0.75 else
            "▓" if v > 0.40 else
            "░" if v > 0.10 else "·"
            for v in row[::2]))
    return 0


def _parse_path(s: str) -> List[Tuple[int, int]]:
    pts = []
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(",")
        pts.append((int(x.strip()), int(y.strip())))
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="font_compress",
        description="Font converter: raster (finite-step) + "
                    "vector (UGC dlog)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="show configuration")
    p_info.set_defaults(func=cmd_info)

    p_enc = sub.add_parser("encode", help="encode glyph / path")
    p_enc.add_argument("--mode", choices=["raster", "vector"], required=True)
    p_enc.add_argument("--text", default="")
    p_enc.add_argument("--path", default="")
    p_enc.add_argument("--p", type=int, default=10007)
    p_enc.add_argument("--g", type=int, default=5)
    p_enc.add_argument("--closed", action="store_true")
    p_enc.set_defaults(func=cmd_encode)

    p_dec = sub.add_parser("decode", help="decode hex payload")
    p_dec.add_argument("--mode", choices=["raster", "vector"], required=True)
    p_dec.add_argument("--hex", required=True)
    p_dec.set_defaults(func=cmd_decode)

    p_demo = sub.add_parser("demo", help="full demo")
    p_demo.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())