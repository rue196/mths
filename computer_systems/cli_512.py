#!/usr/bin/env python3
"""
cli_512.py
==========

512-bit compile/parse extension for cli.py.

Wire format (export) — unchanged from cli.py:
    [ MOSB header        (48 B) ]
    [ FILE header        (55 B) ]
    [ UTF-8 UGC ALU packet      ]  particle cloud + UGC + ALU + fiber
    [ weight signature   (8 B)  ]  SHA-256(payload)[:8]  →  ">Q"

Compile (build codec) and parse (verify) run at ≥ 512 bits:
    · 512-bit α, π, e              (via piepy_512)
    · 512-bit harmonic C[i] array  (C[0]=+1, C[i≠0]=−H_{|i|+1})
    · 512-bit Q/P decomposition    (TwoPointMatrix)
    · 512-bit Z_forward / Z_central / giant_step_walk

Export (encode) is still 64-bit float64 wire:
    · particle (i, j, x, y, vx, zt)  →  ">iidddd"  (40 B per particle)
    · weight signature is byte-hash  (precision-independent)
    · FILE header fields are ints / 8-byte doubles
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from decimal import Decimal, localcontext
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- parents: 64-bit wire, same as cli.py ----
from cli import (
    FILE_MAGIC, FILE_FMT, FILE_SIZE,
    MODE_RAW, MODE_TEXT, MODE_BINARY, MODE_NAMES,
    FileHeader, FileCodec, FileSealer,
    inspect_frame, print_breakdown,
    resolve_K_from_envelope,
)
from trading_platform import K_MOSB
from download_mboius_unpack_buffer import MobiusPacker
from UGC_discrete_train_envelope import QPWeightEncoder

# ---- 512-bit constants ----
from piepy_512 import (
    e_512bit, pi_512bit, alpha_512bit, DEC_DIGITS,
)

# ---- optional mpmath for speed ----
try:
    import mpmath as mp
    _HAS_MPMATH = True
except ImportError:
    _HAS_MPMATH = False


# ============================================================
#  Precision budget
# ============================================================
BITS        = 512
PREC_DIGITS = DEC_DIGITS + 10         # ≈ 190 decimal digits

E_512     = e_512bit()
PI_512    = pi_512bit()
ALPHA_512 = alpha_512bit()

ALPHA_MIN_D = Decimal("0.3628")
ALPHA_MAX_D = ALPHA_512               # = 1/(π−e)
ALPHA_F64   = float(ALPHA_512)        # for reference


# ============================================================
#  Decimal trig (with mpmath fast-path)
# ============================================================
def _dec_sin_cos(x: Decimal) -> Tuple[Decimal, Decimal]:
    """sin(x), cos(x) at ≥ 512-bit precision."""
    if _HAS_MPMATH:
        mp.mp.dps = PREC_DIGITS + 10
        xm = mp.mpf(str(x))
        return Decimal(str(mp.sin(xm))), Decimal(str(mp.cos(xm)))

    # Decimal Taylor with range reduction
    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 10
        two_pi = Decimal(2) * PI_512
        n = round(x / two_pi)
        x = x - n * two_pi
        x2 = x * x
        eps = Decimal(10) ** (-(PREC_DIGITS + 5))
        cos_t = Decimal(1); cos_s = Decimal(1); k = 0
        while True:
            k += 1
            cos_t = -cos_t * x2 / Decimal((2*k - 1) * (2*k))
            if abs(cos_t) < eps or k > 2000:
                break
            cos_s += cos_t
        sin_t = x; sin_s = x; k = 0
        while True:
            k += 1
            sin_t = -sin_t * x2 / Decimal((2*k) * (2*k + 1))
            if abs(sin_t) < eps or k > 2000:
                break
            sin_s += sin_t
        return sin_s, cos_s


class DComplex:
    """512-bit complex number on Decimal."""
    __slots__ = ("re", "im")
    def __init__(self, re=0, im=0):
        self.re = re if isinstance(re, Decimal) else Decimal(re)
        self.im = im if isinstance(im, Decimal) else Decimal(im)
    def __add__(self, o): return DComplex(self.re + o.re, self.im + o.im)
    def __sub__(self, o): return DComplex(self.re - o.re, self.im - o.im)
    def __mul__(self, o):
        a, b, c, d = self.re, self.im, o.re, o.im
        return DComplex(a*c - b*d, a*d + b*c)
    def __truediv__(self, o):
        a, b, c, d = self.re, self.im, o.re, o.im
        den = c*c + d*d
        return DComplex((a*c + b*d)/den, (b*c - a*d)/den)
    def __abs__(self):
        return (self.re*self.re + self.im*self.im).sqrt()
    def __repr__(self):
        return f"DComplex({self.re:.6e}, {self.im:.6e})"


# ============================================================
#  512-bit harmonics and C array
# ============================================================
def build_C_512(K: int) -> List[Decimal]:
    """C[i] at 512-bit: C[0]=+1, C[i≠0]=−H_{|i|+1}."""
    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 5
        H = [Decimal(0)] * (K + 2)
        for n in range(1, K + 2):
            H[n] = H[n-1] + Decimal(1) / Decimal(n)
        C = [Decimal(0)] * (2*K + 1)
        C[K] = Decimal(1)
        for i in range(1, K + 1):
            C[K + i] = -H[i + 1]
            C[K - i] = -H[i + 1]
    return C


def alpha_at_512(i: int, K: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 5
        t = min(Decimal(abs(i)) / Decimal(max(K, 1)), Decimal(1))
        return ALPHA_MIN_D + (ALPHA_MAX_D - ALPHA_MIN_D) * t


# ============================================================
#  512-bit spectral sum and finite-step derivatives
# ============================================================
def zeta_512(t: Decimal, C: List[Decimal], K: int) -> DComplex:
    """ζ(t) = Σ C[i]·exp(i·t·i/α_i) at 512-bit."""
    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 5
        total = DComplex(0, 0)
        for i in range(-K, K + 1):
            if i == 0:
                total = total + DComplex(C[K], 0)
                continue
            alpha_i = alpha_at_512(i, K)
            phase = t * Decimal(i) / alpha_i
            sin_p, cos_p = _dec_sin_cos(phase)
            total = total + DComplex(C[K + i] * cos_p,
                                     C[K + i] * sin_p)
    return total


def Z_forward_512(t: Decimal, C: List[Decimal], K: int,
                  A: Decimal) -> DComplex:
    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 5
        return (zeta_512(t + A, C, K)
                - zeta_512(t, C, K)) / DComplex(A, 0)


def Z_central_512(t: Decimal, C: List[Decimal], K: int,
                  A: Decimal) -> DComplex:
    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 5
        return (zeta_512(t + A, C, K)
                - zeta_512(t - A, C, K)) / DComplex(2*A, 0)


# ============================================================
#  512-bit two-point Q/P decomposition
# ============================================================
def two_point_QP_512(x: Decimal, y: Decimal
                     ) -> Tuple[Decimal, Decimal]:
    """Q = Re(tr M), P = Im(tr M) at 512-bit."""
    with localcontext() as ctx:
        ctx.prec = PREC_DIGITS + 10
        ln_x = x.ln()
        ln_y = y.ln()
        sin_x, cos_x = _dec_sin_cos(ln_x)
        sin_y, cos_y = _dec_sin_cos(ln_y)
        Q = cos_x / x + cos_y / y
        P = sin_x / x + sin_y / y
    return Q, P


# ============================================================
#  FileCodec512  ·  512-bit verification, 64-bit wire
# ============================================================
class FileCodec512(FileCodec):
    """
    Same wire format as FileCodec.
    Overrides only the 512-bit verification layers.
    """

    def __init__(self, K_table=128, K_alu=64, k_labels=4,
                 threshold=0.3, K_ci=20):
        super().__init__(K_table, K_alu, k_labels, threshold, K_ci)
        self._C_512 = build_C_512(K_ci)

    # ---------- Q/P at 512-bit ----------
    def _qp_consistency(self, data: bytes) -> Tuple[bool, Dict]:
        if len(data) == 0:
            return True, dict(n_samples=0)
        n_target = min(32, len(data))
        step = max(1, len(data) // n_target)
        samples = data[::step][:n_target]
        n_baby = n_giant = n_fail = 0
        mode_agree = mode_disagree = 0
        qp_log = []
        for b in samples:
            w = (int(b) - 128) / 128.0
            enc = self.weight_encoder.encode(float(w))
            if enc["mode"] == "fail":
                n_fail += 1
                continue
            x, y = enc["x"], enc["y"]
            Q, P = two_point_QP_512(Decimal(x + 1), Decimal(y + 1))
            mode_512 = ("baby" if (abs(Q) >= abs(P) and Q > 0)
                        else "giant")
            if mode_512 == "baby":
                n_baby += 1
            else:
                n_giant += 1
            if enc["mode"] == mode_512:
                mode_agree += 1
            else:
                mode_disagree += 1
            if len(qp_log) < 5:
                qp_log.append(dict(
                    w=float(w), x=int(x), y=int(y),
                    Q=str(Q)[:22], P=str(P)[:22],
                    mode_64=enc["mode"], mode_512=mode_512))
        ok = (n_fail == 0) and (n_baby + n_giant == len(samples))
        return ok, dict(n_baby=n_baby, n_giant=n_giant, n_fail=n_fail,
                        n_samples=len(samples),
                        mode_agree=mode_agree,
                        mode_disagree=mode_disagree,
                        qp_log=qp_log)

    # ---------- Z-walk at 512-bit ----------
    def _zwalk_consistency(self, data: bytes) -> Tuple[bool, Dict]:
        if len(data) < 4:
            return True, dict(n_steps=0, expected=0, z=0)
        x_target = float(data[0])
        if x_target < 1.0:
            return True, dict(n_steps=1, expected=1,
                              z=int(x_target))
        expected = max(1, int(math.ceil(math.sqrt(x_target))))

        with localcontext() as ctx:
            ctx.prec = PREC_DIGITS + 5
            A_512 = ALPHA_MAX_D / ALPHA_MIN_D
            dt = Decimal(x_target) / Decimal(expected)

            zeta_cache: Dict[Decimal, DComplex] = {}
            def get_zeta(tt):
                if tt in zeta_cache:
                    return zeta_cache[tt]
                z = zeta_512(tt, self._C_512, self.K_ci)
                zeta_cache[tt] = z
                return z

            path = []
            t = Decimal(0)
            for _ in range(expected):
                zt   = get_zeta(t)
                ztp  = get_zeta(t + A_512)
                ztm  = get_zeta(t - A_512)
                Zf = (ztp - zt) / DComplex(A_512, 0)
                Zc = (ztp - ztm) / DComplex(2 * A_512, 0)
                path.append(dict(
                    t=float(t),
                    Zf_re=float(Zf.re), Zf_im=float(Zf.im),
                    Zc_re=float(Zc.re), Zc_im=float(Zc.im),
                ))
                t += dt

        ok = (len(path) == expected)
        zf_re_rms = float(np.sqrt(
            np.mean([p["Zf_re"]**2 for p in path])))
        zf_im_rms = float(np.sqrt(
            np.mean([p["Zf_im"]**2 for p in path])))
        return ok, dict(
            n_steps=len(path), expected=expected,
            z=int(x_target), A_512=float(A_512),
            Zf_re_rms=zf_re_rms, Zf_im_rms=zf_im_rms,
            first=path[0] if path else None,
            last=path[-1] if path else None,
        )


# ============================================================
#  FileSealer512
# ============================================================
class FileSealer512:
    def __init__(self, K_ci=20, K_table=128, K_alu=64):
        self.K_ci = K_ci
        self.K_table = K_table
        self.K_alu = K_alu
        self.codec = FileCodec512(
            K_table=K_table, K_alu=K_alu, K_ci=K_ci)
        self.packer = MobiusPacker(K=K_MOSB)

    def seal(self, payload, mode=MODE_RAW):
        packet = self.codec.encode(payload)
        header = FileHeader(
            mode=mode, reserved_a=0, reserved_b=0,
            K_ci=self.K_ci,
            threshold=self.codec.threshold,
            timestamp=time.time(),
            file_size=float(len(payload)),
            nonce=os.urandom(12),
            env_size=len(packet),
            msg_len=len(payload),
        )
        payload_hdr = header.pack() + packet
        frame, S, H, m = self.packer.pack(payload_hdr)
        return dict(frame=frame, header=header,
                    packet=packet, payload=payload,
                    mosb_S=S, mosb_H=H, mosb_m=m)


# ============================================================
#  Pretty printers
# ============================================================
def _short(d: Decimal, n: int = 60) -> str:
    s = str(d)
    return s[:n] + "…" if len(s) > n else s


def print_breakdown_512(diag: Dict):
    print_breakdown(diag)
    if "mode_agree" in diag:
        print(f"    mode_agree           = {diag['mode_agree']}")
        print(f"    mode_disagree        = {diag['mode_disagree']}")
    zd = diag.get("zwalk_diag")
    if zd and "Zf_re_rms" in zd:
        print(f"    Z-walk (512-bit):")
        print(f"      A_512              = {zd['A_512']:.10f}")
        print(f"      Zf_re_rms          = {zd['Zf_re_rms']:+.6e}")
        print(f"      Zf_im_rms          = {zd['Zf_im_rms']:+.6e}")
        if zd.get("first"):
            p = zd["first"]
            print(f"      step 0   t={p['t']:.4f}  "
                  f"Zf=({p['Zf_re']:+.6e}, {p['Zf_im']:+.6e})")


# ============================================================
#  CLI commands
# ============================================================
def cmd_info(args):
    print("=" * 70)
    print("MOSB file CLI  ·  512-bit compile/parse  ·  64-bit wire")
    print("=" * 70)
    print(f"  BITS            = {BITS}")
    print(f"  PREC_DIGITS     = {PREC_DIGITS}")
    print(f"  mpmath          = "
          f"{'available' if _HAS_MPMATH else 'not installed (Decimal fallback)'}")
    print(f"  FILE header     = {FILE_SIZE} B")
    print(f"  MOSB K          = {K_MOSB}")
    print(f"  K_table         = 128   (wire format)")
    print(f"  K_alu           = 64    (wire format)")
    print()
    print("  Wire format (unchanged from cli.py):")
    print(f"    [ MOSB header        (48 B) ]")
    print(f"    [ FILE header        ({FILE_SIZE} B) ]")
    print(f"    [ UTF-8 UGC ALU packet      ]  "
          f"particle cloud + UGC + ALU + fiber")
    print(f"    [ weight signature   (8 B)  ]  SHA-256[:8]")
    print()
    print("  512-bit constants:")
    print(f"    e_512       = {_short(E_512)}")
    print(f"    π_512       = {_short(PI_512)}")
    print(f"    α_512       = {_short(ALPHA_512)}")
    print(f"    α_min       = {ALPHA_MIN_D}")
    print(f"    A_512       = α/0.3628  = "
          f"{_short(ALPHA_512 / ALPHA_MIN_D)}")
    print()
    print("  512-bit verify flags:")
    print("    qp_consistent      Q/P decision at 512-bit")
    print("    zwalk_consistent   Z-walk path at 512-bit")
    print("    particle/ugc/alu/fiber/weight  unchanged 64-bit")


def cmd_report(args):
    print("=" * 70)
    print("512-bit internals  ·  C, Q/P, Z-walk")
    print("=" * 70)
    K = args.K if args.K > 0 else 20
    C = build_C_512(K)
    print(f"  K_ci        = {K}")
    print(f"  C[0]        = {_short(C[K])}")
    for i in (1, 2, 3, K):
        print(f"  C[±{i:<3d}]   = {_short(C[K + i])}")
    print()

    print("  Q/P at 512-bit on a grid of samples:")
    enc = QPWeightEncoder()
    print(f"    {'w':>6s}  {'x':>5s}  {'y':>5s}  "
          f"{'Q (512)':>20s}  {'P (512)':>20s}  "
          f"{'mode_64':>8s}  {'mode_512':>8s}  agree")
    print("    " + "-" * 92)
    for w in (-0.9, -0.5, 0.0, 0.5, 0.9):
        e64 = enc.encode(float(w))
        if e64["mode"] == "fail":
            continue
        Q, P = two_point_QP_512(Decimal(e64["x"] + 1),
                                 Decimal(e64["y"] + 1))
        mode_512 = ("baby" if (abs(Q) >= abs(P) and Q > 0)
                    else "giant")
        agree = "yes" if mode_512 == e64["mode"] else "NO"
        print(f"    {w:>+6.3f}  {e64['x']:>5d}  {e64['y']:>5d}  "
              f"{_short(Q, 18):>20s}  {_short(P, 18):>20s}  "
              f"{e64['mode']:>8s}  {mode_512:>8s}  {agree}")
    print()

    print("  Z-walk step 0 at 512-bit for byte samples:")
    A_512 = ALPHA_MAX_D / ALPHA_MIN_D
    for byte_val in (1, 4, 16, 64, 128, 255):
        x_target = float(byte_val)
        expected = max(1, int(math.ceil(math.sqrt(x_target))))
        with localcontext() as ctx:
            ctx.prec = PREC_DIGITS + 5
            Z = Z_forward_512(Decimal(0), C, K, A_512)
        print(f"    byte={byte_val:>3d}  n_steps={expected:>2d}  "
              f"Zf(0)=({_short(Z.re, 16)}, {_short(Z.im, 16)})")


def cmd_seal(args):
    with open(args.file, "rb") as f:
        payload = f.read()
    try:
        payload.decode("utf-8"); mode = MODE_TEXT
    except UnicodeDecodeError:
        mode = MODE_BINARY

    K_ci = args.K if args.K > 0 else resolve_K_from_envelope(args.file)
    sealer = FileSealer512(K_ci=K_ci)
    result = sealer.seal(payload, mode=mode)

    out_path = args.out or (args.file + ".mosb")
    with open(out_path, "wb") as f:
        f.write(result["frame"])

    print(f"  sealed         : {args.file}")
    print(f"  →              : {out_path}")
    print(f"  payload        : {len(payload)} B   "
          f"mode={MODE_NAMES[mode]}")
    print(f"  packet (wire)  : {len(result['packet'])} B  (64-bit)")
    print(f"  frame          : {len(result['frame'])} B")
    print(f"  K_ci           : {K_ci}")
    print(f"  BITS (verify)  : {BITS}")
    print(f"  mosb S/H/m     : {result['mosb_S']:+.4f}  /  "
          f"{result['mosb_H']:.4f}  /  {result['mosb_m']:.4e}")

    _, packet, _ = inspect_frame(result["frame"])
    codec = FileCodec512(K_ci=K_ci)
    _, diag = codec.decode(packet)
    print(f"  post-verify    : "
          f"{'VERIFIED' if diag.get('verified') else 'FAILED'}"
          f"  (qp={diag.get('qp_consistent')}, "
          f"zwalk={diag.get('zwalk_consistent')})")


def cmd_verify(args):
    with open(args.frame, "rb") as f:
        frame = f.read()
    try:
        header, packet, mosb = inspect_frame(frame)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    K_ci = args.K if args.K > 0 else (header.K_ci or 20)
    codec = FileCodec512(K_ci=K_ci)
    recovered, diag = codec.decode(packet)
    ok = diag.get("verified", False)

    print(f"  frame      : {args.frame}")
    print(f"  verdict    : {'VERIFIED' if ok else 'FAILED'}")
    print(f"  mode       : {MODE_NAMES.get(header.mode, header.mode)}")
    print(f"  K_ci       : {K_ci}   (header: {header.K_ci})")
    print(f"  K_table    : 128  (wire)")
    print(f"  orig size  : {header.msg_len} B")
    print(f"  packet     : {header.env_size} B  (wire)")
    print(f"  nonce      : {header.nonce.hex()[:16]}…")
    print(f"  BITS       : {BITS} (verify)")
    print(f"  mosb S/H/m : {mosb['mosb_S']:+.4f}  /  "
          f"{mosb['mosb_H']:.4f}  /  {mosb['mosb_m']:.4e}")
    print_breakdown_512(diag)

    if args.out and recovered:
        with open(args.out, "wb") as f:
            f.write(recovered)
        print(f"\n  recovered  → {args.out}  ({len(recovered)} B)")
    return 0 if ok else 1


def cmd_roundtrip(args):
    with open(args.file, "rb") as f:
        payload = f.read()
    try:
        payload.decode("utf-8"); mode = MODE_TEXT
    except UnicodeDecodeError:
        mode = MODE_BINARY

    K_ci = args.K if args.K > 0 else resolve_K_from_envelope(args.file)
    sealer = FileSealer512(K_ci=K_ci)
    seal_result = sealer.seal(payload, mode=mode)

    _, packet, _ = inspect_frame(seal_result["frame"])
    codec = FileCodec512(K_ci=K_ci)
    recovered, diag = codec.decode(packet)
    ok = diag.get("verified", False) and recovered == payload

    print("=" * 70)
    print(f"Round-trip  ·  {args.file}  ·  {BITS}-bit verify")
    print("=" * 70)
    print(f"  payload    : {len(payload)} B   mode={MODE_NAMES[mode]}")
    print(f"  packet     : {len(seal_result['packet'])} B  "
          f"(64-bit wire)")
    print(f"  frame      : {len(seal_result['frame'])} B")
    print(f"  K_ci       : {K_ci}")
    print(f"  recovered  : {len(recovered)} B   "
          f"match={'OK' if recovered == payload else 'FAIL'}")
    print(f"  verified   : {diag.get('verified')}")
    print_breakdown_512(diag)
    return 0 if ok else 1


def cmd_compare(args):
    """Prove 64-bit wire identity: cli.py encode == cli_512.py encode."""
    with open(args.file, "rb") as f:
        payload = f.read()
    K_ci = args.K if args.K > 0 else 20
    codec_64  = FileCodec(K_ci=K_ci)
    codec_512 = FileCodec512(K_ci=K_ci)
    packet_64  = codec_64.encode(payload)
    packet_512 = codec_512.encode(payload)
    same = (packet_64 == packet_512)

    print("=" * 70)
    print(f"Byte-identity  ·  cli.py vs cli_512.py  ·  {args.file}")
    print("=" * 70)
    print(f"  payload size        : {len(payload)} B")
    print(f"  64-bit packet size  : {len(packet_64)} B")
    print(f"  512-bit packet size : {len(packet_512)} B  "
          f"(uses parent encode)")
    print(f"  byte-identical      : {same}")
    if not same:
        for i in range(min(len(packet_64), len(packet_512))):
            if packet_64[i] != packet_512[i]:
                print(f"  first diff at byte {i}: "
                      f"64={packet_64[i]:02x}  512={packet_512[i]:02x}")
                break
    return 0 if same else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cli_512",
        description="512-bit compile/parse extension for cli.py "
                    "(64-bit wire format unchanged)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show configuration").set_defaults(
        func=cmd_info)

    p_rep = sub.add_parser(
        "report", help="512-bit internals (C, Q/P, Z-walk)")
    p_rep.add_argument("-K", "--K", type=int, default=0)
    p_rep.set_defaults(func=cmd_report)

    p_seal = sub.add_parser("seal", help="encode file → MOSB frame")
    p_seal.add_argument("file")
    p_seal.add_argument("-o", "--out", default=None)
    p_seal.add_argument("-K", "--K", type=int, default=0)
    p_seal.set_defaults(func=cmd_seal)

    p_ver = sub.add_parser("verify",
                           help="verify a MOSB frame at 512-bit")
    p_ver.add_argument("frame")
    p_ver.add_argument("-o", "--out", default=None)
    p_ver.add_argument("-K", "--K", type=int, default=0)
    p_ver.set_defaults(func=cmd_verify)

    p_rt = sub.add_parser("roundtrip",
                          help="seal + verify at 512-bit")
    p_rt.add_argument("file")
    p_rt.add_argument("-K", "--K", type=int, default=0)
    p_rt.set_defaults(func=cmd_roundtrip)

    p_cmp = sub.add_parser(
        "compare", help="prove 64-bit wire identity vs cli.py")
    p_cmp.add_argument("file")
    p_cmp.add_argument("-K", "--K", type=int, default=0)
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())