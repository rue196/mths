#!/usr/bin/env python3
"""
mosb_file_cli.py
================

MOSB-framed UTF-8 UGC ALU file sealer / verifier.

Frame layout
------------
    [ MOSB header        (48 B) ]   MobiusPacker framing
    [ FILE header        (55 B) ]   magic "FILB", K_ci, sizes, nonce
    [ UTF-8 UGC ALU packet      ]   particle cloud + UGC + ALU + fiber
    [ weight signature   (8 B)  ]   SHA-256(payload)[:8]  →  ">Q"

Verification on decode
----------------------
    particle_match      UTF8TradeCodec particle cloud consistency
    ugc_match           UGC survivor / non-survivor index lists
    alu_match           ADD / SUB / MUL / DIV z-values
    fiber_match         fiber kernel supertrace
    weight_match        8-byte SHA-256 of the recovered payload
    qp_consistent       QPWeightEncoder over sampled weights
    zwalk_consistent    giant_step_walk completes at ⌈√z⌉

Spectral coefficients (drive the Z-walk)
----------------------------------------
    ζ(t) = Σ_{i=-K}^{K}  C[i] · exp(i · t · i / α_i)
    C[0]   = +1
    C[i≠0] = −H_{|i|+1}
    α_i   ∈ [0.3628, 1/(π−e)]   (linear interp)

K_ci is resolved from the envelope file system
(``file_reader_envelope_sat.EnvelopeFactory``) when available;
otherwise a size-based heuristic is used.  ``-K`` forces a value.

Usage
-----
    python mosb_file_cli.py info
    python mosb_file_cli.py seal   <file>   [-o out.mosb] [-K K_ci]
    python mosb_file_cli.py verify <frame>  [-o recovered] [-K K_ci]
    python mosb_file_cli.py roundtrip <file> [-K K_ci]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- parents ----
from trading_platform import UTF8TradeCodec, K_MOSB
from UGC_discrete_train_envelope import (
    QPWeightEncoder,
    giant_step_walk,
)
from complete_planes_spectral_sum import build_spectral_coefficients
from download_mboius_unpack_buffer import MobiusPacker, parse_size


# ============================================================
#  Constants
# ============================================================
FILE_MAGIC = b"FILB"
# magic(4) mode(1) ra(1) rb(1) K_ci(4) threshold(8) ts(8) filesize(8)
# nonce(12) packet(4) orig(4)   →  55 bytes (matches TRDB / OITH)
FILE_FMT   = ">4sBBBiddd12sII"
FILE_SIZE  = struct.calcsize(FILE_FMT)       # 55

MODE_RAW    = 0
MODE_TEXT   = 1
MODE_BINARY = 2

MODE_NAMES = {MODE_RAW: "raw", MODE_TEXT: "text", MODE_BINARY: "binary"}


# ============================================================
#  FILE header  (55 bytes — same physical layout as TRDB / OITH)
# ============================================================
@dataclass
class FileHeader:
    mode:       int          # 0 raw / 1 text / 2 binary
    reserved_a: int
    reserved_b: int
    K_ci:       int          # spectral K / QP encoder K / Z-walk K
    threshold:  float
    timestamp:  float
    file_size:  float        # original file size (float for uniform layout)
    nonce:      bytes
    env_size:   int          # UTF-8 UGC ALU packet length (with weight sig)
    msg_len:    int          # original file size as int

    def pack(self) -> bytes:
        return struct.pack(
            FILE_FMT, FILE_MAGIC,
            int(self.mode), int(self.reserved_a), int(self.reserved_b),
            int(self.K_ci), float(self.threshold), float(self.timestamp),
            float(self.file_size), self.nonce,
            int(self.env_size), int(self.msg_len),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "FileHeader":
        (magic, mode, ra, rb, K_ci, thr, ts, fs,
         nonce, env_sz, msg_len) = struct.unpack(
            FILE_FMT, data[:FILE_SIZE])
        if magic != FILE_MAGIC:
            raise ValueError(f"bad FILE magic: {magic!r}")
        return cls(
            mode=int(mode), reserved_a=int(ra), reserved_b=int(rb),
            K_ci=int(K_ci), threshold=float(thr), timestamp=float(ts),
            file_size=float(fs), nonce=nonce,
            env_size=int(env_sz), msg_len=int(msg_len),
        )


# ============================================================
#  FileCodec  ·  UTF8TradeCodec + weight sig + Q/P + Z-walk
# ============================================================
class FileCodec:
    """
    Wraps UTF8TradeCodec (from trading_platform.py, which in turn
    delegates to UTF_8_dzeta_dt.py) and appends the three
    inference-server-style verification layers:

        weight sig     8-byte SHA-256 of the payload
        Q/P            QPWeightEncoder on sampled weights
        Z-walk         giant_step_walk over the spectral C array
    """

    def __init__(self,
                 K_table:   int   = 128,      # must be 128 for lossless
                 K_alu:     int   = 64,
                 k_labels:  int   = 4,
                 threshold: float = 0.3,
                 K_ci:      int   = 20):
        if K_table != 128:
            raise ValueError(
                "K_table must be 128 for bijective byte ↔ index mapping")
        self.base = UTF8TradeCodec(
            K_table=K_table, K_alu=K_alu,
            k_labels=k_labels, threshold=threshold)
        self.K_table   = K_table
        self.K_alu     = K_alu
        self.k_labels  = k_labels
        self.threshold = threshold
        self.K_ci      = K_ci
        self.weight_encoder = QPWeightEncoder()
        self._C = build_spectral_coefficients(K_ci)

    # ---------- encode ----------
    def encode(self, payload: bytes) -> bytes:
        base_packet = self.base.encode(payload)
        weight_sig  = self._weight_signature(payload)
        return base_packet + struct.pack(">Q", weight_sig)

    # ---------- decode + verify ----------
    def decode(self, packet: bytes) -> Tuple[bytes, Dict]:
        if len(packet) < 8:
            return b"", dict(verified=False, error="packet < 8 B")

        base_packet     = packet[:-8]
        weight_sig_sent = struct.unpack(">Q", packet[-8:])[0]

        recovered, diag = self.base.decode(base_packet)
        if not diag.get("verified", False):
            diag.update(weight_match=False,
                        qp_consistent=False,
                        zwalk_consistent=False,
                        verified=False)
            return recovered, diag

        weight_sig_recv = self._weight_signature(recovered)
        weight_match    = (weight_sig_sent == weight_sig_recv)

        qp_ok, qp_diag       = self._qp_consistency(recovered)
        zwalk_ok, zwalk_diag = self._zwalk_consistency(recovered)

        diag["weight_match"]     = weight_match
        diag["qp_consistent"]    = qp_ok
        diag["zwalk_consistent"] = zwalk_ok
        diag["qp_diag"]          = qp_diag
        diag["zwalk_diag"]       = zwalk_diag
        diag["verified"] = (weight_match and qp_ok and zwalk_ok)
        return recovered, diag

    # ---------- weight signature ----------
    def _weight_signature(self, data: bytes) -> int:
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")

    # ---------- Q/P consistency ----------
    def _qp_consistency(self, data: bytes) -> Tuple[bool, Dict]:
        if len(data) == 0:
            return True, dict(n_samples=0)
        n_target = min(32, len(data))
        step = max(1, len(data) // n_target)
        samples = data[::step][:n_target]
        n_baby = n_giant = n_fail = 0
        for b in samples:
            w = (int(b) - 128) / 128.0
            enc = self.weight_encoder.encode(float(w))
            if enc["mode"] == "baby":
                n_baby += 1
            elif enc["mode"] == "giant":
                n_giant += 1
            else:
                n_fail += 1
        ok = (n_fail == 0) and (n_baby + n_giant == len(samples))
        return ok, dict(n_baby=n_baby, n_giant=n_giant,
                        n_fail=n_fail, n_samples=len(samples))

    # ---------- Z-walk consistency ----------
    def _zwalk_consistency(self, data: bytes) -> Tuple[bool, Dict]:
        if len(data) < 4:
            return True, dict(n_steps=0, expected=0, z=0)
        x_target = float(data[0])
        if x_target < 1.0:
            return True, dict(n_steps=1, expected=1, z=int(x_target))
        path = giant_step_walk(0.0, x_target, self._C, K=self.K_ci)
        expected = max(1, int(math.ceil(math.sqrt(x_target))))
        ok = (len(path) == expected)
        return ok, dict(n_steps=len(path), expected=expected,
                        z=int(x_target))


# ============================================================
#  Envelope file system  ·  resolve K_ci
# ============================================================
def resolve_K_from_envelope(path: str,
                           k_min: int = 8,
                           k_max: int = 64) -> int:
    """
    Pick K_ci by running the file through the envelope file system.

    If ``file_reader_envelope_sat`` is available, use its kept_qs
    chunk count; otherwise fall back to a size-based heuristic
    (⌈√filesize⌉ chunks).
    """
    n_chunks: int
    try:
        from file_reader_envelope_sat import EnvelopeFactory
        factory = EnvelopeFactory(K=k_max)
        env = factory.read_envelope(path)
        n_chunks = max(1, len(env.kept_qs))
    except Exception:
        size = os.path.getsize(path) if os.path.exists(path) else 0
        n_chunks = max(1, int(math.ceil(math.sqrt(max(size, 1)))))

    k = 1
    while k < n_chunks and k < k_max:
        k <<= 1
    return max(k_min, min(k_max, k))


# ============================================================
#  Sealer  ·  frame assembly
# ============================================================
class FileSealer:
    def __init__(self, K_ci: int = 20,
                 K_table: int = 128, K_alu: int = 64):
        self.K_ci    = K_ci
        self.K_table = K_table
        self.K_alu   = K_alu
        self.codec   = FileCodec(K_table=K_table, K_alu=K_alu, K_ci=K_ci)
        self.packer  = MobiusPacker(K=K_MOSB)

    # ---------- seal ----------
    def seal(self, payload: bytes, mode: int = MODE_RAW) -> Dict:
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
#  Frame inspection (deframe + parse header without decoding)
# ============================================================
def inspect_frame(frame: bytes) -> Tuple[FileHeader, bytes, Dict]:
    packer = MobiusPacker(K=K_MOSB)
    payload, S, H, m, cid = packer.unpack(frame)
    if len(payload) < FILE_SIZE:
        raise ValueError("payload too short for FILE header")
    header = FileHeader.unpack(payload[:FILE_SIZE])
    packet = payload[FILE_SIZE:FILE_SIZE + header.env_size]
    if len(packet) != header.env_size:
        raise ValueError(
            f"packet length mismatch: {len(packet)} != {header.env_size}")
    return header, packet, dict(mosb_S=S, mosb_H=H, mosb_m=m, cid=cid)


# ============================================================
#  Pretty print
# ============================================================
def print_breakdown(diag: Dict):
    keys = ("particle_match", "ugc_match", "alu_match", "fiber_match",
            "weight_match", "qp_consistent", "zwalk_consistent")
    print()
    print("  verification breakdown:")
    for k in keys:
        print(f"    {k:<20s} = {diag.get(k)}")
    if diag.get("qp_diag"):
        print(f"    qp_diag              = {diag['qp_diag']}")
    if diag.get("zwalk_diag"):
        print(f"    zwalk_diag           = {diag['zwalk_diag']}")
    if diag.get("n_survivors") is not None:
        print(f"    survivor_rate        = "
              f"{diag.get('survivor_rate', 0.0):.4f}")


# ============================================================
#  CLI commands
# ============================================================
def cmd_info(args):
    C = build_spectral_coefficients(20)
    print("=" * 70)
    print("MOSB file CLI  ·  UTF-8 UGC ALU + weight sig + Q/P + Z-walk")
    print("=" * 70)
    print(f"  FILE_MAGIC     = {FILE_MAGIC!r}")
    print(f"  FILE header    = {FILE_SIZE} B")
    print(f"  MOSB K         = {K_MOSB}")
    print(f"  K_table        = 128  (fixed; bijective byte ↔ index)")
    print(f"  K_alu          = 64")
    print(f"  K_ci (default) = 20   (overridable with -K)")
    print()
    print("  Spectral coefficients  ζ(t) = Σ C[i]·exp(i·t·i/α_i):")
    print(f"    C[0]   = {C[20]:+.4f}    +H_1")
    print(f"    C[±1]  = {C[21]:+.4f}    −H_2")
    print(f"    C[±20] = {C[40]:+.4f}    −H_21")
    print(f"    α range  ∈ [0.3628, 1/(π−e)] = [0.362800, 2.362283]")
    print()
    print("  Frame layout:")
    print(f"    [ MOSB header    (48 B) ]")
    print(f"    [ FILE header    ({FILE_SIZE} B) ]")
    print(f"    [ UTF-8 UGC ALU packet  ]  via UTF_8_dzeta_dt.py")
    print(f"    [ weight signature (8 B)]")
    print()
    print("  Verification flags on decode:")
    print("    particle_match / ugc_match / alu_match / fiber_match")
    print("    weight_match      8-byte SHA-256 of payload")
    print("    qp_consistent     QPWeightEncoder on sampled weights")
    print("    zwalk_consistent  giant_step_walk completes at ⌈√z⌉")


def cmd_seal(args):
    with open(args.file, "rb") as f:
        payload = f.read()

    try:
        payload.decode("utf-8")
        mode = MODE_TEXT
    except UnicodeDecodeError:
        mode = MODE_BINARY

    K_ci = args.K if args.K > 0 else resolve_K_from_envelope(args.file)

    sealer = FileSealer(K_ci=K_ci)
    result = sealer.seal(payload, mode=mode)

    out_path = args.out or (args.file + ".mosb")
    with open(out_path, "wb") as f:
        f.write(result["frame"])

    # quick post-seal verification (cheap)
    header, packet, mosb = inspect_frame(result["frame"])
    codec = FileCodec(K_ci=K_ci)
    _, diag = codec.decode(packet)

    print(f"  sealed     : {args.file}")
    print(f"  →          : {out_path}")
    print(f"  payload    : {len(payload)} B   mode={MODE_NAMES[mode]}")
    print(f"  packet     : {len(result['packet'])} B")
    print(f"  frame      : {len(result['frame'])} B")
    print(f"  K_ci       : {K_ci}")
    print(f"  K_table    : {sealer.K_table}")
    print(f"  mosb S/H/m : {result['mosb_S']:+.4f}  /  "
          f"{result['mosb_H']:.4f}  /  {result['mosb_m']:.4e}")
    print(f"  post-verify: {'VERIFIED' if diag.get('verified') else 'FAILED'}")


def cmd_verify(args):
    with open(args.frame, "rb") as f:
        frame = f.read()

    try:
        header, packet, mosb = inspect_frame(frame)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    K_ci = args.K if args.K > 0 else (header.K_ci or 20)
    codec = FileCodec(K_ci=K_ci)
    recovered, diag = codec.decode(packet)

    ok = diag.get("verified", False)

    print(f"  frame      : {args.frame}")
    print(f"  verdict    : {'VERIFIED' if ok else 'FAILED'}")
    print(f"  mode       : {MODE_NAMES.get(header.mode, header.mode)}")
    print(f"  K_ci       : {K_ci}   (header: {header.K_ci})")
    print(f"  K_table    : {header.K}   (must be 128)")
    print(f"  orig size  : {header.msg_len} B")
    print(f"  packet     : {header.env_size} B")
    print(f"  nonce      : {header.nonce.hex()[:16]}…")
    print(f"  mosb S/H/m : {mosb['mosb_S']:+.4f}  /  "
          f"{mosb['mosb_H']:.4f}  /  {mosb['mosb_m']:.4e}")
    print_breakdown(diag)

    if args.out and recovered:
        with open(args.out, "wb") as f:
            f.write(recovered)
        print(f"\n  recovered  → {args.out}  ({len(recovered)} B)")

    return 0 if ok else 1


def cmd_roundtrip(args):
    with open(args.file, "rb") as f:
        payload = f.read()

    try:
        payload.decode("utf-8")
        mode = MODE_TEXT
    except UnicodeDecodeError:
        mode = MODE_BINARY

    K_ci = args.K if args.K > 0 else resolve_K_from_envelope(args.file)

    sealer = FileSealer(K_ci=K_ci)
    seal_result = sealer.seal(payload, mode=mode)

    header, packet, mosb = inspect_frame(seal_result["frame"])
    codec = FileCodec(K_ci=K_ci)
    recovered, diag = codec.decode(packet)

    ok = diag.get("verified", False) and recovered == payload

    print("=" * 70)
    print(f"Round-trip  ·  {args.file}")
    print("=" * 70)
    print(f"  payload    : {len(payload)} B   mode={MODE_NAMES[mode]}")
    print(f"  packet     : {len(seal_result['packet'])} B")
    print(f"  frame      : {len(seal_result['frame'])} B")
    print(f"  K_table    : {sealer.K_table}")
    print(f"  K_ci       : {K_ci}")
    print(f"  recovered  : {len(recovered)} B   "
          f"match={'OK' if recovered == payload else 'FAIL'}")
    print(f"  verified   : {diag.get('verified')}")
    print_breakdown(diag)

    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mosb_file_cli",
        description=("MOSB-framed UTF-8 UGC ALU file sealer/verifier "
                     "with Q/P decomposition and Z-walk checks."))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="show configuration")
    p_info.set_defaults(func=cmd_info)

    p_seal = sub.add_parser("seal", help="encode file → MOSB frame")
    p_seal.add_argument("file")
    p_seal.add_argument("-o", "--out", default=None)
    p_seal.add_argument("-K", "--K", type=int, default=0,
                        help="K_ci for spectral sum (0 = auto from envelope)")
    p_seal.set_defaults(func=cmd_seal)

    p_ver = sub.add_parser("verify", help="verify a MOSB frame")
    p_ver.add_argument("frame")
    p_ver.add_argument("-o", "--out", default=None,
                       help="write recovered payload here")
    p_ver.add_argument("-K", "--K", type=int, default=0,
                       help="K_ci override (0 = use header / default)")
    p_ver.set_defaults(func=cmd_verify)

    p_rt = sub.add_parser("roundtrip", help="seal + verify in one go")
    p_rt.add_argument("file")
    p_rt.add_argument("-K", "--K", type=int, default=0,
                      help="K_ci (0 = auto from envelope)")
    p_rt.set_defaults(func=cmd_roundtrip)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())