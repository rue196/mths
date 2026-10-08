#!/usr/bin/env python3
"""
mobius_stream.py
================

Stream large files through a Möbius pack/unpack pipeline with a
bounded download buffer.

    URL ──▶ [buffer] ──▶ pack ──▶ packed stream ──▶ unpack ──▶ output

Each buffer chunk of raw bytes is:
    1. viewed as raw bytes (gibberish)
    2. framed with a Möbius header carrying invariants S, H, m
    3. streamed to the packed output
    4. simultaneously unpacked and appended to the output file

The stream ends when the HTTP source ends — i.e., when the buffer
returns empty.

CLI
---
    python mobius_stream.py <url> -o <output>
        [--buffer 4M] [--K 512]
        [--packed packed.bin]
        [--verify]
        [--only-pack]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import sys
import time
from typing import Iterator, Optional, Tuple


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)                  # ≈ 2.362338
DENSITY = 6.0 / (PI * PI)                 # ≈ 0.607927

MAGIC       = b"MOSB"                      # Möbius stream block
HEADER_FMT  = ">4sIQQddd"                  # magic, K, n_bytes, chunk_id, S, H, m
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # = 48 bytes


# ============================================================
#  Möbius sieve  ·  O(K)
# ============================================================
def mobius_sieve(K: int):
    if K < 1:
        return [0] * (K + 1)
    mu = [0] * (K + 1)
    mu[1] = 1
    primes: list[int] = []
    is_comp = [False] * (K + 1)
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
            else:
                mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Size parser  ·  4M, 1G, 512K, 1GiB
# ============================================================
def parse_size(s: str) -> int:
    s = s.strip().upper()
    units = [
        ("GIB", 1 << 30), ("MIB", 1 << 20), ("KIB", 1 << 10),
        ("GB",  1 << 30), ("MB",  1 << 20), ("KB",  1 << 10),
        ("G",   1 << 30), ("M",   1 << 20), ("K",   1 << 10),
        ("B",   1),
    ]
    for unit, mul in units:
        if s.endswith(unit):
            return int(float(s[:-len(unit)]) * mul)
    return int(s)


# ============================================================
#  Möbius pack / unpack
# ============================================================
class MobiusPacker:
    """
    Packs a byte chunk with a Möbius header.

        pack(chunk)  -> packed_bytes, S, H, m
        unpack(blob) -> chunk, S, H, m, chunk_id

    The header carries K, n_bytes, chunk_id, and the invariants
    S (supertrace), H (entropy), m (mass) computed over the chunk
    through the Möbius sieve.
    """
    def __init__(self, K: int = 512):
        self.K = K
        self.mu = mobius_sieve(K)
        self.kept = [k for k in range(1, K + 1) if self.mu[k] != 0]
        self.L = len(self.kept)
        if self.L == 0:
            raise ValueError("Möbius sieve returned no square-free indices")
        self._chunk_id = 0

    # --- invariants -------------------------------------------------
    @staticmethod
    def _supertrace(arr: bytes) -> float:
        S = 0.0
        for t, b in enumerate(arr):
            val = (b - 128) / 128.0
            S += val if (t % 2 == 0) else -val
        return S

    @staticmethod
    def _entropy_mass(S: float, N: int):
        if S == 0.0 or N <= 0:
            return 0.0, 0.0
        p = abs(S) / N
        if not (0.0 < p < 1.0):
            return 0.0, abs(S)
        H = -ALPHA * p * math.log(p)
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
        return H, m

    # --- pack / unpack ----------------------------------------------
    def pack(self, chunk: bytes) -> Tuple[bytes, float, float, float]:
        n = len(chunk)
        S = self._supertrace(chunk)
        H, m = self._entropy_mass(S, n)
        cid = self._chunk_id
        self._chunk_id += 1
        header = struct.pack(HEADER_FMT, MAGIC, self.K,
                             n, cid, S, H, m)
        return header + bytes(chunk), S, H, m

    def unpack(self, blob: bytes) -> Tuple[bytes, float, float, float, int]:
        if len(blob) < HEADER_SIZE:
            raise ValueError("packed chunk too short")
        magic, K, n, cid, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        if K != self.K:
            raise ValueError(f"K mismatch: {K} != {self.K}")
        payload = blob[HEADER_SIZE:HEADER_SIZE + n]
        if len(payload) != n:
            raise ValueError(
                f"payload length mismatch: {len(payload)} != {n}")
        # re-verify the invariants
        S2 = self._supertrace(payload)
        if abs(S2 - S) > 1e-9:
            raise ValueError(
                f"supertrace mismatch: {S2} != {S}")
        return bytes(payload), S, H, m, cid


# ============================================================
#  Streaming HTTP download
# ============================================================
def stream_url(url: str, buf_bytes: int
               ) -> Iterator[Tuple[bytes, int, Optional[int]]]:
    """
    Yield (chunk, bytes_received, total_or_None).
    Supports http(s):// and local file paths.
    """
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("", "file"):
        path = parsed.path if parsed.scheme == "file" else url
        total = os.path.getsize(path)
        with open(path, "rb") as f:
            received = 0
            while True:
                chunk = f.read(buf_bytes)
                if not chunk:
                    break
                received += len(chunk)
                yield chunk, received, total
        return

    req = urllib.request.Request(
        url, headers={"User-Agent": "mobius-stream/1.0"})
    with urllib.request.urlopen(req) as resp:
        cl = resp.headers.get("Content-Length")
        total = int(cl) if cl and cl.isdigit() else None
        received = 0
        while True:
            chunk = resp.read(buf_bytes)
            if not chunk:
                break
            received += len(chunk)
            yield chunk, received, total


# ============================================================
#  Progress bar
# ============================================================
class ProgressBar:
    def __init__(self, total: Optional[int], width: int = 40):
        self.total = total
        self.width = width
        self.t0 = time.perf_counter()
        self.n = 0

    def update(self, received: int, packed: int,
               S: float, m: float, gib_head: bytes):
        self.n = received
        dt = time.perf_counter() - self.t0
        rate = received / dt if dt > 0 else 0.0
        head = gib_head[:4].hex()
        if self.total:
            frac = received / self.total
            filled = int(frac * self.width)
            bar = "█" * filled + "░" * (self.width - filled)
            pct = frac * 100
            sys.stdout.write(
                f"\r  [{bar}] {pct:6.2f}%  "
                f"in {received/1e6:7.2f} MB  "
                f"out {packed/1e6:7.2f} MB  "
                f"{rate/1e6:6.2f} MB/s  "
                f"gib {head}…  "
                f"S={S:+.3e}  m={m:.2e}")
        else:
            sys.stdout.write(
                f"\r  in {received/1e6:7.2f} MB  "
                f"out {packed/1e6:7.2f} MB  "
                f"{rate/1e6:6.2f} MB/s  "
                f"gib {head}…  "
                f"S={S:+.3e}  m={m:.2e}")
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


# ============================================================
#  Main pipeline
# ============================================================
def run(url: str,
        output: str,
        buf_bytes: int = 4 << 20,
        K: int = 512,
        packed_path: Optional[str] = None,
        verify: bool = False,
        only_pack: bool = False) -> int:

    packer = MobiusPacker(K=K)
    L = packer.L

    print("=" * 78)
    print("Möbius stream  ·  download → buffer → pack → unpack → write")
    print("=" * 78)
    print(f"  URL         : {url}")
    print(f"  output      : {output}")
    print(f"  buffer      : {buf_bytes / (1 << 20):.4f} MiB "
          f"({buf_bytes} bytes)")
    print(f"  K (sieve)   : {K}")
    print(f"  |μ≠0| / K   : {L} / {K}  = {L/K:.4f}   "
          f"(target 6/π² = {DENSITY:.4f})")
    if packed_path:
        print(f"  packed log  : {packed_path}")
    if only_pack:
        print(f"  mode        : pack only (no unpack, no write)")
    elif verify:
        print(f"  verify      : SHA-256 round-trip check")
    print()

    out_f = None if only_pack else open(output, "wb")
    packed_f = open(packed_path, "wb") if packed_path else None

    sha_in = hashlib.sha256() if (verify and not only_pack) else None
    sha_out = hashlib.sha256() if (verify and not only_pack) else None

    bar = ProgressBar(total=None)
    total_in = 0
    total_packed = 0
    chunks_done = 0
    last_S = 0.0
    last_H = 0.0
    last_m = 0.0
    t0 = time.perf_counter()

    try:
        for chunk, received, total in stream_url(url, buf_bytes):
            if bar.total is None and total is not None:
                bar.total = total

            if sha_in is not None:
                sha_in.update(chunk)

            # --- 1. pack ---
            packed_chunk, S, H, m = packer.pack(chunk)
            total_packed += len(packed_chunk)

            if packed_f:
                packed_f.write(packed_chunk)

            # --- 2. unpack (round-trip) ---
            if not only_pack:
                raw_chunk, S2, H2, m2, cid = packer.unpack(packed_chunk)
                if raw_chunk != chunk:
                    raise RuntimeError(
                        f"round-trip mismatch on chunk {cid}")
                sha_out.update(raw_chunk)
                out_f.write(raw_chunk)

            total_in += len(chunk)
            chunks_done += 1
            last_S, last_H, last_m = S, H, m
            bar.update(total_in, total_packed, S, m, chunk)

        bar.finish()

        # --- verification ---
        if verify and not only_pack:
            h_in  = sha_in.hexdigest()
            h_out = sha_out.hexdigest()
            ok = (h_in == h_out)
            print(f"  sha256(in)  = {h_in}")
            print(f"  sha256(out) = {h_out}")
            print(f"  round-trip  = {'PASS' if ok else 'FAIL'}")
            if not ok:
                return 1

        dt = time.perf_counter() - t0
        print()
        print("--- summary ---")
        print(f"  chunks processed   : {chunks_done}")
        print(f"  bytes in           : {total_in:,}")
        print(f"  bytes packed       : {total_packed:,}")
        print(f"  packing ratio      : "
              f"{total_packed / max(total_in, 1):.6f}")
        print(f"  final S            : {last_S:+.6f}")
        print(f"  final H            : {last_H:.6f}")
        print(f"  final m            : {last_m:.6e}")
        print(f"  wall time          : {dt:.3f} s")
        print(f"  throughput (in)    : {total_in / dt / 1e6:.2f} MB/s")
        return 0

    finally:
        if out_f is not None:
            out_f.close()
        if packed_f:
            packed_f.close()


# ============================================================
#  CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="Stream large files through a Möbius pack/unpack "
                    "pipeline with a bounded download buffer.")
    p.add_argument("url",
                   help="HTTP(S) URL, file:// URL, or local path")
    p.add_argument("-o", "--output", required=True,
                   help="output file path")
    p.add_argument("--buffer", default="4M",
                   help="buffer size per chunk (e.g. 4M, 1G); default 4M")
    p.add_argument("--K", type=int, default=512,
                   help="Möbius sieve size K; default 512")
    p.add_argument("--packed", default=None,
                   help="optional path to write the packed stream as a log")
    p.add_argument("--verify", action="store_true",
                   help="SHA-256 verify the round-trip (source vs output)")
    p.add_argument("--only-pack", action="store_true",
                   help="only run the pack stage; skip unpack and write")
    args = p.parse_args()

    buf_bytes = parse_size(args.buffer)
    rc = run(
        url=args.url,
        output=args.output,
        buf_bytes=buf_bytes,
        K=args.K,
        packed_path=args.packed,
        verify=args.verify,
        only_pack=args.only_pack,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()