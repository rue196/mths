#!/usr/bin/env python3
"""
mosb_quicksort_search.py
========================

512-bit quicksort search engine over MOSB frames (cli_512 codec).

Each entry is a (512-bit key, arbitrary-bytes value) pair encoded as
a single MOSB frame — byte-identical wire format to cli_512.py:

    [ MOSB header    48 B ]
    [ FILE header    55 B ]
    [ UTF-8 UGC ALU packet ]  key(64 B) + value
    [ weight sig      8 B ]

Operations
----------
    build(entries)        512-bit parse + encode each entry
    sort()                quicksort on 512-bit integer keys
    search(key)           binary search, O(log n)
    pull(idx)             decode + full 512-bit verification
    search_and_pull(key)  binary search + pull + verify
    verify_all()          full sweep
    save(path) / load(path)    JSON + base64 persistence

Usage
-----
    python mosb_quicksort_search.py demo
    python mosb_quicksort_search.py bench
    python mosb_quicksort_search.py build entries.json -o index.json
    python mosb_quicksort_search.py search index.json "some-key"
    python mosb_quicksort_search.py query  index.json "some-key"
    python mosb_quicksort_search.py verify index.json
"""

from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ---- codecs (unchanged wire) ----
from cli_512 import FileCodec512, BITS, PREC_DIGITS
from cli import (
    FileCodec, FileHeader, FILE_SIZE,
    MODE_RAW, MODE_TEXT, MODE_BINARY, MODE_NAMES,
)
from tcp_mosb_geo import MobiusTransport


# ============================================================
#  MOSBQuickSortSearch
# ============================================================
class MOSBQuickSortSearch:
    """512-bit quicksort search engine over MOSB frames."""

    KEY_BYTES = 64   # 512-bit keys

    def __init__(self, K_ci: int = 12, use_512bit: bool = True):
        self.K_ci = int(K_ci)
        self.use_512bit = bool(use_512bit)
        self._make_codec()
        self.transport = MobiusTransport(K=512)
        self.entries: List[Dict[str, Any]] = []
        self._sort_calls = 0
        self._search_calls = 0
        self._pull_calls = 0

    # --------------------------------------------------------
    def _make_codec(self):
        if self.use_512bit:
            self.codec = FileCodec512(K_table=128, K_alu=64, K_ci=self.K_ci)
        else:
            self.codec = FileCodec(K_table=128, K_alu=64, K_ci=self.K_ci)

    # --------------------------------------------------------
    #  key normalisation  ·  pad/truncate to 64 bytes (512 bits)
    # --------------------------------------------------------
    def _make_key(self, key_in) -> bytes:
        if isinstance(key_in, str):
            key_in = key_in.encode("utf-8")
        if isinstance(key_in, int):
            if key_in < 0:
                raise ValueError("key integer must be non-negative")
            return int(key_in).to_bytes(self.KEY_BYTES, "big",
                                        signed=False)
        if not isinstance(key_in, (bytes, bytearray)):
            raise TypeError(
                f"key must be str / bytes / int, got {type(key_in)}")
        k = bytes(key_in)
        if len(k) >= self.KEY_BYTES:
            return k[:self.KEY_BYTES]
        return k + b"\x00" * (self.KEY_BYTES - len(k))

    @staticmethod
    def hash_key(data) -> bytes:
        """SHA-512 any input to exactly 64 bytes (512 bits)."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha512(bytes(data)).digest()

    # --------------------------------------------------------
    #  encode one entry  ·  512-bit parse
    # --------------------------------------------------------
    def _encode_entry(self, key_in, value_in) -> Dict[str, Any]:
        if isinstance(value_in, str):
            value_in = value_in.encode("utf-8")
        if not isinstance(value_in, (bytes, bytearray)):
            raise TypeError(
                f"value must be str / bytes, got {type(value_in)}")

        key_512 = self._make_key(key_in)
        value   = bytes(value_in)
        payload = key_512 + value

        packet = self.codec.encode(payload)
        header = FileHeader(
            mode=MODE_BINARY, reserved_a=0, reserved_b=0,
            K_ci=self.K_ci,
            threshold=self.codec.threshold,
            timestamp=time.time(),
            file_size=float(len(payload)),
            nonce=os.urandom(12),
            env_size=len(packet),
            msg_len=len(payload),
        )
        mosb = self.transport.frame(header.pack() + packet)
        return {
            "key_512":   key_512,
            "key_int":   int.from_bytes(key_512, "big"),
            "value_bytes": value,
            "payload":   payload,
            "frame":     mosb.bytes_,
            "header":    header,
            "mosb_S":    mosb.S,
            "mosb_H":    mosb.H,
            "mosb_m":    mosb.m,
        }

    # --------------------------------------------------------
    #  build
    # --------------------------------------------------------
    def build(self, entries) -> int:
        self.entries = [self._encode_entry(k, v) for (k, v) in entries]
        return len(self.entries)

    # --------------------------------------------------------
    #  quicksort  ·  median-of-three  ·  512-bit int comparison
    # --------------------------------------------------------
    def _qsort(self, arr, lo, hi):
        while lo < hi:
            mid = (lo + hi) // 2
            a_lo  = arr[lo]["key_int"]
            a_mid = arr[mid]["key_int"]
            a_hi  = arr[hi]["key_int"]
            # median-of-three
            if a_mid < a_lo:
                arr[lo], arr[mid] = arr[mid], arr[lo]
                a_lo, a_mid = a_mid, a_lo
            if a_hi < a_lo:
                arr[lo], arr[hi] = arr[hi], arr[lo]
                a_lo, a_hi = a_hi, a_lo
            if a_hi < a_mid:
                arr[mid], arr[hi] = arr[hi], arr[mid]
                a_mid, a_hi = a_hi, a_mid
            pivot = a_mid
            # partition
            i, j = lo, hi
            while i <= j:
                while arr[i]["key_int"] < pivot:
                    i += 1
                while arr[j]["key_int"] > pivot:
                    j -= 1
                if i <= j:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
                    j -= 1
            # recurse on smaller side, loop on larger (stack O(log n))
            if (j - lo) < (hi - i):
                self._qsort(arr, lo, j)
                lo = i
            else:
                self._qsort(arr, i, hi)
                hi = j

    def sort(self) -> int:
        self._sort_calls += 1
        if self.entries:
            self._qsort(self.entries, 0, len(self.entries) - 1)
        return len(self.entries)

    # --------------------------------------------------------
    #  binary search  ·  O(log n)
    # --------------------------------------------------------
    def search(self, key_in) -> int:
        self._search_calls += 1
        target = int.from_bytes(self._make_key(key_in), "big")
        lo, hi = 0, len(self.entries) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            k = self.entries[mid]["key_int"]
            if k == target:
                return mid
            if k < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    # --------------------------------------------------------
    #  pull  ·  decode + 512-bit verification
    # --------------------------------------------------------
    def pull(self, idx: int, verify: bool = True) -> Optional[Dict]:
        self._pull_calls += 1
        if idx < 0 or idx >= len(self.entries):
            return None
        e = self.entries[idx]
        if not verify:
            return {
                "key_512":     e["key_512"],
                "key_int":     e["key_int"],
                "value_bytes": e["value_bytes"],
                "verified":    None,
                "diag":        None,
            }
        try:
            inner, mosb = self.transport.deframe(e["frame"])
            if len(inner) < FILE_SIZE:
                return {"verified": False, "error": "short payload"}
            header = FileHeader.unpack(inner[:FILE_SIZE])
            packet = inner[FILE_SIZE:FILE_SIZE + header.env_size]
            recovered, diag = self.codec.decode(packet)
            expected = e["key_512"] + e["value_bytes"]
            verified = (bool(diag.get("verified", False))
                        and recovered == expected)
            return {
                "key_512":     e["key_512"],
                "key_int":     e["key_int"],
                "value_bytes": recovered[len(e["key_512"]):]
                               if verified else b"",
                "verified":    verified,
                "diag":        diag,
                "mosb_S":      mosb.S,
                "mosb_H":      mosb.H,
                "mosb_m":      mosb.m,
                "frame_size":  mosb.size,
                "packet_size": len(packet),
                "bits":        BITS if self.use_512bit else 64,
            }
        except Exception as ex:
            return {"verified": False, "error": str(ex)}

    # --------------------------------------------------------
    def search_and_pull(self, key_in) -> Dict:
        idx = self.search(key_in)
        if idx < 0:
            return {"found": False, "verified": False}
        r = self.pull(idx, verify=True) or {"verified": False}
        r["found"] = True
        r["position"] = idx
        return r

    # --------------------------------------------------------
    def verify_all(self) -> Dict:
        n_ok = 0
        failures: List[Dict] = []
        for i in range(len(self.entries)):
            r = self.pull(i, verify=True)
            if r and r.get("verified"):
                n_ok += 1
            else:
                failures.append({
                    "idx":   i,
                    "error": (r.get("error") if r else "pull failed"),
                })
        return {
            "total":    len(self.entries),
            "verified": n_ok,
            "all_ok":   (n_ok == len(self.entries)),
            "failures": failures,
        }

    # --------------------------------------------------------
    #  persistence  ·  JSON + base64
    # --------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "version":    1,
            "K_ci":       self.K_ci,
            "use_512bit": self.use_512bit,
            "bits":       BITS if self.use_512bit else 64,
            "n_entries":  len(self.entries),
            "entries": [
                {
                    "key_b64":   base64.b64encode(e["key_512"]).decode("ascii"),
                    "value_b64": base64.b64encode(e["value_bytes"]).decode("ascii"),
                    "frame_b64": base64.b64encode(e["frame"]).decode("ascii"),
                }
                for e in self.entries
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> int:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("version") != 1:
            raise ValueError(
                f"unsupported index version: {data.get('version')}")
        self.K_ci       = int(data["K_ci"])
        self.use_512bit = bool(data["use_512bit"])
        self._make_codec()
        self.entries = []
        for item in data["entries"]:
            key_512 = base64.b64decode(item["key_b64"])
            value   = base64.b64decode(item["value_b64"])
            frame   = base64.b64decode(item["frame_b64"])
            self.entries.append({
                "key_512":     key_512,
                "key_int":     int.from_bytes(key_512, "big"),
                "value_bytes": value,
                "payload":     key_512 + value,
                "frame":       frame,
                "header":      None,   # not persisted
            })
        return len(self.entries)

    # --------------------------------------------------------
    def stats(self) -> Dict:
        return {
            "n":            len(self.entries),
            "K_ci":         self.K_ci,
            "bits":         BITS if self.use_512bit else 64,
            "use_512bit":   self.use_512bit,
            "sort_calls":   self._sort_calls,
            "search_calls": self._search_calls,
            "pull_calls":   self._pull_calls,
        }


# ============================================================
#  Demo
# ============================================================
def demo() -> int:
    print("=" * 78)
    print("MOSB Quicksort Search Engine  ·  512-bit keys")
    print("=" * 78)
    print(f"  BITS       = {BITS}")
    print(f"  KEY_BYTES  = {MOSBQuickSortSearch.KEY_BYTES}  (512-bit int keys)")
    print()

    engine = MOSBQuickSortSearch(K_ci=12, use_512bit=True)

    # ---- 1. build ----
    print("=" * 78)
    print("1  ·  build  ·  parse 20 entries at 512-bit")
    print("=" * 78)
    entries = [(f"key-{i:03d}".encode(), f"value-{i}".encode())
               for i in range(20)]
    n = engine.build(entries)
    print(f"  built n = {n} entries")
    print(f"  first frame size = {len(engine.entries[0]['frame'])} B")
    print(f"  first packet size = {engine.entries[0]['header'].env_size} B")
    print()

    # ---- 2. sort ----
    print("=" * 78)
    print("2  ·  sort  ·  quicksort on 512-bit integer keys")
    print("=" * 78)
    t0 = time.perf_counter()
    engine.sort()
    dt_sort = (time.perf_counter() - t0) * 1e3
    is_sorted = all(engine.entries[i]["key_int"] <= engine.entries[i+1]["key_int"]
                    for i in range(len(engine.entries) - 1))
    print(f"  sort time = {dt_sort:.2f} ms")
    print(f"  sorted    = {is_sorted}")
    print(f"  first 3 keys:")
    for i in range(min(3, len(engine.entries))):
        raw = engine.entries[i]["key_512"].rstrip(b"\x00")
        pref = engine.entries[i]["key_int"] >> 448
        print(f"    {i}: {raw!r}  int_prefix=0x{pref:x}")
    print()

    # ---- 3. search ----
    print("=" * 78)
    print("3  ·  search  ·  binary search O(log n)")
    print("=" * 78)
    for target in [b"key-010", b"key-000", b"key-019", b"key-999"]:
        t0 = time.perf_counter()
        idx = engine.search(target)
        dt = (time.perf_counter() - t0) * 1e6
        if idx >= 0:
            print(f"  search {target.decode():<10}  →  idx={idx:>3}  ({dt:.1f} µs)")
        else:
            print(f"  search {target.decode():<10}  →  NOT FOUND  ({dt:.1f} µs)")
    print()

    # ---- 4. pull with verify ----
    print("=" * 78)
    print("4  ·  pull  ·  decode + 512-bit verification")
    print("=" * 78)
    idx = engine.search(b"key-015")
    result = engine.pull(idx, verify=True)
    print(f"  pull idx={idx}  (verify=True)")
    print(f"  key_int          = {result['key_int']}")
    print(f"  value            = {result['value_bytes']}")
    print(f"  verified         = {result['verified']}")
    d = result.get("diag") or {}
    print(f"  particle         = {d.get('particle_match')}")
    print(f"  ugc              = {d.get('ugc_match')}")
    print(f"  alu              = {d.get('alu_match')}")
    print(f"  fiber            = {d.get('fiber_match')}")
    print(f"  weight_match     = {d.get('weight_match')}")
    print(f"  qp_consistent    = {d.get('qp_consistent')}")
    print(f"  zwalk_consistent = {d.get('zwalk_consistent')}")
    print(f"  bits             = {result['bits']}")
    print()

    # ---- 5. search_and_pull ----
    print("=" * 78)
    print("5  ·  search_and_pull  ·  combined operation")
    print("=" * 78)
    for target in [b"key-007", b"key-999"]:
        r = engine.search_and_pull(target)
        if r.get("found"):
            print(f"  {target.decode():<10}  found idx={r['position']:>3}  "
                  f"value={r['value_bytes']}  verified={r['verified']}")
        else:
            print(f"  {target.decode():<10}  NOT FOUND")
    print()

    # ---- 6. verify all ----
    print("=" * 78)
    print("6  ·  verify_all  ·  full sweep")
    print("=" * 78)
    t0 = time.perf_counter()
    vr = engine.verify_all()
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  verified  = {vr['verified']}/{vr['total']}")
    print(f"  all_ok    = {vr['all_ok']}")
    print(f"  time      = {dt:.0f} ms  ({dt/max(1, vr['total']):.1f} ms/entry)")
    print()

    # ---- 7. wire identity with cli_512 ----
    print("=" * 78)
    print("7  ·  wire format identity  ·  vs cli_512.py")
    print("=" * 78)
    ref = FileCodec512(K_ci=engine.K_ci)
    entry = engine.entries[0]
    ref_packet = ref.encode(entry["payload"])
    inner, _ = engine.transport.deframe(entry["frame"])
    header = FileHeader.unpack(inner[:FILE_SIZE])
    actual_packet = inner[FILE_SIZE:FILE_SIZE + header.env_size]
    print(f"  payload size       = {len(entry['payload'])} B")
    print(f"  ref packet size    = {len(ref_packet)} B")
    print(f"  actual packet size = {len(actual_packet)} B")
    print(f"  byte-identical     = {ref_packet == actual_packet}")
    print()

    # ---- 8. tamper detection ----
    print("=" * 78)
    print("8  ·  tamper detection  ·  modify one packet byte")
    print("=" * 78)
    saved = engine.entries[3]["frame"]
    bad = bytearray(saved)
    bad[48 + FILE_SIZE + 40] ^= 0x01     # flip a byte inside the UGC ALU packet
    engine.entries[3]["frame"] = bytes(bad)
    r_bad = engine.pull(3, verify=True)
    engine.entries[3]["frame"] = saved
    r_good = engine.pull(3, verify=True)
    print(f"  untampered : verified={r_good['verified']}")
    print(f"  tampered   : verified={r_bad.get('verified')}")
    d = r_bad.get("diag") or {}
    print(f"    particle = {d.get('particle_match')}  "
          f"ugc = {d.get('ugc_match')}  "
          f"alu = {d.get('alu_match')}  "
          f"fiber = {d.get('fiber_match')}")
    print()

    # ---- 9. stats ----
    print("=" * 78)
    print("9  ·  engine stats")
    print("=" * 78)
    for k, v in engine.stats().items():
        print(f"  {k:<14s} = {v}")
    print()

    # ---- 10. complexity ----
    print("=" * 78)
    print("10  ·  complexity")
    print("=" * 78)
    print("  build (encode each)   O(n · |payload| · K_alu) + 512-bit Q/P/Z-walk")
    print("  quicksort             O(n log n)  512-bit int comparison")
    print("  binary search         O(log n)")
    print("  pull (verify)         O(|frame| · K_alu) + 512-bit Q/P/Z-walk")
    print("  verify_all            O(n · pull)")
    print()
    print("Done.")
    return 0


# ============================================================
#  Bench  ·  scaling check
# ============================================================
def bench() -> int:
    print("=" * 78)
    print("Benchmark  ·  quicksort O(n log n)  ·  binary search O(log n)")
    print("=" * 78)
    print(f"  {'n':>6s}  {'build':>10s}  {'sort':>10s}  "
          f"{'search':>10s}  {'verify':>10s}")
    print("  " + "-" * 60)
    for n in (10, 50, 100, 200):
        engine = MOSBQuickSortSearch(K_ci=4, use_512bit=False)
        entries = [(f"key-{i:05d}".encode(), f"v-{i}".encode())
                   for i in range(n)]
        t0 = time.perf_counter(); engine.build(entries)
        t_build = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter(); engine.sort()
        t_sort = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        for _ in range(100):
            engine.search(f"key-{n//2:05d}".encode())
        t_search = (time.perf_counter() - t0) * 1e3 / 100 * 1000   # µs

        t0 = time.perf_counter(); engine.verify_all()
        t_verify = (time.perf_counter() - t0) * 1e3

        print(f"  {n:>6d}  {t_build:>8.1f}ms  {t_sort:>8.2f}ms  "
              f"{t_search:>8.2f}µs  {t_verify:>8.1f}ms")
    print()
    print("  Note: 512-bit Q/P + Z-walk verification dominates build/pull")
    print("        times; sort and search are pure 512-bit integer ops.")
    return 0


# ============================================================
#  CLI
# ============================================================
def cmd_demo(_args) -> int:
    return demo()


def cmd_bench(_args) -> int:
    return bench()


def cmd_build(args) -> int:
    with open(args.entries, "r") as f:
        data = json.load(f)
    entries = [(item["key"], item["value"]) for item in data]
    engine = MOSBQuickSortSearch(K_ci=args.K_ci,
                                 use_512bit=not args.no_512)
    engine.build(entries)
    engine.sort()
    engine.save(args.output)
    print(f"  built {len(engine.entries)} entries")
    print(f"  → {args.output}")
    return 0


def cmd_search(args) -> int:
    engine = MOSBQuickSortSearch(use_512bit=not args.no_512)
    engine.load(args.index)
    idx = engine.search(args.key.encode("utf-8"))
    if idx < 0:
        print(f"  NOT FOUND: {args.key}")
        return 1
    print(f"  found idx={idx}")
    raw = engine.entries[idx]["key_512"].rstrip(b"\x00")
    print(f"  key = {raw!r}")
    return 0


def cmd_query(args) -> int:
    engine = MOSBQuickSortSearch(use_512bit=not args.no_512)
    engine.load(args.index)
    r = engine.search_and_pull(args.key.encode("utf-8"))
    if not r.get("found"):
        print(f"  NOT FOUND: {args.key}")
        return 1
    print(f"  found idx={r['position']}")
    print(f"  verified = {r['verified']}")
    print(f"  value    = {r['value_bytes']}")
    return 0 if r.get("verified") else 2


def cmd_verify(args) -> int:
    engine = MOSBQuickSortSearch(use_512bit=not args.no_512)
    engine.load(args.index)
    vr = engine.verify_all()
    print(f"  verified = {vr['verified']}/{vr['total']}")
    print(f"  all_ok   = {vr['all_ok']}")
    if vr["failures"]:
        for f in vr["failures"]:
            print(f"    idx={f['idx']}  error={f['error']}")
    return 0 if vr["all_ok"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mosb_quicksort_search",
        description="512-bit quicksort search engine over MOSB frames")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo").set_defaults(func=cmd_demo)
    sub.add_parser("bench").set_defaults(func=cmd_bench)

    p_build = sub.add_parser("build",
                             help="build index from entries JSON")
    p_build.add_argument("entries")
    p_build.add_argument("-o", "--output", required=True)
    p_build.add_argument("-K", "--K-ci", type=int, default=12)
    p_build.add_argument("--no-512", action="store_true")
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search",
                              help="binary search on a saved index")
    p_search.add_argument("index")
    p_search.add_argument("key")
    p_search.add_argument("--no-512", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_query = sub.add_parser("query",
                             help="search + pull with 512-bit verification")
    p_query.add_argument("index")
    p_query.add_argument("key")
    p_query.add_argument("--no-512", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_verify = sub.add_parser("verify",
                              help="full 512-bit verification sweep")
    p_verify.add_argument("index")
    p_verify.add_argument("--no-512", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())