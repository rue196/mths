#!/usr/bin/env python3
"""
inference_server_mobius.py
==========================

MOSB-framed inference server with UTF-8 UGC ALU KV cache verification
and UGC discrete-log model weight verification.

Trading platform → inference server mapping
-------------------------------------------
    trading_platform          inference_server_mobius
    ────────────────          ──────────────────────
    broker                    server
    exchange                  GPU-worker (implied)
    investor                  client
    TRDB header (55 B)        OITH header (55 B)
    order JSON                prompt + KV cache
    env_size                  kv_cache_size
    VOLUME_BUFFER             COMPUTE_BUFFER
    volume tier               compute tier

Verification  (adjusted from trading_platform.py)
-------------------------------------------------
    Base  (UTF8TradeCodec, unchanged):
        particle_match  ugc_match  alu_match  fiber_match

    UGC discrete-log extension  (UGC_discrete_train_envelope.py):
        weight_match        8-byte SHA-256 of the payload
        qp_consistent       Q/P decomposition on sampled weights
        zwalk_consistent    giant_step_walk completes at ⌈√z⌉

Output to user
--------------
    [ MOSB header             (48 B) ]
    [ OITH header             (55 B) ]   ← "trading header" replaced
                                           with output token info
    [ UTF-8 UGC ALU packet           ]   ← output tokens + KV cache

Compute allocation
------------------
    COMPUTE_BUFFER[tier] = parse_size(...)  via MobiusPacker
        SHARED        64K
        STANDARD      4M
        DEDICATED     64M
        RESERVED      1G

Imports (unchanged parents)
---------------------------
    trading_platform              UTF8TradeCodec, K_MOSB
    UGC_discrete_train_envelope   QPWeightEncoder, giant_step_walk
    complete_planes_spectral_sum  build_spectral_coefficients
    download_mboius_unpack_buffer MobiusPacker, parse_size
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- trading_platform  ·  reuse the base codec (unchanged) ----
from trading_platform import (
    UTF8TradeCodec,
    K_TRADING_TABLE as K_INFERENCE_TABLE,
    K_TRADING_ALU   as K_INFERENCE_ALU,
    K_MOSB,
)

# ---- UGC discrete-log training envelope (verification extension) ----
from UGC_discrete_train_envelope import (
    QPWeightEncoder,
    giant_step_walk,
)

from complete_planes_spectral_sum import (
    build_spectral_coefficients,
)

# ---- MOSB framing ----
from download_mboius_unpack_buffer import (
    MobiusPacker,
    parse_size,
)


# ============================================================
#  Enums
# ============================================================
class ClientTier(IntEnum):
    ANON    = 0
    AUTH    = 1
    ADMIN   = 2
    SYSTEM  = 3


class ComputeTier(IntEnum):
    SHARED      = 0
    STANDARD    = 1
    DEDICATED   = 2
    RESERVED    = 3


class ModelFamily(IntEnum):
    SMALL   = 0
    MEDIUM  = 1
    LARGE   = 2


# ============================================================
#  Compute tier buffers  (mirror of VOLUME_BUFFER)
# ============================================================
COMPUTE_BUFFER: Dict[ComputeTier, int] = {
    ComputeTier.SHARED:    parse_size("64K"),
    ComputeTier.STANDARD:  parse_size("4M"),
    ComputeTier.DEDICATED: parse_size("64M"),
    ComputeTier.RESERVED:  parse_size("1G"),
}


def classify_compute(tokens: int) -> ComputeTier:
    if tokens < 128:
        return ComputeTier.SHARED
    if tokens < 4096:
        return ComputeTier.STANDARD
    if tokens < 131_072:
        return ComputeTier.DEDICATED
    return ComputeTier.RESERVED


# ============================================================
#  OITH header  (55 bytes — same layout as TRDB)
# ============================================================
OITH_MAGIC = b"OITH"
OITH_FMT   = ">4sBBBiddd12sII"
OITH_SIZE  = struct.calcsize(OITH_FMT)      # 55


@dataclass
class OutputHeader:
    """
    OITH header — replaces TRDB.

    TRDB field          →  OITH field
    ─────────────────────────────────────
    magic (4s)          →  OITH
    role    (B)         →  client_tier
    commodity (B)       →  server_tier
    volume_tier (B)     →  compute_tier
    family_code (i)     →  model_id
    price   (d)         →  temperature
    qty     (d)         →  top_p
    timestamp (d)       →  timestamp
    nonce   (12s)       →  nonce
    env_size (I)        →  kv_cache_size
    msg_len (I)         →  output_token_count
    """
    client_tier:        ClientTier
    server_tier:        ClientTier
    compute_tier:       ComputeTier
    model_id:           int
    temperature:        float
    top_p:              float
    timestamp:          float
    nonce:              bytes
    kv_cache_size:      int
    output_token_count: int

    def pack(self) -> bytes:
        return struct.pack(
            OITH_FMT, OITH_MAGIC,
            int(self.client_tier), int(self.server_tier),
            int(self.compute_tier), int(self.model_id),
            float(self.temperature), float(self.top_p),
            float(self.timestamp), self.nonce,
            int(self.kv_cache_size), int(self.output_token_count),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "OutputHeader":
        (magic, ct, st, compt, mid,
         temp, top_p, ts, nonce, kv_sz, tok_n) = struct.unpack(
            OITH_FMT, data[:OITH_SIZE])
        if magic != OITH_MAGIC:
            raise ValueError(f"bad OITH magic: {magic!r}")
        return cls(
            client_tier=ClientTier(ct), server_tier=ClientTier(st),
            compute_tier=ComputeTier(compt), model_id=int(mid),
            temperature=temp, top_p=top_p, timestamp=ts,
            nonce=nonce, kv_cache_size=kv_sz, output_token_count=tok_n,
        )


# ============================================================
#  Request / response serialization
# ============================================================
def serialize_request(prompt_tokens: List[int],
                     kv_cache: bytes) -> bytes:
    kv_hash = hashlib.sha256(kv_cache).hexdigest()
    return json.dumps({
        "kind": "request",
        "prompt": [int(t) for t in prompt_tokens],
        "kv_hash": kv_hash,
        "kv_len": len(kv_cache),
        "kv_data": kv_cache.hex() if kv_cache else "",
    }, separators=(",", ":")).encode("utf-8")


def serialize_response(output_tokens: List[int],
                      kv_cache: bytes) -> bytes:
    kv_hash = hashlib.sha256(kv_cache).hexdigest()
    return json.dumps({
        "kind": "response",
        "tokens": [int(t) for t in output_tokens],
        "kv_hash": kv_hash,
        "kv_len": len(kv_cache),
        "kv_data": kv_cache.hex() if kv_cache else "",
    }, separators=(",", ":")).encode("utf-8")


def parse_payload(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


# ============================================================
#  InferenceCodec  ·  extends UTF8TradeCodec with UGC discrete-log
# ============================================================
class InferenceCodec:
    """
    Wraps UTF8TradeCodec.  Verification is adjusted to include:

        1. UTF8TradeCodec checks         particle / ugc / alu / fiber
        2. weight signature              8-byte SHA-256 of the payload
        3. Q/P consistency               QPWeightEncoder on sampled weights
        4. Z-walk consistency            giant_step_walk completes at ⌈√z⌉

    The 8-byte weight signature is appended to the packet.
    """

    def __init__(self,
                 K_table:   int = K_INFERENCE_TABLE,
                 K_alu:     int = K_INFERENCE_ALU,
                 k_labels:  int = 4,
                 threshold: float = 0.3,
                 K_ci:      int = 20):
        self.base    = UTF8TradeCodec(K_table=K_table, K_alu=K_alu,
                                       k_labels=k_labels,
                                       threshold=threshold)
        self.K_table = K_table
        self.K_alu   = K_alu
        self.K_ci    = K_ci
        self.weight_encoder = QPWeightEncoder()
        self._C = build_spectral_coefficients(K_ci)

    # --------------------------------------------------------
    def encode(self, payload: bytes) -> bytes:
        base_packet = self.base.encode(payload)
        weight_sig  = self._weight_signature(payload)
        return base_packet + struct.pack(">Q", weight_sig)

    # --------------------------------------------------------
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

        diag['weight_match']     = weight_match
        diag['qp_consistent']    = qp_ok
        diag['zwalk_consistent'] = zwalk_ok
        diag['qp_diag']          = qp_diag
        diag['zwalk_diag']       = zwalk_diag
        diag['verified'] = (weight_match and qp_ok and zwalk_ok)
        return recovered, diag

    # --------------------------------------------------------
    def _weight_signature(self, data: bytes) -> int:
        h = hashlib.sha256(data).digest()[:8]
        return int.from_bytes(h, "big")

    def _qp_consistency(self, data: bytes) -> Tuple[bool, Dict]:
        n_target = min(32, len(data))
        if n_target == 0:
            return True, dict(n_samples=0)
        step    = max(1, len(data) // n_target)
        samples = data[::step][:n_target]

        n_baby = n_giant = n_fail = 0
        for b in samples:
            w = (int(b) - 128) / 128.0
            enc = self.weight_encoder.encode(float(w))
            if enc['mode'] == 'baby':
                n_baby += 1
            elif enc['mode'] == 'giant':
                n_giant += 1
            else:
                n_fail += 1

        ok = (n_fail == 0) and (n_baby + n_giant == len(samples))
        return ok, dict(n_baby=n_baby, n_giant=n_giant, n_fail=n_fail,
                        n_samples=len(samples))

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
#  Tiered compute buffer  (mirror of TieredOrderBuffer)
# ============================================================
class TieredComputeBuffer:
    def __init__(self, tier: ComputeTier):
        self.tier       = tier
        self.max_bytes  = COMPUTE_BUFFER[tier]
        self.buffer     = bytearray()
        self.frames_queued = 0

    def add_frame(self, frame_bytes: bytes) -> List[bytes]:
        self.buffer.extend(frame_bytes)
        self.frames_queued += 1
        out: List[bytes] = []
        while len(self.buffer) >= self.max_bytes:
            out.append(bytes(self.buffer[:self.max_bytes]))
            del self.buffer[:self.max_bytes]
        return out

    def flush(self) -> bytes:
        if self.buffer:
            out = bytes(self.buffer)
            self.buffer.clear()
            return out
        return b""

    def fill_ratio(self) -> float:
        return len(self.buffer) / self.max_bytes


# ============================================================
#  Client
# ============================================================
class InferenceClient:
    def __init__(self, client_id: str, client_tier: ClientTier,
                 model_id: int, temperature: float = 0.7,
                 top_p: float = 0.9):
        self.client_id   = client_id
        self.client_tier = client_tier
        self.model_id    = model_id
        self.temperature = temperature
        self.top_p       = top_p
        self.codec       = InferenceCodec()
        self.packer      = MobiusPacker(K=K_MOSB)
        self.kv_cache    = b""

    # --------------------------------------------------------
    def submit_prompt(self, prompt_tokens: List[int]) -> Dict:
        # 1. serialize
        payload = serialize_request(prompt_tokens, self.kv_cache)

        # 2. encode (UTF8TradeCodec + UGC weight sig)
        packet = self.codec.encode(payload)

        # 3. compute tier
        total = len(prompt_tokens) + len(self.kv_cache) // 8
        tier  = classify_compute(total)

        # 4. OITH header
        header = OutputHeader(
            client_tier=self.client_tier,
            server_tier=ClientTier.SYSTEM,
            compute_tier=tier,
            model_id=self.model_id,
            temperature=self.temperature,
            top_p=self.top_p,
            timestamp=time.time(),
            nonce=os.urandom(12),
            kv_cache_size=len(packet),
            output_token_count=0,   # request → no output tokens
        )

        # 5. MOSB frame
        payload_hdr = header.pack() + packet
        frame, S, H, m = self.packer.pack(payload_hdr)

        return dict(
            frame=frame, header=header, tier=tier,
            packet=packet, mosb_S=S, mosb_H=H, mosb_m=m,
        )

    # --------------------------------------------------------
    def receive_response(self, frame: bytes) -> Dict:
        # 1. MOSB deframe
        payload, S, H, m, cid = self.packer.unpack(frame)

        # 2. OITH header
        if len(payload) < OITH_SIZE:
            raise ValueError("response too short for OITH header")
        header = OutputHeader.unpack(payload[:OITH_SIZE])
        packet = payload[OITH_SIZE:OITH_SIZE + header.kv_cache_size]
        if len(packet) != header.kv_cache_size:
            raise ValueError(
                f"packet length mismatch: {len(packet)} != "
                f"{header.kv_cache_size}")

        # 3. decode + verify
        recovered, diag = self.codec.decode(packet)
        if not diag.get("verified", False):
            raise ValueError(f"response verification failed: {diag}")

        # 4. parse
        response = parse_payload(recovered)

        # 5. update local KV cache
        kv_data = bytes.fromhex(response.get("kv_data", ""))
        self.kv_cache = kv_data

        return dict(
            tokens=response["tokens"],
            kv_cache=kv_data,
            header=header,
            diagnostics=diag,
        )


# ============================================================
#  Server
# ============================================================
class InferenceServer:
    def __init__(self):
        self.codec  = InferenceCodec()
        self.packer = MobiusPacker(K=K_MOSB)
        self.compute_buffers: Dict[ComputeTier, TieredComputeBuffer] = {
            t: TieredComputeBuffer(t) for t in ComputeTier
        }
        self.models:       Dict[int, Dict] = {}
        self.seen_nonces:  set             = set()
        self.completed:    List[Dict]      = []

    # --------------------------------------------------------
    def register_model(self, model_id: int, family: ModelFamily,
                      weights: bytes):
        self.models[model_id] = dict(
            family=family,
            weights=weights,
            weights_hash=hashlib.sha256(weights).hexdigest(),
        )

    # --------------------------------------------------------
    def process_frame(self, frame: bytes) -> Dict:
        # ---- 1. MOSB deframe ----
        try:
            payload, S, H, m, cid = self.packer.unpack(frame)
        except ValueError as e:
            return dict(approved=False, stage=0, reason=f"deframe: {e}")

        # ---- 2. OITH header ----
        if len(payload) < OITH_SIZE:
            return dict(approved=False, stage=0,
                        reason="payload too short for OITH")
        try:
            header = OutputHeader.unpack(payload[:OITH_SIZE])
        except ValueError as e:
            return dict(approved=False, stage=0, reason=f"OITH: {e}")

        packet = payload[OITH_SIZE:OITH_SIZE + header.kv_cache_size]
        if len(packet) != header.kv_cache_size:
            return dict(approved=False, stage=0,
                        reason="packet byte count mismatch")

        # ---- 3. nonce replay ----
        nonce_hex = header.nonce.hex()
        if nonce_hex in self.seen_nonces:
            return dict(approved=False, stage=0,
                        reason="replay detected (nonce reused)")
        self.seen_nonces.add(nonce_hex)

        # ---- 4. tiered compute buffering ----
        buffer = self.compute_buffers[header.compute_tier]
        buffer.add_frame(frame)

        # ---- 5. decode + verify ----
        recovered, diag = self.codec.decode(packet)
        if not diag.get("verified", False):
            return dict(
                approved=False, stage="ugc",
                reason=f"verification failed: {diag}",
                header=header, diagnostics=diag,
                buffer_fill=buffer.fill_ratio(),
            )

        # ---- 6. parse ----
        try:
            request = parse_payload(recovered)
        except Exception as e:
            return dict(approved=False, stage="ugc",
                        reason=f"parse: {e}", header=header)

        # ---- 7. model check ----
        model = self.models.get(header.model_id)
        if model is None:
            return dict(approved=False, stage="model",
                        reason=f"unknown model: {header.model_id}")

        # ---- 8. infer ----
        output_tokens, updated_kv = self._run_inference(
            model, request, header)

        # ---- 9. encode response ----
        response_payload = serialize_response(output_tokens, updated_kv)
        response_packet  = self.codec.encode(response_payload)

        # ---- 10. OITH header for the response ----
        out_header = OutputHeader(
            client_tier=header.client_tier,
            server_tier=header.server_tier,
            compute_tier=header.compute_tier,
            model_id=header.model_id,
            temperature=header.temperature,
            top_p=header.top_p,
            timestamp=time.time(),
            nonce=os.urandom(12),
            kv_cache_size=len(response_packet),
            output_token_count=len(output_tokens),
        )

        # ---- 11. MOSB frame ----
        out_payload = out_header.pack() + response_packet
        out_frame, S2, H2, m2 = self.packer.pack(out_payload)

        # ---- 12. record ----
        self.completed.append(dict(
            header=header, out_header=out_header,
            request=request, output_tokens=output_tokens,
            diagnostics=diag,
        ))

        return dict(
            approved=True, stage="ok",
            reason="inference complete",
            header=header, out_header=out_header,
            frame=out_frame,
            output_tokens=output_tokens,
            diagnostics=diag,
            mosb_S=S, mosb_H=H, mosb_m=m,
            buffer_fill=buffer.fill_ratio(),
        )

    # --------------------------------------------------------
    def _run_inference(self, model: Dict, request: Dict,
                      header: OutputHeader) -> Tuple[List[int], bytes]:
        """Toy inference — 8 output tokens + appended KV cache."""
        prompt  = request["prompt"]
        kv_data = bytes.fromhex(request.get("kv_data", ""))
        n_out   = min(8, max(1, len(prompt)))
        out_tokens: List[int] = []
        for i in range(n_out):
            last = prompt[-1] if prompt else 0
            tok  = (last + i * 7 + header.model_id) % 32000
            out_tokens.append(int(tok))
        updated_kv = kv_data + bytes([t % 256 for t in out_tokens])
        return out_tokens, updated_kv


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("MOSB inference server  ·  UGC discrete-log KV cache verification")
    print("=" * 78)
    print()

    # ---- compute tier buffers ----
    print("=" * 78)
    print("1  ·  compute tier buffers  (parse_size / VOLUME_BUFFER mirror)")
    print("=" * 78)
    for tier, size in COMPUTE_BUFFER.items():
        print(f"  {tier.name:<12s}  {size:>13,} bytes  "
              f"({size / (1 << 20):.4f} MiB)")
    print()

    # ---- setup ----
    server = InferenceServer()
    server.register_model(
        model_id=1, family=ModelFamily.SMALL,
        weights=b"small_weights_" * 100)
    server.register_model(
        model_id=2, family=ModelFamily.MEDIUM,
        weights=b"medium_weights_" * 200)
    print("=" * 78)
    print("2  ·  registered models")
    print("=" * 78)
    for mid, m in server.models.items():
        print(f"  model_id={mid}  "
              f"family={m['family'].name:<6s}  "
              f"hash={m['weights_hash'][:16]}…")
    print()

    alice = InferenceClient("alice", ClientTier.AUTH,    model_id=1)
    bob   = InferenceClient("bob",   ClientTier.AUTH,    model_id=2)
    carol = InferenceClient("carol", ClientTier.ADMIN,   model_id=1)

    # ---- round-trip ----
    print("=" * 78)
    print("3  ·  round-trip  ·  prompt → inference → response")
    print("=" * 78)
    sample_prompt = [100, 200, 300, 400, 500]
    sub = alice.submit_prompt(sample_prompt)

    print(f"  prompt tokens       = {sample_prompt}")
    print(f"  frame size          = {len(sub['frame'])} B")
    print(f"  OITH header         : "
          f"client={sub['header'].client_tier.name}, "
          f"server={sub['header'].server_tier.name}, "
          f"compute={sub['header'].compute_tier.name}, "
          f"model={sub['header'].model_id}")
    print(f"  kv_cache_size       = {sub['header'].kv_cache_size} B")
    print(f"  output_token_count  = {sub['header'].output_token_count}")
    print(f"  mosb S / H / m      = {sub['mosb_S']:+.4f} / "
          f"{sub['mosb_H']:.4f} / {sub['mosb_m']:.4e}")
    print()

    result = server.process_frame(sub["frame"])
    print(f"  server verdict      = "
          f"{'APPROVE' if result['approved'] else 'DECLINE'}")
    print(f"  reason              = {result['reason']}")
    if result["approved"]:
        diag = result["diagnostics"]
        print(f"  particle_match      = {diag['particle_match']}")
        print(f"  ugc_match           = {diag['ugc_match']}")
        print(f"  alu_match           = {diag['alu_match']}")
        print(f"  fiber_match         = {diag['fiber_match']}")
        print(f"  weight_match        = {diag['weight_match']}")
        print(f"  qp_consistent       = {diag['qp_consistent']}  "
              f"{diag['qp_diag']}")
        print(f"  zwalk_consistent    = {diag['zwalk_consistent']}  "
              f"{diag['zwalk_diag']}")
        print(f"  output tokens       = {result['output_tokens']}")
        print(f"  output header       : "
              f"kv_size={result['out_header'].kv_cache_size}, "
              f"tok_count={result['out_header'].output_token_count}")
    print()

    # ---- client receives response ----
    response = alice.receive_response(result["frame"])
    print(f"  alice received      = {response['tokens']}")
    print(f"  alice KV cache      = {len(response['kv_cache'])} B")
    print(f"  alice header        : "
          f"client={response['header'].client_tier.name}, "
          f"model={response['header'].model_id}, "
          f"out_count={response['header'].output_token_count}")
    print()

    # ---- tamper detection ----
    print("=" * 78)
    print("4  ·  tamper detection  (verification breakdown)")
    print("=" * 78)
    sub = alice.submit_prompt([1, 2, 3])
    frame_good = sub["frame"]
    frame_bad  = bytearray(frame_good)
    # flip a byte inside the UGC packet region
    frame_bad[48 + OITH_SIZE + 40] ^= 0x01
    frame_bad = bytes(frame_bad)

    r_good = server.process_frame(frame_good)
    r_bad  = server.process_frame(frame_bad)
    print(f"  untampered  : approved={r_good['approved']}  "
          f"reason={r_good['reason'][:52]}")
    if r_bad.get("diagnostics"):
        d = r_bad["diagnostics"]
        print(f"  tampered    : approved={r_bad['approved']}  "
              f"reason={r_bad['reason'][:60]}")
        print(f"    particle  = {d.get('particle_match')}  "
              f"ugc = {d.get('ugc_match')}  "
              f"alu = {d.get('alu_match')}  "
              f"fiber = {d.get('fiber_match')}")
        print(f"    weight    = {d.get('weight_match')}  "
              f"qp = {d.get('qp_consistent')}  "
              f"zwalk = {d.get('zwalk_consistent')}")
    else:
        print(f"  tampered    : approved={r_bad['approved']}  "
              f"reason={r_bad['reason'][:60]}")
    print()

    # ---- buffer state ----
    print("=" * 78)
    print("5  ·  compute tier buffer state")
    print("=" * 78)
    print(f"  {'tier':<12s}  {'queued':>8s}  "
          f"{'bytes buffered':>16s}  {'fill %':>8s}")
    print("  " + "-" * 54)
    for tier, buf in server.compute_buffers.items():
        print(f"  {tier.name:<12s}  {buf.frames_queued:>8d}  "
              f"{len(buf.buffer):>16,d}  "
              f"{100 * buf.fill_ratio():>7.2f}%")
    print()

    # ---- session summary ----
    print("=" * 78)
    print("6  ·  completed inferences")
    print("=" * 78)
    for i, rec in enumerate(server.completed[-5:]):
        h  = rec["header"]
        oh = rec["out_header"]
        print(f"  #{i+1}  "
              f"model={h.model_id}  "
              f"compute={h.compute_tier.name:<10s}  "
              f"prompt_len={len(rec['request']['prompt']):>3d}  "
              f"out_len={oh.output_token_count:>3d}  "
              f"kv={oh.kv_cache_size:>6d} B")
    print()

    # ---- complexity ----
    print("=" * 78)
    print("7  ·  complexity")
    print("=" * 78)
    print("  UTF-8 UGC ALU encode        O(n_bytes · K_alu)")
    print("  weight signature            O(n_bytes)")
    print("  Q/P consistency (32 samples) O(32) = O(1)")
    print("  Z-walk consistency          O(√z)  z = byte_0")
    print("  MOSB pack / unpack          O(packet_size)")
    print("  OITH header                 O(1)")
    print("  tiered compute buffer flush O(batch_size) per tier")
    print()
    print("  Verification layers (adjusted from trading_platform.py):")
    print("    (1) UTF8TradeCodec   particle / ugc / alu / fiber")
    print("    (2) weight signature 8-byte SHA-256")
    print("    (3) Q/P consistency  QPWeightEncoder on sampled weights")
    print("    (4) Z-walk           giant_step_walk step count")
    print()
    print("  Output frame layout:")
    print("    [ MOSB header           (48 B) ]")
    print("    [ OITH header           (55 B) ]  client/server/compute,")
    print("                                       model, temperature, top_p,")
    print("                                       nonce, kv_size, token_n")
    print("    [ UTF-8 UGC ALU packet  (var) ]  output tokens + KV cache")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()