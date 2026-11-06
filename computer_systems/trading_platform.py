#!/usr/bin/env python3
"""
mobius_trading_platform.py
==========================

Multi-role trading platform  ·  broker ↔ exchange ↔ investor
for futures and commodities  ·  UTF-8 UGC ALU encoding

Encoding stack (replaces SATEnvelopeCipher)
-------------------------------------------
    UTF_8_dzeta_dt.py          UTF8Table / DiscreteParticle
                               build_particle_cloud
                               ugc_survivors / ugc_satisfied
                               ReadOnlyBuffer / ReadOnlyALU
                               fiber_propagate

Each order JSON is encoded as a UTF-8 UGC ALU packet:

    bytes  →  particle cloud (v(i), v(j), dRe/dn, dζ/dt)
           →  UGC boolean gate  (cross-coupling ≥ τ ∧ label match)
           →  ALU over MOSB read-only buffer  (ADD/SUB/MUL/DIV)
           →  fiber kernel signature  exp(−α|z|)
           →  serialized packet (particle data + UGC + ALU + fiber)

Verification on decode
----------------------
    particle consistency   recompute cloud from recovered bytes
    UGC consistency        survivors/non-survivors identical
    ALU consistency        z-values match per op
    fiber consistency      supertrace match
    → all four pass  ⇒  verified

Pipeline (unchanged outer structure)
------------------------------------
    transaction_system.py            account registration + Stage 1 checks
                                     build_sat_envelope (account commitment)
    download_mboius_unpack_buffer.py MobiusPacker + tiered buffers
                                     parse_size VOLUME_BUFFER

MOSB frame layout
-----------------
    [ MOSB header             (48 B) ]
    [ TRDB trading header     (55 B) ]  role, commodity, tier, family
    [ UTF-8 UGC ALU packet           ]  variable length
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- transaction_system · Stage 1 account commitment ----
from transaction_system import (
    mobius_sieve as tx_sieve,
    build_sat_envelope,
    ALPHA, A_STEP, DENSITY, TAU, UGC_K,
)

# ---- UTF-8 UGC ALU codec  ·  the encoding/decoding layer ----
from UTF_8_dzeta_dt import (
    UTF8Table,
    DiscreteParticle,
    ReadOnlyBuffer,
    ReadOnlyALU,
    build_particle_cloud,
    ugc_survivors,
    ugc_satisfied,
    ugc_label,
    fiber_propagate,
    utf8_byte_to_signed_index,
    K_DEFAULT as UTF8_K_DEFAULT,
)

# ---- download_mboius_unpack_buffer · MOSB framing ----
from download_mboius_unpack_buffer import (
    MobiusPacker,
    parse_size,
)


# ============================================================
#  Codec sizing
# ============================================================
K_TRADING_TABLE = 128       # bijective: 256 bytes → 256 distinct i
K_TRADING_ALU   = 64
K_MOSB          = 256


# ============================================================
#  Enums
# ============================================================
class Role(IntEnum):
    BROKER    = 0
    EXCHANGE  = 1
    INVESTOR  = 2
    REGULATOR = 3


class VolumeTier(IntEnum):
    RETAIL        = 0
    STANDARD      = 1
    INSTITUTIONAL = 2
    WHOLESALE     = 3


class Commodity(IntEnum):
    WTI     = 1
    BRENT   = 2
    NATGAS  = 3
    GOLD    = 4
    SILVER  = 5
    COPPER  = 6
    CORN    = 7
    WHEAT   = 8
    SP500   = 9
    NASDAQ  = 10


# ============================================================
#  Commodity specifications
# ============================================================
@dataclass(frozen=True)
class CommoditySpec:
    family_code: int       # 0 = real, 1 = imag, 2 = dc
    tick_size:   float
    lot_size:    float
    max_qty:     float


COMMODITY_SPECS: Dict[Commodity, CommoditySpec] = {
    Commodity.WTI:    CommoditySpec(0, 0.01,   1000,    10_000),
    Commodity.BRENT:  CommoditySpec(0, 0.01,   1000,    10_000),
    Commodity.NATGAS: CommoditySpec(0, 0.001,  10_000,  100_000),
    Commodity.GOLD:   CommoditySpec(1, 0.10,   100,     5_000),
    Commodity.SILVER: CommoditySpec(1, 0.005,  5000,    50_000),
    Commodity.COPPER: CommoditySpec(1, 0.0005, 25_000,  250_000),
    Commodity.CORN:   CommoditySpec(2, 0.25,   5000,    100_000),
    Commodity.WHEAT:  CommoditySpec(2, 0.25,   5000,    100_000),
    Commodity.SP500:  CommoditySpec(2, 0.25,   50,      10_000),
    Commodity.NASDAQ: CommoditySpec(2, 0.25,   20,      10_000),
}


# ============================================================
#  Volume tier → buffer size  (unchanged)
# ============================================================
VOLUME_BUFFER: Dict[VolumeTier, int] = {
    VolumeTier.RETAIL:        parse_size("64K"),
    VolumeTier.STANDARD:      parse_size("4M"),
    VolumeTier.INSTITUTIONAL: parse_size("64M"),
    VolumeTier.WHOLESALE:     parse_size("1G"),
}


def classify_volume(qty: float) -> VolumeTier:
    if qty < 1_000:
        return VolumeTier.RETAIL
    if qty < 100_000:
        return VolumeTier.STANDARD
    if qty < 10_000_000:
        return VolumeTier.INSTITUTIONAL
    return VolumeTier.WHOLESALE


# ============================================================
#  TRDB trading header  (unchanged format — env_size now carries
#  the UTF-8 UGC packet size instead of SAT envelope size)
# ============================================================
TRADING_MAGIC = b"TRDB"
TRADING_FMT   = ">4sBBBiddd12sII"
TRADING_SIZE  = struct.calcsize(TRADING_FMT)      # 55 bytes


@dataclass
class TradingHeader:
    role:         Role
    commodity:    Commodity
    volume_tier:  VolumeTier
    family_code:  int
    price:        float
    qty:          float
    timestamp:    float
    nonce:        bytes
    env_size:     int
    msg_len:      int

    def pack(self) -> bytes:
        return struct.pack(
            TRADING_FMT, TRADING_MAGIC,
            int(self.role), int(self.commodity), int(self.volume_tier),
            int(self.family_code), float(self.price), float(self.qty),
            float(self.timestamp), self.nonce,
            int(self.env_size), int(self.msg_len),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "TradingHeader":
        (magic, role, comm, tier, fam,
         price, qty, ts, nonce, env_sz, msg_len) = struct.unpack(
            TRADING_FMT, data[:TRADING_SIZE])
        if magic != TRADING_MAGIC:
            raise ValueError(f"bad trading magic: {magic!r}")
        return cls(
            role=Role(role), commodity=Commodity(comm),
            volume_tier=VolumeTier(tier), family_code=int(fam),
            price=price, qty=qty, timestamp=ts, nonce=nonce,
            env_size=env_sz, msg_len=msg_len,
        )


# ============================================================
#  Order
# ============================================================
@dataclass
class Order:
    order_id:    str
    role:        Role
    entity:      str
    commodity:   Commodity
    side:        str
    price:       float
    qty:         float
    timestamp:   float
    counterparty: str = ""

    def to_json(self) -> bytes:
        return json.dumps({
            "id": self.order_id, "role": int(self.role),
            "entity": self.entity, "comm": int(self.commodity),
            "side": self.side, "price": self.price, "qty": self.qty,
            "ts": self.timestamp, "counterparty": self.counterparty,
        }, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "Order":
        obj = json.loads(data.decode("utf-8"))
        return cls(
            order_id=obj["id"], role=Role(obj["role"]),
            entity=obj["entity"], commodity=Commodity(obj["comm"]),
            side=obj["side"], price=obj["price"], qty=obj["qty"],
            timestamp=obj["ts"], counterparty=obj.get("counterparty", ""),
        )


# ============================================================
#  Role routing table
# ============================================================
ALLOWED_ROUTES: Dict[Role, Tuple[Role, ...]] = {
    Role.INVESTOR:  (Role.BROKER,),
    Role.BROKER:    (Role.EXCHANGE, Role.INVESTOR),
    Role.EXCHANGE:  (Role.BROKER, Role.REGULATOR),
    Role.REGULATOR: (Role.BROKER, Role.EXCHANGE, Role.INVESTOR),
}


def role_allowed(src: Role, dst: Role) -> bool:
    return dst in ALLOWED_ROUTES[src]


# ============================================================
#  UTF-8 UGC ALU codec  ·  the encoding/decoding layer
# ============================================================
class UTF8TradeCodec:
    """
    Encodes and decodes order bytes via the UTF-8 UGC ALU pipeline.

    Encoding
    --------
        bytes → UTF8Table → particle cloud
              → UGC boolean gate
              → ALU over MOSB read-only buffer
              → fiber kernel signature
              → serialized packet

    Decoding
    --------
        packet → particle i values → recover bytes
               → recompute particle cloud, UGC, ALU, fiber
               → verify all four against the sent packet

    Verification flags
    ------------------
        particle_match   i, x, y reconstructed identically
        ugc_match        survivor/non-survivor index lists match
        alu_match        every ADD/SUB/MUL/DIV z-value matches
        fiber_match      supertrace of fiber output matches

        verified = ∧ (particle, ugc, alu, fiber)
    """

    # op encoding for serialization
    _OP_IDS = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}
    _OP_NAMES = {v: k for k, v in _OP_IDS.items()}

    def __init__(self,
                 K_table:   int = K_TRADING_TABLE,
                 K_alu:     int = K_TRADING_ALU,
                 k_labels:  int = UGC_K,
                 threshold: float = 0.3):
        # K = 128 is required for lossless byte ↔ index inversion.
        if K_table != 128:
            raise ValueError(
                f"K_table must be 128 for byte-exact round-trip; "
                f"got {K_table}")
        self.K_table   = K_table
        self.K_alu     = K_alu
        self.k_labels  = k_labels
        self.threshold = threshold

        self.table = UTF8Table.build(K_table)
        self.alu   = ReadOnlyALU(K_alu)

        # bijective inverse: signed-index n → original byte b
        self._inverse: Dict[int, int] = {}
        for b in range(256):
            n = utf8_byte_to_signed_index(b, K_table)
            self._inverse[n] = b

    # --------------------------------------------------------
    def encode(self, data: bytes) -> bytes:
        """Encode bytes → UTF-8 UGC ALU packet."""
        # 1. particle cloud
        particles = build_particle_cloud(self.table, data)

        # 2. UGC boolean gate
        surv, non = ugc_survivors(
            particles, self.K_table, self.table.mu,
            self.k_labels, self.threshold)

        # 3. MOSB read-only buffer
        buf = ReadOnlyBuffer(K=self.K_table)
        for p in particles:
            buf.append(struct.pack(">dd", p.x, p.y))

        # 4. ALU operations
        alu_results = []
        n_chunks = len(buf.chunks)
        for cid in range(0, min(8, n_chunks) - 1, 2):
            for op in ("ADD", "SUB", "MUL", "DIV"):
                alu_results.append(self.alu.dispatch(buf, cid, cid + 1, op))

        # 5. fiber kernel signature
        signal = np.array([p.vx for p in particles])
        fiber  = fiber_propagate(signal, width=41)
        fiber_S = self._supertrace(fiber)

        # 6. serialize
        return self._serialize(
            particles, surv, non, alu_results, fiber_S, len(data))

    # --------------------------------------------------------
    def decode(self, packet: bytes) -> Tuple[bytes, Dict]:
        """
        Decode UTF-8 UGC ALU packet.

        Returns (recovered_bytes, diagnostics).
        `diagnostics['verified']` is True iff all four
        consistency checks pass.
        """
        try:
            particles, surv_sent, non_sent, alu_sent, fiber_S_sent, n_orig = \
                self._deserialize(packet)
        except Exception as e:
            return b"", dict(error=f"parse: {e}", verified=False)

        # ---- 1. recover bytes from particle i ----
        recovered = bytearray()
        for p in particles:
            if p.i not in self._inverse:
                return b"", dict(
                    error=f"i={p.i} not in inverse table",
                    verified=False)
            recovered.append(self._inverse[p.i])
        recovered = bytes(recovered[:n_orig])

        # ---- 2. rebuild expected particle cloud ----
        expected_particles = build_particle_cloud(self.table, recovered)

        # ---- 3. particle consistency ----
        particle_match = (
            len(particles) == len(expected_particles) and
            all(abs(p.x - e.x) < 1e-9 and abs(p.y - e.y) < 1e-9
                and p.i == e.i
                for p, e in zip(particles, expected_particles)))

        # ---- 4. UGC consistency ----
        surv_chk, non_chk = ugc_survivors(
            expected_particles, self.K_table, self.table.mu,
            self.k_labels, self.threshold)
        ugc_match = (surv_sent == surv_chk and non_sent == non_chk)

        # ---- 5. ALU consistency ----
        buf_chk = ReadOnlyBuffer(K=self.K_table)
        for p in expected_particles:
            buf_chk.append(struct.pack(">dd", p.x, p.y))
        alu_chk = []
        for cid in range(0, min(8, len(buf_chk.chunks)) - 1, 2):
            for op in ("ADD", "SUB", "MUL", "DIV"):
                alu_chk.append(self.alu.dispatch(buf_chk, cid, cid + 1, op))
        alu_match = (
            len(alu_sent) == len(alu_chk) and
            all(abs(a["z"] - b["z"]) < 1e-6
                for a, b in zip(alu_sent, alu_chk)))

        # ---- 6. fiber consistency ----
        signal_chk = np.array([p.vx for p in expected_particles])
        fiber_chk  = fiber_propagate(signal_chk, width=41)
        fiber_S_chk = self._supertrace(fiber_chk)
        fiber_match = abs(fiber_S_sent - fiber_S_chk) < 1e-9

        verified = all([particle_match, ugc_match, alu_match, fiber_match])

        diagnostics = dict(
            n_particles=len(particles),
            n_survivors=len(surv_sent),
            n_non=len(non_sent),
            n_alu=len(alu_sent),
            survivor_rate=(len(surv_sent)
                           / max(1, len(surv_sent) + len(non_sent))),
            particle_match=particle_match,
            ugc_match=ugc_match,
            alu_match=alu_match,
            fiber_match=fiber_match,
            fiber_S=fiber_S_sent,
            verified=verified,
        )
        return recovered, diagnostics

    # --------------------------------------------------------
    @staticmethod
    def _supertrace(arr: np.ndarray) -> float:
        S = 0.0
        for i, v in enumerate(arr):
            S += float(v) if (i % 2 == 0) else -float(v)
        return S

    # --------------------------------------------------------
    def _serialize(self, particles, surv, non, alu, fiber_S, n_original
                   ) -> bytes:
        out = bytearray()
        # header
        out += struct.pack(">IIIdI",
                           len(particles), self.K_table,
                           self.k_labels, self.threshold, n_original)
        # particles
        for p in particles:
            out += struct.pack(">iidddd",
                               int(p.i), int(p.j),
                               float(p.x), float(p.y),
                               float(p.vx), float(p.zt))
        # UGC survivors
        out += struct.pack(">I", len(surv))
        for s in surv:
            out += struct.pack(">I", int(s))
        # UGC non-survivors
        out += struct.pack(">I", len(non))
        for s in non:
            out += struct.pack(">I", int(s))
        # ALU results
        out += struct.pack(">I", len(alu))
        for r in alu:
            out += struct.pack(">Bd",
                               self._OP_IDS[r["op"]], float(r["z"]))
        # fiber signature
        out += struct.pack(">d", float(fiber_S))
        return bytes(out)

    # --------------------------------------------------------
    def _deserialize(self, packet: bytes):
        off = 0
        n_part, K_tab, k_lab, thr, n_orig = struct.unpack(
            ">IIIdI", packet[off:off + 24])
        off += 24
        if K_tab != self.K_table:
            raise ValueError(f"K_table mismatch: {K_tab} != {self.K_table}")
        if k_lab != self.k_labels or abs(thr - self.threshold) > 1e-9:
            raise ValueError("codec parameters mismatch")

        particles: List[DiscreteParticle] = []
        for _ in range(n_part):
            i, j, x, y, vx, zt = struct.unpack(
                ">iidddd", packet[off:off + 40])
            off += 40
            particles.append(DiscreteParticle(
                i=i, j=j, x=x, y=y, vx=vx, zt=zt))

        n_surv = struct.unpack(">I", packet[off:off + 4])[0]; off += 4
        surv = [struct.unpack(">I", packet[off + 4*k:off + 4*k + 4])[0]
                for k in range(n_surv)]
        off += 4 * n_surv

        n_non = struct.unpack(">I", packet[off:off + 4])[0]; off += 4
        non = [struct.unpack(">I", packet[off + 4*k:off + 4*k + 4])[0]
               for k in range(n_non)]
        off += 4 * n_non

        n_alu = struct.unpack(">I", packet[off:off + 4])[0]; off += 4
        alu = []
        for _ in range(n_alu):
            op_id, z = struct.unpack(">Bd", packet[off:off + 9]); off += 9
            alu.append(dict(op=self._OP_NAMES[op_id], z=z))

        fiber_S = struct.unpack(">d", packet[off:off + 8])[0]; off += 8

        return particles, surv, non, alu, fiber_S, n_orig

    # --------------------------------------------------------
    def summary(self) -> Dict:
        return dict(
            K_table=self.K_table,
            K_alu=self.K_alu,
            k_labels=self.k_labels,
            threshold=self.threshold,
            bijective=len(self._inverse) == 256,
        )


# ============================================================
#  Volume-tiered order buffer  (unchanged)
# ============================================================
class TieredOrderBuffer:
    def __init__(self, tier: VolumeTier):
        self.tier = tier
        self.max_bytes = VOLUME_BUFFER[tier]
        self.buffer = bytearray()
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
#  Trading client  ·  encode via UTF-8 UGC ALU
# ============================================================
class TradingClient:
    """
    Builds orders, encodes them via UTF-8 UGC ALU, wraps in
    TRDB header + MOSB frame.
    """
    def __init__(self, role: Role, entity_id: str,
                 commodity: Commodity,
                 K_table: int = K_TRADING_TABLE,
                 K_alu:   int = K_TRADING_ALU):
        self.role       = role
        self.entity_id  = entity_id
        self.commodity  = commodity
        self.spec       = COMMODITY_SPECS[commodity]
        self.codec      = UTF8TradeCodec(K_table=K_table, K_alu=K_alu)
        self.packer     = MobiusPacker(K=K_MOSB)

    # --------------------------------------------------------
    def submit_order(self, side: str, price: float, qty: float,
                     dst_role: Role, dst_entity: str) -> Dict:
        # 1. build order
        order_id = hashlib.sha256(
            (self.entity_id + str(time.time()) + side
             + str(os.urandom(4))).encode()
        ).hexdigest()[:16]
        nonce = os.urandom(12)
        order = Order(
            order_id=order_id, role=self.role,
            entity=self.entity_id, commodity=self.commodity,
            side=side, price=price, qty=qty,
            timestamp=time.time(), counterparty=dst_entity,
        )
        order_bytes = order.to_json()

        # 2. encode via UTF-8 UGC ALU
        packet = self.codec.encode(order_bytes)

        # 3. classify volume tier
        tier = classify_volume(qty)

        # 4. build trading header  (env_size now = packet length)
        header = TradingHeader(
            role=self.role, commodity=self.commodity,
            volume_tier=tier, family_code=self.spec.family_code,
            price=price, qty=qty, timestamp=order.timestamp,
            nonce=nonce, env_size=len(packet),
            msg_len=len(order_bytes),
        )

        # 5. payload = TRDB header + UTF-8 UGC ALU packet
        payload = header.pack() + packet

        # 6. wrap in MOSB frame
        frame, S, H, m = self.packer.pack(payload)

        return dict(
            order=order, frame=frame, header=header, tier=tier,
            packet=packet, mosb_S=S, mosb_H=H, mosb_m=m,
        )


# ============================================================
#  Trading server  ·  decode via UTF-8 UGC ALU + Stage 1
# ============================================================
class TradingServer:
    """
    Receives MOSB trading frames, unpacks through tiered buffers,
    decodes the UTF-8 UGC ALU packet, verifies it, and runs
    Stage 1 deterministic checks.
    """
    def __init__(self,
                 K_table: int = K_TRADING_TABLE,
                 K_alu:   int = K_TRADING_ALU,
                 max_order_age_s: float = 60.0):
        self.K_table = K_table
        self.K_alu   = K_alu
        self.max_age = max_order_age_s

        self.codec  = UTF8TradeCodec(K_table=K_table, K_alu=K_alu)
        self.packer = MobiusPacker(K=K_MOSB)
        self.mu     = tx_sieve(K_table)

        self.accounts: Dict[str, Dict] = {}
        self.seen_nonces: set = set()
        self.volume_buffers: Dict[VolumeTier, TieredOrderBuffer] = {
            t: TieredOrderBuffer(t) for t in VolumeTier
        }
        self.matched_orders: List[Order] = []

    # --------------------------------------------------------
    def register_account(self, entity_id: str, role: Role,
                         commodity: Commodity,
                         balance: float, credit_limit: float = 0.0):
        address_bytes = (entity_id + "|" + role.name).encode("utf-8")
        env = build_sat_envelope(
            self.K_table, self.mu,
            address_bytes=address_bytes,
            frozen=False,
            expiry_ts=time.time() + 86400 * 365,
            nonce=hashlib.sha256(entity_id.encode()).digest()[:16],
            amount=balance,
        )
        self.accounts[entity_id] = dict(
            role=role, commodity=commodity,
            balance=balance, credit_limit=credit_limit,
            frozen=False, envelope=env,
        )

    # --------------------------------------------------------
    def process_frame(self, frame: bytes) -> Dict:
        # ---- 1. MOSB deframe ----
        try:
            payload, S, H, m, cid = self.packer.unpack(frame)
        except ValueError as e:
            return dict(approved=False, stage=0, reason=f"deframe: {e}")

        # ---- 2. TRDB header ----
        if len(payload) < TRADING_SIZE:
            return dict(approved=False, stage=0,
                        reason="payload too short for TRDB header")
        header = TradingHeader.unpack(payload[:TRADING_SIZE])
        packet = payload[TRADING_SIZE:TRADING_SIZE + header.env_size]
        if len(packet) != header.env_size:
            return dict(approved=False, stage=0,
                        reason="packet byte count mismatch",
                        header=header)

        # ---- 3. volume-tiered buffering ----
        buffer = self.volume_buffers[header.volume_tier]
        buffer.add_frame(frame)

        # ---- 4. decode UTF-8 UGC ALU packet ----
        recovered, diag = self.codec.decode(packet)
        if not diag.get("verified", False):
            return dict(
                approved=False, stage="ugc",
                reason=f"UTF-8 UGC verification failed: {diag}",
                header=header, diagnostics=diag,
                buffer_fill=buffer.fill_ratio(),
            )

        # ---- 5. parse Order ----
        try:
            order = Order.from_json(recovered)
        except Exception as e:
            return dict(approved=False, stage="ugc",
                        reason=f"order parse: {e}",
                        header=header, diagnostics=diag)

        # ---- 6. STAGE 1  ·  deterministic ----
        s1_ok, s1_reason = self._stage1(order, header)
        if not s1_ok:
            return dict(
                approved=False, stage=1, reason=s1_reason,
                header=header, order=order,
                diagnostics=diag,
                mosb_S=S, mosb_H=H, mosb_m=m,
                buffer_fill=buffer.fill_ratio(),
            )

        # ---- 7. APPROVE ----
        self.matched_orders.append(order)
        return dict(
            approved=True, stage="ok",
            reason=(f"UGC verified "
                    f"(particle={diag['particle_match']}, "
                    f"ugc={diag['ugc_match']}, "
                    f"alu={diag['alu_match']}, "
                    f"fiber={diag['fiber_match']})"),
            header=header, order=order,
            diagnostics=diag,
            tier=header.volume_tier,
            mosb_S=S, mosb_H=H, mosb_m=m,
            buffer_fill=buffer.fill_ratio(),
        )

    # --------------------------------------------------------
    def _stage1(self, order: Order, header: TradingHeader
                 ) -> Tuple[bool, str]:
        if not role_allowed(order.role, Role.BROKER):
            return False, f"role {order.role.name} cannot send orders"

        rec = self.accounts.get(order.entity)
        if rec is None:
            return False, f"unknown entity: {order.entity}"
        if rec["commodity"] != order.commodity:
            return False, (
                f"commodity mismatch: account={rec['commodity'].name} "
                f"order={order.commodity.name}")
        if rec["frozen"]:
            return False, "account frozen"

        nonce_hex = header.nonce.hex()
        if nonce_hex in self.seen_nonces:
            return False, "replay detected (nonce reused)"
        self.seen_nonces.add(nonce_hex)

        age = time.time() - header.timestamp
        if age > self.max_age:
            return False, f"stale order ({age:.1f} s old)"

        spec = COMMODITY_SPECS[order.commodity]
        if order.price <= 0:
            return False, "non-positive price"
        if order.qty <= 0 or order.qty > spec.max_qty:
            return False, (
                f"qty {order.qty} out of range (max {spec.max_qty})")
        if abs(order.qty % spec.lot_size) > 1e-6:
            return False, (
                f"qty {order.qty} not a multiple of lot {spec.lot_size}")

        notional = order.price * order.qty
        available = rec["balance"] + rec["credit_limit"]
        if notional > available:
            return False, (
                f"insufficient funds: notional={notional:.2f} "
                f"> available={available:.2f}")

        if order.side not in ("BUY", "SELL"):
            return False, f"invalid side: {order.side!r}"

        return True, "ok"


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius trading platform  ·  UTF-8 UGC ALU encoding")
    print("=" * 78)
    print(f"  α              = 1/(π − e)  = {ALPHA:.6f}")
    print(f"  a_step         = α/0.3628   = {A_STEP:.6f}")
    print(f"  6/π²                        = {DENSITY:.6f}")
    print(f"  τ (UGC)                     = 0.3")
    print(f"  K_table                     = {K_TRADING_TABLE}")
    print(f"  K_alu                       = {K_TRADING_ALU}")
    print()

    # ============================================================
    #  UTF-8 UGC ALU codec: direct round-trip
    # ============================================================
    print("=" * 78)
    print("1  ·  UTF-8 UGC ALU codec  ·  round-trip test")
    print("=" * 78)
    codec = UTF8TradeCodec()
    s = codec.summary()
    print(f"  bijective byte ↔ index      : {s['bijective']}")
    print(f"  k_labels                    : {s['k_labels']}")
    print(f"  threshold                   : {s['threshold']}")
    print()

    sample = b"BUY WTI 500 @ 78.45 from investor_alice to broker_morgan"
    packet = codec.encode(sample)
    recovered, diag = codec.decode(packet)
    ok = (recovered == sample and diag["verified"])
    print(f"  original   : {len(sample)} bytes")
    print(f"  packet     : {len(packet)} bytes  "
          f"({len(packet)/len(sample):.2f}×)")
    print(f"  recovered  : {len(recovered)} bytes  "
          f"match={'OK' if recovered == sample else 'FAIL'}")
    print(f"  n_particles: {diag['n_particles']}")
    print(f"  n_survivors: {diag['n_survivors']}  "
          f"rate={diag['survivor_rate']:.4f}")
    print(f"  ALU checks : {diag['n_alu']}")
    print(f"  fiber S    : {diag['fiber_S']:+.6f}")
    print(f"  verified   : {diag['verified']}  "
          f"(particle={diag['particle_match']}, "
          f"ugc={diag['ugc_match']}, "
          f"alu={diag['alu_match']}, "
          f"fiber={diag['fiber_match']})")
    print(f"  round-trip : {'PASS' if ok else 'FAIL'}")
    print()

    # ---- tamper detection ----
    print("=" * 78)
    print("2  ·  tamper detection on the UTF-8 UGC packet")
    print("=" * 78)
    tampered = bytearray(packet)
    tampered[40] ^= 0x01          # flip one byte in the particle data
    r_t, d_t = codec.decode(bytes(tampered))
    print(f"  untampered : verified={diag['verified']}  "
          f"bytes_match={recovered == sample}")
    print(f"  tampered   : verified={d_t.get('verified', False)}  "
          f"particle_match={d_t.get('particle_match')}  "
          f"ugc_match={d_t.get('ugc_match')}  "
          f"alu_match={d_t.get('alu_match')}  "
          f"fiber_match={d_t.get('fiber_match')}")
    print()

    # ============================================================
    #  Volume tier buffers
    # ============================================================
    print("=" * 78)
    print("3  ·  volume-tier buffers (parse_size)")
    print("=" * 78)
    for tier, size in VOLUME_BUFFER.items():
        print(f"  {tier.name:<14s}  {size:>13,} bytes  "
              f"({size / (1 << 20):.4f} MiB)")
    print()

    # ============================================================
    #  Register accounts + trading pipeline
    # ============================================================
    server = TradingServer()
    print("=" * 78)
    print("4  ·  register accounts")
    print("=" * 78)
    server.register_account("investor_alice", Role.INVESTOR,
                            Commodity.WTI,     balance=1_000_000)
    server.register_account("investor_bob",   Role.INVESTOR,
                            Commodity.GOLD,    balance=500_000)
    server.register_account("investor_carol", Role.INVESTOR,
                            Commodity.SP500,   balance=2_000_000)
    server.register_account("broker_morgan",  Role.BROKER,
                            Commodity.WTI,     balance=10_000_000)
    server.register_account("broker_gs",      Role.BROKER,
                            Commodity.GOLD,    balance=20_000_000)
    server.register_account("exchange_cme",   Role.EXCHANGE,
                            Commodity.WTI,     balance=100_000_000)
    for eid, rec in server.accounts.items():
        print(f"  {eid:<16s}  {Role(rec['role']).name:<10s}  "
              f"{Commodity(rec['commodity']).name:<7s}  "
              f"bal={rec['balance']:>15,.0f}")
    print()

    alice  = TradingClient(Role.INVESTOR, "investor_alice", Commodity.WTI)
    bob    = TradingClient(Role.INVESTOR, "investor_bob",   Commodity.GOLD)
    carol  = TradingClient(Role.INVESTOR, "investor_carol", Commodity.SP500)
    morgan = TradingClient(Role.BROKER,   "broker_morgan",  Commodity.WTI)

    print("=" * 78)
    print("5  ·  TRANSACTIONS  (encode → MOSB → decode → Stage 1)")
    print("=" * 78)
    print(f"  {'#':>2}  {'entity':<16s}  {'side':<4s}  "
          f"{'commodity':<7s}  {'qty':>10s}  {'price':>10s}  "
          f"{'tier':>13s}  {'result':<9s}  notes")
    print("  " + "-" * 104)

    cases = [
        (alice,  "BUY",  78.45,    500,  Role.BROKER, "broker_morgan"),
        (alice,  "BUY",  78.50,   5000,  Role.BROKER, "broker_morgan"),
        (alice,  "SELL", 78.55, 500_000, Role.BROKER, "broker_morgan"),
        (bob,    "BUY",  1950.0,    50,  Role.BROKER, "broker_gs"),
        (bob,    "BUY",  1955.0,   500,  Role.BROKER, "broker_gs"),
        (carol,  "BUY",  4500.5,   100,  Role.BROKER, "broker_gs"),
        (carol,  "BUY",  4500.0, 20_000, Role.BROKER, "broker_gs"),
        (morgan, "BUY",  78.40,  100_000,Role.EXCHANGE, "exchange_cme"),
    ]

    for k, (client, side, price, qty, dst_role, dst_entity) in enumerate(cases):
        sub = client.submit_order(
            side=side, price=price, qty=qty,
            dst_role=dst_role, dst_entity=dst_entity)
        r = server.process_frame(sub["frame"])
        outcome = "APPROVE" if r["approved"] else "DECLINE"
        print(f"  {k+1:>2d}  {client.entity_id:<16s}  {side:<4s}  "
              f"{Commodity(client.commodity).name:<7s}  "
              f"{qty:>10.2f}  {price:>10.4f}  "
              f"{sub['tier'].name:>13s}  {outcome:<9s}  "
              f"{r['reason'][:60]}")
    print()

    # ---- MOSB buffer state ----
    print("=" * 78)
    print("6  ·  VOLUME-TIER BUFFER STATE")
    print("=" * 78)
    print(f"  {'tier':<14s}  {'queued':>8s}  {'bytes buffered':>16s}  "
          f"{'fill %':>8s}")
    print("  " + "-" * 54)
    for tier, buf in server.volume_buffers.items():
        print(f"  {tier.name:<14s}  {buf.frames_queued:>8d}  "
              f"{len(buf.buffer):>16,d}  "
              f"{100 * buf.fill_ratio():>7.2f}%")
    print()

    # ---- matched orders breakdown ----
    print("=" * 78)
    print("7  ·  MATCHED ORDERS BY COMMODITY")
    print("=" * 78)
    by_comm: Dict[Commodity, int] = {}
    by_side: Dict[str, int] = {"BUY": 0, "SELL": 0}
    by_tier: Dict[VolumeTier, int] = {t: 0 for t in VolumeTier}
    for o in server.matched_orders:
        by_comm[o.commodity] = by_comm.get(o.commodity, 0) + 1
        by_side[o.side] = by_side.get(o.side, 0) + 1
        by_tier[classify_volume(o.qty)] += 1
    for comm, count in sorted(by_comm.items()):
        print(f"  {Commodity(comm).name:<7s}  {count} order(s)")
    print()
    print(f"  BUY orders     : {by_side['BUY']}")
    print(f"  SELL orders    : {by_side['SELL']}")
    print(f"  RETAIL         : {by_tier[VolumeTier.RETAIL]}")
    print(f"  STANDARD       : {by_tier[VolumeTier.STANDARD]}")
    print(f"  INSTITUTIONAL  : {by_tier[VolumeTier.INSTITUTIONAL]}")
    print(f"  WHOLESALE      : {by_tier[VolumeTier.WHOLESALE]}")
    print()

    # ---- tamper in a real trading frame ----
    print("=" * 78)
    print("8  ·  TAMPER DETECTION  (real trading frame)")
    print("=" * 78)
    sub = alice.submit_order("BUY", 78.50, 500,
                              Role.BROKER, "broker_morgan")
    frame_good = sub["frame"]

    # flip one byte inside the UTF-8 UGC packet region
    # (TRADING_SIZE bytes header + 48 byte MOSB header)
    frame_bad = bytearray(frame_good)
    frame_bad[48 + TRADING_SIZE + 40] ^= 0x01
    frame_bad = bytes(frame_bad)

    r_good = server.process_frame(frame_good)
    r_bad  = server.process_frame(frame_bad)
    print(f"  untampered frame  : approved={r_good['approved']}  "
          f"reason={r_good['reason'][:52]}")
    print(f"  tampered   frame  : approved={r_bad['approved']}  "
          f"reason={r_bad['reason'][:52]}")
    print()

    # ---- complexity ----
    print("=" * 78)
    print("9  ·  COMPLEXITY")
    print("=" * 78)
    print("  UTF-8 encode           O(K_table log K_table + n_bytes)")
    print("    particle cloud       O(n_bytes)")
    print("    UGC gate             O(n_bytes)")
    print("    ALU over buffer      O(n_bytes · K_alu)")
    print("    fiber kernel         O(n_bytes log n_bytes)")
    print("  UTF-8 decode + verify  same")
    print()
    print("  MOSB pack / unpack     O(packet_size)")
    print("  TRDB header            O(1)")
    print("  Stage 1 checks         O(1)")
    print("  Tiered buffer flush    O(batch_size) per tier")
    print()
    print(f"  K_table = {K_TRADING_TABLE} gives bijective byte ↔ index.")
    print(f"  Per-order time ≈ O(n_bytes · K_alu) with K_alu = {K_TRADING_ALU}.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()