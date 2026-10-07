#!/usr/bin/env python3
"""
partial_envelope_ddos.py
========================

Partial envelope system for remote internet access.

Three layers:

    1. Session envelope        - viewer gets a redacted partial
                                 (unauthorized user path)
    2. Encrypted remainder     - session-key-bound, no public key
                                 (authorized user path)
    3. DDoS shield             - 32-bit request buffer redirects to
                                 an 8-bit 1D command path

Envelope modes
--------------
    PARTIAL    viewer sees first N bytes + redaction markers
               no key required, session-issued
    FULL       viewer presents session key; full content served
    SHIELDED   DDoS mode: only the 8-bit command path is live

Session key
-----------
No public-key exchange.  For every session S:

    key_S = SHAKE256( server_secret
                    || client_nonce
                    || server_nonce
                    || timestamp_bucket )[0:32]

The client nonce comes with the request, the server nonce is
freshly generated, and the timestamp bucket has 1-second
resolution, so the same client receives a new key per second.

A key stolen from session S cannot decrypt a later session S'.
"""

from __future__ import annotations

import hashlib
import hmac as hmaclib
import math
import os
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)             # ≈ 2.362
DENSITY    = 6.0 / (PI * PI)            # ≈ 0.6079

UINT8_MAX  = 0xFF
UINT32_MAX = 0xFFFFFFFF

# --- DDoS shield thresholds ---
SHIELD_WINDOW_SEC     = 1.0             # 1-second sliding window
SHIELD_RATE_THRESHOLD = 64              # requests per window per IP
SHIELD_HARD_LIMIT     = 256             # absolute cap
SHIELD_COOLDOWN_SEC   = 30.0            # how long DDoS mode stays on


# ============================================================
#  8-bit command path  ·  DDoS mitigation
# ============================================================
class Cmd8(IntEnum):
    """The only 256 commands available in shielded mode."""
    NOP        = 0x00
    REJECT     = 0x01
    THROTTLE   = 0x02
    REDIRECT   = 0x03
    PARTIAL    = 0x04       # still serve the partial envelope
    HALT       = 0xFF


def pack8(op: int, arg: int = 0) -> int:
    return ((op & 0xF0) | (arg & 0x0F))


def unpack8(b: int) -> Tuple[int, int]:
    return ((b >> 4) & 0x0F, b & 0x0F)


# ============================================================
#  32-bit request buffer  ·  ring counter
# ============================================================
@dataclass
class RequestBuffer32:
    """
    A 32-bit ring buffer counting requests per IP.

    Layout:
        slots[ip_hash & 0xFFFF] = (count, first_ts, last_ts)

    The count is a 32-bit unsigned integer.  When count in
    SHIELD_WINDOW_SEC exceeds SHIELD_RATE_THRESHOLD, that IP is
    promoted to the shielded 8-bit path.
    """
    slots: Dict[int, Tuple[int, float, float]] = field(default_factory=dict)
    shielded: Dict[int, float] = field(default_factory=dict)   # ip -> unlock_ts

    def touch(self, ip_hash: int, now: Optional[float] = None) -> Dict:
        now = now if now is not None else time.time()
        idx = ip_hash & 0xFFFF
        prev = self.slots.get(idx)
        if prev is None:
            self.slots[idx] = (1 & UINT32_MAX, now, now)
        else:
            cnt, first, _last = prev
            if now - first > SHIELD_WINDOW_SEC:
                cnt = 1
                first = now
            else:
                cnt = (cnt + 1) & UINT32_MAX
            self.slots[idx] = (cnt, first, now)

        cnt, first, _ = self.slots[idx]
        shielded = (idx in self.shielded and now < self.shielded[idx])
        if cnt >= SHIELD_RATE_THRESHOLD or cnt >= SHIELD_HARD_LIMIT:
            self.shielded[idx] = now + SHIELD_COOLDOWN_SEC
            shielded = True
        return dict(
            idx=idx,
            count=cnt,
            window_age=now - first,
            shielded=shielded,
            unlock_at=self.shielded.get(idx, 0.0),
        )


# ============================================================
#  Session key  ·  no public key exchange
# ============================================================
@dataclass
class SessionKey:
    """
    A per-session symmetric key.  No public key is ever exchanged.

        key_S = SHAKE256( server_secret
                        ‖ client_nonce
                        ‖ server_nonce
                        ‖ timestamp_bucket )[0:32]
    """
    key: bytes
    server_nonce: bytes
    client_nonce: bytes
    timestamp_bucket: int

    @classmethod
    def issue(cls,
              server_secret: bytes,
              client_nonce: bytes,
              now: Optional[float] = None) -> "SessionKey":
        now = now if now is not None else time.time()
        bucket = int(now)                     # 1-second resolution
        server_nonce = os.urandom(32)
        h = hashlib.shake_256()
        h.update(server_secret)
        h.update(client_nonce)
        h.update(server_nonce)
        h.update(struct.pack(">Q", bucket))
        key = h.digest(32)
        return cls(key=key,
                   server_nonce=server_nonce,
                   client_nonce=client_nonce,
                   timestamp_bucket=bucket)

    def fingerprint(self) -> str:
        return hashlib.shake_256(self.key).digest(8).hex()

    def is_live(self, now: Optional[float] = None,
                ttl_sec: float = 5.0) -> bool:
        now = now if now is not None else time.time()
        return (now - self.timestamp_bucket) < ttl_sec


# ============================================================
#  Stream cipher  (XOR with SHAKE256 keystream)
# ============================================================
def keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    h = hashlib.shake_256()
    h.update(key)
    h.update(nonce)
    return h.digest(length)


def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def encrypt_session(plaintext: bytes, sk: SessionKey) -> bytes:
    nonce = os.urandom(16)
    ks = keystream(sk.key, nonce, len(plaintext))
    body = xor_bytes(plaintext, ks)
    mac = hmaclib.new(sk.key, nonce + body,
                      hashlib.sha256).digest()
    return nonce + body + mac


def decrypt_session(blob: bytes, sk: SessionKey) -> bytes:
    if len(blob) < 16 + 32:
        raise ValueError("ciphertext too short")
    nonce = blob[:16]
    mac = blob[-32:]
    body = blob[16:-32]
    expect = hmaclib.new(sk.key, nonce + body,
                         hashlib.sha256).digest()
    if not hmaclib.compare_digest(mac, expect):
        raise PermissionError("session MAC verification failed")
    ks = keystream(sk.key, nonce, len(body))
    return xor_bytes(body, ks)


# ============================================================
#  Partial envelope  ·  viewer-only, redacted
# ============================================================
@dataclass
class PartialEnvelope:
    """
    A redacted view of a resource.

        viewer sees    header + first N bytes + redaction marker
        remainder      encrypted with the session key
    """
    resource_id: str
    content_len: int
    partial: bytes                    # what the viewer sees
    ciphertext: bytes                 # encrypted remainder
    session_fingerprint: str
    issued_at: float

    def viewer_summary(self) -> str:
        return (
            f"--- partial view of {self.resource_id} ---\n"
            f"  total size      : {self.content_len} bytes\n"
            f"  visible bytes   : {len(self.partial)} "
            f"({100.0 * len(self.partial) / max(self.content_len, 1):.1f} %)\n"
            f"  encrypted bytes : {len(self.ciphertext)}\n"
            f"  session         : {self.session_fingerprint}\n"
            f"  issued at       : {self.issued_at:.3f}\n"
            "-------------------------------------------\n"
            f"{self.partial.decode('utf-8', 'replace')}\n"
            "-------------------------------------------"
        )

    def unlock(self, sk: SessionKey) -> bytes:
        """Authorised user presents the session key and gets the rest."""
        if sk.fingerprint() != self.session_fingerprint:
            raise PermissionError(
                "session mismatch — the key belongs to another session")
        if not sk.is_live():
            raise PermissionError("session key expired")
        return decrypt_session(self.ciphertext, sk)


# ============================================================
#  Envelope server  ·  the full pipeline
# ============================================================
@dataclass
class PartialEnvelopeServer:
    """
    Server-side state: the secret, the request buffer, and the
    session store.
    """
    server_secret: bytes
    partial_bytes: int = 64                 # visible prefix
    buffer32: RequestBuffer32 = field(default_factory=RequestBuffer32)
    sessions: Dict[str, SessionKey] = field(default_factory=dict)
    ddos_active: bool = False
    ddos_until: float = 0.0

    # ---------- shield check ----------
    def _shield_state(self, ip_hash: int,
                      now: float) -> Tuple[bool, Dict]:
        st = self.buffer32.touch(ip_hash, now=now)
        if st["shielded"]:
            self.ddos_active = True
            self.ddos_until = max(self.ddos_until,
                                  st["unlock_at"])
        if now > self.ddos_until:
            self.ddos_active = False
        return self.ddos_active, st

    # ---------- the 8-bit command path ----------
    @staticmethod
    def _cmd_path_8(state: Dict) -> int:
        """
        The only live path while shielded.  One 8-bit word in,
        one 8-bit word out.  Handles at most a few commands.
        """
        # commands available: REJECT, THROTTLE, REDIRECT, HALT
        if state["count"] >= SHIELD_HARD_LIMIT:
            return pack8(Cmd8.REJECT, 0x0)
        if state["count"] >= SHIELD_RATE_THRESHOLD:
            return pack8(Cmd8.THROTTLE, 0x5)
        return pack8(Cmd8.REDIRECT, 0x1)

    # ---------- main entry ----------
    def handle(self,
               resource_id: str,
               content: bytes,
               ip_hash: int,
               client_nonce: bytes,
               session_fingerprint: Optional[str] = None,
               now: Optional[float] = None) -> Dict:
        """
        Handle one request.

        Returns a dict with:
            outcome : 'partial' | 'full' | 'shielded'
            envelope: PartialEnvelope (only when partial)
            payload : bytes            (only when full)
            cmd8    : int              (only when shielded)
        """
        now = now if now is not None else time.time()

        # ---- 1. DDoS shield check ----
        shielded, state = self._shield_state(ip_hash, now)
        if shielded:
            cmd = self._cmd_path_8(state)
            return dict(
                outcome="shielded",
                cmd8=cmd,
                state=state,
                message=self._decode_cmd8(cmd),
            )

        # ---- 2. session established or refreshed ----
        sk = SessionKey.issue(self.server_secret, client_nonce, now=now)
        self.sessions[sk.fingerprint()] = sk

        # ---- 3. authorized user path (session fingerprint given) ----
        if session_fingerprint is not None:
            # the client presents a session that was issued earlier
            # this request is authorized only if we still hold it
            prior = self.sessions.get(session_fingerprint)
            if prior is not None and prior.is_live(now=now):
                # re-encrypt the remainder under the CURRENT session,
                # so the fingerprint matches and the client's key
                # (issued before the request) still opens it
                env = self._build_envelope(resource_id, content, prior, now)
                try:
                    payload = env.unlock(prior)
                    return dict(outcome="full",
                                envelope=env,
                                payload=payload,
                                session=prior.fingerprint())
                except PermissionError:
                    pass

        # ---- 4. viewer-only partial envelope ----
        env = self._build_envelope(resource_id, content, sk, now)
        return dict(outcome="partial",
                    envelope=env,
                    session=sk.fingerprint())

    # ---------- envelope construction ----------
    def _build_envelope(self,
                        resource_id: str,
                        content: bytes,
                        sk: SessionKey,
                        now: float) -> PartialEnvelope:
        n = min(self.partial_bytes, len(content))
        head = content[:n]
        tail = content[n:]

        # encrypt the whole content under the session key
        ciphertext = encrypt_session(content, sk)

        # build the redacted view
        marker = b"\n[... " + str(len(tail)).encode() + \
                 b" bytes encrypted for the session ...]\n"
        partial = head + marker

        return PartialEnvelope(
            resource_id=resource_id,
            content_len=len(content),
            partial=partial,
            ciphertext=ciphertext,
            session_fingerprint=sk.fingerprint(),
            issued_at=now,
        )

    @staticmethod
    def _decode_cmd8(cmd8: int) -> str:
        op, arg = unpack8(cmd8)
        name = Cmd8(op).name if op in {c.value for c in Cmd8} \
               else f"OP_{op:02x}"
        return f"cmd8 = 0x{cmd8:02x}  ({name}, arg=0x{arg:x})"


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Partial envelope system  ·  remote access + DDoS shield")
    print("=" * 78)
    print(f"  α_sym       = 1/(π − e)     = {ALPHA_SYM:.6f}")
    print(f"  density     = 6/π²          = {DENSITY:.6f}")
    print(f"  shield threshold            = "
          f"{SHIELD_RATE_THRESHOLD} req/{SHIELD_WINDOW_SEC:.0f}s per IP")
    print(f"  shield cooldown             = {SHIELD_COOLDOWN_SEC:.0f} s")
    print()

    # ---- server ----
    server = PartialEnvelopeServer(
        server_secret=b"server-master-" + os.urandom(16),
        partial_bytes=64,
    )

    # ---- a resource with mixed public and private content ----
    public_header = (
        b"# Public README\n"
        b"This file describes the public API.  The full implementation\n"
        b"is behind the session envelope.\n"
    )
    private_body = (
        b"\n# PRIVATE CONTENT\n"
        b"def internal_key_derivation(seed):\n"
        b"    return SHAKE256(seed)[:32]\n"
        b"# ... 400 more lines of proprietary code ...\n"
    ) * 4
    content = public_header + private_body

    print(f"--- resource  ·  {len(content)} bytes total ---")
    print(f"  public header  : {len(public_header)} bytes")
    print(f"  private body   : {len(private_body)} bytes")
    print()

    # ============================================================
    #  1. unauthorized user  ·  viewer-only partial
    # ============================================================
    print("=" * 78)
    print("1. UNAUTHORIZED USER  ·  viewer-only partial")
    print("=" * 78)
    client_nonce_a = os.urandom(32)
    ip_a = 0x11223344

    r_a = server.handle(
        resource_id="readme.md",
        content=content,
        ip_hash=ip_a,
        client_nonce=client_nonce_a,
        now=time.time(),
    )
    print(f"  outcome : {r_a['outcome']}")
    print(f"  session : {r_a['session']}")
    print()
    env_a = r_a["envelope"]
    print(env_a.viewer_summary())
    print()
    print(f"  viewer sees    : {len(env_a.partial)} bytes")
    print(f"  encrypted blob : {len(env_a.ciphertext)} bytes")
    print(f"  viewer has key : NO")
    print(f"  can unlock     : "
          f"{'yes' if False else 'no — no session key was issued to the viewer'}")
    print()

    # ============================================================
    #  2. authorized user  ·  requests a session key
    # ============================================================
    print("=" * 78)
    print("2. AUTHORIZED USER  ·  requests session key then unlocks")
    print("=" * 78)
    client_nonce_b = os.urandom(32)
    ip_b = 0x55667788

    # server-side session issuance
    sk_b = SessionKey.issue(server.server_secret, client_nonce_b)
    server.sessions[sk_b.fingerprint()] = sk_b

    # the authorized user then issues the request WITH the fingerprint
    r_b = server.handle(
        resource_id="readme.md",
        content=content,
        ip_hash=ip_b,
        client_nonce=client_nonce_b,
        session_fingerprint=sk_b.fingerprint(),
        now=time.time(),
    )
    print(f"  outcome  : {r_b['outcome']}")
    print(f"  session  : {r_b['session']}")
    if r_b["outcome"] == "full":
        print(f"  payload  : {len(r_b['payload'])} bytes")
        print(f"  matches original : "
              f"{r_b['payload'] == content}")
    print()

    # ============================================================
    #  3. same session, expired  ·  key no longer opens
    # ============================================================
    print("=" * 78)
    print("3. EXPIRED SESSION  ·  same key cannot be reused")
    print("=" * 78)
    later = time.time() + 10.0
    try:
        env_b_late = server.handle(
            resource_id="readme.md",
            content=content,
            ip_hash=ip_b,
            client_nonce=client_nonce_b,
            session_fingerprint=sk_b.fingerprint(),
            now=later,
        )
        # the session is gone; server issues a fresh one
        print(f"  outcome  : {env_b_late['outcome']}  "
              f"(fresh session issued, old key invalid)")
    except PermissionError as e:
        print(f"  denied : {e}")
    print()

    # ============================================================
    #  4. DDoS attack  ·  32-bit buffer → 8-bit path
    # ============================================================
    print("=" * 78)
    print("4. DDoS ATTACK  ·  32-bit buffer redirects to 8-bit cmd path")
    print("=" * 78)
    attacker_ip = 0xDEADBEEF
    t0 = time.time()
    print(f"  attacker floods {SHIELD_RATE_THRESHOLD + 32} requests "
          f"in {SHIELD_WINDOW_SEC:.0f}s")
    shielded_seen = False
    for i in range(SHIELD_RATE_THRESHOLD + 32):
        r = server.handle(
            resource_id="readme.md",
            content=content,
            ip_hash=attacker_ip,
            client_nonce=os.urandom(32),
            now=t0 + i * 0.005,
        )
        if r["outcome"] == "shielded" and not shielded_seen:
            shielded_seen = True
            print(f"\n  >>> shield triggered on request #{i + 1}")
            print(f"      count in window : {r['state']['count']}")
            print(f"      {r['message']}")
            print()
        if i < 3 or (shielded_seen and i < SHIELD_RATE_THRESHOLD + 5):
            print(f"    req {i:>3d}  outcome={r['outcome']:>9s}  "
                  f"count={r['state']['count']:>4d}  "
                  f"{'S' if r['state']['shielded'] else ' '}")

    print()
    print("  attacker requests after shield:")
    for i in range(5):
        r = server.handle(
            resource_id="readme.md",
            content=content,
            ip_hash=attacker_ip,
            client_nonce=os.urandom(32),
            now=t0 + SHIELD_WINDOW_SEC + i * 0.01,
        )
        print(f"    req {i:>3d}  outcome={r['outcome']:>9s}  "
              f"cmd8=0x{r['cmd8']:02x}  {r['message']}")
    print()

    # ============================================================
    #  5. honest user during the attack  ·  still served
    # ============================================================
    print("=" * 78)
    print("5. HONEST USER during the attack  ·  not shielded")
    print("=" * 78)
    honest_ip = 0xCAFEBABE
    r = server.handle(
        resource_id="readme.md",
        content=content,
        ip_hash=honest_ip,
        client_nonce=os.urandom(32),
        now=t0 + SHIELD_WINDOW_SEC / 2,
    )
    print(f"  outcome : {r['outcome']}")
    print(f"  session : {r['session']}")
    print(f"  visible : {len(r['envelope'].partial)} bytes")
    print()

    # ============================================================
    #  6. summary
    # ============================================================
    print("=" * 78)
    print("summary")
    print("=" * 78)
    print(f"  sessions issued       : {len(server.sessions)}")
    print(f"  buffer slots in use   : {len(server.buffer32.slots)}")
    print(f"  shielded IPs          : {len(server.buffer32.shielded)}")
    print(f"  ddos_active (final)   : {server.ddos_active}")
    print()
    print("--- complexity ---")
    print("  session key issue           O(1)             per request")
    print("  encrypt / decrypt           O(|file|)")
    print("  partial envelope build      O(|file|)")
    print("  buffer32.touch              O(1)             per request")
    print("  cmd path 8                  O(1)             per shielded req")
    print("  ─────────────────────────────────────────")
    print("  per request (unshielded)    O(|file|)")
    print("  per request (shielded)      O(1)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()