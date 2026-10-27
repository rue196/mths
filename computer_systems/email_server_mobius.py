#!/usr/bin/env python3
"""
mobius_email_server.py
======================

Email server over Möbius TCP.  Clock sync removed.  SAT envelope
encryption for message payloads.

Wire protocol (length-prefixed MOSB frames):

    HELLO (0x01) : [u16 id_len][client_id][u16 nonce_len][client_nonce]
    ACK   (0x02) : [32B server_fp][16B server_nonce][32B session_check]
    SEND  (0x10) : [u32 env_size][envelope][12B nonce]
                   [u32 msg_len][8B supertrace]
    STATUS(0x11) : [u8 ok][u16 msg_len][msg]
    LIST  (0x20) : [u16 user_len][user]
    LSTRS (0x21) : [u32 count][per-msg: id, from, subj, ts]
    FETCH (0x30) : [u16 msg_id_len][msg_id]
    FTRSP (0x31) : [u32 env_size=0 if missing][envelope][12B nonce]
                   [u32 msg_len][8B supertrace]
    CLOSE (0x04)
    ERROR (0xFF)

SAT envelope
------------
    Envelope is a float64 array of length K.

        carriers  = { q ∈ [0, K) : μ(q) ≠ 0 }      (≈ 6/π² · K)
        capacity  = 4 · |carriers|                 (4 bytes/carrier)

    Each carrier stores 4 raw bytes as
        float = (int32_be + 1) / 2^32    ∈ (0, 1]
    so that 0.0 remains a structural zero (empty slot).

    Encryption:
        keystream  = SHAKE256(b"DATA"  + key + nonce)
        masked     = plaintext ⊕ keystream
        envelope[q_i] = encode_4bytes(masked[i])
        unused carriers get noise from SHAKE256(b"NOISE" + key + nonce)

    Decryption is the exact inverse; the envelope supertrace
        S = Σ (−1)^i · envelope[i]
    is transmitted alongside as a secondary check.
"""

from __future__ import annotations
import hashlib, hmac, math, os, socket, struct, threading, time, json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Optional imports (graceful fallback)
# ============================================================
try:
    from mobius_internet import MobiusTransport
    _HAS_MOSB = True
except ImportError:
    _HAS_MOSB = False


# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)
DENSITY   = 6.0 / (PI * PI)

K_ENVELOPE = 4096          # envelope slots → ~10 KB capacity per email
K_TRANSPORT = 512
BYTES_PER_CARRIER = 4

MSG_HELLO     = 0x01
MSG_ACK       = 0x02
MSG_SEND      = 0x10
MSG_STATUS    = 0x11
MSG_LIST      = 0x20
MSG_LIST_RSP  = 0x21
MSG_FETCH     = 0x30
MSG_FETCH_RSP = 0x31
MSG_CLOSE     = 0x04
MSG_ERROR     = 0xFF


# ============================================================
#  Möbius sieve  (list — bound check via try/except on index)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    mu = [0] * (K + 1)
    mu[1] = 1
    primes: List[int] = []
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
#  MOSB transport (fallback if mobius_internet missing)
# ============================================================
if not _HAS_MOSB:
    @dataclass
    class _MOSBFrame:
        bytes_: bytes
        size:   int
        S:      float
        H:      float
        m:      float

    class MobiusTransport:
        MAGIC = b"MOSB"
        FMT   = ">4sIQQddd"
        SIZE  = struct.calcsize(FMT)

        def __init__(self, K: int = 512):
            self.K = K
            self._cid = 0

        @staticmethod
        def _supertrace(data: bytes) -> float:
            S = 0.0
            for t, b in enumerate(data):
                v = (b - 128) / 128.0
                S += v if t % 2 == 0 else -v
            return S

        @staticmethod
        def _entropy_mass(S: float, n: int) -> Tuple[float, float]:
            if S == 0 or n <= 0:
                return 0.0, 0.0
            p = min(abs(S) / n, 0.9999)
            H = -ALPHA * p * math.log(p) if p > 0 else 0.0
            return H, abs(S) * math.exp(-H)

        def frame(self, payload: bytes) -> _MOSBFrame:
            n = len(payload)
            S = self._supertrace(payload)
            H, m = self._entropy_mass(S, n)
            cid = self._cid
            self._cid += 1
            hdr = struct.pack(self.FMT, self.MAGIC, self.K, n,
                              cid, S, H, m)
            raw = hdr + payload
            return _MOSBFrame(bytes_=raw, size=len(raw), S=S, H=H, m=m)

        def deframe(self, raw: bytes) -> Tuple[bytes, _MOSBFrame]:
            if len(raw) < self.SIZE:
                raise ValueError("frame too short")
            magic, K, n, cid, S, H, m = struct.unpack(
                self.FMT, raw[:self.SIZE])
            if magic != self.MAGIC:
                raise ValueError(f"bad magic: {magic!r}")
            payload = raw[self.SIZE:self.SIZE + n]
            if abs(self._supertrace(payload) - S) > 1e-9:
                raise ValueError("supertrace mismatch")
            return payload, _MOSBFrame(bytes_=raw, size=len(raw),
                                       S=S, H=H, m=m)


# ============================================================
#  SAT envelope cipher
# ============================================================
class SATEnvelopeCipher:
    """
    Encrypts bytes into a SAT envelope of length K.

        carriers  = { q : μ(q) ≠ 0 }        (square-free)
        4 bytes per carrier (uint32 big-endian)

    Each slot holds (v + 1) / 2^32 ∈ (0, 1], so 0.0 is always a
    structural zero (unused) and every non-zero float decodes
    uniquely back to 4 bytes.
    """
    _MASK = 2 ** 32

    def __init__(self, K: int = K_ENVELOPE):
        self.K = K
        mu = mobius_sieve(K)
        self.carriers = [q for q in range(K) if mu[q] != 0]
        self.capacity = len(self.carriers) * BYTES_PER_CARRIER

    # ---- stream ciphers ----
    @staticmethod
    def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
        return hashlib.shake_256(b"DATA" + key + nonce).digest(length)

    @staticmethod
    def _noise(key: bytes, nonce: bytes, length: int) -> bytes:
        return hashlib.shake_256(b"NOISE" + key + nonce).digest(length)

    # ---- 4-byte <-> float encoding ----
    def _encode_4bytes(self, b: bytes) -> float:
        v = int.from_bytes(b, "big")            # 0 .. 2^32 − 1
        return (v + 1) / self._MASK              # (0, 1]

    def _decode_4bytes(self, f: float) -> bytes:
        v = int(round(float(f) * self._MASK)) - 1
        v = max(0, min(self._MASK - 1, v))
        return v.to_bytes(4, "big")

    # ---- encrypt ----
    def encrypt(self, data: bytes, key: bytes, nonce: bytes) -> np.ndarray:
        n = len(data)
        if n > self.capacity:
            raise ValueError(
                f"data too large: {n} bytes > capacity {self.capacity}")

        ks     = self._keystream(key, nonce, n)
        masked = bytes(a ^ b for a, b in zip(data, ks))

        env = np.zeros(self.K, dtype=np.float64)
        for i in range(n):
            q = self.carriers[i // BYTES_PER_CARRIER]
            off = (i % BYTES_PER_CARRIER) * 8
            cur = int(round(env[q] * self._MASK))
            chunk = int.from_bytes(masked[i:i+1], "big")
            # accumulate 4 bytes per carrier
            # simpler: buffer per carrier
        # Re-do per-carrier accumulation
        env = np.zeros(self.K, dtype=np.float64)
        used_carriers = (n + BYTES_PER_CARRIER - 1) // BYTES_PER_CARRIER
        for c in range(used_carriers):
            block = masked[c * BYTES_PER_CARRIER:
                           (c + 1) * BYTES_PER_CARRIER]
            if len(block) < BYTES_PER_CARRIER:
                block = block + b"\x00" * (BYTES_PER_CARRIER - len(block))
            env[self.carriers[c]] = self._encode_4bytes(block)

        # noise-fill the remaining carriers
        pad = len(self.carriers) - used_carriers
        if pad > 0:
            noise = self._noise(key, nonce, pad * BYTES_PER_CARRIER)
            for k in range(pad):
                block = noise[k * BYTES_PER_CARRIER:
                              (k + 1) * BYTES_PER_CARRIER]
                env[self.carriers[used_carriers + k]] = \
                    self._encode_4bytes(block)

        return env

    # ---- decrypt ----
    def decrypt(self, env: np.ndarray, key: bytes, nonce: bytes,
                data_len: int) -> bytes:
        if data_len > self.capacity:
            raise ValueError(
                f"data_len {data_len} > capacity {self.capacity}")

        used_carriers = (data_len + BYTES_PER_CARRIER - 1) // BYTES_PER_CARRIER
        masked = bytearray()
        for c in range(used_carriers):
            block = self._decode_4bytes(env[self.carriers[c]])
            masked.extend(block)

        masked = bytes(masked[:data_len])
        ks = self._keystream(key, nonce, data_len)
        return bytes(a ^ b for a, b in zip(masked, ks))

    # ---- envelope supertrace (secondary check) ----
    def supertrace(self, env: np.ndarray) -> float:
        S = 0.0
        for i, v in enumerate(env):
            S += v if i % 2 == 0 else -v
        return S

    # ---- diagnostics ----
    def summary(self) -> Dict:
        return dict(
            K=self.K,
            carriers=len(self.carriers),
            capacity_bytes=self.capacity,
            density=len(self.carriers) / self.K,
            target_density=DENSITY,
        )


# ============================================================
#  Email message
# ============================================================
@dataclass
class EmailMessage:
    msg_id:     str
    sender:     str
    recipients: List[str]
    subject:    str
    body:       str
    timestamp:  float = field(default_factory=time.time)

    def serialize(self) -> bytes:
        return json.dumps(
            dict(id=self.msg_id,
                 sender=self.sender,
                 recipients=self.recipients,
                 subject=self.subject,
                 body=self.body,
                 ts=self.timestamp),
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "EmailMessage":
        obj = json.loads(data.decode("utf-8"))
        return cls(
            msg_id=obj["id"],
            sender=obj["sender"],
            recipients=obj["recipients"],
            subject=obj["subject"],
            body=obj["body"],
            timestamp=obj["ts"],
        )


# ============================================================
#  Mailbox
# ============================================================
@dataclass
class Mailbox:
    owner:    str
    messages: List[EmailMessage] = field(default_factory=list)

    def deliver(self, msg: EmailMessage) -> None:
        self.messages.append(msg)

    def metadata(self) -> List[Dict]:
        return [dict(msg_id=m.msg_id,
                     sender=m.sender,
                     recipients=m.recipients,
                     subject=m.subject,
                     timestamp=m.timestamp)
                for m in self.messages]

    def fetch(self, msg_id: str) -> Optional[EmailMessage]:
        for m in self.messages:
            if m.msg_id == msg_id:
                return m
        return None


# ============================================================
#  Session key derivation (no clock)
# ============================================================
def derive_session_key(client_id: bytes,
                       client_nonce: bytes,
                       server_fp: bytes,
                       server_nonce: bytes) -> bytes:
    return hashlib.sha256(
        b"SESSION|"
        + client_id + b"|"
        + client_nonce + b"|"
        + server_fp + b"|"
        + server_nonce
    ).digest()


# ============================================================
#  Connection handler
# ============================================================
class EmailConnection:
    def __init__(self, sock: socket.socket,
                 server: "MobiusEmailServer",
                 verbose: bool = False):
        self.sock = sock
        self.server = server
        self.cipher = server.cipher
        self.transport = server.transport_class(K=K_TRANSPORT)
        self.session_key: Optional[bytes] = None
        self.client_id: Optional[str] = None
        self.verbose = verbose
        self._nonces: set = set()

    def _log(self, *args):
        if self.verbose:
            print("  " + " ".join(str(a) for a in args))

    # ---------- framing ----------
    def _xor(self, key: bytes, nonce: bytes, data: bytes) -> bytes:
        stream = hashlib.shake_256(key + nonce).digest(len(data))
        return bytes(a ^ b for a, b in zip(data, stream))

    def _send_msg(self, msg_type: int, payload: bytes,
                  session: bool = True) -> None:
        if session and self.session_key:
            nonce = os.urandom(12)
            ct = self._xor(self.session_key, nonce, payload)
            mac = hmac.new(self.session_key,
                           nonce + ct, hashlib.sha256).digest()
            inner = (bytes([msg_type]) + nonce
                     + struct.pack(">H", len(mac)) + mac + ct)
        else:
            inner = bytes([msg_type]) + payload
        frame = self.transport.frame(inner)
        self.sock.sendall(
            struct.pack(">Q", len(frame.bytes_)) + frame.bytes_)

    def _recv_msg(self) -> Tuple[int, bytes]:
        n = struct.unpack(">Q", self._recv_exact(8))[0]
        raw = self._recv_exact(n)
        inner, _ = self.transport.deframe(raw)
        msg_type = inner[0]

        if self.session_key and msg_type not in (MSG_HELLO, MSG_ACK):
            nonce = inner[1:13]
            mac_len = struct.unpack(">H", inner[13:15])[0]
            mac = inner[15:15 + mac_len]
            ct = inner[15 + mac_len:]
            expected = hmac.new(self.session_key,
                                nonce + ct, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                raise RuntimeError("MAC verification failed")
            nh = nonce.hex()
            if nh in self._nonces:
                raise RuntimeError(f"replay nonce {nh[:8]}")
            self._nonces.add(nh)
            payload = self._xor(self.session_key, nonce, ct)
            return msg_type, payload
        return msg_type, inner[1:]

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    # ---------- handshake ----------
    def handle_hello(self) -> None:
        msg_type, payload = self._recv_msg()
        if msg_type != MSG_HELLO:
            raise RuntimeError(f"expected HELLO, got {msg_type:#x}")

        off = 0
        id_len = struct.unpack(">H", payload[off:off+2])[0]; off += 2
        client_id = payload[off:off+id_len].decode("utf-8"); off += id_len
        nonce_len = struct.unpack(">H", payload[off:off+2])[0]; off += 2
        client_nonce = payload[off:off+nonce_len]

        self.client_id = client_id

        # server-side nonce + fingerprint
        server_nonce = os.urandom(16)
        server_fp = hashlib.sha256(
            self.server.server_id.encode() + server_nonce).digest()

        self.session_key = derive_session_key(
            client_id.encode("utf-8"), client_nonce,
            server_fp, server_nonce)

        check = hashlib.sha256(self.session_key).digest()
        ack_payload = server_fp + server_nonce + check
        self._send_msg(MSG_ACK, ack_payload, session=False)
        self._log(f"HELLO  client={client_id}  "
                  f"key={self.session_key.hex()[:16]}…")

    # ---------- SEND ----------
    def handle_send(self, payload: bytes) -> bytes:
        off = 0
        env_size = struct.unpack(">I", payload[off:off+4])[0]; off += 4
        env_bytes = payload[off:off+env_size]; off += env_size
        nonce = payload[off:off+12]; off += 12
        msg_len = struct.unpack(">I", payload[off:off+4])[0]; off += 4
        S_expected = struct.unpack(">d", payload[off:off+8])[0]; off += 8

        env = np.frombuffer(env_bytes, dtype=np.float64)
        if env.size != self.cipher.K:
            raise ValueError(f"envelope size mismatch: "
                             f"{env.size} != {self.cipher.K}")

        S_actual = self.cipher.supertrace(env)
        if abs(S_actual - S_expected) > 1e-9:
            raise ValueError(
                f"envelope supertrace mismatch: "
                f"{S_actual} != {S_expected}")

        msg_bytes = self.cipher.decrypt(env, self.session_key,
                                        nonce, msg_len)
        msg = EmailMessage.deserialize(msg_bytes)

        delivered: List[str] = []
        for r in msg.recipients:
            self.server.get_or_create_mailbox(r).deliver(msg)
            delivered.append(r)

        self._log(f"SEND   from={msg.sender}  to={delivered}  "
                  f"subj={msg.subject!r}")

        ok_msg = (f"OK delivered to {len(delivered)} mailbox(es) "
                  f"(id={msg.msg_id})")
        return struct.pack(">BH", 1, len(ok_msg)) + ok_msg.encode("utf-8")

    # ---------- LIST ----------
    def handle_list(self, payload: bytes) -> bytes:
        user_len = struct.unpack(">H", payload[:2])[0]
        user = payload[2:2+user_len].decode("utf-8")

        mb = self.server.mailboxes.get(user)
        md = mb.metadata() if mb else []

        out = struct.pack(">I", len(md))
        for m in md:
            mid = m["msg_id"].encode("utf-8")
            snd = m["sender"].encode("utf-8")
            subj = m["subject"].encode("utf-8")
            out += struct.pack(">H", len(mid)) + mid
            out += struct.pack(">H", len(snd)) + snd
            out += struct.pack(">H", len(subj)) + subj
            out += struct.pack(">d", m["timestamp"])

        self._log(f"LIST   user={user}  count={len(md)}")
        return out

    # ---------- FETCH ----------
    def handle_fetch(self, payload: bytes) -> bytes:
        id_len = struct.unpack(">H", payload[:2])[0]
        msg_id = payload[2:2+id_len].decode("utf-8")

        found = None
        for mb in self.server.mailboxes.values():
            m = mb.fetch(msg_id)
            if m:
                found = m
                break

        if not found:
            return struct.pack(">I", 0)

        msg_bytes = found.serialize()
        nonce = os.urandom(12)
        env = self.cipher.encrypt(msg_bytes, self.session_key, nonce)
        env_bytes = env.tobytes()
        S = self.cipher.supertrace(env)

        out = struct.pack(">I", len(env_bytes)) + env_bytes
        out += nonce
        out += struct.pack(">I", len(msg_bytes))
        out += struct.pack(">d", S)

        self._log(f"FETCH  id={msg_id}  from={found.sender}")
        return out

    # ---------- loop ----------
    def run(self) -> None:
        try:
            self.handle_hello()
            while True:
                try:
                    msg_type, payload = self._recv_msg()
                except ConnectionError:
                    break

                if msg_type == MSG_SEND:
                    self._send_msg(MSG_STATUS, self.handle_send(payload))
                elif msg_type == MSG_LIST:
                    self._send_msg(MSG_LIST_RSP, self.handle_list(payload))
                elif msg_type == MSG_FETCH:
                    self._send_msg(MSG_FETCH_RSP, self.handle_fetch(payload))
                elif msg_type == MSG_CLOSE:
                    break
                else:
                    err = f"unknown message type {msg_type:#x}"
                    self._send_msg(MSG_ERROR,
                                   struct.pack(">H", len(err)) + err.encode())
        except Exception as e:
            self._log(f"error: {e}")
            try:
                err = str(e).encode("utf-8")
                self._send_msg(MSG_ERROR,
                               struct.pack(">H", len(err)) + err)
            except Exception:
                pass
        finally:
            try:
                self.sock.close()
            except Exception:
                pass


# ============================================================
#  Server
# ============================================================
class MobiusEmailServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0,
                 server_id: str = "mobius-mail-01",
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.server_id = server_id
        self.verbose = verbose

        self.cipher = SATEnvelopeCipher(K=K_ENVELOPE)
        self.transport_class = MobiusTransport
        self.mailboxes: Dict[str, Mailbox] = {}

        self._sock: Optional[socket.socket] = None
        self._running = False

    def get_or_create_mailbox(self, user: str) -> Mailbox:
        if user not in self.mailboxes:
            self.mailboxes[user] = Mailbox(owner=user)
        return self.mailboxes[user]

    def start(self) -> int:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        if self.verbose:
            s = self.cipher.summary()
            print(f"--- Email server  {self.host}:{self.port} ---")
            print(f"    server_id     = {self.server_id}")
            print(f"    K_envelope    = {s['K']}")
            print(f"    carriers      = {s['carriers']}")
            print(f"    capacity      = {s['capacity_bytes']} bytes / email")
            print(f"    density       = {s['density']:.4f}  "
                  f"(target 6/π² = {s['target_density']:.4f})")
            print(f"    bytes/carrier = {BYTES_PER_CARRIER}")
        return self.port

    def _accept_loop(self):
        while self._running:
            try:
                sock, addr = self._sock.accept()
            except OSError:
                break
            if self.verbose:
                print(f"--- accept {addr} ---")
            threading.Thread(
                target=self._handle, args=(sock, addr), daemon=True
            ).start()

    def _handle(self, sock, addr):
        conn = EmailConnection(sock, self, verbose=self.verbose)
        conn.run()

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass


# ============================================================
#  Client
# ============================================================
class MobiusEmailClient:
    def __init__(self, host: str, port: int, client_id: str,
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.verbose = verbose

        self.sock: Optional[socket.socket] = None
        self.cipher = SATEnvelopeCipher(K=K_ENVELOPE)
        self.transport = MobiusTransport(K=K_TRANSPORT)
        self.session_key: Optional[bytes] = None
        self._nonces: set = set()

    def _log(self, *args):
        if self.verbose:
            print("  " + " ".join(str(a) for a in args))

    def _xor(self, key: bytes, nonce: bytes, data: bytes) -> bytes:
        stream = hashlib.shake_256(key + nonce).digest(len(data))
        return bytes(a ^ b for a, b in zip(data, stream))

    def _send_msg(self, msg_type: int, payload: bytes,
                  session: bool = True) -> None:
        if session and self.session_key:
            nonce = os.urandom(12)
            ct = self._xor(self.session_key, nonce, payload)
            mac = hmac.new(self.session_key,
                           nonce + ct, hashlib.sha256).digest()
            inner = (bytes([msg_type]) + nonce
                     + struct.pack(">H", len(mac)) + mac + ct)
        else:
            inner = bytes([msg_type]) + payload
        frame = self.transport.frame(inner)
        self.sock.sendall(
            struct.pack(">Q", len(frame.bytes_)) + frame.bytes_)

    def _recv_msg(self) -> Tuple[int, bytes]:
        n = struct.unpack(">Q", self._recv_exact(8))[0]
        raw = self._recv_exact(n)
        inner, _ = self.transport.deframe(raw)
        msg_type = inner[0]

        if self.session_key and msg_type not in (MSG_HELLO, MSG_ACK):
            nonce = inner[1:13]
            mac_len = struct.unpack(">H", inner[13:15])[0]
            mac = inner[15:15 + mac_len]
            ct = inner[15 + mac_len:]
            expected = hmac.new(self.session_key,
                                nonce + ct, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected):
                raise RuntimeError("MAC verification failed")
            payload = self._xor(self.session_key, nonce, ct)
            return msg_type, payload
        return msg_type, inner[1:]

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("socket closed")
            buf += chunk
        return buf

    # ---------- connect ----------
    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

        cid = self.client_id.encode("utf-8")
        client_nonce = os.urandom(16)
        payload = (struct.pack(">H", len(cid)) + cid
                   + struct.pack(">H", len(client_nonce)) + client_nonce)
        self._send_msg(MSG_HELLO, payload, session=False)

        msg_type, response = self._recv_msg()
        if msg_type != MSG_ACK:
            raise RuntimeError(f"expected ACK, got {msg_type:#x}")

        server_fp    = response[:32]
        server_nonce = response[32:48]
        check        = response[48:80]

        self.session_key = derive_session_key(
            cid, client_nonce, server_fp, server_nonce)
        expected = hashlib.sha256(self.session_key).digest()
        if not hmac.compare_digest(check, expected):
            raise RuntimeError("session key check mismatch")

        self._log(f"connected  key={self.session_key.hex()[:16]}…")

    # ---------- send ----------
    def send_email(self, sender: str, recipients: List[str],
                   subject: str, body: str) -> str:
        msg = EmailMessage(
            msg_id=hashlib.sha256(
                (sender + "|".join(recipients) + subject
                 + str(time.time())).encode()
            ).hexdigest()[:16],
            sender=sender, recipients=recipients,
            subject=subject, body=body,
        )
        msg_bytes = msg.serialize()
        if len(msg_bytes) > self.cipher.capacity:
            raise ValueError(
                f"email too large: {len(msg_bytes)} > "
                f"{self.cipher.capacity}")

        nonce = os.urandom(12)
        env = self.cipher.encrypt(msg_bytes, self.session_key, nonce)
        S = self.cipher.supertrace(env)
        env_bytes = env.tobytes()

        payload = (struct.pack(">I", len(env_bytes)) + env_bytes
                   + nonce
                   + struct.pack(">I", len(msg_bytes))
                   + struct.pack(">d", S))

        self._send_msg(MSG_SEND, payload)
        msg_type, response = self._recv_msg()
        if msg_type != MSG_STATUS:
            raise RuntimeError(f"expected STATUS, got {msg_type:#x}")

        ok = response[0]
        msg_len = struct.unpack(">H", response[1:3])[0]
        text = response[3:3+msg_len].decode("utf-8")
        self._log(f"SEND {'OK' if ok else 'FAIL'}: {text}")
        return text

    # ---------- list ----------
    def list_mailbox(self, user: str) -> List[Dict]:
        ub = user.encode("utf-8")
        payload = struct.pack(">H", len(ub)) + ub
        self._send_msg(MSG_LIST, payload)
        msg_type, response = self._recv_msg()
        if msg_type != MSG_LIST_RSP:
            raise RuntimeError(f"expected LIST_RSP, got {msg_type:#x}")

        count = struct.unpack(">I", response[:4])[0]
        off = 4
        results = []
        for _ in range(count):
            l = struct.unpack(">H", response[off:off+2])[0]; off += 2
            mid = response[off:off+l].decode("utf-8"); off += l
            l = struct.unpack(">H", response[off:off+2])[0]; off += 2
            snd = response[off:off+l].decode("utf-8"); off += l
            l = struct.unpack(">H", response[off:off+2])[0]; off += 2
            subj = response[off:off+l].decode("utf-8"); off += l
            ts = struct.unpack(">d", response[off:off+8])[0]; off += 8
            results.append(dict(msg_id=mid, sender=snd,
                                subject=subj, ts=ts))
        return results

    # ---------- fetch ----------
    def fetch_email(self, msg_id: str) -> Optional[EmailMessage]:
        mb = msg_id.encode("utf-8")
        payload = struct.pack(">H", len(mb)) + mb
        self._send_msg(MSG_FETCH, payload)
        msg_type, response = self._recv_msg()
        if msg_type != MSG_FETCH_RSP:
            raise RuntimeError(f"expected FETCH_RSP, got {msg_type:#x}")

        env_size = struct.unpack(">I", response[:4])[0]
        if env_size == 0:
            return None
        off = 4
        env_bytes = response[off:off+env_size]; off += env_size
        nonce     = response[off:off+12]; off += 12
        msg_len   = struct.unpack(">I", response[off:off+4])[0]; off += 4
        S_expected = struct.unpack(">d", response[off:off+8])[0]; off += 8

        env = np.frombuffer(env_bytes, dtype=np.float64)
        S_actual = self.cipher.supertrace(env)
        if abs(S_actual - S_expected) > 1e-9:
            raise RuntimeError("fetched envelope supertrace mismatch")

        msg_bytes = self.cipher.decrypt(env, self.session_key,
                                        nonce, msg_len)
        return EmailMessage.deserialize(msg_bytes)

    def close(self) -> None:
        try:
            self._send_msg(MSG_CLOSE, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius email  ·  SAT envelope encryption  ·  no clock")
    print("=" * 78)
    print(f"  α              = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  6/π²                          = {DENSITY:.6f}")
    print(f"  K_envelope                    = {K_ENVELOPE}")
    print(f"  bytes/carrier                 = {BYTES_PER_CARRIER}")
    print()

    server = MobiusEmailServer(port=0, server_id="mobius-mail-01",
                               verbose=True)
    port = server.start()
    print()

    # ---- sender: alice ----
    alice = MobiusEmailClient("127.0.0.1", port,
                              "alice@mobius.local", verbose=True)
    alice.connect()
    print()

    print("--- alice sends ---")
    result = alice.send_email(
        sender="alice@mobius.local",
        recipients=["bob@mobius.local", "carol@mobius.local"],
        subject="Meeting tomorrow",
        body="Hi Bob and Carol,\n\nCan we meet tomorrow at 10am?\n\n- Alice",
    )
    print(f"  result: {result}")
    print()

    # ---- recipient: bob ----
    print("--- bob checks inbox ---")
    bob = MobiusEmailClient("127.0.0.1", port,
                            "bob@mobius.local", verbose=True)
    bob.connect()
    print()

    inbox = bob.list_mailbox("bob@mobius.local")
    print(f"  bob inbox: {len(inbox)} message(s)")
    for m in inbox:
        print(f"    - {m['subject']!r}  from {m['sender']}  "
              f"ts={m['ts']:.0f}")
    print()

    if inbox:
        msg = bob.fetch_email(inbox[0]["msg_id"])
        if msg:
            print(f"  fetched: {msg.subject!r} from {msg.sender}")
            print(f"  body preview: {msg.body[:60]!r}…")
    print()

    # ---- second send: alice replies to herself ----
    print("--- alice sends to herself ---")
    alice.send_email(
        sender="alice@mobius.local",
        recipients=["alice@mobius.local"],
        subject="Test self-delivery",
        body="Testing SAT envelope round-trip.",
    )
    a_inbox = alice.list_mailbox("alice@mobius.local")
    print(f"  alice inbox: {len(a_inbox)} message(s)")
    print()

    # ---- tamper detection ----
    print("--- tamper detection ---")
    env = alice.cipher.encrypt(b"hello", alice.session_key, b"0"*12)
    S_orig = alice.cipher.supertrace(env)
    env_tampered = env.copy()
    env_tampered[env_tampered != 0][0] += 0.1
    S_tampered = alice.cipher.supertrace(env_tampered)
    print(f"  orig      S = {S_orig:+.12f}")
    print(f"  tampered  S = {S_tampered:+.12f}")
    print(f"  detects   : {abs(S_orig - S_tampered) > 1e-9}")
    print()

    alice.close()
    bob.close()
    time.sleep(0.1)
    server.stop()

    # ---- complexity ----
    print("--- complexity ---")
    print(f"  sieve                  O(K log log K)       once per server")
    print(f"  encrypt                O(capacity)           per email")
    print(f"  decrypt                O(data_len)           per fetch")
    print(f"  MOSB frame + verify    O(n)                  per message")
    print(f"  HMAC-SHA256            O(n)                  per message")
    print()
    print(f"  capacity / email       ≈ {server.cipher.capacity} bytes")
    print(f"  envelope wire size     = {K_ENVELOPE * 8} bytes")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()