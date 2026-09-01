#!/usr/bin/env python3
"""
finite_step_packet.py

Packet transmission using finite‑step derivative coding.
Step size: a = 1/(π−e)/0.3628 ≈ 6.511.

Encoder: samples signal at times t = n*a, computes derivative d[n] = (s[n+1]-s[n])/a.
Packet = [s0, d0, d1, ..., d_{N-2}]
Decoder: reconstructs s[n+1] = s[n] + a*d[n].
"""

import math
import numpy as np
import struct
import hashlib

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite derivative step)

def encode_packet(signal):
    """
    Encode a 1D signal into a packet.
    signal: list or numpy array of length N.
    Returns: tuple (packet, metadata)
        packet: list [s0, d0, d1, ..., d_{N-2}]
        metadata: dict with original length, step a, etc.
    """
    N = len(signal)
    if N < 2:
        raise ValueError("Signal must have at least 2 samples.")
    
    s0 = signal[0]
    # Compute finite differences: d[n] = (s[n+1] - s[n]) / a
    d = [(signal[n+1] - signal[n]) / A for n in range(N-1)]
    
    packet = [s0] + d
    metadata = {
        'N': N,
        'step': A,
        'original_signal': signal,   # for verification
        'checksum': basel_checksum_from_signal(signal)
    }
    return packet, metadata

def decode_packet(packet, step=A):
    """
    Reconstruct signal from packet.
    packet: list [s0, d0, d1, ..., d_{N-2}]
    Returns: reconstructed signal (list).
    """
    if len(packet) < 1:
        raise ValueError("Empty packet.")
    s0 = packet[0]
    d = packet[1:]
    recon = [s0]
    for d_val in d:
        recon.append(recon[-1] + step * d_val)
    return recon

def basel_checksum_from_signal(signal):
    """
    Compute a Basel‑style checksum from the signal:
    S = Σ μ(n)/n²? No, we'll use a simple hash of the signal's bytes.
    But for consistency with the Möbius theme, we compute the supertrace.
    """
    # Compute supertrace of the signal
    S = 0.0
    for i, val in enumerate(signal):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    return S

def packet_to_bytes(packet, metadata):
    """
    Convert packet + metadata to bytes for transmission.
    Format: 4 bytes magic, 4 bytes N, 8 bytes step, 8 bytes checksum, then floats.
    """
    N = metadata['N']
    # Use double precision (8 bytes per float)
    fmt = '>I I d d'
    header = struct.pack(fmt, 0x4D4F4249, N, metadata['step'], metadata['checksum'])
    data = struct.pack('>' + 'd' * len(packet), *packet)
    return header + data

def bytes_to_packet(data):
    """
    Convert bytes back to packet and metadata.
    """
    magic, N, step, checksum = struct.unpack('>I I d d', data[:24])
    if magic != 0x4D4F4249:
        raise ValueError("Invalid magic number.")
    packet = list(struct.unpack('>' + 'd' * (N), data[24:24 + 8*N]))
    # N is the number of elements in packet = 1 + (N-1) = N? Actually packet length = N (since d length = N-1, plus s0 -> N)
    # So the number of floats in packet is N.
    metadata = {
        'N': N,
        'step': step,
        'checksum': checksum
    }
    return packet, metadata

def verify_checksum(recon_signal, expected_checksum):
    """Check if the reconstructed signal's checksum matches."""
    S = basel_checksum_from_signal(recon_signal)
    return abs(S - expected_checksum) < 1e-9

# ---------- Demonstration ----------
def demo():
    # Generate a test signal: a sine wave with a trend
    t = np.linspace(0, 50, 200)
    signal = np.sin(0.5 * t) + 0.02 * t

    print("Original signal (first 5):", signal[:5].round(4))

    # Encode
    packet, meta = encode_packet(signal)
    print(f"Packet length: {len(packet)} (original {len(signal)})")

    # Transmit (simulated)
    bytes_data = packet_to_bytes(packet, meta)
    print(f"Packet size: {len(bytes_data)} bytes")

    # Receive and decode
    received_packet, received_meta = bytes_to_packet(bytes_data)
    recon = decode_packet(received_packet, received_meta['step'])

    # Verify checksum
    ok = verify_checksum(recon, received_meta['checksum'])
    print(f"Checksum verification: {'PASS' if ok else 'FAIL'}")

    # Reconstruction error
    error = np.linalg.norm(np.array(signal) - np.array(recon)) / np.linalg.norm(signal)
    print(f"Relative reconstruction error: {error:.2e}")

    # Show first few
    print("Reconstructed (first 5):", recon[:5].round(4))

if __name__ == "__main__":
    demo()