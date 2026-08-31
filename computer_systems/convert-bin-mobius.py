#!/usr/bin/env python3
"""
mobius_file_converter.py

Convert any file to a compressed Möbius memory file (.mobi) and back.
Uses Möbius filtering (square‑free indices), integral kernel convolution,
supertrace, entropy, and compression.

Usage:
  python mobius_file_converter.py compress <input> <output> [--K=256] [--gate=log] [--chunk=4096]
  python mobius_file_converter.py decompress <input> <output>
"""

import math
import sys
import os
import struct
import argparse
import numpy as np
import hashlib

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))
MAGIC = b'MOBI'
VERSION = 1

# ---------- Möbius sieve (O(K)) ----------
def mobius_sieve(K):
    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
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

# ---------- Exponential convolution (two‑pass, O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K, dtype=float)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    b = np.zeros(K, dtype=float)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace and entropy ----------
def supertrace_from_signal(signal):
    S = 0.0
    for idx, val in enumerate(signal):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def mass_from_signal(signal):
    S = supertrace_from_signal(signal)
    H = entropy_from_supertrace(S, len(signal))
    return abs(S) * math.exp(-H)

# ---------- Logic gates ----------
def apply_gate(signal, gate_name):
    if gate_name == 'log':
        return np.log(np.maximum(np.abs(signal), 1e-12))
    elif gate_name == 'exp':
        return np.exp(signal)
    elif gate_name == 'sin':
        return np.sin(signal)
    elif gate_name == 'cos':
        return np.cos(signal)
    elif gate_name == 'sqrt':
        return np.sqrt(np.maximum(signal, 0))
    else:
        return signal

# ---------- Chip compression ----------
def chip_compress(signal, mu, gate_name='log'):
    K = len(signal)
    # Apply gate
    if gate_name:
        signal = apply_gate(signal, gate_name)
    # Convolution
    conv = conv_exp_kernel(signal)
    # Supertrace
    S = supertrace_from_signal(conv)
    H = entropy_from_supertrace(S, K)
    m = mass_from_signal(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    # Keep top M square‑free coefficients
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]
    kept = []
    count = 0
    for idx in idx_sorted:
        n = idx + 1
        if mu[n] != 0:
            kept.append((idx, conv[idx]))
            count += 1
            if count >= M:
                break
    # Reconstruct for error
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / (np.linalg.norm(conv) + 1e-12)
    return kept, S, H, m, error, conv

# ---------- File converter ----------
class MobiusFileConverter:
    def __init__(self, K=256, gate='log', chunk_size=4096):
        self.K = K
        self.gate = gate
        self.chunk_size = chunk_size
        self.mu = mobius_sieve(K)
        self.basel_ref = sum(self.mu[n] / (n*n) for n in range(1, K+1) if self.mu[n] != 0)

    def compress_file(self, input_path, output_path):
        """Compress a file into .mobi format."""
        with open(input_path, 'rb') as f:
            data = f.read()
        orig_size = len(data)

        # Convert bytes to float signal (0..255)
        signal = np.frombuffer(data, dtype=np.uint8).astype(float)

        # Pad to multiple of chunk_size
        pad = (self.chunk_size - (len(signal) % self.chunk_size)) % self.chunk_size
        if pad:
            signal = np.pad(signal, (0, pad), constant_values=0)

        # Reshape into chunks
        num_chunks = len(signal) // self.chunk_size
        chunks = signal.reshape(num_chunks, self.chunk_size)

        # Write header
        header = struct.pack('>4sBBH', MAGIC, VERSION, 0, self.K)
        # We'll write the file in two passes? Better to write header and then data.
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(struct.pack('>Q', orig_size))      # original size
            f.write(struct.pack('>I', self.chunk_size))
            f.write(struct.pack('>I', num_chunks))
            f.write(struct.pack('>I', len(self.gate))) # gate name length
            f.write(self.gate.encode())

            total_kept = 0
            for chunk in chunks:
                kept, S, H, m, error, conv = chip_compress(chunk, self.mu, self.gate)
                M = len(kept)
                total_kept += M
                f.write(struct.pack('>H', M))          # number of kept coefficients
                for idx, val in kept:
                    f.write(struct.pack('>Hd', idx, val))

        comp_size = os.path.getsize(output_path)
        ratio = comp_size / orig_size if orig_size > 0 else 1.0
        print(f"Compressed {orig_size} bytes to {comp_size} bytes (ratio {ratio:.3f})")
        print(f"Total kept coefficients: {total_kept}")

    def decompress_file(self, input_path, output_path):
        """Decompress a .mobi file back to original format (lossy)."""
        with open(input_path, 'rb') as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError("Not a valid Mobius file.")
            version = struct.unpack('>B', f.read(1))[0]
            _ = struct.unpack('>B', f.read(1))[0]  # reserved
            K = struct.unpack('>H', f.read(2))[0]
            self.K = K
            self.mu = mobius_sieve(K)
            orig_size = struct.unpack('>Q', f.read(8))[0]
            chunk_size = struct.unpack('>I', f.read(4))[0]
            num_chunks = struct.unpack('>I', f.read(4))[0]
            gate_len = struct.unpack('>I', f.read(4))[0]
            gate_name = f.read(gate_len).decode()

            # Reconstruct chunks
            signal_recon = np.zeros(num_chunks * chunk_size, dtype=float)
            for c in range(num_chunks):
                M = struct.unpack('>H', f.read(2))[0]
                kept = []
                for _ in range(M):
                    idx, val = struct.unpack('>Hd', f.read(10))
                    kept.append((idx, val))
                # Reconstruct chunk (zero‑padded)
                chunk = np.zeros(chunk_size, dtype=complex)
                for idx, val in kept:
                    chunk[idx] = val
                # Inverse of gate (if any) – not strictly invertible, but we'll just take real part and clip
                chunk_real = np.real(chunk)
                # Clamp to 0..255
                chunk_real = np.clip(chunk_real, 0, 255)
                signal_recon[c*chunk_size:(c+1)*chunk_size] = chunk_real

            # Trim to original size
            signal_recon = signal_recon[:orig_size]
            # Convert to bytes (clamp and round)
            bytes_recon = np.round(np.clip(signal_recon, 0, 255)).astype(np.uint8).tobytes()

            with open(output_path, 'wb') as f:
                f.write(bytes_recon)

        # Compute error (if possible)
        print(f"Decompressed to {output_path} ({orig_size} bytes)")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description='Möbius file converter')
    subparsers = parser.add_subparsers(dest='command', help='compress or decompress')

    compress_parser = subparsers.add_parser('compress')
    compress_parser.add_argument('input', help='Input file to compress')
    compress_parser.add_argument('output', help='Output .mobi file')
    compress_parser.add_argument('--K', type=int, default=256, help='Number of coefficients (power of two)')
    compress_parser.add_argument('--gate', type=str, default='log', help='Logic gate (log, exp, sin, cos, sqrt, none)')
    compress_parser.add_argument('--chunk', type=int, default=4096, help='Chunk size (samples per block)')

    decompress_parser = subparsers.add_parser('decompress')
    decompress_parser.add_argument('input', help='Input .mobi file')
    decompress_parser.add_argument('output', help='Output decompressed file')

    args = parser.parse_args()
    if args.command == 'compress':
        conv = MobiusFileConverter(K=args.K, gate=args.gate, chunk_size=args.chunk)
        conv.compress_file(args.input, args.output)
    elif args.command == 'decompress':
        conv = MobiusFileConverter()  # K will be read from file
        conv.decompress_file(args.input, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()