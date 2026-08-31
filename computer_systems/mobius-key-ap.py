#!/usr/bin/env python3
"""
mobius_file_converter_gui.py

GUI tool to convert any file to .mobi (Möbius compressed) and back.
Uses the key system (compressed μ bits) for integrity.
Provides preview of Möbius bits and SHA‑256 fingerprint.
"""

import math
import sys
import os
import struct
import numpy as np
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))
MAGIC = b'MOBI'
VERSION = 2

# ---------- Möbius functions ----------
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

def compress_mobius(mu):
    K = len(mu) - 1
    if K == 0:
        return struct.pack('>II', 0, 0)
    abs_sum = sum(abs(mu[i]) for i in range(1, K+1))
    bit_len = 2 * K
    num_bytes = (bit_len + 7) // 8
    packed = bytearray(num_bytes)
    for i in range(1, K+1):
        val = mu[i]
        code = 0 if val == 0 else (1 if val == 1 else 2)
        bit_pos = (i - 1) * 2
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        packed[byte_idx] |= (code << (6 - bit_offset))
    header = struct.pack('>II', K, abs_sum)
    return header + bytes(packed)

def decompress_mobius(data):
    if len(data) < 8:
        raise ValueError("Data too short")
    K, abs_sum = struct.unpack('>II', data[:8])
    if K == 0:
        return [0] * (K + 1)
    packed = data[8:]
    mu = [0] * (K + 1)
    for i in range(1, K+1):
        bit_pos = (i - 1) * 2
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        if byte_idx >= len(packed):
            raise ValueError("Insufficient data")
        code = (packed[byte_idx] >> (6 - bit_offset)) & 0b11
        if code == 0:
            mu[i] = 0
        elif code == 1:
            mu[i] = 1
        elif code == 2:
            mu[i] = -1
        else:
            raise ValueError(f"Invalid code {code} at position {i}")
    computed_sum = sum(abs(mu[i]) for i in range(1, K+1))
    if computed_sum != abs_sum:
        raise ValueError(f"Integrity check failed: expected abs_sum={abs_sum}, got {computed_sum}")
    return mu

# ---------- Core compression ----------
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

def chip_compress(signal, mu, gate_name='log'):
    K = len(signal)
    if gate_name:
        signal = apply_gate(signal, gate_name)
    conv = conv_exp_kernel(signal)
    S = supertrace_from_signal(conv)
    H = entropy_from_supertrace(S, K)
    m = mass_from_signal(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
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
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / (np.linalg.norm(conv) + 1e-12)
    return kept, S, H, m, error, conv

# ---------- File converter class ----------
class MobiusFileConverter:
    def __init__(self, K=256, gate='log', chunk_size=4096):
        self.K = K
        self.gate = gate
        self.chunk_size = chunk_size
        self.mu = mobius_sieve(K)
        self.compressed_mu = compress_mobius(self.mu)

    def compress_file(self, input_path, output_path, progress_callback=None):
        with open(input_path, 'rb') as f:
            data = f.read()
        orig_size = len(data)

        signal = np.frombuffer(data, dtype=np.uint8).astype(float)
        pad = (self.chunk_size - (len(signal) % self.chunk_size)) % self.chunk_size
        if pad:
            signal = np.pad(signal, (0, pad), constant_values=0)

        num_chunks = len(signal) // self.chunk_size
        chunks = signal.reshape(num_chunks, self.chunk_size)

        # Build header
        header = struct.pack('>4sBBH', MAGIC, VERSION, 0, self.K)
        header += struct.pack('>Q', orig_size)
        header += struct.pack('>I', self.chunk_size)
        header += struct.pack('>I', num_chunks)
        # Store compressed μ
        header += struct.pack('>H', len(self.compressed_mu))
        header += self.compressed_mu
        # Gate name
        gate_enc = self.gate.encode()
        header += struct.pack('>I', len(gate_enc))
        header += gate_enc

        with open(output_path, 'wb') as f:
            f.write(header)
            total_kept = 0
            for idx_chunk, chunk in enumerate(chunks):
                kept, S, H, m, error, conv = chip_compress(chunk, self.mu, self.gate)
                M = len(kept)
                total_kept += M
                f.write(struct.pack('>H', M))
                for idx, val in kept:
                    f.write(struct.pack('>Hd', idx, val))
                if progress_callback:
                    progress_callback((idx_chunk + 1) / num_chunks)

        comp_size = os.path.getsize(output_path)
        ratio = comp_size / orig_size if orig_size > 0 else 1.0
        return orig_size, comp_size, ratio, total_kept

    def decompress_file(self, input_path, output_path, progress_callback=None):
        with open(input_path, 'rb') as f:
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError("Not a valid Mobius file.")
            version = struct.unpack('>B', f.read(1))[0]
            if version != VERSION:
                raise ValueError(f"Unsupported version: {version}")
            _ = struct.unpack('>B', f.read(1))[0]  # flags
            K = struct.unpack('>H', f.read(2))[0]
            self.K = K
            orig_size = struct.unpack('>Q', f.read(8))[0]
            chunk_size = struct.unpack('>I', f.read(4))[0]
            num_chunks = struct.unpack('>I', f.read(4))[0]
            # Read compressed μ
            mu_len = struct.unpack('>H', f.read(2))[0]
            mu_data = f.read(mu_len)
            self.mu = decompress_mobius(mu_data)
            self.compressed_mu = mu_data
            # Gate name
            gate_len = struct.unpack('>I', f.read(4))[0]
            gate_name = f.read(gate_len).decode()
            self.gate = gate_name

            signal_recon = np.zeros(num_chunks * chunk_size, dtype=float)
            for c in range(num_chunks):
                M = struct.unpack('>H', f.read(2))[0]
                kept = []
                for _ in range(M):
                    idx, val = struct.unpack('>Hd', f.read(10))
                    kept.append((idx, val))
                chunk = np.zeros(chunk_size, dtype=complex)
                for idx, val in kept:
                    chunk[idx] = val
                chunk_real = np.real(chunk)
                chunk_real = np.clip(chunk_real, 0, 255)
                signal_recon[c*chunk_size:(c+1)*chunk_size] = chunk_real
                if progress_callback:
                    progress_callback((c + 1) / num_chunks)

            signal_recon = signal_recon[:orig_size]
            bytes_recon = np.round(np.clip(signal_recon, 0, 255)).astype(np.uint8).tobytes()

            with open(output_path, 'wb') as f:
                f.write(bytes_recon)

        return orig_size

    def get_mobius_bits_hex(self):
        """Return hex string of compressed μ bits (for preview)."""
        return self.compressed_mu.hex()

    def get_file_fingerprint(self, file_path):
        """Return SHA‑256 hex fingerprint of file."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

# ---------- GUI ----------
class MobiusConverterApp:
    def __init__(self, root):
        self.root = root
        root.title("Möbius File Converter")
        root.geometry("700x600")
        self.converter = MobiusFileConverter(K=256, gate='log')

        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.K_val = tk.IntVar(value=256)
        self.gate_val = tk.StringVar(value='log')
        self.chunk_val = tk.IntVar(value=4096)
        self.status = tk.StringVar(value="Ready.")
        self.fingerprint = tk.StringVar(value="")

        # Layout
        tk.Label(root, text="Möbius File Converter", font=('Arial', 16)).pack(pady=10)

        frame_input = tk.Frame(root)
        frame_input.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame_input, text="Input:").pack(side=tk.LEFT)
        tk.Entry(frame_input, textvariable=self.input_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(frame_input, text="Browse", command=self.browse_input).pack(side=tk.RIGHT)

        frame_output = tk.Frame(root)
        frame_output.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame_output, text="Output:").pack(side=tk.LEFT)
        tk.Entry(frame_output, textvariable=self.output_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(frame_output, text="Browse", command=self.browse_output).pack(side=tk.RIGHT)

        tk.Label(root, text="Settings:").pack(anchor=tk.W, padx=10)
        settings_frame = tk.Frame(root)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(settings_frame, text="K (coeffs):").pack(side=tk.LEFT)
        tk.Spinbox(settings_frame, from_=64, to=2048, textvariable=self.K_val, width=6).pack(side=tk.LEFT, padx=5)
        tk.Label(settings_frame, text="Gate:").pack(side=tk.LEFT, padx=(20,0))
        tk.OptionMenu(settings_frame, self.gate_val, 'log', 'exp', 'sin', 'cos', 'sqrt', 'none').pack(side=tk.LEFT, padx=5)
        tk.Label(settings_frame, text="Chunk:").pack(side=tk.LEFT, padx=(20,0))
        tk.Spinbox(settings_frame, from_=512, to=16384, textvariable=self.chunk_val, width=6).pack(side=tk.LEFT, padx=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Compress → .mobi", command=self.do_compress, bg='lightblue', width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Decompress ←", command=self.do_decompress, bg='lightgreen', width=20).pack(side=tk.LEFT, padx=5)

        preview_frame = tk.Frame(root)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(preview_frame, text="Preview (compressed μ bits hex, fingerprint):").pack(anchor=tk.W)
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=6, state=tk.DISABLED)
        self.preview_text.pack(fill=tk.BOTH, expand=True)

        status_frame = tk.Frame(root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(status_frame, textvariable=self.status).pack(side=tk.LEFT)
        tk.Label(status_frame, text="Fingerprint:").pack(side=tk.LEFT, padx=(20,0))
        tk.Entry(status_frame, textvariable=self.fingerprint, width=50, state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.update_preview()

    def browse_input(self):
        f = filedialog.askopenfilename()
        if f:
            self.input_file.set(f)
            # Auto‑set output if empty
            if not self.output_file.get():
                self.output_file.set(os.path.splitext(f)[0] + '.mobi')
            self.update_preview()

    def browse_output(self):
        f = filedialog.asksaveasfilename(defaultextension=".mobi")
        if f:
            self.output_file.set(f)

    def update_preview(self):
        self.preview_text.config(state=tk.NORMAL)
        self.preview_text.delete(1.0, tk.END)
        # Show Mobius bits of current K
        K = self.K_val.get()
        mu = mobius_sieve(K)
        comp = compress_mobius(mu)
        bits_hex = comp.hex()
        self.preview_text.insert(tk.END, f"Compressed μ bits (K={K}):\n{bits_hex}\n\n")
        # If input file is set, show its SHA‑256
        if self.input_file.get() and os.path.exists(self.input_file.get()):
            fp = self.converter.get_file_fingerprint(self.input_file.get())
            self.fingerprint.set(fp[:32] + "...")
            self.preview_text.insert(tk.END, f"Input file SHA‑256:\n{fp}\n")
        else:
            self.fingerprint.set("")
        self.preview_text.config(state=tk.DISABLED)

    def do_compress(self):
        if not self.input_file.get():
            messagebox.showerror("Error", "No input file selected.")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "No output file specified.")
            return
        self.status.set("Compressing...")
        self.root.update()

        # Update converter settings
        self.converter.K = self.K_val.get()
        self.converter.gate = self.gate_val.get()
        self.converter.chunk_size = self.chunk_val.get()
        self.converter.mu = mobius_sieve(self.converter.K)
        self.converter.compressed_mu = compress_mobius(self.converter.mu)

        def task():
            try:
                orig, comp, ratio, kept = self.converter.compress_file(
                    self.input_file.get(), self.output_file.get(),
                    progress_callback=lambda p: self.status.set(f"Compressing... {p*100:.1f}%")
                )
                self.status.set(f"Compressed {orig} → {comp} bytes (ratio {ratio:.3f}), kept {kept} coeffs.")
                # Update preview with the new .mobi file's fingerprint
                fp = self.converter.get_file_fingerprint(self.output_file.get())
                self.fingerprint.set(fp[:32] + "...")
                self.preview_text.config(state=tk.NORMAL)
                self.preview_text.insert(tk.END, f"\nOutput .mobi SHA‑256:\n{fp}\n")
                self.preview_text.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status.set("Error.")
        threading.Thread(target=task, daemon=True).start()

    def do_decompress(self):
        if not self.input_file.get():
            messagebox.showerror("Error", "No input .mobi file selected.")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "No output file specified.")
            return
        self.status.set("Decompressing...")
        self.root.update()

        def task():
            try:
                orig = self.converter.decompress_file(
                    self.input_file.get(), self.output_file.get(),
                    progress_callback=lambda p: self.status.set(f"Decompressing... {p*100:.1f}%")
                )
                self.status.set(f"Decompressed {orig} bytes to {self.output_file.get()}.")
                fp = self.converter.get_file_fingerprint(self.output_file.get())
                self.fingerprint.set(fp[:32] + "...")
                self.preview_text.config(state=tk.NORMAL)
                self.preview_text.insert(tk.END, f"\nDecompressed file SHA‑256:\n{fp}\n")
                self.preview_text.config(state=tk.DISABLED)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status.set("Error.")
        threading.Thread(target=task, daemon=True).start()

# ---------- Main ----------
def main():
    root = tk.Tk()
    app = MobiusConverterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()