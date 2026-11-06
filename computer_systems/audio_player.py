#!/usr/bin/env python3
"""
mobius_audio_player.py
======================

Audio player combining four layers:

    stream    MobiusPacker (inline) + stream_path
              → bytes in, MOSB-framed pack/unpack round-trip, bytes out
    compress  finite_step_audio.MobiusAudioSampler
              → finite-step derivative + Möbius gate + wire throughput
    wire      wire_analog.AnalogWire
              → audio drives electrons on the transmission wire
    spectrum  linear_solver.zeta2 / DualLinearSolver
              → FFT-like spectral ζ(t) = Σ C[i]·exp(i·t·i/α)

Usage
-----
    python mobius_audio_player.py                    # synthetic demo
    python mobius_audio_player.py <audio.bin>        # raw int16 PCM
        [--chunk-size 1024] [--K-stream 512] [--fs 44100]
        [--wire-length 10.0] [--N-electrons 20]
"""

from __future__ import annotations
import argparse
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ---- finite_step_audio: compression layer ----
from finite_step_audio import (
    MobiusAudioSampler,
    SamplingConfig,
    mobius_sieve,
    finite_derivative,
    integrate_signal,
    make_audio,
    snr,
    A_STEP,
    ALPHA_SYM,
    ALPHA_USER,
    DENSITY,
)

# ---- wire_analog: transmission layer ----
from wire_analog import (
    AnalogWire,
    wire_positions,
    travel_lengths,
    coupling_kernel,
    alternating_E_field,
    pi_e,
    PI,
    E,
    ALPHA,
)

# ---- linear_solver: FFT / spectral layer ----
from linear_solver_ternary import (
    DualLinearSolver,
    DualSolution,
    zeta2,
    classify_dual,
    ternary_pack,
    ternary_unpack,
    ALPHA_ODD,
    ALPHA_EVEN,
)



# ============================================================
#  MobiusPacker — inline minimal implementation
# ============================================================
class MobiusPacker:
    """
    Byte-stream packer with Möbius supertrace metadata.

    Each chunk is framed as:

        ┌──────┬──────────┬──────────┬──────────────┐
        │ MOSB │ len (8B) │ cid (8B) │ payload      │
        └──────┴──────────┴──────────┴──────────────┘

    The supertrace S = Σ_even b − Σ_odd b  (over payload bytes)
    drives H (entropy) and m (mass), exactly like the |Ci| pipeline.
    """

    MAGIC = b"MOSB"

    def __init__(self, K: int = 512):
        self.K = int(K)
        self.mu = mobius_sieve(max(K, 256))
        self._cid = 0

    def _meta(self, data: bytes) -> Tuple[float, float, float]:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        if arr.size == 0:
            return 0.0, 0.0, 0.0
        S = float(np.sum(arr[::2]) - np.sum(arr[1::2]))
        p = min(abs(S) / max(arr.size, 1), 0.9999)
        H_ent = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H_ent) if H_ent < 700 else 0.0
        return S, H_ent, m

    def pack(self, data: bytes) -> Tuple[bytes, float, float, float]:
        S, H_ent, m = self._meta(data)
        header = (self.MAGIC
                  + len(data).to_bytes(8, "little")
                  + self._cid.to_bytes(8, "little"))
        self._cid += 1
        return header + data, S, H_ent, m

    def unpack(self, packed: bytes
               ) -> Tuple[bytes, float, float, float, int]:
        if packed[:4] != self.MAGIC:
            raise ValueError(f"bad magic: {packed[:4]!r}")
        length = int.from_bytes(packed[4:12], "little")
        cid    = int.from_bytes(packed[12:20], "little")
        data   = packed[20:20 + length]
        S, H_ent, m = self._meta(data)
        return data, S, H_ent, m, cid


def stream_path(path: str, buf_bytes: int = 65536):
    """Yield (chunk, received, total) — same contract as stream_url."""
    total = os.path.getsize(path)
    received = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf_bytes)
            if not chunk:
                break
            received += len(chunk)
            yield chunk, received, total


# ============================================================
#  Player
# ============================================================
class MobiusAudioPlayer:
    """
    Audio player:
    
        bytes ──[MobiusPacker]── PCM ──[MobiusAudioSampler]── recon
                                    │
                                    ├──[AnalogWire]── wire e-cloud
                                    │
                                    └──[DualLinearSolver / zeta2]── spectrum
    """

    def __init__(self,
                 K_stream:     int = 512,
                 chunk_size:   int = 1024,
                 fs:           int = 44_100,
                 bps:          int = 16,
                 wire_length:  float = 10.0,
                 N_electrons:  int = 20,
                 verbose:      bool = True):
        self.K_stream    = K_stream
        self.chunk_size  = chunk_size
        self.fs          = fs
        self.bps         = bps
        self.verbose     = verbose

        # 1. streaming layer
        self.packer = MobiusPacker(K=K_stream)

        # 2. compression layer
        self.sampler = MobiusAudioSampler(
            SamplingConfig(fs=fs, bps=bps, K=chunk_size,
                           use_mobius=True, use_throughput=True))

        # 3. wire transmission layer
        self.wire = AnalogWire(N=64, length=wire_length,
                               N_electrons=N_electrons)

        # 4. spectrum layer (FFT via ζ)
        self.solver = DualLinearSolver(use_coeff_classify=True,
                                        spectral_samples=8)

        # results
        self.chunks_raw:   List[np.ndarray] = []
        self.chunks_recon: List[np.ndarray] = []
        self.snr_list:     List[float]      = []
        self.stream_info:  List[Dict]       = []
        self.spectra:      List[Dict]       = []
        self.wire_states:  List[Dict]       = []

    # --------------------------------------------------------
    def load_from_path(self,
                       path: str,
                       buf_bytes: int = 8192,
                       max_chunks: Optional[int] = None) -> int:
        """
        Stream a raw int16 PCM file through the pack/unpack/compress
        pipeline and process chunk by chunk.
        """
        n_chunks = 0
        total_in = 0
        total_packed = 0
        buffer = bytearray()
        bytes_per_chunk = self.chunk_size * 2   # int16

        for chunk, received, total in stream_path(path, buf_bytes):
            # 1. pack/unpack round-trip
            packed, S, H_ent, m = self.packer.pack(chunk)
            raw, S2, H2, m2, cid = self.packer.unpack(packed)
            if raw != chunk:
                raise RuntimeError(f"stream mismatch on chunk {cid}")

            buffer.extend(raw)
            total_in     += len(chunk)
            total_packed += len(packed)

            # 2. split into audio frames
            while len(buffer) >= bytes_per_chunk:
                fb = bytes(buffer[:bytes_per_chunk])
                del buffer[:bytes_per_chunk]

                audio = (np.frombuffer(fb, dtype=np.int16)
                         .astype(np.float64) / 32768.0)
                self._process_chunk(audio, S, H_ent, m, cid, n_chunks)
                n_chunks += 1
                if max_chunks and n_chunks >= max_chunks:
                    break
            if max_chunks and n_chunks >= max_chunks:
                break

        if self.verbose:
            print(f"  bytes in        : {total_in:,}")
            print(f"  bytes packed    : {total_packed:,}")
            print(f"  pack ratio      : "
                  f"{total_packed / max(total_in, 1):.6f}")
            print(f"  chunks loaded   : {n_chunks}")
            if self.snr_list:
                print(f"  mean SNR        : "
                      f"{np.mean(self.snr_list):.2f} dB")
        return n_chunks

    # --------------------------------------------------------
    def _process_chunk(self, audio: np.ndarray,
                       S: float, H_ent: float, m: float,
                       cid: int, k: int) -> None:
        # 1. compress + reconstruct (finite_step_audio)
        sampled = self.sampler.sample(audio)
        recon   = self.sampler.reconstruct(sampled)
        s       = snr(audio, recon)

        # 2. wire transmission (wire_analog)
        wire_state = self._wire_transmit(audio)

        # 3. spectral ζ (linear_solver FFT)
        spectrum = self._spectral_analysis(audio)

        # 4. store
        self.chunks_raw.append(audio)
        self.chunks_recon.append(recon)
        self.snr_list.append(s)
        self.stream_info.append(dict(
            S=S, H=H_ent, m=m, cid=cid,
            kept=len(self.sampler.kept),
            K=self.sampler.cfg.K,
        ))
        self.wire_states.append(wire_state)
        self.spectra.append(spectrum)

    # --------------------------------------------------------
    def _wire_transmit(self, audio: np.ndarray) -> Dict:
        """
        Drive the AnalogWire with the chunk's samples.

        The wire's `field_at_time` is temporarily overridden by a
        closure returning the audio samples (×10 gain).  After the
        run, the wire's state is restored so chunks don't compound.
        """
        n_steps = min(len(audio), 64)
        audio_sub = audio[:n_steps] * 10.0

        # save caller-visible state
        saved = (self.wire.e_pos.copy(),
                 self.wire.e_vel.copy(),
                 list(self.wire.history_pos),
                 list(self.wire.history_pi),
                 list(self.wire.history_E),
                 list(self.wire.history_rho))
        original_field_fn = self.wire.field_at_time

        idx = [0]
        def custom_field(t: float) -> float:
            i = idx[0] % len(audio_sub)
            idx[0] += 1
            return float(audio_sub[i])

        # reset wire for chunk-local run
        self.wire.e_pos = np.linspace(0.5,
                                      self.wire.length - 0.5,
                                      self.wire.N_electrons)
        self.wire.e_vel = np.zeros(self.wire.N_electrons)
        self.wire.history_pos  = []
        self.wire.history_pi   = []
        self.wire.history_E    = []
        self.wire.history_rho  = []
        self.wire.field_at_time = custom_field

        try:
            for n in range(n_steps):
                self.wire.step(0.02, n * 0.02)
            state = dict(
                final_pos   = self.wire.e_pos.copy(),
                final_vel   = self.wire.e_vel.copy(),
                rms         = float(np.sqrt(np.mean(audio ** 2))),
                pi_mean     = (float(np.mean(self.wire.history_pi[-1]))
                               if self.wire.history_pi else 0.0),
                E_hist      = list(self.wire.history_E),
            )
        finally:
            self.wire.field_at_time = original_field_fn
            (self.wire.e_pos,
             self.wire.e_vel,
             self.wire.history_pos,
             self.wire.history_pi,
             self.wire.history_E,
             self.wire.history_rho) = saved
        return state

    # --------------------------------------------------------
    def _spectral_analysis(self, audio: np.ndarray) -> Dict:
        """
        FFT-like analysis using linear_solver.zeta2:

            ζ(t) = Σ_{i=−K}^{K} C[i] · exp(i · t · i / α)

        x is embedded symmetrically into C ∈ ℂ^{2K+1} with K = n.
        Also produces the ternary family codes via classify_dual.
        """
        n = len(audio)
        C = np.zeros(2 * n + 1, dtype=np.complex128)
        for i in range(n):
            C[n + i] = audio[i]
            C[n - i] = audio[i]

        t_vals = np.arange(self.solver.spectral_samples, dtype=np.float64)
        zetas = np.array([zeta2(float(t), C, ALPHA_ODD) for t in t_vals])

        mu = self.solver._get_mu(n)
        codes = np.array([classify_dual(i + 1, mu) for i in range(n)],
                        dtype=np.int8)
        packed = ternary_pack(codes.tolist())

        return dict(
            zetas=zetas,
            spectral_K=n,
            codes=codes,
            packed=packed,
        )

    # --------------------------------------------------------
    def summary(self) -> str:
        if not self.chunks_raw:
            return "(no chunks loaded)"
        kept_fracs = [s['kept'] / s['K'] for s in self.stream_info]
        return "\n".join([
            "=" * 78,
            "Playback summary",
            "=" * 78,
            f"  chunks processed   : {len(self.chunks_raw)}",
            f"  mean SNR           : {np.mean(self.snr_list):.2f} dB",
            f"  min SNR            : {np.min(self.snr_list):.2f} dB",
            f"  max SNR            : {np.max(self.snr_list):.2f} dB",
            f"  mean kept fraction : {np.mean(kept_fracs):.4f}  "
            f"(target 6/π² = {DENSITY:.4f})",
            f"  bit rate           : "
            f"{self.sampler.bit_rate()/1e3:.2f} kbps",
            f"  uniform bit rate   : "
            f"{self.sampler.bit_rate_uniform()/1e3:.2f} kbps",
            f"  compression        : "
            f"{100 * self.sampler.compression_ratio():.2f} %",
        ])

    # --------------------------------------------------------
    def animate(self, interval: int = 80) -> FuncAnimation:
        if not self.chunks_raw:
            raise RuntimeError("no chunks loaded — call load_from_path first")

        n          = len(self.chunks_raw)
        chunk_size = self.chunk_size
        t_axis     = np.arange(chunk_size) / self.fs * 1000.0   # ms

        fig = plt.figure(figsize=(16, 9))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

        # (a) waveform
        ax_wave = fig.add_subplot(gs[0, 0])
        ax_wave.plot(t_axis, self.chunks_raw[0],
                     color="#2c3e50", lw=1.0, label="original")
        ax_wave.plot(t_axis, self.chunks_recon[0],
                     color="#e74c3c", lw=1.0, ls="--", label="recon")
        ax_wave.set_xlabel("time (ms)"); ax_wave.set_ylabel("amplitude")
        ax_wave.set_title("Waveform  ·  original vs reconstruction")
        ax_wave.legend(fontsize=8); ax_wave.grid(alpha=0.3)
        ax_wave.set_ylim(-1.1, 1.1)

        # (b) finite-step derivative
        ax_deriv = fig.add_subplot(gs[0, 1])
        d0 = finite_derivative(self.chunks_raw[0], A_STEP)
        ax_deriv.plot(t_axis, d0, color="#c0392b", lw=1.0)
        ax_deriv.set_xlabel("time (ms)"); ax_deriv.set_ylabel("d/dt")
        ax_deriv.set_title(f"Finite-step derivative  (A = {A_STEP:.3f})")
        ax_deriv.grid(alpha=0.3)

        # (c) throughput
        ax_tau = fig.add_subplot(gs[0, 2])
        ax_tau.plot(self.sampler.tau, color="#16a085", lw=1.2)
        ax_tau.fill_between(np.arange(chunk_size), 0,
                            self.sampler.tau,
                            color="#16a085", alpha=0.2)
        ax_tau.set_xlabel("slot k"); ax_tau.set_ylabel("τ_k")
        ax_tau.set_title("Wire throughput τ_k = Π(x_k, y₀)")
        ax_tau.grid(alpha=0.3)

        # (d) spectrum ζ(t)
        ax_spec = fig.add_subplot(gs[1, 0])
        z0 = self.spectra[0]['zetas']
        ax_spec.plot(np.arange(len(z0)), z0, "o-",
                     color="#8e44ad", lw=1.4, ms=7)
        ax_spec.set_xlabel("t (spectral index)")
        ax_spec.set_ylabel("ζ(t)")
        ax_spec.set_title("Spectral ζ(t)  ·  linear_solver (FFT)")
        ax_spec.grid(alpha=0.3)

        # (e) wire electron cloud
        ax_wire = fig.add_subplot(gs[1, 1])
        ax_wire.plot(self.wire_states[0]['final_pos'],
                     np.zeros(self.wire.N_electrons),
                     "o", color="#3498db", ms=9)
        ax_wire.set_xlim(0, self.wire.length)
        ax_wire.set_ylim(-0.5, 0.5)
        ax_wire.set_xlabel("wire position s")
        ax_wire.set_title("Electrons on wire")
        ax_wire.grid(alpha=0.3)

        # (f) per-chunk stats
        ax_stats  = fig.add_subplot(gs[1, 2])
        ax_stats2 = ax_stats.twinx()
        times = np.arange(n)
        snrs  = np.array(self.snr_list)
        kepts = np.array([s['kept'] / s['K'] for s in self.stream_info])
        ax_stats.set_xlim(0, n)
        ax_stats.set_ylim(0, max(60.0, snrs.max() * 1.1))
        ax_stats2.set_ylim(0, 1.0)
        line_snr,  = ax_stats.plot([], [], 'g-',  lw=1.5, label='SNR')
        line_kept, = ax_stats2.plot([], [], 'r--', lw=1.5,
                                     label='kept fraction')
        ax_stats.set_xlabel("chunk index")
        ax_stats.set_ylabel("SNR (dB)", color='g')
        ax_stats2.set_ylabel("kept fraction", color='r')
        ax_stats.set_title("Per-chunk stats")
        ax_stats.grid(alpha=0.3)

        suptitle = fig.suptitle("", fontsize=10)

        def update(k):
            ax_wave.clear()
            ax_wave.plot(t_axis, self.chunks_raw[k],
                         color="#2c3e50", lw=1.0, label="original")
            ax_wave.plot(t_axis, self.chunks_recon[k],
                         color="#e74c3c", lw=1.0, ls="--", label="recon")
            ax_wave.set_xlabel("time (ms)"); ax_wave.set_ylabel("amplitude")
            ax_wave.legend(fontsize=8); ax_wave.grid(alpha=0.3)
            ax_wave.set_ylim(-1.1, 1.1)

            ax_deriv.clear()
            ax_deriv.plot(t_axis,
                          finite_derivative(self.chunks_raw[k], A_STEP),
                          color="#c0392b", lw=1.0)
            ax_deriv.set_xlabel("time (ms)"); ax_deriv.set_ylabel("d/dt")
            ax_deriv.grid(alpha=0.3)

            ax_spec.clear()
            zk = self.spectra[k]['zetas']
            ax_spec.plot(np.arange(len(zk)), zk, "o-",
                         color="#8e44ad", lw=1.4, ms=7)
            ax_spec.set_xlabel("t")
            ax_spec.set_ylabel("ζ(t)")
            ax_spec.set_title(
                f"ζ(t)  ·  K = {self.spectra[k]['spectral_K']}  ·  "
                f"packed = {self.spectra[k]['packed']}")
            ax_spec.grid(alpha=0.3)

            ax_wire.clear()
            wp = self.wire_states[k]['final_pos']
            ax_wire.plot(wp, np.zeros_like(wp), "o",
                         color="#3498db", ms=9)
            ax_wire.set_xlim(0, self.wire.length)
            ax_wire.set_ylim(-0.5, 0.5)
            ax_wire.set_xlabel("wire position s")
            ax_wire.set_title(
                f"Electrons on wire  (RMS = {self.wire_states[k]['rms']:.3f})")
            ax_wire.grid(alpha=0.3)

            line_snr.set_data(times[:k + 1], snrs[:k + 1])
            line_kept.set_data(times[:k + 1], kepts[:k + 1])

            suptitle.set_text(
                f"chunk {k+1}/{n}  ·  "
                f"SNR={self.snr_list[k]:.2f} dB  ·  "
                f"kept={self.stream_info[k]['kept']}/"
                f"{self.stream_info[k]['K']}  ·  "
                f"S={self.stream_info[k]['S']:+.3f}  ·  "
                f"packed={self.spectra[k]['packed']}"
            )
            return line_snr, line_kept, suptitle

        anim = FuncAnimation(fig, update, frames=n,
                             interval=interval, blit=False)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()
        return anim


# ============================================================
#  Synthetic audio + writer
# ============================================================
def make_synthetic_audio(path: str,
                         duration: float = 2.0,
                         fs: int = 44_100) -> int:
    """Write a chord + envelope as raw int16 PCM."""
    n = int(duration * fs)
    t = np.arange(n) / fs
    s = (0.5 * np.sin(2 * np.pi * 440.0 * t)
         + 0.3 * np.sin(2 * np.pi * 554.0 * t + 0.3)
         + 0.2 * np.sin(2 * np.pi * 659.0 * t + 0.7))
    env = 0.5 * (1.0 - np.cos(2 * np.pi * t / duration))
    s = s * env
    pcm = np.clip(s * 32767, -32768, 32767).astype(np.int16)
    with open(path, "wb") as f:
        f.write(pcm.tobytes())
    return n


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius audio player  ·  finite-step  ·  wire  ·  spectral ζ")
    print("=" * 78)
    print(f"  α_sym          = 1/(π − e)     = {ALPHA_SYM:.6f}")
    print(f"  α_user         = 0.3628        = {ALPHA_USER:.4f}")
    print(f"  A (finite step)= α_sym/α_user  = {A_STEP:.6f}")
    print(f"  6/π²                           = {DENSITY:.6f}")
    print()

    tmp = os.path.join(tempfile.gettempdir(), "mobius_demo_audio.bin")
    n_samples = make_synthetic_audio(tmp, duration=2.0, fs=44_100)
    file_size = os.path.getsize(tmp)

    print("--- synthetic audio ---")
    print(f"  path           : {tmp}")
    print(f"  samples        : {n_samples:,}")
    print(f"  bytes          : {file_size:,}")
    print()

    player = MobiusAudioPlayer(
        K_stream=512,
        chunk_size=1024,
        fs=44_100,
        bps=16,
        wire_length=10.0,
        N_electrons=20,
        verbose=True,
    )
    player.load_from_path(tmp, buf_bytes=8192)
    print()
    print(player.summary())
    print()

    player.animate(interval=100)

    try:
        os.remove(tmp)
    except OSError:
        pass


# ============================================================
#  CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="Möbius audio player: finite-step compression + "
                    "wire transmission + spectral ζ analysis")
    p.add_argument("audio", nargs="?", default=None,
                   help="raw int16 PCM file; if omitted, a synthetic "
                        "chord demo is generated")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--K-stream",   type=int, default=512)
    p.add_argument("--fs",         type=int, default=44_100)
    p.add_argument("--bps",        type=int, default=16)
    p.add_argument("--wire-length", type=float, default=10.0)
    p.add_argument("--N-electrons", type=int,  default=20)
    p.add_argument("--interval",   type=int,   default=100)
    args = p.parse_args()

    if args.audio is None:
        demo()
        return

    player = MobiusAudioPlayer(
        K_stream=args.K_stream,
        chunk_size=args.chunk_size,
        fs=args.fs,
        bps=args.bps,
        wire_length=args.wire_length,
        N_electrons=args.N_electrons,
    )
    player.load_from_path(args.audio, buf_bytes=8192)
    print()
    print(player.summary())
    print()
    player.animate(interval=args.interval)


if __name__ == "__main__":
    main()