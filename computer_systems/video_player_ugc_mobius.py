#!/usr/bin/env python3
"""
mobius_video_player.py
======================

Video player combining three layers:

    stream    download_mboius_unpack_buffer.MobiusPacker
              → bytes in, MOSB-framed pack/unpack round-trip, bytes out
    frames    photo_compress.compress_image / decompress_image
              → square-free FFT compression per frame
    motion    ugc_discrete_log.UGCDiscreteLog
              → frames walk the L side of the UGC graph
                y_k = −H_{x_k + 1}  (log trajectory)

Usage
-----
    python mobius_video_player.py                    # synthetic demo
    python mobius_video_player.py <video.bin>        # real raw video file
        [--frame-size H W] [--K-stream 512]
        [--ugc-p 10007] [--ugc-g 5]
"""

from __future__ import annotations
import argparse
import hashlib
import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# ---- project imports ----
from download_mboius_unpack_buffer import (
    MobiusPacker,
    stream_url,
    parse_size,
)
from photo_compress import (
    compress_image,
    decompress_image,
    psnr,
)
from ugc_discrete_log import UGCDiscreteLog


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)
GAMMA   = 0.5772156649015329
DENSITY = 6.0 / (PI * PI)


# ============================================================
#  Player
# ============================================================
class MobiusVideoPlayer:
    """
    Video player:

        bytes ──[MobiusPacker]── frame bytes ──[photo_compress]── recon
                                    │
                                    └──[ugc_discrete_log]── (x, −H_{x+1})
    """
    def __init__(self,
                 K_stream:   int = 512,
                 ugc_p:      int = 10007,
                 ugc_g:      int = 5,
                 frame_size: Tuple[int, int] = (64, 64),
                 verbose:    bool = True):
        self.K_stream = K_stream
        self.frame_size = frame_size
        self.verbose = verbose

        # 1. streaming layer
        self.packer = MobiusPacker(K=K_stream)

        # 2. UGC motion layer
        self.ugc = UGCDiscreteLog(p=ugc_p, g=ugc_g)

        # results
        self.frames_raw:    List[np.ndarray] = []
        self.frames_recon:  List[np.ndarray] = []
        self.psnr_list:     List[float]      = []
        self.stream_info:   List[Dict]       = []
        self.ugc_walk:      List[Tuple[int, float]] = []
        self.ugc_dlog:      List[Tuple[int, float]] = []

    # --------------------------------------------------------
    def load_from_url(self, url: str,
                      buf_bytes: int = 4 << 20,
                      max_frames: Optional[int] = None) -> int:
        H, W = self.frame_size
        frame_bytes_count = H * W      # grayscale uint8

        buffer = bytearray()
        n_frames = 0
        total_in = 0
        total_packed = 0

        for chunk, received, total in stream_url(url, buf_bytes):
            # ---- 1. pack/unpack round-trip ----
            packed, S, H_ent, m = self.packer.pack(chunk)
            raw, S2, H2, m2, cid = self.packer.unpack(packed)
            if raw != chunk:
                raise RuntimeError(f"stream mismatch on chunk {cid}")

            buffer.extend(raw)
            total_in     += len(chunk)
            total_packed += len(packed)

            # ---- 2. split into frames ----
            while len(buffer) >= frame_bytes_count:
                fb = bytes(buffer[:frame_bytes_count])
                del buffer[:frame_bytes_count]

                frame = (np.frombuffer(fb, dtype=np.uint8)
                         .reshape(H, W).astype(np.float64) / 255.0)
                self._process_frame(frame, S, H_ent, m, cid, n_frames)
                n_frames += 1

                if max_frames and n_frames >= max_frames:
                    break

            if max_frames and n_frames >= max_frames:
                break

        if self.verbose:
            print(f"  bytes in        : {total_in:,}")
            print(f"  bytes packed    : {total_packed:,}")
            print(f"  pack ratio      : "
                  f"{total_packed / max(total_in, 1):.6f}")
            print(f"  frames loaded   : {n_frames}")
            if self.psnr_list:
                print(f"  mean PSNR       : "
                      f"{np.mean(self.psnr_list):.2f} dB")
        return n_frames

    # --------------------------------------------------------
    def _process_frame(self, frame: np.ndarray,
                       S: float, H_ent: float, m: float,
                       cid: int, k: int) -> None:
        # ---- 1. compress + reconstruct ----
        payload = compress_image(frame)
        recon   = decompress_image(payload)
        p       = psnr(frame, recon)

        # ---- 2. walk position  (smooth, video-like) ----
        x_walk = k
        y_walk = (-self.ugc.H[x_walk + 1]
                  if x_walk + 1 < len(self.ugc.H)
                  else -math.log(x_walk + 1) - GAMMA)

        # ---- 3. dlog anchor from frame content ----
        fh = hashlib.sha256(frame.tobytes()).digest()
        h_val = int.from_bytes(fh[:8], "big") % self.ugc.p
        if h_val == 0:
            h_val = 1
        x_true, y_true = self.ugc.dlog(h_val)
        if x_true is None:
            x_true, y_true = 0, 0.0

        # ---- 4. store ----
        self.frames_raw.append(frame)
        self.frames_recon.append(recon)
        self.psnr_list.append(p)
        self.stream_info.append(dict(
            S=S, H=H_ent, m=m, cid=cid,
            kept=payload['sqfree'].size,
            N=payload['N'],
        ))
        self.ugc_walk.append((x_walk, y_walk))
        self.ugc_dlog.append((x_true, y_true))

    # --------------------------------------------------------
    def summary(self) -> str:
        if not self.frames_raw:
            return "(no frames loaded)"
        kept_fracs = [s['kept'] / s['N'] for s in self.stream_info]
        lines = [
            "=" * 78,
            "Playback summary",
            "=" * 78,
            f"  frames processed   : {len(self.frames_raw)}",
            f"  mean PSNR          : {np.mean(self.psnr_list):.2f} dB",
            f"  min PSNR           : {np.min(self.psnr_list):.2f} dB",
            f"  max PSNR           : {np.max(self.psnr_list):.2f} dB",
            f"  mean kept fraction : {np.mean(kept_fracs):.4f}  "
            f"(target 6/π² = {DENSITY:.4f})",
            f"  walk x range       : "
            f"{min(x for x, _ in self.ugc_walk)} … "
            f"{max(x for x, _ in self.ugc_walk)}",
            f"  walk y range       : "
            f"{min(y for _, y in self.ugc_walk):.4f} … "
            f"{max(y for _, y in self.ugc_walk):.4f}",
            f"  dlog x range       : "
            f"{min(x for x, _ in self.ugc_dlog)} … "
            f"{max(x for x, _ in self.ugc_dlog)}",
        ]
        return "\n".join(lines)

    # --------------------------------------------------------
    def animate(self, interval: int = 80) -> FuncAnimation:
        if not self.frames_raw:
            raise RuntimeError("no frames loaded — call load_from_url first")

        n = len(self.frames_raw)
        n_ugc = min(64, self.ugc.n)

        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.3, 1])

        # ---------------- (a) frame view ----------------
        ax_frame = fig.add_subplot(gs[0, 0])
        ax_frame.axis('off')
        ax_frame.set_title("Reconstructed frame")
        im = ax_frame.imshow(self.frames_recon[0],
                             cmap='gray', vmin=0, vmax=1)

        # ---------------- (b) UGC graph ----------------
        ax_ugc = fig.add_subplot(gs[0, 1])
        Lx = np.arange(n_ugc); Ly = np.ones(n_ugc)
        Rx = np.arange(n_ugc); Ry = np.zeros(n_ugc)
        ax_ugc.scatter(Lx, Ly, s=25, c='#3498db', zorder=2, label='L side')
        ax_ugc.scatter(Rx, Ry, s=25, c='#e67e22', zorder=2, label='R side')
        for i in range(n_ugc):
            ax_ugc.plot([Lx[i], Rx[i]], [Ly[i], Ry[i]],
                        color='gray', lw=0.3, alpha=0.25)

        # walk trajectory
        walk_xs = [self.ugc_walk[0][0] % n_ugc]
        walk_ys = [1.0]
        traj_line, = ax_ugc.plot(walk_xs, walk_ys, '-',
                                  color='#8e44ad', lw=1.6, alpha=0.7,
                                  label='walk trajectory')
        # current position marker
        cur_marker, = ax_ugc.plot([], [], 'ro', ms=14, zorder=5,
                                   label='current')
        # dlog anchors on L side
        dlog_xs = [self.ugc_dlog[0][0] % n_ugc]
        dlog_ys = [1.0]
        dlog_scatter = ax_ugc.scatter(dlog_xs, dlog_ys, s=60,
                                       marker='x', c='#16a085', zorder=4,
                                       label='dlog(frame_hash)')

        ax_ugc.set_xlim(-1, n_ugc)
        ax_ugc.set_ylim(-0.3, 1.3)
        ax_ugc.set_xlabel('UGC position x')
        ax_ugc.set_ylabel('side')
        ax_ugc.set_title("UGC graph  ·  frames walk L side\n"
                         "walk y = −H_(x+1)   ·   dlog from content")
        ax_ugc.legend(fontsize=8, loc='upper center', ncol=2)
        ax_ugc.grid(alpha=0.3)

        # ---------------- (c) stats ----------------
        ax_stats  = fig.add_subplot(gs[0, 2])
        ax_stats2 = ax_stats.twinx()

        times       = np.arange(n)
        psnrs_arr   = np.array(self.psnr_list)
        kept_arr    = np.array([s['kept'] / s['N'] for s in self.stream_info])

        ax_stats.set_xlim(0, n)
        ax_stats.set_ylim(0, max(60.0, psnrs_arr.max() * 1.1))
        ax_stats2.set_ylim(0, 1.0)

        line_psnr, = ax_stats.plot([], [], 'g-', lw=1.5, label='PSNR')
        line_kept, = ax_stats2.plot([], [], 'r--', lw=1.5, label='kept')

        ax_stats.set_xlabel('frame index')
        ax_stats.set_ylabel('PSNR (dB)', color='g')
        ax_stats2.set_ylabel('kept fraction', color='r')
        ax_stats.set_title("Per-frame stats")
        ax_stats.grid(alpha=0.3)

        suptitle = fig.suptitle("", fontsize=10)

        # ---------------- animation callback ----------------
        def update(k):
            im.set_data(self.frames_recon[k])

            cur_x = self.ugc_walk[k][0] % n_ugc
            cur_marker.set_data([cur_x], [1.0])

            walk_xs.append(self.ugc_walk[k][0] % n_ugc)
            walk_ys.append(1.0)
            traj_line.set_data(walk_xs, walk_ys)

            dx = self.ugc_dlog[k][0] % n_ugc
            dlog_xs.append(dx)
            dlog_ys.append(1.0)
            dlog_scatter.set_offsets(np.array(list(zip(dlog_xs, dlog_ys))))

            line_psnr.set_data(times[:k + 1], psnrs_arr[:k + 1])
            line_kept.set_data(times[:k + 1], kept_arr[:k + 1])

            x_w, y_w = self.ugc_walk[k]
            x_d, y_d = self.ugc_dlog[k]
            suptitle.set_text(
                f"frame {k + 1}/{n}  ·  "
                f"walk (x={x_w}, y={y_w:.4f})  ·  "
                f"dlog (x={x_d}, y={y_d:.4f})  ·  "
                f"PSNR={self.psnr_list[k]:.2f} dB  ·  "
                f"kept={self.stream_info[k]['kept']}/{self.stream_info[k]['N']}  ·  "
                f"S={self.stream_info[k]['S']:+.3f}"
            )
            return im, cur_marker, traj_line, dlog_scatter, line_psnr, line_kept

        anim = FuncAnimation(fig, update, frames=n,
                              interval=interval, blit=False)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()
        return anim


# ============================================================
#  Synthetic frame + video writer
# ============================================================
def make_frame(t: int, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Synthetic grayscale frame: moving gaussian blob + slowly varying bg."""
    H, W = size
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)

    cx = 0.5 + 0.30 * np.cos(t * 0.20)
    cy = 0.5 + 0.30 * np.sin(t * 0.15)
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    blob = np.exp(-r2 / 0.02)

    bg = 0.15 * np.sin(4 * X + t * 0.1) * np.cos(4 * Y + t * 0.07) + 0.5
    return np.clip(bg + blob, 0, 1)


def make_synthetic_video(path: str,
                          n_frames: int = 60,
                          size: Tuple[int, int] = (64, 64)
                          ) -> Tuple[int, Tuple[int, int]]:
    """Write a synthetic grayscale video as concatenated uint8 bytes."""
    with open(path, "wb") as f:
        for t in range(n_frames):
            frame = make_frame(t, size)
            f.write((frame * 255).astype(np.uint8).tobytes())
    return n_frames, size


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius video player  ·  square-free FFT  ·  UGC motion  ·  "
          "stream pack/unpack")
    print("=" * 78)
    print(f"  α              = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  γ (Euler-Mascheroni)         = {GAMMA:.6f}")
    print(f"  6/π²                         = {DENSITY:.6f}")
    print()

    # 1. generate a synthetic raw video on disk
    tmp_video = os.path.join(tempfile.gettempdir(),
                             "mobius_demo_video.bin")
    n_frames, size = make_synthetic_video(tmp_video,
                                          n_frames=60,
                                          size=(64, 64))
    file_size = os.path.getsize(tmp_video)

    print("--- synthetic video ---")
    print(f"  path             : {tmp_video}")
    print(f"  frames           : {n_frames}")
    print(f"  frame size       : {size}  (grayscale uint8)")
    print(f"  bytes / frame    : {size[0] * size[1]}")
    print(f"  total size       : {file_size:,} bytes")
    print()

    # 2. instantiate + load
    player = MobiusVideoPlayer(
        K_stream=512,
        ugc_p=10007,
        ugc_g=5,
        frame_size=size,
        verbose=True,
    )
    print()
    player.load_from_url(tmp_video, buf_bytes=8192, max_frames=None)
    print()
    print(player.summary())
    print()

    # 3. animate
    player.animate(interval=100)

    # cleanup
    try:
        os.remove(tmp_video)
    except OSError:
        pass


# ============================================================
#  CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(
        description="Möbius video player: square-free FFT frames + "
                    "UGC motion + pack/unpack streaming")
    p.add_argument("video", nargs="?", default=None,
                   help="raw video file (concatenated grayscale frames); "
                        "if omitted, a synthetic demo is generated")
    p.add_argument("--frame-size", type=int, nargs=2,
                   default=(64, 64), metavar=("H", "W"))
    p.add_argument("--K-stream", type=int, default=512,
                   help="MobiusPacker sieve size")
    p.add_argument("--ugc-p", type=int, default=10007,
                   help="UGC group prime")
    p.add_argument("--ugc-g", type=int, default=5,
                   help="UGC group generator")
    p.add_argument("--interval", type=int, default=100,
                   help="animation frame interval (ms)")
    args = p.parse_args()

    if args.video is None:
        demo()
        return

    player = MobiusVideoPlayer(
        K_stream=args.K_stream,
        ugc_p=args.ugc_p,
        ugc_g=args.ugc_g,
        frame_size=tuple(args.frame_size),
    )
    player.load_from_url(args.video, buf_bytes=8192)
    print()
    print(player.summary())
    print()
    player.animate(interval=args.interval)


if __name__ == "__main__":
    main()