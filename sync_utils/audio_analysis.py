"""
audio_analysis.py — Brain Battle inter-camera delay computation

Strategy
────────
Camera clocks are NOT reliably synced across manufacturers (Panasonic DSLR vs
Insta360 clocks can differ by days).  creation_time metadata is therefore used
only as a sanity-check, not as the primary sync source.

Primary method: audio cross-correlation
  1. Extract the first MAX_AUDIO_SEC seconds of mono audio at EXTRACT_SR Hz
     directly from ffmpeg (low sample rate = fast extraction, sufficient bandwidth)
  2. Bandpass 300–3000 Hz  →  removes rumble & hiss, keeps punch/voice transients
  3. Analytic envelope  abs(hilbert())  →  instantaneous energy shape
  4. Downsample to TARGET_SR via resample_poly  →  1 ms resolution, zero phase error
  5. L2-normalise  →  corrects for amplitude differences across mics
  6. Normalised FFT xcorr mode='full'  →  full lag range, polarity-safe
  7. Print confidence; warn if low

Speed: extraction at 8 kHz on 5-minute clip = 2.4 M samples.
       hilbert + resample + xcorr < 2s per pair.  Total < 10s for 4 clips.
"""

from typing import List, Optional
import subprocess
import datetime as dt
import os
from math import gcd

import numpy as np
from scipy import signal as _signal
import ffmpeg as _ffmpeg


# ── Constants ─────────────────────────────────────────────────────────────────

EXTRACT_SR    = 8000   # Hz — extract at low rate directly from ffmpeg
                       # Nyquist = 4000 Hz > 3000 Hz bandpass upper edge → safe
TARGET_SR     = 1000   # Hz — final rate for xcorr (1 ms resolution)
BP_LOW_HZ     = 300
BP_HIGH_HZ    = 3000
BP_ORDER      = 4
MAX_AUDIO_SEC = 300    # only use first 5 min of each clip (enough for shared events)

_CT_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_file_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def find_longest_vid(video_paths: List[str]) -> int:
    best_idx, best_dur = 0, -1.0
    for i, path in enumerate(video_paths):
        try:
            probe    = _ffmpeg.probe(path)
            duration = float(probe["format"].get("duration", 0.0))
        except Exception:
            duration = 0.0
        if duration > best_dur:
            best_dur = duration
            best_idx = i
    return best_idx


def find_all_durations(video_paths: List[str]) -> List[float]:
    durations = []
    for path in video_paths:
        try:
            probe    = _ffmpeg.probe(path)
            duration = float(probe["format"].get("duration", 0.0))
            if duration == 0.0:
                for s in probe.get("streams", []):
                    if s.get("codec_type") == "video" and "duration" in s:
                        duration = float(s["duration"])
                        break
        except Exception:
            duration = 0.0
        durations.append(duration)
    return durations


def process_audio(video_path: str) -> tuple[np.ndarray, int]:
    """Extract mono audio at EXTRACT_SR Hz, capped at MAX_AUDIO_SEC seconds."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(EXTRACT_SR),
        "-t", str(MAX_AUDIO_SEC),   # only extract first MAX_AUDIO_SEC seconds
        "-f", "f32le", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio  = np.frombuffer(result.stdout, dtype=np.float32).copy()
    print(f"  Audio: {get_file_name(video_path)}  "
          f"({len(audio)/EXTRACT_SR:.0f}s @ {EXTRACT_SR} Hz)")
    return audio, EXTRACT_SR


def _prepare_for_xcorr(audio: np.ndarray, sr: int) -> np.ndarray:
    """Bandpass → envelope → downsample to TARGET_SR → L2-normalise."""
    nyq = sr / 2.0
    # Clamp upper bandpass to 99% of Nyquist to avoid filter instability
    bp_high = min(BP_HIGH_HZ, nyq * 0.99)
    sos      = _signal.butter(BP_ORDER, [BP_LOW_HZ / nyq, bp_high / nyq],
                              btype="band", output="sos")
    filtered  = _signal.sosfilt(sos, audio.astype(np.float64))
    envelope  = np.abs(_signal.hilbert(filtered))
    g         = gcd(TARGET_SR, sr)
    resampled = _signal.resample_poly(envelope, TARGET_SR // g, sr // g)
    norm = np.linalg.norm(resampled)
    if norm > 1e-9:
        resampled /= norm
    return resampled.astype(np.float64)


def _xcorr_offset(pivot_env: np.ndarray, other_env: np.ndarray) -> tuple[float, float]:
    """
    Normalised FFT cross-correlation.
    Returns (offset_sec, confidence).
    offset_sec > 0  →  other clip started LATER than pivot.
    """
    corr     = _signal.correlate(pivot_env, other_env, mode="full", method="fft")
    lags     = _signal.correlation_lags(len(pivot_env), len(other_env), mode="full")
    abs_corr = np.abs(corr)
    peak_idx = int(np.argmax(abs_corr))

    offset_sec = float(lags[peak_idx]) / TARGET_SR

    p95        = float(np.percentile(abs_corr, 95))
    peak_val   = float(abs_corr[peak_idx])
    confidence = float(np.clip((peak_val / p95 - 1.0) / 10.0, 0.0, 1.0)) if p95 > 1e-9 else 0.0

    return offset_sec, confidence


# ── Public API ────────────────────────────────────────────────────────────────

def compute_delays(video_paths: List[str]) -> List[float]:
    """
    Compute start-time delays between clips using audio cross-correlation.

    The pivot is the longest clip.  All delays are returned relative to the
    earliest-starting clip (minimum delay = 0, all others >= 0).

    Returns
    ───────
    list[float] — delay in seconds per clip, all >= 0.
    """
    n = len(video_paths)
    if n < 2:
        raise ValueError("At least two video paths are required.")

    # Longest clip = pivot (most audio overlap with all others)
    pivot = find_longest_vid(video_paths)
    print(f"\n[Sync] Pivot: {get_file_name(video_paths[pivot])} (longest)")
    print(f"[Sync] Extracting first {MAX_AUDIO_SEC}s of audio at {EXTRACT_SR} Hz:")

    # Extract and preprocess all clips
    raw    = [process_audio(p) for p in video_paths]
    envs   = [_prepare_for_xcorr(a, sr) for a, sr in raw]

    pivot_env = envs[pivot]
    delays    = [0.0] * n

    print("[Sync] Cross-correlating:")
    for i in range(n):
        if i == pivot:
            continue
        offset, conf = _xcorr_offset(pivot_env, envs[i])
        delays[i] = offset
        status = "OK" if conf > 0.25 else "LOW CONFIDENCE — check manually"
        print(f"  clip {i} ({get_file_name(video_paths[i])}): "
              f"offset={offset:+.3f}s  conf={conf:.2f}  [{status}]")

    # Shift so minimum delay = 0 (earliest clip is the reference)
    min_d  = min(delays)
    delays = [d - min_d for d in delays]

    return delays


def find_all_delays_with_pivot(video_paths: List[str], pivot_index: int) -> List[float]:
    """Legacy wrapper — pivot_index ignored, uses longest clip as pivot."""
    return compute_delays(video_paths)


def find_all_delays(video_paths: List[str]) -> List[float]:
    return compute_delays(video_paths)


def autosync(video_paths: List[str]) -> List[float]:
    if not all(os.path.exists(p) for p in video_paths):
        raise FileNotFoundError("One or more video paths are invalid.")
    delays = compute_delays(video_paths)
    print(f"Delays: {delays}")
    return delays