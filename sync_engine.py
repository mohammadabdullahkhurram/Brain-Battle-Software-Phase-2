"""
sync_engine.py  —  Brain Battle synchronisation engine

Responsibilities
────────────────
1. Video-to-video sync
   For each of the 4 video files, extract mono audio and compute the
   time offset relative to the reference video (the longest one) using
   FFT cross-correlation.

2. EEG-to-video sync
   Each Muse-S CSV carries Unix timestamps.  The paired video file has a
   wall-clock creation_time in its metadata (written by the camera at
   record-start).  The offset is simply:

       eeg_offset = video_creation_time_unix - eeg_first_timestamps_unix

   A positive value means the video started AFTER the EEG recording
   began — so we seek the EEG data forward by that many seconds when
   rendering the overlay.

3. Result object
   SyncResult holds every offset plus the aligned "session start" so
   callers can convert any source-local timestamps to a global timeline
   position:

       global_t = source_local_t - source_offset

All heavy work runs inside SyncWorker (QThread) and emits progress /
result signals so the UI stays responsive.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import subprocess
import tempfile
import os

import ffmpeg
import numpy as np
import pandas as pd
from scipy import signal

from PyQt6.QtCore import QThread, pyqtSignal


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VideoInfo:
    path: str
    duration: float          # seconds
    sample_rate: int         # audio sample rate
    audio: np.ndarray        # mono float32
    creation_time: Optional[float] = None   # unix epoch, from metadata


@dataclass
class EEGInfo:
    path: str
    df: pd.DataFrame
    first_timestamps: float   # unix epoch (first row of 'timestamps' column)
    channels: list[str]      # e.g. ['TP9','AF7','TP10','AF8']


@dataclass
class SourceOffset:
    """Offset for a single source relative to the global timeline start."""
    source_id: str
    path: str
    offset_sec: float        # seek this many seconds into the source to reach t=0
    is_reference: bool = False
    method: str = ""         # 'reference' | 'xcorr' | 'creation_time'
    confidence: float = 1.0  # 0-1, based on xcorr peak ratio


@dataclass
class SyncResult:
    """
    Full synchronisation result for one session.

    global_duration  — length of the shortest aligned source (safe play window)
    video_offsets    — list[SourceOffset], one per video
    eeg_offsets      — list[SourceOffset], one per EEG CSV
    warnings         — non-fatal issues encountered
    """
    global_duration: float
    video_offsets: list[SourceOffset] = field(default_factory=list)
    eeg_offsets:   list[SourceOffset] = field(default_factory=list)
    warnings:      list[str]          = field(default_factory=list)

    def all_offsets(self) -> list[SourceOffset]:
        return self.video_offsets + self.eeg_offsets

    def summary(self) -> str:
        lines = [f"Global duration: {self.global_duration:.3f}s", ""]
        for o in self.all_offsets():
            tag = "[REF]" if o.is_reference else "     "
            conf = f"  conf={o.confidence:.2f}" if o.method == "xcorr" else ""
            lines.append(
                f"{tag} {o.source_id:<18} offset={o.offset_sec:+.4f}s  "
                f"({o.method}){conf}"
            )
        if self.warnings:
            lines.append("")
            lines.extend(f"  WARN: {w}" for w in self.warnings)
        return "\n".join(lines)


# ─── Core audio helpers ───────────────────────────────────────────────────────

def _load_video(path: str) -> VideoInfo:
    """
    Load a video using ffmpeg-python only (no moviepy dependency).

    Audio is extracted by piping raw PCM float32 mono @ 44100 Hz out of
    ffmpeg into a numpy array.  Metadata (duration, creation_time) is read
    via ffmpeg.probe().
    """
    TARGET_SR = 44100

    # ── Probe metadata ────────────────────────────────────────────────────────
    probe = ffmpeg.probe(path)

    # Duration — prefer format-level, fall back to first video stream
    duration: float = float(probe["format"].get("duration", 0.0))
    if duration == 0.0:
        for s in probe.get("streams", []):
            if s.get("codec_type") == "video" and "duration" in s:
                duration = float(s["duration"])
                break

    # creation_time — check every stream then the format container
    creation_unix: Optional[float] = None
    candidates = [s.get("tags", {}) for s in probe.get("streams", [])]
    candidates.append(probe.get("format", {}).get("tags", {}))
    for tags in candidates:
        ct = tags.get("creation_time")
        if ct:
            fmt = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in ct else "%Y-%m-%dT%H:%M:%S%z"
            try:
                creation_unix = dt.datetime.strptime(ct, fmt).timestamp()
                break
            except ValueError:
                continue

    # ── Audio extraction via ffmpeg pipe ─────────────────────────────────────
    # Check whether the file has an audio stream at all
    has_audio = any(
        s.get("codec_type") == "audio" for s in probe.get("streams", [])
    )

    if has_audio:
        try:
            out, _ = (
                ffmpeg
                .input(path)
                .audio
                .output("pipe:", format="f32le", acodec="pcm_f32le",
                        ac=1, ar=str(TARGET_SR))
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            mono = np.frombuffer(out, dtype=np.float32).copy()
        except ffmpeg.Error:
            # Audio extraction failed — create silent array (xcorr will flag low confidence)
            mono = np.zeros(int(duration * TARGET_SR), dtype=np.float32)
    else:
        mono = np.zeros(int(duration * TARGET_SR), dtype=np.float32)

    return VideoInfo(
        path=path,
        duration=duration,
        sample_rate=TARGET_SR,
        audio=mono,
        creation_time=creation_unix,
    )


def _load_eeg(path: str) -> EEGInfo:
    """Load a muselsl CSV.  Expected columns: timestamps, TP9, AF7, TP10, AF8."""
    df = pd.read_csv(path)

    if "timestamps" not in df.columns:
        raise ValueError(f"EEG CSV '{path}' has no 'timestamps' column.")

    eeg_channels = [c for c in ["TP9", "AF7", "TP10", "AF8"] if c in df.columns]
    if not eeg_channels:
        raise ValueError(f"EEG CSV '{path}' contains none of the expected EEG channels.")

    return EEGInfo(
        path=path,
        df=df,
        first_timestamps=float(df["timestamps"].iloc[0]),
        channels=eeg_channels,
    )


def _xcorr_offset(
    ref_audio: np.ndarray,
    other_audio: np.ndarray,
    ref_sr: int,
    other_sr: int,
) -> tuple[float, float]:
    """
    Compute the time offset (seconds) of other_audio relative to ref_audio
    using FFT cross-correlation.

    Returns (offset_seconds, confidence).
    A positive offset means other_audio starts LATER than ref_audio —
    i.e. seek other_audio forward by offset_seconds to align with ref.

    Confidence is the ratio of the peak correlation to mean correlation,
    normalised to [0, 1].  Values below ~0.3 suggest the two clips may
    not share a common audio event and xcorr may be unreliable.
    """
    # Resample other_audio to ref sample rate if needed
    if ref_sr != other_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(ref_sr, other_sr)
        other_audio = resample_poly(other_audio, ref_sr // g, other_sr // g)

    # Use the shorter length to keep "same" mode symmetric
    n = min(len(ref_audio), len(other_audio))
    a = ref_audio[:n].astype(np.float64)
    b = other_audio[:n].astype(np.float64)

    # Normalise to unit variance to make confidence comparable across clips
    def _norm(x):
        s = x.std()
        return x / s if s > 1e-9 else x

    a, b = _norm(a), _norm(b)

    corr = signal.correlate(a, b, mode="same", method="fft")
    lags = signal.correlation_lags(len(a), len(b), mode="same")

    peak_idx = int(np.argmax(np.abs(corr)))
    offset_samples = lags[peak_idx]
    offset_sec = float(offset_samples) / ref_sr

    # Confidence: peak-to-mean ratio (clipped to [0, 1])
    peak_val = float(np.abs(corr[peak_idx]))
    mean_val = float(np.abs(corr).mean())
    confidence = float(np.clip((peak_val / mean_val - 1.0) / 50.0, 0.0, 1.0))

    return offset_sec, confidence


# ─── Main sync function ───────────────────────────────────────────────────────

def compute_sync(
    video_paths: list[str],
    eeg_paths: list[str],
    eeg_video_pairs: Optional[list[tuple[int, int]]] = None,
    progress_cb=None,
) -> SyncResult:
    """
    Compute full synchronisation for a session.

    Parameters
    ──────────
    video_paths      — list of 1-4 video file paths
    eeg_paths        — list of 0-2 EEG CSV file paths
    eeg_video_pairs  — optional list of (eeg_index, video_index) pairs
                       indicating which video was recorded alongside which EEG.
                       If None, pairs are assigned by index order (eeg[0]↔video[0], etc.)
    progress_cb      — optional callable(int, str) receiving (percent, message)

    Returns
    ───────
    SyncResult with all offsets relative to a global timeline.
    """
    warnings: list[str] = []

    def _progress(pct: int, msg: str):
        if progress_cb:
            progress_cb(pct, msg)

    # ── 1. Load videos ────────────────────────────────────────────────────────
    _progress(0, "Loading video files…")
    video_infos: list[VideoInfo] = []
    for i, path in enumerate(video_paths):
        _progress(int(5 + i * 15), f"Loading video {i+1}/{len(video_paths)}…")
        video_infos.append(_load_video(path))

    # ── 2. Choose reference video (longest) ──────────────────────────────────
    ref_idx = int(np.argmax([v.duration for v in video_infos]))
    ref = video_infos[ref_idx]

    _progress(20, f"Reference video: #{ref_idx+1} ({ref.duration:.1f}s)")

    # ── 3. Compute video-to-video offsets via xcorr ───────────────────────────
    video_offsets: list[SourceOffset] = []
    for i, vi in enumerate(video_infos):
        source_id = f"VIDEO_{i+1:02d}"
        if i == ref_idx:
            video_offsets.append(SourceOffset(
                source_id=source_id,
                path=vi.path,
                offset_sec=0.0,
                is_reference=True,
                method="reference",
                confidence=1.0,
            ))
            continue

        pct = 20 + int((i / len(video_infos)) * 50)
        _progress(pct, f"Cross-correlating video {i+1} vs reference…")

        off, conf = _xcorr_offset(ref.audio, vi.audio, ref.sample_rate, vi.sample_rate)

        if conf < 0.15:
            warnings.append(
                f"VIDEO_{i+1:02d}: low xcorr confidence ({conf:.2f}). "
                "Consider checking audio levels."
            )

        video_offsets.append(SourceOffset(
            source_id=source_id,
            path=vi.path,
            offset_sec=off,
            is_reference=False,
            method="xcorr",
            confidence=conf,
        ))

    # ── 4. Load EEG files ─────────────────────────────────────────────────────
    _progress(70, "Loading EEG files…")
    eeg_infos: list[EEGInfo] = []
    for i, path in enumerate(eeg_paths):
        _progress(70 + i * 5, f"Loading EEG {i+1}/{len(eeg_paths)}…")
        eeg_infos.append(_load_eeg(path))

    # ── 5. Compute EEG offsets via creation_time delta ────────────────────────
    _progress(80, "Aligning EEG timestamps to video…")

    if eeg_video_pairs is None:
        eeg_video_pairs = [(i, i) for i in range(len(eeg_infos))]

    eeg_offsets: list[SourceOffset] = []
    for eeg_i, (eeg_idx, vid_idx) in enumerate(eeg_video_pairs):
        eeg = eeg_infos[eeg_idx]
        vi  = video_infos[vid_idx]
        source_id = f"EEG_{eeg_idx+1:02d}"

        if vi.creation_time is None:
            # Fallback: no creation_time in video — offset is unknown, set 0
            warnings.append(
                f"EEG_{eeg_idx+1:02d}: paired video has no creation_time metadata. "
                "EEG offset set to 0 — manual alignment may be needed."
            )
            eeg_offsets.append(SourceOffset(
                source_id=source_id,
                path=eeg.path,
                offset_sec=0.0,
                method="fallback",
                confidence=0.0,
            ))
            continue

        # video_creation_time is the Unix time when the camera started recording.
        # eeg.first_timestamps is the Unix time of the first EEG sample.
        # delta > 0  → video started after EEG → EEG has extra data at the front
        # delta < 0  → video started before EEG → EEG starts mid-video
        delta = vi.creation_time - eeg.first_timestamps

        # Also incorporate the video's own offset relative to the reference
        vid_offset = video_offsets[vid_idx].offset_sec

        # EEG offset in global timeline terms:
        # To find EEG data at global_t, read EEG at: global_t + delta - vid_offset
        eeg_global_offset = delta - vid_offset

        eeg_offsets.append(SourceOffset(
            source_id=source_id,
            path=eeg.path,
            offset_sec=eeg_global_offset,
            method="creation_time",
            confidence=0.95,
        ))

    # ── 6. Compute global duration ────────────────────────────────────────────
    _progress(90, "Computing global duration…")
    # Safe play window = shortest available content across all aligned sources
    durations: list[float] = []
    for vi, off in zip(video_infos, video_offsets):
        durations.append(vi.duration - abs(off.offset_sec))
    global_duration = min(durations) if durations else 0.0

    _progress(100, "Sync complete.")

    return SyncResult(
        global_duration=global_duration,
        video_offsets=video_offsets,
        eeg_offsets=eeg_offsets,
        warnings=warnings,
    )


def get_eeg_window(
    eeg_info: EEGInfo,
    eeg_offset: SourceOffset,
    global_t: float,
    window_sec: float = 10.0,
) -> pd.DataFrame:
    """
    Return EEG rows visible at a given global timeline position.

    Parameters
    ──────────
    eeg_info    — loaded EEGInfo object
    eeg_offset  — the SourceOffset computed for this EEG file
    global_t    — current playhead position in global timeline (seconds from t=0)
    window_sec  — how many seconds of EEG data to return

    Returns
    ───────
    DataFrame slice containing rows within [global_t, global_t + window_sec]
    with a 'global_t' column added for easy plotting.
    """
    df = eeg_info.df.copy()

    # Convert raw Unix timestamps to global timeline seconds
    # global_t_of_sample = (sample_unix - eeg_first_ts) - eeg_offset.offset_sec
    df["global_t"] = (df["timestamps"] - eeg_info.first_timestamps) - eeg_offset.offset_sec

    mask = (df["global_t"] >= global_t) & (df["global_t"] < global_t + window_sec)
    return df[mask].reset_index(drop=True)


# ─── QThread worker ──────────────────────────────────────────────────────────

class SyncWorker(QThread):
    """
    Runs compute_sync() off the main thread.

    Signals
    ───────
    progress(int, str)     — percent complete + status message
    finished(SyncResult)   — emitted when sync succeeds
    error(str)             — emitted on exception
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)   # SyncResult
    error    = pyqtSignal(str)

    def __init__(
        self,
        video_paths: list[str],
        eeg_paths: list[str],
        eeg_video_pairs: Optional[list[tuple[int, int]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.video_paths    = video_paths
        self.eeg_paths      = eeg_paths
        self.eeg_video_pairs = eeg_video_pairs

    def run(self):
        try:
            result = compute_sync(
                video_paths=self.video_paths,
                eeg_paths=self.eeg_paths,
                eeg_video_pairs=self.eeg_video_pairs,
                progress_cb=lambda pct, msg: self.progress.emit(pct, msg),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
