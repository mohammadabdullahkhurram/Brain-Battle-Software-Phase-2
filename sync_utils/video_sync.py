"""
video_sync.py — Brain Battle per-clip alignment via stream-copy trim

Two-stage alignment for sub-frame accuracy without re-encoding
──────────────────────────────────────────────────────────────
Stage 1  (generate_single_preview)
  ffmpeg -ss <seek> -i <clip> -c copy -avoid_negative_ts make_zero
  Seeks to the nearest keyframe BEFORE seek_sec and copies from there.
  Fast: completes in < 5s regardless of clip length or resolution.
  Limitation: output starts at the keyframe, which may be up to 1 GOP
  (≈ 2s at 23.98 fps) EARLIER than the desired seek point.

Stage 2  (get_first_frame_pts)
  ffprobe reads the PTS of the first video packet in the temp file.
  residual[i] = actual_first_pts[i]   (seconds from file start to desired sync point)
  This is passed back to _on_sync_finished as the per-clip VLC start offset.

Stage 3  (in _on_sync_finished / _prebuffer_done)
  VLC loads the temp file, then set_time(residual_ms[i]) to skip the
  pre-keyframe leader.  VLC can seek within a playing file to any frame —
  no re-encode needed.  Sub-frame accurate after this correction.
"""

import ffmpeg
import numpy as np
import datetime
import subprocess
import re
import os


def generate_single_preview(
    video_path: str,
    delay: float,
    duration: float,
    output_path: str,
    max_delay: float = 0,
) -> tuple[list, float, float]:
    """
    Build an FFmpeg stream-copy command that coarsely trims clip i
    to start at the shared sync point.

    Returns
    ───────
    (command, new_duration, seek_sec)
      command      — compiled ffmpeg command list
      new_duration — approximate output duration (seconds)
      seek_sec     — the requested seek point; caller uses this with
                     get_first_frame_pts() to compute the fine residual
    """
    seek_sec = round(max_delay - delay, 3)

    if seek_sec > 0:
        inp = ffmpeg.input(video_path, ss=seek_sec)
    else:
        inp = ffmpeg.input(video_path)

    out = ffmpeg.output(
        inp,
        output_path,
        vcodec="copy",
        acodec="copy",
        avoid_negative_ts="make_zero",
    )

    command      = out.compile()
    new_duration = max(duration - seek_sec, 0.0)
    return command, new_duration, seek_sec


def get_first_frame_pts(video_path: str) -> float:
    """
    Return the PTS (seconds) of the first video frame in a file.

    After a stream-copy seek, the file starts at the nearest keyframe
    which may be up to 1 GOP before the desired seek point.  This function
    reads the actual first-frame timestamp so the caller can compute
    residual = first_pts  (how many ms into the file VLC should start).

    Returns 0.0 if the file doesn't exist or ffprobe fails.
    """
    if not os.path.exists(video_path):
        return 0.0
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "packet=pts_time",
                "-of", "csv=p=0",
                "-read_intervals", "%+#1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        text = result.stdout.decode().strip()
        if text:
            return float(text.split("\n")[0].strip())
    except Exception:
        pass
    return 0.0


def run_ffmpeg_subprocess(ffmpeg_command: list, resulting_duration: float, debug: bool = False):
    """Run a compiled FFmpeg command, printing progress."""
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except OSError:
        print("OSError running FFmpeg")
        return
    except ValueError as e:
        print(f"ValueError: {e}")
        return
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return

    end_time_str = str(datetime.timedelta(seconds=max(resulting_duration, 0)))
    print(f"FFmpeg starting... (expected: {end_time_str})")
    for line in ffmpeg_process.stdout:
        if debug:
            print(line, end="")
        if line.startswith("out_time="):
            progress = line.split("=")[1].strip()
            print(f"  Progress: {progress} / {end_time_str}", end="\r")
    ffmpeg_process.wait()
    print("\nFFmpeg finished.")


def generate_video_sync_command(
    all_video_paths: list[str],
    delays: list[float],
    durations: list[float],
    output_file_name_with_extension: str,
    video_end_time: float | None = None,
) -> tuple[list, float]:
    """Grid export with filter-based alignment (requires decode; used for export only)."""
    videos        = [ffmpeg.input(path) for path in all_video_paths]
    video_streams = [v.video for v in videos]
    audio_streams = [v.audio for v in videos]
    max_delay     = max(delays)

    for i, video in enumerate(video_streams):
        seek = round(max_delay - delays[i], 2)
        if seek > 0:
            video_streams[i] = ffmpeg.trim(video, start=seek).setpts("PTS-STARTPTS")
            audio_streams[i] = (
                ffmpeg.filter(audio_streams[i], "atrim", start=seek)
                      .filter("asetpts", "PTS-STARTPTS")
            )
            durations[i] = max(durations[i] - seek, 0.0)

    xstacked      = ffmpeg.filter(video_streams, "xstack", inputs="4", layout="0_0|0_h0|w0_0|w0_h0")
    max_delay_idx = int(np.argmax(delays))
    audio_stream  = audio_streams[max_delay_idx]
    video_scaled  = (
        ffmpeg.filter(xstacked, "scale", 3840, 2160)
              .filter("fps", fps=60, round="up")
    )
    stop_duration = video_end_time if video_end_time else min(d for d in durations if d > 0)
    out = ffmpeg.output(
        audio_stream, video_scaled,
        output_file_name_with_extension,
        acodec="aac", vcodec="libx264", pix_fmt="yuv420p",
        crf=21, preset="superfast", progress="pipe:1", to=stop_duration,
    )
    return out.compile(), stop_duration


def generate_grid_command(
    all_video_paths: list[str],
    delays: list[float],
    durations: list[float],
    output_file_name_with_extension: str,
    video_end_time: float | None = None,
    font_path: str | None = None,
) -> tuple[list, float]:
    return generate_video_sync_command(
        all_video_paths, delays, durations,
        output_file_name_with_extension, video_end_time,
    )