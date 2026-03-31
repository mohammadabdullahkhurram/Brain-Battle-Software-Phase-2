import ffmpeg
import pandas as pd
import datetime as dt
import subprocess
import os


def _parse_creation_time(creation_time: str) -> float:
    """
    Parse an ISO-8601 creation_time string from ffmpeg metadata into a Unix timestamp.

    Handles both formats written by cameras and phones:
      - With fractional seconds:  "2025-03-25T14:03:00.123456Z"
      - Without fractional seconds: "2025-03-25T14:03:00Z"

    Raises ValueError if neither format matches.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return dt.datetime.strptime(creation_time, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(
        f"creation_time '{creation_time}' does not match any known ISO-8601 format. "
        "Expected 'YYYY-MM-DDTHH:MM:SS[.ffffff]+HH:MM' or 'YYYY-MM-DDTHH:MM:SSZ'."
    )


def compare_video_eeg(video_path: str, csv_path: str, video_duration: float) -> tuple[float, float]:
    """
    Compares EEG data's timecode with that of the video.

    Returns the new start and end time (in seconds) to cut the video so that
    it covers exactly the same real-world time window as the EEG recording.

    Note that for this to work, the video should have accurate creation_time metadata.

    Arguments:
    - video_path (str): Path to the video.
    - csv_path (str): Path to the CSV file that contains EEG data (muselsl format,
                      must have a 'timestamps' column of Unix epoch floats).
    - video_duration (float): Duration of the video in seconds.

    Returns:
    - start_time (float): New start time of the video (seconds from video start).
    - end_time (float): New end time of the video (seconds from video start).
    """

    # ── Extract video creation time ───────────────────────────────────────────
    # Note: metadata tag location varies by camera/encoder; check streams first,
    # then the format container.
    video_metadata = ffmpeg.probe(video_path)
    creation_time = None
    for stream in video_metadata.get("streams", []):
        ct = stream.get("tags", {}).get("creation_time")
        if ct:
            creation_time = ct
            break
    if not creation_time:
        creation_time = video_metadata.get("format", {}).get("tags", {}).get("creation_time")
    if not creation_time:
        raise ValueError("No creation_time found in video metadata")

    # FIX: use robust parser that handles both .ffffff and no-fractional-seconds forms
    video_creation_timestamp = _parse_creation_time(creation_time)

    # ── Read EEG CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    if "timestamps" not in df.columns:
        raise ValueError(f"EEG CSV '{csv_path}' has no 'timestamps' column.")

    initial_csv_timestamp = float(df["timestamps"].iloc[0])
    last_csv_timestamp    = float(df["timestamps"].iloc[-1])

    # Duration of the EEG recording itself
    eeg_duration = last_csv_timestamp - initial_csv_timestamp

    # time_difference > 0 → EEG started BEFORE the video
    # time_difference < 0 → EEG started AFTER the video (rare, but handled)
    time_difference = initial_csv_timestamp - video_creation_timestamp

    if time_difference >= 0:
        # ── Common case: EEG was already running when the camera started ──────
        # The EEG has |time_difference| seconds of data before the video begins.
        # We seek the video to t=0 (its natural start) and run for eeg_duration,
        # clamped to the actual video length.
        start_time = 0.0
        end_time   = min(eeg_duration, video_duration)
    else:
        # ── EEG started after the camera ──────────────────────────────────────
        # |time_difference| seconds of video elapsed before EEG began.
        # Seek the video forward to the point the EEG started, then run for
        # eeg_duration, clamped so we never exceed the video's end.
        start_time = abs(time_difference)
        end_time   = min(start_time + eeg_duration, video_duration)

    return (start_time, end_time)


def cut_video_from_start_end(video_path: str, start_time: float, end_time: float, temp_output_path: str) -> None:
    """
    Cuts the video so that it starts at start_time seconds and ends at end_time.

    Removes the output file first if it already exists (ffmpeg refuses to overwrite
    by default, and the -y flag is not always reliable across platforms).

    Blocks until ffmpeg finishes.

    Arguments:
    - video_path (str): Path to the source video.
    - start_time (float): Start point in seconds (must be >= 0).
    - end_time (float): End point in seconds (must be > start_time).
    """
    # ── Validate times ────────────────────────────────────────────────────────
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time:.3f}")
    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time:.3f}) must be greater than start_time ({start_time:.3f})"
        )

    # ── Remove stale output if present ───────────────────────────────────────
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    eeg_sync_video = ffmpeg.input(video_path, ss=start_time, to=end_time)
    out_cmd = ffmpeg.output(eeg_sync_video, temp_output_path, vcodec="copy", acodec="copy").compile()

    end_time_str = str(dt.timedelta(seconds=end_time))

    print("FFMPEG is starting, please wait.")
    try:
        ffmpeg_process = subprocess.Popen(
            out_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except OSError:
        print("OSError: could not launch ffmpeg.")
        return
    except ValueError as exc:
        print(f"ValueError: Cannot run FFMPEG with given parameters. {exc}")
        return

    for line in ffmpeg_process.stdout:
        print(line, end="")
        for elem in line.split(" "):
            if elem.startswith("time="):
                progress = elem.split("=")[1].strip()
                print(f"Progress: currently at {progress} / going until {end_time_str}", end="\r")

    ffmpeg_process.wait()
    print("\nFFMPEG has finished processing.")
