import ffmpeg
import numpy as np
import datetime
import subprocess
import tempfile
import time  # Import time module for delay

def generate_single_preview(video_path: str, delay: float, duration: float, output_path: str, max_delay: float = 0):
    """
        Cuts a given video by its corresponding delay

        Returns:
        - out (str) -> The output command to run with ffmpeg.
        - new_duration (float) -> new duration of the video
    """
    # output_file_name = f"temporary_vid_{datetime.datetime.now().strftime("%d%m%Y%H%M%S")}"
    
    video = ffmpeg.input(video_path)
    audio = video.audio

    new_duration = duration

    # If the delay of the video is negative:
    if delay < 0:
        # We trim the video from the start
        video_cut = ffmpeg.trim(video, start=round(abs(delay), 3)).setpts("PTS-STARTPTS")
        audio_cut = ffmpeg.filter(audio, "atrim", start=round(abs(delay), 3)).filter("asetpts", "PTS-STARTPTS")

        new_duration -= round(delay)

    else: # If the delay is positive
        # We delay/push the videos start by the amount
        video_cut = ffmpeg.filter(video, "tpad", start_duration=round(delay, 3))
        audio_cut = ffmpeg.filter(audio, "adelay", delays=int(round(delay, 3) * 1000), all=1)

    out = ffmpeg.output(audio_cut, video_cut, output_path, acodec="aac", vcodec="libx264", pix_fmt="yuv420p", crf=23, preset="ultrafast", progress="pipe:1", ss=max_delay)

    command = out.compile()

    return command, new_duration

def generate_single_preview_optimized(video_path: str, delay: float, duration: float, output_path: str, max_delay: float = 0):
    """
        DO NOT USE IT IS BUGGY
        Cuts a given video by its corresponding delay

        Returns:
        - out (str) -> The output command to run with ffmpeg.
        - new_duration (float) -> new duration of the video
    """
    # output_file_name = f"temporary_vid_{datetime.datetime.now().strftime("%d%m%Y%H%M%S")}"
    
    video = ffmpeg.input(video_path, hwaccel="cuda", hwaccel_output_format="cuda")
    audio = video.audio

    new_duration = duration

    video_cut_due_to_delay = 0

    # If the delay of the video is negative:
    if delay < 0:
        # We trim the video from the start
        video_cut_due_to_delay = round(abs(delay), 2)

        new_duration -= video_cut_due_to_delay

    resulting_cut = max_delay + video_cut_due_to_delay

    out = ffmpeg.output(audio, video, output_path, acodec="copy", vcodec="h264_nvenc", progress="pipe:1", ss=resulting_cut)

    command = out.compile()

    return command, new_duration

"""
def generate_grid_command(all_video_paths: list[str], delays: list[float], durations: list[float], output_file_name_with_extension: str, video_end_time: float | None = None):
    videos = [ffmpeg.input(path) for path in all_video_paths]

    video_streams = [video.video for video in videos]

    # Handling audio seperately
    audio_streams = [video.audio for video in videos]

    xstacked = ffmpeg.filter(videos, "xstack", inputs="4", layout="0_0|0_h0|w0_0|w0_h0")

    # Trying to split xstack

    # We take the audio stream with the maximum delay.
    maximum_delay = np.argmax(delays)
    audio_stream = audio_streams[maximum_delay]

    # Scale down the video to 1080p 60FPS
    # Normally, we are supposed to output 4K 60FPS
    video_scaled = ffmpeg.filter(xstacked, "scale", 3840, 2160).filter('fps', fps=60, round='up')

    # The output uses ultrafast preset for demonstration purposes.
    # The video will continue until the shortest video ends, i.e set stop_duration = min(durations)
    # If you want to test it out for a shorter time, change the "stop_duration" to amount of seconds you want (e.g 10 secs)
    if video_end_time:
        stop_duration = video_end_time
    else:
        stop_duration = min(durations)

    out = ffmpeg.output(audio_stream, video_scaled, output_file_name_with_extension, acodec="aac", vcodec="libx264", pix_fmt="yuv420p", crf=24, preset="superfast", progress="pipe:1", to=stop_duration)

    # Compile command will return the terminal command to run FFMPEG as a string
    command = out.compile()
    resulting_duration = stop_duration

    return command, resulting_duration
"""

def generate_grid_command(all_video_paths: list[str], delays: list[float], durations: list[float], output_file_name_with_extension: str, video_end_time: float | None = None):
    videos = [ffmpeg.input(path) for path in all_video_paths]
    xstacked = ffmpeg.filter(videos, "xstack", inputs=4, layout="0_0|0_h0|w0_0|w0_h0")

    max_delay_idx = np.argmax(delays)
    audio_stream = videos[max_delay_idx].audio

    video_scaled = (
        xstacked
        .filter("scale", 3840, 2160)
        .filter("fps", fps=60, round="up")
    )

    text_string = (
        "Time: %{eif:trunc(t/60):d}m" 
        "%{eif:mod(t,60):d}s"          
        "%{eif:mod(t*1000,1000):d}ms "
        "Frame: %{n}"
    )

    video_with_timestamp = ffmpeg.filter(
        video_scaled,
        "drawtext",
        fontfile="C:/Users/Asus/OneDrive/Desktop/CIP/Synchronization/downloaded_fonts/Droid_Sans_Mono_Slashed_400.ttf",
        text=text_string,
        x="(w-tw)/2",
        y="h-(2*lh)",
        fontsize="72",
        fontcolor="white",
        box="1",
        boxcolor="black@0.4",
        expansion="normal"
    )

    stop_duration = video_end_time if video_end_time else min(durations)

    out = (
        ffmpeg
        .output(
            audio_stream,
            video_with_timestamp,
            output_file_name_with_extension,
            acodec="aac",
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=21,
            preset="superfast",
            progress="pipe:1",
            to=stop_duration
        )
    )

    command = out.compile()
    print("\n--- Compiled FFmpeg Command ---")
    print(command)
    
    time.sleep(5)
    
    return command, stop_duration

def generate_video_sync_command(all_video_paths: list[str], delays: list[float], durations: list[float], output_file_name_with_extension: str, video_end_time: float | None = None):
    """
        Generate the FFMPEG command that can be used to
        synchronize and put all four videos into a grid.

        The function takes all video paths, their delays, and their durations.
        It applies FFMPEG filters trim, and tpad accordingly to match the videos with each other.
        After which, it exports it to 4K 60 fps video.

        Please be mindful that indexes should be consistent between arrays, if first videos path is at index 0,
        its delay and duration should be at index 0 in the respective lists.
        
        Arguments:
        - all_video_paths (list[str]): List of paths that lead to the videos that should be processed
        - delays (list[float]): List of the delays for each video.
        - durations (list[float]): List of the durations of each video.
        - (optional) video_end_time (float): specify the end time of the video so that FFMPEG does not process the whole video.
          can be used for testing purposes.

        Returns:
        - The FFMPEG terminal command that should be passed to the FFMPEG subprocess.
        - The duration of the resulting video
    """
    # Load videos into ffmpeg
    videos = [ffmpeg.input(path) for path in all_video_paths]

    video_streams = [video.video for video in videos]

    # Handling audio seperately
    audio_streams = [video.audio for video in videos]

    # We handle each video individually.
    for index, video in enumerate(video_streams):

        # If the delay of the vide is negative:
        if delays[index] < 0:
            # We trim the video from the start
            video_streams[index] = ffmpeg.trim(video, start=round(abs(delays[index]), 2)).setpts("PTS-STARTPTS")
            
            # Update the corresponding video's duration
            durations[index] -= round(delays[index])

        else: # If the delay is positive
            # We delay/push the videos start by the amount
            video_streams[index] = ffmpeg.filter(video, "tpad", start_duration=round(delays[index], 2))

            audio_streams[index] = ffmpeg.filter(audio_streams[index], "apad", pad_dur=round(delays[index], 2))
            
    # Put the videos on grid 2x2
    xstacked = ffmpeg.filter(video_streams, "xstack", inputs="4", layout="0_0|0_h0|w0_0|w0_h0")

    # Trying to split xstack

    # We cut the whole video by the longest positive delay
    # So that the videos begin at the same time.
    video_start_adjusted = ffmpeg.trim(xstacked, start=max(delays)).setpts("PTS-STARTPTS")

    # We take the audio stream with the maximum delay.
    maximum_delay = np.argmax(delays)
    audio_stream = audio_streams[maximum_delay]

    # Cut the start of audio similarly
    # This works because there will always be at least one 0 or larger delay.
    # Thus no need to account for negatives.
    audios_start_adjusted = ffmpeg.filter(audio_stream, "atrim", start=max(delays)).filter("asetpts", "PTS-STARTPTS")

    # Scale down the video to 1080p 60FPS
    # Normally, we are supposed to output 4K 60FPS
    video_scaled = ffmpeg.filter(video_start_adjusted, "scale", 3840, 2160).filter('fps', fps=60, round='up')

    # The output uses ultrafast preset for demonstration purposes.
    # The video will continue until the shortest video ends, i.e set stop_duration = min(durations)
    # If you want to test it out for a shorter time, change the "stop_duration" to amount of seconds you want (e.g 10 secs)
    if video_end_time:
        stop_duration = video_end_time
    else:
        stop_duration = min(durations)

    out = ffmpeg.output(audio_stream, video_scaled, output_file_name_with_extension, acodec="aac", vcodec="libx264", pix_fmt="yuv420p", crf=21, preset="superfast", progress="pipe:1", to=stop_duration)

    # Compile command will return the terminal command to run FFMPEG as a string
    command = out.compile()
    resulting_duration = stop_duration - max(delays)

    return command, resulting_duration

def run_ffmpeg_subprocess(ffmpeg_command: str, resulting_duration: float, debug: bool = False):
    import subprocess, datetime
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
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

    end_time = str(datetime.timedelta(seconds=resulting_duration))
    print("FFmpeg starting...")
    for line in ffmpeg_process.stdout:
        if debug:
            print(line, end="")
        if line.startswith("out_time="):
            progress = line.split("=")[1].strip()
            print(f"Progress: {progress} / {end_time}")


    print("FFmpeg finished.")

    ffmpeg_process.terminate()

    # IMPORTANT: you have to quit the FFMPEG process itself from Task Manager to stop it
    # It will also not continue if the file exists already, make sure to delete the file if it looks like its doing nothing.


def get_first_frame_pts(video_path: str) -> float:
    """
    Return the PTS (presentation timestamp) of the first video frame in seconds.
    Used by MainEditorSyncWorker to account for files that don't start at t=0.
    Falls back to 0.0 on any error.
    """
    import subprocess, json
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time",
            "-of", "json",
            "-read_intervals", "%+#1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        packets = data.get("packets", [])
        if packets:
            return float(packets[0].get("pts_time", 0.0))
    except Exception:
        pass
    return 0.0