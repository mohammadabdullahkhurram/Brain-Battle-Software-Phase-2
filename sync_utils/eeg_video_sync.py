import ffmpeg
import pandas as pd
import datetime as dt
import os

def compare_video_eeg(video_path: str, csv_path: str, video_duration: float) -> tuple[float, float]:
    """
        Compares EEG data's timecode with that of the video.

        Returns the new start and end time of the video to match it with the EEG.

        Note that for this to work, the video should have accurate timecode.

        Arguments:
        - video_path (str): Path to the video.
        - eeg_path (str): Path to the CSV file that contains EEG data.
        - video_duration (float): Duration of the video in seconds.

        Returns:
        - start_time (float): New start time of the video.
        - end_time (float): New end time of the video.

        TODO Implement matching the last timecode of the EEG with the video.
    """

    # Extract video creation time
    # Note that the metadata might change depending on the camera.
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

    # Convert datetime into timestamp.
    video_creation_dt = dt.datetime.strptime(creation_time, "%Y-%m-%dT%H:%M:%S.%f%z")
    video_creation_timestamp = video_creation_dt.timestamp()

    # Read the CSV
    df = pd.read_csv(csv_path)

    # Extract initial timestamp from the CSV
    initial_csv_timestamp = df["timestamps"][0]
    last_csv_timestamp = df["timestamps"].iloc[-1]
        
    # Calculate the adjustment needed
    time_difference = initial_csv_timestamp - video_creation_timestamp

    # EEG data is the last thing that will start before session starts.
    # Therefore 99% of the time the above calculation will be a positive value.

    # The new length of the video
    # We will add it to start time to find the end time
    new_duration = last_csv_timestamp - initial_csv_timestamp

    if time_difference > 0:
        # Video starts later than CSV
        start_time = abs(time_difference)
        end_time = start_time + new_duration
    else:
        # Note: This does not really make sense,
        # Someone should check it out.
        # Video starts earlier or at the same time as CSV
        start_time = 0
        end_time = video_duration - abs(time_difference)

    return (start_time, end_time)

def cut_video_from_start_end(video_path: str, start_time: float, end_time: float, temp_output_path: str) -> str:
    """
        Opens a FFMPEG subprocess and cuts the video provided
        so that it starts at start_time seconds and ends at end_time.

        This function will check if the file already exists in the folder and remove it if necessary.

        Opens a pipe to get updates from the FFMPEG subprocess and prints it to standard output.

        This function blocks until FFMPEG subprocess is done.

        Arguments:
        - video_path (str): Path to the video
        - start_time (float): When to start the video, in seconds
        - end_time (float): When to end the video, in seconds

        
        TODO Check initially if start and end times are valid.
        TODO Better temp file handling
    """


    # current_time = dt.datetime.now().strftime("%d%m%Y%H%M%S")
    # temp_file_name = f"temporary_{current_time}"

    eeg_sync_video = ffmpeg.input(video_path, ss=start_time, to=end_time)
    out = ffmpeg.output(eeg_sync_video, temp_output_path, vcodec="copy", acodec="copy").compile()
    import subprocess
    import datetime # For printing progress
    import os

    # Run the FFMPEG subprocess
    try:
        ffmpeg_process = subprocess.Popen(out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    except OSError:
        print("OSError")
        return
    except ValueError:
        print("ValueError: Cannot run FFMPEG with given parameters.")
        return

    end_time = str(datetime.timedelta(seconds=end_time)) # Used for printing the last point of the video
    # Print the progress that ffmpeg outputs
    print("FFMPEG is starting, please wait a couple minutes.")
    for line in ffmpeg_process.stdout:
        # for all output from ffmpeg
        print(line)

        # If you would like to print it in a simpler format
        # This shows how much of the video you processed
        all_info = line.split(" ") 
        for elem in all_info:
            if elem.startswith("time"):
                progress = elem.split("=")[1]
                print(f"Progress: currently at {progress} / going until {end_time}", end="\r")

    print("FFMPEG has finished processing.")