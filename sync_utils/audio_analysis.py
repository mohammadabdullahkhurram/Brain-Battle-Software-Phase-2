from typing import List
from moviepy import VideoFileClip
import numpy as np
from scipy import signal
import os

"""
STEPS:

1. FIGURE OUT LONGEST VIDEO
2. FIGURE OUT ALL DELAYS COMPARED TO LONGEST VIDEO
3. SHIFT ALL VIDEOS ACCORDING TO DELAY (TO BE DONE) 
    i.e. if delay = 1s, shift 2nd video forward by 1s
    i.e. if delay = -1s, shift 2nd back by 1s
4. yay :D autosync done

def autosync():
    def find_longest_vid() -> int video_num
    def find_all_delays -> list[floats]
    def shift_all_videos -> None but it shifts using moviepy or whatever gpt cooks
"""

# Function to get the file name from a path
def get_file_name(path):
    file_name = os.path.basename(path)
    file = os.path.splitext(file_name)
    return file[0]

# # Function to initialize and update the Pygame progress bar
# def show_progress_bar(video_path, total_frames, frame_callback):
#     # Initialize Pygame
#     pygame.init()

#     # Set up Pygame window
#     screen_width, screen_height = 500, 100
#     screen = pygame.display.set_mode((screen_width, screen_height))
#     pygame.display.set_caption(f"Processing Audio of {get_file_name(video_path)}")

#     # Colors
#     background_color = (30, 30, 30)
#     progress_bar_color = (0, 150, 255)
#     text_color = (255, 255, 255)

#     # Font
#     font = pygame.font.Font("Fonts/Droid_Sans_Mono_Slashed_400.ttf", 24)

#     # Progress bar variables
#     progress_bar_width = screen_width - 40
#     progress_bar_height = 20
#     progress_x = 20
#     progress_y = 50

#     # Initialize frame count
#     frame_count = 0
#     running = True

#     while running and frame_count < total_frames:
#         # Event loop to handle quitting the window
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#                 pygame.quit()
#                 return

#         # Process the next frame via callback
#         frame_count = frame_callback(frame_count)

#         # Calculate progress
#         progress = frame_count / total_frames

#         # Draw progress bar
#         screen.fill(background_color)
#         progress_text = font.render(f"Processing: {int(progress * 100)}%", True, text_color)
#         screen.blit(progress_text, (20, 10))
#         pygame.draw.rect(
#             screen,
#             progress_bar_color,
#             (progress_x, progress_y, progress * progress_bar_width, progress_bar_height),
#         )

#         # Update display
#         pygame.display.flip()

#     pygame.quit()

def find_longest_vid(video_paths) -> int:
    return_val = 0
    max_duration = -1
    for i, video_path in enumerate(video_paths):
        with VideoFileClip(video_path) as video:
            duration = video.duration
        if duration > max_duration:
            max_duration = duration
            return_val = i

    return return_val

def process_audio(video_path):
    """Extract mono audio via ffmpeg pipe at guaranteed 44100 Hz.
    Avoids MoviePy resampling ambiguity — identical to sync_engine.py approach."""
    import subprocess
    import numpy as _np2
    EXTRACT_SR = 44100
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(EXTRACT_SR),
        "-f", "f32le", "-"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = _np2.frombuffer(result.stdout, dtype=_np2.float32).copy()
    print(f"Audio processing complete for {get_file_name(video_path)} (sr={EXTRACT_SR}, samples={len(audio)}).")
    return audio, EXTRACT_SR


def find_all_delays(video_paths: List[str]) -> List[float]:
    """
        Extracts audio clips from video paths and calculates cross-correlations of the audio signals
        around a pivot video. Returns the amount that the videos are delayed (in seconds) compared to
        the pivot video.

        The point where the cross correlation is the highest is the point where audios match.

        This function chooses the pivot as the video that has the longest duration.

        If you would like to specify the pivot yourself, please see find_all_delays_with_pivot() 

        Arguments:
            - video_paths (list(str)): All video paths as strings.

        Returns:
            - list(float): List of the amount of times that the other videos are delayed compared to
                           the pivot video. The pivot by definition has a delay of 0 seconds.
        
        Notes:
        - Longest video by duration is the pivot.
        - Negative values of delay mean that the videos start earlier than the pivot. This can be solved
          by pushing the pivot by the amount of delay, however since we are matching more than 2 videos
          this is not recommended. Trim the start of the video that is delayed, by the absolute value of the delay,
          to match it with the pivot.
        - Positive delays mean that the video starts later compared to the pivot, and thus should be
          "pushed forward" by the amount of the delay to match with the pivot.
    """

    if len(video_paths) < 2:
        raise ValueError("At least two video paths are required for autosync.")

    delays = [0.0] * len(video_paths)
    longest_index = find_longest_vid(video_paths)
    results = [process_audio(path) for path in video_paths]
    processed_audios = [r[0] for r in results]
    sample_rates     = [r[1] for r in results]

    for i in range(len(video_paths)):
        if i != longest_index:
            corr = signal.correlate(processed_audios[longest_index], 
                                    processed_audios[i], 
                                    mode="same", 
                                    method="fft")
            
            lags = signal.correlation_lags(len(processed_audios[longest_index]), 
                                           len(processed_audios[i]), 
                                           mode="same")
            
            max_corr = np.argmax(corr)
            sr = sample_rates[longest_index]  # use actual sample rate
            time_delay = lags[max_corr] / sr
            delays[i] = time_delay

    return delays

def find_all_durations(video_paths: List[str]) -> List[float]:
    return [VideoFileClip(path).duration for path in video_paths]


def find_all_delays_with_pivot(video_paths: List[str], pivot_index: int) -> List[float]:
    """
        A modified version of the find_all_delays function
        Instead of choosing the longest video as the pivot,
        it takes the index of the pivot as an argument.

        This would allow the user to choose which specific video
        that they want to synchronize the others to.

        Arguments:
            - video_paths (list(str)): All video paths as strings
            - pivot_index (int): the index of the video path, that will be taken as pivot.

        Returns:
            - list(float): List of the amount of times that the other videos are delayed compared to
                           the pivot video. The pivot by definition has a delay of 0 seconds.
        
        Notes:
        - Negative values of delay mean that the videos start earlier than the pivot. This can be solved
          by pushing the pivot by the amount of delay, however since we are matching more than 2 videos
          this is not recommended. Trim the start of the video that is delayed, by the absolute value of the delay,
          to match it with the pivot.
        - Positive delays mean that the video starts later compared to the pivot, and thus should be
          "pushed forward" by the amount of the delay to match with the pivot.
    """
    
    if len(video_paths) < 2:
        raise ValueError("At least two video paths are required for autosync.")

    delays = [0.0] * len(video_paths)
    results = [process_audio(path) for path in video_paths]
    processed_audios = [r[0] for r in results]
    sample_rates     = [r[1] for r in results]

    for i in range(len(video_paths)):
        if i != pivot_index:
            corr = signal.correlate(processed_audios[pivot_index], 
                                    processed_audios[i], 
                                    mode="valid", 
                                    method="fft")
            
            lags = signal.correlation_lags(len(processed_audios[pivot_index]), 
                                           len(processed_audios[i]), 
                                           mode="valid")
            
            max_corr = np.argmax(corr)
            sr = sample_rates[pivot_index]  # use actual sample rate
            time_delay = lags[max_corr] / sr
            delays[i] = time_delay

    return delays

def autosync(video_paths: List[str]) -> List[float]:
    """Main function to calculate and return delays for autosync."""
    
    if not all(os.path.exists(path) for path in video_paths):
        raise FileNotFoundError("One or more video paths are invalid.")
    delays = find_all_delays(video_paths)
    print(f"Delays calculated, the delays are: {delays}")
    return delays