# BRAIN BATTLE
### Neuroscience Sports Capture System
**Constructor University Bremen — Community Impact Project**
`v1.0  //  Software Phase 2`

---

## Table of Contents

1. [Overview](#1-overview)
2. [File Structure](#2-file-structure)
3. [Installation](#3-installation)
4. [Hardware Setup](#4-hardware-setup)
5. [Tab 01 — Live Monitor](#5-tab-01--live-monitor)
6. [Tab 02 — Sync + Review](#6-tab-02--sync--review)
7. [Tab 03 — Labelling](#7-tab-03--labelling)
8. [Label Configuration](#8-label-configuration)
9. [Sync Engine — Technical Reference](#9-sync-engine--technical-reference)
10. [Module Reference](#10-module-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Known Limitations](#12-known-limitations)
13. [Dependencies](#13-dependencies)
14. [Project Information](#14-project-information)

---

## 1. Overview

Brain Battle is a CIP (Community Impact Project) developed at Constructor University Bremen. The system captures, synchronises, reviews, and labels neuroscience data collected during boxing sessions — specifically EEG brain activity from Muse-S headbands worn by two boxers, combined with multi-angle camera footage from up to four cameras.

The goal is to provide researchers with a complete, time-aligned dataset: every punch, clinch, knockdown, or round event is timestamped against both the EEG signals and all camera angles, enabling post-session analysis of the neurological correlates of combat sports performance.

### 1.1 What the System Records

- EEG brainwave data (delta, theta, alpha, beta, gamma bands) from **2x Muse-S headbands** worn by both boxers simultaneously
- Video footage from up to **4 cameras**: 2x DSLR cameras and 2x action/360 cameras
- **Event labels** placed manually during review: punch types, round markers, dominant fighter periods, referee stoppages
- **Synchronisation offsets** between all sources so every timestamp maps to a common global timeline

### 1.2 Workflow at a Glance

```
1. Hardware setup      →  Connect Muse-S headbands, configure cameras
2. Tab 01 LIVE MONITOR →  Start session, monitor EEG + live camera feeds
3. Tab 02 SYNC+REVIEW  →  Upload files, AUTO-SYNC, review synced playback
4. Tab 03 LABELLING    →  Place event labels on the timeline
5. Export              →  Quad-grid merged video + labels CSV + ZIP archive
```

---

## 2. File Structure

All files must be in the same directory.

```
Software Phase 2/
├── brain_battle_main.py   # Main application — all three tab UIs and main window
├── sync_engine.py         # Audio cross-correlation sync engine + SyncWorker QThread
├── eeg_module.py          # Live EEG streaming (EEGLiveWidget) + review (EEGReviewWidget)
├── rtmp_module.py         # RTMP live stream viewer using ffplay (Tab 01 camera feeds)
├── vlc_module.py          # VLC video playback slots + SyncPlaybackController (Tabs 02/03)
├── export_module.py       # ffmpeg quad-grid video export, labels CSV, ZIP packaging
├── labels_config.py       # Static label definitions — read on startup, never modified
├── labels_data.json       # Auto-generated — live label palette state (created on first run)
└── Muse Recordings/       # Auto-created — EEG CSV output from recording sessions
```

> **Note:** `labels_data.json` is created automatically the first time you add or remove a label in the app. If it does not exist, the app falls back to the defaults in `labels_config.py`.

---

## 3. Installation

### 3.1 System Requirements

- macOS (primary target), Windows, or Linux
- Python 3.10 or higher
- `ffmpeg` installed and on PATH
- VLC Media Player installed (for video playback in Tabs 02 and 03)
- Bluetooth adapter (for Muse-S EEG headband connection)

### 3.2 Python Dependencies

```bash
pip install PyQt6 matplotlib pylsl muselsl scipy pandas ffmpeg-python numpy python-vlc
```

On macOS, install `ffmpeg` and VLC via Homebrew:

```bash
brew install ffmpeg
brew install --cask vlc
```

### 3.3 MonaServer2 (RTMP Live Camera Streams)

The RTMP camera feed feature in Tab 01 requires **MonaServer2** as a local RTMP relay. You need one instance per camera (four total for a full setup). Download from the [MonaServer2 GitHub repository](https://github.com/MonaSolutions/MonaServer2).

Configure each instance with a unique port by editing `MonaServer.ini`:

| Instance | Port | Camera      |
|----------|------|-------------|
| Mona 1   | 1935 | DSLR 01     |
| Mona 2   | 9999 | DSLR 02     |
| Mona 3   | 9998 | 360 Cam 01  |
| Mona 4   | 9997 | 360 Cam 02  |

> **Warning:** eduroam and other enterprise WiFi networks block RTMP ports. Use a personal hotspot or regular WiFi router.

### 3.4 Running the App

```bash
python3 brain_battle_main.py
```

---

## 4. Hardware Setup

### 4.1 Muse-S EEG Headbands

Brain Battle is pre-configured for two specific Muse-S devices:

| Device Name  | Role    | Widget   |
|--------------|---------|----------|
| `MuseS-7538` | BOXER A | EEG 1    |
| `MuseS-7564` | BOXER B | EEG 2    |

**To connect a headband:**

1. Power on the Muse-S (hold button until LED blinks)
2. In Tab 01, click **SELECT DEVICE** on the EEG widget
3. Click **SCAN FOR DEVICES** — scan takes up to 20 seconds
4. Select the device from the list and click **CONNECT**
5. Status dot turns green when the LSL stream is live

> **Warning:** If the device disconnects mid-recording, the app shows a red **DISCONNECTED** banner and automatically saves whatever EEG data was captured. Click **RECONNECT** to re-establish the stream.

### 4.2 DSLR Cameras (Lumix / Sony)

DSLR cameras do not natively support RTMP streaming. Use an **HDMI capture card** (e.g. Elgato Cam Link 4K or a generic USB capture card):

```
Camera HDMI out  →  Capture card  →  Mac USB  →  ffmpeg  →  MonaServer  →  App
```

**Steps:**

1. Set camera to **Clean HDMI output** (disable overlays and menus on HDMI output)
2. Plug capture card into Mac USB
3. Verify macOS sees it: QuickTime → New Movie Recording → check dropdown
4. Find device index:
   ```bash
   ffmpeg -f avfoundation -list_devices true -i ""
   ```
5. Push feed to MonaServer:
   ```bash
   # Camera 1 (capture card at index 1)
   ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "1" \
     -vcodec libx264 -preset ultrafast -tune zerolatency \
     -f flv rtmp://localhost:1935

   # Camera 2 (capture card at index 2)
   ffmpeg -f avfoundation -framerate 30 -video_size 1920x1080 -i "2" \
     -vcodec libx264 -preset ultrafast -tune zerolatency \
     -f flv rtmp://localhost:9999
   ```

### 4.3 Insta360 GO 3S

The Insta360 GO 3S **does not support RTMP live streaming**. It records locally to its SD card only. The 360 camera slots in Tab 01 are reserved for future FPV cameras. Use GO 3S recordings as offline sources uploaded in Tab 02 after the session.

---

## 5. Tab 01 — Live Monitor

The Live Monitor tab is the primary interface during an active boxing session. It shows live camera feeds, real-time EEG waveforms, and controls session recording.

### 5.1 Camera Grid

The 2×2 grid shows four `RTMPSlot` panels. Each slot displays the RTMP URL, connection status, and the live video feed once connected.

| Control         | Description                                                              |
|-----------------|--------------------------------------------------------------------------|
| `LOCAL IP`      | Auto-detected on startup. All RTMP URLs are pre-filled using this IP.    |
| `CONNECT ALL`   | Connects all four slots simultaneously. Toggles to DISCONNECT ALL.       |
| URL field       | Editable per slot. Change IP or port for custom MonaServer configurations.|
| `CONNECT` btn   | Per-slot toggle. Shows **RECONNECT** in red if stream is lost.           |

> **Note (macOS):** ffplay opens each stream in a floating window rather than embedding in the slot panel — macOS does not support foreign window embedding. The slot shows a green **LIVE** indicator when the stream is active.

### 5.2 EEG Streams

Two `EEGLiveWidget` panels sit below the camera grid — one per Muse-S headband. Toggle between:

- **WAVEFORM** — scrolling 30-second rolling trace of all four EEG channels (TP9, AF7, AF8, TP10)
- **BAND POWER** — live bar chart of relative delta/theta/alpha/beta/gamma power, computed from the last 4 seconds using Welch PSD

### 5.3 Session Recording

1. Enter a session name in the text field (e.g. `Round 1`)
2. Click **START RECORDING** — button turns red and shows **STOP RECORDING**
3. The REC indicator illuminates and the session timer starts
4. Click **STOP RECORDING** when done

EEG recordings are saved to `Muse Recordings/` (auto-created next to `brain_battle_main.py`):

```
Muse Recordings/DD.MM.YYYY HH-MM TZ SessionName Muse S XXXX.csv
```

---

## 6. Tab 02 — Sync + Review

Upload all recorded files here to align them to a common timeline and review footage with synchronised EEG overlay.

### 6.1 Uploading Sources

Six source rows in the left panel — four for video, two for EEG CSV:

| Source         | Accepted Formats        |
|----------------|-------------------------|
| DSLR 01 / 02   | `.mp4`, `.mov`, `.avi`  |
| INSTA360 01/02 | `.mp4`, `.mov`, `.avi`  |
| MUSE-S 01 / 02 | `.csv` (Muse-S output)  |

At least **two video files** must be uploaded before AUTO-SYNC can run. EEG files are optional.

### 6.2 AUTO-SYNC

Click **AUTO-SYNC** to compute synchronisation offsets. The sync engine runs in a background thread and reports progress via the status bar.

#### Video-to-Video Sync

1. Extracts mono audio from each video via `ffmpeg-python`
2. Designates the longest video as the reference
3. Computes **FFT cross-correlation** of each video's audio against the reference using `scipy.signal`
4. The cross-correlation peak position gives the time offset in seconds
5. Normalised peak height is used as the confidence metric

#### EEG-to-Video Sync

Muse-S CSVs contain Unix timestamps. Camera files contain `creation_time` metadata (written at record-start). The EEG offset is:

```
eeg_global_offset = (video_creation_time_unix - eeg_first_timestamp_unix) - video_xcorr_offset
```

#### Confidence Badges

| Badge        | Meaning                                          |
|--------------|--------------------------------------------------|
| `✓` (green)  | High confidence — sharp peak, reliable offset    |
| `~` (yellow) | Medium confidence — usable but review recommended |
| `?` (red)    | Low confidence — poor audio overlap              |

### 6.3 Synchronised Review

After sync completes, video slots are populated and the EEG overlay strips are loaded.

| Control             | Description                                              |
|---------------------|----------------------------------------------------------|
| `▶  PLAY / ⏸  PAUSE`| Play/pause all four video slots in lockstep              |
| Timeline slider     | Scrub all videos and EEG playheads simultaneously        |
| Timecode display    | Current global timeline position in `HH:MM:SS.mmm`      |
| Duration display    | Total session duration from the longest source           |

The `SyncPlaybackController` polls the reference player every 100ms and re-seeks drifted players if drift exceeds **150ms**.

### 6.4 Exporting Synced Data

Click **EXPORT SYNCED DATA** to package the session as a named ZIP. The ZIP contains:

```
export_name.zip
├── merged_video.mp4    # Quad-grid 2×2 composite, H.264 encoded
├── labels.csv          # Event labels placed in Tab 03
├── sync_offsets.csv    # Per-source offsets, methods, confidence scores
├── eeg/
│   ├── eeg_01.csv      # Original Muse-S recording — Boxer A
│   └── eeg_02.csv      # Original Muse-S recording — Boxer B
├── videos/
│   ├── video_01.mp4    # Original camera files
│   └── ...
└── README.txt          # Session metadata
```

> **Note:** The quad-grid video merge runs via `ffmpeg` in a background thread with a progress dialog. For long sessions this can take several minutes. If `ffmpeg` is not found the video merge is skipped; all other files are still zipped.

### 6.5 Proceed to Labelling

Click **PROCEED TO LABELLING** to carry all loaded videos, EEG data, and sync offsets into Tab 03. No re-uploading is needed.

---

## 7. Tab 03 — Labelling

Frame-by-frame review with event label placement. All four camera angles and both EEG streams play back simultaneously.

### 7.1 Label Palette

The left panel shows the label palette. Click any label to select it. Predefined labels:

| Label              | Colour               | Description                 |
|--------------------|----------------------|-----------------------------|
| `ROUND_START`      | `#22c55e` (green)    | Start of a boxing round     |
| `ROUND_END`        | `#ef4444` (red)      | End of a boxing round       |
| `PUNCH_JAB`        | `#3b82f6` (blue)     | Jab punch thrown            |
| `PUNCH_CROSS`      | `#60a5fa` (lt. blue) | Cross punch thrown          |
| `PUNCH_HOOK`       | `#818cf8` (indigo)   | Hook punch thrown           |
| `PUNCH_UPPERCUT`   | `#a78bfa` (purple)   | Uppercut thrown             |
| `CLINCH`           | `#f59e0b` (amber)    | Fighters in clinch          |
| `KNOCKDOWN`        | `#ef4444` (red)      | Boxer knocked down          |
| `CORNER_CUT`       | `#6b7280` (grey)     | Corner cut between rounds   |
| `REF_STOPPAGE`     | `#f97316` (orange)   | Referee stops the fight     |
| `DOMINANT_BOXER_A` | `#06b6d4` (teal)     | Boxer A is dominant         |
| `DOMINANT_BOXER_B` | `#ec4899` (pink)     | Boxer B is dominant         |

### 7.2 Adding and Removing Labels

- **Add:** Type a name in the input field at the bottom of the palette and press **Enter** or **+**. Names are auto-uppercased, spaces become underscores. A colour is auto-assigned from a 12-colour rotation pool (never repeats a colour already in use).
- **Remove:** Click **×** next to any chip. The change is saved immediately to `labels_data.json`.

> **Warning:** Removing a label from the palette does not remove already-placed instances of that label in the current session.

### 7.3 Placing Labels

1. Select a label from the palette
2. Navigate to the event using the timeline slider or playback controls
3. Click **PLACE LABEL AT PLAYHEAD**

Each entry in the **PLACED LABELS** panel shows a colour dot, timestamp (`HH:MM:SS.mmm`), and label name. Click **×** to remove. The counter in the header updates automatically.

### 7.4 Playback Controls

| Control             | Description                                    |
|---------------------|------------------------------------------------|
| `⏮`                 | Rewind to beginning (seeks to `t=0`)           |
| `▶  PLAY / ⏸  PAUSE`| Play/pause all videos and EEG streams          |
| `⏭`                 | Skip forward                                   |
| Timeline slider     | Seek all sources to any point                  |

### 7.5 Exporting Labels

Click **EXPORT LABELS CSV** in the toolbar. Output columns:

| Column          | Description                                             |
|-----------------|---------------------------------------------------------|
| `timestamp_sec` | Event time as a float (seconds from session start)      |
| `timestamp_str` | Human-readable `HH:MM:SS.mmm` format                   |
| `label`         | Label name string                                       |
| `color`         | Hex colour code of the label                            |

---

## 8. Label Configuration

Labels are managed through two files:

### 8.1 `labels_config.py` — Static Defaults

Defines the default label set loaded when `labels_data.json` does not exist. **This file is never modified by the running application.** Edit it directly to change the factory defaults.

To add a permanent default label, append to `_DEFAULT_LABELS`:

```python
('MY_LABEL', '#hexcolour'),
```

To reset to factory defaults, delete `labels_data.json`.

### 8.2 `labels_data.json` — Live State

Auto-generated JSON array of `[name, colour]` pairs representing the current palette:

```json
[
  ["ROUND_START", "#22c55e"],
  ["PUNCH_JAB", "#3b82f6"],
  ...
]
```

Written by `save()` in `labels_config.py` every time a label is added or removed.

### 8.3 Auto-Colour Pool

New labels are assigned the next unused colour from this rotation:

```
#e8ff00  #06b6d4  #34d399  #f97316
#a78bfa  #f43f5e  #38bdf8  #fb923c
#4ade80  #c084fc  #fbbf24  #67e8f9
```

---

## 9. Sync Engine — Technical Reference

### 9.1 Algorithm

Implemented in `sync_engine.py`, runs inside a `SyncWorker(QThread)`.

**Step 1 — Audio Extraction**
`ffmpeg-python` extracts mono audio from each video as a `float32` PCM array resampled to 44,100 Hz. `creation_time` metadata is extracted for EEG alignment.

**Step 2 — Reference Selection**
The longest video is the reference. All other offsets are computed relative to it.

**Step 3 — Cross-Correlation**
```python
# Simplified
correlation = scipy.signal.fftconvolve(ref_audio, other_audio[::-1])
offset_samples = correlation.argmax() - (len(other_audio) - 1)
offset_sec = offset_samples / sample_rate
confidence = correlation.max() / correlation.std()
```

**Step 4 — EEG Alignment**
```
eeg_global_offset = (video_creation_time_unix - eeg_first_timestamp_unix)
                    - video_xcorr_offset
```

### 9.2 Global Timeline Conversion

```python
# Source-local  →  global timeline
global_t = source_local_t - source_offset

# Global timeline  →  video seek position
local_seek_time = global_t + video_offset
```

### 9.3 SyncResult Object

```python
@dataclass
class SyncResult:
    video_offsets:    list[SourceOffset]
    eeg_offsets:      list[SourceOffset]
    global_duration:  float              # seconds

@dataclass
class SourceOffset:
    source_id:    str
    offset_sec:   float
    confidence:   float
    method:       str    # 'xcorr' or 'metadata'
    is_reference: bool
```

---

## 10. Module Reference

### `eeg_module.py`

| Class / Function      | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `EEGLiveWidget`       | Live EEG display. Handles scanning, connection, streaming, and recording.   |
| `EEGReviewWidget`     | Offline EEG viewer. Loads a DataFrame; responds to `set_playhead()`.        |
| `_LSLReaderThread`    | QThread reading from `pylsl.StreamInlet`. Detects disconnect after 5s silence.|
| `_MuseScanThread`     | QThread running `muselsl list` as subprocess to find nearby devices.        |
| `DevicePickerDialog`  | Modal device selector. Already-claimed devices are greyed out.              |
| `_RecordStopper`      | QThread sending `SIGINT` to `muselsl record` to flush CSV before exit.      |

### `rtmp_module.py`

| Class / Function   | Description                                                               |
|--------------------|---------------------------------------------------------------------------|
| `RTMPSlot`         | Single camera slot. Launches `ffplay` with low-latency flags.             |
| `RTMPConfigPanel`  | 2×2 grid of `RTMPSlot`s with auto-detected local IP and CONNECT ALL toggle.|
| `_StreamMonitor`   | QThread watching the `ffplay` subprocess. Emits `disconnected()` on exit. |
| `get_local_ip()`   | Auto-detects local WiFi IP using a UDP socket.                            |

**ffplay flags used (matching existing Brain Battle shell scripts):**
```
-an -flags low_delay -fflags nobuffer -framedrop
-strict experimental -probesize 32 -analyzeduration 0 -sync ext
```

### `vlc_module.py`

| Class / Function            | Description                                                          |
|-----------------------------|----------------------------------------------------------------------|
| `VLCVideoSlot`              | VLC-backed video panel. Handles load/play/pause/seek with offsets.   |
| `SyncPlaybackController`    | Owns ≤4 slots. 100ms timer re-syncs drifted players (tolerance 150ms).|
| `_attach_player_to_widget`  | Platform-aware surface binding: `set_nsobject` / `set_hwnd` / `set_xwindow`.|

### `export_module.py`

| Class                       | Description                                                             |
|-----------------------------|-------------------------------------------------------------------------|
| `VideoExportWorker`         | QThread. `ffmpeg filter_complex` 2×2 grid merge. Parses `time=` for progress. |
| `LabelsExporter`            | Synchronous CSV writer (`timestamp_sec`, `timestamp_str`, `label`, `color`). |
| `FullSessionExportWorker`   | QThread. Orchestrates video + labels + offsets + EEG + originals → ZIP. |
| `ExportProgressDialog`      | Modal progress dialog. Live status + progress bar. Closeable on finish.  |

**ffmpeg export command structure:**
```bash
ffmpeg -ss <offset> -t <duration> -i video1 \
       -ss <offset> -t <duration> -i video2 \
       -ss <offset> -t <duration> -i video3 \
       -ss <offset> -t <duration> -i video4 \
       -filter_complex "
         [0:v]scale=960:540,pad=960:540,setsar=1[v0];
         [1:v]scale=960:540,pad=960:540,setsar=1[v1];
         [2:v]scale=960:540,pad=960:540,setsar=1[v2];
         [3:v]scale=960:540,pad=960:540,setsar=1[v3];
         [v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]
       " \
       -map "[out]" -c:v libx264 -preset fast -crf 23 output.mp4
```

---

## 11. Troubleshooting

| Issue                          | Resolution                                                                 | Notes                          |
|--------------------------------|----------------------------------------------------------------------------|--------------------------------|
| EEG device not found           | Run `muselsl list` in terminal. Check Bluetooth is on and headband is powered. | Scan timeout is 20 seconds     |
| EEG recording not saved        | Ensure `ffmpeg` is installed. Check `Muse Recordings/` folder exists.      | App creates folder automatically|
| Stream disconnects mid-session | App saves data to that point. Click **RECONNECT** in the EEG slot.         | Red banner appears              |
| AUTO-SYNC fails                | Ensure videos have audio tracks. Check `ffmpeg` is installed and on PATH.  |                                |
| Video playback blank           | Ensure VLC app is installed. `python-vlc` requires the VLC bundle.         | `brew install --cask vlc`      |
| RTMP stream won't connect      | Verify MonaServer is running. Check firewall. All devices on same WiFi.    | Do not use eduroam             |
| `labels_data.json` error       | Delete `labels_data.json` — app recreates from defaults on next launch.    |                                |
| Export ZIP missing video       | `ffmpeg` required for video merge. Other files still exported without it.  | `brew install ffmpeg`          |
| `Monospace` font warning       | Harmless Qt warning. Replace `Monospace` with `Courier New` in stylesheet if it bothers you. | Does not affect functionality  |

---

## 12. Known Limitations

- **macOS:** Live camera feeds open in floating `ffplay` windows — macOS blocks foreign window embedding into Qt widgets.
- **Insta360 GO 3S:** Does not support RTMP live streaming. Records to SD card only; upload footage in Tab 02 post-session.
- **VLC dependency:** `python-vlc` wraps VLC's shared libraries from the installed app bundle. VLC must be installed as a macOS `.app`, not just the Python package.
- **Audio-based sync:** The cross-correlation sync assumes all recordings share an audible common event. For best results, include a clear audio cue (bell, clap, countdown) within the first 30 seconds of all recordings.
- **No-audio sources:** If a camera has no audio track, the sync engine assigns a zero offset and marks confidence as low.
- **Export speed:** The quad-grid video merge uses `-preset fast` and `-crf 23`. For a 10-minute session expect 2–5 minutes of encoding time on a modern Mac.

---

## 13. Dependencies

| Package           | Version     | Purpose                                            |
|-------------------|-------------|----------------------------------------------------|
| `PyQt6`           | ≥ 6.4       | GUI framework                                      |
| `matplotlib`      | ≥ 3.7       | EEG waveform and band power rendering              |
| `pylsl`           | ≥ 1.16      | Lab Streaming Layer — receives EEG data            |
| `muselsl`         | ≥ 2.2       | Muse-S BLE streaming and recording CLI             |
| `scipy`           | ≥ 1.10      | FFT cross-correlation for sync engine              |
| `pandas`          | ≥ 2.0       | EEG CSV loading and manipulation                   |
| `ffmpeg-python`   | ≥ 0.2       | Python bindings for ffmpeg (audio extraction)      |
| `numpy`           | ≥ 1.24      | Audio array processing                             |
| `python-vlc`      | ≥ 3.0.18    | Python bindings for VLC (video playback)           |
| `ffmpeg` (binary) | ≥ 5.0       | Required by ffmpeg-python, muselsl, and ffplay     |
| VLC (app)         | ≥ 3.0       | Required by python-vlc for video rendering         |
| MonaServer2       | latest      | Local RTMP server for live camera feeds            |

Install all Python packages:
```bash
pip install PyQt6 matplotlib pylsl muselsl scipy pandas ffmpeg-python numpy python-vlc
```

---

## 14. Project Information

| Field        | Value                                                  |
|--------------|--------------------------------------------------------|
| Project name | Brain Battle — Neuroscience Sports Capture System      |
| Institution  | Constructor University Bremen                          |
| Type         | Community Impact Project (CIP)                         |
| Phase        | Software Phase 2                                       |
| Version      | v1.0                                                   |
| GitHub       | https://github.com/Brain-Battle                        |
| Hardware     | 2× Muse-S (MuseS-7538, MuseS-7564), 2× DSLR, 2× Insta360 GO 3S |

---

*Brain Battle — Constructor University Bremen*