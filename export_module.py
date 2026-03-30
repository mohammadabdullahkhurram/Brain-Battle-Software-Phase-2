"""
export_module.py  —  Brain Battle export pipeline

Two exports
───────────

1. VideoExportWorker  (QThread)
   Merges up to 4 synced video files into a single 2×2 quad-grid MP4
   using ffmpeg filter_complex. Each video is trimmed to its sync offset
   and scaled to a common resolution before compositing.

   Output: one MP4 file chosen by the user.

2. LabelsExportWorker  (QThread)
   Writes all placed labels to a CSV with columns:
       timestamp_sec, timestamp_str, label, color

   Output: one CSV file chosen by the user.

3. FullSessionExport
   Combines both into a single zip:
   - merged_video.mp4
   - labels.csv
   - sync_offsets.csv
   - eeg/eeg_01.csv, eeg/eeg_02.csv
   - README.txt

All workers emit:
   progress(int, str)   — percent + status message
   finished(str)        — output path on success
   error(str)           — error message on failure
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import zipfile
import datetime
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QFileDialog,
    QInputDialog, QMessageBox, QWidget,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> Optional[str]:
    import shutil
    return shutil.which("ffmpeg")


def _fmt_ts(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS.mmm string for ffmpeg -ss."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ─── 1. Video export worker ──────────────────────────────────────────────────

class VideoExportWorker(QThread):
    """
    Merges up to 4 synced videos into a quad-grid MP4 using ffmpeg.

    Layout (2×2):
        [video 1] [video 2]
        [video 3] [video 4]

    Each video is:
      - Trimmed from its sync offset (so all start at global t=0)
      - Scaled to CELL_W × CELL_H  (default 960×540, giving 1920×1080 output)
      - Padded with black if shorter than the reference

    If fewer than 4 videos are provided, empty black cells fill the gaps.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    CELL_W = 960
    CELL_H = 540

    def __init__(
        self,
        video_paths:  list[str],
        offset_secs:  list[float],
        output_path:  str,
        global_duration: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self.video_paths      = video_paths
        self.offset_secs      = offset_secs
        self.output_path      = output_path
        self.global_duration  = global_duration

    def run(self):
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            self.error.emit(
                "ffmpeg not found. Install with: brew install ffmpeg"
            )
            return

        try:
            self._export(ffmpeg)
        except Exception as exc:
            self.error.emit(str(exc))

    def _export(self, ffmpeg: str):
        n = len(self.video_paths)
        self.progress.emit(5, "Preparing ffmpeg filter graph…")

        # ── Build input args ──────────────────────────────────────────────
        # Each video is started at its offset with -ss for fast seeking
        input_args = []
        for i, (path, offset) in enumerate(
            zip(self.video_paths, self.offset_secs)
        ):
            if offset > 0:
                input_args += ["-ss", _fmt_ts(offset)]
            if self.global_duration > 0:
                input_args += ["-t", f"{self.global_duration:.3f}"]
            input_args += ["-i", path]

        # Fill missing slots with lavfi nullsrc (black)
        null_inputs = []
        for i in range(n, 4):
            dur = self.global_duration if self.global_duration > 0 else 3600
            null_inputs += [
                "-f", "lavfi",
                "-t", f"{dur:.3f}",
                "-i", f"color=black:s={self.CELL_W}x{self.CELL_H}:r=30",
            ]

        self.progress.emit(15, "Building filter graph…")

        # ── Filter complex ────────────────────────────────────────────────
        # Scale each input to CELL_W×CELL_H, then xstack 2×2
        filter_parts = []
        scale_labels = []
        for i in range(4):
            lbl = f"v{i}"
            filter_parts.append(
                f"[{i}:v]scale={self.CELL_W}:{self.CELL_H}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={self.CELL_W}:{self.CELL_H}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1[{lbl}]"
            )
            scale_labels.append(f"[{lbl}]")

        stack_inputs = "".join(scale_labels)
        filter_parts.append(
            f"{stack_inputs}xstack=inputs=4:"
            f"layout=0_0|w0_0|0_h0|w0_h0[out]"
        )

        filter_complex = ";".join(filter_parts)

        self.progress.emit(25, "Running ffmpeg — this may take a while…")

        cmd = (
            [ffmpeg, "-y"]
            + input_args
            + null_inputs
            + [
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                self.output_path,
            ]
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Parse ffmpeg progress from stderr
        total_dur = self.global_duration or 1.0
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            line = line.decode(errors="replace")
            if "time=" in line:
                # Parse current time from ffmpeg output
                try:
                    t_str = line.split("time=")[1].split()[0]
                    parts = t_str.split(":")
                    if len(parts) == 3:
                        elapsed = (
                            float(parts[0]) * 3600
                            + float(parts[1]) * 60
                            + float(parts[2])
                        )
                        pct = min(95, int(25 + (elapsed / total_dur) * 70))
                        self.progress.emit(pct, f"Encoding… {t_str}")
                except Exception:
                    pass

        proc.wait()

        if proc.returncode != 0:
            err = (proc.stderr.read() or b"").decode(errors="replace")[-300:]
            self.error.emit(f"ffmpeg failed:\n{err}")
            return

        self.progress.emit(100, "Video export complete.")
        self.finished.emit(self.output_path)


# ─── 2. Labels CSV export ─────────────────────────────────────────────────────

class LabelsExporter:
    """
    Synchronous (non-threaded) labels CSV writer.
    Called from the main thread — fast enough not to need a worker.
    """

    @staticmethod
    def export(labels: list[dict], output_path: str):
        """
        labels: list of dicts with keys: timestamp_sec, timestamp_str, label, color
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp_sec", "timestamp_str", "label", "color"],
            )
            writer.writeheader()
            writer.writerows(labels)


# ─── 3. Full session zip export worker ───────────────────────────────────────

class FullSessionExportWorker(QThread):
    """
    Packages everything into a named zip:
      merged_video.mp4   — quad-grid merged video (optional, skipped if no ffmpeg)
      labels.csv         — all placed labels with timestamps
      sync_offsets.csv   — source offsets from sync engine
      eeg/eeg_01.csv     — EEG recording files
      eeg/eeg_02.csv
      videos/            — original video files (copied)
      README.txt
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(
        self,
        zip_path:        str,
        zip_name:        str,
        video_paths:     list[str],
        offset_secs:     list[float],
        eeg_paths:       list[str],
        labels:          list[dict],
        sync_offsets_rows: list[dict],
        global_duration: float,
        merge_video:     bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.zip_path           = zip_path
        self.zip_name           = zip_name
        self.video_paths        = video_paths
        self.offset_secs        = offset_secs
        self.eeg_paths          = eeg_paths
        self.labels             = labels
        self.sync_offsets_rows  = sync_offsets_rows
        self.global_duration    = global_duration
        self.merge_video        = merge_video

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self.error.emit(str(exc))

    def _run(self):
        import io

        self.progress.emit(2, "Preparing export…")

        with tempfile.TemporaryDirectory() as tmp:
            files_to_zip: list[tuple[str, str]] = []  # (disk_path, zip_arcname)

            # ── Merged video ──────────────────────────────────────────────
            merged_path = None
            if self.merge_video and self.video_paths and _find_ffmpeg():
                merged_path = os.path.join(tmp, "merged_video.mp4")
                self.progress.emit(5, "Merging video — this may take a while…")

                done = {"ok": False, "err": ""}

                worker = VideoExportWorker(
                    video_paths=self.video_paths,
                    offset_secs=self.offset_secs,
                    output_path=merged_path,
                    global_duration=self.global_duration,
                )

                def on_prog(pct, msg):
                    # Map 5-85% to video merge progress
                    self.progress.emit(5 + int(pct * 0.80), msg)

                def on_done(path):
                    done["ok"] = True

                def on_err(msg):
                    done["err"] = msg

                worker.progress.connect(on_prog)
                worker.finished.connect(on_done)
                worker.error.connect(on_err)
                worker.start()
                worker.wait()  # blocking — we're already in a thread

                if done["ok"] and os.path.exists(merged_path):
                    files_to_zip.append((merged_path, "merged_video.mp4"))
                else:
                    self.progress.emit(85, f"Video merge skipped: {done['err'][:60]}")
            else:
                self.progress.emit(85, "Skipping video merge…")

            # ── Labels CSV ────────────────────────────────────────────────
            self.progress.emit(86, "Writing labels CSV…")
            labels_path = os.path.join(tmp, "labels.csv")
            LabelsExporter.export(self.labels, labels_path)
            files_to_zip.append((labels_path, "labels.csv"))

            # ── Sync offsets CSV ──────────────────────────────────────────
            self.progress.emit(88, "Writing sync offsets…")
            offsets_path = os.path.join(tmp, "sync_offsets.csv")
            if self.sync_offsets_rows:
                with open(offsets_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["source", "offset_sec", "method",
                                    "confidence", "is_reference"],
                    )
                    writer.writeheader()
                    writer.writerows(self.sync_offsets_rows)
                files_to_zip.append((offsets_path, "sync_offsets.csv"))

            # ── EEG files ─────────────────────────────────────────────────
            self.progress.emit(90, "Adding EEG files…")
            for i, eeg_path in enumerate(self.eeg_paths):
                if eeg_path and os.path.exists(eeg_path):
                    files_to_zip.append((eeg_path, f"eeg/eeg_{i+1:02d}.csv"))

            # ── Original videos ───────────────────────────────────────────
            self.progress.emit(92, "Adding original videos…")
            for i, vpath in enumerate(self.video_paths):
                if vpath and os.path.exists(vpath):
                    ext = os.path.splitext(vpath)[1]
                    files_to_zip.append((vpath, f"videos/video_{i+1:02d}{ext}"))

            # ── README ────────────────────────────────────────────────────
            self.progress.emit(95, "Writing README…")
            created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            readme_lines = [
                "Brain Battle — Session Export",
                f"Name:            {self.zip_name}",
                f"Created:         {created}",
                f"Global duration: {self.global_duration:.3f}s",
                f"Videos:          {len([v for v in self.video_paths if v])}",
                f"EEG files:       {len([e for e in self.eeg_paths if e])}",
                f"Labels placed:   {len(self.labels)}",
                "",
                "Contents:",
                "  merged_video.mp4   — quad-grid synced video",
                "  labels.csv         — event labels with timestamps",
                "  sync_offsets.csv   — per-source sync offsets",
                "  eeg/               — raw EEG CSV recordings",
                "  videos/            — original camera files",
            ]
            readme_path = os.path.join(tmp, "README.txt")
            with open(readme_path, "w") as f:
                f.write("\n".join(readme_lines))
            files_to_zip.append((readme_path, "README.txt"))

            # ── Write zip ─────────────────────────────────────────────────
            self.progress.emit(97, "Writing zip file…")
            with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for disk_path, arcname in files_to_zip:
                    zf.write(disk_path, arcname)

        self.progress.emit(100, "Export complete.")
        self.finished.emit(self.zip_path)


# ─── 4. Export progress dialog ───────────────────────────────────────────────

class ExportProgressDialog(QDialog):
    """
    Modal dialog that shows export progress.
    Runs FullSessionExportWorker and displays progress bar + status.
    """

    def __init__(self, worker: FullSessionExportWorker, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting…")
        self.setMinimumWidth(420)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background-color: #0e0f11; }
            QLabel  { color: #d4d0c8; font-family: 'Courier New'; font-size: 12px; }
            QProgressBar {
                background: #0a0b0d; border: 1px solid #2a2d35; height: 6px;
                text-align: center; color: transparent;
            }
            QProgressBar::chunk { background: #e8ff00; }
            QPushButton {
                background: #161820; border: 1px solid #3a3d48; color: #d4d0c8;
                font-family: 'Courier New'; font-size: 11px; padding: 7px 20px;
            }
        """)

        self._worker = worker
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        self._status = QLabel("Preparing…")
        layout.addWidget(self._status)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        layout.addWidget(self._bar)

        self._close_btn = QPushButton("CLOSE")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.accept)
        layout.addWidget(self._close_btn)

        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.start()

    def _on_progress(self, pct: int, msg: str):
        self._bar.setValue(pct)
        self._status.setText(msg)

    def _on_finished(self, path: str):
        self._bar.setValue(100)
        self._status.setText(f"Done  ✓\n{path}")
        self._status.setStyleSheet(
            "color: #22c55e; font-family: 'Courier New'; font-size: 12px;"
        )
        self._close_btn.setEnabled(True)

    def _on_error(self, msg: str):
        self._status.setText(f"Error: {msg[:120]}")
        self._status.setStyleSheet(
            "color: #ef4444; font-family: 'Courier New'; font-size: 12px;"
        )
        self._close_btn.setEnabled(True)
