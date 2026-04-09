import os
os.environ["QT_MEDIA_BACKEND"] = "ffmpeg"
os.environ["QT_FFMPEG_DISABLE_HW_DECODING"] = "1"

from pathlib import Path
import sys
import tempfile
import datetime as _dt

import numpy as _np
import vlc

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QFrame, QSizePolicy, QFileDialog,
    QListWidget, QListWidgetItem, QScrollArea,
    QSplitter, QSlider, QComboBox, QLineEdit,
    QGroupBox, QProgressBar, QStatusBar, QToolBar,
    QSpacerItem, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QPainter

from sync_engine import SyncWorker, SyncResult
from sync_utils.audio_analysis import compute_delays, find_all_durations
from sync_utils.video_sync import generate_single_preview, get_first_frame_pts, run_ffmpeg_subprocess
from sync_utils.eeg_video_sync import compare_video_eeg, cut_video_from_start_end
from eeg_module import EEGLiveWidget, EEGReviewWidget
from rtmp_module import RTMPConfigPanel
from vlc_module import VLCVideoSlot, SyncPlaybackController
from export_module import FullSessionExportWorker, LabelsExporter, ExportProgressDialog
from labels_config import LABELS as PREDEFINED_LABELS, next_colour, save as save_labels

_INT32_MAX = 2_147_483_647

def _clamp_ms(ms) -> int:
    return int(max(0, min(int(ms), _INT32_MAX)))


from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink, QVideoFrame
from PyQt6.QtGui import QImage
from PyQt6.QtCore import pyqtSlot


class VideoSurface(QWidget):
    """QPainter-based video surface — no native NSView created."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #050506;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._st     = None
        self._bufs   = None
        self._vw     = 1
        self._vh     = 1
        self._vpitch = 4
        self._timer  = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def attach_vmem(self, bufs, st, w, h, pitch):
        self._bufs = bufs; self._st = st
        self._vw = w; self._vh = h; self._vpitch = pitch

    def _tick(self):
        st = self._st
        if st is None or not st['fresh']:
            return
        st['fresh'] = False
        st['r'], st['d'] = st['d'], st['r']
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtCore import QRectF
        painter = QPainter(self)
        st, bufs = self._st, self._bufs
        if st is not None and bufs is not None:
            img = QImage(bufs[st['d']], self._vw, self._vh, self._vpitch, QImage.Format.Format_RGB32)
            sw, sh = self.width(), self.height()
            scale  = min(sw / self._vw, sh / self._vh)
            dw, dh = self._vw * scale, self._vh * scale
            painter.drawImage(QRectF((sw - dw) / 2, (sh - dh) / 2, dw, dh), img)
        else:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
        painter.end()


def _set_high_z_position(ns_handle: int, z: float = 9000.0) -> None:
    """Raise an NSView's CALayer zPosition above VLC's Metal layer."""
    try:
        import ctypes
        lib = ctypes.CDLL('/usr/lib/libobjc.dylib')
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.objc_msgSend.restype     = ctypes.c_void_p
        sel = lib.sel_registerName
        msg = lib.objc_msgSend
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        msg(ns_handle, sel(b'setWantsLayer:'), True)
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        layer = msg(ns_handle, sel(b'layer'))
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]
        msg(layer, sel(b'setZPosition:'), ctypes.c_double(z))
    except Exception as exc:
        print(f'[zPosition] {exc}')


def _clip_nsview_layer(ns_handle: int) -> None:
    """Enable masksToBounds on an NSView's CALayer to clip VLC's Metal sublayer."""
    try:
        import ctypes
        lib = ctypes.CDLL('/usr/lib/libobjc.dylib')
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.objc_msgSend.restype     = ctypes.c_void_p
        sel = lib.sel_registerName
        msg = lib.objc_msgSend
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        msg(ns_handle, sel(b'setWantsLayer:'), True)
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        layer = msg(ns_handle, sel(b'layer'))
        msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
        msg(layer, sel(b'setMasksToBounds:'), True)
    except Exception as exc:
        print(f'[CALayer] {exc}')


APP_STYLE = """
QMainWindow, QWidget { background-color: #0e0f11; color: #d4d0c8; font-family: 'Courier New', monospace; }
QTabWidget::pane { border: none; border-top: 1px solid #2a2d35; background-color: #0e0f11; top: 0px; }
QTabWidget::tab-bar { alignment: left; }
QTabBar::tab { background-color: #161820; color: #6b7280; border: 1px solid #2a2d35; border-bottom: none; padding: 10px 28px; font-family: 'Courier New', monospace; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; min-width: 160px; }
QTabBar::tab:selected { background-color: #0e0f11; color: #e8ff00; border-top: 2px solid #e8ff00; }
QTabBar::tab:hover:!selected { background-color: #1a1c24; color: #a0a8b0; }
QPushButton { background-color: #161820; color: #d4d0c8; border: 1px solid #3a3d48; padding: 8px 20px; font-family: 'Courier New', monospace; font-size: 11px; letter-spacing: 1px; }
QPushButton:hover { background-color: #1e2030; border-color: #6b7280; color: #ffffff; }
QPushButton:disabled { color: #3a3d48; border-color: #2a2d35; }
QFrame#panel { background-color: #13141a; border: 1px solid #2a2d35; }
QFrame#video_slot { background-color: #0a0b0d; border: 1px solid #2a2d35; }
QFrame#video_slot:hover { border-color: #4a4f60; }
QLabel#panel_title { color: #e8ff00; font-size: 10px; letter-spacing: 3px; font-family: 'Courier New', monospace; padding: 0px; }
QLabel#timecode { color: #e8ff00; font-size: 13px; font-family: 'Courier New', monospace; letter-spacing: 2px; }
QSlider::groove:horizontal { height: 2px; background: #2a2d35; }
QSlider::sub-page:horizontal { background: #e8ff00; height: 2px; }
QSlider::handle:horizontal { background: #e8ff00; width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; }
QListWidget { background-color: #0a0b0d; border: 1px solid #2a2d35; color: #d4d0c8; font-family: 'Courier New', monospace; font-size: 11px; outline: none; }
QListWidget::item { padding: 8px 12px; border-bottom: 1px solid #1a1c24; }
QListWidget::item:selected { background-color: #1a1f0a; color: #e8ff00; border-left: 2px solid #e8ff00; }
QScrollBar:vertical { background: #0e0f11; width: 6px; }
QScrollBar::handle:vertical { background: #2a2d35; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QProgressBar { background-color: #0a0b0d; border: 1px solid #2a2d35; height: 4px; text-align: center; color: transparent; }
QProgressBar::chunk { background-color: #e8ff00; }
QStatusBar { background-color: #0a0b0d; border-top: 1px solid #2a2d35; color: #6b7280; font-family: 'Courier New', monospace; font-size: 10px; }
QSplitter::handle { background-color: #2a2d35; width: 1px; }
QGroupBox { border: 1px solid #2a2d35; margin-top: 18px; padding: 12px; font-family: 'Courier New', monospace; font-size: 10px; letter-spacing: 2px; color: #6b7280; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #6b7280; letter-spacing: 2px; }
QComboBox { background-color: #161820; border: 1px solid #2a2d35; color: #d4d0c8; padding: 6px 10px; font-family: 'Courier New', monospace; font-size: 11px; }
QLineEdit { background-color: #0a0b0d; border: 1px solid #2a2d35; color: #d4d0c8; padding: 6px 10px; font-family: 'Courier New', monospace; font-size: 11px; }
QLineEdit:focus { border-color: #e8ff00; }
"""

def make_divider():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("background-color: #2a2d35; max-height: 1px; border: none;")
    return line

def make_panel_title(text):
    lbl = QLabel(text); lbl.setObjectName("panel_title"); return lbl

def make_source_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #9ca3af; font-size: 11px; font-family: 'Courier New'; letter-spacing: 2px; padding-left: 12px;")
    return lbl


class VideoSlot(QFrame):
    upload_requested = pyqtSignal(int)
    def __init__(self, slot_index, label="CAM", parent=None):
        super().__init__(parent)
        self.slot_index = slot_index; self.label = label
        self.setObjectName("video_slot")
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build()
    def _build(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        top_bar = QWidget(); top_bar.setFixedHeight(26)
        top_bar.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
        tl = QHBoxLayout(top_bar); tl.setContentsMargins(10,0,10,0)
        self.status_dot = QLabel("●"); self.status_dot.setObjectName("status_dot_idle")
        self.slot_label = QLabel(f"{self.label} {self.slot_index:02d}")
        self.slot_label.setStyleSheet("color: #4a4f60; font-size: 10px; font-family: 'Courier New'; letter-spacing: 2px;")
        self.filename_label = QLabel("NO SOURCE")
        self.filename_label.setStyleSheet("color: #3a3d48; font-size: 9px; font-family: 'Courier New'; letter-spacing: 1px;")
        tl.addWidget(self.status_dot); tl.addSpacing(6); tl.addWidget(self.slot_label)
        tl.addSpacing(12); tl.addWidget(self.filename_label); tl.addStretch()
        layout.addWidget(top_bar)
        self.main_area = QWidget()
        self.main_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_area.setStyleSheet("background-color: #0a0b0d;")
        ml = QVBoxLayout(self.main_area); ml.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder = QLabel(f"[ {self.label} {self.slot_index:02d} ]")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #2a2d35; font-size: 16px; font-family: 'Courier New'; letter-spacing: 4px;")
        ml.addWidget(self.placeholder); layout.addWidget(self.main_area)
    def set_active(self, filename=""):
        self.status_dot.setStyleSheet("color: #22c55e; font-size: 10px;")
        if filename:
            self.filename_label.setText(filename.split("/")[-1][:28])
            self.filename_label.setStyleSheet("color: #6b7280; font-size: 9px; font-family: 'Courier New'; letter-spacing: 1px;")
        self.placeholder.setText("")
    def set_idle(self):
        self.status_dot.setStyleSheet("color: #3a3d48; font-size: 10px;")
        self.filename_label.setText("NO SOURCE"); self.placeholder.setText(f"[ {self.label} {self.slot_index:02d} ]")


class EEGPanel(QFrame):
    def __init__(self, channel_id=1, parent=None):
        super().__init__(parent); self.channel_id = channel_id
        self.setObjectName("panel"); self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); self._build()
    def _build(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        top_bar = QWidget(); top_bar.setFixedHeight(26)
        top_bar.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
        tb = QHBoxLayout(top_bar); tb.setContentsMargins(10,0,10,0)
        dot = QLabel("●"); dot.setStyleSheet("color: #3a3d48; font-size: 10px;")
        lbl = QLabel(f"EEG CHANNEL {self.channel_id}  —  MUSE-S")
        lbl.setStyleSheet("color: #4a4f60; font-size: 10px; font-family: 'Courier New'; letter-spacing: 2px;")
        self.hz_label = QLabel("256 HZ")
        self.hz_label.setStyleSheet("color: #3a3d48; font-size: 9px; font-family: 'Courier New'; letter-spacing: 1px;")
        tb.addWidget(dot); tb.addSpacing(6); tb.addWidget(lbl); tb.addStretch(); tb.addWidget(self.hz_label)
        layout.addWidget(top_bar)
        ph = QLabel(f"EEG plot will render here — channel {self.channel_id}")
        ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph.setStyleSheet("color: #2a2d35; font-size: 11px; font-family: 'Courier New'; letter-spacing: 2px; padding: 20px;")
        layout.addWidget(ph)


class LiveMonitorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._build()
    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        toolbar = QWidget(); toolbar.setFixedHeight(44)
        toolbar.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
        tb = QHBoxLayout(toolbar); tb.setContentsMargins(16,0,16,0); tb.setSpacing(8)
        tb.addWidget(make_panel_title("LIVE MONITOR")); tb.addSpacing(24)
        self.rec_indicator = QLabel("● REC")
        self.rec_indicator.setStyleSheet("color: #3a3d48; font-size: 10px; font-family: 'Courier New'; letter-spacing: 2px;")
        tb.addWidget(self.rec_indicator); tb.addStretch()
        self.session_timer = QLabel("00:00:00"); self.session_timer.setObjectName("timecode")
        tb.addWidget(self.session_timer); tb.addSpacing(16)
        self.session_name_input = QLineEdit()
        self.session_name_input.setPlaceholderText("Session name, e.g. Round 1")
        self.session_name_input.setFixedWidth(220); self.session_name_input.setFixedHeight(30)
        self.session_name_input.setStyleSheet("background:#0a0b0d; border:1px solid #3a3d48; color:#d4d0c8; font-family:'Courier New'; font-size:11px; padding:0 10px;")
        tb.addWidget(self.session_name_input); tb.addSpacing(8)
        self.btn_start_stop = QPushButton("START RECORDING")
        self.btn_start_stop.setStyleSheet("background-color: #1a1f0a; border: 1px solid #e8ff00; color: #e8ff00; font-family: 'Courier New'; font-size: 10px; letter-spacing: 2px; padding: 6px 18px;")
        self.btn_start_stop.clicked.connect(self._toggle_recording); self._is_recording = False
        tb.addWidget(self.btn_start_stop); root.addWidget(toolbar)
        splitter = QSplitter(Qt.Orientation.Vertical); splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2d35; }")
        self.rtmp_panel = RTMPConfigPanel(); splitter.addWidget(self.rtmp_panel)
        eeg_widget = QWidget(); eeg_layout = QVBoxLayout(eeg_widget)
        eeg_layout.setContentsMargins(12,8,12,12); eeg_layout.setSpacing(8)
        eeg_header = QHBoxLayout(); eeg_header.addWidget(make_source_label("EEG STREAMS"))
        eeg_header.addStretch(); eeg_header.addWidget(make_source_label("2 MUSE-S DEVICES"))
        eeg_layout.addLayout(eeg_header)
        eeg_row = QHBoxLayout(); eeg_row.setSpacing(6)
        self.eeg_widget_1 = EEGLiveWidget(channel_id=1); self.eeg_widget_2 = EEGLiveWidget(channel_id=2)
        eeg_row.addWidget(self.eeg_widget_1); eeg_row.addWidget(self.eeg_widget_2)
        eeg_layout.addLayout(eeg_row); splitter.addWidget(eeg_widget)
        splitter.setSizes([480, 200]); root.addWidget(splitter)
        self._session_elapsed = 0
        self._session_timer = QTimer(self); self._session_timer.setInterval(1000)
        self._session_timer.timeout.connect(self._tick_timer)
    def _toggle_recording(self):
        if not self._is_recording: self._start_recording()
        else: self._stop_recording()
    def _start_recording(self):
        import datetime, re
        session_name = self.session_name_input.text().strip() or "Session"
        session_name_safe = re.sub(r'[<>:"/\\|?*]', '-', session_name)
        now = datetime.datetime.now()
        try: tz_name = now.astimezone().tzname() or "LOCAL"
        except: tz_name = "LOCAL"
        timestamp_str = now.strftime(f"%d.%m.%Y %H-%M {tz_name}")
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Muse Recordings")
        os.makedirs(base_dir, exist_ok=True)
        def _filename(widget, slot_num):
            device = widget._connected_device
            raw_name = device["name"] if device else f"Muse-{slot_num}"
            friendly = raw_name.replace("MuseS-", "Muse S ").replace("Muse-S-", "Muse S ").replace("-", " ")
            return os.path.join(base_dir, f"{timestamp_str} {session_name_safe} {friendly}.csv")
        path1 = _filename(self.eeg_widget_1, 1); path2 = _filename(self.eeg_widget_2, 2)
        self._is_recording = True; self._session_elapsed = 0; self._session_timer.start()
        self.rec_indicator.setStyleSheet("color: #ef4444; font-size: 10px; font-family: 'Courier New'; letter-spacing: 2px;")
        self.btn_start_stop.setText("STOP RECORDING")
        self.btn_start_stop.setStyleSheet("background-color: transparent; border: 1px solid #ef4444; color: #ef4444; font-family: 'Courier New'; font-size: 10px; letter-spacing: 2px; padding: 6px 18px;")
        self.session_name_input.setEnabled(False)
        self.eeg_widget_1.start_recording(save_path=path1); self.eeg_widget_2.start_recording(save_path=path2)
    def _stop_recording(self):
        self._session_timer.stop(); self._is_recording = False
        self.rec_indicator.setStyleSheet("color: #3a3d48; font-size: 10px; font-family: 'Courier New'; letter-spacing: 2px;")
        self.btn_start_stop.setText("START RECORDING")
        self.btn_start_stop.setStyleSheet("background-color: #1a1f0a; border: 1px solid #e8ff00; color: #e8ff00; font-family: 'Courier New'; font-size: 10px; letter-spacing: 2px; padding: 6px 18px;")
        self.session_name_input.setEnabled(True)
        self.eeg_widget_1.stop_recording(); self.eeg_widget_2.stop_recording()
    def _tick_timer(self):
        self._session_elapsed += 1
        h = self._session_elapsed // 3600; m = (self._session_elapsed % 3600) // 60; s = self._session_elapsed % 60
        self.session_timer.setText(f"{h:02d}:{m:02d}:{s:02d}")


class SourceRow(QWidget):
    file_loaded = pyqtSignal(int, str)
    def __init__(self, source_id, label, accept_filter, parent=None):
        super().__init__(parent); self.source_id = source_id; self.file_path = None
        self.accept_filter = accept_filter; self._build(label)
    def _build(self, label):
        layout = QHBoxLayout(self); layout.setContentsMargins(0,3,0,3); layout.setSpacing(6)
        self.status = QLabel("○"); self.status.setFixedWidth(14)
        self.status.setStyleSheet("color: #6b7280; font-size: 12px; font-family: 'Courier New';")
        self.type_lbl = QLabel(label); self.type_lbl.setFixedWidth(100)
        self.type_lbl.setStyleSheet("color: #9ca3af; font-size: 10px; font-family: 'Courier New'; letter-spacing: 1px;")
        self.file_lbl = QLabel("—")
        self.file_lbl.setStyleSheet("color: #6b7280; font-size: 10px; font-family: 'Courier New';")
        self.file_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.file_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.offset_lbl = QLabel(""); self.offset_lbl.setVisible(False)
        btn = QPushButton("UPLOAD"); btn.setFixedWidth(72); btn.setFixedHeight(26)
        btn.setStyleSheet("background-color: #161820; border: 1px solid #4a4f60; color: #9ca3af; font-family: 'Courier New'; font-size: 10px; padding: 0 6px;")
        btn.clicked.connect(self._pick_file)
        layout.addWidget(self.status); layout.addWidget(self.type_lbl); layout.addWidget(self.file_lbl); layout.addWidget(btn)
    def _pick_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", self.accept_filter)
        if path:
            self.file_path = path
            name = path.split("/")[-1]; short = name[:12] + "…" if len(name) > 12 else name
            self.file_lbl.setText(short); self.file_lbl.setStyleSheet("color: #d4d0c8; font-size: 10px; font-family: 'Courier New';")
            self.status.setText("●"); self.status.setStyleSheet("color: #22c55e; font-size: 12px; font-family: 'Courier New';")
            self.file_loaded.emit(self.source_id, path)
    def set_offset(self, seconds):
        if seconds == 0.0:
            self.offset_lbl.setText("BASE"); self.offset_lbl.setStyleSheet("color: #e8ff00; font-size: 10px; font-family: 'Courier New'; text-align: right;")
        else:
            sign = "+" if seconds > 0 else ""
            self.offset_lbl.setText(f"{sign}{seconds:.3f}s"); self.offset_lbl.setStyleSheet("color: #4a9eff; font-size: 10px; font-family: 'Courier New'; text-align: right;")
    def set_confidence(self, confidence, method):
        if method != "xcorr": return
        if confidence >= 0.6: badge, color = "✓", "#22c55e"
        elif confidence >= 0.3: badge, color = "~", "#f59e0b"
        else: badge, color = "?", "#ef4444"
        current = self.type_lbl.text().rstrip(" ✓~?")
        self.type_lbl.setText(f"{current} {badge}")
        self.type_lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-family: 'Courier New'; letter-spacing: 1px;")


import pandas as pd


class MainEditorSyncWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, list)
    error    = pyqtSignal(str)

    def __init__(self, video_paths, eeg_paths=None, pivot_index=None, parent=None):
        super().__init__(parent)
        self.video_paths = video_paths; self.eeg_paths = eeg_paths or []
        self.pivot_index = pivot_index; self._temp_dir = tempfile.TemporaryDirectory()

    def run(self):
        try:
            original_video_paths = list(self.video_paths)
            self.progress.emit(5, "Analyzing durations...")
            durations = find_all_durations(original_video_paths)

            if self.eeg_paths:
                self.progress.emit(10, "EEG data detected — computing EEG window offset...")
                try:
                    pivot = int(_np.argmax(durations))
                    eeg_start, eeg_end = compare_video_eeg(original_video_paths[pivot], self.eeg_paths[0], durations[pivot])
                    self.progress.emit(15, f"EEG window: {eeg_start:.1f}s → {eeg_end:.1f}s in pivot video")
                    print(f"[EEG sync] pivot video EEG window: {eeg_start:.3f}s → {eeg_end:.3f}s")
                except Exception as e:
                    self.progress.emit(15, f"EEG offset skipped: {e}"); print(f"[EEG sync warning] {e}")

            self.progress.emit(20, "Computing inter-camera delays via audio xcorr...")
            raw_delays = compute_delays(original_video_paths)
            min_delay = min(raw_delays)
            norm_delays = [d - min_delay for d in raw_delays]
            max_norm_delay = max(norm_delays)
            print(f"Normalised delays: {[round(d,3) for d in norm_delays]}  (max={max_norm_delay:.3f}s)")

            self.progress.emit(90, "Computing VLC seek offsets...")
            seek_ms = [_clamp_ms((max_norm_delay - d) * 1000) for d in norm_delays]
            for i, (path, s) in enumerate(zip(original_video_paths, seek_ms)):
                print(f"  clip {i} ({os.path.basename(path)}): delay={norm_delays[i]:.3f}s  seek={s}ms")

            self.progress.emit(100, "Sync complete.")
            self.finished.emit(original_video_paths, norm_delays + [max_norm_delay] + [float(s) for s in seek_ms])
        except Exception as e:
            self.error.emit(str(e))


class SyncReviewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.media_players      = [None, None, None, None]
        self._video_widgets_qt  = [None, None, None, None]
        self._vmem_states       = [None, None, None, None]
        self.video_paths        = [None, None, None, None]
        self.video_widgets      = []
        self.eeg_file_name_1 = None; self.eeg_file_name_2 = None
        self.sync_result = None; self._worker = None
        self._temp_paths = []; self._delays = []; self._residuals_ms = []
        self._post_sync_offsets_ms = [0, 0, 0, 0]
        self._pos_timer = QTimer(self); self._pos_timer.setInterval(100)
        self._pos_timer.timeout.connect(self._poll_position)
        self._app_ref = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        toolbar = QWidget(); toolbar.setFixedHeight(44)
        toolbar.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
        tb = QHBoxLayout(toolbar); tb.setContentsMargins(16,0,16,0); tb.setSpacing(8)
        tb.addWidget(make_panel_title("SYNC + REVIEW")); tb.addStretch()
        self.sync_btn = QPushButton("AUTO-SYNC"); self.sync_btn.setFixedHeight(28)
        self.sync_btn.setStyleSheet("background-color: #1a1f0a; border: 1px solid #e8ff00; color: #e8ff00; font-family: 'Courier New'; font-size: 10px; letter-spacing: 2px; padding: 0 16px;")
        self.export_btn = QPushButton("EXPORT SYNCED DATA"); self.export_btn.setFixedHeight(28)
        self.export_btn.setStyleSheet("background-color: transparent; border: 1px solid #4a9eff; color: #4a9eff; font-family: 'Courier New'; font-size: 10px; letter-spacing: 1px; padding: 0 14px;")
        self.export_btn.clicked.connect(self._export_synced_data)
        tb.addWidget(self.sync_btn); tb.addWidget(self.export_btn); root.addWidget(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.setHandleWidth(1)
        upload_widget = QWidget(); upload_widget.setFixedWidth(400)
        upload_widget.setStyleSheet("background-color: #0e0f11;")
        upload_layout = QVBoxLayout(upload_widget); upload_layout.setContentsMargins(10,16,10,16); upload_layout.setSpacing(6)
        upload_layout.addWidget(make_source_label("DATA SOURCES")); upload_layout.addSpacing(6)
        header = QHBoxLayout(); header.setSpacing(6)
        for txt, w in [("", 14), ("TYPE", 100), ("FILE", 0), ("", 80)]:
            h = QLabel(txt); h.setStyleSheet("color: #3a3d48; font-size: 9px; font-family: 'Courier New'; letter-spacing: 2px;")
            if w: h.setFixedWidth(w)
            else: h.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            header.addWidget(h)
        upload_layout.addLayout(header); upload_layout.addWidget(make_divider())
        self.sources = []
        source_defs = [
            ("DSLR  01","Video Files (*.mp4 *.mov *.avi *.MP4 *.MOV)"),("DSLR  02","Video Files (*.mp4 *.mov *.avi *.MP4 *.MOV)"),
            ("INSTA360  01","Video Files (*.mp4 *.mov *.avi *.MP4 *.MOV)"),("INSTA360  02","Video Files (*.mp4 *.mov *.avi *.MP4 *.MOV)"),
            ("MUSE-S  01","CSV Files (*.csv)"),("MUSE-S  02","CSV Files (*.csv)"),
        ]
        for i, (lbl, filt) in enumerate(source_defs):
            row = SourceRow(i, lbl, filt); upload_layout.addWidget(row)
            if i == 3: upload_layout.addWidget(make_divider())
            self.sources.append(row); row.file_loaded.connect(self._on_file_loaded)
        upload_layout.addWidget(make_divider())
        self.sync_status = QLabel("AWAITING SOURCES")
        self.sync_status.setStyleSheet("color: #3a3d48; font-size: 10px; font-family: 'Courier New'; letter-spacing: 2px; padding: 8px 0;")
        upload_layout.addWidget(self.sync_status)
        self.sync_progress = QProgressBar(); self.sync_progress.setValue(0)
        self.sync_progress.setFixedHeight(3); self.sync_progress.setVisible(False)
        upload_layout.addWidget(self.sync_progress); upload_layout.addStretch()
        self.goto_label_btn = QPushButton("PROCEED TO LABELLING  →"); self.goto_label_btn.setEnabled(False)
        self.goto_label_btn.setStyleSheet("background-color: #1a1f0a; border: 1px solid #e8ff00; color: #e8ff00; font-family: 'Courier New'; font-size: 10px; letter-spacing: 2px; padding: 10px;")
        upload_layout.addWidget(self.goto_label_btn); splitter.addWidget(upload_widget)

        preview_widget = QWidget(); preview_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout = QVBoxLayout(preview_widget); preview_layout.setContentsMargins(0,0,0,0); preview_layout.setSpacing(0)
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right_scroll.setStyleSheet("QScrollArea { background: #000; border: none; } QScrollBar:vertical { background: #0a0b0d; width: 8px; border: none; } QScrollBar::handle:vertical { background: #2a2d35; border-radius: 4px; min-height: 24px; } QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; border: none; }")
        right_content = QWidget(); right_content.setStyleSheet("background-color: #000;")
        right_vbox = QVBoxLayout(right_content); right_vbox.setContentsMargins(0,0,0,0); right_vbox.setSpacing(0)

        class _AspectVideoArea(QWidget):
            HEADER_H = 26; SPACING = 2
            def resizeEvent(self, event):
                super().resizeEvent(event)
                cell_w = max(100, (self.width() - self.SPACING) // 2); cell_h = int(cell_w * 9 / 16)
                self.setFixedHeight((cell_h + self.HEADER_H) * 2 + self.SPACING)

        video_area = _AspectVideoArea(); video_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        video_area.setStyleSheet("background-color: #000;")
        va_layout = QVBoxLayout(video_area); va_layout.setContentsMargins(0,0,0,0); va_layout.setSpacing(0)
        vgrid = QGridLayout(); vgrid.setSpacing(2); vgrid.setContentsMargins(0,0,0,0)
        vgrid.setRowStretch(0,1); vgrid.setRowStretch(1,1); vgrid.setColumnStretch(0,1); vgrid.setColumnStretch(1,1)

        def _make_video_slot(cam_label, placeholder_text):
            slot = QWidget(); slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            slot_layout = QVBoxLayout(slot); slot_layout.setContentsMargins(0,0,0,0); slot_layout.setSpacing(0)
            hdr = QWidget(); hdr.setFixedHeight(26)
            hdr.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
            hdr_layout = QHBoxLayout(hdr); hdr_layout.setContentsMargins(8,0,8,0); hdr_layout.setSpacing(6)
            dot = QLabel("●"); dot.setStyleSheet("color: #3a3d48; font-size: 10px;"); dot.setFixedWidth(12)
            name_lbl = QLabel(cam_label); name_lbl.setStyleSheet("color: #9ca3af; font-size: 9px; font-family: 'Courier New'; letter-spacing: 2px;")
            status_lbl = QLabel("— NO FILE"); status_lbl.setStyleSheet("color: #3a3d48; font-size: 9px; font-family: 'Courier New';")
            hdr_layout.addWidget(dot); hdr_layout.addWidget(name_lbl); hdr_layout.addWidget(status_lbl); hdr_layout.addStretch()
            slot_layout.addWidget(hdr)
            vlc_container = QWidget(); vlc_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            vlc_container.setStyleSheet("background-color: #0a0b0d;")
            slot_layout.addWidget(vlc_container, 1)
            ph = QLabel(placeholder_text, vlc_container); ph.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ph.setStyleSheet("color: #3a3d48; font-size: 10px; font-family: 'Courier New'; letter-spacing: 1px; background-color: transparent;")
            ph.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True); ph.show()
            def _resize_container(event, _ph=ph, _c=vlc_container): _ph.setGeometry(0, 0, _c.width(), _c.height())
            vlc_container.resizeEvent = _resize_container
            def _resize_slot(event, _hdr=hdr): _hdr.raise_()
            slot.resizeEvent = _resize_slot
            slot._dot = dot; slot._status = status_lbl; slot._ph = ph; slot._vlc_container = vlc_container
            return slot, vlc_container

        slot1, self.video_display_1 = _make_video_slot("DSLR  01", "[ DSLR 01  —  UPLOAD VIDEO TO VIEW ]")
        slot2, self.video_display_2 = _make_video_slot("DSLR  02", "[ DSLR 02  —  UPLOAD VIDEO TO VIEW ]")
        slot3, self.video_display_3 = _make_video_slot("INSTA360  01", "[ INSTA360 01  —  UPLOAD VIDEO TO VIEW ]")
        slot4, self.video_display_4 = _make_video_slot("INSTA360  02", "[ INSTA360 02  —  UPLOAD VIDEO TO VIEW ]")
        self._video_slots = [slot1, slot2, slot3, slot4]
        self.video_widgets = [self.video_display_1, self.video_display_2, self.video_display_3, self.video_display_4]
        vgrid.addWidget(slot1,0,0); vgrid.addWidget(slot2,0,1); vgrid.addWidget(slot3,1,0); vgrid.addWidget(slot4,1,1)
        va_layout.addLayout(vgrid); right_vbox.addWidget(video_area)

        eeg_row_widget = QWidget(); eeg_row_widget.setFixedHeight(320)
        eeg_row_widget.setStyleSheet("background-color: #0e0f11;")
        eeg_row_layout = QVBoxLayout(eeg_row_widget); eeg_row_layout.setContentsMargins(0,0,0,0); eeg_row_layout.setSpacing(0)
        eeg_row_layout.addWidget(make_source_label("EEG OVERLAY"))
        eeg_cols_widget = QWidget(); eeg_cols_widget.setStyleSheet("background-color: #0e0f11;")
        eeg_cols = QHBoxLayout(eeg_cols_widget); eeg_cols.setContentsMargins(12,4,12,8); eeg_cols.setSpacing(6)
        self.eeg_review_1 = EEGReviewWidget(channel_id=1); self.eeg_review_2 = EEGReviewWidget(channel_id=2)
        eeg_cols.addWidget(self.eeg_review_1); eeg_cols.addWidget(self.eeg_review_2)
        eeg_row_layout.addWidget(eeg_cols_widget, 1); right_vbox.addWidget(eeg_row_widget)
        right_scroll.setWidget(right_content); preview_layout.addWidget(right_scroll, stretch=1)

        controls = QWidget(); controls.setFixedHeight(52)
        controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        controls.setStyleSheet("background-color: #0e0f11; border-top: 2px solid #2a2d35;")
        ctrl_layout = QHBoxLayout(controls); ctrl_layout.setContentsMargins(16,0,16,0); ctrl_layout.setSpacing(10)
        self.play_btn = QPushButton("▶  PLAY"); self.play_btn.setFixedWidth(110); self.play_btn.setFixedHeight(34)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet("background-color: #161820; border: 1px solid #4a4f60; color: #d4d0c8; font-family: 'Courier New'; font-size: 11px; letter-spacing: 1px; padding: 6px;")
        self.play_btn.clicked.connect(self._toggle_play)
        self.tc_label = QLabel("00:00:00.000"); self.tc_label.setObjectName("timecode"); self.tc_label.setFixedWidth(110)
        self.time_slider = QSlider(Qt.Orientation.Horizontal); self.time_slider.setRange(0,0); self.time_slider.setValue(0)
        self.time_slider.sliderMoved.connect(self._set_position)
        self.duration_label = QLabel("--:--:--.---")
        self.duration_label.setStyleSheet("color: #9ca3af; font-size: 11px; font-family: 'Courier New'; letter-spacing: 1px;")
        self.duration_label.setFixedWidth(90)
        ctrl_layout.addWidget(self.play_btn); ctrl_layout.addWidget(self.tc_label)
        ctrl_layout.addWidget(self.time_slider); ctrl_layout.addWidget(self.duration_label)
        splitter.addWidget(preview_widget); splitter.setSizes([400, 1200])
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1)
        root.addWidget(splitter, stretch=1); root.addWidget(controls)
        self.sync_btn.clicked.connect(self._run_sync); self.goto_label_btn.clicked.connect(self._request_label_tab)

    # ── Video loading — VLC vmem → QPainter (no native NSView / Metal) ──────────
    #
    # VLC decodes the video and writes BGRA pixels into a ctypes buffer.
    # A display callback fires for each frame, wraps the buffer in a QImage,
    # and schedules an update() on the VideoSurface widget.  VideoSurface draws
    # with QPainter inside paintEvent — it is a plain QWidget with no native
    # NSView, so it is fully clipped by its parent and can never overflow the
    # header or tab bar regardless of macOS compositor behaviour.

    def _load_video_vmem(self, video_num, path):
        import ctypes
        i = video_num - 1; self.video_paths[i] = path
        container = self.video_widgets[i]

        # ── Tear down old player ───────────────────────────────────────────────
        old = self.media_players[i]
        if old is not None: old.stop()
        old_vw = self._video_widgets_qt[i]
        if old_vw is not None:
            if container.layout(): container.layout().removeWidget(old_vw)
            old_vw.setParent(None); old_vw.deleteLater()
            self._video_widgets_qt[i] = None

        # ── Create VideoSurface inside container ───────────────────────────────
        surface = VideoSurface(container)
        if container.layout() is None:
            from PyQt6.QtWidgets import QVBoxLayout as _VBL
            _lay = _VBL(container); _lay.setContentsMargins(0,0,0,0); _lay.setSpacing(0)
        container.layout().addWidget(surface); surface.show()
        self._video_widgets_qt[i] = surface

        # ── Probe video dimensions via ffprobe ────────────────────────────────
        width, height = 1920, 1080
        try:
            import ffmpeg as _ff
            probe = _ff.probe(path)
            vs = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
            if vs: width = int(vs.get("width", 1920)); height = int(vs.get("height", 1080))
        except Exception: pass

        pitch = width * 4

        # Triple-buffer: slots w(rite), r(eady), d(isplay)
        import ctypes as _ct
        bufs = [(ctypes.c_uint8 * (pitch * height))() for _ in range(3)]
        st   = {'w': 0, 'r': 1, 'd': 2, 'fresh': False}
        surface.attach_vmem(bufs, st, width, height, pitch)

        # ── VLC vmem callbacks (called on VLC decode thread) ──────────────────
        @vlc.CallbackDecorators.VideoLockCb
        def _lock(opaque, planes):
            planes[0] = ctypes.cast(bufs[st['w']], ctypes.c_void_p)
            return None

        @vlc.CallbackDecorators.VideoUnlockCb
        def _unlock(opaque, picture, planes):
            # Rotate write→ready; mark fresh for the display tick
            st['w'], st['r'] = st['r'], st['w']
            st['fresh'] = True

        @vlc.CallbackDecorators.VideoDisplayCb
        def _display(opaque, picture):
            pass  # VideoSurface._tick() handles promotion r→d and update()

        # ── Create VLC player ─────────────────────────────────────────────────
        player = vlc.MediaPlayer()
        player.video_set_callbacks(_lock, _unlock, _display, None)
        player.video_set_format("RV32", width, height, pitch)
        media = vlc.Media(Path(path).as_uri()); player.set_media(media)
        # Keep callbacks + buffers alive (GC protection)
        player._vmem_refs = (_lock, _unlock, _display, bufs, st)
        self.media_players[i] = player

        # Wire VLC events for slider / duration
        em = player.event_manager()
        em.event_attach(vlc.EventType.MediaPlayerTimeChanged,   lambda e, _i=i: self._vlc_time_changed(_i, e))
        em.event_attach(vlc.EventType.MediaPlayerLengthChanged, lambda e, _i=i: self._vlc_length_changed(_i, e))

    def _raise_ui_above_vlc(self) -> None:
        pass  # vmem uses no native view — no raise needed

    def _vlc_time_changed(self, player_idx, event):
        if self.sync_result is not None: return
        p = self.media_players[player_idx]
        if p:
            ms = p.get_time()
            if ms >= 0: QTimer.singleShot(0, lambda: self._position_changed(ms))

    def _vlc_length_changed(self, player_idx, event):
        p = self.media_players[player_idx]
        if p:
            ms = p.get_length()
            if ms > 0: QTimer.singleShot(0, lambda: self._duration_changed(ms))

    def _on_file_loaded(self, source_id, path):
        if source_id < 4:
            self._load_video_vmem(source_id + 1, path); self.play_btn.setEnabled(True)
            slot = self._video_slots[source_id]; slot._ph.hide()
            fname = path.split("/")[-1]; short = fname[:12] + "…" if len(fname) > 12 else fname
            slot._dot.setStyleSheet("color: #22c55e; font-size: 10px;")
            slot._status.setText(f"— {short}"); slot._status.setStyleSheet("color: #d4d0c8; font-size: 9px; font-family: 'Courier New';")
        elif source_id == 4:
            self.eeg_file_name_1 = path
            try: df = pd.read_csv(path); self.eeg_review_1.load(df, offset_sec=0.0)
            except Exception as e: print(f"[EEG 1] {e}")
        elif source_id == 5:
            self.eeg_file_name_2 = path
            try: df = pd.read_csv(path); self.eeg_review_2.load(df, offset_sec=0.0)
            except Exception as e: print(f"[EEG 2] {e}")

    def _raise_overlays(self): pass

    def _toggle_play(self):
        any_playing = any(p and p.is_playing() for p in self.media_players)
        if any_playing:
            for p in self.media_players:
                if p: p.pause()
            self.play_btn.setText("▶  PLAY"); self._pos_timer.stop()
        else:
            for p in self.media_players:
                if p: p.play()
            self.play_btn.setText("⏸  PAUSE"); self._pos_timer.start()

    def _set_position(self, position_ms):
        seek_targets = getattr(self, "_residuals_ms", []); max_seek = max(seek_targets) if seek_targets else 0
        for i, p in enumerate(self.media_players):
            if p:
                if seek_targets and i < len(seek_targets): p.set_time(_clamp_ms(position_ms - max_seek + seek_targets[i]))
                else: p.set_time(_clamp_ms(position_ms))
        self._update_tc(position_ms)
        self.eeg_review_1.set_playhead(position_ms / 1000.0); self.eeg_review_2.set_playhead(position_ms / 1000.0)

    def _position_changed(self, ms):
        if self.sync_result is not None: return
        self.time_slider.blockSignals(True); self.time_slider.setValue(_clamp_ms(ms)); self.time_slider.blockSignals(False)
        self._update_tc(ms)

    def _duration_changed(self, ms):
        if self.sync_result is not None: return
        if ms > 0:
            self.time_slider.setRange(0, _clamp_ms(ms))
            secs = ms / 1000.0; h = int(secs//3600); m = int((secs%3600)//60); s = secs%60
            self.duration_label.setText(f"{h:02d}:{m:02d}:{s:06.3f}")

    def _poll_position(self):
        seek_targets = getattr(self, "_residuals_ms", []); max_seek = max(seek_targets) if seek_targets else 0
        for i, p in enumerate(self.media_players):
            if p:
                t = p.get_time()
                if t < 0: continue
                my_seek = seek_targets[i] if seek_targets and i < len(seek_targets) else 0
                global_t = _clamp_ms(t - my_seek + max_seek)
                self.time_slider.blockSignals(True); self.time_slider.setValue(global_t); self.time_slider.blockSignals(False)
                self._update_tc(global_t)
                self.eeg_review_1.set_playhead(global_t / 1000.0); self.eeg_review_2.set_playhead(global_t / 1000.0)
                break

    def _update_tc(self, ms):
        secs = ms / 1000.0; h = int(secs//3600); m = int((secs%3600)//60); s = secs%60
        self.tc_label.setText(f"{h:02d}:{m:02d}:{s:06.3f}")

    def _run_sync(self):
        video_paths = [s.file_path for s in self.sources[:4] if s.file_path]
        if len(video_paths) < 2:
            QMessageBox.warning(self, "Not enough sources", "Upload at least 2 video files before syncing."); return
        self.sync_btn.setEnabled(False); self.goto_label_btn.setEnabled(False)
        self.sync_progress.setVisible(True); self.sync_progress.setValue(0)
        self._set_sync_status("INITIALISING…", "#e8ff00")
        eeg_paths = [s.file_path for s in self.sources[4:] if s.file_path]
        self._worker = MainEditorSyncWorker(video_paths, eeg_paths=eeg_paths, parent=self)
        self._worker.progress.connect(self._on_sync_progress)
        self._worker.finished.connect(self._on_sync_finished)
        self._worker.error.connect(self._on_sync_error)
        self._worker.start()

    def _on_sync_progress(self, pct, msg):
        self.sync_progress.setValue(pct); self._set_sync_status(msg.upper(), "#e8ff00")

    def _on_sync_finished(self, video_paths, delays_with_sentinel):
        self.sync_progress.setValue(100)
        n_clips = len(video_paths)
        delays = list(delays_with_sentinel[:n_clips])
        max_norm_delay = float(delays_with_sentinel[n_clips]) if len(delays_with_sentinel) > n_clips else 0.0
        seek_ms_list = [int(delays_with_sentinel[n_clips+1+i]) if len(delays_with_sentinel) > n_clips+1+i else 0 for i in range(n_clips)]
        self._temp_paths = video_paths; self._delays = delays; self._residuals_ms = seek_ms_list

        from sync_engine import SourceOffset, SyncResult as _SR
        video_sources = [s for s in self.sources[:4] if s.file_path]
        video_offsets = [SourceOffset(source_id=f"VIDEO_{i+1:02d}", path=src.file_path, offset_sec=delays[i] if i < len(delays) else 0.0, is_reference=(delays[i] == 0.0 if i < len(delays) else True), method="xcorr", confidence=1.0) for i, src in enumerate(video_sources)]
        eeg_sources = [s for s in self.sources[4:] if s.file_path]
        eeg_offsets = [SourceOffset(source_id=f"EEG_{i+1:02d}", path=s.file_path, offset_sec=0.0, method="creation_time", confidence=0.95) for i, s in enumerate(eeg_sources)]
        global_duration = 0.0
        for i, path in enumerate(video_paths):
            if path and os.path.exists(path):
                try:
                    import ffmpeg as _ffmpeg; probe = _ffmpeg.probe(path)
                    d = float(probe["format"].get("duration", 0.0))
                    seek_s = seek_ms_list[i] / 1000.0 if i < len(seek_ms_list) else 0.0
                    content_dur = max(d - seek_s, 0.0)
                    global_duration = content_dur if global_duration == 0.0 else min(global_duration, content_dur)
                except Exception: pass
        self.sync_result = _SR(global_duration=max(global_duration, 0.0), video_offsets=video_offsets, eeg_offsets=eeg_offsets)

        target_seeks = [_clamp_ms(s) for s in seek_ms_list]
        for i, path in enumerate(video_paths):
            p = self.media_players[i] if i < len(self.media_players) else None
            if p and path:
                media = vlc.Media(Path(path).as_uri()); p.set_media(media); p.play()

        _poll_count = [0]
        def _wait_and_seek():
            _poll_count[0] += 1
            active = [p for p in self.media_players if p]
            if not all(p.get_length() > 0 for p in active) and _poll_count[0] < 100:
                QTimer.singleShot(100, _wait_and_seek); return
            for i, p in enumerate(self.media_players):
                if p:
                    seek = target_seeks[i] if i < len(target_seeks) else 0
                    p.pause(); p.set_time(seek); print(f"  player {i}: set_time({seek}ms)")
            max_seek_ms = max(target_seeks) if target_seeks else 0
            content_end_ms = next((p.get_length() for p in self.media_players if p and p.get_length() > 0), 0)
            if content_end_ms > max_seek_ms:
                self.time_slider.setRange(_clamp_ms(max_seek_ms), _clamp_ms(content_end_ms))
                self.time_slider.setValue(_clamp_ms(max_seek_ms))
                secs = (content_end_ms - max_seek_ms) / 1000.0; h = int(secs//3600); m = int((secs%3600)//60); s = secs%60
                self.duration_label.setText(f"{h:02d}:{m:02d}:{s:05.2f}")
            else: self.time_slider.setRange(0, 0)
        QTimer.singleShot(200, _wait_and_seek)

        self.time_slider.setRange(0, 0); self.time_slider.setValue(0)
        self.play_btn.setText("▶  PLAY"); self.play_btn.setEnabled(True); self._post_sync_offsets_ms = [0]*4
        for i, (src_row, delay) in enumerate(zip(self.sources[:4], delays)): src_row.set_offset(delay)
        eeg_sources = [s for s in self.sources[4:] if s.file_path]
        for i, widget in enumerate([self.eeg_review_1, self.eeg_review_2]):
            if i < len(eeg_sources):
                try: df = pd.read_csv(eeg_sources[i].file_path); widget.load(df, offset_sec=0.0)
                except Exception as e: print(f"EEG {i+1} error: {e}")
        self._set_sync_status("SYNC COMPLETE  ✓", "#22c55e"); self.sync_btn.setEnabled(True); self.goto_label_btn.setEnabled(True)
        print(f"\nSync offsets (seek per player): {seek_ms_list} ms")

    def _on_sync_error(self, msg):
        self.sync_progress.setValue(0); self._set_sync_status(f"ERROR: {msg[:60]}", "#ef4444")
        self.sync_btn.setEnabled(True); QMessageBox.critical(self, "Sync failed", msg)

    def _set_sync_status(self, text, color):
        self.sync_status.setText(text)
        self.sync_status.setStyleSheet(f"color: {color}; font-size: 10px; font-family: 'Courier New'; letter-spacing: 2px; padding: 8px 0;")

    def _export_synced_data(self):
        from PyQt6.QtWidgets import QInputDialog
        if not self.sync_result: QMessageBox.warning(self, "No sync data", "Run AUTO-SYNC before exporting."); return
        zip_name, ok = QInputDialog.getText(self, "Export", "Name this export:", text="Brain_Battle_Session")
        if not ok or not zip_name.strip(): return
        zip_name = zip_name.strip().replace(" ", "_")
        if not zip_name.endswith(".zip"): zip_name += ".zip"
        save_path, _ = QFileDialog.getSaveFileName(self, "Save", zip_name, "ZIP (*.zip)")
        if not save_path: return
        video_paths = [s.file_path or "" for s in self.sources[:4]]
        offset_secs = [self.sync_result.video_offsets[i].offset_sec if i < len(self.sync_result.video_offsets) else 0.0 for i in range(4)]
        eeg_paths = [s.file_path or "" for s in self.sources[4:]]
        sync_rows = [{"source": o.source_id, "offset_sec": o.offset_sec, "method": o.method, "confidence": round(o.confidence,3), "is_reference": o.is_reference} for o in self.sync_result.all_offsets()]
        worker = FullSessionExportWorker(zip_path=save_path, zip_name=zip_name, video_paths=video_paths, offset_secs=offset_secs, eeg_paths=eeg_paths, labels=[], sync_offsets_rows=sync_rows, global_duration=self.sync_result.global_duration, merge_video=True, parent=self)
        ExportProgressDialog(worker, parent=self).exec()

    def _request_label_tab(self):
        parent = self.parent()
        while parent and not isinstance(parent, QTabWidget): parent = parent.parent()
        if not parent: return
        parent.setCurrentIndex(2); label_tab = parent.widget(2)
        if not self.sync_result or not hasattr(label_tab, "load_session_data"): return
        video_data = [(s.file_path, self.sync_result.video_offsets[i].offset_sec if i < len(self.sync_result.video_offsets) else 0.0) for i, s in enumerate(self.sources[:4])]
        eeg_data = []
        for i, s in enumerate(self.sources[4:]):
            if s.file_path:
                try: df = pd.read_csv(s.file_path); off = self.sync_result.eeg_offsets[i].offset_sec if i < len(self.sync_result.eeg_offsets) else 0.0; eeg_data.append((df, off))
                except Exception: eeg_data.append((None, 0.0))
            else: eeg_data.append((None, 0.0))
        label_tab.load_session_data(video_data, eeg_data, self.sync_result)


class LabelChip(QWidget):
    clicked = pyqtSignal(str)
    def __init__(self, label_name, color, parent=None):
        super().__init__(parent); self.label_name = label_name; self._name = label_name; self.color = color
        layout = QHBoxLayout(self); layout.setContentsMargins(0,2,0,2)
        btn = QPushButton(label_name); btn.setFixedHeight(28)
        btn.setStyleSheet(f"background-color: transparent; border: 1px solid {color}; color: {color}; font-family: 'Courier New'; font-size: 11px; letter-spacing: 1px; padding: 0 10px; text-align: left;")
        btn.clicked.connect(lambda: self.clicked.emit(self.label_name)); layout.addWidget(btn)


class LabelMarker(QWidget):
    removed = pyqtSignal()
    def __init__(self, label_name, color, timestamp_str, parent=None):
        super().__init__(parent); self.label_name = label_name; self.timestamp_str = timestamp_str; self._color = color
        self.setStyleSheet("background-color: #0e0f11;")
        layout = QHBoxLayout(self); layout.setContentsMargins(8,5,8,5); layout.setSpacing(10)
        dot = QLabel("▶"); dot.setFixedWidth(12); dot.setStyleSheet(f"color: {color}; font-size: 9px; border: none;")
        ts = QLabel(timestamp_str); ts.setFixedWidth(88); ts.setStyleSheet("color: #9ca3af; font-size: 10px; font-family: 'Courier New'; border: none;")
        name = QLabel(label_name); name.setStyleSheet(f"color: {color}; font-size: 10px; font-family: 'Courier New'; letter-spacing: 1px; border: none;")
        name.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        remove = QPushButton("×"); remove.setFixedSize(20,20)
        remove.setStyleSheet("background: transparent; border: none; color: #6b7280; font-size: 14px; padding: 0;")
        remove.clicked.connect(self._remove)
        layout.addWidget(dot); layout.addWidget(ts); layout.addWidget(name); layout.addWidget(remove)
    def _remove(self): self.removed.emit(); self.deleteLater()


class LabellingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self._fake_ts = 0.0; self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)
        toolbar = QWidget(); toolbar.setFixedHeight(44)
        toolbar.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
        tb = QHBoxLayout(toolbar); tb.setContentsMargins(16,0,16,0); tb.setSpacing(8)
        tb.addWidget(make_panel_title("EVENT LABELLING")); tb.addStretch()
        self.export_csv_btn = QPushButton("EXPORT LABELS CSV"); self.export_csv_btn.setFixedHeight(28)
        self.export_csv_btn.setStyleSheet("background-color: #1a1f0a; border: 1px solid #e8ff00; color: #e8ff00; font-family: 'Courier New'; font-size: 10px; letter-spacing: 2px; padding: 0 14px;")
        tb.addWidget(self.export_csv_btn); root.addWidget(toolbar)
        body = QHBoxLayout(); body.setContentsMargins(0,0,0,0); body.setSpacing(0)

        left = QWidget(); left.setFixedWidth(230); left.setStyleSheet("background-color: #0e0f11; border-right: 1px solid #2a2d35;")
        left_outer = QVBoxLayout(left); left_outer.setContentsMargins(0,0,0,0); left_outer.setSpacing(0)
        palette_header = QWidget(); palette_header.setFixedHeight(36)
        palette_header.setStyleSheet("background-color: #0a0b0d; border-bottom: 1px solid #2a2d35;")
        ph = QHBoxLayout(palette_header); ph.setContentsMargins(12,0,8,0); ph.setSpacing(6)
        ph.addWidget(make_source_label("LABEL PALETTE")); ph.addStretch(); left_outer.addWidget(palette_header)
        self.active_label_display = QLabel("NO LABEL SELECTED"); self.active_label_display.setFixedHeight(30)
        self.active_label_display.setStyleSheet("color: #4a4f60; font-size: 10px; font-family: 'Courier New'; padding: 0 12px; background: #0a0b0d; border-bottom: 1px solid #2a2d35;")
        left_outer.addWidget(self.active_label_display)
        chip_scroll = QScrollArea(); chip_scroll.setWidgetResizable(True)
        chip_scroll.setStyleSheet("border: none; background: #0e0f11;")
        chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._palette_widget = QWidget(); self._palette_widget.setStyleSheet("background: #0e0f11;")
        self._palette_layout = QVBoxLayout(self._palette_widget); self._palette_layout.setContentsMargins(10,8,10,8)
        self._palette_layout.setSpacing(3); self._palette_layout.addStretch()
        chip_scroll.setWidget(self._palette_widget); left_outer.addWidget(chip_scroll)
        add_bar = QWidget(); add_bar.setFixedHeight(40); add_bar.setStyleSheet("background-color: #0a0b0d; border-top: 1px solid #2a2d35;")
        ab = QHBoxLayout(add_bar); ab.setContentsMargins(10,6,10,6); ab.setSpacing(6)
        self.custom_input = QLineEdit(); self.custom_input.setPlaceholderText("add label…")
        self.custom_input.setStyleSheet("background: #161820; border: 1px solid #2a2d35; color: #d4d0c8; font-family: 'Courier New'; font-size: 11px; padding: 0 8px;")
        add_btn = QPushButton("+"); add_btn.setFixedSize(28,26)
        add_btn.setStyleSheet("background:#1a1f0a; border:1px solid #e8ff00; color:#e8ff00; font-size:16px; padding:0;")
        ab.addWidget(self.custom_input); ab.addWidget(add_btn); left_outer.addWidget(add_bar)
        self.active_label = None; self.active_color = "#e8ff00"
        for name, color in PREDEFINED_LABELS: self._add_chip_to_palette(name, color)
        add_btn.clicked.connect(self._add_custom_label); self.custom_input.returnPressed.connect(self._add_custom_label)
        body.addWidget(left)

        center = QWidget(); center.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout = QVBoxLayout(center); center_layout.setContentsMargins(0,0,0,0); center_layout.setSpacing(0)
        video_area = QWidget(); video_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_area.setStyleSheet("background-color: #0a0b0d;")
        va = QVBoxLayout(video_area); va.setContentsMargins(12,12,12,8); va.setSpacing(6)
        va.addWidget(make_source_label("SYNCHRONIZED PLAYBACK"))
        vg = QGridLayout(); vg.setSpacing(6)
        self.label_slots = []
        for i, ln in enumerate(["DSLR","DSLR","360","360"]):
            sl = VLCVideoSlot(i+1, label=ln); vg.addWidget(sl, *divmod(i,2)); self.label_slots.append(sl)
        self._label_sync_ctrl = SyncPlaybackController(); va.addLayout(vg)
        lbl_inner_splitter = QSplitter(Qt.Orientation.Vertical); lbl_inner_splitter.setHandleWidth(2)
        lbl_inner_splitter.setStyleSheet("QSplitter::handle { background: #2a2d35; }"); lbl_inner_splitter.addWidget(video_area)
        eeg_strip = QWidget(); eeg_strip.setMinimumHeight(380); eeg_strip.setStyleSheet("background-color: #0e0f11;")
        eeg_strip_layout = QVBoxLayout(eeg_strip); eeg_strip_layout.setContentsMargins(12,6,12,6); eeg_strip_layout.setSpacing(4)
        eeg_strip_layout.addWidget(make_source_label("EEG OVERLAY"))
        eeg_strip_row = QHBoxLayout(); eeg_strip_row.setSpacing(6)
        self.eeg_label_1 = EEGReviewWidget(channel_id=1); self.eeg_label_2 = EEGReviewWidget(channel_id=2)
        eeg_strip_row.addWidget(self.eeg_label_1); eeg_strip_row.addWidget(self.eeg_label_2)
        eeg_strip_layout.addLayout(eeg_strip_row); lbl_inner_splitter.addWidget(eeg_strip)
        lbl_inner_splitter.setSizes([320,380]); lbl_inner_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(lbl_inner_splitter, stretch=1)

        tl_container = QWidget(); tl_container.setFixedHeight(110)
        tl_container.setStyleSheet("background-color: #0e0f11; border-top: 1px solid #2a2d35;")
        tl_layout = QVBoxLayout(tl_container); tl_layout.setContentsMargins(16,8,16,8); tl_layout.setSpacing(6)
        tl_header = QHBoxLayout(); tl_header.addWidget(make_source_label("TIMELINE")); tl_header.addStretch()
        self.tl_tc = QLabel("00:00:00.000"); self.tl_tc.setObjectName("timecode"); tl_header.addWidget(self.tl_tc)
        tl_layout.addLayout(tl_header)
        self.tl_slider = QSlider(Qt.Orientation.Horizontal); self.tl_slider.setRange(0,10000); self.tl_slider.setValue(0)
        self.tl_slider.valueChanged.connect(self._on_timeline_scrub); tl_layout.addWidget(self.tl_slider)
        play_row = QHBoxLayout(); play_row.setSpacing(8)
        rewind_btn = QPushButton("⏮"); rewind_btn.setFixedSize(32,28)
        rewind_btn.setStyleSheet("background-color: #161820; border: 1px solid #2a2d35; color: #d4d0c8; font-size: 12px; padding: 0;")
        rewind_btn.clicked.connect(lambda: self._label_sync_ctrl.seek(0.0)); play_row.addWidget(rewind_btn)
        self._play_btn_lbl = QPushButton("▶  PLAY"); self._play_btn_lbl.setFixedWidth(110); self._play_btn_lbl.setFixedHeight(28)
        self._play_btn_lbl.setEnabled(False)
        self._play_btn_lbl.setStyleSheet("background-color: #161820; border: 1px solid #2a2d35; color: #d4d0c8; font-size: 12px; padding: 0;")
        self._play_btn_lbl.clicked.connect(self._toggle_play_label); play_row.addWidget(self._play_btn_lbl)
        skip_btn = QPushButton("⏭"); skip_btn.setFixedSize(32,28)
        skip_btn.setStyleSheet("background-color: #161820; border: 1px solid #2a2d35; color: #d4d0c8; font-size: 12px; padding: 0;")
        play_row.addWidget(skip_btn)
        self._label_pos_timer = QTimer(self); self._label_pos_timer.setInterval(100)
        self._label_pos_timer.timeout.connect(self._update_label_timeline)
        place_btn = QPushButton("PLACE LABEL AT PLAYHEAD"); place_btn.setFixedHeight(28)
        place_btn.setStyleSheet("background-color: #1a1f0a; border: 1px solid #e8ff00; color: #e8ff00; font-family: 'Courier New'; font-size: 9px; letter-spacing: 2px; padding: 0 14px;")
        place_btn.clicked.connect(self._place_label); play_row.addWidget(place_btn); play_row.addStretch()
        self._end_lbl = QLabel("--:--:--.---")
        self._end_lbl.setStyleSheet("color: #9ca3af; font-size: 11px; font-family: 'Courier New'; letter-spacing: 1px;")
        play_row.addWidget(self._end_lbl); tl_layout.addLayout(play_row); center_layout.addWidget(tl_container)
        body.addWidget(center)

        right = QWidget(); right.setFixedWidth(280); right.setStyleSheet("background-color: #0e0f11; border-left: 1px solid #2a2d35;")
        right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(0)
        right_header = QWidget(); right_header.setFixedHeight(36); right_header.setStyleSheet("background-color: #0a0b0d; border-bottom: 1px solid #2a2d35;")
        rh = QHBoxLayout(right_header); rh.setContentsMargins(14,0,14,0)
        rh.addWidget(make_source_label("PLACED LABELS")); rh.addStretch()
        self.label_count = QLabel("0"); self.label_count.setStyleSheet("color: #e8ff00; font-size: 10px; font-family: 'Courier New';")
        rh.addWidget(self.label_count); right_layout.addWidget(right_header)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("border: none; background-color: #0e0f11;")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.label_list_widget = QWidget(); self.label_list_widget.setStyleSheet("background-color: #0e0f11;")
        self.placed_layout = QVBoxLayout(self.label_list_widget); self.placed_layout.setContentsMargins(0,4,0,4)
        self.placed_layout.setSpacing(0); self.placed_layout.addStretch()
        scroll.setWidget(self.label_list_widget); right_layout.addWidget(scroll)
        body.addWidget(right); root.addLayout(body)
        self.export_csv_btn.clicked.connect(self._export_labels); QTimer.singleShot(0, self._do_recount)

    def _add_chip_to_palette(self, name, color):
        row = QWidget(); row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row); rl.setContentsMargins(0,1,0,1); rl.setSpacing(4)
        chip = LabelChip(name, color); chip.clicked.connect(self._select_label); rl.addWidget(chip)
        rm = QPushButton("×"); rm.setFixedSize(22,24)
        rm.setStyleSheet("background: transparent; border: none; color: #6b7280; font-size: 14px; padding: 0;")
        rm.clicked.connect(lambda checked, r=row, n=name: self._remove_chip(r, n)); rl.addWidget(rm)
        self._palette_layout.insertWidget(self._palette_layout.count()-1, row)

    def _remove_chip(self, row, name):
        row.deleteLater()
        if self.active_label == name:
            self.active_label = None; self.active_color = "#e8ff00"; self.active_label_display.setText("NO LABEL SELECTED")
            self.active_label_display.setStyleSheet("color: #4a4f60; font-size: 10px; font-family: 'Courier New'; padding: 0 12px; background: #0a0b0d; border-bottom: 1px solid #2a2d35;")
        QTimer.singleShot(50, lambda: save_labels(self._current_labels()))

    def _current_labels(self):
        labels = []
        for i in range(self._palette_layout.count()-1):
            item = self._palette_layout.itemAt(i)
            if item and item.widget():
                chips = item.widget().findChildren(LabelChip)
                if chips: labels.append((chips[0]._name, chips[0].color))
        return labels

    def _add_custom_label(self):
        name = self.custom_input.text().strip().upper().replace(" ","_")
        if not name: return
        existing = set()
        for i in range(self._palette_layout.count()-1):
            item = self._palette_layout.itemAt(i)
            if item and item.widget():
                children = item.widget().findChildren(LabelChip)
                if children: existing.add(children[0]._name)
        if name in existing: self._select_label(name); self.custom_input.clear(); return
        color = next_colour(self._current_labels()); self._add_chip_to_palette(name, color)
        self.custom_input.clear(); self._select_label(name); save_labels(self._current_labels())

    def _select_label(self, name):
        self.active_label = name; color = "#e8ff00"
        for i in range(self._palette_layout.count()-1):
            item = self._palette_layout.itemAt(i)
            if item and item.widget():
                chips = item.widget().findChildren(LabelChip)
                if chips and chips[0]._name == name: color = chips[0].color; break
        self.active_color = color; self.active_label_display.setText(f"● {name}")
        self.active_label_display.setStyleSheet(f"color: {color}; font-size: 11px; font-family: 'Courier New'; padding: 0 12px; background: #0a0b0d; border-bottom: 1px solid {color};")

    def _toggle_play_label(self):
        if self._label_sync_ctrl.is_playing():
            self._label_sync_ctrl.pause(); self._play_btn_lbl.setText("▶  PLAY"); self._label_pos_timer.stop()
        else:
            self._label_sync_ctrl.play(); self._play_btn_lbl.setText("⏸  PAUSE"); self._label_pos_timer.start()

    def _update_label_timeline(self):
        global_t = self._label_sync_ctrl.current_global_t(); self._fake_ts = global_t
        h = int(global_t//3600); m = int((global_t%3600)//60); s = global_t%60
        self.tl_tc.setText(f"{h:02d}:{m:02d}:{s:06.3f}")
        loaded = [sl for sl in self.label_slots if sl.is_loaded()]
        dur = (loaded[0].get_duration_sec() if loaded else 1.0) or 1.0
        self.tl_slider.blockSignals(True); self.tl_slider.setValue(int((global_t/dur)*10000)); self.tl_slider.blockSignals(False)
        self.eeg_label_1.set_playhead(global_t); self.eeg_label_2.set_playhead(global_t)

    def _on_timeline_scrub(self, value):
        loaded = [sl for sl in self.label_slots if sl.is_loaded()]
        dur = (loaded[0].get_duration_sec() if loaded else 1.0) or 1.0
        total_secs = (value/10000)*dur; h = int(total_secs//3600); m = int((total_secs%3600)//60); s = total_secs%60
        self.tl_tc.setText(f"{h:02d}:{m:02d}:{s:06.3f}"); self._fake_ts = total_secs
        self._label_sync_ctrl.seek(total_secs); self.eeg_label_1.set_playhead(total_secs); self.eeg_label_2.set_playhead(total_secs)

    def load_session_data(self, video_slots_data, eeg_data, sync_result):
        for i, (path, offset) in enumerate(video_slots_data):
            if i < len(self.label_slots) and path: self.label_slots[i].load(path, offset_sec=offset)
        loaded = [s for s in self.label_slots if s.is_loaded()]; offsets = [s._offset_sec for s in loaded]
        dur = sync_result.global_duration if sync_result else 0.0
        self._label_sync_ctrl.attach(loaded, offsets_sec=offsets, global_duration=dur)
        if dur > 0:
            h = int(dur//3600); m = int((dur%3600)//60); s = dur%60; self._end_lbl.setText(f"{h:02d}:{m:02d}:{s:06.3f}")
        for i, (df, offset) in enumerate(eeg_data):
            if i == 0 and df is not None: self.eeg_label_1.load(df, offset_sec=offset)
            elif i == 1 and df is not None: self.eeg_label_2.load(df, offset_sec=offset)
        self._play_btn_lbl.setEnabled(True)

    def load_eeg_data(self, df1, offset1, df2, offset2):
        if df1 is not None: self.eeg_label_1.load(df1, offset_sec=offset1)
        if df2 is not None: self.eeg_label_2.load(df2, offset_sec=offset2)

    def _place_label(self):
        if not self.active_label: return
        ts = self._fake_ts; h = int(ts//3600); m = int((ts%3600)//60); s = ts%60
        ts_str = f"{h:02d}:{m:02d}:{s:06.3f}"
        marker = LabelMarker(self.active_label, self.active_color, ts_str)
        marker.removed.connect(self._update_label_count)
        self.placed_layout.insertWidget(self.placed_layout.count()-1, marker); self._update_label_count()

    def _update_label_count(self): QTimer.singleShot(0, self._do_recount)

    def _do_recount(self):
        if not hasattr(self,'placed_layout') or not hasattr(self,'label_count'): return
        n = sum(1 for i in range(self.placed_layout.count()) if isinstance(self.placed_layout.itemAt(i).widget(), LabelMarker))
        self.label_count.setText(str(n))

    def get_labels(self):
        labels = []
        for i in range(self.placed_layout.count()-1):
            item = self.placed_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), LabelMarker):
                marker = item.widget(); ts_str = marker.timestamp_str; name = marker.label_name
                try: parts = ts_str.split(":"); ts_sec = float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
                except Exception: ts_sec = 0.0
                labels.append({"timestamp_sec": round(ts_sec,3), "timestamp_str": ts_str, "label": name, "color": marker._color})
        return labels

    def _export_labels(self):
        labels = self.get_labels()
        if not labels: QMessageBox.information(self, "No labels", "Place some labels before exporting."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export labels", "labels.csv", "CSV (*.csv)")
        if path:
            try: LabelsExporter.export(labels, path); QMessageBox.information(self, "Exported", f"Saved {len(labels)} labels to:\n{path}")
            except Exception as e: QMessageBox.critical(self, "Export failed", str(e))


class BrainBattleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BRAIN BATTLE  //  DATA CAPTURE + SYNC + LABELLING")
        self.setMinimumSize(1280, 800); self.resize(1440, 900); self._build()

    def _build(self):
        self.setStyleSheet(APP_STYLE)
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)

        header = QWidget(); header.setFixedHeight(38)
        header.setStyleSheet("background-color: #0a0b0d; border-bottom: 1px solid #2a2d35;")
        h_layout = QHBoxLayout(header); h_layout.setContentsMargins(16,0,16,0)
        logo = QLabel("BB//"); logo.setStyleSheet("color: #e8ff00; font-size: 16px; font-family: 'Courier New'; font-weight: bold; letter-spacing: 2px;")
        title = QLabel("BRAIN BATTLE  —  NEUROSCIENCE SPORTS CAPTURE SYSTEM"); title.setStyleSheet("color: #3a3d48; font-size: 10px; font-family: 'Courier New'; letter-spacing: 3px;")
        version = QLabel("v0.1.0  //  CONSTRUCTOR UNIVERSITY BREMEN"); version.setStyleSheet("color: #2a2d35; font-size: 9px; font-family: 'Courier New'; letter-spacing: 1px;")
        h_layout.addWidget(logo); h_layout.addSpacing(12); h_layout.addWidget(title); h_layout.addStretch(); h_layout.addWidget(version)
        self._header = header  # saved so _raise_ui_above_vlc() can lift it above VLC
        layout.addWidget(header)

        self.tabs = QTabWidget(); self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(False); self.tabs.setContentsMargins(0,0,0,0)
        self.live_tab = LiveMonitorTab(); self.sync_tab = SyncReviewTab(); self.sync_tab._app_ref = self
        self.label_tab = LabellingTab()
        self.tabs.addTab(self.live_tab, "01  LIVE MONITOR"); self.tabs.addTab(self.sync_tab, "02  SYNC + REVIEW"); self.tabs.addTab(self.label_tab, "03  LABELLING")
        layout.addWidget(self.tabs)
        sb = QStatusBar(); sb.showMessage("READY  //  NO ACTIVE SESSION  —  6 SOURCES EXPECTED: 2× DSLR  2× INSTA360  2× MUSE-S")
        self.setStatusBar(sb)


if __name__ == "__main__":
    import traceback as _tb
    app = QApplication(sys.argv); app.setApplicationName("Brain Battle")
    def _excepthook(exc_type, exc_value, exc_tb):
        print("\n=== UNHANDLED EXCEPTION ==="); _tb.print_exception(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook
    try:
        window = BrainBattleApp(); window.show()
    except Exception:
        _tb.print_exc(); input("\nPress Enter to exit..."); sys.exit(1)
    sys.exit(app.exec())