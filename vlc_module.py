"""
vlc_module.py  —  Brain Battle

VLCVideoSlot  — a single video display slot backed by QMediaPlayer /
                QVideoWidget.  Used exclusively by LabellingTab.

SyncPlaybackController  — coordinates play/pause/seek across a list of
                          VLCVideoSlot objects so they all stay in sync.

Changes from original
─────────────────────
- SyncPlaybackController completely rewritten to operate on VLCVideoSlot
  objects (it previously used QMediaPlayer directly, which caused a type
  mismatch since LabellingTab passes VLCVideoSlots).
- SyncPlaybackController.attach() now accepts an optional global_duration
  float (seconds) instead of global_duration_ms (int ms) to match the
  call site in LabellingTab.load_session_data().
- SyncPlaybackController.current_global_t() replaces current_pos_ms() and
  returns seconds — used by LabellingTab._update_label_timeline().
- SyncPlaybackController.seek() now takes seconds (float) not ms (int) to
  match _on_timeline_scrub() and _update_label_timeline() call sites.
- Removed QMediaPlayer import from the top of the module (no longer used
  by SyncPlaybackController; VLCVideoSlot imports it locally).
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QSizePolicy,
)


# ─── VLCVideoSlot ─────────────────────────────────────────────────────────────

class VLCVideoSlot(QFrame):
    """
    Used exclusively by LabellingTab.
    Each slot wraps a QMediaPlayer + QVideoWidget, created fresh on load().
    """
    clicked         = pyqtSignal()
    positionChanged = pyqtSignal(int)   # ms
    durationChanged = pyqtSignal(int)   # ms

    def __init__(self, slot_index: int, label: str = "CAM", parent=None):
        super().__init__(parent)
        self.slot_index        = slot_index
        self.label             = label
        self._path:            Optional[str]          = None
        self._offset_sec:      float                  = 0.0
        self._duration_ms:     int                    = 0
        self._pending_seek_ms: Optional[int]          = None
        self._player:          Optional[QMediaPlayer] = None
        self._audio:           Optional[QAudioOutput] = None
        self._video_widget:    Optional[QVideoWidget] = None

        self.setObjectName("video_slot")
        self.setMinimumSize(200, 130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        top = QWidget()
        top.setFixedHeight(26)
        top.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(10, 0, 10, 0)
        tl.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(12)
        self._dot.setStyleSheet("color: #3a3d48; font-size: 11px;")

        self._slot_lbl = QLabel(f"{self.label} {self.slot_index:02d}")
        self._slot_lbl.setStyleSheet(
            "color: #6b7280; font-size: 11px; font-family: 'Courier New'; letter-spacing: 1px;"
        )
        self._file_lbl = QLabel("NO SOURCE")
        self._file_lbl.setStyleSheet(
            "color: #4a4f60; font-size: 9px; font-family: 'Courier New';"
        )
        self._file_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self._offset_lbl = QLabel("")
        self._offset_lbl.setStyleSheet(
            "color: #4a4f60; font-size: 9px; font-family: 'Courier New';"
        )

        tl.addWidget(self._dot)
        tl.addWidget(self._slot_lbl)
        tl.addSpacing(8)
        tl.addWidget(self._file_lbl)
        tl.addStretch()
        tl.addWidget(self._offset_lbl)
        self._root_layout.addWidget(top)

        self._container = QWidget()
        self._container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._container.setStyleSheet("background-color: #050506;")
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.addWidget(self._container)

        self._placeholder = QLabel(f"[ {self.label} {self.slot_index:02d} ]")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #1e2030; font-size: 14px; font-family: 'Courier New'; letter-spacing: 4px;"
        )
        self._container_layout.addWidget(self._placeholder)

    # ── Internal callbacks ────────────────────────────────────────────────────

    def _on_duration_changed(self, ms: int):
        ms = int(ms)
        if ms > 0:
            self._duration_ms = ms
            self.durationChanged.emit(ms)
            if self._pending_seek_ms is not None:
                target = min(self._pending_seek_ms, ms - 100)
                self._player.setPosition(max(0, target))
                self._pending_seek_ms = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, path: str, offset_sec: float = 0.0):
        self._path            = path
        self._offset_sec      = offset_sec
        self._duration_ms     = 0
        self._pending_seek_ms = None

        # Teardown previous player
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
        if self._video_widget is not None:
            self._container_layout.removeWidget(self._video_widget)
            self._video_widget.setParent(None)
            self._video_widget.deleteLater()
            self._video_widget = None

        self._player = QMediaPlayer()
        self._audio  = QAudioOutput()
        self._audio.setVolume(1.0)
        self._player.setAudioOutput(self._audio)

        self._video_widget = QVideoWidget(self.window())
        self._video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._player.setVideoOutput(self._video_widget)
        self._container_layout.addWidget(self._video_widget)

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(
            lambda ms: self.positionChanged.emit(int(ms))
        )

        self._placeholder.hide()
        self._player.setSource(QUrl.fromLocalFile(path))

        name = path.replace("\\", "/").split("/")[-1][:32]
        self._file_lbl.setText(name)
        self._file_lbl.setStyleSheet(
            "color: #9ca3af; font-size: 9px; font-family: 'Courier New';"
        )
        self._dot.setStyleSheet("color: #22c55e; font-size: 11px;")
        self._slot_lbl.setStyleSheet(
            "color: #d4d0c8; font-size: 11px; font-family: 'Courier New'; letter-spacing: 1px;"
        )
        if offset_sec != 0.0:
            sign = "+" if offset_sec >= 0 else ""
            self._offset_lbl.setText(f"{sign}{offset_sec:.3f}s")
            self._offset_lbl.setStyleSheet(
                "color: #4a9eff; font-size: 9px; font-family: 'Courier New';"
            )
        else:
            self._offset_lbl.setText("BASE")
            self._offset_lbl.setStyleSheet(
                "color: #e8ff00; font-size: 9px; font-family: 'Courier New';"
            )

    def play(self):
        if self._player and self._path:
            self._player.play()

    def pause(self):
        if self._player:
            self._player.pause()

    def seek(self, global_t: float):
        """Seek to global_t seconds (accounting for this slot's offset)."""
        if not self._path or not self._player:
            return
        local_ms = max(0, int((global_t + self._offset_sec) * 1000))
        if self._duration_ms > 0:
            local_ms = min(local_ms, self._duration_ms - 100)
            self._player.setPosition(local_ms)
        else:
            self._pending_seek_ms = local_ms

    def get_global_t(self) -> float:
        """Return the current playhead position in global seconds."""
        if not self._path or not self._player:
            return 0.0
        return max(0.0, self._player.position() / 1000.0 - self._offset_sec)

    def get_duration_sec(self) -> float:
        return self._duration_ms / 1000.0 if self._duration_ms > 0 else 0.0

    def is_loaded(self) -> bool:
        return self._path is not None

    def is_playing(self) -> bool:
        if not self._player:
            return False
        return (
            self._player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        )

    def set_idle(self):
        if self._player:
            self._player.stop()
            self._player.setSource(QUrl())
        if self._video_widget is not None:
            self._container_layout.removeWidget(self._video_widget)
            self._video_widget.setParent(None)
            self._video_widget.deleteLater()
            self._video_widget = None
        self._player          = None
        self._audio           = None
        self._path            = None
        self._offset_sec      = 0.0
        self._duration_ms     = 0
        self._pending_seek_ms = None
        self._placeholder.setText(f"[ {self.label} {self.slot_index:02d} ]")
        self._placeholder.show()
        self._file_lbl.setText("NO SOURCE")
        self._file_lbl.setStyleSheet(
            "color: #4a4f60; font-size: 9px; font-family: 'Courier New';"
        )
        self._dot.setStyleSheet("color: #3a3d48; font-size: 11px;")
        self._offset_lbl.setText("")


# ─── SyncPlaybackController ───────────────────────────────────────────────────

class SyncPlaybackController:
    """
    Coordinates play / pause / seek across a list of VLCVideoSlot objects
    so that all loaded slots stay in sync.

    Usage
    ─────
        ctrl = SyncPlaybackController()
        ctrl.attach(slots, offsets_sec=[0.0, -1.2, 0.3], global_duration=300.0)
        ctrl.play()
        ctrl.seek(42.5)   # jump to 42.5 s on the global timeline
        t = ctrl.current_global_t()
    """

    def __init__(self):
        self._slots:            list[VLCVideoSlot] = []
        self._offsets_sec:      list[float]        = []
        self._playing:          bool               = False
        self._saved_global_t:   float              = 0.0
        self._global_duration:  float              = 0.0
        self._ref_idx:          int                = 0

        # 100 ms poll — drives timeline scrub from playback position
        self._poll = QTimer()
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._refresh)

    # ── Configuration ─────────────────────────────────────────────────────────

    def attach(
        self,
        slots: list[VLCVideoSlot],
        offsets_sec: list[float] | None = None,
        global_duration: float = 0.0,
    ):
        """
        Register VLCVideoSlot objects.

        offsets_sec  — per-slot sync offset in seconds (same sign convention
                       as SourceOffset.offset_sec: positive = slot started
                       later than global t=0 → seek slot forward).
        global_duration — total session length in seconds; 0 = unknown.
        """
        if self._playing:
            for s in self._slots:
                if s.is_loaded():
                    s.pause()
            self._playing = False
            self._poll.stop()

        self._slots           = list(slots)
        self._offsets_sec     = list(offsets_sec) if offsets_sec else [0.0] * len(slots)
        self._saved_global_t  = 0.0
        self._global_duration = global_duration

        # Reference slot = first loaded slot with offset ≈ 0
        self._ref_idx = 0
        for i, (slot, off) in enumerate(zip(self._slots, self._offsets_sec)):
            if slot.is_loaded() and abs(off) < 1e-6:
                self._ref_idx = i
                break

        self._seek_all(0.0)

    # ── Playback ──────────────────────────────────────────────────────────────

    def play(self):
        self._seek_all(self._saved_global_t)
        for slot in self._slots:
            if slot.is_loaded():
                slot.play()
        self._playing = True
        self._poll.start()

    def pause(self):
        self._refresh()   # snapshot position before pausing
        for slot in self._slots:
            if slot.is_loaded():
                slot.pause()
        self._playing = False
        self._poll.stop()

    def toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def seek(self, global_t: float):
        """Seek all slots to global_t (seconds on the global timeline)."""
        self._saved_global_t = global_t
        self._seek_all(global_t)

    # ── State ─────────────────────────────────────────────────────────────────

    def is_playing(self) -> bool:
        return self._playing

    def current_global_t(self) -> float:
        """Current position in seconds on the global timeline."""
        ref = self._get_ref()
        if ref and ref.is_loaded():
            return ref.get_global_t()
        return self._saved_global_t

    # ── Internal ──────────────────────────────────────────────────────────────

    def _seek_all(self, global_t: float):
        for slot in self._slots:
            if slot.is_loaded():
                slot.seek(global_t)

    def _get_ref(self) -> Optional[VLCVideoSlot]:
        if self._slots and self._ref_idx < len(self._slots):
            return self._slots[self._ref_idx]
        return None

    def _refresh(self):
        ref = self._get_ref()
        if ref and ref.is_loaded():
            self._saved_global_t = ref.get_global_t()
