"""
vlc_module.py  —  Brain Battle

SyncPlaybackController only.
Operates directly on a list of QMediaPlayer objects — no wrapper classes.
Mirrors MainEditor.py's play_pause_all_videos() and set_position() exactly.
"""

from __future__ import annotations
from PyQt6.QtCore import QTimer
from PyQt6.QtMultimedia import QMediaPlayer


class SyncPlaybackController:
    """
    Mirrors MainEditor.py:

        def play_pause_all_videos(self):
            for player in self.media_players:
                if player:
                    if player.state() == QMediaPlayer.PlayingState:
                        player.pause()
                    else:
                        player.play()

        def set_position(self, position):
            for player in self.media_players:
                if player:
                    player.setPosition(position)
    """

    def __init__(self):
        self._players:       list                  = []   # list of QMediaPlayer|None
        self._offsets_ms:    list[int]             = []   # offset per player in ms
        self._playing:       bool                  = False
        self._saved_pos_ms:  int                   = 0    # position saved on pause
        self._ref_idx:       int                   = 0    # reference player index

        # Read-only poll timer — never seeks during playback
        self._poll = QTimer()
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._refresh)

    # ── Configuration ─────────────────────────────────────────────────────────

    def attach(self, players: list, offsets_ms: list[int] = None,
               global_duration_ms: int = 0):
        """
        Register the list of QMediaPlayer objects (mirrors self.media_players
        in MainEditor.py).  offsets_ms is the per-player seek offset after sync.
        """
        # Stop anything playing first
        if self._playing:
            for p in self._players:
                if p:
                    p.pause()
            self._playing = False
            self._poll.stop()

        self._players      = players
        self._offsets_ms   = offsets_ms if offsets_ms else [0] * len(players)
        self._saved_pos_ms = 0

        # Reference = player with offset 0 (or first loaded player)
        self._ref_idx = 0
        for i, off in enumerate(self._offsets_ms):
            if off == 0 and players[i] is not None:
                self._ref_idx = i
                break

        # Pre-seek all players to their start positions
        self._seek_all(0)

    # ── Playback — mirrors MainEditor.py exactly ───────────────────────────────

    def play(self):
        """
        Mirrors:
            for player in self.media_players:
                if player:
                    player.play()
        With a pre-seek to the saved position first so resume works correctly.
        """
        self._seek_all(self._saved_pos_ms)
        for p in self._players:
            if p:
                p.play()
        self._playing = True
        self._poll.start()

    def pause(self):
        """
        Mirrors:
            for player in self.media_players:
                if player:
                    player.pause()
        Saves position before pausing.
        """
        self._refresh()   # snapshot position first
        for p in self._players:
            if p:
                p.pause()
        self._playing = False
        self._poll.stop()

    def toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def seek(self, global_pos_ms: int):
        """
        Mirrors:
            def set_position(self, position):
                for player in self.media_players:
                    if player:
                        player.setPosition(position)
        """
        self._saved_pos_ms = global_pos_ms
        self._seek_all(global_pos_ms)

    # ── State ──────────────────────────────────────────────────────────────────

    def is_playing(self) -> bool:
        return self._playing

    def current_pos_ms(self) -> int:
        """Current position in global ms (reference player minus its offset)."""
        ref = self._get_ref()
        if ref:
            raw = ref.position()
            return max(0, raw - self._offsets_ms[self._ref_idx])
        return self._saved_pos_ms

    # ── Internal ───────────────────────────────────────────────────────────────

    def _seek_all(self, global_pos_ms: int):
        """Seek every player to global_pos_ms + that player's own offset."""
        for i, p in enumerate(self._players):
            if p:
                local_ms = max(0, global_pos_ms + self._offsets_ms[i])
                p.setPosition(local_ms)

    def _get_ref(self):
        if self._players and self._ref_idx < len(self._players):
            return self._players[self._ref_idx]
        return None

    def _refresh(self):
        ref = self._get_ref()
        if ref:
            raw = ref.position()
            self._saved_pos_ms = max(0, raw - self._offsets_ms[self._ref_idx])


# ─── VLCVideoSlot — used by LabellingTab only ────────────────────────────────

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QSizePolicy,
)
from typing import Optional


class VLCVideoSlot(QFrame):
    """
    Used exclusively by LabellingTab.
    Mirrors MainEditor.py upload_video() — creates QMediaPlayer + QVideoWidget
    fresh on each load(), parents the video widget to self (the frame).
    """
    clicked         = pyqtSignal()
    positionChanged = pyqtSignal(int)
    durationChanged = pyqtSignal(int)

    def __init__(self, slot_index: int, label: str = "CAM", parent=None):
        super().__init__(parent)
        self.slot_index       = slot_index
        self.label            = label
        self._path:           Optional[str]          = None
        self._offset_sec:     float                  = 0.0
        self._duration_ms:    int                    = 0
        self._pending_seek_ms: Optional[int]         = None
        self._player:         Optional[QMediaPlayer] = None
        self._audio:          Optional[QAudioOutput] = None
        self._video_widget:   Optional[QVideoWidget] = None

        self.setObjectName("video_slot")
        self.setMinimumSize(200, 130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._build()

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
        self._file_lbl.setStyleSheet("color: #4a4f60; font-size: 9px; font-family: 'Courier New';")
        self._file_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._offset_lbl = QLabel("")
        self._offset_lbl.setStyleSheet("color: #4a4f60; font-size: 9px; font-family: 'Courier New';")

        tl.addWidget(self._dot)
        tl.addWidget(self._slot_lbl)
        tl.addSpacing(8)
        tl.addWidget(self._file_lbl)
        tl.addStretch()
        tl.addWidget(self._offset_lbl)
        self._root_layout.addWidget(top)

        self._container = QWidget()
        self._container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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

    def _on_duration_changed(self, ms):
        ms = int(ms)
        if ms > 0:
            self._duration_ms = ms
            self.durationChanged.emit(ms)
            if self._pending_seek_ms is not None:
                target = min(self._pending_seek_ms, ms - 100)
                self._player.setPosition(max(0, target))
                self._pending_seek_ms = None

    def load(self, path: str, offset_sec: float = 0.0):
        self._path            = path
        self._offset_sec      = offset_sec
        self._duration_ms     = 0
        self._pending_seek_ms = None

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
        self._file_lbl.setStyleSheet("color: #9ca3af; font-size: 9px; font-family: 'Courier New';")
        self._dot.setStyleSheet("color: #22c55e; font-size: 11px;")
        self._slot_lbl.setStyleSheet(
            "color: #d4d0c8; font-size: 11px; font-family: 'Courier New'; letter-spacing: 1px;"
        )
        if offset_sec != 0.0:
            sign = "+" if offset_sec >= 0 else ""
            self._offset_lbl.setText(f"{sign}{offset_sec:.3f}s")
            self._offset_lbl.setStyleSheet("color: #4a9eff; font-size: 9px; font-family: 'Courier New';")
        else:
            self._offset_lbl.setText("BASE")
            self._offset_lbl.setStyleSheet("color: #e8ff00; font-size: 9px; font-family: 'Courier New';")

    def play(self):
        if self._player and self._path:
            self._player.play()

    def pause(self):
        if self._player:
            self._player.pause()

    def seek(self, global_t: float):
        if not self._path or not self._player:
            return
        local_ms = max(0, int((global_t + self._offset_sec) * 1000))
        if self._duration_ms > 0:
            local_ms = min(local_ms, self._duration_ms - 100)
            self._player.setPosition(local_ms)
        else:
            self._pending_seek_ms = local_ms

    def get_global_t(self) -> float:
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
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

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
        self._file_lbl.setStyleSheet("color: #4a4f60; font-size: 9px; font-family: 'Courier New';")
        self._dot.setStyleSheet("color: #3a3d48; font-size: 11px;")
        self._offset_lbl.setText("")