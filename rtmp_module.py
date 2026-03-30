"""
rtmp_module.py  —  Brain Battle RTMP live stream viewer

Approach
────────
We run ffplay as a subprocess with low-latency flags (identical to the
existing Brain Battle shell scripts) and embed its window inside the Qt
widget using the native window ID. This avoids the VLC/python-vlc
dependency and works identically to the existing workflow.

On macOS ffplay cannot be embedded (no foreign window reparenting), so
we fall back to rendering the stream in a detached floating ffplay window
and showing a "STREAMING — external window" indicator in the slot.

On Windows / Linux, XCB / Win32 embedding works via winId().

Each RTMPSlot widget:
  - Has an editable URL field (pre-filled with default)
  - CONNECT / DISCONNECT button
  - Status indicator (idle / connecting / live / error)
  - Slot label (DSLR 01, DSLR 02, 360 03, 360 04)

RTMPConfigPanel (used inside LiveMonitorTab):
  - Shows a 2×2 grid of RTMPSlot widgets
  - "Connect All" button
  - Local IP auto-detection helper

Default RTMP URLs (matching MonaServer2 defaults from the docs):
  Slot 1 (DSLR 01):  rtmp://<ip>:1935
  Slot 2 (DSLR 02):  rtmp://<ip>:9999
  Slot 3 (360 03):   rtmp://<ip>:9998
  Slot 4 (360 04):   rtmp://<ip>:9997
"""

from __future__ import annotations

import subprocess
import sys
import socket
import platform
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QProcess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy, QLineEdit,
    QGridLayout,
)
from PyQt6.QtGui import QWindow

# ─── Constants ────────────────────────────────────────────────────────────────

# Same low-latency flags used in the existing Brain Battle shell scripts
_FFPLAY_FLAGS = [
    "-an",
    "-flags", "low_delay",
    "-fflags", "nobuffer",
    "-framedrop",
    "-strict", "experimental",
    "-probesize", "32",
    "-analyzeduration", "0",
    "-sync", "ext",
]

_IS_MAC = platform.system() == "Darwin"

_DEFAULT_PORTS = [1935, 9999, 9998, 9997]

_SLOT_LABELS = ["DSLR 01", "DSLR 02", "360  03", "360  04"]


# ─── IP detection ─────────────────────────────────────────────────────────────

def get_local_ip() -> str:
    """Return the local WLAN IP address (best guess)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.1.1"


# ─── ffplay health monitor thread ────────────────────────────────────────────

class _StreamMonitor(QThread):
    """
    Watches the ffplay subprocess. Emits disconnected() if the process
    exits unexpectedly.
    """
    disconnected = pyqtSignal(str)   # reason string

    def __init__(self, proc: subprocess.Popen, parent=None):
        super().__init__(parent)
        self._proc = proc

    def run(self):
        try:
            stdout, stderr = self._proc.communicate(timeout=None)
            # If we get here the process exited
            if self._proc.returncode != 0:
                err = (stderr or b"").decode(errors="replace")[-200:]
                self.disconnected.emit(err or "ffplay exited unexpectedly")
            else:
                self.disconnected.emit("Stream ended")
        except Exception as e:
            self.disconnected.emit(str(e))


# ─── Single RTMP slot widget ──────────────────────────────────────────────────

class RTMPSlot(QFrame):
    """
    One camera slot in the Live Monitor grid.

    On macOS: ffplay opens in a separate floating window (macOS does not
    support reparenting foreign windows). The slot shows a status banner.

    On Linux/Windows: ffplay is embedded via winId() of the video_frame.
    """

    def __init__(self, slot_index: int, label: str = "CAM",
                 default_url: str = "", parent=None):
        super().__init__(parent)
        self.slot_index  = slot_index
        self.label       = label
        self.default_url = default_url
        self.setObjectName("video_slot")
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._proc:    Optional[subprocess.Popen] = None
        self._monitor: Optional[_StreamMonitor]   = None
        self._is_live: bool = False

        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Top bar ──
        top = QWidget()
        top.setFixedHeight(28)
        top.setStyleSheet("background-color: #0e0f11; border-bottom: 1px solid #2a2d35;")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(10, 0, 10, 0)
        tl.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(12)
        self._dot.setStyleSheet("color: #4a4f60; font-size: 12px;")

        self._slot_lbl = QLabel(f"{self.label}")
        self._slot_lbl.setStyleSheet(
            "color: #9ca3af; font-size: 12px; font-family: 'Courier New'; letter-spacing: 1px;"
        )

        self._status_lbl = QLabel("NO STREAM")
        self._status_lbl.setStyleSheet(
            "color: #9ca3af; font-size: 12px; font-family: 'Courier New';"
        )

        tl.addWidget(self._dot)
        tl.addWidget(self._slot_lbl)
        tl.addSpacing(8)
        tl.addWidget(self._status_lbl)
        tl.addStretch()
        layout.addWidget(top)

        # ── URL bar ──
        url_bar = QWidget()
        url_bar.setFixedHeight(34)
        url_bar.setStyleSheet("background-color: #0a0b0d; border-bottom: 1px solid #1a1c24;")
        ul = QHBoxLayout(url_bar)
        ul.setContentsMargins(8, 0, 8, 0)
        ul.setSpacing(6)

        self._url_input = QLineEdit(self.default_url)
        self._url_input.setStyleSheet(
            "background: transparent; border: none; color: #9ca3af;"
            "font-family: 'Courier New'; font-size: 11px;"
        )
        self._url_input.setPlaceholderText("rtmp://ip:port")

        self._connect_btn = QPushButton("CONNECT")
        self._connect_btn.setFixedSize(100, 24)
        self._connect_btn.setStyleSheet(
            "background:#1a1f0a; border:1px solid #e8ff00; color:#e8ff00;"
            "font-family:'Courier New'; font-size:11px; letter-spacing:1px; padding: 0 8px;"
        )
        self._connect_btn.clicked.connect(self._toggle)

        ul.addWidget(self._url_input)
        ul.addWidget(self._connect_btn)
        layout.addWidget(url_bar)

        # ── Video frame (embedding target on Linux/Windows) ──
        self._video_frame = QWidget()
        self._video_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_frame.setStyleSheet("background-color: #050506;")
        layout.addWidget(self._video_frame)

        # ── Placeholder overlay ──
        ph_layout = QVBoxLayout(self._video_frame)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder = QLabel(f"[ {self.label} ]")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #1e2030; font-size: 15px; font-family: 'Courier New'; letter-spacing: 4px;"
        )
        ph_layout.addWidget(self._placeholder)

    # ── Stream control ────────────────────────────────────────────────────────

    def _toggle(self):
        if self._is_live:
            self.disconnect_stream()
        else:
            self.connect_stream(self._url_input.text().strip())

    def connect_stream(self, url: str = ""):
        if self._is_live:
            return
        url = url or self._url_input.text().strip()
        if not url:
            self._set_status("NO URL", "#ef4444")
            return

        self._url_input.setText(url)
        self._set_status("CONNECTING…", "#f59e0b")

        # Build ffplay command
        ffplay_bin = self._find_ffplay()
        if not ffplay_bin:
            self._set_status("ffplay NOT FOUND", "#ef4444")
            return

        cmd = [ffplay_bin] + _FFPLAY_FLAGS

        if _IS_MAC:
            # macOS: floating window — no embedding
            cmd += ["-window_title", f"Brain Battle — {self.label}", url]
        else:
            # Linux / Windows: embed into video_frame
            win_id = int(self._video_frame.winId())
            cmd += [
                "-wid" if sys.platform == "win32" else "-wid",
                str(win_id),
                url,
            ]

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Monitor subprocess for unexpected exits
        self._monitor = _StreamMonitor(self._proc, parent=self)
        self._monitor.disconnected.connect(self._on_stream_lost)
        self._monitor.start()

        self._is_live = True
        self._placeholder.setVisible(_IS_MAC)  # hide on Linux/Win (ffplay fills it)
        self._set_live()

    def disconnect_stream(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._monitor:
            self._monitor.quit()
            self._monitor = None
        self._is_live = False
        self._set_idle()
        self._placeholder.setVisible(True)

    def _on_stream_lost(self, reason: str):
        self._proc = None
        self._is_live = False
        self._set_status(f"LOST — {reason[:30]}", "#ef4444")
        self._dot.setStyleSheet("color: #ef4444; font-size: 11px;")
        self._connect_btn.setText("RECONNECT")
        self._connect_btn.setStyleSheet(
            "background:#2a0f0f; border:1px solid #ef4444; color:#ef4444;"
            "font-family:'Courier New'; font-size:9px; letter-spacing:1px;"
        )
        self._placeholder.setVisible(True)
        self._placeholder.setText(f"[ {self.label} ]\nSTREAM LOST")
        self._placeholder.setStyleSheet(
            "color: #ef4444; font-size: 12px; font-family: 'Courier New';"
            "letter-spacing: 2px; text-align: center;"
        )

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _set_live(self):
        self._dot.setStyleSheet("color: #22c55e; font-size: 11px;")
        self._slot_lbl.setStyleSheet(
            "color: #d4d0c8; font-size: 11px; font-family: 'Courier New'; letter-spacing: 1px;"
        )
        if _IS_MAC:
            self._set_status("LIVE — external window", "#22c55e")
            self._placeholder.setText(f"[ {self.label} ]\nLIVE — see floating window")
            self._placeholder.setStyleSheet(
                "color: #22c55e; font-size: 11px; font-family: 'Courier New';"
                "letter-spacing: 1px; text-align: center;"
            )
        else:
            self._set_status("LIVE", "#22c55e")
        self._connect_btn.setText("DISCONNECT")
        self._connect_btn.setStyleSheet(
            "background:transparent; border:1px solid #ef4444; color:#ef4444;"
            "font-family:'Courier New'; font-size:9px; letter-spacing:1px;"
        )
        self._url_input.setEnabled(False)

    def _set_idle(self):
        self._dot.setStyleSheet("color: #4a4f60; font-size: 12px;")
        self._slot_lbl.setStyleSheet(
            "color: #9ca3af; font-size: 12px; font-family: 'Courier New'; letter-spacing: 1px;"
        )
        self._set_status("NO STREAM", "#3a3d48")
        self._connect_btn.setText("CONNECT")
        self._connect_btn.setStyleSheet(
            "background:#1a1f0a; border:1px solid #e8ff00; color:#e8ff00;"
            "font-family:'Courier New'; font-size:11px; letter-spacing:1px; padding: 0 8px;"
        )
        self._url_input.setEnabled(True)
        self._placeholder.setText(f"[ {self.label} ]")
        self._placeholder.setStyleSheet(
            "color: #1e2030; font-size: 15px; font-family: 'Courier New'; letter-spacing: 4px;"
        )

    def _set_status(self, text: str, color: str):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"color: {color}; font-size: 9px; font-family: 'Courier New';"
        )

    @staticmethod
    def _find_ffplay() -> Optional[str]:
        """Locate ffplay binary on PATH."""
        import shutil
        return shutil.which("ffplay")


# ─── RTMP config panel ────────────────────────────────────────────────────────

class RTMPConfigPanel(QWidget):
    """
    2×2 grid of RTMPSlot widgets with a header bar that shows
    the detected local IP and a Connect All / Disconnect All button.

    Used inside LiveMonitorTab replacing the placeholder VideoSlot grid.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slots: list[RTMPSlot] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar ──
        header = QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet("background-color: #0a0b0d; border-bottom: 1px solid #2a2d35;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.setSpacing(12)

        self._ip_lbl = QLabel("LOCAL IP: detecting…")
        self._ip_lbl.setStyleSheet(
            "color: #9ca3af; font-size: 12px; font-family: 'Courier New';"
        )

        self._all_btn = QPushButton("CONNECT ALL")
        self._all_btn.setFixedHeight(26)
        self._all_btn.setStyleSheet(
            "background:#1a1f0a; border:1px solid #e8ff00; color:#e8ff00;"
            "font-family:'Courier New'; font-size:11px; padding:0 18px;"
        )
        self._all_connected = False
        self._all_btn.clicked.connect(self._toggle_all)

        hl.addWidget(self._ip_lbl)
        hl.addStretch()
        hl.addWidget(self._all_btn)
        layout.addWidget(header)

        # ── 2×2 grid ──
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)

        local_ip = get_local_ip()
        self._ip_lbl.setText(f"LOCAL IP: {local_ip}")

        for i in range(4):
            port = _DEFAULT_PORTS[i]
            url  = f"rtmp://{local_ip}:{port}" if port != 1935 else f"rtmp://{local_ip}"
            slot = RTMPSlot(
                slot_index=i + 1,
                label=_SLOT_LABELS[i],
                default_url=url,
            )
            row, col = divmod(i, 2)
            grid.addWidget(slot, row, col)
            self._slots.append(slot)

        layout.addLayout(grid)

        # Refresh IP after a moment (network may not be ready at init)
        QTimer.singleShot(1500, self._refresh_ip)

    def _refresh_ip(self):
        local_ip = get_local_ip()
        self._ip_lbl.setText(f"LOCAL IP: {local_ip}")
        for i, slot in enumerate(self._slots):
            port = _DEFAULT_PORTS[i]
            current = slot._url_input.text()
            # Only update if still at default (user hasn't edited it)
            if not slot._is_live and (
                current.startswith("rtmp://192.168") or
                current.startswith("rtmp://10.") or
                not current
            ):
                url = f"rtmp://{local_ip}:{port}" if port != 1935 else f"rtmp://{local_ip}"
                slot._url_input.setText(url)

    def _toggle_all(self):
        if self._all_connected:
            for slot in self._slots:
                slot.disconnect_stream()
            self._all_connected = False
            self._all_btn.setText("CONNECT ALL")
            self._all_btn.setStyleSheet(
                "background:#1a1f0a; border:1px solid #e8ff00; color:#e8ff00;"
                "font-family:'Courier New'; font-size:11px; padding:0 18px;"
            )
        else:
            for slot in self._slots:
                if not slot._is_live:
                    slot.connect_stream()
            self._all_connected = True
            self._all_btn.setText("DISCONNECT ALL")
            self._all_btn.setStyleSheet(
                "background:transparent; border:1px solid #ef4444; color:#ef4444;"
                "font-family:'Courier New'; font-size:11px; padding:0 18px;"
            )

    def get_slots(self) -> list[RTMPSlot]:
        return self._slots
