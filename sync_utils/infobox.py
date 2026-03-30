from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt


class Infobox(QWidget):
    def __init__(self, window_title: str = "Info", parent=None):
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.setMinimumWidth(480)
        self.setMinimumHeight(100)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

    def update_message(self, message: str):
        self.label.setText(message)
        QApplication.processEvents()
