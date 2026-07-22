"""Fase 5: main window -- audio-only live note list, no camera/overlay yet.

Deliberately minimal: a status label, a start/stop button, and a table of confirmed
notes (newest on top). This validates the desktop-app plumbing (worker thread -> model
-> Qt signals -> UI) before the camera/ArUco overlay (a separate, later piece that
needs Dario physically present with his guitar and a webcam) gets built on top.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from audio_worker import AudioWorker


class MainWindow(QMainWindow):
    def __init__(self, source: str = "live", wav_path: Path | None = None):
        super().__init__()
        self.setWindowTitle("Fretcast — live (audio-only)")
        self.resize(480, 640)

        self.worker = AudioWorker(source=source, wav_path=wav_path)
        self.worker.note_detected.connect(self._on_note)
        self.worker.status_changed.connect(self._on_status)
        self._running = False

        central = QWidget()
        layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.status_label = QLabel("gestopt")
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._toggle)
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()
        top_bar.addWidget(self.start_button)
        layout.addLayout(top_bar)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["tijd", "snaar", "fret", "noot"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        self.setCentralWidget(central)

    def _toggle(self) -> None:
        if self._running:
            self.worker.stop()
            self.worker.wait(2000)
            self.start_button.setText("Start")
            self._running = False
        else:
            self.table.setRowCount(0)
            self.worker.start()
            self.start_button.setText("Stop")
            self._running = True

    def _on_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_note(self, time_sec: float, string: int, fret: int, pitch: int, name: str) -> None:
        self.table.insertRow(0)
        self.table.setItem(0, 0, QTableWidgetItem(f"{time_sec:.2f}s"))
        self.table.setItem(0, 1, QTableWidgetItem(str(string)))
        self.table.setItem(0, 2, QTableWidgetItem(str(fret)))
        self.table.setItem(0, 3, QTableWidgetItem(name))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._running:
            self.worker.stop()
            self.worker.wait(2000)
        event.accept()
