"""Fretcast desktop app entrypoint (audio-only live note viewer).

Usage:
    python -m core.gui.app --source wav --wav-path D:/FretcastData/egdb/audio_DI/104.wav
    python -m core.gui.app --source live
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

from main_window import MainWindow  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["live", "wav"], default="live")
    ap.add_argument("--wav-path", type=Path, help="required when --source wav")
    args = ap.parse_args()

    if args.source == "wav" and not args.wav_path:
        ap.error("--wav-path is required when --source wav")

    app = QApplication(sys.argv)
    window = MainWindow(source=args.source, wav_path=args.wav_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
