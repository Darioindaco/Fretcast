"""Fase 5: QThread audio worker driving StreamingTranscriber, live or from a WAV file.

Runs the whole audio-capture -> streaming-inference chain on a background thread and
emits a Qt signal per newly-confirmed note. Qt's signal/slot mechanism is already
thread-safe for getting results to the UI thread (auto-queued across threads) -- no
separate IPC layer needed now that everything lives in one process (an earlier draft,
core/scripts/websocket_server.py, bridged to a separate JS/Expo renderer over a
websocket; that approach was dropped in favor of a single Python desktop app, see the
plan this shipped under).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(CORE_DIR / "scripts"))

from PySide6.QtCore import QThread, Signal  # noqa: E402

from run_baseline import load_model  # noqa: E402
from streaming_infer import StreamingTranscriber  # noqa: E402

STREAMING_CHECKPOINT = Path("D:/FretcastData/artifacts/streaming_v1/best_model.pth")
STREAMING_RUN_CONFIG = Path("D:/FretcastData/artifacts/streaming_v1/run_configuration.json")

WAV_BLOCK_SIZE = 512  # samples at tx_config.SAMPLE_RATE, matches validate_streaming.py's default


class AudioWorker(QThread):
    # time_sec, string (1=high e..6=low E), fret, midi_pitch, note name (e.g. "B3")
    note_detected = Signal(float, int, int, int, str)
    status_changed = Signal(str)

    def __init__(self, source: str = "live", wav_path: Path | None = None, parent=None):
        super().__init__(parent)
        self.source = source
        self.wav_path = wav_path
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        self.status_changed.emit("model laden...")
        model_handle = load_model(STREAMING_CHECKPOINT, STREAMING_RUN_CONFIG)
        transcriber = StreamingTranscriber(model_handle)

        if self.source == "wav":
            self._run_wav(transcriber, model_handle)
        else:
            self._run_live(transcriber, model_handle)

    def _emit_events(self, events) -> None:
        for e in events:
            self.note_detected.emit(e.time, e.string, e.fret, e.pitch, e.name)

    def _run_wav(self, transcriber: StreamingTranscriber, model_handle) -> None:
        import librosa

        tx_config = model_handle[2]
        audio, _ = librosa.load(str(self.wav_path), sr=tx_config.SAMPLE_RATE, mono=True)
        block_seconds = WAV_BLOCK_SIZE / tx_config.SAMPLE_RATE
        start = time.monotonic()
        self.status_changed.emit(f"afspelen: {self.wav_path.name}")

        for i, block_start in enumerate(range(0, len(audio), WAV_BLOCK_SIZE)):
            if self._stop_requested:
                self.status_changed.emit("gestopt")
                return
            block = audio[block_start : block_start + WAV_BLOCK_SIZE]
            self._emit_events(transcriber.push_audio(block))
            sleep_seconds = start + (i + 1) * block_seconds - time.monotonic()
            if sleep_seconds > 0:
                self.msleep(int(sleep_seconds * 1000))

        self._emit_events(transcriber.finalize())
        self.status_changed.emit("afspelen klaar")

    def _run_live(self, transcriber: StreamingTranscriber, model_handle) -> None:
        import numpy as np
        import sounddevice as sd
        from scipy.signal import resample_poly

        from live_capture import NATIVE_SAMPLE_RATE, find_input_device

        tx_config = model_handle[2]
        downsample_factor = NATIVE_SAMPLE_RATE // tx_config.SAMPLE_RATE
        device_index = find_input_device()

        def callback(indata, frames, time_info, status):
            mono = indata[:, 0].astype(np.float32)
            resampled = resample_poly(mono, up=1, down=downsample_factor).astype(np.float32)
            self._emit_events(transcriber.push_audio(resampled))

        self.status_changed.emit("luistert (WASAPI exclusive)...")
        with sd.InputStream(
            device=device_index,
            samplerate=NATIVE_SAMPLE_RATE,
            channels=1,
            blocksize=1024,
            dtype="float32",
            extra_settings=sd.WasapiSettings(exclusive=True),
            callback=callback,
        ):
            while not self._stop_requested:
                sd.sleep(200)

        self._emit_events(transcriber.finalize())
        self.status_changed.emit("gestopt")
