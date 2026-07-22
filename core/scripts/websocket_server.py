"""Fase 5 bridge: broadcast live TabEvents over a local WebSocket to the frontend.

Per the V02 plan's own architecture consequence (fase-3-plan discussion): the existing
Expo/RN app becomes a pure renderer, driven by this local Python inference server over a
websocket, instead of its own legacy JS audio pipeline (HPS/chroma chord detection).
This is the wiring itself -- no camera/ArUco overlay wiring yet (fase 5's own AR-neck
part), just getting real detected notes from the Python model to the frontend at all.

Two source modes:
  --source live      real audio via live_capture.py's WASAPI capture (needs Dario at
                      his interface, actually playing -- can't be tested by the agent)
  --source wav PATH  replay a WAV file with real-time-like pacing (sleeps between
                      chunks to match wall-clock time), so the whole server+frontend
                      pipeline can be validated end-to-end without a guitar or the
                      Focusrite plugged in -- what the agent actually tests with.

Wire format (one JSON object per line, sent to every connected client for each new
TabEvent): {"time": 1.347, "string": 3, "fret": 4, "pitch": 59, "name": "B3"}

Usage:
    python core/scripts/websocket_server.py --source wav D:/FretcastData/egdb/audio_DI/104.wav
    python core/scripts/websocket_server.py --source live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import websockets  # noqa: E402

from run_baseline import load_model  # noqa: E402
from streaming_infer import StreamingTranscriber  # noqa: E402
from fretcast.tab import TabEvent  # noqa: E402

STREAMING_CHECKPOINT = Path("D:/FretcastData/artifacts/streaming_v1/best_model.pth")
STREAMING_RUN_CONFIG = Path("D:/FretcastData/artifacts/streaming_v1/run_configuration.json")

CONNECTED_CLIENTS: set = set()


def event_to_json(e: TabEvent) -> str:
    return json.dumps({"time": round(e.time, 4), "string": e.string, "fret": e.fret, "pitch": e.pitch, "name": e.name})


async def broadcast(message: str) -> None:
    if CONNECTED_CLIENTS:
        await asyncio.gather(*(c.send(message) for c in CONNECTED_CLIENTS), return_exceptions=True)


async def handle_client(websocket) -> None:
    CONNECTED_CLIENTS.add(websocket)
    print(f"client connected ({len(CONNECTED_CLIENTS)} total)")
    try:
        async for _ in websocket:
            pass  # this server is broadcast-only; ignore anything a client sends
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        print(f"client disconnected ({len(CONNECTED_CLIENTS)} total)")


async def run_wav_source(wav_path: Path, model_handle, block_size: int = 512) -> None:
    """Replay a WAV file through StreamingTranscriber with real-time-like pacing."""
    import librosa

    tx_config = model_handle[2]
    audio, _ = librosa.load(str(wav_path), sr=tx_config.SAMPLE_RATE, mono=True)
    transcriber = StreamingTranscriber(model_handle)
    block_seconds = block_size / tx_config.SAMPLE_RATE
    start = time.monotonic()

    print(f"replaying {wav_path.name} ({len(audio) / tx_config.SAMPLE_RATE:.1f}s) at real-time pace...")
    for i, block_start in enumerate(range(0, len(audio), block_size)):
        block = audio[block_start : block_start + block_size]
        events = transcriber.push_audio(block)
        for e in events:
            print(f"  [{time.monotonic() - start:6.2f}s] {e}")
            await broadcast(event_to_json(e))
        target_time = start + (i + 1) * block_seconds
        await asyncio.sleep(max(0.0, target_time - time.monotonic()))

    for e in transcriber.finalize():
        print(f"  [final] {e}")
        await broadcast(event_to_json(e))
    print("replay finished.")


async def run_live_source(model_handle) -> None:
    """Real WASAPI capture (see live_capture.py) -- runs the callback on a background
    thread (sounddevice's own model) and forwards events into the asyncio event loop."""
    import sounddevice as sd
    from scipy.signal import resample_poly
    from live_capture import NATIVE_SAMPLE_RATE, find_input_device

    tx_config = model_handle[2]
    downsample_factor = NATIVE_SAMPLE_RATE // tx_config.SAMPLE_RATE
    transcriber = StreamingTranscriber(model_handle)
    device_index = find_input_device()
    loop = asyncio.get_running_loop()
    start = time.monotonic()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        mono = indata[:, 0].astype(np.float32)
        resampled = resample_poly(mono, up=1, down=downsample_factor).astype(np.float32)
        events = transcriber.push_audio(resampled)
        for e in events:
            print(f"  [{time.monotonic() - start:6.2f}s] {e}")
            asyncio.run_coroutine_threadsafe(broadcast(event_to_json(e)), loop)

    print(f"opening WASAPI exclusive capture on device {device_index}...")
    with sd.InputStream(
        device=device_index,
        samplerate=NATIVE_SAMPLE_RATE,
        channels=1,
        blocksize=1024,
        dtype="float32",
        extra_settings=sd.WasapiSettings(exclusive=True),
        callback=callback,
    ):
        print("listening (Ctrl+C to stop)...")
        while True:
            await asyncio.sleep(0.2)


async def main_async(args) -> None:
    model_handle = load_model(args.checkpoint, args.run_config)

    async with websockets.serve(handle_client, "localhost", args.port):
        print(f"websocket server listening on ws://localhost:{args.port}")
        if args.source == "wav":
            await run_wav_source(args.wav_path, model_handle)
        else:
            await run_live_source(model_handle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["wav", "live"], required=True)
    ap.add_argument("--wav-path", type=Path, help="required when --source wav")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--checkpoint", type=Path, default=STREAMING_CHECKPOINT)
    ap.add_argument("--run-config", type=Path, default=STREAMING_RUN_CONFIG)
    args = ap.parse_args()

    if args.source == "wav" and not args.wav_path:
        ap.error("--wav-path is required when --source wav")

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
