"""Fase 3: real audio capture, feeding StreamingTranscriber from a live interface.

This machine has no ASIO backend at all (`sd.query_hostapis()` only lists MME/
DirectSound/WASAPI/WDM-KS -- confirmed, not a config error; the standard pip
`sounddevice`/PortAudio wheel doesn't ship ASIO support for licensing reasons). WASAPI
exclusive mode is the substitute: it bypasses the Windows audio mixer directly, giving
latency in the same ballpark as ASIO without needing a custom PortAudio build against
Focusrite's ASIO SDK.

WASAPI exclusive mode needs the device's own native sample rate -- Dario's Focusrite
reports 44100 Hz as its default (`sd.query_devices()`), not the 22050 Hz the model was
trained on. This captures at 44100 and downsamples 2:1 (exact ratio, cheap polyphase
resample) before handing audio to StreamingTranscriber.

I (the agent) cannot play guitar into this to verify real transcription accuracy --
that needs a human at the interface. What I *can* and did verify: the device opens in
WASAPI exclusive mode and the callback -> resample -> StreamingTranscriber pipeline
runs without exceptions on a few seconds of whatever's actually coming into the input
(see the --duration smoke test in this session's notes). Run this yourself and play to
see real detections.

Usage:
    python core/scripts/live_capture.py                  # runs until Ctrl+C
    python core/scripts/live_capture.py --duration 10     # runs for 10 seconds then stops
    python core/scripts/live_capture.py --list-devices    # show available input devices
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from run_baseline import load_model  # noqa: E402
from streaming_infer import StreamingTranscriber  # noqa: E402

NATIVE_SAMPLE_RATE = 44100  # Focusrite's reported default_samplerate; must match its native rate for exclusive mode

# The causal (unidirectional + lookahead) fase-3 model, NOT run_baseline's own
# DEFAULT_CHECKPOINT (that's the fase-1/2 bidirectional model, which needs the whole
# clip non-causally -- using it here would silently defeat the entire point of fase 3).
DEFAULT_CHECKPOINT = Path("D:/FretcastData/artifacts/streaming_v1/best_model.pth")
DEFAULT_RUN_CONFIG = Path("D:/FretcastData/artifacts/streaming_v1/run_configuration.json")


def find_input_device(name_substring: str = "Focusrite") -> int:
    """Find a WASAPI input device by (partial) name. Raises if none found."""
    wasapi_apis = [i for i, api in enumerate(sd.query_hostapis()) if api["name"] == "Windows WASAPI"]
    if not wasapi_apis:
        raise RuntimeError("no WASAPI host API found on this system")
    wasapi_index = wasapi_apis[0]

    for i, dev in enumerate(sd.query_devices()):
        if (
            dev["hostapi"] == wasapi_index
            and dev["max_input_channels"] > 0
            and name_substring.lower() in dev["name"].lower()
        ):
            return i
    raise RuntimeError(f"no WASAPI input device matching {name_substring!r} found -- run with --list-devices")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device-name", default="Focusrite", help="substring to match the WASAPI input device")
    ap.add_argument("--duration", type=float, default=None, help="seconds to run; omit to run until Ctrl+C")
    ap.add_argument("--block-size", type=int, default=1024, help="native-rate samples per callback (~23ms at 44.1kHz)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    device_index = find_input_device(args.device_name)
    device_info = sd.query_devices(device_index)
    print(f"using device {device_index}: {device_info['name']} (native {NATIVE_SAMPLE_RATE} Hz, WASAPI exclusive)")

    model_handle = load_model(args.checkpoint, args.run_config)
    tx_config = model_handle[2]
    if int(NATIVE_SAMPLE_RATE) % tx_config.SAMPLE_RATE != 0:
        raise RuntimeError(
            f"native rate {NATIVE_SAMPLE_RATE} isn't an integer multiple of the model's "
            f"{tx_config.SAMPLE_RATE} Hz -- the simple resample_poly(down=N) path below assumes it is"
        )
    downsample_factor = NATIVE_SAMPLE_RATE // tx_config.SAMPLE_RATE

    transcriber = StreamingTranscriber(model_handle)
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        mono = indata[:, 0].astype(np.float32)
        resampled = resample_poly(mono, up=1, down=downsample_factor).astype(np.float32)
        events = transcriber.push_audio(resampled)
        for e in events:
            elapsed = time.time() - start_time
            print(f"[{elapsed:6.2f}s]  {e}")

    wasapi_settings = sd.WasapiSettings(exclusive=True)
    try:
        with sd.InputStream(
            device=device_index,
            samplerate=NATIVE_SAMPLE_RATE,
            channels=1,
            blocksize=args.block_size,
            dtype="float32",
            extra_settings=wasapi_settings,
            callback=callback,
        ):
            print("listening" + (f" for {args.duration}s" if args.duration else " (Ctrl+C to stop)") + "...")
            if args.duration:
                sd.sleep(int(args.duration * 1000))
            else:
                while True:
                    sd.sleep(200)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        for e in transcriber.finalize():
            print(f"[final]  {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
