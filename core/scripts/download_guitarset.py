"""Download the GuitarSet subset the trimplexx pipeline actually uses.

process_single_track() (external/trimplexx/python/data_processing/preparation.py)
reads track.audio_mix_path (mono-pickup mix) with a fallback to audio_mic_path.
It never touches the hexaphonic pickup audio, so we skip audio_hex_debleeded and
audio_hex_original -- that cuts the download from 8.2 GB to ~1.4 GB.

Everything lands under D:\FretcastData\mir_datasets_storage (kept off the small C:
drive; see core/README.md for why).
"""
from __future__ import annotations

import mirdata

DATA_HOME = "D:/FretcastData/mir_datasets_storage"


def main() -> None:
    ds = mirdata.initialize("guitarset", data_home=DATA_HOME)
    # "index" must be listed explicitly: passing partial_download replaces the
    # dataset's default list rather than extending it, and the index (track id ->
    # file path manifest) is itself a downloadable remote, separate from the audio.
    ds.download(partial_download=["index", "annotations", "audio_mic", "audio_mix"])
    print("validating...")
    missing = ds.validate()
    print("done.", "issues:" if missing else "no issues found.", missing or "")


if __name__ == "__main__":
    main()
