"""Download EGDB audio_DI + audio_label files individually, one request at a time.

gdown's bulk `download_folder()` hammers Google Drive fast enough to trip its
per-file "too many people have viewed this today" throttle partway through (hit
this directly: a first attempt got 59/240 labels before every remaining file started
returning FileURLRetrievalError). Individual `gdown.download(id=...)` calls with a
short sleep between them do not trip it -- confirmed by downloading all 20 audio_DI
files for tracks 100-119 this way without a single failure.

Reads the (drive_file_id, filename) pairs from *_section.log files that were captured
from an earlier `download_folder(skip_download=True)` listing pass (full 240-entry
listing, download step never run) -- see D:/FretcastData/egdb_listing.log.

Usage:
    python core/scripts/download_egdb_files.py --kind label
    python core/scripts/download_egdb_files.py --kind audio
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import gdown

LISTING_LOG = Path("D:/FretcastData/egdb_listing.log")
OUT_DIR = Path("D:/FretcastData/egdb")

SECTION_HEADERS = {
    "label": "audio_label",
    "audio": "audio_DI",
}
SUFFIX = {"label": "midi", "audio": "wav"}


def parse_section(kind: str) -> dict[int, str]:
    header = SECTION_HEADERS[kind]
    suffix = SUFFIX[kind]
    lines = LISTING_LOG.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if header in l and l.startswith("Retrieving folder"))
    ids: dict[int, str] = {}
    pattern = re.compile(rf"^Processing file (\S+) (\d+)\.{suffix}$")
    for line in lines[start + 1 :]:
        if line.startswith("Retrieving folder"):
            break
        m = pattern.match(line)
        if m:
            ids[int(m.group(2))] = m.group(1)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["label", "audio"], required=True)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ids = parse_section(args.kind)
    subdir = OUT_DIR / ("audio_label" if args.kind == "label" else "audio_DI")
    subdir.mkdir(parents=True, exist_ok=True)
    suffix = SUFFIX[args.kind]

    todo = sorted(n for n in ids if not (subdir / f"{n}.{suffix}").exists())
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(ids)} total, {len(ids) - len(todo)} already on disk, {len(todo)} to fetch")

    ok = fail = 0
    for n in todo:
        out = subdir / f"{n}.{suffix}"
        try:
            gdown.download(id=ids[n], output=str(out), quiet=True)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"FAIL {n}: {e}")
        time.sleep(args.sleep)
    print(f"done: {ok} ok, {fail} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
