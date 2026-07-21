"""Load / save ground-truth tab labels for the scripted Phase 0 exercises.

CSV format (comment header carries the tempo so beats convert to seconds):

    # fretcast exercise ground truth
    # bpm: 80
    # string numbering: 1=high E (thinnest) .. 6=low E (thickest)
    beat,string,fret
    0,6,1
    1,6,2

`beat` is quarter-note beats from the count-in's downbeat. With the take recorded
to the same click, onset time in seconds is simply beat * 60 / bpm — no manual
annotation of audio required, which is the whole point of scripting the exercises.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .tab import TabEvent


def load_exercise(path: str | Path) -> tuple[list[TabEvent], int]:
    """Read an exercise CSV. Returns (events sorted by time, bpm)."""
    path = Path(path)
    bpm: int | None = None
    rows: list[tuple[float, int, int]] = []

    with path.open(newline="", encoding="utf-8") as fh:
        data_lines: list[str] = []
        for raw in fh:
            line = raw.strip()
            if line.startswith("#"):
                key, _, val = line[1:].partition(":")
                if key.strip().lower() == "bpm":
                    bpm = int(val.strip())
                continue
            if line:
                data_lines.append(raw)

        if bpm is None:
            raise ValueError(f"{path}: missing '# bpm:' header")

        reader = csv.DictReader(data_lines)
        for r in reader:
            beat = float(r["beat"])
            rows.append((beat, int(r["string"]), int(r["fret"])))

    sec_per_beat = 60.0 / bpm
    events = [TabEvent(time=beat * sec_per_beat, string=s, fret=f) for beat, s, f in rows]
    events.sort(key=lambda e: e.time)
    return events, bpm


def save_predictions(path: str | Path, events: list[TabEvent]) -> None:
    """Write model predictions in the same schema (seconds, not beats)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time_sec", "string", "fret"])
        for e in sorted(events, key=lambda e: e.time):
            w.writerow([f"{e.time:.4f}", e.string, e.fret])


def load_predictions(path: str | Path) -> list[TabEvent]:
    """Read predictions written as time_sec,string,fret."""
    path = Path(path)
    events: list[TabEvent] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            events.append(
                TabEvent(time=float(r["time_sec"]), string=int(r["string"]), fret=int(r["fret"]))
            )
    events.sort(key=lambda e: e.time)
    return events
