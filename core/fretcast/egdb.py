"""Parse EGDB (Chen et al., ISMIR 2022) label MIDIs into TabEvents.

EGDB ships one MIDI per tab with 7 tracks: track 0 is tempo/key/chord markers only,
tracks 1-6 are one per string, each on its own MIDI channel. Track index N (1-6)
already matches Fretcast's own string convention (1 = high e .. 6 = low E, see
tab.py) -- confirmed empirically against track 1's note range (fret 4-8 against an
open high-e), not stated in any EGDB doc found. No string-numbering translation
needed here, unlike the trimplexx model output (see run_baseline.py).

A single constant tempo is assumed (true for every EGDB file inspected: one
set_tempo in track 0, none elsewhere) -- ticks convert to seconds via that tempo
directly rather than a general tempo-map walk.
"""
from __future__ import annotations

from pathlib import Path

import mido

from .tab import STANDARD_TUNING, TabEvent, fretcast_string_to_trimplexx

DEFAULT_MIDI_TEMPO = 500_000  # MIDI spec default (120 BPM), used only if a file omits set_tempo


def _seconds_per_tick(mid: mido.MidiFile) -> float:
    tempo = DEFAULT_MIDI_TEMPO
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break
    return tempo / 1_000_000 / mid.ticks_per_beat


def load_egdb_label(midi_path: str | Path) -> list[TabEvent]:
    """Read one EGDB label .midi file, return its notes sorted by onset time."""
    mid = mido.MidiFile(str(midi_path))
    seconds_per_tick = _seconds_per_tick(mid)

    events: list[TabEvent] = []
    for string, track in enumerate(mid.tracks[1:], start=1):
        if string not in STANDARD_TUNING:
            continue  # defensive: exactly 6 string tracks expected
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                fret = msg.note - STANDARD_TUNING[string]
                if fret < 0:
                    continue  # below the open string: annotation/tuning mismatch, skip
                events.append(TabEvent(time=tick * seconds_per_tick, string=string, fret=fret))

    events.sort(key=lambda e: e.time)
    return events


def load_egdb_notes_for_training(midi_path: str | Path) -> list[tuple[float, float, int, int, int]]:
    """Read one EGDB label .midi file into trimplexx's own training-label tuple shape.

    Returns (onset_sec, offset_sec, string_idx, fret, pitch_midi) rows, string_idx in
    trimplexx's own 0=low E..5=high e convention (matching
    data_processing.preparation.extract_annotations_from_jams's output for GuitarSet) so
    these rows can sit in the exact same processed-data format trimplexx's own
    GuitarSetTabDataset already reads. Each string track is monophonic in EGDB (one note
    active at a time), so note_on/note_off pairing per track is a simple single-slot
    state machine, not a stack.
    """
    mid = mido.MidiFile(str(midi_path))
    seconds_per_tick = _seconds_per_tick(mid)

    notes: list[tuple[float, float, int, int, int]] = []
    for string, track in enumerate(mid.tracks[1:], start=1):
        if string not in STANDARD_TUNING:
            continue
        trimplexx_string = fretcast_string_to_trimplexx(string)
        tick = 0
        active_note: int | None = None
        active_start_tick: int | None = None
        for msg in track:
            tick += msg.time
            is_note_off = msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)
            if msg.type == "note_on" and msg.velocity > 0:
                if active_note is not None:
                    # defensive: shouldn't happen on a monophonic string track, but
                    # close the still-open note at this new onset rather than drop it
                    notes.append(
                        (
                            active_start_tick * seconds_per_tick,
                            tick * seconds_per_tick,
                            trimplexx_string,
                            active_note - STANDARD_TUNING[string],
                            active_note,
                        )
                    )
                active_note = msg.note
                active_start_tick = tick
            elif is_note_off and active_note == msg.note and active_start_tick is not None:
                fret = active_note - STANDARD_TUNING[string]
                if fret >= 0:
                    notes.append(
                        (
                            active_start_tick * seconds_per_tick,
                            tick * seconds_per_tick,
                            trimplexx_string,
                            fret,
                            active_note,
                        )
                    )
                active_note = None
                active_start_tick = None

    notes.sort(key=lambda n: n[0])
    return notes
