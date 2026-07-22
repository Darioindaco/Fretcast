"""Fretboard math: pitch <-> (string, fret) for a standard-tuned 6-string guitar.

String numbering (Fretcast V02 convention):
    1 = high E (thinnest)  open MIDI 64 (E4)
    2 = B                  open MIDI 59 (B3)
    3 = G                  open MIDI 55 (G3)
    4 = D                  open MIDI 50 (D3)
    5 = A                  open MIDI 45 (A2)
    6 = low E (thickest)   open MIDI 40 (E2)

This is the deterministic core of the disambiguation candidate set (plan section 2,
step 3): a detected pitch maps to 1-4 physical (string, fret) positions, and the
ranking layer then chooses among them. Nothing here is learned — it is pure tuning
arithmetic, so it is exact and needs no data.
"""
from __future__ import annotations

from dataclasses import dataclass

# string index -> open-string MIDI note
STANDARD_TUNING: dict[int, int] = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
NUM_STRINGS = 6
DEFAULT_MAX_FRET = 24
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def position_to_pitch(string: int, fret: int, tuning: dict[int, int] = STANDARD_TUNING) -> int:
    """MIDI note produced by fretting `fret` on `string`. Open string = fret 0."""
    if string not in tuning:
        raise ValueError(f"invalid string {string!r}; expected one of {sorted(tuning)}")
    if fret < 0:
        raise ValueError(f"invalid fret {fret!r}; must be >= 0")
    return tuning[string] + fret


def pitch_to_positions(
    midi: int,
    tuning: dict[int, int] = STANDARD_TUNING,
    max_fret: int = DEFAULT_MAX_FRET,
) -> list[tuple[int, int]]:
    """All playable (string, fret) positions for a MIDI pitch, low string first.

    Returns 1-4 candidates for typical guitar pitches. The length of this list is
    exactly the disambiguation problem: len 1 => no ambiguity, len > 1 => the
    ranking layer must choose.
    """
    positions: list[tuple[int, int]] = []
    for s in sorted(tuning):  # 1 (high E) .. 6 (low E)
        fret = midi - tuning[s]
        if 0 <= fret <= max_fret:
            positions.append((s, fret))
    return positions


def fretcast_string_to_trimplexx(string: int) -> int:
    """Fretcast string number (1=high e..6=low E) -> trimplexx string index (0=low E..5=high e)."""
    return 6 - string


def trimplexx_string_to_fretcast(string_idx: int) -> int:
    """Inverse of `fretcast_string_to_trimplexx` (the mapping is its own inverse)."""
    return 6 - string_idx


def midi_to_name(midi: int) -> str:
    """MIDI number -> scientific pitch name, e.g. 64 -> 'E4'."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


@dataclass(frozen=True)
class TabEvent:
    """A single fretted note in time.

    time:   onset in seconds from the start of the take
    string: 1 (high E) .. 6 (low E)
    fret:   0 (open) .. max_fret
    """

    time: float
    string: int
    fret: int

    @property
    def pitch(self) -> int:
        return position_to_pitch(self.string, self.fret)

    @property
    def name(self) -> str:
        return midi_to_name(self.pitch)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TabEvent(t={self.time:.3f}s, s{self.string} f{self.fret}, {self.name})"
