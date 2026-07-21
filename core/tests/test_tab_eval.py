"""Self-tests for the fretboard math and the eval harness. Pure stdlib, no pytest.

Run: python core/tests/test_tab_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fretcast.eval import evaluate  # noqa: E402
from fretcast.labels import load_exercise  # noqa: E402
from fretcast.tab import TabEvent, midi_to_name, pitch_to_positions, position_to_pitch  # noqa: E402

EXERCISE_DIR = Path(__file__).resolve().parents[1] / "exercises"
_checks = 0


def check(cond: bool, msg: str) -> None:
    global _checks
    _checks += 1
    if not cond:
        raise AssertionError(msg)


def test_position_pitch_roundtrip() -> None:
    check(position_to_pitch(1, 0) == 64, "open high E should be MIDI 64 (E4)")
    check(position_to_pitch(6, 0) == 40, "open low E should be MIDI 40 (E2)")
    check(position_to_pitch(6, 12) == 52, "low E fret 12 should be E3 (52)")
    check(midi_to_name(64) == "E4", "MIDI 64 should name as E4")
    check(midi_to_name(40) == "E2", "MIDI 40 should name as E2")


def test_candidate_positions() -> None:
    # E4 is the classic ambiguous pitch: open high E, or several fretted spots.
    pos = pitch_to_positions(64)
    check((1, 0) in pos, "E4 must include open high E (s1 f0)")
    check((2, 5) in pos, "E4 must include s2 f5")
    check((3, 9) in pos, "E4 must include s3 f9")
    check(len(pos) >= 3, "E4 should have >= 3 candidate positions")
    # A very low note is unambiguous.
    check(pitch_to_positions(40) == [(6, 0)], "E2 has exactly one position")


def test_devil_pairs_are_actually_ambiguous() -> None:
    # Every pair in ex3 must be two positions of the SAME pitch, else the test set lies.
    events, _ = load_exercise(EXERCISE_DIR / "ex3_devil_pairs.csv")
    check(len(events) == 12, "devil pairs should have 12 notes (6 pairs)")
    for i in range(0, len(events), 2):
        a, b = events[i], events[i + 1]
        check(a.pitch == b.pitch, f"pair {i//2+1} not same pitch: {a} vs {b}")
        check((a.string, a.fret) != (b.string, b.fret), f"pair {i//2+1} identical position")


def test_eval_perfect_and_ambiguity() -> None:
    ref, bpm = load_exercise(EXERCISE_DIR / "ex3_devil_pairs.csv")
    check(bpm == 80, "ex3 bpm should be 80")

    # Perfect prediction -> everything 100%.
    perfect = evaluate(ref, list(ref))
    check(perfect.onset_f1 == 1.0, "perfect onset F1")
    check(perfect.position_acc == 1.0, "perfect position accuracy")
    check(perfect.disambiguation_rate == 1.0, "perfect disambiguation")

    # Pitch-perfect but flips every note to the OTHER position in its pair:
    # pitch stays 100%, position collapses, disambiguation -> 0%, all flagged.
    flipped: list[TabEvent] = []
    for i in range(0, len(ref), 2):
        a, b = ref[i], ref[i + 1]
        flipped.append(TabEvent(time=a.time, string=b.string, fret=b.fret))
        flipped.append(TabEvent(time=b.time, string=a.string, fret=a.fret))
    res = evaluate(ref, flipped)
    check(res.pitch_acc == 1.0, f"flipped should keep pitch 100%, got {res.pitch_acc}")
    check(res.position_acc == 0.0, f"flipped should drop position to 0, got {res.position_acc}")
    check(res.disambiguation_rate == 0.0, "flipped disambiguation should be 0")
    check(len(res.ambiguity_failures) == 12, "all 12 notes should be ambiguity failures")


def test_onset_miss_and_insertion() -> None:
    ref, _ = load_exercise(EXERCISE_DIR / "ex1_chromatic.csv")
    # Drop one ref note from the prediction (a deletion) and add a spurious one.
    pred = list(ref[:-1]) + [TabEvent(time=ref[-1].time + 10.0, string=1, fret=1)]
    res = evaluate(ref, pred)
    check(res.n_matched == len(ref) - 1, "one note should go unmatched")
    check(res.onset_recall < 1.0, "recall must reflect the missed note")
    check(res.onset_precision < 1.0, "precision must reflect the inserted note")


def test_pitch_autocorr_synthetic() -> None:
    # Synthesize A3 (220 Hz) + a weaker octave harmonic; tracker should lock the fundamental.
    import math

    from fretcast.mono_baseline import pitch_autocorr

    rate = 22050
    n = 1024
    frame = [
        math.sin(2 * math.pi * 220.0 * i / rate) + 0.4 * math.sin(2 * math.pi * 440.0 * i / rate)
        for i in range(n)
    ]
    pr = pitch_autocorr(frame, rate)
    check(abs(pr.freq - 220.0) < 3.0, f"expected ~220 Hz, got {pr.freq:.1f}")
    check(pr.clarity > 0.7, f"clean tone should have high clarity, got {pr.clarity:.2f}")


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print(f"\n{_checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
