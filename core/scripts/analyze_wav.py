"""Monophonic baseline pass over a real WAV — no model, no numpy required.

    python core/scripts/analyze_wav.py "path/to/take.wav"

Prints file info, a dry-DI stereo check, and the detected notes with their candidate
(string, fret) positions so the disambiguation problem is visible per note.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fretcast.mono_baseline import transcribe_mono  # noqa: E402
from fretcast.tab import midi_to_name  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav")
    ap.add_argument("--clarity", type=float, default=0.6, help="min pitch clarity 0..1")
    args = ap.parse_args()

    notes, info = transcribe_mono(args.wav, clarity_min=args.clarity)

    print(f"file       : {Path(args.wav).name}")
    print(f"channels   : {info['channels']}   {info['bits']}-bit {'float' if info['float'] else 'PCM'}"
          f"   {info['samplerate']} Hz")
    if "lr_correlation" in info:
        c = info["lr_correlation"]
        verdict = "dual-mono (dry)" if c > 0.999 else "true stereo (has processing)" if c < 0.98 else "near-mono"
        print(f"L/R corr   : {c:.4f}  -> {verdict}")
    print(f"notes      : {len(notes)} detected")
    print("-" * 68)
    print(f"{'t (s)':>7}  {'note':>4}  {'clarity':>7}  {'display':>8}   candidates (string,fret)")
    for nt in notes:
        cand = "  ".join(f"s{s}f{f}" for s, f in nt.candidates)
        amb = "" if len(nt.candidates) == 1 else f"  [{len(nt.candidates)}-way]"
        print(f"{nt.event.time:7.2f}  {midi_to_name(nt.event.pitch):>4}  {nt.clarity:7.2f}  "
              f"s{nt.event.string}f{nt.event.fret:<2}    {cand}{amb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
