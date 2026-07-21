"""Evaluation harness — the measurement backbone for Fretcast V02 fases 1-4.

Compares a model's predicted tab against scripted ground truth and reports the
metrics that actually decide the project:

  onset F1            did we detect the note at all (insertions / deletions)
  pitch accuracy      right note, regardless of where it was played  (the "easy" part)
  string accuracy     right string on matched notes
  fret accuracy       right fret on matched notes
  position accuracy   right string AND fret  (this is the real tab metric)
  disambiguation rate of the correct-PITCH notes, how many got the right POSITION
                      -> this is the number the disambiguation layer (plan 2) must move

Notes are aligned to ground truth greedily by onset time within a tolerance window.
Pure stdlib so it runs on a bare Python install, before any model is trained.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tab import TabEvent

DEFAULT_ONSET_TOL = 0.05  # 50 ms — generous enough for click-aligned takes


@dataclass
class EvalResult:
    n_ref: int
    n_pred: int
    n_matched: int
    onset_precision: float
    onset_recall: float
    onset_f1: float
    pitch_acc: float
    string_acc: float
    fret_acc: float
    position_acc: float
    disambiguation_rate: float
    # matched pairs where the pitch was right but the position was wrong: the
    # notes a vision tiebreaker (plan section 3) would exist to rescue.
    ambiguity_failures: list[tuple[TabEvent, TabEvent]] = field(default_factory=list)

    def summary(self) -> str:
        pct = lambda x: f"{100 * x:5.1f}%"
        return (
            f"onset      P {pct(self.onset_precision)}  R {pct(self.onset_recall)}  "
            f"F1 {pct(self.onset_f1)}   ({self.n_matched}/{self.n_ref} ref, {self.n_pred} pred)\n"
            f"pitch      {pct(self.pitch_acc)}\n"
            f"string     {pct(self.string_acc)}\n"
            f"fret       {pct(self.fret_acc)}\n"
            f"position   {pct(self.position_acc)}   (string AND fret)\n"
            f"disambig   {pct(self.disambiguation_rate)}   (right position | right pitch)\n"
            f"ambiguity failures: {len(self.ambiguity_failures)}"
        )


def _match(ref: list[TabEvent], pred: list[TabEvent], tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one onset matching: each ref to the nearest free pred within tol."""
    ref_order = sorted(range(len(ref)), key=lambda i: ref[i].time)
    used_pred: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for ri in ref_order:
        best_pi, best_dt = -1, tol
        for pi, p in enumerate(pred):
            if pi in used_pred:
                continue
            dt = abs(p.time - ref[ri].time)
            if dt <= best_dt:
                best_dt, best_pi = dt, pi
        if best_pi >= 0:
            used_pred.add(best_pi)
            pairs.append((ri, best_pi))
    return pairs


def evaluate(
    ref: list[TabEvent],
    pred: list[TabEvent],
    onset_tol: float = DEFAULT_ONSET_TOL,
) -> EvalResult:
    """Score predicted tab `pred` against ground truth `ref`."""
    pairs = _match(ref, pred, onset_tol)
    n_matched = len(pairs)

    onset_precision = n_matched / len(pred) if pred else 0.0
    onset_recall = n_matched / len(ref) if ref else 0.0
    onset_f1 = (
        2 * onset_precision * onset_recall / (onset_precision + onset_recall)
        if (onset_precision + onset_recall) > 0
        else 0.0
    )

    pitch_ok = string_ok = fret_ok = position_ok = 0
    pitch_right_total = 0
    position_right_given_pitch = 0
    ambiguity_failures: list[tuple[TabEvent, TabEvent]] = []

    for ri, pi in pairs:
        r, p = ref[ri], pred[pi]
        same_pitch = r.pitch == p.pitch
        same_string = r.string == p.string
        same_fret = r.fret == p.fret
        same_position = same_string and same_fret

        pitch_ok += same_pitch
        string_ok += same_string
        fret_ok += same_fret
        position_ok += same_position

        if same_pitch:
            pitch_right_total += 1
            if same_position:
                position_right_given_pitch += 1
            else:
                # correct note, wrong string/fret: the disambiguation miss
                ambiguity_failures.append((r, p))

    n = n_matched or 1  # guard div-by-zero; rates are 0 when nothing matched
    return EvalResult(
        n_ref=len(ref),
        n_pred=len(pred),
        n_matched=n_matched,
        onset_precision=onset_precision,
        onset_recall=onset_recall,
        onset_f1=onset_f1,
        pitch_acc=pitch_ok / n if n_matched else 0.0,
        string_acc=string_ok / n if n_matched else 0.0,
        fret_acc=fret_ok / n if n_matched else 0.0,
        position_acc=position_ok / n if n_matched else 0.0,
        disambiguation_rate=(
            position_right_given_pitch / pitch_right_total if pitch_right_total else 0.0
        ),
        ambiguity_failures=ambiguity_failures,
    )
