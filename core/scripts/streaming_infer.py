"""Fase 3: hardware-agnostic streaming inference core.

`StreamingTranscriber` simulates what a live ASIO callback will eventually feed it --
small blocks of raw audio arriving one at a time -- without needing real audio hardware
to validate the logic. Feed it blocks via `push_audio()` (see `simulate_live_wav` below
for a WAV-file stand-in).

First version of this file recomputed trimplexx's own `note_conversion_utils.
frames_to_notes_for_eval` on the whole rolling buffer every push. That decoder force-
closes any still-sustaining note at the last frame of whatever buffer it's given --
correct for a fixed offline clip, wrong for a growing/rolling buffer, where "the last
frame" is just wherever this particular push happened to stop, not a real note
boundary. Every push before a sustained note actually ended would re-discover it as a
"new" note starting right after the previous push's artificial cutoff -- confirmed
empirically: near-duplicate (string, fret) pairs a consistent 70-160ms apart, matching
old buffer-edge positions, not random jitter (a straightforward time-based dedup /
trailing-margin filter doesn't fix this, because the confusion originates from a PAST
push's forced closure, not the current push's edge).

This version instead keeps `PerStringSegmenter` state ACROSS pushes: a note is only
considered finished when a genuine re-onset, fret change, or silence classification
says so, never because a particular push's buffer happened to end there. Frames too
close to the current buffer's trailing edge (not yet past the model's own lookahead)
are simply left unprocessed until a later push has enough context for them -- no
forced decisions, no artifacts.

What this deliberately does NOT do yet: real ASIO/WASAPI capture (a separate small
piece once this core is validated -- this machine's sounddevice/PortAudio build only
exposes MME/DirectSound/WASAPI host APIs, no ASIO backend at all; the plan's own
"ASIO-input" assumption needs revisiting, WASAPI exclusive mode is the realistic
near-term substitute). It also recomputes the full CQT over the whole rolling raw-audio
buffer on every push rather than truly incrementally -- correct but computationally
wasteful; fine for validating correctness, not yet the final low-latency implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_baseline import TRIMPLEXX_PY  # noqa: E402
from fretcast.tab import TabEvent, trimplexx_string_to_fretcast  # noqa: E402

sys.path.insert(0, str(TRIMPLEXX_PY))
import librosa  # noqa: E402


class PerStringSegmenter:
    """Incremental version of note_conversion_utils.frames_to_notes_for_eval's inner
    per-string state machine. State (the currently-active note, if any) persists across
    `feed_frame` calls -- the whole point: a note is only closed by a genuine re-onset,
    fret change, or silence classification, never by "this is the last frame I was
    given right now."
    """

    def __init__(self, silence_fret_idx: int, min_note_duration_frames: int, max_fret_value: int):
        self.silence_fret_idx = silence_fret_idx
        self.min_note_duration_frames = min_note_duration_frames
        self.max_fret_value = max_fret_value
        self.active_start_frame: int | None = None
        self.active_fret: int | None = None

    def feed_frame(self, frame_idx: int, is_onset: bool, fret_value: int) -> tuple[int, int, int] | None:
        """Process one new (absolute) frame. Returns (start_frame, end_frame, fret) if a
        note just concluded, else None."""
        finished = None
        if self.active_start_frame is not None:
            terminate = (
                (is_onset and frame_idx > self.active_start_frame)
                or fret_value == self.silence_fret_idx
                or fret_value != self.active_fret
            )
            if terminate:
                duration = frame_idx - self.active_start_frame
                if (
                    duration >= self.min_note_duration_frames
                    and self.active_fret != self.silence_fret_idx
                    and 0 <= self.active_fret <= self.max_fret_value
                ):
                    finished = (self.active_start_frame, frame_idx, self.active_fret)
                self.active_start_frame = None
                self.active_fret = None

        if is_onset and fret_value != self.silence_fret_idx:
            self.active_start_frame = frame_idx
            self.active_fret = fret_value

        return finished

    def finalize(self, end_frame: int) -> tuple[int, int, int] | None:
        """Force-close whatever's still active -- call this once, only when the real
        stream truly ends (not on every push)."""
        if self.active_start_frame is None:
            return None
        duration = end_frame - self.active_start_frame
        finished = None
        if (
            duration >= self.min_note_duration_frames
            and self.active_fret != self.silence_fret_idx
            and 0 <= self.active_fret <= self.max_fret_value
        ):
            finished = (self.active_start_frame, end_frame, self.active_fret)
        self.active_start_frame = None
        self.active_fret = None
        return finished


class StreamingTranscriber:
    def __init__(
        self,
        model_handle,
        filter_scale: float = 1.0,
        trailing_guard_frames: int | None = None,
        recompute_hop_frames: int = 4,
    ):
        model, device, tx_config, lookahead_frames = model_handle
        self.model_handle = model_handle
        self.model = model
        self.device = device
        self.tx_config = tx_config
        self.filter_scale = filter_scale
        self.lookahead_frames = lookahead_frames
        self.lookahead_seconds = lookahead_frames * tx_config.HOP_LENGTH / tx_config.SAMPLE_RATE

        # Frames within this many of the buffer's trailing edge lack full context (both
        # the model's own lookahead and the CQT filters' need for local future samples
        # under librosa's centered windowing) and are left unprocessed until a later
        # push has grown the buffer past them.
        self.trailing_guard_frames = (
            trailing_guard_frames if trailing_guard_frames is not None else lookahead_frames + 4
        )

        # Rolling raw-audio buffer sized to comfortably cover the lowest CQT filter's
        # analysis window (the ~420ms low-E figure from the fase-3 latency discussion)
        # plus headroom for several pushes worth of new audio.
        freqs = librosa.cqt_frequencies(
            n_bins=tx_config.N_BINS_CQT, fmin=tx_config.FMIN_CQT, bins_per_octave=tx_config.BINS_PER_OCTAVE_CQT
        )
        lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=tx_config.SAMPLE_RATE, filter_scale=filter_scale)
        low_bin_samples = int(lengths[0])
        hop = tx_config.HOP_LENGTH
        self.max_buffer_samples = ((low_bin_samples + 8 * hop) // hop + 1) * hop  # keep it hop-aligned

        # Audio can arrive in whatever tiny blocks the hardware callback delivers (e.g.
        # 512 samples / 23ms), but recomputing the full CQT + model forward pass over the
        # whole rolling buffer that often is wasteful -- >95% of that buffer's audio was
        # already transformed on the previous call. Only actually recompute once at least
        # `recompute_hop_frames` worth of genuinely new audio has arrived; audio in
        # between just accumulates in the buffer cheaply. This is what fixed the track-115
        # timeout (many minutes -> seconds) without changing any correctness behavior --
        # push_audio's return value is unaffected, it just returns [] more often between
        # recomputes.
        self.recompute_hop_samples = recompute_hop_frames * hop
        self.samples_since_recompute = 0

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.buffer_start_sample = 0  # absolute sample index (multiple of hop_length) of buffer[0]
        self.next_frame_to_process = 0  # absolute frame index not yet fed to any segmenter

        silence_fret_idx = tx_config.MAX_FRETS + tx_config.FRET_SILENCE_CLASS_OFFSET
        self.segmenters = {
            s: PerStringSegmenter(silence_fret_idx, tx_config.MIN_NOTE_DURATION_FRAMES, tx_config.MAX_FRETS)
            for s in range(tx_config.DEFAULT_NUM_STRINGS)
        }
        self._finalized = False

    def _run_model(self):
        cqt = librosa.cqt(
            y=self.audio_buffer,
            sr=self.tx_config.SAMPLE_RATE,
            hop_length=self.tx_config.HOP_LENGTH,
            fmin=self.tx_config.FMIN_CQT,
            n_bins=self.tx_config.N_BINS_CQT,
            bins_per_octave=self.tx_config.BINS_PER_OCTAVE_CQT,
            filter_scale=self.filter_scale,
        )
        log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        features = torch.tensor(log_cqt, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            onset_logits, fret_logits = self.model(features)
            onset_probs = torch.sigmoid(onset_logits)[0].cpu().numpy()
            fret_preds = torch.argmax(fret_logits, dim=-1)[0].cpu().numpy()
        return onset_probs, fret_preds

    def _event_from_finished(self, string_idx: int, finished: tuple[int, int, int]) -> TabEvent:
        start_frame, _end_frame, fret = finished
        time_sec = start_frame * self.tx_config.HOP_LENGTH / self.tx_config.SAMPLE_RATE
        return TabEvent(
            time=max(0.0, time_sec - self.lookahead_seconds),
            string=trimplexx_string_to_fretcast(string_idx),
            fret=int(fret),
        )

    def push_audio(self, block: np.ndarray) -> list[TabEvent]:
        """Feed one new block of mono float32 audio at tx_config.SAMPLE_RATE.
        Returns newly-confirmed TabEvents since the previous call (absolute stream time) --
        may be empty simply because not enough new audio has accumulated yet to trigger
        a recompute (see `recompute_hop_samples`), not necessarily because nothing happened.
        """
        self.audio_buffer = np.concatenate([self.audio_buffer, block.astype(np.float32)])
        self.samples_since_recompute += len(block)

        if self.samples_since_recompute < self.recompute_hop_samples:
            return []
        self.samples_since_recompute = 0
        return self._recompute_and_process()

    def _recompute_and_process(self) -> list[TabEvent]:
        hop = self.tx_config.HOP_LENGTH
        if len(self.audio_buffer) > self.max_buffer_samples:
            excess = len(self.audio_buffer) - self.max_buffer_samples
            trim = (excess // hop) * hop  # hop-aligned, keeps frame bookkeeping exact
            # never trim past a frame we haven't processed yet
            oldest_needed_sample = (self.next_frame_to_process * hop) - self.buffer_start_sample
            trim = max(0, min(trim, oldest_needed_sample))
            if trim > 0:
                self.audio_buffer = self.audio_buffer[trim:]
                self.buffer_start_sample += trim

        if len(self.audio_buffer) < hop * 4:
            return []

        onset_probs, fret_preds = self._run_model()
        num_local_frames = onset_probs.shape[0]
        buffer_start_frame = self.buffer_start_sample // hop
        last_trustable_frame = buffer_start_frame + num_local_frames - 1 - self.trailing_guard_frames

        new_events: list[TabEvent] = []
        frame = self.next_frame_to_process
        while frame <= last_trustable_frame:
            local_idx = frame - buffer_start_frame
            for string_idx in range(self.tx_config.DEFAULT_NUM_STRINGS):
                is_onset = bool(onset_probs[local_idx, string_idx] > 0.5)
                fret_value = int(fret_preds[local_idx, string_idx])
                finished = self.segmenters[string_idx].feed_frame(frame, is_onset, fret_value)
                if finished:
                    new_events.append(self._event_from_finished(string_idx, finished))
            frame += 1
        self.next_frame_to_process = frame

        return new_events

    def finalize(self) -> list[TabEvent]:
        """Call once when the stream truly ends, to flush any still-sustaining notes."""
        if self._finalized:
            return []
        self._finalized = True

        events = self._recompute_and_process()  # process whatever hadn't hit the hop threshold yet

        end_frame = self.next_frame_to_process
        for string_idx, segmenter in self.segmenters.items():
            finished = segmenter.finalize(end_frame)
            if finished:
                events.append(self._event_from_finished(string_idx, finished))
        return events


def simulate_live_wav(wav_path, model_handle, block_size: int = 512, filter_scale: float = 1.0):
    """Feed a WAV file through StreamingTranscriber in fixed-size blocks, as if it were
    arriving live from an audio callback. Returns (all_events, transcriber)."""
    tx_config = model_handle[2]
    audio, _ = librosa.load(str(wav_path), sr=tx_config.SAMPLE_RATE, mono=True)
    transcriber = StreamingTranscriber(model_handle, filter_scale=filter_scale)

    events: list[TabEvent] = []
    for start in range(0, len(audio), block_size):
        block = audio[start : start + block_size]
        events.extend(transcriber.push_audio(block))
    events.extend(transcriber.finalize())

    return sorted(events, key=lambda e: e.time), transcriber
