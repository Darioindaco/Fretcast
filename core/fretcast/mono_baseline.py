"""Monophonic baseline transcriber — dependency-free, pure stdlib.

This is NOT the V02 model. It is the cheap, always-available front end the plan calls
for (section 2, steps 1-2): energy onset detection + per-onset autocorrelation pitch,
then each pitch mapped to its candidate (string, fret) positions via tab.py.

Purpose:
  - sanity-check a real recording before any model exists
  - serve as the pitch-verification layer that vetoes a CRNN's fret guess whose implied
    pitch disagrees with what was actually sounding
  - expose the disambiguation problem directly: every note prints its candidate positions

Honest limits: it is MONOPHONIC. On a chord it locks onto the strongest periodicity and
reports one note. It reads float32 and PCM16 WAV without numpy.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .tab import TabEvent, midi_to_name, pitch_to_positions

GUITAR_FMIN = 70.0   # a touch below low E (82 Hz), headroom for detune
GUITAR_FMAX = 700.0  # comfortably covers fingerpicked lead range


# ── WAV I/O (float32 + PCM16, no numpy) ──────────────────────────────────────

def read_wav_mono(path: str | Path) -> tuple[list[float], int, dict]:
    """Read a WAV to a mono float list. Returns (samples, samplerate, info).

    info carries channel count and, for stereo, the L/R correlation — a quick tell
    for whether this is a dry mono-summed DI or has stereo processing on it.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")

    tag = ch = rate = bits = 0
    data: bytes = b""
    i = 12
    while i + 8 <= len(raw):
        cid = raw[i : i + 4]
        sz = struct.unpack("<I", raw[i + 4 : i + 8])[0]
        body = raw[i + 8 : i + 8 + sz]
        if cid == b"fmt ":
            tag, ch, rate, _br, _al, bits = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        i += 8 + sz + (sz & 1)

    n_bytes_per = bits // 8
    n_frames = len(data) // (n_bytes_per * ch)

    if tag == 3 and bits == 32:  # IEEE float
        flat = struct.unpack(f"<{n_frames * ch}f", data[: n_frames * ch * 4])
    elif tag == 1 and bits == 16:  # PCM16
        ints = struct.unpack(f"<{n_frames * ch}h", data[: n_frames * ch * 2])
        flat = tuple(v / 32768.0 for v in ints)
    else:
        raise ValueError(f"unsupported WAV: tag={tag} bits={bits}")

    info: dict = {"channels": ch, "samplerate": rate, "bits": bits, "float": tag == 3}

    if ch == 1:
        mono = list(flat)
    else:
        left = flat[0::ch]
        right = flat[1::ch]
        info["lr_correlation"] = _correlation(left, right)
        mono = [(l + r) * 0.5 for l, r in zip(left, right)]

    return mono, rate, info


def _correlation(a, b) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    ma = sum(a[:n]) / n
    mb = sum(b[:n]) / n
    num = da = db = 0.0
    step = max(1, n // 20000)  # subsample; correlation is a coarse tell
    cnt = 0
    for i in range(0, n, step):
        x = a[i] - ma
        y = b[i] - mb
        num += x * y
        da += x * x
        db += y * y
        cnt += 1
    return num / (da**0.5 * db**0.5) if da > 0 and db > 0 else 0.0


def downsample(x: list[float], factor: int) -> list[float]:
    """Averaging decimation by an integer factor (crude anti-alias, fine for pitch)."""
    if factor <= 1:
        return x
    out = []
    for i in range(0, len(x) - factor + 1, factor):
        out.append(sum(x[i : i + factor]) / factor)
    return out


# ── FFT (radix-2, pure Python) ────────────────────────────────────────────────

def _fft(re: list[float], im: list[float]) -> None:
    """In-place iterative radix-2 Cooley-Tukey FFT. len must be a power of 2."""
    import math

    n = len(re)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    length = 2
    while length <= n:
        ang = -2 * math.pi / length
        wr, wi = math.cos(ang), math.sin(ang)
        half = length >> 1
        for i in range(0, n, length):
            cr, ci = 1.0, 0.0
            for k in range(half):
                a = i + k
                b = a + half
                tr = re[b] * cr - im[b] * ci
                ti = re[b] * ci + im[b] * cr
                re[b] = re[a] - tr
                im[b] = im[a] - ti
                re[a] += tr
                im[a] += ti
                cr, ci = cr * wr - ci * wi, cr * wi + ci * wr
        length <<= 1


# ── Onset detection (spectral flux — catches plucks over sustain) ─────────────

def detect_onsets(
    x: list[float],
    rate: int,
    win: int = 1024,
    hop: int = 256,
    min_ioi_s: float = 0.07,
    sensitivity: float = 1.5,
) -> list[int]:
    """Onset sample indices via spectral flux (plan section 2, step 1).

    Flux = sum of positive frame-to-frame magnitude increases. Unlike energy, this
    fires on a new note's attack even while previous notes still ring — the normal
    case in fingerpicking. Adaptive peak-picking over a local window follows.
    """
    import math

    hann = [0.5 * (1 - math.cos(2 * math.pi * i / (win - 1))) for i in range(win)]
    prev_mag: list[float] | None = None
    flux: list[float] = []
    for start in range(0, len(x) - win, hop):
        re = [x[start + i] * hann[i] for i in range(win)]
        im = [0.0] * win
        _fft(re, im)
        half = win // 2
        mag = [math.hypot(re[b], im[b]) for b in range(half)]
        if prev_mag is None:
            flux.append(0.0)
        else:
            flux.append(sum(m - p for m, p in zip(mag, prev_mag) if m > p))
        prev_mag = mag

    # adaptive peak-picking: local maximum above a moving mean + sensitivity*std
    onsets: list[int] = []
    min_gap = max(1, int(min_ioi_s * rate / hop))
    last = -min_gap
    w = 20  # +/- frames for the local statistics window
    for k in range(1, len(flux) - 1):
        lo, hi = max(0, k - w), min(len(flux), k + w)
        local = flux[lo:hi]
        mean = sum(local) / len(local)
        std = (sum((v - mean) ** 2 for v in local) / len(local)) ** 0.5
        thr = mean + sensitivity * std
        is_peak = flux[k] > flux[k - 1] and flux[k] >= flux[k + 1]
        if is_peak and flux[k] > thr and (k - last) >= min_gap:
            onsets.append(k * hop)
            last = k
    return onsets


# ── Pitch via normalized autocorrelation ──────────────────────────────────────

@dataclass
class PitchResult:
    freq: float
    clarity: float  # normalized autocorr peak, 0..1


def pitch_autocorr(frame: list[float], rate: int) -> PitchResult:
    """Estimate f0 in [GUITAR_FMIN, GUITAR_FMAX] by peak of normalized autocorrelation."""
    n = len(frame)
    lag_min = max(2, int(rate / GUITAR_FMAX))
    lag_max = min(n - 1, int(rate / GUITAR_FMIN))

    r0 = sum(v * v for v in frame)
    if r0 < 1e-9:
        return PitchResult(-1.0, 0.0)

    best_lag, best_val = -1, 0.0
    prev = 0.0
    rising = False
    for lag in range(lag_min, lag_max):
        s = 0.0
        for j in range(0, n - lag):
            s += frame[j] * frame[j + lag]
        norm = s / r0
        # take the first strong local maximum, not the global one (avoids octave-down)
        if norm > prev:
            rising = True
        elif rising and norm < prev and prev > 0.5:
            best_lag, best_val = lag - 1, prev
            break
        if norm > best_val:
            best_lag, best_val = lag, norm
        prev = norm

    if best_lag < 1:
        return PitchResult(-1.0, 0.0)

    # parabolic interpolation around best_lag for sub-sample precision
    def ac(l: int) -> float:
        return sum(frame[j] * frame[j + l] for j in range(0, n - l)) / r0

    if lag_min < best_lag < lag_max - 1:
        a, b, c = ac(best_lag - 1), best_val, ac(best_lag + 1)
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        best_lag += shift

    return PitchResult(rate / best_lag, best_val)


def freq_to_midi(freq: float) -> int:
    import math

    return round(69 + 12 * math.log2(freq / 440.0))


# ── Top-level ─────────────────────────────────────────────────────────────────

@dataclass
class Note:
    event: TabEvent
    freq: float
    clarity: float
    candidates: list[tuple[int, int]]


def transcribe_mono(
    path: str | Path, clarity_min: float = 0.6, preemph: float = 0.0
) -> tuple[list[Note], dict]:
    """Full monophonic pass. Returns (notes, file_info).

    preemph (0..~1): first-difference high-pass y[n] = x[n] - preemph*x[n-1] applied to
    the onset window before pitch tracking. It suppresses ringing low strings, so on
    two-voice fingerpicking the plain pass tends to report the BASS voice and a preemph
    pass (e.g. 0.97) tends to surface the MELODY voice. Neither captures both at once —
    that is the polyphony the V02 CRNN exists to solve.
    """
    import math

    mono, rate, info = read_wav_mono(path)

    # work at ~22 kHz: enough for guitar, ~2x faster autocorrelation
    factor = max(1, rate // 22050)
    x = downsample(mono, factor)
    wrate = rate // factor

    onsets = detect_onsets(x, wrate)
    win = 1024
    skip = int(0.03 * wrate)  # skip the pick transient before measuring pitch
    hann = [0.5 * (1 - math.cos(2 * math.pi * i / (win - 1))) for i in range(win)]

    notes: list[Note] = []
    for on in onsets:
        start = on + skip
        if start + win > len(x):
            break
        if preemph and start >= 1:
            seg = x[start - 1 : start + win]
            seg = [seg[i] - preemph * seg[i - 1] for i in range(1, win + 1)]
        else:
            seg = x[start : start + win]
        frame = [seg[i] * hann[i] for i in range(win)]
        pr = pitch_autocorr(frame, wrate)
        if pr.freq <= 0 or pr.clarity < clarity_min:
            continue
        midi = freq_to_midi(pr.freq)
        cands = pitch_to_positions(midi)
        if not cands:
            continue
        # display position = lowest fret candidate (NOT a real disambiguation choice)
        s, f = min(cands, key=lambda sf: sf[1])
        notes.append(
            Note(
                event=TabEvent(time=on / wrate, string=s, fret=f),
                freq=pr.freq,
                clarity=pr.clarity,
                candidates=cands,
            )
        )
    return notes, info
