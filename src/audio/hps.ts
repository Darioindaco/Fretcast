/**
 * Harmonic Product Spectrum (HPS) — spec Section 6
 *
 * RECORDING MODE ONLY. Never use during calibration.
 * Superior for chord (polyphonic) detection.
 *
 * Window:    4096 samples at 22050Hz + 2× zero-padding → FFT size 8192
 * Resolution: ~5.4Hz at 22050Hz / 8192 bins = 2.69Hz/bin × 2 = ~5.4Hz effective
 * Hop size:  220 samples (10ms)
 * Peaks:     up to 6 (one per string)
 */

import { SAMPLE_RATE, HPS_WINDOW } from './AudioCapture';

const FFT_SIZE = HPS_WINDOW * 2; // 8192 — 2× zero-padded
const HPS_HARMONICS = 5;         // downsample by 2×, 3×, 4×, 5×

/**
 * Compute FFT magnitude spectrum using Cooley-Tukey radix-2 DIT.
 * Input length must be power of 2.
 */
function fft(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  // FFT butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let re = 1, im = 0;
      for (let k = 0; k < len >> 1; k++) {
        const uRe = real[i + k];
        const uIm = imag[i + k];
        const vRe = real[i + k + (len >> 1)] * re - imag[i + k + (len >> 1)] * im;
        const vIm = real[i + k + (len >> 1)] * im + imag[i + k + (len >> 1)] * re;
        real[i + k] = uRe + vRe;
        imag[i + k] = uIm + vIm;
        real[i + k + (len >> 1)] = uRe - vRe;
        imag[i + k + (len >> 1)] = uIm - vIm;
        const newRe = re * wRe - im * wIm;
        im = re * wIm + im * wRe;
        re = newRe;
      }
    }
  }
}

/**
 * Apply Hann window to reduce spectral leakage.
 */
function hannWindow(buffer: Float32Array): Float64Array {
  const result = new Float64Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    result[i] = buffer[i] * 0.5 * (1 - Math.cos((2 * Math.PI * i) / (buffer.length - 1)));
  }
  return result;
}

export interface HPSResult {
  /** Up to 6 detected frequencies in Hz, ascending order */
  frequencies: number[];
  /** Magnitude of each detected peak */
  magnitudes: number[];
  /**
   * 12-bin chroma vector — energy per pitch class (0=C … 11=B), normalised 0..1.
   * Sums spectral energy from ALL octaves per pitch class.
   * More robust than peak-picking for chord matching.
   */
  chroma: Float64Array;
}

/**
 * Run HPS on a buffer.
 * Buffer should be HPS_WINDOW (4096) samples.
 * Returns up to 6 peaks (one per guitar string).
 */
export function runHPS(buffer: Float32Array): HPSResult {
  if (buffer.length < HPS_WINDOW) return { frequencies: [], magnitudes: [], chroma: new Float64Array(12) };

  // Zero-pad to FFT_SIZE (8192)
  const real = new Float64Array(FFT_SIZE);
  const imag = new Float64Array(FFT_SIZE);
  const windowed = hannWindow(buffer.subarray(0, HPS_WINDOW));
  for (let i = 0; i < HPS_WINDOW; i++) real[i] = windowed[i];

  fft(real, imag);

  // Magnitude spectrum (first half only)
  const halfSize = FFT_SIZE / 2;
  const mag = new Float64Array(halfSize);
  for (let i = 0; i < halfSize; i++) {
    mag[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
  }

  // Harmonic Product Spectrum
  const hps = new Float64Array(mag);
  for (let h = 2; h <= HPS_HARMONICS; h++) {
    for (let i = 0; i < halfSize; i++) {
      const srcIdx = Math.floor(i * h);
      if (srcIdx < halfSize) {
        hps[i] *= mag[srcIdx];
      } else {
        hps[i] = 0;
      }
    }
  }

  // Frequency resolution
  const freqPerBin = SAMPLE_RATE / FFT_SIZE;

  // Guitar frequency range: E2 (82Hz) to E5 (1318Hz)
  const minBin = Math.floor(60 / freqPerBin);
  const maxBin = Math.ceil(1400 / freqPerBin);

  // Find up to 6 peaks with minimum spacing (~50Hz between strings)
  const minBinSpacing = Math.floor(40 / freqPerBin);
  const peaks: Array<{ bin: number; mag: number }> = [];

  // Simple peak picking: local maxima above noise floor
  const noiseFloor = computeNoiseFloor(hps, minBin, maxBin);

  for (let i = minBin + 1; i < maxBin - 1; i++) {
    if (
      hps[i] > hps[i - 1] &&
      hps[i] > hps[i + 1] &&
      hps[i] > noiseFloor * 1.5
    ) {
      // Check minimum distance from existing peaks
      const tooClose = peaks.some((p) => Math.abs(p.bin - i) < minBinSpacing);
      if (!tooClose) {
        peaks.push({ bin: i, mag: hps[i] });
      }
    }
  }

  // Sort by magnitude descending, take top 6
  peaks.sort((a, b) => b.mag - a.mag);
  const top6 = peaks.slice(0, 6);

  // Sub-bin interpolation + convert to Hz
  const result: Array<{ freq: number; mag: number }> = top6.map(({ bin, mag }) => {
    // Parabolic interpolation
    let interpolatedBin = bin;
    if (bin > 0 && bin < halfSize - 1) {
      const alpha = hps[bin - 1];
      const beta = hps[bin];
      const gamma = hps[bin + 1];
      const denom = alpha - 2 * beta + gamma;
      if (denom !== 0) {
        interpolatedBin = bin + 0.5 * (alpha - gamma) / denom;
      }
    }
    return { freq: interpolatedBin * freqPerBin, mag };
  });

  // Sort by frequency ascending
  result.sort((a, b) => a.freq - b.freq);

  // Chroma from the raw magnitude spectrum (not HPS — captures all harmonics)
  const chromaFreqPerBin = SAMPLE_RATE / FFT_SIZE;
  const chroma = computeChroma(mag, chromaFreqPerBin);

  return {
    frequencies: result.map((r) => r.freq),
    magnitudes: result.map((r) => r.mag),
    chroma,
  };
}

/**
 * Compute 12-bin chroma vector from FFT magnitude spectrum.
 * Sums energy per pitch class across all octaves (60Hz – 2000Hz).
 * Normalised so the maximum bin = 1.0.
 */
function computeChroma(mag: Float64Array, freqPerBin: number): Float64Array {
  const chroma = new Float64Array(12);
  for (let i = 1; i < mag.length; i++) {
    const freq = i * freqPerBin;
    if (freq < 60 || freq > 2000) continue;
    const midiFloat = 12 * Math.log2(freq / 440) + 69;
    const pc = ((Math.round(midiFloat) % 12) + 12) % 12;
    chroma[pc] += mag[i] * mag[i]; // accumulate energy
  }
  // Normalise to 0..1
  const max = Math.max(...chroma);
  if (max > 0) for (let i = 0; i < 12; i++) chroma[i] /= max;
  return chroma;
}

function computeNoiseFloor(hps: Float64Array, minBin: number, maxBin: number): number {
  let sum = 0;
  let count = 0;
  for (let i = minBin; i < maxBin; i++) {
    sum += hps[i];
    count++;
  }
  return count > 0 ? sum / count : 1;
}
