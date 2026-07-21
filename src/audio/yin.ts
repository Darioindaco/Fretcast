/**
 * YIN pitch detection algorithm — spec Section 6
 *
 * CALIBRATION MODE ONLY. Deactivate completely when recording starts.
 * Superior for single note detection.
 * Window: 2048 samples at 22050Hz (~93ms), resolution ~10.7Hz
 *
 * Reference: de Cheveigné & Kawahara (2002)
 */

import { SAMPLE_RATE, YIN_WINDOW } from './AudioCapture';

const YIN_THRESHOLD = 0.15;

/**
 * Computes the difference function d'(τ) for the YIN algorithm.
 * Uses the cumulative mean normalised difference to avoid octave errors.
 */
function cumulativeMeanNormalisedDifference(buffer: Float32Array): Float32Array {
  const W = buffer.length >> 1; // half window
  const yinBuffer = new Float32Array(W);

  // Step 1 & 2: difference function
  let runningSum = 0;
  yinBuffer[0] = 1;

  for (let tau = 1; tau < W; tau++) {
    let s = 0;
    for (let j = 0; j < W; j++) {
      const delta = buffer[j] - buffer[j + tau];
      s += delta * delta;
    }
    runningSum += s;
    // Step 3: cumulative mean normalised difference
    yinBuffer[tau] = s === 0 ? 0 : (s * tau) / runningSum;
  }

  return yinBuffer;
}

/**
 * Parabolic interpolation for sub-sample precision.
 */
function parabolicInterpolation(yinBuffer: Float32Array, tauEstimate: number): number {
  const betterTau =
    tauEstimate !== yinBuffer.length - 1 && tauEstimate !== 0
      ? tauEstimate +
        (yinBuffer[tauEstimate + 1] - yinBuffer[tauEstimate - 1]) /
          (2 * (2 * yinBuffer[tauEstimate] - yinBuffer[tauEstimate - 1] - yinBuffer[tauEstimate + 1]))
      : tauEstimate;
  return betterTau;
}

export interface YINResult {
  frequency: number;   // Hz, -1 if no pitch detected
  confidence: number;  // 0..1 (1 - probability)
}

/**
 * Run YIN on a buffer.
 * Buffer should be YIN_WINDOW (2048) samples.
 */
export function runYIN(buffer: Float32Array): YINResult {
  if (buffer.length < YIN_WINDOW) {
    return { frequency: -1, confidence: 0 };
  }

  const yinBuffer = cumulativeMeanNormalisedDifference(buffer);
  const W = yinBuffer.length;

  // Step 4: absolute threshold — find first tau below threshold
  let tauEstimate = -1;
  for (let tau = 2; tau < W; tau++) {
    if (yinBuffer[tau] < YIN_THRESHOLD) {
      // Continue searching for a deeper minimum
      while (tau + 1 < W && yinBuffer[tau + 1] < yinBuffer[tau]) {
        tau++;
      }
      tauEstimate = tau;
      break;
    }
  }

  if (tauEstimate === -1) {
    // No pitch — return minimum point as fallback
    let minVal = Infinity;
    for (let i = 2; i < W; i++) {
      if (yinBuffer[i] < minVal) { minVal = yinBuffer[i]; tauEstimate = i; }
    }
  }

  // Step 5: parabolic interpolation
  const betterTau = parabolicInterpolation(yinBuffer, tauEstimate);

  // Step 6: best local estimate (already done by threshold step)
  const frequency = betterTau > 0 ? SAMPLE_RATE / betterTau : -1;
  const confidence = tauEstimate >= 0 ? 1 - yinBuffer[tauEstimate] : 0;

  return { frequency, confidence };
}

/**
 * Convert frequency to nearest note name.
 */
export function freqToNoteName(freq: number): string {
  if (freq <= 0) return '—';
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const midiNote = Math.round(12 * Math.log2(freq / 440) + 69);
  return noteNames[((midiNote % 12) + 12) % 12];
}
