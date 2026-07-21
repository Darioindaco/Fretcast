/**
 * HFC Onset Detector — spec Section 6
 *
 * High-Frequency Content onset detector.
 * Window: 512 samples, hop: 64 samples
 * Threshold: mean + 1.5 × stddev (adaptive)
 * Minimum inter-onset interval: 150ms
 */

import { SAMPLE_RATE } from './AudioCapture';

const THRESHOLD_MULTIPLIER = 1.5;
const MIN_IOI_SAMPLES = Math.round(0.15 * SAMPLE_RATE); // 150ms in samples

export interface OnsetEvent {
  timestamp: number;   // ms from session start
  hfc: number;         // HFC value at onset
  sampleIndex: number; // absolute sample index
}

export class OnsetDetector {
  private hfcHistory: number[] = [];
  private historyWindow = 43; // ~1s of HFC history for adaptive threshold
  private lastOnsetSample = -MIN_IOI_SAMPLES;
  private sampleCount = 0;
  private readonly onOnset: (event: OnsetEvent) => void;

  constructor(onOnset: (event: OnsetEvent) => void) {
    this.onOnset = onOnset;
  }

  reset(): void {
    this.hfcHistory = [];
    this.lastOnsetSample = -MIN_IOI_SAMPLES;
    this.sampleCount = 0;
    // reset timestamp
  }

  /**
   * Process a hop of audio samples.
   * hop should be HFC_HOP (64) samples.
   */
  processHop(hop: Float32Array): void {
    // Compute HFC for this hop using full HFC_WINDOW
    // (caller should pass windowed frames; hop is the new samples)
    const hfc = this.computeHFC(hop);
    this.hfcHistory.push(hfc);
    if (this.hfcHistory.length > this.historyWindow) {
      this.hfcHistory.shift();
    }

    // Adaptive threshold
    const { mean, std } = this.stats(this.hfcHistory);
    const threshold = mean + THRESHOLD_MULTIPLIER * std;

    // Onset detection: HFC exceeds threshold + minimum IOI
    const currentSample = this.sampleCount;
    const ioiSamples = currentSample - this.lastOnsetSample;

    if (
      hfc > threshold &&
      ioiSamples >= MIN_IOI_SAMPLES &&
      this.hfcHistory.length >= 4 // need some history for valid threshold
    ) {
      this.lastOnsetSample = currentSample;
      const timestamp = (currentSample / SAMPLE_RATE) * 1000; // ms from start
      this.onOnset({ timestamp, hfc, sampleIndex: currentSample });
    }

    this.sampleCount += hop.length;
  }

  private computeHFC(buffer: Float32Array): number {
    // High-Frequency Content: sum of k * |X(k)|^2
    // Approximated in time domain as emphasised energy in high-frequency region
    let hfc = 0;
    for (let i = 0; i < buffer.length; i++) {
      // Weight by position (emphasises later samples in window = higher freq)
      hfc += (i + 1) * buffer[i] * buffer[i];
    }
    return hfc;
  }

  private stats(arr: number[]): { mean: number; std: number } {
    if (arr.length === 0) return { mean: 0, std: 0 };
    const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    return { mean, std: Math.sqrt(variance) };
  }
}
