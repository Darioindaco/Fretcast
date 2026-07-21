/**
 * AudioEngine — spec Sections 6, 11
 *
 * Central audio pipeline. Enforces the YIN/HPS mode switch explicitly.
 *
 * RULE (non-negotiable, spec Section 6):
 *   YIN  = calibration mode ONLY
 *   HPS  = recording mode ONLY
 *   The switch is explicit here — never implicit.
 *   YIN is fully deactivated before HPS starts.
 *   HPS is fully deactivated before YIN starts.
 *   Using HPS during calibration introduces missing fundamental errors
 *   that corrupt the anchor capture.
 */

import { AudioCapture, SAMPLE_RATE, HPS_HOP, HPS_WINDOW, YIN_WINDOW } from './AudioCapture';
import { runYIN, YINResult } from './yin';
import { runHPS, HPSResult } from './hps';
import { OnsetDetector, OnsetEvent } from './onset';
import { matchChordFromChroma, ChordMatch } from './chords';
import { AudioMode } from '../types';

export interface YINCallback {
  (result: YINResult, timestamp: number): void;
}

export interface ChordDetectedCallback {
  (matches: ChordMatch[], onsetEvent: OnsetEvent, hpsResult: HPSResult): void;
}

export class AudioEngine {
  private capture: AudioCapture | null = null;
  private onsetDetector: OnsetDetector | null = null;

  // Current mode — starts IDLE
  private mode: AudioMode = 'IDLE';

  // Rolling HPS window buffer (accumulates hops into full window)
  private hpsWindowBuffer: Float32Array = new Float32Array(HPS_WINDOW);
  private hpsBufferFill = 0;

  // Rolling YIN buffer
  private yinWindowBuffer: Float32Array = new Float32Array(YIN_WINDOW);
  private yinBufferFill = 0;

  private yinCallback: YINCallback | null = null;
  private chordCallback: ChordDetectedCallback | null = null;

  /** Start audio capture from a MediaStream */
  async startFromStream(stream: MediaStream): Promise<void> {
    this.capture = new AudioCapture();
    await this.capture.start(stream, HPS_HOP, (hop) => this.onHop(hop));
  }

  /**
   * Explicit mode switch — spec Section 6.
   * Calling setMode('HPS') fully deactivates YIN first.
   * Calling setMode('YIN') fully deactivates HPS first.
   */
  setMode(mode: AudioMode): void {
    if (this.mode === mode) return;

    // Full deactivation of previous mode
    if (this.mode === 'YIN') {
      this.yinWindowBuffer = new Float32Array(YIN_WINDOW);
      this.yinBufferFill = 0;
    } else if (this.mode === 'HPS') {
      this.hpsWindowBuffer = new Float32Array(HPS_WINDOW);
      this.hpsBufferFill = 0;
      this.onsetDetector?.reset();
    }

    this.mode = mode;

    // Initialize new mode
    if (mode === 'HPS') {
      this.onsetDetector = new OnsetDetector((event) => this.onOnset(event));
    } else if (mode === 'YIN') {
      this.onsetDetector = null;
    }
  }

  onYINResult(cb: YINCallback): void { this.yinCallback = cb; }
  onChordDetected(cb: ChordDetectedCallback): void { this.chordCallback = cb; }

  stop(): void {
    this.capture?.stop();
    this.capture = null;
    this.mode = 'IDLE';
    this.onsetDetector = null;
  }

  get currentMode(): AudioMode { return this.mode; }
  get sampleRate(): number { return this.capture?.sampleRate ?? SAMPLE_RATE; }

  // ── Private: process each hop ────────────────────────────────────────────

  private onHop(hop: Float32Array): void {
    switch (this.mode) {
      case 'YIN':
        this.processYINHop(hop);
        break;
      case 'HPS':
        this.processHPSHop(hop);
        break;
      case 'IDLE':
        break;
    }
  }

  /** YIN: accumulate into YIN_WINDOW, run when full */
  private processYINHop(hop: Float32Array): void {
    for (let i = 0; i < hop.length; i++) {
      if (this.yinBufferFill < YIN_WINDOW) {
        this.yinWindowBuffer[this.yinBufferFill++] = hop[i];
      }
    }

    if (this.yinBufferFill >= YIN_WINDOW) {
      const result = runYIN(this.yinWindowBuffer);
      const timestamp = Date.now();
      this.yinCallback?.(result, timestamp);

      // Slide window: shift by hop size for overlap
      const hopSize = hop.length;
      this.yinWindowBuffer.copyWithin(0, hopSize);
      this.yinBufferFill -= hopSize;
    }
  }

  /** HPS: accumulate hops, run HPS per window + feed onset detector */
  private processHPSHop(hop: Float32Array): void {
    // Feed onset detector
    this.onsetDetector?.processHop(hop);

    // Accumulate into HPS window buffer
    for (let i = 0; i < hop.length; i++) {
      if (this.hpsBufferFill < HPS_WINDOW) {
        this.hpsWindowBuffer[this.hpsBufferFill++] = hop[i];
      }
    }

    if (this.hpsBufferFill >= HPS_WINDOW) {
      // Run HPS on current window
      // (results are used by onOnset via the ring buffer — Phase 5)
      // For now store the last HPS result for use by onset handler
      this.lastHPSResult = runHPS(this.hpsWindowBuffer);

      // Slide window by hop size
      this.hpsWindowBuffer.copyWithin(0, HPS_HOP);
      this.hpsBufferFill -= HPS_HOP;
    }
  }

  private lastHPSResult: HPSResult = { frequencies: [], magnitudes: [], chroma: new Float64Array(12) };

  private onOnset(event: OnsetEvent): void {
    // Spec Section 4 / Section 8: read chord 60ms after onset so the
    // attack transient has decayed and the sustained chord is stable.
    setTimeout(() => {
      const result = this.lastHPSResult;
      if (result.chroma == null) return;

      // Bass note = lowest detected frequency (first peak, ascending order)
      const bassFreq = result.frequencies[0] ?? 0;

      const matches = matchChordFromChroma(result.chroma, bassFreq);
      if (matches.length === 0) return;

      this.chordCallback?.(matches, event, result);
    }, 60);
  }
}

// Singleton for use across the app
export const audioEngine = new AudioEngine();
