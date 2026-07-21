/**
 * useAudioPipeline — wires AudioEngine into React
 *
 * Spec Section 6 + Section 11:
 * - On mount: explicit YIN → HPS switch
 * - Audio starts 200ms after mount
 * - On unmount: HPS → IDLE
 */

import { useEffect, useRef } from 'react';
import { audioEngine } from '../audio/AudioEngine';
import { useFretcastStore } from '../store';
import { AudioMode, DetectedChord } from '../types';

interface Options {
  mode: AudioMode;
  stream: MediaStream | null;
  sessionStartMs: number;
  onOnsetTimestamp?: (ts: number) => void;
}

export function useAudioPipeline({ mode, stream, sessionStartMs, onOnsetTimestamp }: Options) {
  const addDetectedChord = useFretcastStore((s) => s.addDetectedChord);
  const setLiveChord = useFretcastStore((s) => s.setLiveChord);
  const startedRef = useRef(false);

  useEffect(() => {
    if (stream == null || mode === 'IDLE') return;
    if (startedRef.current) return;

    let cancelled = false;

    const start = async () => {
      // Spec Section 11: audio starts 200ms after mount
      await new Promise((r) => setTimeout(r, 200));
      if (cancelled) return;

      await audioEngine.startFromStream(stream);

      // Spec Section 6: explicit mode switch
      audioEngine.setMode(mode);

      startedRef.current = true;

      if (mode === 'HPS') {
        audioEngine.onChordDetected((matches, onsetEvent) => {
          if (matches.length === 0) return;

          const best = matches[0];
          if (best.score < 0.3) return; // below minimum confidence

          const confidence = Math.min(best.score * 1.4, 1.0); // scale to 0..1

          const chord: DetectedChord = {
            timestamp: sessionStartMs + onsetEvent.timestamp,
            plateauTimestamp: sessionStartMs + onsetEvent.timestamp,
            name: best.chord.name,
            root: best.chord.root,
            quality: best.chord.quality,
            fretPosition: null,       // filled by fusion engine Phase 5
            voicing: best.chord.pitchClasses.map(pcToNote),
            confidence,
            audioConfidence: confidence,
            visionConfidence: 0,      // filled by fusion engine Phase 5
            corrected: false,
            heroFrameTimestamp: sessionStartMs + onsetEvent.timestamp,
          };

          addDetectedChord(chord);
          setLiveChord(chord);
          onOnsetTimestamp?.(onsetEvent.timestamp);
        });
      }

      if (mode === 'YIN') {
        audioEngine.onYINResult((result) => {
          if (result.confidence > 0.7 && result.frequency > 0) {
            // YIN results consumed by calibration screen — Phase 3
          }
        });
      }
    };

    start().catch(console.error);

    return () => {
      cancelled = true;
      // Spec Section 6: explicit deactivation on unmount
      audioEngine.setMode('IDLE');
      audioEngine.stop();
      startedRef.current = false;
    };
  }, [stream, mode]); // eslint-disable-line react-hooks/exhaustive-deps
}

const PC_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function pcToNote(pc: number): string { return PC_NOTES[pc % 12]; }
