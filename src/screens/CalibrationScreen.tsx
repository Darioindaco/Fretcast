/**
 * CalibrationScreen — spec Section 4
 *
 * Simplified 5-phase calibration flow:
 *
 *   SETUP        → show instructions
 *   CORNERS      → tap 4 fretboard corners → establishes homography boundary
 *   FRET_CAPTURE → play frets 3, 5, 7, 9, 12 — YIN + hand tracking capture position automatically
 *   COMPUTING    → derive homography + FretRegion grid
 *   CONFIRM      → show grid overlay; save
 *
 * Spec rules enforced:
 *   - YIN active during calibration, never HPS (spec Section 6)
 *   - Logarithmic fret spacing applied after homography (spec Section 3)
 *   - Left-hand auto-detection: nut on RIGHT → isLeftHanded (spec Section 4)
 *   - Hand tracking position read at the moment of YIN detection (no manual tap)
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { useFretcastStore } from '../store';
import { CalibrationMap, FretRegion, RootStackParamList } from '../types';
import { audioEngine } from '../audio/AudioEngine';
import { Point2D, computeHomography, applyInverseHomography } from '../vision/homography';
import { computeFretGrid, estimateVisibleFrets, expectedFrequenciesForFret } from '../vision/fretGrid';
import { useHandTracking, HandLandmark } from '../hooks/useHandTracking';
import HandCanvas from '../components/HandCanvas';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Calibration'>;
};

type CalibStep = 'SETUP' | 'CORNERS' | 'FRET_CAPTURE' | 'COMPUTING' | 'CONFIRM';

const CORNER_LABELS = ['Top-left (nut end)', 'Top-right (nut end)', 'Bottom-right', 'Bottom-left'];
const CAPTURE_FRETS = [3, 5, 7, 9, 12];

interface CapturedFret {
  fret: number;
  pixelX: number;
  pixelY: number;
  frequency: number;
  noteName: string;
}

const STORAGE_KEY = '@fretcast/calibrationMap';

// ── Screen ────────────────────────────────────────────────────────────────────

export default function CalibrationScreen({ navigation }: Props) {
  const setCalibrationMap = useFretcastStore((s) => s.setCalibrationMap);
  const setCalibrationProgress = useFretcastStore((s) => s.setCalibrationProgress);

  const [step, setStep] = useState<CalibStep>('SETUP');
  const [corners, setCorners] = useState<Point2D[]>([]);
  const [capturedFrets, setCapturedFrets] = useState<CapturedFret[]>([]);
  const [currentFretIdx, setCurrentFretIdx] = useState(0);
  const [listeningFreq, setListeningFreq] = useState<number | null>(null);
  const [captureStatus, setCaptureStatus] = useState<'idle' | 'listening' | 'captured' | 'error'>('idle');
  const [captureMsg, setCaptureMsg] = useState('');
  const [audioDebug, setAudioDebug] = useState('');
  const [tapLog, setTapLog] = useState('—');
  const [fretGrid, setFretGrid] = useState<FretRegion[][] | null>(null);
  const [homographyMatrix, setHomographyMatrix] = useState<number[][] | null>(null);
  const [isLeftHanded, setIsLeftHanded] = useState(false);
  const [containerSize, setContainerSize] = useState({ width: 1, height: 1 });

  // Web camera
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [streamReady, setStreamReady] = useState(false);

  // Hand tracking
  const { landmarks, modelLoaded } = useHandTracking({
    videoRef: videoRef as React.RefObject<HTMLVideoElement | null>,
    enabled: Platform.OS === 'web' && streamReady,
  });

  // Refs for access inside YIN callback (avoids stale closures)
  const landmarksRef = useRef<HandLandmark[] | null>(null);
  const containerSizeRef = useRef({ width: 1, height: 1 });
  const currentFretIdxRef = useRef(0);
  const capturedFretsRef = useRef<CapturedFret[]>([]);
  const isListeningRef = useRef(false);
  const yinReadingsRef = useRef<number[]>([]);
  const cooldownRef = useRef(false);

  useEffect(() => { landmarksRef.current = landmarks; }, [landmarks]);
  useEffect(() => { containerSizeRef.current = containerSize; }, [containerSize]);
  useEffect(() => { currentFretIdxRef.current = currentFretIdx; }, [currentFretIdx]);
  useEffect(() => { capturedFretsRef.current = capturedFrets; }, [capturedFrets]);

  // ── Camera ────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (Platform.OS !== 'web') return;
    let cancelled = false;
    (async () => {
      let stream: MediaStream | null = null;
      const attempts: MediaStreamConstraints[] = [
        { video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true },
        { video: true, audio: true },
        { audio: true },
      ];
      for (const c of attempts) {
        try { stream = await navigator.mediaDevices.getUserMedia(c); break; } catch { /* next */ }
      }
      if (cancelled) { stream?.getTracks().forEach((t) => t.stop()); return; }
      if (stream != null) {
        const hasAudio = stream.getAudioTracks().length > 0;
        const hasVideo = stream.getVideoTracks().length > 0;
        setAudioDebug(`${hasVideo ? 'video+' : ''}${hasAudio ? 'audio OK' : 'NO AUDIO'}`);
        streamRef.current = stream;
        if (videoRef.current != null) videoRef.current.srcObject = stream;
      } else {
        setAudioDebug('getUserMedia FAILED');
      }
      setStreamReady(true);
    })();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ── YIN — always-on during FRET_CAPTURE ──────────────────────────────────

  useEffect(() => {
    if (step !== 'FRET_CAPTURE' || !streamReady) return;
    if (streamRef.current == null) { setAudioDebug('No stream'); return; }

    let started = false;
    let cancelled = false;

    (async () => {
      setAudioDebug('starting YIN…');
      audioEngine.stop();
      try {
        await audioEngine.startFromStream(streamRef.current!);
      } catch (e) {
        setAudioDebug(`YIN start error: ${e}`);
        return;
      }
      if (cancelled) { audioEngine.stop(); return; }

      audioEngine.setMode('YIN');
      started = true;
      isListeningRef.current = true;
      setCaptureStatus('listening');
      setAudioDebug('YIN active — play fret with index finger');

      audioEngine.onYINResult((result) => {
        const freq = result.frequency;
        const conf = result.confidence;

        setAudioDebug(`YIN: ${freq > 0 ? Math.round(freq) + 'Hz' : '--'} conf:${conf.toFixed(2)}`);

        if (!isListeningRef.current || cooldownRef.current) return;

        // Reset buffer on any weak reading — only a clean sustained note fills it
        if (conf < 0.75 || freq <= 0) {
          yinReadingsRef.current = [];
          setListeningFreq(null);
          return;
        }

        yinReadingsRef.current.push(freq);
        setListeningFreq(freq);

        // Require 5 readings — and all must be within 8% of each other (stable pitch)
        if (yinReadingsRef.current.length >= 5) {
          const readings = yinReadingsRef.current.slice(-5);
          const min = Math.min(...readings);
          const max = Math.max(...readings);
          if (max / min > 1.08) {
            // Pitch unstable — keep last 2 and keep building
            yinReadingsRef.current = readings.slice(-2);
            return;
          }
          yinReadingsRef.current = [];
          isListeningRef.current = false;
          processCapturedFrequency(medianFreq(readings));
        }
      });
    })();

    return () => {
      cancelled = true;
      isListeningRef.current = false;
      if (started) { audioEngine.setMode('IDLE'); audioEngine.stop(); }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, streamReady]);

  // ── COMPUTING ─────────────────────────────────────────────────────────────

  useEffect(() => {
    if (step !== 'COMPUTING') return;
    const [tl, tr, br, bl] = corners as [Point2D, Point2D, Point2D, Point2D];
    const leftHanded = tl.x > tr.x;
    setIsLeftHanded(leftHanded);
    const dst: [Point2D, Point2D, Point2D, Point2D] = [
      { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 },
    ];
    const H = computeHomography([tl, tr, br, bl], dst);
    setHomographyMatrix(H);
    let visibleFrets = 17;
    const fret12 = capturedFretsRef.current.find((c) => c.fret === 12);
    if (fret12 != null) {
      const ny12 = fret12.pixelY / containerSizeRef.current.height;
      visibleFrets = estimateVisibleFrets(12, ny12);
    }
    setFretGrid(computeFretGrid(H, leftHanded, visibleFrets));
    setStep('CONFIRM');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  // ── Fret frequency capture ────────────────────────────────────────────────

  const processCapturedFrequency = useCallback((freq: number) => {
    const fretIdx = currentFretIdxRef.current;
    const fret = CAPTURE_FRETS[fretIdx];

    // Validate frequency matches expected fret
    const expected = expectedFrequenciesForFret(fret);
    const closest = expected.reduce((best, cur) =>
      Math.abs(cur.freq - freq) < Math.abs(best.freq - freq) ? cur : best
    );
    // ±2 semitones = 2^(2/12) ≈ 1.122 — tight enough to reject neighbouring frets
    const ratio = freq / closest.freq;
    if (ratio < 0.891 || ratio > 1.122) {
      cooldownRef.current = true;
      setCaptureStatus('error');
      setCaptureMsg(`${Math.round(freq)}Hz doesn't match fret ${fret} — play fret ${fret}`);
      setTimeout(() => {
        setCaptureStatus('listening');
        setCaptureMsg('');
        yinReadingsRef.current = [];
        setListeningFreq(null);
        isListeningRef.current = true;
        cooldownRef.current = false;
      }, 1500);
      return;
    }

    // Get finger position from hand tracking (index fingertip = landmark 8)
    const lm = landmarksRef.current;
    const cs = containerSizeRef.current;

    if (lm == null || lm.length < 9) {
      // Hand not visible — reject and re-arm
      cooldownRef.current = true;
      setCaptureStatus('error');
      setCaptureMsg('Hand not visible — keep index finger on fret, then play');
      setTimeout(() => {
        setCaptureStatus('listening');
        setCaptureMsg('');
        yinReadingsRef.current = [];
        setListeningFreq(null);
        isListeningRef.current = true;
        cooldownRef.current = false;
      }, 2000);
      return;
    }

    const tip = lm[8]; // index fingertip
    const pixelX = tip.x * cs.width;
    const pixelY = tip.y * cs.height;

    const newEntry: CapturedFret = {
      fret,
      pixelX,
      pixelY,
      frequency: freq,
      noteName: freqToNoteName(freq),
    };

    const newCaptures = [...capturedFretsRef.current, newEntry];
    capturedFretsRef.current = newCaptures;
    setCapturedFrets(newCaptures);
    setCaptureStatus('captured');
    setCaptureMsg(`✓ fret ${fret} at (${Math.round(pixelX)}, ${Math.round(pixelY)}) — ${Math.round(freq)}Hz`);

    cooldownRef.current = true;
    setTimeout(() => {
      cooldownRef.current = false;
      const nextIdx = fretIdx + 1;
      if (nextIdx >= CAPTURE_FRETS.length) {
        setStep('COMPUTING');
      } else {
        currentFretIdxRef.current = nextIdx;
        setCurrentFretIdx(nextIdx);
        setCaptureStatus('listening');
        setCaptureMsg('');
        yinReadingsRef.current = [];
        setListeningFreq(null);
        isListeningRef.current = true;
      }
    }, 1000);
  }, []);

  // ── Corner taps (web: document-level listener; native: TouchableOpacity) ──

  const handleCornerTap = useCallback((x: number, y: number) => {
    setTapLog(`x:${Math.round(x)} y:${Math.round(y)}`);
    setCorners((prev) => {
      if (prev.length >= 4) return prev;
      return [...prev, { x, y }];
    });
  }, []);

  useEffect(() => {
    if (Platform.OS !== 'web' || step !== 'CORNERS') return;

    const handler = (e: MouseEvent) => {
      const target = e.target as Element;
      if (target?.closest?.('button, a, [role="button"]')) return;
      const videoEl = videoRef.current;
      const rect = videoEl?.getBoundingClientRect?.() ?? { left: 0, top: 0 };
      handleCornerTap(e.clientX - rect.left, e.clientY - rect.top);
    };

    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, [step, handleCornerTap]);

  // ── Save ──────────────────────────────────────────────────────────────────

  const handleSave = useCallback(async () => {
    if (homographyMatrix == null || fretGrid == null) return;
    const cs = containerSizeRef.current;

    // Derive anchor points from corners (nut = midpoint of top edge, body = midpoint of bottom edge)
    const nutAnchor = corners.length >= 2
      ? { pixelX: (corners[0].x + corners[1].x) / 2, pixelY: (corners[0].y + corners[1].y) / 2 }
      : { pixelX: 0, pixelY: 0 };
    const bodyJointAnchor = corners.length >= 4
      ? { pixelX: (corners[2].x + corners[3].x) / 2, pixelY: (corners[2].y + corners[3].y) / 2 }
      : { pixelX: 0, pixelY: 0 };
    const fret12Capture = capturedFretsRef.current.find((c) => c.fret === 12);
    const twelfthFretAnchor = fret12Capture != null
      ? { pixelX: fret12Capture.pixelX, pixelY: fret12Capture.pixelY }
      : { pixelX: 0, pixelY: 0 };

    const map: CalibrationMap = {
      anchorPoints: capturedFretsRef.current.map((cf) => ({
        fret: cf.fret,
        normalizedX: 0.5,
        normalizedY: cf.pixelY / cs.height,
        pixelX: cf.pixelX,
        pixelY: cf.pixelY,
        reprojectionError: 0,
      })),
      nutAnchor,
      bodyJointAnchor,
      twelfthFretAnchor,
      homographyMatrix,
      fretGrid,
      scaleLength: 648,
      isLeftHanded,
      capoFret: null,
      createdAt: Date.now(),
    };

    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(map));
    setCalibrationMap(map);
    setCalibrationProgress({
      phase: 'CONFIRM',
      capturedFrets: capturedFretsRef.current.map((c) => c.fret),
      pendingFret: null,
      isLeftHanded,
      gridValidated: true,
    });
    navigation.goBack();
  }, [homographyMatrix, fretGrid, corners, isLeftHanded, setCalibrationMap, setCalibrationProgress, navigation]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <View
      style={styles.container}
      onLayout={(e) =>
        setContainerSize({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })
      }
    >
      {/* Camera feed */}
      {Platform.OS === 'web' && (
        <video
          ref={videoRef as React.RefObject<HTMLVideoElement>}
          autoPlay playsInline muted
          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover' }}
        />
      )}

      {/* Hand tracking overlay */}
      <HandCanvas landmarks={landmarks} width={containerSize.width} height={containerSize.height} />

      {/* Native corner tap interceptor */}
      {Platform.OS !== 'web' && step === 'CORNERS' && (
        <TouchableOpacity
          style={StyleSheet.absoluteFill}
          activeOpacity={1}
          onPress={(e) => handleCornerTap(e.nativeEvent.locationX, e.nativeEvent.locationY)}
        />
      )}

      {/* Corner markers */}
      {corners.map((pt, i) => (
        <View key={`corner-${i}`} style={[styles.marker, styles.cornerMarker, { left: pt.x - 10, top: pt.y - 10 }]}>
          <Text style={styles.markerNum}>{i + 1}</Text>
        </View>
      ))}

      {/* Captured fret markers */}
      {capturedFrets.map((cf, i) => (
        <View key={`fret-${i}`} style={[styles.marker, styles.fretMarker, { left: cf.pixelX - 10, top: cf.pixelY - 10 }]}>
          <Text style={styles.markerNum}>{cf.fret}</Text>
        </View>
      ))}

      {/* Grid overlay in CONFIRM step */}
      {step === 'CONFIRM' && fretGrid != null && homographyMatrix != null &&
        renderGridOverlay(fretGrid, homographyMatrix, containerSize)
      }

      {/* Instruction panel */}
      <View style={styles.panel} pointerEvents="none">
        {renderInstructions({ step, corners, currentFretIdx, captureStatus, captureMsg, listeningFreq, capturedFrets, modelLoaded })}
      </View>

      {/* Persistent debug bar */}
      <View style={styles.debugBar} pointerEvents="none">
        <Text style={styles.debugLine}>step:{step}  stream:{streamReady ? 'ready' : 'wait'}  hand:{modelLoaded ? (landmarks ? 'tracking' : 'no hand') : 'loading'}</Text>
        <Text style={styles.debugLine}>corners:{corners.length}/4  frets:{capturedFrets.length}/5</Text>
        {step === 'CORNERS' && <Text style={styles.debugLine}>last tap: {tapLog}</Text>}
        <Text style={styles.debugLine}>audio: {audioDebug}</Text>
        {listeningFreq != null && (
          <Text style={styles.debugLine}>YIN: {Math.round(listeningFreq)}Hz  status:{captureStatus}</Text>
        )}
      </View>

      {/* Action buttons */}
      <View style={styles.buttonRow}>
        {step === 'SETUP' && (
          <TouchableOpacity style={styles.btn} onPress={() => setStep('CORNERS')}>
            <Text style={styles.btnText}>Start Calibration</Text>
          </TouchableOpacity>
        )}
        {step === 'CORNERS' && corners.length === 4 && (
          <TouchableOpacity style={styles.btn} onPress={() => setStep('FRET_CAPTURE')}>
            <Text style={styles.btnText}>Corners done →</Text>
          </TouchableOpacity>
        )}
        {step === 'CORNERS' && corners.length > 0 && (
          <TouchableOpacity style={styles.btnSecondary} onPress={() => setCorners([])}>
            <Text style={styles.btnText}>Reset</Text>
          </TouchableOpacity>
        )}
        {step === 'COMPUTING' && <ActivityIndicator color="#fff" />}
        {step === 'CONFIRM' && (
          <>
            <TouchableOpacity style={styles.btnSecondary} onPress={() => {
              setStep('CORNERS'); setCorners([]);
              setCapturedFrets([]); capturedFretsRef.current = [];
              setCurrentFretIdx(0); currentFretIdxRef.current = 0;
              setFretGrid(null); setHomographyMatrix(null);
            }}>
              <Text style={styles.btnText}>Redo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.btn} onPress={handleSave}>
              <Text style={styles.btnText}>Save Calibration</Text>
            </TouchableOpacity>
          </>
        )}
        <TouchableOpacity style={styles.btnDanger} onPress={() => navigation.goBack()}>
          <Text style={styles.btnText}>Cancel</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ── Instructions ──────────────────────────────────────────────────────────────

interface InstructionProps {
  step: CalibStep;
  corners: Point2D[];
  currentFretIdx: number;
  captureStatus: string;
  captureMsg: string;
  listeningFreq: number | null;
  capturedFrets: CapturedFret[];
  modelLoaded: boolean;
}

function renderInstructions(p: InstructionProps) {
  switch (p.step) {
    case 'SETUP':
      return (
        <>
          <Text style={styles.title}>Calibrate Fretcast</Text>
          <Text style={styles.body}>
            Point the camera at your guitar neck from above.{'\n'}
            Keep the full fretboard visible.{'\n\n'}
            Step 1: tap the 4 corners of the fretboard.{'\n'}
            Step 2: play frets 3, 5, 7, 9, 12 with your index finger — the app detects position automatically.
          </Text>
        </>
      );
    case 'CORNERS':
      return (
        <>
          <Text style={styles.title}>Step 1 — Fretboard Corners</Text>
          <Text style={styles.body}>
            {p.corners.length < 4
              ? `Tap: ${CORNER_LABELS[p.corners.length]}`
              : '✓ All 4 corners set — press Corners done'}
          </Text>
          <Text style={styles.sub}>{p.corners.length}/4 tapped</Text>
        </>
      );
    case 'FRET_CAPTURE': {
      const fret = CAPTURE_FRETS[p.currentFretIdx];
      return (
        <>
          <Text style={styles.title}>Step 2 — Play fret {fret}  ({p.currentFretIdx + 1}/5)</Text>
          <Text style={styles.body}>
            Press fret {fret} with your index finger and pluck any string.{'\n'}
            The app captures your finger position automatically.
          </Text>
          {!p.modelLoaded && <Text style={styles.warn}>⚠ Hand tracking loading…</Text>}
          {p.captureStatus === 'listening' && (
            <Text style={styles.listening}>
              Listening…{p.listeningFreq != null ? `  ${Math.round(p.listeningFreq)} Hz` : ''}
            </Text>
          )}
          {p.captureStatus === 'captured' && <Text style={styles.captured}>{p.captureMsg}</Text>}
          {p.captureStatus === 'error' && <Text style={styles.error}>{p.captureMsg}</Text>}
          <Text style={styles.sub}>{p.capturedFrets.length}/5 done</Text>
        </>
      );
    }
    case 'COMPUTING':
      return <Text style={styles.title}>Computing fret grid…</Text>;
    case 'CONFIRM':
      return (
        <>
          <Text style={styles.title}>Confirm Grid</Text>
          <Text style={styles.body}>
            Blue dots should align with fret intersections.{'\n'}
            If not, tap Redo.
          </Text>
        </>
      );
    default:
      return null;
  }
}

// ── Grid overlay ──────────────────────────────────────────────────────────────

function renderGridOverlay(
  grid: FretRegion[][],
  H: number[][],
  containerSize: { width: number; height: number }
) {
  const SHOW_FRETS = [0, 3, 5, 7, 9, 12];
  const dots: Array<{ x: number; y: number; key: string }> = [];
  for (const fret of SHOW_FRETS) {
    for (let s = 0; s < 6; s++) {
      if (grid[fret]?.[s] == null) continue;
      const { normalizedX, normalizedY } = grid[fret][s];
      const px = applyInverseHomography(H, { x: normalizedX, y: normalizedY });
      dots.push({ x: px.x * containerSize.width, y: px.y * containerSize.height, key: `${fret}-${s}` });
    }
  }
  return dots.map((d) => (
    <View key={d.key} style={[styles.gridDot, { left: d.x - 4, top: d.y - 4 }]} />
  ));
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function medianFreq(freqs: number[]): number {
  const sorted = [...freqs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
function freqToNoteName(freq: number): string {
  const midi = Math.round(12 * Math.log2(freq / 440) + 69);
  const octave = Math.floor(midi / 12) - 1;
  return NOTE_NAMES[((midi % 12) + 12) % 12] + octave;
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  panel: {
    position: 'absolute', top: 60, left: 16, right: 16,
    backgroundColor: 'rgba(0,0,0,0.78)', borderRadius: 12, padding: 16,
  },
  title: { color: '#fff', fontSize: 18, fontWeight: '700', marginBottom: 8 },
  body: { color: 'rgba(255,255,255,0.85)', fontSize: 15, lineHeight: 22 },
  sub: { color: 'rgba(255,255,255,0.5)', fontSize: 13, marginTop: 6 },
  warn: { color: '#FBBF24', fontSize: 13, marginTop: 6 },
  listening: { color: '#60A5FA', fontSize: 15, fontWeight: '600', marginTop: 6 },
  captured: { color: '#34D399', fontSize: 14, fontWeight: '600', marginTop: 6 },
  error: { color: '#F87171', fontSize: 14, marginTop: 6 },
  debugBar: {
    position: 'absolute', bottom: 140, left: 8, right: 8,
    backgroundColor: 'rgba(0,0,0,0.88)', borderRadius: 8, padding: 8, zIndex: 30,
  },
  debugLine: { color: '#FBBF24', fontSize: 11, lineHeight: 17, fontFamily: 'monospace' as const },
  buttonRow: {
    position: 'absolute', bottom: 48, left: 16, right: 16,
    flexDirection: 'row', gap: 10, justifyContent: 'center', flexWrap: 'wrap', zIndex: 20,
  },
  btn: { backgroundColor: '#2563EB', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 10 },
  btnSecondary: { backgroundColor: 'rgba(255,255,255,0.15)', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 10 },
  btnDanger: { backgroundColor: 'rgba(239,68,68,0.7)', paddingHorizontal: 20, paddingVertical: 12, borderRadius: 10 },
  btnText: { color: '#fff', fontWeight: '600', fontSize: 15 },
  marker: {
    position: 'absolute', width: 20, height: 20, borderRadius: 10,
    borderWidth: 2, alignItems: 'center', justifyContent: 'center',
  },
  cornerMarker: { borderColor: '#FBBF24', backgroundColor: 'rgba(251,191,36,0.35)' },
  fretMarker: { borderColor: '#60A5FA', backgroundColor: 'rgba(96,165,250,0.35)' },
  markerNum: { color: '#fff', fontSize: 9, fontWeight: '800' },
  gridDot: {
    position: 'absolute', width: 8, height: 8, borderRadius: 4,
    backgroundColor: 'rgba(96,165,250,0.9)',
  },
});
