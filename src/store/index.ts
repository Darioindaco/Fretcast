import { create } from 'zustand';
import {
  AudioMode,
  CalibrationMap,
  CalibrationProgress,
  DetectedChord,
  ProcessingStatus,
  RecordingSession,
  RecordingState,
  VisionFrame,
} from '../types';

// ─── Calibration slice ───────────────────────────────────────────────────────

interface CalibrationSlice {
  calibrationMap: CalibrationMap | null;
  calibrationProgress: CalibrationProgress | null;
  isCalibrated: boolean;
  setCalibrationMap: (map: CalibrationMap) => void;
  setCalibrationProgress: (progress: CalibrationProgress) => void;
  clearCalibration: () => void;
}

// ─── Recording slice ─────────────────────────────────────────────────────────

interface RecordingSlice {
  recordingState: RecordingState;
  activeSession: RecordingSession | null;
  audioMode: AudioMode;
  setRecordingState: (state: RecordingState) => void;
  setActiveSession: (session: RecordingSession | null) => void;
  /** Spec Section 6: switch must be explicit — never implicit */
  setAudioMode: (mode: AudioMode) => void;
}

// ─── Vision slice ─────────────────────────────────────────────────────────────

interface VisionSlice {
  /** Ring buffer: last 200ms of VisionFrames (spec Section 7) */
  visionRingBuffer: VisionFrame[];
  /**
   * Live homography updated every frame by Phase 4 LK tracker.
   * null = not calibrated or tracking not started.
   * Consumers should fall back to calibrationMap.homographyMatrix when null.
   */
  liveHomography: number[][] | null;
  pushVisionFrame: (frame: VisionFrame) => void;
  setLiveHomography: (h: number[][] | null) => void;
  clearVisionBuffer: () => void;
}

// ─── Detection slice ─────────────────────────────────────────────────────────

interface DetectionSlice {
  detectedChords: DetectedChord[];
  liveChord: DetectedChord | null;
  addDetectedChord: (chord: DetectedChord) => void;
  updateChord: (timestamp: number, patch: Partial<DetectedChord>) => void;
  setLiveChord: (chord: DetectedChord | null) => void;
  clearChords: () => void;
}

// ─── Processing slice ─────────────────────────────────────────────────────────

interface ProcessingSlice {
  processingStatus: ProcessingStatus;
  setProcessingStep: (status: ProcessingStatus) => void;
}

// ─── Root store ───────────────────────────────────────────────────────────────

type FretcastStore =
  CalibrationSlice &
  RecordingSlice &
  VisionSlice &
  DetectionSlice &
  ProcessingSlice;

const VISION_BUFFER_WINDOW_MS = 200;

export const useFretcastStore = create<FretcastStore>((set, get) => ({
  // ── Calibration ──────────────────────────────────────────────────────────
  calibrationMap: null,
  calibrationProgress: null,
  isCalibrated: false,

  setCalibrationMap: (map) =>
    set({ calibrationMap: map, isCalibrated: true }),

  setCalibrationProgress: (progress) =>
    set({ calibrationProgress: progress }),

  clearCalibration: () =>
    set({ calibrationMap: null, calibrationProgress: null, isCalibrated: false }),

  // ── Recording ────────────────────────────────────────────────────────────
  recordingState: 'IDLE',
  activeSession: null,
  audioMode: 'IDLE',

  setRecordingState: (recordingState) => set({ recordingState }),
  setActiveSession: (activeSession) => set({ activeSession }),
  setAudioMode: (audioMode) => set({ audioMode }),

  // ── Vision ring buffer (200ms window) ────────────────────────────────────
  visionRingBuffer: [],
  liveHomography: null,

  pushVisionFrame: (frame) => {
    const now = frame.timestamp;
    const cutoff = now - VISION_BUFFER_WINDOW_MS;
    const prev = get().visionRingBuffer;
    const pruned = prev.filter((f) => f.timestamp >= cutoff);
    set({ visionRingBuffer: [...pruned, frame] });
  },

  setLiveHomography: (liveHomography) => set({ liveHomography }),

  clearVisionBuffer: () => set({ visionRingBuffer: [] }),

  // ── Detection ────────────────────────────────────────────────────────────
  detectedChords: [],
  liveChord: null,

  addDetectedChord: (chord) =>
    set((s) => ({ detectedChords: [...s.detectedChords, chord] })),

  updateChord: (timestamp, patch) =>
    set((s) => ({
      detectedChords: s.detectedChords.map((c) =>
        c.timestamp === timestamp ? { ...c, ...patch } : c
      ),
    })),

  setLiveChord: (liveChord) => set({ liveChord }),

  clearChords: () => set({ detectedChords: [], liveChord: null }),

  // ── Processing ────────────────────────────────────────────────────────────
  processingStatus: { step: 'IDLE', progressPercent: 0, errorMessage: null },

  setProcessingStep: (processingStatus) => set({ processingStatus }),
}));
