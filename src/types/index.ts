// ─── FRETCAST — Section 10 Data Types ───────────────────────────────────────
// These types are the single source of truth.
// Never use `any`. All fields match the spec exactly.

export interface AnchorPoint {
  fret: number;
  normalizedX: number;
  normalizedY: number;
  pixelX: number;
  pixelY: number;
  reprojectionError: number;
}

export interface FretRegion {
  fret: number;
  string: number;
  normalizedX: number;
  normalizedY: number;
  radiusPx: number;
}

export interface CalibrationMap {
  anchorPoints: AnchorPoint[];
  nutAnchor: { pixelX: number; pixelY: number };
  bodyJointAnchor: { pixelX: number; pixelY: number };
  twelfthFretAnchor: { pixelX: number; pixelY: number };
  homographyMatrix: number[][];
  fretGrid: FretRegion[][];
  scaleLength: number;
  isLeftHanded: boolean;
  capoFret: number | null;
  createdAt: number;
  guitarNickname?: string;
}

export type FingerName = 'index' | 'middle' | 'ring' | 'pinky';

export interface FingerPosition {
  finger: FingerName;
  fret: number | null;
  string: number | null;
  pixelX: number;
  pixelY: number;
  confidence: number;
}

export interface VisionFrame {
  timestamp: number;
  fingerPositions: FingerPosition[];
  fingerVelocities: number[];
  frameConfidence: number;
  handVisible: boolean;
  anchorsStable: boolean;
}

export interface DetectedChord {
  timestamp: number;
  plateauTimestamp: number;
  name: string;
  root: string;
  quality: string;
  fretPosition: number | null;
  voicing: string[];
  confidence: number;
  audioConfidence: number;
  visionConfidence: number;
  corrected: boolean;
  heroFrameTimestamp: number;
}

export interface AnalysisResult {
  chords: DetectedChord[];
  key: string;
  relativeKey: string;
  bpm: number;
  timeSignature: string;
  progression: string;
  romanNumerals: string;
  tabString: string;
  duration: number;
}

// ─── Audio pipeline types ────────────────────────────────────────────────────

/** Spec Section 6: YIN = calibration only, HPS = recording only. Switch is explicit. */
export type AudioMode = 'YIN' | 'HPS' | 'IDLE';

export interface ChordDetectedEvent {
  chord: DetectedChord;
  plateauTimestamp: number;
}

export interface OnsetEvent {
  timestamp: number;
  hfc: number;
}

// ─── Calibration flow types ──────────────────────────────────────────────────

export type CalibrationPhase =
  | 'SETUP'
  | 'FRETBOARD_DETECTION'
  | 'ANCHOR_IDENTIFICATION'
  | 'ANCHOR_CAPTURE'
  | 'GRID_COMPUTATION'
  | 'CONFIRM';

export interface CalibrationProgress {
  phase: CalibrationPhase;
  capturedFrets: number[];        // which of [3,5,7,9,12] are done
  pendingFret: number | null;     // fret currently being captured
  isLeftHanded: boolean;
  gridValidated: boolean;
}

// ─── Recording session types ─────────────────────────────────────────────────

export type RecordingState = 'IDLE' | 'RECORDING' | 'STOPPING';

export interface RecordingSession {
  startTimestamp: number;
  videoUri: string | null;
  overlayUri: string | null;
  mergedVideoUri: string | null;
  analysisResult: AnalysisResult | null;
  durationMs: number;
}

// ─── Processing pipeline types ───────────────────────────────────────────────

export type ProcessingStep =
  | 'IDLE'
  | 'ANALYZING_AUDIO'
  | 'RENDERING_OVERLAY'
  | 'MERGING_VIDEO'
  | 'COMPLETE'
  | 'ERROR';

export interface ProcessingStatus {
  step: ProcessingStep;
  progressPercent: number;
  errorMessage: string | null;
}

// ─── Navigation param types ──────────────────────────────────────────────────

export type RootStackParamList = {
  Home: undefined;
  Recording: undefined;
  Processing: { sessionId: string };
  Review: { sessionId: string };
  Calibration: undefined;
};
