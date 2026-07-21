/**
 * useFretPosition
 *
 * Projects the hand wrist (landmark 0) onto the neck centre line
 * (nut → bridge) and maps that position to an estimated fret number
 * using the calibrated anchor points.
 *
 * Combined with the audio chord name from HPS this gives the exact
 * voicing — e.g. "C major at fret 3" vs "C major open".
 *
 * The neck line is derived from the existing homography:
 *   nut    = applyInverseHomography(H, { x: 0.5, y: 0 })
 *   bridge = applyInverseHomography(H, { x: 0.5, y: 1 })
 *
 * No new calibration needed.
 */

import { useMemo } from 'react';
import { CalibrationMap } from '../types';
import { applyHomography, applyInverseHomography } from '../vision/homography';
import { HandLandmark } from './useHandTracking';

export interface FretPositionResult {
  /** Rounded fret number (0 = open). null when hand not visible. */
  fret: number | null;
  /** Exact (sub-fret) position for smooth UI. null when hand not visible. */
  fretExact: number | null;
  /** 0 = nut, 1 = bridge end. null when hand not visible. */
  t: number | null;
  /** Neck line endpoints in display pixels (stable from calibration). */
  nutPixel:    { x: number; y: number } | null;
  bridgePixel: { x: number; y: number } | null;
  /** Wrist position projected onto the neck line, in display pixels. */
  handOnLinePixel: { x: number; y: number } | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** t = scalar projection of `pt` onto the segment (start → end), 0..1 */
function projectT(
  pt:    { x: number; y: number },
  start: { x: number; y: number },
  end:   { x: number; y: number },
): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len2 = dx * dx + dy * dy;
  if (len2 < 1) return 0;
  return ((pt.x - start.x) * dx + (pt.y - start.y) * dy) / len2;
}

/**
 * Map normalised-Y (0 = nut, 1 = bridge) → fret number.
 * Uses the calibrated anchor points (frets 3, 5, 7, 9, 12) + nut at 0.
 * Interpolates between known points; clamps outside.
 */
function normYToFret(normY: number, anchorPoints: CalibrationMap['anchorPoints']): number {
  const refs = [
    { fret: 0, y: 0 },
    ...anchorPoints.map((a) => ({ fret: a.fret, y: a.normalizedY })),
  ].sort((a, b) => a.y - b.y);

  if (normY <= refs[0].y) return refs[0].fret;
  if (normY >= refs[refs.length - 1].y) return refs[refs.length - 1].fret;

  for (let i = 0; i < refs.length - 1; i++) {
    if (normY >= refs[i].y && normY <= refs[i + 1].y) {
      const span = refs[i + 1].y - refs[i].y;
      const t    = span > 0 ? (normY - refs[i].y) / span : 0;
      return refs[i].fret + t * (refs[i + 1].fret - refs[i].fret);
    }
  }

  return 0;
}

// ── Hook ──────────────────────────────────────────────────────────────────────

const EMPTY: FretPositionResult = {
  fret: null, fretExact: null, t: null,
  nutPixel: null, bridgePixel: null, handOnLinePixel: null,
};

export function useFretPosition(
  landmarks:      HandLandmark[] | null,
  calibrationMap: CalibrationMap | null,
  displayWidth:   number,
  displayHeight:  number,
): FretPositionResult {
  return useMemo(() => {
    if (landmarks == null || calibrationMap == null || displayWidth <= 1) return EMPTY;

    const H = calibrationMap.homographyMatrix;

    // Neck centre line in display pixels
    const nutPixel    = applyInverseHomography(H, { x: 0.5, y: 0 });
    const bridgePixel = applyInverseHomography(H, { x: 0.5, y: 1 });

    // Wrist (landmark 0) is the most stable point for neck position
    const wrist = landmarks[0];
    const wristPixel = { x: wrist.x * displayWidth, y: wrist.y * displayHeight };

    // Project wrist onto neck line
    const t = projectT(wristPixel, nutPixel, bridgePixel);

    // Point ON the line (for drawing the dot)
    const dx = bridgePixel.x - nutPixel.x;
    const dy = bridgePixel.y - nutPixel.y;
    const handOnLinePixel = {
      x: nutPixel.x + t * dx,
      y: nutPixel.y + t * dy,
    };

    // Normalised Y via homography → fret number
    const normPos    = applyHomography(H, wristPixel);
    const fretExact  = normYToFret(normPos.y, calibrationMap.anchorPoints);
    const fret       = Math.round(fretExact);

    return { fret, fretExact, t, nutPixel, bridgePixel, handOnLinePixel };
  }, [landmarks, calibrationMap, displayWidth, displayHeight]);
}
