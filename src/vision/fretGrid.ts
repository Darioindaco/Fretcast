/**
 * Fret grid computation — spec Section 3
 *
 * Logarithmic fret spacing applied AFTER homography in normalized space.
 * Non-negotiable order (spec Section 3):
 *   raw pixel → homography → normalized → logarithmic spacing → FretRegion grid
 *
 * Formula: fret_position = 1 - 1/2^(n/12)
 *   Fret 0  → 0.000 (nut)
 *   Fret 12 → 0.500 (octave, halfway)
 *   Fret 24 → 0.750
 */

import { FretRegion } from '../types';
import { applyInverseHomography } from './homography';

export const N_FRETS = 24;
export const N_STRINGS = 6;

/**
 * Standard tuning MIDI notes, string 0 = high E (E4=64), string 5 = low E (E2=40).
 */
export const STANDARD_TUNING_MIDI: number[] = [64, 59, 55, 50, 45, 40];

/**
 * Normalized fret Y position: 1 - 1/2^(fret/12), nut = 0.
 * For the visible fretboard region we scale to [0..1] where 1 = last visible fret.
 *
 * visibleFrets: how many frets are in view (e.g. 17 for standard guitar photo).
 * If unknown, pass 0 → raw logarithmic scale (12th fret = 0.5).
 */
export function fretNormalizedY(fret: number, visibleFrets: number): number {
  const raw = fret === 0 ? 0 : 1 - Math.pow(0.5, fret / 12);
  if (visibleFrets <= 0) return raw;
  const maxY = 1 - Math.pow(0.5, visibleFrets / 12);
  return raw / maxY;
}

/**
 * Compute FretRegion[N_FRETS][N_STRINGS] from homography matrix.
 *
 * @param homographyMatrix  3×3 H mapping pixel → normalized fretboard space
 * @param isLeftHanded      mirror string order when true
 * @param visibleFrets      highest fret visible in the camera frame (0 = use raw log scale)
 */
export function computeFretGrid(
  homographyMatrix: number[][],
  isLeftHanded: boolean,
  visibleFrets: number = 0
): FretRegion[][] {
  const grid: FretRegion[][] = [];

  for (let fret = 0; fret < N_FRETS; fret++) {
    grid[fret] = [];
    const normalizedY = fretNormalizedY(fret, visibleFrets);

    for (let string = 0; string < N_STRINGS; string++) {
      // String 0 = high E (left in standard), string 5 = low E (right)
      const stringNorm = string / (N_STRINGS - 1);
      const normalizedX = isLeftHanded ? 1 - stringNorm : stringNorm;

      // Pixel center via inverse homography
      const px = applyInverseHomography(homographyMatrix, { x: normalizedX, y: normalizedY });

      grid[fret][string] = {
        fret,
        string,
        normalizedX,
        normalizedY,
        radiusPx: fretAdaptiveRadius(fret),
        // pixelX and pixelY are derived on-screen, store in normalizedX/Y for portability
      } as FretRegion;

      // Suppress unused px — used for pixel-space overlay in Phase 6
      void px;
    }
  }

  return grid;
}

/**
 * Estimate visibleFrets from two known fret positions in normalized space.
 *
 * Given that fretA is at normalizedY_A and fretB is at normalizedY_B,
 * solve for the scale such that fretNormalizedY matches.
 *
 * Used to calibrate the grid from YIN anchor captures.
 */
/**
 * Estimate visibleFrets from one known fret position in normalized space.
 * Uses the reference fret's expected logarithmic position to derive the scale.
 */
export function estimateVisibleFrets(
  fretRef: number,
  normalizedYRef: number
): number {
  if (normalizedYRef <= 0) return 17; // fallback

  // rawRef = 1 - 1/2^(fretRef/12)
  const rawRef = fretRef === 0 ? 0 : 1 - Math.pow(0.5, fretRef / 12);

  // maxY = rawRef / normalizedYRef
  const maxY = rawRef / normalizedYRef;

  // maxY = 1 - 1/2^(visibleFrets/12) → visibleFrets = -12 * log2(1 - maxY)
  const clamped = Math.max(0.01, Math.min(0.99, maxY));
  const visibleFrets = -12 * Math.log2(1 - clamped);
  return Math.round(Math.max(12, Math.min(24, visibleFrets)));
}

/**
 * Expected frequency for (string, fret) in standard tuning.
 * string 0 = high E, string 5 = low E.
 */
export function expectedFrequency(string: number, fret: number): number {
  const midi = STANDARD_TUNING_MIDI[string] + fret;
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/**
 * All strings × expected frequency for a given fret.
 * Returns [string, freq] pairs sorted by frequency ascending.
 */
export function expectedFrequenciesForFret(fret: number): Array<{ string: number; freq: number }> {
  return STANDARD_TUNING_MIDI.map((_, s) => ({
    string: s,
    freq: expectedFrequency(s, fret),
  })).sort((a, b) => a.freq - b.freq);
}

/**
 * Fret-adaptive validation radius (spec Section 3).
 * Fret 3: 8px | Fret 5: 6px | Fret 7: 5px | Fret 9: 4px | Fret 12+: 3px
 */
export function fretAdaptiveRadius(fret: number): number {
  if (fret <= 3) return 8;
  if (fret <= 5) return 6;
  if (fret <= 7) return 5;
  if (fret <= 9) return 4;
  return 3;
}
