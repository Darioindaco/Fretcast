/**
 * useFretboardTracking — Phase 4 (LK optical flow + hand masking)
 *
 * Tracks the calibrated fretboard using Lucas-Kanade optical flow.
 * MediaPipe hand landmarks are used to mask out the hand so occlusion
 * doesn't break tracking.
 *
 * Per frame:
 *   1. Capture grayscale video frame at display resolution
 *   2. LK track all stored feature points from prev → curr frame
 *   3. Discard points that drifted into the hand polygon
 *   4. If < MIN_FEATURES remain, top up with fresh Shi-Tomasi detections
 *   5. Pick 4 well-spread points (one per fretboard quadrant)
 *   6. computeHomography(currentDisplayPositions, storedNormalisedPositions)
 *   7. Push to store as liveHomography
 *
 * No model training required — everything reuses existing opticalFlow.ts
 * and homography.ts code.
 */

import { RefObject, useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import { useFretcastStore } from '../store';
import {
  Point2D,
  applyHomography,
  applyInverseHomography,
  computeHomography,
} from '../vision/homography';
import { FretCorner } from '../components/FretboardCanvas';
import { HandLandmark } from './useHandTracking';
import { toGrayscale, lkTrack, detectShiTomasi, GrayscaleFrame } from '../vision/opticalFlow';

const NORM_CORNERS: [Point2D, Point2D, Point2D, Point2D] = [
  { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 },
];

const MIN_FEATURES = 6;   // re-detect if fewer good points survive
const MAX_FEATURES = 24;  // max features to maintain

// ── Geometry helpers ──────────────────────────────────────────────────────────

/** Ray-casting point-in-polygon test */
function inPolygon(x: number, y: number, poly: Point2D[]): boolean {
  let inside = false;
  const n = poly.length;
  let j = n - 1;
  for (let i = 0; i < n; i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}

/**
 * Rough hand outline from MediaPipe 21 landmarks.
 * Traces wrist → thumb → fingertips → pinky base → back to wrist.
 */
function handOutline(lm: HandLandmark[], w: number, h: number): Point2D[] {
  const idx = [0, 1, 2, 3, 4, 8, 12, 16, 20, 19, 18, 17, 13, 9, 5, 0];
  return idx.map((i) => ({ x: lm[i].x * w, y: lm[i].y * h }));
}

// ── Video capture ─────────────────────────────────────────────────────────────

let _capCanvas: HTMLCanvasElement | null = null;

function captureGray(video: HTMLVideoElement, w: number, h: number): GrayscaleFrame | null {
  if (!_capCanvas) _capCanvas = document.createElement('canvas');
  if (_capCanvas.width !== w || _capCanvas.height !== h) {
    _capCanvas.width  = w;
    _capCanvas.height = h;
  }
  const ctx = _capCanvas.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, w, h);
  return toGrayscale(ctx.getImageData(0, 0, w, h));
}

// ── Feature selection ─────────────────────────────────────────────────────────

/**
 * Pick the 4 tracked points that best cover the corners of the feature
 * bounding box in normalised fretboard space.
 *
 * No hand-masking here — that was the cause of instant tracking loss
 * (the hand covers most visible frets so all candidates got excluded).
 * Hand polygon is only used for the visual overlay.
 *
 * Requires MIN_SPREAD in both axes to avoid a degenerate homography.
 */
const MIN_SPREAD = 0.12; // minimum normalised spread in x and y

function pick4Corners(
  displayPts: Point2D[],
  normPts:    Point2D[],
): { src: [Point2D, Point2D, Point2D, Point2D]; dst: [Point2D, Point2D, Point2D, Point2D] } | null {
  if (displayPts.length < 4) return null;

  const xs = normPts.map((p) => p.x);
  const ys = normPts.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  if (maxX - minX < MIN_SPREAD || maxY - minY < MIN_SPREAD) return null;

  // For each corner of the bounding box, pick the closest feature
  const targets = [
    { tx: minX, ty: minY },
    { tx: maxX, ty: minY },
    { tx: maxX, ty: maxY },
    { tx: minX, ty: maxY },
  ];

  const used     = new Set<number>();
  const selected: { dp: Point2D; np: Point2D }[] = [];

  for (const t of targets) {
    let bestD = Infinity, bestI = -1;
    for (let i = 0; i < normPts.length; i++) {
      if (used.has(i)) continue;
      const d = (normPts[i].x - t.tx) ** 2 + (normPts[i].y - t.ty) ** 2;
      if (d < bestD) { bestD = d; bestI = i; }
    }
    if (bestI < 0) return null;
    used.add(bestI);
    selected.push({ dp: displayPts[bestI], np: normPts[bestI] });
  }

  return {
    src: selected.map((s) => s.dp) as [Point2D, Point2D, Point2D, Point2D],
    dst: selected.map((s) => s.np) as [Point2D, Point2D, Point2D, Point2D],
  };
}

// ── Hook ──────────────────────────────────────────────────────────────────────

interface Options {
  videoRef: RefObject<HTMLVideoElement | null>;
  landmarks: HandLandmark[] | null;
  displayWidth: number;
  displayHeight: number;
  enabled: boolean;
}

export interface FretboardTrackingResult {
  isTracking:    boolean;
  corners:       FretCorner[];
  features:      FretCorner[];
  trackedCount:  number;
  totalFeatures: number;
  modelLoaded:   boolean;   // always true — no model needed
  loadError:     string | null;
}

export function useFretboardTracking({
  videoRef,
  landmarks,
  displayWidth,
  displayHeight,
  enabled,
}: Options): FretboardTrackingResult {
  const calibrationMap    = useFretcastStore((s) => s.calibrationMap);
  const setLiveHomography = useFretcastStore((s) => s.setLiveHomography);

  const [isTracking, setIsTracking] = useState(false);
  const [corners,    setCorners]    = useState<FretCorner[]>([]);
  const [features,   setFeatures]   = useState<FretCorner[]>([]);

  // Mutable tracking state
  const prevFrameRef   = useRef<GrayscaleFrame | null>(null);
  const displayPtsRef  = useRef<Point2D[]>([]);   // current positions in display pixels
  const normPtsRef     = useRef<Point2D[]>([]);   // fixed normalised fretboard coords
  const liveHRef       = useRef<number[][] | null>(null);
  const rafRef         = useRef<number | null>(null);
  const cancelRef      = useRef(false);
  const landmarksRef   = useRef<HandLandmark[] | null>(null);
  const frameCountRef  = useRef(0);

  // Keep landmarks available inside the rAF loop without restarting it
  useEffect(() => { landmarksRef.current = landmarks; }, [landmarks]);

  // ── Reset on calibration change ──────────────────────────────────────────
  useEffect(() => {
    prevFrameRef.current  = null;
    displayPtsRef.current = [];
    normPtsRef.current    = [];

    if (calibrationMap == null) {
      liveHRef.current = null;
      setLiveHomography(null);
      return;
    }
    liveHRef.current = calibrationMap.homographyMatrix;
    setLiveHomography(calibrationMap.homographyMatrix);
  }, [calibrationMap, setLiveHomography]);

  // ── rAF loop ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!enabled || Platform.OS !== 'web') return;
    if (calibrationMap == null) return;
    if (displayWidth <= 1 || displayHeight <= 1) return;

    cancelRef.current = false;

    function loop() {
      if (cancelRef.current) return;

      const video = videoRef.current;
      if (video == null || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      const H = liveHRef.current ?? calibrationMap!.homographyMatrix;

      const currFrame = captureGray(video, displayWidth, displayHeight);
      if (currFrame == null) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      // ── Initial detection (first frame or after full loss) ──────────────
      if (prevFrameRef.current == null) {
        const feats = detectShiTomasi(
          currFrame,
          (x, y) => {
            const n = applyHomography(H, { x, y });
            return n.x >= 0.02 && n.x <= 0.98 && n.y >= 0.02 && n.y <= 0.98;
          },
          MAX_FEATURES, 8, 10,  // low minQuality — fretboard may have subtle texture
        );
        displayPtsRef.current = feats.map((f) => ({ x: f.x, y: f.y }));
        normPtsRef.current    = feats.map((f) => applyHomography(H, { x: f.x, y: f.y }));
        prevFrameRef.current  = currFrame;
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      // ── LK track ─────────────────────────────────────────────────────────
      const tracked = lkTrack(prevFrameRef.current, currFrame, displayPtsRef.current, {
        winRadius:   5,
        maxIter:    15,
        eps:        0.05,
        minQuality:  1,   // very low — let maxDrift be the gate
        maxDrift:   25,
      });
      prevFrameRef.current = currFrame;

      // Retain only well-tracked points
      const keepDp:   Point2D[] = [];
      const keepNp:   Point2D[] = [];
      const keepQual: number[]  = [];
      for (let i = 0; i < tracked.length; i++) {
        if (tracked[i].found) {
          keepDp.push({ x: tracked[i].x, y: tracked[i].y });
          keepNp.push(normPtsRef.current[i]);
          keepQual.push(tracked[i].quality);
        }
      }

      // ── Top-up when features drop below threshold ─────────────────────
      if (keepDp.length < MIN_FEATURES) {
        const H_now = liveHRef.current ?? calibrationMap!.homographyMatrix;
        const existing = new Set(keepDp.map((p) => `${Math.round(p.x)},${Math.round(p.y)}`));

        const newFeats = detectShiTomasi(
          currFrame,
          (x, y) => {
            if (existing.has(`${Math.round(x)},${Math.round(y)}`)) return false;
            const n = applyHomography(H_now, { x, y });
            return n.x >= 0.02 && n.x <= 0.98 && n.y >= 0.02 && n.y <= 0.98;
          },
          MAX_FEATURES - keepDp.length, 8, 10,
        );
        for (const f of newFeats) {
          keepDp.push({ x: f.x, y: f.y });
          keepNp.push(applyHomography(H_now, { x: f.x, y: f.y }));
          keepQual.push(f.quality);
        }
      }

      displayPtsRef.current = keepDp;
      normPtsRef.current    = keepNp;

      // ── Pick 4 spread points → homography ────────────────────────────────
      // Hand polygon is only used to colour-code the debug overlay, not to
      // exclude points from selection (exclusion caused instant tracking loss).
      const lm    = landmarksRef.current;
      const hPoly = lm && lm.length >= 21 ? handOutline(lm, displayWidth, displayHeight) : [];

      const picked = pick4Corners(keepDp, keepNp);

      if (picked != null) {
        try {
          const newH = computeHomography(picked.src, picked.dst);
          liveHRef.current = newH;
          setLiveHomography(newH);

          // Throttle React state updates to ~10 fps
          frameCountRef.current++;
          if (frameCountRef.current % 3 === 0) {
            const c4: FretCorner[] = NORM_CORNERS.map((nc) => {
              const px = applyInverseHomography(newH, nc);
              return { x: px.x, y: px.y, quality: 999, found: true };
            });
            setCorners(c4);
            setIsTracking(true);
            setFeatures(keepDp.map((p, i) => ({
              x: p.x, y: p.y,
              quality: keepQual[i],
              // Green = on fretboard, orange = under hand (for debug)
              found: hPoly.length === 0 || !inPolygon(p.x, p.y, hPoly),
            })));
          }
        } catch {
          // Degenerate quad — keep last homography
        }
      } else {
        setIsTracking(false);
      }

      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      cancelRef.current = true;
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      prevFrameRef.current  = null;
      displayPtsRef.current = [];
      normPtsRef.current    = [];
    };
  }, [enabled, calibrationMap, displayWidth, displayHeight, setLiveHomography, videoRef]);

  return {
    isTracking,
    corners,
    features,
    trackedCount:  features.filter((f) => f.found).length,
    totalFeatures: features.length,
    modelLoaded:   true,
    loadError:     null,
  };
}
