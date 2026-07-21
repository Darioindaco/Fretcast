/**
 * Lucas-Kanade iterative optical flow — Phase 4
 *
 * Tracks sparse feature points from one frame to the next.
 * Single-level implementation with bilinear interpolation.
 * Adequate for <15px inter-frame motion (normal guitar/camera movement at 30fps).
 *
 * Algorithm:
 *   For each tracked point (x, y):
 *     Assume optical flow is locally constant in a window around (x, y).
 *     Minimise Σ (Ix·u + Iy·v + It)² over the window.
 *     Solution: [u,v]ᵀ = -(AᵀA)⁻¹ Aᵀb, refined iteratively.
 *   Quality = minimum eigenvalue of AᵀA — low = untextured = unreliable.
 */

export interface GrayscaleFrame {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
}

export interface TrackedPoint {
  x: number;
  y: number;
  /** Min eigenvalue of gradient matrix. Higher = more reliable. */
  quality: number;
  /** false if quality too low or displacement exceeded maxDrift */
  found: boolean;
}

// ── Conversion ────────────────────────────────────────────────────────────────

/** Convert RGBA ImageData to single-channel grayscale (BT.601) */
export function toGrayscale(imageData: ImageData): GrayscaleFrame {
  const { width, height, data } = imageData;
  const gray = new Uint8ClampedArray(width * height);
  for (let i = 0; i < width * height; i++) {
    gray[i] = (data[i * 4] * 77 + data[i * 4 + 1] * 150 + data[i * 4 + 2] * 29) >> 8;
  }
  return { width, height, data: gray };
}

// ── Sampling ──────────────────────────────────────────────────────────────────

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** Bilinear interpolation at fractional (x, y) — clamps to frame boundaries */
function blerp(f: GrayscaleFrame, x: number, y: number): number {
  const x0 = clamp(Math.floor(x), 0, f.width - 1);
  const y0 = clamp(Math.floor(y), 0, f.height - 1);
  const x1 = Math.min(x0 + 1, f.width - 1);
  const y1 = Math.min(y0 + 1, f.height - 1);
  const fx = x - x0;
  const fy = y - y0;
  const w = f.width;
  return (
    f.data[y0 * w + x0] * (1 - fx) * (1 - fy) +
    f.data[y0 * w + x1] *      fx  * (1 - fy) +
    f.data[y1 * w + x0] * (1 - fx) *      fy  +
    f.data[y1 * w + x1] *      fx  *      fy
  );
}

// ── Core LK for one point ─────────────────────────────────────────────────────

function trackOnePoint(
  prev: GrayscaleFrame,
  curr: GrayscaleFrame,
  x: number,
  y: number,
  winR: number,
  maxIter: number,
  eps: number,
): { dx: number; dy: number; quality: number } {
  const W = 2 * winR + 1;
  const N = W * W;
  const Ix = new Float32Array(N);
  const Iy = new Float32Array(N);
  let a11 = 0, a12 = 0, a22 = 0;

  // Build gradient matrix AᵀA and precompute spatial gradients from prev frame
  let k = 0;
  for (let dy = -winR; dy <= winR; dy++) {
    for (let dx = -winR; dx <= winR; dx++) {
      const px = clamp(x + dx, 1, prev.width - 2);
      const py = clamp(y + dy, 1, prev.height - 2);
      const gx = (blerp(prev, px + 1, py) - blerp(prev, px - 1, py)) * 0.5;
      const gy = (blerp(prev, px, py + 1) - blerp(prev, px, py - 1)) * 0.5;
      Ix[k] = gx;
      Iy[k] = gy;
      a11 += gx * gx;
      a12 += gx * gy;
      a22 += gy * gy;
      k++;
    }
  }

  const det = a11 * a22 - a12 * a12;
  const trace = a11 + a22;
  // Min eigenvalue of 2×2 symmetric matrix
  const quality = (trace - Math.sqrt(Math.max(0, trace * trace - 4 * det))) * 0.5;

  if (quality < 1e-4) return { dx: 0, dy: 0, quality };

  // Iteratively solve for displacement (vx, vy)
  let vx = 0;
  let vy = 0;

  for (let iter = 0; iter < maxIter; iter++) {
    let b1 = 0;
    let b2 = 0;
    k = 0;

    for (let dy = -winR; dy <= winR; dy++) {
      for (let dx = -winR; dx <= winR; dx++) {
        const px = clamp(x + dx, 1, prev.width - 2);
        const py = clamp(y + dy, 1, prev.height - 2);
        const cx = clamp(px + vx, 0, curr.width - 1);
        const cy = clamp(py + vy, 0, curr.height - 1);
        // Temporal gradient at current estimate
        const It = blerp(curr, cx, cy) - blerp(prev, px, py);
        b1 += Ix[k] * It;
        b2 += Iy[k] * It;
        k++;
      }
    }

    // [u,v]ᵀ += -(AᵀA)⁻¹ [b1,b2]ᵀ
    const d = det + 1e-9;
    const ddx = (a22 * (-b1) - a12 * (-b2)) / d;
    const ddy = (a11 * (-b2) - a12 * (-b1)) / d;
    vx += ddx;
    vy += ddy;

    if (ddx * ddx + ddy * ddy < eps * eps) break;
  }

  return { dx: vx, dy: vy, quality };
}

// ── Shi-Tomasi feature detection ─────────────────────────────────────────────

export interface DetectedFeature {
  x: number;
  y: number;
  quality: number;
}

/**
 * Detect good features to track (Shi-Tomasi) within a frame.
 *
 * @param frame       Grayscale frame to search
 * @param inRegion    Optional predicate — if provided, only pixels where this returns
 *                    true are considered. Use to restrict detection to the fretboard area.
 * @param maxPoints   Maximum number of features to return
 * @param minQuality  Minimum min-eigenvalue threshold (lower = more features)
 * @param minDist     Minimum pixel distance between returned features (grid-based)
 */
export function detectShiTomasi(
  frame: GrayscaleFrame,
  inRegion: ((x: number, y: number) => boolean) | null,
  maxPoints: number,
  minQuality: number,
  minDist: number,
): DetectedFeature[] {
  const { width, height, data } = frame;
  const W = 3; // gradient window half-radius
  const qualMap = new Float32Array(width * height);

  // Compute min-eigenvalue map
  for (let y = W + 1; y < height - W - 1; y++) {
    for (let x = W + 1; x < width - W - 1; x++) {
      if (inRegion != null && !inRegion(x, y)) continue;

      let a11 = 0, a12 = 0, a22 = 0;
      for (let dy = -W; dy <= W; dy++) {
        for (let dx = -W; dx <= W; dx++) {
          const px = x + dx;
          const py = y + dy;
          const idx = py * width + px;
          const gx = (data[idx + 1] - data[idx - 1]) * 0.5;
          const gy = (data[idx + width] - data[idx - width]) * 0.5;
          a11 += gx * gx;
          a12 += gx * gy;
          a22 += gy * gy;
        }
      }
      const trace = a11 + a22;
      const det   = a11 * a22 - a12 * a12;
      qualMap[y * width + x] = (trace - Math.sqrt(Math.max(0, trace * trace - 4 * det))) * 0.5;
    }
  }

  // Grid-based selection: best point per cell — O(N) and ensures spatial spread
  const cellSize = Math.max(1, Math.floor(minDist * 0.8));
  const cols = Math.ceil(width  / cellSize);
  const rows = Math.ceil(height / cellSize);
  const cellBest = new Float32Array(cols * rows);
  const cellPt: (DetectedFeature | null)[] = new Array(cols * rows).fill(null);

  for (let y = W + 1; y < height - W - 1; y++) {
    const row = Math.floor(y / cellSize);
    for (let x = W + 1; x < width - W - 1; x++) {
      const q = qualMap[y * width + x];
      if (q < minQuality) continue;
      const ci = row * cols + Math.floor(x / cellSize);
      if (q > cellBest[ci]) {
        cellBest[ci] = q;
        cellPt[ci] = { x, y, quality: q };
      }
    }
  }

  return (cellPt.filter(Boolean) as DetectedFeature[])
    .sort((a, b) => b.quality - a.quality)
    .slice(0, maxPoints);
}

// ── Public API ────────────────────────────────────────────────────────────────

export interface LKOptions {
  /** Half-window radius. Default 8 → 17×17 window. */
  winRadius?: number;
  /** Max iterations per point. Default 30. */
  maxIter?: number;
  /** Convergence threshold (pixels). Default 0.03. */
  eps?: number;
  /** Min eigenvalue to consider point trackable. Default 50. */
  minQuality?: number;
  /** Max pixel displacement allowed per frame. Default 40. */
  maxDrift?: number;
}

/**
 * Track `points` from `prev` to `curr`.
 * Returns new positions with quality scores.
 */
export function lkTrack(
  prev: GrayscaleFrame,
  curr: GrayscaleFrame,
  points: ReadonlyArray<{ x: number; y: number }>,
  opts: LKOptions = {},
): TrackedPoint[] {
  const {
    winRadius = 8,
    maxIter   = 30,
    eps       = 0.03,
    minQuality = 50,
    maxDrift  = 40,
  } = opts;

  return points.map(({ x, y }) => {
    const { dx, dy, quality } = trackOnePoint(prev, curr, x, y, winRadius, maxIter, eps);
    const found = quality >= minQuality && (dx * dx + dy * dy) <= maxDrift * maxDrift;
    return { x: x + dx, y: y + dy, quality, found };
  });
}
