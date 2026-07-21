/**
 * Homography computation — spec Section 3
 *
 * DLT (Direct Linear Transformation) for 4-point correspondences.
 * Maps pixel coordinates → normalized [0,1]×[0,1] fretboard space.
 *
 * Order is non-negotiable (spec Section 3):
 *   raw pixel → homography → normalized → logarithmic fret spacing
 * Never apply logarithmic spacing before the homography step.
 */

export interface Point2D {
  x: number;
  y: number;
}

/**
 * Compute 3×3 homography matrix from 4 pixel → normalized correspondences.
 * src[i] maps to dst[i] for i in 0..3.
 *
 * Returns H such that [x', y', 1]^T ∝ H · [x, y, 1]^T
 */
export function computeHomography(
  src: [Point2D, Point2D, Point2D, Point2D],
  dst: [Point2D, Point2D, Point2D, Point2D]
): number[][] {
  // Build 8×8 system Ah = b (DLT)
  const A: number[][] = [];
  const b: number[] = [];

  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const { x: xp, y: yp } = dst[i];

    // Row for x'
    A.push([x, y, 1, 0, 0, 0, -xp * x, -xp * y]);
    b.push(xp);

    // Row for y'
    A.push([0, 0, 0, x, y, 1, -yp * x, -yp * y]);
    b.push(yp);
  }

  const h = gaussianElimination(A, b);

  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

/**
 * Apply homography: pixel → normalized.
 */
export function applyHomography(H: number[][], pt: Point2D): Point2D {
  const w = H[2][0] * pt.x + H[2][1] * pt.y + H[2][2];
  return {
    x: (H[0][0] * pt.x + H[0][1] * pt.y + H[0][2]) / w,
    y: (H[1][0] * pt.x + H[1][1] * pt.y + H[1][2]) / w,
  };
}

/**
 * Apply inverse homography: normalized → pixel.
 */
export function applyInverseHomography(H: number[][], pt: Point2D): Point2D {
  return applyHomography(invertMatrix3(H), pt);
}

// ── Private helpers ───────────────────────────────────────────────────────────

function gaussianElimination(A: number[][], b: number[]): number[] {
  const n = A.length;
  // Augmented matrix [A|b]
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > Math.abs(M[maxRow][col])) maxRow = row;
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];

    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-10) continue;

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / pivot;
      for (let k = col; k <= n; k++) {
        M[row][k] -= factor * M[col][k];
      }
    }
  }

  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

function invertMatrix3(m: number[][]): number[][] {
  const det =
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

  if (Math.abs(det) < 1e-10) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  return [
    [
      (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
      (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
      (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
    ],
    [
      (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
      (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
      (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
    ],
    [
      (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
      (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
      (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
    ],
  ];
}
