/**
 * FretboardCanvas — draws tracked fretboard features and the derived quad.
 * Web-only. Positioned absoluteFill over the camera video.
 *
 * Draws:
 *   - All tracked feature points (coloured by found/lost)
 *   - Quadrilateral connecting the 4 derived corners
 *   - Corner circles with TL/TR/BR/BL labels
 *   - Debug text: tracked count and corner coords
 */

import React, { useEffect, useRef } from 'react';
import { Platform } from 'react-native';

export interface FretCorner {
  x: number;
  y: number;
  quality: number;
  found: boolean;
}

interface Props {
  /** All tracked feature points (small dots) */
  features: FretCorner[];
  /** Derived 4 corners: TL, TR, BR, BL (larger circles + quad lines) */
  corners: FretCorner[];
  width: number;
  height: number;
}

const CORNER_LABELS = ['TL', 'TR', 'BR', 'BL'];

export default function FretboardCanvas({ features, corners, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (Platform.OS !== 'web') return;
    const canvas = canvasRef.current;
    if (canvas == null) return;

    canvas.width  = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (ctx == null) return;
    ctx.clearRect(0, 0, width, height);

    const trackedCount = features.filter((f) => f.found).length;
    const totalCount   = features.length;
    const allCornersOk = corners.length === 4 && corners.every((c) => c.found);

    // ── Feature point dots ────────────────────────────────────────────────
    for (const f of features) {
      ctx.beginPath();
      ctx.arc(f.x, f.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = f.found ? 'rgba(52,211,153,0.8)' : 'rgba(239,68,68,0.5)';
      ctx.fill();
    }

    // ── Fretboard quad (corners derived from live H) ──────────────────────
    if (corners.length === 4) {
      const quadColor = allCornersOk
        ? 'rgba(52,211,153,0.7)'
        : 'rgba(251,191,36,0.6)';

      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      ctx.lineTo(corners[1].x, corners[1].y);
      ctx.lineTo(corners[2].x, corners[2].y);
      ctx.lineTo(corners[3].x, corners[3].y);
      ctx.closePath();
      ctx.strokeStyle = quadColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Corner circles + labels
      corners.forEach((c, i) => {
        ctx.beginPath();
        ctx.arc(c.x, c.y, 8, 0, Math.PI * 2);
        ctx.strokeStyle = allCornersOk ? '#34D399' : '#FBBF24';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.font = '10px monospace';
        ctx.fillStyle = allCornersOk ? '#34D399' : '#FBBF24';
        ctx.fillText(CORNER_LABELS[i], c.x + 10, c.y + 4);
      });
    }

    // ── Debug text ────────────────────────────────────────────────────────
    ctx.font = '11px monospace';
    ctx.fillStyle = 'rgba(251,191,36,0.9)';
    ctx.fillText(
      `tracking: ${trackedCount}/${totalCount} features`,
      8, height - 44,
    );
    if (corners.length === 4) {
      ctx.fillText(
        `TL(${Math.round(corners[0].x)},${Math.round(corners[0].y)})  TR(${Math.round(corners[1].x)},${Math.round(corners[1].y)})`,
        8, height - 30,
      );
      ctx.fillText(
        `BL(${Math.round(corners[3].x)},${Math.round(corners[3].y)})  BR(${Math.round(corners[2].x)},${Math.round(corners[2].y)})`,
        8, height - 16,
      );
    }
  }, [features, corners, width, height]);

  if (Platform.OS !== 'web') return null;

  return (
    <canvas
      ref={canvasRef as React.RefObject<HTMLCanvasElement>}
      style={{
        position: 'absolute', top: 0, left: 0,
        width: '100%', height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}
