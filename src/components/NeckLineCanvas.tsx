/**
 * NeckLineCanvas — draws the nut→bridge neck centreline and hand position dot.
 * Web-only. Positioned absoluteFill over the camera video.
 */

import React, { useEffect, useRef } from 'react';
import { Platform } from 'react-native';

interface Props {
  nutPixel:        { x: number; y: number } | null;
  bridgePixel:     { x: number; y: number } | null;
  handOnLinePixel: { x: number; y: number } | null;
  fret:            number | null;
  width:           number;
  height:          number;
}

export default function NeckLineCanvas({
  nutPixel,
  bridgePixel,
  handOnLinePixel,
  fret,
  width,
  height,
}: Props) {
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

    if (nutPixel == null || bridgePixel == null) return;

    // ── Neck centre line ──────────────────────────────────────────────────
    ctx.beginPath();
    ctx.moveTo(nutPixel.x,    nutPixel.y);
    ctx.lineTo(bridgePixel.x, bridgePixel.y);
    ctx.strokeStyle = 'rgba(255,255,255,0.45)';
    ctx.lineWidth   = 2;
    ctx.setLineDash([6, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Nut endpoint
    ctx.beginPath();
    ctx.arc(nutPixel.x, nutPixel.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fill();

    // ── Hand dot ─────────────────────────────────────────────────────────
    if (handOnLinePixel != null) {
      // Outer glow ring
      ctx.beginPath();
      ctx.arc(handOnLinePixel.x, handOnLinePixel.y, 14, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(52,211,153,0.4)';
      ctx.lineWidth   = 3;
      ctx.stroke();

      // Filled dot
      ctx.beginPath();
      ctx.arc(handOnLinePixel.x, handOnLinePixel.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#34D399';
      ctx.fill();

      // Fret label next to dot
      if (fret != null) {
        ctx.font      = 'bold 14px monospace';
        ctx.fillStyle = '#34D399';
        ctx.fillText(`fret ${fret}`, handOnLinePixel.x + 18, handOnLinePixel.y + 5);
      }
    }
  }, [nutPixel, bridgePixel, handOnLinePixel, fret, width, height]);

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
