/**
 * HandCanvas — draws MediaPipe hand landmarks on a transparent canvas overlay.
 * Web-only. Positioned absoluteFill over the camera video.
 *
 * Draws:
 *   - Skeleton connections (white lines)
 *   - Finger tip dots (blue — index, middle, ring, pinky)
 *   - Wrist dot (white)
 */

import React, { useEffect, useRef } from 'react';
import { Platform } from 'react-native';
import { HandLandmark, HAND_CONNECTIONS } from '../hooks/useHandTracking';

// Finger tip indices for colour coding
const TIP_INDICES = new Set([4, 8, 12, 16, 20]);

interface Props {
  landmarks: HandLandmark[] | null;
  /** Visible dimensions of the video element (CSS px) */
  width: number;
  height: number;
}

export default function HandCanvas({ landmarks, width, height }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (Platform.OS !== 'web') return;
    const canvas = canvasRef.current;
    if (canvas == null) return;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (ctx == null) return;

    ctx.clearRect(0, 0, width, height);

    if (landmarks == null || landmarks.length === 0) return;

    // Scale normalised coords to canvas size
    const sx = (lm: HandLandmark) => lm.x * width;
    const sy = (lm: HandLandmark) => lm.y * height;

    // Draw skeleton connections
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      if (landmarks[a] == null || landmarks[b] == null) continue;
      ctx.beginPath();
      ctx.moveTo(sx(landmarks[a]), sy(landmarks[a]));
      ctx.lineTo(sx(landmarks[b]), sy(landmarks[b]));
      ctx.stroke();
    }

    // Draw landmark dots
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      const isTip = TIP_INDICES.has(i);
      const radius = isTip ? 6 : 3;

      ctx.beginPath();
      ctx.arc(sx(lm), sy(lm), radius, 0, 2 * Math.PI);

      if (i === 0) {
        // Wrist
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
      } else if (isTip) {
        // Finger tips — colour by finger
        const tipColors = ['#FBBF24', '#60A5FA', '#34D399', '#F472B6', '#A78BFA'];
        const tipIdx = [4, 8, 12, 16, 20].indexOf(i);
        ctx.fillStyle = tipColors[tipIdx] ?? '#fff';
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.55)';
      }

      ctx.fill();
    }
  }, [landmarks, width, height]);

  if (Platform.OS !== 'web') return null;

  return (
    <canvas
      ref={canvasRef as React.RefObject<HTMLCanvasElement>}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
}
