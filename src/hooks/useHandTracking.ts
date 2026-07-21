/**
 * useHandTracking — spec Section 7
 *
 * Loads MediaPipe HandLandmarker from CDN via an injected <script type="module">.
 * This bypasses Metro bundler completely — Metro cannot process MediaPipe's WASM.
 *
 * MediaPipe config (spec Section 7):
 *   numHands: 1 | confidence thresholds 0.7/0.7/0.6 | runningMode: VIDEO
 *
 * Returns landmarks for HandCanvas overlay + pushes VisionFrames to store.
 */

import { useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';
import { useFretcastStore } from '../store';
import { VisionFrame } from '../types';

export interface HandLandmark {
  x: number; // normalised 0..1
  y: number;
  z: number;
}

export interface HandTrackingResult {
  landmarks: HandLandmark[] | null;
  modelLoaded: boolean;
  loadError: string | null;
}

interface Options {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  enabled: boolean;
}

export const HAND_CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const FINGER_TIPS  = [8, 12, 16, 20] as const;
const FINGER_NAMES = ['index', 'middle', 'ring', 'pinky'] as const;

// ── CDN loader (runs once, caches on window) ─────────────────────────────────

const MP_READY_EVENT = 'fretcast:mp:loaded';
const MP_ERROR_EVENT = 'fretcast:mp:error';

function injectMediaPipeCDN(): void {
  const win = window as any;
  if (win.__fretcastMPInjected) return;
  win.__fretcastMPInjected = true;

  const script = document.createElement('script');
  script.type = 'module';
  // Top-level await is in modern browsers (Chrome 89+, Safari 15+, FF 89+)
  script.textContent = `
    try {
      const { HandLandmarker, FilesetResolver } =
        await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs');
      window.__fretcastMP = { HandLandmarker, FilesetResolver };
      window.dispatchEvent(new CustomEvent('${MP_READY_EVENT}'));
    } catch (e) {
      window.__fretcastMPErr = String(e);
      window.dispatchEvent(new CustomEvent('${MP_ERROR_EVENT}'));
    }
  `;
  document.head.appendChild(script);
}

async function waitForMediaPipe(): Promise<{ HandLandmarker: any; FilesetResolver: any }> {
  const win = window as any;
  if (win.__fretcastMP?.HandLandmarker) return win.__fretcastMP;

  return new Promise((resolve, reject) => {
    const onOk = () => {
      window.removeEventListener(MP_ERROR_EVENT, onErr);
      resolve((window as any).__fretcastMP);
    };
    const onErr = () => {
      window.removeEventListener(MP_READY_EVENT, onOk);
      reject(new Error((window as any).__fretcastMPErr ?? 'MediaPipe CDN load failed'));
    };
    window.addEventListener(MP_READY_EVENT, onOk, { once: true });
    window.addEventListener(MP_ERROR_EVENT, onErr, { once: true });
  });
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useHandTracking({ videoRef, enabled }: Options): HandTrackingResult {
  const pushVisionFrame = useFretcastStore((s) => s.pushVisionFrame);
  const [landmarks, setLandmarks] = useState<HandLandmark[] | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const landmarkerRef = useRef<any>(null);
  const rafRef        = useRef<number | null>(null);
  const cancelledRef  = useRef(false);

  useEffect(() => {
    if (!enabled || Platform.OS !== 'web') return;

    cancelledRef.current = false;

    async function init() {
      // 1. Inject CDN script (no-op if already done)
      injectMediaPipeCDN();

      // 2. Wait for MediaPipe to load (up to 30s)
      let mp: { HandLandmarker: any; FilesetResolver: any };
      try {
        mp = await waitForMediaPipe();
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error('[HandTracking] CDN load failed:', msg);
        setLoadError(msg);
        return;
      }

      if (cancelledRef.current) return;

      // 3. Poll until video element is ready (up to 10s)
      let video: HTMLVideoElement | null = null;
      for (let i = 0; i < 100; i++) {
        video = videoRef.current;
        if (video != null && video.readyState >= 1) break;
        await new Promise((r) => setTimeout(r, 100));
        if (cancelledRef.current) return;
      }
      if (video == null) {
        setLoadError('Video element not ready after 10s');
        return;
      }

      // 4. Create HandLandmarker
      try {
        const vision = await mp.FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
        );
        landmarkerRef.current = await mp.HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 1,
          minHandDetectionConfidence: 0.7,
          minHandPresenceConfidence: 0.7,
          minTrackingConfidence: 0.6,
        });
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error('[HandTracking] HandLandmarker init failed:', msg);
        setLoadError(msg);
        return;
      }

      if (cancelledRef.current) return;
      setModelLoaded(true);

      // 5. Detection loop
      const v = video;
      function detect() {
        if (cancelledRef.current) return;
        if (v.readyState >= 2 && landmarkerRef.current != null) {
          try {
            const results = landmarkerRef.current.detectForVideo(v, performance.now());
            const lm: HandLandmark[] | null =
              results.landmarks?.length > 0 ? results.landmarks[0] : null;

            setLandmarks(lm);

            const score: number = results.handednesses?.[0]?.[0]?.score ?? (lm ? 0.8 : 0);

            const frame: VisionFrame = {
              timestamp: Date.now(),
              fingerPositions: lm
                ? FINGER_TIPS.map((li, si) => ({
                    finger: FINGER_NAMES[si],
                    fret: null,
                    string: null,
                    pixelX: lm[li].x * (v.videoWidth  || v.clientWidth),
                    pixelY: lm[li].y * (v.videoHeight || v.clientHeight),
                    confidence: score,
                  }))
                : [],
              fingerVelocities: lm ? [0, 0, 0, 0] : [],
              frameConfidence: lm ? score : 0,
              handVisible: lm != null,
              anchorsStable: lm != null,
            };
            pushVisionFrame(frame);
          } catch {
            // Skip frame
          }
        }
        rafRef.current = requestAnimationFrame(detect);
      }
      rafRef.current = requestAnimationFrame(detect);
    }

    init();

    return () => {
      cancelledRef.current = true;
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      landmarkerRef.current?.close?.();
      landmarkerRef.current = null;
      setLandmarks(null);
      setModelLoaded(false);
      setLoadError(null);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  return { landmarks, modelLoaded, loadError };
}
