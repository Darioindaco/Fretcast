/**
 * guitarNeckDetector — Phase 4 AI tracking (YOLOv8n-pose)
 *
 * Loads a YOLOv8n-pose ONNX model that predicts the 4 exact corners of the
 * guitar fretboard. Returns them directly — no similarity-transform
 * approximation needed. Caller feeds the 4 corners straight into
 * computeHomography().
 *
 * Model spec (train with the Colab notebook):
 *   Architecture : YOLOv8n-pose, 1 class, 4 keypoints
 *   Keypoint order: 0=TL, 1=TR, 2=BR, 3=BL  (matches NORM_CORNERS)
 *   Input        : [1, 3, 640, 640] float32 CHW, normalised [0,1]
 *   Output       : [1, 17, 8400]
 *     rows 0-3   : bbox xc, yc, w, h  (in 640px input space)
 *     row  4     : detection confidence
 *     rows 5-7   : kp0 (TL) — x, y (640px space), visibility [0,1]
 *     rows 8-10  : kp1 (TR)
 *     rows 11-13 : kp2 (BR)
 *     rows 14-16 : kp3 (BL)
 *
 * Hosting: put best.onnx in the web/ folder → served at /guitar_neck.onnx
 */

export interface FretboardKeypoint {
  /** Normalised 0..1 in display-pixel space */
  x: number;
  y: number;
  /** Sigmoid visibility confidence [0,1]. Reject if < 0.5 */
  visibility: number;
}

export interface FretboardKeypoints {
  /** [TL, TR, BR, BL] — matches NORM_CORNERS = [(0,0),(1,0),(1,1),(0,1)] */
  corners: [FretboardKeypoint, FretboardKeypoint, FretboardKeypoint, FretboardKeypoint];
  /** Detection box confidence */
  confidence: number;
}

// ── CDN loader (same pattern as MediaPipe) ────────────────────────────────────

const ORT_READY_EVENT = 'fretcast:ort:loaded';
const ORT_ERROR_EVENT = 'fretcast:ort:error';

function injectONNXRuntime(): void {
  const win = window as any;
  if (win.__fretcastORTInjected) return;
  win.__fretcastORTInjected = true;

  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
  script.onload = () => {
    (window as any).ort.env.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    window.dispatchEvent(new CustomEvent(ORT_READY_EVENT));
  };
  script.onerror = () => {
    (window as any).__fretcastORTErr = 'ONNX Runtime CDN load failed';
    window.dispatchEvent(new CustomEvent(ORT_ERROR_EVENT));
  };
  document.head.appendChild(script);
}

function waitForORT(): Promise<any> {
  const win = window as any;
  if (win.ort?.InferenceSession) return Promise.resolve(win.ort);
  return new Promise((resolve, reject) => {
    const onOk = () => {
      window.removeEventListener(ORT_ERROR_EVENT, onErr);
      resolve((window as any).ort);
    };
    const onErr = () => {
      window.removeEventListener(ORT_READY_EVENT, onOk);
      reject(new Error((window as any).__fretcastORTErr ?? 'ONNX Runtime load failed'));
    };
    window.addEventListener(ORT_READY_EVENT, onOk,  { once: true });
    window.addEventListener(ORT_ERROR_EVENT, onErr, { once: true });
  });
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

const INPUT_SIZE = 640;

let _canvas: HTMLCanvasElement | null = null;
function getCanvas(): HTMLCanvasElement {
  if (!_canvas && typeof document !== 'undefined') _canvas = document.createElement('canvas');
  return _canvas!;
}

/** Resize video to 640×640 (stretch, no letterbox) → CHW float32 [0,1] */
function videoToTensor(video: HTMLVideoElement): Float32Array {
  const c = getCanvas();
  c.width  = INPUT_SIZE;
  c.height = INPUT_SIZE;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const stride = INPUT_SIZE * INPUT_SIZE;
  const t = new Float32Array(3 * stride);
  for (let i = 0; i < stride; i++) {
    t[i]              = data[i * 4]     / 255;
    t[stride     + i] = data[i * 4 + 1] / 255;
    t[2 * stride + i] = data[i * 4 + 2] / 255;
  }
  return t;
}

// ── Postprocessing ────────────────────────────────────────────────────────────

const CONF_THRESHOLD  = 0.35;
const MIN_VISIBILITY  = 0.5;   // reject keypoints below this

/**
 * Parse YOLOv8-pose output [1, 17, 8400] and return the top detection.
 * Converts coordinates from 640px input space → normalised display space.
 * Returns null if no detection passes CONF_THRESHOLD or any keypoint
 * visibility < MIN_VISIBILITY.
 */
function parsePoseOutput(
  output: Float32Array,
  videoW: number,
  videoH: number,
  displayW: number,
  displayH: number,
): FretboardKeypoints | null {
  const N = 8400;

  let bestConf = CONF_THRESHOLD;
  let bestIdx  = -1;
  for (let i = 0; i < N; i++) {
    const c = output[4 * N + i];
    if (c > bestConf) { bestConf = c; bestIdx = i; }
  }
  if (bestIdx < 0) return null;

  // objectFit: cover — map 640px model input → display pixels
  const scaleX     = displayW / videoW;
  const scaleY     = displayH / videoH;
  const coverScale = Math.max(scaleX, scaleY);
  const offX       = (displayW - videoW * coverScale) / 2;
  const offY       = (displayH - videoH * coverScale) / 2;

  // Since we stretched video to 640×640 (no letterbox), model coords are in
  // [0,640] corresponding to [0, videoW] / [0, videoH] video space.
  const toDisplayX = (px640: number) =>
    (px640 / INPUT_SIZE) * videoW * coverScale + offX;
  const toDisplayY = (py640: number) =>
    (py640 / INPUT_SIZE) * videoH * coverScale + offY;

  // Rows 5-16: kp0..kp3, each taking 3 consecutive rows (x, y, vis)
  const rawCorners = [0, 1, 2, 3].map((k) => {
    const rowBase = 5 + k * 3;
    return {
      x:          toDisplayX(output[rowBase       * N + bestIdx]) / displayW,
      y:          toDisplayY(output[(rowBase + 1) * N + bestIdx]) / displayH,
      visibility: output[(rowBase + 2) * N + bestIdx],
    };
  });

  // Reject if any keypoint is unreliable
  if (rawCorners.some((kp) => kp.visibility < MIN_VISIBILITY)) return null;

  return {
    corners:    rawCorners as FretboardKeypoints['corners'],
    confidence: bestConf,
  };
}

// ── Session cache ─────────────────────────────────────────────────────────────

let _sessionPromise: Promise<any> | null = null;

/**
 * Load (or return cached) ONNX inference session.
 * modelUrl: e.g. "/guitar_neck.onnx" (put best.onnx in Expo's web/ folder)
 */
export function loadGuitarNeckModel(modelUrl: string): Promise<any> {
  if (_sessionPromise) return _sessionPromise;
  _sessionPromise = (async () => {
    injectONNXRuntime();
    const ort = await waitForORT();
    return ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgl', 'wasm'],
    });
  })();
  return _sessionPromise;
}

/**
 * Run one detection pass and return the 4 fretboard corner keypoints
 * in normalised display space, or null if nothing passes confidence/visibility
 * thresholds.
 */
export async function runFretboardDetection(
  session: any,
  video: HTMLVideoElement,
  displayW: number,
  displayH: number,
): Promise<FretboardKeypoints | null> {
  const ort = (window as any).ort;
  if (!ort) return null;

  const tensorData = videoToTensor(video);
  const input      = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const results    = await session.run({ images: input });

  const output = results['output0']?.data as Float32Array | undefined;
  if (!output) return null;

  return parsePoseOutput(
    output,
    video.videoWidth  || displayW,
    video.videoHeight || displayH,
    displayW,
    displayH,
  );
}
