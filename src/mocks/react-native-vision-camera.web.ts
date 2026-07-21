// Web stub for react-native-vision-camera
// On web, camera is handled by the browser getUserMedia API directly in screens
import { forwardRef } from 'react';

export const Camera = forwardRef((_props: Record<string, unknown>, _ref: unknown) => null);
Camera.displayName = 'Camera';

export function useCameraDevice(_position: string) { return { id: 'web', position: _position }; }
export function useCameraDevices() { return []; }
export function useCameraFormat(_device: unknown, _templates: unknown) { return undefined; }
export function useCameraPermission() {
  return { hasPermission: true, requestPermission: async () => true };
}
export function useMicrophonePermission() {
  return { hasPermission: true, requestPermission: async () => true };
}
export function useFrameProcessor(_fn: unknown, _deps: unknown) { return undefined; }
export type VideoFile = { path: string; duration: number };
export type CameraCaptureError = { code: string; message: string };
export type CameraDevice = { id: string; position: string };
