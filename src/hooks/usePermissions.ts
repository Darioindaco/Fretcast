import { useEffect, useState } from 'react';
import { useCameraPermission, useMicrophonePermission } from 'react-native-vision-camera';

export type PermissionStatus = 'checking' | 'granted' | 'denied';

/**
 * Requests camera + microphone permissions on mount.
 * Returns combined status used by HomeScreen and RecordingScreen.
 */
export function usePermissions() {
  const { hasPermission: hasCam, requestPermission: requestCam } = useCameraPermission();
  const { hasPermission: hasMic, requestPermission: requestMic } = useMicrophonePermission();
  const [status, setStatus] = useState<PermissionStatus>('checking');

  useEffect(() => {
    (async () => {
      let cam = hasCam;
      let mic = hasMic;

      if (!cam) cam = await requestCam();
      if (!mic) mic = await requestMic();

      setStatus(cam && mic ? 'granted' : 'denied');
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return status;
}
