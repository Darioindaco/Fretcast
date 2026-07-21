import React, { useEffect, useRef, useState } from 'react';
import { Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useIsFocused } from '@react-navigation/native';
import { useFretcastStore } from '../store';
import { RootStackParamList } from '../types';
import { useAudioPipeline } from '../hooks/useAudioPipeline';
import { useHandTracking } from '../hooks/useHandTracking';
import { useFretboardTracking } from '../hooks/useFretboardTracking';
import { useFretPosition } from '../hooks/useFretPosition';
import HandCanvas from '../components/HandCanvas';
import FretboardCanvas from '../components/FretboardCanvas';
import NeckLineCanvas from '../components/NeckLineCanvas';
import { audioEngine } from '../audio/AudioEngine';

// Native-only import — not used on web
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
} from 'react-native-vision-camera';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Home'>;
};

/**
 * Spec Section 11 — HOME
 * Full screen camera preview. Always warm.
 * Live chord detection shown without needing to press record.
 * Calibration banner only shown once Phase 3 is built (hidden for now).
 */
export default function HomeScreen({ navigation }: Props) {
  const liveChord = useFretcastStore((s) => s.liveChord);
  const setAudioMode = useFretcastStore((s) => s.setAudioMode);
  const isCalibrated = useFretcastStore((s) => s.isCalibrated);
  const calibrationMap = useFretcastStore((s) => s.calibrationMap);

  const isFocused = useIsFocused();

  const webVideoRef = useRef<HTMLVideoElement | null>(null);
  const webStreamRef = useRef<MediaStream | null>(null);
  const [streamReady, setStreamReady] = useState(false);
  const sessionStartMs = useRef(Date.now());
  const [videoSize, setVideoSize] = useState({ width: 1, height: 1 });

  // Native camera
  const device = useCameraDevice('back');
  const format = useCameraFormat(device, [
    { videoResolution: { width: 1920, height: 1080 } },
    { fps: 30 },
  ]);

  // Web: start camera + mic stream when screen is focused
  useEffect(() => {
    if (Platform.OS !== 'web' || !isFocused) return;

    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: true,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        webStreamRef.current = stream;
        if (webVideoRef.current != null) webVideoRef.current.srcObject = stream;
        setStreamReady(true);
      } catch {
        setStreamReady(true);
      }
    })();

    return () => {
      cancelled = true;
      webStreamRef.current?.getTracks().forEach((t) => t.stop());
      webStreamRef.current = null;
    };
  }, [isFocused]);

  // Spec Section 6: HPS active on Home for live preview — pause when not focused
  // (CalibrationScreen needs exclusive mic access for YIN)
  useEffect(() => {
    if (!isFocused) {
      audioEngine.setMode('IDLE');
      audioEngine.stop();
      setStreamReady(false);
      return;
    }
    setAudioMode('HPS');
    return () => setAudioMode('IDLE');
  }, [isFocused, setAudioMode]);

  // Live audio detection pipeline
  useAudioPipeline({
    mode: 'HPS',
    stream: streamReady ? webStreamRef.current : null,
    sessionStartMs: sessionStartMs.current,
  });

  // Hand tracking overlay (spec Section 7)
  const { landmarks, modelLoaded, loadError } = useHandTracking({
    videoRef: webVideoRef as React.RefObject<HTMLVideoElement | null>,
    enabled: Platform.OS === 'web' && streamReady,
  });

  // Phase 4: fretboard tracking — LK optical flow + hand masking
  const { isTracking, trackedCount, totalFeatures, corners, features } = useFretboardTracking({
    videoRef: webVideoRef as React.RefObject<HTMLVideoElement | null>,
    landmarks,
    displayWidth: videoSize.width,
    displayHeight: videoSize.height,
    enabled: Platform.OS === 'web' && streamReady && isCalibrated && calibrationMap != null,
  });

  // Fret position from wrist + calibration homography
  const { fret, nutPixel, bridgePixel, handOnLinePixel } = useFretPosition(
    landmarks,
    calibrationMap,
    videoSize.width,
    videoSize.height,
  );

  return (
    <View
      style={styles.container}
      onLayout={(e) => setVideoSize({ width: e.nativeEvent.layout.width, height: e.nativeEvent.layout.height })}
    >
      {Platform.OS === 'web' ? (
        <video
          ref={webVideoRef as React.RefObject<HTMLVideoElement>}
          autoPlay
          playsInline
          muted
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%', height: '100%', objectFit: 'cover',
          }}
        />
      ) : (
        device != null && (
          <Camera
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={true}
            video={false}
            audio={false}
            format={format}
          />
        )
      )}

      {/* Hand tracking overlay */}
      <HandCanvas landmarks={landmarks} width={videoSize.width} height={videoSize.height} />

      {/* Neck centre line overlay — only shown when calibrated */}
      {isCalibrated && (
        <NeckLineCanvas
          nutPixel={nutPixel}
          bridgePixel={bridgePixel}
          handOnLinePixel={handOnLinePixel}
          fret={fret}
          width={videoSize.width}
          height={videoSize.height}
        />
      )}

      {/* Fretboard tracking overlay — quad + all feature points */}
      {isCalibrated && (
        <FretboardCanvas
          features={features}
          corners={corners}
          width={videoSize.width}
          height={videoSize.height}
        />
      )}

      {/* Status badges */}
      {Platform.OS === 'web' && streamReady && (
        <View style={styles.badgeStack} pointerEvents="none">
          {/* Hand tracking */}
          {loadError != null
            ? <Text style={[styles.badgeText, { color: '#F87171' }]}>Hand: {loadError.slice(0, 50)}</Text>
            : !modelLoaded
              ? <Text style={styles.badgeText}>Hand tracking loading…</Text>
              : <Text style={[styles.badgeText, { color: '#34D399' }]}>Hand tracking active</Text>
          }
          {/* Fretboard tracking (only shown when calibrated) */}
          {isCalibrated && (
            isTracking
              ? <Text style={[styles.badgeText, { color: '#34D399' }]}>
                  Fretboard: {trackedCount}/{totalFeatures} pts
                </Text>
              : <Text style={[styles.badgeText, { color: '#F87171' }]}>
                  Fretboard lost — show guitar
                </Text>
          )}
        </View>
      )}

      {/* Live chord display */}
      {liveChord != null && (
        <View style={styles.chordOverlay}>
          <Text style={styles.chordText}>
            {liveChord.name}
            {liveChord.confidence < 0.75 ? '?' : ''}
          </Text>
          <Text style={styles.confidenceText}>
            {Math.round(liveChord.confidence * 100)}%
          </Text>
          {fret != null && (
            <Text style={styles.fretText}>fret {fret}</Text>
          )}
        </View>
      )}

      {/* Record button + calibrate */}
      <View style={styles.bottomBar}>
        <TouchableOpacity
          style={styles.calibrateButton}
          onPress={() => navigation.navigate('Calibration')}
          activeOpacity={0.8}
        >
          <Text style={styles.calibrateText}>{isCalibrated ? 'Recalibrate' : 'Calibrate'}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.recordButton}
          onPress={() => navigation.navigate('Recording')}
          activeOpacity={0.8}
        >
          <View style={styles.recordButtonInner} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  chordOverlay: {
    position: 'absolute',
    top: 60,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  chordText: {
    color: '#fff',
    fontSize: 72,
    fontWeight: '700',
  },
  confidenceText: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 14,
    marginTop: 4,
  },
  fretText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 13,
    marginTop: 6,
    letterSpacing: 0.5,
  },
  bottomBar: {
    position: 'absolute',
    bottom: 48,
    left: 0,
    right: 0,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 24,
  },
  calibrateButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  calibrateText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: '600',
  },
  recordButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordButtonInner: { width: 52, height: 52, borderRadius: 26, backgroundColor: '#EF4444' },
  badgeStack: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    gap: 3,
  },
  badgeText: { color: 'rgba(255,255,255,0.6)', fontSize: 11 },
});
