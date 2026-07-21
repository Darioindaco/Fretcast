import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useShallow } from 'zustand/react/shallow';
import { useFretcastStore } from '../store';
import { RootStackParamList } from '../types';
import { useAudioPipeline } from '../hooks/useAudioPipeline';

// Native-only — resolved to no-op mock on web via metro.config.js
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
  type VideoFile,
} from 'react-native-vision-camera';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Recording'>;
};

const MAX_DURATION_MS = 90_000;
const WARN_DURATION_MS = 80_000;

export default function RecordingScreen({ navigation }: Props) {
  const { setAudioMode, audioMode, setActiveSession } = useFretcastStore(
    useShallow((s) => ({
      setAudioMode: s.setAudioMode,
      audioMode: s.audioMode,
      setActiveSession: s.setActiveSession,
    }))
  );
  const liveChord = useFretcastStore((s) => s.liveChord);

  // ── Native camera ────────────────────────────────────────────────────────
  const nativeCameraRef = useRef<InstanceType<typeof Camera> | null>(null);
  const device = useCameraDevice('back');
  const format = useCameraFormat(device, [
    { videoResolution: { width: 1920, height: 1080 } },
    { fps: 30 },
  ]);

  // ── Web camera ───────────────────────────────────────────────────────────
  const webVideoRef = useRef<HTMLVideoElement | null>(null);
  const webStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [webReady, setWebReady] = useState(false);

  // ── Recording state ──────────────────────────────────────────────────────
  const [isRecording, setIsRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [warned, setWarned] = useState(false);
  const startTimeRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stoppingRef = useRef(false);

  // ── Mount: explicit YIN → HPS switch (spec Section 6) ───────────────────
  useEffect(() => {
    setAudioMode('HPS');
    return () => {
      setAudioMode('IDLE');
      if (timerRef.current != null) clearInterval(timerRef.current);
    };
  }, [setAudioMode]);

  // ── Audio pipeline (Phase 2) ─────────────────────────────────────────────
  useAudioPipeline({
    mode: 'HPS',
    stream: webStreamRef.current,
    sessionStartMs: startTimeRef.current,
  });

  // ── Web: start camera stream ─────────────────────────────────────────────
  useEffect(() => {
    if (Platform.OS !== 'web') return;

    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: true,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        webStreamRef.current = stream;
        if (webVideoRef.current != null) {
          webVideoRef.current.srcObject = stream;
        }
        setWebReady(true);
      } catch (e) {
        console.warn('[RecordingScreen] getUserMedia failed:', e);
        setWebReady(true); // still allow UI to render
      }
    })();

    return () => {
      cancelled = true;
      webStreamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ── Timer ────────────────────────────────────────────────────────────────
  const stopTimer = useCallback(() => {
    if (timerRef.current != null) { clearInterval(timerRef.current); timerRef.current = null; }
  }, []);

  // ── Stop recording ───────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (!isRecording || stoppingRef.current) return;
    stoppingRef.current = true;
    stopTimer();
    if (Platform.OS === 'web') {
      mediaRecorderRef.current?.stop();
    } else {
      nativeCameraRef.current?.stopRecording();
    }
  }, [isRecording, stopTimer]);

  // ── Start recording ──────────────────────────────────────────────────────
  const startRecording = useCallback(() => {
    if (stoppingRef.current || isRecording) return;
    setIsRecording(true);
    setElapsedMs(0);
    setWarned(false);
    startTimeRef.current = Date.now();

    timerRef.current = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      setElapsedMs(elapsed);
      if (elapsed >= WARN_DURATION_MS) setWarned(true);
      if (elapsed >= MAX_DURATION_MS) stopRecording();
    }, 500);

    if (Platform.OS === 'web') {
      const stream = webStreamRef.current;
      if (stream == null) { setIsRecording(false); stopTimer(); return; }

      chunksRef.current = [];
      // Use webm on web — mp4 requires native h264_videotoolbox (spec Section 9)
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        setIsRecording(false);
        stoppingRef.current = false;
        setActiveSession({
          startTimestamp: startTimeRef.current,
          videoUri: url,
          overlayUri: null,
          mergedVideoUri: null,
          analysisResult: null,
          durationMs: Date.now() - startTimeRef.current,
        });
        navigation.replace('Processing', { sessionId: url });
      };
      recorder.start(100);
    } else {
      nativeCameraRef.current?.startRecording({
        fileType: 'mp4',
        videoCodec: 'h264',
        onRecordingFinished: (video: VideoFile) => {
          stopTimer();
          setIsRecording(false);
          stoppingRef.current = false;
          setActiveSession({
            startTimestamp: startTimeRef.current,
            videoUri: video.path,
            overlayUri: null,
            mergedVideoUri: null,
            analysisResult: null,
            durationMs: Date.now() - startTimeRef.current,
          });
          navigation.replace('Processing', { sessionId: video.path });
        },
        onRecordingError: (error) => {
          stopTimer();
          setIsRecording(false);
          stoppingRef.current = false;
          console.error('[Recording] error:', error);
        },
      });
    }
  }, [isRecording, stopTimer, stopRecording, setActiveSession, navigation]);

  // ── Auto-start: web waits for stream, native waits for camera init ───────
  useEffect(() => {
    if (Platform.OS === 'web') {
      if (!webReady) return;
      const t = setTimeout(startRecording, 300);
      return () => clearTimeout(t);
    } else {
      if (device == null) return;
      const t = setTimeout(startRecording, 400);
      return () => clearTimeout(t);
    }
  }, [webReady, device]); // eslint-disable-line react-hooks/exhaustive-deps

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  };

  return (
    <View style={styles.container}>
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
            ref={nativeCameraRef}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={true}
            video={true}
            audio={true}
            format={format}
            videoStabilizationMode="auto"
          />
        )
      )}

      {/* Live chord — Phase 6 */}
      <View style={styles.chordOverlay}>
        <Text style={styles.chordText}>{liveChord?.name ?? ''}</Text>
      </View>

      {/* Timer + mode indicator */}
      <View style={styles.topRight}>
        <Text style={[styles.timerText, warned && styles.timerWarn]}>
          {formatTime(elapsedMs)}
        </Text>
        <Text style={styles.modeText}>{audioMode}</Text>
        {isRecording && <View style={styles.recDot} />}
      </View>

      {warned && (
        <View style={styles.warnBanner}>
          <Text style={styles.warnText}>10 seconds remaining</Text>
        </View>
      )}

      <View style={styles.bottomBar}>
        <TouchableOpacity
          style={[styles.stopButton, !isRecording && styles.stopButtonDisabled]}
          onPress={stopRecording}
          disabled={!isRecording}
          activeOpacity={0.7}
        >
          <View style={styles.stopButtonInner} />
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  chordOverlay: {
    position: 'absolute', top: 60, left: 0, right: 0, alignItems: 'center',
    pointerEvents: 'none' as 'none',
  },
  chordText: { color: '#fff', fontSize: 52, fontWeight: '700' },
  topRight: {
    position: 'absolute', top: 16, right: 16, alignItems: 'flex-end', gap: 4,
    pointerEvents: 'none' as 'none',
  },
  timerText: { color: '#fff', fontSize: 14, fontWeight: '500' },
  timerWarn: { color: '#F59E0B' },
  modeText: { color: '#22C55E', fontSize: 10, fontFamily: 'monospace' },
  recDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#EF4444' },
  warnBanner: {
    position: 'absolute', bottom: 140, left: 0, right: 0, alignItems: 'center',
    pointerEvents: 'none' as 'none',
  },
  warnText: { color: '#F59E0B', fontSize: 14, fontWeight: '600' },
  bottomBar: { position: 'absolute', bottom: 48, left: 0, right: 0, alignItems: 'center' },
  stopButton: {
    width: 72, height: 72, borderRadius: 36,
    borderWidth: 4, borderColor: '#fff', alignItems: 'center', justifyContent: 'center',
  },
  stopButtonDisabled: { opacity: 0.5 },
  stopButtonInner: { width: 28, height: 28, borderRadius: 4, backgroundColor: '#EF4444' },
});
