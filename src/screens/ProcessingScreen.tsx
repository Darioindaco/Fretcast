import { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { useShallow } from 'zustand/react/shallow';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { useFretcastStore } from '../store';
import { RootStackParamList } from '../types';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Processing'>;
  route: RouteProp<RootStackParamList, 'Processing'>;
};

/**
 * Spec Section 11 — PROCESSING
 * Blurred last frame background (Phase 7).
 * Real pipeline status text per step.
 * Target: < 16s for 60s recording.
 *
 * Pipeline order (spec Section 9):
 *   1. Audio analysis (HPS → chords, key, BPM)
 *   2. Skia transparent overlay render (< 5s)
 *   3. FFmpeg hardware merge via h264_videotoolbox (< 6s)
 *   Total target: < 11s
 */
export default function ProcessingScreen({ navigation, route }: Props) {
  const { processingStatus, setProcessingStep } = useFretcastStore(
    useShallow((s) => ({
      processingStatus: s.processingStatus,
      setProcessingStep: s.setProcessingStep,
    }))
  );

  useEffect(() => {
    // Processing pipeline — implemented in Phase 7
    // Stub: navigate to review after placeholder delay
    const timer = setTimeout(() => {
      navigation.replace('Review', { sessionId: route.params.sessionId });
    }, 2000);

    return () => clearTimeout(timer);
  }, [navigation, route.params.sessionId, setProcessingStep]);

  const stepLabels: Record<string, string> = {
    IDLE: 'Starting…',
    ANALYZING_AUDIO: 'Analyzing audio…',
    RENDERING_OVERLAY: 'Rendering overlay…',
    MERGING_VIDEO: 'Merging video…',
    COMPLETE: 'Done',
    ERROR: 'Error',
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <Text style={styles.stepText}>
          {stepLabels[processingStatus.step] ?? 'Processing…'}
        </Text>
        <View style={styles.progressBar}>
          <View
            style={[
              styles.progressFill,
              { width: `${processingStatus.progressPercent}%` },
            ]}
          />
        </View>
        {processingStatus.errorMessage != null && (
          <Text style={styles.errorText}>{processingStatus.errorMessage}</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  content: {
    alignItems: 'center',
    gap: 16,
    paddingHorizontal: 32,
    width: '100%',
  },
  stepText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '500',
  },
  progressBar: {
    width: '100%',
    height: 4,
    backgroundColor: '#333',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#22C55E',
    borderRadius: 2,
  },
  errorText: {
    color: '#EF4444',
    fontSize: 14,
    textAlign: 'center',
  },
});
