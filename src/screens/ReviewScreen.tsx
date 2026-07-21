import { useRef } from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { VideoView, useVideoPlayer } from 'expo-video';
import { useFretcastStore } from '../store';
import { RootStackParamList } from '../types';

type Props = {
  navigation: NativeStackNavigationProp<RootStackParamList, 'Review'>;
  route: RouteProp<RootStackParamList, 'Review'>;
};

/**
 * Spec Section 11 — REVIEW
 * Video player: auto-play muted, loop.
 * Analysis bar: Key · BPM · Duration · Chords.
 * Chord list: confidence bar + edit per chord.
 * Tab: monospace, horizontal scroll.
 * Share: Video · Screenshot · Tab card.
 * MIDI: stubbed button, disabled ("coming in v2").
 */
export default function ReviewScreen({ navigation }: Props) {
  const detectedChords = useFretcastStore((s) => s.detectedChords);
  const activeSession = useFretcastStore((s) => s.activeSession);
  const videoRef = useRef<VideoView>(null);

  const videoUri = activeSession?.mergedVideoUri ?? activeSession?.videoUri ?? null;
  const analysisResult = activeSession?.analysisResult;

  // Spec Section 11: auto-play muted, loop
  const player = useVideoPlayer(videoUri ?? '', (p) => {
    p.loop = true;
    p.muted = true;
    p.play();
  });

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Video player — auto-play muted, loop */}
      {videoUri != null ? (
        <VideoView
          ref={videoRef}
          player={player}
          style={styles.videoPlayer}
          contentFit="cover"
          nativeControls={false}
        />
      ) : (
        <View style={styles.videoPlaceholder}>
          <Text style={styles.placeholderText}>No video recorded</Text>
        </View>
      )}

      {/* Analysis bar */}
      <View style={styles.analysisBar}>
        <Text style={styles.analysisPill}>Key: {analysisResult?.key ?? '—'}</Text>
        <Text style={styles.analysisPill}>BPM: {analysisResult?.bpm ?? '—'}</Text>
        <Text style={styles.analysisPill}>
          {analysisResult?.duration != null
            ? `${Math.round(analysisResult.duration)}s`
            : '—'}
        </Text>
        <Text style={styles.analysisPill}>{detectedChords.length} chords</Text>
      </View>

      {/* Chord list with confidence bars */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Chords</Text>
        {detectedChords.length === 0 ? (
          <Text style={styles.emptyText}>No chords detected</Text>
        ) : (
          detectedChords.map((chord, idx) => (
            <View key={idx} style={styles.chordRow}>
              <View style={styles.chordRowLeft}>
                <Text style={styles.chordName}>
                  {chord.name}{chord.confidence < 0.75 ? '?' : ''}
                </Text>
                <Text style={styles.chordMeta}>
                  {chord.fretPosition != null ? `Fret ${chord.fretPosition}` : ''}
                  {chord.corrected ? '  (edited)' : ''}
                </Text>
              </View>
              <View style={styles.confidenceBar}>
                <View
                  style={[
                    styles.confidenceFill,
                    { width: `${Math.round(chord.confidence * 100)}%` },
                  ]}
                />
              </View>
              <Text style={styles.confidenceLabel}>
                {Math.round(chord.confidence * 100)}%
              </Text>
            </View>
          ))
        )}
      </View>

      {/* Tab — monospace, horizontal scroll */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Tab</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator>
          <Text style={styles.tabText}>
            {analysisResult?.tabString ?? '— available after Phase 7 —'}
          </Text>
        </ScrollView>
      </View>

      {/* Share actions */}
      <View style={styles.shareRow}>
        <TouchableOpacity style={styles.shareButton}>
          <Text style={styles.shareButtonText}>Video</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.shareButton}>
          <Text style={styles.shareButtonText}>Screenshot</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.shareButton}>
          <Text style={styles.shareButtonText}>Tab Card</Text>
        </TouchableOpacity>
        {/* Spec Section 12: MIDI stub — disabled, "coming in v2" */}
        <TouchableOpacity
          style={[styles.shareButton, styles.shareButtonDisabled]}
          disabled
        >
          <Text style={[styles.shareButtonText, styles.shareButtonTextDisabled]}>
            MIDI{'\n'}v2
          </Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity
        style={styles.doneButton}
        onPress={() => navigation.navigate('Home')}
      >
        <Text style={styles.doneButtonText}>Done</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  content: {
    paddingBottom: 48,
  },
  videoPlayer: {
    width: '100%',
    height: 260,
    backgroundColor: '#000',
  },
  videoPlaceholder: {
    height: 260,
    backgroundColor: '#111',
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderText: {
    color: '#555',
    fontSize: 14,
  },
  analysisBar: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 8,
    flexWrap: 'wrap',
  },
  analysisPill: {
    backgroundColor: '#1a1a1a',
    color: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    fontSize: 13,
    fontWeight: '500',
  },
  section: {
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 8,
  },
  sectionTitle: {
    color: '#888',
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 8,
  },
  emptyText: {
    color: '#555',
    fontSize: 14,
  },
  chordRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#222',
    gap: 12,
  },
  chordRowLeft: {
    width: 96,
  },
  chordName: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  chordMeta: {
    color: '#666',
    fontSize: 11,
  },
  confidenceBar: {
    flex: 1,
    height: 4,
    backgroundColor: '#222',
    borderRadius: 2,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    backgroundColor: '#22C55E',
    borderRadius: 2,
  },
  confidenceLabel: {
    color: '#666',
    fontSize: 11,
    width: 36,
    textAlign: 'right',
  },
  tabText: {
    color: '#ccc',
    fontSize: 12,
    fontFamily: 'monospace',
    lineHeight: 18,
    paddingBottom: 8,
  },
  shareRow: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingTop: 16,
    gap: 8,
  },
  shareButton: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
  },
  shareButtonDisabled: {
    opacity: 0.35,
  },
  shareButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
    textAlign: 'center',
  },
  shareButtonTextDisabled: {
    color: '#666',
  },
  doneButton: {
    marginHorizontal: 16,
    marginTop: 16,
    backgroundColor: '#fff',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  doneButtonText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '600',
  },
});
