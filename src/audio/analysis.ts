/**
 * BPM detection + Key detection — spec Section 6
 *
 * BPM:  Median inter-onset interval → 60000/medianIOI, clamp 40-240
 * Key:  Krumhansl-Schmuckler algorithm, weighted by chord duration
 */

import { DetectedChord } from '../types';

// ── BPM ─────────────────────────────────────────────────────────────────────

/**
 * Spec Section 6: BPM = median IOI → 60000/medianIOI, clamp 40-240
 */
export function detectBPM(onsetTimestampsMs: number[]): number {
  if (onsetTimestampsMs.length < 2) return 120; // default

  const iois: number[] = [];
  for (let i = 1; i < onsetTimestampsMs.length; i++) {
    iois.push(onsetTimestampsMs[i] - onsetTimestampsMs[i - 1]);
  }

  const sorted = [...iois].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const medianIOI = sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];

  if (medianIOI <= 0) return 120;

  const bpm = 60000 / medianIOI;
  return Math.max(40, Math.min(240, Math.round(bpm)));
}

// ── Key (Krumhansl-Schmuckler) ───────────────────────────────────────────────

/**
 * Krumhansl-Schmuckler key profiles (major and minor).
 * Reference: Krumhansl (1990) "Cognitive Foundations of Musical Pitch"
 */
const KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function rotateProfile(profile: number[], shift: number): number[] {
  const n = profile.length;
  return profile.map((_, i) => profile[(i - shift + n) % n]);
}

function pearsonCorrelation(a: number[], b: number[]): number {
  const n = a.length;
  const meanA = a.reduce((s, v) => s + v, 0) / n;
  const meanB = b.reduce((s, v) => s + v, 0) / n;
  let num = 0, denA = 0, denB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    num += da * db;
    denA += da * da;
    denB += db * db;
  }
  return denA > 0 && denB > 0 ? num / Math.sqrt(denA * denB) : 0;
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/**
 * Spec Section 6: Krumhansl-Schmuckler key detection, weighted by duration.
 * Returns e.g. "C major" or "A minor"
 */
export function detectKey(chords: DetectedChord[]): { key: string; relativeKey: string } {
  if (chords.length === 0) return { key: 'C major', relativeKey: 'A minor' };

  // Build pitch class duration vector (12 bins)
  const pcDuration = new Array(12).fill(0) as number[];

  for (let i = 0; i < chords.length; i++) {
    const chord = chords[i];
    const nextTimestamp = i < chords.length - 1
      ? chords[i + 1].timestamp
      : chord.timestamp + 1000; // assume 1s for last chord
    const duration = nextTimestamp - chord.timestamp;

    // Add duration to all pitch classes in this chord's voicing
    for (const note of chord.voicing) {
      const pc = NOTE_NAMES.indexOf(note.replace(/\d/, '').replace('b', '#') );
      if (pc >= 0) pcDuration[pc] += duration;
    }
  }

  // Correlate against all 24 key profiles
  let bestScore = -Infinity;
  let bestKey = 'C';
  let bestMode = 'major';

  for (let root = 0; root < 12; root++) {
    const majorProfile = rotateProfile(KS_MAJOR, root);
    const minorProfile = rotateProfile(KS_MINOR, root);

    const majorScore = pearsonCorrelation(pcDuration, majorProfile);
    const minorScore = pearsonCorrelation(pcDuration, minorProfile);

    if (majorScore > bestScore) {
      bestScore = majorScore;
      bestKey = NOTE_NAMES[root];
      bestMode = 'major';
    }
    if (minorScore > bestScore) {
      bestScore = minorScore;
      bestKey = NOTE_NAMES[root];
      bestMode = 'minor';
    }
  }

  const key = `${bestKey} ${bestMode}`;

  // Relative key: major → relative minor is 9 semitones up (or 3 down)
  // minor → relative major is 3 semitones up
  const rootIdx = NOTE_NAMES.indexOf(bestKey);
  const relativeRootIdx = bestMode === 'major'
    ? (rootIdx + 9) % 12
    : (rootIdx + 3) % 12;
  const relativeMode = bestMode === 'major' ? 'minor' : 'major';
  const relativeKey = `${NOTE_NAMES[relativeRootIdx]} ${relativeMode}`;

  return { key, relativeKey };
}
