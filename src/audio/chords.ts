/**
 * Chord lookup table — spec Section 6
 *
 * 12 roots × [maj, min, 7, maj7, min7, dim, aug, sus2, sus4, add9, power]
 * = 132 chord types
 *
 * Each entry: sorted pitch class set (mod 12)
 *
 * BASS NOTE WEIGHTING (spec Section 6):
 * Do NOT use plain Jaccard similarity.
 * Lowest stable detected frequency gets ×1.5 multiplier.
 * Distinguishes C major from C/E slash chord.
 */

export interface ChordEntry {
  name: string;
  root: string;
  quality: string;
  /** Sorted pitch class set (0=C, 1=C#, ..., 11=B) */
  pitchClasses: number[];
}

const ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/**
 * Interval sets for each quality (semitones above root)
 */
const QUALITIES: Array<{ suffix: string; intervals: number[] }> = [
  { suffix: '',      intervals: [0, 4, 7] },           // major
  { suffix: 'm',     intervals: [0, 3, 7] },           // minor
  { suffix: '7',     intervals: [0, 4, 7, 10] },       // dominant 7
  { suffix: 'maj7',  intervals: [0, 4, 7, 11] },       // major 7
  { suffix: 'm7',    intervals: [0, 3, 7, 10] },       // minor 7
  { suffix: 'dim',   intervals: [0, 3, 6] },           // diminished
  { suffix: 'aug',   intervals: [0, 4, 8] },           // augmented
  { suffix: 'sus2',  intervals: [0, 2, 7] },           // suspended 2
  { suffix: 'sus4',  intervals: [0, 5, 7] },           // suspended 4
  { suffix: 'add9',  intervals: [0, 4, 7, 14] },       // add9 (14 mod 12 = 2)
  { suffix: '5',     intervals: [0, 7] },              // power chord
];

export const CHORD_TABLE: ChordEntry[] = ROOTS.flatMap((root, rootIdx) =>
  QUALITIES.map(({ suffix, intervals }) => ({
    name: root + suffix,
    root,
    quality: suffix || 'maj',
    pitchClasses: intervals.map((i) => (rootIdx + i) % 12).sort((a, b) => a - b),
  }))
);

/**
 * Convert frequency to pitch class (0=C … 11=B)
 */
export function freqToPitchClass(freq: number): number {
  if (freq <= 0) return -1;
  const midiNote = 12 * Math.log2(freq / 440) + 69;
  return ((Math.round(midiNote) % 12) + 12) % 12;
}

/**
 * Convert frequency to MIDI note number (float)
 */
export function freqToMidi(freq: number): number {
  return 12 * Math.log2(freq / 440) + 69;
}

export interface ChordMatch {
  chord: ChordEntry;
  score: number;
  bassNote: number; // pitch class of lowest detected note
}

/**
 * Match chord from 12-bin chroma vector using cosine similarity.
 *
 * Spec Section 6: bass note weighting ×1.5 applied to the chroma template.
 * Chroma-based matching is far more robust than peak-picking because:
 *  - Weak thirds (common on guitar) still contribute energy to the chroma
 *  - Octave doublings cancel out
 *  - No threshold tuning required
 *
 * Returns top 3 matches sorted by score descending.
 */
export function matchChordFromChroma(
  chroma: Float64Array,
  bassFrequency: number
): ChordMatch[] {
  if (chroma.length !== 12) return [];

  const bassPC = bassFrequency > 0 ? freqToPitchClass(bassFrequency) : -1;

  const scores: ChordMatch[] = CHORD_TABLE.map((entry) => {
    // Build weighted chord template
    const template = new Float64Array(12);
    for (const pc of entry.pitchClasses) {
      // Bass note weighting: ×1.5 on lowest chord note (spec Section 6)
      template[pc] = pc === bassPC ? 1.5 : 1.0;
    }

    // Cosine similarity between chroma and template
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < 12; i++) {
      dot += chroma[i] * template[i];
      normA += chroma[i] * chroma[i];
      normB += template[i] * template[i];
    }
    const cosine = normA > 0 && normB > 0
      ? dot / (Math.sqrt(normA) * Math.sqrt(normB))
      : 0;

    // Power chord penalty: rarely intended in normal strumming
    const powerPenalty = entry.quality === '5' ? 0.12 : 0;

    return {
      chord: entry,
      score: Math.max(0, cosine - powerPenalty),
      bassNote: bassPC,
    };
  });

  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, 3);
}

/**
 * Match detected frequencies to chord table.
 *
 * Spec Section 6:
 * - Identify lowest stable detected frequency (bass note)
 * - Apply ×1.5 multiplier to bass note pitch class in matching
 * - Distinguishes C major (bass=C) from C/E (bass=E)
 *
 * Returns top 3 matches sorted by score descending.
 */
export function matchChord(
  frequencies: number[],
  magnitudes: number[]
): ChordMatch[] {
  if (frequencies.length === 0) return [];

  // Pitch classes of detected notes, with magnitudes
  const detected: Array<{ pc: number; weight: number }> = frequencies
    .map((f, i) => ({ pc: freqToPitchClass(f), weight: magnitudes[i] ?? 1 }))
    .filter((d) => d.pc >= 0);

  if (detected.length === 0) return [];

  // Identify lowest note (smallest frequency = lowest index)
  const bassPC = detected[0].pc;

  // Apply bass note weighting (spec Section 6: ×1.5)
  const weightedDetected = detected.map((d) => ({
    pc: d.pc,
    weight: d.pc === bassPC ? d.weight * 1.5 : d.weight,
  }));

  const totalWeight = weightedDetected.reduce((s, d) => s + d.weight, 0);

  const detectedPCSet = new Set(weightedDetected.map((d) => d.pc));
  const numDetected = detectedPCSet.size;

  // Score each chord in the table
  const scores: ChordMatch[] = CHORD_TABLE.map((entry) => {
    const chordPCSet = new Set(entry.pitchClasses);

    // Weighted intersection (detected notes that appear in chord)
    let intersection = 0;
    let matchedCount = 0;
    for (const d of weightedDetected) {
      if (chordPCSet.has(d.pc)) { intersection += d.weight; matchedCount++; }
    }

    // Precision: fraction of detected notes in this chord
    const precision = totalWeight > 0 ? intersection / totalWeight : 0;

    // Recall: fraction of chord notes that were detected
    const recall = entry.pitchClasses.length > 0
      ? matchedCount / entry.pitchClasses.length
      : 0;

    // F-score (harmonic mean of precision and recall)
    const fScore = precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : 0;

    // Complexity bonus: when all detected notes fit inside a chord AND
    // that chord has more notes than detected, prefer the larger chord.
    // This prevents C5 from beating C major when only root+fifth detected.
    // Power chord (2 notes) gets a penalty when 2+ real notes detected.
    const allDetectedInChord = [...detectedPCSet].every((pc) => chordPCSet.has(pc));
    const complexityBonus = allDetectedInChord && entry.pitchClasses.length > numDetected
      ? 0.05 * (entry.pitchClasses.length - numDetected)
      : 0;

    // Power chord penalty: only valid when explicitly playing 2 strings
    // In normal strumming, power chords are very rare
    const powerPenalty = entry.quality === '5' ? 0.15 : 0;

    const score = Math.max(0, fScore + complexityBonus - powerPenalty);

    return { chord: entry, score, bassNote: bassPC };
  });

  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, 3);
}
