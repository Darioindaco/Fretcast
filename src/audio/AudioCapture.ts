/**
 * AudioCapture — spec Section 6
 *
 * Web: Web Audio API, ScriptProcessorNode at 22050Hz
 * iOS: Swift module (Phase 2 iOS — stub here, implemented natively)
 *
 * Sample rate: 22050Hz (sufficient for guitar E2=82.4Hz to E5=1318Hz,
 * halves FFT cost vs 44.1kHz — spec Section 6)
 */

export const SAMPLE_RATE = 22050;
export const HPS_WINDOW = 4096;
export const HPS_HOP = 220;       // 10ms at 22050Hz
export const YIN_WINDOW = 2048;

type AudioCallback = (buffer: Float32Array) => void;

export class AudioCapture {
  private ctx: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private callback: AudioCallback | null = null;
  private hopBuffer: Float32Array = new Float32Array(0);

  async start(mediaStream: MediaStream, hopSize: number, onBuffer: AudioCallback): Promise<void> {
    this.callback = onBuffer;
    void mediaStream; // stored via source node
    this.hopBuffer = new Float32Array(0);

    // Web Audio context at 22050Hz — spec Section 6
    this.ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)({
      sampleRate: SAMPLE_RATE,
    });

    this.source = this.ctx.createMediaStreamSource(mediaStream);

    // ScriptProcessor bufferSize must be power of 2; use 2048 and accumulate
    const bufferSize = 2048;
    this.processor = this.ctx.createScriptProcessor(bufferSize, 1, 1);

    this.processor.onaudioprocess = (e: AudioProcessingEvent) => {
      const input = e.inputBuffer.getChannelData(0);
      this.accumulate(input, hopSize);
    };

    this.source.connect(this.processor);
    this.processor.connect(this.ctx.destination);
  }

  private accumulate(chunk: Float32Array, hopSize: number): void {
    // Concatenate incoming chunk to rolling buffer
    const combined = new Float32Array(this.hopBuffer.length + chunk.length);
    combined.set(this.hopBuffer);
    combined.set(chunk, this.hopBuffer.length);
    this.hopBuffer = combined;

    // Emit complete hops
    while (this.hopBuffer.length >= hopSize) {
      const hop = this.hopBuffer.slice(0, hopSize);
      this.hopBuffer = this.hopBuffer.slice(hopSize);
      if (this.callback != null) this.callback(hop);
    }
  }

  stop(): void {
    this.processor?.disconnect();
    this.source?.disconnect();
    this.ctx?.close();
    this.processor = null;
    this.source = null;
    this.ctx = null;
    this.callback = null;
  }

  get sampleRate(): number {
    return this.ctx?.sampleRate ?? SAMPLE_RATE;
  }
}
