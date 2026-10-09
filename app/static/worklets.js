// Two AudioWorklet processors: microphone capture and agent playback.

// Float32 at the context rate -> PCM16 frames of 20ms, posted to the main
// thread for the WebSocket.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Int16Array(320); // 20ms @ 16kHz
    this.filled = 0;
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      const clamped = Math.max(-1, Math.min(1, channel[i]));
      this.frame[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      if (this.filled === this.frame.length) {
        const copy = this.frame.slice();
        this.port.postMessage(copy.buffer, [copy.buffer]);
        this.filled = 0;
      }
    }
    return true;
  }
}

// Plays queued PCM16. The queue is deliberately short and `flush` empties it
// synchronously: on a barge-in, anything still queued is audio the caller
// would hear after they began speaking.
class Playback extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.port.onmessage = ({ data }) => {
      if (data.cmd === "push") this.queue.push(new Int16Array(data.buf));
      else if (data.cmd === "flush") { this.queue.length = 0; this.offset = 0; }
    };
  }
  process(_inputs, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      const head = this.queue[0];
      if (!head) { out[i] = 0; continue; }
      out[i] = head[this.offset++] / 0x8000;
      if (this.offset >= head.length) { this.queue.shift(); this.offset = 0; }
    }
    return true;
  }
}

registerProcessor("capture", Capture);
registerProcessor("playback", Playback);
