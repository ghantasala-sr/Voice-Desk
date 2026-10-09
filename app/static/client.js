// Browser client: microphone -> 16 kHz PCM16 -> WebSocket, and audio back.
//
// Two things here are not incidental:
//
//  * AudioWorklet, not ScriptProcessor. ScriptProcessor is deprecated and
//    glitches under load, which reads as a bad agent rather than a bad buffer.
//  * A SHORT playback jitter buffer that can be flushed instantly. Every
//    millisecond of buffered audio is a millisecond of barge-in latency the
//    caller actually hears.

const SAMPLE_RATE = 16000;

const els = {
  mic: document.getElementById("mic"),
  status: document.getElementById("status"),
  transcript: document.getElementById("transcript"),
  events: document.getElementById("events"),
  stages: document.getElementById("stages"),
};

let ws = null;
let audioCtx = null;
let micStream = null;
let playback = null;
let running = false;
let agentBubble = null;

els.mic.onclick = () => (running ? stop() : start());

async function start() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    setStatus("connected");
    ws.send(JSON.stringify({ type: "start", domain: "ticketing", source: "browser" }));
    await startAudio();
    running = true;
    els.mic.textContent = "End call";
    els.mic.classList.add("stop");
  };
  ws.onclose = () => { setStatus("disconnected"); stop(); };
  ws.onerror = () => setStatus("connection error");
  ws.onmessage = (e) =>
    typeof e.data === "string" ? onControl(JSON.parse(e.data)) : playback?.push(e.data);
}

function stop() {
  running = false;
  els.mic.textContent = "Start call";
  els.mic.classList.remove("stop");
  micStream?.getTracks().forEach((t) => t.stop());
  audioCtx?.close();
  audioCtx = micStream = playback = null;
  if (ws?.readyState === WebSocket.OPEN) ws.close();
}

async function startAudio() {
  // Echo cancellation is required, not optional: without it the agent hears
  // itself through the laptop speakers and barge-in fires continuously.
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true,
             channelCount: 1, sampleRate: SAMPLE_RATE },
  });

  audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  await audioCtx.audioWorklet.addModule("/static/worklets.js");

  const capture = new AudioWorkletNode(audioCtx, "capture");
  capture.port.onmessage = (e) => ws?.readyState === WebSocket.OPEN && ws.send(e.data);
  audioCtx.createMediaStreamSource(micStream).connect(capture);

  const player = new AudioWorkletNode(audioCtx, "playback");
  player.connect(audioCtx.destination);
  playback = {
    push: (buf) => player.port.postMessage({ cmd: "push", buf }, [buf]),
    flush: () => player.port.postMessage({ cmd: "flush" }),
  };
}

function onControl(msg) {
  switch (msg.type) {
    case "call_started":
      setStatus(`call ${msg.call_id.slice(0, 8)}`);
      break;
    case "transcript":
      addTranscript(msg);
      break;
    case "cancel_speech":
      // Flush immediately. Anything still buffered is audio the caller would
      // hear AFTER they started talking.
      playback?.flush();
      agentBubble?.classList.add("cancelled");
      break;
    case "event":
      addEvent(msg.name, msg.payload);
      break;
    case "metrics":
      renderStages(msg);
      break;
    case "call_ended":
      setStatus(`ended: ${msg.outcome}`);
      stop();
      break;
    case "error":
      addEvent("error", { message: msg.message });
      setStatus(msg.message);
      break;
  }
}

function addTranscript({ role, text, segment_id }) {
  // Consecutive agent segments belong to one spoken answer, so they join one
  // bubble — that is what the caller experienced.
  if (role === "agent" && agentBubble && segment_id > 0) {
    agentBubble.textContent += " " + text;
    return;
  }
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  els.transcript.append(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  agentBubble = role === "agent" ? div : null;
}

function addEvent(name, payload = {}) {
  const div = document.createElement("div");
  div.className = `ev ${name}`;
  const detail = Object.entries(payload)
    .filter(([, v]) => v !== null && typeof v !== "object")
    .map(([k, v]) => `${k}=${v}`)
    .join(" ");
  div.innerHTML = `<b>${name}</b> ${detail}`;
  els.events.prepend(div);
}

// The latency strip. Endpointing is shown separately from everything else,
// because it is a VAD setting rather than anything the harness earned.
const STAGE_ORDER = [
  ["endpoint", "vad"], ["stt_final", "stt"], ["retrieval", "rag"],
  ["llm_ttft", "llm"], ["sentence_buffer", "buffer"], ["tts_ttfb", "tts"],
];

function renderStages({ stages = {}, voice_to_voice_ms, stop_reason }) {
  els.stages.innerHTML = "";
  for (const [key, label] of STAGE_ORDER) {
    if (stages[key] === undefined) continue;
    const span = document.createElement("span");
    span.className = `stage ${key}`;
    span.textContent = `${label} ${stages[key]}ms`;
    els.stages.append(span);
  }
  const total = document.createElement("span");
  total.className = "stage total";
  total.textContent = voice_to_voice_ms
    ? `voice-to-voice ${voice_to_voice_ms}ms · ${stop_reason}`
    : `${stop_reason}`;
  els.stages.append(total);
}

const setStatus = (text) => (els.status.textContent = text);
