// VoiceDesk console: a live view over Postgres via app/observe.py.
//
// Rules this file keeps:
//
//  * textContent only. Transcripts and event payloads are caller-controlled
//    text; nothing from the API is ever parsed as HTML.
//  * endpoint_ms is drawn on its own track. It is the VAD silence setting,
//    not harness work, and blending it into the harness bar would make a VAD
//    change look like a harness speed-up (docs/10 §2).
//  * The SSE stream is the clock. Panels re-fetch when an event says their
//    data changed, rather than polling every call on a timer.

const STATS_EVERY_MS = 5000;
const FEED_MAX = 600;
const DETAIL_DEBOUNCE_MS = 300;

const HARNESS_STAGES = [
  ["stt_final_ms", "stt"], ["retrieval_ms", "retrieval"], ["llm_ttft_ms", "llm ttft"],
  ["sentence_buffer_ms", "sentence"], ["checker_ms", "checker"], ["tts_ttfb_ms", "tts ttfb"],
  ["playout_ms", "playout"],
];
const TOOL_STATES = ["pending", "dispatched", "committed"];

const $ = (id) => document.getElementById(id);
const state = {
  calls: [],             // summaries from /api/calls, newest first
  selected: null,        // call_id
  follow: true,          // auto-select the newest call until the user clicks one
  detail: null,          // /api/calls/{id} response for `selected`
  live: [],              // events from the stream, all calls, capped
  seen: new Set(),       // event_ids already in `live`
  source: "",
};

// -- tiny DOM helper -------------------------------------------------------
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = String(v);
    else if (k === "title") node.title = String(v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const fmtMs = (ms) => (ms === null || ms === undefined ? "—" : `${Math.round(ms)} ms`);
const fmtNum = (x, d = 3) => (x === null || x === undefined ? "—" : Number(x).toFixed(d));
const shortId = (id) => (id ? id.slice(-6) : "—");
function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}
function fmtWhen(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const secs = (Date.now() - d.getTime()) / 1000;
  if (secs < 60) return `${Math.max(0, Math.round(secs))}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  return d.toLocaleString([], { hour12: false });
}

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  const body = await res.json().catch(() => ({}));
  $("db-status").hidden = res.status !== 503;
  if (res.status === 503) $("db-status").textContent = body.error || "database unavailable";
  if (!res.ok) throw Object.assign(new Error(body.error || res.statusText), { status: res.status });
  return body;
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// -- health strip ----------------------------------------------------------
function tile(key, value, ...rows) {
  return el("div", { class: "tile" }, el("div", { class: "k", text: key }),
    el("div", { class: "v", text: value }), ...rows);
}
const sub = (text) => el("div", { class: "s", text });

function distTile(key, dist, emptyText = "none") {
  const entries = Object.entries(dist.counts || {});
  const rows = entries.length
    ? entries.map(([k, n]) => el("div", { class: "row" }, el("span", { text: k }),
        el("span", { text: `${n} · ${pct(n, dist.n)}` })))
    : [sub(emptyText)];
  return tile(key, `n=${dist.n}`, ...rows);
}
const pct = (n, total) => (total ? `${((100 * n) / total).toFixed(0)}%` : "—");

function latTile(key, lat, note) {
  return tile(key, lat.n ? `${fmtMs(lat.p50)}` : "—",
    sub(`p50 · p95 ${fmtMs(lat.p95)}`), sub(`n=${lat.n}${note ? " · " + note : ""}`));
}

async function refreshStats() {
  let s;
  try {
    s = await getJSON("/api/stats?window_minutes=60");
  } catch {
    return;
  }
  $("window").textContent = s.window_minutes;
  const dup = s.duplicate_side_effects;
  const dupTile = tile("duplicate side effects", String(dup.count),
    sub(dup.count === 0 ? "committed, same idempotency key · all time" : "BUG: same side effect committed twice"),
    ...dup.keys.map((k) => sub(`${k.idempotency_key} ×${k.n}`)));
  dupTile.classList.add(dup.count === 0 ? "dup-ok" : "dup-bad");

  $("tiles").replaceChildren(
    dupTile,
    tile("calls", `n=${s.calls.n}`, sub(`${s.calls.active} active (no ended_at)`)),
    distTile("stop reasons (turns)", s.stop_reasons),
    latTile("first audio · harness", s.latency.first_audio_ms, "excl. endpoint"),
    latTile("endpoint · VAD", s.latency.endpoint_ms, "a setting"),
    latTile("voice-to-voice", s.latency.voice_to_voice_ms, "incl. endpoint"),
    distTile("checker verdicts", s.checker_verdicts),
    distTile("tool status", s.tool_status),
    distTile("handoffs by reason", s.handoffs),
    tile("chunks ingested", String(s.chunks_ingested), sub("all time")),
  );
}

// -- calls list ------------------------------------------------------------
async function refreshCalls() {
  const q = state.source ? `&source=${encodeURIComponent(state.source)}` : "";
  try {
    state.calls = (await getJSON(`/api/calls?limit=100${q}`)).calls;
  } catch {
    return;
  }
  renderCalls();
  if (state.follow && state.calls.length && state.calls[0].call_id !== state.selected) {
    select(state.calls[0].call_id, { user: false });
  } else if (!state.calls.length && !state.selected) {
    renderEmpty();
  }
}
const refreshCallsSoon = debounce(refreshCalls, 250);

function outcomeBadge(call) {
  const outcome = call.outcome || (call.ended_at ? "ended" : "live");
  return el("span", { class: `badge ${outcome}`, text: outcome });
}

function renderCalls() {
  const items = state.calls.map((c) =>
    el("li", { class: c.call_id === state.selected ? "selected" : "", onclick: () => select(c.call_id, { user: true }) },
      el("div", { class: "top" },
        el("span", { class: "mono", text: shortId(c.call_id), title: c.call_id }), outcomeBadge(c)),
      el("div", { class: "meta", text:
        `${c.source} · ${c.domain} · ${c.turn_count} turn${c.turn_count === 1 ? "" : "s"} · ` +
        `${fmtWhen(c.started_at)}${c.duration_ms !== null ? " · " + (c.duration_ms / 1000).toFixed(1) + "s" : ""}` })));
  $("calls").replaceChildren(...items);
  if (!items.length) $("calls").append(el("li", { class: "dim", text: "no calls" }));
}

function setFollow(on) {
  state.follow = on;
  $("follow").classList.toggle("on", on);
  $("follow").textContent = on ? "following newest" : "follow newest call";
}

async function select(callId, { user }) {
  if (user) setFollow(false);
  state.selected = callId;
  state.detail = null;
  renderCalls();
  $("feed-title").textContent = `Live events · call ${shortId(callId)}`;
  await refreshDetail();
}

// -- call detail -----------------------------------------------------------
async function refreshDetail() {
  const callId = state.selected;
  if (!callId) return;
  let detail;
  try {
    detail = await getJSON(`/api/calls/${encodeURIComponent(callId)}`);
  } catch (err) {
    if (err.status === 404) {
      $("timeline").replaceChildren(el("div", { class: "empty dim", text: "This call no longer exists." }));
    }
    return;
  }
  if (callId !== state.selected) return; // a newer selection won the race
  state.detail = detail;
  renderDetail();
  renderFeed();
}
const refreshDetailSoon = debounce(refreshDetail, DETAIL_DEBOUNCE_MS);

function renderEmpty() {
  $("call-head").replaceChildren();
  $("timeline").replaceChildren($("empty-hint").content.cloneNode(true));
}

function renderDetail() {
  const { call, turns, tool_executions: tools, checker_verdicts: verdicts, handoffs } = state.detail;
  $("call-head").replaceChildren(
    el("div", { class: "title" }, el("span", { class: "mono", text: call.call_id }), outcomeBadge(call)),
    el("div", { class: "facts" },
      fact("source", call.source), fact("domain", call.domain), fact("turns", call.turn_count),
      fact("verified", call.caller_verified ? "yes" : "no"), fact("order", call.order_id),
      fact("started", fmtTime(call.started_at)),
      fact("duration", call.duration_ms !== null ? `${(call.duration_ms / 1000).toFixed(1)} s` : "in progress"),
      fact("handoff reason", call.handoff_reason), fact("rephrases", call.rephrase_count)));

  const byTurn = (rows) => {
    const m = new Map();
    for (const r of rows) {
      const k = r.turn_id || "";
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(r);
    }
    return m;
  };
  const toolsBy = byTurn(tools), verdictsBy = byTurn(verdicts), handoffsBy = byTurn(handoffs);

  // One scale per call, so bar lengths are comparable between turns.
  const scale = Math.max(1, ...turns.map((t) =>
    Math.max((t.endpoint_ms || 0), stageSum(t), t.first_audio_ms || 0)));

  const cards = turns.map((t) => turnCard(t, scale, toolsBy.get(t.turn_id) || [],
    verdictsBy.get(t.turn_id) || [], handoffsBy.get(t.turn_id) || []));

  // Rows not attached to a turn still belong on screen.
  const orphanTools = toolsBy.get("") || [], orphanHandoffs = handoffsBy.get("") || [];
  if (orphanTools.length || orphanHandoffs.length) {
    cards.push(el("div", { class: "turn" }, el("div", { class: "turn-head" }, el("span", { class: "seq", text: "call-level" })),
      orphanTools.length ? [el("h3", { text: "tool executions" }), toolList(orphanTools)] : null,
      orphanHandoffs.map(handoffBox)));
  }
  if (!cards.length) cards.push(el("div", { class: "empty dim", text: "Call started — no turns persisted yet." }));
  $("timeline").replaceChildren(...cards);
}

function fact(label, value) {
  if (value === null || value === undefined || value === "") return null;
  return el("span", {}, `${label} `, el("b", { text: value }));
}

const stageSum = (t) => HARNESS_STAGES.reduce((acc, [k]) => acc + (t[k] || 0), 0);

function turnCard(t, scale, tools, verdicts, handoffs) {
  return el("div", { class: "turn" },
    el("div", { class: "turn-head" },
      el("span", { class: "seq", text: `turn ${t.seq}` }),
      el("span", { class: "badge", text: t.trigger }),
      el("span", { class: `badge ${t.stop_reason || "live"}`, text: t.stop_reason || "in progress" }),
      t.ack_bypass_used ? el("span", { class: "badge", text: "ack bypass" }) : null,
      el("span", { class: "dim small", text: fmtTime(t.created_at) }),
      t.trace_id ? el("span", { class: "trace", text: `trace ${t.trace_id}`, title: "Jaeger trace id" }) : null),
    t.user_text ? el("div", { class: "bubble user" }, el("span", { class: "who", text:
      `caller${t.stt_confidence !== null ? ` · stt ${fmtNum(t.stt_confidence, 2)}` : ""}` }), t.user_text) : null,
    agentBubble(t, verdicts),
    waterfall(t, scale),
    retrievalBlock(t),
    verdicts.length ? [el("h3", { text: "checker verdicts" }), verdictList(verdicts)] : null,
    tools.length ? [el("h3", { text: "tool executions" }), toolList(tools)] : null,
    t.handoff_signals && t.handoff_signals.length
      ? el("div", { class: "kv" }, el("span", { class: "dim", text: "handoff signals" }),
          ...t.handoff_signals.map((s) => el("span", { class: "badge handoff", text: s })),
          t.handoff_score !== null ? el("span", { class: "dim", text: `score ${fmtNum(t.handoff_score, 2)}` }) : null)
      : null,
    handoffs.map(handoffBox));
}

// generated ⊇ approved ⊇ spoken — but NOT as plain strings. approved_text is
// stored in SPOKEN form ("twenty-five dollars", citations stripped) because the
// transport's played-character count is measured against what TTS received,
// while generated_text is the model's raw output ("$25 [K1]"). Diffing those
// two strings marks every normalized sentence as rejected. What the checker
// actually withheld is recorded per segment, so that is what is shown.
function agentBubble(t, verdicts = []) {
  const gen = t.generated_text || "", appr = t.approved_text || "", spoken = t.spoken_text || "";
  if (!gen && !appr && !spoken) return null;
  const withheld = verdicts.filter((v) => v.turn_id === t.turn_id && v.action_taken !== "released");
  const body = el("div", { class: "bubble agent" }, el("span", { class: "who", text:
    `agent · heard ${spoken.length}/${appr.length} approved chars · ${t.llm_iterations} llm iter` }));
  const extras = [];
  if (appr.startsWith(spoken)) {
    body.append(el("span", { text: spoken }));
    if (appr.length > spoken.length) {
      body.append(el("span", { class: "unheard", text: appr.slice(spoken.length), title: "approved, not heard (barge-in)" }));
    }
  } else {
    body.append(el("span", { text: spoken }));
    extras.push(el("div", { class: "altline" }, "approved: ", el("span", { class: "unheard", text: appr })));
  }
  for (const v of withheld) {
    extras.push(el("div", { class: "altline" }, `withheld by checker (${v.verdict}): `,
      el("span", { class: "withheld", text: v.segment_text })));
  }
  if (gen && gen.trim() !== appr.trim()) {
    // Not struck through: this is the same answer before citation stripping
    // and spoken-form normalization, not a rejection.
    extras.push(el("div", { class: "altline dim" }, "model output: ", el("span", { text: gen.trim() })));
  }
  const legend = el("div", { class: "legend" }, el("span", { class: "l-heard", text: "heard" }),
    el("span", { class: "l-unheard", text: "approved, not heard" }),
    el("span", { class: "l-withheld", text: "withheld by checker" }));
  return [body, ...extras, (appr !== spoken || withheld.length) ? legend : null];
}

function waterfall(t, scale) {
  const pctOf = (ms) => `${(100 * ms) / scale}%`;
  const epTrack = el("div", { class: "track" });
  if (t.endpoint_ms !== null) {
    epTrack.append(el("div", { class: "seg endpoint", style: `left:0;width:${pctOf(t.endpoint_ms)}`,
      title: `endpoint (VAD silence) ${t.endpoint_ms} ms`, text: `${t.endpoint_ms}` }));
  }
  const hTrack = el("div", { class: "track" });
  let x = 0;
  for (const [key, label] of HARNESS_STAGES) {
    const ms = t[key];
    if (ms === null || ms === undefined) continue;
    hTrack.append(el("div", { class: `seg ${key}`, style: `left:${pctOf(x)};width:${pctOf(ms)}`,
      title: `${label} ${ms} ms`, text: ms >= scale * 0.06 ? label : "" }));
    x += ms;
  }
  if (t.first_audio_ms !== null) {
    hTrack.append(el("div", { class: "mark", style: `left:calc(${pctOf(t.first_audio_ms)} - 1px)`,
      title: `first audio ${t.first_audio_ms} ms` }));
  }
  const ran = HARNESS_STAGES.filter(([k]) => t[k] !== null).map(([, l]) => l);
  return el("div", { class: "waterfall" },
    el("span", { class: "lbl", text: "endpoint · VAD" }), epTrack,
    el("span", { class: "tot", text: fmtMs(t.endpoint_ms) }),
    el("span", { class: "lbl", text: "harness" }), hTrack,
    el("span", { class: "tot", text: `first audio ${fmtMs(t.first_audio_ms)}` }),
    el("span", { class: "wf-note", text:
      `voice-to-voice ${fmtMs(t.voice_to_voice_ms)} (includes endpoint) · ` +
      (ran.length ? `stages: ${ran.join(", ")}` : "no harness stages recorded") +
      " · green tick = first audio" }));
}

function retrievalBlock(t) {
  const chunks = t.retrieved_chunks || [];
  const any = chunks.length || t.retrieval_top_vec_score !== null || t.retrieval_top_score !== null;
  if (!any) return null;
  return [
    el("h3", { text: `retrieval${t.retrieval_ms !== null ? ` · ${t.retrieval_ms} ms` : ""}` }),
    el("div", { class: "kv" },
      el("span", { class: "primary", title: "raw dense cosine of the best chunk — primary sufficiency signal" },
        "top vec ", el("b", { text: fmtNum(t.retrieval_top_vec_score) })),
      el("span", {}, "vec margin ", el("b", { text: fmtNum(t.retrieval_vec_margin) })),
      el("span", {}, "top lex ", el("b", { text: fmtNum(t.retrieval_top_lex_score) })),
      el("span", { class: "dim", title: "fused RRF rank-1 score — ranking provenance, weak signal" },
        "fused ", el("b", { text: fmtNum(t.retrieval_top_score, 4) })),
      el("span", { class: "dim", title: "fused(1) - fused(2); measured inverted, not a sufficiency signal" },
        "rank margin ", el("b", { text: fmtNum(t.retrieval_margin, 4) }))),
    ...chunks.map((c, i) => el("details", { class: "chunk" },
      el("summary", {}, el("span", { class: "rank", text: `#${i + 1}` }),
        c.heading_path || `(chunk ${c.chunk_id} not in corpus)`),
      el("div", { class: "snippet", text: c.snippet || "" }),
      el("div", { class: "dim small mono", text: c.chunk_id }))),
  ];
}

function verdictList(verdicts) {
  return el("div", { class: "list" }, ...verdicts.map((v) => el("div", { class: "item" },
    el("span", { class: "dim mono", text: `seg ${v.segment_index}` }),
    el("span", { class: `badge ${v.verdict.startsWith("fail") ? "fail" : v.verdict}`, text: v.verdict }),
    el("span", { class: `badge ${v.action_taken}`, text: v.action_taken }),
    v.seeded_class ? el("span", { class: "badge interrupted", text: `seeded: ${v.seeded_class}` }) : null,
    el("span", { class: "dim", text: fmtMs(v.checker_ms) }),
    el("span", { class: "text", text: v.segment_text }))));
}

function toolStates(status) {
  const path = TOOL_STATES.includes(status) ? TOOL_STATES : [...TOOL_STATES.slice(0, 2), status];
  const idx = path.indexOf(status);
  return el("span", { class: "states" }, ...path.map((s, i) =>
    el("span", { class: i === idx ? `current ${s}` : i < idx ? "reached" : "", text: s })));
}

function toolList(tools) {
  return el("div", { class: "list" }, ...tools.map((x) => el("div", { class: "item" },
    el("b", { text: x.tool_name }),
    toolStates(x.status),
    x.side_effecting ? el("span", { class: "badge handoff", text: "side-effecting" }) : el("span", { class: "badge", text: "read" }),
    x.attempt > 1 ? el("span", { class: "badge interrupted", text: `attempt ${x.attempt}` }) : null,
    el("span", { class: "dim", text: fmtMs(x.duration_ms) }),
    el("span", { class: "dim mono small", text: `key ${x.idempotency_key}`, title: JSON.stringify(x.args_canonical) }),
    x.error_type ? el("span", { class: "text", text: `${x.error_type}: ${x.error_detail || ""}` }) : null)));
}

function handoffBox(h) {
  return el("div", { class: "handoff-box" },
    el("div", {}, el("span", { class: "badge handoff", text: "handoff" }), " ",
      el("b", { text: h.reason }), el("span", { class: "dim", text: ` · ${h.rule_type}${h.score !== null ? ` · score ${fmtNum(h.score, 2)}` : ""}` })),
    el("div", { class: "dim", text: h.summary_text }));
}

// -- live feed -------------------------------------------------------------
function colorOf(type) {
  if (type === "barge_in" || type === "context_trimmed") return "amber";
  if (type === "error" || type === "budget_exceeded") return "red";
  if (type.startsWith("tool_")) return "blue";
  if (type === "checker_verdict") return "purple";
  if (type.startsWith("handoff_")) return "orange";
  if (type === "call_started" || type === "call_ended") return "green";
  return "";
}

function feedEvents() {
  if (!state.selected) return state.live;
  const merged = new Map();
  for (const e of (state.detail && state.detail.events) || []) merged.set(e.event_id, e);
  for (const e of state.live) if (e.call_id === state.selected) merged.set(e.event_id, e);
  return [...merged.values()].sort((a, b) => a.event_id - b.event_id);
}

let lastRendered = 0;
function renderFeed() {
  const feed = $("feed");
  const hidePartials = $("hide-partials").checked;
  const atBottom = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 40;
  const events = feedEvents().filter((e) => !(hidePartials && e.event_type === "stt_partial")).slice(-FEED_MAX);
  const rows = events.map((e) => {
    const payload = JSON.stringify(e.payload);
    const row = el("div", { class: `ev ${colorOf(e.event_type)}${e.event_id > lastRendered ? " fresh" : ""}` },
      el("span", { class: "t", text: fmtTime(e.ts) }),
      el("span", { class: "ty", text: e.event_type }),
      state.selected ? null : el("span", { class: "cid", text: shortId(e.call_id), title: e.call_id }),
      el("span", { class: "pl", text: payload.length > 140 ? payload.slice(0, 140) + "…" : payload }));
    row.addEventListener("click", () => {
      const open = row.querySelector("pre");
      if (open) open.remove();
      else row.append(el("pre", { text: `event ${e.event_id} · turn ${e.turn_id || "—"}\n${JSON.stringify(e.payload, null, 2)}` }));
    });
    return row;
  });
  feed.replaceChildren(...rows);
  if (!rows.length) feed.append(el("div", { class: "dim", style: "padding:10px 0", text: "waiting for events…" }));
  lastRendered = Math.max(lastRendered, ...events.map((e) => e.event_id), 0);
  if (atBottom) feed.scrollTop = feed.scrollHeight;
}
const renderFeedSoon = debounce(renderFeed, 80);

function onEvent(e) {
  if (state.seen.has(e.event_id)) return;
  state.seen.add(e.event_id);
  state.live.push(e);
  if (state.live.length > FEED_MAX) {
    for (const old of state.live.splice(0, state.live.length - FEED_MAX)) state.seen.delete(old.event_id);
  }
  const known = state.calls.some((c) => c.call_id === e.call_id);
  if (e.event_type === "call_started" || e.event_type === "call_ended" || e.event_type === "turn_completed" || !known) {
    refreshCallsSoon();
  }
  if (e.call_id === state.selected) refreshDetailSoon();
  renderFeedSoon();
}

function connectStream() {
  // EventSource reconnects by itself and sends Last-Event-ID, which the
  // server uses as the cursor — no events are skipped across a reconnect.
  const source = new EventSource("/api/stream");
  const status = $("stream-status");
  source.onopen = () => { status.textContent = "live"; status.className = "pill ok"; };
  source.onerror = () => { status.textContent = "reconnecting…"; status.className = "pill warn"; };
  source.onmessage = (msg) => {
    try { onEvent(JSON.parse(msg.data)); } catch { /* a malformed frame is not worth a crash */ }
  };
}

// -- boot ------------------------------------------------------------------
$("follow").addEventListener("click", () => {
  setFollow(!state.follow);
  if (state.follow && state.calls.length) select(state.calls[0].call_id, { user: false });
});
$("source-filter").addEventListener("change", (ev) => { state.source = ev.target.value; refreshCalls(); });
$("hide-partials").addEventListener("change", renderFeed);
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && state.selected) {
    // Escape returns to the all-calls feed.
    state.selected = null; state.detail = null; setFollow(false);
    $("feed-title").textContent = "Live events · all calls";
    $("call-head").replaceChildren();
    $("timeline").replaceChildren(el("div", { class: "empty dim", text: "Select a call." }));
    renderCalls(); renderFeed();
  }
});

refreshStats();
setInterval(refreshStats, STATS_EVERY_MS);
setInterval(renderCalls, 15000); // keep "n s ago" honest
refreshCalls().then(() => { if (!state.calls.length) renderEmpty(); });
renderFeed();
connectStream();
