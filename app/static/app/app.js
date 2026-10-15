// VoiceDesk app: Talk (text chat with the production agent), Voice, Insights,
// Results. Vanilla JS, no build step.
//
// Every piece of text that came from the server — replies, transcripts,
// chunk headings, READMEs, cluster labels — is inserted with textContent or
// createTextNode. Nothing here assigns innerHTML.
"use strict";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function append(parent, ...children) {
  for (const child of children) {
    if (child == null || child === false) continue;
    parent.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return parent;
}

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

function pct(v, digits) {
  if (!isNum(v)) return "—";
  const p = v * 100;
  const d = digits != null ? digits : (Math.abs(p) < 10 && p !== 0 ? 1 : 0);
  return p.toFixed(d) + "%";
}

function fixed(v, d = 2) { return isNum(v) ? v.toFixed(d) : "—"; }

function isRate(v) { return v && typeof v === "object" && isNum(v.value); }

// A proportion always travels with its n (docs/15 §4).
function rateLines(rate) {
  if (!isRate(rate)) return ["no data"];
  const lines = [];
  const frac = isNum(rate.successes) && isNum(rate.n) ? `${rate.successes}/${rate.n} = ` : "";
  lines.push(`${frac}${pct(rate.value)}` + (isNum(rate.n) && !frac ? ` (n=${rate.n})` : ""));
  if (Array.isArray(rate.ci95) && rate.ci95.length === 2) {
    lines.push(`95% CI [${pct(rate.ci95[0])}, ${pct(rate.ci95[1])}]`);
  }
  return lines;
}
window.vdRateLines = rateLines;

function badgeClass(word) {
  const good = ["committed", "released", "pass", "answered", "contained", "confirmed", "healthy", "cited"];
  const bad = ["failed", "replaced", "error", "unknown", "no_coverage", "tool_failure",
               "fail_ungrounded_entity", "fail_unsupported_claim", "declined"];
  const warn = ["awaiting confirmation", "interrupted", "budget_exceeded", "weak_coverage",
                "held_out_coverage", "reconciled", "cancelled", "proposed"];
  const hand = ["handoff"];
  if (good.includes(word)) return "badge good";
  if (bad.includes(word)) return "badge badc";
  if (warn.includes(word)) return "badge warnc";
  if (hand.includes(word)) return "badge hand";
  if (word === "threshold_too_strict" || word === "dispatched") return "badge info";
  return "badge";
}

function badge(word, label) { return el("span", badgeClass(word), label || String(word).replace(/_/g, " ")); }

async function getJSON(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  let body = null;
  try { body = await res.json(); } catch { body = null; }
  return { status: res.status, ok: res.ok, body };
}

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

function emptyState(...lines) {
  const box = el("div", "empty-state");
  for (const line of lines) append(box, typeof line === "string" ? el("div", null, line) : line);
  return box;
}

function commandHint() {
  const p = el("div");
  append(p, "Run ", el("code", null, "make simulate && make gap-report && make cycle"));
  return p;
}

// ---------------------------------------------------------------------------
// routing
// ---------------------------------------------------------------------------
const VIEWS = ["talk", "voice", "insights", "results"];
const loaded = new Set();

function route() {
  const hash = location.hash.replace(/^#/, "");
  const [view, ...rest] = hash.split("/");
  const name = VIEWS.includes(view) ? view : "talk";
  for (const v of VIEWS) $("view-" + v).hidden = v !== name;
  document.querySelectorAll("nav a[data-view]").forEach((a) => {
    a.classList.toggle("active", a.dataset.view === name);
    if (a.dataset.view === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  window.vdCharts.hideTip();

  if (name === "talk") Talk.ensure();
  if (name === "voice" && !loaded.has("voice")) { loaded.add("voice"); Voice.load(); }
  if (name === "insights" && !loaded.has("insights")) { loaded.add("insights"); Insights.load(); }
  if (name === "results") {
    const run = rest.length ? decodeURIComponent(rest.join("/")) : null;
    Results.show(run);
  }
}

// ---------------------------------------------------------------------------
// status (shared)
// ---------------------------------------------------------------------------
let statusPromise = null;
function loadStatus() {
  if (!statusPromise) {
    statusPromise = getJSON("/api/status").then((r) => (r.ok ? r.body : null)).catch(() => null);
  }
  return statusPromise;
}

async function paintStatus() {
  const status = await loadStatus();
  const pill = $("db-pill");
  if (!status) { pill.textContent = "status unavailable"; pill.className = "pill bad"; return; }
  if (status.database) {
    pill.textContent = isNum(status.chunks) ? `db · ${status.chunks} chunks` : "db up";
    pill.className = "pill ok";
    pill.title = "Postgres reachable — chat sessions are recorded and appear in the console";
  } else {
    pill.textContent = "no database";
    pill.className = "pill bad";
    pill.title = "Postgres unreachable — no retrieval, nothing recorded. Run `make up && make ingest`.";
  }
}

// ---------------------------------------------------------------------------
// Talk
// ---------------------------------------------------------------------------
const EXAMPLES = [
  { text: "Can I get a refund if I can't make it?" },
  { text: "What's in the VIP package?" },
  { text: "I want to send two tickets to friend@example.com", step: "transfer 1/2" },
  { text: "It's order T K 8 8 2 1 3, sam at example dot com", step: "transfer 2/2" },
  { text: "Let me talk to a manager", step: "handoff" },
];

const STAGE_ORDER = ["endpoint", "stt_final", "retrieval", "llm_ttft", "sentence_buffer", "checker",
                     "first_speak", "first_content", "first_audio"];

const Talk = {
  ws: null,
  live: false,
  ended: true,
  turns: new Map(),
  pending: null,
  awaiting: null,

  ensure() {
    if (!this.ws) this.connect();
  },

  connect() {
    this.reset();
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/chat`);
    this.ws = ws;
    this.ended = false;
    this.setState("connecting…", "live");
    ws.onmessage = (e) => {
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (ws === this.ws) this.onMessage(msg);
    };
    ws.onclose = () => {
      if (ws !== this.ws) return;
      if (!this.ended) {
        this.note("Disconnected from the agent.", true);
        this.finish("disconnected");
      }
    };
    ws.onerror = () => {};
  },

  reset() {
    this.turns.clear();
    this.pending = null;
    this.awaiting = null;
    this.live = false;
    const t = $("transcript");
    t.replaceChildren($("talk-empty"));
    $("talk-empty").hidden = false;
    $("call-id").textContent = "";
    $("call-recorded").hidden = true;
    $("stack-line").hidden = true;
    $("restart-row").hidden = true;
    $("composer").hidden = false;
    $("examples").hidden = false;
    $("end-call").hidden = true;
    this.setComposer(false);
    this.renderExamples();
  },

  setState(text, cls) {
    const pill = $("call-state");
    pill.textContent = text;
    pill.className = "pill " + (cls || "");
  },

  setComposer(enabled) {
    $("input").disabled = !enabled;
    $("send").disabled = !enabled;
    document.querySelectorAll("#examples .chip").forEach((c) => (c.disabled = !enabled));
    $("input").placeholder = this.awaiting
      ? "Reply in your own words — e.g. “yes” or “no, change the email”"
      : "Ask about refunds, tickets, the venue…";
  },

  renderExamples() {
    const box = $("examples");
    box.replaceChildren();
    for (const ex of EXAMPLES) {
      const chip = el("button", "chip");
      chip.type = "button";
      if (ex.step) chip.appendChild(el("span", "step", ex.step));
      chip.appendChild(document.createTextNode(ex.text));
      chip.disabled = true;
      chip.onclick = () => this.send(ex.text);
      box.appendChild(chip);
    }
  },

  scroll() {
    const t = $("transcript");
    t.scrollTop = t.scrollHeight;
  },

  note(text, isError) {
    const n = el("div", "system-note" + (isError ? " err" : ""), text);
    $("transcript").appendChild(n);
    this.scroll();
  },

  send(text) {
    text = (text || "").trim();
    if (!text || !this.live || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    $("talk-empty").hidden = true;
    const msg = el("div", "msg user");
    msg.appendChild(el("div", "bubble", text));
    $("transcript").appendChild(msg);
    this.ws.send(JSON.stringify({ type: "user", text }));
    $("input").value = "";
    autosize();
    if (!this.pending) this.pending = this.agentMessage();
    this.scroll();
  },

  agentMessage() {
    const msg = el("div", "msg agent");
    const bubble = el("div", "bubble pending");
    msg.appendChild(bubble);
    $("transcript").appendChild(msg);
    return { msg, bubble, segments: 0, details: null };
  },

  turnEntry(turnId) {
    if (!turnId) return this.pending || (this.pending = this.agentMessage());
    let entry = this.turns.get(turnId);
    if (!entry) {
      entry = this.pending || this.agentMessage();
      this.pending = null;
      this.turns.set(turnId, entry);
    }
    return entry;
  },

  onMessage(msg) {
    switch (msg.type) {
      case "status":
        this.setState("connecting…", "live");
        $("stack-line").textContent = msg.detail || "";
        $("stack-line").hidden = !msg.detail;
        break;
      case "ready": {
        this.live = true;
        this.setState("live", "ok");
        $("call-id").textContent = "call " + String(msg.call_id || "").slice(0, 10);
        $("call-id").title = msg.call_id || "";
        const rec = $("call-recorded");
        rec.hidden = false;
        rec.textContent = msg.recorded ? "recorded · source text" : "not recorded";
        rec.className = "pill " + (msg.recorded ? "ok" : "warn");
        rec.title = msg.recorded ? "Saved to Postgres; visible in the console" : "No database — kept in memory only";
        $("stack-line").textContent = (msg.stack || []).join(" · ");
        $("stack-line").hidden = !(msg.stack || []).length;
        $("end-call").hidden = false;
        this.paintAgent(msg.agent);
        this.setComposer(true);
        $("input").focus();
        break;
      }
      case "turn_started":
        this.turnEntry(msg.turn_id);
        break;
      case "say": {
        const entry = this.turnEntry(msg.turn_id);
        if (entry.segments) entry.bubble.appendChild(document.createTextNode(" "));
        entry.bubble.appendChild(el("span", "seg", msg.text));
        entry.segments++;
        this.scroll();
        break;
      }
      case "turn":
        this.onTurn(msg.turn_id, msg.details || {});
        break;
      case "handoff":
        this.onHandoff(msg);
        break;
      case "ended":
        this.finish(msg.outcome);
        break;
      case "error":
        this.note(msg.message || "Something went wrong.", true);
        if (msg.code === "internal") this.finish("error");
        break;
      default:
        break; // "event" frames feed the console; the panel uses the turn summary.
    }
  },

  paintAgent(agent) {
    const banner = $("agent-banner");
    banner.replaceChildren();
    if (!agent) { banner.hidden = true; return; }
    if (agent.offline) {
      banner.className = "banner";
      append(banner, el("b", null, "Offline agent"), " — answers are quoted from the knowledge base; set ",
             el("code", null, "ANTHROPIC_API_KEY"), " for Claude.");
      banner.title = agent.notice || "";
    } else {
      banner.className = "banner claude";
      // Name the family that actually runs — never assume Claude.
      const model = agent.model || "";
      const family = model.startsWith("gemini") ? "Gemini"
        : model.startsWith("claude") ? "Claude"
        : model.startsWith("gpt") || agent.provider === "openai" ? "OpenAI" : "Live model";
      const where = agent.provider.startsWith("vertex_") ? " on Google Cloud Vertex AI" : "";
      append(banner, el("b", null, family), ` — ${model}${where}`);
    }
    banner.hidden = false;
  },

  onTurn(turnId, d) {
    const entry = this.turnEntry(turnId);
    entry.bubble.classList.remove("pending");
    if (!entry.segments) entry.bubble.appendChild(el("span", "dim", "(no reply)"));
    // Acknowledgments and safe lines are fixed templates, not the agent's
    // answer: they are spoken but never enter approved_text.
    if (typeof d.approved_text === "string") {
      entry.bubble.querySelectorAll(".seg").forEach((seg) => {
        const filler = !d.approved_text.includes(seg.textContent);
        seg.classList.toggle("filler", filler);
        if (filler) seg.title = "fixed line — not checked, not part of the answer";
      });
    }

    this.awaiting = d.awaiting_confirmation || null;
    entry.msg.classList.toggle("confirm", !!this.awaiting);
    if (this.awaiting) {
      entry.bubble.prepend(el("span", "confirm-head", "Confirmation needed · " +
                                String(this.awaiting.tool || "").replace(/_/g, " ")));
      entry.bubble.appendChild(el("div", "confirm-hint",
        "Nothing has changed yet. Type your answer — “yes”, “no”, or a correction."));
    }
    entry.msg.appendChild(detailsPanel(d));
    this.setComposer(this.live);
    this.scroll();
  },

  onHandoff(msg) {
    const card = el("div", "handoff-card");
    append(card, el("h3", null, "Handed off to a human agent"));
    const facts = el("div", "facts");
    append(facts, el("span", null, "reason "), el("b", null, String(msg.reason || "—").replace(/_/g, " ")));
    if (msg.rule_type) append(facts, el("span", null, `${msg.rule_type} rule`));
    card.appendChild(facts);
    append(card, el("div", "dim small", "What the human sees before they say hello:"));
    card.appendChild(el("pre", null, msg.summary_text || "(no summary)"));
    $("transcript").appendChild(card);
    this.scroll();
  },

  finish(outcome) {
    this.ended = true;
    this.live = false;
    this.awaiting = null;
    if (this.pending) {
      this.pending.bubble.classList.remove("pending");
      if (!this.pending.segments) this.pending.msg.remove();
      this.pending = null;
    }
    this.setState("ended · " + String(outcome || "closed"), outcome === "error" ? "bad" : "");
    this.note(`Call ended — outcome: ${outcome || "closed"}.`);
    this.setComposer(false);
    $("composer").hidden = true;
    $("examples").hidden = true;
    $("end-call").hidden = true;
    // Not focused: a handoff can end the call within the same Enter key press
    // that sent the message, and a focused button would catch its keypress.
    $("restart-row").hidden = false;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.close();
  },

  hangUp() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "end" }));
  },
};

function detailsPanel(d) {
  const wrap = el("details", "why");
  const summary = el("summary");
  const r = d.retrieval || {};
  const chunks = r.chunks || [];
  const cited = chunks.filter((c) => c.cited).length;
  const checker = d.checker || [];
  const replaced = checker.filter((c) => c.action === "replaced").length;
  const tools = d.tools || [];
  const stages = d.stages_ms || {};
  const firstContent = isNum(stages.first_content) ? stages.first_content : stages.first_speak;

  const bits = ["details"];
  if (chunks.length) bits.push(`${chunks.length} chunks${cited ? ` (${cited} cited)` : ""}`);
  if (checker.length) bits.push(replaced ? `checker: ${replaced}/${checker.length} replaced`
                                         : `checker: ${checker.length}/${checker.length} released`);
  for (const t of tools) bits.push(`${t.tool}: ${t.status}`);
  if (isNum(r.top_vec_score)) bits.push(`top vec ${fixed(r.top_vec_score)}`);
  if (isNum(firstContent)) bits.push(`${firstContent} ms`);
  summary.appendChild(el("span", null, bits.join(" · ")));
  if (d.stop_reason && d.stop_reason !== "answered") summary.appendChild(badge(d.stop_reason));
  wrap.appendChild(summary);

  const body = el("div", "why-body");
  wrap.appendChild(body);

  // -- knowledge
  const h = el("h4", null, "Knowledge retrieved");
  h.appendChild(el("span", "scope", "this call"));
  body.appendChild(h);
  const kv = el("div", "kv small");
  append(kv,
    append(el("span"), "top vec ", el("b", null, fixed(r.top_vec_score, 3)),
           el("span", "dim", " (sufficiency signal)")),
    append(el("span"), "top lex ", el("b", null, fixed(r.top_lex_score, 3))),
    append(el("span"), "fused ", el("b", null, fixed(r.top_score, 4))));
  body.appendChild(kv);
  if (!chunks.length) body.appendChild(el("div", "dim small", "No chunks retrieved."));
  for (const c of chunks) {
    const row = el("div", "row small");
    append(row, el("span", "rank", "#" + c.rank), el("span", "grow", c.heading || c.chunk_id),
           c.cited ? badge("cited") : null);
    body.appendChild(row);
  }

  // -- checker
  body.appendChild(el("h4", null, "Grounding checker"));
  if (!checker.length) body.appendChild(el("div", "dim small", "No model-written segments were checked this turn."));
  for (const v of checker) {
    const row = el("div", "row small");
    const text = el("span", "grow" + (v.action === "replaced" ? " withheld" : ""), v.text);
    append(row, badge(v.action), badge(v.verdict), text);
    if (v.reason) row.appendChild(el("span", "dim", v.reason));
    body.appendChild(row);
  }

  // -- tools
  if (tools.length) {
    body.appendChild(el("h4", null, "Tool calls"));
    for (const t of tools) {
      const row = el("div", "row small");
      append(row, el("b", null, t.tool), badge(t.status));
      if (t.error) row.appendChild(badge("error", t.error));
      if (t.duplicate_suppressed) row.appendChild(badge("cancelled", "duplicate suppressed"));
      if (t.args) row.appendChild(el("span", "args grow", JSON.stringify(t.args)));
      body.appendChild(row);
    }
  }

  // -- signals
  if ((d.signals || []).length) {
    body.appendChild(el("h4", null, "Handoff signals"));
    const row = el("div", "row small");
    for (const s of d.signals) row.appendChild(badge("handoff", s.replace(/_/g, " ")));
    body.appendChild(row);
  }

  // -- latency
  const keys = Object.keys(stages).sort((a, b) => {
    const ia = STAGE_ORDER.indexOf(a), ib = STAGE_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  if (keys.length) {
    const lh = el("h4", null, "Latency stages (ms)");
    lh.appendChild(el("span", "dim", "text mode — no audio stages"));
    body.appendChild(lh);
    const max = Math.max(...keys.map((k) => stages[k] || 0), 1);
    const grid = el("div", "stages small");
    for (const k of keys) {
      const bar = el("div", "bar");
      bar.style.width = `${Math.max(1, (100 * (stages[k] || 0)) / max)}%`;
      const track = el("div");
      track.appendChild(bar);
      append(grid, el("span", "dim", k.replace(/_/g, " ")), track, el("span", "v", String(stages[k])));
    }
    body.appendChild(grid);
  }
  const meta = el("div", "dim small");
  meta.style.marginTop = "8px";
  meta.textContent = `stop reason: ${d.stop_reason || "—"} · LLM calls: ${d.llm_iterations ?? "—"}` +
    (d.ack_bypass_used ? " · acknowledgment spoken" : "") + (d.budget_exceeded ? " · budget exceeded" : "");
  body.appendChild(meta);
  return wrap;
}

function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = Math.min(140, t.scrollHeight) + "px";
}

// ---------------------------------------------------------------------------
// Voice
// ---------------------------------------------------------------------------
const Voice = {
  async load() {
    const body = $("voice-body");
    const status = await loadStatus();
    body.replaceChildren();
    if (!status) {
      body.appendChild(emptyState("Could not read /api/status."));
      return;
    }
    const v = status.voice || {};
    if (v.ready) {
      const p = el("p", "small");
      const a = el("a", null, "open it in its own tab ↗");
      a.href = "/voice";
      a.target = "_blank";
      append(p, "Allow microphone access when asked, or ", a, ".");
      const frame = el("iframe", "voice-frame");
      frame.src = "/voice";
      frame.title = "Voice client";
      frame.allow = "microphone; autoplay";
      body.append(p, frame);
      return;
    }
    const card = el("div", "card");
    append(card, el("h3", null, "Voice needs a few things this server doesn't have yet"));
    const list = el("ul", "need");
    for (const name of v.missing || []) {
      const li = el("li");
      append(li, el("code", null, name), " — add it to ", el("code", null, ".env"));
      list.appendChild(li);
    }
    if (!v.audio_extra) {
      const li = el("li");
      append(li, "the audio extra — ", el("code", null, ".venv/bin/pip install -e '.[audio]'"));
      list.appendChild(li);
    }
    card.appendChild(list);
    const p = el("p", "dim small");
    append(p, "Then restart the server. The pipeline uses Deepgram for streaming speech-to-text and ",
           "Cartesia for text-to-speech; the conversation itself runs in the same harness as ",
           el("b", null, "Talk"), ", which needs none of these — try it there.");
    card.appendChild(p);
    const p2 = el("p", "small");
    const a = el("a", null, "Open the voice client anyway ↗");
    a.href = "/voice";
    p2.appendChild(a);
    card.appendChild(p2);
    body.appendChild(card);
  },
};

// ---------------------------------------------------------------------------
// Insights
// ---------------------------------------------------------------------------
const DIAGNOSIS = {
  no_coverage: "No coverage — the knowledge base has nothing on this",
  held_out_coverage: "Held-out coverage — a document exists but is not ingested",
  weak_coverage: "Weak coverage — retrieved, but scores too low to answer",
  threshold_too_strict: "Threshold too strict — good matches still handed off",
  tool_failure: "Tool failure — the backend, not the knowledge",
  healthy: "Healthy",
};

const Insights = {
  async load() {
    this.gap();
    this.cycle();
  },

  async gap() {
    const body = $("gap-body");
    let res;
    try { res = await getJSON("/api/gap-report/latest"); } catch { res = { status: 0 }; }
    body.replaceChildren();
    if (res.status === 503) {
      body.appendChild(emptyState("The database is unavailable, so there is no gap report to read.",
                                  append(el("div"), "Start it with ", el("code", null, "make up"), ", then:"),
                                  commandHint()));
      return;
    }
    if (!res.ok || !res.body) {
      body.appendChild(emptyState("No gap report yet.", commandHint()));
      return;
    }
    const report = res.body;
    const clusters = Array.isArray(report.clusters) ? report.clusters.slice() : [];
    const facts = el("div", "facts");
    append(facts,
      append(el("span"), "report ", el("b", null, report.report_id || "—")),
      append(el("span"), "generated ", el("b", null, fmtDate(report.generated_at))),
      append(el("span"), "window ", el("b", null, `${fmtDate(report.window_start)} → ${fmtDate(report.window_end)}`)),
      append(el("span"), "calls ", el("b", null, `n=${report.call_count ?? "—"}`)),
      append(el("span"), "clusters ", el("b", null, String(clusters.length))));
    body.appendChild(facts);

    if (!clusters.length) {
      body.appendChild(emptyState("This report has no clusters."));
      return;
    }
    if (clusters.every((c) => isNum(c.priority))) clusters.sort((a, b) => b.priority - a.priority);
    const ranked = clusters.filter((c) => isNum(c.priority) && c.priority > 0);
    const rest = clusters.filter((c) => !(isNum(c.priority) && c.priority > 0));

    // What to do first: the report's own ranked actions.
    const actions = (Array.isArray(report.actions) ? report.actions : [])
      .slice().sort((x, y) => (x.rank ?? 99) - (y.rank ?? 99));
    if (actions.length) {
      const card = el("div", "card");
      card.style.marginTop = "12px";
      append(card, el("h3", null, "Ranked actions"),
             el("div", "dim small", "What the report says to change, in order. Troubled calls are of n=" +
                `${report.call_count ?? "—"} in the window.`));
      const top = actions.slice(0, 5);
      for (const a of top) card.appendChild(actionRow(a, report.call_count));
      if (actions.length > top.length) {
        const more = el("details", "raw small");
        more.appendChild(el("summary", null, `${actions.length - top.length} more actions`));
        for (const a of actions.slice(top.length)) more.appendChild(actionRow(a, report.call_count));
        card.appendChild(more);
      }
      body.appendChild(card);
    }

    if (ranked.length) {
      const chartCard = el("div", "card");
      chartCard.style.marginTop = "12px";
      append(chartCard, el("h3", null, "Cluster priority"),
             el("div", "dim small", "Higher = fix first. Hover a bar for its diagnosis and n." +
                (rest.length ? ` ${rest.length} clusters at priority 0 are not drawn.` : "")));
      body.appendChild(chartCard);
      window.vdCharts.hbar(chartCard, ranked.map((c) => ({
        label: clusterName(c),
        value: c.priority,
        valueLabel: fixed(c.priority, 1),
        tip: [clusterName(c),
              `priority ${fixed(c.priority, 2)}`,
              String(c.diagnosis || "—").replace(/_/g, " "),
              `failure rate ${pct(c.failure_rate)} of n=${c.n_calls ?? "—"} calls`],
      })), { title: "Cluster priority", tickFormat: (v) => fixed(v, v < 10 ? 1 : 0) });
    }
    body.appendChild(tableView(
      ["#", "cluster", "diagnosis", "priority", "failure rate", "n calls", "share"],
      clusters.map((c, i) => [String(i + 1), clusterName(c), String(c.diagnosis || "—"),
                              fixed(c.priority, 2), pct(c.failure_rate), String(c.n_calls ?? "—"),
                              pct(c.share)])));

    ranked.forEach((c, i) => body.appendChild(clusterCard(c, i + 1, report.call_count)));
    if (rest.length) {
      const more = el("details", "raw");
      more.style.marginTop = "12px";
      more.appendChild(el("summary", null, `${rest.length} clusters with priority 0 ` +
        `(${rest.filter((c) => c.diagnosis === "healthy").length} healthy)`));
      rest.forEach((c, i) => more.appendChild(clusterCard(c, ranked.length + i + 1, report.call_count)));
      body.appendChild(more);
    }
  },

  async cycle() {
    const body = $("cycle-body");
    let res;
    try { res = await getJSON("/api/feedback-cycle/latest"); } catch { res = { status: 0 }; }
    body.replaceChildren();
    if (!res.ok || !res.body || !res.body.metrics) {
      body.appendChild(emptyState("No feedback cycle yet.", commandHint()));
      return;
    }
    const m = res.body.metrics;
    const facts = el("div", "facts");
    const runLink = el("a", null, res.body.run);
    runLink.href = "#results/" + encodeURIComponent(res.body.run);
    let seeds = null;
    if (Array.isArray(m.seeds)) seeds = `${m.seeds.length} seeds`;
    else if (m.seeds && typeof m.seeds === "object") {
      seeds = `seed ${m.seeds.seed ?? "—"}` + (m.seeds.identical ? " · identical both runs" : "");
    }
    append(facts,
      append(el("span"), "run ", runLink),
      append(el("span"), "calls per run ", el("b", null, `n=${m.n_calls ?? "—"}`)),
      seeds ? append(el("span"), el("b", null, seeds)) : null,
      m.agent ? append(el("span"), "agent ", el("b", null, String(m.agent) +
        (m.agent === "extractive" ? " (offline)" : ""))) : null,
      m.corpus_hash_before ? append(el("span"), "corpus ", el("b", null,
        `${String(m.corpus_hash_before).slice(0, 8)} → ${String(m.corpus_hash_after || "—").slice(0, 8)}`)) : null);
    body.appendChild(facts);

    const change = m.change || {};
    const ch = el("div", "card");
    ch.style.marginTop = "12px";
    append(ch, el("h3", null, "The change that was made"));
    const row = el("div", "row");
    append(row, badge("dispatched", String(change.type || "—").replace(/_/g, " ")));
    if (change.target) append(row, el("code", null, String(change.target)));
    ch.appendChild(row);
    if (change.source) ch.appendChild(el("div", "dim small", `from ${change.source}`));
    if (Array.isArray(change.skipped_higher_ranked) && change.skipped_higher_ranked.length) {
      ch.appendChild(el("div", "dim small", "skipped higher-ranked: " + change.skipped_higher_ranked
        .map((x) => `#${x.rank} ${String(x.diagnosis || "").replace(/_/g, " ")} ${x.target || ""}`).join("; ")));
    }
    body.appendChild(ch);

    const before = m.before || {}, after = m.after || {};
    const metrics = [
      { key: "containment", label: "Containment", up: true, role: "headline" },
      { key: "resolved_rate", label: "Resolved", up: true, role: "headline" },
      { key: "handoff_rate", label: "Handoff rate", up: false, role: "guardrail" },
      { key: "wrong_answer_rate", label: "Wrong answers", up: false, role: "guardrail" },
    ].filter((spec) => isRate(before[spec.key]) || isRate(after[spec.key]));

    const tiles = el("div", "nums");
    tiles.style.marginTop = "12px";
    for (const spec of metrics) tiles.appendChild(deltaTile(spec, before[spec.key], after[spec.key]));
    const dup = m.duplicate_side_effects;
    if (dup && typeof dup === "object") {
      const t = el("div", "num");
      t.style.minWidth = "180px";
      const bad = (dup.after ?? 0) > 0;
      append(t, el("div", "k", "Duplicate side effects · guardrail"),
             el("div", "v " + (bad ? "delta-bad" : ""), `${dup.before ?? "—"} → ${dup.after ?? "—"}`),
             el("div", "n", "count, both runs; must stay 0"));
      tiles.appendChild(t);
    }
    body.appendChild(tiles);
    if (isRate(before.wrong_answer_rate)) {
      body.appendChild(el("div", "dim small", "Wrong answers are measured over contained calls only, so their n differs."));
    }

    if (metrics.length) {
      const chartCard = el("div", "card");
      chartCard.style.marginTop = "12px";
      append(chartCard, el("h3", null, "Before vs after"),
             el("div", "dim small", "Same simulated callers, same seed. The headline rates should rise; the guardrails should not."));
      body.appendChild(chartCard);
      window.vdCharts.beforeAfter(chartCard, metrics.map((spec) => ({
        label: spec.label, note: spec.role, before: before[spec.key], after: after[spec.key],
      })), { title: "Before and after the change" });
      body.appendChild(tableView(["metric", "before", "after"], metrics.map((spec) => [
        spec.label, rateLines(before[spec.key]).join(" · "), rateLines(after[spec.key]).join(" · ")])));
    }

    const paired = m.paired;
    if (paired && typeof paired === "object") {
      const card = el("div", "card");
      card.style.marginTop = "12px";
      append(card, el("h3", null, "Paired comparison"),
             el("div", "dim small", "The same caller before and after — gains and losses per call, McNemar exact test."));
      const rows = [];
      for (const [k, v] of Object.entries(paired)) {
        if (v && typeof v === "object") {
          rows.push([k.replace(/_/g, " "), `gained ${v.gained ?? "—"} · lost ${v.lost ?? "—"}`,
                     isNum(v.mcnemar_exact_p) ? `p = ${v.mcnemar_exact_p.toExponential(2)}` : ""]);
        } else {
          rows.push([k.replace(/_/g, " "), String(v), ""]);
        }
      }
      card.appendChild(plainTable(["outcome", "calls", "test"], rows));
      body.appendChild(card);
    }

    const intents = m.by_intent && typeof m.by_intent === "object" ? Object.entries(m.by_intent) : [];
    if (intents.length) {
      const card = el("div", "card");
      card.style.marginTop = "12px";
      card.appendChild(el("h3", null, "Containment by intent"));
      const target = new Set();
      const frac = (x, key) => (x && isNum(x[key]) && isNum(x.n) && x.n ? `${x[key]}/${x.n} = ${pct(x[key] / x.n)}` : "—");
      const rows = intents.map(([intent, v]) => {
        const b = (v && v.before) || {}, a = (v && v.after) || {};
        if (isNum(a.contained) && isNum(b.contained) && a.contained !== b.contained) target.add(intent);
        return [intent.replace(/_/g, " "), frac(b, "contained"), frac(a, "contained")];
      });
      card.appendChild(plainTable(["intent", "before (contained/n)", "after (contained/n)"], rows));
      if (target.size) card.appendChild(el("div", "dim small", "Changed: " + [...target].map((t) => t.replace(/_/g, " ")).join(", ")));
      body.appendChild(card);
    }

    const acc = m.gap_report_accuracy;
    if (acc && typeof acc === "object") {
      const card = el("div", "card");
      card.style.marginTop = "12px";
      append(card, el("h3", null, "Was the gap report right?"));
      if (acc.note) card.appendChild(el("div", "dim small", String(acc.note)));
      card.appendChild(plainTable(["measure", "value"], Object.entries(acc)
        .filter(([k]) => k !== "note")
        .map(([k, v]) => [k.replace(/_/g, " "), describeValue(v)])));
      body.appendChild(card);
    }
  },
};

// Render an arbitrary metrics.json value as one readable line.
function describeValue(v) {
  if (v == null) return "—";
  if (isRate(v)) return rateLines(v).join(" · ");
  if (Array.isArray(v)) return v.length ? v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ") : "none";
  if (typeof v === "object") {
    return Object.entries(v).map(([k, x]) => `${k.replace(/_/g, " ")} ${isNum(x) ? (Number.isInteger(x) ? x : fixed(x)) : describeValue(x)}`).join(" · ");
  }
  if (isNum(v)) return Number.isInteger(v) ? String(v) : fixed(v, 3);
  return String(v);
}

function deltaTile(spec, b, a) {
  const tile = el("div", "num");
  tile.style.minWidth = "180px";
  append(tile, el("div", "k", `${spec.label} · ${spec.role}`));
  const v = el("div", "v", `${pct(b && b.value)} → ${pct(a && a.value)}`);
  tile.appendChild(v);
  if (isRate(a) && isRate(b)) {
    const d = (a.value - b.value) * 100;
    const good = spec.up ? d > 0 : d < 0;
    const span = el("div", "n " + (d === 0 ? "" : good ? "delta-good" : "delta-bad"),
                    `${d > 0 ? "+" : ""}${d.toFixed(1)} pts ${d === 0 ? "" : good ? "(better)" : "(worse)"}`);
    tile.appendChild(span);
  }
  tile.appendChild(el("div", "n", `n=${(b && b.n) ?? "—"} before · n=${(a && a.n) ?? "—"} after`));
  return tile;
}

function plainTable(headers, rows) {
  const wrap = el("div", "scroll-x");
  const t = el("table", "t");
  const tr = el("tr");
  for (const h of headers) tr.appendChild(el("th", null, h));
  append(t, append(el("thead"), tr));
  const tb = el("tbody");
  for (const r of rows) {
    const row = el("tr");
    for (const c of r) row.appendChild(el("td", null, c));
    tb.appendChild(row);
  }
  t.appendChild(tb);
  wrap.appendChild(t);
  return wrap;
}

// The table view every chart carries, for screen readers and exact values.
function tableView(headers, rows) {
  const d = el("details", "raw small");
  d.style.marginTop = "6px";
  d.appendChild(el("summary", null, "table view"));
  d.appendChild(plainTable(headers, rows));
  return d;
}

function clusterName(c) {
  const label = c.label || (c.cluster_id != null ? `cluster ${c.cluster_id}` : "cluster");
  return c.majority_intent && label === "miscellaneous" ? `miscellaneous (mostly ${c.majority_intent.replace(/_/g, " ")})` : label;
}

function actionRow(a, callCount) {
  const row = el("div", "cluster");
  row.style.borderTop = "1px solid var(--line)";
  row.style.paddingTop = "8px";
  const head = el("div", "cluster-head");
  append(head, el("span", "pos", "#" + (a.rank ?? "—")), badge(a.diagnosis || "unknown"));
  if (a.target) head.appendChild(el("code", null, String(a.target)));
  if (isNum(a.priority)) head.appendChild(el("span", "dim small", `priority ${fixed(a.priority, 1)}`));
  if (isNum(a.troubled_calls)) {
    head.appendChild(el("span", "dim small",
      `${a.troubled_calls} troubled calls of n=${callCount ?? "—"}` + (isNum(a.share_of_calls) ? ` (${pct(a.share_of_calls)})` : "")));
  }
  row.appendChild(head);
  row.appendChild(el("div", "small", String(a.action || "—")));
  if (Array.isArray(a.example_utterances) && a.example_utterances.length) {
    const ul = el("ul", "quotes small");
    for (const u of a.example_utterances.slice(0, 2)) ul.appendChild(el("li", null, `“${u}”`));
    row.appendChild(ul);
  }
  return row;
}

function clusterCard(c, position, callCount) {
  const card = el("div", "card cluster");
  const head = el("div", "cluster-head");
  append(head, el("span", "pos", "#" + position), el("h3", null, clusterName(c)),
         badge(c.diagnosis || "unknown"));
  card.appendChild(head);
  if (DIAGNOSIS[c.diagnosis]) card.appendChild(el("div", "dim small", DIAGNOSIS[c.diagnosis]));

  const nums = el("div", "nums");
  nums.style.marginTop = "8px";
  const failures = isNum(c.failure_rate) && isNum(c.n_calls) ? Math.round(c.failure_rate * c.n_calls) : null;
  const tile = (k, v, n) => append(el("div", "num"), el("div", "k", k), el("div", "v", v), n ? el("div", "n", n) : null);
  append(nums,
    tile("failure rate", pct(c.failure_rate), failures != null ? `${failures}/${c.n_calls} calls` : `n=${c.n_calls ?? "—"}`),
    tile("calls", String(c.n_calls ?? "—"), isNum(c.share) ? `${pct(c.share)} of n=${callCount ?? "—"}` : null),
    tile("priority", fixed(c.priority, 2)),
    isNum(c.trouble_rate) ? tile("trouble rate", pct(c.trouble_rate), `of n=${c.n_calls ?? "—"}`) : null,
    tile("median top vec", fixed(c.median_top_vec_score, 3), "retrieval sufficiency"),
    c.dominant_handoff_reason ? tile("main handoff reason", String(c.dominant_handoff_reason).replace(/_/g, " ")) : null);
  card.appendChild(nums);

  if (Array.isArray(c.keywords) && c.keywords.length) {
    const kw = el("div", "kw");
    for (const k of c.keywords.slice(0, 10)) kw.appendChild(el("span", "badge", String(k)));
    card.appendChild(kw);
  }
  if (Array.isArray(c.example_utterances) && c.example_utterances.length) {
    const ul = el("ul", "quotes");
    for (const u of c.example_utterances.slice(0, 4)) ul.appendChild(el("li", null, `“${u}”`));
    card.appendChild(ul);
  }
  const docs = Array.isArray(c.covering_docs) ? c.covering_docs : [];
  const extra = el("div", "dim small");
  extra.style.marginTop = "6px";
  if (docs.length) extra.appendChild(el("div", null,
    "covering docs: " + docs.map((d) => `${d.doc} (${d.count})`).join(", ")));
  const held = c.held_out_match;
  if (held && held.doc) {
    const how = isNum(held.similarity) ? `similarity ${fixed(held.similarity, 3)}`
      : isNum(held.calls) ? `best match for ${held.calls} of n=${c.n_calls ?? "—"} calls` : "";
    extra.appendChild(el("div", null, `held-out match: ${held.doc}` + (how ? ` (${how})` : "")));
  }
  if (c.majority_intent) extra.appendChild(el("div", null, `majority intent: ${String(c.majority_intent).replace(/_/g, " ")}`));
  if (extra.childNodes.length) card.appendChild(extra);
  if (c.suggested_action) {
    const act = el("div", "action small");
    append(act, el("b", null, "Suggested action: "), String(c.suggested_action));
    card.appendChild(act);
  }
  return card;
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
function headlineTiles(numbers) {
  const box = el("div", "nums");
  for (const h of numbers || []) {
    const tile = el("div", "num");
    const frac = isNum(h.successes) && isNum(h.n) ? `${h.successes}/${h.n}` : `n=${h.n ?? "—"}`;
    const ci = Array.isArray(h.ci95) ? ` · CI [${fixed(h.ci95[0])}, ${fixed(h.ci95[1])}]` : "";
    append(tile, el("div", "k", h.label), el("div", "v", fixed(h.value)), el("div", "n", frac + ci));
    box.appendChild(tile);
  }
  return box;
}

const Results = {
  cards: null,

  async show(name) {
    $("results-list").hidden = !!name;
    $("result-detail").hidden = !name;
    if (name) return this.detail(name);
    if (!this.cards) await this.list();
  },

  async list() {
    const grid = $("results-grid");
    let res;
    try { res = await getJSON("/api/results"); } catch { res = { ok: false }; }
    grid.replaceChildren();
    if (!res.ok || !res.body || !(res.body.runs || []).length) {
      grid.appendChild(emptyState("No run directories under results/ yet."));
      return;
    }
    this.cards = res.body.runs;
    for (const run of this.cards) {
      const card = el("button", "run-card");
      card.type = "button";
      card.onclick = () => { location.hash = "#results/" + encodeURIComponent(run.name); };
      append(card, el("div", "meta", `${run.date || ""}  ${run.name}`));
      card.appendChild(el("h3", null, run.title || run.name));
      if (run.summary) card.appendChild(el("div", "sum", run.summary));
      else if (!run.has_readme) card.appendChild(el("div", "sum", "No README — metrics only."));
      if (run.headline && run.headline.length) card.appendChild(headlineTiles(run.headline));
      const flags = [];
      if (run.flags && run.flags.sampled) flags.push(badge("interrupted", "sampled — not a full result"));
      if (run.flags && run.flags.timing_valid === false) flags.push(badge("interrupted", "timing not valid"));
      if (run.pngs && run.pngs.length) flags.push(badge("", `${run.pngs.length} figure${run.pngs.length > 1 ? "s" : ""}`));
      if (flags.length) card.appendChild(append(el("div", "row"), ...flags));
      grid.appendChild(card);
    }
  },

  async detail(name) {
    const box = $("result-detail");
    box.replaceChildren(el("p", "dim", "Loading…"));
    let res;
    try { res = await getJSON("/api/results/" + encodeURIComponent(name)); } catch { res = { ok: false }; }
    box.replaceChildren();
    const back = el("a", "back", "← all runs");
    back.href = "#results";
    box.appendChild(back);
    if (!res.ok || !res.body) {
      box.appendChild(emptyState(`No run named “${name}”.`));
      return;
    }
    const run = res.body;
    const files = (run.pngs || []).map((f) => `/api/results/${encodeURIComponent(run.name)}/files/${encodeURIComponent(f)}`);
    append(box, el("div", "meta mono small dim", `results/${run.name}/`));
    if (run.headline && run.headline.length) {
      const h = headlineTiles(run.headline);
      h.style.margin = "10px 0 4px";
      box.appendChild(h);
      box.appendChild(el("div", "dim small", "Headline numbers from metrics.json — the README has the context."));
    }
    const md = el("article", "md");
    if (run.readme) {
      md.appendChild(window.renderMarkdown(run.readme, {
        resolveLink: (href) => {
          const clean = href.replace(/^\.\//, "");
          const i = (run.pngs || []).indexOf(clean);
          return i >= 0 ? files[i] : null;
        },
      }));
    } else {
      md.appendChild(el("p", "dim", "This run has no README.md."));
    }
    box.appendChild(md);
    (run.pngs || []).forEach((f, i) => {
      const fig = el("figure", "figure");
      const img = el("img");
      img.src = files[i];
      img.alt = f;
      img.loading = "lazy";
      append(fig, img, el("figcaption", null, f));
      box.appendChild(fig);
    });
    if (run.metrics) {
      const raw = el("details", "raw");
      raw.appendChild(el("summary", null, "metrics.json"));
      raw.appendChild(el("pre", null, JSON.stringify(run.metrics, null, 2)));
      box.appendChild(raw);
    }
    window.scrollTo(0, 0);
  },
};

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  Talk.send($("input").value);
});
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    Talk.send($("input").value);
  }
});
$("input").addEventListener("input", autosize);
$("restart").addEventListener("click", () => { Talk.ws = null; Talk.connect(); });
$("end-call").addEventListener("click", () => Talk.hangUp());
window.addEventListener("hashchange", route);
paintStatus();
route();
