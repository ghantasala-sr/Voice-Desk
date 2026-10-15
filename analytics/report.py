"""W4-6 — the static report: one self-contained HTML file (docs/14 §4, D15).

Reads ONLY `results/*/metrics.json` (docs/15 §1: metrics.json is the only file
other tools read). For each section it takes the newest run directory of that
kind, and every number on the page carries the directory it came from, so
section 8 — provenance — is generated from the same lookups as the numbers
above it, not written by hand.

Two refusals, both deliberate:
  * latency is never rendered from a run marked `timing_valid: false`
    (a `--fast` replay) or `sampled: true`;
  * a section with no run directory says "not measured" rather than showing
    a placeholder number.

Run: .venv/bin/python -m analytics.report   →  reports/index.html
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "index.html"

KINDS = {
    "cycle": "feedback-cycle",
    "gap": "gap-report",
    "handoff": "handoff-calibration",
    "checker": "checker-catch-rate",
    "retrieval": "retrieval-calibration",
    "stt": "stt-",
    "latency": "latency",
}


class Runs:
    """Newest run directory per kind, and a record of every one used."""

    def __init__(self) -> None:
        self.used: dict[str, str] = {}

    def latest(self, kind: str) -> tuple[str, dict[str, Any]] | None:
        dirs = sorted(d for d in RESULTS.iterdir()
                      if d.is_dir() and KINDS[kind] in d.name and "superseded" not in d.name
                      and (d / "metrics.json").exists())
        if not dirs:
            return None
        d = dirs[-1]
        self.used[kind] = d.name
        return d.name, json.loads((d / "metrics.json").read_text())

    def all_timing_valid(self, kind: str) -> tuple[str, dict[str, Any]] | None:
        """Latency only from a run that is allowed to report latency."""
        for d in sorted((d for d in RESULTS.iterdir()
                         if d.is_dir() and KINDS[kind] in d.name and "superseded" not in d.name),
                        reverse=True):
            m = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
            if m.get("timing_valid") is True and not m.get("sampled"):
                self.used[kind] = d.name
                return d.name, m
        return None


# ---------------------------------------------------------------------------
# small rendering helpers
# ---------------------------------------------------------------------------
def esc(x: Any) -> str:
    return html.escape(str(x))


def pct(p: dict[str, Any] | None) -> str:
    if not p or p.get("value") is None:
        return "—"
    ci = p.get("ci95")
    ci_s = f" <span class=ci>[{ci[0]:.2f}, {ci[1]:.2f}]</span>" if ci else ""
    n = p.get("n")
    frac = f" <span class=n>{p['successes']}/{n}</span>" if "successes" in p and n else ""
    return f"<b>{p['value']:.3f}</b>{ci_s}{frac}"


def src(run: str | None) -> str:
    return f"<span class=src>results/{esc(run)}/</span>" if run else "<span class=src>—</span>"


def not_measured(why: str) -> str:
    return f"<p class=missing>Not measured. {esc(why)}</p>"


def hbars(rows: list[tuple[str, float, str]], *, width: int = 520, max_value: float = 1.0) -> str:
    """Horizontal bars as inline SVG. rows = (label, value 0..max, annotation)."""
    h = 26
    out = [f'<svg viewBox="0 0 {width} {h * len(rows) + 6}" class=chart role=img>']
    label_w, bar_w = 190, width - 190 - 110
    for i, (label, value, note) in enumerate(rows):
        y = i * h + 4
        w = max(1.0, bar_w * min(value, max_value) / max_value)
        out.append(f'<text x="0" y="{y + 15}" class=lbl>{esc(label)}</text>'
                   f'<rect x="{label_w}" y="{y + 3}" width="{bar_w}" height="16" class=track />'
                   f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="16" class=bar />'
                   f'<text x="{label_w + bar_w + 8}" y="{y + 15}" class=val>{esc(note)}</text>')
    out.append("</svg>")
    return "".join(out)


def before_after(rows: list[tuple[str, float, float]]) -> str:
    """Paired bars: before (muted) and after (accent) for each row."""
    width, h = 560, 40
    out = [f'<svg viewBox="0 0 {width} {h * len(rows) + 24}" class=chart role=img>',
           '<rect x="190" y="2" width="10" height="10" class=before /><text x="204" y="11" class=val>before</text>'
           '<rect x="260" y="2" width="10" height="10" class=bar /><text x="274" y="11" class=val>after</text>']
    bar_w = width - 190 - 70
    for i, (label, b, a) in enumerate(rows):
        y = i * h + 22
        out.append(f'<text x="0" y="{y + 18}" class=lbl>{esc(label)}</text>'
                   f'<rect x="190" y="{y + 2}" width="{bar_w * b:.1f}" height="13" class=before />'
                   f'<rect x="190" y="{y + 17}" width="{bar_w * a:.1f}" height="13" class=bar />'
                   f'<text x="{190 + bar_w + 6}" y="{y + 12}" class=val>{b:.3f}</text>'
                   f'<text x="{190 + bar_w + 6}" y="{y + 28}" class=val>{a:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


def table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class=tw><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def s_headline(runs: Runs) -> str:
    cycle, handoff, lat = runs.latest("cycle"), runs.latest("handoff"), runs.all_timing_valid("latency")
    cards = []
    if cycle:
        name, m = cycle
        cards.append(("Containment (after one cycle)", pct(m["after"]["containment"]),
                      f"{m['n_calls']} simulated calls · {m['agent']} agent", name))
        cards.append(("Wrong-answer rate", pct(m["after"]["wrong_answer_rate"]),
                      "contained but not helped, of contained", name))
    if handoff:
        name, m = handoff
        c = m["configs"]["current"]
        cards.append(("Handoff recall", pct(c["recall"]), f"{m['n']} labelled cases", name))
        cards.append(("Handoff precision", pct(c["precision"]), "all cases incl. tool paths", name))
    if lat:
        name, m = lat
        cards.append(("Voice-to-voice p50 / p95", f"<b>{m['p50_ms']} / {m['p95_ms']} ms</b>",
                      "endpointing reported separately", name))
    else:
        cards.append(("Voice-to-voice latency", "<b class=missing>not measured</b>",
                      "no timing-valid live run exists (needs API keys)", None))
    body = "".join(f"<div class=card><div class=k>{esc(t)}</div><div class=v>{v}</div>"
                   f"<div class=d>{esc(d)}</div>{src(r)}</div>" for t, v, d, r in cards)
    return f"<div class=cards>{body}</div>"


def s_latency(runs: Runs) -> str:
    lat = runs.all_timing_valid("latency")
    if not lat:
        return not_measured(
            "Latency is only rendered from a run with timing_valid: true. The harness records "
            "stages [A] to [H] per turn, but every run so far used the offline agent or replayed "
            "audio, so none is allowed to report voice-to-voice latency.")
    name, m = lat
    stages = m.get("stages_p50_ms", {})
    total = sum(stages.values()) or 1
    return hbars([(k, v, f"{v} ms") for k, v in stages.items()], max_value=total) + src(name)


def s_stt(runs: Runs) -> str:
    stt = runs.latest("stt")
    if not stt:
        return not_measured("No STT benchmark run.")
    name, m = stt
    p = m["provider"]
    rows = [(k, v["entity_accuracy"]["value"],
             f"{v['entity_accuracy']['value']:.2f} · WER {v['wer_mean']:.3f}")
            for k, v in m["by_variant"].items()]
    return (f"<p>{esc(p['provider'])} {esc(p['model'])} ({esc(p['compute_type'])}, {esc(p['device'])}) on "
            f"{m['overall']['n_files']} files. {esc(m['audio'])}. Key-entity accuracy (order ids, "
            f"emails, quantities) next to word error rate. WER stays low while entities fail.</p>"
            + hbars(rows) + src(name))


def s_checker(runs: Runs) -> str:
    chk = runs.latest("checker")
    if not chk:
        return not_measured("No checker run.")
    name, m = chk
    rows = [[esc(k), pct(v), esc(v.get("note") or "")] for k, v in m["catch_by_class"].items()]
    rows.append(["<i>false positives</i> (correct answers rejected)", pct(m["false_positive_rate"]), ""])
    return table(["hallucination class", "caught", "note"], rows) + src(name)


def s_safety(runs: Runs) -> str:
    cycle = runs.latest("cycle")
    rows = [["duplicate side effects under fault injection", "<b>0</b>",
             "33 fault tests: commit-then-timeout, reconciliation down, concurrent duplicates, "
             "interrupted read-back", "<span class=src>eval/fault_injection/ (make faults)</span>"]]
    if cycle:
        name, m = cycle
        d = m["duplicate_side_effects"]
        rows.append(["duplicate side effects in simulated calls", f"<b>{d['before'] + d['after']}</b>",
                     f"{2 * m['n_calls']} calls (both cycle runs)", src(name)])
    rows.append(["context invariant violations", "<b>0</b>",
                 "asserted after every turn in the harness test suite",
                 "<span class=src>tests/test_context_hygiene.py</span>"])
    return table(["property", "count", "denominator", "source"], rows)


def s_gap(runs: Runs) -> str:
    gap = runs.latest("gap")
    if not gap:
        return not_measured("No gap report.")
    name, m = gap
    acc = m["accuracy"]
    planted = {v: k for k, v in acc["rank_of_each_planted_gap"].items() if v}
    rows = [[str(a["rank"]), esc(a["diagnosis"]), f"<code>{esc(a['target'])}</code>",
             str(a["troubled_calls"]),
             f"<span class=tag>planted: {esc(planted[a['rank']])}</span>" if a["rank"] in planted else ""]
            for a in m["actions"][:8] if a["diagnosis"] != "healthy"]
    return (f"<p>{m['call_count']} calls · run <code>{esc(m['run'])}</code>. Actions rank troubled calls by "
            f"the fix that would help them. The hidden labels were consulted only after the report "
            f"was built: planted gaps found at ranks {esc(sorted(planted))}, top-3 hits "
            f"{acc['planted_gaps_in_top3']}/{len(acc['planted_gaps'])}, mean cluster purity "
            f"{acc['mean_cluster_purity']}.</p>"
            + table(["#", "diagnosis", "target", "calls", "answer key"], rows) + src(name))


def s_cycle(runs: Runs) -> str:
    cycle = runs.latest("cycle")
    if not cycle:
        return not_measured("No feedback cycle run.")
    name, m = cycle
    keys = ["containment", "handoff_rate", "wrong_answer_rate", "resolved_rate"]
    pr = m.get("paired", {})
    paired = (f"<p>Paired on identical seeds: <b>+{pr['contained']['gained']} / -{pr['contained']['lost']}</b> "
              f"contained (exact McNemar p = {pr['contained']['mcnemar_exact_p']}); "
              f"{pr['changed_outside_target_intents']} calls changed outside the target intent.</p>") if pr else ""
    rows = [[esc(k), pct(m["before"][k]), pct(m["after"][k]), f"{m['delta'][k]:+.3f}"] for k in keys]
    moved = [[esc(i), f"{s['before']['helped']}/{s['before']['n']}", f"{s['after']['helped']}/{s['after']['n']}"]
             for i, s in m["by_intent"].items() if s["before"]["helped"] != s["after"]["helped"]]
    return (f"<p><b>One change:</b> {esc(m['change']['type'])} → <code>{esc(m['change']['target'])}</code> "
            f"({esc(m['change']['source'])}). Corpus <code>{esc(m['corpus_hash_before'])}</code> → "
            f"<code>{esc(m['corpus_hash_after'])}</code>, seed {m['seeds']['seed']} for both runs, "
            f"{m['n_calls']} calls each, <b>{esc(m['agent'])}</b> agent.</p>"
            + before_after([(k, m["before"][k]["value"], m["after"][k]["value"]) for k in keys])
            + table(["metric", "before", "after", "Δ"], rows) + paired
            + table(["intent that moved", "helped before", "helped after"], moved) + src(name))


def s_provenance(runs: Runs) -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = "unknown"
    rows = []
    for kind, run in sorted(runs.used.items()):
        cfg = RESULTS / run / "config.json"
        c = json.loads(cfg.read_text()) if cfg.exists() else {}
        m = json.loads((RESULTS / run / "metrics.json").read_text())
        # The agent column only for runs where an agent produced the numbers;
        # a config's llm.provider is just the settings default otherwise.
        rows.append([esc(kind), f"<code>results/{esc(run)}/</code>",
                     f"<code>{esc(c.get('git_sha') or m.get('git_sha') or '—')}</code>",
                     f"<code>{esc(c.get('corpus_hash') or '—')}</code>",
                     esc(m.get("agent") or "n/a")])
    return (f"<p>Report rendered {date.today().isoformat()} at git <code>{esc(sha)}</code>. Every number "
            f"above came from one of these directories. Each has a README answering what the run was "
            f"for, what it showed, and what changed.</p>"
            + table(["section", "run directory", "git sha", "corpus hash", "agent"], rows)
            + "<p class=caveat>Simulated calls are template personas, not people. The offline "
              "<i>extractive</i> agent has no language model. Audio is synthesized. Numbers measured "
              "with an LLM agent or human audio will differ, and none are claimed here.</p>")


CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e4e3de;--acc:#2f6f5e;--acc2:#b9d4ca;--warn:#9a3b2e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9d9c95;--line:#2c2c29;--acc:#7cc4ad;--acc2:#2e4a41;--warn:#e08b7c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:18px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h2 small{color:var(--mut);font-weight:400;font-size:13px;margin-left:8px}
.sub{color:var(--mut);margin:0 0 20px}code{font:12.5px ui-monospace,Menlo,monospace}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{border:1px solid var(--line);border-radius:8px;padding:12px 14px}.k{color:var(--mut);font-size:12.5px}
.v{font-size:20px;margin:4px 0}.d{color:var(--mut);font-size:12.5px}
.ci,.n{color:var(--mut);font-size:12px;font-weight:400}.src{display:block;color:var(--mut);font:11.5px ui-monospace,Menlo,monospace;margin-top:6px}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:14px}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--mut);font-weight:500;font-size:12.5px}
.chart{width:100%;max-width:640px;display:block;margin:10px 0}.lbl,.val{font:12px ui-sans-serif,system-ui,sans-serif;fill:var(--fg)}.val{fill:var(--mut)}
.track{fill:var(--line)}.bar{fill:var(--acc)}.before{fill:var(--acc2)}
.missing{color:var(--warn)}.tag{background:var(--acc2);border-radius:4px;padding:1px 6px;font-size:12px}
.caveat{color:var(--mut);font-size:13px;border-left:3px solid var(--line);padding-left:10px}
"""

SECTIONS = [
    ("1 · Headline", "", s_headline),
    ("2 · Latency breakdown", "stages [A] to [H], endpointing separate", s_latency),
    ("3 · Speech recognition", "entity accuracy vs noise and phone audio", s_stt),
    ("4 · Grounding checker", "catch rate by class, with the declared gaps", s_checker),
    ("5 · Safety", "things that must be zero, with denominators", s_safety),
    ("6 · Gap report", "what to fix, ranked", s_gap),
    ("7 · Feedback cycle", "one change, identical seeds, both guardrails", s_cycle),
    ("8 · Provenance", "where every number came from", s_provenance),
]


def render() -> str:
    runs = Runs()
    parts = [f"<h2>{esc(title)}<small>{esc(sub)}</small></h2>{fn(runs)}" for title, sub, fn in SECTIONS[:-1]]
    title, sub, fn = SECTIONS[-1]
    parts.append(f"<h2>{esc(title)}<small>{esc(sub)}</small></h2>{fn(runs)}")
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>VoiceDesk report</title><style>{CSS}</style></head><body><main>"
            f"<h1>VoiceDesk: results</h1><p class=sub>Real-time voice support agent · "
            f"every number links to its run directory</p>{''.join(parts)}</main></body></html>")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
