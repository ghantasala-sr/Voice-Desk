"""W4-3 / W4-4 / W4-5 — find what the knowledge base is missing, from calls.

    calls -> first utterances -> embed -> HDBSCAN clusters -> per-cluster
    trouble -> diagnosis -> ranked actions

It reads ONLY what production would see: call outcomes, handoff reasons,
retrieval scores, and the caller's first words. It never reads the
`simulation_label` events — those are the answer key, used afterwards by
`score_against_labels()` to check whether the report found the gaps that were
planted. A clustering job nobody validated is a job you cannot trust.

Why HDBSCAN rather than k-means: the number of intents is unknown, cluster
sizes are wildly unequal, and k-means would force the long tail into whichever
large cluster sits nearest. Unclustered calls become "miscellaneous", and a
large miscellaneous bucket is itself a finding.

Cluster NAMES come from distinctive TF-IDF terms, not a model: offline there is
no LLM to name them, and keywords are at least honest about what they are.

Run: .venv/bin/python -m analytics.gap_report [--run <simulation_run_id>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from core import db
from core.ids import new_id
from core.settings import get_settings
from harness.policy import retrieval_sufficiency_threshold
from retrieval.chunk import chunk_document
from retrieval.embed import get_embedder

ROOT = Path(__file__).resolve().parent.parent
MIN_CLUSTER = 6
#: An unindexed document must beat what IS indexed by this much, and be a
#: decent match in absolute terms, before the report blames the indexing.
HELD_OUT_MARGIN = 0.03
HELD_OUT_FLOOR = 0.66
TOOL_REASONS = {"tool_failure", "tool_state_unknown", "verification_failed", "readback_unclear"}
RETRIEVAL_REASONS = {"retrieval_very_weak", "retrieval_insufficient", "retrieval_unavailable"}


# ---------------------------------------------------------------------------
# Loading — production-visible data only
# ---------------------------------------------------------------------------
async def load_calls(run: str | None) -> tuple[str, list[dict[str, Any]]]:
    if run is None:
        latest = await db.fetch_one(
            "SELECT config_snapshot->>'simulation_run' AS run FROM calls"
            " WHERE source = 'simulation' AND config_snapshot ? 'simulation_run'"
            " ORDER BY started_at DESC LIMIT 1")
        if not latest:
            raise SystemExit("no simulation runs found — run `make simulate` first")
        run = latest["run"]
    rows = await db.fetch_all(
        """
        SELECT c.call_id, c.outcome, c.handoff_reason, c.rephrase_count, c.started_at,
               (SELECT t.user_text FROM turns t WHERE t.call_id = c.call_id ORDER BY seq LIMIT 1)
                   AS first_utterance,
               (SELECT t.retrieval_top_vec_score FROM turns t WHERE t.call_id = c.call_id
                  ORDER BY seq LIMIT 1) AS first_top_vec,
               (SELECT t.retrieved_chunk_ids[1] FROM turns t WHERE t.call_id = c.call_id
                  ORDER BY seq LIMIT 1) AS first_top_chunk,
               EXISTS (SELECT 1 FROM tool_executions x WHERE x.call_id = c.call_id) AS used_tools,
               EXISTS (SELECT 1 FROM turns t WHERE t.call_id = c.call_id
                       AND 'rephrase_repeat' = ANY(t.handoff_signals)) AS rephrased
        FROM calls c
        WHERE c.source = 'simulation' AND c.config_snapshot->>'simulation_run' = %(run)s
        ORDER BY c.started_at
        """,
        {"run": run},
    )
    return run, [r for r in rows if r["first_utterance"]]


async def held_out_chunks(domain: str) -> list[tuple[str, str]]:
    """(doc path, chunk text) for every document recorded but NOT indexed."""
    rows = await db.fetch_all(
        "SELECT path, title FROM documents WHERE domain_config = %(d)s AND NOT ingested", {"d": domain})
    corpus = ROOT / "domains" / domain / "corpus"
    out = []
    for row in rows:
        text = (corpus / row["path"]).read_text(encoding="utf-8")
        out.extend((row["path"], c.embed_text) for c in chunk_document(text, title=row["title"]))
    return out


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def cluster(vectors: np.ndarray) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    # Unit vectors, so euclidean distance orders exactly like cosine distance.
    return HDBSCAN(min_cluster_size=MIN_CLUSTER, min_samples=3).fit_predict(vectors)


def name_clusters(texts: list[str], labels: np.ndarray) -> dict[int, list[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids = sorted(set(labels) - {-1})
    docs = [" ".join(t for t, lab in zip(texts, labels, strict=True) if lab == cid) for cid in ids]
    if not docs:
        return {}
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = tfidf.fit_transform(docs)
    vocab = np.array(tfidf.get_feature_names_out())
    names = {}
    for row, cid in enumerate(ids):
        scores = matrix[row].toarray().ravel()
        names[cid] = [str(t) for t in vocab[np.argsort(scores)[::-1][:4]] if scores[np.argsort(scores)[::-1][0]] > 0]
    return names


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
ACTIONS = {
    "held_out_coverage": lambda t: (f"Ingest {t}: it exists, answers these questions better than "
                                    "anything indexed, and is not in the knowledge base."),
    "weak_coverage": lambda t: (f"Rewrite or expand {t} — it is what these calls retrieve, and they "
                                "still fail. Use the callers' own words (see examples)."),
    "threshold_too_strict": lambda t: ("Retrieval looks strong but these calls hand off on retrieval "
                                       "signals — lower the threshold before adding documents."),
    "tool_failure": lambda t: f"Investigate {str(t).replace('_', ' ')} — fix the tool, not the corpus.",
    "no_coverage": lambda t: "Write a document covering this topic — nothing indexed answers it.",
    "healthy": lambda t: "No action.",
}


def attribute(c: dict[str, Any], threshold: float) -> tuple[str, str | None]:
    """Which fix would help THIS call? Each diagnosis is a different fix."""
    reason = c["handoff_reason"]
    if not c["troubled"]:
        return "healthy", None
    if reason in TOOL_REASONS:
        return "tool_failure", reason
    if c["held_out"]:
        return "held_out_coverage", c["held_out"]["doc"]
    if (c["first_top_vec"] or 0) >= threshold and reason in RETRIEVAL_REASONS:
        return "threshold_too_strict", "retrieval threshold"
    if c["top_doc"]:
        return "weak_coverage", c["top_doc"]
    return "no_coverage", None


async def build_report(run: str | None) -> dict[str, Any]:
    settings = get_settings()
    await db.open_pool(settings)
    run, calls = await load_calls(run)
    embedder = get_embedder(settings)
    threshold = retrieval_sufficiency_threshold(settings.domain)

    chunk_rows = await db.fetch_all(
        "SELECT c.chunk_id, d.path FROM chunks c JOIN documents d USING (doc_id)")
    chunk_doc = {r["chunk_id"]: r["path"] for r in chunk_rows}
    held = await held_out_chunks(settings.domain)

    texts = [c["first_utterance"] for c in calls]
    # Utterances are QUERIES to BGE; chunks are passages. Keep the asymmetry.
    q_vectors = np.array([embedder.embed_query(t) for t in texts])
    labels = cluster(q_vectors)
    names = name_clusters(texts, labels)

    # ---- per call: is it in trouble, and what would fix it? ------------------
    held_sims = (np.array(embedder.embed_passages([t for _, t in held])) @ q_vectors.T
                 if held else None)  # (held chunks, calls)
    for i, c in enumerate(calls):
        c["cluster"] = int(labels[i])
        c["weak"] = (c["first_top_vec"] or 0) < threshold and not c["used_tools"]
        c["failed"] = c["outcome"] in ("handoff", "dropped") or c["rephrased"]
        c["troubled"] = c["weak"] or c["failed"]
        c["top_doc"] = chunk_doc.get(c["first_top_chunk"])
        c["held_out"] = None
        if held_sims is not None:
            best = int(np.argmax(held_sims[:, i]))
            sim = float(held_sims[best, i])
            # An unindexed document must beat what IS indexed, by a margin,
            # and be a decent match in absolute terms.
            if sim >= HELD_OUT_FLOOR and sim > (c["first_top_vec"] or 0) + HELD_OUT_MARGIN:
                c["held_out"] = {"doc": held[best][0], "similarity": round(sim, 3)}
        c["diagnosis"], c["target"] = attribute(c, threshold)

    # ---- clusters: the narrative, with examples -----------------------------
    clusters = []
    for cid in sorted(set(labels), key=lambda x: (x == -1, x)):
        members = [c for c in calls if c["cluster"] == cid]
        n = len(members)
        troubled = [m for m in members if m["troubled"]]
        top_vec = sorted(m["first_top_vec"] or 0 for m in members)
        covering = Counter(m["top_doc"] for m in members if m["top_doc"])
        targets = Counter((m["diagnosis"], m["target"]) for m in troubled)
        (diagnosis, target), _ = targets.most_common(1)[0] if targets else (("healthy", None), 0)
        if len(troubled) / n < 0.35:
            diagnosis, target = "healthy", None
        held_votes = Counter(m["held_out"]["doc"] for m in members if m["held_out"])
        clusters.append({
            "cluster_id": int(cid),
            "label": " / ".join(names.get(cid, [])) if cid != -1 else "miscellaneous",
            "keywords": names.get(cid, []),
            "n_calls": n,
            "share": round(n / len(calls), 4),
            "failure_rate": round(sum(m["failed"] for m in members) / n, 4),
            "weak_retrieval_rate": round(sum(m["weak"] for m in members) / n, 4),
            "trouble_rate": round(len(troubled) / n, 4),
            "median_top_vec_score": round(top_vec[len(top_vec) // 2], 4) if top_vec else None,
            "handoff_reasons": dict(Counter(m["handoff_reason"] for m in members if m["handoff_reason"])),
            "dominant_handoff_reason": (Counter(m["handoff_reason"] for m in members
                                                if m["handoff_reason"]).most_common(1) or [[None]])[0][0],
            "covering_docs": [{"doc": d, "count": k} for d, k in covering.most_common(3)],
            "held_out_match": ({"doc": held_votes.most_common(1)[0][0],
                                "calls": held_votes.most_common(1)[0][1]} if held_votes else None),
            "diagnosis": diagnosis,
            "target": target,
            "suggested_action": ACTIONS[diagnosis](target),
            "priority": round(len(troubled) * min(1.0, n / 10), 2),
            "example_utterances": list(dict.fromkeys(m["first_utterance"] for m in members))[:5],
            "call_ids": [m["call_id"] for m in members],
        })
    clusters.sort(key=lambda c: -c["priority"])

    # ---- actions: troubled calls summed by the fix that would help them -----
    # The unit of work in a gap report is a DOCUMENT to ingest, rewrite or
    # write — not a phrasing. Ranking clusters instead let one topic, split
    # across five phrasings, lose to a single well-worded cluster.
    by_target: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for c in calls:
        if c["troubled"]:
            by_target.setdefault((c["diagnosis"], c["target"]), []).append(c)
    actions = []
    for (diagnosis, target), members in by_target.items():
        n = len(members)
        actions.append({
            "diagnosis": diagnosis, "target": target, "action": ACTIONS[diagnosis](target),
            "troubled_calls": n, "share_of_calls": round(n / len(calls), 4),
            "priority": round(n * min(1.0, n / 10), 2),
            "cluster_ids": sorted({m["cluster"] for m in members}),
            "cluster_id": Counter(m["cluster"] for m in members).most_common(1)[0][0],
            "example_utterances": list(dict.fromkeys(m["first_utterance"] for m in members))[:5],
            "call_ids": [m["call_id"] for m in members],
        })
    actions.sort(key=lambda a: -a["priority"])
    for rank, action in enumerate(actions, 1):
        action["rank"] = rank

    report = {
        "report_id": new_id(), "run": run, "generated_at": datetime.now(UTC).isoformat(),
        "call_count": len(calls), "clusters": clusters, "actions": actions,
        "threshold_top_vec_score": threshold,
        "window_start": min(c["started_at"] for c in calls).isoformat(),
        "window_end": max(c["started_at"] for c in calls).isoformat(),
    }
    report["accuracy"] = await score_against_labels(report)
    await db.execute(
        """INSERT INTO gap_reports (report_id, window_start, window_end, call_count, clusters, actions)
           VALUES (%(id)s, %(s)s, %(e)s, %(n)s, %(c)s::jsonb, %(a)s::jsonb)""",
        {"id": report["report_id"], "s": report["window_start"], "e": report["window_end"],
         "n": len(calls), "c": json.dumps(_public(report["clusters"])),
         "a": json.dumps(_public(report["actions"]))},
    )
    await db.close_pool()
    return report


def _public(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop per-call ids before storing — the report is about topics, not callers."""
    return [{k: v for k, v in c.items() if k != "call_ids"} for c in items]


# ---------------------------------------------------------------------------
# W4-5 — score the report against the hidden answer key
# ---------------------------------------------------------------------------
async def score_against_labels(report: dict[str, Any]) -> dict[str, Any]:
    """The ONLY function that reads simulation_label — after the report exists."""
    rows = await db.fetch_all(
        "SELECT call_id, payload FROM call_events WHERE event_type = 'simulation_label'"
        " AND payload->>'run' = %(run)s", {"run": report["run"]})
    truth = {r["call_id"]: r["payload"] for r in rows}
    intents = yaml.safe_load((ROOT / "eval" / "caller_sim" / "intents.yaml").read_text())["intents"]
    uncovered = {i["id"] for i in intents if i["covered"] is False}

    by_intent: dict[str, list[dict[str, Any]]] = {}
    for label in truth.values():
        by_intent.setdefault(label["intent"], []).append(label)
    resolved = {k: sum(x["satisfied"] for x in v) / len(v) for k, v in by_intent.items()}

    for c in report["clusters"]:
        here = Counter(truth[cid]["intent"] for cid in c["call_ids"] if cid in truth)
        c["majority_intent"], top = here.most_common(1)[0] if here else ("?", 0)
        c["purity"] = round(top / max(1, sum(here.values())), 3)
    for a in report["actions"]:
        here = Counter(truth[cid]["intent"] for cid in a["call_ids"] if cid in truth)
        a["majority_intent"] = here.most_common(1)[0][0] if here else "?"

    gold = {i["id"]: i.get("gold_docs", []) for i in intents}
    gap_ranks = {
        g: next((a["rank"] for a in report["actions"]
                 if a["diagnosis"] == "held_out_coverage" and a["target"] in gold[g]), None)
        for g in sorted(uncovered)
    }
    top3 = report["actions"][:3]
    false_positives = [a["majority_intent"] for a in top3
                       if resolved.get(a["majority_intent"], 0) >= 0.8]
    ranked = [c for c in report["clusters"] if c["cluster_id"] != -1]
    return {
        "planted_gaps": sorted(uncovered),
        "rank_of_each_planted_gap": gap_ranks,
        "planted_gaps_in_top3": sum(1 for r in gap_ranks.values() if r and r <= 3),
        "false_positives_in_top3": false_positives,
        "mean_cluster_purity": round(float(np.mean([c["purity"] for c in ranked])) if ranked else 0.0, 3),
        "hidden_resolved_rate_by_intent": {k: round(v, 3) for k, v in sorted(resolved.items())},
        "note": "scored with the simulator's hidden labels, AFTER the report was built",
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def render_text(report: dict[str, Any]) -> str:
    n_clusters = len([c for c in report["clusters"] if c["cluster_id"] != -1])
    misc = next((c for c in report["clusters"] if c["cluster_id"] == -1), None)
    lines = [f"GAP REPORT · run {report['run']} · {report['call_count']} calls · {n_clusters} clusters"
             + (f" · {misc['n_calls']} unclustered" if misc else ""), "",
             "RANKED ACTIONS (troubled calls attributed to the fix that would help them)", ""]
    for a in report["actions"]:
        if a["diagnosis"] == "healthy":
            continue
        lines += [f"#{a['rank']}  {a['diagnosis'].upper()}  →  {a['target']}",
                  f"    {a['troubled_calls']} troubled calls ({a['share_of_calls']:.1%} of all) "
                  f"across {len(a['cluster_ids'])} cluster(s)   priority {a['priority']}",
                  f"    {a['action']}"]
        lines += [f'      "{u}"' for u in a["example_utterances"][:3]] + [""]
    acc = report["accuracy"]
    lines += ["SCORED AGAINST THE HIDDEN LABELS (after the fact):",
              f"  planted gaps whose document is a top-3 action: {acc['planted_gaps_in_top3']}/{len(acc['planted_gaps'])}",
              f"  rank of each: {acc['rank_of_each_planted_gap']}",
              f"  false positives in top 3: {acc['false_positives_in_top3'] or 'none'}",
              f"  mean cluster purity: {acc['mean_cluster_purity']}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the gap report for a simulation run.")
    parser.add_argument("--run", default=None)
    parser.add_argument("--label", default=None,
                        help="output directory label (default: the run's label)")
    args = parser.parse_args()
    report = asyncio.run(build_report(args.run))
    label = args.label or report["run"].rsplit("-", 1)[0]
    out = ROOT / "results" / f"{date.today().isoformat()}-gap-report-{label}"
    if out.exists():
        raise SystemExit(f"{out.relative_to(ROOT)} exists — runs are never overwritten; pass --label")
    out.mkdir(parents=True, exist_ok=True)
    public = {**report, "clusters": _public(report["clusters"]), "actions": _public(report["actions"])}
    (out / "metrics.json").write_text(json.dumps(public, indent=2, default=str))
    text = render_text(report)
    (out / "report.txt").write_text(text + "\n")
    print(f"run: {out.relative_to(ROOT)}\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
