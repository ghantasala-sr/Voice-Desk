"""W4-7 — one feedback cycle, run exactly to the protocol in docs/14 §5.

    0  freeze      the simulation seed and the agent are fixed for both runs
    1  baseline    simulate N calls against the documented corpus
    2  gap report  on the baseline calls only
    3  ONE change  the highest-ranked action that can be applied automatically
                   (ingesting a held-out document). Actions that need a human
                   author — "rewrite this section" — are listed, not faked.
    4  re-measure  re-ingest, recompute the corpus hash, rerun the SAME seeds
    5  report      both numbers AND both guardrails: wrong-answer rate and
                   handoff rate. Containment always rises if you hand off less;
                   the claim only means something if the guardrails hold.

One change, because with two you cannot attribute the delta.

The corpus is restored to the documented baseline afterwards, so the repo's
held-out design (docs/00 D4) and its tests stay true. To ADOPT the change, set
`ingested: true` for that document in corpus_manifest.yaml.

Run: .venv/bin/python -m eval.feedback_cycle --n 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from math import comb
from pathlib import Path
from typing import Any

import yaml

from core import db
from core.settings import get_settings
from eval.caller_sim.simulate import run as simulate
from providers import offline_fallback
from retrieval.ingest import ingest

ROOT = Path(__file__).resolve().parent.parent
AUTOMATABLE = {"held_out_coverage"}


async def reingest(settings: Any, overrides: dict[str, bool] | None) -> str:
    await db.open_pool(settings)
    try:
        report = await ingest(settings.domain, settings=settings, overrides=overrides)
    finally:
        await db.close_pool()
    return report.corpus_hash


async def gap_report(run: str) -> dict[str, Any]:
    from analytics.gap_report import build_report

    return await build_report(run)


def paired(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]],
           target_intents: set[str]) -> dict[str, Any]:
    """Identical seeds make every call a matched pair, so the right test is
    McNemar on the discordant pairs, not two overlapping Wilson intervals."""
    after = {r["index"]: r for r in after_rows}
    out: dict[str, Any] = {}
    for key, ok in (("contained", lambda r: r["outcome"] == "contained"),
                    ("helped", lambda r: bool(r["satisfied"]))):
        gained = sum(1 for r in before_rows if not ok(r) and ok(after[r["index"]]))
        lost = sum(1 for r in before_rows if ok(r) and not ok(after[r["index"]]))
        n, k = gained + lost, min(gained, lost)
        p = min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2**n) if n else 1.0
        out[key] = {"gained": gained, "lost": lost, "mcnemar_exact_p": float(f"{p:.3g}")}
    out["changed_outside_target_intents"] = sum(
        1 for r in before_rows if r["intent"] not in target_intents
        and (r["outcome"], r["satisfied"]) != (after[r["index"]]["outcome"], after[r["index"]]["satisfied"]))
    return out


def delta(before: dict[str, Any], after: dict[str, Any]) -> float | None:
    if before.get("value") is None or after.get("value") is None:
        return None
    return round(after["value"] - before["value"], 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one feedback cycle.")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    settings = get_settings()
    settings = offline_fallback(settings)

    # 1. baseline, against the documented corpus
    hash_before = asyncio.run(reingest(settings, None))
    print(f"[1] baseline · corpus {hash_before} · agent {settings.llm_provider}")
    before = asyncio.run(simulate(args.n, "cycle-before", args.seed, settings))
    before_rows = before.pop("_rows")

    # 2. gap report on the baseline only
    report = asyncio.run(gap_report(before["run"]))
    actions = [a for a in report["actions"] if a["diagnosis"] != "healthy"]
    print(f"[2] gap report · {len(actions)} actions")
    for a in actions[:5]:
        print(f"      #{a['rank']} {a['diagnosis']} -> {a['target']} ({a['troubled_calls']} calls)")

    # 3. ONE change: the highest-ranked action we can apply without a human
    chosen = next((a for a in actions if a["diagnosis"] in AUTOMATABLE), None)
    if chosen is None:
        print("no automatable action — nothing to change")
        return 1
    skipped = [a for a in actions if a["rank"] < chosen["rank"]]
    print(f"[3] change · ingest {chosen['target']} (action #{chosen['rank']}); "
          f"{len(skipped)} higher-ranked action(s) need a human author")

    # 4. re-measure with the SAME seeds
    try:
        hash_after = asyncio.run(reingest(settings, {chosen["target"]: True}))
        print(f"[4] re-run · corpus {hash_after}")
        after = asyncio.run(simulate(args.n, "cycle-after", args.seed, settings))
        after_rows = after.pop("_rows")
    finally:
        restored = asyncio.run(reingest(settings, None))
        print(f"    corpus restored to documented baseline · {restored}")

    # 5. report — both numbers and both guardrails
    by_intent = {}
    for intent in sorted({r["intent"] for r in before_rows}):
        b = [r for r in before_rows if r["intent"] == intent]
        a = [r for r in after_rows if r["intent"] == intent]
        by_intent[intent] = {
            "before": {"contained": sum(r["outcome"] == "contained" for r in b),
                       "helped": sum(r["satisfied"] for r in b), "n": len(b)},
            "after": {"contained": sum(r["outcome"] == "contained" for r in a),
                      "helped": sum(r["satisfied"] for r in a), "n": len(a)},
        }
    # The intents the change is FOR: those whose gold document was ingested.
    # (Not "whichever improved" — that would make the spillover count circular.)
    intents_spec = yaml.safe_load((ROOT / "eval" / "caller_sim" / "intents.yaml").read_text())["intents"]
    target_intents = {i["id"] for i in intents_spec if chosen["target"] in i.get("gold_docs", [])}
    metric_keys = ("containment", "handoff_rate", "wrong_answer_rate", "resolved_rate")
    result = {
        "run": f"{date.today().isoformat()}-feedback-cycle",
        "agent": settings.llm_provider,
        "change": {"type": "ingest_held_out_document", "target": chosen["target"],
                   "source": f"gap report {report['report_id']} action #{chosen['rank']}",
                   "skipped_higher_ranked": [{"rank": a["rank"], "diagnosis": a["diagnosis"],
                                              "target": a["target"]} for a in skipped]},
        "corpus_hash_before": hash_before, "corpus_hash_after": hash_after,
        "seeds": {"seed": args.seed, "identical": True}, "n_calls": args.n,
        "before": {k: before[k] for k in metric_keys},
        "after": {k: after[k] for k in metric_keys},
        "delta": {k: delta(before[k], after[k]) for k in metric_keys},
        "duplicate_side_effects": {"before": before["duplicate_side_effects"],
                                   "after": after["duplicate_side_effects"]},
        "paired": {**paired(before_rows, after_rows, target_intents),
                   "target_intents": sorted(target_intents)},
        "by_intent": by_intent,
        "gap_report_accuracy": report["accuracy"],
        "gap_report_actions": [{k: a[k] for k in ("rank", "diagnosis", "target", "troubled_calls")}
                               for a in actions],
        "simulation_runs": {"before": before["run"], "after": after["run"]},
    }
    out = ROOT / "results" / result["run"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "raw.jsonl").write_text("\n".join(json.dumps({"phase": "before", **r}) for r in before_rows)
                                   + "\n" + "\n".join(json.dumps({"phase": "after", **r}) for r in after_rows) + "\n")

    print(f"\n[5] FEEDBACK CYCLE · {out.relative_to(ROOT)}")
    print(f"    change: ingested {chosen['target']}   corpus {hash_before} -> {hash_after}   seeds identical")
    print(f"    {'':20} {'before':>22} {'after':>22} {'delta':>8}")
    for k in metric_keys:
        b, a = before[k], after[k]
        print(f"    {k:20} {b['value']:.3f} [{b['ci95'][0]:.2f},{b['ci95'][1]:.2f}]   "
              f"{a['value']:.3f} [{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}]   {result['delta'][k]:+.3f}")
    pr = result["paired"]
    print(f"    paired: contained +{pr['contained']['gained']}/-{pr['contained']['lost']} "
          f"(McNemar p={pr['contained']['mcnemar_exact_p']}); "
          f"{pr['changed_outside_target_intents']} calls changed outside {sorted(target_intents)}")
    for intent in sorted(i for i, s in by_intent.items() if s["before"]["helped"] != s["after"]["helped"]):
        s = by_intent[intent]
        print(f"    {intent}: helped {s['before']['helped']}/{s['before']['n']} -> {s['after']['helped']}/{s['after']['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
