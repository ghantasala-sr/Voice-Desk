"""W3-8 — handoff precision/recall for the DETERMINISTIC policy. docs/09 §5.

Every labelled case runs through the real harness — real retrieval, real
pre-gate, real soft scoring — with the model's reply held constant as one
neutral sentence that the checker cannot fail. What this measures is
therefore exactly the part of the handoff decision that does not depend on
the model: hard utterance rules plus retrieval signals.

What it cannot measure, and the README says so: handoffs that happen because
the MODEL's answer tripped the checker, or because the model itself says it
will fetch a colleague. That needs a live model (blocked: no API key).

Configurations compared (each one is also an ablation):

  current           handoff.yaml: two retrieval levels + hard rules
  one_level         no "very weak" level: retrieval can never hand off alone
  no_patterns       hard utterance rules disabled

Two views of every config. "all" is every labelled case. "no_tool_path"
drops cases where the real agent would call a tool: in production that marks
the turn transactional and suppresses the retrieval signal, but with a
neutral reply no tool runs — so in "all" those cases are false positives the
system would not actually produce.

Run: .venv/bin/python -m eval.calibrate_handoff
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from core import db
from core.settings import get_settings
from eval.dataset import Case, all_cases
from eval.metrics import precision_recall
from eval.text_driver import TextDriver
from harness.retriever import PostgresRetriever
from providers import config_snapshot
from providers.llm.replay import ReplayProvider
from retrieval.embed import get_embedder

ROOT = Path(__file__).resolve().parent.parent
NEUTRAL = [{"text": "Let me see what I can find on that. "}, {"end": {"stop_reason": "end_turn"}}]

CONFIGS: dict[str, dict[str, Any]] = {
    "current": {},
    "one_level": {"very_weak": 0.0},
    "no_patterns": {"patterns": False},
}


def tool_path(case: Case) -> bool:
    return case.expected.get("type") == "tool" or case.id == "disclose_without_verification"


async def decide(case: Case, retriever: PostgresRetriever, config: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings().model_copy(update={"handoff_enabled": True, "checker_enabled": True})
    driver = TextDriver(llm=ReplayProvider(events=NEUTRAL), settings=settings, retriever=retriever)
    policy = driver.harness.policy
    policy.weights.update(config.get("weights", {}))
    if config.get("patterns") is False:
        policy._patterns = []
    if "very_weak" in config:
        policy.very_weak_threshold = config["very_weak"]
    async with driver:
        result = await driver.say(case.utterance)
    turn = result.turn
    return {
        "case_id": case.id, "group": case.group, "label": case.should_escalate,
        "expected_reason": case.expected.get("escalate_reason"),
        "handed_off": driver.call.outcome == "handoff",
        "reason": driver.call.handoff_reason,
        "llm_calls": driver.harness.llm.call_count,
        "signals": list(turn.handoff_signals) if turn else [],
        "top_vec_score": turn.retrieval_top_vec_score if turn else None,
    }


async def run() -> dict[str, Any]:
    settings = get_settings()
    await db.open_pool(settings)
    retriever = PostgresRetriever(domain=settings.domain, embedder=get_embedder(settings),
                                  settings=settings)
    cases = all_cases()
    results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, config in CONFIGS.items():
        decisions = [await decide(case, retriever, config) for case in cases]
        for d in decisions:
            rows.append({"config": name, **d})
        pr = precision_recall([d["handed_off"] for d in decisions], [d["label"] for d in decisions])
        kept = [d for d, c in zip(decisions, cases, strict=True) if not tool_path(c)]
        pr_kept = precision_recall([d["handed_off"] for d in kept], [d["label"] for d in kept])
        by_group = Counter()
        for d in decisions:
            if d["label"]:
                by_group[(d["group"], d["handed_off"])] += 1
        results[name] = {
            **pr,
            "no_tool_path": {k: pr_kept[k] for k in ("precision", "recall", "f1", "n", "positives")},
            "false_negatives": [d["case_id"] for d in decisions if d["label"] and not d["handed_off"]],
            "false_positives": [d["case_id"] for d in decisions if d["handed_off"] and not d["label"]],
            "recall_by_group": {
                g: f"{by_group[(g, True)]}/{by_group[(g, True)] + by_group[(g, False)]}"
                for g in sorted({g for g, _ in by_group})
            },
            "hard_rule_llm_calls": sum(
                d["llm_calls"] for d in decisions if d["handed_off"] and d["reason"]
                and d["reason"] in ("explicit_human_request", "payment_dispute", "distress")
            ),
        }
    await db.close_pool()
    return {"run": f"{date.today().isoformat()}-handoff-calibration", "n": len(cases),
            "positives": sum(c.should_escalate for c in cases), "configs": results,
            "_rows": rows, "_config": config_snapshot(settings)}


def main() -> int:
    result = asyncio.run(run())
    out = ROOT / "results" / result["run"]
    out.mkdir(parents=True, exist_ok=True)
    rows, config = result.pop("_rows"), result.pop("_config")
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"run: {out.relative_to(ROOT)}   n={result['n']}  positives={result['positives']}")
    for name, r in result["configs"].items():
        p, rc = r["precision"], r["recall"]
        print(f"\n  [{name}]  precision {p['value']} {p['ci95']} ({p['successes']}/{p['n']})"
              f"   recall {rc['value']} {rc['ci95']} ({rc['successes']}/{rc['n']})   F1 {r['f1']}")
        k = r["no_tool_path"]
        print(f"    no-tool-path view (n={k['n']}): precision {k['precision']['value']} {k['precision']['ci95']}"
              f"  recall {k['recall']['value']} {k['recall']['ci95']}  F1 {k['f1']}")
        print(f"    recall by group: {r['recall_by_group']}")
        print(f"    false positives ({len(r['false_positives'])}): {r['false_positives']}")
        print(f"    false negatives ({len(r['false_negatives'])}): {r['false_negatives']}")
        print(f"    LLM calls spent on hard-rule handoffs: {r['hard_rule_llm_calls']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
