"""W3-7 — calibrate the "retrieval is insufficient" threshold. docs/05 §6.

Runs real retrieval over every eval case labelled covered or uncovered, then
for each candidate signal sweeps the rule

    flag as insufficient  <=>  score < threshold

and picks the LOWEST threshold that flags at least TARGET_RECALL of the
uncovered queries. Recall on uncovered is the priority: a missed one is a
confident answer from the wrong document. The false-alarm rate on covered
queries is what that recall costs, and is reported beside it, always.

Writes results/<date>-retrieval-calibration/. The README in that directory is
written by hand after reading the numbers.

Run: .venv/bin/python -m eval.calibrate_retrieval
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from core import db
from core.settings import get_settings
from eval.dataset import all_cases
from eval.metrics import Proportion
from providers import config_snapshot
from retrieval.embed import get_embedder
from retrieval.ingest import corpus_hash, load_manifest
from retrieval.search import search

TARGET_RECALL = 0.90
SIGNALS = ["top_vec_score", "top_lex_score", "vec_margin", "top_score", "margin"]
ROOT = Path(__file__).resolve().parent.parent


def auc(covered: list[float], uncovered: list[float]) -> float:
    """P(score of a random covered case > score of a random uncovered one).

    0.5 is a coin toss; below 0.5 the signal runs the wrong way round, which
    is exactly what the fused margin did in the week-2 study.
    """
    pairs = len(covered) * len(uncovered)
    if not pairs:
        return float("nan")
    wins = sum(
        1.0 if c > u else 0.5 if c == u else 0.0 for c in covered for u in uncovered
    )
    return wins / pairs


def operating_point(covered: list[float], uncovered: list[float]) -> dict[str, Any]:
    """Lowest threshold whose recall on uncovered reaches TARGET_RECALL."""
    candidates = sorted(set(covered) | set(uncovered))
    candidates = [c + 1e-6 for c in candidates]  # "< t" must include the value itself
    for threshold in candidates:
        caught = sum(u < threshold for u in uncovered)
        if caught / len(uncovered) >= TARGET_RECALL:
            false_alarms = sum(c < threshold for c in covered)
            return {
                "threshold": round(threshold, 4),
                "recall_uncovered": Proportion(caught, len(uncovered)).as_dict(),
                "false_alarm_covered": Proportion(false_alarms, len(covered)).as_dict(),
            }
    return {"threshold": None}


async def run() -> dict[str, Any]:
    settings = get_settings()
    await db.open_pool(settings)
    embedder = get_embedder(settings)

    rows: list[dict[str, Any]] = []
    for case in all_cases():
        if case.retrieval not in ("covered", "uncovered"):
            continue
        result = await search(case.utterance, domain=settings.domain, embedder=embedder,
                              settings=settings)
        docs = [c.doc_path for c in result.chunks]
        rows.append({
            "case_id": case.id, "source": case.source_file, "label": case.retrieval,
            "utterance": case.utterance, "gold_docs": case.citation_docs,
            "top1_doc": docs[0] if docs else None, "retrieved_docs": docs,
            "top1_hit": bool(case.citation_docs) and bool(docs) and docs[0] in case.citation_docs,
            "topk_hit": bool(case.citation_docs) and any(d in case.citation_docs for d in docs),
            **{s: getattr(result, s) for s in SIGNALS},
        })
    await db.close_pool()

    covered = [r for r in rows if r["label"] == "covered"]
    uncovered = [r for r in rows if r["label"] == "uncovered"]
    gold = [r for r in covered if r["gold_docs"]]

    signals = {}
    for signal in SIGNALS:
        c = [r[signal] for r in covered]
        u = [r[signal] for r in uncovered]
        signals[signal] = {
            "auc": round(auc(c, u), 4),
            "covered_median": round(sorted(c)[len(c) // 2], 5),
            "uncovered_median": round(sorted(u)[len(u) // 2], 5),
            "operating_point": operating_point(c, u),
        }

    manifest = load_manifest(settings.domain)
    return {
        "run": f"{date.today().isoformat()}-retrieval-calibration",
        "n_covered": len(covered), "n_uncovered": len(uncovered),
        "target_recall_on_uncovered": TARGET_RECALL,
        "accuracy": {
            "top1_document": Proportion(sum(r["top1_hit"] for r in gold), len(gold)).as_dict(),
            "topk_document": Proportion(sum(r["topk_hit"] for r in gold), len(gold)).as_dict(),
        },
        "signals": signals,
        "corpus": {"ingested": [d["path"] for d in manifest["documents"] if d["ingested"]]},
        "_rows": rows,
        "_config": config_snapshot(settings),
    }


def _corpus_hash() -> str:
    """Same definition as retrieval/ingest.py: ingested documents only."""
    import hashlib

    settings = get_settings()
    manifest = load_manifest(settings.domain)
    corpus = ROOT / "domains" / settings.domain / "corpus"
    entries = [
        (d["path"], hashlib.sha256((corpus / d["path"]).read_bytes()).hexdigest()[:16])
        for d in manifest["documents"] if d["ingested"]
    ]
    return corpus_hash(entries)


def plot(rows: list[dict[str, Any]], signal: str, threshold: float | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    covered = [r[signal] for r in rows if r["label"] == "covered"]
    uncovered = [r[signal] for r in rows if r["label"] == "uncovered"]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=140)
    bins = 20
    ax.hist(covered, bins=bins, alpha=0.65, label=f"covered (n={len(covered)})", color="#2a78d6")
    ax.hist(uncovered, bins=bins, alpha=0.65, label=f"uncovered (n={len(uncovered)})", color="#e07a1f")
    if threshold is not None:
        ax.axvline(threshold, color="#222", linestyle="--", linewidth=1.2,
                   label=f"threshold {threshold:.3f}")
    ax.set_xlabel(signal)
    ax.set_ylabel("cases")
    ax.set_title("Retrieval sufficiency: covered vs uncovered queries")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    result = asyncio.run(run())
    out = ROOT / "results" / result["run"]
    out.mkdir(parents=True, exist_ok=True)

    rows = result.pop("_rows")
    config = result.pop("_config")
    config["corpus_hash"] = _corpus_hash()
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    best = result["signals"]["top_vec_score"]["operating_point"].get("threshold")
    plot(rows, "top_vec_score", best, out / "retrieval_calibration.png")

    print(f"run: {out.relative_to(ROOT)}")
    print(f"  covered n={result['n_covered']}  uncovered n={result['n_uncovered']}")
    print(f"  top-1 document accuracy {result['accuracy']['top1_document']}")
    print(f"\n  {'signal':16} {'AUC':>6}  {'cov med':>9} {'unc med':>9}  threshold  recall(unc)  false alarm(cov)")
    for name, s in result["signals"].items():
        op = s["operating_point"]
        if op.get("threshold") is None:
            print(f"  {name:16} {s['auc']:6.3f}  — no threshold reaches the target")
            continue
        print(f"  {name:16} {s['auc']:6.3f}  {s['covered_median']:9.4f} {s['uncovered_median']:9.4f}"
              f"  {op['threshold']:9.4f}  {op['recall_uncovered']['value']:.2f} "
              f"[{op['recall_uncovered']['ci95'][0]:.2f},{op['recall_uncovered']['ci95'][1]:.2f}]"
              f"   {op['false_alarm_covered']['value']:.2f} "
              f"({op['false_alarm_covered']['successes']}/{op['false_alarm_covered']['n']})")
    misses = [r for r in rows if r["label"] == "covered" and r["gold_docs"] and not r["top1_hit"]]
    for r in misses:
        print(f"  top-1 miss: {r['case_id']} -> {r['top1_doc']} (wanted {r['gold_docs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
