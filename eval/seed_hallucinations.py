"""W3-9 — checker catch rate BY HALLUCINATION CLASS. docs/08 §6.

Takes known-good answers grounded in the real corpus and corrupts them one
class at a time, deterministically, then runs every segment through the
checker exactly as the turn loop does. Reports, per class, how many seeded
corruptions were caught — and, in the same table, the false-positive rate on
the uncorrupted answers. A checker that rejects everything scores 100% catch
and is useless; the false-positive row is what keeps the catch rate honest.

The two classes the checker is NOT designed to catch stay in the table with
their numbers (docs/00 D10):

  contradiction  an inverted or contradicted covered fact shares every
                 entity and most vocabulary with its chunk; no deterministic
                 rule here sees it
  binding   a number that IS in the retrieved text, attached to the wrong fact
  citation  a wrong [K] tag — stripped before checking, by design

Run: .venv/bin/python -m eval.seed_hallucinations
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from eval.metrics import Proportion
from harness.checker import Checker, GroundingSources, load_lexicon
from harness.normalize import strip_citations
from harness.tools.mocks.ticketing import EVENTS
from retrieval.chunk import chunk_document

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "domains" / "ticketing"
LEXICON = load_lexicon("ticketing")

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
INVENTED_VENUES = ["Seaview Arena", "Riverside Hall", "Copper Dome", "Maple Theatre"]


def chunks_of(doc_prefix: str) -> list[str]:
    manifest = yaml.safe_load((CORPUS / "corpus_manifest.yaml").read_text())
    entry = next(d for d in manifest["documents"] if d["path"].startswith(doc_prefix))
    text = (CORPUS / "corpus" / entry["path"]).read_text()
    return [c.embed_text for c in chunk_document(text, title=entry["title"])]


def doc_numbers(chunks: list[str]) -> set[int]:
    return {int(n) for n in re.findall(r"\b\d+\b", " ".join(chunks))}


# ---------------------------------------------------------------------------
# Seed generators — each returns (corrupted_answer, chunks, tool_texts, sufficient)
# ---------------------------------------------------------------------------
def seed_numeric(doc: str, answer: str) -> list[tuple[str, ...]]:
    chunks = chunks_of(doc)
    present = doc_numbers(chunks)
    out = []
    for match in re.finditer(r"\b\d+\b", answer):
        value = int(match.group())
        replacement = next(v for v in (value + 11, value * 3 + 1, value + 97) if v not in present)
        out.append(answer[: match.start()] + str(replacement) + answer[match.end():])
    return [(s, chunks, [], True) for s in out]


def seed_entity(doc: str, answer: str) -> list[tuple[str, ...]]:
    chunks = chunks_of(doc)
    for venue in ("Harbour Pavilion", "Kestrel Amphitheatre", "Kestrel"):
        if venue in answer:
            return [(answer.replace(venue, INVENTED_VENUES[len(venue) % 4], 1), chunks, [], True)]
    return []


def date_cases() -> list[tuple[str, str, list[str]]]:
    """Known-good date answers grounded by a lookup result, plus corruptions."""
    cases = []
    for event in EVENTS.values():
        _, month, day = (int(x) for x in event["date"].split("-"))
        tool = json.dumps({"event_name": event["name"], "event_date": event["date"]})
        good = f"Your show is on {MONTHS[month - 1]} {day}."
        bad = f"Your show is on {MONTHS[month - 1]} {day % 28 + 3}."
        cases.append((good, bad, [tool]))
    return cases


# Fabricated rules in topics NO ingested document covers, each paired with the
# question that would provoke it. Checked against what retrieval ACTUALLY
# returns for that question — the only fair test of Rule 2, which exists for
# exactly this case.
POLICY_CLAIMS = [
    ("Can I bring a folding chair to the outdoor show?",
     "You can bring a folding chair to any outdoor show."),
    ("Is there a dress code for the late session?",
     "You must wear smart dress for the late session at Lumen Hall."),
    ("Do you sell gift vouchers?",
     "You can buy gift vouchers for any amount on our website."),
    ("Do I get my parking money back if the show is cancelled?",
     "We always refund parking when a show is cancelled."),
    ("Can I smoke anywhere at the venue?",
     "You can smoke in the designated areas at every venue."),
    ("Do young kids get in for free?",
     "Children under five always get in free."),
    ("What if I lose my phone with the tickets on it?",
     "We will always reissue lost tickets at the gate free of charge."),
    ("Can I meet the band after the show?",
     "You can meet the band after every show as long as you ask at the merch stand."),
]

# Covered facts, inverted or contradicted. Retrieval is strong for these, and
# they share their chunk's vocabulary by construction — the declared gap.
CONTRADICTIONS = [
    ("01", "Tickets are non-refundable once the order is complete.",
           "Tickets are refundable once the order is complete."),
    ("02", "Will-call orders can't be transferred.", "Will-call orders can be transferred."),
    ("05", "Backpacks of any size are refused at Harbour Pavilion.",
           "Backpacks of any size are allowed at Harbour Pavilion."),
    ("07", "Emotional support animals are not permitted.",
           "Emotional support animals are permitted."),
    ("09", "Resale tickets cannot be exchanged.", "Resale tickets can be exchanged."),
    ("08", "We do not accept cash.", "We accept cash."),
    ("08", "One order can only use one payment method.",
           "You can split a single order across as many cards as you like."),
    ("09", "Listings stay up until the event starts.",
           "You can resell tickets any time, even after the event has started."),
    ("05", "Outside food and drink is not allowed inside.",
           "You can bring outside food for children at no charge."),
]

# Correct answers to covered questions whose retrieval scored WEAK in the
# calibration run. Rule 2 can only fire under weak retrieval, so this — not
# the strong-retrieval known-good set — is where its false positives live.
KNOWN_GOOD_WEAK = [
    ("Do you have anything for people who are hard of hearing?",
     "Harbour Pavilion, Northgate Arena and Lumen Hall lend assistive listening receivers from guest services."),
    ("Can I bring my guide dog?",
     "You can bring your guide dog. Trained service animals are welcome and don't need to be registered."),
    ("My card keeps getting rejected every time I try to check out.",
     "The most common reason is a billing postcode that doesn't match the one your bank has on file."),
    ("The show got cancelled last week. When am I actually going to see the money?",
     "Cancelled events are refunded automatically to your original payment method."),
    ("I've got a spare ticket — how do I get it over to my sister?",
     "Open your order, choose Transfer, and enter your sister's email address."),
    ("What do I need to bring with me to pick my tickets up at the box office?",
     "Bring photo ID that matches the name on the order to the box office."),
    ("Something came up and I can't go on Saturday anymore. Do I get my money back?",
     "Tickets aren't refundable if you can't attend, but you can transfer them to someone else at no charge."),
    ("They moved the date and I can't make the new one. What are my options?",
     "If you can't make the new date, you can request a refund within 30 days of it being announced."),
    ("What's the most I'm allowed to resell for?",
     "Where the organizer has set a cap, you can list at up to face value plus 10 percent."),
    ("Is there somewhere to leave my bag at Lumen Hall?",
     "Lumen Hall has bag check inside the main entrance at 10 dollars per item."),
]

OVERLAP_SWEEP = (0.25, 0.35, 0.50, 0.65, 0.80)


BINDING = [
    # A number present in the retrieved text, attached to the wrong fact.
    ("01", "Refunds are processed within 10 business days."),
    ("04", "Will-call opens 48 hours before doors."),
    ("06", "Parking at Kestrel Amphitheatre costs 25 dollars."),
    ("08", "Duplicate charges drop off within 18 business days."),
]


def check(answer: str, chunks: list[str], tools: list[str], sufficient: bool,
          overlap: float | None = None, caller: list[str] | None = None) -> str:
    from harness.policy import claim_overlap_min

    overlap = claim_overlap_min("ticketing") if overlap is None else overlap
    clean, _ = strip_citations(answer)
    checker = Checker(
        GroundingSources(chunks=chunks, tool_results=tools, caller=caller or []),
        lexicon=LEXICON, retrieval_sufficient=sufficient, claim_overlap_min=overlap,
    )
    return checker.check(clean).verdict


def retrieve_for(questions: list[str]) -> dict[str, tuple[list[str], bool, float]]:
    """Real hybrid retrieval: chunks the turn loop would see, and whether the
    calibrated sufficiency threshold would call it sufficient."""
    import asyncio

    from core import db
    from core.settings import get_settings
    from harness.policy import retrieval_sufficiency_threshold
    from retrieval.embed import get_embedder
    from retrieval.search import search

    async def run() -> dict[str, tuple[list[str], bool, float]]:
        settings = get_settings()
        await db.open_pool(settings)
        embedder = get_embedder(settings)
        threshold = retrieval_sufficiency_threshold(settings.domain)
        out = {}
        for q in questions:
            r = await search(q, domain=settings.domain, embedder=embedder, settings=settings)
            out[q] = ([f"{c.heading_path}\n{c.text}" for c in r.chunks],
                      r.top_vec_score >= threshold, r.top_vec_score)
        await db.close_pool()
        return out

    return asyncio.run(run())


def main() -> int:
    known = yaml.safe_load((ROOT / "eval" / "dataset" / "checker_known_good.yaml").read_text())["answers"]
    rows: list[dict[str, Any]] = []

    def record(cls: str, answer: str, chunks: list[str], tools: list[str], sufficient: bool,
               seeded: bool, caller: list[str] | None = None) -> None:
        # The turn loop passes everything the caller said as a grounding
        # source; so does this.
        verdict = check(answer, chunks, tools, sufficient, caller=caller)
        rows.append({"class": cls, "seeded": seeded, "answer": answer, "verdict": verdict,
                     "caught": verdict.startswith("fail")})

    for entry in known:
        record("known_good", entry["answer"], chunks_of(entry["doc"]), [], True, False)
        for s in seed_numeric(entry["doc"], entry["answer"]):
            record("numeric", *s, True)
        for s in seed_entity(entry["doc"], entry["answer"]):
            record("entity", *s, True)
        # Citation: a wrong tag. Stripped before checking, so never caught.
        record("citation", entry["answer"].rstrip(".") + " [K4].", chunks_of(entry["doc"]), [], True, True)

    for good, bad, tools in date_cases():
        record("known_good", good, [], tools, True, False)
        record("date", bad, [], tools, True, True)
    retrieved = retrieve_for([q for q, _ in POLICY_CLAIMS] + [q for q, _ in KNOWN_GOOD_WEAK])
    for question, claim in POLICY_CLAIMS:
        chunks, sufficient, score = retrieved[question]
        record("policy_claim", claim, chunks, [], sufficient, True, caller=[question])
        rows[-1]["top_vec_score"] = round(score, 3)
    for doc, good, bad in CONTRADICTIONS:
        record("known_good", good, chunks_of(doc), [], True, False)
        record("contradiction", bad, chunks_of(doc), [], True, True)
    for question, answer in KNOWN_GOOD_WEAK:
        chunks, sufficient, score = retrieved[question]
        record("known_good_weak", answer, chunks, [], sufficient, False, caller=[question])
        rows[-1]["top_vec_score"] = round(score, 3)
    for doc, claim in BINDING:
        record("binding", claim, chunks_of(doc), [], True, True)

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)

    order = ["numeric", "date", "entity", "policy_claim", "contradiction", "binding", "citation"]
    expected = {"contradiction": "declared gap", "binding": "declared gap",
                "citation": "not an enforcement target"}
    table = {
        cls: {**Proportion(sum(r["caught"] for r in by_class[cls]), len(by_class[cls])).as_dict(),
              "note": expected.get(cls, "")}
        for cls in order
    }
    good = by_class["known_good"] + by_class["known_good_weak"]
    false_positive = Proportion(sum(r["caught"] for r in good), len(good)).as_dict()

    # Rule 2's trade-off, made visible: catch on fabricated claims against
    # false positives on correct answers under weak retrieval.
    sweep = []
    for overlap in OVERLAP_SWEEP:
        caught = sum(
            check(claim, retrieved[q][0], [], retrieved[q][1], overlap, caller=[q]).startswith("fail")
            for q, claim in POLICY_CLAIMS
        )
        fps = sum(
            check(a, retrieved[q][0], [], retrieved[q][1], overlap, caller=[q]).startswith("fail")
            for q, a in KNOWN_GOOD_WEAK
        )
        sweep.append({"claim_overlap_min": overlap,
                      "policy_claim_caught": Proportion(caught, len(POLICY_CLAIMS)).as_dict(),
                      "weak_retrieval_false_positives": Proportion(fps, len(KNOWN_GOOD_WEAK)).as_dict()})

    run = f"{date.today().isoformat()}-checker-catch-rate"
    out = ROOT / "results" / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "metrics.json").write_text(json.dumps(
        {"run": run, "catch_by_class": table, "false_positive_rate": false_positive,
         "rule2_overlap_sweep": sweep}, indent=2))

    print(f"run: results/{run}\n")
    print(f"  {'class':14} {'caught':>9}  {'rate':>5}  95% CI         note")
    for cls in order:
        t = table[cls]
        rate = f"{t['value']:.2f}" if t["value"] is not None else "—"
        print(f"  {cls:14} {t['successes']:>3}/{t['n']:<5}  {rate:>5}  "
              f"[{t['ci95'][0]:.2f}, {t['ci95'][1]:.2f}]   {t['note']}")
    fp = false_positive
    print(f"  {'-' * 60}\n  false positives on known-good: {fp['successes']}/{fp['n']}"
          f"  [{fp['ci95'][0]:.2f}, {fp['ci95'][1]:.2f}]")
    print("\n  Rule 2 sweep  (overlap min -> fabricated claims caught | correct weak-retrieval answers rejected)")
    for row in sweep:
        c, f = row["policy_claim_caught"], row["weak_retrieval_false_positives"]
        print(f"    {row['claim_overlap_min']:.2f}   caught {c['successes']}/{c['n']}   false positives {f['successes']}/{f['n']}")
    fp_rows = [r for r in rows if r["class"] in ("known_good", "known_good_weak") and r["caught"]]
    for r in fp_rows:
        print(f"  false positive: {r['answer']!r} -> {r['verdict']}")
    missed = [r for r in rows if r["seeded"] and not r["caught"] and r["class"] in ("numeric", "date", "entity", "policy_claim")]
    for r in missed:
        print(f"  missed {r['class']}: {r['answer']!r} -> {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
