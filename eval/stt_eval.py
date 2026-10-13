"""W3-10 — STT accuracy and cost under noise and phone audio. docs/12 §4-5.

Transcribes every file in audio_manifest.json and scores it against the
utterance text: WER, and accuracy on KEY ENTITIES (order ids, emails,
quantities) scored per entity. Entity accuracy is the number that matters for
a support agent — a 3% WER that drops the order number is a failed call.

Framed as the plan says (docs/00 D14): what SELF-HOSTING ASR costs on a
real-time path. faster-whisper is a batch model; its decode time per second of
audio is reported beside its accuracy. The streaming Deepgram side of the
comparison needs a key and is not run.

Run: .venv/bin/python -m eval.stt_eval [--model small.en] [--sample N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from eval.dataset import DATASET, load_cases
from eval.metrics import Proportion, entity_hits, percentiles, wer
from providers import git_sha

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["clean", "snr20", "snr10", "snr5", "telephony_clean", "telephony_snr10"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--sample", type=int, default=0, help="fixed-seed subset; tagged in metrics")
    args = parser.parse_args()

    from providers.stt.faster_whisper import FasterWhisperProvider

    stt = FasterWhisperProvider(model_size=args.model)
    cases = {c.id: c for c in load_cases()}
    files = json.loads((DATASET / "audio_manifest.json").read_text())["files"]
    if args.sample:
        files = random.Random(7).sample(files, args.sample)

    rows: list[dict[str, Any]] = []
    for i, entry in enumerate(files, 1):
        case = cases[entry["case_id"]]
        transcript, elapsed = stt.transcribe_timed(str(DATASET / entry["path"]))
        correct, total = entity_hits(case.key_entities, transcript.text)
        rows.append({
            **{k: entry[k] for k in ("case_id", "voice", "variant")},
            "noise": entry.get("noise"), "seconds": entry["seconds"],
            "reference": case.utterance, "hypothesis": transcript.text,
            "wer": wer(case.utterance, transcript.text),
            "entity_correct": correct, "entity_total": total,
            "decode_s": round(elapsed, 3), "rtf": round(elapsed / entry["seconds"], 4),
            "min_word_conf": min((p for _, p in transcript.word_confidences), default=None),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        ent_c = sum(r["entity_correct"] for r in subset)
        ent_t = sum(r["entity_total"] for r in subset)
        return {
            "n_files": len(subset),
            "wer_mean": round(sum(r["wer"] for r in subset) / len(subset), 4),
            "wer_p50_p95": percentiles([r["wer"] for r in subset]),
            "entity_accuracy": Proportion(ent_c, ent_t).as_dict(),
            "rtf_p50_p95": percentiles([r["rtf"] for r in subset]),
        }

    by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by["variant"][r["variant"]].append(r)
        by["voice"][r["voice"]].append(r)
        if r["noise"]:
            by["noise"][r["noise"]].append(r)

    run = f"{date.today().isoformat()}-stt-faster-whisper-{args.model.replace('.', '')}"
    out = ROOT / "results" / run
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run": run, "sampled": bool(args.sample), "provider": stt.describe(), "git_sha": git_sha(),
        "audio": "synthesized with macOS `say`, 3 voices — cleaner than human speech",
        "overall": summarize(rows),
        "by_variant": {v: summarize(by["variant"][v]) for v in ORDER if by["variant"][v]},
        "by_voice": {v: summarize(x) for v, x in sorted(by["voice"].items())},
        "by_noise_type": {v: summarize(x) for v, x in sorted(by["noise"].items())},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "raw.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"\nrun: results/{run}   files={len(rows)}   model={args.model} {stt.compute_type} {stt.device}")
    print(f"  {'variant':18} {'WER mean':>9} {'entity acc':>28} {'RTF p50':>8} {'RTF p95':>8}")
    for v, m in metrics["by_variant"].items():
        e = m["entity_accuracy"]
        acc = (f"{e['value']:.2f} [{e['ci95'][0]:.2f},{e['ci95'][1]:.2f}] ({e['successes']}/{e['n']})"
               if e["n"] else "—  (no entities)")
        print(f"  {v:18} {m['wer_mean']:9.3f}   {acc:>26}  "
              f"{m['rtf_p50_p95']['p50']:8.3f} {m['rtf_p50_p95']['p95']:8.3f}")
    print("  by voice:", {v: m["wer_mean"] for v, m in metrics["by_voice"].items()})
    print("  by noise:", {v: m["wer_mean"] for v, m in metrics["by_noise_type"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
