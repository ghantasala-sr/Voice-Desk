# results/

One directory per run. Never edit a run after the fact — supersede it with a new one.

```
results/<YYYY-MM-DD>-<label>/
  README.md     ← hand-written. Three questions, below.
  config.json   ← config snapshot + git sha + corpus hash
  metrics.json  ← computed by eval/metrics.py. The only file other tools read.
  raw.jsonl     ← one line per (case, voice, variant)
  failures.md   ← every failure with transcript, retrieval, verdicts
  *.png         ← calibration plots, PR curves
  audio/        ← TTS output for failed cases
```

Every run README answers exactly three questions:

1. What was this run for?
2. What did it show?
3. What changed as a result?

## The rule

**Never report a number without the run directory that produced it.**

Full reporting rules — `n` and confidence intervals on every proportion, both latency
numbers with the endpointing constant, labelled denominators, replay vs live, synthesized
vs human audio, simulated vs real calls — are in
[../docs/15-results-and-claims.md](../docs/15-results-and-claims.md) §4.

A run with `timing_valid: false` (a `--fast` replay) or `sampled: true` (a `--sample N`
run) may not be quoted as a full result.
