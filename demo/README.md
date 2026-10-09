# Demo scripts

Each beat is a scripted conversation run through the REAL harness — real
retrieval over the ingested corpus, the grounding checker, the tool executor,
the idempotency ledger, handoff — with only the LLM replayed from a fixture so
the run is deterministic and free. Every call is recorded to Postgres, so the
console shows it live.

```bash
make up && make ingest
make dev            # in another terminal; console at http://localhost:8088/console
make demo-scripts
```

| Beat | Script | What to watch in the console |
|---|---|---|
| 1 | `1_policy_answer.yaml` | retrieval scores, `[K1]` recorded in cited chunks but not spoken, checker `pass` per segment |
| 2 | `2_transfer_interrupted.yaml` | read-back spoken → caller cuts in → `readback_rejected`, ledger `cancelled` → asked again → `yes` → commit-then-timeout → `tool_reconciled` → one `committed` row, attempt 2 |
| 3 | `3_out_of_policy_handoff.yaml` | `handoff_triggered` with zero model calls on the turn; the summary's "has NOT promised" line |
| 4 | `4_checker_catch.yaml` | a `fail_ungrounded_entity` verdict; the invented fee appears in generated text but never in spoken text |

The duplicate-side-effects tile stays at 0 throughout.

The LLM is replayed because the point of these scripts is the harness's
behaviour, which must be identical on every run. `make chat` runs the same
stack against the live model.
