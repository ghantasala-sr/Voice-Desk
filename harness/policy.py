"""The domain's handoff policy file, loaded once.

The checker and the handoff scorer both read thresholds from
`domains/<domain>/handoff.yaml`. Values tagged PROVISIONAL are placeholders
until week-3 calibration replaces them (docs/05 §6, docs/09 §5).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

DOMAINS_ROOT = Path(__file__).resolve().parent.parent / "domains"


@functools.lru_cache(maxsize=8)
def load_policy(domain: str) -> dict[str, Any]:
    path = DOMAINS_ROOT / domain / "handoff.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def retrieval_sufficiency_threshold(domain: str) -> float:
    """Raw dense cosine below which retrieval is treated as insufficient.

    Deliberately the dense score and not the fused margin, which measured
    inverted (results/2026-10-08-retrieval-signal-selection/).
    """
    signal = load_policy(domain).get("soft_signals", {}).get("retrieval_insufficient", {})
    return float(signal.get("threshold_top_vec_score", 0.65))


def claim_overlap_min(domain: str) -> float:
    return float(load_policy(domain).get("checker", {}).get("claim_overlap_min", 0.25))
