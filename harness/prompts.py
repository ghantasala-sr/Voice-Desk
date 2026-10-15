"""System prompt assembly.

Split into two halves at the prompt-cache breakpoint:

  static   persona, style, grounding rules, authority rules, tool descriptions.
           Byte-identical on every turn of every call, so after turn 1 it is a
           cache read rather than a fresh prefill.
  dynamic  retrieved chunks and verification state. Changes every turn, never
           cached.

If `cache_read_tokens` stays at zero on turn 2, the breakpoint is in the wrong
place and the latency work is being wasted. That is asserted in the week-1
acceptance check for W1-6, because it is otherwise invisible.

Week 1 builds the static half only. W2-4 adds the <knowledge> block.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

DOMAINS_ROOT = Path(__file__).resolve().parent.parent / "domains"


@functools.lru_cache(maxsize=8)
def load_domain(domain: str) -> dict[str, Any]:
    path = DOMAINS_ROOT / domain / "prompts.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"no prompts.yaml for domain {domain!r} (looked in {path}). "
            f"Available: {[p.name for p in DOMAINS_ROOT.iterdir() if p.is_dir()]}"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=8)
def static_system_prompt(domain: str) -> str:
    """The cacheable half. Must be byte-stable — do not interpolate anything
    call-specific here."""
    config = load_domain(domain)
    sections = [
        config["persona"].strip(),
        config["style"].strip(),
        config["grounding"].strip(),
        config["authority"].strip(),
        config["verification"].strip(),
        *([config["tools"].strip()] if config.get("tools") else []),
    ]
    return "\n\n".join(sections)


def dynamic_system_prompt(
    *,
    domain: str,
    caller_verified: bool,
    knowledge_block: str = "",
    action_note: str | None = None,
) -> str:
    """The per-turn half: verification state, any action outcome, knowledge."""
    lines = [
        "CALL CONTEXT",
        f"- Caller verified: {'yes' if caller_verified else 'no'}",
    ]
    if action_note:
        lines.append(f"- {action_note}")
    if knowledge_block:
        lines += ["", knowledge_block]
    return "\n".join(lines)


def fallback_line(domain: str, key: str) -> str:
    return load_domain(domain)["fallback_lines"][key]


def acknowledgments(domain: str) -> list[str]:
    return list(load_domain(domain).get("acknowledgments", []))


def safe_line(domain: str, kind: str) -> str:
    """The fixed line spoken in place of a segment the checker rejected."""
    return load_domain(domain)["safe_lines"][kind]
