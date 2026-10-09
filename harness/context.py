"""Conversation messages, and the hygiene that runs after every barge-in.

Message format is provider-neutral and block-structured:

    {"role": "user",      "content": [{"type": "text", "text": "..."}]}
    {"role": "assistant", "content": [{"type": "text", "text": "..."},
                                      {"type": "tool_use", "id": "...",
                                       "name": "...", "input": {...}}]}
    {"role": "user",      "content": [{"type": "tool_result",
                                       "tool_use_id": "...", "content": "..."}]}

Providers translate from this. The harness never learns a vendor's shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from harness.state import CallState, TurnState

# Rough speaking rate, used to convert a playout trim in milliseconds into
# characters. ~165 wpm at ~5 characters per word.
CHARS_PER_SECOND = 14.0

_WORD_BOUNDARY = re.compile(r"\s")


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------
def user_text(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def assistant_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def append_user(messages: list[dict[str, Any]], text: str) -> None:
    """Append a user message, merging into the previous one if it is also a
    user message.

    This matters after an interruption where the caller heard nothing: the
    assistant message is removed entirely, leaving history ending on a user
    message. Appending the caller's next utterance naively would produce two
    consecutive user messages, which every provider rejects.
    """
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"].append({"type": "text", "text": text})
        return
    messages.append(user_text(text))


def text_of(message: dict[str, Any]) -> str:
    return "".join(
        block.get("text", "")
        for block in message.get("content", [])
        if block.get("type") == "text"
    )


# ---------------------------------------------------------------------------
# spoken_text reconstruction
# ---------------------------------------------------------------------------
def estimate_spoken_text(approved: str, played_chars: int | None, trim_ms: int) -> str:
    """What we believe the caller actually heard.

    Deliberately conservative. We subtract `trim_ms` worth of characters from
    the transport's reported offset and then truncate to a word boundary
    BEFORE that point — so the estimate errs toward "the caller heard less".

    That is the safe direction. If we underestimate, the agent may repeat
    itself, which is mildly awkward. If we overestimate, the agent believes it
    told the caller something it never said, and then refers back to it — which
    is the failure this whole mechanism exists to prevent.
    """
    if not approved:
        return ""
    if played_chars is None:
        # The transport could not tell us. Assume nothing was heard.
        return ""

    trim_chars = int((trim_ms / 1000) * CHARS_PER_SECOND)
    cutoff = max(0, min(len(approved), played_chars) - trim_chars)
    if cutoff <= 0:
        return ""

    head = approved[:cutoff]
    if cutoff < len(approved) and not _WORD_BOUNDARY.match(approved[cutoff]):
        # We landed mid-word; drop the partial word.
        space = head.rfind(" ")
        head = head[:space] if space > 0 else ""
    return head.strip()


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------
@dataclass
class TrimReport:
    messages_before: int
    messages_after: int
    chars_dropped: int
    orphan_tool_results_dropped: int
    assistant_message_removed: bool


def trim_on_interrupt(call: CallState, turn: TurnState, *, played_chars: int | None,
                      playout_trim_ms: int, history_offset: int = 0) -> TrimReport:
    """Rewrite history so it reflects what the caller actually heard.

    Steps, in order (docs/06 §5):
      1. Finalize spoken_text from the transport offset.
      2. Rewrite this turn's assistant message to spoken_text — or remove it
         entirely if nothing was heard. An empty assistant message is an API
         error on most providers.
      3. Drop orphaned tool_results, along with their tool_use blocks.
      4. Resolve in-flight tool executions: pending -> cancelled. Anything
         already dispatched is LEFT ALONE to resolve or reconcile (week 2,
         docs/07 §3.2). Cancelling a dispatched call is how duplicate side
         effects get created.
    """
    before = len(call.messages)

    turn.spoken_text = estimate_spoken_text(turn.approved_text, played_chars, playout_trim_ms)

    # Text before `history_offset` already went into history alongside a
    # tool_use block in an earlier iteration of this turn. Only the tail since
    # then is this message's to rewrite. (If the cut landed before the offset,
    # the earlier text stays: it was approved and almost certainly heard.)
    tail = turn.spoken_text[history_offset:].strip() if len(turn.spoken_text) > history_offset else ""
    removed = _rewrite_last_assistant(call, tail if history_offset else turn.spoken_text)
    orphans = drop_orphan_tool_results(call.messages)

    for execution in call.tool_executions:
        if execution.status == "pending":
            execution.status = "cancelled"

    return TrimReport(
        messages_before=before,
        messages_after=len(call.messages),
        # What the trim actually took away: text approved for speech that the
        # caller did not hear. Measuring history before/after would be wrong —
        # on an interrupted turn the approved text was never in history.
        chars_dropped=len(turn.approved_text) - len(turn.spoken_text),
        orphan_tool_results_dropped=orphans,
        assistant_message_removed=removed,
    )


def _rewrite_last_assistant(call: CallState, spoken: str) -> bool:
    """Make history say exactly what the caller heard. Returns True if an
    assistant message was removed.

    Three cases, and the third is the one that is easy to miss:

      * An assistant message for this turn exists and the caller heard some of
        it -> rewrite it down to spoken_text.
      * It exists and the caller heard none of it -> remove it. An empty
        assistant message is an API error on most providers.
      * It does NOT exist yet, because the interruption landed before the turn
        finished and only _finish_normally appends it -> append it now. Without
        this the agent has no record of speech the caller actually heard, which
        is the exact failure this whole mechanism exists to prevent.
    """
    for index in range(len(call.messages) - 1, -1, -1):
        role = call.messages[index]["role"]
        if role == "user":
            break  # no assistant message for this turn
        if role != "assistant":
            continue
        if spoken:
            call.messages[index] = assistant_text(spoken)
            return False
        del call.messages[index]
        return True

    if spoken:
        call.messages.append(assistant_text(spoken))
    return False


def drop_orphan_tool_results(messages: list[dict[str, Any]]) -> int:
    """Remove tool_result blocks with no surviving tool_use, and vice versa.

    An orphaned tool_result is a hard API error on most providers and a silent
    confusion on the rest, so this runs after every trim rather than being
    left to chance.
    """
    tool_use_ids = {
        block["id"]
        for message in messages
        for block in message.get("content", [])
        if block.get("type") == "tool_use"
    }
    result_ids = {
        block["tool_use_id"]
        for message in messages
        for block in message.get("content", [])
        if block.get("type") == "tool_result"
    }
    unmatched_results = result_ids - tool_use_ids
    unmatched_uses = tool_use_ids - result_ids

    dropped = 0
    for message in messages:
        kept = []
        for block in message.get("content", []):
            if block.get("type") == "tool_result" and block["tool_use_id"] in unmatched_results:
                dropped += 1
                continue
            if block.get("type") == "tool_use" and block["id"] in unmatched_uses:
                dropped += 1
                continue
            kept.append(block)
        message["content"] = kept

    # A message whose blocks were all dropped must not survive as empty.
    messages[:] = [m for m in messages if m.get("content")]
    return dropped


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------
class ContextInvariantError(AssertionError):
    """History is in a state a provider will reject or misread."""


def assert_invariant(messages: list[dict[str, Any]]) -> None:
    """Checked after every trim. Violations raise in tests rather than
    failing opaquely at the provider three steps later."""
    previous_role: str | None = None
    open_tool_uses: set[str] = set()

    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")

        if role not in ("user", "assistant"):
            raise ContextInvariantError(f"message {index}: bad role {role!r}")
        if not content:
            raise ContextInvariantError(f"message {index}: empty content")

        for block in content:
            kind = block.get("type")
            if kind == "text" and not block.get("text", "").strip():
                raise ContextInvariantError(f"message {index}: empty text block")
            if kind == "tool_use":
                open_tool_uses.add(block["id"])
            if kind == "tool_result":
                if block["tool_use_id"] not in open_tool_uses:
                    raise ContextInvariantError(
                        f"message {index}: tool_result for unknown tool_use "
                        f"{block['tool_use_id']!r}"
                    )
                open_tool_uses.discard(block["tool_use_id"])

        # Roles must alternate. Tool results live inside a user message, so
        # this holds even mid-tool-loop.
        if role == previous_role:
            raise ContextInvariantError(
                f"message {index}: two consecutive {role!r} messages"
            )
        previous_role = role

    if open_tool_uses:
        raise ContextInvariantError(f"tool_use with no tool_result: {sorted(open_tool_uses)}")
