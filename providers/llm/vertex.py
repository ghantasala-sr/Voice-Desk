"""Google Cloud Vertex AI implementations of LLMProvider.

Two models behind one cloud, both authenticated by gcloud Application Default
Credentials (`gcloud auth application-default login`) — no key in `.env`:

  * `vertex_gemini` — Gemini through the google-genai SDK. Its own provider,
    because the wire shape differs from Anthropic's in four ways absorbed here:

      - System prompt is `system_instruction`, one string. Static half first so
        the prefix is stable for Vertex's implicit caching.
      - Tool calls arrive COMPLETE (a `function_call` part with parsed args),
        not as JSON fragments — rule 1 of the interface holds trivially.
      - Tool results go back as `function_response` parts, which need the
        tool NAME, not just the id. The name is recovered from the matching
        tool_use block earlier in the history.
      - Thinking models sign their function calls (`thought_signature`), and
        the signature must be echoed on the next request or the call fails.
        The harness's Anthropic-shaped history has no slot for it, so the
        provider keeps signatures keyed by tool-call id.

  * `vertex_claude` — Claude through `AsyncAnthropicVertex`. Identical Messages
    API, so it is the Anthropic provider with a different client.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncIterator
from typing import Any

from core.ids import new_id
from core.settings import Settings
from providers.llm.base import (
    LLMEvent,
    LLMProvider,
    LLMRequest,
    StreamEnd,
    TextDelta,
    ToolCallDelta,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def resolve_project(settings: Settings) -> str:
    """VD_GCP_PROJECT, then GOOGLE_CLOUD_PROJECT, then ADC, then gcloud config."""
    if settings.gcp_project:
        return settings.gcp_project
    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return os.environ["GOOGLE_CLOUD_PROJECT"]
    try:
        import google.auth

        _, project = google.auth.default()
        if project:
            return str(project)
    except Exception:
        pass
    try:
        project = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        project = ""
    if not project:
        raise RuntimeError(
            "No Google Cloud project. Set VD_GCP_PROJECT, or run "
            "`gcloud config set project <id>` and `gcloud auth application-default login`."
        )
    return project


def claude_on_vertex(settings: Settings) -> LLMProvider:
    from anthropic import AsyncAnthropicVertex

    from providers.llm.anthropic import AnthropicProvider

    client = AsyncAnthropicVertex(project_id=resolve_project(settings),
                                  region=settings.vertex_location)
    provider = AnthropicProvider(settings, client=client)
    provider.name = "vertex_claude"
    return provider


class VertexGeminiProvider(LLMProvider):
    name = "vertex_gemini"

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        from providers import effective_model

        self.model = effective_model(settings)
        if client is None:
            from google import genai

            client = genai.Client(vertexai=True, project=resolve_project(settings),
                                  location=settings.vertex_location)
        self._client = client
        self._signatures: dict[str, bytes] = {}
        self._thinking_budget = settings.llm_thinking_budget

    async def aclose(self) -> None:
        close = getattr(getattr(self._client, "aio", None), "aclose", None)
        if close is not None:
            await close()

    def _config(self, request: LLMRequest) -> Any:
        from google.genai import types

        system = request.system_static
        if request.system_dynamic:
            system = f"{system}\n\n{request.system_dynamic}"
        config: dict[str, Any] = {
            "system_instruction": system,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
            # The harness runs the tool loop; the SDK must never call tools itself.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
            # The harness's deadline, enforced at the HTTP layer. Without it a
            # slow Vertex response held one measured turn for 35.8 s instead of
            # failing over to the harness's retry / fallback line.
            "http_options": types.HttpOptions(timeout=request.timeout_ms),
        }
        budget = self._thinking_budget
        if budget is None and "flash" in self.model and self.model.startswith("gemini-2.5"):
            # Thinking adds first-token latency that a phone call cannot afford,
            # and on 2.5 Flash it can be switched off entirely.
            budget = 0
        if budget is not None:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
        if request.tools:
            config["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t.get("description", ""),
                                          parameters_json_schema=t["parameters"])
                for t in request.tools])]
        return types.GenerateContentConfig(**config)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMEvent]:
        input_tokens = output_tokens = cache_read = 0
        stop_reason = "end_turn"
        saw_tool = False
        stream: Any = None
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=to_gemini_contents(request.messages, self._signatures),
                config=self._config(request),
            )
            async for chunk in stream:
                usage = getattr(chunk, "usage_metadata", None)
                if usage is not None:
                    input_tokens = usage.prompt_token_count or input_tokens
                    output_tokens = usage.candidates_token_count or output_tokens
                    cache_read = usage.cached_content_token_count or cache_read
                for candidate in chunk.candidates or []:
                    for part in (candidate.content.parts if candidate.content else None) or []:
                        if getattr(part, "thought", False):
                            continue  # reasoning summaries are not speech
                        if part.function_call is not None:
                            call = part.function_call
                            call_id = call.id or f"gem_{new_id()}"
                            if part.thought_signature:
                                self._signatures[call_id] = part.thought_signature
                            saw_tool = True
                            yield ToolCallDelta(call_id=call_id, name=call.name or "",
                                                arguments=dict(call.args or {}))
                        elif part.text:
                            yield TextDelta(text=part.text)
                    reason = getattr(candidate.finish_reason, "name", candidate.finish_reason)
                    if reason == "MAX_TOKENS":
                        stop_reason = "max_tokens"
                    elif reason not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
                        # SAFETY, MALFORMED_FUNCTION_CALL, ...: not an answer.
                        yield StreamEnd(stop_reason="error", input_tokens=input_tokens,
                                        output_tokens=output_tokens, cache_read_tokens=cache_read,
                                        error=f"gemini finish_reason={reason}")
                        return
        except Exception as exc:  # auth, quota (429), network, timeout
            yield StreamEnd(stop_reason="error", input_tokens=input_tokens,
                            output_tokens=output_tokens, cache_read_tokens=cache_read,
                            error=f"{type(exc).__name__}: {exc}")
            return
        finally:
            # Rule 2: closing our iterator must close the upstream stream.
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

        yield StreamEnd(stop_reason="tool_use" if saw_tool and stop_reason == "end_turn" else stop_reason,
                        input_tokens=input_tokens, output_tokens=output_tokens,
                        cache_read_tokens=cache_read)


def to_gemini_contents(messages: list[dict[str, Any]],
                       signatures: dict[str, bytes] | None = None) -> list[Any]:
    """Neutral (Anthropic-shaped) history -> Gemini `Content` list."""
    from google.genai import types

    signatures = signatures or {}
    names: dict[str, str] = {}
    contents: list[types.Content] = []
    for message in messages:
        blocks = message.get("content", [])
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        role = "model" if message["role"] == "assistant" else "user"
        parts: list[types.Part] = []
        for b in blocks:
            if b.get("type") == "text" and b.get("text"):
                parts.append(types.Part(text=b["text"]))
            elif b.get("type") == "tool_use":
                names[b["id"]] = b["name"]
                parts.append(types.Part(
                    function_call=types.FunctionCall(id=b["id"], name=b["name"],
                                                     args=b.get("input", {})),
                    thought_signature=signatures.get(b["id"])))
            elif b.get("type") == "tool_result":
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=b["tool_use_id"], name=names.get(b["tool_use_id"], "tool"),
                    response={"result": str(b.get("content", ""))})))
        if parts:
            contents.append(types.Content(role=role, parts=parts))
    return contents
