"""Provider construction and the config snapshot.

Three interfaces, each with at least two implementations. The second
implementation is not decoration — it is what proves the interface is an
abstraction rather than a wrapper around one vendor's SDK (docs/04).
"""

from __future__ import annotations

import subprocess
from typing import Any

from core.settings import Settings
from providers.llm.base import LLMProvider
from providers.stt.base import STTProvider
from providers.tts.base import TTSProvider


def offline_fallback(settings: Settings) -> Settings:
    """Claude when a key is set; otherwise the offline extractive agent.

    Every result and the web app say which one ran — a fallback that is not
    reported would make offline numbers look like Claude's."""
    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        return settings.model_copy(update={"llm_provider": "extractive"})
    return settings


def build_llm(settings: Settings, **kwargs: Any) -> LLMProvider:
    match settings.llm_provider:
        case "anthropic":
            from providers.llm.anthropic import AnthropicProvider

            return AnthropicProvider(settings)
        case "openai":
            from providers.llm.openai import OpenAIProvider

            return OpenAIProvider(settings)
        case "replay":
            from providers.llm.replay import ReplayProvider

            return ReplayProvider(**kwargs)
        case "extractive":
            from providers.llm.extractive import ExtractiveProvider

            return ExtractiveProvider()
        case other:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown LLM provider: {other}")


def build_stt(settings: Settings) -> STTProvider:
    match settings.stt_provider:
        case "deepgram":
            from providers.stt.deepgram import DeepgramProvider

            return DeepgramProvider(settings)
        case "faster_whisper":
            from providers.stt.faster_whisper import FasterWhisperProvider

            return FasterWhisperProvider(settings)
        case "null":
            from providers.stt.null import NullSTT

            return NullSTT()
        case other:  # pragma: no cover
            raise ValueError(f"unknown STT provider: {other}")


def build_tts(settings: Settings) -> TTSProvider:
    match settings.tts_provider:
        case "cartesia":
            from providers.tts.cartesia import CartesiaProvider

            return CartesiaProvider(settings)
        case "null":
            from providers.tts.null import NullTTS

            return NullTTS()
        case other:  # pragma: no cover
            raise ValueError(f"unknown TTS provider: {other}")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def config_snapshot(settings: Settings, *, corpus_hash: str = "") -> dict[str, Any]:
    """Everything that could change a number.

    Written to `calls.config_snapshot` and `eval_runs.config_snapshot`. A number
    without its snapshot is not a result (docs/15 §1).
    """
    return {
        "git_sha": git_sha(),
        "llm": {
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        },
        "stt": {"provider": settings.stt_provider},
        "tts": {"provider": settings.tts_provider, "voice_id": settings.cartesia_voice_id},
        "embed": {"model": settings.embed_model},
        "retrieval": {"top_k": settings.retrieval_top_k, "rrf_k": settings.rrf_k},
        "vad": {
            "stop_secs": settings.vad_stop_secs,
            "start_secs": settings.vad_start_secs,
            "confidence": settings.vad_confidence,
        },
        "budgets": {
            "turn_budget_ms": settings.turn_budget_ms,
            "tool_timeout_ms": settings.tool_timeout_ms,
            "max_tool_iterations": settings.max_tool_iterations,
            "playout_trim_ms": settings.playout_trim_ms,
        },
        "domain": settings.domain,
        "corpus_hash": corpus_hash,
    }
