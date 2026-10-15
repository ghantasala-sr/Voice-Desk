"""Configuration, loaded once from the environment and `.env`.

Rule: no magic numbers in code. Anything a tuning log might change lives here
or in a domain YAML file. See docs/02-repo-and-tooling.md §3.

Every non-secret has a working default, so a minimal `.env` only needs
ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import functools
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal[
    "anthropic", "openai", "replay", "extractive", "vertex_gemini", "vertex_claude"
]
STTProviderName = Literal["deepgram", "faster_whisper", "null"]
TTSProviderName = Literal["cartesia", "null"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- provider keys ------------------------------------------------------
    anthropic_api_key: SecretStr | None = Field(None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")
    deepgram_api_key: SecretStr | None = Field(None, validation_alias="DEEPGRAM_API_KEY")
    cartesia_api_key: SecretStr | None = Field(None, validation_alias="CARTESIA_API_KEY")
    cartesia_voice_id: str = Field("", validation_alias="CARTESIA_VOICE_ID")

    # -- provider selection -------------------------------------------------
    llm_provider: LLMProviderName = Field("anthropic", validation_alias="VD_LLM_PROVIDER")
    llm_model: str = Field("claude-sonnet-5", validation_alias="VD_LLM_MODEL")
    # Google Cloud Vertex AI. Auth is gcloud Application Default Credentials
    # (`gcloud auth application-default login`), never a key in .env. The
    # project falls back to GOOGLE_CLOUD_PROJECT, then `gcloud config`.
    gcp_project: str | None = Field(None, validation_alias="VD_GCP_PROJECT")
    vertex_location: str = Field("global", validation_alias="VD_VERTEX_LOCATION")
    # Gemini thinking tokens. None = provider default (0 on 2.5 Flash, where it
    # can be switched off; the model's own default elsewhere).
    llm_thinking_budget: int | None = Field(None, validation_alias="VD_LLM_THINKING_BUDGET")
    llm_temperature: float = Field(0.2, validation_alias="VD_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(512, validation_alias="VD_LLM_MAX_TOKENS")
    llm_timeout_ms: int = Field(10_000, validation_alias="VD_LLM_TIMEOUT_MS")

    stt_provider: STTProviderName = Field("deepgram", validation_alias="VD_STT_PROVIDER")
    tts_provider: TTSProviderName = Field("cartesia", validation_alias="VD_TTS_PROVIDER")

    # -- database -----------------------------------------------------------
    database_url: str = Field(
        "postgresql://voicedesk:voicedesk@localhost:5434/voicedesk",
        validation_alias="VD_DATABASE_URL",
    )
    db_pool_min: int = Field(1, validation_alias="VD_DB_POOL_MIN")
    db_pool_max: int = Field(8, validation_alias="VD_DB_POOL_MAX")

    # -- domain -------------------------------------------------------------
    domain: str = Field("ticketing", validation_alias="VD_DOMAIN")

    # -- turn-taking (docs/10 §2) -------------------------------------------
    # vad_stop_secs is the largest single term in voice-to-voice latency.
    # Tuned by W1-9; do not lower it to improve a latency number.
    vad_stop_secs: float = Field(0.4, validation_alias="VD_VAD_STOP_SECS")
    vad_start_secs: float = Field(0.2, validation_alias="VD_VAD_START_SECS")
    vad_confidence: float = Field(0.7, validation_alias="VD_VAD_CONFIDENCE")
    silence_timeout_ms: int = Field(8_000, validation_alias="VD_SILENCE_TIMEOUT_MS")

    # -- budgets (docs/01 §5) -----------------------------------------------
    turn_budget_ms: int = Field(2_500, validation_alias="VD_TURN_BUDGET_MS")
    tool_filler_ms: int = Field(700, validation_alias="VD_TOOL_FILLER_MS")
    tool_timeout_ms: int = Field(4_000, validation_alias="VD_TOOL_TIMEOUT_MS")
    max_tool_iterations: int = Field(3, validation_alias="VD_MAX_TOOL_ITERATIONS")
    readback_timeout_ms: int = Field(6_000, validation_alias="VD_READBACK_TIMEOUT_MS")
    ack_after_ms: int = Field(350, validation_alias="VD_ACK_AFTER_MS")

    # The grounding checker. ON by default and in production. It is only
    # switched off by tests that exercise loop mechanics with ungrounded
    # fixture text; checker behaviour itself is tested against real chunks.
    checker_enabled: bool = Field(True, validation_alias="VD_CHECKER_ENABLED")
    # Same reasoning: ON in production, off only for loop-mechanics tests
    # whose fixtures would otherwise trip the retrieval signal on every turn.
    handoff_enabled: bool = Field(True, validation_alias="VD_HANDOFF_ENABLED")
    # benign | slow | flaky | ambiguous | blackout — harness/tools/mocks/faults.py
    tool_fault_profile: str = Field("benign", validation_alias="VD_TOOL_FAULT_PROFILE")
    max_verification_failures: int = Field(2, validation_alias="VD_MAX_VERIFICATION_FAILURES")

    # How much audio handed to the transport we assume the caller did NOT hear
    # when they interrupted. Errs toward "they heard less" — the safe direction.
    playout_trim_ms: int = Field(120, validation_alias="VD_PLAYOUT_TRIM_MS")

    # -- retrieval (week 2) -------------------------------------------------
    embed_model: str = Field("BAAI/bge-small-en-v1.5", validation_alias="VD_EMBED_MODEL")
    retrieval_top_k: int = Field(4, validation_alias="VD_RETRIEVAL_TOP_K")
    rrf_k: int = Field(60, validation_alias="VD_RRF_K")

    # -- observability ------------------------------------------------------
    tracing_enabled: bool = Field(True, validation_alias="VD_TRACING_ENABLED")
    otlp_endpoint: str = Field(
        "http://localhost:4319", validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    service_name: str = Field("voicedesk", validation_alias="OTEL_SERVICE_NAME")
    log_level: str = Field("INFO", validation_alias="VD_LOG_LEVEL")
    log_json: bool = Field(False, validation_alias="VD_LOG_JSON")

    # -- helpers ------------------------------------------------------------
    def require(self, field: str) -> str:
        """Read a secret, failing with a useful message instead of a 401 later."""
        value = getattr(self, field, None)
        if value is None or (isinstance(value, SecretStr) and not value.get_secret_value()):
            env_name = type(self).model_fields[field].validation_alias
            raise RuntimeError(
                f"{env_name} is not set. Add it to .env (see .env.example)."
            )
        return value.get_secret_value() if isinstance(value, SecretStr) else str(value)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Cached so `.env` is read once.

    Tests that need different values construct `Settings(...)` directly rather
    than mutating this, so they cannot leak configuration into each other.
    """
    return Settings()
