"""OpenTelemetry setup and the span helper the harness uses.

Two independent observability systems, on purpose (docs/11):

  * OTel -> Jaeger    debugging ONE call. Rich, hierarchical, expires.
  * Postgres tables   measuring MANY calls. Flat, queryable, permanent.

Traces answer "why did this call go wrong". SQL answers "how often does this
go wrong". One system doing both does neither well.

If OpenTelemetry is not installed, or tracing is disabled, `span()` degrades to
a no-op context manager so nothing else has to care.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from core.settings import Settings

ATTR_PREFIX = "voicedesk."

_tracer: Any = None
_enabled = False


def configure_tracing(settings: Settings) -> None:
    """Call once at process start. Safe to call when OTel is absent."""
    global _tracer, _enabled

    if not settings.tracing_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover - optional at week 1
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("voicedesk")
    _enabled = True


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[_Span]:
    """Open a span. Attribute keys are prefixed with `voicedesk.` automatically.

    Attributes with a None value are dropped rather than recorded as "None",
    which keeps Jaeger's attribute filter usable.
    """
    if not _enabled or _tracer is None:
        yield _NoopSpan()
        return

    with _tracer.start_as_current_span(name) as otel_span:
        wrapper = _Span(otel_span)
        wrapper.set(**attributes)
        try:
            yield wrapper
        except Exception as exc:
            from opentelemetry.trace import Status, StatusCode

            otel_span.record_exception(exc)
            otel_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def current_trace_id() -> str | None:
    """Hex trace id, written to `turns.trace_id` — the join into Jaeger."""
    if not _enabled:
        return None
    from opentelemetry import trace

    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")


class _Span:
    def __init__(self, otel_span: Any) -> None:
        self._span = otel_span

    def set(self, **attributes: Any) -> None:
        for key, value in attributes.items():
            if value is None:
                continue
            self._span.set_attribute(f"{ATTR_PREFIX}{key}", value)

    def event(self, name: str, **attributes: Any) -> None:
        self._span.add_event(
            name, {f"{ATTR_PREFIX}{k}": v for k, v in attributes.items() if v is not None}
        )

    def mark_cancelled(self) -> None:
        """A span ended by barge-in must say so.

        Otherwise an interrupted turn looks identical to a fast one in the
        trace view, and the latency histogram quietly fills with truncated
        turns.
        """
        self.set(cancelled=True)


class _NoopSpan:
    def set(self, **attributes: Any) -> None: ...
    def event(self, name: str, **attributes: Any) -> None: ...
    def mark_cancelled(self) -> None: ...
