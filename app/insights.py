"""Read-only API behind the app's Insights, Results and status banner.

Two sources, kept apart on purpose:

  * `results/<run>/` — measured runs. Every number the Results and Insights
    views show comes from a run directory, so it can be quoted with its run
    name (docs/15). The README is returned as raw markdown text; the browser
    renders it into DOM nodes and never treats it as HTML.
  * Postgres `gap_reports` — the latest week-4 gap report.

Path handling is the only security-relevant part. A run name or file name is
never joined onto a path as given: it must EQUAL one of the names produced by
listing the directory, so `..`, encoded separators and symlinks out of
`results/` all reduce to "not found".

`/api/status` reports which keys exist as booleans. Key values never leave
the process.
"""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from app.chat import agent_settings
from core import db
from core.logging import get_logger
from core.settings import get_settings

log = get_logger(__name__)

router = APIRouter()

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Run directories are dated slugs; anything else in results/ is not a run.
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_SHOWED = re.compile(r"what (it|did it|this run) show|final numbers", re.I)

# Headline numbers per card. Enough to say what a run measured; the README
# carries the rest.
HEADLINE_LIMIT = 4
SUMMARY_CHARS = 320


def _not_found(detail: str = "not found") -> JSONResponse:
    return JSONResponse({"error": detail}, status_code=404)


# ---------------------------------------------------------------------------
# results/
# ---------------------------------------------------------------------------
def _runs(root: Path | None = None) -> dict[str, Path]:
    """Every run directory, by name. The only source of valid names."""
    root = root or RESULTS
    if not root.is_dir():
        return {}
    resolved_root = root.resolve()
    runs: dict[str, Path] = {}
    for entry in root.iterdir():
        if not _RUN_NAME.match(entry.name) or entry.is_symlink() or not entry.is_dir():
            continue
        if entry.resolve().parent != resolved_root:
            continue
        runs[entry.name] = entry
    return runs


def _pngs(run: Path) -> list[str]:
    resolved = run.resolve()
    return sorted(
        f.name for f in run.iterdir()
        if f.suffix.lower() == ".png" and f.is_file() and not f.is_symlink()
        and f.resolve().parent == resolved
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _plain(markdown: str) -> str:
    """Markdown line(s) -> plain text, for a card's one-paragraph summary."""
    text = re.sub(r"`([^`]*)`", r"\1", markdown)
    text = re.sub(r"\*\*([^*]*)\*\*|__([^_]*)__", lambda m: m.group(1) or m.group(2) or "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?<!\w)[*_]([^*_]+)[*_](?!\w)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def readme_outline(markdown: str) -> tuple[str, str]:
    """(first heading, first paragraph of the "What it showed" section)."""
    title = ""
    lines = markdown.splitlines()
    for line in lines:
        if line.startswith("#"):
            title = _plain(line.lstrip("#"))
            break

    in_section = False
    paragraph: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if line.startswith("#"):
            if paragraph:
                break
            heading = line.lstrip("#").strip()
            if line.startswith("## "):
                in_section = bool(_SHOWED.search(heading))
            elif not line.startswith("### "):
                in_section = False
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        # Tables and rules make poor one-line summaries; prose follows them.
        if stripped.startswith("|") or set(stripped) <= {"-", " "}:
            if paragraph:
                break
            continue
        paragraph.append(stripped.lstrip("-* ").strip() if stripped[:2] in ("- ", "* ") else stripped)
    summary = _plain(" ".join(paragraph))
    if len(summary) > SUMMARY_CHARS:
        summary = summary[: SUMMARY_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return title, summary


def _is_rate(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("value"), int | float) and "n" in value


def headline_numbers(metrics: Any, limit: int = HEADLINE_LIMIT) -> list[dict[str, Any]]:
    """The shallowest rate objects ({value, n, successes, ci95}) in metrics.json.

    Breadth-first, so a run's top-level numbers come before per-variant
    breakdowns. Every number returned carries its n.
    """
    found: list[dict[str, Any]] = []
    queue: list[tuple[list[str], Any]] = [([], metrics)]
    while queue and len(found) < limit:
        path, node = queue.pop(0)
        if _is_rate(node):
            found.append({
                "label": " · ".join(path[-2:]).replace("_", " "),
                "value": node["value"],
                "n": node.get("n"),
                "successes": node.get("successes"),
                "ci95": node.get("ci95"),
            })
            continue
        if isinstance(node, dict):
            queue.extend(([*path, str(k)], v) for k, v in node.items())
    return found


def _run_card(name: str, run: Path) -> dict[str, Any]:
    readme = run / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    title, summary = readme_outline(text)
    metrics = _read_json(run / "metrics.json") if (run / "metrics.json").is_file() else None
    date = _DATE.match(name)
    return {
        "name": name,
        "date": date.group(1) if date else None,
        "title": title,
        "summary": summary,
        "has_readme": bool(text),
        "headline": headline_numbers(metrics) if metrics is not None else [],
        "flags": {k: metrics.get(k) for k in ("sampled", "timing_valid")
                  if isinstance(metrics, dict) and k in metrics},
        "pngs": _pngs(run),
    }


@router.get("/api/results")
async def list_results() -> dict[str, Any]:
    runs = _runs()
    return {"runs": [_run_card(name, runs[name]) for name in sorted(runs, reverse=True)]}


@router.get("/api/results/{name}")
async def get_result(name: str) -> Any:
    runs = _runs()
    run = runs.get(name)
    if run is None:
        return _not_found("unknown run")
    readme = run / "README.md"
    metrics_path = run / "metrics.json"
    return {
        **_run_card(name, run),
        "readme": readme.read_text(encoding="utf-8") if readme.is_file() else "",
        "metrics": _read_json(metrics_path) if metrics_path.is_file() else None,
    }


@router.get("/api/results/{name}/files/{file}")
async def get_result_file(name: str, file: str) -> Any:
    run = _runs().get(name)
    if run is None or file not in _pngs(run):
        return _not_found()
    return FileResponse(run / file, media_type="image/png")


@router.get("/api/feedback-cycle/latest")
async def latest_feedback_cycle() -> Any:
    runs = _runs()
    # `<date>-feedback-cycle` (offline agent) or `<date>-feedback-cycle-<agent>`;
    # superseded runs are kept on disk but never shown as the latest.
    cycles = sorted((n for n in runs if "-feedback-cycle" in n and "superseded" not in n
                     and (runs[n] / "metrics.json").is_file()), reverse=True)
    for name in cycles:
        metrics = _read_json(runs[name] / "metrics.json")
        if isinstance(metrics, dict):
            return {"run": name, "has_readme": (runs[name] / "README.md").is_file(),
                    "metrics": metrics}
    return _not_found("no feedback cycle yet")


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
def _db_open(request: Request) -> bool:
    if not getattr(request.app.state, "db_ready", False):
        return False
    try:
        db.pool()
    except RuntimeError:
        return False
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str | bytes):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


@router.get("/api/gap-report/latest")
async def latest_gap_report(request: Request) -> Any:
    if not _db_open(request):
        return JSONResponse({"error": "database unavailable"}, status_code=503)
    try:
        row = await db.fetch_one(
            """
            SELECT report_id, generated_at, window_start, window_end, call_count,
                   clusters, actions
            FROM gap_reports ORDER BY generated_at DESC LIMIT 1
            """
        )
    except Exception as exc:
        log.warning("gap_report_unavailable", detail=str(exc))
        return JSONResponse({"error": "database unavailable"}, status_code=503)
    if row is None:
        return _not_found("no gap report yet")
    report = {
        "report_id": row["report_id"],
        "generated_at": _jsonable(row["generated_at"]),
        "window_start": _jsonable(row["window_start"]),
        "window_end": _jsonable(row["window_end"]),
        "call_count": row["call_count"],
    }
    for key in ("clusters", "actions"):
        value = row[key] if isinstance(row[key], list) else _jsonable(row[key])
        report[key] = value if isinstance(value, list) else []
    return report


@router.get("/api/status")
async def status(request: Request) -> dict[str, Any]:
    settings = get_settings()
    _, agent = agent_settings(settings)

    def has(field: str) -> bool:
        value = getattr(settings, field, None)
        return bool(value is not None and value.get_secret_value())

    keys = {
        "anthropic": has("anthropic_api_key"),
        "openai": has("openai_api_key"),
        "deepgram": has("deepgram_api_key"),
        "cartesia": has("cartesia_api_key"),
    }

    # The live voice pipeline (app/pipeline.py) always uses Deepgram and
    # Cartesia. With no Anthropic key it falls back to the offline extractive
    # agent (providers.offline_fallback), so only an OpenAI choice needs a key.
    missing: list[str] = []
    if not keys["deepgram"]:
        missing.append("DEEPGRAM_API_KEY")
    if not keys["cartesia"]:
        missing.append("CARTESIA_API_KEY")
    if not settings.cartesia_voice_id:
        missing.append("CARTESIA_VOICE_ID")
    if settings.llm_provider == "openai" and not keys["openai"]:
        missing.append(f"{settings.llm_provider.upper()}_API_KEY")
    audio_extra = importlib.util.find_spec("pipecat") is not None

    database = _db_open(request)
    chunks: int | None = None
    if database:
        try:
            row = await db.fetch_one(
                "SELECT count(*) AS n FROM chunks WHERE domain_config = %(d)s",
                {"d": settings.domain},
            )
            chunks = int(row["n"]) if row else 0
        except Exception as exc:
            log.warning("status_chunk_count_failed", detail=str(exc))

    return {
        "domain": settings.domain,
        "agent": agent,
        "keys": keys,
        "voice": {
            "ready": not missing and audio_extra,
            "missing": missing,
            "audio_extra": audio_extra,
        },
        "database": database,
        "chunks": chunks,
    }
