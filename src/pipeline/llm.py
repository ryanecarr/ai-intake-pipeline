"""Thin Anthropic client wrapper for the triage pipeline.

Centralizes model selection and structured-output calls so the step modules stay
focused on prompts. Two model tiers are used (see the design doc):

* ``HAIKU`` — cheap, fast, deterministic structured extraction (classification,
  signal extraction). The high-volume structured steps run on the smallest capable
  model to keep cost proportional to where quality matters.
* ``OPUS`` — nuanced generation that a human will read and act on (the customer
  response draft, the risk/confidence assessment).

**Failure handling.** Transient failures (connection errors, timeouts, 429, 5xx) are
retried by the SDK with exponential backoff — we only *tune* that here
(``MAX_RETRIES``), rather than hand-rolling a retry loop. Anything that ultimately
fails — exhausted retries, or a non-retryable error like an invalid/revoked key, a
bad model ID, or an unparseable structured response — raises :class:`TriageStepError`,
which aborts the run (fail-fast). We never emit a partial or misleading report.
"""

from __future__ import annotations

import os
from typing import Type, TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

# Exact model IDs (no date suffix).
HAIKU = "claude-haiku-4-5"
OPUS = "claude-opus-4-8"

# SDK retry/timeout tuning. The SDK already retries connection errors, timeouts,
# 429, and 5xx with exponential backoff; we just raise the ceiling from the default 2
# and set an explicit per-request timeout so a hung call fails in bounded time.
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120.0  # seconds

T = TypeVar("T", bound=BaseModel)

load_dotenv()

_client = None


class TriageStepError(RuntimeError):
    """A pipeline step failed; the run is aborted (fail-fast)."""


def get_client() -> "anthropic.Anthropic":
    """Lazily build a shared client, with a friendly error if the key is missing."""
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise TriageStepError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
                "Anthropic API key."
            )
        _client = anthropic.Anthropic(
            max_retries=MAX_RETRIES,
            timeout=REQUEST_TIMEOUT,
        )
    return _client


def extract_structured(
    *,
    step: str,
    model: str,
    system: str,
    user: str,
    schema: Type[T],
    max_tokens: int,
    thinking: bool = False,
) -> T:
    """Run one structured-output call and return a validated pydantic instance.

    ``step`` is a human-readable label used in error messages so a failure points at
    the stage that failed. ``thinking`` enables adaptive thinking (Opus tier only);
    effort is left at its default (``high`` on Opus) to avoid an output_config
    conflict with structured outputs.

    Raises :class:`TriageStepError` on any unrecoverable failure.
    """
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=schema,
    )
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    try:
        response = get_client().messages.parse(**kwargs)
    except anthropic.APIError as exc:  # transient retries already exhausted, or terminal
        raise TriageStepError(_api_error_message(step, exc)) from exc
    except Exception as exc:  # e.g. the model's output failed schema validation
        raise TriageStepError(
            f"{step} failed while producing structured output: {exc}"
        ) from exc

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        stop = getattr(response, "stop_reason", "unknown")
        request_id = getattr(response, "_request_id", None)
        raise TriageStepError(
            f"{step} failed: the model returned no valid structured output "
            f"(stop_reason={stop}, request_id={request_id})."
        )
    return parsed


def _api_error_message(step: str, exc: "anthropic.APIError") -> str:
    """Translate an SDK error into an actionable, step-attributed message."""
    request_id = getattr(exc, "request_id", None)
    prefix = f"{step} failed"
    if request_id:
        prefix += f" (request_id={request_id})"

    if isinstance(exc, anthropic.AuthenticationError):
        reason = "authentication error — the API key is missing, invalid, or revoked."
    elif isinstance(exc, anthropic.PermissionDeniedError):
        reason = "permission/billing error — the key lacks access or a spend limit was hit."
    elif isinstance(exc, anthropic.NotFoundError):
        reason = "model or endpoint not found — check the model ID."
    elif isinstance(exc, anthropic.RateLimitError):
        reason = "rate limited and automatic retries were exhausted — wait and rerun."
    elif isinstance(exc, anthropic.BadRequestError):
        reason = f"invalid request — {exc}."
    elif isinstance(exc, anthropic.APIConnectionError):
        reason = "could not reach the Anthropic API after retries — check your network."
    elif isinstance(exc, anthropic.APIStatusError):
        # Remaining 5xx / overloaded after the SDK exhausted its retries.
        reason = (
            f"the Anthropic API returned a server error ({exc.status_code}) after retries "
            "— try again shortly."
        )
    else:
        reason = str(exc)

    return f"{prefix}: {reason}"
