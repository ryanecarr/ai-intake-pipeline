"""Customer complaint triage pipeline.

A hybrid prompt-chain: LLM steps classify the complaint, extract escalation
signals, draft the customer response, and assess risk/confidence; deterministic
Python applies the routing rules and renders the markdown report.

Public entry point: :func:`run_complaint_triage`.
"""

from __future__ import annotations

from pathlib import Path

from .assess import assess, build_human_judgment_items, cap_confidence
from .classify import classify
from .data_loader import load_context
from .draft import draft_response
from .escalate import determine_escalations, extract_signals
from .llm import TriageStepError
from .report import write_report

__all__ = ["run_complaint_triage", "TriageStepError"]


def run_complaint_triage(verbose: bool = True, **paths: Path) -> Path:
    """Run the full triage pipeline and write the markdown report.

    ``paths`` optionally overrides ``complaint_path`` / ``guidelines_path`` /
    ``policies_path`` (see :func:`load_context`). Returns the path to the report.
    """

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("• Loading inputs and running data-integrity checks…")
    ctx = load_context(**paths)
    for flag in ctx.flags:
        log(f"    ⚠ {flag.title}")

    log("• Classifying complaint (Haiku)…")
    classification = classify(ctx)
    log(
        f"    type={classification.complaint_type} "
        f"sentiment={classification.sentiment} urgency={classification.urgency}"
    )

    log("• Extracting escalation signals (Haiku)…")
    extraction = extract_signals(ctx)
    log("• Applying routing rules (deterministic)…")
    escalations = determine_escalations(ctx, extraction)
    for det in escalations:
        log(f"    {det.category}: {det.status.replace('_', ' ')}")

    log("• Drafting customer response (Opus)…")
    draft = draft_response(ctx, classification, escalations)

    log("• Assessing risk and confidence (Opus)…")
    assessment = assess(ctx, classification, escalations)
    cap_confidence(ctx, escalations, assessment)  # deterministic cap on hard signals
    items = build_human_judgment_items(ctx, escalations, assessment)
    log(f"    confidence={assessment.confidence_level}; {len(items)} human-judgment item(s)")

    path = write_report(ctx, classification, escalations, draft, assessment, items)
    log(f"• Report written to {path}")
    return path
