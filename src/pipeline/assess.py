"""Step 4 — risk + confidence assessment, and the human-judgment merge (Opus tier).

The model produces the risk narrative, a qualitative confidence level with reasons,
and any *additional* soft concerns it noticed. Deterministic code then merges those
with the hard flags (data-integrity checks + indeterminate escalation triggers) into a
single, de-duplicated "items requiring human judgment" list for the report.
"""

from __future__ import annotations

from typing import List

from . import llm
from .data_loader import TriageContext
from .escalate import Determination
from .schemas import Assessment, Classification

SYSTEM = (
    "You are a risk reviewer for {carrier}'s complaint-triage process. Assess the downside "
    "if this complaint is mishandled, and judge how confident the triage output is. Be "
    "honest about uncertainty: unverified identity or missing data should lower confidence. "
    "The known data-integrity flags are listed for you — factor them into confidence, and "
    "only add NEW concerns a human should weigh that are not already listed."
)


# Confidence ordering for the deterministic cap (higher = more confident).
_CONFIDENCE_RANK = {"Low": 1, "Medium": 2, "High": 3}


def assess(
    ctx: TriageContext,
    classification: Classification,
    escalations: List[Determination],
) -> Assessment:
    user = _build_prompt(ctx, classification, escalations)
    return llm.extract_structured(
        step="Risk/confidence assessment",
        model=llm.OPUS,
        system=SYSTEM.format(carrier=ctx.carrier),
        user=user,
        schema=Assessment,
        max_tokens=4096,
        thinking=True,
    )


def cap_confidence(
    ctx: TriageContext,
    escalations: List[Determination],
    assessment: Assessment,
) -> None:
    """Deterministically cap the model's confidence based on hard signals.

    The model decides confidence, but it cannot rate the triage above what the
    structured facts allow. Unresolved identity (or a missing policy) caps confidence at
    Low — coverage conclusions are unreliable until identity is confirmed. Otherwise, any
    escalation trigger that could not be evaluated caps confidence at Medium. Mutates the
    assessment in place; only lowers, never raises.
    """
    cap = None
    reason = None

    if any(f.key in ("policy_holder_mismatch", "policy_not_found") for f in ctx.flags):
        cap, reason = (
            "Low",
            "Confidence capped at Low: the policy-holder identity is unresolved, so any "
            "coverage-based conclusion is unreliable until identity is verified.",
        )
    elif any(det.indeterminate for det in escalations):
        cap, reason = (
            "Medium",
            "Confidence capped at Medium: one or more escalation triggers could not be "
            "evaluated and require human review.",
        )

    if cap and _CONFIDENCE_RANK[assessment.confidence_level] > _CONFIDENCE_RANK[cap]:
        assessment.confidence_level = cap
        assessment.confidence_reasons = [reason] + list(assessment.confidence_reasons)


def build_human_judgment_items(
    ctx: TriageContext,
    escalations: List[Determination],
    assessment: Assessment,
) -> List[str]:
    """Merge deterministic flags, indeterminate triggers, and LLM-noticed concerns.

    De-dupes across all three sources: indeterminate triggers that share a ``key`` with a
    deterministic flag (e.g. the missing claim amount) are dropped, and the model's softer
    concerns are skipped when they restate an item already listed.
    """
    items: List[str] = []
    seen_keys = set()

    # 1. Hard data-integrity flags.
    for flag in ctx.flags:
        items.append(f"{flag.title}: {flag.detail}")
        if flag.key:
            seen_keys.add(flag.key)

    # 2. Escalation triggers that could not be evaluated (skip ones a flag already covers).
    for det in escalations:
        for trig in det.indeterminate:
            if trig.key and trig.key in seen_keys:
                continue
            items.append(
                f"{det.category} — '{trig.trigger}' could not be evaluated: {trig.evidence}"
            )
            if trig.key:
                seen_keys.add(trig.key)

    # 3. Softer concerns the model surfaced (skip near-duplicates of the above).
    for extra in assessment.additional_human_judgment_items:
        if not _is_duplicate(extra, items):
            items.append(extra)

    return items


def _is_duplicate(candidate: str, existing: List[str]) -> bool:
    c = candidate.strip().lower()
    return any(c and (c in e.lower() or e.lower() in c) for e in existing)


def _build_prompt(
    ctx: TriageContext,
    classification: Classification,
    escalations: List[Determination],
) -> str:
    complaint = ctx.complaint
    fired = []
    for det in escalations:
        label = det.status.replace("_", " ").upper()
        triggers = "; ".join(f.trigger for f in det.fired) or "—"
        fired.append(f"- {det.category}: {label} ({triggers})")
    fired_text = "\n".join(fired)

    flags = "\n".join(f"- {f.title}: {f.detail}" for f in ctx.flags) or "- None."

    return f"""Assess the triage of this complaint.

COMPLAINT (type={classification.complaint_type}, sentiment={classification.sentiment}, urgency={classification.urgency}):
\"\"\"{complaint['complaint_text']}\"\"\"

CONTEXT:
- Customer tenure: {complaint.get('customer_tenure_years')} years; prior complaints: {complaint.get('prior_complaints')}
- Claim status: {complaint.get('current_claim_status')}; denial reason on file: {complaint.get('denial_reason_on_file')}

ESCALATION DETERMINATIONS:
{fired_text}

KNOWN DATA-INTEGRITY FLAGS (already captured — factor into confidence, do not repeat):
{flags}

Provide:
1. risk_assessment — concrete consequences if this is mishandled (regulatory, legal, reputational, customer-retention).
2. confidence_level — High / Medium / Low in the overall triage output.
3. confidence_reasons — short bullet reasons for that level, referencing the flags where relevant.
4. additional_human_judgment_items — only NEW concerns not already in the known flags above."""
