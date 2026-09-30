"""Step 3 — draft the customer response (Opus tier).

This is the step a human will read and send, so it runs on the strongest model with
adaptive thinking. It is grounded in: the classification, the fired escalations, the
tone guidance matched to the customer's sentiment, and — importantly — the deterministic
data-integrity flags, so the draft does not confirm coverage details or make promises
that depend on an unverified identity.
"""

from __future__ import annotations

from typing import Dict, List

from . import llm
from .data_loader import TriageContext
from .escalate import Determination
from .schemas import Classification, DraftResponse

SYSTEM = (
    "You are a senior customer-resolution specialist at {carrier}, an insurance carrier, "
    "drafting a response to a customer complaint. Write a response that a {carrier} team "
    "member can send with minimal editing, signed on behalf of {carrier}. Be empathetic and "
    "specific, follow the provided tone guidance exactly, and never invent facts. If the "
    "customer's identity or coverage is unverified, do not confirm coverage specifics or make "
    "commitments that depend on them — instead commit to verifying and following up."
)

# Map a sentiment to the tone-guideline key in routing_guidelines.json.
_TONE_KEY = {
    "hostile": "hostile_or_angry",
    "angry": "hostile_or_angry",
    "upset": "upset_or_frustrated",
    "frustrated": "upset_or_frustrated",
    "concerned": "concerned_or_calm",
    "calm": "concerned_or_calm",
}

# Flags that mean the complainant's authority on the policy is unconfirmed.
_IDENTITY_FLAGS = {"policy_holder_mismatch", "policy_not_found"}


def draft_response(
    ctx: TriageContext,
    classification: Classification,
    escalations: List[Determination],
) -> DraftResponse:
    tone = _tone_guidance(ctx.routing, classification.sentiment)
    user = _build_prompt(ctx, classification, escalations, tone)
    return llm.extract_structured(
        step="Response drafting",
        model=llm.OPUS,
        system=SYSTEM.format(carrier=ctx.carrier),
        user=user,
        schema=DraftResponse,
        max_tokens=8192,
        thinking=True,
    )


def _tone_guidance(routing: Dict, sentiment: str) -> str:
    key = _TONE_KEY.get(sentiment, "concerned_or_calm")
    return routing["response_tone_guidelines"][key]


def _build_prompt(
    ctx: TriageContext,
    classification: Classification,
    escalations: List[Determination],
    tone: str,
) -> str:
    complaint = ctx.complaint
    fired = _escalation_summary(escalations)
    caveats = (
        "\n".join(f"- {f.title}: {f.detail}" for f in ctx.flags)
        or "- None."
    )
    if any(f.key in _IDENTITY_FLAGS for f in ctx.flags):
        coverage_rule = (
            "- Do NOT state a final coverage determination or confirm policy specifics — the "
            "complainant's identity/authority on the policy is not yet verified. Explain that "
            "you need to confirm authority, and commit to reviewing and following up."
        )
    else:
        coverage_rule = (
            "- Do not promise an outcome that has not been decided. Commit to a review or an "
            "answer, and to following up."
        )

    return f"""Draft a customer response for this complaint.

CUSTOMER: {complaint.get('customer_name')}  (claim {complaint.get('claim_reference')}, policy {complaint.get('policy_number')})
COMPLAINT TEXT:
\"\"\"{complaint['complaint_text']}\"\"\"

CLASSIFICATION:
- Type: {classification.complaint_type}
- Sentiment: {classification.sentiment}
- Urgency: {classification.urgency} — {ctx.routing['urgency_levels'].get(classification.urgency, '')}

ESCALATIONS THAT APPLY:
{fired}

TONE GUIDANCE (follow this exactly):
{tone}

DATA-INTEGRITY CAVEATS (do not contradict these in the response):
{caveats}

Requirements for the response:
- Open by acknowledging the customer's specific concern; do not lead with policy language.
- Commit to a concrete next step with a timeframe no longer than the urgency level's response window above. If the customer asked for a specific contact or deadline, address that request directly.
{coverage_rule}
- Keep it professional, warm, and free of jargon.

Also provide a one-line recommended handling path and the role that should own the response."""


def _escalation_summary(escalations: List[Determination]) -> str:
    lines: List[str] = []
    for det in escalations:
        if det.status == "required":
            triggers = "; ".join(f.trigger for f in det.fired)
            lines.append(f"- {det.category}: REQUIRED ({triggers})")
        elif det.status == "needs_review":
            triggers = "; ".join(f.trigger for f in det.indeterminate)
            lines.append(f"- {det.category}: NEEDS REVIEW — could not be evaluated ({triggers})")
        else:
            lines.append(f"- {det.category}: not required")
    return "\n".join(lines)
