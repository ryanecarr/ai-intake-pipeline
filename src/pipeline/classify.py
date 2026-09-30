"""Step 1 — classify the complaint (type / sentiment / urgency).

Runs on the Haiku tier with structured output. The schema enums already constrain
the model to the controlled vocabulary; the prompt supplies the guideline
definitions so the urgency choice is grounded in the carrier's own SLA language.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from . import llm
from .data_loader import TriageContext
from .schemas import Classification

SYSTEM = (
    "You are a customer-complaint triage classifier for {carrier}, an insurance carrier. "
    "Classify the complaint accurately and conservatively using only the carrier's controlled "
    "vocabularies. Base every decision on what the complaint actually says; quote the text "
    "as evidence. Do not infer escalation actions here — only classify type, sentiment, and urgency."
)


def classify(ctx: TriageContext) -> Classification:
    routing = ctx.routing
    user = _build_prompt(ctx.complaint, routing)
    return llm.extract_structured(
        step="Classification",
        model=llm.HAIKU,
        system=SYSTEM.format(carrier=ctx.carrier),
        user=user,
        schema=Classification,
        max_tokens=2048,
    )


def _build_prompt(complaint: Dict[str, Any], routing: Dict[str, Any]) -> str:
    urgency_defs = json.dumps(routing["urgency_levels"], indent=2)
    categories = ", ".join(routing["complaint_categories"])
    sentiments = ", ".join(routing["sentiment_categories"])

    return f"""Classify the following customer complaint.

COMPLAINT TEXT:
\"\"\"{complaint['complaint_text']}\"\"\"

CONTEXT:
- Customer tenure: {complaint.get('customer_tenure_years')} years
- Prior complaints: {complaint.get('prior_complaints')}
- Current claim status: {complaint.get('current_claim_status')}
- Denial reason on file: {complaint.get('denial_reason_on_file')}

ALLOWED COMPLAINT CATEGORIES: {categories}
ALLOWED SENTIMENTS (most to least severe): {sentiments}

URGENCY LEVEL DEFINITIONS:
{urgency_defs}

Choose the single best complaint_type, the sentiment that best matches the customer's
tone, and the urgency level justified by the definitions above. Provide a brief rationale
and short verbatim quotes that support your choices."""
