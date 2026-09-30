"""Pydantic schemas for each structured LLM step in the triage pipeline.

These models double as the JSON schemas the Anthropic structured-outputs feature
constrains the model to. Enums mirror the controlled vocabularies in
``data/complaints/routing_guidelines.json`` so the model can only emit valid values.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

# --- Controlled vocabularies (kept in sync with routing_guidelines.json) ------

ComplaintType = Literal[
    "claim_denial",
    "claim_delay",
    "coverage_question",
    "adjuster_conduct",
    "billing_or_premium",
    "cancellation_or_nonrenewal",
    "service_experience",
]

Sentiment = Literal[
    "hostile",
    "angry",
    "upset",
    "frustrated",
    "concerned",
    "calm",
]

Urgency = Literal["critical", "high", "standard", "low"]

# Atomic signals the classifier extracts from the complaint *text*. Each name maps
# to one escalation trigger in routing_guidelines.json. Deterministic Python (not the
# model) decides escalations from these signals — see escalate.py.
SignalName = Literal[
    "requests_supervisor",              # explicitly asks for supervisor/manager
    "threatens_legal_or_attorney",      # threatens legal action / attorney involvement
    "threatens_regulatory_complaint",   # threatens regulator / CFPB / BBB complaint
    "intent_to_cancel",                 # intends to cancel / has cancelled
    "eo_exposure",                      # potential errors & omissions exposure
    "media_or_social_threat",           # references media / social media
    "consulting_or_retained_attorney",  # is consulting / has retained an attorney
    "references_litigation",            # references a lawsuit / litigation
    "bodily_injury_allegation",         # alleges bodily injury
    "discrimination_allegation",        # alleges discriminatory treatment
    "intent_to_file_regulator",         # states intent to file with the insurance regulator
    "vulnerable_customer",              # elderly / disability / hospitalized / bereaved
]

ConfidenceLevel = Literal["High", "Medium", "Low"]


# --- Step 1: classification ---------------------------------------------------


class Classification(BaseModel):
    """Type, sentiment, and urgency of the complaint, with supporting evidence."""

    complaint_type: ComplaintType = Field(
        description="Primary complaint category from the allowed list."
    )
    sentiment: Sentiment = Field(description="Customer's emotional state.")
    urgency: Urgency = Field(description="Response-urgency level.")
    rationale: str = Field(
        description="One or two sentences explaining the classification choices."
    )
    evidence_quotes: List[str] = Field(
        description="Short verbatim quotes from the complaint supporting the classification."
    )


# --- Step 2: escalation signal extraction (LLM half of the hybrid) ------------


class DetectedSignal(BaseModel):
    name: SignalName = Field(description="The signal that is present in the complaint.")
    quote: str = Field(
        description="Verbatim text from the complaint that evidences this signal."
    )


class SignalExtraction(BaseModel):
    """Only the signals that ARE present, each with a supporting quote.

    Deterministic code maps these to escalation determinations; the model does not
    decide escalations itself.
    """

    signals: List[DetectedSignal] = Field(
        description="Every escalation-relevant signal present in the complaint text."
    )


# --- Step 3: response drafting (Opus) -----------------------------------------


class DraftResponse(BaseModel):
    recommended_path: str = Field(
        description="The recommended response/handling path, one or two sentences."
    )
    recommended_owner: str = Field(
        description="Role that should own the response (e.g. 'Claims Supervisor')."
    )
    customer_response: str = Field(
        description="The full drafted response to send to the customer."
    )


# --- Step 4: risk + confidence assessment (Opus) ------------------------------


class Assessment(BaseModel):
    risk_assessment: str = Field(
        description="What happens if this complaint is mishandled — concrete consequences."
    )
    confidence_level: ConfidenceLevel = Field(
        description="Overall confidence in the triage output."
    )
    confidence_reasons: List[str] = Field(
        description="Short reasons justifying the confidence level."
    )
    additional_human_judgment_items: List[str] = Field(
        description="Softer concerns the model noticed that a human should weigh before acting."
    )
