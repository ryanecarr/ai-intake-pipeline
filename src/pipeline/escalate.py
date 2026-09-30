"""Step 2 — escalation determination (the hybrid core).

The LLM (Haiku) only *extracts evidence*: which escalation-relevant signals are
present in the complaint text, each with a quote. Deterministic Python then applies
the routing rules from ``routing_guidelines.json`` to decide the three escalations.
Keeping the decision in auditable code (rather than letting a model "decide" a
regulatory filing) is the central design choice of the pipeline.

Trigger wording and thresholds come from the guidelines file; this module maps each
trigger id to the signals (or structured data) that satisfy it. Removing a trigger
from the guidelines disables it.

Two regulatory triggers depend on structured data rather than text:
* high-value denial — needs a claim amount; a denied claim with none -> indeterminate.
* repeat complaint — evaluated from ``prior_complaint_dates`` when present; prior
  complaints without dates -> indeterminate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from . import llm
from .data_loader import TriageContext, claim_amount, claim_amount_needed
from .schemas import SignalExtraction

SYSTEM = (
    "You extract escalation-relevant signals from an insurance customer complaint. "
    "Report ONLY signals that are genuinely present, each with a short verbatim quote as "
    "evidence. Do not decide what should be escalated — that is handled separately. Be "
    "precise: do not report a signal unless the text clearly supports it."
)

# Human-readable descriptions of each signal, sent to the model so it knows what to look for.
_SIGNAL_GUIDE = {
    "requests_supervisor": "Customer asks for a supervisor, manager, or someone more senior than their current contact.",
    "threatens_legal_or_attorney": "Customer threatens legal action or attorney involvement (including conditional threats).",
    "threatens_regulatory_complaint": "Customer threatens a complaint to a regulator or consumer agency (state insurance department/commissioner, CFPB, BBB).",
    "intent_to_cancel": "Customer intends to cancel or switch carriers, or has already cancelled.",
    "eo_exposure": "Facts suggest potential errors & omissions (E&O) exposure, e.g. the customer says coverage was never explained or was misrepresented.",
    "media_or_social_threat": "Customer threatens to go to the media, post on social media, or post reviews.",
    "consulting_or_retained_attorney": "Customer says they are already consulting, or have retained, an attorney.",
    "references_litigation": "Customer references a lawsuit, litigation, or suing.",
    "bodily_injury_allegation": "Customer alleges bodily injury.",
    "discrimination_allegation": "Customer alleges discriminatory treatment.",
    "intent_to_file_regulator": "Customer states they will file, or have filed, a complaint with the state insurance regulator/commissioner.",
    "vulnerable_customer": "The customer or insured is described as elderly, disabled, hospitalized, or recently bereaved.",
}

# Guideline trigger id -> signals that satisfy it, per escalation category.
_SIGNAL_RULES = {
    "supervisor_escalation": [
        ("requested_supervisor", ["requests_supervisor"]),
        ("legal_threat", ["threatens_legal_or_attorney", "consulting_or_retained_attorney", "references_litigation"]),
        ("regulator_threat", ["threatens_regulatory_complaint", "intent_to_file_regulator"]),
        ("cancellation", ["intent_to_cancel"]),
        ("eo_exposure", ["eo_exposure"]),
        ("public_threat", ["media_or_social_threat"]),
    ],
    "legal_review": [
        ("attorney_involved", ["consulting_or_retained_attorney"]),
        ("litigation", ["references_litigation"]),
        ("bodily_injury", ["bodily_injury_allegation"]),
        ("discrimination", ["discrimination_allegation"]),
    ],
    "regulatory_flag": [
        ("regulator_complaint", ["intent_to_file_regulator"]),
        ("vulnerable_customer", ["vulnerable_customer"]),
    ],
}

_CATEGORY_LABELS = {
    "supervisor_escalation": "Supervisor escalation",
    "legal_review": "Legal review",
    "regulatory_flag": "Regulatory flag",
}

# Deterministic keyword backstop for the HIGHEST-stakes triggers. Signal extraction is
# the one place the deterministic guarantee can fail silently: if the LLM overlooks a
# signal, the rule never fires and the category would read "not required". For these
# triggers we also scan the raw text; if a keyword is present but the covering signal
# was NOT flagged, we force the affected category to needs_review (never auto-escalate —
# a keyword can be a false positive). Each rule lists the signals that would normally
# cover it, the keyword patterns, and the (category, trigger id) targets it protects.
_BACKSTOP_RULES = [
    {
        "label": "supervisor/manager request",
        "patterns": [r"\bsupervisor\b", r"\bmanager\b"],
        "covered_by": {"requests_supervisor"},
        "targets": [("supervisor_escalation", "requested_supervisor")],
    },
    {
        "label": "attorney / legal action",
        "patterns": [r"\battorney\b", r"\blawyer\b", r"\blawsuit\b", r"\blitigat", r"\bsue\b", r"legal action"],
        "covered_by": {
            "threatens_legal_or_attorney",
            "consulting_or_retained_attorney",
            "references_litigation",
        },
        "targets": [
            ("supervisor_escalation", "legal_threat"),
            ("legal_review", "attorney_involved"),
        ],
    },
    {
        "label": "regulator complaint",
        "patterns": [
            r"department of insurance",
            r"insurance department",
            r"insurance commissioner",
            r"\bDOI\b",
            r"\bCFPB\b",
            r"\bBBB\b",
            r"\bregulator",
        ],
        "covered_by": {"intent_to_file_regulator", "threatens_regulatory_complaint"},
        "targets": [
            ("supervisor_escalation", "regulator_threat"),
            ("regulatory_flag", "regulator_complaint"),
        ],
    },
    {
        "label": "cancellation",
        "patterns": [r"\bcancel"],
        "covered_by": {"intent_to_cancel"},
        "targets": [("supervisor_escalation", "cancellation")],
    },
]


@dataclass
class FiredTrigger:
    trigger: str
    evidence: str
    key: str = ""  # stable id for de-duping against deterministic flags


@dataclass
class Determination:
    category: str
    fired: List[FiredTrigger] = field(default_factory=list)
    indeterminate: List[FiredTrigger] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Three-state outcome derived from the triggers.

        A determination with only indeterminate triggers is ``needs_review`` — it is
        neither confirmed required nor safely "not required".
        """
        if self.fired:
            return "required"
        if self.indeterminate:
            return "needs_review"
        return "not_required"

    @property
    def required(self) -> bool:
        return self.status == "required"


def extract_signals(ctx: TriageContext) -> SignalExtraction:
    guide = "\n".join(f"- {name}: {desc}" for name, desc in _SIGNAL_GUIDE.items())
    user = (
        "Identify which of the following signals are present in the complaint. Return only "
        "the ones that ARE present, each with a supporting verbatim quote.\n\n"
        f"SIGNALS:\n{guide}\n\n"
        f'COMPLAINT TEXT:\n"""{ctx.complaint["complaint_text"]}"""'
    )
    return llm.extract_structured(
        step="Escalation signal extraction",
        model=llm.HAIKU,
        system=SYSTEM,
        user=user,
        schema=SignalExtraction,
        max_tokens=2048,
    )


def determine_escalations(
    ctx: TriageContext, extraction: SignalExtraction
) -> List[Determination]:
    """Apply routing rules to extracted signals + structured data."""
    present: Dict[str, str] = {s.name: s.quote for s in extraction.signals}
    triggers = _trigger_text(ctx)

    by_key: Dict[str, Determination] = {}
    for category, rules in _SIGNAL_RULES.items():
        det = Determination(category=_CATEGORY_LABELS[category])
        for trigger_id, signal_names in rules:
            text = triggers[category].get(trigger_id)
            if text is None:
                continue  # trigger not in the guidelines -> disabled
            evidence = _first_present(present, signal_names)
            if evidence is not None:
                det.fired.append(FiredTrigger(trigger=text, evidence=evidence))
        by_key[category] = det

    _add_regulatory_data_rules(by_key["regulatory_flag"], ctx, triggers["regulatory_flag"])
    _apply_keyword_backstop(
        by_key, triggers, ctx.complaint.get("complaint_text", ""), set(present.keys())
    )
    return list(by_key.values())


def _trigger_text(ctx: TriageContext) -> Dict[str, Dict[str, str]]:
    """Trigger descriptions from the guidelines, with thresholds filled in."""
    raw = ctx.routing.get("escalation_triggers", {})
    thresholds = {k: f"{v:,}" if isinstance(v, int) else v for k, v in ctx.thresholds.items()}
    return {
        category: {tid: text.format(**thresholds) for tid, text in raw.get(category, {}).items()}
        for category in _SIGNAL_RULES
    }


def _apply_keyword_backstop(
    by_key: Dict[str, Determination],
    triggers: Dict[str, Dict[str, str]],
    complaint_text: str,
    present_signals: set,
) -> None:
    """Catch high-stakes signals the LLM may have missed (see _BACKSTOP_RULES).

    If a covering signal was NOT flagged but its keyword appears in the text, add an
    indeterminate trigger so the category becomes needs_review and the item reaches the
    human-judgment list — rather than silently reading "not required". Never fires an
    escalation outright; a keyword match only forces human review.
    """
    text = complaint_text or ""

    for rule in _BACKSTOP_RULES:
        if rule["covered_by"] & present_signals:
            continue  # the LLM already flagged a covering signal — nothing missed
        if not any(re.search(p, text, re.IGNORECASE) for p in rule["patterns"]):
            continue  # keyword absent — rule does not apply
        for category, trigger_id in rule["targets"]:
            det = by_key.get(category)
            trigger_text = triggers.get(category, {}).get(trigger_id)
            if det is None or trigger_text is None:
                continue
            key = f"backstop:{rule['label']}"
            if any(t.key == key for t in det.indeterminate):
                continue  # idempotent
            det.indeterminate.append(
                FiredTrigger(
                    trigger=f"Possible missed signal: {trigger_text}",
                    evidence=(
                        f"The complaint text matches the '{rule['label']}' keyword check, "
                        "but signal extraction did not flag it — verify manually."
                    ),
                    key=key,
                )
            )


def _first_present(present: Dict[str, str], names: List[str]) -> Optional[str]:
    for name in names:
        if name in present:
            return present[name] or f"(signal: {name})"
    return None


def _add_regulatory_data_rules(
    det: Determination, ctx: TriageContext, triggers: Dict[str, str]
) -> None:
    """Structured-data regulatory triggers (high-value denial, repeat complaints)."""
    complaint = ctx.complaint

    # High-value denial requires BOTH a denial and an amount above the threshold. A
    # non-denied claim (or no claim at all) does not apply; a denied claim with no
    # amount is indeterminate rather than firing or silently passing.
    trigger = triggers.get("high_value_denial")
    threshold = ctx.thresholds.get("denied_claim_amount")
    if trigger and threshold is not None and claim_amount_needed(complaint):
        status = complaint.get("current_claim_status")
        amount = claim_amount(complaint)
        if amount is None:
            evidence = (
                "Claim status and amount are unavailable — cannot evaluate this threshold."
                if status is None
                else "No claim amount available in the data — cannot evaluate this threshold."
            )
            det.indeterminate.append(
                FiredTrigger(trigger=trigger, evidence=evidence, key="claim_amount_unknown")
            )
        elif amount > threshold:
            if status == "denied":
                det.fired.append(
                    FiredTrigger(
                        trigger=trigger,
                        evidence=f"Denied claim with amount ${amount:,.0f} on file.",
                    )
                )
            else:  # amount is over the line but the claim status is unknown
                det.indeterminate.append(
                    FiredTrigger(
                        trigger=trigger,
                        evidence=f"Claim amount ${amount:,.0f} is above the threshold, but claim status is unknown.",
                        key="claim_status_unknown",
                    )
                )
        # else: amount at or below the threshold -> trigger does not fire.

    # Repeat complaint within the window: evaluated from dates when we have them.
    trigger = triggers.get("repeat_complaint")
    window = ctx.thresholds.get("repeat_complaint_window_days")
    prior = complaint.get("prior_complaints")
    if not trigger or window is None or not (isinstance(prior, int) and prior > 0):
        return

    days = _days_since_prior_complaints(complaint)
    if days is None:
        det.indeterminate.append(
            FiredTrigger(
                trigger=trigger,
                evidence=(
                    f"{prior} prior complaint(s) on record, but no dates available to confirm "
                    f"the {window}-day window."
                ),
                key="repeat_complaint_window",
            )
        )
    elif min(days) <= window:
        det.fired.append(
            FiredTrigger(
                trigger=trigger,
                evidence=f"Prior complaint received {min(days)} days before this one.",
            )
        )


def _days_since_prior_complaints(complaint: Dict[str, Any]) -> Optional[List[int]]:
    """Days between each prior complaint and this one; None if dates are unavailable."""
    try:
        received = date.fromisoformat(complaint["received_date"])
        priors = [date.fromisoformat(d) for d in complaint.get("prior_complaint_dates") or []]
    except (KeyError, TypeError, ValueError):
        return None
    if not priors:
        return None
    return [(received - p).days for p in priors]
