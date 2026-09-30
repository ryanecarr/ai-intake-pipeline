"""Ingest + enrich step (deterministic).

Loads the three JSON inputs, resolves the referenced policy record, and runs the
data-integrity checks whose results must surface in the report's confidence and
"items requiring human judgment" sections. No LLM calls happen here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root resolved from this file: src/pipeline/data_loader.py -> repo root.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

COMPLAINT_PATH = DATA_DIR / "complaints" / "complaint_water_damage.json"
GUIDELINES_PATH = DATA_DIR / "complaints" / "routing_guidelines.json"
POLICIES_PATH = DATA_DIR / "policies" / "policy_records.json"


@dataclass
class Flag:
    """A deterministic finding for the human-judgment / confidence sections.

    ``key`` is a stable identifier used to (a) dedupe against indeterminate escalation
    triggers that describe the same root cause and (b) drive the deterministic
    confidence cap. ``title``/``detail`` are for display.
    """

    title: str
    detail: str
    key: str = ""


@dataclass
class TriageContext:
    complaint: Dict[str, Any]
    guidelines: Dict[str, Any]
    policy: Optional[Dict[str, Any]]
    flags: List[Flag] = field(default_factory=list)

    @property
    def routing(self) -> Dict[str, Any]:
        return self.guidelines["routing_guidelines"]

    @property
    def carrier(self) -> str:
        return self.routing.get("carrier_name") or "the insurance carrier"

    @property
    def thresholds(self) -> Dict[str, Any]:
        return self.routing.get("regulatory_thresholds", {})


def _load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def _find_policy(policies: List[Dict[str, Any]], policy_number: str) -> Optional[Dict[str, Any]]:
    for record in policies:
        if record.get("policy_number") == policy_number:
            return record
    return None


def load_context(
    complaint_path: Path = COMPLAINT_PATH,
    guidelines_path: Path = GUIDELINES_PATH,
    policies_path: Path = POLICIES_PATH,
) -> TriageContext:
    """Load inputs, resolve the policy, and compute deterministic flags."""
    complaint = _load_json(complaint_path)
    guidelines = _load_json(guidelines_path)
    policies = _load_json(policies_path)

    policy = _find_policy(policies, complaint.get("policy_number", ""))
    ctx = TriageContext(complaint=complaint, guidelines=guidelines, policy=policy)
    ctx.flags = _compute_flags(ctx)
    return ctx


def _compute_flags(ctx: TriageContext) -> List[Flag]:
    """Hard data-integrity checks that must reach the human reviewer."""
    complaint, policy = ctx.complaint, ctx.policy
    flags: List[Flag] = []

    # 1. Policy resolution.
    if policy is None:
        flags.append(
            Flag(
                key="policy_not_found",
                title="Policy record not found",
                detail=(
                    f"No policy matching {complaint.get('policy_number')!r} exists in "
                    "the policy records. Coverage assessment cannot be verified."
                ),
            )
        )
    else:
        # 2. Policy-holder identity mismatch — the complainant is not the holder of
        #    record for the cited policy (e.g. a relative writing on their behalf).
        #    Nothing policy-specific should be disclosed until authority is confirmed.
        complainant = (complaint.get("customer_name") or "").strip()
        holder = (policy.get("holder_name") or "").strip()
        if complainant.lower() != holder.lower():
            flags.append(
                Flag(
                    key="policy_holder_mismatch",
                    title="Policy-holder identity mismatch",
                    detail=(
                        f"Complaint is from {complainant!r}, but policy "
                        f"{policy.get('policy_number')} is held by {holder!r}. Verify "
                        "the complainant's identity and authority on this policy before "
                        "discussing or acting on any coverage determination."
                    ),
                )
            )

    # 3. Denied claim with no amount -> the high-value-denial regulatory trigger is
    #    indeterminate (see escalate.py, which also records this on the determination).
    threshold = ctx.thresholds.get("denied_claim_amount")
    if threshold is not None and claim_amount_needed(complaint) and claim_amount(complaint) is None:
        flags.append(
            Flag(
                key="claim_amount_unknown",
                title="Claim amount unknown",
                detail=(
                    "No claim/loss amount is present in the complaint data, so the "
                    f"regulatory trigger 'denied claim above ${threshold:,}' cannot be "
                    "evaluated. Confirm the claim value to determine whether a regulatory "
                    "flag is required."
                ),
            )
        )

    return flags


def claim_amount_needed(complaint: Dict[str, Any]) -> bool:
    """True when the high-value-denial rule could apply: a claim that is (or may be) denied."""
    status = complaint.get("current_claim_status")
    has_claim = bool(complaint.get("claim_reference")) or status is not None
    return has_claim and status in (None, "denied")


def claim_amount(complaint: Dict[str, Any]) -> Optional[float]:
    """Return the claim/loss amount if present; otherwise None."""
    for key in ("claim_amount", "loss_amount", "amount_claimed", "claim_value"):
        value = complaint.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None
