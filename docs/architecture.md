# Architecture — Complaint Triage Pipeline

A **hybrid prompt-chain**: the LLM does what it's good at (reading unstructured complaint
text — classification, evidence extraction, drafting, risk judgment), while **deterministic
Python owns every consequential decision** (applying the escalation rules, the
data-integrity checks, and assembling the report). We deliberately do not let a model
"decide" a regulatory or legal escalation when the rules are explicit and auditable.

## Flow

```mermaid
flowchart TD
    subgraph Inputs["data/ (JSON inputs)"]
        A1[complaint_*.json]
        A2[routing_guidelines.json]
        A3[policy_records.json]
    end

    A1 & A2 & A3 --> D0

    D0["**Ingest & enrich** (deterministic)\ndata_loader.py\n• resolve policy by number\n• integrity checks → flags"]

    D0 -->|TriageContext + flags| S1
    D0 -->|flag: policy-holder mismatch| FLAGS
    D0 -->|flag: claim amount unknown| FLAGS

    S1["**1. Classify** (Haiku, structured)\nclassify.py\ntype · sentiment · urgency + quotes"]
    S1 --> S2

    S2["**2a. Extract signals** (Haiku, structured)\nescalate.py\npresent signals + evidence quotes"]
    S2 --> R{"**2b. Apply routing rules**\n(deterministic)\nsignals + structured data\n+ keyword backstop"}

    R -->|supervisor / legal / regulatory\ndeterminations| S3
    R -->|"indeterminate triggers\n(e.g. missing claim amount)" | FLAGS

    S3["**3. Draft response** (Opus + adaptive thinking)\ndraft.py\ntone-matched letter + path/owner\n(constrained by integrity caveats)"]
    S3 --> S4

    S4["**4. Assess** (Opus + adaptive thinking)\nassess.py\nrisk · confidence + reasons · soft flags\n→ deterministic confidence cap"]
    S4 -->|soft human-judgment flags| FLAGS

    FLAGS["**Merge** human-judgment items\n(deterministic + LLM, de-duped)"]

    S4 --> OUT
    FLAGS --> OUT
    S3 --> OUT
    R --> OUT
    S1 --> OUT

    OUT["**5. Render report** (deterministic)\nreport.py"]
    OUT --> FILE[("output/{complaint_id}_triage.md")]
```

## Why this shape

| Concern | Decision |
| --- | --- |
| Escalation must be **auditable** | Rules applied in Python using trigger wording and thresholds from `routing_guidelines.json`; the LLM only supplies evidence (signals + quotes). |
| **Cost** | Fast, inexpensive Haiku for the structured extraction steps; Opus only for the two steps a human reads (draft + risk). |
| **Edge cases** must surface | Deterministic data-integrity checks (policy-holder mismatch, missing claim amount) are guaranteed to appear in the confidence + human-judgment sections, not dependent on the model noticing. |
| **Indeterminate ≠ false** | Triggers that can't be evaluated (e.g. no claim amount on a denied claim) are reported as *indeterminate* and routed to human judgment, never silently dropped. |
| **Missed signals** | A keyword backstop forces *needs review* when high-stakes keywords appear but the LLM didn't flag the covering signal. |
| **Don't over-promise** | The draft step receives the integrity caveats so it won't confirm coverage for an unverified identity. |

## Models

| Step | Model | Why |
| --- | --- | --- |
| Classify, extract signals | `claude-haiku-4-5` | Constrained structured output; fast and inexpensive at the high-frequency steps. |
| Draft response, assess risk/confidence | `claude-opus-4-8` (adaptive thinking) | Nuanced, human-facing generation and judgment where quality matters most. |
