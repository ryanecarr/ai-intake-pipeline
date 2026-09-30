# Design Document — Customer Complaint Triage Pipeline

**Author:** Ryan Carr · **Entry point:** `python src/run.py [--complaint <file>]`

This pipeline triages an incoming insurance customer complaint and produces a markdown
report a complaints-handling team member can act on without re-reading the original
complaint. It ingests the complaint, classifies it, applies the carrier's routing and
escalation rules, drafts a customer response, assesses risk and confidence, and writes
`output/<complaint_id>_triage.md`. The full flow diagram is in
[`docs/architecture.md`](architecture.md).

All sample data (the carrier "Harborline Mutual", customers, policies) is fictional.

---

## 1. Tech stack

- **Language:** Python 3.9+.
- **LLM:** Anthropic Python SDK (`anthropic`) via the Messages API, using **structured
  outputs** (`messages.parse` with Pydantic schemas) so every LLM step returns validated,
  typed data instead of free text to parse.
- **Validation/modeling:** Pydantic v2 — the schemas double as the JSON schemas that
  constrain the model and as the runtime contracts between pipeline steps.
- **Config:** `python-dotenv` for the API key; routing rules, trigger wording, and
  thresholds live in `data/complaints/routing_guidelines.json`.
- **No external services** — no database, vector store, or third-party APIs. Inputs are
  three JSON files; the only network call is to the Anthropic API.

Module layout (`src/pipeline/`): `data_loader` (ingest + integrity checks) → `classify`
→ `escalate` (LLM signals + deterministic rules) → `draft` → `assess` → `report`, wired by
`run_complaint_triage()` in `__init__.py`.

## 2. Models used and why

| Step | Model | Rationale |
| --- | --- | --- |
| Classification, escalation-signal extraction | **`claude-haiku-4-5`** | Constrained, structured, high-frequency extraction. The cheapest tier that does it reliably. |
| Response drafting, risk/confidence assessment | **`claude-opus-4-8`** (adaptive thinking) | These two outputs are read and acted on by a human and require nuance and judgment (tone-matched empathy, careful non-commitment on unverified facts, multi-front risk reasoning). |

Tiering keeps cost proportional to where quality actually matters. Sampling parameters and
`budget_tokens` are intentionally not set; thinking is adaptive.

## 3. Agentic pattern and reasoning

**Chosen: a hybrid, code-orchestrated prompt-chain.** Each stage is a discrete step with a
typed contract; control flow lives in Python. The defining choice is the **boundary
between the LLM and deterministic code**:

- The **LLM reads unstructured text** — classifies, extracts escalation *signals* with
  supporting quotes, drafts the response, and judges risk.
- **Deterministic Python makes every consequential decision** — it applies the escalation
  rules from `routing_guidelines.json`, runs the data-integrity checks, and assembles the
  report.

The model never "decides" whether to escalate to legal or raise a regulatory flag. It
supplies evidence; auditable code maps that evidence to the carrier's published triggers.
For a regulated insurer, an escalation decision a compliance reviewer can read line-by-line
is worth far more than one buried in a prompt. Trigger wording and thresholds are read from
the guidelines file, so a rule change (e.g. a new high-value-denial threshold) is a config
edit, not a code change.

**Alternatives considered and rejected:**

- *Single mega-prompt* — fastest to build, but opaque, untestable, and impossible to audit.
- *Pure-LLM chain* — cleaner than one prompt, but the escalation logic would live in a
  prompt rather than in code, losing auditability and determinism.
- *Autonomous tool-use agent* — the flow here is fixed and known; an agent that chooses its
  own trajectory adds cost and nondeterminism for no benefit.

**Reliability touches:**

- Every LLM step uses **structured outputs** — no brittle parsing.
- Triggers that cannot be evaluated are reported as **indeterminate** and routed to human
  judgment rather than silently treated as false. Each escalation has a third
  **needs-review** state, so a case with only unevaluable triggers is never rendered a clean
  "not required".
- The model proposes a confidence level, but a **deterministic cap** lowers it when hard
  signals demand it (unresolved policy-holder identity → at most Low; any unevaluable
  trigger → at most Medium), so confidence can never out-run the facts.
- Signal extraction is backstopped by a **deterministic keyword scan** on the
  highest-stakes triggers (supervisor, attorney/legal, regulator, cancellation). If a
  keyword is in the text but the LLM didn't flag the signal, the category is forced to
  *needs_review* rather than silently reading "not required". It never auto-escalates,
  since a keyword can be a false positive.
- The pipeline is **fail-fast**. The SDK already retries transient failures (connection
  errors, timeouts, `429`, `5xx`) with exponential backoff — we tune that
  (`max_retries=4`, explicit timeout) rather than hand-rolling a loop. Anything that still
  fails raises a clear, step-attributed error and aborts the run. For a compliance
  artifact, no report is safer than a partial or misleading one.

### Sample inputs

Two sample complaints exercise different paths:

- **`complaint_water_damage.json`** (default) — a denied $18,750 homeowners water-damage
  claim, written by the policyholder's son on behalf of his 81-year-old father. It is
  designed to exercise: the **policy-holder identity mismatch** check (complainant ≠ holder
  of record, so the draft must not discuss coverage specifics); both **structured-data
  regulatory rules** (denied claim above the threshold, and a prior complaint inside the
  repeat window); a **vulnerable-customer** signal; and a *conditional* legal threat ("if I
  have to get a lawyer involved"), which should escalate to a supervisor without
  triggering legal review.
- **`complaint_premium_question.json`** — a calm question about a renewal premium
  increase, with no claim. It checks that the pipeline does **not** over-escalate: no
  triggers should fire and no claim-amount flag should appear for a non-claim complaint.

## 4. Assumptions

- **Identity is surfaced, not resolved.** A complainant who is not the holder of record is
  flagged rather than assumed to be authorized; coverage facts are treated as unverified
  until a human confirms the complainant's authority on the policy.
- **The high-value-denial regulatory trigger requires both a denial and the amount.** A
  non-denied claim, or a complaint with no claim, does not apply; a denied claim with no
  amount on file is *indeterminate* (not false) and flagged for confirmation.
- **The repeat-complaint trigger is evaluated from `prior_complaint_dates`** against the
  complaint's `received_date`. If prior complaints exist without dates, it is
  indeterminate.
- **A threatened attorney is not a retained attorney.** "Threatens legal action" fires the
  supervisor trigger; legal review requires an attorney who is actually retained or
  consulted, or litigation, per the guideline wording.
- **Tone** is selected deterministically from sentiment via the
  `response_tone_guidelines` map (e.g. angry → `hostile_or_angry`).
- **One complaint per run**; output filename derives from `complaint_id`.

## 5. What I would do next

- **Tests + eval harness.** Unit-test the deterministic rule mapper (the auditable core)
  and add a golden-file test for the renderer. Build a small labeled set to measure
  classification/extraction accuracy and to safely tune prompts.
- **Verification pass.** The keyword backstop already guards the highest-stakes triggers
  against a missed signal; the next step is self-consistency (run extraction more than once
  and lower confidence on disagreement) and/or an LLM-as-judge check on classification and
  signal extraction before they drive escalations.
- **Resolve identity instead of only flagging.** Look up authorized parties on the policy
  (e.g. a power of attorney or named contact) to distinguish a family member acting on the
  holder's behalf from a wrong-policy or fraud concern.
- **Read the attachments.** Complaints reference attachments (denial letters, discharge
  summaries, photos); document understanding could extract claim amounts and dates to
  resolve indeterminate triggers automatically.
- **Partly rule-drive urgency.** When supervisor, legal, and regulatory escalations all
  fire, a deterministic override to `critical` may be warranted.
- **Scale + cost.** Batch many complaints (Messages Batches API at 50% cost) and apply
  prompt caching to the shared guidelines prefix.
- **Operational hardening.** Structured audit logging of every determination and its
  evidence, and per-step token/cost telemetry.
