# AI Intake Pipeline — Insurance Complaint Triage

An AI pipeline that triages insurance customer complaints. Given a complaint, the carrier's
routing rules, and policy records, it classifies the complaint, decides supervisor / legal /
regulatory escalations, drafts a customer response, assesses risk and confidence, and writes
a markdown triage report a team member can act on immediately.

The core design choice: **the LLM reads text, deterministic code makes decisions.** Claude
classifies the complaint and extracts escalation *signals* with verbatim evidence; auditable
Python maps those signals to the carrier's escalation rules. Anything that can't be
evaluated is surfaced for human review instead of being silently treated as "no".

> All sample data — the carrier "Harborline Mutual", customers, and policies — is fictional.

## Pipeline

1. **Ingest & enrich** (deterministic) — load inputs, resolve the policy, run data-integrity
   checks (e.g. complainant is not the policyholder, missing claim amount).
2. **Classify** (Claude Haiku, structured output) — complaint type, sentiment, urgency, with
   quoted evidence.
3. **Escalate** — Haiku extracts signals; Python applies the routing rules, structured-data
   thresholds, and a keyword backstop for missed high-stakes signals. Each escalation is
   *required*, *needs review*, or *not required*.
4. **Draft response** (Claude Opus) — tone-matched to sentiment, constrained so it never
   confirms coverage for an unverified identity.
5. **Assess** (Claude Opus) — risk if mishandled, confidence level (deterministically
   capped by hard data problems), and items needing human judgment.
6. **Render** (deterministic) — `output/<complaint_id>_triage.md`.

See [`docs/architecture.md`](docs/architecture.md) for the flow diagram and
[`docs/DESIGN.md`](docs/DESIGN.md) for model choices, trade-offs, and assumptions.

## Running it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add your ANTHROPIC_API_KEY
```

```bash
python src/run.py                                                          # default sample
python src/run.py --complaint data/complaints/complaint_premium_question.json
```

Options: `--complaint`, `--guidelines`, `--policies` point at alternative input files.

## Repository layout

```
data/
  complaints/
    complaint_water_damage.json      high-stakes sample: denied claim, identity mismatch
    complaint_premium_question.json  low-stakes sample: should not escalate
    routing_guidelines.json          categories, SLAs, escalation triggers, thresholds, tone
  policies/
    policy_records.json              policy lookup data
docs/
  architecture.md                    Mermaid flow diagram
  DESIGN.md                          design decisions and trade-offs
src/
  run.py                             CLI entry point
  pipeline/                          data_loader → classify → escalate → draft → assess → report
output/                              generated triage reports
```

## Customizing the rules

Escalation trigger wording and the regulatory thresholds (`denied_claim_amount`,
`repeat_complaint_window_days`) live in `routing_guidelines.json`. Removing a trigger there
disables it. The mapping from triggers to extracted signals is in
`src/pipeline/escalate.py`.
