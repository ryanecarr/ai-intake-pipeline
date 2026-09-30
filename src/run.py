"""
AI Intake Pipeline — complaint triage entry point.

Usage:
    python src/run.py
    python src/run.py --complaint data/complaints/complaint_premium_question.json
"""

import argparse
import sys
from pathlib import Path

from pipeline import TriageStepError, run_complaint_triage
from pipeline.data_loader import COMPLAINT_PATH, GUIDELINES_PATH, POLICIES_PATH


def main():
    parser = argparse.ArgumentParser(description="Triage an insurance customer complaint.")
    parser.add_argument("--complaint", type=Path, default=COMPLAINT_PATH, help="Complaint JSON to triage")
    parser.add_argument("--guidelines", type=Path, default=GUIDELINES_PATH, help="Routing guidelines JSON")
    parser.add_argument("--policies", type=Path, default=POLICIES_PATH, help="Policy records JSON")
    args = parser.parse_args()

    print(f"Triaging {args.complaint}\n")
    try:
        path = run_complaint_triage(
            complaint_path=args.complaint,
            guidelines_path=args.guidelines,
            policies_path=args.policies,
        )
    except TriageStepError as exc:
        # Fail-fast: a step failed, so no (partial, misleading) report is written.
        print(f"\n❌ Pipeline aborted — {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\n✅ Done. Triage report: {path}")


if __name__ == "__main__":
    main()
