#!/usr/bin/env python3
"""Validate retrieval benchmark question bank and expert annotation CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

QUESTION_REQUIRED = {
    "question_id",
    "question_text",
    "question_type",
    "expected_answer_mode",
    "priority_tier",
}

ANNOTATION_REQUIRED = {
    "batch_id",
    "expert_id",
    "question_id",
    "question_text",
    "selected_rank",
    "resource_title",
    "resource_url",
    "resource_role",
    "selection_rationale",
    "selection_confidence",
    "corpus_gap_flag",
}

VALID_ROLES = {
    "primary_source",
    "latest_data",
    "policy_context",
    "implementation_example",
    "regional_context",
    "methodology",
    "case_example",
}

VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_GAP = {"yes", "no"}


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_headers(rows: list[dict], required: set[str]) -> list[str]:
    if not rows:
        return ["CSV contains no rows."]
    missing = sorted(required - set(rows[0].keys()))
    return [f"Missing required columns: {', '.join(missing)}"] if missing else []


def _validate_questions(rows: list[dict]) -> tuple[list[str], set[str]]:
    errors = _check_headers(rows, QUESTION_REQUIRED)
    question_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        qid = (row.get("question_id") or "").strip()
        text = (row.get("question_text") or "").strip()
        if not qid:
            errors.append(f"questions:{index}: empty question_id")
        if qid in question_ids:
            errors.append(f"questions:{index}: duplicate question_id {qid}")
        question_ids.add(qid)
        if not text:
            errors.append(f"questions:{index}: empty question_text for {qid}")
    return errors, question_ids


def _validate_annotations(rows: list[dict], question_ids: set[str]) -> list[str]:
    errors = _check_headers(rows, ANNOTATION_REQUIRED)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        qid = (row.get("question_id") or "").strip()
        expert_id = (row.get("expert_id") or "").strip()
        title = (row.get("resource_title") or "").strip()
        url = (row.get("resource_url") or "").strip()
        rationale = (row.get("selection_rationale") or "").strip()
        role = (row.get("resource_role") or "").strip()
        confidence = (row.get("selection_confidence") or "").strip().lower()
        gap = (row.get("corpus_gap_flag") or "").strip().lower()
        if qid not in question_ids:
            errors.append(f"annotations:{index}: unknown question_id {qid}")
        if not expert_id:
            errors.append(f"annotations:{index}: empty expert_id")
        if not title:
            errors.append(f"annotations:{index}: empty resource_title")
        if not url:
            errors.append(f"annotations:{index}: empty resource_url")
        if not rationale:
            errors.append(f"annotations:{index}: empty selection_rationale")
        if role not in VALID_ROLES:
            errors.append(f"annotations:{index}: invalid resource_role {role}")
        if confidence not in VALID_CONFIDENCE:
            errors.append(f"annotations:{index}: invalid selection_confidence {confidence}")
        if gap not in VALID_GAP:
            errors.append(f"annotations:{index}: invalid corpus_gap_flag {gap}")
        try:
            rank = int((row.get("selected_rank") or "").strip())
        except ValueError:
            errors.append(f"annotations:{index}: selected_rank is not an integer")
            rank = None
        if rank is not None and not 1 <= rank <= 5:
            errors.append(f"annotations:{index}: selected_rank out of range {rank}")
        grouped[(expert_id, qid)].append(row)

    for (expert_id, qid), items in grouped.items():
        if not 3 <= len(items) <= 5:
            errors.append(
                f"annotations: expert={expert_id} question={qid} has {len(items)} rows; expected 3 to 5"
            )
            continue
        ranks = []
        for row in items:
            try:
                ranks.append(int((row.get("selected_rank") or "").strip()))
            except ValueError:
                pass
        if len(set(ranks)) != len(ranks):
            errors.append(f"annotations: expert={expert_id} question={qid} has duplicate ranks")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            errors.append(
                f"annotations: expert={expert_id} question={qid} ranks must be contiguous starting at 1; got {sorted(ranks)}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Question bank CSV")
    parser.add_argument("--annotations", required=True, help="Annotation CSV")
    args = parser.parse_args()

    question_rows = _load_csv(Path(args.questions))
    annotation_rows = _load_csv(Path(args.annotations))

    question_errors, question_ids = _validate_questions(question_rows)
    annotation_errors = _validate_annotations(annotation_rows, question_ids)
    errors = question_errors + annotation_errors

    report = {
        "questions": len(question_rows),
        "annotation_rows": len(annotation_rows),
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
