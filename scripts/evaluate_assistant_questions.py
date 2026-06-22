#!/usr/bin/env python3
"""Evaluate assistant-kit questions against retrieval and optional model answers."""

from __future__ import annotations

import argparse
from html import unescape
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_extract_text(item) for item in content)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, dict) and isinstance(text.get("value"), str):
            return text["value"]
        nested = content.get("content")
        if nested is not None:
            return _extract_text(nested)
    return ""


def _request_json(url: str, *, api_key: str, timeout: int) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url=url, method="GET")
    request.add_header("Accept", "application/json")
    request.add_header("X-Api-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"detail": body}
        return error.code, payload


def _request_model(
    *,
    base_url: str,
    api_key: str,
    assistant_id: str,
    query: str,
    timeout: int,
) -> tuple[int, list[dict[str, Any]], str, list[dict[str, Any]]]:
    path = f"/assistants/{urllib.parse.quote(assistant_id, safe='')}/model"
    url = f"{base_url.rstrip('/')}{path}"
    payload = json.dumps([{"role": "human", "content": query}]).encode("utf-8")
    request = urllib.request.Request(url=url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/x-ndjson")
    request.add_header("X-Api-Key", api_key)
    chunks: list[dict[str, Any]] = []
    answer_text = ""
    documents: list[dict[str, Any]] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunks.append(chunk)
            answer_text += _extract_text(chunk.get("content"))
            if isinstance(chunk.get("documents"), list):
                documents = chunk["documents"]
        return response.status, chunks, answer_text, documents


def _normalize_url(value: object) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _doc_rank(documents: list[dict[str, Any]], expected: dict[str, Any]) -> int | None:
    expected_url = _normalize_url(expected.get("url"))
    expected_title = str(expected.get("title") or "").strip().lower()
    for index, document in enumerate(documents, start=1):
        url = _normalize_url(document.get("url"))
        title = str(document.get("title") or "").strip().lower()
        if expected_url and url == expected_url:
            return index
        if expected_title and (expected_title == title or expected_title in title):
            return index
    return None


def _evaluate_question(
    *,
    question: dict[str, Any],
    retrieve_payload: dict[str, Any],
    answer_text: str | None,
) -> dict[str, Any]:
    documents = retrieve_payload.get("documents") if isinstance(retrieve_payload, dict) else []
    if not isinstance(documents, list):
        documents = []

    expected_documents = question.get("expected_documents") or []
    document_checks = []
    for expected in expected_documents:
        if not isinstance(expected, dict):
            continue
        rank = _doc_rank(documents, expected)
        max_rank = int(expected.get("max_rank") or expected.get("rank") or 5)
        document_checks.append(
            {
                "title": expected.get("title"),
                "url": expected.get("url"),
                "rank": rank,
                "max_rank": max_rank,
                "passed": rank is not None and rank <= max_rank,
            }
        )

    term_checks = []
    if answer_text is not None:
        normalized_answer = unescape(answer_text).lower()
        for term in question.get("expected_answer_terms") or []:
            alternatives = term if isinstance(term, list) else [term]
            term_text = " / ".join(str(item) for item in alternatives)
            passed = any(str(item).lower() in normalized_answer for item in alternatives)
            term_checks.append(
                {
                    "term": term_text,
                    "passed": passed,
                }
            )

    return {
        "question_id": question.get("question_id"),
        "query": question.get("query"),
        "retrieved_documents": [
            {
                "rank": index,
                "title": document.get("title"),
                "url": document.get("url"),
                "year": document.get("year"),
            }
            for index, document in enumerate(documents[:8], start=1)
            if isinstance(document, dict)
        ],
        "document_checks": document_checks,
        "term_checks": term_checks,
        "answer_excerpt": (answer_text or "")[:1000] if answer_text is not None else None,
        "passed": all(check["passed"] for check in document_checks + term_checks),
    }


def _write_markdown_report(path: Path, results: list[dict[str, Any]], *, include_model: bool) -> None:
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    lines = [
        "# SGP AI Question Evaluation",
        "",
        f"- Questions: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        f"- Model answer terms checked: {'yes' if include_model else 'no'}",
        "",
    ]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.extend(
            [
                f"## {status}: {result['question_id']}",
                "",
                f"Query: {result['query']}",
                "",
                "Top retrieved documents:",
            ]
        )
        for document in result["retrieved_documents"][:5]:
            lines.append(
                f"- {document['rank']}. {document.get('title') or '(untitled)'}"
            )
        if result["document_checks"]:
            lines.append("")
            lines.append("Expected document checks:")
            for check in result["document_checks"]:
                mark = "PASS" if check["passed"] else "FAIL"
                rank = check["rank"] if check["rank"] is not None else "not found"
                lines.append(
                    f"- {mark}: {check['title']} rank={rank}, max_rank={check['max_rank']}"
                )
        if result["term_checks"]:
            lines.append("")
            lines.append("Expected answer terms:")
            for check in result["term_checks"]:
                mark = "PASS" if check["passed"] else "FAIL"
                lines.append(f"- {mark}: {check['term']}")
        if result.get("answer_excerpt"):
            excerpt = " ".join(str(result["answer_excerpt"]).split())
            lines.append("")
            lines.append("Answer excerpt:")
            lines.append(f"> {excerpt}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        default=str(ROOT / "assistant_kits" / "sgp_ai" / "eval" / "questions.yaml"),
        help="Path to assistant question YAML.",
    )
    parser.add_argument("--base-url", default=os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default=str(ROOT / "tmp" / "sgp_ai_question_eval"),
        help="Output prefix; .json and .md will be written.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Pass --api-key or set API_KEY in .env.", file=sys.stderr)
        return 2

    questions_path = Path(args.questions)
    payload = yaml.safe_load(questions_path.read_text(encoding="utf-8")) or {}
    assistant_id = str(payload.get("assistant_id") or "sgp_ai")
    questions = payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        print(f"No questions found in {questions_path}", file=sys.stderr)
        return 2

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for index, question in enumerate(questions, start=1):
        query = str(question.get("query") or "")
        retrieve_path = f"/assistants/{urllib.parse.quote(assistant_id, safe='')}/debug/retrieve"
        retrieve_url = (
            f"{args.base_url.rstrip('/')}{retrieve_path}?"
            + urllib.parse.urlencode({"query": query, "limit": args.limit})
        )
        retrieve_status, retrieve_payload = _request_json(
            retrieve_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        answer_text: str | None = None
        model_status = None
        model_documents = []
        if not args.skip_model:
            model_status, _chunks, answer_text, model_documents = _request_model(
                base_url=args.base_url,
                api_key=args.api_key,
                assistant_id=assistant_id,
                query=query,
                timeout=args.timeout,
            )
        result = _evaluate_question(
            question=question,
            retrieve_payload=retrieve_payload,
            answer_text=answer_text,
        )
        result.update(
            {
                "retrieve_status": retrieve_status,
                "model_status": model_status,
                "answer_length": len(answer_text or ""),
                "model_documents": [
                    {
                        "title": document.get("title"),
                        "url": document.get("url"),
                        "year": document.get("year"),
                    }
                    for document in model_documents[:8]
                    if isinstance(document, dict)
                ],
            }
        )
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{index}/{len(questions)}] {status} {result['question_id']}")

    json_path = output_prefix.with_suffix(".json")
    markdown_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(markdown_path, results, include_model=not args.skip_model)
    passed = sum(1 for result in results if result["passed"])
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(json.dumps({"questions": len(results), "passed": passed, "failed": len(results) - passed}))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
