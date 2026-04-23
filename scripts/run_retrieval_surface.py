#!/usr/bin/env python3
"""
Run a fixed prompt set against `/debug/retrieve` and `/model`, then save a combined
diagnostic report for review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

DEFAULT_PROMPTS = [
    "what is the connection between sustainable energy and climate change mitigation",
    "tell me more about feed in tariff",
    "tell me more about access to electricity",
    "tell me more about grid infrastructure",
]


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


def _request_json(
    *,
    url: str,
    api_key: str,
    timeout: int,
) -> tuple[int, dict]:
    request = urllib.request.Request(url=url, method="GET")
    request.add_header("Accept", "application/json")
    request.add_header("X-Api-Key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"detail": body}
        return error.code, payload


def _request_model_stream(
    *,
    base_url: str,
    api_key: str,
    graph_version: str,
    query: str,
    timeout: int,
) -> tuple[int, list[dict], dict[str, str]]:
    path = "/model"
    if graph_version != "default":
        path += "?" + urllib.parse.urlencode({"graph_version": graph_version})
    url = f"{base_url.rstrip('/')}{path}"
    payload = json.dumps([{"role": "human", "content": query}]).encode("utf-8")
    request = urllib.request.Request(url=url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/x-ndjson")
    request.add_header("X-Api-Key", api_key)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunks = []
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            chunks.append(json.loads(line))
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"x-request-id", "x-kg-timing", "server-timing", "content-type"}
        }
        return response.status, chunks, headers


def _summarize_model_chunks(chunks: list[dict]) -> dict:
    answer_text = ""
    documents = []
    ideas = []
    graph = None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("graph") is not None:
            graph_payload = chunk.get("graph") or {}
            graph = {
                "node_count": len(graph_payload.get("nodes") or []),
                "edge_count": len(graph_payload.get("edges") or []),
                "sample_nodes": [
                    node.get("name")
                    for node in (graph_payload.get("nodes") or [])[:5]
                    if isinstance(node, dict)
                ],
            }
        text = _extract_text(chunk.get("content"))
        if text:
            answer_text += text
        if isinstance(chunk.get("documents"), list):
            documents = chunk["documents"]
        if isinstance(chunk.get("ideas"), list):
            ideas = chunk["ideas"]
    return {
        "answer_text": answer_text,
        "answer_length": len(answer_text),
        "documents": documents,
        "ideas": ideas,
        "graph": graph,
        "chunk_count": len(chunks),
    }


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return args.prompt
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as handle:
            prompts = [
                line.strip()
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            ]
        return prompts or DEFAULT_PROMPTS
    return DEFAULT_PROMPTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", ""),
        help="API key (defaults to API_KEY env var)",
    )
    parser.add_argument(
        "--graph-version",
        default="default",
        choices=["default", "v1", "v2"],
        help="Optional graph version to send to /model",
    )
    parser.add_argument("--prompt", action="append", help="Prompt to run, repeatable.")
    parser.add_argument("--prompts-file", default=None, help="Optional newline-separated prompt file.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--output-prefix",
        default="tmp/retrieval_surface",
        help="Output prefix; .json and .txt will be written.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Pass --api-key or set API_KEY in env.", file=sys.stderr)
        return 2

    prompts = _load_prompts(args)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    results = []
    text_sections = []

    for index, prompt in enumerate(prompts, start=1):
        retrieve_params = urllib.parse.urlencode({"query": prompt, "limit": 12})
        retrieve_url = f"{args.base_url.rstrip('/')}/debug/retrieve?{retrieve_params}"
        retrieve_status, retrieve_payload = _request_json(
            url=retrieve_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        model_status, model_chunks, model_headers = _request_model_stream(
            base_url=args.base_url,
            api_key=args.api_key,
            graph_version=args.graph_version,
            query=prompt,
            timeout=args.timeout,
        )
        model_summary = _summarize_model_chunks(model_chunks)
        result = {
            "index": index,
            "query": prompt,
            "retrieve_status": retrieve_status,
            "retrieve_payload": retrieve_payload,
            "model_status": model_status,
            "model_headers": model_headers,
            "model_summary": model_summary,
            "model_chunks": model_chunks,
        }
        results.append(result)

        text_sections.extend(
            [
                f"=== QUERY {index} ===",
                prompt,
                "",
                "RETRIEVE_DEBUG",
                json.dumps(retrieve_payload, indent=2, ensure_ascii=False),
                "",
                "MODEL_SUMMARY",
                json.dumps(
                    {
                        "status": model_status,
                        "headers": model_headers,
                        **model_summary,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "MODEL_RAW_NDJSON",
            ]
        )
        text_sections.extend(json.dumps(chunk, ensure_ascii=False) for chunk in model_chunks)
        text_sections.extend(["", ""])

        selected_path = (
            retrieve_payload.get("debug", {}).get("selected", {}).get("path")
            if isinstance(retrieve_payload, dict)
            else None
        )
        selected_docs = len(retrieve_payload.get("documents") or []) if isinstance(retrieve_payload, dict) else 0
        print(
            f"[{index}/{len(prompts)}] "
            f"retrieve={retrieve_status} model={model_status} "
            f"path={selected_path or '-'} docs={selected_docs} "
            f"stream_docs={len(model_summary['documents'])} ideas={len(model_summary['ideas'])}"
        )

    json_path = output_prefix.with_suffix(".json")
    text_path = output_prefix.with_suffix(".txt")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text("\n".join(text_sections), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
