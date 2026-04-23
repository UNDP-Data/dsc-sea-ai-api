#!/usr/bin/env python3
"""
Run a compact set of `/model` prompts and dump raw NDJSON responses to a text file.
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


def _request_stream(
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

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            chunks: list[dict] = []
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
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"detail": body}
        return error.code, [payload], {}
    except urllib.error.URLError as error:
        return 0, [{"detail": f"Request failed: {error}"}], {}


def _summarize_chunks(chunks: list[dict]) -> dict:
    answer_text = ""
    latest_documents = []
    latest_ideas = []
    graph_summary = None
    graph_chunks = 0
    text_chunks = 0
    document_chunks = 0
    ideas_chunks = 0

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("detail") and len(chunks) == 1:
            return {
                "answer_text": "",
                "answer_length": 0,
                "documents": [],
                "ideas": [],
                "graph": None,
                "chunk_count": len(chunks),
                "stream_stats": {
                    "graph_chunks": 0,
                    "text_chunks": 0,
                    "document_chunks": 0,
                    "ideas_chunks": 0,
                },
                "final_chunk": chunk,
            }
        if chunk.get("graph") is not None:
            graph_chunks += 1
            graph = chunk.get("graph") or {}
            nodes = graph.get("nodes") if isinstance(graph, dict) else []
            edges = graph.get("edges") if isinstance(graph, dict) else []
            graph_summary = {
                "node_count": len(nodes or []),
                "edge_count": len(edges or []),
                "sample_nodes": [
                    node.get("name")
                    for node in (nodes or [])[:5]
                    if isinstance(node, dict) and node.get("name")
                ],
            }
        text = _extract_text(chunk.get("content"))
        if text:
            text_chunks += 1
            answer_text += text
        if isinstance(chunk.get("documents"), list):
            document_chunks += 1
            latest_documents = chunk["documents"]
        if isinstance(chunk.get("ideas"), list):
            ideas_chunks += 1
            latest_ideas = chunk["ideas"]

    return {
        "answer_text": answer_text,
        "answer_length": len(answer_text),
        "documents": latest_documents,
        "ideas": latest_ideas,
        "graph": graph_summary,
        "chunk_count": len(chunks),
        "stream_stats": {
            "graph_chunks": graph_chunks,
            "text_chunks": text_chunks,
            "document_chunks": document_chunks,
            "ideas_chunks": ideas_chunks,
        },
        "final_chunk": chunks[-1] if chunks else None,
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
    parser = argparse.ArgumentParser(description="Dump raw /model NDJSON streams for debugging.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", ""),
        help="API key header value (defaults to API_KEY env var)",
    )
    parser.add_argument(
        "--graph-version",
        default="default",
        choices=["default", "v1", "v2"],
        help="Optional graph version query parameter",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to run. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional newline-separated prompt file.",
    )
    parser.add_argument(
        "--output",
        default="tmp/model_stream_debug.txt",
        help="Output text file path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Pass --api-key or set API_KEY in env.", file=sys.stderr)
        return 2

    prompts = _load_prompts(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    for index, query in enumerate(prompts, start=1):
        status, chunks, headers = _request_stream(
            base_url=args.base_url,
            api_key=args.api_key,
            graph_version=args.graph_version,
            query=query,
            timeout=args.timeout,
        )
        summary = _summarize_chunks(chunks)
        sections.extend(
            [
                f"=== QUERY {index} ===",
                query,
                "",
                "SUMMARY",
                json.dumps(
                    {
                        "status_code": status,
                        "headers": headers,
                        **summary,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "RAW_NDJSON_CHUNKS",
            ]
        )
        if chunks:
            sections.extend(
                json.dumps(chunk, ensure_ascii=False)
                for chunk in chunks
            )
        else:
            sections.append("<no chunks>")
        sections.extend(["", ""])

        print(
            f"[{index}/{len(prompts)}] status={status} chunks={len(chunks)} "
            f"documents={summary['stream_stats']['document_chunks']} "
            f"ideas={summary['stream_stats']['ideas_chunks']} "
            f"text={summary['stream_stats']['text_chunks']}"
        )

    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote debug output to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
