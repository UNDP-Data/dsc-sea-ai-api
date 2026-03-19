#!/usr/bin/env python3
"""
Evaluate `/graph/v2` outputs across a query set and report tuning metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

DEFAULT_QUERIES = [
    "climate change mitigation",
    "solar energy transition policy",
    "grid resilience in africa",
    "clean cooking access",
    "decarbonization and industry",
    "renewable energy finance",
    "energy poverty and gender",
    "electric mobility infrastructure",
    "distributed solar mini grids",
    "energy storage policy",
    "rural electrification",
    "methane mitigation in power sector",
]


@dataclass
class QueryMetrics:
    query: str
    status_code: int
    nodes: int
    edges: int
    central_nodes: int
    secondary_nodes: int
    periphery_nodes: int
    stage1_edges: int
    stage2_edges: int
    stage3_edges: int
    stage2_parents: int
    stage3_parents: int
    branch_counts_stage1: dict[str, int]
    warnings: list[str]


def _request_json(url: str, api_key: str, timeout: int = 45) -> tuple[int, dict]:
    request = urllib.request.Request(url=url, method="GET")
    request.add_header("X-Api-Key", api_key)
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"detail": body}
        return error.code, payload


def _evaluate_graph(query: str, payload: dict, status_code: int) -> QueryMetrics:
    if status_code != 200:
        return QueryMetrics(
            query=query,
            status_code=status_code,
            nodes=0,
            edges=0,
            central_nodes=0,
            secondary_nodes=0,
            periphery_nodes=0,
            stage1_edges=0,
            stage2_edges=0,
            stage3_edges=0,
            stage2_parents=0,
            stage3_parents=0,
            branch_counts_stage1={},
            warnings=[f"request_failed: {payload}"],
        )

    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    tiers = Counter(node.get("tier") for node in nodes)
    node_tiers = {node.get("name"): node.get("tier") for node in nodes}

    def infer_stage(edge: dict) -> int | None:
        if edge.get("level") is not None:
            try:
                return int(edge.get("level"))
            except Exception:
                return None
        source_tier = node_tiers.get(edge.get("subject"))
        target_tier = node_tiers.get(edge.get("object"))
        if source_tier == "central" and target_tier == "secondary":
            return 1
        if source_tier == "secondary" and target_tier == "secondary":
            return 2
        if source_tier == "secondary" and target_tier == "periphery":
            return 3
        if source_tier == "central" and target_tier == "central":
            return 4
        return None

    stage_by_edge = [infer_stage(edge) for edge in edges]
    stage_edges = Counter(stage for stage in stage_by_edge if stage is not None)
    stage1_counts: dict[str, int] = defaultdict(int)
    for edge, stage in zip(edges, stage_by_edge):
        if stage == 1:
            stage1_counts[edge.get("subject", "")] += 1

    stage2_parents = len({edge.get("subject") for edge, stage in zip(edges, stage_by_edge) if stage == 2})
    stage3_parents = len({edge.get("subject") for edge, stage in zip(edges, stage_by_edge) if stage == 3})

    warnings: list[str] = []
    central = tiers.get("central", 0)
    if not (1 <= central <= 3):
        warnings.append(f"central_count_out_of_range={central}")

    for parent, count in stage1_counts.items():
        if count < 3 or count > 6:
            warnings.append(f"stage1_branch_count_{parent}={count}")

    if stage2_parents > 5:
        warnings.append(f"stage2_parents_exceeds_limit={stage2_parents}")
    if stage3_parents > 4:
        warnings.append(f"stage3_parents_exceeds_limit={stage3_parents}")

    if tiers.get("secondary", 0) == 0:
        warnings.append("no_secondary_nodes")

    return QueryMetrics(
        query=query,
        status_code=status_code,
        nodes=len(nodes),
        edges=len(edges),
        central_nodes=central,
        secondary_nodes=tiers.get("secondary", 0),
        periphery_nodes=tiers.get("periphery", 0),
        stage1_edges=stage_edges.get(1, 0),
        stage2_edges=stage_edges.get(2, 0),
        stage3_edges=stage_edges.get(3, 0),
        stage2_parents=stage2_parents,
        stage3_parents=stage3_parents,
        branch_counts_stage1=dict(sorted(stage1_counts.items())),
        warnings=warnings,
    )


def _load_queries(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUERIES
    with open(path, "r", encoding="utf-8") as file:
        data = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]
    return data or DEFAULT_QUERIES


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate /graph/v2 over a query set.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key header value")
    parser.add_argument("--queries-file", default=None, help="Optional newline-separated query list")
    parser.add_argument("--output", default=None, help="Optional JSON file for full metrics output")
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Pass --api-key or set API_KEY in env.", file=sys.stderr)
        return 2

    queries = _load_queries(args.queries_file)
    metrics: list[QueryMetrics] = []

    for query in queries:
        params = urllib.parse.urlencode({"query": query})
        url = f"{args.base_url.rstrip('/')}/graph/v2?{params}"
        status, payload = _request_json(url, api_key=args.api_key)
        metric = _evaluate_graph(query, payload, status)
        metrics.append(metric)

    print("\nV2 Query Evaluation\n")
    for metric in metrics:
        warning_text = "; ".join(metric.warnings) if metric.warnings else "OK"
        print(
            f"- {metric.query}\n"
            f"  status={metric.status_code} nodes={metric.nodes} edges={metric.edges} "
            f"tiers(c/s/p)={metric.central_nodes}/{metric.secondary_nodes}/{metric.periphery_nodes} "
            f"levels(1/2/3)={metric.stage1_edges}/{metric.stage2_edges}/{metric.stage3_edges} "
            f"parents(2/3)={metric.stage2_parents}/{metric.stage3_parents}\n"
            f"  warnings={warning_text}"
        )

    total = len(metrics)
    with_warnings = sum(1 for item in metrics if item.warnings)
    failures = sum(1 for item in metrics if item.status_code != 200)
    print("\nSummary")
    print(f"- queries={total}")
    print(f"- failures={failures}")
    print(f"- with_warnings={with_warnings}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump([asdict(item) for item in metrics], file, indent=2)
        print(f"- wrote={args.output}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
