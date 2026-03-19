"""
V2 staged graph builder.
"""

from __future__ import annotations

import colorsys
import heapq
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from time import perf_counter

import networkx as nx

from ..database import Client
from .types import EdgeV2, GraphV2, NodeV2

CENTRAL_COLOUR = "#E5E7EB"
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "link",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


@dataclass(frozen=True)
class BranchingConfig:
    """
    Tuning configuration for staged branching.
    """

    first_secondary_min: int = 3
    first_secondary_max: int = 6
    second_parents_max: int = 5
    second_secondary_min: int = 2
    second_secondary_max: int = 4
    final_parents_max: int = 4
    final_periphery_min: int = 2
    final_periphery_max: int = 4


@dataclass
class Candidate:
    """
    Intermediate neighbour candidate in staged selection.
    """

    node_name: str
    parent: str
    predicate: str
    description: str | None
    weight: float
    score: float
    domain: str
    bridge_score: float = 0.0


@dataclass(frozen=True)
class LexicalCandidate:
    node_name: str
    score: float
    contains_phrase: bool
    token_count: int
    weight: float


@dataclass
class EdgeRecord:
    subject: str
    predicate: str
    object: str
    description: str | None
    weight: float
    level: int


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _token_set(text: str) -> set[str]:
    return set(_normalise(text).split())


def _lexical_similarity(text_a: str, text_b: str) -> float:
    """
    Lightweight lexical similarity in [0, 1].
    """
    tokens_a = _token_set(text_a)
    tokens_b = _token_set(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    ratio = SequenceMatcher(None, _normalise(text_a), _normalise(text_b)).ratio()
    return max(jaccard, ratio)


def _is_near_synonym(text_a: str, text_b: str) -> bool:
    """
    Approximate synonym / near-duplicate detector.
    """
    normal_a = _normalise(text_a)
    normal_b = _normalise(text_b)
    if not normal_a or not normal_b:
        return False
    if normal_a == normal_b:
        return True
    if normal_a in normal_b or normal_b in normal_a:
        shorter = min(len(normal_a), len(normal_b))
        longer = max(len(normal_a), len(normal_b))
        if longer > 0 and (shorter / longer) >= 0.82:
            return True
    tokens_a = _token_set(text_a)
    tokens_b = _token_set(text_b)
    overlap = len(tokens_a & tokens_b) / max(1, min(len(tokens_a), len(tokens_b)))
    return overlap >= 0.92 or SequenceMatcher(None, normal_a, normal_b).ratio() >= 0.93


def _score_candidate(query: str, edge_weight: float, node_weight: float, text: str) -> float:
    """
    Combined relevance score for staged branching.
    """
    edge_component = edge_weight / (edge_weight + 1.0)
    node_component = node_weight / (node_weight + 1.0)
    text_component = _lexical_similarity(query, text)
    return 0.55 * edge_component + 0.2 * node_component + 0.25 * text_component


def _target_count(size: int, minimum: int, maximum: int) -> int:
    """
    Pick count bounds by candidate pool size.
    """
    if size <= 0:
        return 0
    target = minimum + (size // 6)
    return max(minimum, min(maximum, target))


def _generate_dark_rainbow(n: int) -> list[str]:
    """
    Generate a dark rainbow palette distributed evenly.
    """
    if n <= 0:
        return []
    colours: list[str] = []
    for index in range(n):
        hue = index / n
        red, green, blue = colorsys.hls_to_rgb(hue, 0.38, 0.80)
        colours.append(f"#{int(red*255):02X}{int(green*255):02X}{int(blue*255):02X}")
    return colours


def _add_node(
    registry: dict[str, NodeV2],
    name: str,
    description: str | None,
    tier: str,
    weight: float,
    colour: str,
) -> None:
    """
    Insert or upgrade node tier in the registry.
    """
    if name not in registry:
        registry[name] = NodeV2(
            name=name,
            description=description,
            tier=tier,  # type: ignore[arg-type]
            weight=max(0.0, float(weight)),
            colour=colour,
        )
        return

    current = registry[name]
    priority = {"central": 3, "secondary": 2, "periphery": 1}
    chosen_tier = tier if priority[tier] > priority[current.tier] else current.tier
    registry[name] = NodeV2(
        name=name,
        description=current.description or description,
        tier=chosen_tier,  # type: ignore[arg-type]
        weight=max(current.weight, float(weight)),
        colour=current.colour if current.tier == "central" else colour,
    )


def _add_edge(
    registry: dict[tuple[str, str, str], EdgeRecord],
    subject: str,
    predicate: str,
    object_name: str,
    description: str | None,
    weight: float,
    level: int,
) -> None:
    key = (subject, object_name, predicate)
    if key in registry:
        existing = registry[key]
        existing.weight = max(existing.weight, weight)
        existing.level = min(existing.level, level)
        return
    registry[key] = EdgeRecord(
        subject=subject,
        predicate=predicate or "related_to",
        object=object_name,
        description=description,
        weight=max(0.0, float(weight)),
        level=max(1, level),
    )


def _bridge_bonus(graph: nx.DiGraph, node_name: str, centres: list[str]) -> float:
    """
    Reward nodes that connect to multiple central concepts.
    """
    if len(centres) <= 1 or node_name not in graph:
        return 0.0
    neighbours = set(graph.predecessors(node_name)) | set(graph.successors(node_name))
    shared = len(set(centres) & neighbours)
    return min(1.0, shared / max(1, len(centres) - 1))


def _pick_diverse(
    candidates: list[Candidate], target: int, blocked: set[str], parent_name: str
) -> list[Candidate]:
    """
    Pick candidates with domain diversity and duplicate suppression.
    """
    by_domain: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.node_name)):
        by_domain[candidate.domain].append(candidate)

    selected: list[Candidate] = []
    domains = sorted(
        by_domain.keys(), key=lambda domain: -by_domain[domain][0].score if by_domain[domain] else 0.0
    )
    while len(selected) < target and domains:
        still_open = []
        for domain in domains:
            group = by_domain[domain]
            while group:
                candidate = group.pop(0)
                if candidate.node_name in blocked:
                    continue
                if any(_is_near_synonym(candidate.node_name, existing) for existing in blocked):
                    continue
                if _is_near_synonym(candidate.node_name, parent_name):
                    continue
                if any(_is_near_synonym(candidate.node_name, item.node_name) for item in selected):
                    continue
                selected.append(candidate)
                blocked.add(candidate.node_name)
                break
            if group:
                still_open.append(domain)
            if len(selected) >= target:
                break
        domains = still_open
    return selected


def _collect_neighbours(
    graph: nx.DiGraph,
    parent: str,
    query: str,
    blocked: set[str],
    centres: list[str],
) -> list[Candidate]:
    """
    Collect scored neighbours around a parent node.
    """
    candidates: dict[str, Candidate] = {}
    parent_description = graph.nodes[parent].get("description", "") if parent in graph else ""
    for source, target, data in graph.out_edges(parent, data=True):
        neighbour = target
        if neighbour in blocked:
            continue
        edge_weight = float(data.get("weight", 0.0) or 0.0)
        node_weight = float(graph.nodes[neighbour].get("weight", 0.0) or 0.0)
        node_description = graph.nodes[neighbour].get("description", None)
        text = f"{neighbour} {node_description or ''} {parent_description}"
        score = _score_candidate(query, edge_weight, node_weight, text)
        predicate = str(data.get("predicate", "related_to"))
        domain = predicate.split()[0].strip().lower() or "related"
        candidate = Candidate(
            node_name=neighbour,
            parent=source,
            predicate=predicate,
            description=data.get("description"),
            weight=edge_weight,
            score=score,
            domain=domain,
            bridge_score=_bridge_bonus(graph, neighbour, centres),
        )
        if neighbour not in candidates or candidate.score > candidates[neighbour].score:
            candidates[neighbour] = candidate

    for source, target, data in graph.in_edges(parent, data=True):
        neighbour = source
        if neighbour in blocked:
            continue
        edge_weight = float(data.get("weight", 0.0) or 0.0)
        node_weight = float(graph.nodes[neighbour].get("weight", 0.0) or 0.0)
        node_description = graph.nodes[neighbour].get("description", None)
        text = f"{neighbour} {node_description or ''} {parent_description}"
        score = _score_candidate(query, edge_weight, node_weight, text)
        predicate = str(data.get("predicate", "related_to"))
        domain = predicate.split()[0].strip().lower() or "related"
        candidate = Candidate(
            node_name=neighbour,
            parent=parent,
            predicate=predicate,
            description=data.get("description"),
            weight=edge_weight,
            score=score,
            domain=domain,
            bridge_score=_bridge_bonus(graph, neighbour, centres),
        )
        if neighbour not in candidates or candidate.score > candidates[neighbour].score:
            candidates[neighbour] = candidate

    return sorted(
        candidates.values(),
        key=lambda item: (-(item.score + 0.12 * item.bridge_score), item.node_name),
    )


def _collect_two_hop_candidates(
    graph: nx.DiGraph,
    parent: str,
    query: str,
    blocked: set[str],
    centres: list[str],
) -> list[Candidate]:
    """
    Controlled fallback candidate pool from 2-hop neighbourhood.
    """
    if parent not in graph:
        return []
    undirected = graph.to_undirected()
    paths = nx.single_source_shortest_path(undirected, source=parent, cutoff=2)
    candidates: dict[str, Candidate] = {}
    for target, path in paths.items():
        if target == parent or len(path) != 3:
            continue
        if target in blocked:
            continue
        mid = path[1]
        edge_data_1 = graph.get_edge_data(parent, mid) or graph.get_edge_data(mid, parent) or {}
        edge_data_2 = graph.get_edge_data(mid, target) or graph.get_edge_data(target, mid) or {}
        edge_weight = (
            float(edge_data_1.get("weight", 0.0) or 0.0)
            + float(edge_data_2.get("weight", 0.0) or 0.0)
        ) / 2.0
        edge_weight *= 0.85
        node_weight = float(graph.nodes[target].get("weight", 0.0) or 0.0)
        node_description = graph.nodes[target].get("description", "")
        text = f"{target} {node_description} via {mid}"
        base_score = _score_candidate(query, edge_weight, node_weight, text)
        score = 0.92 * base_score + 0.08 * _bridge_bonus(graph, target, centres)
        candidate = Candidate(
            node_name=target,
            parent=parent,
            predicate=f"via {mid}",
            description=f"Two-hop relation via {mid}",
            weight=max(0.05, edge_weight),
            score=score,
            domain=f"via_{_normalise(mid).replace(' ', '_')[:18] or 'bridge'}",
            bridge_score=_bridge_bonus(graph, target, centres),
        )
        if target not in candidates or candidate.score > candidates[target].score:
            candidates[target] = candidate
    return sorted(candidates.values(), key=lambda item: (-item.score, item.node_name))


def _central_target(query: str) -> int:
    terms = [term for term in re.findall(r"[a-zA-Z0-9]+", query) if len(term) > 2]
    if len(terms) >= 10:
        return 3
    if len(terms) >= 5:
        return 2
    return 1


def _allow_additional_central(best_distance: float, candidate_distance: float) -> bool:
    """
    Gate 2nd/3rd central nodes by semantic confidence relative to the top hit.
    """
    best = max(0.0, float(best_distance))
    candidate = max(0.0, float(candidate_distance))
    return candidate <= (best * 1.35 + 0.05)


def _query_token_set(query: str) -> set[str]:
    return {token for token in _token_set(query) if token not in QUERY_STOPWORDS and len(token) > 2}


def _find_lexical_central_nodes(graph: nx.Graph, query: str, target: int) -> list[str]:
    """
    Fast in-memory lexical central selection to avoid expensive remote vector lookup when possible.
    """
    if target <= 0:
        return []
    norm_query = _normalise(query)
    query_tokens = _query_token_set(query)
    if not norm_query or not query_tokens:
        return []

    candidates: list[LexicalCandidate] = []
    for name in graph.nodes:
        if not isinstance(name, str):
            continue
        norm_name = _normalise(name)
        if not norm_name:
            continue
        name_tokens = set(norm_name.split())
        if not name_tokens:
            continue
        contains_phrase = len(name_tokens) >= 2 and norm_name in norm_query
        overlap = len(name_tokens & query_tokens)
        if overlap == 0 and not contains_phrase:
            continue
        coverage = overlap / max(1, len(name_tokens))
        jaccard = overlap / max(1, len(name_tokens | query_tokens))
        ratio = SequenceMatcher(None, norm_name, norm_query).ratio()
        contains_bonus = 1.0 if contains_phrase else 0.0
        phrase_len_bonus = min(0.12, 0.03 * max(0, len(name_tokens) - 1))
        score = (
            0.47 * coverage
            + 0.16 * jaccard
            + 0.21 * ratio
            + 0.10 * contains_bonus
            + phrase_len_bonus
        )
        if score < 0.35:
            continue
        candidates.append(
            LexicalCandidate(
                node_name=name,
                score=score,
                contains_phrase=contains_phrase,
                token_count=len(name_tokens),
                weight=float(graph.nodes[name].get("weight", 0.0) or 0.0),
            )
        )

    if not candidates:
        return []

    top = heapq.nlargest(
        24,
        candidates,
        key=lambda item: (
            item.score,
            1.0 if item.contains_phrase else 0.0,
            item.token_count,
            item.weight,
        ),
    )
    top.sort(
        key=lambda item: (
            -item.score,
            -int(item.contains_phrase),
            -item.token_count,
            -item.weight,
            item.node_name,
        )
    )
    best = top[0]
    strong_enough = best.score >= 0.73 or (best.contains_phrase and best.score >= 0.64)
    if not strong_enough:
        return []

    selected = [best.node_name]
    best_score = best.score
    for candidate in top[1:]:
        if len(selected) >= target:
            break
        if any(_is_near_synonym(candidate.node_name, existing) for existing in selected):
            continue
        if candidate.score < max(0.56, best_score * 0.78):
            continue
        selected.append(candidate.node_name)
    return selected


async def _find_central_nodes(
    client: Client, graph: nx.Graph, query: str, timings: dict[str, float] | None = None
) -> list[str]:
    """
    Select 1-3 central nodes from vector search candidates.
    """
    lexical_start = perf_counter()

    # Fast path: if the query already matches a node name, skip remote embedding call.
    normalised_query = _normalise(query)
    if normalised_query:
        exact_matches = [
            name
            for name in graph.nodes
            if isinstance(name, str) and _normalise(name) == normalised_query
        ]
        if exact_matches:
            best = max(
                exact_matches,
                key=lambda name: float(graph.nodes[name].get("weight", 0.0) or 0.0),
            )
            if timings is not None:
                timings["central_exact_match_ms"] = round(
                    (perf_counter() - lexical_start) * 1000, 2
                )
                timings["central_embed_ms"] = 0.0
                timings["central_vector_search_ms"] = 0.0
                timings["central_lexical_ms"] = 0.0
            return [best]
    if timings is not None:
        timings["central_exact_match_ms"] = round((perf_counter() - lexical_start) * 1000, 2)

    lexical_candidate_start = perf_counter()
    target = _central_target(query)
    lexical_nodes = _find_lexical_central_nodes(graph, query, target=target)
    if timings is not None:
        timings["central_lexical_ms"] = round((perf_counter() - lexical_candidate_start) * 1000, 2)
    if lexical_nodes:
        if timings is not None:
            timings["central_embed_ms"] = 0.0
            timings["central_vector_search_ms"] = 0.0
        return lexical_nodes

    table = await client.connection.open_table("nodes")
    embed_start = perf_counter()
    vector = await client.embedder.aembed_query(query)
    if timings is not None:
        timings["central_embed_ms"] = round((perf_counter() - embed_start) * 1000, 2)
    vector_start = perf_counter()
    rows = await table.vector_search(vector).limit(10).select(["name", "_distance"]).to_list()
    if timings is not None:
        timings["central_vector_search_ms"] = round((perf_counter() - vector_start) * 1000, 2)
    selected: list[str] = []
    best_distance: float | None = None
    for row in rows:
        name = row["name"]
        if name not in graph.nodes:
            continue
        if any(_is_near_synonym(name, existing) for existing in selected):
            continue
        row_distance = float(row.get("_distance", 1.0) or 1.0)
        if not selected:
            selected.append(name)
            best_distance = row_distance
            if len(selected) >= target:
                break
            continue
        if best_distance is None:
            best_distance = row_distance
        if not _allow_additional_central(best_distance, row_distance):
            continue
        selected.append(name)
        if len(selected) >= target:
            break
    return selected


async def _collect_semantic_candidates(
    client: Client,
    graph: nx.Graph,
    parent: str,
    query: str,
    blocked: set[str],
    centres: list[str],
) -> list[Candidate]:
    """
    Final fallback pool from semantic nearest-neighbour search over all nodes.
    """
    try:
        table = await client.connection.open_table("nodes")
        semantic_query = f"{query} {parent}"
        vector = await client.embedder.aembed_query(semantic_query)
        rows = (
            await table.vector_search(vector)
            .limit(64)
            .select(["name", "description", "weight", "_distance"])
            .to_list()
        )
    except Exception:
        return []

    parent_description = graph.nodes[parent].get("description", "") if parent in graph else ""
    candidates: dict[str, Candidate] = {}
    for row in rows:
        name = row.get("name")
        if not name or name not in graph:
            continue
        if name in blocked:
            continue
        if _is_near_synonym(name, parent):
            continue

        distance = float(row.get("_distance", 1.0) or 1.0)
        semantic_score = 1.0 / (1.0 + max(0.0, distance))
        node_weight = float(graph.nodes[name].get("weight", 0.0) or 0.0)
        node_description = graph.nodes[name].get("description", "")
        lexical_score = _lexical_similarity(query, f"{name} {node_description} {parent_description}")
        score = 0.65 * semantic_score + 0.20 * lexical_score + 0.15 * (
            node_weight / (node_weight + 1.0)
        )

        edge_data = graph.get_edge_data(parent, name) or graph.get_edge_data(name, parent) or {}
        edge_weight = float(edge_data.get("weight", score) or score)
        candidate = Candidate(
            node_name=name,
            parent=parent,
            predicate=str(edge_data.get("predicate", "semantic_related")),
            description=edge_data.get(
                "description", "Semantic relation from nearest-neighbour node search"
            ),
            weight=max(0.05, edge_weight),
            score=score,
            domain="semantic",
            bridge_score=_bridge_bonus(graph, name, centres),
        )
        if name not in candidates or candidate.score > candidates[name].score:
            candidates[name] = candidate

    return sorted(candidates.values(), key=lambda item: (-item.score, item.node_name))


async def build_subgraph_v2(
    client: Client,
    graph: nx.Graph,
    query: str,
    config: BranchingConfig | None = None,
    timings: dict[str, float] | None = None,
) -> GraphV2:
    """
    Build staged V2 subgraph in parallel to the current implementation.
    """
    total_start = perf_counter()
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        if timings is not None:
            timings["total_ms"] = round((perf_counter() - total_start) * 1000, 2)
        return GraphV2(nodes=[], edges=[])
    config = config or BranchingConfig()

    central_start = perf_counter()
    try:
        try:
            central_nodes = await _find_central_nodes(client, graph, query, timings=timings)
        except TypeError:
            # Compatibility for monkeypatched test doubles that accept old signature.
            central_nodes = await _find_central_nodes(client, graph, query)
    except Exception:
        if timings is not None:
            timings["central_selection_ms"] = round((perf_counter() - central_start) * 1000, 2)
            timings["total_ms"] = round((perf_counter() - total_start) * 1000, 2)
        return GraphV2(nodes=[], edges=[])
    if timings is not None:
        timings["central_selection_ms"] = round((perf_counter() - central_start) * 1000, 2)

    if not central_nodes:
        if timings is not None:
            timings["total_ms"] = round((perf_counter() - total_start) * 1000, 2)
        return GraphV2(nodes=[], edges=[])

    nodes: dict[str, NodeV2] = {}
    edges: dict[tuple[str, str, str], EdgeRecord] = {}
    branch_by_node: dict[str, str] = {}
    scored_secondary: dict[str, float] = {}
    branched: set[str] = set()
    selected_names = set(central_nodes)

    for central in central_nodes:
        central_weight = float(graph.nodes[central].get("weight", 0.0) or 0.0)
        _add_node(
            registry=nodes,
            name=central,
            description=graph.nodes[central].get("description"),
            tier="central",
            weight=central_weight,
            colour=CENTRAL_COLOUR,
        )

    # Keep central nodes connected by default to avoid detached graph components.
    ordered_centrals = sorted(set(central_nodes))
    for index, source in enumerate(ordered_centrals):
        for target in ordered_centrals[index + 1 :]:
            edge_data = graph.get_edge_data(source, target) or graph.get_edge_data(target, source) or {}
            _add_edge(
                registry=edges,
                subject=source,
                predicate=str(edge_data.get("predicate", "central_related_to")),
                object_name=target,
                description=edge_data.get(
                    "description",
                    "Auto-linked central nodes to keep the graph connected.",
                ),
                weight=float(edge_data.get("weight", 1.0) or 1.0),
                level=4,
            )

    # Stage 1: 3-6 secondary nodes per central.
    stage1_start = perf_counter()
    first_stage: list[Candidate] = []
    for central in central_nodes:
        blocked = selected_names.copy()
        direct_candidates = [
            item
            for item in _collect_neighbours(graph, central, query, blocked, central_nodes)
            if not _is_near_synonym(item.node_name, central)
        ]
        target = _target_count(
            len(direct_candidates),
            minimum=config.first_secondary_min,
            maximum=config.first_secondary_max,
        )
        selected = _pick_diverse(
            direct_candidates, target, selected_names, parent_name=central
        )
        if len(selected) < config.first_secondary_min:
            fallback_pool = [
                item
                for item in _collect_two_hop_candidates(
                    graph, central, query, selected_names, central_nodes
                )
                if not _is_near_synonym(item.node_name, central)
                and item.node_name not in {node.node_name for node in selected}
            ]
            selected.extend(
                _pick_diverse(
                    fallback_pool,
                    config.first_secondary_min - len(selected),
                    selected_names,
                    parent_name=central,
                )
            )
        if len(selected) < config.first_secondary_min:
            semantic_pool = [
                item
                for item in await _collect_semantic_candidates(
                    client, graph, central, query, selected_names, central_nodes
                )
                if item.node_name not in {node.node_name for node in selected}
            ]
            selected.extend(
                _pick_diverse(
                    semantic_pool,
                    config.first_secondary_min - len(selected),
                    selected_names,
                    parent_name=central,
                )
            )
        first_stage.extend(selected)
        for candidate in selected:
            scored_secondary[candidate.node_name] = candidate.score
            _add_edge(
                registry=edges,
                subject=central,
                predicate=candidate.predicate,
                object_name=candidate.node_name,
                description=candidate.description,
                weight=candidate.weight,
                level=1,
            )
    if timings is not None:
        timings["stage1_ms"] = round((perf_counter() - stage1_start) * 1000, 2)

    first_stage_names = [item.node_name for item in first_stage]
    palette = _generate_dark_rainbow(len(first_stage_names))
    colour_by_branch = {name: palette[index] for index, name in enumerate(first_stage_names)}
    for candidate in first_stage:
        branch_colour = colour_by_branch.get(candidate.node_name, "#3B82F6")
        branch_by_node[candidate.node_name] = candidate.node_name
        node_weight = float(graph.nodes[candidate.node_name].get("weight", 0.0) or 0.0)
        _add_node(
            registry=nodes,
            name=candidate.node_name,
            description=graph.nodes[candidate.node_name].get("description"),
            tier="secondary",
            weight=node_weight,
            colour=branch_colour,
        )

    # Stage 2: top 5 secondary branchers, add 2-4 nodes each (still secondary).
    stage2_start = perf_counter()
    second_stage_parents = sorted(
        first_stage_names, key=lambda name: scored_secondary.get(name, 0.0), reverse=True
    )[: config.second_parents_max]
    for parent in second_stage_parents:
        branched.add(parent)
        blocked = selected_names.copy()
        candidates = _collect_neighbours(graph, parent, query, blocked, central_nodes)
        target = _target_count(
            len(candidates),
            minimum=config.second_secondary_min,
            maximum=config.second_secondary_max,
        )
        selected = _pick_diverse(candidates, target, selected_names, parent_name=parent)
        branch_root = branch_by_node.get(parent, parent)
        branch_colour = colour_by_branch.get(branch_root, "#3B82F6")
        for candidate in selected:
            scored_secondary[candidate.node_name] = candidate.score
            branch_by_node[candidate.node_name] = branch_root
            node_weight = float(graph.nodes[candidate.node_name].get("weight", 0.0) or 0.0)
            _add_node(
                registry=nodes,
                name=candidate.node_name,
                description=graph.nodes[candidate.node_name].get("description"),
                tier="secondary",
                weight=node_weight,
                colour=branch_colour,
            )
            _add_edge(
                registry=edges,
                subject=parent,
                predicate=candidate.predicate,
                object_name=candidate.node_name,
                description=candidate.description,
                weight=candidate.weight,
                level=2,
            )
    if timings is not None:
        timings["stage2_ms"] = round((perf_counter() - stage2_start) * 1000, 2)

    # Stage 3: pick up to 4 unbranched secondary nodes, add 2-4 periphery nodes.
    stage3_start = perf_counter()
    all_secondary = [name for name, node in nodes.items() if node.tier == "secondary"]
    unbranched_secondary = [name for name in all_secondary if name not in branched]
    final_parents = sorted(
        unbranched_secondary, key=lambda name: scored_secondary.get(name, 0.0), reverse=True
    )[: config.final_parents_max]
    for parent in final_parents:
        blocked = selected_names.copy()
        candidates = _collect_neighbours(graph, parent, query, blocked, central_nodes)
        target = _target_count(
            len(candidates),
            minimum=config.final_periphery_min,
            maximum=config.final_periphery_max,
        )
        if target == 0:
            continue
        selected = _pick_diverse(candidates, target, selected_names, parent_name=parent)
        branch_root = branch_by_node.get(parent, parent)
        branch_colour = colour_by_branch.get(branch_root, "#3B82F6")
        for candidate in selected:
            branch_by_node[candidate.node_name] = branch_root
            node_weight = float(graph.nodes[candidate.node_name].get("weight", 0.0) or 0.0)
            _add_node(
                registry=nodes,
                name=candidate.node_name,
                description=graph.nodes[candidate.node_name].get("description"),
                tier="periphery",
                weight=node_weight,
                colour=branch_colour,
            )
            _add_edge(
                registry=edges,
                subject=parent,
                predicate=candidate.predicate,
                object_name=candidate.node_name,
                description=candidate.description,
                weight=candidate.weight,
                level=3,
            )
    if timings is not None:
        timings["stage3_ms"] = round((perf_counter() - stage3_start) * 1000, 2)
        timings["total_ms"] = round((perf_counter() - total_start) * 1000, 2)

    tier_order = {"central": 0, "secondary": 1, "periphery": 2}
    ordered_edges = sorted(edges.values(), key=lambda item: (item.level, item.subject, item.object))
    return GraphV2(
        nodes=sorted(nodes.values(), key=lambda item: (tier_order[item.tier], item.name)),
        edges=[
            EdgeV2(
                subject=edge.subject,
                predicate=edge.predicate,
                object=edge.object,
                description=edge.description,
                weight=edge.weight,
            )
            for edge in ordered_edges
        ],
    )
