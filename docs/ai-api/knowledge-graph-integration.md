# Knowledge Graph Integration

## Summary

The API loads graph data from LanceDB tables into a NetworkX directed graph during FastAPI application startup. Graph endpoints return either the legacy V1 graph schema or the staged V2 schema. `/model` also streams graph data in parallel with answer text.

## Graph data loading

Confirmed startup flow:

- FastAPI lifespan in `main.py` opens LanceDB using `database.get_connection()`.
- A `database.Client` is created.
- `client.get_knowledge_graph()` loads graph data into memory.
- The graph is stored in request state as `request.state.graph`.

Confirmed LanceDB tables:

- `nodes`
- `edges`

Confirmed `get_knowledge_graph()` behavior:

- Opens `nodes` table.
- Selects `name`, `description`, and `weight`.
- Adds nodes to a `networkx.DiGraph`.
- Opens `edges` table.
- Selects `subject`, `object`, `predicate`, `description`, and `weight`.
- If `level` exists in the edge schema, it is also selected.
- Adds edges to the directed graph.
- If `nodes` table is absent, returns an empty directed graph.
- If `edges` table is absent, returns graph with nodes only.

Notebook source path:

- `main.ipynb` reads node data from `abfs://datasets/nodes-v25-09-25.parquet`.
- `main.ipynb` reads edge data from `abfs://datasets/edges-v25-09-25.parquet`.
- Node names are embedded using Azure OpenAI embeddings.
- Nodes and edges are written to LanceDB tables `nodes` and `edges`.

Unknown / requires confirmation:

- The upstream method used to extract nodes and edges from publications.
- Whether graph tables in production match the notebook parquet versions.
- Whether graph extraction had human review.

## Entity and relation structure inferred from code

V1 node model:

- `name`: concept/entity name.
- `description`: optional concept description.
- `neighbourhood`: hop-distance tier from central node.
- `weight`: node relevance or importance.
- `colour`: visualization color.
- `vector`: node embedding excluded from API response.

V1 edge model:

- `subject`: source node name.
- `predicate`: relation label.
- `object`: target node name.
- `description`: relation description.
- `weight`: edge importance.
- `level`: hop distance from central node in V1 response.

V2 node model:

- `name`
- `description`
- `tier`: `central`, `secondary`, or `periphery`
- `weight`
- `colour`

V2 edge model:

- `subject`
- `predicate`
- `object`
- `description`
- `weight`

In notebook cell for edges, `predicate` is inserted with constant value `relates_to` if not already present. Current `get_knowledge_graph()` expects a `predicate` column in `edges`.

## Graph endpoint behavior

### `/nodes`

Searches the in-memory graph by substring match against node names. Does not call LanceDB directly in current endpoint implementation. Returns sorted nodes up to `limit`.

### `/nodes/{name}`

Performs case-insensitive exact node-name lookup in the in-memory graph. Returns 404 if not found.

### `/graph`

Uses V1 graph assembly:

- If called directly, query parameter `query` is used.
- `Client.find_subgraph()` resolves central node(s) by lexical match first, then vector search over `nodes` table.
- Extracts neighborhood nodes using utility functions.
- Prunes to closest nodes and pruned edges.
- Returns `Graph` with V1 `neighbourhood` and edge `level` fields.

### `/graph/v2`

Uses staged V2 graph assembly:

- Finds 1 to 3 central nodes.
- Fast path exact node-name match.
- Lexical central-node selection.
- Vector central-node fallback over LanceDB `nodes` table.
- Stage 1 selects secondary nodes for each central node.
- Stage 2 expands top secondary branchers.
- Stage 3 adds periphery nodes.
- Uses semantic nearest-neighbor fallback if branch expansion is too sparse.
- Adds timing headers:
  - `X-KG-Timing`
  - `Server-Timing`

## How graph output relates to AI responses

`/model` produces graph and answer concurrently:

- `produce_graph()` creates a dedicated LanceDB connection to avoid contention.
- For `graph_version=v1`, it first calls `genai.extract_entities(user_query)` then `graph_client.find_subgraph()`.
- For `graph_version=v2`, it directly calls `kg_v2.build_subgraph_v2()`.
- The graph response is streamed as an `AssistantResponse` chunk with empty `content` and populated `graph`.
- Text answer generation does not depend on graph generation finishing successfully.
- If graph generation times out or fails, `/model` streams an empty graph fallback.

Known behavior:

- The graph is visual context, not necessarily citation evidence.
- RAG documents and graph nodes are separate data products.
- The frontend must handle graph chunks arriving before, during, or after text chunks.

## Graph visualization API dependencies

Confirmed frontend dependencies from templates:

- The KG tester renders graph data using client-side D3 logic in `frontend/templates/kg_tester.html` and `templates/kg_tester.html`.
- The tester can display graph returned inside `/model` NDJSON stream.
- Legacy template can call `/graph` or `/graph/v2` directly.

Confirmed API response dependencies:

- V1 graph uses `nodes[].neighbourhood` and `edges[].level`.
- V2 graph uses `nodes[].tier`; V2 edges do not include `level` in the API model.

## Known limitations

- Graph provenance is not exposed in API responses.
- Graph extraction process is not implemented in this repository; only notebook loading from prebuilt parquet is visible.
- Graph data is loaded into memory at startup; runtime updates in LanceDB require app restart or new loading logic.
- V2 semantic fallback depends on Azure OpenAI embeddings and LanceDB vector search.
- If `nodes` or `edges` tables are missing, graph responses may be empty or node-only.
- Node and edge quality depends on upstream extraction, which is Unknown / requires confirmation.
