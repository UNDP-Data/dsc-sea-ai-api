# AI API Catalog

## Scope

This catalog covers endpoints exposed by the main FastAPI application in `main.py`, the included Moonshot router in `src/moonshot.py`, and the standalone local tester app in `frontend/kg_tester_app.py`.

Authentication notes:

- Core API endpoints that declare `Depends(authenticate)` require `X-Api-Key`.
- FastAPI returns `403` when the `X-Api-Key` header is missing, because `APIKeyHeader` uses `auto_error=True`.
- `src/security.py` returns `401` for an invalid key and `500` if server-side `API_KEY` is not configured.
- Moonshot endpoints use origin checks and rate limits, not `X-Api-Key`, except where otherwise noted.
- Static/template endpoints are unauthenticated.

## Main API endpoints

### `GET /`

- Purpose: Serve the repository documentation page from `README.md` rendered as HTML.
- Request schema: None.
- Response schema: HTML.
- Auth requirements: None.
- Internal functions/services: `readme()` and `markdown` rendering in `main.py`.
- Error responses: Unknown / requires confirmation. File read or markdown rendering errors are not explicitly handled.
- Frontend consumers: Browser users; developer documentation.

### `GET /changelog`

- Purpose: Serve changelog page from `CHANGELOG.md` rendered as HTML.
- Request schema: None.
- Response schema: HTML.
- Auth requirements: None.
- Internal functions/services: `changelog()` and `markdown` rendering in `main.py`.
- Error responses: Unknown / requires confirmation. File read or markdown rendering errors are not explicitly handled.
- Frontend consumers: Browser users; developer documentation.

### `GET /kg-tester`

- Purpose: Serve the legacy KG tester template from `templates/kg_tester.html`.
- Request schema: None.
- Response schema: HTML.
- Auth requirements: None.
- Internal functions/services: Jinja2 templates in `main.py`.
- Error responses: Template rendering errors are not explicitly handled.
- Frontend consumers: Local/manual graph testing.

### `GET /favicon.ico`

- Purpose: Return empty favicon response to avoid browser 404 noise.
- Request schema: None.
- Response schema: Empty response.
- Auth requirements: None.
- Internal functions/services: `favicon()` in `main.py`.
- Error responses: None expected.
- Frontend consumers: Browsers.

### `GET /static/{path}`

- Purpose: Serve files from local `static/` directory.
- Request schema: Path parameter handled by `StaticFiles`.
- Response schema: Static file response.
- Auth requirements: None.
- Internal functions/services: FastAPI `StaticFiles` mounted at `/static`.
- Error responses: Static file 404 if asset is missing.
- Frontend consumers: Static assets for served pages.

### `GET /nodes`

- Purpose: Search graph nodes by exact or semantic matching.
- Request schema:
  - Query parameters from `SearchParameters`:
    - `query`: string, minimum length 2.
    - `limit`: integer, default 10, min 1, max 100.
    - `method`: `exact` or `vector`, default `vector`.
- Response schema: `list[Node]`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.search_nodes()`.
  - `request.state.client.search_nodes()`.
  - LanceDB `nodes` table.
  - Azure OpenAI embeddings for vector mode.
- Error responses:
  - `403` missing API key.
  - `401` invalid API key.
  - `500` missing server API key.
  - Validation errors for invalid query params.
- Frontend consumers: Knowledge Graph search UI, if connected; legacy tester may use graph endpoints rather than this endpoint directly.

### `GET /nodes/{name}`

- Purpose: Fetch a single graph node by name.
- Request schema:
  - Path parameter `name`: string.
- Response schema: `Node`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.get_node()`.
  - `request.state.client.get_node()`.
  - LanceDB `nodes` table.
- Error responses:
  - `404` if node is not found.
  - Auth errors as above.
- Frontend consumers: Knowledge Graph node detail UI, if connected.

### `GET /documents`

- Purpose: Search canonical publication records.
- Request schema:
  - Query parameters from `DocumentSearchParameters`:
    - `query`: optional string, min length 2.
    - `limit`: integer, default 10, min 1, max 100.
    - `year`: optional integer.
    - `source`: optional string.
    - `topic`: optional string.
- Response schema: `list[Document]`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.search_documents()`.
  - `request.state.client.search_documents()`.
  - LanceDB `documents` table.
- Error responses:
  - Auth errors as above.
  - Validation errors for invalid query params.
  - Unknown / requires confirmation for missing `documents` table behavior.
- Frontend consumers: Unknown / requires confirmation. Could support reference browsing or debugging.

### `GET /sources`

- Purpose: List source registry rows from the corpus metadata layer.
- Request schema: None.
- Response schema: `list[SourceRecord]`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.list_sources()`.
  - `request.state.client.list_sources()`.
  - LanceDB `sources` table.
- Error responses:
  - Auth errors as above.
  - Unknown / requires confirmation for missing `sources` table behavior.
- Frontend consumers: Unknown / requires confirmation.

### `GET /graph`

- Purpose: Return a v1 subgraph around entities relevant to a query.
- Request schema:
  - Query parameters from `GraphParameters`:
    - `query`: string, min length 2.
    - `hops`: integer, default 2, min 0, max 3.
- Response schema: `Graph`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.query_knowledge_graph()`.
  - `genai.extract_entities()`.
  - `request.state.client.get_knowledge_graph()` loaded during app lifespan.
  - `KnowledgeGraph.find_subgraph()`.
  - NetworkX graph traversal.
- Error responses:
  - Auth errors as above.
  - Validation errors for invalid params.
  - Unknown / requires confirmation for graph-load failures.
- Frontend consumers: Legacy `templates/kg_tester.html` can call this endpoint.

### `GET /graph/v2`

- Purpose: Return a v2 graph optimized for frontend visualization using central, secondary, and periphery tiers.
- Request schema:
  - Query parameters from `GraphV2Parameters`:
    - `query`: string, min length 2.
    - `max_central_nodes`: integer, default 3, min 1, max 10.
    - `max_secondary_nodes`: integer, default 12, min 0, max 30.
    - `max_periphery_nodes`: integer, default 24, min 0, max 60.
    - `min_edge_weight`: optional float, min 0.
    - `include_periphery`: boolean, default true.
    - `search_method`: `hybrid`, `semantic`, or `exact`, default `hybrid`.
- Response schema: `GraphV2`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.query_knowledge_graph_v2()`.
  - `kg_v2.build_subgraph_v2()`.
  - LanceDB `nodes` and `edges` tables through `Client`.
  - Azure OpenAI embeddings for semantic/hybrid matching.
- Error responses:
  - Auth errors as above.
  - Validation errors for invalid params.
- Frontend consumers: Legacy `templates/kg_tester.html`; likely Knowledge Graph experience.

### `POST /model`

- Purpose: Stream the main assistant response, graph payloads, document references, and suggested follow-up ideas.
- Request schema:
  - JSON array of `Message` objects.
  - Each message has:
    - `role`: `human` or `assistant`.
    - `content`: string, max length 16,384.
  - Last message must have role `human`.
  - Optional request header `X-Graph-Version`: `v1`, `v2`, or `default`.
  - Optional request header `X-Request-Id`; generated if absent.
- Response schema:
  - NDJSON stream of `AssistantResponse` chunks.
  - Chunk fields can include:
    - `role`: usually `assistant`.
    - `content`: streamed answer text.
    - `ideas`: follow-up suggestions on final/blocked chunks.
    - `documents`: frontend-compatible reference objects.
    - `graph`: `Graph` or `GraphV2` object.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.ask_model()`.
  - `genai.assess_scope()`.
  - `genai.get_answer()`.
  - `genai.extract_entities()` for graph v1.
  - `kg_v2.build_subgraph_v2()` for graph v2.
  - `Client.retrieve_chunks()` for RAG retrieval.
  - Azure OpenAI chat model.
  - Azure OpenAI embeddings for retrieval and graph search.
  - LanceDB `chunks`, `documents`, `sources`, `nodes`, and `edges` tables.
- Error responses:
  - `400` if message array is empty.
  - `400` if last message role is not `human`.
  - Auth errors as above.
  - Streamed fallback chunks on graph timeout or answer timeout.
  - Unhandled internal exceptions may produce `500`.
- Frontend consumers:
  - `frontend/kg_tester_app.py` proxies to this endpoint.
  - Energy AI / SEA frontend likely consumes this endpoint; exact frontend repository unknown / requires confirmation.

### `GET /debug/tables`

- Purpose: Inspect LanceDB table presence/counts for operational debugging.
- Request schema: None.
- Response schema: JSON object with table status/count fields.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.debug_tables()`.
  - LanceDB table open/count operations.
- Error responses:
  - Auth errors as above.
  - Table errors are returned per table where handled.
- Frontend consumers: Developer/operator only.
- OpenAPI visibility: `include_in_schema=False`.

### `GET /debug/retrieve`

- Purpose: Run retrieval for one query and return selected documents, selected chunks, and debug diagnostics.
- Request schema:
  - Query parameter `query`: string, min length 2.
  - Query parameter `limit`: integer, default 20, min 1, max 50.
- Response schema:
  - JSON object with `query`, `limit`, `document_count`, `chunk_count`, `documents`, `chunks`, and `debug`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services:
  - `main.debug_retrieve()`.
  - `Client.retrieve_chunks(debug={})`.
- Error responses:
  - Auth errors as above.
  - Validation errors for invalid params.
  - Retrieval exceptions can propagate as `500`.
- Frontend consumers: Developer/operator only.
- OpenAPI visibility: `include_in_schema=False`.

## Moonshot endpoints

### `GET /api/moonshot/health`

- Purpose: Report whether Moonshot model provider configuration is available.
- Request schema: None.
- Response schema: `MoonshotHealthResponse`:
  - `ok`: boolean.
  - `configured`: boolean.
  - `provider`: optional string.
  - `parseModel`: optional string.
  - `synopsisModel`: optional string.
- Auth requirements: None.
- Internal functions/services: `get_settings()` in `src/moonshot.py`.
- Error responses: Unknown / requires confirmation.
- Frontend consumers: Moonshot dashboard or health checks.

### `POST /api/moonshot/prodoc`

- Purpose: Resolve a project Prodoc PDF URL from project id/title metadata.
- Request schema: `ProdocResolveRequest`:
  - `projectId`: string, max 64.
  - `title`: optional string, max 500.
  - `verticalFunded`: optional boolean.
- Response schema: `ProdocResolveResponse`:
  - `url`: optional string.
  - `blobName`: optional string.
  - `matches`: list.
- Auth requirements:
  - No `X-Api-Key`.
  - Origin check through `ALLOWED_ORIGINS` when configured; loopback origins allowed.
- Internal functions/services:
  - `resolve_prodoc_blob()`.
  - Azure Blob public/container listing and matching.
- Error responses:
  - `400` invalid project id.
  - `403` disallowed origin.
  - Upstream/listing failures can map to gateway errors.
- Frontend consumers: Moonshot frontend.

### `GET /api/moonshot/prodoc/download`

- Purpose: Download a resolved Prodoc PDF by `projectId`.
- Request schema:
  - Query parameter `projectId`: string.
  - Optional `title`: string.
  - Optional `verticalFunded`: boolean.
- Response schema: PDF response.
- Auth requirements:
  - No `X-Api-Key`.
  - Origin check.
- Internal functions/services:
  - `resolve_prodoc_blob()`.
  - `httpx` download from Prodoc container.
- Error responses:
  - `400` invalid project id.
  - `404` if no Prodoc is found.
  - `502` if blob fetch fails.
  - `403` disallowed origin.
- Frontend consumers: Moonshot frontend.

### `GET /api/moonshot/prodoc/download-url`

- Purpose: Proxy-download a Prodoc PDF from a validated Prodoc container URL.
- Request schema:
  - Query parameter `url`: string.
- Response schema: PDF response.
- Auth requirements:
  - No `X-Api-Key`.
  - Origin check.
- Internal functions/services:
  - URL validation in `src/moonshot.py`.
  - `httpx` download.
- Error responses:
  - `400` if URL is outside allowed Prodoc prefix or not a PDF.
  - `502` if fetch fails.
  - `403` disallowed origin.
- Frontend consumers: Moonshot frontend.

### `POST /api/moonshot/parse-query`

- Purpose: Convert natural-language dashboard query text into structured filter selections.
- Request schema: `ParseQueryRequest`:
  - `query`: string, max 500.
  - `locale`: optional string, max 32.
  - `filterCatalog`: dictionary.
- Response schema: `ParseQueryResponse`:
  - `filters`: dictionary.
  - `unresolvedTerms`: list of strings.
- Auth requirements:
  - No `X-Api-Key`.
  - Origin check.
  - Per-origin/IP rate limit.
- Internal functions/services:
  - `require_openai_client()`.
  - Azure OpenAI or OpenAI chat completions with JSON schema.
  - Deterministic filter overrides.
- Error responses:
  - `403` disallowed origin.
  - `429` rate limit exceeded.
  - `503` if model provider unavailable or auth/upstream provider failures occur.
  - `502` invalid model output.
  - `500` unexpected internal errors.
- Frontend consumers: Moonshot dashboard natural-language filter UI.

### `POST /api/moonshot/project-synopsis`

- Purpose: Generate a concise synopsis from supplied project dashboard context.
- Request schema: `ProjectSynopsisRequest`:
  - `query`: optional string, max 500.
  - `locale`: optional string.
  - `filters`: dictionary.
  - `summaryMetrics`: dictionary.
  - `projectContext`: dictionary.
- Response schema: `ProjectSynopsisResponse`:
  - `synopsis`: string.
- Auth requirements:
  - No `X-Api-Key`.
  - Origin check.
  - Per-origin/IP rate limit.
- Internal functions/services:
  - `require_openai_client()`.
  - Azure OpenAI or OpenAI chat completions.
  - Supplied dashboard context only.
- Error responses:
  - `403` disallowed origin.
  - `429` rate limit exceeded.
  - `503` model provider unavailable/upstream auth errors.
  - `502` invalid model output.
  - `500` unexpected internal errors.
- Frontend consumers: Moonshot dashboard.

## FastAPI documentation endpoints

### `GET /openapi.json`

- Purpose: FastAPI OpenAPI schema.
- Auth requirements: None.
- OpenAPI visibility: internal FastAPI route, not included in generated schema.

### `GET /docs`

- Purpose: Swagger UI.
- Auth requirements: None.

### `GET /docs/oauth2-redirect`

- Purpose: Swagger UI OAuth redirect helper.
- Auth requirements: None.

### `GET /redoc`

- Purpose: ReDoc UI.
- Auth requirements: None.

## Standalone local tester app endpoints

These endpoints exist in `frontend/kg_tester_app.py`, not in the main app unless the standalone tester is launched with `make run-tester`.

### `GET /`

- Purpose: Redirect to `/kg-tester`.
- Auth requirements: Loopback-only app access; no API key for page load.

### `GET /kg-tester`

- Purpose: Serve `frontend/templates/kg_tester.html`.
- Auth requirements: Loopback-only app access.

### `POST /kg-tester/api/model`

- Purpose: Proxy a browser tester request to the configured local or remote API `/model` endpoint.
- Request schema: JSON body from tester frontend containing model messages and target options.
- Response schema: NDJSON stream proxied from upstream `/model`.
- Auth requirements:
  - Tester app enforces loopback access.
  - Proxy sends `X-Api-Key` using `KG_TESTER_API_KEY` or `API_KEY` from environment.
- Internal functions/services:
  - `frontend.kg_tester_app.proxy_model()`.
  - `httpx.AsyncClient.stream()`.
- Error responses:
  - `403` if non-loopback client accesses tester.
  - `404` or upstream errors if configured API base URL is wrong.
  - Gateway errors for upstream failures.
- Frontend consumers: Local KG tester browser page.

## Multi-assistant RAG endpoints

These endpoints expose the reusable profile-based RAG system. The legacy SEA endpoints remain available and internally map to the `sea` assistant profile.

### `GET /assistants`

- Purpose: List configured RAG assistant profiles.
- Request schema: None.
- Response schema: JSON array of assistant metadata objects with `assistant_id`, `display_name`, `domain_description`, `tables`, and `default`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services: `list_profiles()` from `src.rag_system`.
- Error responses: Auth errors as above.
- Frontend consumers: Future platform assistant picker or operational diagnostics.

### `POST /assistants/{assistant_id}/model`

- Purpose: Stream a profile-specific publication-backed assistant response.
- Request schema: Same as `POST /model`: JSON array of `Message` objects.
- Response schema: Same NDJSON `AssistantResponse` chunks as `/model`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services: profile loader, profile-specific scope guard, `Client.retrieve_chunks()` using profile table namespace, `genai.get_answer()` with profile prompts.
- Error responses: `404` unknown assistant profile, `400` invalid messages, auth errors as above.
- Frontend consumers: Future topic-specific RAG assistants. In v1, non-SEA assistants do not emit Knowledge Graph chunks.

### `GET /assistants/{assistant_id}/documents`

- Purpose: Search canonical document metadata for a specific assistant profile.
- Request schema: `pattern` and `limit` query parameters, same behavior as `/documents`.
- Response schema: `list[Document]`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services: profile loader, profile-specific LanceDB `documents` table.
- Error responses: `404` unknown assistant profile, auth errors as above.
- Frontend consumers: Future topic-specific document/reference browsing.

### `GET /assistants/{assistant_id}/sources`

- Purpose: List source registry rows for a specific assistant profile.
- Request schema: None.
- Response schema: `list[SourceRecord]`.
- Auth requirements: `X-Api-Key`.
- Internal functions/services: profile loader, profile-specific LanceDB `sources` table.
- Error responses: `404` unknown assistant profile, auth errors as above.
- Frontend consumers: Future topic-specific source browsing or diagnostics.

### `GET /assistants/{assistant_id}/debug/retrieve`

- Purpose: Inspect profile-specific document/chunk retrieval decisions for a query.
- Request schema: `query` and `limit` query parameters.
- Response schema: Debug JSON with `assistant_id`, selected documents, selected chunks, and retrieval diagnostics.
- Auth requirements: `X-Api-Key`.
- Internal functions/services: profile loader, profile-specific `Client.retrieve_chunks()`.
- Error responses: `404` unknown assistant profile, auth errors as above.
- Frontend consumers: Developer/operator only.
