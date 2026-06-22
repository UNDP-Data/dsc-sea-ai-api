# AI Data and Storage

## Runtime storage

Confirmed runtime storage backend:

- LanceDB connected at `az://lancedb`.
- Storage options are derived from Azure Blob environment variables.

Credential options in `src/database.py`:

- Preferred full SAS URL:
  - `STORAGE_SAS_URL`
- Or account plus credential:
  - `STORAGE_ACCOUNT_NAME`
  - `STORAGE_ACCOUNT_KEY`
- Or account plus SAS token:
  - `STORAGE_ACCOUNT_NAME`
  - `STORAGE_SAS_TOKEN`

If storage credentials are absent, `get_storage_options()` raises a runtime error.

## LanceDB tables

Tables referenced by the API:

- `chunks`: publication chunk content and vectors.
- `documents`: canonical document metadata.
- `sources`: source registry metadata.
- `nodes`: knowledge graph node records and vectors.
- `edges`: knowledge graph edge records.
- `sdg7`: structured SDG7 indicator data loaded from UNSD API in notebook.

`/debug/tables` reports existence and row counts for these tables.

## Input corpus locations

Confirmed from `main.ipynb`:

- Publication corpus parquet:
  - `abfs://datasets/corpus-v25-06-27.parquet`
- Graph nodes parquet:
  - `abfs://datasets/nodes-v25-09-25.parquet`
- Graph edges parquet:
  - `abfs://datasets/edges-v25-09-25.parquet`
- UNSD SDG API:
  - `https://unstats.un.org/SDGAPI/v1/sdg/GeoArea/List`
  - `https://unstats.un.org/SDGAPI/v1/sdg/Goal/DataCSV`

Confirmed local data files:

- `data/corpus/sources.yaml`: source registry seed/reference.
- `data/corpus/sample_manifest.yaml`: sample manual corpus manifest.
- `data/retrieval_benchmark/*`: expert annotation templates, batch plan, launch email, and corpus inventory export.

Unknown / requires confirmation:

- Whether the Azure Blob `datasets` location is the authoritative source of truth.
- Whether raw PDFs or HTML snapshots are stored anywhere.
- Whether `data/chunks-{VERSION}.parquet`, `data/nodes-{VERSION}.parquet`, or `data/edges-{VERSION}.parquet` outputs are committed or archived in deployment workflows.

## Corpus source registry

Source definitions in `src/corpus.py` and `data/corpus/sources.yaml` include:

- `tracking_sdg7`: Tracking SDG7 / ESMAP.
- `undp`: UNDP Publications.
- `world_bank`: World Bank Publications.
- `iea`: International Energy Agency.
- `seforall`: Sustainable Energy for All.
- `irena`: IRENA.
- `unep`: UNEP.
- `manual_external`: fallback for manual/external documents.

Observed current bootstrap output from prior local run showed:

- `chunks`: 82,705 rows.
- `documents`: 243 rows.
- `sources`: 2 rows.

These counts are environment-dependent and should be verified with `/debug/tables` or `scripts/validate_corpus_tables.py`.

## Cache files and in-memory caches

Confirmed runtime cache:

- `_DOCUMENT_INDEX_CACHE` in `src/database.py` caches approved document index rows per connection id.
- TTL environment variable:
  - `DOCUMENT_INDEX_CACHE_TTL_SECONDS`
- Default TTL:
  - 300 seconds.

Python/test caches in repo:

- `__pycache__/`
- `.pytest_cache/`

Operational/debug output files:

- `tmp/model_stream_debug.txt`
- `tmp/retrieval_surface*.json`
- `tmp/retrieval_surface*.txt`

These are local debugging artifacts, not confirmed production cache files.

## Logs

The service uses Python logging via module loggers.

Confirmed logs include:

- model request start/finish;
- blocked model request category/reason/query;
- graph build timeout/failure;
- publication retrieval scheduled/started/completed/timed out/failed;
- stream idle timeout and watchdog timeout;
- retrieval branch diagnostics and warnings;
- Moonshot provider errors mapped to HTTP errors.

Important privacy note:

- Logs include raw user query strings for `/model` in several log messages.
- No database-backed conversation logging is implemented in this repository.
- Unknown / requires confirmation: production log retention, access controls, redaction policy, and monitoring backend.

## Environment variables

Core API:

- `API_KEY`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_MODEL`
- `AZURE_OPENAI_EMBED_MODEL`
- `AZURE_OPENAI_TIMEOUT`
- `STORAGE_SAS_URL`
- `STORAGE_ACCOUNT_NAME`
- `STORAGE_ACCOUNT_KEY`
- `STORAGE_SAS_TOKEN`

Model and retrieval timeouts:

- `MODEL_SCOPE_GUARD_ENABLED`
- `MODEL_GRAPH_TIMEOUT_SECONDS`
- `MODEL_STREAM_IDLE_TIMEOUT_SECONDS`
- `MODEL_TOOLS_PREP_TIMEOUT_SECONDS`
- `MODEL_STREAM_WATCHDOG_SECONDS`
- `MODEL_PUBLICATION_RETRIEVAL_TIMEOUT_SECONDS`
- `MODEL_PUBLICATION_HEARTBEAT_SECONDS`
- `RETRIEVE_CHUNKS_TIMEOUT_SECONDS`
- `RETRIEVE_CHUNKS_LEXICAL_TIMEOUT_SECONDS`
- `RETRIEVE_DOCUMENT_INDEX_TIMEOUT_SECONDS`
- `RETRIEVE_CHUNKS_VARIANT_TIMEOUT_SECONDS`
- `RETRIEVE_CURRENT_DATA_ENRICHMENT_TIMEOUT_SECONDS`
- `GRAPH_VECTOR_TIMEOUT_SECONDS`
- `DOCUMENT_INDEX_CACHE_TTL_SECONDS`

KG tester:

- `KG_TESTER_API_KEY`
- `KG_TESTER_LOCAL_API_BASE_URL`
- `KG_TESTER_REMOTE_API_BASE_URL`

Moonshot router:

- `AZURE_OPENAI_MOONSHOT_PARSE_DEPLOYMENT`
- `AZURE_OPENAI_PARSE_DEPLOYMENT`
- `AZURE_OPENAI_MOONSHOT_SYNOPSIS_DEPLOYMENT`
- `AZURE_OPENAI_SYNOPSIS_DEPLOYMENT`
- `AZURE_OPENAI_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_PARSE_MODEL`
- `OPENAI_SYNOPSIS_MODEL`
- `ALLOWED_ORIGINS`
- `MOONSHOT_RATE_LIMIT_WINDOW_SECONDS`
- `MOONSHOT_PARSE_RATE_LIMIT`
- `MOONSHOT_SYNOPSIS_RATE_LIMIT`

## Storage limitations and risks

- Runtime requires Azure Blob-backed LanceDB. Local fallback storage is not implemented.
- Full source-corpus ingestion from raw documents is not implemented in app code.
- Notebook uses `connection.create_table("chunks", data=df_chunks)` without visible `mode="overwrite"` in the publication section; behavior when a table already exists requires confirmation.
- LanceDB/Rust Azure storage errors can appear under storage/network issues. Debugging may require checking Azure Blob access, SAS expiry, account key, and LanceDB table schemas.
