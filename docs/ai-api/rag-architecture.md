# RAG Architecture

## Summary

The RAG implementation is centered on a LanceDB-backed publication corpus. `/model` streams a draft or publication-check notice, retrieves publication chunks and document metadata, streams `documents[]` separately, then streams a publication-grounded continuation using the retrieved excerpts.

The current code supports both:

- legacy chunk-first retrieval from the `chunks` table;
- newer document-first retrieval when a `documents` table exists.

## Corpus ingestion assumptions

Confirmed from `main.ipynb`:

- The original publication corpus is read from an Azure Blob parquet file:
  - `abfs://datasets/corpus-v25-06-27.parquet`
- The source parquet is expected to contain one row per document with fields shown in notebook output:
  - `title`
  - `year`
  - `language`
  - `url`
  - `summary`
  - `text`
- Notebook comments say publications are relevant UNDP Publications and SDG 7 reports.
- The upstream process that created `corpus-v25-06-27.parquet` is not implemented in this repository.

Confirmed from scripts:

- `scripts/bootstrap_corpus_tables.py` builds `sources` and `documents` from the existing `chunks` table.
- `scripts/import_corpus_manifest.py` imports manually prepared YAML source/document/chunk records.
- `data/corpus/sample_manifest.yaml` is an example manual import manifest.
- `scripts/validate_corpus_tables.py` validates table existence and metadata completeness.
- `scripts/export_documents_inventory.py` exports the canonical `documents` table to CSV.

Unknown / requires confirmation:

- Which PDF extraction tool was used to create the source corpus parquet.
- Whether source PDFs/HTML snapshots are archived anywhere.
- Whether corpus licensing/robots checks were performed before inclusion.
- Whether manual editorial approval existed for all source documents.

## Document preprocessing

Confirmed from `main.ipynb`:

1. Read corpus parquet from Azure Blob.
2. Normalize whitespace in each document text using `str.replace(r"\s+", " ", regex=True)`.
3. Preserve non-text metadata as chunk metadata.
4. Split each document twice using `TokenTextSplitter`:
   - 768-token chunks with 25 percent overlap.
   - 256-token chunks with 25 percent overlap.
5. Build `df_chunks` with metadata plus `content`.
6. Sort by `year` descending and `title` ascending.
7. Drop duplicate chunk content.
8. Filter out chunks where more than 25 percent of whitespace-separated content appears numeric.
9. Generate embeddings in batches of 128.
10. Write local parquet `data/chunks-{VERSION}.parquet`.
11. Create LanceDB table `chunks` from `df_chunks`.

Current document-centric metadata enrichment in `src/corpus.py` additionally infers:

- source/publisher from URL domain;
- document type;
- series name/series id;
- topic tags;
- SDG tags;
- sector tags;
- audience tags;
- region/country tags;
- quality score;
- content hash;
- stable document id;
- stable chunk id if chunks are rewritten with provenance.

## Chunking strategy

Original notebook strategy:

- Large chunks: 768 tokens, 25 percent overlap.
- Smaller chunks: 256 tokens, 25 percent overlap.
- Both sets are combined.
- Duplicate content is removed.
- Numeric-heavy chunks are filtered.

Current manifest import strategy:

- If explicit chunks are provided in YAML, they are used.
- If no explicit chunks are provided, one chunk is generated from document `content` or `summary`.
- Enriched chunks receive `document_id`, `chunk_id`, `chunk_index`, `content_type`, `section_title`, page fields, token count, and chunk summary.

Unknown / requires confirmation:

- Whether production `chunks` has been rewritten with document provenance. `validate_corpus_tables.py` reports this.
- Whether page-level provenance exists in the original corpus.

## Embeddings

Confirmed embedding provider:

- `AzureOpenAIEmbeddings` from `langchain_openai`.
- Model from `AZURE_OPENAI_EMBED_MODEL`.
- Endpoint from `AZURE_OPENAI_ENDPOINT`.
- API key from `AZURE_OPENAI_KEY`.
- Embedding dimensions set to `1024` in `src/genai.py`.

Confirmed embedding uses:

- Publication chunk embedding during notebook ingestion.
- Query embedding during vector chunk retrieval.
- Node embedding in notebook and graph V2 semantic fallback.

Unknown / requires confirmation:

- Exact embedding model deployed in production.
- Whether old corpus embeddings were generated with the same current `AZURE_OPENAI_EMBED_MODEL` and 1024 dimensions.

## Vector store

Confirmed vector/data store:

- LanceDB `0.25.0`.
- Asynchronous connection to `az://lancedb`.
- Azure Blob credentials passed through LanceDB storage options.

Confirmed tables referenced by code:

- `chunks`
- `documents`
- `sources`
- `nodes`
- `edges`
- `sdg7`

## Retrieval flow

High-level flow in `Client.retrieve_chunks`:

1. Open `chunks` table.
2. Detect chunk schema fields.
3. Optionally open `documents` table.
4. Build `RetrievalProfile`:
   - normalized query;
   - tokens/phrases;
   - explicit years;
   - recency preference;
   - query intent;
   - current-data flag;
   - data focus;
   - country/region scopes;
   - geographic fallback regions.
5. Build prioritized retrieval queries.
6. If `documents` table exists:
   - load approved document index rows using cached `get_document_index_rows`;
   - score document rows;
   - enforce geography filtering;
   - seed latest 2025 Tracking SDG7 report for current SDG7 access data queries;
   - shortlist documents;
   - retrieve chunks within shortlisted documents using lexical and vector search;
   - select chunk rows;
   - optionally add curated fallback chunks for high-value SDG7 current-data metrics.
7. If document-first path fails:
   - run lexical search over `chunks`;
   - then vector search over `chunks`;
   - apply geography filter;
   - select chunks/documents.
8. Return `(list[Chunk], list[Document])`.

Geography handling:

- Query country/region aliases are extracted from user query.
- Country queries allow exact country and global; if no exact country exists, parent region and global are allowed.
- Region queries allow exact region and global.
- Sibling countries or unrelated regions are rejected.

Current-data handling:

- Queries such as `how many people lack access to energy` are treated as current-data queries.
- Latest Tracking SDG7 report gets a strong boost/seed.
- A trusted fallback metric chunk can provide the 2025 SDG7 electricity-access figure: 666 million people without electricity in 2023.
- Broader SDG7 context fallback chunks exist for clean cooking and 2030 progress context.

## Prompt templates

Prompts are stored in `src/prompts.yaml`:

- `extract_entities`: extracts KG entities from user query.
- `answer_question`: ReAct agent prompt for answering with tools.
- `draft_answer`: concise initial answer without citations or URLs.
- `answer_with_publications`: continuation prompt using retrieved publication excerpts.
- `suggest_ideas`: generates three follow-up query ideas.

Additional runtime prompt construction in `src/genai.py`:

- Publication continuation prompt includes conversation history, initial answer, publication excerpts, metric-specific constraints, and instructions not to output source lists or raw URLs.
- Trusted metric instructions prevent replacing the 666 million SDG7 electricity-access metric with older approximate totals.

## Citation/source handling

Confirmed behavior:

- Retrieved `Document` objects are streamed as a separate `documents[]` chunk from `/model`.
- `AssistantResponse` serializes documents into compact frontend-compatible shape:
  - `title`
  - `year`
  - `language`
  - `url`
  - `summary`
- Prompts instruct the model not to output source lists, inline citations, or raw URLs in answer body.
- `/debug/retrieve` returns full selected documents and chunks for inspection.

Unknown / requires confirmation:

- Whether frontend displays `documents[]` as citations, references, or supporting sources.
- Whether page-level citations are required by governance policy.

## Response generation

`/model` streams NDJSON and runs graph generation and answer generation concurrently.

Answer generation flow:

1. Deterministic scope guard runs before generation.
2. If blocked, returns empty graph, refusal text, and scope-safe ideas.
3. If allowed, graph production and answer production start concurrently.
4. `genai.get_answer` starts query-idea generation in parallel.
5. If not deferred, it streams `draft_answer` output.
6. It emits a publication-check bridge.
7. It waits for publication retrieval, sending heartbeat chunks while waiting.
8. If documents exist, it streams a `documents[]` chunk.
9. If chunks exist, it streams a publication-grounded continuation.
10. It emits final ideas.

For current-data queries where `should_defer_to_publications` is true, the initial draft answer is skipped and the stream starts with `I will check the publications for the latest data.`

## Guardrails and response constraints

Confirmed guardrails:

- API key authentication on protected core endpoints.
- Deterministic scope guard before `/model` generation.
- Prompt-probe and harmful request blocking.
- Off-topic blocking for clearly unrelated prompts.
- Output HTML encoding for streamed content, ideas, graph strings, and document metadata.
- Markdown images are neutralized.
- Document hrefs are restricted to HTTP(S).
- Answer prompts prohibit raw URLs and source lists in answer body.
- RAG prompt tells model to stay within named country/region scope.
- Retrieval enforces geography filters when scope is detected.
- Timeouts and fallback responses exist for graph generation, publication retrieval, and streaming.

Known limitations:

- Scope guard is deterministic keyword/phrase based, not a trained semantic classifier.
- RAG corpus ingestion from raw PDFs is not reproducible from code in this repository.
- Citation granularity is document-level unless chunk page metadata exists.
- Current trusted SDG7 fallback facts are hard-coded for selected high-value metrics.
- The ReAct `retrieve_chunks` tool still exists, but current `/model` path mainly uses scheduled publication retrieval rather than relying only on the agent to call the tool.
