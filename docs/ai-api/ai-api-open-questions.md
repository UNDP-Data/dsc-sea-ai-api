# AI API Open Questions

## Corpus and ingestion

- What is the authoritative process for creating `abfs://datasets/corpus-v25-06-27.parquet` from raw PDFs or web pages?
- Where are the raw source PDFs, HTML snapshots, extraction logs, and extraction versions stored?
- Which tool was used for PDF extraction before the corpus parquet was created?
- Who approved the current source list and document inclusion policy?
- Are licensing, robots, and usage restrictions tracked for every publication?
- Is `main.ipynb` still the production ingestion process or only a historical notebook?
- Should production chunks be rewritten in place with `document_id`/`chunk_id`, or should a versioned LanceDB path be created first?
- What is the expected corpus size after the next ingestion expansion?
- Should curated expert annotations become a retrieval-evaluation benchmark, a training signal, or both?

## Retrieval and citations

- Should `documents[]` be described to users as citations, references, sources, or supporting documents?
- Is document-level citation sufficient, or does the platform require page/section-level citations?
- Should the frontend preserve all streamed document updates or replace the visible references list with the newest `documents[]` payload?
- What minimum number of relevant documents should be expected for common policy/concept queries?
- What is the official ranking policy for flagship reports such as Tracking SDG7?
- Should trusted metric fallback facts be moved from code into a governed data table?

## Knowledge Graph

- What is the authoritative upstream pipeline for `nodes-v25-09-25.parquet` and `edges-v25-09-25.parquet`?
- How are node descriptions, weights, and edge weights generated and validated?
- Are KG entities manually curated, model-generated, or both?
- What graph size and response-size limits should the frontend expect?
- Which graph version is production default for the platform frontend: v1, v2, or header-controlled default?

## Governance

- What privacy notice covers user messages sent to the API and model providers?
- What is the retention period for application logs containing raw user queries?
- Should raw queries be redacted or hashed before logging?
- Is there an approved model-provider data-processing agreement for the deployed Azure/OpenAI tenant?
- Is there a required evaluation gate before production deployment?
- Is there an incident-response process for hallucinations, unsafe responses, or source-quality failures?
- Should `/model` have per-user or per-IP rate limits?
- Should the deterministic scope guard be supplemented with a semantic classifier?

## Operations and deployment

- Does production use the Dockerfile, Azure App Service Oryx build, or another startup command?
- Where are production environment variables managed?
- Who owns `AZURE_WEBAPP_PUBLISH_PROFILE` and secret rotation?
- What monitoring platform receives application logs and metrics?
- What are expected SLOs for `/model`, `/graph/v2`, and Moonshot endpoints?
- Are LanceDB tables versioned or backed up before ingestion changes?
- What rollback process exists if retrieval quality regresses after a corpus update?

## Frontend integration

- Which production frontend repositories consume `/model`, `/graph/v2`, `/documents`, and `/sources`?
- What exact NDJSON streaming contract does each frontend assume?
- Are graph chunks and document chunks expected before or after answer text?
- Does the frontend sanitize AI output independently in addition to API-side HTML encoding?
- How are blocked-scope responses displayed to users?

## Moonshot

- Is Moonshot part of the same production service long term or should it be split into a separate API?
- What frontend origins should be configured in `ALLOWED_ORIGINS` for production?
- What is the ownership and lifecycle of the Prodoc blob container?
- Are Moonshot endpoints covered by the same security review as the core AI API?
