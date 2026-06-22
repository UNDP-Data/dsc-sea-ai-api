# Codex Handoff for This Assistant Kit

This document is the primary instruction source for Codex when this assistant kit is copied into another repo.

If a user says, “check the assistant kit in this repo and follow the documentation,” start here.

## Objective

Create or refine a local publication-backed RAG assistant prototype using this kit folder only. Preserve the exact kit structure and data surface so the finished kit can be installed back into the original backend and deployed under `/assistants/{assistant_id}/...`.

The backend runtime is not part of this kit. Do not recreate or fork the API unless explicitly asked. The original backend owns routing, authentication, Azure OpenAI clients, LanceDB storage access, streaming, response encoding, and deployment.

## Required folder structure

Do not rename or move these files:

```text
assistant.yaml
corpus/manifest.yaml
eval/questions.yaml
README.md
CODEX_HANDOFF.md
tests/
```

The original backend installer expects this shape. Adding extra assistant-specific notes or fixtures is acceptable, but the required files must remain in place.

## Files Codex may edit

Edit only assistant-owned files:

```text
assistant.yaml
corpus/manifest.yaml
eval/questions.yaml
README.md
CODEX_HANDOFF.md
tests/**
```

Do not add backend runtime code, FastAPI routes, authentication code, Azure client code, LanceDB repository code, or generic RAG framework code inside this kit. If a future assistant truly needs new runtime behavior, document the requirement in `README.md` so it can be implemented in the original backend’s shared `src/rag_system` layer.

## Data surface that must remain stable

The kit must remain compatible with the original backend’s assistant-kit installer and profile loader.

### `assistant.yaml`

Required fields:

```yaml
assistant_id: my_assistant
display_name: My Assistant
domain_description: >
  Domain this assistant supports.
refusal_guidance: >
  User-facing message for out-of-scope questions.

tables:
  chunks: my_assistant_chunks
  documents: my_assistant_documents
  sources: my_assistant_sources

prompts:
  draft_answer: >
    Initial answer prompt.
  answer_with_publications: >
    Publication-grounded continuation prompt.
  suggest_ideas: >
    Follow-up suggestion prompt.
```

Rules:

- `assistant_id` must start with a lowercase letter and contain only lowercase letters, digits, and underscores.
- For non-SEA assistants, tables must be exactly:
  - `{assistant_id}_chunks`
  - `{assistant_id}_documents`
  - `{assistant_id}_sources`
- Keep source definitions, tag rules, and scope terms in `assistant.yaml` rather than hard-coding behavior elsewhere.
- Do not use SEA table names (`chunks`, `documents`, `sources`) unless the assistant id is exactly `sea`.

### `corpus/manifest.yaml`

Required structure:

```yaml
assistant_id: my_assistant
sources:
  - source_id: my_source
    name: Source Name
    organization: Source Organization
    authority_tier: trusted
    base_url: https://example.org
    ingestion_method: manual_manifest
    review_policy: hybrid_editorial

documents:
  - source_id: my_source
    title: Example Publication
    url: https://example.org/publication
    year: 2026
    language: en
    summary: >
      Publication-level summary.
    topic_tags:
      - example topic
    region_codes:
      - global
    audience_tags:
      - policy-makers
    chunks:
      - section_title: Overview
        content: >
          Prepared publication text chunk.
```

Rules:

- `assistant_id` must match `assistant.yaml`.
- `sources` must contain at least one source.
- `documents` must contain at least one document.
- Every document must include `source_id`, `title`, `url`, `year`, `language`, and `summary`.
- Every document must include either `content` or a non-empty `chunks` list.
- Every chunk must include non-empty `content`.
- Use prepared text/chunks. Raw PDF parsing is outside this kit’s v1 workflow.

### `eval/questions.yaml`

Use this to capture the expected retrieval behavior while prototyping:

```yaml
assistant_id: my_assistant
questions:
  - question_id: q1
    query: What should the assistant answer?
    expected_documents:
      - rank: 1
        title: Best Source Title
        url: https://example.org/best-source
        reason: Why this source should be top-ranked.
```

## Local prototyping in another repo

When working in another repo, treat this folder as the assistant product surface.

Recommended process:

1. Read `assistant.yaml`, `corpus/manifest.yaml`, and `eval/questions.yaml`.
2. Rename the assistant by updating `assistant_id`, display name, table names, prompts, scope terms, and manifest `assistant_id` consistently.
3. Replace template sources/documents/chunks with the new publication corpus metadata and prepared text.
4. Add evaluation questions that represent expected user queries and top resources.
5. Keep the folder structure unchanged.
6. Do not modify backend runtime assumptions or response shapes.

## Validation

The authoritative validator lives in the original backend repo. After editing this kit, validate it from the original backend repo:

```bash
python scripts/validate_assistant_kit.py --kit /absolute/path/to/this/kit
```

A valid kit returns JSON with:

```json
{
  "ok": true,
  "assistant_id": "my_assistant",
  "tables": {
    "chunks": "my_assistant_chunks",
    "documents": "my_assistant_documents",
    "sources": "my_assistant_sources"
  }
}
```

If validation fails, fix the kit files rather than changing the validator.

## Deploy back into the backend

Run this from the repo where this kit is being prototyped:

```bash
python3 "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend/scripts/deploy_assistant_kit.py" \
  --kit ./my_assistant_kit
```

Replace `./sgp_ai` with the path to this kit folder if it has a different name.

This installs versionable files into the original backend:

```text
config/rag_profiles/{assistant_id}.yaml
assistant_kits/{assistant_id}/**
```

To also write the manifest into assistant-specific LanceDB tables, add corpus import flags:

```bash
python3 "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend/scripts/deploy_assistant_kit.py" \
  --kit ./my_assistant_kit \
  --import-corpus \
  --include-chunks
```

This writes to:

```text
{assistant_id}_sources
{assistant_id}_documents
{assistant_id}_chunks
```

The deploy-back command copies the kit into the backend but does not alter this prototype repo. Continue development here and rerun the command whenever the kit is ready to sync back.

## Deployment back to Azure

Deployment is performed from the original backend repo, not from this copied repo:

1. Install the refined kit back into the backend.
2. Run backend tests.
3. Commit `config/rag_profiles/{assistant_id}.yaml` and `assistant_kits/{assistant_id}/`.
4. Merge to `main`.
5. The existing Azure Web App workflow deploys the backend.

The deployed assistant is available at:

```text
POST /assistants/{assistant_id}/model
GET  /assistants/{assistant_id}/documents
GET  /assistants/{assistant_id}/sources
GET  /assistants/{assistant_id}/debug/retrieve
```

SEA legacy endpoints remain unchanged.

## Short prompt the user can give Codex in another repo

The user should only need to say:

```text
Check the assistant kit in this repo and review the documentation there on how to use it. Follow it to create a local version of the RAG assistant that we can prototype here before deploying it back to the original backend. Preserve the documented data surface and folder structure so it can be copied back cleanly.
```

Then read this file and proceed according to the steps above.
