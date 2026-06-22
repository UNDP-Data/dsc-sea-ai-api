# Copyable Assistant Kit Template

This folder is the portable unit for building a new publication-backed RAG assistant in another repo and later deploying it back through the original backend.

Start by reading `CODEX_HANDOFF.md`. It contains the full cross-repo workflow and the stable data surface that must be preserved.

## Required files

```text
assistant.yaml
corpus/manifest.yaml
eval/questions.yaml
README.md
CODEX_HANDOFF.md
tests/
```

## What to edit

- `assistant.yaml`: assistant id, table names, prompts, source rules, tagging rules, and scope terms.
- `corpus/manifest.yaml`: prepared source, document, and chunk records.
- `eval/questions.yaml`: prototype/evaluation questions and expected top resources.
- `README.md`: assistant-specific notes.
- `tests/`: assistant-specific fixtures or notes.

## What not to edit here

Do not add backend runtime code in this folder. The backend owns FastAPI routes, auth, Azure OpenAI clients, LanceDB access, streaming, response encoding, and deployment.

## Naming rule

For non-SEA assistants, table names must use this namespace:

```text
{assistant_id}_chunks
{assistant_id}_documents
{assistant_id}_sources
```

## Validate from the original backend

```bash
python scripts/validate_assistant_kit.py --kit /absolute/path/to/this/kit
```

## Deploy back into the backend from the prototype repo

```bash
python3 "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend/scripts/deploy_assistant_kit.py" --kit ./my_assistant_kit
```

Replace `./sgp_ai` with the local kit folder path. Add `--import-corpus --include-chunks` only when you want to write the manifest into LanceDB.
