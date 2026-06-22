# Assistant Kits

This directory contains copyable publication-backed assistant kits.

## What to copy to another repo

Copy exactly one kit folder, not the full backend.

Recommended starting point:

```text
assistant_kits/_template/
```

The copied folder must keep this structure:

```text
my_assistant_kit/
  assistant.yaml
  corpus/manifest.yaml
  eval/questions.yaml
  README.md
  CODEX_HANDOFF.md
  tests/
```

`assistant.yaml` and `corpus/manifest.yaml` are the files required to bring the assistant back into this backend later.

## Export to another local repo

From this backend repo:

```bash
python scripts/export_assistant_kit.py \
  --kit assistant_kits/_template \
  --target /path/to/other-repo/my_assistant_kit
```

If the destination already exists and should be replaced/updated:

```bash
python scripts/export_assistant_kit.py \
  --kit assistant_kits/_template \
  --target /path/to/other-repo/my_assistant_kit \
  --overwrite
```

## What to tell Codex in the other repo

The detailed instructions live inside the copied kit, especially `CODEX_HANDOFF.md`. Use a short prompt like this:

```text
Check the assistant kit in this repo and review the documentation there on how to use it. Follow it to create a local version of the RAG assistant that we can prototype here before deploying it back to the original backend. Preserve the documented data surface and folder structure so it can be copied back cleanly.
```

## Bring a refined kit back into this backend

Preferred: run the deploy-back wrapper from inside the other repo:

```bash
python3 "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend/scripts/deploy_assistant_kit.py" \
  --kit ./my_assistant_kit
```

To also import the manifest into assistant-specific LanceDB tables:

```bash
python3 "/Users/ben/Documents/UNDP/SEH/dsc-energy-ai-backend/scripts/deploy_assistant_kit.py" \
  --kit ./my_assistant_kit \
  --import-corpus \
  --include-chunks
```

Equivalent from this backend repo:

```bash
python scripts/install_assistant_kit.py --kit /path/to/other-repo/my_assistant_kit --overwrite
```

## Deployment path

Deployment remains this backend repo:

1. Install the refined kit back here.
2. Commit the changed `assistant_kits/{assistant_id}/` folder and `config/rag_profiles/{assistant_id}.yaml`.
3. Merge to `main`.
4. Existing GitHub Actions deploys the same Azure Web App.

The new assistant is served at:

```text
/assistants/{assistant_id}/model
/assistants/{assistant_id}/documents
/assistants/{assistant_id}/sources
/assistants/{assistant_id}/debug/retrieve
```
