# Sample Publications Assistant Kit

This kit proves the copyable assistant workflow with an isolated non-SEA profile.

Start by reading `CODEX_HANDOFF.md`. It contains the full cross-repo workflow and the stable data surface that must be preserved.

- Assistant id: `sample`
- Runtime route: `/assistants/sample/model`
- Tables: `sample_chunks`, `sample_documents`, `sample_sources`

Validate from the original backend:

```bash
python scripts/validate_assistant_kit.py --kit assistant_kits/sample
```

Install profile only:

```bash
python scripts/install_assistant_kit.py --kit assistant_kits/sample
```

Install and import sample corpus:

```bash
python scripts/install_assistant_kit.py --kit assistant_kits/sample --import-corpus --include-chunks
```
