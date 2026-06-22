# SGP Assistant Test Notes

Use `eval/questions.yaml` as the first retrieval smoke-test surface for this
prototype.

Recommended checks after installing the kit in the source backend:

```bash
python3 scripts/validate_assistant_kit.py \
  --kit "/Users/ben/Documents/UNDP/SGP/SGP AI/sgp_ai"
```

After importing the corpus and starting the backend, use debug retrieval for the
eval questions:

```bash
curl -sS -H "X-Api-Key: $API_KEY" \
  "http://127.0.0.1:8000/assistants/sgp_ai/debug/retrieve?query=Who%20are%20the%20OP8%20guidelines%20written%20for"
```

Expected behavior:

- OP8 governance questions retrieve the OP8 Operational Guidelines first.
- Latest annual-results questions retrieve the 2024-2025 Annual Monitoring Report first.
- Biodiversity activity questions retrieve the SGP Biodiversity Area of Work first.
- Capacity-development questions retrieve the SGP Capacity Development Approach first.
- 2023-2024 priority-group questions retrieve the 2023-2024 Annual Monitoring Report first.
