# dsc-sea-ai-api

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/github/license/undp-data/dsc-sea-ai-api)](https://github.com/undp-data/dsc-sea-ai-api/blob/main/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Build and deploy Python app to Azure Web App](https://github.com/UNDP-Data/dsc-sea-ai-api/actions/workflows/azure-webapps-python.yml/badge.svg)](https://github.com/UNDP-Data/dsc-sea-ai-api/actions/workflows/azure-webapps-python.yml)

A Python API to serve data from the knowledge graph for the Sustainable Energy Academy.

> [!WARNING]  
> The package is currently undergoing a major revamp. Some features may be missing or not working as intended. Feel free to [open an issue](https://github.com/UNDP-Data/dsc-sea-ai-api/issues).

## Table of Contents

- [Getting Started](#getting-started)
- [API Structure](#api-structure)
- [KG Subgraph Tester](#kg-subgraph-tester)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

Follow the steps below to run the API locally.

1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment.
3. Create and populate the `.env` file based on `.env.example`.
4. Run `make install` to install project dependencies.
5. To launch the API, run `make run`. The API will be running at http://127.0.0.1:8000.

```bash
git clone https://github.com/UNDP-Data/dsc-sea-ai-api
cd dsc-sea-ai-api
python -m venv .venv
source .venv/bin/activate
make install
make run
# INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

For Azure Blob auth you can use either:
- `STORAGE_SAS_URL` (full SAS URL), or
- `STORAGE_ACCOUNT_NAME` with `STORAGE_ACCOUNT_KEY` / `STORAGE_SAS_TOKEN`.

Optional local tester env (separate tester app):
- `KG_TESTER_API_KEY` (or fallback `API_KEY`) for tester proxy authentication to backend.
- `KG_TESTER_LOCAL_API_BASE_URL` for local backend target (default `http://127.0.0.1:8000`).
- `KG_TESTER_REMOTE_API_BASE_URL` to prefill remote backend target.

## API Structure

All protected endpoints require `X-Api-Key`.

### V1 Graph Endpoint

- `GET /graph?query=<text>[&hops=<int>]`
- Response:
  - `nodes[]` with `name`, `description`, `neighbourhood`, `weight`, `colour`
  - `edges[]` with `subject`, `predicate`, `object`, `description`, `weight`

### V2 Graph Endpoint

- `GET /graph/v2?query=<text>`
- Response:
  - `nodes[]` with `name`, `description`, `tier`, `weight`, `colour`
    - `tier` is one of: `central`, `secondary`, `periphery`
  - `edges[]` with `subject`, `predicate`, `object`, `description`, `weight`
  - `level` is not returned in the API payload.

### Model Endpoint Graph Version

- `POST /model[?graph_version=v1|v2]`
- `graph_version` default is `v2`.
- Request body must include at least one message; empty lists return `400`.
- Response includes `X-Request-Id` header for request correlation.
- The stream is NDJSON; graph and text generation run in parallel.
- Chunk order is not fixed. The graph chunk usually arrives early, but clients must handle it arriving before, during, or after text deltas.
- To avoid stalled streams under slow storage/model conditions, `/model` applies:
  - `MODEL_GRAPH_TIMEOUT_SECONDS` (graph build timeout),
  - `MODEL_STREAM_IDLE_TIMEOUT_SECONDS` (max idle gap between streamed chunks),
  - `MODEL_TOOLS_PREP_TIMEOUT_SECONDS` (SQL/RAG tool preparation timeout),
  - `MODEL_STREAM_WATCHDOG_SECONDS` (overall no-progress watchdog for stream completion),
  - `RETRIEVE_CHUNKS_TIMEOUT_SECONDS` (RAG chunk retrieval timeout).
- This controls the schema of `graph` returned in streamed `/model` chunks:
  - `v1`: legacy graph schema (`neighbourhood` on nodes, `level` on edges)
  - `v2`: staged graph schema (`tier` on nodes, no edge `level`)

### Internal KG Module Layout

Knowledge graph logic is versioned under:
- `src/kg/v1.py` for V1 graph assembly
- `src/kg/v2.py` for V2 staged graph assembly
- `src/kg/types.py` for V2 response models

Compatibility adapters are retained:
- `src/entities_v2.py`
- `src/graph_v2.py`

## KG Subgraph Tester

The KG tester is now a separate frontend app in `frontend/`. The backend API deployment
does not include the tester route.

1. Start the API backend locally (`make run`).
2. Start the tester frontend app (`make run-tester`).
3. Open [http://127.0.0.1:8010/kg-tester](http://127.0.0.1:8010/kg-tester).
4. Enter:
   - target (`Local server` or `Remote API`),
   - graph version (`default`, `v1`, or `v2`, sent as `graph_version` to `/model` when selected),
   - a graph `query` (for example `climate change mitigation`),
   - remote API base URL (only when target is `Remote API`).
5. Submit to call `/model` and view:
   - streamed answer text (delta chunks),
   - query ideas (clickable chips that trigger a new query),
   - graph payload returned in the stream,
   - an interactive D3 force graph,
   - the raw JSON response payload.

Interaction shortcuts:
- Press `Enter` in the form to run the current query.
- Click a graph node to run a follow-up query (`tell me more about <node>`).
- Click an idea chip to run that suggestion immediately.

Notes:
- The tester proxy uses `KG_TESTER_API_KEY` (or `API_KEY`) from environment and does not expose
  API keys in browser JavaScript.
- The tester proxy forwards `X-Request-Id` (or generates one) to simplify backend trace correlation.
- The tester app calls backend APIs server-to-server, so browser CORS is not required for remote targets.
- No extra CORS proxy is needed when using the standalone tester app.
- The tester app is local-only by design (loopback clients only).

Example local startup:

```bash
# terminal 1
make run

# terminal 2
export KG_TESTER_API_KEY="$API_KEY"
make run-tester
```

`make run-tester` binds to `127.0.0.1:8010` by default.

Pre-commit validation:

```bash
python3 -m py_compile main.py src/security.py src/database.py src/genai.py src/entities.py src/kg/__init__.py src/kg/v1.py src/kg/v2.py frontend/kg_tester_app.py tests/test_model.py tests/test_genai.py
make test   # requires dev dependencies installed
```

To evaluate V2 output quality over a representative query set:

```bash
python3 scripts/evaluate_graph_v2.py --api-key "$API_KEY"
```

Optional flags:
- `--queries-file path/to/queries.txt` for custom query sets
- `--output /tmp/graph_v2_eval.json` to persist full metrics

## Deployment

The project is hooked up to CI/CD via GitHub Actions.
- Workflow: `.github/workflows/azure-webapps-python.yml`
- Azure Web App name: `sea-ai-api`
- A push to `main` triggers deployment.
- The `frontend/` KG tester app is intentionally separate and is not required for API deployment.

## Contributing

All contributions must follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
The codebase is formatted with `black` and `isort`. Use the provided [Makefile](./Makefile) for these
routine operations.

1. Clone or fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Ensure your code is properly formatted (`make format`)
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature-branch`)
7. Open a pull request

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](./LICENSE) file.
