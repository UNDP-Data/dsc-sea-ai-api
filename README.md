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
3. Create and populate the `.env` file base on `.env.example`.
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

Optional local tester env:
- `KG_TESTER_REMOTE_API_BASE_URL` to prefill the remote API target in `/kg-tester`.

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

### Internal KG Module Layout

Knowledge graph logic is versioned under:
- `src/kg/v1.py` for V1 graph assembly
- `src/kg/v2.py` for V2 staged graph assembly
- `src/kg/types.py` for V2 response models

Compatibility adapters are retained:
- `src/entities_v2.py`
- `src/graph_v2.py`

## KG Subgraph Tester

The API includes a built-in testing page for the current knowledge graph subgraph system.

1. Start the API locally (`make run`).
2. Open [http://127.0.0.1:8000/kg-tester](http://127.0.0.1:8000/kg-tester).
3. Enter:
   - target (`Local server` or `Remote API`),
   - endpoint version (`/graph` or `/graph/v2`),
   - a graph `query` (for example `climate change mitigation`),
   - remote API base URL (only when target is `Remote API`),
   - your `X-Api-Key` value (from `API_KEY` in your environment).
4. Submit to call the selected graph endpoint and view:
   - an interactive D3 force graph,
   - the raw JSON response payload.

Notes:
- In browser-based `Remote API` mode, the remote API must allow CORS from your local tester origin and allow header `X-Api-Key`.

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
