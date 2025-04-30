# dsc-sea-ai-api

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/github/license/undp-data/dsc-sea-ai-api)](https://github.com/undp-data/dsc-sea-ai-api/blob/main/LICENSE)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)

A Python API to serve data from the knowledge graph for the Sustainable Energy Academy.

> [!WARNING]  
> The package is currently undergoing a major revamp. Some features may be missing or not working as intended. Feel free to [open an issue](https://github.com/UNDP-Data/dsc-sea-ai-api/issues).

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

Follow the steps below to run the API locally.

1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment.
3. Create and populate the `.env` file base on `.env.example`.
4. Run `make install` to install project dependencies.
5. To launch the API, run `python main.py`. The API will be running at http://127.0.0.1:5000.

```bash
git clone https://github.com/UNDP-Data/dsc-sea-ai-api
cd dsc-sea-ai-api
python -m venv .venv
source .venv/bin/activate
make install
python main.py
# Running on http://127.0.0.1:5000
```

## Deployment

The project is hooked up to a CI/CD pipeline. Committing to `main` branch will trigger deployment to Azure Web App service. A pull request is required to change the branch.

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
