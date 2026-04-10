# Contributing to eachmind

Thank you for your interest in contributing to eachmind! This project is in its early concept stage, and we welcome all forms of contribution.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Install in development mode: `pip install -e ".[dev]"`
4. Create a branch for your changes
5. Make your changes and add tests
6. Run the test suite: `pytest`
7. Submit a pull request

## Development Setup

```bash
git clone https://github.com/your-username/eachmind.git
cd eachmind
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest ruff mypy
```

## Running Tests

```bash
pytest
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

## Type Checking

```bash
mypy eachmind
```

## Areas Where Help Is Needed

- Core primitive implementations and API refinement
- Storage backend adapters (SQLite, Redis, etc.)
- Integration examples with agent frameworks
- Documentation and tutorials
- Drift measurement algorithms
- Consolidation strategies

## Code of Conduct

Be kind, be constructive, be collaborative. We're building something new together.
