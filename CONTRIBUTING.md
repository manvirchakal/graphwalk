# Contributing to graphwalk

## Setup

```bash
uv sync --all-extras      # base deps, every optional extra, and the dev group
uvx pre-commit install    # optional: run the checks below on every commit
```

Add extras as needed (`uv sync --extra llm`, and so on). Use `uv add` / `uv add --optional <extra>`
to change dependencies, and commit the updated `uv.lock`.

## Checks

Run these before every commit. CI runs the same set on Python 3.12 and 3.13 with all
extras, runs the tests again with no extras, and builds and smoke-tests the wheel:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright            # strict on src/
uv run pytest
```

## Tests

- The default `uv run pytest` is **offline**: `pytest-socket` blocks all INET sockets, so
  any accidental network call fails the test. Use the fake decision, LLM, and embedder
  backends.
- `@pytest.mark.live` tests call real APIs. Run them with `uv run pytest --run-live`
  (needs the relevant API key in the environment).
- `@pytest.mark.neo4j` tests need a running Neo4j. Run them with `uv run pytest --run-neo4j`.
- Numerical logic (beam scoring, sampling, guardrails) is tested with exact expected values.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
`test:`, `refactor:`, `chore:`, `ci:`. Never commit secrets; `.env` is gitignored and
`.env.example` documents every variable. Add user-visible changes to `CHANGELOG.md` under
"Unreleased". Report security issues privately (see `SECURITY.md`).

## Design

Read [`docs/design.md`](docs/design.md) before larger changes. Protocol changes
(`GraphStore`, `DecisionBackend`, `Embedder`, `LLMBackend`, `Source`) need a design-doc update.
