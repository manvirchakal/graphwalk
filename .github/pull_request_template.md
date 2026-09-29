## What and why

<!-- What this changes and the problem it solves. Link issues. -->

## How it was tested

<!-- New or changed tests; for eval changes, the run directory and its summary. -->

## Checklist

- [ ] `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest` pass
- [ ] Default tests make no network calls
- [ ] `CHANGELOG.md` updated under "Unreleased" (user-visible changes)
- [ ] `docs/design.md` updated if a protocol changed
