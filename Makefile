.PHONY: check ci lint type test fix compile sync-dev

check: lint type test

ci: check

sync-dev:
	uv sync --group dev

lint:
	uv run ruff check .

type:
	uv run mypy

test:
	uv run pytest --cov=awesome_feature_navigation --cov-report=term-missing --cov-fail-under=92

fix:
	uv run ruff check . --fix

compile:
	uv run python -m compileall src tests
