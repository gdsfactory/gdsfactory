install:
	uv sync --extra docs --extra dev

update-pre:
	pre-commit autoupdate --bleeding-edge

build:
	rm -rf dist
	pip install build
	python -m build

docs:
	uv run python .github/write_components.py
	uv run jb build docs

pics:
	uv run gfp build-pics

test:
	uv run gfp test

.PHONY: drc doc docs build
