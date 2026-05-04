install:
	uv sync --extra docs --extra dev

test:
	uv run pytest -s tests/test_cells.py

test-force:
	uv run pytest -s tests/test_cells.py --force-regen

test-fail-fast:
	uv run pytest -s tests/test_cells.py -x

update-pre:
	pre-commit autoupdate --bleeding-edge

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

build:
	rm -rf dist
	pip install build
	python -m build

jupytext:
	jupytext docs/**/*.ipynb --to py

notebooks:
	jupytext docs/**/*.py --to ipynb

docs:
	uv run python .github/write_cells.py
	uv run jb build docs

.PHONY: drc doc docs
