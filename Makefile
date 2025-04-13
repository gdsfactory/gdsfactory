help:
	@echo 'make install:          Install package'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

uv:
	curl -LsSf https://astral.sh/uv/0.4.30/install.sh | sh

install:
	uv sync --all-extras

install310:
	uv sync --all-extras

dev:
	uv venv -p 3.12
	uv sync --all-extras
	uv pip install -e .
	uv run pre-commit install
	uv run gf install-klayout-genericpdk
	uv run gf install-git-diff

install-kfactory-dev:
	uv pip install git+https://github.com/gdsfactory/kfactory --force-reinstall

update-pre:
	pre-commit autoupdate

test-data:
	git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test_klayout test-data-gds

test-data-gds:
	git clone git@github.com:gdsfactory/gdsfactory-test-data.git -b test_klayout test-data-gds

test: test-data-gds
	pytest -s

test-force:
	uv run pytest -n logical --force-regen -s

uv-test: test-data-gds
	uv run pytest -s -n logical

cov:
	uv run pytest --cov=gdsfactory --cov-report=term-missing:skip-covered

dev-cov:
	uv run pytest -s -n logical --cov=gdsfactory --cov-report=term-missing:skip-covered --durations=10

test-samples:
	uv run pytest tests/test_samples.py

docker-debug:
	docker run -it joamatab/gdsfactory sh

docker-build:
	docker build -t joamatab/gdsfactory .

docker-run:
	docker run \
		-p 8888:8888 \
		-p 8082:8082 \
		-e JUPYTER_ENABLE_LAB=yes \
		joamatab/gdsfactory:latest

build:
	rm -rf dist
	pip install build
	python -m build

upload-devpi:
	pip install devpi-client wheel
	devpi upload --format=bdist_wheel,sdist.tgz

upload-twine: build
	pip install twine
	twine upload dist/*

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

docs:
	uv run python docs/write_cells.py
	uv run jb build docs

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

notebooks:
	jupytext docs/notebooks/*.py --to ipynb

clean:
	rm -rf .venv
	find src -name "*.c" | xargs rm -rf
	find src -name "*.pyc" | xargs rm -rf
	find src -name "*.so" | xargs rm -rf
	find src -name "*.pyd" | xargs rm -rf
	find . -name "*.egg_info" | xargs rm -rf
	find . -name ".ipynb_checkpoints" | xargs rm -rf
	find . -name ".mypy_cache" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf
	find . -name ".ruff_cache" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "builds" | xargs rm -rf
	find . -name "dist" -not -path "*node_modules*" | xargs rm -rf

.PHONY: gdsdiff build conda gdslib docs doc install
