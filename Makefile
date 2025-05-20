uv: ## install uv
	curl -LsSf https://astral.sh/uv/install.sh | sh

install: ## Install all dependencies using uv
	uv sync --all-extras --no-extra full

dev: ## Set up dev environment and install pre-commit
	uv venv -p 3.12
	uv sync --all-extras --no-extra full
	uv pip install -e .
	uv run pre-commit install
	uv run gf install-klayout-genericpdk
	uv run gf install-git-diff

install-kfactory-dev: ## Force-reinstall kfactory from GitHub
	uv pip install git+https://github.com/gdsfactory/kfactory --force-reinstall

update-pre: ## Update pre-commit hooks
	pre-commit autoupdate

test-data: ## Clone test data from GitHub (HTTPS)
	git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test_klayout test-data-gds

test-data-gds: ## Clone test data from GitHub (SSH)
	git clone git@github.com:gdsfactory/gdsfactory-test-data.git -b test_klayout test-data-gds

test: test-data-gds ## Run tests
	uv run pytest -s

test-force: ## Run tests with force-regen
	uv run pytest -n logical --force-regen -s

cov: ## Run tests with coverage
	uv run pytest --cov=gdsfactory --cov-report=term-missing:skip-covered

dev-cov: ### Run tests in parallel with coverage
	uv run pytest -s -n logical --cov=gdsfactory --cov-report=term-missing:skip-covered --durations=10

test-samples: ## Test that samples run without error
	uv run pytest tests/test_samples.py

docker-debug: ## Start a debug shell in Docker
	docker run -it joamatab/gdsfactory sh

docker-build: ## Build Docker image
	docker build -t joamatab/gdsfactory .

docker-run: ## Run Docker container
	docker run \
		-p 8888:8888 \
		-p 8082:8082 \
		-e JUPYTER_ENABLE_LAB=yes \
		joamatab/gdsfactory:latest

build: ## Build python package
	rm -rf dist
	pip install build
	python -m build

upload-devpi: ## Upload package to devpi
	pip install devpi-client wheel
	devpi upload --format=bdist_wheel,sdist.tgz

upload-twine: build ## Upload package to PyPI using twine
	pip install twine
	twine upload dist/*

autopep8: ## Format python files with autopep8
	autopep8 --in-place --aggressive --aggressive **/*.py

docs: ## Build documentation
	uv run python docs/write_cells.py
	uv run jb build docs

git-rm-merged: ## Delete merged git branches
	git branch -D `git branch --merged | grep -v \* | xargs`

notebooks: ## Convert python scripts to Jupyter notebooks
	jupytext docs/notebooks/*.py --to ipynb

clean: ## Remove build, cache and temporary files
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

# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help: ## this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
