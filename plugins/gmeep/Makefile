help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make waveguide:        Build a sample waveguide'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install:
	bash install.sh

test:
	pytest

test-force:
	echo 'Regenerating component metadata for regression test. Make sure there are not any unwanted regressions because this will overwrite them'
	pytest --force-regen

update:
	pre-commit autoupdate --bleeding-edge

cov:
	pytest --cov=gmeep

mypy:
	mypy . --ignore-missing-imports

lint:
	tox -e flake8

pylint:
	pylint --rcfile .pylintrc gmeep/

lintdocs:
	flake8 --select RST

lintdocs2:
	pydocstyle gmeep

doc8:
	doc8 docs/

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

codestyle:
	pycodestyle --max-line-length=88

.PHONY: build
