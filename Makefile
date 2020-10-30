help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make waveguide:        Build a sample waveguide'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install: gdslib
	bash install.sh

test:
	pytest

test-force:
	echo 'Regenerating component metadata for regression test. Make sure there are not any unwanted regressions because this will overwrite them'
	pytest --force-regen

cov:
	pytest --cov=pp

venv:
	python3 -m venv env

meep:
	conda config --add channels conda-forge
	pip install ipykernel
	conda install -y pymeep


mypy:
	mypy . --ignore-missing-imports

build:
	python setup.py sdist bdist_wheel

devpi-release:
	pip install devpi-client wheel
	devpi upload --format=bdist_wheel,sdist.tgz

release:
	git push origin --tags


.PHONY: build
