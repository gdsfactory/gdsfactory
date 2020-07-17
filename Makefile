help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make waveguide:        Build a sample waveguide'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install: gdslib
	bash install.sh

waveguide:
	python pp/components/waveguide.py

gdslib:
	git clone https://github.com/gdsfactory/gdslib.git

test:
	pytest

test-force:
	echo 'Regenerating component metadata for regression test. Make sure there are not any unwanted regressions because this will overwrite them'
	pytest --force-regen

venv:
	python3 -m venv env

pyenv3:
	pyenv shell 3.7.2
	virtualenv venv
	source venv/bin/activate
	python -V # Print out python version for debugging
	which python # Print out which python for debugging
	python setup.py develop

mypy:
	mypy . --ignore-missing-imports

build:
	python setup.py sdist bdist_wheel

devpi-release:
	pip install devpi-client wheel
	devpi upload --format=bdist_wheel,sdist.tgz

release:
	git push origin --tags


.PHONY: gdsdiff build
