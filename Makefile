help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make waveguide:        Build a sample waveguide'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install: gdslib
	pip install -r requirements.txt --upgrade
	pip install -e .
	python install_klive.py
	python install_gdsdiff.py
	python install_generic_tech.py
	pip install pre-commit
	pre-commit install

waveguide:
	python pp/components/waveguide.py


gdslib:
	git clone https://github.com/gdsfactory/gdslib.git

test:
	pyflakes pp && pytest

test-force:
	echo 'we are going to fix the metadata of all components'
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

release:
	pip install devpi-client wheel
	devpi upload --format=bdist_wheel,sdist.tgz


.PHONY: gdsdiff build
