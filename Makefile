help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make gitdiff:          Git diff GDS shows the boolean operation in klayout'
	@echo 'make notebooks:        Download notebook samples repo'

install: gdslib
	pip install -r requirements.txt --upgrade
	pip install -e .
	python install_klive.py
	python install_gdsdiff.py
	python install_generic_tech.py
	pip install pre-commit
	pre-commit install

install3:
	pyenv shell 3.7.2
	virtualenv venv
	source venv/bin/activate
	python -V # Print out python version for debugging
	which python # Print out which python for debugging
	python setup.py develop

gdslib:
	git clone https://github.com/PsiQ/gdslib.git

gitdiff:
	cd gdsdiff
	python install.py

test:
	pyflakes pp && pytest

hook:
	cp .hooks/pre-commit .git/hooks/pre-commit
	cp .hooks/pre-push .git/hooks/pre-push

hook-lint:
	cp .hooks/pre-commit .git/hooks/pre-commit

hook-pytest:
	cp .hooks/pre-push .git/hooks/pre-push

unhook:
	rm .git/hooks/*

notebooks:
	pip install jupyterlab
	mkdir -p $(HOME)/notebooks
	ln -sf $(PWD)/notebooks $(HOME)/notebooks/gdsfactory

venv:
	python3 -m venv env

mypy:
	mypy . --ignore-missing-imports

waveguide:
	python pp/components/waveguide.py

clean:
	rm -rf build

build:
	python setup.py sdist bdist_wheel

release:
	pip install devpi-client wheel
	devpi upload --format=bdist_wheel,sdist.tgz

lint:
	pyflakes pp

.PHONY: gdsdiff build
