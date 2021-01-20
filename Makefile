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
	pytest -x

test-force:
	echo 'Regenerating component metadata for regression test. Make sure there are not any unwanted regressions because this will overwrite them'
	pytest --force-regen

test-notebooks:
	py.test --nbval notebooks

cov:
	pytest --cov=pp

venv:
	python3 -m venv env


pyenv3:
	pyenv shell 3.7.2
	virtualenv venv
	source venv/bin/activate
	python -V # Print out python version for debugging
	which python # Print out which python for debugging
	python setup.py develop

conda:
	conda env create -f environment.yml
	echo 'conda env installed, run `conda activate pp` to activate it'

mypy:
	mypy . --ignore-missing-imports

build:
	python setup.py sdist bdist_wheel

devpi-release:
	pip install devpi-client wheel
	devpi upload --format=bdist_wheel,sdist.tgz

release:
	git push origin --tags

lint:
	tox -e flake8

pylint:
	pylint --rcfile .pylintrc pp/

lintdocs:
	flake8 --select RST

lintdocs2:
	pydocstyle pp

doc8:
	doc8 docs/

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

codestyle:
	pycodestyle --max-line-length=88

.PHONY: gdsdiff build conda
