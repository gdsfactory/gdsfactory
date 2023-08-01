help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install:
	pip install -e .[full,dev,docs] pre-commit
	pre-commit install
	gf install klayout-genericpdk
	gf install git-diff

dev: install
	pip install gplugins[tidy3d]

update-pre:
	pre-commit autoupdate --bleeding-edge

test-data:
	git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test-data test-data

data-download: test-data
	echo 'Make sure you git pull inside test-data folder'

data-download-old:
	aws s3 sync s3://gdslib data --no-sign-request

data-clean:
	aws s3 rm data s3://gdslib/gds

test:
	pytest -s

test-force:
	pytest --force-regen -s

test-watch:
	ptw

test-notebooks:
	py.test --nbval notebooks

diff:
	python gdsfactory/merge_cells.py

cov:
	pytest --cov=gdsfactory

venv:
	python3 -m venv env

pipenv:
	pip install pipenv --user
	pipenv install

pyenv3:
	pyenv shell 3.7.2
	virtualenv venv
	source venv/bin/activate
	python -V # Print out python version for debugging
	which python # Print out which python for debugging
	python setup.py develop

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

conda:
	conda env create -f environment.yml
	echo 'conda env installed, run `conda activate gdsfactory` to activate it'

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

release:
	git push
	git push origin --tags

lint:
	tox -e flake8

pylint:
	pylint --rcfile .pylintrc gdsfactory/

lintdocs:
	flake8 --select RST

pydocstyle:
	pydocstyle gdsfactory

doc8:
	doc8 docs/

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

codestyle:
	pycodestyle --max-line-length=88

doc:
	python .github/write_components_doc.py

docs:
	jb build docs

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

link:
	lygadgets_link gdsfactory/klayout

constructor:
	conda install constructor -y
	constructor conda

notebooks:
	jupytext gdsfactory/samples/notebooks/*.md --to ipynb notebooks/

jupytext:
	jupytext **/*.ipynb --to py

jupytext-clean:
	jupytext docs/**/*.py --to py

notebooks:
	# jupytext docs/notebooks/*.py --to ipynb
	# jupytext docs/notebooks/*.ipynb --to to
	jupytext --pipe black docs/notebooks/*.py

.PHONY: gdsdiff build conda gdslib docs doc
