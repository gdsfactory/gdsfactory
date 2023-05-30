help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

full: plugins
	pip install -e .[docs,dev,full,gmsh,tidy3d,devsim,meow,sax,database,kfactory]

all: plugins install full

install:
	pip install -e .[full,dev,kfactory] pre-commit
	pre-commit install
	gf install klayout-genericpdk
	gf install git-diff

dev: full
	pre-commit install
	gf install klayout-genericpdk

mamba:
	bash conda/mamba.sh

patch:
	bumpversion patch
	python docs/write_components_doc.py

minor:
	bumpversion minor
	python docs/write_components_doc.py

major:
	bumpversion major
	python docs/write_components_doc.py

plugins:
	conda install -c conda-forge pymeep=*=mpi_mpich_* nlopt -y
	conda install -c conda-forge slepc4py=*=complex* -y
	pip install -e .[tidy3d,ray,sax,devsim,meow,database,femwell]

plugins-conda:
	conda install -c conda-forge pymeep=*=mpi_mpich_* nlopt -y
	conda install -c conda-forge slepc4py=*=complex* -y
	pip install jax jaxlib numpy femwell --upgrade
	pip install -e .[tidy3d,ray,sax,devsim,meow,database]

plugins-mamba:
	mamba install -c conda-forge pymeep=*=mpi_mpich_* nlopt -y
	mamba install -c conda-forge slepc4py=*=complex* -y
	pip install jax jaxlib numpy femwell --upgrade
	pip install -e .[tidy3d,ray,sax,devsim,meow]

plugins-debian: plugins
	sudo apt-get update
	sudo apt-get install -y python3-gmsh gmsh

thermal:
	conda install python-gmsh

gmsh:
	pip install trimesh mapbox_earcut gmsh meshio pygmsh pyvista h5py

meep:
	conda install pymeep=*=mpi_mpich_* -y

sax:
	pip install jax jaxlib

publish:
	anaconda upload environment.yml

update-pre:
	pre-commit autoupdate --bleeding-edge

gds:
	python gdsfactory/components/straight.py

data-upload:
	echo 'no need to upload'
	# aws s3 sync data s3://gdslib
	# gh release upload v6.90.3 data/gds/*.gds --clobber
	# gh release upload v6.90.3 data/sp/*.npz --clobber
	# gh release upload v6.90.3 data/sp/*.yml --clobber
	# gh release upload v6.90.3 data/modes/*.msh --clobber
	# gh release upload v6.90.3 data/modes/*.npz --clobber

test-data:
	git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test-data test-data

data-download: test-data
	echo 'Make sure you git pull inside test-data folder'
	# aws s3 sync s3://gdslib data --no-sign-request
	# gh release download v6.90.3 -D data/gds/*.gds --clobber
	# gh release download v6.90.3 data/sp/*.npz --clobber
	# gh release download v6.90.3 data/sp/*.yml --clobber
	# gh release download v6.90.3 data/modes/*.msh --clobber
	# gh release download v6.90.3 data/modes/*.npz --clobber

data-clean:
	aws s3 rm data s3://gdslib/gds

test:
	pytest -s

test-force:
	pytest --force-regen -s

test-watch:
	ptw

test-meep:
	pytest gdsfactory/simulation/gmeep

test-tidy3d:
	pytest gdsfactory/simulation/gtidy3d

test-gmsh:
	pytest gdsfactory/simulation/gmsh

test-femwell:
	pytest gdsfactory/simulation/fem

test-plugins:
	pytest gdsfactory/simulation/gmeep gdsfactory/simulation/modes gdsfactory/simulation/lumerical gdsfactory/simulation/gmsh tests/test_klayout gdsfactory/simulation/fem

test-plugins-no-tidy3d:
	pytest gdsfactory/simulation/gmeep gdsfactory/simulation/modes gdsfactory/simulation/lumerical gdsfactory/simulation/gmsh tests/test_klayout gdsfactory/simulation/fem

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

mypy:
	mypy gdsfactory --ignore-missing-imports

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
	python docs/write_components_doc.py

docs:
	jb build docs

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

link:
	lygadgets_link gdsfactory/klayout

constructor:
	conda install constructor -y
	constructor conda

nbqa:
	nbqa blacken-docs docs/notebooks/**/*.ipynb --nbqa-md
	nbqa blacken-docs docs/notebooks/*.ipynb --nbqa-md
	nbqa isort docs/notebooks/*.ipynb --float-to-top
	nbqa isort docs/notebooks/**/*.ipynb --float-to-top
	nbqa ruff --fix docs/notebooks/*.ipynb
	nbqa ruff --fix docs/**/*.ipynb
	nbqa autopep8 -i docs/notebooks/*.ipynb
	nbqa autopep8 -i docs/notebooks/**/*.ipynb

jupytext:
	jupytext **/*.ipynb --to py

jupytext-clean:
	jupytext docs/**/*.py --to py

notebooks:
	jupytext docs/notebooks/**/*.py --to ipynb
	jupytext docs/notebooks/*.py --to ipynb

.PHONY: gdsdiff build conda gdslib docs doc
