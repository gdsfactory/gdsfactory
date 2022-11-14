help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

full: gdslib
	pip install -r requirements_dev.txt
	pip install -r requirements_full.txt
	pip install -e .
	pip install -r requirements_tidy3d.txt
	pip install -r requirements_sipann.txt
	pip install -r requirements_devsim.txt

install: gdslib
	pip install -r requirements_dev.txt
	pip install -r requirements_full.txt
	pip install -e .
	pre-commit install
	gf tool install

mamba:
	bash mamba.sh

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
	pip install -e .[tidy3d]
	pip install jax jaxlib
	mamba install pymeep=*=mpi_mpich_* -y
	pip install -r requirements_sipann.txt
	pip install --upgrade "protobuf<=3.20.1"

plugins-debian:
	sudo apt install libgl1-mesa-glx -y
	pip install -e .[tidy3d]
	pip install jax jaxlib
	mamba install pymeep=*=mpi_mpich_* -y
	pip install -r requirements_sipann.txt
	pip install --upgrade "protobuf<=3.20.1"

thermal:
	mamba install python-gmsh

meep:
	mamba install pymeep=*=mpi_mpich_* -y

sax:
	pip install jax jaxlib

update:
	pur
	pur -r requirements_dev.txt

publish:
	anaconda upload environment.yml

update-pre:
	pre-commit autoupdate --bleeding-edge

gds:
	python gdsfactory/components/straight.py

gdslib:
	git clone https://github.com/gdsfactory/gdslib.git -b data

test:
	flake8 gdsfactory
	pytest -s

test-force:
	echo 'Regenerating component metadata for regression test. Make sure there are not any unwanted regressions because this will overwrite them'
	rm -rf gdslib/gds/gds_ref
	rm -rf gdsfactory/samples/pdk/test_fab_c.gds
	pytest --force-regen

test-meep:
	pytest gdsfactory/simulation/gmeep

test-tidy3d:
	pytest gdsfactory/simulation/gtidy3d

test-plugins:
	pytest gdsfactory/simulation/gmeep gdsfactory/simulation/modes gdsfactory/simulation/lumerical gdsfactory/simulation/simphony gdsfactory/simulation/gtidy3d

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
	python setup.py sdist bdist_wheel

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

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

link:
	lygadgets_link gdsfactory/klayout

spell:
	codespell -i 3 -w -L TE,TE/TM,te,ba,FPR,fpr_spacing

.PHONY: gdsdiff build conda
