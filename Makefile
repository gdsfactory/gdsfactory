help:
	@echo 'make install:          Install package'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install:
	pip install -e .[dev,docs] pre-commit
	# pip install git+https://github.com/gdsfactory/kfactory --force-reinstall
	gf install-klayout-genericpdk
	gf install-git-diff

update-pre:
	pre-commit autoupdate

test-data:
	git clone https://github.com/gdsfactory/gdsfactory-test-data.git -b test_klayout test-data-gds

test-data-gds:
	git clone git@github.com:gdsfactory/gdsfactory-test-data.git -b test_klayout test-data-gds

test: test-data-gds
	pytest -s

test-force:
	pytest --force-regen -s

cov:
	pytest --cov=gdsfactory

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

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

docs:
	python docs/write_cells.py
	jb build docs

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

notebooks:
	jupytext docs/notebooks/*.py --to ipynb

.PHONY: gdsdiff build conda gdslib docs doc install
