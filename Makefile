help:
	@echo 'make install:          Install package, hook, notebooks and gdslib'
	@echo 'make test:             Run tests with pytest'
	@echo 'make test-force:       Rebuilds regression test'

install:
	pip install -e .[dev,docs,cad] pre-commit
	pre-commit install
	gf install-klayout-genericpdk
	gf install-git-diff

dev: install

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

release:
	git push
	git push origin --tags

autopep8:
	autopep8 --in-place --aggressive --aggressive **/*.py

doc:
	python .github/write_components_doc.py

docs: notebooks
	npm install -g mystmd
	pydoc-markdown -p gdsfactory -m gdsfactory.routing > docs/api.md
	pydoc-markdown -m gdsfactory.routing.* > docs/api_routing.md
	pydoc-markdown -m gdsfactory.components > docs/api_components.md

git-rm-merged:
	git branch -D `git branch --merged | grep -v \* | xargs`

constructor:
	conda install constructor -y
	constructor conda

notebooks:
	python -m ipykernel install --name kernel_name --user
	GDSFACTORY_DISPLAY_TYPE=klayout jupytext docs/notebooks/*.py --to ipynb --execute

.PHONY: gdsdiff build conda gdslib docs doc
