format:
	jupytext --pipe black *.ipynb
	jupytext --pipe black **/*.ipynb

sync:
	jupytext --sync *.ipynb

python:
	jupytext **/*.ipynb --to py

notebooks:
	jupytext *.py --to ipynb
	jupytext **/*.py --to ipynb

markdown:
	jupytext *.py --to md
	jupytext **/*.py --to md
	jupytext **/*.py --to md

clean:
	nbstripout --drop-empty-cells *.ipynb
	nbstripout --drop-empty-cells **/*.ipynb

rm:
	rm *.py
	rm **/*.py
