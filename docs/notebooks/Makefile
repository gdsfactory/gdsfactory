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
	jupytext *.ipynb --to md
	jupytext **/*.ipynb --to md
	jupytext **/*.ipynb --to md

clean:
	nbstripout --drop-empty-cells *.ipynb
	nbstripout --drop-empty-cells **/*.ipynb

rm:
	rm *.ipynb
	rm **/*.ipynb
