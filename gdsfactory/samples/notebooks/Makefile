format:
	jupytext --pipe black *.ipynb
	jupytext --pipe black **/*.ipynb

sync:
	jupytext --sync *.ipynb

python:
	jupytext **/*.ipynb --to py

notebooks:
	jupytext *.md --to ipynb
	jupytext **/*.md --to ipynb

markdown:
	jupytext *.py --to md
	jupytext **/*.py --to md
	jupytext **/*.py --to md

clean:
	nbstripout --drop-empty-cells *.ipynb
	nbstripout --drop-empty-cells **/*.ipynb

rm:
	rm *.ipynb
	rm **/*.ipynb


set:
	jupytext --set-formats md,ipynb **/*.md
