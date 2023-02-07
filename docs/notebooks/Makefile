format:
	jupytext --pipe black *.ipynb
	jupytext --pipe black plugins/**/*.ipynb

sync:
	jupytext --sync *.ipynb

python:
	jupytext *.ipynb --to py

markdown:
	jupytext *.ipynb --to md
	jupytext **/*.ipynb --to md
	jupytext plugins/**/*.ipynb --to md

clean:
	nbstripout --drop-empty-cells *.ipynb
	nbstripout --drop-empty-cells **/*.ipynb
